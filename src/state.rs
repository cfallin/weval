//! State tracking.

use crate::image::Image;
use crate::value::{Value, ValueTags};
use std::collections::BTreeMap;
use walrus::{ir::InstrSeqType, FunctionId, FunctionKind, GlobalId, LocalId, Module, ModuleTypes};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    /// Memory overlay. We store only aligned u32s here.
    pub mem_overlay: BTreeMap<u32, Value>,
    /// Global values.
    pub globals: BTreeMap<GlobalId, Value>,
    /// Local values.
    pub locals: BTreeMap<LocalId, Value>,
    /// Operand stack. May be partial: describes the suffix of the
    /// operand stack. This allows meets to work more easily when a
    /// block returns results to its parent block.
    pub stack: Vec<Value>,
    /// Loop stack, with a known PC per loop level.
    pub loop_pcs: Vec<Option<u64>>,
}

fn map_meet_with<K: PartialEq + Eq + PartialOrd + Ord + Copy>(
    this: &mut BTreeMap<K, Value>,
    other: &BTreeMap<K, Value>,
) -> bool {
    let mut changed = false;
    for (k, val) in this.iter_mut() {
        if let Some(other_val) = other.get(k) {
            let met = Value::meet(*val, *other_val);
            changed |= (met != *val);
            *val = met;
        } else {
            *val = Value::Top;
            // N.B.: not changed: Top should not have an effect (every
            // non-present key is conceptually already Top).
        }
    }
    for other_k in other.keys() {
        if !this.contains_key(other_k) {
            // `Runtime` is a "bottom" value in the semilattice.
            this.insert(*other_k, Value::Runtime(ValueTags::default()));
            changed = true;
        }
    }
    changed
}

impl State {
    pub fn initial(module: &Module, im: &Image, func: FunctionId, args: Vec<Value>) -> State {
        let arg_locals = match &module.funcs.get(func).kind {
            FunctionKind::Local(lf) => &lf.args[..],
            _ => panic!("Getting state for non-local function"),
        };
        let locals = arg_locals.iter().cloned().zip(args.into_iter()).collect();
        let globals = im
            .globals
            .iter()
            .map(|(id, val)| (*id, Value::Concrete(*val, ValueTags::default())))
            .collect();

        State {
            mem_overlay: BTreeMap::new(),
            globals,
            locals,
            stack: vec![],
            loop_pcs: vec![],
        }
    }

    pub fn meet_with(&mut self, other: &State) -> bool {
        let mut changed = false;
        changed |= map_meet_with(&mut self.mem_overlay, &other.mem_overlay);
        changed |= map_meet_with(&mut self.globals, &other.globals);
        changed |= map_meet_with(&mut self.locals, &other.locals);

        // Meet stacks. Zip stacks backward, since they describe the
        // suffix of the stack. Grow our stack to the size of
        // `other`'s stack at least, filling in Top values as needed.
        if self.stack.len() < other.stack.len() {
            let diff = other.stack.len() - self.stack.len();
            self.stack.resize(other.stack.len(), Value::Top);
            self.stack.rotate_right(diff);
            changed = true;
        }
        for (this_stack, other_stack) in self.stack.iter_mut().zip(other.stack.iter()) {
            let val = *this_stack;
            let met = Value::meet(*this_stack, *other_stack);
            if met != val {
                *this_stack = met;
                changed = true;
            }
        }
        changed
    }

    /// Create a clone of the state to flow into a block, taking args
    /// from the parent block's stack.
    pub fn subblock_state(&mut self, ty: InstrSeqType, tys: &ModuleTypes) -> State {
        match ty {
            InstrSeqType::Simple(_) => self.clone(),
            InstrSeqType::MultiValue(ty) => {
                let ty = tys.get(ty);
                let n_params = ty.params().len();
                let n_rets = ty.results().len();
                let mut ret = self.clone();
                // Split off params.
                let param_vals = self.stack.split_off(self.stack.len() - n_params);
                ret.stack = param_vals;
                // Create `Top` values for result in this (fallthrough) state.
                for _ in 0..n_rets {
                    self.stack.push(Value::Top);
                }
                ret
            }
        }
    }

    /// Pop N values.
    pub fn popn(&mut self, n: usize) {
        assert!(self.stack.len() >= n);
        self.stack.truncate(self.stack.len() - n);
    }

    /// Push N copies of the given value.
    pub fn pushn(&mut self, n: usize, val: Value) {
        self.stack.resize(self.stack.len() + n, val);
    }
}
