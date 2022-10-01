//! State tracking.

use crate::image::Image;
use crate::value::{Value, WasmVal};
use std::collections::BTreeMap;
use walrus::{FunctionId, FunctionKind, GlobalId, InstrSeqType, LocalId, Module, ModuleTypes};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    /// Memory overlay. We store only aligned u32s here.
    mem_overlay: BTreeMap<u32, Value>,
    /// Global values.
    globals: BTreeMap<GlobalId, Value>,
    /// Local values.
    locals: BTreeMap<LocalId, Value>,
    /// Operand stack.
    stack: Vec<Value>,
}

fn map_meet_with<K: PartialEq + Eq + PartialOrd + Ord + Copy>(
    this: &mut BTreeMap<K, Value>,
    other: &BTreeMap<K, Value>,
) {
    for (k, val) in this.iter_mut() {
        if let Some(other_val) = other.get(k) {
            *val = Value::meet(*val, *other_val);
        } else {
            *val = Value::Runtime;
        }
    }
    for other_k in other.keys() {
        if !this.contains_key(other_k) {
            this.insert(*other_k, Value::Runtime);
        }
    }
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
            .map(|(id, val)| (*id, Value::Concrete(*val)))
            .collect();

        State {
            mem_overlay: BTreeMap::new(),
            globals,
            locals,
            stack: vec![],
        }
    }

    pub fn meet_with(&mut self, other: &State) {
        map_meet_with(&mut self.mem_overlay, &other.mem_overlay);
        map_meet_with(&mut self.globals, &other.globals);
        map_meet_with(&mut self.locals, &other.locals);
        assert_eq!(self.stack.len(), other.stack.len());
        for (this_stack, other_stack) in self.stack.iter_mut().zip(other.stack.iter()) {
            *this_stack = Value::meet(*this_stack, *other_stack);
        }
    }

    pub fn enter_block(&mut self, ty: InstrSeqType, tys: &ModuleTypes) -> State {
        match ty {
            InstrSeqType::Simple(_) => self.clone(),
            InstrSeqType::MultiValue(ty) => {
                let ty = tys.get(ty);
                let n_params = ty.params().len();
                let mut ret = self.clone();
                // Split off params.
                let param_vals = self.stack.split_off(self.stack.len() - n_params);
                ret.stack = param_vals;
                ret
            }
        }
    }

    pub fn leave_block(&mut self, ty: InstrSeqType, tys: &ModuleTypes) {
        match ty {
            InstrSeqType::Simple(None) => {}
            InstrSeqType::Simple(Some(ret_ty)) => {
                assert_eq!(self.stack.len(), 1);
            }
            InstrSeqType::MultiValue(ty) => {
                let ty = tys.get(ty);
                let n_rets = ty.results().len();
                assert_eq!(n_rets, self.stack.len());
            }
        }
    }
}
