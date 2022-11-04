//! State tracking.

use crate::image::Image;
use crate::value::{AbstractValue, ValueTags};
use std::collections::BTreeMap;
use waffle::{Func, FunctionBody, Global, Module, Value};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    /// Memory overlay. We store only aligned u32s here.
    pub mem_overlay: BTreeMap<u32, AbstractValue>,
    /// Global values.
    pub globals: BTreeMap<Global, AbstractValue>,
    /// Values of SSA `Value`s.
    pub values: BTreeMap<Value, AbstractValue>,
    /// Loop stack, with a known PC per loop level.
    pub loop_pcs: Vec<Option<u64>>,
}

fn map_meet_with<K: PartialEq + Eq + PartialOrd + Ord + Copy>(
    this: &mut BTreeMap<K, AbstractValue>,
    other: &BTreeMap<K, AbstractValue>,
) -> bool {
    let mut changed = false;
    for (k, val) in this.iter_mut() {
        if let Some(other_val) = other.get(k) {
            let met = AbstractValue::meet(*val, *other_val);
            changed |= met != *val;
            *val = met;
        } else {
            *val = AbstractValue::Top;
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
    pub fn initial(
        module: &Module,
        im: &Image,
        func: &FunctionBody,
        args: Vec<AbstractValue>,
    ) -> State {
        let mut values = BTreeMap::new();
        // Block params of first block get values of args.
        for ((_, arg), arg_val) in func.blocks[func.entry].params.iter().zip(args.iter()) {
            values.insert(*arg, arg_val.clone());
        }
        let globals = im
            .globals
            .iter()
            .map(|(id, val)| (*id, AbstractValue::Concrete(*val, ValueTags::default())))
            .collect();

        State {
            mem_overlay: BTreeMap::new(),
            globals,
            values,
            loop_pcs: vec![],
        }
    }

    pub fn meet_with(&mut self, other: &State) -> bool {
        let mut changed = false;
        changed |= map_meet_with(&mut self.mem_overlay, &other.mem_overlay);
        changed |= map_meet_with(&mut self.globals, &other.globals);
        changed |= map_meet_with(&mut self.values, &other.values);

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
}
