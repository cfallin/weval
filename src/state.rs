//! State tracking.

use crate::image::Image;
use crate::value::{Value, WasmVal};
use std::collections::BTreeMap;
use walrus::{FunctionId, FunctionKind, GlobalId, LocalId, Module};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    /// Memory overlay. We store only aligned u32s here.
    mem_overlay: BTreeMap<u32, Value>,
    /// Global values.
    globals: BTreeMap<GlobalId, Value>,
    /// Local values.
    locals: BTreeMap<LocalId, Value>,
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
        }
    }

    pub fn meet_with(&mut self, other: &State) {
        map_meet_with(&mut self.mem_overlay, &other.mem_overlay);
        map_meet_with(&mut self.globals, &other.globals);
        map_meet_with(&mut self.locals, &other.locals);
    }
}
