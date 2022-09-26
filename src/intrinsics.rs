//! Discovery of intrinsics.

use walrus::{ExportItem, FunctionId, FunctionKind, Module};

pub struct Intrinsics {
    assume_const: Option<FunctionId>,
    assume_const_memory: Option<FunctionId>,
}

impl Intrinsics {
    pub fn find(module: &Module) -> Intrinsics {
        Intrinsics {
            assume_const: find_exported_func(module, "weval.assume.const"),
            assume_const_memory: find_exported_func(module, "weval.assume.const.memory"),
        }
    }
}

pub fn find_exported_func(module: &Module, name: &str) -> Option<FunctionId> {
    module
        .exports
        .iter()
        .find(|ex| &ex.name == name)
        .and_then(|ex| match &ex.item {
            &ExportItem::Function(f) => Some(f),
            _ => None,
        })
}

pub fn find_global_data_by_exported_func(module: &Module, name: &str) -> Option<u32> {
    let f = find_exported_func(module, name)?;
    let lf = match &module.funcs.get(f).kind {
        FunctionKind::Local(lf) => lf,
        _ => return None,
    };
    let body = lf.block(lf.entry_block());
    if body.len() == 1 && body[0].0.is_const() {
        match body[0].0.unwrap_const().value {
            walrus::ir::Value::I32(i) => {
                return Some(i as u32);
            }
            _ => {}
        }
    }
    None
}
