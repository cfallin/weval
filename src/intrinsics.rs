//! Discovery of intrinsics.

use walrus::{ExportItem, FunctionId, FunctionKind, Module, ValType};

pub struct Intrinsics {
    pub assume_const_memory: Option<FunctionId>,
    pub loop_pc32: Option<FunctionId>,
    pub loop_pc64: Option<FunctionId>,
}

impl Intrinsics {
    pub fn find(module: &Module) -> Intrinsics {
        Intrinsics {
            assume_const_memory: find_exported_func(
                module,
                "weval.assume.const.memory",
                &[ValType::I32],
                &[ValType::I32],
            ),
            loop_pc32: find_exported_func(
                module,
                "weval.loop.pc32",
                &[ValType::I32],
                &[ValType::I32],
            ),
            loop_pc64: find_exported_func(
                module,
                "weval.loop.pc64",
                &[ValType::I64],
                &[ValType::I64],
            ),
        }
    }
}

fn export_sig_matches(
    module: &Module,
    f: FunctionId,
    in_tys: &[ValType],
    out_tys: &[ValType],
) -> bool {
    let sig_ty = module.funcs.get(f).ty();
    let (params, results) = module.types.params_results(sig_ty);
    params == in_tys && results == out_tys
}

pub fn find_exported_func(
    module: &Module,
    name: &str,
    in_tys: &[ValType],
    out_tys: &[ValType],
) -> Option<FunctionId> {
    module
        .exports
        .iter()
        .find(|ex| &ex.name == name)
        .and_then(|ex| match &ex.item {
            &ExportItem::Function(f) if export_sig_matches(module, f, in_tys, out_tys) => Some(f),
            _ => None,
        })
}

pub fn find_global_data_by_exported_func(module: &Module, name: &str) -> Option<u32> {
    let f = find_exported_func(module, name, &[], &[ValType::I32])?;
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
