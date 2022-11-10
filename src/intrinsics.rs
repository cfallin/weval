//! Discovery of intrinsics.

use waffle::{ExportKind, Func, FuncDecl, Module, Operator, Terminator, Type, ValueDef};

#[derive(Clone, Debug)]
pub struct Intrinsics {
    pub assume_const_memory: Option<Func>,
    pub loop_header: Option<Func>,
    pub loop_pc32_update: Option<Func>,
}

impl Intrinsics {
    pub fn find(module: &Module) -> Intrinsics {
        Intrinsics {
            assume_const_memory: find_exported_func(
                module,
                "weval.assume.const.memory",
                &[Type::I32],
                &[Type::I32],
            ),
            // TODO: return a unique token from the header intrinsic
            // to pass into pc32_update
            loop_header: find_exported_func(module, "weval.loop.header", &[], &[]),
            loop_pc32_update: find_exported_func(
                module,
                "weval.loop.pc32.update",
                &[Type::I32],
                &[Type::I32],
            ),
        }
    }
}

fn export_sig_matches(module: &Module, f: Func, in_tys: &[Type], out_tys: &[Type]) -> bool {
    let sig = module.func(f).sig();
    let sig = module.signature(sig);
    &sig.params[..] == in_tys && &sig.returns[..] == out_tys
}

pub fn find_exported_func(
    module: &Module,
    name: &str,
    in_tys: &[Type],
    out_tys: &[Type],
) -> Option<Func> {
    module
        .exports()
        .find(|ex| &ex.name == name)
        .and_then(|ex| match &ex.kind {
            &ExportKind::Func(f) if export_sig_matches(module, f, in_tys, out_tys) => Some(f),
            _ => None,
        })
}

pub fn find_global_data_by_exported_func(module: &Module, name: &str) -> Option<u32> {
    let f = find_exported_func(module, name, &[], &[Type::I32])?;
    let body = match module.func(f) {
        FuncDecl::Body(_, body) => body,
        _ => return None,
    };
    // Find the `return`; its value should be an I32Const.
    match &body.blocks[body.entry].terminator {
        Terminator::Return { values } => {
            assert_eq!(values.len(), 1);
            match &body.values[values[0]] {
                ValueDef::Operator(Operator::I32Const { value }, _, _) => Some(*value as u32),
                _ => None,
            }
        }
        Terminator::Br { target } => {
            assert_eq!(target.args.len(), 1);
            let val = match &body.values[target.args[0]] {
                ValueDef::Operator(Operator::I32Const { value }, _, _) => *value as u32,
                _ => return None,
            };
            match &body.blocks[target.block].terminator {
                Terminator::Return { values }
                    if values.len() == 1 && values[0] == body.blocks[target.block].params[0].1 =>
                {
                    Some(val)
                }
                _ => None,
            }
        }
        _ => None,
    }
}
