//! Discovery of intrinsics.

use waffle::{ExportKind, Func, ImportKind, Module, Operator, Terminator, Type, ValueDef};

#[derive(Clone, Debug)]
pub struct Intrinsics {
    pub read_reg: Option<Func>,
    pub write_reg: Option<Func>,
    pub push_context: Option<Func>,
    pub pop_context: Option<Func>,
    pub update_context: Option<Func>,
    pub context_bucket: Option<Func>,
    pub abort_specialization: Option<Func>,
    pub trace_line: Option<Func>,
    pub assert_const32: Option<Func>,
    pub specialize_value: Option<Func>,
    pub print: Option<Func>,
    pub dispatch_point: Option<Func>,
    pub dispatch_point_get_func: Option<Func>,
    pub dispatch_point_set_func: Option<Func>,
}

impl Intrinsics {
    pub fn find(module: &Module) -> Intrinsics {
        Intrinsics {
            read_reg: find_imported_intrinsic(module, "read.reg", &[Type::I64], &[Type::I64]),
            write_reg: find_imported_intrinsic(module, "write.reg", &[Type::I64, Type::I64], &[]),
            push_context: find_imported_intrinsic(module, "push.context", &[Type::I32], &[]),
            pop_context: find_imported_intrinsic(module, "pop.context", &[], &[]),
            update_context: find_imported_intrinsic(module, "update.context", &[Type::I32], &[]),
            context_bucket: find_imported_intrinsic(module, "context.bucket", &[Type::I32], &[]),
            abort_specialization: find_imported_intrinsic(
                module,
                "abort.specialization",
                &[Type::I32, Type::I32],
                &[],
            ),
            trace_line: find_imported_intrinsic(module, "trace.line", &[Type::I32], &[]),
            assert_const32: find_imported_intrinsic(
                module,
                "assert.const32",
                &[Type::I32, Type::I32],
                &[],
            ),
            specialize_value: find_imported_intrinsic(
                module,
                "specialize.value",
                &[Type::I32, Type::I32, Type::I32],
                &[Type::I32],
            ),
            print: find_imported_intrinsic(
                module,
                "print",
                &[Type::I32, Type::I32, Type::I32],
                &[],
            ),
            dispatch_point: find_imported_intrinsic(
                module,
                "dispatch.point",
                &[Type::I32],
                &[Type::I32],
            ),
            dispatch_point_get_func: find_imported_intrinsic(
                module,
                "dispatch.point.get.func",
                &[Type::I32],
                &[Type::I32],
            ),
            dispatch_point_set_func: find_imported_intrinsic(
                module,
                "dispatch.point.set.func",
                &[Type::I32, Type::I32],
                &[],
            ),
        }
    }
}

fn sig_matches(module: &Module, f: Func, in_tys: &[Type], out_tys: &[Type]) -> bool {
    let sig = module.funcs[f].sig();
    let sig = &module.signatures[sig];
    &sig.params[..] == in_tys && &sig.returns[..] == out_tys
}

pub fn find_imported_intrinsic(
    module: &Module,
    name: &str,
    in_tys: &[Type],
    out_tys: &[Type],
) -> Option<Func> {
    module
        .imports
        .iter()
        .find(|im| im.module == "weval" && im.name == name)
        .and_then(|im| match &im.kind {
            &ImportKind::Func(f) if sig_matches(module, f, in_tys, out_tys) => Some(f),
            _ => None,
        })
}

pub fn find_exported_func(
    module: &Module,
    name: &str,
    in_tys: &[Type],
    out_tys: &[Type],
) -> Option<Func> {
    module
        .exports
        .iter()
        .find(|ex| ex.name == name)
        .and_then(|ex| match &ex.kind {
            &ExportKind::Func(f) if sig_matches(module, f, in_tys, out_tys) => Some(f),
            _ => None,
        })
}

pub fn find_global_data_by_exported_func(module: &Module, name: &str) -> Option<u32> {
    let f = find_exported_func(module, name, &[], &[Type::I32])?;
    let mut body = module.funcs[f].clone();
    body.parse(module).unwrap();
    let body = body.body()?;

    // Find the `return`; its value should be an I32Const or maybe an
    // I32ADd of a global (the GOT memory base) and an I32Const.

    let extract_const_value = |value| match &body.values[value] {
        ValueDef::Operator(Operator::I32Const { value }, _, _) => Some(*value as u32),
        ValueDef::Operator(Operator::I32Add, args, _) => {
            let args = &body.arg_pool[*args];
            match (&body.values[args[0]], &body.values[args[1]]) {
                (
                    ValueDef::Operator(Operator::GlobalGet { global_index }, _, _),
                    ValueDef::Operator(Operator::I32Const { value }, _, _),
                ) => match module.globals[*global_index].value {
                    Some(global_value) => Some((global_value as u32) + *value),
                    _ => None,
                },
                _ => None,
            }
        }
        _ => None,
    };

    match &body.blocks[body.entry].terminator {
        Terminator::Return { values } => {
            assert_eq!(values.len(), 1);
            extract_const_value(values[0])
        }
        Terminator::Br { target } => {
            assert_eq!(target.args.len(), 1);
            let val = extract_const_value(target.args[0])?;
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
