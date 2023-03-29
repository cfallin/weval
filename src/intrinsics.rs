//! Discovery of intrinsics.

use waffle::{ExportKind, Func, Module, Operator, Terminator, Type, ValueDef};

#[derive(Clone, Debug)]
pub struct Intrinsics {
    pub assume_const_memory: Option<Func>,
    pub make_symbolic_ptr: Option<Func>,
    pub flush_to_mem: Option<Func>,
    pub push_context: Option<Func>,
    pub pop_context: Option<Func>,
    pub update_context: Option<Func>,
    pub abort_specialization: Option<Func>,
    pub trace_line: Option<Func>,
    pub assert_const32: Option<Func>,
    pub assert_const_memory: Option<Func>,
    pub switch_value: Option<Func>,
    pub switch_default: Option<Func>,
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
            make_symbolic_ptr: find_exported_func(
                module,
                "weval.make.symbolic.ptr",
                &[Type::I32],
                &[Type::I32],
            ),
            flush_to_mem: find_exported_func(module, "weval.flush.to.mem", &[], &[]),
            push_context: find_exported_func(module, "weval.push.context", &[Type::I32], &[]),
            pop_context: find_exported_func(module, "weval.pop.context", &[], &[]),
            update_context: find_exported_func(module, "weval.update.context", &[Type::I32], &[]),
            abort_specialization: find_exported_func(
                module,
                "weval.abort.specialization",
                &[Type::I32, Type::I32],
                &[],
            ),
            trace_line: find_exported_func(module, "weval.trace.line", &[Type::I32], &[]),
            assert_const32: find_exported_func(
                module,
                "weval.assert.const32",
                &[Type::I32, Type::I32],
                &[],
            ),
            assert_const_memory: find_exported_func(
                module,
                "weval.assert.const.memory",
                &[Type::I32, Type::I32],
                &[],
            ),
            switch_value: find_exported_func(
                module,
                "weval.switch.value",
                &[Type::I32, Type::I32],
                &[Type::I32],
            ),
            switch_default: find_exported_func(
                module,
                "weval.switch.default",
                &[Type::I32],
                &[Type::I32],
            ),
        }
    }
}

fn export_sig_matches(module: &Module, f: Func, in_tys: &[Type], out_tys: &[Type]) -> bool {
    let sig = module.funcs[f].sig();
    let sig = &module.signatures[sig];
    &sig.params[..] == in_tys && &sig.returns[..] == out_tys
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
        .find(|ex| &ex.name == name)
        .and_then(|ex| match &ex.kind {
            &ExportKind::Func(f) if export_sig_matches(module, f, in_tys, out_tys) => Some(f),
            _ => None,
        })
}

pub fn find_global_data_by_exported_func(module: &Module, name: &str) -> Option<u32> {
    let f = find_exported_func(module, name, &[], &[Type::I32])?;
    let mut body = module.funcs[f].clone();
    body.parse(module).unwrap();
    let body = body.body()?;

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
