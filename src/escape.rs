use std::collections::HashSet;
/// Shadow-stack escape analysis optimization.
///
/// Determines whether pointers derived from global 0 (the shadow
/// stack used by LLVM-generated Wasm) are actually used anywhere (by
/// any operator other than another update to the stack pointer); if
/// not, deletes all global.sets and then does DCE to remove all
/// stack-pointer manipulation. This turns out to be useful when
/// weval'ing ICs when partial evaluation has removed all uses of some
/// dynamic on-stack data structure, like an opcode reader.
use waffle::cfg::CFGInfo;
use waffle::entity::EntityRef;
use waffle::pool::ListRef;
use waffle::{FunctionBody, Operator, Terminator, Type, Value, ValueDef};

enum EscapeAnalysisResult {
    Escapes,
    NonEscaping(HashSet<Value>),
}

fn shadow_stack_escapes(func: &FunctionBody, cfg: &CFGInfo) -> EscapeAnalysisResult {
    let mut tainted = HashSet::new();
    for (block_rpo, &block) in cfg.rpo.entries() {
        for &inst in &func.blocks[block].insts {
            match &func.values[inst] {
                &ValueDef::Operator(Operator::GlobalGet { global_index }, _, _)
                | &ValueDef::Operator(Operator::GlobalSet { global_index }, _, _)
                    if global_index.index() == 0 =>
                {
                    log::trace!("tainted because global.get/set: {}", inst);
                    tainted.insert(inst);
                }
                &ValueDef::Operator(Operator::I32Add, args, _)
                | &ValueDef::Operator(Operator::I32Sub, args, _) => {
                    let args = &func.arg_pool[args];
                    if args.iter().any(|arg| tainted.contains(arg)) {
                        log::trace!("tainted because of arg: {}", inst);
                        tainted.insert(inst);
                    }
                }
                &ValueDef::Operator(_, args, _) => {
                    let args = &func.arg_pool[args];
                    if args.iter().any(|arg| tainted.contains(arg)) {
                        log::trace!("shadow stack escape due to inst {}", inst);
                        return EscapeAnalysisResult::Escapes;
                    }
                }
                &ValueDef::PickOutput(val, _, _) | &ValueDef::Alias(val)
                    if tainted.contains(&val) =>
                {
                    log::trace!(
                        "taint on {} propagates to {} because of alias or pick",
                        val,
                        inst
                    );
                    tainted.insert(inst);
                }
                _ => {}
            }
        }

        match &func.blocks[block].terminator {
            &Terminator::CondBr { cond, .. } | &Terminator::Select { value: cond, .. } => {
                if tainted.contains(&cond) {
                    log::trace!(
                        "taint on input to conditional branch causes escape: {}",
                        cond
                    );
                    return EscapeAnalysisResult::Escapes;
                }
            }
            &Terminator::Return { ref values } => {
                if values.iter().any(|v| tainted.contains(v)) {
                    log::trace!("taint on return value causes escape");
                    return EscapeAnalysisResult::Escapes;
                }
            }
            _ => {}
        }
        let mut escaped_via_term = false;
        func.blocks[block].terminator.visit_targets(|target| {
            for (arg, (_, param)) in target
                .args
                .iter()
                .zip(func.blocks[target.block].params.iter())
            {
                if tainted.contains(arg) {
                    let target_rpo = cfg.rpo_pos[target.block].unwrap();
                    if target_rpo.index() <= block_rpo.index() {
                        log::trace!(
                            "taint traveling on backedge from {} to {} ({} to {}) causes escape",
                            arg,
                            param,
                            block,
                            target.block
                        );
                        escaped_via_term = true;
                    }
                    tainted.insert(*param);
                }
            }
        });
        if escaped_via_term {
            return EscapeAnalysisResult::Escapes;
        }
    }

    EscapeAnalysisResult::NonEscaping(tainted)
}

pub fn remove_shadow_stack_if_non_escaping(func: &mut FunctionBody, cfg: &CFGInfo) {
    if let EscapeAnalysisResult::NonEscaping(values_to_remove) = shadow_stack_escapes(func, &cfg) {
        log::trace!("removing shadow stack operations: {:?}", values_to_remove);
        let ty_u32 = func.type_pool.single(Type::I32);
        let const_zero = func.values.push(ValueDef::Operator(
            Operator::I32Const { value: 0 },
            ListRef::default(),
            ty_u32,
        ));
        func.blocks[func.entry].insts.push(const_zero);
        for block in func.blocks.values_mut() {
            block.insts.retain(|v| !values_to_remove.contains(v));
            block.terminator.update_targets(|target| {
                for arg in &mut target.args {
                    if values_to_remove.contains(arg) {
                        assert_eq!(func.values[*arg].ty(&func.type_pool), Some(Type::I32));
                        *arg = const_zero;
                    }
                }
            });
        }
    }
}
