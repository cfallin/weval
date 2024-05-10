//! Dead-code elimination pass.

use fxhash::FxHashSet;
use waffle::{cfg::CFGInfo, Block, FunctionBody, Terminator, Value, ValueDef};

/// Scan backwards over a block, marking as used the inputs to any
/// instruction that itself is used (or for a branch arg, for which
/// any target's corresponding blockparam is used). Returns `true` if
/// any changes occurred to the used-value set.
fn scan_block(
    func: &FunctionBody,
    block: Block,
    used: &mut FxHashSet<Value>,
    used_branch_args: &mut FxHashSet<(Block, usize, usize)>,
) -> bool {
    let mut changed = false;

    let mut target_idx = 0;
    func.blocks[block].terminator.visit_targets(|target| {
        let succ_params = &func.blocks[target.block].params;
        for (i, (&arg, &(_, param))) in target.args.iter().zip(succ_params.iter()).enumerate() {
            if used.contains(&param) {
                changed |= used.insert(arg);
                used_branch_args.insert((block, target_idx, i));
            }
        }
        target_idx += 1;
    });
    match &func.blocks[block].terminator {
        Terminator::CondBr { cond: value, .. } | Terminator::Select { value, .. } => {
            changed |= used.insert(*value);
        }
        Terminator::Return { values } => {
            for &value in values {
                changed |= used.insert(value);
            }
        }
        Terminator::Br { .. } | Terminator::Unreachable | Terminator::None => {}
    }

    for &inst in func.blocks[block].insts.iter().rev() {
        match &func.values[inst] {
            ValueDef::BlockParam(..) => {
                // Nothing: blockparam-block arg linkages are handled
                // on the branch side above.
            }
            ValueDef::Alias(orig) => {
                if used.contains(&inst) {
                    changed |= used.insert(*orig);
                }
            }
            ValueDef::PickOutput(value, ..) => {
                if used.contains(&inst) {
                    changed |= used.insert(*value);
                }
            }
            ValueDef::Trace(_, args) => {
                for &arg in &func.arg_pool[*args] {
                    changed |= used.insert(arg);
                }
            }
            ValueDef::Operator(op, args, _) => {
                if !op.is_pure() || used.contains(&inst) {
                    for &arg in &func.arg_pool[*args] {
                        changed |= used.insert(arg);
                    }
                }
            }
            ValueDef::Placeholder(..) | ValueDef::None => {
                // Nothing.
            }
        }
    }

    changed
}

pub fn run(func: &mut FunctionBody, cfg: &CFGInfo) {
    // For any unreachable blocks, empty their contents and
    // terminators, and remove all blockparams (and there will then be
    // no targets with branch args to adjust because only an
    // unreachable block can branch to an unreachable block).
    for (block, block_def) in func.blocks.entries_mut() {
        if cfg.rpo_pos[block].is_none() {
            block_def.insts.clear();
            block_def.params.clear();
            block_def.terminator = Terminator::Unreachable;
        }
    }

    // Now compute value uses.
    let mut used = FxHashSet::default();
    let mut used_branch_args = FxHashSet::default();
    loop {
        let mut changed = false;
        for &block in cfg.rpo.values().rev() {
            changed |= scan_block(func, block, &mut used, &mut used_branch_args);
        }
        if !changed {
            break;
        }
    }

    // Now delete any values that aren't used from `insts`, `params`
    // and targets' `args`.
    for (block, block_def) in func.blocks.entries_mut() {
        block_def.params.retain(|(_ty, param)| used.contains(param));
        block_def.insts.retain(|inst| used.contains(inst));
        let mut target_idx = 0;
        block_def.terminator.update_targets(|target| {
            for i in (0..target.args.len()).rev() {
                if !used_branch_args.contains(&(block, target_idx, i)) {
                    target.args.remove(i);
                }
            }
            target_idx += 1;
        });
    }
}
