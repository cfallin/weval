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
    let mark_used = |used: &mut FxHashSet<Value>, mut arg: Value| -> bool {
        let mut changed = false;
        changed |= used.insert(arg);
        while let ValueDef::Alias(orig) = &func.values[arg] {
            arg = *orig;
            changed |= used.insert(arg);
        }
        changed
    };

    log::trace!("DCE: scanning {}", block);
    let mut changed = false;

    let mut target_idx = 0;
    func.blocks[block].terminator.visit_targets(|target| {
        log::trace!(" -> considering succ {}", target.block);
        let succ_params = &func.blocks[target.block].params;
        for (i, (&arg, &(_, param))) in target.args.iter().zip(succ_params.iter()).enumerate() {
            if used.contains(&param) {
                log::trace!(
                    "  -> succ blockparam {} is used; marking arg {} used",
                    param,
                    arg
                );
                changed |= mark_used(used, arg);
                used_branch_args.insert((block, target_idx, i));
            }
        }
        target_idx += 1;
    });
    match &func.blocks[block].terminator {
        Terminator::CondBr { cond: value, .. } | Terminator::Select { value, .. } => {
            log::trace!(" -> marking branch input {} used", value);
            changed |= mark_used(used, *value);
        }
        Terminator::Return { values } => {
            for &value in values {
                log::trace!(" -> marking return value {} used", value);
                changed |= mark_used(used, value);
            }
        }
        Terminator::Br { .. } | Terminator::Unreachable | Terminator::None => {}
    }

    for &inst in func.blocks[block].insts.iter().rev() {
        match &func.values[inst] {
            ValueDef::BlockParam(..) | ValueDef::Alias(..) => {
                // Nothing.
            }
            ValueDef::PickOutput(value, ..) => {
                if used.contains(&inst) {
                    log::trace!(" -> marking pick-output src {} used", value);
                    changed |= mark_used(used, *value);
                }
            }
            ValueDef::Trace(_, args) => {
                for &arg in &func.arg_pool[*args] {
                    log::trace!(" -> marking trace arg {} used", arg);
                    changed |= mark_used(used, arg);
                }
            }
            ValueDef::Operator(op, args, _) => {
                if !op.is_pure() || used.contains(&inst) {
                    for &arg in &func.arg_pool[*args] {
                        log::trace!(" -> marking arg {} used", arg);
                        changed |= mark_used(used, arg);
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
            log::trace!("removing unreachable block {}", block);
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
        log::trace!("done with all blocks; changed = {}", changed);
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
