//! Partial evaluation.

/* TODO:

- if/else scheme
- breaks out of blocks, and forward-edge state merging
- loops and per-state loop unrolling
- inlining
- "memory renaming": connecting symbolic ops through the operand-stack
  memory region
- more general memory-region handling: symbolic but unique
  (non-escaped) pointers, stack, operand-stack region, ...

*/

use crate::directive::Directive;
use crate::image::Image;
use crate::intrinsics::Intrinsics;
use crate::state::State;
use crate::value::Value;
use std::collections::HashMap;
use walrus::{
    ir::Instr, ir::InstrSeq, ir::InstrSeqId, ActiveData, ActiveDataLocation, DataKind, Function,
    FunctionBuilder, FunctionKind, LocalFunction, Module, ModuleTypes,
};

/// Partially evaluates according to the given directives.
pub fn partially_evaluate(
    module: &mut Module,
    im: &mut Image,
    directives: &[Directive],
) -> anyhow::Result<()> {
    let intrinsics = Intrinsics::find(module);
    let mut mem_updates = HashMap::new();
    for directive in directives {
        if let Some(idx) = partially_evaluate_func(module, im, &intrinsics, directive)? {
            // Update memory image.
            mem_updates.insert(directive.func_index_out_addr, idx);
        }
    }

    // Update memory: regenerate as one large data segment for heap 0 from the image.
    if let Some(main_heap_image) = im
        .main_heap
        .and_then(|main_heap| im.memories.get_mut(&main_heap))
    {
        for (addr, value) in mem_updates {
            let addr = addr as usize;
            if (addr + 4) > main_heap_image.image.len() {
                log::warn!(
                    "Cannot store function index for new function: address {:x} is out-of-bounds",
                    addr
                );
                continue;
            }
            main_heap_image.image[addr..(addr + 4)].copy_from_slice(&value.to_le_bytes()[..]);
        }
        Ok(())
    } else {
        anyhow::bail!("No image for main heap: cannot update");
    }
}

fn partially_evaluate_func(
    module: &mut Module,
    image: &Image,
    intrinsics: &Intrinsics,
    directive: &Directive,
) -> anyhow::Result<Option<u32>> {
    let lf = match &module.funcs.get(directive.func).kind {
        FunctionKind::Local(lf) => lf,
        _ => return Ok(None),
    };
    let (param_tys, result_tys) = module.types.params_results(lf.ty());
    let param_tys = param_tys.to_vec();
    let result_tys = result_tys.to_vec();

    let mut builder = FunctionBuilder::new(&mut module.types, &param_tys[..], &result_tys[..]);
    let state = State::initial(
        module,
        image,
        directive.func,
        directive.const_params.clone(),
    );
    let from_seq = OrigSeqId(lf.entry_block());
    let into_seq = OutSeqId(builder.func_body_id());
    let mut ctx = EvalCtx {
        generic_fn: lf,
        builder: &mut builder,
        intrinsics,
        image,
        seq_map: HashMap::new(),
        tys: &module.types,
    };

    let exit_state = ctx.eval_seq(state, from_seq, into_seq)?;
    // TODO: implicit return if `exit_state.fallthrough` is `Some`.
    drop(exit_state);

    let specialized_fn = builder.finish(lf.args.clone(), &mut module.funcs);

    Ok(Some(specialized_fn.index() as u32))
}

/// Newtype around seq IDs in original (generic) function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct OrigSeqId(InstrSeqId);
/// Newtype around seq IDs in output (specialized) function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct OutSeqId(InstrSeqId);

struct EvalCtx<'a> {
    generic_fn: &'a LocalFunction,
    builder: &'a mut FunctionBuilder,
    intrinsics: &'a Intrinsics,
    image: &'a Image,
    /// Map from seq ID in original function to specialized function.
    seq_map: HashMap<OrigSeqId, OutSeqId>,
    tys: &'a ModuleTypes,
}

/// Evaluation result.
#[derive(Clone, Debug)]
struct EvalResult {
    /// The output state for fallthrough. `None` if unreachable.
    fallthrough: Option<State>,
    /// Any taken-edge states.
    taken: HashMap<OrigSeqId, State>,
}

impl EvalResult {
    /// Returns `true` if merged-into seq was branched to.
    fn merge_subblock(&mut self, sub_seq: EvalResult, this_seq: OrigSeqId) -> bool {
        let state = self
            .fallthrough
            .as_mut()
            .expect("Cannot merge into unreachable state");
        if let Some(fallthrough) = sub_seq.fallthrough {
            state.meet_with(&fallthrough);
        }
        let mut this_seq_taken = false;
        for (taken_target, taken_state) in sub_seq.taken {
            if taken_target == this_seq {
                this_seq_taken = true;
                state.meet_with(&taken_state);
            } else if let Some(our_taken_state) = self.taken.get_mut(&taken_target) {
                our_taken_state.meet_with(&taken_state);
            } else {
                self.taken.insert(taken_target, taken_state);
            }
        }
        this_seq_taken
    }
}

impl<'a> EvalCtx<'a> {
    fn eval_seq(
        &mut self,
        state: State,
        from_seq: OrigSeqId,
        into_seq: OutSeqId,
    ) -> anyhow::Result<EvalResult> {
        self.seq_map.insert(from_seq, into_seq);

        let mut result = EvalResult {
            fallthrough: Some(state),
            taken: HashMap::new(),
        };

        for (instr, _) in &self.generic_fn.block(from_seq.0).instrs {
            match instr {
                Instr::Block(b) => {
                    // Create a new output seq and recursively eval.
                    let ty = self.generic_fn.block(b.seq).ty;
                    let sub_from_seq = OrigSeqId(b.seq);
                    let sub_into_seq = OutSeqId(self.builder.dangling_instr_seq(ty).id());
                    let sub_state = result
                        .fallthrough
                        .as_mut()
                        .unwrap()
                        .subblock_state(ty, &self.tys);
                    let sub_result = self.eval_seq(sub_state, sub_from_seq, sub_into_seq)?;
                    let block_used = result.merge_subblock(sub_result, from_seq);

                    if block_used {
                        // This `block` was actually branched to, so
                        // we need to keep it.
                        self.builder
                            .instr_seq(into_seq.0)
                            .instr(Instr::Block(walrus::ir::Block {
                                seq: sub_into_seq.0,
                            }));
                    } else {
                        // This `block` was not actually branched to,
                        // so we can remove it and just inline the
                        // resulting instructions directly (i.e.,
                        // discard the layering).
                        let instrs =
                            std::mem::take(self.builder.instr_seq(sub_into_seq.0).instrs_mut());
                        self.builder
                            .instr_seq(into_seq.0)
                            .instrs_mut()
                            .extend(instrs.into_iter());
                    }
                }
                Instr::Loop(l) => {
                    todo!("Loops are special")
                }
                Instr::Call(c) => {
                    // TODO: enqueue a directive for a specialized function?
                }
                Instr::CallIndirect(ci) => {}
                Instr::LocalGet(lg) => {}
                Instr::LocalSet(ls) => {}
                Instr::LocalTee(lt) => {}
                Instr::GlobalGet(gg) => {}
                Instr::GlobalSet(gs) => {}
                Instr::Const(c) => {}
                Instr::Binop(b) => {}
                Instr::Unop(u) => {}
                Instr::Select(s) => {}
                Instr::Unreachable(u) => {}
                Instr::Br(b) => {}
                Instr::BrIf(bi) => {}
                Instr::IfElse(ie) => {}
                Instr::BrTable(bt) => {}
                Instr::Drop(d) => {}
                Instr::Return(r) => {}
                Instr::MemorySize(ms) => {}
                Instr::MemoryGrow(mg) => {}
                Instr::Load(l) => {}
                Instr::Store(s) => {}
                _ => {
                    anyhow::bail!("Unsupported instruction: {:?}", instr);
                }
            }
        }

        self.seq_map.remove(&from_seq);

        Ok(result)
    }
}
