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
    FunctionBuilder, FunctionKind, LocalFunction, Module,
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
    let from_seq = lf.entry_block();
    let into_seq = builder.func_body_id();
    let mut ctx = EvalCtx {
        generic_fn: lf,
        builder: &mut builder,
        intrinsics,
        image,
        seq_map: HashMap::new(),
        inbound_state: HashMap::new(),
    };

    let (exit_state, exit_op_stack) = ctx.eval_seq(&state, from_seq, into_seq, &[])?;
    drop(exit_state);
    drop(exit_op_stack);

    let specialized_fn = builder.finish(lf.args.clone(), &mut module.funcs);

    Ok(Some(specialized_fn.index() as u32))
}

struct EvalCtx<'a> {
    generic_fn: &'a LocalFunction,
    builder: &'a mut FunctionBuilder,
    intrinsics: &'a Intrinsics,
    image: &'a Image,
    seq_map: HashMap<InstrSeqId, InstrSeqId>,
    inbound_state: HashMap<InstrSeqId, Vec<State>>,
}

impl<'a> EvalCtx<'a> {
    fn eval_seq(
        &mut self,
        state: &State,
        from_seq: InstrSeqId,
        into_seq: InstrSeqId,
        stack: &[Value],
    ) -> anyhow::Result<(State, Vec<Value>)> {
        let mut state = state.clone();
        if let Some(states) = self.inbound_state.remove(&from_seq) {
            for s in &states {
                state.meet_with(s);
            }
        }
        self.seq_map.insert(from_seq, into_seq);

        let mut stack = stack.iter().cloned().collect::<Vec<_>>();
        for (instr, _) in &self.generic_fn.block(from_seq).instrs {
            match instr {
                Instr::Block(b) => {}
                Instr::Loop(l) => {}
                Instr::Call(c) => {}
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
        // TODO: trim stack according to block's return type
        Ok((state, stack))
    }
}
