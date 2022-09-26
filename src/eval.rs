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
use std::collections::HashMap;
use walrus::{
    ir::InstrSeq, ir::InstrSeqId, ActiveData, ActiveDataLocation, DataKind, Function,
    FunctionBuilder, FunctionKind, Module,
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
        if let Some(idx) = partially_evaluate_func(module, im, &intrinsics, directive) {
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
    im: &Image,
    intrinsics: &Intrinsics,
    directive: &Directive,
) -> Option<u32> {
    let lf = match &module.funcs.get(directive.func).kind {
        FunctionKind::Local(lf) => lf,
        _ => return None,
    };
    let (param_tys, result_tys) = module.types.params_results(lf.ty());
    let param_tys = param_tys.to_vec();
    let result_tys = result_tys.to_vec();

    let mut builder = FunctionBuilder::new(&mut module.types, &param_tys[..], &result_tys[..]);
    let mut state = State::initial(module, im, directive.func, directive.const_params.clone());
    let into_seq = builder.func_body_id();
    let _exit_state = partially_evaluate_seq(
        im,
        intrinsics,
        &mut builder,
        &state,
        lf.entry_block(),
        into_seq,
    );

    None
}

fn partially_evaluate_seq(
    im: &Image,
    intrinsics: &Intrinsics,
    builder: &mut FunctionBuilder,
    state: &State,
    seq: InstrSeqId,
    into_seq: InstrSeqId,
) -> State {
    for i in 0..builder.instr_seq(seq).instrs().len() {
        let instr = builder.instr_seq(seq).instrs()[i].0.clone();
        builder.instr_seq(into_seq).instr(instr);
    }
    state.clone()
}
