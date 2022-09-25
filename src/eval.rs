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
use std::collections::HashMap;
use walrus::{ActiveData, ActiveDataLocation, DataKind, Module};

/// Partially evaluates according to the given directives.
pub fn partially_evaluate(
    module: &mut Module,
    im: &mut Image,
    directives: &[Directive],
) -> anyhow::Result<()> {
    let mut mem_updates = HashMap::new();
    for directive in directives {
        if let Some(idx) = partially_evaluate_func(module, im, directive) {
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

fn partially_evaluate_func(module: &mut Module, im: &Image, directive: &Directive) -> Option<u32> {
    None
}
