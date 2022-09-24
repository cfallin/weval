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
use walrus::Module;

pub fn partially_evaluate(
    module: &mut Module,
    im: &Image,
    directives: &[Directive],
) -> anyhow::Result<()> {
    Ok(())
}
