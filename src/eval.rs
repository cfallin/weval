//! Partial evaluation.

use crate::directive::Directive;
use crate::heap::Summaries;
use walrus::Module;

pub fn partially_evaluate(
    module: &mut Module,
    heaps: &Summaries,
    directives: &[Directive],
) -> anyhow::Result<()> {
    Ok(())
}
