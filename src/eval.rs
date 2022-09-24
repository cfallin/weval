//! Partial evaluation.

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
