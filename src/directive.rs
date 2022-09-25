//! Partial-evaluation directives.

use crate::image::Image;
use crate::value::Value;
use walrus::{FunctionId, Module};

pub struct Directive {
    /// Evaluate the given function.
    pub func: FunctionId,
    /// Evaluate with the given parameter values fixed.
    pub const_params: Vec<Value>,
    /// Place the ID of the resulting specialized function at the
    /// given address in memory.
    pub func_index_out_addr: u32,
}

pub fn collect(module: &Module, im: &Image) -> anyhow::Result<Vec<Directive>> {
    Ok(vec![])
}
