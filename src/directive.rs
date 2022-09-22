//! Partial-evaluation directives.

use crate::heap::Summaries;
use walrus::{FunctionId, Module};

pub struct Directive {
    /// Evaluate the given function.
    func: FunctionId,
    /// Evaluate with the given parameter values fixed.
    const_params: Vec<(usize, i64)>,
    /// Place the ID of the resulting specialized function at the
    /// given address in memory.
    func_index_out_addr: u32,
}

pub fn collect(module: &Module, heaps: &Summaries) -> anyhow::Result<Vec<Directive>> {
    Ok(vec![])
}
