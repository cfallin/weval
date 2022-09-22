//! Static heap image summary.

use std::collections::HashMap;
use walrus::{MemoryId, Module};

#[derive(Clone, Debug)]
pub struct Summaries {
    heaps: HashMap<MemoryId, Summary>,
}

#[derive(Clone, Debug)]
pub struct Summary {
    image: Vec<u8>,
    len: usize,
}

pub fn build_summaries(module: &Module) -> anyhow::Result<Summaries> {
    Ok(Summaries {
        heaps: HashMap::default(),
    })
}
