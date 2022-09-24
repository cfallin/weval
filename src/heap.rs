//! Static heap image summary.

use std::collections::HashMap;
use walrus::{ActiveData, ActiveDataLocation, DataKind, Memory, MemoryId, Module};

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
        heaps: module
            .memories
            .iter()
            .flat_map(|mem| maybe_summarize(module, mem).map(|summary| (mem.id(), summary)))
            .collect(),
    })
}

fn maybe_summarize(module: &Module, mem: &Memory) -> Option<Summary> {
    const WASM_PAGE: usize = 1 << 16;
    let len = (mem.initial as usize) * WASM_PAGE;
    let mut image = vec![0; len];

    for &segment_id in &mem.data_segments {
        let segment = module.data.get(segment_id);
        match segment.kind {
            DataKind::Passive => continue,
            DataKind::Active(ActiveData {
                memory,
                location: ActiveDataLocation::Relative(..),
            }) => {
                return None;
            }
            DataKind::Active(ActiveData {
                memory,
                location: ActiveDataLocation::Absolute(offset),
            }) => {
                let offset = offset as usize;
                image[offset..(offset + segment.value.len())].copy_from_slice(&segment.value[..]);
            }
        }
    }

    Some(Summary { image, len })
}
