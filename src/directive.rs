//! Partial-evaluation directives.

use crate::image::Image;
use crate::intrinsics::find_global_data_by_exported_func;
use waffle::{Func, Memory, Module};

#[derive(Clone, Debug)]
pub struct Directive {
    /// Evaluate the given function.
    pub func: Func,
    /// Evaluate with the given func_ctx.
    pub func_ctx: u64,
    /// Evaluate with the given pc_ctx.
    pub pc_ctx: u64,
    /// Place the ID of the resulting specialized function at the
    /// given address in memory.
    pub func_index_out_addr: u32,
}

pub fn collect(module: &Module, im: &mut Image) -> anyhow::Result<Vec<Directive>> {
    // Is there a function called "weval.pending.head" and one called
    // "weval.freelist.head"? If so, are both function bodies simple
    // constants? These provide the addresses of two singly-linked
    // lists; we move requests from one to the other and accumulate
    // directive information.

    let (pending_head_addr, freelist_head_addr) = match (
        find_global_data_by_exported_func(module, "weval.pending.head"),
        find_global_data_by_exported_func(module, "weval.freelist.head"),
    ) {
        (Some(x), Some(y)) => (x, y),
        _ => {
            return Ok(vec![]);
        }
    };

    let heap = match im.main_heap {
        Some(heap) => heap,
        None => return Ok(vec![]),
    };

    let mut head = im.read_u32(heap, pending_head_addr)?;
    let mut freelist = im.read_u32(heap, freelist_head_addr)?;
    let mut directives = vec![];
    while head != 0 {
        directives.push(decode_weval_req(im, heap, head)?);
        let next = im.read_u32(heap, head)?;
        im.write_u32(heap, head, freelist)?;
        im.write_u32(heap, freelist_head_addr, head)?;
        freelist = head;
        head = next;
    }

    Ok(directives)
}

fn decode_weval_req(im: &Image, heap: Memory, head: u32) -> anyhow::Result<Directive> {
    let func_table_index = im.read_u32(heap, head + 4)?;
    let func = im.func_ptr(func_table_index)?;
    let func_ctx = im.read_u64(heap, head + 8)?;
    let pc_ctx = im.read_u64(heap, head + 16)?;
    let func_index_out_addr = im.read_u32(heap, head + 24)?;

    Ok(Directive {
        func,
        func_ctx,
        pc_ctx,
        func_index_out_addr,
    })
}
