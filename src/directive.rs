//! Partial-evaluation directives.

use crate::image::Image;
use crate::intrinsics::find_global_data_by_exported_func;
use crate::value::{Value, ValueTags, WasmVal};
use walrus::{FunctionId, Module};

#[derive(Clone, Debug)]
pub struct Directive {
    /// Evaluate the given function.
    pub func: FunctionId,
    /// Evaluate with the given parameter values fixed.
    pub const_params: Vec<Value>,
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

    let mut head = im.read_heap_u32(pending_head_addr)?;
    let mut freelist = im.read_heap_u32(freelist_head_addr)?;
    let mut directives = vec![];
    while head != 0 {
        directives.push(decode_weval_req(module, im, head)?);
        let next = im.read_heap_u32(head)?;
        im.write_heap_u32(head, freelist)?;
        im.write_heap_u32(freelist_head_addr, head)?;
        freelist = head;
        head = next;
    }

    Ok(directives)
}

fn decode_weval_req(module: &Module, im: &Image, head: u32) -> anyhow::Result<Directive> {
    let func_index = im.read_heap_u32(head + 4)?;
    let mut arg_ptr = im.read_heap_u32(head + 8)?;
    let nargs = im.read_heap_u32(head + 12)?;
    let func_index_out_addr = im.read_heap_u32(head + 16)?;

    let mut const_params = vec![];
    for i in 0..nargs {
        let is_specialized = im.read_heap_u32(arg_ptr)?;
        let ty = im.read_heap_u32(arg_ptr + 4)?;
        let tags = ValueTags::default();
        let value = if is_specialized != 0 {
            match ty {
                0 => Value::Concrete(WasmVal::I32(im.read_heap_u32(arg_ptr + 8)?), tags),
                1 => Value::Concrete(WasmVal::I64(im.read_heap_u64(arg_ptr + 8)?), tags),
                2 => Value::Concrete(WasmVal::F32(im.read_heap_u32(arg_ptr + 8)?), tags),
                3 => Value::Concrete(WasmVal::F64(im.read_heap_u64(arg_ptr + 8)?), tags),
                _ => anyhow::bail!("Invalid type"),
            }
        } else {
            Value::Runtime(tags)
        };
        const_params.push(value);
        arg_ptr += 16;
    }

    let func = module
        .functions()
        .find(|f| f.id().index() == func_index as usize)
        .ok_or_else(|| anyhow::anyhow!("Cannot find function index {}", func_index))?
        .id();

    Ok(Directive {
        func,
        const_params,
        func_index_out_addr,
    })
}
