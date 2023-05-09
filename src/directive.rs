//! Partial-evaluation directives.

use crate::image::Image;
use crate::intrinsics::find_global_data_by_exported_func;
use crate::value::{AbstractValue, ValueTags, WasmVal};
use waffle::{Func, Memory, Module};

#[derive(Clone, Debug)]
pub struct Directive {
    /// Evaluate the given function.
    pub func: Func,
    /// Evaluate with the given parameter values fixed.
    pub const_params: Vec<AbstractValue>,
    /// Place the ID of the resulting specialized function at the
    /// given address in memory.
    pub func_index_out_addr: u32,
}

pub fn collect(module: &Module, im: &mut Image) -> anyhow::Result<Vec<Directive>> {
    // Is there a function called "weval.pending.head"?  If so, is the
    // function body a simple constant? This provides the address of a
    // doubly-linked list; we process requests and unlink them.

    let pending_head_addr = match find_global_data_by_exported_func(module, "weval.pending.head") {
        Some(x) => x,
        _ => {
            return Ok(vec![]);
        }
    };

    let heap = match im.main_heap {
        Some(heap) => heap,
        None => return Ok(vec![]),
    };

    let mut head = im.read_u32(heap, pending_head_addr)?;
    let mut directives = vec![];
    while head != 0 {
        directives.push(decode_weval_req(im, heap, head)?);
        let next = im.read_u32(heap, head)?;
        let prev = im.read_u32(heap, head + 4)?;
        if next != 0 {
            im.write_u32(heap, next + 4, prev)?;
        }
        if prev != 0 {
            im.write_u32(heap, prev, next)?;
        } else {
            im.write_u32(heap, pending_head_addr, next)?;
        }
        im.write_u32(heap, head, 0)?;
        im.write_u32(heap, head + 4, 0)?;
        head = next;
    }

    Ok(directives)
}

fn decode_weval_req(im: &Image, heap: Memory, head: u32) -> anyhow::Result<Directive> {
    let func_table_index = im.read_u32(heap, head + 8)?;
    let func = im.func_ptr(func_table_index)?;
    let mut arg_ptr = im.read_u32(heap, head + 12)?;
    let nargs = im.read_u32(heap, head + 16)?;
    let func_index_out_addr = im.read_u32(heap, head + 20)?;

    let mut const_params = vec![];
    for _ in 0..nargs {
        let is_specialized = im.read_u32(heap, arg_ptr)?;
        let ty = im.read_u32(heap, arg_ptr + 4)?;
        let tags = ValueTags::default();
        let value = if is_specialized != 0 {
            match ty {
                0 => AbstractValue::Concrete(WasmVal::I32(im.read_u32(heap, arg_ptr + 8)?), tags),
                1 => AbstractValue::Concrete(WasmVal::I64(im.read_u64(heap, arg_ptr + 8)?), tags),
                2 => AbstractValue::Concrete(WasmVal::F32(im.read_u32(heap, arg_ptr + 8)?), tags),
                3 => AbstractValue::Concrete(WasmVal::F64(im.read_u64(heap, arg_ptr + 8)?), tags),
                _ => anyhow::bail!("Invalid type: {}", ty),
            }
        } else {
            AbstractValue::Runtime(None, tags)
        };
        const_params.push(value);
        arg_ptr += 16;
    }

    Ok(Directive {
        func,
        const_params,
        func_index_out_addr,
    })
}
