//! Partial-evaluation directives.

use crate::image::Image;
use crate::intrinsics::find_global_data_by_exported_func;
use crate::value::{AbstractValue, WasmVal};
use std::sync::Arc;
use waffle::{Func, Memory, Module};

#[derive(Clone, Debug)]
pub struct Directive {
    /// Evaluate the given function.
    pub func: Func,
    /// Evaluate with the given parameter values fixed.
    pub const_params: Vec<AbstractValue>,
    /// Evaluate with the given symbolic memory buffers.
    pub const_memory: Vec<Option<MemoryBuffer>>,
    /// Place the ID of the resulting specialized function at the
    /// given address in memory.
    pub func_index_out_addr: u32,
}

/// A "symbolic pointer" backing buffer: if we are specializing a
/// function assuming a given argument which is a pointer has fixed
/// *contents* (but not necessarily a constant pointer value), this
/// allows us to give the backing data directly.
#[derive(Clone, Debug)]
pub struct MemoryBuffer {
    /// The bytes in memory at this pointer.
    data: Arc<Vec<u8>>,
}

impl MemoryBuffer {
    pub fn read_size(&self, offset: u32, size: u32) -> anyhow::Result<u64> {
        let offset = usize::try_from(offset).unwrap();
        let size = usize::try_from(size).unwrap();
        if offset + size >= self.data.len() {
            anyhow::bail!("Out of bounds");
        }
        let slice = &self.data[offset..(offset + size)];
        Ok(match size {
            1 => u64::from(slice[0]),
            2 => u64::from(u16::from_le_bytes([slice[0], slice[1]])),
            4 => u64::from(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])),
            8 => u64::from_le_bytes([
                slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
            ]),
            _ => unreachable!(),
        })
    }
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
    let mut const_memory = vec![];
    for _ in 0..nargs {
        let is_specialized = im.read_u32(heap, arg_ptr)?;
        let ty = im.read_u32(heap, arg_ptr + 4)?;
        let (value, mem) = if is_specialized != 0 {
            match ty {
                0 => (
                    AbstractValue::Concrete(WasmVal::I32(im.read_u32(heap, arg_ptr + 8)?)),
                    None,
                ),
                1 => (
                    AbstractValue::Concrete(WasmVal::I64(im.read_u64(heap, arg_ptr + 8)?)),
                    None,
                ),
                2 => (
                    AbstractValue::Concrete(WasmVal::F32(im.read_u32(heap, arg_ptr + 8)?)),
                    None,
                ),
                3 => (
                    AbstractValue::Concrete(WasmVal::F64(im.read_u64(heap, arg_ptr + 8)?)),
                    None,
                ),
                4 => {
                    let ptr = im.read_u32(heap, arg_ptr + 8)?;
                    let len = im.read_u32(heap, arg_ptr + 12)?;
                    let data = MemoryBuffer {
                        data: Arc::new(im.read_slice(heap, ptr, len)?.to_vec()),
                    };
                    (AbstractValue::Runtime(None), Some(data))
                }
                _ => anyhow::bail!("Invalid type: {}", ty),
            }
        } else {
            (AbstractValue::Runtime(None), None)
        };
        const_params.push(value);
        const_memory.push(mem);
        arg_ptr += 16;
    }

    Ok(Directive {
        func,
        const_params,
        const_memory,
        func_index_out_addr,
    })
}
