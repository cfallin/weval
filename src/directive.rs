//! Partial-evaluation directives.

use crate::image::Image;
use crate::intrinsics::find_global_data_by_exported_func;
use crate::value::{AbstractValue, WasmVal};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use waffle::{Func, Memory, Module};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Directive {
    /// User-given ID for the weval'd function.
    pub user_id: u32,
    /// Evaluate the given function.
    #[serde(skip)]
    pub func: Func,
    /// Evaluate with the given arguments, encoded as a bytestring.
    pub args: Vec<u8>,
    /// Place the ID of the resulting specialized function at the
    /// given address in memory.
    #[serde(skip)]
    pub func_index_out_addr: u32,
}

#[derive(Clone, Debug)]
pub struct DirectiveArgs {
    /// Evaluate with the given parameter values fixed.
    pub const_params: Vec<AbstractValue>,
    /// Evaluate with the given symbolic memory buffers.
    pub const_memory: Vec<Option<MemoryBuffer>>,
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

/// Saved directive, able to be reinjected later in a new context.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedDirective {
    /// User-defined weval site.
    pub user_id: u32,
    /// Serialized argument request string.
    pub args: Vec<u8>,
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
    log::trace!("head = {:#x}", head);
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
        log::trace!("head = {:#x}", head);
    }

    Ok(directives)
}

fn decode_weval_req(im: &Image, heap: Memory, head: u32) -> anyhow::Result<Directive> {
    let user_id = im.read_u32(heap, head + 8)?;
    let func_table_index = im.read_u32(heap, head + 12)?;
    let func = im.func_ptr(func_table_index)?;
    let arg_ptr = im.read_u32(heap, head + 16)?;
    let arg_len = im.read_u32(heap, head + 20)?;
    let func_index_out_addr = im.read_u32(heap, head + 24)?;
    let args = im.read_slice(heap, arg_ptr, arg_len)?.to_vec();
    Ok(Directive {
        user_id,
        func,
        args,
        func_index_out_addr,
    })
}

impl DirectiveArgs {
    /// Decode an argument-request bytestring.
    pub fn decode(bytes: &[u8]) -> anyhow::Result<DirectiveArgs> {
        let mut const_params = vec![];
        let mut const_memory = vec![];
        let mut arg_ptr = 0;

        let read_u32 = |addr| {
            u32::from_le_bytes([
                bytes[addr],
                bytes[addr + 1],
                bytes[addr + 2],
                bytes[addr + 3],
            ])
        };
        let read_u64 = |addr| {
            u64::from_le_bytes([
                bytes[addr],
                bytes[addr + 1],
                bytes[addr + 2],
                bytes[addr + 3],
                bytes[addr + 4],
                bytes[addr + 5],
                bytes[addr + 6],
                bytes[addr + 7],
            ])
        };

        while arg_ptr < bytes.len() {
            let is_specialized = read_u32(arg_ptr);
            let ty = read_u32(arg_ptr + 4);
            let (value, mem, arg_len) = if is_specialized != 0 {
                match ty {
                    0 => (
                        AbstractValue::Concrete(WasmVal::I32(read_u32(arg_ptr + 8))),
                        None,
                        16,
                    ),
                    1 => (
                        AbstractValue::Concrete(WasmVal::I64(read_u64(arg_ptr + 8))),
                        None,
                        16,
                    ),
                    2 => (
                        AbstractValue::Concrete(WasmVal::F32(read_u32(arg_ptr + 8))),
                        None,
                        16,
                    ),
                    3 => (
                        AbstractValue::Concrete(WasmVal::F64(read_u64(arg_ptr + 8))),
                        None,
                        16,
                    ),
                    4 => {
                        let len = read_u32(arg_ptr + 8);
                        let padded_len = read_u32(arg_ptr + 12);
                        let data = MemoryBuffer {
                            data: Arc::new(
                                bytes[arg_ptr..(arg_ptr + usize::try_from(len).unwrap())].to_vec(),
                            ),
                        };
                        (AbstractValue::Runtime(None), Some(data), 16 + padded_len)
                    }
                    _ => anyhow::bail!("Invalid type: {}", ty),
                }
            } else {
                (AbstractValue::Runtime(None), None, 16)
            };
            const_params.push(value);
            const_memory.push(mem);
            arg_ptr += usize::try_from(arg_len).unwrap();
        }

        Ok(DirectiveArgs {
            const_params,
            const_memory,
        })
    }
}
