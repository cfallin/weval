//! Static module image summary.

use crate::value::WasmVal;
use std::collections::BTreeMap;
use waffle::{Func, Global, Memory, MemoryData, MemorySegment, Module, Table};

#[derive(Clone, Debug)]
pub struct Image {
    pub memories: BTreeMap<Memory, MemImage>,
    pub globals: BTreeMap<Global, WasmVal>,
    pub tables: BTreeMap<Table, Vec<Func>>,
    pub stack_pointer: Option<Global>,
    pub main_heap: Option<Memory>,
    pub main_table: Option<Table>,
}

#[derive(Clone, Debug)]
pub struct MemImage {
    pub image: Vec<u8>,
}

impl MemImage {
    pub fn len(&self) -> usize {
        self.image.len()
    }
}

pub fn build_image(module: &Module) -> anyhow::Result<Image> {
    Ok(Image {
        memories: module
            .memories
            .entries()
            .flat_map(|(id, mem)| maybe_mem_image(mem).map(|image| (id, image)))
            .collect(),
        globals: module
            .globals
            .entries()
            .flat_map(|(global_id, data)| match data.value {
                Some(bits) => Some((global_id, WasmVal::from_bits(data.ty, bits)?)),
                _ => None,
            })
            .collect(),
        tables: module
            .tables
            .entries()
            .map(|(id, data)| (id, data.func_elements.clone().unwrap_or(vec![])))
            .collect(),
        // HACK: assume first global is shadow stack pointer.
        stack_pointer: module.globals.iter().next(),
        // HACK: assume first memory is main heap.
        main_heap: module.memories.iter().next(),
        // HACK: assume first table is used for function pointers.
        main_table: module.tables.iter().next(),
    })
}

const WASM_PAGE: usize = 1 << 16;

fn maybe_mem_image(mem: &MemoryData) -> Option<MemImage> {
    let len = mem.initial_pages * WASM_PAGE;
    let mut image = vec![0; len];

    for segment in &mem.segments {
        image[segment.offset..(segment.offset + segment.data.len())]
            .copy_from_slice(&segment.data[..]);
    }

    Some(MemImage { image })
}

pub fn update(module: &mut Module, im: &Image) {
    for (&mem_id, mem) in &im.memories {
        module.memories[mem_id].segments.clear();
        module.memories[mem_id].segments.push(MemorySegment {
            offset: 0,
            data: mem.image.clone(),
        });
        let image_pages = mem.image.len() / WASM_PAGE;
        module.memories[mem_id].initial_pages =
            std::cmp::max(module.memories[mem_id].initial_pages, image_pages);
    }
}

impl Image {
    pub fn can_read(&self, memory: Memory, addr: u32, size: u32) -> bool {
        let end = match addr.checked_add(size) {
            Some(end) => end,
            None => return false,
        };
        let image = match self.memories.get(&memory) {
            Some(image) => image,
            None => return false,
        };
        (end as usize) <= image.len()
    }

    pub fn main_heap(&self) -> anyhow::Result<Memory> {
        self.main_heap
            .ok_or_else(|| anyhow::anyhow!("no main heap"))
    }

    pub fn read_slice(&self, id: Memory, addr: u32, len: u32) -> anyhow::Result<&[u8]> {
        let image = self.memories.get(&id).unwrap();
        let addr = usize::try_from(addr).unwrap();
        let len = usize::try_from(len).unwrap();
        if addr + len >= image.len() {
            anyhow::bail!("Out of bounds");
        }
        Ok(&image.image[addr..(addr + len)])
    }

    pub fn read_u8(&self, id: Memory, addr: u32) -> anyhow::Result<u8> {
        let image = self.memories.get(&id).unwrap();
        image
            .image
            .get(addr as usize)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Out of bounds"))
    }

    pub fn read_u16(&self, id: Memory, addr: u32) -> anyhow::Result<u16> {
        let image = self.memories.get(&id).unwrap();
        let addr = addr as usize;
        if (addr + 2) > image.len() {
            anyhow::bail!("Out of bounds");
        }
        let slice = &image.image[addr..(addr + 2)];
        Ok(u16::from_le_bytes([slice[0], slice[1]]))
    }

    pub fn read_u32(&self, id: Memory, addr: u32) -> anyhow::Result<u32> {
        let image = self.memories.get(&id).unwrap();
        let addr = addr as usize;
        if (addr + 4) > image.len() {
            anyhow::bail!("Out of bounds");
        }
        let slice = &image.image[addr..(addr + 4)];
        Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    pub fn read_u64(&self, id: Memory, addr: u32) -> anyhow::Result<u64> {
        let low = self.read_u32(id, addr)?;
        let high = self.read_u32(id, addr + 4)?;
        Ok((high as u64) << 32 | (low as u64))
    }

    pub fn read_u128(&self, id: Memory, addr: u32) -> anyhow::Result<u128> {
        let low = self.read_u64(id, addr)?;
        let high = self.read_u64(id, addr + 8)?;
        Ok((high as u128) << 64 | (low as u128))
    }

    pub fn read_size(&self, id: Memory, addr: u32, size: u8) -> anyhow::Result<u64> {
        match size {
            1 => self.read_u8(id, addr).map(|x| x as u64),
            2 => self.read_u16(id, addr).map(|x| x as u64),
            4 => self.read_u32(id, addr).map(|x| x as u64),
            8 => self.read_u64(id, addr),
            _ => panic!("bad size"),
        }
    }

    pub fn read_str(&self, id: Memory, mut addr: u32) -> anyhow::Result<String> {
        let mut bytes = vec![];
        loop {
            let byte = self.read_u8(id, addr)?;
            if byte == 0 {
                break;
            }
            bytes.push(byte);
            addr += 1;
        }
        Ok(std::str::from_utf8(&bytes[..])?.to_owned())
    }

    pub fn write_u8(&mut self, id: Memory, addr: u32, value: u8) -> anyhow::Result<()> {
        let image = self.memories.get_mut(&id).unwrap();
        *image
            .image
            .get_mut(addr as usize)
            .ok_or_else(|| anyhow::anyhow!("Out of bounds"))? = value;
        Ok(())
    }

    pub fn write_u32(&mut self, id: Memory, addr: u32, value: u32) -> anyhow::Result<()> {
        let image = self.memories.get_mut(&id).unwrap();
        let addr = addr as usize;
        if (addr + 4) > image.len() {
            anyhow::bail!("Out of bounds");
        }
        let slice = &mut image.image[addr..(addr + 4)];
        slice.copy_from_slice(&value.to_le_bytes()[..]);
        Ok(())
    }

    pub fn func_ptr(&self, idx: u32) -> anyhow::Result<Func> {
        let table = self
            .main_table
            .ok_or_else(|| anyhow::anyhow!("no main table"))?;
        Ok(self
            .tables
            .get(&table)
            .unwrap()
            .get(idx as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("func ptr out of bounds"))?)
    }

    pub fn append_data(&mut self, id: Memory, data: Vec<u8>) {
        let image = self.memories.get_mut(&id).unwrap();
        let orig_len = image.len();
        let data_len = data.len();
        let padded_len = (data_len + WASM_PAGE - 1) & !(WASM_PAGE - 1);
        let padding = padded_len - data_len;
        image
            .image
            .extend(data.into_iter().chain(std::iter::repeat(0).take(padding)));
        log::debug!(
            "Appending data ({} bytes, {} padding): went from {} bytes to {} bytes",
            data_len,
            padding,
            orig_len,
            image.len()
        );
    }
}
