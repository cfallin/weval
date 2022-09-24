//! State tracking.

use crate::image::Image;
use crate::value::{Value, WasmVal};
use std::collections::BTreeMap;
use walrus::{GlobalId, LocalId};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    /// Memory overlay. We store only aligned u32s here.
    mem_overlay: BTreeMap<u32, Value>,
    /// Global values.
    globals: BTreeMap<GlobalId, Value>,
    /// Local values.
    locals: BTreeMap<LocalId, Value>,
}

impl State {
    pub fn initial(im: &Image) -> State {
        todo!()
    }
}
