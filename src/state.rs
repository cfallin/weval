//! State tracking.
//!
//! Constant-propagation / function specialization state consists of
//! *abstract values* for each value in the program, indicating
//! whether we know that value to be a constant (and can specialize
//! the function body on it) or not.
//!
//! Because we replicate basic blocks according to "loop PC", this
//! state is *context-sensitive*: every piece of the state is indexed
//! on the "context", which is a stack of PCs as indicated by program
//! intrinsics.
//!
//! Per context, there are two halves to the state:
//!
//! - The *SSA values*, which are *flow-insensitive*, i.e. have the
//!   same value everywhere (are not indexed on program-point). The
//!   flow-insensitivity arises from the fact that each value is
//!   defined exactly once.
//!
//! - The *global state*, consisting of an overlay of abstract values
//!   on memory addresses (the "memory overlay") and abstract values for
//!   Wasm globals, which is *flow-sensitive*: because this state can be
//!   updated by certain instructions, we need to track it indexed by
//!   both context and program-point. Fortunately this piece of the
//!   state is usually small relative to the flow-insensitive part.
//!
//! The lookup of any particular piece of state in the
//! flow-insensitive part works via the "context stack". First we look
//! to see if the value is defined with the most specific context we
//! have (all nested unrolled loops' PCs); if not found, we pop a PC
//! off the context stack and try again. This lets us see values from
//! blocks outside of the loop. The flow-sensitive part of state does
//! not need to do this, and in fact cannot, because we have to
//! examine the state at a given program point and using a different
//! context implies leaving the current loop.

use crate::image::Image;
use crate::value::{AbstractValue, ValueTags};
use std::collections::BTreeMap;
use std::collections::{hash_map::Entry, HashMap};
use waffle::entity::{EntityRef, EntityVec, PerEntity};
use waffle::{Block, FunctionBody, Global, Value};

waffle::declare_entity!(Context, "context");

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContextElem(pub Option<u64>);

/// Arena of contexts.
#[derive(Clone, Debug, Default)]
pub struct Contexts {
    contexts: EntityVec<Context, (Context, ContextElem)>,
    dedup: HashMap<(Context, ContextElem), Context>, // map from (parent, tail_elem) to ID
}

impl Contexts {
    pub fn create(&mut self, parent: Option<Context>, elem: ContextElem) -> Context {
        let parent = parent.unwrap_or(Context::invalid());
        match self.dedup.entry((parent, elem)) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let id = self.contexts.push((parent, elem));
                *v.insert(id)
            }
        }
    }
}

/// The flow-insensitive part of the satte.
#[derive(Clone, Debug, Default)]
pub struct SSAState {
    /// Values of SSA `Value`s.
    pub values: BTreeMap<Value, AbstractValue>,
}

/// The flow-sensitive part of the state.
#[derive(Clone, Debug)]
pub struct ProgPointState {
    /// Memory overlay. We store only aligned u32s here.
    pub mem_overlay: BTreeMap<u32, AbstractValue>,
    /// Global values.
    pub globals: BTreeMap<Global, AbstractValue>,
}

/// The state for a function body.
#[derive(Clone, Debug)]
pub struct FunctionState {
    pub contexts: Contexts,
    pub state: PerEntity<Context, PerContextState>,
}

/// The state for one context.
#[derive(Clone, Debug, Default)]
pub struct PerContextState {
    pub ssa: SSAState,
    pub block_entry: BTreeMap<Block, ProgPointState>,
}

/// State carried during a pass through a block.
#[derive(Clone, Debug)]
pub struct PointState {
    pub context: Context,
    pub flow: ProgPointState,
}

fn map_meet_with<K: PartialEq + Eq + PartialOrd + Ord + Copy>(
    this: &mut BTreeMap<K, AbstractValue>,
    other: &BTreeMap<K, AbstractValue>,
) -> bool {
    let mut changed = false;
    for (k, val) in this.iter_mut() {
        if let Some(other_val) = other.get(k) {
            let met = AbstractValue::meet(*val, *other_val);
            changed |= met != *val;
            *val = met;
        } else {
            *val = AbstractValue::Top;
            // N.B.: not changed: Top should not have an effect (every
            // non-present key is conceptually already Top).
        }
    }
    for other_k in other.keys() {
        if !this.contains_key(other_k) {
            // `Runtime` is a "bottom" value in the semilattice.
            this.insert(*other_k, AbstractValue::Runtime(ValueTags::default()));
            changed = true;
        }
    }
    changed
}

impl FunctionState {
    pub fn new(_im: &Image, _func: &FunctionBody, _args: Vec<AbstractValue>) -> FunctionState {
        todo!()
    }
}
