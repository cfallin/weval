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
use std::collections::{hash_map::Entry, HashMap};
use std::collections::{BTreeMap, BTreeSet};
use waffle::entity::{EntityRef, EntityVec, PerEntity};
use waffle::{Block, FunctionBody, Global, Type, Value};

waffle::declare_entity!(Context, "context");

pub type PC = Option<u32>;

/// One element in the context stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContextElem {
    Root,
    Loop(PC),
}

/// Arena of contexts.
#[derive(Clone, Default, Debug)]
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
                log::trace!("create context: {}: parent {} leaf {:?}", id, parent, elem);
                *v.insert(id)
            }
        }
    }

    pub fn parent(&self, context: Context) -> Context {
        self.contexts[context].0
    }

    pub fn leaf_element(&self, context: Context) -> ContextElem {
        self.contexts[context].1
    }
}

/// The flow-insensitive part of the satte.
#[derive(Clone, Debug, Default)]
pub struct SSAState {
    /// AbstractValues in specialized function of generic function's
    /// SSA `Value`s.
    pub values: BTreeMap<Value, AbstractValue>,
}

/// The flow-sensitive part of the state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgPointState {
    /// Memory overlay. We store only aligned u32s here.
    pub mem_overlay: BTreeMap<SymbolicAddr, MemValue>,
    /// Global values.
    pub globals: BTreeMap<Global, AbstractValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolicAddr(pub u32, pub i64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemValue {
    Value {
        addr: Value,
        offset: u32,
        data: Value,
        ty: Type,
    },
    Conflict,
}

impl MemValue {
    fn meet(a: MemValue, b: MemValue) -> MemValue {
        match (a, b) {
            (a, b) if a == b => a,
            _ => MemValue::Conflict,
        }
    }

    pub fn unwrap(&self) -> Option<(Value, Value, u32, Type)> {
        match self {
            MemValue::Value {
                addr,
                data,
                offset,
                ty,
            } => Some((*addr, *data, *offset, *ty)),
            _ => None,
        }
    }
}

/// The state for a function body during analysis.
#[derive(Clone, Debug)]
pub struct FunctionState {
    pub contexts: Contexts,
    pub state: PerEntity<Context, PerContextState>,
}

/// The state for one context.
#[derive(Clone, Debug, Default)]
pub struct PerContextState {
    pub ssa: SSAState,
    /// State at top of basic block.
    pub block_entry: BTreeMap<Block, ProgPointState>,
    /// State at end of basic block, just prior to terminator and
    /// successor-specific updates.
    pub block_exit: BTreeMap<Block, ProgPointState>,
}

/// State carried during a pass through a block.
#[derive(Clone, Debug)]
pub struct PointState {
    pub context: Context,
    pub pending_context: Option<Context>,
    pub flow: ProgPointState,
}

fn map_meet_with<
    K: PartialEq + Eq + PartialOrd + Ord + Copy,
    V: Copy + PartialEq + Eq,
    Meet: Fn(V, V) -> V,
>(
    this: &mut BTreeMap<K, V>,
    other: &BTreeMap<K, V>,
    meet: Meet,
    bot: V,
) -> bool {
    let mut changed = false;
    for (k, val) in this.iter_mut() {
        if let Some(other_val) = other.get(k) {
            let met = meet(*val, *other_val);
            changed |= met != *val;
            *val = met;
        } else {
            let old = *val;
            *val = bot;
            changed |= old != *val;
        }
    }
    for other_k in other.keys() {
        if !this.contains_key(other_k) {
            this.insert(*other_k, bot);
            changed = true;
        }
    }
    changed
}

fn set_union<K: PartialEq + Eq + PartialOrd + Ord + Copy>(
    this: &mut BTreeSet<K>,
    other: &BTreeSet<K>,
) -> bool {
    let mut inserted = false;
    for &elt in other {
        inserted |= this.insert(elt);
    }
    inserted
}

impl ProgPointState {
    pub fn entry(im: &Image) -> ProgPointState {
        let globals = im
            .globals
            .keys()
            .map(|global| (*global, AbstractValue::Runtime(ValueTags::default())))
            .collect();
        ProgPointState {
            mem_overlay: BTreeMap::new(),
            globals,
        }
    }

    pub fn meet_with(&mut self, other: &ProgPointState) -> bool {
        let mut changed = false;
        changed |= map_meet_with(
            &mut self.mem_overlay,
            &other.mem_overlay,
            MemValue::meet,
            MemValue::Conflict,
        );

        // TODO: check mem overlay for overlapping values of different
        // types

        changed |= map_meet_with(
            &mut self.globals,
            &other.globals,
            AbstractValue::meet,
            AbstractValue::Runtime(ValueTags::default()),
        );
        changed
    }

    /// the corresponding blockparam value.
    pub fn update_at_block_entry<F: FnMut(SymbolicAddr, Type) -> Value>(
        &mut self,
        _get_blockparam: &mut F,
    ) {
        // TODO
    }
}

impl FunctionState {
    pub fn new() -> FunctionState {
        FunctionState {
            contexts: Contexts::default(),
            state: PerEntity::default(),
        }
    }

    pub fn init_args(
        &mut self,
        orig_body: &FunctionBody,
        im: &Image,
        args: &[AbstractValue],
    ) -> (Context, ProgPointState) {
        // For each blockparam of the entry block, set the value of the SSA arg.
        debug_assert_eq!(args.len(), orig_body.blocks[orig_body.entry].params.len());
        let ctx = self.contexts.create(None, ContextElem::Root);
        for ((_, orig_value), abs) in orig_body.blocks[orig_body.entry]
            .params
            .iter()
            .zip(args.iter())
        {
            self.state[ctx].ssa.values.insert(*orig_value, *abs);
        }
        (ctx, ProgPointState::entry(im))
    }
}
