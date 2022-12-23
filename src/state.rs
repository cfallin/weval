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
    /// Escaped symbolic labels. We can't track any memory relative to
    /// these bases anymore at this program point.
    pub escaped_labels: BTreeSet<u32>,
    /// Global values.
    pub globals: BTreeMap<Global, AbstractValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolicAddr(pub u32, pub i64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemValue {
    Value(Value, Type),
    BlockParam(Type),
    Escaped,
}

impl MemValue {
    fn meet(a: MemValue, b: MemValue) -> MemValue {
        match (a, b) {
            (a, b) if a == b => a,
            (MemValue::Value(_, t1), MemValue::Value(_, t2)) if t1 == t2 => {
                MemValue::BlockParam(t1)
            }
            _ => MemValue::Escaped,
        }
    }

    fn ty(&self) -> Option<Type> {
        match self {
            &MemValue::Value(_, t) => Some(t),
            &MemValue::BlockParam(t) => Some(t),
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
    pub block_entry: BTreeMap<Block, ProgPointState>,
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
            escaped_labels: BTreeSet::new(),
            globals,
        }
    }

    pub fn meet_with(&mut self, other: &ProgPointState) -> bool {
        let mut changed = false;
        changed |= map_meet_with(
            &mut self.mem_overlay,
            &other.mem_overlay,
            MemValue::meet,
            MemValue::Escaped,
        );

        // TODO: check mem overlay for overlapping values of different
        // types

        changed |= map_meet_with(
            &mut self.globals,
            &other.globals,
            AbstractValue::meet,
            AbstractValue::Runtime(ValueTags::default()),
        );
        changed |= set_union(&mut self.escaped_labels, &other.escaped_labels);
        changed
    }

    pub fn escape_label(&mut self, label: u32) {
        log::trace!("escaping label {}", label);
        if !self.escaped_labels.insert(label) {
            return;
        }

        for (_k, v) in self
            .mem_overlay
            .range_mut(SymbolicAddr(label, i64::MIN)..SymbolicAddr(label + 1, i64::MIN))
        {
            *v = MemValue::Escaped;
        }
    }

    /// Invoked at block entry; may lift a merged-blockparam state to
    /// the corresponding blockparam value.
    pub fn update_at_block_entry<F: FnMut(SymbolicAddr, Type) -> Value>(
        &mut self,
        get_blockparam: &mut F,
    ) {
        // Any `MemValue::BlockParam` needs a blockparam, and an
        // upgrade to `Value`.
        for (addr, val) in &mut self.mem_overlay {
            if let MemValue::BlockParam(ty) = val {
                let blockparam = get_blockparam(*addr, *ty);
                *val = MemValue::Value(blockparam, *ty);
            }
        }
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
