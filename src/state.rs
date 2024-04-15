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
use crate::value::{AbstractValue, WasmVal};
use fxhash::FxHashMap as HashMap;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use waffle::entity::{EntityRef, EntityVec, PerEntity};
use waffle::{Block, FunctionBody, Global, Type, Value};

waffle::declare_entity!(Context, "context");

pub type PC = u32;

/// One element in the context stack.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContextElem {
    Root,
    Loop(PC),
    PendingSpecialize(Value, u32, u32),
    Specialized(Value, u32),
    Stack(Vec<StackEntry>),
}

/// Arena of contexts.
#[derive(Clone, Default, Debug)]
pub struct Contexts {
    contexts: EntityVec<Context, (Context, ContextElem)>,
    pub(crate) context_bucket: PerEntity<Context, Option<u32>>,
    dedup: HashMap<(Context, ContextElem), Context>, // map from (parent, tail_elem) to ID
}

impl Contexts {
    pub fn create(&mut self, parent: Option<Context>, elem: ContextElem) -> Context {
        let parent = parent.unwrap_or(Context::invalid());
        match self.dedup.entry((parent, elem.clone())) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let id = self.contexts.push((parent, elem.clone()));
                log::trace!("create context: {}: parent {} leaf {:?}", id, parent, elem);
                *v.insert(id)
            }
        }
    }

    pub fn parent(&self, context: Context) -> Context {
        self.contexts[context].0
    }

    pub fn leaf_element(&self, context: Context) -> ContextElem {
        self.contexts[context].1.clone()
    }

    pub fn pop_one_loop(&self, mut context: Context) -> Context {
        loop {
            match &self.contexts[context] {
                (parent, ContextElem::Loop(_)) => return *parent,
                (_, ContextElem::Root) => return context,
                (parent, _) => {
                    context = *parent;
                }
            }
        }
    }

    pub fn pop_stack_and_pending_specialization(
        &self,
        mut context: Context,
    ) -> Option<ContextElem> {
        let mut ret = None;
        loop {
            match &self.contexts[context] {
                (parent, ce @ ContextElem::PendingSpecialize(..)) => {
                    ret = Some(ce.clone());
                    context = *parent;
                }
                (parent, ContextElem::Stack(..)) => {
                    context = *parent;
                }
                _ => break,
            }
        }
        ret
    }

    pub fn push_stack(&mut self, context: Context, stack: &[StackEntry]) -> Context {
        if stack.is_empty() {
            context
        }else {
            self.create(Some(context), ContextElem::Stack(stack.to_vec()))
        }
    }
}

/// The flow-insensitive part of the satte.
#[derive(Clone, Debug, Default)]
pub struct SSAState {}

/// The flow-sensitive part of the state.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ProgPointState {
    /// Specialization registers.
    pub regs: BTreeMap<u64, RegValue>,
    /// Global values.
    pub globals: BTreeMap<Global, AbstractValue>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegValue {
    Value {
        data: Value,
        abs: AbstractValue,
        ty: Type,
    },
    Merge {
        ty: Type,
        abs: AbstractValue,
    },
    Conflict,
}

impl RegValue {
    fn meet(a: &RegValue, b: &RegValue) -> RegValue {
        match (a, b) {
            (a, b) if a == b => a.clone(),
            (
                RegValue::Value {
                    ty: ty1, abs: abs1, ..
                },
                RegValue::Value {
                    ty: ty2, abs: abs2, ..
                },
            ) if ty1 == ty2 => RegValue::Merge {
                ty: *ty1,
                abs: AbstractValue::meet(abs1, abs2),
            },
            (RegValue::Merge { ty: ty1, abs: abs1 }, RegValue::Merge { ty: ty2, abs: abs2 })
                if ty1 == ty2 =>
            {
                RegValue::Merge {
                    ty: *ty1,
                    abs: AbstractValue::meet(abs1, abs2),
                }
            }
            (
                RegValue::Merge { ty, abs },
                RegValue::Value {
                    ty: ty1, abs: abs1, ..
                },
            )
            | (
                RegValue::Value {
                    ty: ty1, abs: abs1, ..
                },
                RegValue::Merge { ty, abs },
            ) if ty == ty1 => RegValue::Merge {
                ty: *ty,
                abs: AbstractValue::meet(abs, abs1),
            },
            _ => {
                log::trace!("Values {:?} and {:?} meeting to Conflict", a, b);
                RegValue::Conflict
            }
        }
    }

    pub fn value(&self) -> Option<Value> {
        match self {
            RegValue::Value { data, .. } => Some(*data),
            _ => None,
        }
    }

    pub fn ty(&self) -> Option<Type> {
        match self {
            RegValue::Value { ty, .. } => Some(*ty),
            RegValue::Merge { ty, .. } => Some(*ty),
            _ => None,
        }
    }
}

/// An entry on the virtualized stack.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StackEntry {
    /// A value explicitly pushed on the stack. Not yet sync'd
    /// (written) to real memory.
    Value { stackptr: Value, value: Value },
    // TODO: OtherMem too.
}

/// The state for a function body during analysis.
#[derive(Clone, Debug, Default)]
pub struct FunctionState {
    pub contexts: Contexts,
    /// AbstractValues in specialized function, indexed by specialized
    /// Value.
    pub values: PerEntity<Value, AbstractValue>,
    /// Block-entry abstract values, indexed by specialized Block.
    pub block_entry: PerEntity<Block, ProgPointState>,
    /// Block-exit abstract values, indexed by specialized Block.
    pub block_exit: PerEntity<Block, ProgPointState>,
}

/// State carried during a pass through a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointState {
    pub context: Context,
    pub pending_context: Option<Context>,
    pub flow: ProgPointState,
    /// Virtualized stack. This is logically part of the context as well.
    pub stack: Vec<StackEntry>,
}

fn map_meet_with<
    K: PartialEq + Eq + PartialOrd + Ord + Copy,
    V: Clone + PartialEq + Eq,
    Meet: Fn(&V, &V) -> V,
>(
    this: &mut BTreeMap<K, V>,
    other: &BTreeMap<K, V>,
    meet: Meet,
    bot: Option<V>,
) -> bool {
    let mut changed = false;
    let mut to_remove = vec![];
    for (k, val) in this.iter_mut() {
        if let Some(other_val) = other.get(k) {
            let met = meet(val, other_val);
            changed |= met != *val;
            *val = met;
        } else {
            let old = val.clone();
            if let Some(bot) = bot.as_ref() {
                *val = bot.clone();
                changed |= old != *val;
            } else {
                to_remove.push(k.clone());
                changed = true;
            }
        }
    }
    for k in to_remove {
        this.remove(&k);
    }
    for other_k in other.keys() {
        if !this.contains_key(other_k) {
            if let Some(bot) = bot.as_ref() {
                this.insert(*other_k, bot.clone());
            } else {
                this.remove(other_k);
            }
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
            .iter()
            .enumerate()
            .map(|(i, (global, init_val))| {
                if i == 0 {
                    (*global, AbstractValue::Runtime(None))
                } else if let &WasmVal::I32(addr) = init_val {
                    // GOT base global.
                    (*global, AbstractValue::StaticMemory(addr))
                } else {
                    (*global, AbstractValue::Runtime(None))
                }
            })
            .collect();
        ProgPointState {
            regs: BTreeMap::new(),
            globals,
        }
    }

    pub fn meet_with(&mut self, other: &ProgPointState) -> bool {
        let mut changed = false;
        changed |= map_meet_with(&mut self.regs, &other.regs, RegValue::meet, None);

        changed |= map_meet_with(
            &mut self.globals,
            &other.globals,
            AbstractValue::meet,
            Some(AbstractValue::Runtime(None)),
        );
        changed
    }

    pub fn update_across_edge(&mut self) {
        for value in self.regs.values_mut() {
            if let RegValue::Value { ty, abs, .. } = value {
                // Ensure all specialization-register values become
                // blockparams, even if only one pred.
                *value = RegValue::Merge {
                    ty: *ty,
                    abs: abs.clone(),
                };
            }
        }
    }

    pub fn update_at_block_entry<
        C,
        GB: FnMut(&mut C, u64, Type) -> Value,
        RB: FnMut(&mut C, u64),
    >(
        &mut self,
        ctx: &mut C,
        get_blockparam: &mut GB,
        remove_blockparam: &mut RB,
    ) -> anyhow::Result<()> {
        let mut to_remove = vec![];
        for (&idx, value) in &mut self.regs {
            match value {
                RegValue::Value { .. } => {}
                RegValue::Merge { ty, abs } => {
                    let param = get_blockparam(ctx, idx, *ty);
                    *value = RegValue::Value {
                        data: param,
                        ty: *ty,
                        abs: abs.clone(),
                    };
                }
                RegValue::Conflict => {
                    remove_blockparam(ctx, idx);
                    to_remove.push(idx);
                }
            }
        }
        for to_remove in to_remove {
            self.regs.remove(&to_remove);
        }
        Ok(())
    }
}

impl FunctionState {
    pub fn new() -> FunctionState {
        FunctionState::default()
    }

    pub fn init(&mut self, im: &Image) -> (Context, ProgPointState) {
        let ctx = self.contexts.create(None, ContextElem::Root);
        (ctx, ProgPointState::entry(im))
    }

    pub fn set_args(
        &mut self,
        orig_body: &FunctionBody,
        args: &[AbstractValue],
        ctx: Context,
        value_map: &HashMap<(Context, Value), Value>,
    ) {
        // For each blockparam of the entry block, set the value of the SSA arg.
        debug_assert_eq!(args.len(), orig_body.blocks[orig_body.entry].params.len());
        for ((_, orig_value), abs) in orig_body.blocks[orig_body.entry]
            .params
            .iter()
            .zip(args.iter())
        {
            let spec_value = *value_map.get(&(ctx, *orig_value)).unwrap();
            self.values[spec_value] = abs.clone();
        }
    }
}
