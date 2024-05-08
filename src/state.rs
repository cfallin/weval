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
    Specialized(Value, u32),
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
}

/// The flow-sensitive part of the state.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ProgPointState {
    /// Specialization registers.
    pub regs: BTreeMap<RegSlot, RegValue>,
    /// Global values.
    pub globals: BTreeMap<Global, AbstractValue>,
    /// Virtualized stack values (grows downward: we insert at the
    /// beginning, so indices are consistent with the API's
    /// definitions and merging takes a common prefix).
    ///
    /// Each entry is an (address, data) pair.
    pub stack: Vec<(RegValue, RegValue)>,
    /// Virtualized locals, with (address, data) pairs for spilling
    /// back to memory at sync points.
    pub locals: BTreeMap<u32, (RegValue, RegValue)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegSlot {
    Register(u32),
    LocalAddr(u32),
    LocalData(u32),
    StackData(u32),
    StackAddr(u32),
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
                panic!("Values {:?} and {:?} meeting to Conflict", a, b);
            }
        }
    }

    pub fn value(&self) -> Option<Value> {
        match self {
            RegValue::Value { data, .. } => Some(*data),
            _ => None,
        }
    }

    pub fn ty(&self) -> Type {
        match self {
            RegValue::Value { ty, .. } => *ty,
            RegValue::Merge { ty, .. } => *ty,
        }
    }
}

/// The state for a function body during analysis.
#[derive(Clone, Debug, Default)]
pub struct FunctionState {
    pub contexts: Contexts,
    /// AbstractValues in specialized function, indexed by specialized
    /// Value.
    pub values: PerEntity<Value, AbstractValue>,
    /// Block-entry abstract values, indexed by specialized Block.
    pub block_entry: PerEntity<Block, Option<ProgPointState>>,
    /// Block-exit abstract values, indexed by specialized Block.
    pub block_exit: PerEntity<Block, Option<ProgPointState>>,
    /// Specialization values (constant args).
    pub specialization_globals: Vec<AbstractValue>,
}

/// State carried during a pass through a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointState {
    pub context: Context,
    pub pending_context: Option<Context>,
    pub pending_specialize: Option<(Value, u32, u32)>,
    pub flow: ProgPointState,
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
        let globals: BTreeMap<Global, AbstractValue> = im
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
            stack: vec![],
            locals: BTreeMap::new(),
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

        if other.stack.len() < self.stack.len() {
            changed = true;
            self.stack.truncate(other.stack.len());
        }
        for (this, other) in self.stack.iter_mut().zip(other.stack.iter()) {
            let new_addr = RegValue::meet(&this.0, &other.0);
            changed |= new_addr != this.0;
            this.0 = new_addr;
            let new_data = RegValue::meet(&this.1, &other.1);
            changed |= new_data != this.1;
            this.1 = new_data;
        }

        changed |= map_meet_with(
            &mut self.locals,
            &other.locals,
            |(a0, a1), (b0, b1)| (RegValue::meet(a0, b0), RegValue::meet(a1, b1)),
            None,
        );

        changed
    }

    pub fn update_across_edge(&mut self) {
        let create_merge = |value: &mut RegValue| {
            if let RegValue::Value { ty, abs, .. } = value {
                // Ensure all specialization-register values become
                // blockparams, even if only one pred.
                *value = RegValue::Merge {
                    ty: *ty,
                    abs: abs.clone(),
                };
            }
        };

        for value in self.regs.values_mut() {
            create_merge(value);
        }
        for (addr, data) in &mut self.stack {
            create_merge(addr);
            create_merge(data);
        }
        for (addr, data) in self.locals.values_mut() {
            create_merge(addr);
            create_merge(data);
        }
    }

    pub fn update_at_block_entry<C, GB: FnMut(&mut C, RegSlot, Type) -> Value>(
        &mut self,
        ctx: &mut C,
        get_blockparam: &mut GB,
    ) -> anyhow::Result<()> {
        let mut handle_value = |slot: RegSlot, value: &mut RegValue| match value {
            RegValue::Value { .. } => {}
            RegValue::Merge { ty, abs } => {
                let param = get_blockparam(ctx, slot, *ty);
                *value = RegValue::Value {
                    data: param,
                    ty: *ty,
                    abs: abs.clone(),
                };
            }
        };
        for (&idx, value) in &mut self.regs {
            handle_value(idx, value);
        }
        for (i, (addr, data)) in self.stack.iter_mut().enumerate() {
            handle_value(RegSlot::StackAddr(i as u32), addr);
            handle_value(RegSlot::StackData(i as u32), data);
        }
        for (i, (addr, value)) in self.locals.iter_mut() {
            handle_value(RegSlot::LocalAddr(*i), addr);
            handle_value(RegSlot::LocalData(*i), value);
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
        num_globals: usize,
        args: &[AbstractValue],
        ctx: Context,
        value_map: &HashMap<(Context, Value), Value>,
    ) {
        // For each blockparam of the entry block, set the value of the SSA arg.
        debug_assert_eq!(args.len(), orig_body.blocks[orig_body.entry].params.len());
        for ((_, orig_value), abs) in orig_body.blocks[orig_body.entry]
            .params
            .iter()
            .zip(args.iter().skip(num_globals))
        {
            let spec_value = *value_map.get(&(ctx, *orig_value)).unwrap();
            self.values[spec_value] = abs.clone();
        }

        // Set specialization globals, if any.
        for i in 0..num_globals {
            self.specialization_globals.push(args[i].clone());
        }
    }
}
