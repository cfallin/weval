//! Partial evaluation.

/* TODO:
- inlining
- "memory renaming": connecting symbolic ops through the operand-stack
  memory region
- more general memory-region handling: symbolic but unique
  (non-escaped) pointers, stack, operand-stack region, ...
*/

use crate::directive::Directive;
use crate::image::Image;
use crate::intrinsics::Intrinsics;
use crate::state::*;
use crate::value::{AbstractValue, ValueTags, WasmVal};
use std::collections::{
    btree_map::Entry as BTreeEntry, hash_map::Entry as HashEntry, HashMap, HashSet,
};
use waffle::cfg::CFGInfo;
use waffle::{
    entity::EntityRef, Block, BlockTarget, FunctionBody, Module, Operator, Terminator, Type, Value,
    ValueDef,
};

struct Evaluator<'a> {
    /// Original function body.
    generic: &'a FunctionBody,
    /// Intrinsic function indices.
    intrinsics: &'a Intrinsics,
    /// Memory image.
    image: &'a Image,
    /// Domtree for function body.
    cfg: CFGInfo,
    /// State of SSA values and program points:
    /// - per context:
    ///   - per SSA number, an abstract value
    ///   - per block, entry state for that block
    state: FunctionState,
    /// New function body.
    func: FunctionBody,
    /// Map of (ctx, block_in_generic) to specialized block_in_func.
    block_map: HashMap<(Context, Block), Block>,
    /// Map of (ctx, value_in_generic) to specialized value_in_func.
    value_map: HashMap<(Context, Value), Value>,
    /// Queue of blocks to (re)compute. List of (block_in_generic,
    /// ctx, block_in_func).
    queue: Vec<(Block, Context, Block)>,
    /// Set to deduplicate `queue`.
    queue_set: HashSet<(Block, Context)>,
}

/// Partially evaluates according to the given directives.
pub fn partially_evaluate(
    module: &mut Module,
    im: &mut Image,
    directives: &[Directive],
) -> anyhow::Result<()> {
    let intrinsics = Intrinsics::find(module);
    log::trace!("intrinsics: {:?}", intrinsics);
    let mut mem_updates = HashMap::new();
    for directive in directives {
        log::trace!("Processing directive {:?}", directive);
        if let Some(idx) = partially_evaluate_func(module, im, &intrinsics, directive)? {
            log::trace!("New func index {}", idx);
            // Update memory image.
            mem_updates.insert(directive.func_index_out_addr, idx);
        }
    }

    // Update memory.
    let heap = im.main_heap()?;
    for (addr, value) in mem_updates {
        im.write_u32(heap, addr, value)?;
    }
    Ok(())
}

fn partially_evaluate_func(
    module: &mut Module,
    image: &Image,
    intrinsics: &Intrinsics,
    directive: &Directive,
) -> anyhow::Result<Option<u32>> {
    // Get function body.
    let body = module
        .func(directive.func)
        .body()
        .ok_or_else(|| anyhow::anyhow!("Attempt to specialize an import"))?;
    let sig = module.func(directive.func).sig();

    // Compute CFG info.
    let cfg = CFGInfo::new(body);

    // Build the evaluator.
    let mut evaluator = Evaluator {
        generic: body,
        intrinsics,
        image,
        cfg,
        state: FunctionState::new(),
        func: FunctionBody::new(module, sig),
        block_map: HashMap::new(),
        value_map: HashMap::new(),
        queue: vec![],
        queue_set: HashSet::new(),
    };
    evaluator.state.init_args(
        body,
        &mut evaluator.func,
        image,
        &directive.const_params[..],
    );
    evaluator.queue.push((
        evaluator.generic.entry,
        Context::default(),
        evaluator.func.entry,
    ));
    evaluator
        .queue_set
        .insert((evaluator.generic.entry, Context::default()));
    evaluator.evaluate();

    log::debug!("Adding func:\n{}", evaluator.func.display("| "));
    let func = module.add_func(sig, evaluator.func);
    Ok(Some(func.index() as u32))
}

fn const_operator(ty: Type, value: WasmVal) -> Option<Operator> {
    match (ty, value) {
        (Type::I32, WasmVal::I32(k)) => Some(Operator::I32Const { value: k as i32 }),
        (Type::I64, WasmVal::I64(k)) => Some(Operator::I64Const { value: k as i64 }),
        (Type::F32, WasmVal::F32(k)) => Some(Operator::F32Const {
            value: waffle::wasmparser::Ieee32::from_bits(k as u32),
        }),
        (Type::F64, WasmVal::F64(k)) => Some(Operator::F64Const {
            value: waffle::wasmparser::Ieee64::from_bits(k),
        }),
        _ => None,
    }
}

impl<'a> Evaluator<'a> {
    fn evaluate(&mut self) {
        while let Some((orig_block, ctx, new_block)) = self.queue.pop() {
            self.queue_set.remove(&(orig_block, ctx));
            self.evaluate_block(orig_block, ctx, new_block);
        }
    }

    fn evaluate_block(&mut self, orig_block: Block, ctx: Context, new_block: Block) {
        // Clear the block body each time we rebuild it -- we may be
        // recomputing a specialization with an existing output.
        self.func.blocks[new_block].insts.clear();

        // Create program-point state.
        let mut state = PointState {
            context: ctx,
            flow: self.state.state[ctx]
                .block_entry
                .get(&orig_block)
                .cloned()
                .unwrap(),
        };

        // Do the actual constant-prop, carrying the state across the
        // block and updating flow-sensitive state, and updating SSA
        // vals as well.
        self.evaluate_block_body(orig_block, &mut state, new_block);
        self.evaluate_term(orig_block, &mut state, new_block);
    }

    /// For a given value in the generic function, accessed in the
    /// given context, find its abstract value and SSA value in the
    /// specialized function.
    fn use_value(&self, mut context: Context, orig_val: Value) -> (Value, AbstractValue) {
        loop {
            if let Some((val, abs)) = self.state.state[context].ssa.values.get(&orig_val) {
                return (*val, *abs);
            }
            assert_ne!(context, Context::default());
            context = self.state.contexts.parent(context);
        }
    }

    fn def_value(
        &mut self,
        block: Block,
        context: Context,
        orig_val: Value,
        val: Value,
        abs: AbstractValue,
    ) {
        let changed = self.state.state[context]
            .ssa
            .values
            .insert(orig_val, (val, abs))
            .map(|(_, old_abs)| abs != old_abs)
            .unwrap_or(true);

        if changed {
            if let &ValueDef::BlockParam(dest_block, _, _) = &self.generic.values[orig_val] {
                // We just updated a blockparam. If the block it is
                // attached to is not dominated by our current block,
                // then it won't be seen in this pass, so let's
                // enqueue the block.
                if !self.cfg.dominates(block, dest_block) {
                    self.enqueue_block_if_existing(dest_block, context);
                }
            }
        }
    }

    fn enqueue_block_if_existing(&mut self, orig_block: Block, context: Context) {
        if let Some(block) = self.block_map.get(&(context, orig_block)).copied() {
            if self.queue_set.insert((orig_block, context)) {
                self.queue.push((orig_block, context, block));
            }
        }
    }

    fn evaluate_block_body(&mut self, orig_block: Block, state: &mut PointState, new_block: Block) {
        // Reused below for each instruction.
        let mut arg_abs_values = vec![];
        let mut arg_values = vec![];

        for &inst in &self.generic.blocks[orig_block].insts {
            let input_ctx = state.context;
            if let Some((result_value, result_abs)) = match &self.generic.values[inst] {
                ValueDef::Alias(_) => {
                    // Don't generate any new code; uses will be
                    // rewritten. (We resolve aliases when
                    // transcribing to specialized blocks, in other
                    // words.)
                    None
                }
                ValueDef::PickOutput(val, idx, ty) => {
                    // Directly transcribe.
                    let (val, _) = self.use_value(state.context, *val);
                    Some((
                        ValueDef::PickOutput(val, *idx, *ty),
                        AbstractValue::Runtime(ValueTags::default()),
                    ))
                }
                ValueDef::Operator(op, args, tys) => {
                    // Collect AbstractValues for args.
                    arg_abs_values.clear();
                    arg_values.clear();
                    for &arg in args {
                        let arg = self.generic.resolve_alias(arg);
                        let (val, abs) = self.use_value(state.context, arg);
                        arg_abs_values.push(abs);
                        arg_values.push(val);
                    }

                    // Eval the transfer-function for this operator.
                    let (result_abs_value, is_intrinsic_ident) =
                        self.abstract_eval(*op, &arg_abs_values[..], &arg_values[..], state);
                    // Transcribe either the original operation, or a
                    // constant, to the output.

                    match (is_intrinsic_ident, result_abs_value) {
                        (_, AbstractValue::Top) => unreachable!(),
                        (true, av) => Some((ValueDef::Alias(arg_values[0]), av)),
                        (_, AbstractValue::Concrete(bits, t)) if tys.len() == 1 => {
                            if let Some(const_op) = const_operator(tys[0], bits) {
                                Some((
                                    ValueDef::Operator(const_op, vec![], tys.clone()),
                                    AbstractValue::Concrete(bits, t),
                                ))
                            } else {
                                Some((
                                    ValueDef::Operator(
                                        *op,
                                        std::mem::take(&mut arg_values),
                                        tys.clone(),
                                    ),
                                    AbstractValue::Runtime(t),
                                ))
                            }
                        }
                        (_, av) => Some((
                            ValueDef::Operator(*op, std::mem::take(&mut arg_values), tys.clone()),
                            AbstractValue::Runtime(av.tags()),
                        )),
                    }
                }
                _ => unreachable!(
                    "Invalid ValueDef in `insts` array for {} at {}",
                    orig_block, inst
                ),
            } {
                let result_value = self.func.add_value(result_value);
                self.value_map.insert((input_ctx, inst), result_value);
                self.func.blocks[new_block].insts.push(result_value);

                self.def_value(orig_block, input_ctx, inst, result_value, result_abs);
            }
        }
    }

    fn meet_into_block_entry(
        &mut self,
        block: Block,
        context: Context,
        state: &ProgPointState,
    ) -> bool {
        match self.state.state[context].block_entry.entry(block) {
            BTreeEntry::Vacant(v) => {
                v.insert(state.clone());
                true
            }
            BTreeEntry::Occupied(mut o) => o.get_mut().meet_with(state),
        }
    }

    fn target_block(&mut self, state: &PointState, orig_block: Block, target: Block) -> Block {
        match self.block_map.entry((state.context, target)) {
            HashEntry::Vacant(v) => {
                let block = self.func.add_block();
                for (ty, _) in &self.generic.blocks[target].params {
                    self.func.add_blockparam(block, *ty);
                }
                self.queue_set.insert((target, state.context));
                self.queue.push((target, state.context, block));
                self.state.state[state.context]
                    .block_entry
                    .insert(target, state.flow.clone());
                *v.insert(block)
            }
            HashEntry::Occupied(o) => {
                let target_specialized = *o.get();
                let changed = self.meet_into_block_entry(target, state.context, &state.flow);
                // If we *don't* dominate this block and input changed, then re-enqueue.
                if changed && !self.cfg.dominates(orig_block, target) {
                    if self.queue_set.insert((target, state.context)) {
                        self.queue.push((target, state.context, target_specialized));
                    }
                }
                target_specialized
            }
        }
    }

    fn evaluate_block_target(
        &mut self,
        orig_block: Block,
        state: &PointState,
        target: &BlockTarget,
    ) -> BlockTarget {
        let mut args = vec![];
        for (blockparam, arg) in self.generic.blocks[target.block]
            .params
            .iter()
            .map(|(_, val)| *val)
            .zip(target.args.iter().copied())
        {
            let (val, abs) = self.use_value(state.context, arg);
            args.push(val);
            self.def_value(orig_block, state.context, blockparam, val, abs);
        }

        let target_block = self.target_block(state, orig_block, target.block);
        BlockTarget {
            block: target_block,
            args,
        }
    }

    fn evaluate_term(&mut self, orig_block: Block, state: &mut PointState, new_block: Block) {
        let new_term = match &self.generic.blocks[orig_block].terminator {
            &Terminator::None => Terminator::None,
            &Terminator::CondBr {
                cond,
                ref if_true,
                ref if_false,
            } => {
                let (cond, abs_cond) = self.use_value(state.context, cond);
                match abs_cond.is_const_truthy() {
                    Some(true) => Terminator::Br {
                        target: self.evaluate_block_target(orig_block, state, if_true),
                    },
                    Some(false) => Terminator::Br {
                        target: self.evaluate_block_target(orig_block, state, if_false),
                    },
                    None => Terminator::CondBr {
                        cond,
                        if_true: self.evaluate_block_target(orig_block, state, if_true),
                        if_false: self.evaluate_block_target(orig_block, state, if_false),
                    },
                }
            }
            &Terminator::Br { ref target } => Terminator::Br {
                target: self.evaluate_block_target(orig_block, state, target),
            },
            &Terminator::Select {
                value,
                ref targets,
                ref default,
            } => {
                let (value, abs_value) = self.use_value(state.context, value);
                if let Some(selector) = abs_value.is_const_u32() {
                    let selector = selector as usize;
                    let target = if selector < targets.len() {
                        &targets[selector]
                    } else {
                        default
                    };
                    Terminator::Br {
                        target: self.evaluate_block_target(orig_block, state, target),
                    }
                } else {
                    let targets = targets
                        .iter()
                        .map(|target| self.evaluate_block_target(orig_block, state, target))
                        .collect::<Vec<_>>();
                    let default = self.evaluate_block_target(orig_block, state, default);
                    Terminator::Select {
                        value,
                        targets,
                        default,
                    }
                }
            }
            &Terminator::Return { ref values } => {
                let values = values
                    .iter()
                    .map(|&value| self.use_value(state.context, value).0)
                    .collect::<Vec<_>>();
                Terminator::Return { values }
            }
            &Terminator::Unreachable => Terminator::Unreachable,
        };
        self.func.blocks[new_block].terminator = new_term;
    }

    fn abstract_eval(
        &mut self,
        op: Operator,
        abs: &[AbstractValue],
        values: &[Value],
        state: &mut PointState,
    ) -> (AbstractValue, bool) {
        debug_assert_eq!(abs.len(), values.len());

        if let Some(ret) = self.abstract_eval_intrinsic(op, abs, values, state) {
            return (ret, true);
        }

        let ret = match abs.len() {
            0 => self.abstract_eval_nullary(op, state),
            1 => self.abstract_eval_unary(op, abs[0], values[0], state),
            2 => self.abstract_eval_binary(op, abs[0], abs[1], values[0], values[1], state),
            3 => self.abstract_eval_ternary(
                op, abs[0], abs[1], abs[2], values[0], values[1], values[2], state,
            ),
            _ => AbstractValue::Runtime(ValueTags::default()),
        };
        (ret, false)
    }

    fn abstract_eval_intrinsic(
        &mut self,
        op: Operator,
        abs: &[AbstractValue],
        _values: &[Value],
        state: &mut PointState,
    ) -> Option<AbstractValue> {
        match op {
            Operator::Call { function_index } => {
                if Some(function_index) == self.intrinsics.assume_const_memory {
                    Some(abs[0].with_tags(ValueTags::const_memory()))
                } else if Some(function_index) == self.intrinsics.loop_pc32 {
                    let pc = abs[0].is_const_u32().map(|pc| pc as u64);
                    state.context = self
                        .state
                        .contexts
                        .create(Some(state.context), ContextElem(pc));
                    Some(abs[0])
                } else if Some(function_index) == self.intrinsics.loop_pc64 {
                    let pc = abs[0].is_const_u64();
                    state.context = self
                        .state
                        .contexts
                        .create(Some(state.context), ContextElem(pc));
                    Some(abs[0])
                } else if Some(function_index) == self.intrinsics.loop_end {
                    state.context = self.state.contexts.parent(state.context);
                    Some(AbstractValue::Runtime(ValueTags::default()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn abstract_eval_nullary(&mut self, op: Operator, state: &mut PointState) -> AbstractValue {
        match op {
            Operator::GlobalGet { global_index } => state
                .flow
                .globals
                .get(&global_index)
                .cloned()
                .unwrap_or(AbstractValue::Runtime(ValueTags::default())),
            Operator::I32Const { .. }
            | Operator::I64Const { .. }
            | Operator::F32Const { .. }
            | Operator::F64Const { .. } => {
                AbstractValue::Concrete(WasmVal::try_from(op).unwrap(), ValueTags::default())
            }
            _ => AbstractValue::Runtime(ValueTags::default()),
        }
    }

    fn abstract_eval_unary(
        &mut self,
        op: Operator,
        x: AbstractValue,
        _x_val: Value,
        state: &mut PointState,
    ) -> AbstractValue {
        match (op, x) {
            (Operator::GlobalSet { global_index }, av) => {
                state.flow.globals.insert(global_index, av);
                AbstractValue::Runtime(ValueTags::default())
            }
            (Operator::I32Eqz, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                AbstractValue::Concrete(WasmVal::I32(if k == 0 { 1 } else { 0 }), t)
            }
            (Operator::I64Eqz, AbstractValue::Concrete(WasmVal::I64(k), t)) => {
                AbstractValue::Concrete(WasmVal::I64(if k == 0 { 1 } else { 0 }), t)
            }
            (Operator::I32Extend8S, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                AbstractValue::Concrete(WasmVal::I32(k as i8 as i32 as u32), t)
            }
            (Operator::I32Extend16S, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                AbstractValue::Concrete(WasmVal::I32(k as i16 as i32 as u32), t)
            }
            (Operator::I64Extend8S, AbstractValue::Concrete(WasmVal::I64(k), t)) => {
                AbstractValue::Concrete(WasmVal::I64(k as i8 as i64 as u64), t)
            }
            (Operator::I64Extend16S, AbstractValue::Concrete(WasmVal::I64(k), t)) => {
                AbstractValue::Concrete(WasmVal::I64(k as i16 as i64 as u64), t)
            }
            (Operator::I64Extend32S, AbstractValue::Concrete(WasmVal::I64(k), t)) => {
                AbstractValue::Concrete(WasmVal::I64(k as i32 as i64 as u64), t)
            }
            (Operator::I32Clz, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                AbstractValue::Concrete(WasmVal::I32(k.leading_zeros()), t)
            }
            (Operator::I64Clz, AbstractValue::Concrete(WasmVal::I64(k), t)) => {
                AbstractValue::Concrete(WasmVal::I64(k.leading_zeros() as u64), t)
            }
            (Operator::I32Ctz, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                AbstractValue::Concrete(WasmVal::I32(k.trailing_zeros()), t)
            }
            (Operator::I64Ctz, AbstractValue::Concrete(WasmVal::I64(k), t)) => {
                AbstractValue::Concrete(WasmVal::I64(k.trailing_zeros() as u64), t)
            }
            (Operator::I32Popcnt, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                AbstractValue::Concrete(WasmVal::I32(k.count_ones()), t)
            }
            (Operator::I64Popcnt, AbstractValue::Concrete(WasmVal::I64(k), t)) => {
                AbstractValue::Concrete(WasmVal::I64(k.count_ones() as u64), t)
            }
            (Operator::I32WrapI64, AbstractValue::Concrete(WasmVal::I64(k), t)) => {
                AbstractValue::Concrete(WasmVal::I32(k as u32), t)
            }
            (Operator::I64ExtendI32S, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                AbstractValue::Concrete(WasmVal::I64(k as i32 as i64 as u64), t)
            }
            (Operator::I64ExtendI32U, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                AbstractValue::Concrete(WasmVal::I64(k as u64), t)
            }

            (Operator::I32Load { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I32Load8U { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I32Load8S { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I32Load16U { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I32Load16S { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
                if t.contains(ValueTags::const_memory()) =>
            {
                let size = match op {
                    Operator::I32Load { .. } => 4,
                    Operator::I32Load8U { .. } => 1,
                    Operator::I32Load8S { .. } => 1,
                    Operator::I32Load16U { .. } => 2,
                    Operator::I32Load16S { .. } => 2,
                    _ => unreachable!(),
                };
                let conv = |x: u64| match op {
                    Operator::I32Load { .. } => x as u32,
                    Operator::I32Load8U { .. } => x as u8 as u32,
                    Operator::I32Load8S { .. } => x as i8 as i32 as u32,
                    Operator::I32Load16U { .. } => x as u16 as u32,
                    Operator::I32Load16S { .. } => x as i16 as i32 as u32,
                    _ => unreachable!(),
                };

                self.image
                    .read_size(memory.memory, k + memory.offset as u32, size)
                    .map(|data| AbstractValue::Concrete(WasmVal::I32(conv(data)), t))
                    .unwrap_or(AbstractValue::Runtime(ValueTags::default()))
            }

            (Operator::I64Load { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I64Load8U { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I64Load8S { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I64Load16U { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I64Load16S { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I64Load32U { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I64Load32S { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
                if t.contains(ValueTags::const_memory()) =>
            {
                let size = match op {
                    Operator::I64Load { .. } => 8,
                    Operator::I64Load8U { .. } => 1,
                    Operator::I64Load8S { .. } => 1,
                    Operator::I64Load16U { .. } => 2,
                    Operator::I64Load16S { .. } => 2,
                    Operator::I64Load32U { .. } => 4,
                    Operator::I64Load32S { .. } => 4,
                    _ => unreachable!(),
                };
                let conv = |x: u64| match op {
                    Operator::I64Load { .. } => x,
                    Operator::I64Load8U { .. } => x as u8 as u64,
                    Operator::I64Load8S { .. } => x as i8 as i64 as u64,
                    Operator::I64Load16U { .. } => x as u16 as u64,
                    Operator::I64Load16S { .. } => x as i16 as i64 as u64,
                    Operator::I64Load32U { .. } => x as u32 as u64,
                    Operator::I64Load32S { .. } => x as i32 as i64 as u64,
                    _ => unreachable!(),
                };

                self.image
                    .read_size(memory.memory, k + memory.offset as u32, size)
                    .map(|data| AbstractValue::Concrete(WasmVal::I64(conv(data)), t))
                    .unwrap_or(AbstractValue::Runtime(ValueTags::default()))
            }

            // TODO: FP and SIMD
            _ => AbstractValue::Runtime(ValueTags::default()),
        }
    }

    fn abstract_eval_binary(
        &mut self,
        op: Operator,
        x: AbstractValue,
        y: AbstractValue,
        _x_val: Value,
        _y_val: Value,
        _state: &mut PointState,
    ) -> AbstractValue {
        match (x, y) {
            (AbstractValue::Concrete(v1, tag1), AbstractValue::Concrete(v2, tag2)) => {
                let tags = tag1.meet(tag2);
                match (op, v1, v2) {
                    // 32-bit comparisons.
                    (Operator::I32Eq, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 == k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I32Ne, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 != k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I32LtS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I32(if (k1 as i32) < (k2 as i32) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I32LtU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 < k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I32GtS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I32(if (k1 as i32) > (k2 as i32) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I32GtU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 > k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I32LeS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I32(if (k1 as i32) <= (k2 as i32) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I32LeU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 <= k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I32GeS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I32(if (k1 as i32) >= (k2 as i32) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I32GeU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 >= k2 { 1 } else { 0 }), tags)
                    }

                    // 64-bit comparisons.
                    (Operator::I64Eq, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 == k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I64Ne, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 != k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I64LtS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(if (k1 as i64) < (k2 as i64) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I64LtU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 < k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I64GtS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(if (k1 as i64) > (k2 as i64) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I64GtU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 > k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I64LeS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(if (k1 as i64) <= (k2 as i64) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I64LeU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 <= k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I64GeS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(if (k1 as i64) >= (k2 as i64) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I64GeU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 >= k2 { 1 } else { 0 }), tags)
                    }

                    // 32-bit integer arithmetic.
                    (Operator::I32Add, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_add(k2)), tags)
                    }
                    (Operator::I32Sub, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_sub(k2)), tags)
                    }
                    (Operator::I32Mul, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_mul(k2)), tags)
                    }
                    (Operator::I32DivU, WasmVal::I32(k1), WasmVal::I32(k2)) if k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_div(k2)), tags)
                    }
                    (Operator::I32DivS, WasmVal::I32(k1), WasmVal::I32(k2))
                        if k2 != 0 && (k1 != 0x8000_0000 || k2 != 0xffff_ffff) =>
                    {
                        AbstractValue::Concrete(
                            WasmVal::I32((k1 as i32).wrapping_div(k2 as i32) as u32),
                            tags,
                        )
                    }
                    (Operator::I32RemU, WasmVal::I32(k1), WasmVal::I32(k2)) if k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_rem(k2)), tags)
                    }
                    (Operator::I32RemS, WasmVal::I32(k1), WasmVal::I32(k2))
                        if k2 != 0 && (k1 != 0x8000_0000 || k2 != 0xffff_ffff) =>
                    {
                        AbstractValue::Concrete(
                            WasmVal::I32((k1 as i32).wrapping_rem(k2 as i32) as u32),
                            tags,
                        )
                    }
                    (Operator::I32And, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1 & k2), tags)
                    }
                    (Operator::I32Or, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1 | k2), tags)
                    }
                    (Operator::I32Xor, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1 ^ k2), tags)
                    }
                    (Operator::I32Shl, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_shl(k2 & 0x1f)), tags)
                    }
                    (Operator::I32ShrU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_shr(k2 & 0x1f)), tags)
                    }
                    (Operator::I32ShrS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I32((k1 as i32).wrapping_shr(k2 & 0x1f) as u32),
                            tags,
                        )
                    }
                    (Operator::I32Rotl, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        let amt = k2 & 0x1f;
                        let result = k1.wrapping_shl(amt) | k1.wrapping_shr(32 - amt);
                        AbstractValue::Concrete(WasmVal::I32(result), tags)
                    }
                    (Operator::I32Rotr, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        let amt = k2 & 0x1f;
                        let result = k1.wrapping_shr(amt) | k1.wrapping_shl(32 - amt);
                        AbstractValue::Concrete(WasmVal::I32(result), tags)
                    }

                    // 64-bit integer arithmetic.
                    (Operator::I64Add, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_add(k2)), tags)
                    }
                    (Operator::I64Sub, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_sub(k2)), tags)
                    }
                    (Operator::I64Mul, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_mul(k2)), tags)
                    }
                    (Operator::I64DivU, WasmVal::I64(k1), WasmVal::I64(k2)) if k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_div(k2)), tags)
                    }
                    (Operator::I64DivS, WasmVal::I64(k1), WasmVal::I64(k2))
                        if k2 != 0
                            && (k1 != 0x8000_0000_0000_0000 || k2 != 0xffff_ffff_ffff_ffff) =>
                    {
                        AbstractValue::Concrete(
                            WasmVal::I64((k1 as i64).wrapping_div(k2 as i64) as u64),
                            tags,
                        )
                    }
                    (Operator::I64RemU, WasmVal::I64(k1), WasmVal::I64(k2)) if k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_rem(k2)), tags)
                    }
                    (Operator::I64RemS, WasmVal::I64(k1), WasmVal::I64(k2))
                        if k2 != 0
                            && (k1 != 0x8000_0000_0000_0000 || k2 != 0xffff_ffff_ffff_ffff) =>
                    {
                        AbstractValue::Concrete(
                            WasmVal::I64((k1 as i64).wrapping_rem(k2 as i64) as u64),
                            tags,
                        )
                    }
                    (Operator::I64And, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1 & k2), tags)
                    }
                    (Operator::I64Or, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1 | k2), tags)
                    }
                    (Operator::I64Xor, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1 ^ k2), tags)
                    }
                    (Operator::I64Shl, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(k1.wrapping_shl((k2 & 0x3f) as u32)),
                            tags,
                        )
                    }
                    (Operator::I64ShrU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(k1.wrapping_shr((k2 & 0x3f) as u32)),
                            tags,
                        )
                    }
                    (Operator::I64ShrS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64((k1 as i64).wrapping_shr((k2 & 0x3f) as u32) as u64),
                            tags,
                        )
                    }
                    (Operator::I64Rotl, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        let amt = (k2 & 0x3f) as u32;
                        let result = k1.wrapping_shl(amt) | k1.wrapping_shr(64 - amt);
                        AbstractValue::Concrete(WasmVal::I64(result), tags)
                    }
                    (Operator::I64Rotr, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        let amt = (k2 & 0x3f) as u32;
                        let result = k1.wrapping_shr(amt) | k1.wrapping_shl(64 - amt);
                        AbstractValue::Concrete(WasmVal::I64(result), tags)
                    }

                    // TODO: FP and SIMD ops.
                    _ => AbstractValue::Runtime(ValueTags::default()),
                }
            }
            _ => AbstractValue::Runtime(ValueTags::default()),
        }
    }

    fn abstract_eval_ternary(
        &mut self,
        op: Operator,
        x: AbstractValue,
        y: AbstractValue,
        z: AbstractValue,
        _x_val: Value,
        _y_val: Value,
        _z_val: Value,
        _state: &mut PointState,
    ) -> AbstractValue {
        match (op, x) {
            (Operator::Select, AbstractValue::Concrete(v, _t))
            | (Operator::TypedSelect { .. }, AbstractValue::Concrete(v, _t)) => {
                if v.is_truthy() {
                    y
                } else {
                    z
                }
            }
            _ => AbstractValue::Runtime(ValueTags::default()),
        }
    }
}
