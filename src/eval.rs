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
    btree_map::Entry as BTreeEntry, hash_map::Entry as HashEntry, HashMap, HashSet, VecDeque,
};
use waffle::cfg::CFGInfo;
use waffle::entity::EntityRef;
use waffle::{
    Block, BlockTarget, Func, FunctionBody, Module, Operator, Table, Terminator, Type, Value,
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
    /// Dependencies for updates: some use in a given block with a
    /// given context occurs of a value defined in another block at
    /// another context.
    block_deps: HashMap<(Context, Block), HashSet<(Context, Block)>>,
    /// Map of (ctx, value_in_generic) to specialized value_in_func.
    value_map: HashMap<(Context, Value), Value>,
    /// Map of (ctx, block, sym_addr) to blockparam value.
    mem_blockparam_map: HashMap<(Context, Block, SymbolicAddr), Value>,
    /// Queue of blocks to (re)compute. List of (block_in_generic,
    /// ctx, block_in_func).
    queue: VecDeque<(Block, Context, Block)>,
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
        log::info!("Processing directive {:?}", directive);
        if let Some(idx) = partially_evaluate_func(module, im, &intrinsics, directive)? {
            // Append to table.
            let func_table = module.table_mut(Table::from(0));
            let table_idx = {
                let func_table_elts = func_table.func_elements.as_mut().unwrap();
                let table_idx = func_table_elts.len();
                func_table_elts.push(idx);
                table_idx
            } as u32;
            if func_table.max.is_some() && table_idx >= func_table.max.unwrap() {
                func_table.max = Some(table_idx + 1);
            }
            log::info!("New func index {} -> table index {}", idx, table_idx);
            log::info!(" -> writing to 0x{:x}", directive.func_index_out_addr);
            // Update memory image.
            mem_updates.insert(directive.func_index_out_addr, table_idx);
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
) -> anyhow::Result<Option<Func>> {
    // Get function body.
    let body = module
        .func(directive.func)
        .body()
        .ok_or_else(|| anyhow::anyhow!("Attempt to specialize an import"))?;
    let sig = module.func(directive.func).sig();

    log::trace!("Specializing: {}", directive.func);
    log::trace!("body:\n{}", body.display("| "));

    // Compute CFG info.
    let cfg = CFGInfo::new(body);

    log::trace!("CFGInfo: {:?}", cfg);

    // Build the evaluator.
    let mut evaluator = Evaluator {
        generic: body,
        intrinsics,
        image,
        cfg,
        state: FunctionState::new(),
        func: FunctionBody::new(module, sig),
        block_map: HashMap::new(),
        block_deps: HashMap::new(),
        value_map: HashMap::new(),
        mem_blockparam_map: HashMap::new(),
        queue: VecDeque::new(),
        queue_set: HashSet::new(),
    };
    let (ctx, entry_state) = evaluator
        .state
        .init_args(body, image, &directive.const_params[..]);
    log::trace!("after init_args, state is {:?}", evaluator.state);
    let specialized_entry = evaluator.create_block(evaluator.generic.entry, ctx, entry_state);
    evaluator.func.entry = specialized_entry;
    evaluator
        .queue
        .push_back((evaluator.generic.entry, ctx, specialized_entry));
    evaluator.queue_set.insert((evaluator.generic.entry, ctx));
    evaluator.evaluate()?;

    log::debug!("Adding func:\n{}", evaluator.func.display("| "));
    let func = module.add_func(sig, evaluator.func);
    Ok(Some(func))
}

fn const_operator(ty: Type, value: WasmVal) -> Option<Operator> {
    match (ty, value) {
        (Type::I32, WasmVal::I32(k)) => Some(Operator::I32Const { value: k }),
        (Type::I64, WasmVal::I64(k)) => Some(Operator::I64Const { value: k }),
        (Type::F32, WasmVal::F32(k)) => Some(Operator::F32Const { value: k }),
        (Type::F64, WasmVal::F64(k)) => Some(Operator::F64Const { value: k }),
        _ => None,
    }
}

enum EvalResult {
    Unhandled,
    Elide,
    Alias(AbstractValue, Value),
    Normal(AbstractValue),
}
impl EvalResult {
    fn is_handled(&self) -> bool {
        match self {
            &EvalResult::Unhandled => false,
            _ => true,
        }
    }
}

impl<'a> Evaluator<'a> {
    fn evaluate(&mut self) -> anyhow::Result<()> {
        while let Some((orig_block, ctx, new_block)) = self.queue.pop_front() {
            self.queue_set.remove(&(orig_block, ctx));
            self.evaluate_block(orig_block, ctx, new_block)?;
        }
        self.finalize()?;
        Ok(())
    }

    fn evaluate_block(
        &mut self,
        orig_block: Block,
        ctx: Context,
        new_block: Block,
    ) -> anyhow::Result<()> {
        // Clear the block body each time we rebuild it -- we may be
        // recomputing a specialization with an existing output.
        self.func.blocks[new_block].insts.clear();

        log::trace!(
            "evaluate_block: orig {} ctx {} new {}",
            orig_block,
            ctx,
            new_block
        );
        debug_assert_eq!(self.block_map.get(&(ctx, orig_block)), Some(&new_block));

        // Create program-point state.
        let mut state = PointState {
            context: ctx,
            pending_context: None,
            flow: self.state.state[ctx]
                .block_entry
                .get(&orig_block)
                .cloned()
                .unwrap(),
        };
        log::trace!(" -> state = {:?}", state);

        state.flow.update_at_block_entry(&mut |addr, ty| {
            *self
                .mem_blockparam_map
                .entry((ctx, orig_block, addr))
                .or_insert_with(|| {
                    let param = self.func.add_placeholder(ty);
                    log::trace!(
                        "new blockparam {} for addr {:?} on block {}",
                        param,
                        addr,
                        new_block
                    );
                    param
                })
        });

        // Do the actual constant-prop, carrying the state across the
        // block and updating flow-sensitive state, and updating SSA
        // vals as well.
        self.evaluate_block_body(orig_block, &mut state, new_block)
            .map_err(|e| {
                e.context(anyhow::anyhow!(
                    "Evaluating block body {} in func:\n{}",
                    orig_block,
                    self.generic.display("| ")
                ))
            })?;

        // Store the exit state at this point for later use.
        self.state.state[ctx]
            .block_exit
            .insert(orig_block, state.flow.clone());

        self.evaluate_term(orig_block, &mut state, new_block);

        Ok(())
    }

    /// For a given value in the generic function, accessed in the
    /// given context and at the given block, find its abstract value
    /// and SSA value in the specialized function.
    fn use_value(
        &mut self,
        context: Context,
        orig_block: Block,
        orig_val: Value,
    ) -> (Value, AbstractValue) {
        log::trace!(
            "using value {} at block {} in context {}",
            orig_val,
            orig_block,
            context
        );
        if let Some(&abs) = self.state.state[context].ssa.values.get(&orig_val) {
            log::trace!(" -> found abstract  value {:?} at context {}", abs, context);
            let &val = self.value_map.get(&(context, orig_val)).unwrap();
            log::trace!(" -> runtime value {}", val);
            return (val, abs);
        }
        panic!(
            "Could not find value for {} in context {}",
            orig_val, context
        );
    }

    fn def_value(
        &mut self,
        block: Block,
        context: Context,
        orig_val: Value,
        val: Value,
        abs: AbstractValue,
    ) -> bool {
        log::trace!(
            "defining val {} in block {} context {} with specialized val {} abs {:?}",
            orig_val,
            block,
            context,
            val,
            abs
        );
        self.value_map.insert((context, orig_val), val);
        match self.state.state[context].ssa.values.entry(orig_val) {
            BTreeEntry::Vacant(v) => {
                v.insert(abs);
                true
            }
            BTreeEntry::Occupied(mut o) => {
                let val_abs = o.get_mut();
                let updated = AbstractValue::meet(*val_abs, abs);
                let changed = updated != *val_abs;
                log::trace!(
                    " -> meet: cur {:?} input {:?} result {:?} (changed: {})",
                    val_abs,
                    abs,
                    updated,
                    changed,
                );
                *val_abs = updated;
                changed
            }
        }
    }

    fn enqueue_block_if_existing(&mut self, orig_block: Block, context: Context) {
        if let Some(block) = self.block_map.get(&(context, orig_block)).copied() {
            if self.queue_set.insert((orig_block, context)) {
                self.queue.push_back((orig_block, context, block));
            }
        }
    }

    fn evaluate_block_body(
        &mut self,
        orig_block: Block,
        state: &mut PointState,
        new_block: Block,
    ) -> anyhow::Result<()> {
        // Reused below for each instruction.
        let mut arg_abs_values = vec![];
        let mut arg_values = vec![];

        log::debug!("evaluate_block_body: {}: state {:?}", orig_block, state);
        for &(_, param) in &self.generic.blocks[orig_block].params {
            let (_, abs) = self.use_value(state.context, orig_block, param);
            log::debug!(" -> param {}: {:?}", param, abs);
        }

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
                    let (val, _) = self.use_value(state.context, orig_block, *val);
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
                        let (val, abs) = self.use_value(state.context, orig_block, arg);
                        arg_abs_values.push(abs);
                        arg_values.push(val);
                    }

                    // Eval the transfer-function for this operator.
                    let result = self.abstract_eval(
                        orig_block,
                        inst,
                        *op,
                        &arg_abs_values[..],
                        &arg_values[..],
                        state,
                    )?;
                    // Transcribe either the original operation, or a
                    // constant, to the output.

                    match result {
                        EvalResult::Unhandled => unreachable!(),
                        EvalResult::Alias(av, val) => Some((ValueDef::Alias(val), av)),
                        EvalResult::Elide => None,
                        EvalResult::Normal(AbstractValue::Concrete(bits, t)) if tys.len() == 1 => {
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
                        EvalResult::Normal(av) => Some((
                            ValueDef::Operator(*op, std::mem::take(&mut arg_values), tys.clone()),
                            av,
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
                self.func.append_to_block(new_block, result_value);

                self.def_value(orig_block, input_ctx, inst, result_value, result_abs);
            }
        }

        Ok(())
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

    fn create_block(
        &mut self,
        orig_block: Block,
        context: Context,
        state: ProgPointState,
    ) -> Block {
        let block = self.func.add_block();
        log::trace!(
            "create_block: orig_block {} context {} -> {}",
            orig_block,
            context,
            block
        );
        for &(ty, param) in &self.generic.blocks[orig_block].params {
            let new_param = self.func.add_blockparam(block, ty);
            log::trace!(" -> blockparam {} maps to {}", param, new_param);
            self.value_map.insert((context, param), new_param);
        }
        self.block_map.insert((context, orig_block), block);
        self.state.state[context]
            .block_entry
            .insert(orig_block, state);
        block
    }

    fn target_block(
        &mut self,
        state: &PointState,
        orig_block: Block,
        target: Block,
    ) -> (Block, Context) {
        log::trace!(
            "targeting block {} from {}, in context {}",
            target,
            orig_block,
            state.context
        );

        let target_context = state.pending_context.unwrap_or(state.context);
        log::trace!(" -> new context {}", target_context);

        match self.block_map.entry((target_context, target)) {
            HashEntry::Vacant(_) => {
                let block = self.create_block(target, target_context, state.flow.clone());
                log::trace!(" -> created block {}", block);
                self.block_map.insert((target_context, target), block);
                self.queue_set.insert((target, target_context));
                self.queue.push_back((target, target_context, block));
                (block, target_context)
            }
            HashEntry::Occupied(o) => {
                let target_specialized = *o.get();
                log::trace!(" -> already existing block {}", target_specialized);
                let changed = self.meet_into_block_entry(target, target_context, &state.flow);
                if changed {
                    log::trace!("   -> changed");
                    if self.queue_set.insert((target, target_context)) {
                        self.queue
                            .push_back((target, target_context, target_specialized));
                    }
                }
                (target_specialized, target_context)
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
        let mut abs_args = vec![];
        log::trace!(
            "evaluate target: block {} context {} to {:?}",
            orig_block,
            state.context,
            target
        );

        let (target_block, target_ctx) = self.target_block(state, orig_block, target.block);

        for &arg in &target.args {
            let arg = self.generic.resolve_alias(arg);
            let (val, abs) = self.use_value(state.context, orig_block, arg);
            args.push(val);
            abs_args.push(abs);
            log::debug!(
                "blockparam: block {} context {}: arg {} has val {} abs {:?}",
                orig_block,
                state.context,
                arg,
                val,
                abs
            );
        }

        // Parallel-move semantics: read all uses above, then write
        // all defs below.
        let mut changed = false;
        for (blockparam, abs) in self.generic.blocks[target.block]
            .params
            .iter()
            .map(|(_, val)| *val)
            .zip(abs_args.iter())
        {
            let &val = self.value_map.get(&(target_ctx, blockparam)).unwrap();
            log::debug!(
                "blockparam: updating with new def: block {} context {} param {} val {} abstract {:?}",
                target.block, target_ctx, blockparam, val, abs);
            changed |= self.def_value(orig_block, target_ctx, blockparam, val, *abs);
        }

        // If blockparam inputs changed, re-enqueue target for evaluation.
        if changed {
            self.enqueue_block_if_existing(target.block, target_ctx);
        }

        BlockTarget {
            block: target_block,
            args,
        }
    }

    fn evaluate_term(&mut self, orig_block: Block, state: &mut PointState, new_block: Block) {
        log::trace!(
            "evaluating terminator: block {} context {} specialized block {}: {:?}",
            orig_block,
            state.context,
            new_block,
            self.generic.blocks[orig_block].terminator
        );
        let new_term = match &self.generic.blocks[orig_block].terminator {
            &Terminator::None => Terminator::None,
            &Terminator::CondBr {
                cond,
                ref if_true,
                ref if_false,
            } => {
                let (cond, abs_cond) = self.use_value(state.context, orig_block, cond);
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
                let (value, abs_value) = self.use_value(state.context, orig_block, value);
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
                    .map(|&value| self.use_value(state.context, orig_block, value).0)
                    .collect::<Vec<_>>();
                Terminator::Return { values }
            }
            &Terminator::Unreachable => Terminator::Unreachable,
        };
        // Note: we don't use `set_terminator`, because it adds edges;
        // we add edges once, in a separate pass at the end.
        self.func.blocks[new_block].terminator = new_term;
    }

    fn abstract_eval(
        &mut self,
        orig_block: Block,
        orig_inst: Value,
        op: Operator,
        abs: &[AbstractValue],
        values: &[Value],
        state: &mut PointState,
    ) -> anyhow::Result<EvalResult> {
        log::debug!(
            "abstract eval of {} {}: op {:?} abs {:?} state {:?}",
            orig_block,
            orig_inst,
            op,
            abs,
            state
        );

        debug_assert_eq!(abs.len(), values.len());

        let intrinsic_result = self.abstract_eval_intrinsic(orig_block, op, abs, values, state);
        if intrinsic_result.is_handled() {
            return Ok(intrinsic_result);
        }

        let mem_overlay_result =
            self.abstract_eval_mem_overlay(orig_inst, op, abs, values, state)?;
        if mem_overlay_result.is_handled() {
            return Ok(mem_overlay_result);
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

        Ok(EvalResult::Normal(ret))
    }

    fn abstract_eval_intrinsic(
        &mut self,
        _orig_block: Block,
        op: Operator,
        abs: &[AbstractValue],
        values: &[Value],
        state: &mut PointState,
    ) -> EvalResult {
        match op {
            Operator::Call { function_index } => {
                if Some(function_index) == self.intrinsics.assume_const_memory {
                    EvalResult::Alias(abs[0].with_tags(ValueTags::const_memory()), values[0])
                } else if Some(function_index) == self.intrinsics.make_symbolic_ptr {
                    let label_index = values[0].index() as u32;
                    log::trace!(
                        "value {} is getting symbolic label {}",
                        values[0],
                        label_index
                    );
                    EvalResult::Alias(AbstractValue::SymbolicPtr(label_index, 0), values[0])
                } else if Some(function_index) == self.intrinsics.push_context {
                    let pc = abs[0].is_const_u32();
                    let instantaneous_context = state.pending_context.unwrap_or(state.context);
                    let child = self
                        .state
                        .contexts
                        .create(Some(instantaneous_context), ContextElem::Loop(pc));
                    state.pending_context = Some(child);
                    log::trace!("push context (pc {:?}): now {}", pc, child);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.pop_context {
                    let instantaneous_context = state.pending_context.unwrap_or(state.context);
                    let parent = match self.state.contexts.leaf_element(instantaneous_context) {
                        ContextElem::Root => instantaneous_context,
                        _ => self.state.contexts.parent(instantaneous_context),
                    };
                    state.pending_context = Some(parent);
                    log::trace!("pop context: now {}", parent);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.update_context {
                    let pc = abs[0].is_const_u32();
                    let instantaneous_context = state.pending_context.unwrap_or(state.context);
                    let sibling = match self.state.contexts.leaf_element(instantaneous_context) {
                        ContextElem::Root => instantaneous_context,
                        _ => self.state.contexts.create(
                            Some(self.state.contexts.parent(instantaneous_context)),
                            ContextElem::Loop(pc),
                        ),
                    };
                    state.pending_context = Some(sibling);
                    log::trace!("update context (pc {:?}): now {}", pc, sibling);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.flush_to_mem {
                    // Just elide it for now. TODO: handle properly.
                    EvalResult::Elide
                } else {
                    EvalResult::Unhandled
                }
            }
            _ => EvalResult::Unhandled,
        }
    }

    fn abstract_eval_mem_overlay(
        &mut self,
        inst: Value,
        op: Operator,
        abs: &[AbstractValue],
        vals: &[Value],
        state: &mut PointState,
    ) -> anyhow::Result<EvalResult> {
        match (op, abs) {
            (
                Operator::I32Load { memory } | Operator::I64Load { memory },
                &[AbstractValue::SymbolicPtr(label, off)],
            ) => {
                let off = off + (memory.offset as i64);
                let expected_ty = match op {
                    Operator::I32Load { .. } => Type::I32,
                    Operator::I64Load { .. } => Type::I64,
                    _ => anyhow::bail!("Bad load type to symbolic ptr"),
                };
                log::trace!("load from symbolic loc {:?}", abs[0]);
                match state.flow.mem_overlay.get(&SymbolicAddr(label, off)) {
                    Some(MemValue::Value { data, ty, .. }) if *ty == expected_ty => {
                        let abs = self.state.state[state.context]
                            .ssa
                            .values
                            .get(data)
                            .cloned()
                            .unwrap_or(AbstractValue::default());
                        log::trace!(" -> have value {} with abs {:?}", data, abs);
                        return Ok(EvalResult::Alias(abs, *data));
                    }
                    Some(MemValue::Value { .. }) => {
                        anyhow::bail!("Type punning in memory renaming");
                    }
                    Some(MemValue::Conflict) | None => {
                        anyhow::bail!(" -> escaped or no value");
                    }
                }
            }
            (
                Operator::I32Store { memory } | Operator::I64Store { memory },
                &[AbstractValue::SymbolicPtr(label, off), _],
            ) => {
                let off = off + (memory.offset as i64);
                let data_ty = match op {
                    Operator::I32Store { .. } => Type::I32,
                    Operator::I64Store { .. } => Type::I64,
                    _ => anyhow::bail!("Bad store type to symbolic ptr"),
                };
                log::trace!("store to symbolic loc {:?}: value {}", abs[0], vals[1]);
                // TODO: check for overlapping values
                state.flow.mem_overlay.insert(
                    SymbolicAddr(label, off),
                    MemValue::Value {
                        addr: vals[0],
                        offset: memory.offset,
                        data: vals[1],
                        ty: data_ty,
                    },
                );

                // Elide the store.
                return Ok(EvalResult::Elide);
            }
            (
                Operator::I32Add,
                &[AbstractValue::SymbolicPtr(label, off), AbstractValue::Concrete(WasmVal::I32(k), _)]
                | &[AbstractValue::Concrete(WasmVal::I32(k), _), AbstractValue::SymbolicPtr(label, off)],
            ) => {
                // TODO: check for wraparound
                return Ok(EvalResult::Normal(AbstractValue::SymbolicPtr(
                    label,
                    off + (k as i32 as i64),
                )));
            }
            (
                Operator::I32Sub,
                &[AbstractValue::SymbolicPtr(label, off), AbstractValue::Concrete(WasmVal::I32(k), _)]
                | &[AbstractValue::Concrete(WasmVal::I32(k), _), AbstractValue::SymbolicPtr(label, off)],
            ) => {
                // TODO: check for wraparound
                return Ok(EvalResult::Normal(AbstractValue::SymbolicPtr(
                    label,
                    off - (k as i32 as i64),
                )));
            }
            _ => {}
        }

        // If not handled above, any symbolic ptr value flowing into
        // an operator must cause the label to be marked as escaped.
        for &a in abs {
            if let AbstractValue::SymbolicPtr(_, _) = a {
                log::trace!(
                    "abs {:?} flowing into op {} (args {:?}) causes label escape",
                    a,
                    op,
                    abs,
                );
                anyhow::bail!(
                    "Escaped symbolic pointer! inst {} has inputs {:?}",
                    inst,
                    abs
                );
            }
        }

        Ok(EvalResult::Unhandled)
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
        match (op, z) {
            (Operator::Select, AbstractValue::Concrete(v, _t))
            | (Operator::TypedSelect { .. }, AbstractValue::Concrete(v, _t)) => {
                if v.is_truthy() {
                    x
                } else {
                    y
                }
            }
            _ => AbstractValue::Runtime(ValueTags::default()),
        }
    }

    fn add_blockparam_mem_args(&mut self) -> anyhow::Result<()> {
        // TODO: examine mem_overlay in block input state of each
        // specialized block, and create blockparams for all values
        // that in the end were `BlockParam`.
        //
        // Also check for escaping symbolic pointers: any values that
        // are SymboilcPtr in a block arg but Runtime at blockparam of
        // succ.
        Ok(())
    }

    fn finalize(&mut self) -> anyhow::Result<()> {
        for block in self.func.blocks.iter() {
            debug_assert!(self.func.blocks[block].succs.is_empty());
            let term = self.func.blocks[block].terminator.clone();
            term.visit_successors(|succ| self.func.add_edge(block, succ));
        }

        // Add blockparam args for symbolic addrs to each branch.
        self.add_blockparam_mem_args()?;

        self.func.validate().unwrap();

        Ok(())
    }
}
