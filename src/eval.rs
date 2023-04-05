//! Partial evaluation.

use crate::directive::Directive;
use crate::image::Image;
use crate::intrinsics::Intrinsics;
use crate::state::*;
use crate::value::{AbstractValue, ValueTags, WasmVal};
use crate::Options;
use fxhash::FxHashMap as HashMap;
use fxhash::FxHashSet as HashSet;
use rayon::prelude::*;
use std::collections::{btree_map::Entry as BTreeEntry, hash_map::Entry as HashEntry, VecDeque};
use waffle::cfg::CFGInfo;
use waffle::entity::EntityRef;
use waffle::{
    entity::PerEntity, Block, BlockTarget, FuncDecl, FunctionBody, Module, Operator, Signature,
    SourceLoc, Table, Terminator, Type, Value, ValueDef,
};

struct Evaluator<'a> {
    /// Module.
    module: &'a Module<'a>,
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
    /// Reverse map from specialized block to its original ctx/block.
    block_rev_map: PerEntity<Block, (Context, Block)>,
    /// Dependencies for updates: some use in a given block with a
    /// given context occurs of a value defined in another block at
    /// another context.
    block_deps: HashMap<(Context, Block), HashSet<(Context, Block)>>,
    /// Map of (ctx, value_in_generic) to specialized value_in_func.
    value_map: HashMap<(Context, Value), Value>,
    /// Map of (ctx, block, sym_addr) to blockparams for address and
    /// mem-renmaed value.
    mem_blockparam_map: HashMap<(Context, Block, SymbolicAddr), (Value, Value)>,
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
    opts: &Options,
) -> anyhow::Result<()> {
    let intrinsics = Intrinsics::find(module);
    log::trace!("intrinsics: {:?}", intrinsics);
    let mut mem_updates = HashMap::default();

    let mut funcs = HashMap::default();
    for directive in directives {
        if !funcs.contains_key(&directive.func) {
            let mut f = module.clone_and_expand_body(directive.func)?;
            f.optimize();
            f.convert_to_max_ssa();
            if opts.add_tracing {
                waffle::passes::trace::run(&mut f);
            }
            if opts.run_pre {
                module.replace_body(directive.func, f.clone());
            }
            funcs.insert(directive.func, f);
        }
    }

    if opts.run_pre {
        return Ok(());
    }

    let bodies = directives
        .par_iter()
        .map(|directive| {
            let generic = funcs.get(&directive.func).unwrap();
            partially_evaluate_func(module, generic, im, &intrinsics, directive)
                .map(|tuple| (directive, tuple))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    for (directive, (body, sig, name)) in bodies {
        // Add function to module.
        let func = module.funcs.push(FuncDecl::Body(sig, name, body));
        // Append to table.
        let func_table = &mut module.tables[Table::from(0)];
        let table_idx = {
            let func_table_elts = func_table.func_elements.as_mut().unwrap();
            let table_idx = func_table_elts.len();
            func_table_elts.push(func);
            table_idx
        } as u32;
        if func_table.max.is_some() && table_idx >= func_table.max.unwrap() {
            func_table.max = Some(table_idx + 1);
        }
        log::info!("New func index {} -> table index {}", func, table_idx);
        log::info!(" -> writing to 0x{:x}", directive.func_index_out_addr);
        // Update memory image.
        mem_updates.insert(directive.func_index_out_addr, table_idx);
    }

    // Update memory.
    let heap = im.main_heap()?;
    for (addr, value) in mem_updates {
        im.write_u32(heap, addr, value)?;
    }

    Ok(())
}

fn partially_evaluate_func(
    module: &Module,
    generic: &FunctionBody,
    image: &Image,
    intrinsics: &Intrinsics,
    directive: &Directive,
) -> anyhow::Result<(FunctionBody, Signature, String)> {
    let orig_name = module.funcs[directive.func].name();
    let sig = module.funcs[directive.func].sig();

    log::debug!("Specializing: {}", directive.func);
    log::debug!("body:\n{}", generic.display("| ", Some(module)));

    // Compute CFG info.
    let cfg = CFGInfo::new(generic);

    log::trace!("CFGInfo: {:?}", cfg);

    // Build the evaluator.
    let func = FunctionBody::new(module, sig);
    let mut evaluator = Evaluator {
        module,
        generic,
        intrinsics,
        image,
        cfg,
        state: FunctionState::new(),
        func,
        block_map: HashMap::default(),
        block_rev_map: PerEntity::default(),
        block_deps: HashMap::default(),
        value_map: HashMap::default(),
        mem_blockparam_map: HashMap::default(),
        queue: VecDeque::new(),
        queue_set: HashSet::default(),
    };
    let (ctx, entry_state) = evaluator
        .state
        .init_args(generic, image, &directive.const_params[..]);
    log::trace!("after init_args, state is {:?}", evaluator.state);
    let specialized_entry = evaluator.create_block(evaluator.generic.entry, ctx, entry_state);
    evaluator.func.entry = specialized_entry;
    evaluator
        .queue
        .push_back((evaluator.generic.entry, ctx, specialized_entry));
    evaluator.queue_set.insert((evaluator.generic.entry, ctx));
    evaluator.evaluate()?;

    log::debug!(
        "Adding func:\n{}",
        evaluator.func.display_verbose("| ", Some(module))
    );
    let name = format!("{} (specialized)", orig_name);
    evaluator.func.optimize();
    Ok((evaluator.func, sig, name))
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

fn store_operator(ty: Type) -> Option<Operator> {
    let memory = waffle::MemoryArg {
        memory: waffle::Memory::new(0),
        align: 0,
        offset: 0,
    };
    match ty {
        Type::I32 => Some(Operator::I32Store { memory }),
        Type::I64 => Some(Operator::I64Store { memory }),
        Type::F32 => Some(Operator::F32Store { memory }),
        Type::F64 => Some(Operator::F64Store { memory }),
        _ => None,
    }
}

#[derive(Debug)]
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
        while let Some((orig_block, ctx, new_block)) = self.queue.pop_back() {
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
            pending_context: PendingContext::None,
            flow: self.state.state[ctx]
                .block_entry
                .get(&orig_block)
                .cloned()
                .unwrap(),
        };
        log::trace!(" -> state = {:?}", state);

        state.flow.update_at_block_entry(
            &mut self.mem_blockparam_map,
            &mut |mem_blockparam_map, sym_addr, ty| {
                *mem_blockparam_map
                    .entry((ctx, orig_block, sym_addr))
                    .or_insert_with(|| {
                        let param = self.func.add_placeholder(ty);
                        let addr = self.func.add_placeholder(Type::I32);
                        log::trace!(
                            "new blockparams for data {} addr {:?} for symbolic-addr {:?} on block {} (ctx {} orig {})",
                            param,
                            addr,
                            sym_addr,
                            new_block,
                            ctx,
                            orig_block,
                        );
                        (addr, param)
                    },)
            },
            &mut |mem_blockparam_map, sym_addr| {
                mem_blockparam_map.remove(&(ctx, orig_block, sym_addr));
            })?;

        // Do the actual constant-prop, carrying the state across the
        // block and updating flow-sensitive state, and updating SSA
        // vals as well.
        self.evaluate_block_body(orig_block, &mut state, new_block)
            .map_err(|e| {
                e.context(anyhow::anyhow!(
                    "Evaluating block body {} in func:\n{}",
                    orig_block,
                    self.generic.display("| ", None)
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
        if let Some(abs) = self.state.state[context].ssa.values.get(&orig_val) {
            log::trace!(" -> found abstract  value {:?} at context {}", abs, context);
            let &val = self.value_map.get(&(context, orig_val)).unwrap();
            log::trace!(" -> runtime value {}", val);
            return (val, abs.clone());
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
        log::debug!(
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
                let updated = AbstractValue::meet(val_abs, &abs);
                let changed = updated != *val_abs;
                log::debug!(
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

        log::trace!("evaluate_block_body: {}: state {:?}", orig_block, state);
        for &(_, param) in &self.generic.blocks[orig_block].params {
            let (_, abs) = self.use_value(state.context, orig_block, param);
            log::trace!(" -> param {}: {:?}", param, abs);
        }

        for &inst in &self.generic.blocks[orig_block].insts {
            let input_ctx = state.context;
            log::trace!(
                "inst {} in context {} -> {:?}",
                inst,
                input_ctx,
                self.generic.values[inst]
            );
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
                        AbstractValue::Runtime(Some(inst), ValueTags::default()),
                    ))
                }
                ValueDef::Operator(op, args, tys) => {
                    // Collect AbstractValues for args.
                    arg_abs_values.clear();
                    arg_values.clear();
                    for &arg in args {
                        log::trace!(" * arg {}", arg);
                        let arg = self.generic.resolve_alias(arg);
                        log::trace!(" -> resolves to arg {}", arg);
                        let (val, abs) = self.use_value(state.context, orig_block, arg);
                        arg_abs_values.push(abs);
                        arg_values.push(val);
                    }
                    let loc = self.generic.source_locs[inst];

                    // Eval the transfer-function for this operator.
                    let result = self.abstract_eval(
                        orig_block,
                        new_block,
                        inst,
                        *op,
                        loc,
                        &arg_abs_values[..],
                        &arg_values[..],
                        &args[..],
                        &tys[..],
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
                                    AbstractValue::Runtime(None, t),
                                ))
                            }
                        }
                        EvalResult::Normal(av) => Some((
                            ValueDef::Operator(*op, std::mem::take(&mut arg_values), tys.clone()),
                            av,
                        )),
                    }
                }
                ValueDef::Trace(id, args) => {
                    let mut arg_values = vec![];
                    for &arg in args {
                        let arg = self.generic.resolve_alias(arg);
                        let (val, _abs) = self.use_value(state.context, orig_block, arg);
                        arg_values.push(val);
                    }
                    Some((
                        ValueDef::Trace(*id, std::mem::take(&mut arg_values)),
                        AbstractValue::Runtime(None, ValueTags::default()),
                    ))
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
        let mut state = state.clone();
        state.update_across_edge();

        match self.state.state[context].block_entry.entry(block) {
            BTreeEntry::Vacant(v) => {
                v.insert(state);
                true
            }
            BTreeEntry::Occupied(mut o) => o.get_mut().meet_with(&state),
        }
    }

    fn context_desc(&self, ctx: Context) -> String {
        match self.state.contexts.leaf_element(ctx) {
            ContextElem::Root => "root".to_owned(),
            ContextElem::Loop(pc) => format!("PC {:?}", pc),
        }
    }

    fn create_block(
        &mut self,
        orig_block: Block,
        context: Context,
        mut state: ProgPointState,
    ) -> Block {
        state.update_across_edge();
        let block = self.func.add_block();
        self.func.blocks[block].desc = format!(
            "Orig {} ctx {} ({})",
            orig_block,
            context,
            self.context_desc(context)
        );
        log::debug!(
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
        self.block_rev_map[block] = (context, orig_block);
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
        pending_context: &PendingContext,
    ) -> (Block, Context) {
        log::debug!(
            "targeting block {} from {}, in context {}",
            target,
            orig_block,
            state.context
        );

        let target_context = pending_context.single().unwrap_or(state.context);
        log::trace!(" -> new context {}", target_context);

        log::trace!(
            "target_block: from orig {} ctx {} to {} ctx {}",
            orig_block,
            state.context,
            target,
            target_context
        );

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

    fn evaluate_block_target<F: Fn(AbstractValue) -> AbstractValue>(
        &mut self,
        orig_block: Block,
        state: &PointState,
        target: &BlockTarget,
        pending_context: &PendingContext,
        abs_map: F,
    ) -> BlockTarget {
        let mut args = vec![];
        let mut abs_args = vec![];
        let mut abs_generic_args = vec![];
        log::trace!(
            "evaluate target: block {} context {} to {:?}",
            orig_block,
            state.context,
            target
        );

        let (target_block, target_ctx) =
            self.target_block(state, orig_block, target.block, pending_context);

        for &arg in &target.args {
            let arg = self.generic.resolve_alias(arg);
            abs_generic_args.push(arg);
            let (val, abs) = self.use_value(state.context, orig_block, arg);
            log::trace!(
                "blockparam: block {} context {}: arg {} has val {} abs {:?}",
                orig_block,
                state.context,
                arg,
                val,
                abs,
            );
            args.push(val);
            abs_args.push(abs_map(abs));
        }

        // Parallel-move semantics: read all uses above, then write
        // all defs below.
        let mut changed = false;
        for ((blockparam, abs), generic_arg) in self.generic.blocks[target.block]
            .params
            .iter()
            .map(|(_, val)| *val)
            .zip(abs_args.iter())
            .zip(abs_generic_args.iter())
        {
            let &val = self.value_map.get(&(target_ctx, blockparam)).unwrap();
            log::debug!(
                "blockparam: updating with new def: block {} context {} param {} val {} abstract {:?} from branch arg {}",
                target.block, target_ctx, blockparam, val, abs, generic_arg);
            changed |= self.def_value(orig_block, target_ctx, blockparam, val, abs.clone());
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
                        target: self.evaluate_block_target(
                            orig_block,
                            state,
                            if_true,
                            &state.pending_context,
                            |abs| abs,
                        ),
                    },
                    Some(false) => Terminator::Br {
                        target: self.evaluate_block_target(
                            orig_block,
                            state,
                            if_false,
                            &state.pending_context,
                            |abs| abs,
                        ),
                    },
                    None => Terminator::CondBr {
                        cond,
                        if_true: self.evaluate_block_target(
                            orig_block,
                            state,
                            if_true,
                            &state.pending_context,
                            |abs| abs,
                        ),
                        if_false: self.evaluate_block_target(
                            orig_block,
                            state,
                            if_false,
                            &state.pending_context,
                            |abs| abs,
                        ),
                    },
                }
            }
            &Terminator::Br { ref target } => match &state.pending_context {
                PendingContext::Switch(index, contexts, default_context) => {
                    let targets = contexts
                        .iter()
                        .enumerate()
                        .map(|(i, &context)| {
                            self.evaluate_block_target(
                                orig_block,
                                state,
                                target,
                                &PendingContext::Single(context),
                                |abs| abs.remap_switch(*index, i),
                            )
                        })
                        .collect::<Vec<BlockTarget>>();
                    let default = self.evaluate_block_target(
                        orig_block,
                        state,
                        target,
                        &PendingContext::Single(*default_context),
                        |abs| abs.remap_switch(*index, contexts.len()),
                    );
                    log::debug!("switch on terminator due to Switch pending context: index {} targets {:?} default {:?}", index, targets, default);
                    Terminator::Select {
                        value: *index,
                        targets,
                        default,
                    }
                }
                PendingContext::Single(..) | PendingContext::None => Terminator::Br {
                    target: self.evaluate_block_target(
                        orig_block,
                        state,
                        target,
                        &state.pending_context,
                        |abs| abs,
                    ),
                },
                PendingContext::IncompleteSwitch => Terminator::Unreachable,
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
                        target: self.evaluate_block_target(
                            orig_block,
                            state,
                            target,
                            &state.pending_context,
                            |abs| abs,
                        ),
                    }
                } else {
                    let targets = targets
                        .iter()
                        .map(|target| {
                            self.evaluate_block_target(
                                orig_block,
                                state,
                                target,
                                &state.pending_context,
                                |abs| abs,
                            )
                        })
                        .collect::<Vec<_>>();
                    let default = self.evaluate_block_target(
                        orig_block,
                        state,
                        default,
                        &state.pending_context,
                        |abs| abs,
                    );
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
        new_block: Block,
        orig_inst: Value,
        op: Operator,
        loc: SourceLoc,
        abs: &[AbstractValue],
        values: &[Value],
        orig_values: &[Value],
        tys: &[Type],
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

        let intrinsic_result = self.abstract_eval_intrinsic(
            orig_block,
            new_block,
            orig_inst,
            op,
            loc,
            abs,
            values,
            orig_values,
            state,
        );
        if intrinsic_result.is_handled() {
            log::debug!(" -> intrinsic: {:?}", intrinsic_result);
            return Ok(intrinsic_result);
        }

        let switch_result = self.abstract_eval_switch(
            orig_block,
            new_block,
            orig_inst,
            op,
            loc,
            abs,
            values,
            orig_values,
            tys,
            state,
        )?;
        if switch_result.is_handled() {
            log::debug!(" -> switch: {:?}", switch_result);
            return Ok(switch_result);
        }

        let mem_overlay_result =
            self.abstract_eval_mem_overlay(orig_inst, new_block, op, abs, values, tys, state)?;
        if mem_overlay_result.is_handled() {
            log::debug!(" -> overlay: {:?}", intrinsic_result);
            return Ok(mem_overlay_result);
        }

        let ret = if op.is_call() {
            self.flush_to_mem(state, new_block);
            log::debug!(" -> call");
            AbstractValue::Runtime(Some(orig_inst), ValueTags::default())
        } else {
            match abs.len() {
                0 => self.abstract_eval_nullary(orig_inst, op, state),
                1 => self.abstract_eval_unary(orig_inst, op, &abs[0], orig_values[0], state)?,
                2 => self.abstract_eval_binary(orig_inst, op, &abs[0], &abs[1]),
                3 => self.abstract_eval_ternary(orig_inst, op, &abs[0], &abs[1], &abs[2]),
                _ => {
                    let tags = abs
                        .iter()
                        .map(|av| av.tags().sticky())
                        .reduce(|a, b| a.meet(b))
                        .unwrap();
                    AbstractValue::Runtime(Some(orig_inst), tags)
                }
            }
        };

        log::debug!(" -> result: {:?}", ret);
        Ok(EvalResult::Normal(ret))
    }

    fn abstract_eval_intrinsic(
        &mut self,
        orig_block: Block,
        new_block: Block,
        _orig_inst: Value,
        op: Operator,
        loc: SourceLoc,
        abs: &[AbstractValue],
        values: &[Value],
        orig_values: &[Value],
        state: &mut PointState,
    ) -> EvalResult {
        match op {
            Operator::Call { function_index } => {
                if Some(function_index) == self.intrinsics.assume_const_memory {
                    EvalResult::Alias(abs[0].with_tags(ValueTags::const_memory()), values[0])
                } else if Some(function_index) == self.intrinsics.assume_const_memory_transitive {
                    EvalResult::Alias(
                        abs[0].with_tags(
                            ValueTags::const_memory() | ValueTags::const_memory_transitive(),
                        ),
                        values[0],
                    )
                } else if Some(function_index) == self.intrinsics.make_symbolic_ptr {
                    let label_index = values[0].index() as u32;
                    log::trace!(
                        "value {} is getting symbolic label {}",
                        values[0],
                        label_index
                    );
                    EvalResult::Alias(AbstractValue::SymbolicPtr(label_index, 0), values[0])
                } else if Some(function_index) == self.intrinsics.push_context {
                    let pc = abs[0]
                        .is_const_u32()
                        .expect("PC should not be a runtime value");
                    let instantaneous_context =
                        state.pending_context.single().unwrap_or(state.context);
                    let child = self
                        .state
                        .contexts
                        .create(Some(instantaneous_context), ContextElem::Loop(pc));
                    state.pending_context = PendingContext::Single(child);
                    log::trace!("push context (pc {:?}): now {}", pc, child);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.pop_context {
                    let instantaneous_context =
                        state.pending_context.single().unwrap_or(state.context);
                    let parent = match self.state.contexts.leaf_element(instantaneous_context) {
                        ContextElem::Root => instantaneous_context,
                        _ => self.state.contexts.parent(instantaneous_context),
                    };
                    state.pending_context = PendingContext::Single(parent);
                    log::trace!("pop context: now {}", parent);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.update_context {
                    log::trace!("update context at {}: PC is {:?}", orig_values[0], abs[0]);
                    if self.state.contexts.leaf_element(state.context) == ContextElem::Root {
                        let loc = self.module.debug.source_locs[loc];
                        let file = &self.module.debug.source_files[loc.file];
                        panic!(
                            "update_context in root context; loc {}:{}:{}",
                            file, loc.line, loc.col
                        );
                    }
                    let instantaneous_context =
                        state.pending_context.single().unwrap_or(state.context);
                    let mut make_context =
                        |pc: u32| match self.state.contexts.leaf_element(instantaneous_context) {
                            ContextElem::Root => instantaneous_context,
                            _ => self.state.contexts.create(
                                Some(self.state.contexts.parent(instantaneous_context)),
                                ContextElem::Loop(pc),
                            ),
                        };

                    let pending_context = if let Some(pc) = abs[0].is_const_u32() {
                        PendingContext::Single(make_context(pc))
                    } else if let AbstractValue::Switch(index, pcs, default_pc, _) = &abs[0] {
                        log::debug!(
                            "Pending context becomes switch: index {} pcs {:?} default {:?}",
                            index,
                            pcs,
                            default_pc
                        );
                        PendingContext::Switch(
                            *index,
                            pcs.iter()
                                .map(|pc| make_context(pc.integer_value().unwrap() as u32))
                                .collect(),
                            make_context(default_pc.integer_value().unwrap() as u32),
                        )
                    } else if abs[0].is_switch_value() || abs[0].is_switch_default() {
                        log::debug!("Pending context becomes IncompleteSwitch");
                        PendingContext::IncompleteSwitch
                    } else {
                        panic!("PC is a runtime value: {:?}", abs[0]);
                    };
                    log::trace!("update context: now {:?}", pending_context);
                    state.pending_context = pending_context;
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.flush_to_mem {
                    self.flush_to_mem(state, new_block);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.abort_specialization {
                    let line_num = abs[0].is_const_u32().unwrap_or(0);
                    let fatal = abs[1].is_const_u32().unwrap_or(0);
                    log::trace!("abort-specialization point: line {}", line_num);
                    if fatal != 0 {
                        panic!("Specialization reached a point it shouldn't have!");
                    }
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.trace_line {
                    let line_num = abs[0].is_const_u32().unwrap_or(0);
                    log::info!("trace: line number {}: current context {} at block {}, pending context {:?}",
                                line_num, state.context, orig_block, state.pending_context);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.assert_const32 {
                    log::trace!("assert_const32: abs {:?} line {:?}", abs[0], abs[1]);
                    if abs[0].is_const_u32().is_none()
                        && !abs[0].is_switch_value()
                        && !abs[0].is_switch_default()
                    {
                        panic!(
                            "weval_assert_const32() failed: {:?}: line {:?}",
                            abs[0], abs[1]
                        );
                    }
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.assert_const_memory {
                    log::trace!("assert_const_memory: abs {:?} line {:?}", abs[0], abs[1]);
                    if !abs[0].tags().contains(ValueTags::const_memory()) {
                        panic!("weval_assert_const_memory() failed: line {:?}", abs[1]);
                    }
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.switch_value {
                    if let AbstractValue::Concrete(WasmVal::I32(limit), tags) = &abs[1] {
                        let vals = (0..(*limit)).map(WasmVal::I32).collect::<Vec<_>>();
                        log::info!("Producing switch value for {}: {:?}", orig_values[0], vals);
                        EvalResult::Alias(AbstractValue::SwitchValue(vals, *tags), values[0])
                    } else {
                        panic!("Limit to weval_switch_value() is not a constant");
                    }
                } else if Some(function_index) == self.intrinsics.switch_default {
                    if let AbstractValue::Concrete(WasmVal::I32(default), tags) = &abs[0] {
                        log::info!(
                            "Producing switch default for {}: {}",
                            orig_values[0],
                            default,
                        );
                        EvalResult::Alias(
                            AbstractValue::SwitchDefault(WasmVal::I32(*default), *tags),
                            values[0],
                        )
                    } else {
                        panic!("Default index arg to weval_switch_default() is not a constant");
                    }
                } else if Some(function_index) == self.intrinsics.switch {
                    if let AbstractValue::SwitchValueAndDefault(vals, default, tags) = &abs[1] {
                        let index = values[0];
                        log::info!("Tagging switch with index {}: {:?}", orig_values[0], abs[1]);
                        EvalResult::Alias(
                            AbstractValue::Switch(index, vals.clone(), *default, *tags),
                            values[1],
                        )
                    } else {
                        panic!("Default index arg to weval_switch_default() is not a constant");
                    }
                } else if Some(function_index) == self.intrinsics.print {
                    let message_ptr = abs[0].is_const_u32().unwrap();
                    let message = self
                        .image
                        .read_str(self.image.main_heap.unwrap(), message_ptr)
                        .unwrap();
                    let line = abs[1].is_const_u32().unwrap();
                    let val = abs[2].clone();
                    log::debug!("print: line {}: {}: {:?}", line, message, val);
                    EvalResult::Elide
                } else {
                    EvalResult::Unhandled
                }
            }
            _ => EvalResult::Unhandled,
        }
    }

    fn flush_to_mem(&mut self, state: &mut PointState, new_block: Block) {
        for (_, value) in std::mem::take(&mut state.flow.mem_overlay) {
            match value {
                MemValue::Value {
                    data,
                    ty,
                    addr,
                    dirty,
                    abs: _,
                } if dirty => {
                    let store = self.func.add_value(ValueDef::Operator(
                        store_operator(ty).unwrap(),
                        vec![addr, data],
                        vec![],
                    ));
                    self.func.append_to_block(new_block, store);
                }
                _ => {}
            }
        }
    }

    fn flush_on_edge(&mut self, from: Block, to: Block, succ_idx: usize) {
        let (from_ctx, from_orig_block) = self.block_rev_map[from];
        let from_state = self.state.state[from_ctx]
            .block_exit
            .get(&from_orig_block)
            .unwrap();
        let (to_ctx, to_orig_block) = self.block_rev_map[to];
        let to_state = self.state.state[to_ctx]
            .block_exit
            .get(&to_orig_block)
            .unwrap();

        let mut edge_block = None;
        for (sym_addr, value) in &from_state.mem_overlay {
            let succ_value = to_state.mem_overlay.get(sym_addr);
            match (value, succ_value) {
                (MemValue::Value { .. }, Some(MemValue::Value { .. })) => {
                    // Nothing necessary: value will be passed as blockparam.
                }
                (MemValue::Value { .. }, Some(MemValue::TypedMerge(..))) => {
                    // Nothing necessary: value will be passed as blockparam.
                }
                (
                    MemValue::Value {
                        data,
                        ty,
                        addr,
                        dirty,
                        abs: _,
                    },
                    other,
                ) if *dirty => {
                    // We need to flush the value back to memory, if
                    // "dirty" (different from what is already in
                    // memory).
                    let store = self.func.add_value(ValueDef::Operator(
                        store_operator(*ty).unwrap(),
                        vec![*addr, *data],
                        vec![],
                    ));
                    log::trace!("from block {} to block {}: symbolic addr {:?} had addr {} data {}, but not in succ: {:?}",
                                from, to, sym_addr, addr, data, other);
                    let block = *edge_block.get_or_insert_with(|| {
                        let edge_block = self.func.split_edge(from, to, succ_idx);
                        self.func.blocks[edge_block].desc = format!("Edge from {} to {}", from, to);
                        edge_block
                    });
                    log::trace!(" -> appending store {} to edge block {}", store, block);
                    self.func.append_to_block(block, store);
                }
                _ => {}
            }
        }
    }

    fn abstract_eval_mem_overlay(
        &mut self,
        inst: Value,
        new_block: Block,
        op: Operator,
        abs: &[AbstractValue],
        vals: &[Value],
        tys: &[Type],
        state: &mut PointState,
    ) -> anyhow::Result<EvalResult> {
        match (op, abs) {
            (
                Operator::I32Load { memory } | Operator::I64Load { memory },
                &[AbstractValue::SymbolicPtr(label, off)],
            ) if memory.memory.index() == 0 => {
                let off = off + (memory.offset as i64);
                let expected_ty = match op {
                    Operator::I32Load { .. } => Type::I32,
                    Operator::I64Load { .. } => Type::I64,
                    _ => anyhow::bail!("Bad load type to symbolic ptr"),
                };
                log::trace!("load from symbolic loc {:?}", abs[0]);
                match state.flow.mem_overlay.get(&SymbolicAddr(label, off)) {
                    Some(MemValue::Value {
                        data,
                        ty,
                        addr: _,
                        dirty: _,
                        abs,
                    }) if *ty == expected_ty => {
                        log::trace!(" -> have value {} with abs {:?}", data, abs);
                        return Ok(EvalResult::Alias(abs.clone(), *data));
                    }
                    None => {
                        // Create the original load, so we have access
                        // to its value; then insert it into the mem
                        // overlay.
                        let l = self.func.add_value(ValueDef::Operator(
                            op.clone(),
                            vals.to_vec(),
                            tys.to_vec(),
                        ));
                        self.func.append_to_block(new_block, l);
                        state.flow.mem_overlay.insert(
                            SymbolicAddr(label, off),
                            MemValue::Value {
                                data: l,
                                ty: tys[0],
                                addr: vals[0],
                                dirty: false,
                                abs: abs[0].clone(),
                            },
                        );
                        return Ok(EvalResult::Alias(abs[0].clone(), l));
                    }
                    Some(v) => {
                        anyhow::bail!("Bad MemValue: {:?}", v);
                    }
                }
            }
            (
                Operator::I32Store { memory } | Operator::I64Store { memory },
                &[AbstractValue::SymbolicPtr(label, off), _],
            ) if memory.memory.index() == 0 => {
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
                        data: vals[1],
                        ty: data_ty,
                        addr: vals[0],
                        dirty: true,
                        abs: abs[1].clone(),
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

        // If a symbolic-pointer-tainted value (that has escaped and
        // lost precision) reaches a load or store otherwise, we need
        // to flag the error.
        //
        // We exclude calls, because we flush all memory-renamed
        // values back to memory before every call.
        if !op.is_call() && op.accesses_memory() {
            for a in abs {
                if a.tags().contains(ValueTags::symbolic_ptr_taint()) {
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
        }

        Ok(EvalResult::Unhandled)
    }

    fn abstract_eval_nullary(
        &mut self,
        orig_inst: Value,
        op: Operator,
        state: &mut PointState,
    ) -> AbstractValue {
        match op {
            Operator::GlobalGet { global_index } => state
                .flow
                .globals
                .get(&global_index)
                .cloned()
                .unwrap_or(AbstractValue::Runtime(
                    Some(orig_inst),
                    ValueTags::default(),
                )),
            Operator::I32Const { .. }
            | Operator::I64Const { .. }
            | Operator::F32Const { .. }
            | Operator::F64Const { .. } => {
                AbstractValue::Concrete(WasmVal::try_from(op).unwrap(), ValueTags::default())
            }
            _ => AbstractValue::Runtime(Some(orig_inst), ValueTags::default()),
        }
    }

    fn abstract_eval_unary(
        &mut self,
        orig_inst: Value,
        op: Operator,
        x: &AbstractValue,
        orig_x_val: Value,
        state: &mut PointState,
    ) -> anyhow::Result<AbstractValue> {
        let result = match (op, x) {
            (Operator::GlobalSet { global_index }, av) => {
                state.flow.globals.insert(global_index, av.clone());
                Ok(AbstractValue::Runtime(
                    Some(orig_inst),
                    ValueTags::default(),
                ))
            }
            (Operator::I32Eqz, AbstractValue::Concrete(WasmVal::I32(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I32(if *k == 0 { 1 } else { 0 }), *t),
            ),
            (Operator::I64Eqz, AbstractValue::Concrete(WasmVal::I64(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I64(if *k == 0 { 1 } else { 0 }), *t),
            ),
            (Operator::I32Extend8S, AbstractValue::Concrete(WasmVal::I32(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I32(*k as i8 as i32 as u32), *t),
            ),
            (Operator::I32Extend16S, AbstractValue::Concrete(WasmVal::I32(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I32(*k as i16 as i32 as u32), *t),
            ),
            (Operator::I64Extend8S, AbstractValue::Concrete(WasmVal::I64(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I64(*k as i8 as i64 as u64), *t),
            ),
            (Operator::I64Extend16S, AbstractValue::Concrete(WasmVal::I64(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I64(*k as i16 as i64 as u64), *t),
            ),
            (Operator::I64Extend32S, AbstractValue::Concrete(WasmVal::I64(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I64(*k as i32 as i64 as u64), *t),
            ),
            (Operator::I32Clz, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                Ok(AbstractValue::Concrete(WasmVal::I32(k.leading_zeros()), *t))
            }
            (Operator::I64Clz, AbstractValue::Concrete(WasmVal::I64(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I64(k.leading_zeros() as u64), *t),
            ),
            (Operator::I32Ctz, AbstractValue::Concrete(WasmVal::I32(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I32(k.trailing_zeros()), *t),
            ),
            (Operator::I64Ctz, AbstractValue::Concrete(WasmVal::I64(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I64(k.trailing_zeros() as u64), *t),
            ),
            (Operator::I32Popcnt, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                Ok(AbstractValue::Concrete(WasmVal::I32(k.count_ones()), *t))
            }
            (Operator::I64Popcnt, AbstractValue::Concrete(WasmVal::I64(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I64(k.count_ones() as u64), *t),
            ),
            (Operator::I32WrapI64, AbstractValue::Concrete(WasmVal::I64(k), t)) => {
                Ok(AbstractValue::Concrete(WasmVal::I32(*k as u32), *t))
            }
            (Operator::I64ExtendI32S, AbstractValue::Concrete(WasmVal::I32(k), t)) => Ok(
                AbstractValue::Concrete(WasmVal::I64(*k as i32 as i64 as u64), *t),
            ),
            (Operator::I64ExtendI32U, AbstractValue::Concrete(WasmVal::I32(k), t)) => {
                Ok(AbstractValue::Concrete(WasmVal::I64(*k as u64), *t))
            }

            (Operator::I32Load { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I32Load8U { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I32Load8S { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I32Load16U { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
            | (Operator::I32Load16S { memory }, AbstractValue::Concrete(WasmVal::I32(k), t))
                if t.contains(ValueTags::const_memory()) =>
            {
                use anyhow::Context;

                log::trace!(
                    "load of addr {:?} offset {} (orig value {}) with const_memory tag",
                    x,
                    memory.offset,
                    orig_x_val,
                );
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

                let val = self
                    .image
                    .read_size(memory.memory, k + memory.offset as u32, size)
                    .with_context(|| {
                        format!("Out-of-bounds constant load: value {} is {}", orig_x_val, k)
                    })?;
                // N.B.: memory const-ness is *not* transitive unless
                // specified as such!  The user needs to opt in at
                // each level of indirection.
                let tags = if t.contains(ValueTags::const_memory_transitive()) {
                    ValueTags::const_memory() | ValueTags::const_memory_transitive()
                } else {
                    ValueTags::default()
                };
                let val = AbstractValue::Concrete(WasmVal::I32(conv(val)), tags);
                log::trace!(" -> produces {:?}", val);
                Ok(val)
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

                let val = self
                    .image
                    .read_size(memory.memory, k + memory.offset as u32, size)?;
                // N.B.: memory const-ness is *not* transitive unless
                // specified as such!  The user needs to opt in at
                // each level of indirection.
                let tags = if t.contains(ValueTags::const_memory_transitive()) {
                    ValueTags::const_memory() | ValueTags::const_memory_transitive()
                } else {
                    ValueTags::default()
                };
                let val = AbstractValue::Concrete(WasmVal::I64(conv(val)), tags);
                log::trace!(" -> produces {:?}", val);
                Ok(val)
            }

            // TODO: FP and SIMD
            _ => Ok(AbstractValue::Runtime(
                Some(orig_inst),
                ValueTags::default(),
            )),
        };

        result.map(|av| av.prop_sticky_tags(x))
    }

    fn abstract_eval_binary(
        &mut self,
        orig_inst: Value,
        op: Operator,
        x: &AbstractValue,
        y: &AbstractValue,
    ) -> AbstractValue {
        let result = match (x, y) {
            (AbstractValue::Concrete(v1, tag1), AbstractValue::Concrete(v2, tag2)) => {
                let tags = tag1.meet(*tag2);
                let derived_ptr_tags = if tag1.contains(ValueTags::const_memory())
                    || tag2.contains(ValueTags::const_memory())
                {
                    tags | ValueTags::const_memory()
                } else {
                    tags
                };
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
                            WasmVal::I32(if (*k1 as i32) < (*k2 as i32) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I32LtU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 < k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I32GtS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I32(if (*k1 as i32) > (*k2 as i32) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I32GtU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 > k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I32LeS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I32(if (*k1 as i32) <= (*k2 as i32) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I32LeU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 <= k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I32GeS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I32(if (*k1 as i32) >= (*k2 as i32) { 1 } else { 0 }),
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
                            WasmVal::I64(if (*k1 as i64) < (*k2 as i64) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I64LtU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 < k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I64GtS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(if (*k1 as i64) > (*k2 as i64) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I64GtU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 > k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I64LeS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(if (*k1 as i64) <= (*k2 as i64) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I64LeU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 <= k2 { 1 } else { 0 }), tags)
                    }
                    (Operator::I64GeS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(if (*k1 as i64) >= (*k2 as i64) { 1 } else { 0 }),
                            tags,
                        )
                    }
                    (Operator::I64GeU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 >= k2 { 1 } else { 0 }), tags)
                    }

                    // 32-bit integer arithmetic.
                    (Operator::I32Add, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I32(k1.wrapping_add(*k2)),
                            derived_ptr_tags,
                        )
                    }
                    (Operator::I32Sub, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I32(k1.wrapping_sub(*k2)),
                            derived_ptr_tags,
                        )
                    }
                    (Operator::I32Mul, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_mul(*k2)), tags)
                    }
                    (Operator::I32DivU, WasmVal::I32(k1), WasmVal::I32(k2)) if *k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_div(*k2)), tags)
                    }
                    (Operator::I32DivS, WasmVal::I32(k1), WasmVal::I32(k2))
                        if *k2 != 0 && (*k1 != 0x8000_0000 || *k2 != 0xffff_ffff) =>
                    {
                        AbstractValue::Concrete(
                            WasmVal::I32((*k1 as i32).wrapping_div(*k2 as i32) as u32),
                            tags,
                        )
                    }
                    (Operator::I32RemU, WasmVal::I32(k1), WasmVal::I32(k2)) if *k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_rem(*k2)), tags)
                    }
                    (Operator::I32RemS, WasmVal::I32(k1), WasmVal::I32(k2))
                        if *k2 != 0 && (*k1 != 0x8000_0000 || *k2 != 0xffff_ffff) =>
                    {
                        AbstractValue::Concrete(
                            WasmVal::I32((*k1 as i32).wrapping_rem(*k2 as i32) as u32),
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
                            WasmVal::I32((*k1 as i32).wrapping_shr(*k2 & 0x1f) as u32),
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
                        AbstractValue::Concrete(
                            WasmVal::I64(k1.wrapping_add(*k2)),
                            derived_ptr_tags,
                        )
                    }
                    (Operator::I64Sub, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(k1.wrapping_sub(*k2)),
                            derived_ptr_tags,
                        )
                    }
                    (Operator::I64Mul, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_mul(*k2)), tags)
                    }
                    (Operator::I64DivU, WasmVal::I64(k1), WasmVal::I64(k2)) if *k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_div(*k2)), tags)
                    }
                    (Operator::I64DivS, WasmVal::I64(k1), WasmVal::I64(k2))
                        if *k2 != 0
                            && (*k1 != 0x8000_0000_0000_0000 || *k2 != 0xffff_ffff_ffff_ffff) =>
                    {
                        AbstractValue::Concrete(
                            WasmVal::I64((*k1 as i64).wrapping_div(*k2 as i64) as u64),
                            tags,
                        )
                    }
                    (Operator::I64RemU, WasmVal::I64(k1), WasmVal::I64(k2)) if *k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_rem(*k2)), tags)
                    }
                    (Operator::I64RemS, WasmVal::I64(k1), WasmVal::I64(k2))
                        if *k2 != 0
                            && (*k1 != 0x8000_0000_0000_0000 || *k2 != 0xffff_ffff_ffff_ffff) =>
                    {
                        AbstractValue::Concrete(
                            WasmVal::I64((*k1 as i64).wrapping_rem(*k2 as i64) as u64),
                            tags,
                        )
                    }
                    (Operator::I64And, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(*k1 & *k2), tags)
                    }
                    (Operator::I64Or, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(*k1 | *k2), tags)
                    }
                    (Operator::I64Xor, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(*k1 ^ *k2), tags)
                    }
                    (Operator::I64Shl, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(k1.wrapping_shl((*k2 & 0x3f) as u32)),
                            tags,
                        )
                    }
                    (Operator::I64ShrU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64(k1.wrapping_shr((*k2 & 0x3f) as u32)),
                            tags,
                        )
                    }
                    (Operator::I64ShrS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(
                            WasmVal::I64((*k1 as i64).wrapping_shr((*k2 & 0x3f) as u32) as u64),
                            tags,
                        )
                    }
                    (Operator::I64Rotl, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        let amt = (*k2 & 0x3f) as u32;
                        let result = k1.wrapping_shl(amt) | k1.wrapping_shr(64 - amt);
                        AbstractValue::Concrete(WasmVal::I64(result), tags)
                    }
                    (Operator::I64Rotr, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        let amt = (*k2 & 0x3f) as u32;
                        let result = k1.wrapping_shr(amt) | k1.wrapping_shl(64 - amt);
                        AbstractValue::Concrete(WasmVal::I64(result), tags)
                    }

                    // TODO: FP and SIMD ops.
                    _ => AbstractValue::Runtime(Some(orig_inst), ValueTags::default()),
                }
            }
            _ => AbstractValue::Runtime(Some(orig_inst), ValueTags::default()),
        };

        result.prop_sticky_tags(x).prop_sticky_tags(y)
    }

    fn abstract_eval_ternary(
        &mut self,
        orig_inst: Value,
        op: Operator,
        x: &AbstractValue,
        y: &AbstractValue,
        z: &AbstractValue,
    ) -> AbstractValue {
        let result = match (op, z) {
            (Operator::Select, AbstractValue::Concrete(v, _t))
            | (Operator::TypedSelect { .. }, AbstractValue::Concrete(v, _t)) => {
                if v.is_truthy() {
                    x.clone()
                } else {
                    y.clone()
                }
            }
            _ => AbstractValue::Runtime(Some(orig_inst), ValueTags::default()),
        };

        result
            .prop_sticky_tags(x)
            .prop_sticky_tags(y)
            .prop_sticky_tags(z)
    }

    fn abstract_eval_switch(
        &mut self,
        orig_block: Block,
        new_block: Block,
        orig_inst: Value,
        op: Operator,
        loc: SourceLoc,
        abs: &[AbstractValue],
        values: &[Value],
        orig_values: &[Value],
        tys: &[Type],
        state: &mut PointState,
    ) -> anyhow::Result<EvalResult> {
        // If exactly one of the AbstractValues is a SwitchValue,
        // recursively invoke abstract_eval for each value. If all
        // results are Concrete, and if no state was updated, then we
        // can return that this was handled properly, and build a
        // SwitchValue back from the individual "lanes".
        if abs.iter().filter(|abs| abs.is_switch_value()).count() == 1 {
            log::info!(" -> exactly one arg is a SwitchValue");
            let index = abs.iter().position(|abs| abs.is_switch_value()).unwrap();
            let pre = &abs[0..index];
            let post = &abs[(index + 1)..];
            let (switch_values, tags) = match &abs[index] {
                AbstractValue::SwitchValue(values, tags) => (values, *tags),
                _ => unreachable!(),
            };

            let mut new_values = vec![];
            let mut private_state = state.clone();
            let mut all_tags = ValueTags::default();
            for &value in switch_values.iter().chain(def_value.iter()) {
                let mut abs = vec![];
                abs.extend(pre.iter().cloned());
                abs.push(AbstractValue::Concrete(value, tags));
                abs.extend(post.iter().cloned());
                log::debug!("SwitchValue: inst {} op {:?} abs {:?}", orig_inst, op, abs);
                let result = self.abstract_eval(
                    orig_block,
                    new_block,
                    orig_inst,
                    op,
                    loc,
                    &abs[..],
                    values,
                    orig_values,
                    tys,
                    &mut private_state,
                )?;
                if private_state != *state {
                    return Ok(EvalResult::Unhandled);
                }
                match result {
                    EvalResult::Normal(AbstractValue::Concrete(val, tags)) => {
                        new_values.push(val);
                        all_tags = all_tags | tags;
                    }
                    _ => return Ok(EvalResult::Unhandled),
                }
            }

            let def_value = if def_value.is_some() {
                new_values.pop()
            } else {
                None
            };

            log::info!(
                " -> new_values: {:?}, def_value {:?}",
                new_values,
                def_value
            );
            return Ok(EvalResult::Normal(AbstractValue::SwitchValue(
                index, new_values, def_value, all_tags,
            )));
        }

        // Likewise, but for SwitchDefault.
        if abs.iter().filter(|abs| abs.is_switch_default()).count() == 1 {
            log::info!(" -> exactly one arg is a SwitchDefault");
            let index = abs.iter().position(|abs| abs.is_switch_default()).unwrap();
            let pre = &abs[0..index];
            let post = &abs[(index + 1)..];
            let (index, def_value, tags) = match &abs[index] {
                AbstractValue::SwitchDefault(index, def_value, tags) => (*index, *def_value, *tags),
                _ => unreachable!(),
            };

            let mut abs = vec![];
            abs.extend(pre.iter().cloned());
            abs.push(AbstractValue::Concrete(def_value, tags));
            abs.extend(post.iter().cloned());
            log::debug!("SwitchDefault inst {} op {:?} abs {:?}", orig_inst, op, abs);
            let mut private_state = state.clone();
            let result = self.abstract_eval(
                orig_block,
                new_block,
                orig_inst,
                op,
                loc,
                &abs[..],
                values,
                orig_values,
                tys,
                &mut private_state,
            )?;
            if private_state != *state {
                return Ok(EvalResult::Unhandled);
            }
            match result {
                EvalResult::Normal(AbstractValue::Concrete(val, tags)) => {
                    log::debug!(" -> value {:?} tags {:?}", val, tags);
                    return Ok(EvalResult::Normal(AbstractValue::SwitchDefault(
                        index, val, tags,
                    )));
                }
                _ => {}
            }
        }

        Ok(EvalResult::Unhandled)
    }

    fn add_blockparam_mem_args(&mut self) -> anyhow::Result<()> {
        // Examine mem_overlay in block input state of each
        // specialized block, and create blockparams for all values
        // that in the end were `BlockParam`.
        for (&(ctx, orig_block), &block) in &self.block_map {
            let succ_state = self.state.state[ctx].block_entry.get(&orig_block).unwrap();

            for (&addr, val) in &succ_state.mem_overlay {
                let ty = val.to_type().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Inconsistent type on symbolic addr {:?} at block {}",
                        addr,
                        orig_block
                    )
                })?;
                let addr_blockparam = self.func.add_blockparam(block, Type::I32);
                let val_blockparam = self.func.add_blockparam(block, ty);
                let (orig_addr, orig_val) = *self
                    .mem_blockparam_map
                    .get(&(ctx, orig_block, addr))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "placeholder val not found for addr {:?} at block {} (ctx {} orig {})",
                            addr,
                            block,
                            ctx,
                            orig_block,
                        )
                    })?;
                self.func.set_alias(orig_addr, addr_blockparam);
                self.func.set_alias(orig_val, val_blockparam);
            }

            for pred_idx in 0..self.func.blocks[block].preds.len() {
                let pred = self.func.blocks[block].preds[pred_idx];
                let (pred_ctx, pred_orig_block) = self.block_rev_map[pred];
                let pred_state = self.state.state[pred_ctx]
                    .block_exit
                    .get(&pred_orig_block)
                    .unwrap();
                let pred_succ_idx = self.func.blocks[block].pos_in_pred_succ[pred_idx];

                for &addr in succ_state.mem_overlay.keys() {
                    let pred_val = pred_state.mem_overlay.get(&addr).unwrap();
                    let (pred_addr, pred_val) = pred_val.to_addr_and_value().unwrap();
                    self.func.blocks[pred]
                        .terminator
                        .update_target(pred_succ_idx, |target| {
                            target.args.push(pred_addr);
                            target.args.push(pred_val);
                        });
                }
            }
        }

        Ok(())
    }

    fn add_blockparam_mem_spills(&mut self) -> anyhow::Result<()> {
        for block in self.func.blocks.iter() {
            for succ_idx in 0..self.func.blocks[block].succs.len() {
                let succ = self.func.blocks[block].succs[succ_idx];
                self.flush_on_edge(block, succ, succ_idx);
            }
        }

        Ok(())
    }

    fn finalize(&mut self) -> anyhow::Result<()> {
        self.func.recompute_edges();

        // Add blockparam args for symbolic addrs to each branch.
        self.add_blockparam_mem_args()?;
        // Add spills of symbolic addrs on edges as needed to get
        // consistent (non-conflicting) state on inputs.
        self.add_blockparam_mem_spills()?;

        self.func.validate().unwrap();

        Ok(())
    }
}
