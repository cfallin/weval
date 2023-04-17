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
use std::collections::{hash_map::Entry as HashEntry, VecDeque};
use waffle::cfg::CFGInfo;
use waffle::entity::EntityRef;
use waffle::{
    entity::PerEntity, Block, BlockDef, BlockTarget, FuncDecl, FunctionBody, Module, Operator,
    Signature, SourceLoc, Table, Terminator, Type, Value, ValueDef,
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

pub struct PartialEvalResult<'a> {
    pub orig_module: Option<Module<'a>>,
    pub module: Module<'a>,
}

/// Partially evaluates according to the given directives. Returns
/// clone of original module, with tracing added.
pub fn partially_evaluate<'a>(
    mut module: Module<'a>,
    im: &mut Image,
    directives: &[Directive],
    opts: &Options,
    mut progress: Option<indicatif::ProgressBar>,
) -> anyhow::Result<PartialEvalResult<'a>> {
    let intrinsics = Intrinsics::find(&module);
    log::trace!("intrinsics: {:?}", intrinsics);
    let mut mem_updates = HashMap::default();

    // Sort directives by out-address, and remove duplicates.
    let mut directives = directives.to_vec();
    directives.sort_by_key(|d| d.func_index_out_addr);
    directives.dedup_by_key(|d| d.func_index_out_addr);

    let mut funcs = HashMap::default();
    for directive in &directives {
        if !funcs.contains_key(&directive.func) {
            let mut f = module.clone_and_expand_body(directive.func)?;
            f.convert_to_max_ssa(None);
            if opts.run_diff {
                waffle::passes::trace::run(&mut f);
                module.replace_body(directive.func, f.clone());
            }
            split_blocks_at_specialization_points(&mut f, &intrinsics);
            funcs.insert(directive.func, f);
        }
    }

    let mut orig_module = if opts.run_diff {
        Some(module.clone())
    } else {
        None
    };

    if let Some(p) = progress.as_mut() {
        p.set_length(directives.len() as u64);
    }

    let directives = directives
        .iter()
        .zip(std::iter::from_fn(move || {
            Some(progress.as_ref().map(|bar| bar.clone()))
        }))
        .collect::<Vec<_>>();
    let bodies = directives
        .par_iter()
        .flat_map(|(directive, progress)| {
            let generic = funcs.get(&directive.func).unwrap();
            let result = match partially_evaluate_func(&module, generic, im, &intrinsics, directive)
            {
                Ok(result) => result,
                Err(e) => return Some(Err(e)),
            };

            if let Some(p) = progress {
                p.inc(1);
            }
            if let Some((body, sig, name)) = result {
                let decl = if opts.run_diff {
                    FuncDecl::Body(sig, name, body)
                } else {
                    let body = match body.compile() {
                        Ok(body) => body,
                        Err(e) => return Some(Err(e)),
                    };
                    FuncDecl::Compiled(sig, name, body)
                };
                Some(Ok((directive, decl)))
            } else {
                None
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    for (directive, decl) in bodies {
        // Add function to module.
        let func = module.funcs.push(decl);
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
        log::debug!("New func index {} -> table index {}", func, table_idx);
        log::debug!(" -> writing to 0x{:x}", directive.func_index_out_addr);
        
        // If we're doing differential testing, append to *original
        // module*'s function table too, but with the generic function
        // index.
        if opts.run_diff {
            let orig_func_table = &mut orig_module.as_mut().unwrap().tables[Table::from(0)];
            let orig_func_table_elts = orig_func_table.func_elements.as_mut().unwrap();
            assert_eq!(table_idx, orig_func_table_elts.len() as u32);
            orig_func_table_elts.push(directive.func);
            if orig_func_table.max.is_some() && table_idx >= orig_func_table.max.unwrap() {
                orig_func_table.max = Some(table_idx + 1);
            }
        }
        
        // Update memory image.
        mem_updates.insert(directive.func_index_out_addr, table_idx);
    }

    // Update memory.
    let heap = im.main_heap()?;
    for (addr, value) in mem_updates {
        im.write_u32(heap, addr, value)?;
    }

    Ok(PartialEvalResult {
        orig_module,
        module,
    })
}

fn partially_evaluate_func(
    module: &Module,
    generic: &FunctionBody,
    image: &Image,
    intrinsics: &Intrinsics,
    directive: &Directive,
) -> anyhow::Result<Option<(FunctionBody, Signature, String)>> {
    let orig_name = module.funcs[directive.func].name();
    let sig = module.funcs[directive.func].sig();

    log::info!("Specializing: {:?}", directive);
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
    let (ctx, entry_state) = evaluator.state.init(image);
    log::trace!("after init_args, state is {:?}", evaluator.state);
    let specialized_entry = evaluator.create_block(evaluator.generic.entry, ctx, entry_state);
    evaluator.func.entry = specialized_entry;
    evaluator
        .queue
        .push_back((evaluator.generic.entry, ctx, specialized_entry));
    evaluator.queue_set.insert((evaluator.generic.entry, ctx));
    evaluator.state.set_args(
        evaluator.generic,
        &directive.const_params[..],
        ctx,
        &evaluator.value_map,
    );
    let success = evaluator.evaluate()?;
    if !success {
        return Ok(None);
    }

    log::info!("Specialization of {:?} done", directive);
    log::debug!(
        "Adding func:\n{}",
        evaluator.func.display_verbose("| ", Some(module))
    );
    let name = format!("{} (specialized)", orig_name);
    evaluator.func.optimize();
    Ok(Some((evaluator.func, sig, name)))
}

// Split at every `weval_specialize_value()` call. Requires max-SSA
// input, and creates max-SSA output.
fn split_blocks_at_specialization_points(func: &mut FunctionBody, intrinsics: &Intrinsics) {
    for block in 0..func.blocks.len() {
        let block = Block::new(block);
        for i in 0..func.blocks[block].insts.len() {
            let inst = func.blocks[block].insts[i];
            if let ValueDef::Operator(Operator::Call { function_index }, _, _) = &func.values[inst]
            {
                if Some(*function_index) == intrinsics.specialize_value {
                    log::trace!("Splitting at weval_specialize_value for inst {}", inst);

                    // Split the block here!
                    // 1. Create a new block with the remainder of the instructions.
                    // 2. Create a blockparam for every param of the
                    //    original block, and every value def in the
                    //    original block.
                    // 3. Create an unconditional jump from original
                    //    to split-off block.
                    // 4. Rewrite all insts' args (and terminator's
                    //    args) in the split-off block to use the
                    //    blockparams.

                    // Split *after* the call (the `i + 1`).
                    let split_insts = func.blocks[block].insts.split_off(i + 1);
                    let new_block = func.blocks.push(BlockDef::default());
                    log::trace!(" -> new block: {}", new_block);
                    func.blocks[new_block].insts = split_insts;
                    let term = std::mem::take(&mut func.blocks[block].terminator);
                    func.blocks[new_block].terminator = term;
                    func.blocks[new_block].desc = format!(
                        "Split from {} at value specialization on {}",
                        func.blocks[block].desc, inst
                    );

                    let mut target = BlockTarget {
                        block: new_block,
                        args: vec![],
                    };
                    let mut remap = HashMap::default();
                    let mut new_param_idx = 0;
                    for param_idx in 0..func.blocks[block].params.len() {
                        let (ty, orig_param) = func.blocks[block].params[param_idx];
                        let new_param =
                            func.values
                                .push(ValueDef::BlockParam(new_block, new_param_idx, ty));
                        new_param_idx += 1;
                        func.blocks[new_block].params.push((ty, new_param));
                        remap.insert(orig_param, new_param);
                        target.args.push(orig_param);
                    }
                    for inst in 0..func.blocks[block].insts.len() {
                        let inst = func.blocks[block].insts[inst];
                        match &func.values[inst] {
                            ValueDef::Operator(_, _, tys) if tys.len() == 1 => {
                                let ty = tys[0];
                                let new_param = func.values.push(ValueDef::BlockParam(
                                    new_block,
                                    new_param_idx,
                                    ty,
                                ));
                                new_param_idx += 1;
                                func.blocks[new_block].params.push((ty, new_param));
                                remap.insert(inst, new_param);
                                target.args.push(inst);
                            }
                            _ => {}
                        }
                    }

                    for new_inst in 0..func.blocks[new_block].insts.len() {
                        let new_inst = func.blocks[new_block].insts[new_inst];
                        func.values[new_inst].update_uses(|u| {
                            if let Some(r) = remap.get(u) {
                                *u = *r;
                            }
                        });
                    }
                    func.blocks[new_block].terminator.update_uses(|u| {
                        if let Some(r) = remap.get(u) {
                            *u = *r;
                        }
                    });

                    func.blocks[block].terminator = Terminator::Br { target };
                }

                break;
            }
        }
    }

    log::trace!("After splitting:\n{}\n", func.display_verbose("| ", None));
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

const MAX_BLOCKS: usize = 100_000;
const MAX_VALUES: usize = 1_000_000;

impl<'a> Evaluator<'a> {
    fn evaluate(&mut self) -> anyhow::Result<bool> {
        while let Some((orig_block, ctx, new_block)) = self.queue.pop_back() {
            if self.func.blocks.len() > MAX_BLOCKS || self.func.values.len() > MAX_VALUES {
                return Ok(false);
            }
            self.queue_set.remove(&(orig_block, ctx));
            self.evaluate_block(orig_block, ctx, new_block)?;
        }
        self.finalize()?;
        Ok(true)
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
            flow: self.state.block_entry[new_block].clone(),
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
        self.state.block_exit[new_block] = state.flow.clone();

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
        if let Some(&val) = self.value_map.get(&(context, orig_val)) {
            let abs = &self.state.values[val];
            log::trace!(" -> found abstract  value {:?} at context {}", abs, context);
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
        let val_abs = &mut self.state.values[val];
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
        _block: Block,
        _context: Context,
        new_block: Block,
        state: &ProgPointState,
    ) -> bool {
        let mut state = state.clone();
        state.update_across_edge();

        self.state.block_entry[new_block].meet_with(&state)
    }

    fn context_desc(&self, ctx: Context) -> String {
        match self.state.contexts.leaf_element(ctx) {
            ContextElem::Root => "root".to_owned(),
            ContextElem::Loop(pc) => format!("PC {:?}", pc),
            ContextElem::PendingSpecialize(index, lo, hi) => {
                format!("Pending Specialization of {}: {}..={}", index, lo, hi)
            }
            ContextElem::Specialized(index, val) => format!("Specialization of {}: {}", index, val),
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
        self.func.blocks[block]
            .params
            .reserve(self.generic.blocks[orig_block].params.len());
        for &(ty, param) in &self.generic.blocks[orig_block].params {
            let new_param = self.func.add_blockparam(block, ty);
            log::trace!(" -> blockparam {} maps to {}", param, new_param);
            self.value_map.insert((context, param), new_param);
        }
        self.block_map.insert((context, orig_block), block);
        self.block_rev_map[block] = (context, orig_block);
        self.state.block_entry[block] = state;
        block
    }

    fn target_block(
        &mut self,
        state: &PointState,
        orig_block: Block,
        target: Block,
        target_context: Context,
    ) -> Block {
        log::debug!(
            "targeting block {} from {}, in context {}",
            target,
            orig_block,
            state.context
        );

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
                block
            }
            HashEntry::Occupied(o) => {
                let target_specialized = *o.get();
                log::trace!(" -> already existing block {}", target_specialized);
                let changed = self.meet_into_block_entry(
                    target,
                    target_context,
                    target_specialized,
                    &state.flow,
                );
                if changed {
                    log::trace!("   -> changed");
                    if self.queue_set.insert((target, target_context)) {
                        self.queue
                            .push_back((target, target_context, target_specialized));
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
        target_ctx: Context,
        target: &BlockTarget,
    ) -> BlockTarget {
        let n_args = self.generic.blocks[orig_block].params.len();
        let mut args = Vec::with_capacity(n_args);
        let mut abs_args = Vec::with_capacity(n_args);
        log::trace!(
            "evaluate target: block {} context {} to {:?}",
            orig_block,
            state.context,
            target
        );

        let target_block = self.target_block(state, orig_block, target.block, target_ctx);

        for &arg in &target.args {
            let arg = self.generic.resolve_alias(arg);
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
            abs_args.push(abs);
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

            let abs = if let ContextElem::Specialized(index, val) =
                self.state.contexts.leaf_element(target_ctx)
            {
                if index == blockparam {
                    log::trace!(
                        "Specialized context into block {} context {}: index {} becomes val {}",
                        target_block,
                        target_ctx,
                        index,
                        val
                    );
                    AbstractValue::Concrete(WasmVal::I32(val), ValueTags::default())
                } else {
                    abs.clone()
                }
            } else {
                abs.clone()
            };

            log::debug!(
                "blockparam: updating with new def: block {} context {} param {} val {} abstract {:?}",
                target.block, target_ctx, blockparam, val, abs);
            changed |= self.def_value(orig_block, target_ctx, blockparam, val, abs);
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
        let new_context = state.pending_context.unwrap_or(state.context);

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
                        target: self.evaluate_block_target(orig_block, state, new_context, if_true),
                    },
                    Some(false) => Terminator::Br {
                        target: self.evaluate_block_target(
                            orig_block,
                            state,
                            new_context,
                            if_false,
                        ),
                    },
                    None => Terminator::CondBr {
                        cond,
                        if_true: self.evaluate_block_target(
                            orig_block,
                            state,
                            new_context,
                            if_true,
                        ),
                        if_false: self.evaluate_block_target(
                            orig_block,
                            state,
                            new_context,
                            if_false,
                        ),
                    },
                }
            }
            &Terminator::Br { ref target } => {
                if let ContextElem::PendingSpecialize(index, lo, hi) =
                    self.state.contexts.leaf_element(new_context)
                {
                    log::trace!(
                        "Branch to target {} with PendingSpecialize on {}",
                        target.block,
                        index
                    );
                    let parent = self.state.contexts.parent(new_context);
                    assert_eq!(*target.args.last().unwrap(), index);
                    let target_index = self.generic.blocks[target.block].params.last().unwrap().1;
                    let mut targets: Vec<BlockTarget> = (lo..=hi)
                        .map(|i| {
                            let c = self
                                .state
                                .contexts
                                .create(Some(parent), ContextElem::Specialized(target_index, i));
                            log::trace!(" -> created new context {} for index {}", c, i);
                            self.evaluate_block_target(orig_block, state, c, target)
                        })
                        .collect();
                    let default = targets.pop().unwrap();
                    let (value, _) = self.use_value(state.context, orig_block, index);
                    Terminator::Select {
                        value,
                        targets,
                        default,
                    }
                } else {
                    Terminator::Br {
                        target: self.evaluate_block_target(orig_block, state, new_context, target),
                    }
                }
            }
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
                        target: self.evaluate_block_target(orig_block, state, new_context, target),
                    }
                } else {
                    let targets = targets
                        .iter()
                        .map(|target| {
                            self.evaluate_block_target(orig_block, state, new_context, target)
                        })
                        .collect::<Vec<_>>();
                    let default =
                        self.evaluate_block_target(orig_block, state, new_context, default);
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
        orig_inst: Value,
        op: Operator,
        _loc: SourceLoc,
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
                    let parent = self.state.contexts.pop_one_loop(instantaneous_context);
                    state.pending_context = Some(parent);
                    log::trace!("pop context: now {}", parent);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.update_context {
                    log::trace!("update context at {}: PC is {:?}", orig_values[0], abs[0]);
                    let instantaneous_context = state.pending_context.unwrap_or(state.context);
                    let parent = self.state.contexts.pop_one_loop(instantaneous_context);
                    let pending_context = if let Some(pc) = abs[0].is_const_u32() {
                        Some(
                            self.state
                                .contexts
                                .create(Some(parent), ContextElem::Loop(pc)),
                        )
                    } else {
                        panic!("PC is a runtime value: {:?}", abs[0]);
                    };
                    log::trace!("update context: now {:?}", pending_context);
                    state.pending_context = pending_context;
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.specialize_value {
                    let instantaneous_context = state.pending_context.unwrap_or(state.context);
                    let lo = abs[1].is_const_u32().unwrap();
                    let hi = abs[2].is_const_u32().unwrap();
                    let child = self.state.contexts.create(
                        Some(instantaneous_context),
                        ContextElem::PendingSpecialize(orig_inst, lo, hi),
                    );
                    log::trace!(
                        "Creating pending-specize context for index {} lo {} hi {}",
                        orig_inst,
                        lo,
                        hi
                    );
                    state.pending_context = Some(child);
                    EvalResult::Alias(abs[0].clone(), values[0])
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
                    log::debug!("trace: line number {}: current context {} at block {}, pending context {:?}",
                                line_num, state.context, orig_block, state.pending_context);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.assert_const32 {
                    log::trace!("assert_const32: abs {:?} line {:?}", abs[0], abs[1]);
                    if abs[0].is_const_u32().is_none() {
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
                } else if Some(function_index) == self.intrinsics.print {
                    let message_ptr = abs[0].is_const_u32().unwrap();
                    let message = self
                        .image
                        .read_str(self.image.main_heap.unwrap(), message_ptr)
                        .unwrap();
                    let line = abs[1].is_const_u32().unwrap();
                    let val = abs[2].clone();
                    log::trace!("print: line {}: {}: {:?}", line, message, val);
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
        let from_state = &self.state.block_exit[from];
        let to_state = &self.state.block_exit[to];

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

    fn add_blockparam_mem_args(&mut self) -> anyhow::Result<()> {
        // Examine mem_overlay in block input state of each
        // specialized block, and create blockparams for all values
        // that in the end were `BlockParam`.
        for (&(ctx, orig_block), &block) in &self.block_map {
            let succ_state = &self.state.block_entry[block];

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
                let pred_state = &self.state.block_exit[pred];
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

        #[cfg(debug_assertions)]
        self.func.validate().unwrap();

        Ok(())
    }
}
