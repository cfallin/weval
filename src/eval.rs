//! Partial evaluation.

use crate::cache::{Cache, CacheData};
use crate::directive::{Directive, DirectiveArgs};
use crate::image::Image;
use crate::intrinsics::{find_global_data_by_exported_func, Intrinsics};
use crate::liveness::Liveness;
use crate::state::*;
use crate::stats::SpecializationStats;
use crate::value::{AbstractValue, WasmVal};
use fxhash::FxHashMap as HashMap;
use fxhash::FxHashSet as HashSet;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{hash_map::Entry as HashEntry, BTreeSet, VecDeque};
use std::sync::Mutex;
use waffle::{
    cfg::CFGInfo, entity::EntityRef, entity::PerEntity, pool::ListRef, Block, BlockDef,
    BlockTarget, FuncDecl, FunctionBody, Memory, MemoryArg, Module, Operator, Signature, SourceLoc,
    Table, Terminator, Type, Value, ValueDef,
};

struct Evaluator<'a> {
    /// Module.
    module: &'a Module<'a>,
    /// Original function body.
    generic: &'a FunctionBody,
    /// The specialization directive.
    directive: &'a Directive,
    /// The argument string from the directive, parsed.
    directive_args: DirectiveArgs,
    /// Intrinsic function indices.
    intrinsics: &'a Intrinsics,
    /// Memory image.
    image: &'a Image,
    /// Domtree for function body.
    cfg: &'a CFGInfo,
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
    /// Map of (ctx, value_in_generic) to specialized value_in_func.
    value_map: HashMap<(Context, Value), Value>,
    /// Dependency map from a given value to any blocks (in
    /// specialized function) that must be re-evaluated if it changes.
    value_dep_blocks: HashMap<(Context, Value), BTreeSet<Block>>,
    /// Map of (ctx, block, idx) to blockparams for specialization-register values.
    reg_map: HashMap<(Context, Block, RegSlot), Value>,
    /// Queue of blocks to (re)compute. List of (block_in_generic,
    /// ctx, block_in_func).
    queue: VecDeque<(Block, Context, Block)>,
    /// Set to deduplicate `queue`.
    queue_set: HashSet<(Block, Context)>,
    /// Stats accumulated during specialization.
    stats: SpecializationStats,
}

pub struct PartialEvalResult<'a> {
    pub module: Module<'a>,
    pub global_base: usize,
    pub stats: Vec<SpecializationStats>,
}

/// Partially evaluates according to the given directives. Returns
/// clone of original module, with tracing added.
pub fn partially_evaluate<'a>(
    mut module: Module<'a>,
    im: &mut Image,
    directives: &[Directive],
    mut progress: Option<indicatif::ProgressBar>,
    output_ir: Option<std::path::PathBuf>,
    cache: &Cache,
) -> anyhow::Result<PartialEvalResult<'a>> {
    let intrinsics = Intrinsics::find(&module);
    log::trace!("intrinsics: {:?}", intrinsics);

    // Sort directives by out-address, and remove duplicates.
    let mut directives = directives.to_vec();
    directives.sort_by_key(|d| d.func_index_out_addr);
    directives.dedup_by_key(|d| d.func_index_out_addr);

    if let Some(p) = progress.as_mut() {
        p.set_length(directives.len() as u64);
    }

    // Result of compilation.
    let mut bodies: Vec<(Cow<Directive>, FuncDecl, String, bool)> = vec![];

    // Filter out directives that can be directly fulfilled by the cache.
    let mut cache_ctx = cache.thread()?;
    let mut remaining_directives = vec![];
    for directive in directives {
        let key = bincode::serialize(&directive).unwrap();
        if let Some(data) = cache_ctx.lookup(&key)? {
            bodies.push((
                Cow::Owned(directive),
                FuncDecl::Compiled(Signature::new(data.sig as usize), data.name, data.body),
                String::new(),
                true,
            ));

            if let Some(progress) = progress.as_ref() {
                progress.inc(1);
            }
        } else {
            remaining_directives.push(directive);
        }
    }
    directives = remaining_directives;

    if let Some(p) = progress.as_mut() {
        p.tick();
    }

    // Expand function bodies of any function named in a directive.
    let mut funcs = HashMap::default();
    for directive in &directives {
        if !funcs.contains_key(&directive.func) {
            let mut f = module.clone_and_expand_body(directive.func)?;

            if let Some(path) = &output_ir {
                let mut generic_ir_file = path.clone();
                generic_ir_file.push(&format!("generic_{}.txt", directive.func));
                std::fs::write(
                    &generic_ir_file,
                    format!("{}", f.display_verbose("", Some(&module))),
                )
                .unwrap();
            }

            let stats = Mutex::new(SpecializationStats::new(directive.func, &f));

            split_blocks_at_intrinsic_calls(&mut f, &intrinsics);

            f.recompute_edges();
            let cfg = CFGInfo::new(&f);
            let cut_blocks = find_cut_blocks(&f, &cfg, &intrinsics);

            f.convert_to_max_ssa(Some(cut_blocks));

            funcs.insert(directive.func, (f, cfg, stats));
        }
    }

    let global_base = module.globals.len();

    let progress_ref = progress.as_ref();
    bodies.extend(
        directives
            .par_iter()
            .flat_map(|directive| {
                let (generic, cfg, stats) = funcs.get(&directive.func).unwrap();
                let result = match partially_evaluate_func(
                    &module,
                    generic,
                    cfg,
                    im,
                    &intrinsics,
                    directive,
                ) {
                    Ok(result) => result,
                    Err(e) => {
                        log::warn!("Failed to evaluate function: {e:?}");
                        return None;
                    }
                };

                if let Some(p) = progress_ref {
                    p.inc(1);
                }
                if let Some((body, sig, name, spec_stats)) = result {
                    stats.lock().unwrap().add_specialization(&spec_stats);
                    let ir = if output_ir.is_some() {
                        use std::fmt::Write;
                        let cfg = CFGInfo::new(&body);
                        let liveness = Liveness::new(&body, &cfg);
                        let mut s = String::new();
                        writeln!(&mut s, "# Liveness:").unwrap();
                        for (block, _) in body.blocks.entries() {
                            let mut live = liveness.block_start[block]
                                .iter()
                                .cloned()
                                .collect::<Vec<_>>();
                            live.sort();
                            writeln!(&mut s, "# {}: {:?}", block, live).unwrap();
                        }
                        writeln!(&mut s, "").unwrap();
                        writeln!(&mut s, "{}", body.display_verbose("", Some(&module))).unwrap();
                        s
                    } else {
                        String::new()
                    };
                    let decl = {
                        let body = match body.compile() {
                            Ok(body) => body,
                            Err(e) => return Some(Err(e)),
                        };
                        FuncDecl::Compiled(sig, name, body.into_raw_body())
                    };
                    Some(Ok((Cow::Borrowed(directive), decl, ir, false)))
                } else {
                    log::warn!("Failed to weval for directive {:?}", directive);
                    None
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
    );

    if let Some(p) = progress.as_mut() {
        p.finish_and_clear();
    }

    eprintln!("Inserting results into cache...");

    // Compute memory updates.
    let mut mem_updates = HashMap::default();
    for (directive, decl, ir, cache_hit) in bodies {
        // Add to cache.
        if !cache_hit && cache.can_insert() {
            let key = bincode::serialize(&directive)?;
            let (sig, name, body) = match &decl {
                FuncDecl::Compiled(sig, name, body) => (sig, name, body),
                _ => unreachable!(),
            };
            let data = CacheData {
                sig: sig.index() as u32,
                name: name.clone(),
                body: body.clone(),
            };
            cache_ctx.insert(&key, data)?;
        }

        // Add function to module.
        let func = module.funcs.push(decl);
        // Append to table.
        let func_table = &mut module.tables[Table::from(0)];
        let table_idx = {
            let func_table_elts = func_table.func_elements.as_mut().unwrap();
            let table_idx = func_table_elts.len();
            func_table_elts.push(func);
            table_idx
        } as u64;
        func_table.initial = std::cmp::max(func_table.initial, table_idx + 1);
        if func_table.max.is_some() && table_idx >= func_table.max.unwrap() {
            func_table.max = Some(table_idx + 1);
        }
        log::info!("New func index {} -> table index {}", func, table_idx);

        if let Some(path) = &output_ir {
            let mut specialized_ir_file = path.clone();
            specialized_ir_file.push(&format!("specialized_{}_to_{}.txt", directive.func, func));
            std::fs::write(&specialized_ir_file, ir).unwrap();
        }

        // Update memory image with an output function index.
        log::info!(" -> writing to 0x{:x}", directive.func_index_out_addr);
        mem_updates.insert(directive.func_index_out_addr, table_idx);
    }

    // Update memory.
    let heap = im.main_heap()?;
    for (addr, value) in mem_updates {
        im.write_u32(heap, addr, value as u32)?;
    }

    // Update the `weval_is_wevaled` flag, if it exists and is exported.
    if let Some(is_wevaled) = find_global_data_by_exported_func(&module, "weval.is.wevaled") {
        log::info!("updating `is_wevaled` flag at {:#x} to 1", is_wevaled);
        im.write_u32(heap, is_wevaled, 1)?;
    }

    let mut stats = funcs
        .drain()
        .map(|(_, (_, _, stats))| stats.into_inner().unwrap())
        .collect::<Vec<_>>();
    stats.sort_by_key(|stats| stats.generic);

    Ok(PartialEvalResult {
        module,
        global_base,
        stats,
    })
}

fn partially_evaluate_func(
    module: &Module,
    generic: &FunctionBody,
    cfg: &CFGInfo,
    image: &Image,
    intrinsics: &Intrinsics,
    directive: &Directive,
) -> anyhow::Result<Option<(FunctionBody, Signature, String, SpecializationStats)>> {
    let directive_args = DirectiveArgs::decode(&directive.args[..])?;
    let orig_name = module.funcs[directive.func].name();
    let sig = module.funcs[directive.func].sig();

    log::info!("Specializing: {:?}", directive);
    log::info!("Args: {:?}", directive_args);
    log::debug!("body:\n{}", generic.display("| ", Some(module)));

    // Build the evaluator.
    let func = FunctionBody::new(module, sig);
    let mut evaluator = Evaluator {
        module,
        generic,
        directive,
        directive_args,
        intrinsics,
        image,
        cfg,
        state: FunctionState::new(),
        func,
        block_map: HashMap::default(),
        block_rev_map: PerEntity::default(),
        value_map: HashMap::default(),
        value_dep_blocks: HashMap::default(),
        reg_map: HashMap::default(),
        queue: VecDeque::new(),
        queue_set: HashSet::default(),
        stats: SpecializationStats::default(),
    };
    let (ctx, entry_state) = evaluator.state.init(image);
    log::trace!("after init_args, state is {:?}", evaluator.state);

    let specialized_entry = evaluator.create_block(evaluator.generic.entry, ctx, entry_state);
    evaluator
        .queue
        .push_back((evaluator.generic.entry, ctx, specialized_entry));
    evaluator.queue_set.insert((evaluator.generic.entry, ctx));
    evaluator.state.set_args(
        evaluator.generic,
        evaluator.directive.num_globals as usize,
        &evaluator.directive_args.const_params[..],
        ctx,
        &evaluator.value_map,
    );

    let pre_entry = evaluator.create_pre_entry(specialized_entry);
    evaluator.func.entry = pre_entry;

    let success = evaluator.evaluate()?;
    if !success {
        return Ok(None);
    }

    let name = format!("{} (specialized)", orig_name);
    let cfg = CFGInfo::new(&evaluator.func);
    crate::escape::remove_shadow_stack_if_non_escaping(&mut evaluator.func, &cfg);
    evaluator.func.optimize(&waffle::OptOptions {
        gvn: false,
        cprop: false,
        redundant_blockparams: true,
    });
    crate::constant_offsets::run(&mut evaluator.func, &cfg);
    waffle::passes::resolve_aliases::run(&mut evaluator.func);
    evaluator.func.optimize(&waffle::OptOptions {
        gvn: false,
        cprop: false,
        redundant_blockparams: true,
    });
    crate::dce::run(&mut evaluator.func, &cfg);

    accumulate_stats_from_func(&mut evaluator.stats, &evaluator.func);

    log::info!("Specialization of {:?} done", directive);
    log::debug!(
        "Adding func:\n{}",
        evaluator.func.display_verbose("| ", Some(module))
    );
    Ok(Some((evaluator.func, sig, name, evaluator.stats)))
}

// Split at every `weval_specialize_value()` call and
// `weval_pop_context()` call. Requires max-SSA input, and creates
// max-SSA output.
fn split_blocks_at_intrinsic_calls(func: &mut FunctionBody, intrinsics: &Intrinsics) {
    for block in 0..func.blocks.len() {
        let block = Block::new(block);
        for i in 0..func.blocks[block].insts.len() {
            let inst = func.blocks[block].insts[i];
            if let ValueDef::Operator(Operator::Call { function_index }, _, _) = &func.values[inst]
            {
                if Some(*function_index) == intrinsics.specialize_value
                    || Some(*function_index) == intrinsics.pop_context
                {
                    log::trace!("Splitting at weval intrinsic for inst {}", inst);

                    // Split the block here!  Split *after* the call
                    // (the `i + 1`).
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

                    let target = BlockTarget {
                        block: new_block,
                        args: vec![],
                    };
                    func.blocks[block].terminator = Terminator::Br { target };
                }

                break;
            }
        }
    }

    log::trace!("After splitting:\n{}\n", func.display_verbose("| ", None));
}

fn find_cut_blocks(
    func: &FunctionBody,
    cfg: &CFGInfo,
    intrinsics: &Intrinsics,
) -> std::collections::HashSet<Block> {
    let mut blocks = std::collections::HashSet::default();

    // Find the blocks that change context on out-edges.
    let mut change_ctx_blocks = HashSet::default();
    'blocks: for (block, blockdata) in func.blocks.entries() {
        for &inst in &blockdata.insts {
            if let ValueDef::Operator(Operator::Call { function_index }, ..) = &func.values[inst] {
                if Some(*function_index) == intrinsics.update_context
                    || Some(*function_index) == intrinsics.push_context
                    || Some(*function_index) == intrinsics.pop_context
                    || Some(*function_index) == intrinsics.specialize_value
                {
                    change_ctx_blocks.insert(block);
                    continue 'blocks;
                }
            }
        }
    }

    // For each block, we'll find a "highest same-context ancestor" in
    // the domtree.
    let mut highest_same_ctx_ancestor: PerEntity<Block, Block> = PerEntity::default();
    let mut queue = vec![func.entry];
    let mut queue_set = HashSet::default();
    queue_set.insert(func.entry);
    highest_same_ctx_ancestor[func.entry] = func.entry;
    while let Some(block) = queue.pop() {
        queue_set.remove(&block);

        func.blocks[block].terminator.visit_targets(|target| {
            let succ = target.block;
            let current = highest_same_ctx_ancestor[succ];
            let inbound = if change_ctx_blocks.contains(&block) {
                blocks.insert(succ);
                block
            } else {
                highest_same_ctx_ancestor[block]
            };
            let new = if !cfg.dominates(inbound, succ) {
                blocks.insert(succ);
                succ
            } else if current.is_invalid() || cfg.dominates(current, inbound) {
                inbound
            } else {
                current
            };

            let changed = new != current;
            highest_same_ctx_ancestor[succ] = new;
            log::trace!("highest same-context ancestor for {}: {}", succ, new);
            if changed {
                if queue_set.insert(succ) {
                    queue.push(succ);
                }
            }
        });
    }

    log::trace!("cut blocks = {:?}", blocks);
    blocks
}

fn meet_ancestors(cfg: &CFGInfo, a: Block, b: Block) -> Block {
    if cfg.dominates(a, b) {
        a
    } else if cfg.dominates(b, a) {
        b
    } else {
        assert!(cfg.domtree[a].is_valid());
        meet_ancestors(cfg, cfg.domtree[a], b)
    }
}

fn accumulate_stats_from_func(stats: &mut SpecializationStats, func: &FunctionBody) {
    let (blocks, insts, reachable_blocks) = crate::stats::count_reachable_blocks_and_insts(func);
    stats.specialized_blocks += blocks;
    stats.specialized_insts += insts;

    // Compute liveness over all blocks and find the live-over-edge count.
    let cfg = CFGInfo::new(func);
    let liveness = Liveness::new(func, &cfg);
    for &block in &reachable_blocks {
        stats.live_value_at_block_start += liveness.block_start[block].len();
    }
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
    let memory = MemoryArg {
        memory: Memory::new(0),
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

fn load_operator(ty: Type) -> Option<Operator> {
    let memory = MemoryArg {
        memory: Memory::new(0),
        align: 0,
        offset: 0,
    };
    match ty {
        Type::I32 => Some(Operator::I32Load { memory }),
        Type::I64 => Some(Operator::I64Load { memory }),
        Type::F32 => Some(Operator::F32Load { memory }),
        Type::F64 => Some(Operator::F64Load { memory }),
        _ => None,
    }
}

#[derive(Debug)]
enum EvalResult {
    Unhandled,
    Elide,
    Alias(AbstractValue, Value),
    Normal(AbstractValue),
    NewBlock(Block, AbstractValue, Value),
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
                log::info!(
                    " -> too many blocks or values: {} blocks {} values",
                    self.func.blocks.len(),
                    self.func.values.len()
                );
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
            pending_specialize: None,
            flow: self.state.block_entry[new_block].clone(),
        };
        log::trace!(" -> state = {:?}", state);

        state.flow.update_at_block_entry(
            &mut self.reg_map,
            &mut |reg_map, regslot, ty| {
                *reg_map
                    .entry((ctx, orig_block, regslot))
                    .or_insert_with(|| {
                        let param = self.func.add_placeholder(ty);
                        log::trace!(
                            "new blockparam {} of ty {:?} for reg slot {:?} on block {} (ctx {} orig {})",
                            param,
                            ty,
                            regslot,
                            new_block,
                            ctx,
                            orig_block,
                        );
                        param
                    })
            }
        )?;

        // Do the actual constant-prop, carrying the state across the
        // block and updating flow-sensitive state, and updating SSA
        // vals as well.
        let new_block = self
            .evaluate_block_body(orig_block, &mut state, new_block)
            .map_err(|e| {
                e.context(anyhow::anyhow!(
                    "Evaluating block body {} in func:\n{}",
                    orig_block,
                    self.generic.display("| ", Some(self.module))
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
        new_block: Block,
        orig_val: Value,
    ) -> (Value, AbstractValue) {
        log::trace!(
            "using value {} at block {} in context {}",
            orig_val,
            orig_block,
            context
        );
        if let Some(&val) = self.value_map.get(&(context, orig_val)) {
            if self.cfg.def_block[orig_val] != orig_block {
                self.value_dep_blocks
                    .entry((context, orig_val))
                    .or_default()
                    .insert(new_block);
            }
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

        if changed {
            if let Some(deps) = self.value_dep_blocks.get(&(context, orig_val)) {
                for &new_block in deps {
                    let (ctx, block) = self.block_rev_map[new_block];
                    if self.queue_set.insert((block, ctx)) {
                        self.queue.push_back((block, ctx, new_block));
                    }
                }
            }
        }

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
        mut new_block: Block,
    ) -> anyhow::Result<Block> {
        // Reused below for each instruction.
        let mut arg_abs_values = vec![];

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
                    let (val, _) = self.use_value(state.context, orig_block, new_block, *val);
                    Some((
                        ValueDef::PickOutput(val, *idx, *ty),
                        AbstractValue::Runtime(Some(inst)),
                    ))
                }
                ValueDef::Operator(op, args, tys) => {
                    let args_slice = &self.generic.arg_pool[*args];
                    let tys_slice = &self.generic.type_pool[*tys];

                    // Collect AbstractValues for args.
                    arg_abs_values.clear();
                    let mut arg_values = self.func.arg_pool.allocate(args.len(), Value::invalid());
                    for (i, &arg) in args_slice.iter().enumerate() {
                        log::trace!(" * arg {}", arg);
                        let arg = self.generic.resolve_alias(arg);
                        log::trace!(" -> resolves to arg {}", arg);
                        let (val, abs) = self.use_value(state.context, orig_block, new_block, arg);
                        arg_abs_values.push(abs);
                        self.func.arg_pool[arg_values][i] = val;
                    }
                    let loc = self.generic.source_locs[inst];

                    // Eval the transfer-function for this operator.
                    let result = self.abstract_eval(
                        orig_block,
                        new_block,
                        inst,
                        *op,
                        loc,
                        /* abstract values = */ &arg_abs_values[..],
                        /* new values = */ arg_values,
                        /* orig_values = */ args_slice,
                        tys_slice,
                        state,
                    )?;
                    // Transcribe either the original operation, or a
                    // constant, to the output.

                    let specialized_tys = self
                        .func
                        .type_pool
                        .from_iter(self.generic.type_pool[*tys].iter().cloned());
                    match result {
                        EvalResult::Unhandled => unreachable!(),
                        EvalResult::Alias(av, val) => Some((ValueDef::Alias(val), av)),
                        EvalResult::Elide => None,
                        EvalResult::Normal(AbstractValue::Concrete(bits)) if tys.len() == 1 => {
                            if let Some(const_op) = const_operator(tys_slice[0], bits) {
                                Some((
                                    ValueDef::Operator(
                                        const_op,
                                        ListRef::default(),
                                        specialized_tys,
                                    ),
                                    AbstractValue::Concrete(bits),
                                ))
                            } else {
                                Some((
                                    ValueDef::Operator(
                                        *op,
                                        std::mem::take(&mut arg_values),
                                        specialized_tys,
                                    ),
                                    AbstractValue::Runtime(Some(inst)),
                                ))
                            }
                        }
                        EvalResult::Normal(AbstractValue::StaticMemory(addr)) if tys.len() == 1 => {
                            let const_op =
                                const_operator(tys_slice[0], WasmVal::I32(addr)).unwrap();
                            Some((
                                ValueDef::Operator(const_op, ListRef::default(), specialized_tys),
                                AbstractValue::StaticMemory(addr),
                            ))
                        }
                        EvalResult::Normal(av) => Some((
                            ValueDef::Operator(
                                *op,
                                std::mem::take(&mut arg_values),
                                specialized_tys,
                            ),
                            av,
                        )),
                        EvalResult::NewBlock(block, av, value) => {
                            new_block = block;
                            Some((ValueDef::Alias(value), av))
                        }
                    }
                }
                ValueDef::Trace(id, args) => {
                    let new_args = self.func.arg_pool.allocate(args.len(), Value::invalid());
                    let args_slice = &self.generic.arg_pool[*args];
                    for (i, &arg) in args_slice.iter().enumerate() {
                        let arg = self.generic.resolve_alias(arg);
                        let (val, _abs) = self.use_value(state.context, orig_block, new_block, arg);
                        self.func.arg_pool[new_args][i] = val;
                    }
                    Some((
                        ValueDef::Trace(*id, new_args),
                        AbstractValue::Runtime(Some(inst)),
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
                self.func.source_locs[result_value] = self.generic.source_locs[inst];

                self.def_value(orig_block, input_ctx, inst, result_value, result_abs);
            }
        }

        Ok(new_block)
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
        _new_block: Block,
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
        new_block: Block,
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

        let target_block =
            self.target_block(state, orig_block, new_block, target.block, target_ctx);

        for &arg in &target.args {
            let arg = self.generic.resolve_alias(arg);
            let (val, abs) = self.use_value(state.context, orig_block, new_block, arg);
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
                    AbstractValue::Concrete(WasmVal::I32(val))
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
                assert!(!state.pending_specialize.is_some());
                let (cond, abs_cond) = self.use_value(state.context, orig_block, new_block, cond);
                // Update pending context with new stack if necessary.
                match abs_cond.as_const_truthy() {
                    Some(true) => Terminator::Br {
                        target: self.evaluate_block_target(
                            orig_block,
                            new_block,
                            state,
                            new_context,
                            if_true,
                        ),
                    },
                    Some(false) => Terminator::Br {
                        target: self.evaluate_block_target(
                            orig_block,
                            new_block,
                            state,
                            new_context,
                            if_false,
                        ),
                    },
                    None => Terminator::CondBr {
                        cond,
                        if_true: self.evaluate_block_target(
                            orig_block,
                            new_block,
                            state,
                            new_context,
                            if_true,
                        ),
                        if_false: self.evaluate_block_target(
                            orig_block,
                            new_block,
                            state,
                            new_context,
                            if_false,
                        ),
                    },
                }
            }
            &Terminator::Br { ref target } => {
                if let Some((index, lo, hi)) = state.pending_specialize.take() {
                    log::trace!(
                        "Branch to target {} with PendingSpecialize on {}",
                        target.block,
                        index
                    );
                    let index_of_value = target.args.iter().position(|&arg| arg == index).unwrap();
                    let target_specialized_value =
                        self.generic.blocks[target.block].params[index_of_value].1;
                    let mut targets: Vec<BlockTarget> = (lo..=hi)
                        .map(|i| {
                            let c = self.state.contexts.create(
                                Some(new_context),
                                ContextElem::Specialized(target_specialized_value, i),
                            );
                            log::trace!(" -> created new context {} for index {}", c, i);
                            self.evaluate_block_target(orig_block, new_block, state, c, target)
                        })
                        .collect();
                    let default = targets.pop().unwrap();
                    let (value, _) = self.use_value(state.context, orig_block, new_block, index);
                    Terminator::Select {
                        value,
                        targets,
                        default,
                    }
                } else {
                    // Update pending context with new stack if necessary.
                    Terminator::Br {
                        target: self.evaluate_block_target(
                            orig_block,
                            new_block,
                            state,
                            new_context,
                            target,
                        ),
                    }
                }
            }
            &Terminator::Select {
                value,
                ref targets,
                ref default,
            } => {
                assert!(!state.pending_specialize.is_some());
                let (value, abs_value) =
                    self.use_value(state.context, orig_block, new_block, value);
                if let Some(selector) = abs_value.as_const_u32() {
                    let selector = selector as usize;
                    let target = if selector < targets.len() {
                        &targets[selector]
                    } else {
                        default
                    };
                    Terminator::Br {
                        target: self.evaluate_block_target(
                            orig_block,
                            new_block,
                            state,
                            new_context,
                            target,
                        ),
                    }
                } else {
                    let targets = targets
                        .iter()
                        .map(|target| {
                            self.evaluate_block_target(
                                orig_block,
                                new_block,
                                state,
                                new_context,
                                target,
                            )
                        })
                        .collect::<Vec<_>>();
                    let default = self.evaluate_block_target(
                        orig_block,
                        new_block,
                        state,
                        new_context,
                        default,
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
                    .map(|&value| {
                        self.use_value(state.context, orig_block, new_block, value)
                            .0
                    })
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
        values: ListRef<Value>,
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

        let reg_result =
            self.abstract_eval_regs(orig_inst, new_block, op, abs, values, tys, state)?;
        if reg_result.is_handled() {
            log::debug!(" -> specialization regs: {:?}", reg_result);
            return Ok(reg_result);
        }

        let ret = if op.is_call() {
            log::debug!(" -> call");
            AbstractValue::Runtime(Some(orig_inst))
        } else {
            match abs.len() {
                0 => self.abstract_eval_nullary(orig_inst, op, state),
                1 => self.abstract_eval_unary(orig_inst, op, &abs[0], orig_values[0], state)?,
                2 => self.abstract_eval_binary(orig_inst, op, &abs[0], &abs[1]),
                3 => self.abstract_eval_ternary(orig_inst, op, &abs[0], &abs[1], &abs[2]),
                _ => AbstractValue::Runtime(Some(orig_inst)),
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
        values: ListRef<Value>,
        orig_values: &[Value],
        state: &mut PointState,
    ) -> EvalResult {
        match op {
            Operator::Call { function_index } => {
                if Some(function_index) == self.intrinsics.push_context {
                    let pc = abs[0]
                        .as_const_u32_or_mem_offset()
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
                    let pending_context = if let Some(pc) = abs[0].as_const_u32_or_mem_offset() {
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
                } else if Some(function_index) == self.intrinsics.context_bucket {
                    let instantaneous_context = state.pending_context.unwrap_or(state.context);
                    let bucket = abs[0].as_const_u32().unwrap();
                    self.state.contexts.context_bucket[instantaneous_context] = Some(bucket);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.specialize_value {
                    let lo = abs[1].as_const_u32().unwrap();
                    let hi = abs[2].as_const_u32().unwrap();
                    log::trace!(
                        "Creating pending-specialize state for index {} lo {} hi {}",
                        orig_inst,
                        lo,
                        hi
                    );
                    state.pending_specialize = Some((orig_inst, lo, hi));
                    EvalResult::Alias(abs[0].clone(), self.func.arg_pool[values][0])
                } else if Some(function_index) == self.intrinsics.abort_specialization {
                    let line_num = abs[0].as_const_u32().unwrap_or(0);
                    let fatal = abs[1].as_const_u32().unwrap_or(0);
                    log::trace!("abort-specialization point: line {}", line_num);
                    if fatal != 0 {
                        panic!("Specialization reached a point it shouldn't have!");
                    }
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.trace_line {
                    let line_num = abs[0].as_const_u32().unwrap_or(0);
                    log::debug!("trace: line number {}: current context {} at block {}, pending context {:?}",
                                line_num, state.context, orig_block, state.pending_context);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.assert_const32 {
                    log::trace!("assert_const32: abs {:?} line {:?}", abs[0], abs[1]);
                    if abs[0].as_const_u32().is_none() {
                        panic!(
                            "weval_assert_const32() failed: {:?}: line {:?}",
                            abs[0], abs[1]
                        );
                    }
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.print {
                    let message_ptr = abs[0].as_const_u32().unwrap();
                    let message = self
                        .image
                        .read_str(self.image.main_heap.unwrap(), message_ptr)
                        .unwrap();
                    let line = abs[1].as_const_u32().unwrap();
                    let val = abs[2].clone();
                    log::info!("print: line {}: {}: {:?}", line, message, val);
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.read_specialization_global {
                    let index = abs[0].as_const_u32().unwrap() as usize;
                    let i64_ty = self.func.single_type_list(Type::I64);
                    let value = self.func.add_value(ValueDef::Operator(
                        Operator::I64Const { value: 0 },
                        ListRef::default(),
                        i64_ty,
                    ));
                    self.func.blocks[new_block].insts.push(value);
                    let state = self.state.specialization_globals[index].clone();
                    log::trace!(
                        "read_specialization_global: index {}: state = {:?}",
                        index,
                        state
                    );
                    EvalResult::Alias(state, value)
                } else if Some(function_index) == self.intrinsics.push_stack {
                    let stackptr = self.func.arg_pool[values][0];
                    let value = self.func.arg_pool[values][1];
                    log::trace!(
                        "push_stack: value {}, current stack is {:?}",
                        value,
                        state.flow.stack,
                    );
                    log::trace!("push_stack: value {} stackptr {}", value, stackptr);
                    state.flow.stack.insert(
                        0,
                        (
                            RegValue::Value {
                                data: stackptr,
                                ty: Type::I32,
                                abs: abs[0].clone(),
                            },
                            RegValue::Value {
                                data: value,
                                ty: Type::I64,
                                abs: abs[1].clone(),
                            },
                        ),
                    );
                    self.stats.virtstack_writes += 1;
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.pop_stack {
                    log::trace!("pop_stack: current stack is {:?}", state.flow.stack);
                    self.stats.virtstack_reads += 1;
                    if state.flow.stack.len() > 0 {
                        let (_, reg) = state.flow.stack.remove(0);
                        let (value, abs) = match reg {
                            RegValue::Value { data, abs, .. } => (data, abs),
                            _ => unreachable!(),
                        };
                        EvalResult::Alias(abs, value)
                    } else {
                        let ptr = self.func.arg_pool[values][0];
                        let i64_ty = self.func.single_type_list(Type::I64);
                        let args = self.func.arg_pool.single(ptr);
                        let load = self.func.add_value(ValueDef::Operator(
                            Operator::I64Load {
                                memory: MemoryArg {
                                    align: 1,
                                    offset: 0,
                                    memory: self.image.main_heap().unwrap(),
                                },
                            },
                            args,
                            i64_ty,
                        ));
                        self.stats.virtstack_reads_mem += 1;
                        self.func.blocks[new_block].insts.push(load);
                        EvalResult::Alias(AbstractValue::Runtime(None), load)
                    }
                } else if Some(function_index) == self.intrinsics.read_stack {
                    let idx = abs[1].as_const_u32().unwrap();
                    log::trace!(
                        "read_stack: index {}, current stack is {:?}",
                        idx,
                        state.flow.stack
                    );
                    self.stats.virtstack_reads += 1;
                    if let Some((_, data)) = state.flow.stack.get(idx as usize) {
                        let (value, abs) = match data {
                            RegValue::Value { data, abs, .. } => (*data, abs.clone()),
                            _ => unreachable!(),
                        };
                        EvalResult::Alias(abs, value)
                    } else {
                        let ptr = self.func.arg_pool[values][0];
                        let i64_ty = self.func.single_type_list(Type::I64);
                        let args = self.func.arg_pool.single(ptr);
                        let load = self.func.add_value(ValueDef::Operator(
                            Operator::I64Load {
                                memory: MemoryArg {
                                    align: 1,
                                    offset: 0,
                                    memory: self.image.main_heap().unwrap(),
                                },
                            },
                            args,
                            i64_ty,
                        ));
                        self.func.blocks[new_block].insts.push(load);
                        self.stats.virtstack_reads_mem += 1;
                        EvalResult::Alias(AbstractValue::Runtime(None), load)
                    }
                } else if Some(function_index) == self.intrinsics.write_stack {
                    let stackptr = self.func.arg_pool[values][0];
                    let idx = abs[1].as_const_u32().unwrap();
                    let value = self.func.arg_pool[values][2];
                    log::trace!(
                        "write_stack: index {}, value {}, current stack is {:?}",
                        idx,
                        value,
                        state.flow.stack
                    );
                    let addr_value = RegValue::Value {
                        data: stackptr,
                        abs: abs[0].clone(),
                        ty: Type::I32,
                    };
                    let data_value = RegValue::Value {
                        data: value,
                        abs: abs[2].clone(),
                        ty: Type::I64,
                    };
                    self.stats.virtstack_writes += 1;
                    if let Some((addr, data)) = state.flow.stack.get_mut(idx as usize) {
                        log::trace!("write_stack: value {} stackptr {}", value, stackptr);
                        *addr = addr_value;
                        *data = data_value;
                    } else if idx == 0 && state.flow.stack.is_empty() {
                        state.flow.stack.push((addr_value, data_value));
                    } else {
                        let args = self.func.arg_pool.double(stackptr, value);
                        let store = self.func.add_value(ValueDef::Operator(
                            Operator::I64Store {
                                memory: MemoryArg {
                                    align: 1,
                                    offset: 0,
                                    memory: self.image.main_heap().unwrap(),
                                },
                            },
                            args,
                            ListRef::default(),
                        ));
                        self.stats.virtstack_writes_mem += 1;
                        self.func.blocks[new_block].insts.push(store);
                    }
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.sync_stack {
                    log::trace!("sync_stack current stack is {:?}", state.flow.stack);

                    for (addr, data) in state.flow.stack.drain(..) {
                        let addr = addr.value().unwrap();
                        let data = data.value().unwrap();
                        log::trace!("sync_stack: value {} stackptr {}", addr, data);
                        let args = self.func.arg_pool.double(addr, data);
                        let store = self.func.add_value(ValueDef::Operator(
                            Operator::I64Store {
                                memory: MemoryArg {
                                    align: 1,
                                    offset: 0,
                                    memory: self.image.main_heap().unwrap(),
                                },
                            },
                            args,
                            ListRef::default(),
                        ));
                        self.func.blocks[new_block].insts.push(store);
                        self.stats.virtstack_writes_mem += 1;
                    }

                    for (_, (addr, data)) in std::mem::take(&mut state.flow.locals) {
                        let addr = addr.value().unwrap();
                        let data = data.value().unwrap();
                        log::trace!("sync_stack: local addr {} data {}", addr, data);
                        let args = self.func.arg_pool.double(addr, data);
                        let store = self.func.add_value(ValueDef::Operator(
                            Operator::I64Store {
                                memory: MemoryArg {
                                    align: 1,
                                    offset: 0,
                                    memory: self.image.main_heap().unwrap(),
                                },
                            },
                            args,
                            ListRef::default(),
                        ));
                        self.func.blocks[new_block].insts.push(store);
                        self.stats.local_writes_mem += 1;
                    }
                    EvalResult::Elide
                } else if Some(function_index) == self.intrinsics.read_local {
                    self.stats.local_reads += 1;
                    let ptr = self.func.arg_pool[values][0];
                    let idx = abs[1].as_const_u32().unwrap();
                    match state.flow.locals.get(&idx) {
                        None => {
                            let args = self.func.arg_pool.single(ptr);
                            let i64_ty = self.func.single_type_list(Type::I64);
                            let load = self.func.add_value(ValueDef::Operator(
                                Operator::I64Load {
                                    memory: MemoryArg {
                                        align: 1,
                                        offset: 0,
                                        memory: self.image.main_heap().unwrap(),
                                    },
                                },
                                args,
                                i64_ty,
                            ));
                            self.func.blocks[new_block].insts.push(load);
                            self.stats.local_reads_mem += 1;
                            EvalResult::Alias(AbstractValue::Runtime(None), load)
                        }
                        Some((_, RegValue::Value { data, abs, .. })) => {
                            EvalResult::Alias(abs.clone(), *data)
                        }
                        _ => unreachable!(),
                    }
                } else if Some(function_index) == self.intrinsics.write_local {
                    self.stats.local_writes += 1;
                    let ptr = self.func.arg_pool[values][0];
                    let idx = abs[1].as_const_u32().unwrap();
                    let data = self.func.arg_pool[values][2];
                    state.flow.locals.insert(
                        idx,
                        (
                            RegValue::Value {
                                data: ptr,
                                abs: abs[0].clone(),
                                ty: Type::I32,
                            },
                            RegValue::Value {
                                data,
                                abs: abs[2].clone(),
                                ty: Type::I64,
                            },
                        ),
                    );
                    EvalResult::Elide
                } else {
                    EvalResult::Unhandled
                }
            }
            _ => EvalResult::Unhandled,
        }
    }

    fn abstract_eval_regs(
        &mut self,
        _inst: Value,
        _new_block: Block,
        op: Operator,
        abs: &[AbstractValue],
        vals: ListRef<Value>,
        _tys: &[Type],
        state: &mut PointState,
    ) -> anyhow::Result<EvalResult> {
        match op {
            Operator::Call { function_index }
                if Some(function_index) == self.intrinsics.read_reg =>
            {
                let idx = abs[0].as_const_u64().expect("Non-constant register number");
                log::trace!("load from specialization reg {}", idx);
                let slot = RegSlot::Register(idx as u32);
                match state.flow.regs.get(&slot) {
                    Some(RegValue::Value { data, abs, .. }) => {
                        log::trace!(" -> have value {} with abs {:?}", data, abs);
                        return Ok(EvalResult::Alias(abs.clone(), *data));
                    }
                    Some(v) => {
                        anyhow::bail!(
                            "Specialization register {} in bad state {:?} at read",
                            idx,
                            v
                        );
                    }
                    None => {
                        anyhow::bail!("Specialization register {} not set", idx);
                    }
                }
            }
            Operator::Call { function_index }
                if Some(function_index) == self.intrinsics.write_reg =>
            {
                let idx = abs[0].as_const_u64().expect("Non-constant register number");
                let data = self.func.arg_pool[vals][1];
                log::trace!(
                    "store to specialization reg {} value {} abs {:?}",
                    idx,
                    data,
                    abs[1]
                );
                let slot = RegSlot::Register(idx as u32);
                state.flow.regs.insert(
                    slot,
                    RegValue::Value {
                        data,
                        ty: Type::I64,
                        abs: abs[1].clone(),
                    },
                );

                // Elide the store.
                return Ok(EvalResult::Elide);
            }
            _ => {}
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
                .unwrap_or(AbstractValue::Runtime(Some(orig_inst))),
            Operator::I32Const { .. }
            | Operator::I64Const { .. }
            | Operator::F32Const { .. }
            | Operator::F64Const { .. } => AbstractValue::Concrete(WasmVal::try_from(op).unwrap()),
            _ => AbstractValue::Runtime(Some(orig_inst)),
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
        match (op, x) {
            (Operator::GlobalSet { global_index }, av) => {
                state.flow.globals.insert(global_index, av.clone());
                Ok(AbstractValue::Runtime(Some(orig_inst)))
            }
            (Operator::I32Eqz, AbstractValue::Concrete(WasmVal::I32(k))) => {
                Ok(AbstractValue::Concrete(WasmVal::I32(if *k == 0 {
                    1
                } else {
                    0
                })))
            }
            (Operator::I64Eqz, AbstractValue::Concrete(WasmVal::I64(k))) => {
                Ok(AbstractValue::Concrete(WasmVal::I64(if *k == 0 {
                    1
                } else {
                    0
                })))
            }
            (Operator::I32Extend8S, AbstractValue::Concrete(WasmVal::I32(k))) => Ok(
                AbstractValue::Concrete(WasmVal::I32(*k as i8 as i32 as u32)),
            ),
            (Operator::I32Extend16S, AbstractValue::Concrete(WasmVal::I32(k))) => Ok(
                AbstractValue::Concrete(WasmVal::I32(*k as i16 as i32 as u32)),
            ),
            (Operator::I64Extend8S, AbstractValue::Concrete(WasmVal::I64(k))) => Ok(
                AbstractValue::Concrete(WasmVal::I64(*k as i8 as i64 as u64)),
            ),
            (Operator::I64Extend16S, AbstractValue::Concrete(WasmVal::I64(k))) => Ok(
                AbstractValue::Concrete(WasmVal::I64(*k as i16 as i64 as u64)),
            ),
            (Operator::I64Extend32S, AbstractValue::Concrete(WasmVal::I64(k))) => Ok(
                AbstractValue::Concrete(WasmVal::I64(*k as i32 as i64 as u64)),
            ),
            (Operator::I32Clz, AbstractValue::Concrete(WasmVal::I32(k))) => {
                Ok(AbstractValue::Concrete(WasmVal::I32(k.leading_zeros())))
            }
            (Operator::I64Clz, AbstractValue::Concrete(WasmVal::I64(k))) => Ok(
                AbstractValue::Concrete(WasmVal::I64(k.leading_zeros() as u64)),
            ),
            (Operator::I32Ctz, AbstractValue::Concrete(WasmVal::I32(k))) => {
                Ok(AbstractValue::Concrete(WasmVal::I32(k.trailing_zeros())))
            }
            (Operator::I64Ctz, AbstractValue::Concrete(WasmVal::I64(k))) => Ok(
                AbstractValue::Concrete(WasmVal::I64(k.trailing_zeros() as u64)),
            ),
            (Operator::I32Popcnt, AbstractValue::Concrete(WasmVal::I32(k))) => {
                Ok(AbstractValue::Concrete(WasmVal::I32(k.count_ones())))
            }
            (Operator::I64Popcnt, AbstractValue::Concrete(WasmVal::I64(k))) => {
                Ok(AbstractValue::Concrete(WasmVal::I64(k.count_ones() as u64)))
            }
            (Operator::I32WrapI64, AbstractValue::Concrete(WasmVal::I64(k))) => {
                Ok(AbstractValue::Concrete(WasmVal::I32(*k as u32)))
            }
            (Operator::I32WrapI64, AbstractValue::ConcreteMemory(buf, off)) => {
                Ok(AbstractValue::ConcreteMemory(buf.clone(), *off))
            }
            (Operator::I64ExtendI32S, AbstractValue::Concrete(WasmVal::I32(k))) => Ok(
                AbstractValue::Concrete(WasmVal::I64(*k as i32 as i64 as u64)),
            ),
            (Operator::I64ExtendI32U, AbstractValue::Concrete(WasmVal::I32(k))) => {
                Ok(AbstractValue::Concrete(WasmVal::I64(*k as u64)))
            }

            (Operator::I32Load { memory }, AbstractValue::ConcreteMemory(buf, offset))
            | (Operator::I32Load8U { memory }, AbstractValue::ConcreteMemory(buf, offset))
            | (Operator::I32Load8S { memory }, AbstractValue::ConcreteMemory(buf, offset))
            | (Operator::I32Load16U { memory }, AbstractValue::ConcreteMemory(buf, offset))
            | (Operator::I32Load16S { memory }, AbstractValue::ConcreteMemory(buf, offset)) => {
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

                let offset = offset
                    .checked_add(memory.offset)
                    .ok_or_else(|| anyhow::anyhow!("Invalid offset"))?;
                let mem = self.directive_args.const_memory[buf.0 as usize]
                    .as_ref()
                    .unwrap();
                let val = mem.read_size(offset, size)?;
                let val = AbstractValue::Concrete(WasmVal::I32(conv(val)));
                log::trace!(" -> produces {:?}", val);
                Ok(val)
            }

            (Operator::I64Load { memory }, AbstractValue::ConcreteMemory(buf, offset))
            | (Operator::I64Load8U { memory }, AbstractValue::ConcreteMemory(buf, offset))
            | (Operator::I64Load8S { memory }, AbstractValue::ConcreteMemory(buf, offset))
            | (Operator::I64Load16U { memory }, AbstractValue::ConcreteMemory(buf, offset))
            | (Operator::I64Load16S { memory }, AbstractValue::ConcreteMemory(buf, offset))
            | (Operator::I64Load32U { memory }, AbstractValue::ConcreteMemory(buf, offset))
            | (Operator::I64Load32S { memory }, AbstractValue::ConcreteMemory(buf, offset)) => {
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

                let offset = offset
                    .checked_add(memory.offset)
                    .ok_or_else(|| anyhow::anyhow!("Invalid offset"))?;

                let mem = self.directive_args.const_memory[buf.0 as usize]
                    .as_ref()
                    .unwrap();
                let val = mem.read_size(offset, size)?;
                let val = AbstractValue::Concrete(WasmVal::I64(conv(val)));
                log::trace!(" -> produces {:?}", val);
                Ok(val)
            }

            (Operator::I32Load { memory }, AbstractValue::StaticMemory(addr)) => {
                let addr = addr.checked_add(memory.offset).unwrap();
                let val = self.image.read_u32(self.image.main_heap()?, addr)?;
                Ok(AbstractValue::Concrete(WasmVal::I32(val)))
            }
            (Operator::I64Load { memory }, AbstractValue::StaticMemory(addr)) => {
                let addr = addr.checked_add(memory.offset).unwrap();
                let val = self.image.read_u64(self.image.main_heap()?, addr)?;
                Ok(AbstractValue::Concrete(WasmVal::I64(val)))
            }

            // TODO: FP and SIMD
            _ => Ok(AbstractValue::Runtime(Some(orig_inst))),
        }
    }

    fn abstract_eval_binary(
        &mut self,
        orig_inst: Value,
        op: Operator,
        x: &AbstractValue,
        y: &AbstractValue,
    ) -> AbstractValue {
        match (x, y) {
            (AbstractValue::Concrete(v1), AbstractValue::Concrete(v2)) => {
                match (op, v1, v2) {
                    // 32-bit comparisons.
                    (Operator::I32Eq, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 == k2 { 1 } else { 0 }))
                    }
                    (Operator::I32Ne, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 != k2 { 1 } else { 0 }))
                    }
                    (Operator::I32LtS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if (*k1 as i32) < (*k2 as i32) {
                            1
                        } else {
                            0
                        }))
                    }
                    (Operator::I32LtU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 < k2 { 1 } else { 0 }))
                    }
                    (Operator::I32GtS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if (*k1 as i32) > (*k2 as i32) {
                            1
                        } else {
                            0
                        }))
                    }
                    (Operator::I32GtU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 > k2 { 1 } else { 0 }))
                    }
                    (Operator::I32LeS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if (*k1 as i32) <= (*k2 as i32) {
                            1
                        } else {
                            0
                        }))
                    }
                    (Operator::I32LeU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 <= k2 { 1 } else { 0 }))
                    }
                    (Operator::I32GeS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if (*k1 as i32) >= (*k2 as i32) {
                            1
                        } else {
                            0
                        }))
                    }
                    (Operator::I32GeU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(if k1 >= k2 { 1 } else { 0 }))
                    }

                    // 64-bit comparisons.
                    (Operator::I64Eq, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 == k2 { 1 } else { 0 }))
                    }
                    (Operator::I64Ne, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 != k2 { 1 } else { 0 }))
                    }
                    (Operator::I64LtS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if (*k1 as i64) < (*k2 as i64) {
                            1
                        } else {
                            0
                        }))
                    }
                    (Operator::I64LtU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 < k2 { 1 } else { 0 }))
                    }
                    (Operator::I64GtS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if (*k1 as i64) > (*k2 as i64) {
                            1
                        } else {
                            0
                        }))
                    }
                    (Operator::I64GtU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 > k2 { 1 } else { 0 }))
                    }
                    (Operator::I64LeS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if (*k1 as i64) <= (*k2 as i64) {
                            1
                        } else {
                            0
                        }))
                    }
                    (Operator::I64LeU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 <= k2 { 1 } else { 0 }))
                    }
                    (Operator::I64GeS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if (*k1 as i64) >= (*k2 as i64) {
                            1
                        } else {
                            0
                        }))
                    }
                    (Operator::I64GeU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(if k1 >= k2 { 1 } else { 0 }))
                    }

                    // 32-bit integer arithmetic.
                    (Operator::I32Add, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_add(*k2)))
                    }
                    (Operator::I32Sub, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_sub(*k2)))
                    }
                    (Operator::I32Mul, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_mul(*k2)))
                    }
                    (Operator::I32DivU, WasmVal::I32(k1), WasmVal::I32(k2)) if *k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_div(*k2)))
                    }
                    (Operator::I32DivS, WasmVal::I32(k1), WasmVal::I32(k2))
                        if *k2 != 0 && (*k1 != 0x8000_0000 || *k2 != 0xffff_ffff) =>
                    {
                        AbstractValue::Concrete(WasmVal::I32(
                            (*k1 as i32).wrapping_div(*k2 as i32) as u32
                        ))
                    }
                    (Operator::I32RemU, WasmVal::I32(k1), WasmVal::I32(k2)) if *k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_rem(*k2)))
                    }
                    (Operator::I32RemS, WasmVal::I32(k1), WasmVal::I32(k2))
                        if *k2 != 0 && (*k1 != 0x8000_0000 || *k2 != 0xffff_ffff) =>
                    {
                        AbstractValue::Concrete(WasmVal::I32(
                            (*k1 as i32).wrapping_rem(*k2 as i32) as u32
                        ))
                    }
                    (Operator::I32And, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1 & k2))
                    }
                    (Operator::I32Or, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1 | k2))
                    }
                    (Operator::I32Xor, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1 ^ k2))
                    }
                    (Operator::I32Shl, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_shl(k2 & 0x1f)))
                    }
                    (Operator::I32ShrU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(k1.wrapping_shr(k2 & 0x1f)))
                    }
                    (Operator::I32ShrS, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        AbstractValue::Concrete(WasmVal::I32(
                            (*k1 as i32).wrapping_shr(*k2 & 0x1f) as u32
                        ))
                    }
                    (Operator::I32Rotl, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        let amt = k2 & 0x1f;
                        let result = k1.wrapping_shl(amt) | k1.wrapping_shr(32 - amt);
                        AbstractValue::Concrete(WasmVal::I32(result))
                    }
                    (Operator::I32Rotr, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                        let amt = k2 & 0x1f;
                        let result = k1.wrapping_shr(amt) | k1.wrapping_shl(32 - amt);
                        AbstractValue::Concrete(WasmVal::I32(result))
                    }

                    // 64-bit integer arithmetic.
                    (Operator::I64Add, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_add(*k2)))
                    }
                    (Operator::I64Sub, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_sub(*k2)))
                    }
                    (Operator::I64Mul, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_mul(*k2)))
                    }
                    (Operator::I64DivU, WasmVal::I64(k1), WasmVal::I64(k2)) if *k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_div(*k2)))
                    }
                    (Operator::I64DivS, WasmVal::I64(k1), WasmVal::I64(k2))
                        if *k2 != 0
                            && (*k1 != 0x8000_0000_0000_0000 || *k2 != 0xffff_ffff_ffff_ffff) =>
                    {
                        AbstractValue::Concrete(WasmVal::I64(
                            (*k1 as i64).wrapping_div(*k2 as i64) as u64
                        ))
                    }
                    (Operator::I64RemU, WasmVal::I64(k1), WasmVal::I64(k2)) if *k2 != 0 => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_rem(*k2)))
                    }
                    (Operator::I64RemS, WasmVal::I64(k1), WasmVal::I64(k2))
                        if *k2 != 0
                            && (*k1 != 0x8000_0000_0000_0000 || *k2 != 0xffff_ffff_ffff_ffff) =>
                    {
                        AbstractValue::Concrete(WasmVal::I64(
                            (*k1 as i64).wrapping_rem(*k2 as i64) as u64
                        ))
                    }
                    (Operator::I64And, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(*k1 & *k2))
                    }
                    (Operator::I64Or, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(*k1 | *k2))
                    }
                    (Operator::I64Xor, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(*k1 ^ *k2))
                    }
                    (Operator::I64Shl, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_shl((*k2 & 0x3f) as u32)))
                    }
                    (Operator::I64ShrU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(k1.wrapping_shr((*k2 & 0x3f) as u32)))
                    }
                    (Operator::I64ShrS, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        AbstractValue::Concrete(WasmVal::I64(
                            (*k1 as i64).wrapping_shr((*k2 & 0x3f) as u32) as u64,
                        ))
                    }
                    (Operator::I64Rotl, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        let amt = (*k2 & 0x3f) as u32;
                        let result = k1.wrapping_shl(amt) | k1.wrapping_shr(64 - amt);
                        AbstractValue::Concrete(WasmVal::I64(result))
                    }
                    (Operator::I64Rotr, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                        let amt = (*k2 & 0x3f) as u32;
                        let result = k1.wrapping_shr(amt) | k1.wrapping_shl(64 - amt);
                        AbstractValue::Concrete(WasmVal::I64(result))
                    }

                    // TODO: FP and SIMD ops.
                    _ => AbstractValue::Runtime(Some(orig_inst)),
                }
            }

            // ptr OP const | const OP ptr (commutative cases)
            (
                AbstractValue::ConcreteMemory(buf, offset),
                AbstractValue::Concrete(WasmVal::I32(k)),
            )
            | (
                AbstractValue::Concrete(WasmVal::I32(k)),
                AbstractValue::ConcreteMemory(buf, offset),
            ) if op == Operator::I32Add => {
                AbstractValue::ConcreteMemory(buf.clone(), offset.wrapping_add(*k))
            }
            (AbstractValue::StaticMemory(addr), AbstractValue::Concrete(WasmVal::I32(k)))
            | (AbstractValue::Concrete(WasmVal::I32(k)), AbstractValue::StaticMemory(addr))
                if op == Operator::I32Add =>
            {
                AbstractValue::StaticMemory(addr.wrapping_add(*k))
            }

            // ptr OP const (non-commutative cases)
            (
                AbstractValue::ConcreteMemory(buf, offset),
                AbstractValue::Concrete(WasmVal::I32(k)),
            ) if op == Operator::I32Sub => {
                AbstractValue::ConcreteMemory(buf.clone(), offset.wrapping_sub(*k))
            }

            // ptr OP ptr
            (
                AbstractValue::ConcreteMemory(buf1, offset1),
                AbstractValue::ConcreteMemory(buf2, offset2),
            ) if op == Operator::I32Sub && buf1 == buf2 => {
                AbstractValue::Concrete(WasmVal::I32(offset1.wrapping_sub(*offset2)))
            }

            _ => AbstractValue::Runtime(Some(orig_inst)),
        }
    }

    fn abstract_eval_ternary(
        &mut self,
        orig_inst: Value,
        op: Operator,
        x: &AbstractValue,
        y: &AbstractValue,
        z: &AbstractValue,
    ) -> AbstractValue {
        match (op, z) {
            (Operator::Select, AbstractValue::Concrete(v))
            | (Operator::TypedSelect { .. }, AbstractValue::Concrete(v)) => {
                if v.is_truthy() {
                    x.clone()
                } else {
                    y.clone()
                }
            }
            // Concrete-memory symbolic pointers are always truthy.
            (Operator::Select, AbstractValue::ConcreteMemory(..))
            | (Operator::TypedSelect { .. }, AbstractValue::ConcreteMemory(..)) => x.clone(),
            _ => AbstractValue::Runtime(Some(orig_inst)),
        }
    }

    fn add_blockparam_reg_args(&mut self) -> anyhow::Result<()> {
        // Examine regs in block input state of each
        // specialized block, and create blockparams for all values
        // that in the end were `BlockParam`.
        for (&(ctx, orig_block), &block) in &self.block_map {
            let succ_state = &self.state.block_entry[block];

            let mut regs = vec![];
            let mut handle_value = |idx: RegSlot, val: &RegValue| -> anyhow::Result<()> {
                let ty = val.ty();
                let val_blockparam = self.func.add_blockparam(block, ty);
                let orig_val = *self.reg_map.get(&(ctx, orig_block, idx)).ok_or_else(|| {
                    anyhow::anyhow!(
                        "placeholder val not found for reg idx {:?} at block {} (ctx {} orig {})",
                        idx,
                        block,
                        ctx,
                        orig_block,
                    )
                })?;
                self.func.set_alias(orig_val, val_blockparam);
                regs.push(idx);
                Ok(())
            };

            for (&idx, val) in &succ_state.regs {
                handle_value(idx, val)?;
            }
            for (i, (addr, data)) in succ_state.stack.iter().enumerate() {
                handle_value(RegSlot::StackAddr(i as u32), addr)?;
                handle_value(RegSlot::StackData(i as u32), data)?;
            }
            for (&i, (addr, data)) in succ_state.locals.iter() {
                handle_value(RegSlot::LocalAddr(i), addr)?;
                handle_value(RegSlot::LocalData(i), data)?;
            }

            for pred_idx in 0..self.func.blocks[block].preds.len() {
                let pred = self.func.blocks[block].preds[pred_idx];
                let pred_state = &self.state.block_exit[pred];
                let pred_succ_idx = self.func.blocks[block].pos_in_pred_succ[pred_idx];

                for &idx in &regs {
                    let pred_reg = match idx {
                        RegSlot::Register(_) => pred_state.regs.get(&idx).as_ref().unwrap(),
                        RegSlot::StackAddr(i) => &pred_state.stack.get(i as usize).unwrap().0,
                        RegSlot::StackData(i) => &pred_state.stack.get(i as usize).unwrap().1,
                        RegSlot::LocalAddr(i) => &pred_state.locals.get(&i).unwrap().0,
                        RegSlot::LocalData(i) => &pred_state.locals.get(&i).unwrap().1,
                    };
                    let pred_val = pred_reg.value().unwrap();
                    self.func.blocks[pred]
                        .terminator
                        .update_target(pred_succ_idx, |target| {
                            target.args.push(pred_val);
                        });
                }
            }
        }

        Ok(())
    }

    fn insert_stack_syncs(&mut self) {
        // For each edge, look at known stack depth of pred and
        // succ. If succ's range is smaller, read regs from pred and
        // sync at end of pred.
        //
        // Also look at `locals` and find locals present in pred and
        // not in some succ, and sync them.
        for (_, &block) in &self.block_map {
            if self.func.blocks[block].succs.is_empty() {
                continue;
            }

            let pred_state = &self.state.block_exit[block];
            let pred_depth = pred_state.stack.len();
            let succ_min_depth = self.func.blocks[block]
                .succs
                .iter()
                .map(|succ| self.state.block_entry[*succ].stack.len())
                .min()
                .unwrap();

            for i in succ_min_depth..pred_depth {
                let addr = pred_state.stack[i].0.value().unwrap();
                let data = pred_state.stack[i].1.value().unwrap();
                log::trace!(
                    "spilling {} back to real stack memory: addr {} data {}",
                    i,
                    addr,
                    data
                );
                let args = self.func.arg_pool.double(addr, data);
                let store = self.func.add_value(ValueDef::Operator(
                    Operator::I64Store {
                        memory: MemoryArg {
                            align: 1,
                            offset: 0,
                            memory: self.image.main_heap().unwrap(),
                        },
                    },
                    args,
                    ListRef::default(),
                ));
                self.func.blocks[block].insts.push(store);
            }

            let locals_to_sync = pred_state
                .locals
                .keys()
                .filter(|key| {
                    self.func.blocks[block]
                        .succs
                        .iter()
                        .any(|succ| !self.state.block_entry[*succ].locals.contains_key(key))
                })
                .cloned()
                .collect::<Vec<_>>();
            for local in locals_to_sync {
                let (addr, data) = pred_state.locals.get(&local).unwrap();
                let addr = addr.value().unwrap();
                let data = data.value().unwrap();
                log::trace!(
                    "spilling local {} back to real locals memory: addr {} data {}",
                    local,
                    addr,
                    data
                );
                let args = self.func.arg_pool.double(addr, data);
                let store = self.func.add_value(ValueDef::Operator(
                    Operator::I64Store {
                        memory: MemoryArg {
                            align: 1,
                            offset: 0,
                            memory: self.image.main_heap().unwrap(),
                        },
                    },
                    args,
                    ListRef::default(),
                ));
                self.func.blocks[block].insts.push(store);
            }
        }
    }

    fn create_pre_entry(&mut self, specialized_entry: Block) -> Block {
        // Define a pre-entry block that ties supposedly
        // specialized-on-constant params to actual constants. This "bakes
        // in" the values so we don't need to provide them to the
        // specialized function. (The signature remains the same, we just
        // ignore the actual passed-in values.)
        let pre_entry = self.func.add_block();
        let mut pre_entry_args = vec![];
        for (ty, _) in &self.generic.blocks[self.generic.entry].params {
            let param = self.func.add_blockparam(pre_entry, *ty);
            pre_entry_args.push(param);
        }
        for (i, abs) in self
            .directive_args
            .const_params
            .iter()
            .skip(self.directive.num_globals as usize)
            .enumerate()
        {
            let ty = self.generic.blocks[self.generic.entry].params[i].0;
            match ty {
                Type::I32 => {
                    if let Some(value) = abs.as_const_u32() {
                        let tys = self.func.single_type_list(Type::I32);
                        let const_op = self.func.add_value(ValueDef::Operator(
                            Operator::I32Const { value },
                            ListRef::default(),
                            tys,
                        ));
                        self.func.append_to_block(pre_entry, const_op);
                        pre_entry_args[i] = const_op;
                    }
                }
                Type::I64 => {
                    if let Some(value) = abs.as_const_u64() {
                        let tys = self.func.single_type_list(Type::I64);
                        let const_op = self.func.add_value(ValueDef::Operator(
                            Operator::I64Const { value },
                            ListRef::default(),
                            tys,
                        ));
                        self.func.append_to_block(pre_entry, const_op);
                        pre_entry_args[i] = const_op;
                    }
                }
                _ => {}
            }
        }

        self.func.blocks[pre_entry].terminator = Terminator::Br {
            target: BlockTarget {
                block: specialized_entry,
                args: pre_entry_args,
            },
        };

        pre_entry
    }

    fn finalize(&mut self) -> anyhow::Result<()> {
        self.func.recompute_edges();

        self.add_blockparam_reg_args()?;
        self.insert_stack_syncs();

        #[cfg(debug_assertions)]
        self.func.validate().unwrap();

        Ok(())
    }
}
