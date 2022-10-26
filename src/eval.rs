//! Partial evaluation.

/* TODO:

- if/else scheme
- breaks out of blocks, and forward-edge state merging
- loops and per-state loop unrolling
- inlining
- "memory renaming": connecting symbolic ops through the operand-stack
  memory region
- more general memory-region handling: symbolic but unique
  (non-escaped) pointers, stack, operand-stack region, ...

*/

use crate::directive::Directive;
use crate::image::Image;
use crate::intrinsics::Intrinsics;
use crate::stackify::{rewrite_br_targets, stackify};
use crate::state::State;
use crate::value::{Value, ValueTags, WasmVal};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use walrus::{
    ir::BinaryOp, ir::ExtendedLoad, ir::Instr, ir::InstrSeqId, ir::InstrSeqType, ir::LoadKind,
    ir::UnaryOp, FunctionBuilder, FunctionKind, LocalFunction, Module, ModuleFunctions,
    ModuleTypes,
};

/// Partially evaluates according to the given directives.
pub fn partially_evaluate(
    module: &mut Module,
    im: &mut Image,
    directives: &[Directive],
) -> anyhow::Result<()> {
    let intrinsics = Intrinsics::find(module);
    let mut mem_updates = HashMap::new();
    for directive in directives {
        log::trace!("Processing directive {:?}", directive);
        if let Some(idx) = partially_evaluate_func(module, im, &intrinsics, directive)? {
            log::trace!("New func index {}", idx);
            // Update memory image.
            mem_updates.insert(directive.func_index_out_addr, idx);
        }
    }

    // Update memory: regenerate as one large data segment for heap 0 from the image.
    if let Some(main_heap_image) = im
        .main_heap
        .and_then(|main_heap| im.memories.get_mut(&main_heap))
    {
        for (addr, value) in mem_updates {
            let addr = addr as usize;
            if (addr + 4) > main_heap_image.image.len() {
                log::warn!(
                    "Cannot store function index for new function: address {:x} is out-of-bounds",
                    addr
                );
                continue;
            }
            main_heap_image.image[addr..(addr + 4)].copy_from_slice(&value.to_le_bytes()[..]);
        }
        Ok(())
    } else {
        anyhow::bail!("No image for main heap: cannot update");
    }
}

fn partially_evaluate_func(
    module: &mut Module,
    image: &Image,
    intrinsics: &Intrinsics,
    directive: &Directive,
) -> anyhow::Result<Option<u32>> {
    let lf = match &module.funcs.get(directive.func).kind {
        FunctionKind::Local(lf) => lf,
        _ => {
            log::trace!(
                "Cannot partially evaluate func {}: not a local func",
                directive.func.index(),
            );
            return Ok(None);
        }
    };
    let (param_tys, result_tys) = module.types.params_results(lf.ty());
    let param_tys = param_tys.to_vec();
    let result_tys = result_tys.to_vec();

    let mut builder = FunctionBuilder::new(&mut module.types, &param_tys[..], &result_tys[..]);
    let state = State::initial(
        module,
        image,
        directive.func,
        directive.const_params.clone(),
    );
    let from_seq = OrigSeqId(lf.entry_block());
    let into_seq = OutSeqId(builder.func_body_id());
    let mut ctx = EvalCtx {
        generic_fn: lf,
        builder: &mut builder,
        intrinsics,
        image,
        seq_map: HashMap::new(),
        tys: &module.types,
        funcs: &module.funcs,
    };

    let exit_state = ctx.eval_seq(state, from_seq, into_seq)?;
    if exit_state.fallthrough.is_some() {
        builder.instr_seq(into_seq.0).return_();
    }
    drop(exit_state);

    let specialized_fn = builder.finish(lf.args.clone(), &mut module.funcs);

    Ok(Some(specialized_fn.index() as u32))
}

/// Newtype around seq IDs in original (generic) function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct OrigSeqId(InstrSeqId);
/// Newtype around seq IDs in output (specialized) function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct OutSeqId(InstrSeqId);

struct EvalCtx<'a> {
    generic_fn: &'a LocalFunction,
    builder: &'a mut FunctionBuilder,
    intrinsics: &'a Intrinsics,
    image: &'a Image,
    /// Map from seq ID in original function to specialized function.
    seq_map: HashMap<OrigSeqId, OutSeqId>,
    tys: &'a ModuleTypes,
    funcs: &'a ModuleFunctions,
}

/// Evaluation result.
#[derive(Clone, Debug)]
struct EvalResult {
    /// The output state for fallthrough. `None` if unreachable.
    fallthrough: Option<State>,
    /// Any taken-edge states.
    taken: Vec<TakenEdge>,
}

#[derive(Clone, Debug)]
struct TakenEdge {
    target: OrigSeqId,
    state: State,
    seq: OutSeqId,
    instr: usize,
}

impl EvalResult {
    /// Returns `true` if merged-into seq was branched to.
    fn merge_subblock(&mut self, sub_seq: EvalResult, this_seq: OrigSeqId) -> bool {
        let state = self
            .fallthrough
            .as_mut()
            .expect("Cannot merge into unreachable state");
        if let Some(fallthrough) = sub_seq.fallthrough {
            state.meet_with(&fallthrough);
        }
        let mut this_seq_taken = false;
        for taken in sub_seq.taken {
            if taken.target == this_seq {
                this_seq_taken = true;
                state.meet_with(&taken.state);
            } else {
                self.taken.push(taken);
            }
        }
        this_seq_taken
    }

    pub fn cur(&mut self) -> &mut State {
        self.fallthrough
            .as_mut()
            .expect("Should be in reachable code when calling cur()")
    }
}

impl<'a> EvalCtx<'a> {
    fn eval_seq(
        &mut self,
        state: State,
        from_seq: OrigSeqId,
        into_seq: OutSeqId,
    ) -> anyhow::Result<EvalResult> {
        log::trace!(
            "eval_seq: from {:?} into {:?} state {:?}",
            from_seq,
            into_seq,
            state
        );

        self.seq_map.insert(from_seq, into_seq);

        let mut result = EvalResult {
            fallthrough: Some(state),
            taken: vec![],
        };

        for (instr, _) in &self.generic_fn.block(from_seq.0).instrs {
            if result.fallthrough.is_none() {
                break;
            }
            match instr {
                Instr::Block(b) => {
                    // Create a new output seq and recursively eval.
                    let ty = self.generic_fn.block(b.seq).ty;
                    let sub_from_seq = OrigSeqId(b.seq);
                    let sub_into_seq = OutSeqId(self.builder.dangling_instr_seq(ty).id());
                    let sub_state = result.cur().subblock_state(ty, &self.tys);
                    let sub_result = self.eval_seq(sub_state, sub_from_seq, sub_into_seq)?;
                    let block_used = result.merge_subblock(sub_result, from_seq);

                    if block_used {
                        // This `block` was actually branched to, so
                        // we need to keep it.
                        self.builder
                            .instr_seq(into_seq.0)
                            .instr(Instr::Block(walrus::ir::Block {
                                seq: sub_into_seq.0,
                            }));
                    } else {
                        // This `block` was not actually branched to,
                        // so we can remove it and just inline the
                        // resulting instructions directly (i.e.,
                        // discard the layering).
                        let instrs =
                            std::mem::take(self.builder.instr_seq(sub_into_seq.0).instrs_mut());
                        self.builder
                            .instr_seq(into_seq.0)
                            .instrs_mut()
                            .extend(instrs.into_iter());
                    }
                }
                Instr::Loop(l) => {
                    // Create the initial sub-block state.
                    let ty = self.generic_fn.block(l.seq).ty;
                    let sub_from_seq = OrigSeqId(l.seq);
                    let mut sub_state = result.cur().subblock_state(ty, &self.tys);
                    sub_state.loop_pcs.push(None);

                    // Create a map of PC -> iters.
                    #[derive(Debug)]
                    struct IterState {
                        input: State,
                        code: Option<OutSeqId>,
                        output: Option<EvalResult>,
                        next_pcs: HashSet<PC>,
                    }
                    type PC = Option<u64>;
                    let mut iters: HashMap<PC, IterState> = HashMap::new();

                    iters.insert(
                        None,
                        IterState {
                            input: sub_state,
                            code: None,
                            output: None,
                            next_pcs: HashSet::new(),
                        },
                    );
                    let mut workqueue: Vec<PC> = vec![];
                    let mut workqueue_set: HashSet<PC> = HashSet::new();
                    workqueue.push(None);
                    workqueue_set.insert(None);

                    while let Some(iter_pc) = workqueue.pop() {
                        workqueue_set.remove(&iter_pc);

                        // Get the current (input-state, output-code, output-state) for this PC.
                        let iter_state = iters.get_mut(&iter_pc).unwrap();
                        let in_state = iter_state.input.clone();

                        // Allocate an InstrSeq for the output, or
                        // reuse the one that was used last time we
                        // evaluated this iter.
                        let sub_into_seq = match iter_state.code {
                            Some(seq) => {
                                self.builder.instr_seq(seq.0).instrs_mut().clear();
                                seq
                            }
                            None => {
                                let new_seq = OutSeqId(self.builder.dangling_instr_seq(ty).id());
                                iter_state.code = Some(new_seq);
                                new_seq
                            }
                        };

                        // Evaluate the loop body.
                        let mut sub_result = self.eval_seq(in_state, sub_from_seq, sub_into_seq)?;

                        // Examine taken edges out of the sub_result
                        // for any other iters we need to add to the
                        // workqueue.
                        sub_result.taken.retain(|taken| {
                            if taken.target.0 != l.seq {
                                true
                            } else {
                                // Pick the PC out of the taken-state.
                                let taken_pc = taken.state.loop_pcs.last().cloned().unwrap();

                                iters.get_mut(&iter_pc).unwrap().next_pcs.insert(taken_pc);

                                // If there's an existing entry in
                                // `iters`, meet the state into it; if
                                // input state changed, re-enqueue the
                                // iter for processing. Otherwise, add a
                                // new entry and enqueue for processing.
                                let (dest_seq, needs_enqueue) = match iters.entry(taken_pc) {
                                    Entry::Occupied(mut o) => {
                                        let changed = o.get_mut().input.meet_with(&taken.state);
                                        // Re-enqueue only if input state changed.
                                        (o.get().code.unwrap().0, changed)
                                    }
                                    Entry::Vacant(v) => {
                                        let code = self.builder.dangling_instr_seq(ty).id();
                                        v.insert(IterState {
                                            input: taken.state.clone(),
                                            code: Some(OutSeqId(code)),
                                            output: None,
                                            next_pcs: HashSet::new(),
                                        });
                                        (code, true)
                                    }
                                };

                                rewrite_br_targets(
                                    &mut self.builder.instr_seq(taken.seq.0).instrs_mut()
                                        [taken.instr]
                                        .0,
                                    |target| {
                                        if target == l.seq {
                                            dest_seq
                                        } else {
                                            target
                                        }
                                    },
                                );

                                if needs_enqueue {
                                    if workqueue_set.insert(taken_pc) {
                                        workqueue.push(taken_pc);
                                    }
                                }

                                false
                            }
                        });

                        // Save the output state. (Look up in hashmap
                        // again to avoid holding the borrow over the
                        // above.)
                        let iter_state = iters.get_mut(&iter_pc).unwrap();
                        iter_state.output = Some(sub_result);
                    }

                    // Stitch together stamped-out specialized
                    // iterations, resolving branches to specific
                    // other iters and breaks out of the
                    // loop. Encapsulate the whole thing in a block
                    // and replace fallthroughs from individual iters
                    // with breaks out of the block.
                    let entry = iters.get(&None).unwrap().code.unwrap().0;
                    let body = stackify(
                        self.builder,
                        iters.values().map(|iter| iter.code.unwrap().0),
                        entry,
                    )?;

                    // Merge all out-states from iters: into our own
                    // current fallthrough when an iter falls through
                    // (*not* when it branches to the loop seqid), and
                    // into takens for all others. Note that we should
                    // assert that we've handled branches to the loop
                    // label in the process above.
                    for (_, state) in iters.into_iter() {
                        result.merge_subblock(state.output.unwrap(), from_seq);
                    }

                    // Emit a block with the stackified body.
                    self.builder
                        .instr_seq(into_seq.0)
                        .instr(Instr::Block(walrus::ir::Block { seq: body }));
                }
                Instr::Call(c) => {
                    // TODO: enqueue a directive for a specialized function?
                    // TODO: potentially invalidate operand stack caching?

                    // Determine if this is an intrinsic; handle specially if so.
                    if Some(c.func) == self.intrinsics.assume_const_memory {
                        // Add the tag to the value, but omit the call
                        // to the intrinsic in the specialized
                        // function.
                        let arg = result.cur().stack.pop().unwrap();
                        result
                            .cur()
                            .stack
                            .push(arg.with_tags(ValueTags::const_memory()));
                    } else if Some(c.func) == self.intrinsics.loop_pc32 {
                        // Set the known PC for the innermost loop in our state, if any.
                        if let &Value::Concrete(WasmVal::I32(new_pc), _) =
                            result.cur().stack.last().unwrap()
                        {
                            if let Some(pc) = result.cur().loop_pcs.last_mut() {
                                *pc = Some(new_pc as u64);
                            }
                        }
                        // Omit the actual intrinsic call, and leave the value on the stack.
                    } else if Some(c.func) == self.intrinsics.loop_pc64 {
                        // Set the known PC for the innermost loop in our state, if any.
                        if let &Value::Concrete(WasmVal::I64(new_pc), _) =
                            result.cur().stack.last().unwrap()
                        {
                            if let Some(pc) = result.cur().loop_pcs.last_mut() {
                                *pc = Some(new_pc);
                            }
                        }
                        // Omit the actual intrinsic call, and leave the value on the stack.
                    } else {
                        // For now, just emit the instruction directly,
                        // providing `Runtime` return values.
                        let callee = c.func;
                        let callee_type = self.funcs.get(callee).ty();
                        let (callee_args, callee_rets) = self.tys.params_results(callee_type);
                        result.cur().popn(callee_args.len());
                        self.builder.instr_seq(into_seq.0).call(callee);
                        result
                            .cur()
                            .pushn(callee_rets.len(), Value::Runtime(ValueTags::default()));
                    }
                }
                Instr::CallIndirect(ci) => {
                    // TODO: we can devirtualize when we have a concrete value.

                    let callee_type = ci.ty;
                    let callee_table = ci.table;
                    let (callee_args, callee_rets) = self.tys.params_results(callee_type);
                    result.cur().popn(callee_args.len());
                    self.builder
                        .instr_seq(into_seq.0)
                        .call_indirect(callee_type, callee_table);
                    result
                        .cur()
                        .pushn(callee_rets.len(), Value::Runtime(ValueTags::default()));
                }
                Instr::LocalGet(lg) => {
                    self.builder.instr_seq(into_seq.0).local_get(lg.local);
                    let value = result
                        .cur()
                        .locals
                        .get(&lg.local)
                        .cloned()
                        .unwrap_or(Value::Runtime(ValueTags::default()));
                    result.cur().stack.push(value);
                }
                Instr::LocalSet(ls) => {
                    self.builder.instr_seq(into_seq.0).local_set(ls.local);
                    let value = result.cur().stack.pop().unwrap();
                    result.cur().locals.insert(ls.local, value);
                }
                Instr::LocalTee(lt) => {
                    self.builder.instr_seq(into_seq.0).local_tee(lt.local);
                    let value = result.cur().stack.last().cloned().unwrap();
                    result.cur().locals.insert(lt.local, value);
                }
                Instr::GlobalGet(gg) => {
                    self.builder.instr_seq(into_seq.0).global_get(gg.global);
                    let value = result
                        .cur()
                        .globals
                        .get(&gg.global)
                        .cloned()
                        .unwrap_or(Value::Runtime(ValueTags::default()));
                    result.cur().stack.push(value);
                }
                Instr::GlobalSet(gs) => {
                    self.builder.instr_seq(into_seq.0).global_set(gs.global);
                    let value = result.cur().stack.pop().unwrap();
                    result.cur().globals.insert(gs.global, value);
                }
                Instr::Const(c) => {
                    self.builder.instr_seq(into_seq.0).const_(c.value);
                    let value = WasmVal::from(c.value);
                    result
                        .cur()
                        .stack
                        .push(Value::Concrete(value, ValueTags::default()));
                }
                Instr::Binop(b) => {
                    self.builder.instr_seq(into_seq.0).binop(b.op);
                    let arg0 = result.cur().stack.pop().unwrap();
                    let arg1 = result.cur().stack.pop().unwrap();
                    let ret = interpret_binop(b.op, arg0, arg1);
                    result.cur().stack.push(ret);
                }
                Instr::Unop(u) => {
                    self.builder.instr_seq(into_seq.0).unop(u.op);
                    let arg = result.cur().stack.pop().unwrap();
                    let ret = interpret_unop(u.op, arg);
                    result.cur().stack.push(ret);
                }
                Instr::Select(s) => {
                    self.builder.instr_seq(into_seq.0).select(s.ty);
                    let selector = result.cur().stack.pop().unwrap();
                    let b = result.cur().stack.pop().unwrap();
                    let a = result.cur().stack.pop().unwrap();
                    if let Value::Concrete(v, _) = selector {
                        result.cur().stack.push(if v.is_truthy() { a } else { b });
                    } else {
                        result
                            .cur()
                            .stack
                            .push(Value::Runtime(ValueTags::default()));
                    }
                }
                Instr::Unreachable(_) => {
                    self.builder.instr_seq(into_seq.0).unreachable();
                }
                Instr::Br(b) => {
                    // Don't handle block args for now.
                    let ty = self.generic_fn.block(b.block).ty;
                    assert_eq!(ty, InstrSeqType::Simple(None));

                    let instr = self.builder.instr_seq(into_seq.0).instrs().len();
                    self.builder
                        .instr_seq(into_seq.0)
                        .instr(Instr::Br(walrus::ir::Br {
                            block: self.seq_map.get(&OrigSeqId(b.block)).cloned().unwrap().0,
                        }));

                    let state = result.cur().clone();
                    result.taken.push(TakenEdge {
                        target: OrigSeqId(b.block),
                        state,
                        seq: into_seq,
                        instr,
                    });

                    result.fallthrough = None;
                }
                Instr::BrIf(brif) => {
                    // Don't handle block args for now.
                    let ty = self.generic_fn.block(brif.block).ty;
                    assert_eq!(ty, InstrSeqType::Simple(None));

                    // Pop the value off the stack. If known, we can
                    // either ignore this branch (if zero) or treat it
                    // like an unconditional one (if one). Otherwise,
                    // if runtime value, emit a br_if.
                    let cond = result.cur().stack.pop().unwrap();
                    let (instr, did_branch) = match cond.is_const_truthy() {
                        Some(true) => {
                            // Unconditional branch. Drop cond value
                            // that would have been used by br_if,
                            // then emit a br.
                            self.builder.instr_seq(into_seq.0).drop();
                            let instr = self.builder.instr_seq(into_seq.0).instrs().len();
                            self.builder
                                .instr_seq(into_seq.0)
                                .instr(Instr::Br(walrus::ir::Br {
                                    block: self
                                        .seq_map
                                        .get(&OrigSeqId(brif.block))
                                        .cloned()
                                        .unwrap()
                                        .0,
                                }));

                            result.fallthrough = None;

                            (instr, true)
                        }
                        Some(false) => {
                            // Never-taken branch: just emit a `drop`
                            // to drop the constant-zero value on the
                            // stack.
                            self.builder.instr_seq(into_seq.0).drop();

                            (0, false)
                        }
                        None => {
                            // Known at runtime only: emit a br_if.
                            let instr = self.builder.instr_seq(into_seq.0).instrs().len();
                            self.builder.instr_seq(into_seq.0).instr(Instr::BrIf(
                                walrus::ir::BrIf {
                                    block: self
                                        .seq_map
                                        .get(&OrigSeqId(brif.block))
                                        .cloned()
                                        .unwrap()
                                        .0,
                                },
                            ));

                            (instr, true)
                        }
                    };

                    if did_branch {
                        let state = result.cur().clone();
                        result.taken.push(TakenEdge {
                            target: OrigSeqId(brif.block),
                            state,
                            seq: into_seq,
                            instr,
                        });
                    }
                }
                Instr::BrTable(bt) => {
                    let selector = result.cur().stack.pop().unwrap();

                    if let Some(k) = selector.is_const_u32() {
                        // Known constant selector: drop, then emit an uncond br.
                        self.builder.instr_seq(into_seq.0).drop();
                        let instr = self.builder.instr_seq(into_seq.0).instrs().len();
                        let target = if (k as usize) < bt.blocks.len() {
                            bt.blocks[k as usize]
                        } else {
                            bt.default
                        };
                        let block = self.seq_map.get(&OrigSeqId(target)).cloned().unwrap().0;
                        self.builder
                            .instr_seq(into_seq.0)
                            .instr(Instr::Br(walrus::ir::Br { block }));
                        let state = result.cur().clone();
                        result.taken.push(TakenEdge {
                            target: OrigSeqId(target),
                            state,
                            seq: into_seq,
                            instr,
                        });
                        result.fallthrough = None;
                    } else {
                        let instr = self.builder.instr_seq(into_seq.0).instrs().len();
                        let blocks = bt
                            .blocks
                            .iter()
                            .map(|&block| self.seq_map.get(&OrigSeqId(block)).cloned().unwrap().0)
                            .collect::<Vec<InstrSeqId>>()
                            .into_boxed_slice();
                        let default = self.seq_map.get(&OrigSeqId(bt.default)).cloned().unwrap().0;
                        self.builder
                            .instr_seq(into_seq.0)
                            .instr(Instr::BrTable(walrus::ir::BrTable { blocks, default }));

                        for &block in bt.blocks.iter() {
                            let state = result.cur().clone();
                            result.taken.push(TakenEdge {
                                target: OrigSeqId(block),
                                state,
                                seq: into_seq,
                                instr,
                            });
                        }
                        let state = result.cur().clone();
                        result.taken.push(TakenEdge {
                            target: OrigSeqId(bt.default),
                            state,
                            seq: into_seq,
                            instr,
                        });
                    }
                }
                Instr::IfElse(ie) => {
                    let ty = self.generic_fn.block(ie.consequent).ty;
                    debug_assert_eq!(ty, self.generic_fn.block(ie.alternative).ty);
                    // Don't handle block args for now.
                    assert_eq!(ty, InstrSeqType::Simple(None));

                    let cond = result.cur().stack.pop().unwrap();

                    let (consequent_from_seq, alternative_from_seq) = match cond.is_const_truthy() {
                        Some(true) => (Some(OrigSeqId(ie.consequent)), None),
                        Some(false) => (None, Some(OrigSeqId(ie.alternative))),
                        None => (
                            Some(OrigSeqId(ie.consequent)),
                            Some(OrigSeqId(ie.alternative)),
                        ),
                    };

                    let mut f =
                        |sub_from_seq: Option<OrigSeqId>| -> anyhow::Result<Option<OutSeqId>> {
                            sub_from_seq
                                .map(|sub_from_seq| {
                                    let sub_into_seq =
                                        OutSeqId(self.builder.dangling_instr_seq(ty).id());
                                    let sub_sub_state = result.cur().subblock_state(ty, &self.tys);
                                    let sub_sub_result =
                                        self.eval_seq(sub_sub_state, sub_from_seq, sub_into_seq)?;
                                    result.merge_subblock(sub_sub_result, from_seq);
                                    Ok(sub_into_seq)
                                })
                                .transpose()
                        };

                    let consequent_into_seq = f(consequent_from_seq)?;
                    let alternative_into_seq = f(alternative_from_seq)?;

                    match (consequent_into_seq, alternative_into_seq) {
                        (Some(t), Some(f)) => {
                            self.builder.instr_seq(into_seq.0).instr(Instr::IfElse(
                                walrus::ir::IfElse {
                                    consequent: t.0,
                                    alternative: f.0,
                                },
                            ));
                        }
                        (Some(seq), None) | (None, Some(seq)) => {
                            self.builder.instr_seq(into_seq.0).drop();
                            self.builder
                                .instr_seq(into_seq.0)
                                .instr(Instr::Block(walrus::ir::Block { seq: seq.0 }));
                        }
                        (None, None) => unreachable!(),
                    }
                }
                Instr::Drop(_) => {
                    self.builder.instr_seq(into_seq.0).drop();
                    result.cur().stack.pop();
                }
                Instr::Return(_) => {
                    result.cur().stack.pop();
                    self.builder.instr_seq(into_seq.0).return_();
                }
                Instr::MemorySize(ms) => {
                    self.builder.instr_seq(into_seq.0).memory_size(ms.memory);
                    result
                        .cur()
                        .stack
                        .push(Value::Runtime(ValueTags::default()));
                }
                Instr::MemoryGrow(mg) => {
                    result.cur().stack.pop();
                    self.builder.instr_seq(into_seq.0).memory_grow(mg.memory);
                    result
                        .cur()
                        .stack
                        .push(Value::Runtime(ValueTags::default()));
                }
                Instr::Load(l) => {
                    // Get the address from the stack.
                    let addr = result.cur().stack.pop().unwrap();
                    let offset = l.arg.offset;
                    let size = (l.kind.width() / 8) as u32;

                    // Is it a known constant, with the `const_memory`
                    // `ValueTag`? If so, we can read bytes from the
                    // heap image to satisfy this load.
                    let value = match addr {
                        Value::Concrete(WasmVal::I32(base), tags)
                            if tags.contains(ValueTags::const_memory())
                                && self.image.can_read(l.memory, base + offset, size) =>
                        {
                            match l.kind {
                                LoadKind::I32 { .. } => Value::Concrete(
                                    WasmVal::I32(self.image.read_u32(l.memory, base + offset)?),
                                    tags,
                                ),
                                LoadKind::I64 { .. } => Value::Concrete(
                                    WasmVal::I64(self.image.read_u64(l.memory, base + offset)?),
                                    tags,
                                ),
                                LoadKind::I32_8 { kind } => {
                                    let u8val = self.image.read_u8(l.memory, base + offset)?;
                                    let ext = match kind {
                                        ExtendedLoad::SignExtend => (u8val as i8) as i32 as u32,
                                        ExtendedLoad::ZeroExtend
                                        | ExtendedLoad::ZeroExtendAtomic => u8val as u32,
                                    };
                                    Value::Concrete(WasmVal::I32(ext), tags)
                                }
                                LoadKind::I32_16 { kind } => {
                                    let u16val = self.image.read_u16(l.memory, base + offset)?;
                                    let ext = match kind {
                                        ExtendedLoad::SignExtend => (u16val as i16) as i32 as u32,
                                        ExtendedLoad::ZeroExtend
                                        | ExtendedLoad::ZeroExtendAtomic => u16val as u32,
                                    };
                                    Value::Concrete(WasmVal::I32(ext), tags)
                                }
                                LoadKind::I64_8 { kind } => {
                                    let u8val = self.image.read_u8(l.memory, base + offset)?;
                                    let ext = match kind {
                                        ExtendedLoad::SignExtend => (u8val as i8) as i64 as u64,
                                        ExtendedLoad::ZeroExtend
                                        | ExtendedLoad::ZeroExtendAtomic => u8val as u64,
                                    };
                                    Value::Concrete(WasmVal::I64(ext), tags)
                                }
                                LoadKind::I64_16 { kind } => {
                                    let u16val = self.image.read_u16(l.memory, base + offset)?;
                                    let ext = match kind {
                                        ExtendedLoad::SignExtend => (u16val as i16) as i64 as u64,
                                        ExtendedLoad::ZeroExtend
                                        | ExtendedLoad::ZeroExtendAtomic => u16val as u64,
                                    };
                                    Value::Concrete(WasmVal::I64(ext), tags)
                                }
                                LoadKind::I64_32 { kind } => {
                                    let u32val = self.image.read_u32(l.memory, base + offset)?;
                                    let ext = match kind {
                                        ExtendedLoad::SignExtend => (u32val as i32) as i64 as u64,
                                        ExtendedLoad::ZeroExtend
                                        | ExtendedLoad::ZeroExtendAtomic => u32val as u64,
                                    };
                                    Value::Concrete(WasmVal::I64(ext), tags)
                                }
                                LoadKind::F32 => Value::Concrete(
                                    WasmVal::F32(self.image.read_u32(l.memory, base + offset)?),
                                    tags,
                                ),
                                LoadKind::F64 => Value::Concrete(
                                    WasmVal::F64(self.image.read_u64(l.memory, base + offset)?),
                                    tags,
                                ),
                                LoadKind::V128 => Value::Concrete(
                                    WasmVal::V128(self.image.read_u128(l.memory, base + offset)?),
                                    tags,
                                ),
                            }
                        }
                        // TODO: handle loads from renamed operand stack.
                        _ => Value::Runtime(ValueTags::default()),
                    };
                    result.cur().stack.push(value);
                    self.builder
                        .instr_seq(into_seq.0)
                        .instr(Instr::Load(l.clone()));
                }
                Instr::Store(s) => {
                    // TODO: handle stores to renamed operand stack.
                    result.cur().stack.pop();
                    result.cur().stack.pop();
                    self.builder
                        .instr_seq(into_seq.0)
                        .instr(Instr::Store(s.clone()));
                }
                _ => {
                    anyhow::bail!("Unsupported instruction: {:?}", instr);
                }
            }
        }

        self.seq_map.remove(&from_seq);

        Ok(result)
    }
}

fn interpret_unop(op: UnaryOp, arg: Value) -> Value {
    if let Value::Concrete(v, tags) = arg {
        match (op, v) {
            (UnaryOp::I32Eqz, WasmVal::I32(k)) => {
                Value::Concrete(WasmVal::I32(std::cmp::min(k, 1)), tags)
            }
            (UnaryOp::I64Eqz, WasmVal::I64(k)) => {
                Value::Concrete(WasmVal::I64(std::cmp::min(k, 1)), tags)
            }
            (UnaryOp::I32Clz, WasmVal::I32(k)) => {
                Value::Concrete(WasmVal::I32(k.leading_zeros()), tags)
            }
            (UnaryOp::I64Clz, WasmVal::I64(k)) => {
                Value::Concrete(WasmVal::I64(k.leading_zeros() as u64), tags)
            }
            (UnaryOp::I32Ctz, WasmVal::I32(k)) => {
                Value::Concrete(WasmVal::I32(k.trailing_zeros()), tags)
            }
            (UnaryOp::I64Ctz, WasmVal::I64(k)) => {
                Value::Concrete(WasmVal::I64(k.trailing_zeros() as u64), tags)
            }
            (UnaryOp::I32Popcnt, WasmVal::I32(k)) => {
                Value::Concrete(WasmVal::I32(k.count_ones()), tags)
            }
            (UnaryOp::I64Popcnt, WasmVal::I64(k)) => {
                Value::Concrete(WasmVal::I64(k.count_ones() as u64), tags)
            }
            (UnaryOp::I32WrapI64, WasmVal::I64(k)) => Value::Concrete(WasmVal::I32(k as u32), tags),
            (UnaryOp::I32Extend8S, WasmVal::I32(k)) => {
                Value::Concrete(WasmVal::I32(k as u8 as i8 as i32 as u32), tags)
            }
            (UnaryOp::I32Extend16S, WasmVal::I32(k)) => {
                Value::Concrete(WasmVal::I32(k as u16 as i16 as i32 as u32), tags)
            }
            (UnaryOp::I64Extend8S, WasmVal::I64(k)) => {
                Value::Concrete(WasmVal::I64(k as u8 as i8 as i64 as u64), tags)
            }
            (UnaryOp::I64Extend16S, WasmVal::I64(k)) => {
                Value::Concrete(WasmVal::I64(k as u16 as i16 as i64 as u64), tags)
            }
            (UnaryOp::I64Extend32S, WasmVal::I64(k)) => {
                Value::Concrete(WasmVal::I64(k as u32 as i32 as i64 as u64), tags)
            }
            (UnaryOp::I64ExtendSI32, WasmVal::I32(k)) => {
                Value::Concrete(WasmVal::I64(k as i32 as i64 as u64), tags)
            }
            (UnaryOp::I64ExtendUI32, WasmVal::I32(k)) => {
                Value::Concrete(WasmVal::I64(k as u64), tags)
            }
            // TODO: FP and SIMD ops
            _ => Value::Runtime(ValueTags::default()),
        }
    } else {
        Value::Runtime(ValueTags::default())
    }
}

fn interpret_binop(op: BinaryOp, arg0: Value, arg1: Value) -> Value {
    match (arg0, arg1) {
        (Value::Concrete(v1, tag1), Value::Concrete(v2, tag2)) => {
            let tags = tag1.meet(tag2);
            match (op, v1, v2) {
                // 32-bit comparisons.
                (BinaryOp::I32Eq, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(if k1 == k2 { 1 } else { 0 }), tags)
                }
                (BinaryOp::I32Ne, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(if k1 != k2 { 1 } else { 0 }), tags)
                }
                (BinaryOp::I32LtS, WasmVal::I32(k1), WasmVal::I32(k2)) => Value::Concrete(
                    WasmVal::I32(if (k1 as i32) < (k2 as i32) { 1 } else { 0 }),
                    tags,
                ),
                (BinaryOp::I32LtU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(if k1 < k2 { 1 } else { 0 }), tags)
                }
                (BinaryOp::I32GtS, WasmVal::I32(k1), WasmVal::I32(k2)) => Value::Concrete(
                    WasmVal::I32(if (k1 as i32) > (k2 as i32) { 1 } else { 0 }),
                    tags,
                ),
                (BinaryOp::I32GtU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(if k1 > k2 { 1 } else { 0 }), tags)
                }
                (BinaryOp::I32LeS, WasmVal::I32(k1), WasmVal::I32(k2)) => Value::Concrete(
                    WasmVal::I32(if (k1 as i32) <= (k2 as i32) { 1 } else { 0 }),
                    tags,
                ),
                (BinaryOp::I32LeU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(if k1 <= k2 { 1 } else { 0 }), tags)
                }
                (BinaryOp::I32GeS, WasmVal::I32(k1), WasmVal::I32(k2)) => Value::Concrete(
                    WasmVal::I32(if (k1 as i32) >= (k2 as i32) { 1 } else { 0 }),
                    tags,
                ),
                (BinaryOp::I32GeU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(if k1 >= k2 { 1 } else { 0 }), tags)
                }

                // 64-bit comparisons.
                (BinaryOp::I64Eq, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(if k1 == k2 { 1 } else { 0 }), tags)
                }
                (BinaryOp::I64Ne, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(if k1 != k2 { 1 } else { 0 }), tags)
                }
                (BinaryOp::I64LtS, WasmVal::I64(k1), WasmVal::I64(k2)) => Value::Concrete(
                    WasmVal::I64(if (k1 as i64) < (k2 as i64) { 1 } else { 0 }),
                    tags,
                ),
                (BinaryOp::I64LtU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(if k1 < k2 { 1 } else { 0 }), tags)
                }
                (BinaryOp::I64GtS, WasmVal::I64(k1), WasmVal::I64(k2)) => Value::Concrete(
                    WasmVal::I64(if (k1 as i64) > (k2 as i64) { 1 } else { 0 }),
                    tags,
                ),
                (BinaryOp::I64GtU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(if k1 > k2 { 1 } else { 0 }), tags)
                }
                (BinaryOp::I64LeS, WasmVal::I64(k1), WasmVal::I64(k2)) => Value::Concrete(
                    WasmVal::I64(if (k1 as i64) <= (k2 as i64) { 1 } else { 0 }),
                    tags,
                ),
                (BinaryOp::I64LeU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(if k1 <= k2 { 1 } else { 0 }), tags)
                }
                (BinaryOp::I64GeS, WasmVal::I64(k1), WasmVal::I64(k2)) => Value::Concrete(
                    WasmVal::I64(if (k1 as i64) >= (k2 as i64) { 1 } else { 0 }),
                    tags,
                ),
                (BinaryOp::I64GeU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(if k1 >= k2 { 1 } else { 0 }), tags)
                }

                // 32-bit integer arithmetic.
                (BinaryOp::I32Add, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(k1.wrapping_add(k2)), tags)
                }
                (BinaryOp::I32Sub, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(k1.wrapping_sub(k2)), tags)
                }
                (BinaryOp::I32Mul, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(k1.wrapping_mul(k2)), tags)
                }
                (BinaryOp::I32DivU, WasmVal::I32(k1), WasmVal::I32(k2)) if k2 != 0 => {
                    Value::Concrete(WasmVal::I32(k1.wrapping_div(k2)), tags)
                }
                (BinaryOp::I32DivS, WasmVal::I32(k1), WasmVal::I32(k2))
                    if k2 != 0 && (k1 != 0x8000_0000 || k2 != 0xffff_ffff) =>
                {
                    Value::Concrete(
                        WasmVal::I32((k1 as i32).wrapping_div(k2 as i32) as u32),
                        tags,
                    )
                }
                (BinaryOp::I32RemU, WasmVal::I32(k1), WasmVal::I32(k2)) if k2 != 0 => {
                    Value::Concrete(WasmVal::I32(k1.wrapping_rem(k2)), tags)
                }
                (BinaryOp::I32RemS, WasmVal::I32(k1), WasmVal::I32(k2))
                    if k2 != 0 && (k1 != 0x8000_0000 || k2 != 0xffff_ffff) =>
                {
                    Value::Concrete(
                        WasmVal::I32((k1 as i32).wrapping_rem(k2 as i32) as u32),
                        tags,
                    )
                }
                (BinaryOp::I32And, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(k1 & k2), tags)
                }
                (BinaryOp::I32Or, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(k1 | k2), tags)
                }
                (BinaryOp::I32Xor, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(k1 ^ k2), tags)
                }
                (BinaryOp::I32Shl, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(k1.wrapping_shl(k2 & 0x1f)), tags)
                }
                (BinaryOp::I32ShrU, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    Value::Concrete(WasmVal::I32(k1.wrapping_shr(k2 & 0x1f)), tags)
                }
                (BinaryOp::I32ShrS, WasmVal::I32(k1), WasmVal::I32(k2)) => Value::Concrete(
                    WasmVal::I32((k1 as i32).wrapping_shr(k2 & 0x1f) as u32),
                    tags,
                ),
                (BinaryOp::I32Rotl, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    let amt = k2 & 0x1f;
                    let result = k1.wrapping_shl(amt) | k1.wrapping_shr(32 - amt);
                    Value::Concrete(WasmVal::I32(result), tags)
                }
                (BinaryOp::I32Rotr, WasmVal::I32(k1), WasmVal::I32(k2)) => {
                    let amt = k2 & 0x1f;
                    let result = k1.wrapping_shr(amt) | k1.wrapping_shl(32 - amt);
                    Value::Concrete(WasmVal::I32(result), tags)
                }

                // 64-bit integer arithmetic.
                (BinaryOp::I64Add, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(k1.wrapping_add(k2)), tags)
                }
                (BinaryOp::I64Sub, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(k1.wrapping_sub(k2)), tags)
                }
                (BinaryOp::I64Mul, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(k1.wrapping_mul(k2)), tags)
                }
                (BinaryOp::I64DivU, WasmVal::I64(k1), WasmVal::I64(k2)) if k2 != 0 => {
                    Value::Concrete(WasmVal::I64(k1.wrapping_div(k2)), tags)
                }
                (BinaryOp::I64DivS, WasmVal::I64(k1), WasmVal::I64(k2))
                    if k2 != 0 && (k1 != 0x8000_0000_0000_0000 || k2 != 0xffff_ffff_ffff_ffff) =>
                {
                    Value::Concrete(
                        WasmVal::I64((k1 as i64).wrapping_div(k2 as i64) as u64),
                        tags,
                    )
                }
                (BinaryOp::I64RemU, WasmVal::I64(k1), WasmVal::I64(k2)) if k2 != 0 => {
                    Value::Concrete(WasmVal::I64(k1.wrapping_rem(k2)), tags)
                }
                (BinaryOp::I64RemS, WasmVal::I64(k1), WasmVal::I64(k2))
                    if k2 != 0 && (k1 != 0x8000_0000_0000_0000 || k2 != 0xffff_ffff_ffff_ffff) =>
                {
                    Value::Concrete(
                        WasmVal::I64((k1 as i64).wrapping_rem(k2 as i64) as u64),
                        tags,
                    )
                }
                (BinaryOp::I64And, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(k1 & k2), tags)
                }
                (BinaryOp::I64Or, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(k1 | k2), tags)
                }
                (BinaryOp::I64Xor, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(k1 ^ k2), tags)
                }
                (BinaryOp::I64Shl, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(k1.wrapping_shl((k2 & 0x3f) as u32)), tags)
                }
                (BinaryOp::I64ShrU, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    Value::Concrete(WasmVal::I64(k1.wrapping_shr((k2 & 0x3f) as u32)), tags)
                }
                (BinaryOp::I64ShrS, WasmVal::I64(k1), WasmVal::I64(k2)) => Value::Concrete(
                    WasmVal::I64((k1 as i64).wrapping_shr((k2 & 0x3f) as u32) as u64),
                    tags,
                ),
                (BinaryOp::I64Rotl, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    let amt = (k2 & 0x3f) as u32;
                    let result = k1.wrapping_shl(amt) | k1.wrapping_shr(64 - amt);
                    Value::Concrete(WasmVal::I64(result), tags)
                }
                (BinaryOp::I64Rotr, WasmVal::I64(k1), WasmVal::I64(k2)) => {
                    let amt = (k2 & 0x3f) as u32;
                    let result = k1.wrapping_shr(amt) | k1.wrapping_shl(64 - amt);
                    Value::Concrete(WasmVal::I64(result), tags)
                }

                // TODO: FP and SIMD ops.
                _ => Value::Runtime(ValueTags::default()),
            }
        }
        _ => Value::Runtime(ValueTags::default()),
    }
}
