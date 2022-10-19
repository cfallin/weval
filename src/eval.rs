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
use crate::state::State;
use crate::value::{Value, ValueTags, WasmVal};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use walrus::{
    ir::BinaryOp, ir::Instr, ir::InstrSeqId, ir::UnaryOp, FunctionBuilder, FunctionKind,
    LocalFunction, Module, ModuleFunctions, ModuleTypes,
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
        if let Some(idx) = partially_evaluate_func(module, im, &intrinsics, directive)? {
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
        _ => return Ok(None),
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
    // TODO: implicit return if `exit_state.fallthrough` is `Some`.
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
    taken: Vec<(OrigSeqId, State)>,
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
        for (taken_target, taken_state) in sub_seq.taken {
            if taken_target == this_seq {
                this_seq_taken = true;
                state.meet_with(&taken_state);
            } else {
                self.taken.push((taken_target, taken_state));
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
        self.seq_map.insert(from_seq, into_seq);

        let mut result = EvalResult {
            fallthrough: Some(state),
            taken: vec![],
        };

        for (instr, _) in &self.generic_fn.block(from_seq.0).instrs {
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
                        sub_result.taken.retain(|(to_seq, to_state)| {
                            if to_seq.0 != l.seq {
                                true
                            } else {
                                // Pick the PC out of the taken-state.
                                let taken_pc = to_state.loop_pcs.last().cloned().unwrap();

                                iters.get_mut(&iter_pc).unwrap().next_pcs.insert(taken_pc);

                                // If there's an existing entry in
                                // `iters`, meet the state into it; if
                                // input state changed, re-enqueue the
                                // iter for processing. Otherwise, add a
                                // new entry and enqueue for processing.
                                let needs_enqueue = match iters.entry(taken_pc) {
                                    Entry::Occupied(mut o) => {
                                        let changed = o.get_mut().input.meet_with(&to_state);
                                        // Re-enqueue only if input state changed.
                                        changed
                                    }
                                    Entry::Vacant(v) => {
                                        v.insert(IterState {
                                            input: to_state.clone(),
                                            code: None,
                                            output: None,
                                            next_pcs: HashSet::new(),
                                        });
                                        true
                                    }
                                };

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
                    todo!();

                    // Merge all out-states from iters: into our own
                    // current fallthrough when an iter falls through
                    // (*not* when it branches to the loop seqid), and
                    // into takens for all others. Note that we should
                    // assert that we've handled branches to the loop
                    // label in the process above.
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
                Instr::Unreachable(u) => {
                    self.builder.instr_seq(into_seq.0).unreachable();
                }
                Instr::Br(b) => {}
                Instr::BrIf(bi) => {}
                Instr::IfElse(ie) => {}
                Instr::BrTable(bt) => {}
                Instr::Drop(d) => {
                    self.builder.instr_seq(into_seq.0).drop();
                    result.cur().stack.pop();
                }
                Instr::Return(r) => {}
                Instr::MemorySize(ms) => {}
                Instr::MemoryGrow(mg) => {}
                Instr::Load(l) => {}
                Instr::Store(s) => {}
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
    if let Value::Concrete(v, _) = arg {
        match op {
            UnaryOp::I32Eqz => Value::Concrete(
                if v.integer_value().unwrap() == 0 {
                    WasmVal::I32(1)
                } else {
                    WasmVal::I32(0)
                },
                ValueTags::default(),
            ),
            UnaryOp::I32Clz => Value::Runtime(ValueTags::default()),
            UnaryOp::I32Ctz => Value::Runtime(ValueTags::default()),
            UnaryOp::I64Eqz => Value::Concrete(
                if v.integer_value().unwrap() == 0 {
                    WasmVal::I64(1)
                } else {
                    WasmVal::I64(0)
                },
                ValueTags::default(),
            ),
            UnaryOp::I64Clz => Value::Runtime(ValueTags::default()),
            UnaryOp::I64Ctz => Value::Runtime(ValueTags::default()),
            _ => Value::Runtime(ValueTags::default()),
        }
    } else {
        Value::Runtime(ValueTags::default())
    }
}

fn interpret_binop(op: BinaryOp, arg0: Value, arg1: Value) -> Value {
    Value::Runtime(ValueTags::default())
}
