//! Stackify implementation to produce structured control flow from an
//! arbitrary CFG.
//!
//! Note on algorithm:
//!
//! - We sort in RPO, then mark loops, then place blocks within loops
//!   or at top level to give forward edges appropriate targets.
//!
//!  - The RPO sort order we choose is quite special: we need loop
//!    bodies to be placed contiguously, without blocks that do not
//!    belong to the loop in the middle. Otherwise we may not be able
//!    to properly nest a block to allow a forward edge.
//!
//! Consider the following CFG:
//!
//! ```plain
//!           1
//!           |
//!           2 <-.
//!         / |   |
//!        |  3 --'
//!        |  |
//!        `> 4
//!           |
//!           5
//! ```
//!
//! A normal RPO sort may produce 1, 2, 4, 5, 3 or 1, 2, 3, 4, 5
//! depending on which child order it chooses from block 2. (If it
//! visits 3 first, it will emit it first in postorder hence it comes
//! last.)
//!
//! One way of ensuring we get the right order would be to compute the
//! loop nest and make note of loops when choosing children to visit,
//! but we really would rather not do that, since we don't otherwise
//! have the infrastructure to compute that or the need for it.
//!
//! Instead, we keep a "pending" list: as we have nodes on the stack
//! during postorder traversal, we keep a list of other children that
//! we will visit once we get back to a given level. If another node
//! is pending, and is a successor we are considering, we visit it
//! *first* in postorder, so it is last in RPO. This is a way to
//! ensure that (e.g.) block 4 above is visited first when considering
//! successors of block 2.

use std::collections::{HashMap, HashSet};
use walrus::ir::{self, Instr, InstrSeqId, InstrSeqType};
use walrus::FunctionBuilder;

struct CFG<'a> {
    entry: InstrSeqId,
    succs: HashMap<InstrSeqId, HashSet<InstrSeqId>>,
    seqs: &'a HashSet<InstrSeqId>,
}

impl<'a> CFG<'a> {
    fn compute(
        builder: &mut FunctionBuilder,
        entry: InstrSeqId,
        seqs: &'a HashSet<InstrSeqId>,
    ) -> CFG<'a> {
        let mut workqueue = vec![entry];
        let mut succs = HashMap::new();
        let mut visited = HashSet::new();
        while let Some(seq) = workqueue.pop() {
            visited.insert(seq);
            let this_succs = succs.entry(seq).or_insert_with(HashSet::new);
            let mut visit_succ = |succ| {
                if seqs.contains(&succ) {
                    this_succs.insert(succ);
                    if visited.insert(succ) {
                        workqueue.push(succ);
                    }
                }
            };
            for (inst, _) in builder.instr_seq(seq).instrs() {
                match inst {
                    Instr::Br(br) => {
                        visit_succ(br.block);
                    }
                    Instr::BrIf(brif) => {
                        visit_succ(brif.block);
                    }
                    Instr::BrTable(brtable) => {
                        visit_succ(brtable.default);
                        for &block in &brtable.blocks[..] {
                            visit_succ(block);
                        }
                    }
                    _ => {}
                }
            }
        }

        CFG { entry, succs, seqs }
    }

    fn succs<'b>(&'b self, seq: InstrSeqId) -> impl Iterator<Item = InstrSeqId> + 'b {
        debug_assert!(self.seqs.contains(&seq));
        self.succs
            .get(&seq)
            .into_iter()
            .flat_map(|succs| succs.iter())
            .copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RPOIndex(u32);
impl RPOIndex {
    fn index(self) -> usize {
        self.0 as usize
    }
    fn prev(self) -> RPOIndex {
        RPOIndex(self.0.checked_sub(1).unwrap())
    }
}

struct RPO {
    order: Vec<InstrSeqId>,
    rev: HashMap<InstrSeqId, RPOIndex>,
}

impl RPO {
    fn compute(cfg: &CFG<'_>) -> RPO {
        let mut postorder = vec![];
        let mut visited = HashSet::new();
        let mut pending = vec![];
        let mut pending_idx = HashMap::new();
        visited.insert(cfg.entry);
        Self::visit(
            cfg,
            cfg.entry,
            &mut visited,
            &mut pending,
            &mut pending_idx,
            &mut postorder,
        );
        postorder.reverse();

        let mut rev = HashMap::new();
        for (i, block) in postorder.iter().copied().enumerate() {
            rev.insert(block, RPOIndex(i as u32));
        }

        RPO {
            order: postorder,
            rev,
        }
    }

    fn visit(
        cfg: &CFG<'_>,
        block: InstrSeqId,
        visited: &mut HashSet<InstrSeqId>,
        pending: &mut Vec<InstrSeqId>,
        pending_idx: &mut HashMap<InstrSeqId, usize>,
        postorder: &mut Vec<InstrSeqId>,
    ) {
        // `pending` is a Vec, not a Set; we prioritize based on
        // position (first in pending go first in postorder -> last in
        // RPO). A case with nested loops to show why this matters:
        //
        // TODO example

        let pending_top = pending.len();
        pending.extend(cfg.succs(block));

        // Sort new entries in `pending` by index at which they appear
        // earlier. Those that don't appear in `pending` at all should
        // be visited last (to appear in RPO first), so we want `None`
        // values to sort first here (hence the "unwrap or MAX"
        // idiom).  Then those that appear earlier in `pending` should
        // be visited earlier here to appear later in RPO, so they
        // sort later.
        pending[pending_top..]
            .sort_by_key(|entry| pending_idx.get(entry).copied().unwrap_or(usize::MAX));

        // Above we placed items in order they are to be visited;
        // below we pop off the end, so we reverse here.
        pending[pending_top..].reverse();

        // Now update indices in `pending_idx`: insert entries for
        // those seqs not yet present.
        for i in pending_top..pending.len() {
            pending_idx.entry(pending[i]).or_insert(i);
        }

        for _ in 0..(pending.len() - pending_top) {
            let succ = pending.pop().unwrap();
            if pending_idx.get(&succ) == Some(&pending.len()) {
                pending_idx.remove(&succ);
            }

            if visited.insert(succ) {
                Self::visit(cfg, succ, visited, pending, pending_idx, postorder);
            }
        }
        postorder.push(block);
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = InstrSeqId> + 'a {
        self.order.iter().copied()
    }

    pub fn iter_with_index<'a>(&'a self) -> impl Iterator<Item = (RPOIndex, InstrSeqId)> + 'a {
        self.order
            .iter()
            .copied()
            .enumerate()
            .map(|(i, block)| (RPOIndex(i as u32), block))
    }

    pub fn index_of_block(&self, block: InstrSeqId) -> Option<RPOIndex> {
        self.rev.get(&block).copied()
    }
}

/// Start and end marks for loops.
#[derive(Debug)]
struct Marks(HashMap<RPOIndex, Vec<Mark>>);

// Sorting-order note: Loop comes second, so Blocks sort first with
// smaller regions first. Thus, *reverse* sort order places loops
// outermost then larger blocks before smaller blocks.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Mark {
    Block { last_inclusive: RPOIndex },
    Loop { last_inclusive: RPOIndex },
}

impl Marks {
    fn compute(cfg: &CFG<'_>, rpo: &RPO) -> anyhow::Result<Marks> {
        let mut marks = HashMap::new();

        // Pass 1: Place loop markers.
        for (rpo_seq, seq) in rpo.iter_with_index() {
            for succ in cfg.succs(seq) {
                let rpo_succ = rpo.index_of_block(succ).unwrap();
                if rpo_succ <= rpo_seq {
                    let header_marks = marks.entry(rpo_succ).or_insert_with(|| vec![]);
                    debug_assert!(header_marks.len() <= 1);
                    if header_marks.is_empty() {
                        // Newly-discovered loop header.
                        header_marks.push(Mark::Loop {
                            last_inclusive: rpo_seq,
                        });
                    } else {
                        // Already-existing loop header. Adjust `last_inclusive`.
                        match &mut header_marks[0] {
                            Mark::Loop { last_inclusive } => {
                                *last_inclusive = std::cmp::max(*last_inclusive, rpo_seq);
                            }
                            Mark::Block { .. } => unreachable!(),
                        }
                    }
                }
            }
        }

        // Pass 2: compute location of innermost loop for each block.
        let mut innermost_loop: Vec<Option<RPOIndex>> = vec![None; rpo.order.len()];
        let mut loop_stack: Vec<(RPOIndex, RPOIndex)> = vec![];
        for (rpo_seq, _seq) in rpo.iter_with_index() {
            while let Some(innermost) = loop_stack.last() {
                if innermost.1 >= rpo_seq {
                    break;
                }
                loop_stack.pop();
            }

            if let Some(marks) = marks.get(&rpo_seq) {
                debug_assert_eq!(marks.len(), 1);
                let rpo_loop_end = match &marks[0] {
                    &Mark::Loop { last_inclusive } => last_inclusive,
                    _ => unreachable!(),
                };
                if let Some(innermost) = loop_stack.last() {
                    if rpo_loop_end > innermost.1 {
                        anyhow::bail!("Improperly nested loops!");
                    }
                }
                loop_stack.push((rpo_seq, rpo_loop_end));
            }

            innermost_loop[rpo_seq.index()] = loop_stack.last().map(|lp| lp.0);
        }

        // Pass 3: place block markers.
        for (rpo_seq, seq) in rpo.iter_with_index() {
            for succ in cfg.succs(seq) {
                let rpo_succ = rpo.index_of_block(succ).unwrap();
                if rpo_succ > rpo_seq {
                    // Determine the innermost loop for the target,
                    // and add the block just inside the loop.
                    let block_start = innermost_loop[rpo_succ.index()].unwrap_or(RPOIndex(0));
                    let start_marks = marks.entry(block_start).or_insert_with(|| vec![]);
                    let mark = Mark::Block {
                        last_inclusive: rpo_succ.prev(),
                    };
                    start_marks.push(mark);
                }
            }
        }

        // Sort markers at each block.
        for marklist in marks.values_mut() {
            marklist.sort();
            marklist.dedup();
            marklist.reverse();
        }

        Ok(Marks(marks))
    }
}

pub fn stackify(
    builder: &mut FunctionBuilder,
    seqs: impl IntoIterator<Item = InstrSeqId>,
    entry: InstrSeqId,
) -> anyhow::Result<InstrSeqId> {
    let seqs: HashSet<InstrSeqId> = seqs.into_iter().collect();
    let cfg = CFG::compute(builder, entry, &seqs);
    let rpo = RPO::compute(&cfg);
    let marks = Marks::compute(&cfg, &rpo)?;

    let mut target_rewrites: HashMap<InstrSeqId, InstrSeqId> = HashMap::new();
    let empty_ty = InstrSeqType::Simple(None);
    let top = builder.dangling_instr_seq(empty_ty).id();
    let mut cur = top;
    let mut stack: Vec<(RPOIndex, InstrSeqId)> = vec![];

    for (rpo_seq, seq) in rpo.iter_with_index() {
        // Pop back up to seq if any regions ended.
        while let Some(entry) = stack.last() {
            if entry.0 >= rpo_seq {
                break;
            }
            cur = entry.1;
            stack.pop();
        }

        if let Some(block_marks) = marks.0.get(&rpo_seq) {
            for mark in block_marks {
                match mark {
                    &Mark::Loop { last_inclusive } => {
                        // Create loop. Add rewrite target from `seq`
                        // to newly created seq. Add stack entry.
                        let subseq = builder.dangling_instr_seq(empty_ty).id();
                        target_rewrites.insert(seq, subseq);
                        builder
                            .instr_seq(cur)
                            .instr(Instr::Loop(ir::Loop { seq: subseq }));
                        stack.push((last_inclusive, cur));
                        cur = subseq;
                    }
                    &Mark::Block { last_inclusive } => {
                        // Create block. Add rewrite target from block
                        // following `last_inclusive` to newly created
                        // seq. Add stack entry.
                        let subseq = builder.dangling_instr_seq(empty_ty).id();
                        let following = rpo.order[last_inclusive.index() + 1];
                        target_rewrites.insert(following, subseq);
                        builder
                            .instr_seq(cur)
                            .instr(Instr::Block(ir::Block { seq: subseq }));
                        stack.push((last_inclusive, cur));
                        cur = subseq;
                    }
                }
            }
        }

        copy_instrs(builder, &mut target_rewrites, seq, cur);

        // Fallthrough: break out of the topmost block.
        builder.instr_seq(cur).instr(Instr::Br(ir::Br { block: top }));
    }

    Ok(top)
}

pub fn rewrite_br_targets<F: Fn(InstrSeqId) -> InstrSeqId>(instr: &mut Instr, f: F) {
    match instr {
        &mut Instr::Br(ref mut br) => {
            br.block = f(br.block);
        }
        &mut Instr::BrIf(ref mut brif) => {
            brif.block = f(brif.block);
        }
        &mut Instr::BrTable(ref mut brtable) => {
            for block in brtable.blocks.iter_mut() {
                *block = f(*block);
            }
            brtable.default = f(brtable.default);
        }
        _ => {}
    }
}

fn copy_instrs(
    builder: &mut FunctionBuilder,
    target_rewrites: &mut HashMap<InstrSeqId, InstrSeqId>,
    from: InstrSeqId,
    into: InstrSeqId,
) {
    let rewrite = |rewrites: &HashMap<InstrSeqId, InstrSeqId>, block: InstrSeqId| {
        rewrites.get(&block).copied().unwrap_or(block)
    };
    for i in 0..builder.instr_seq(from).instrs().len() {
        // Copy instruction, rewriting targets.
        let mut instr = builder.instr_seq(from).instrs()[i].clone().0;
        rewrite_br_targets(&mut instr, |target| rewrite(target_rewrites, target));
        let instr = match instr {
            Instr::Loop(lp) => {
                let subseq = builder.dangling_instr_seq(InstrSeqType::Simple(None)).id();
                target_rewrites.insert(lp.seq, subseq);
                copy_instrs(builder, target_rewrites, lp.seq, subseq);
                Instr::Loop(ir::Loop { seq: subseq })
            }
            Instr::Block(block) => {
                let subseq = builder.dangling_instr_seq(InstrSeqType::Simple(None)).id();
                target_rewrites.insert(block.seq, subseq);
                copy_instrs(builder, target_rewrites, block.seq, subseq);
                Instr::Block(ir::Block { seq: subseq })
            }
            Instr::IfElse(ifelse) => {
                let sub_consequent = builder.dangling_instr_seq(InstrSeqType::Simple(None)).id();
                target_rewrites.insert(ifelse.consequent, sub_consequent);
                let sub_alternative = builder.dangling_instr_seq(InstrSeqType::Simple(None)).id();
                target_rewrites.insert(ifelse.alternative, sub_alternative);
                copy_instrs(builder, target_rewrites, ifelse.consequent, sub_consequent);
                copy_instrs(
                    builder,
                    target_rewrites,
                    ifelse.alternative,
                    sub_alternative,
                );
                Instr::IfElse(ir::IfElse {
                    consequent: sub_consequent,
                    alternative: sub_alternative,
                })
            }
            i => i,
        };
        builder.instr_seq(into).instr(instr);
    }
}
