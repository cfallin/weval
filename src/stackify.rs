//! Stackify implementation to produce structured control flow from an
//! arbitrary CFG.

use std::collections::{HashMap, HashSet};
use walrus::ir::{Instr, InstrSeqId};
use walrus::{FunctionBuilder, InstrSeqBuilder};

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
        visited.insert(cfg.entry);
        Self::visit(cfg, cfg.entry, &mut visited, &mut postorder);
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
        postorder: &mut Vec<InstrSeqId>,
    ) {
        for succ in cfg.succs(block) {
            if visited.insert(succ) {
                Self::visit(cfg, succ, visited, postorder);
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

#[derive(Debug)]
enum Mark {
    Loop { last_inclusive: RPOIndex },
    Block { last_inclusive: RPOIndex },
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
        for (rpo_seq, seq) in rpo.iter_with_index() {
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
                }
            }
        }

        // Sort markers at each block.

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

    // TODO: iterate through blocks, building up seqs.

    // TODO: rewrite targets: forward edges need to branch to seqid of
    // block that ends just before actual target.

    todo!()
}
