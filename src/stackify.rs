//! Stackify implementation to produce structured control flow from an
//! arbitrary CFG.

use std::collections::{HashMap, HashSet};
use walrus::ir::{Instr, InstrSeqId};
use walrus::{FunctionBuilder, InstrSeqBuilder};

struct CFG {
    entry: InstrSeqId,
    succs: HashMap<InstrSeqId, HashSet<InstrSeqId>>,
}

impl CFG {
    fn compute(
        builder: &mut FunctionBuilder,
        entry: InstrSeqId,
        seqs: &HashSet<InstrSeqId>,
    ) -> CFG {
        let mut workqueue = vec![entry];
        let mut succs = HashMap::new();
        let mut visited = HashSet::new();
        while let Some(seq) = workqueue.pop() {
            visited.insert(seq);
            let mut this_succs = succs.entry(seq).or_insert_with(|| HashSet::new());
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

        CFG { entry, succs }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RPOIndex(u32);
impl RPOIndex {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

pub struct RPO {
    order: Vec<InstrSeqId>,
    rev: HashMap<InstrSeqId, RPOIndex>,
}

impl RPO {
    pub fn new(cfg: &CFG) -> RPO {
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
        cfg: &CFG,
        block: InstrSeqId,
        visited: &mut HashSet<InstrSeqId>,
        postorder: &mut Vec<InstrSeqId>,
    ) {
        if let Some(succs) = cfg.succs.get(&block) {
            for &succ in succs {
                if visited.insert(succ) {
                    Self::visit(cfg, succ, visited, postorder);
                }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label(u32);
