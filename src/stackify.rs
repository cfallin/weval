//! Stackify implementation to produce structured control flow from an
//! arbitrary CFG.

use std::collections::HashSet;

pub trait Block:
    Clone + Copy + std::fmt::Debug + PartialEq + Eq + PartialOrd + Ord + std::hash::Hash
{
}

pub trait CFG {
    type Block: Block;
    type Successors: Iterator<Item = Self::Block>;
    fn entry(&self) -> Self::Block;
    fn successors(&self, block: Self::Block) -> Self::Successors;
}

pub struct RPO<B: Block>(Vec<B>);

impl<B: Block> RPO<B> {
    pub fn new<C: CFG<Block = B>>(cfg: &C) -> RPO<B> {
        let mut postorder = vec![];
        let mut visited = HashSet::new();
        let entry = cfg.entry();
        visited.insert(entry);
        Self::visit(cfg, entry, &mut visited, &mut postorder);
        postorder.reverse();
        RPO(postorder)
    }

    fn visit<C: CFG<Block = B>>(
        cfg: &C,
        block: B,
        visited: &mut HashSet<B>,
        postorder: &mut Vec<B>,
    ) {
        for succ in cfg.successors(block) {
            if visited.insert(succ) {
                Self::visit(cfg, succ, visited, postorder);
            }
        }
        postorder.push(block);
    }
}
