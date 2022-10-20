//! Stackify implementation to produce structured control flow from an
//! arbitrary CFG.

use std::collections::{HashMap, HashSet};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RPOIndex(u32);
impl RPOIndex {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

pub struct RPO<B: Block> {
    order: Vec<B>,
    rev: HashMap<B, RPOIndex>,
}

impl<B: Block> RPO<B> {
    pub fn new<C: CFG<Block = B>>(cfg: &C) -> RPO<B> {
        let mut postorder = vec![];
        let mut visited = HashSet::new();
        let entry = cfg.entry();
        visited.insert(entry);
        Self::visit(cfg, entry, &mut visited, &mut postorder);
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

    fn visit<C: CFG<Block = B>>(
        cfg: &C,
        block: B,
        visited: &mut HashSet<B>,
        postorder: &mut Vec<B>,
    ) {
        let succs = cfg.successors(block).collect::<Vec<_>>();
        for succ in succs.into_iter().rev() {
            if visited.insert(succ) {
                Self::visit(cfg, succ, visited, postorder);
            }
        }
        postorder.push(block);
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = B> + 'a {
        self.order.iter().copied()
    }

    pub fn iter_with_index<'a>(&'a self) -> impl Iterator<Item = (RPOIndex, B)> + 'a {
        self.order
            .iter()
            .copied()
            .enumerate()
            .map(|(i, block)| (RPOIndex(i as u32), block))
    }

    pub fn index_of_block(&self, block: B) -> Option<RPOIndex> {
        self.rev.get(&block).copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label(u32);

#[derive(Clone, Debug)]
pub enum Node<B: Block> {
    /// A block, with its translated successors. (`None` means fallthrough.)
    Single { block: B, succs: Vec<Option<Label>> },
    /// A "block" region, allowing forward edges to its tail, with a
    /// sequence of child nodes.
    Region { label: Label, body: Vec<Node<B>> },
    /// A "loop", region, allowing backward edges to its head, with a
    /// sequence of child nodes.
    Loop { label: Label, body: Vec<Node<B>> },
}

pub struct Stackifier<B: Block> {
    /// RPO traversal.
    rpo: RPO<B>,
    /// Nested loop regions: sorted by start-block, with last block
    /// (inclusive).
    loops: Vec<(RPOIndex, RPOIndex)>,
    /// Loop ranges.
    loop_ranges: HashMap<RPOIndex, RPOIndex>,
    /// Innermost loop.
    innermost_loop: Vec<Option<RPOIndex>>,
    /// Loop parent relations, by header block.
    loop_parent: HashMap<RPOIndex, RPOIndex>,
    /// Forward-edge targets by loop (or toplevel, if key is `None`).
    forward_targets: HashMap<Option<RPOIndex>, HashSet<RPOIndex>>,
    /// In-progress state: next label.
    next_label: u32,
    /// In-progress state: labels assigned to blocks.
    labels: HashMap<RPOIndex, Label>,
}

impl<B: Block> Stackifier<B> {
    /// Builds auxiliary data structures.
    fn new<C: CFG<Block = B>>(cfg: &C) -> anyhow::Result<Stackifier<B>> {
        let rpo = RPO::new(cfg);
        // Scan RPO once and find latest block to branch backward to a
        // given loop header.
        let mut loop_backedge_origins = HashMap::new();
        for (idx, block) in rpo.iter_with_index() {
            for succ in cfg.successors(block) {
                let succ_rpo = rpo.index_of_block(succ).unwrap();
                if succ_rpo <= idx {
                    // succ_rpo is the loop header.
                    let entry = loop_backedge_origins.entry(succ_rpo).or_insert(idx);
                    *entry = std::cmp::max(*entry, idx);
                }
            }
        }

        // Compute loop regions from the loop backedge spans.
        let mut loops: Vec<_> = loop_backedge_origins
            .into_iter()
            .map(|(header, last_tail)| (header, last_tail))
            .collect();
        loops.sort();
        let loop_ranges = loops
            .iter()
            .copied()
            .collect::<HashMap<RPOIndex, RPOIndex>>();

        // Check proper nesting, and build innermost-loop and loop-parent data.
        let mut loop_stack: Vec<(RPOIndex, RPOIndex)> = vec![];
        let mut innermost_loop = vec![None; rpo.order.len()];
        let mut loop_parent = HashMap::new();
        for &(start, end) in &loops {
            while !loop_stack.is_empty() && start > loop_stack.last().unwrap().1 {
                loop_stack.pop();
            }
            if !loop_stack.is_empty() {
                let parent = loop_stack.last().unwrap();
                if end > parent.1 {
                    anyhow::bail!("Improperly nested loops");
                }
                loop_parent.insert(start, parent.0);
            }
            loop_stack.push((start, end));

            for i in start.index()..=end.index() {
                innermost_loop[i] = Some(start);
            }
        }

        // Determine blocks nested under each loop.
        let mut forward_targets = HashMap::new();
        for (rpo_block, block) in rpo.iter_with_index() {
            for succ in cfg.successors(block) {
                let rpo_succ = rpo.index_of_block(succ).unwrap();
                if rpo_succ > rpo_block {
                    // Check proper nesting.
                    let succ_header = innermost_loop[rpo_succ.index()];
                    if let Some(header) = succ_header {
                        if rpo_block < header {
                            anyhow::bail!("Improperly nested forward edge");
                        }
                    }
                    // Add the forward-edge record.
                    forward_targets
                        .entry(succ_header)
                        .or_insert_with(|| HashSet::new())
                        .insert(rpo_succ);
                }
            }
        }

        Ok(Stackifier {
            rpo,
            next_label: 0,
            loops,
            loop_ranges,
            innermost_loop,
            loop_parent,
            forward_targets,
            labels: HashMap::new(),
        })
    }

    pub fn compute<C: CFG<Block = B>>(cfg: &C) -> anyhow::Result<Node<B>> {
        // Compute auxiliary data structures.
        let mut this = Self::new(cfg)?;

        Ok(this
            .stackify_range(RPOIndex(0), RPOIndex((this.rpo.order.len() - 1) as u32))
            .into_iter()
            .next()
            .unwrap())
    }

    fn get_label(&mut self) -> Label {
        let index = self.next_label;
        self.next_label += 1;
        Label(index)
    }

    fn stackify_range(&mut self, first: RPOIndex, last: RPOIndex) -> Vec<Node<B>> {
        // Is this the start of a loop?
        if let Some(&loop_end) = self.loop_ranges.get(&first) {
            assert!(loop_end <= last);

            // Create labels for the loop and all forward targets.
            let label = self.get_label();
            self.labels.insert(first, label);
            let forward_targets = self
                .forward_targets
                .remove(&Some(first))
                .unwrap_or_default();
            let mut forward_targets: Vec<RPOIndex> = forward_targets.into_iter().collect();
            forward_targets.sort();
            for &target in &forward_targets {
                let label = self.get_label();
                self.labels.insert(target, label);
            }

            // Stackify the body, recursively.
            todo!()
        }
        // Otherwise, if this is the top level, do we have top-level
        // forward-edge targets?
        else if first.index() == 0 {
            todo!()
        }
        // Otherwise, just a sequence of individual nodes.
        else {
            todo!()
        }
    }
}
