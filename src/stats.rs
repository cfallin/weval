//! Post-specialization stats.

use crate::state::{Context, Contexts};
use waffle::{entity::PerEntity, Block, Func, FunctionBody};

#[derive(Clone, Debug, Default)]
pub struct SpecializationStats {
    pub generic: Func,
    pub generic_blocks: usize,
    pub generic_insts: usize,
    pub specializations: usize,
    pub specialized_blocks: usize,
    pub specialized_insts: usize,
    pub blocks_and_insts_by_bucket: std::collections::BTreeMap<Option<u32>, (usize, usize)>,
}

impl SpecializationStats {
    pub fn new(generic: Func, body: &FunctionBody) -> Self {
        let (generic_blocks, generic_insts) = count_reachable_blocks_and_insts(body, |_, _| ());
        Self { generic, generic_blocks, generic_insts, ..Default::default() }
    }

    pub fn add_specialization(
        &mut self,
        body: &FunctionBody,
        block_rev_map: &PerEntity<Block, (Context, Block)>,
        contexts: &Contexts,
    ) {
        self.specializations += 1;
        let (blocks, insts) = count_reachable_blocks_and_insts(body, |block, insts| {
            let (context, _) = block_rev_map[block];
            let bucket = contexts.context_bucket[context];
            let pair = self
                .blocks_and_insts_by_bucket
                .entry(bucket)
                .or_insert((0, 0));
            pair.0 += 1;
            pair.1 += insts;
        });
        self.specialized_blocks += blocks;
        self.specialized_insts += insts;
    }
}

fn count_reachable_blocks_and_insts<F: FnMut(Block, usize)>(
    body: &FunctionBody,
    mut visit: F,
) -> (usize, usize) {
    let mut queue = vec![body.entry];
    let mut visited = queue.iter().copied().collect::<fxhash::FxHashSet<_>>();
    let mut insts = 0;
    while let Some(block) = queue.pop() {
        let block_insts = body.blocks[block].insts.len();
        visit(block, block_insts);
        insts += block_insts;
        body.blocks[block].terminator.visit_successors(|succ| {
            if visited.insert(succ) {
                queue.push(succ);
            }
        });
    }

    (visited.len(), insts)
}
