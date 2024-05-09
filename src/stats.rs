//! Post-specialization stats.

use fxhash::FxHashSet;
use waffle::{Block, Func, FunctionBody};

/// Stats per original/generic function.
#[derive(Clone, Debug, Default)]
pub struct SpecializationStats {
    // --- stats computed once, for the generic function.
    pub generic: Func,
    pub generic_blocks: usize,
    pub generic_insts: usize,

    // --- stats accumulated over specializations:
    pub specializations: usize,
    pub specialized_blocks: usize,
    pub specialized_insts: usize,
    pub virtstack_reads: usize,
    pub virtstack_writes: usize,
    pub virtstack_reads_mem: usize,
    pub virtstack_writes_mem: usize,
    pub local_reads: usize,
    pub local_writes: usize,
    pub local_reads_mem: usize,
    pub local_writes_mem: usize,
    pub live_value_at_block_start: usize,
}

impl SpecializationStats {
    pub fn new(generic: Func, body: &FunctionBody) -> Self {
        let mut ret = Self::default();
        ret.generic = generic;
        let (blocks, insts, _) = count_reachable_blocks_and_insts(body);
        ret.generic_blocks = blocks;
        ret.generic_insts = insts;
        ret
    }

    pub fn add_specialization(&mut self, stats: &SpecializationStats) {
        self.specializations += 1;
        self.specialized_blocks += stats.specialized_blocks;
        self.specialized_insts += stats.specialized_insts;
        self.virtstack_reads += stats.virtstack_reads;
        self.virtstack_reads_mem += stats.virtstack_reads_mem;
        self.virtstack_writes += stats.virtstack_writes;
        self.virtstack_writes_mem += stats.virtstack_writes_mem;
        self.local_reads += stats.local_reads;
        self.local_reads_mem += stats.local_reads_mem;
        self.local_writes += stats.local_writes;
        self.local_writes_mem += stats.local_writes_mem;
        self.live_value_at_block_start += stats.live_value_at_block_start;
    }
}

pub fn count_reachable_blocks_and_insts(body: &FunctionBody) -> (usize, usize, FxHashSet<Block>) {
    let mut queue = vec![body.entry];
    let mut visited = queue.iter().cloned().collect::<FxHashSet<_>>();
    let mut insts = 0;
    while let Some(block) = queue.pop() {
        let block_insts = body.blocks[block].insts.len();
        insts += block_insts;
        body.blocks[block].terminator.visit_successors(|succ| {
            if visited.insert(succ) {
                queue.push(succ);
            }
        });
    }

    (visited.len(), insts, visited)
}
