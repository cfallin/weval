//! Constant-offset "remat" pass: rewrite x+k to local additions off
//! of one base, to minimize live value / register pressure. Also push
//! these offsets into loads/stores where possible.

use fxhash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, VecDeque};
use waffle::{
    cfg::CFGInfo, entity::PerEntity, pool::ListRef, Block, FunctionBody, MemoryArg, Operator, Type,
    Value, ValueDef,
};

/// Dataflow analysis lattice: a value is either some original SSA
/// value plus an offset, or else an arbitrary runtime value.
///
/// Constants and additions deal only in 32-bit space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
enum AbsValue {
    /// Not yet computed.
    #[default]
    Top,
    /// A constant value alone.
    Constant(u32),
    /// A constant plus some other SSA value.
    Offset(Value, u32),
    /// Bottom value: value is "itself", no other description.
    Bottom,
}

impl AbsValue {
    fn meet(a: AbsValue, b: AbsValue) -> AbsValue {
        use AbsValue::*;
        match (a, b) {
            (a, b) if a == b => a,
            (x, Top) | (Top, x) => x,
            _ => Bottom,
        }
    }
}

fn is_load_or_store(op: &Operator) -> bool {
    match op {
        Operator::I32Load { .. }
        | Operator::I32Load8S { .. }
        | Operator::I32Load8U { .. }
        | Operator::I32Load16S { .. }
        | Operator::I32Load16U { .. }
        | Operator::I64Load { .. }
        | Operator::I64Load8S { .. }
        | Operator::I64Load8U { .. }
        | Operator::I64Load16S { .. }
        | Operator::I64Load16U { .. }
        | Operator::I64Load32S { .. }
        | Operator::I64Load32U { .. } => true,
        Operator::I32Store { .. }
        | Operator::I32Store8 { .. }
        | Operator::I32Store16 { .. }
        | Operator::I64Store { .. }
        | Operator::I64Store8 { .. }
        | Operator::I64Store16 { .. }
        | Operator::I64Store32 { .. } => true,
        _ => false,
    }
}

fn update_load_or_store_memarg<F: Fn(&mut MemoryArg)>(op: &mut Operator, f: F) {
    match op {
        Operator::I32Load { memory }
        | Operator::I32Load8S { memory }
        | Operator::I32Load8U { memory }
        | Operator::I32Load16S { memory }
        | Operator::I32Load16U { memory }
        | Operator::I64Load { memory }
        | Operator::I64Load8S { memory }
        | Operator::I64Load8U { memory }
        | Operator::I64Load16S { memory }
        | Operator::I64Load16U { memory }
        | Operator::I64Load32S { memory }
        | Operator::I64Load32U { memory }
        | Operator::I32Store { memory }
        | Operator::I32Store8 { memory }
        | Operator::I32Store16 { memory }
        | Operator::I64Store { memory }
        | Operator::I64Store8 { memory }
        | Operator::I64Store16 { memory }
        | Operator::I64Store32 { memory } => f(memory),
        _ => {}
    }
}

pub fn run(func: &mut FunctionBody, cfg: &CFGInfo) {
    waffle::passes::resolve_aliases::run(func);
    log::trace!(
        "constant_offsets pass running on:\n{}",
        func.display_verbose("| ", None)
    );
    // Compute a fixpoint analysis: which values are some original SSA
    // value plus an offset?
    let mut values: PerEntity<Value, AbsValue> = PerEntity::default();

    let mut workqueue: VecDeque<Block> = VecDeque::new();
    let mut workqueue_set: FxHashSet<Block> = FxHashSet::default();
    let mut visited: FxHashSet<Block> = FxHashSet::default();

    workqueue.push_back(func.entry);
    workqueue_set.insert(func.entry);

    for &(_, param) in &func.blocks[func.entry].params {
        values[param] = AbsValue::Bottom;
    }

    while let Some(block) = workqueue.pop_front() {
        log::trace!("processing {}", block);
        workqueue_set.remove(&block);

        for &inst in &func.blocks[block].insts {
            log::trace!("block {} value {}: {:?}", block, inst, func.values[inst]);
            match &func.values[inst] {
                ValueDef::BlockParam(..) => {
                    unreachable!();
                }

                ValueDef::Alias(orig) => {
                    values[inst] = values[*orig];
                }

                ValueDef::Operator(op, args, tys) if tys.len() == 1 => {
                    let args = &func.arg_pool[*args];
                    log::trace!(" -> args = {:?}", args);

                    match op {
                        Operator::I32Const { value } => {
                            values[inst] = AbsValue::Constant(*value);
                        }
                        Operator::I32Add => {
                            let x = args[0];
                            let y = args[1];
                            values[inst] = match (values[x], values[y]) {
                                (AbsValue::Top, _) | (_, AbsValue::Top) => AbsValue::Top,
                                (AbsValue::Constant(k1), AbsValue::Constant(k2)) => {
                                    AbsValue::Constant(k1.wrapping_add(k2))
                                }
                                (AbsValue::Offset(base, k1), AbsValue::Constant(k2))
                                | (AbsValue::Constant(k2), AbsValue::Offset(base, k1)) => {
                                    AbsValue::Offset(base, k1.wrapping_add(k2))
                                }
                                (_, AbsValue::Constant(k)) => AbsValue::Offset(x, k),
                                (AbsValue::Constant(k), _) => AbsValue::Offset(y, k),
                                _ => AbsValue::Bottom,
                            };
                        }
                        Operator::I32Sub => {
                            // Like the addition case, but no commutativity.
                            let x = args[0];
                            let y = args[1];
                            values[inst] = match (values[x], values[y]) {
                                (AbsValue::Top, _) | (_, AbsValue::Top) => AbsValue::Top,
                                (AbsValue::Constant(k1), AbsValue::Constant(k2)) => {
                                    AbsValue::Constant(k1.wrapping_sub(k2))
                                }
                                (AbsValue::Offset(base, k1), AbsValue::Constant(k2)) => {
                                    AbsValue::Offset(base, k1.wrapping_sub(k2))
                                }
                                (_, AbsValue::Constant(k)) => {
                                    AbsValue::Offset(x, 0u32.wrapping_sub(k))
                                }
                                (AbsValue::Offset(base1, k1), AbsValue::Offset(base2, k2))
                                    if base1 == base2 =>
                                {
                                    AbsValue::Constant(k1.wrapping_sub(k2))
                                }
                                (_, AbsValue::Offset(base, k)) if base == x => {
                                    AbsValue::Constant(0u32.wrapping_sub(k))
                                }
                                _ => AbsValue::Bottom,
                            };
                        }
                        _ => {
                            values[inst] = AbsValue::Bottom;
                        }
                    }
                }

                _ => {
                    values[inst] = AbsValue::Bottom;
                }
            }
            log::trace!(" -> values[{}] = {:?}", inst, values[inst]);
        }

        func.blocks[block].terminator.visit_targets(|target| {
            let mut changed = false;
            let succ_params = &func.blocks[target.block].params;
            for (&arg, &(_, blockparam)) in target.args.iter().zip(succ_params.iter()) {
                let new = AbsValue::meet(values[arg], values[blockparam]);
                log::trace!(" -> block {} target {}: arg {} to blockparam {}: value {:?} -> {:?}",
                            block, target.block, arg, blockparam, values[blockparam], new);
                changed |= new != values[blockparam];
                values[blockparam] = new;
            }

            if changed ||
                visited.insert(target.block) ||
                (block != target.block && cfg.dominates(block, target.block)) {
                log::trace!(" -> at least one blockparam changed, or we dominate target block; enqueuing {}",
                            target.block);
                if workqueue_set.insert(target.block) {
                    workqueue.push_back(target.block);
                }
            }
        });
    }

    // Find the set of all values used as addresses to loads/stores.
    let mut used_as_addr = FxHashSet::default();
    for (_, def) in func.values.entries() {
        if let ValueDef::Operator(op, args, _) = def {
            if is_load_or_store(op) {
                used_as_addr.insert(func.arg_pool[*args][0]);
            }
        }
    }

    // For any value that's a base, compute the minimum offset (when
    // interpreted as an i32).
    let mut min_offset_from: BTreeMap<Value, i32> = BTreeMap::new();
    for value in func.values.iter() {
        if used_as_addr.contains(&value) {
            if let AbsValue::Offset(base, off) = values[value] {
                let signed = off as i32;
                let min = min_offset_from.entry(base).or_insert(0);
                *min = std::cmp::min(*min, signed);
            }
        }
    }

    // Create an i32add or i32sub computing one shared base (with only
    // positive offsets needed) for each address-related offset-from
    // value in `min_offset_from`. For a value `x`, with `y =
    // offset_base[x]`, we have `y + min_offset_from[x] == x`.
    //
    // Note that we don't insert these values into any blocks yet; we
    // do that when we see the original defs below (i.e., insert `y`
    // just after `x` is defined). `offset_base_const` facilitates
    // this (we need to insert the i32const first).
    let mut offset_base: BTreeMap<Value, Value> = BTreeMap::new();
    let mut offset_base_const: BTreeMap<Value, Value> = BTreeMap::new();
    let i32_ty = func.single_type_list(Type::I32);
    for (&value, &offset) in &min_offset_from {
        assert!(offset <= 0);
        if offset != 0 {
            let k = func.add_value(ValueDef::Operator(
                Operator::I32Const {
                    value: (-offset) as u32,
                },
                ListRef::default(),
                i32_ty,
            ));
            let args = func.arg_pool.double(value, k);
            let add = func.add_value(ValueDef::Operator(Operator::I32Sub, args, i32_ty));
            offset_base_const.insert(value, k);
            offset_base.insert(value, add);
            log::trace!(
                "created common base {} (and const {}) associated with offset-from value {}",
                k,
                add,
                value
            );
        } else {
            offset_base.insert(value, value);
        }
    }

    // Now, for each value that's an Offset, rewrite it to an add
    // instruction.
    for (block, block_def) in func.blocks.entries_mut() {
        log::trace!("rewriting in block {}", block);
        let mut computed_offsets: FxHashMap<AbsValue, Value> = FxHashMap::default();
        let mut new_insts = vec![];

        for (_, param) in &block_def.params {
            // Insert the common-base computations where needed.
            if let Some(common_base_const) = offset_base_const.get(&param) {
                new_insts.push(*common_base_const);
                new_insts.push(*offset_base.get(&param).unwrap());
            }
        }

        for inst in std::mem::take(&mut block_def.insts) {
            log::trace!("visiting inst {}: {:?}", inst, values[inst]);

            // Handle loads/stores.
            if let ValueDef::Operator(op, args, tys) = &func.values[inst] {
                if is_load_or_store(op) {
                    let args = &func.arg_pool[*args];
                    let tys = *tys;
                    let addr = args[0];
                    log::trace!("load/store with addr {}", addr);
                    if let AbsValue::Offset(base, this_offset) = values[addr] {
                        log::trace!("inst {} is a load/store with addr that is offset from base {}; pushing offset into instruction", inst, base);
                        // Update the offset embedded in the Operator
                        // and use the `base` value instead as the
                        // address arg.
                        let mut op = op.clone();
                        let mut args = args.iter().cloned().collect::<Vec<_>>();
                        let common_base = *offset_base.get(&base).unwrap();
                        let offset = *min_offset_from.get(&base).unwrap();
                        assert!(offset <= 0);
                        let addend = (-offset) as u32;
                        update_load_or_store_memarg(&mut op, |memory| {
                            memory.offset =
                                memory.offset.wrapping_add(addend).wrapping_add(this_offset)
                        });
                        args[0] = common_base;
                        let args = func.arg_pool.from_iter(args.into_iter());
                        func.values[inst] = ValueDef::Operator(op, args, tys);
                    }
                }
            }

            // If this value is a constant according to analysis above
            // (perhaps because it is the difference between x+k1 and
            // x+k2) but not an I32Const then make it so.
            if let AbsValue::Constant(k) = values[inst] {
                if !matches!(
                    func.values[inst],
                    ValueDef::Operator(Operator::I32Const { .. }, _, _)
                ) {
                    func.values[inst] = ValueDef::Operator(
                        Operator::I32Const { value: k },
                        ListRef::default(),
                        i32_ty,
                    );
                }
            }

            // Recompute this particular value if appropriate.
            if let AbsValue::Offset(base, offset) = values[inst] {
                let computed_offset = *computed_offsets.entry(values[inst]).or_insert_with(|| {
                    if offset == 0 {
                        base
                    } else {
                        let offset = offset as i32;
                        let (op, value) = if offset > 0 {
                            (Operator::I32Add, offset as u32)
                        } else {
                            (Operator::I32Sub, (-offset) as u32)
                        };
                        let k = func.values.push(ValueDef::Operator(
                            Operator::I32Const { value },
                            ListRef::default(),
                            i32_ty,
                        ));
                        new_insts.push(k);
                        let args = func.arg_pool.double(base, k);
                        let add = func.values.push(ValueDef::Operator(op, args, i32_ty));
                        func.source_locs[k] = func.source_locs[inst];
                        func.source_locs[add] = func.source_locs[inst];
                        log::trace!(" -> recomputed as {}", add);
                        new_insts.push(add);
                        add
                    }
                });
                log::trace!(" -> rewrite to {}", computed_offset);
                func.values[inst] = ValueDef::Alias(computed_offset);
            } else {
                new_insts.push(inst);
            }

            // Insert the common-base computations where needed.
            if let Some(common_base_const) = offset_base_const.get(&inst) {
                new_insts.push(*common_base_const);
                new_insts.push(*offset_base.get(&inst).unwrap());
            }
        }
        block_def.insts = new_insts;
    }
}
