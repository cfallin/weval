//! Constant-offset "remat" pass: rewrite x+k to local additions off
//! of one base, to minimize live value / register pressure.

use fxhash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::VecDeque;
use waffle::{
    cfg::CFGInfo, entity::PerEntity, pool::ListRef, Block, FunctionBody, Operator, Type, Value,
    ValueDef,
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

pub fn run(func: &mut FunctionBody, cfg: &CFGInfo) {
    // Compute a fixpoint analysis: which values are some original SSA
    // value plus an offset?
    let mut values: PerEntity<Value, AbsValue> = PerEntity::default();

    let mut workqueue: VecDeque<Block> = VecDeque::new();
    let mut workqueue_set: FxHashSet<Block> = FxHashSet::default();

    workqueue.push_back(func.entry);
    workqueue_set.insert(func.entry);

    while let Some(block) = workqueue.pop_front() {
        workqueue_set.remove(&block);

        for &inst in &func.blocks[block].insts {
            match &func.values[inst] {
                ValueDef::BlockParam(..) => {
                    unreachable!();
                }

                ValueDef::Alias(orig) => {
                    values[inst] = values[*orig];
                }

                ValueDef::Operator(op, args, tys) if tys.len() == 1 => {
                    let args = &func.arg_pool[*args];

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
        }

        func.blocks[block].terminator.visit_targets(|target| {
            let mut changed = false;
            let succ_params = &func.blocks[target.block].params;
            for (&arg, &(_, blockparam)) in target.args.iter().zip(succ_params.iter()) {
                let arg = func.resolve_alias(arg);
                let new = AbsValue::meet(values[arg], values[blockparam]);
                changed |= new != values[blockparam];
                values[blockparam] = new;
            }

            if changed || cfg.dominates(block, target.block) {
                if workqueue_set.insert(block) {
                    workqueue.push_back(block);
                }
            }
        });
    }

    // Now, for each value that's an Offset, rewrite it to an add
    // instruction. (We don't bother removing the original
    // instructions: that will happen with a later DCE pass.)
    let i32_ty = func.single_type_list(Type::I32);
    for block_def in func.blocks.values_mut() {
        let mut computed_offsets: FxHashMap<AbsValue, Value> = FxHashMap::default();
        let mut rewrite: FxHashMap<Value, Value> = FxHashMap::default();
        let mut new_insts = vec![];
        for inst in std::mem::take(&mut block_def.insts) {
            if let AbsValue::Offset(base, offset) = values[inst] {
                let computed_offset = *computed_offsets.entry(values[inst]).or_insert_with(|| {
                    let k = func.values.push(ValueDef::Operator(
                        Operator::I32Const { value: offset },
                        ListRef::default(),
                        i32_ty,
                    ));
                    new_insts.push(k);
                    let args = func.arg_pool.double(base, k);
                    let add = func
                        .values
                        .push(ValueDef::Operator(Operator::I32Add, args, i32_ty));
                    new_insts.push(add);
                    add
                });
                rewrite.insert(inst, computed_offset);
            } else {
                let value = match &func.values[inst] {
                    ValueDef::Operator(op, args, tys)
                        if func.arg_pool[*args]
                            .iter()
                            .any(|&arg| rewrite.contains_key(&arg)) =>
                    {
                        let new_args = func.arg_pool[*args]
                            .iter()
                            .map(|&arg| rewrite.get(&arg).cloned().unwrap_or(arg))
                            .collect::<SmallVec<[Value; 4]>>();
                        let args = func.arg_pool.from_iter(new_args.into_iter());
                        func.values.push(ValueDef::Operator(op.clone(), args, *tys))
                    }
                    _ => inst,
                };
                new_insts.push(value);
            }
        }
        block_def.insts = new_insts;
    }
}
