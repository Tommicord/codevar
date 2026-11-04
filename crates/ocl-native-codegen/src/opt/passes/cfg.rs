//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Simplification of the control-flow graph.
//!
//! The pass makes the graph smaller in three steps:
//!
//! * **A branch on a known condition folds away.**  The conditional
//!   branch becomes an unconditional one, and the merge instruction
//!   that introduced the selection goes with it — the merge must stay
//!   immediately in front of the branch.  The branch of a loop header
//!   is left alone: the loop's structure depends on it.
//! * **A block nobody reaches is removed.**  Its instructions go with
//!   it, but the values they defined stay defined: they become copies
//!   of the undefined value in the entry block.  A merge that loses
//!   one of its incoming edges keeps the value of the edges that
//!   remain.
//! * **A block that falls through into its single successor is merged
//!   into it.**  The successor's instructions move into the block, so
//!   the pair becomes one.  The pass refuses the merge when either
//!   block holds a merge, when the successor has another predecessor,
//!   or when a value the block defines is used where the successor
//!   does not dominate the use.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{BasicBlock, BlockId, ConstValue, Inst, Module, Op, TypeId, ValueId};

use crate::error::Result;
use crate::opt::util::{
    Cfg, ConstSource, Resolver, block_id, collect_operands, definitions, dominates, dominators, merge_targets,
};

/// The merges of a block that lose an incoming edge: the position of
/// the merge, the value it defines, its type, and the value that
/// replaces it.
type Collapse = (usize, ValueId, TypeId, Option<ValueId>);

/// The collapses of every block.
type Collapses = BTreeMap<usize, Vec<Collapse>>;

/// Simplifies the control flow of every function.
pub(crate) fn run(module: &mut Module) -> Result<()> {
    let functions: Vec<ValueId> = module.functions().to_vec();
    for func in functions {
        simplify_function(module, func)?;
    }
    Ok(())
}

/// Simplifies the control flow of one function.
fn simplify_function(module: &mut Module, func: ValueId) -> Result<()> {
    fold_branches(module, func)?;
    remove_unreachable(module, func)?;
    merge_blocks(module, func)?;
    Ok(())
}

/// Folds the branches whose condition is known.
fn fold_branches(module: &mut Module, func: ValueId) -> Result<()> {
    let Some(body) = module
        .function(func)
        .and_then(|f| f.body.as_ref())
    else {
        return Ok(());
    };
    let defs = definitions(body);
    let resolver = Resolver::new(module, body, &defs);

    // The blocks whose branch folds away, and the edge that remains.
    let mut folds: Vec<(usize, bool)> = Vec::new();
    for (block_index, block) in body.iter().enumerate() {
        let Some(last) = block.insts.last() else {
            continue;
        };
        let Op::BranchConditional { cond, .. } = &last.op else {
            continue;
        };
        let Some(ConstValue::Bool(bit)) = resolver.payload(*cond) else {
            continue;
        };
        // The branch of a loop header keeps the loop's shape.
        if block
            .insts
            .iter()
            .any(|inst| matches!(inst.op, Op::LoopMerge { .. }))
        {
            continue;
        }
        folds.push((block_index, bit));
    }
    if folds.is_empty() {
        return Ok(());
    }

    let Some(function) = module.function_mut(func) else {
        return Ok(());
    };
    let Some(body) = function.body.as_mut() else {
        return Ok(());
    };
    for (block_index, bit) in folds {
        let Some(block) = body.get_mut(block_index) else {
            continue;
        };
        let Some(last) = block.insts.last_mut() else {
            continue;
        };
        let Op::BranchConditional { then, other, .. } = &last.op else {
            continue;
        };
        let target = if bit { *then } else { *other };
        last.op = Op::Branch { target };
        // The merge of the selection goes with the branch: it must
        // stay immediately in front of it.
        if block.insts.len() >= 2 {
            let position = block.insts.len() - 2;
            if matches!(block.insts[position].op, Op::SelectionMerge { .. }) {
                block.insts.remove(position);
            }
        }
    }
    Ok(())
}

/// Removes the blocks nobody reaches.
fn remove_unreachable(module: &mut Module, func: ValueId) -> Result<()> {
    let Some(body) = module
        .function(func)
        .and_then(|f| f.body.as_ref())
    else {
        return Ok(());
    };
    let cfg = Cfg::build(body);
    let dead: BTreeSet<usize> = (0..cfg.len())
        .filter(|index| {
            !cfg.reachable
                .get(*index)
                .copied()
                .unwrap_or(false)
        })
        .collect();
    if dead.is_empty() {
        return Ok(());
    }
    remove_dead_blocks(module, func, &dead)
}

/// Removes the blocks in `dead`.
///
/// The values the dead blocks defined stay defined: they become copies
/// of the undefined value in the entry block.  A merge that loses one
/// of its incoming edges keeps the value of the edges that remain.
pub(crate) fn remove_dead_blocks(module: &mut Module, func: ValueId, dead: &BTreeSet<usize>) -> Result<()> {
    let Some(body) = module
        .function(func)
        .and_then(|f| f.body.as_ref())
    else {
        return Ok(());
    };
    // The values the dead blocks define have to stay defined, and the
    // merges that lose an edge have to be rewritten.
    let mut kills: Vec<(ValueId, TypeId)> = Vec::new();
    let mut collapses: Collapses = BTreeMap::new();
    let mut needed: BTreeSet<TypeId> = BTreeSet::new();
    for (block_index, block) in body.iter().enumerate() {
        if dead.contains(&block_index) {
            for inst in &block.insts {
                if let (Some(result), Some(ty)) = (inst.result, inst.ty) {
                    kills.push((result, ty));
                    needed.insert(ty);
                }
            }
            continue;
        }
        for (inst_index, inst) in block.insts.iter().enumerate() {
            let Op::Phi { incomings } = &inst.op else {
                continue;
            };
            let (Some(result), Some(ty)) = (inst.result, inst.ty) else {
                continue;
            };
            let kept: Vec<(ValueId, BlockId)> = incomings
                .iter()
                .copied()
                .filter(|(_, predecessor)| !dead.contains(&predecessor.index()))
                .collect();
            if kept.len() >= 2 {
                continue;
            }
            let replacement = kept.first().map(|(value, _)| *value);
            needed.insert(ty);
            collapses
                .entry(block_index)
                .or_default()
                .push((inst_index, result, ty, replacement));
        }
    }

    // The undefined values the rewrites need.
    let mut undefs: BTreeMap<TypeId, ValueId> = BTreeMap::new();
    for ty in needed {
        let Ok(value) = module.intern_const(ty, ConstValue::Undef) else {
            continue;
        };
        undefs.insert(ty, value);
    }

    let Some(function) = module.function_mut(func) else {
        return Ok(());
    };
    let Some(body) = function.body.as_mut() else {
        return Ok(());
    };

    // The merges that lost an edge.
    for (block_index, list) in &collapses {
        let Some(block) = body.get_mut(*block_index) else {
            continue;
        };
        let old = core::mem::take(&mut block.insts);
        let mut phis: Vec<Inst> = Vec::new();
        let mut copies: Vec<Inst> = Vec::new();
        let mut rest: Vec<Inst> = Vec::new();
        let mut collapsing: BTreeMap<usize, (ValueId, TypeId, Option<ValueId>)> = BTreeMap::new();
        for (inst_index, result, ty, replacement) in list.iter().copied() {
            collapsing.insert(inst_index, (result, ty, replacement));
        }
        for (inst_index, inst) in old.into_iter().enumerate() {
            if let Some((result, ty, replacement)) = collapsing.get(&inst_index) {
                let operand = match replacement {
                    Some(value) => *value,
                    None => match undefs.get(ty) {
                        Some(value) => *value,
                        None => continue,
                    },
                };
                copies.push(Inst::def(*result, *ty, Op::CopyObject { operand }));
                continue;
            }
            if let Op::Phi { incomings } = &inst.op {
                let kept: Vec<(ValueId, BlockId)> = incomings
                    .iter()
                    .copied()
                    .filter(|(_, predecessor)| !dead.contains(&predecessor.index()))
                    .collect();
                if kept.len() >= 2 {
                    let (Some(result), Some(ty)) = (inst.result, inst.ty) else {
                        continue;
                    };
                    phis.push(Inst::def(result, ty, Op::Phi { incomings: kept }));
                    continue;
                }
            }
            rest.push(inst);
        }
        let mut fresh = phis;
        fresh.extend(copies);
        fresh.extend(rest);
        block.insts = fresh;
    }

    // The values the dead blocks defined become copies of the
    // undefined value in the entry block.
    if !kills.is_empty()
        && let Some(entry) = body.first_mut()
    {
        let mut fresh: Vec<Inst> = Vec::with_capacity(entry.insts.len() + kills.len());
        let terminator = entry.insts.len().saturating_sub(1);
        for (index, inst) in core::mem::take(&mut entry.insts)
            .into_iter()
            .enumerate()
        {
            if index == terminator {
                for (result, ty) in &kills {
                    let Some(undef) = undefs.get(ty) else {
                        continue;
                    };
                    fresh.push(Inst::def(*result, *ty, Op::CopyObject { operand: *undef }));
                }
            }
            fresh.push(inst);
        }
        entry.insts = fresh;
    }

    // The dead blocks go, and the handles that name them are
    // renumbered.
    remove_blocks(body, dead);
    Ok(())
}

/// Merges the blocks that fall through into their successor.
fn merge_blocks(module: &mut Module, func: ValueId) -> Result<()> {
    loop {
        let Some(body) = module
            .function(func)
            .and_then(|f| f.body.as_ref())
        else {
            return Ok(());
        };
        let cfg = Cfg::build(body);
        let dom = dominators(&cfg);

        // The blocks a merge instruction names: they keep their shape.
        let mut named: BTreeSet<usize> = BTreeSet::new();
        for block in body {
            for inst in &block.insts {
                let mut targets = Vec::new();
                merge_targets(&inst.op, &mut targets);
                named.extend(targets);
            }
        }

        let mut found = None;
        for block in 1..cfg.len() {
            let Some(preds) = cfg.preds.get(block) else {
                continue;
            };
            if preds.len() != 1 {
                continue;
            }
            let predecessor = preds[0];
            let Some(last) = body.get(block).and_then(|b| b.insts.last()) else {
                continue;
            };
            let Op::Branch { target } = last.op else {
                continue;
            };
            let successor = target.index();
            if successor >= cfg.len() || successor == block {
                continue;
            }
            if cfg
                .preds
                .get(successor)
                .map_or(0, |list| list.len())
                != 1
            {
                continue;
            }
            let Some(source) = body.get(block) else {
                continue;
            };
            let Some(target_block) = body.get(successor) else {
                continue;
            };
            // A block that holds a merge keeps its shape.
            if source
                .insts
                .iter()
                .any(|inst| matches!(inst.op, Op::SelectionMerge { .. } | Op::LoopMerge { .. }))
            {
                continue;
            }
            if holds_phi(source) || holds_phi(target_block) {
                continue;
            }
            if named.contains(&block) {
                continue;
            }
            if !values_stay_dominated(body, &dom, block, successor) {
                continue;
            }
            found = Some((block, successor, predecessor));
            break;
        }

        let Some((block, successor, predecessor)) = found else {
            return Ok(());
        };

        // The move: the block's instructions join the successor, and
        // the predecessor points at the successor directly.
        let Some(function) = module.function_mut(func) else {
            return Ok(());
        };
        let Some(body) = function.body.as_mut() else {
            return Ok(());
        };
        let Some(source) = body.get_mut(block) else {
            return Ok(());
        };
        let mut moved = core::mem::take(&mut source.insts);
        moved.pop();
        let Some(target_block) = body.get_mut(successor) else {
            return Ok(());
        };
        let existing = core::mem::take(&mut target_block.insts);
        let mut fresh = Vec::with_capacity(moved.len() + existing.len());
        fresh.extend(moved);
        fresh.extend(existing);
        target_block.insts = fresh;

        if let Some(predecessor_block) = body.get_mut(predecessor) {
            for inst in predecessor_block.insts.iter_mut() {
                match &mut inst.op {
                    Op::Branch { target } => {
                        if target.index() == block {
                            *target = block_id(successor);
                        }
                    }
                    Op::BranchConditional { then, other, .. } => {
                        if then.index() == block {
                            *then = block_id(successor);
                        }
                        if other.index() == block {
                            *other = block_id(successor);
                        }
                    }
                    _ => {}
                }
            }
        }

        // The merged block goes, and the handles are renumbered.
        let dead: BTreeSet<usize> = BTreeSet::from([block]);
        remove_blocks(body, &dead);
    }
}

/// True when the block holds a merge.
fn holds_phi(block: &BasicBlock) -> bool {
    block
        .insts
        .iter()
        .any(|inst| matches!(inst.op, Op::Phi { .. }))
}

/// True when every value the block defines is used where the successor
/// dominates the use.
fn values_stay_dominated(body: &[BasicBlock], dom: &[Vec<bool>], block: usize, successor: usize) -> bool {
    let mut defined: BTreeSet<ValueId> = BTreeSet::new();
    let Some(source) = body.get(block) else {
        return false;
    };
    for inst in &source.insts {
        if let Some(result) = inst.result {
            defined.insert(result);
        }
    }
    if defined.is_empty() {
        return true;
    }
    for (index, other) in body.iter().enumerate() {
        if index == block || index == successor {
            continue;
        }
        for inst in &other.insts {
            let mut operands = Vec::new();
            collect_operands(&inst.op, &mut operands);
            for operand in operands {
                if !defined.contains(&operand) {
                    continue;
                }
                if !dominates(dom, successor, index) {
                    return false;
                }
            }
        }
    }
    true
}

/// Removes the blocks in `dead` and renumbers the handles that name
/// them.
fn remove_blocks(body: &mut Vec<BasicBlock>, dead: &BTreeSet<usize>) {
    let len = body.len();
    let mut remap: Vec<usize> = Vec::with_capacity(len);
    let mut next = 0usize;
    for index in 0..len {
        if dead.contains(&index) {
            remap.push(usize::MAX);
        } else {
            remap.push(next);
            next += 1;
        }
    }
    let translate = |handle: BlockId| -> BlockId {
        match remap.get(handle.index()) {
            Some(&new) if new != usize::MAX => block_id(new),
            _ => handle,
        }
    };
    for block in body.iter_mut() {
        for inst in block.insts.iter_mut() {
            match &mut inst.op {
                Op::Branch { target } => *target = translate(*target),
                Op::BranchConditional { then, other, .. } => {
                    *then = translate(*then);
                    *other = translate(*other);
                }
                Op::SelectionMerge { target, .. } => *target = translate(*target),
                Op::LoopMerge { merge, cont, .. } => {
                    *merge = translate(*merge);
                    *cont = translate(*cont);
                }
                Op::Phi { incomings } => {
                    for (_, predecessor) in incomings.iter_mut() {
                        *predecessor = translate(*predecessor);
                    }
                }
                _ => {}
            }
        }
    }
    let mut index = 0usize;
    body.retain(|_| {
        let keep = !dead.contains(&index);
        index += 1;
        keep
    });
}
