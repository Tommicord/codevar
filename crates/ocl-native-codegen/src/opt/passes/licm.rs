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

//! Movement of the loop-invariant computations out of their loop.
//!
//! A loop is found through its merge instruction: the block that
//! carries the merge is the header, and the block the merge names as
//! the continue target ends the iteration.  The blocks of the loop are
//! the ones that can reach the continue block without passing through
//! the header.
//!
//! A computation is invariant when it has no effect of its own and
//! every operand it reads is either defined outside the loop or is
//! itself invariant.  The operands must also reach the block that
//! enters the loop, so that the computation keeps its meaning when it
//! moves there.
//!
//! The pass moves the invariant computations into the block that
//! enters the loop, in the order the loop runs them, and leaves the
//! rest of the loop alone.  It refuses to move anything when the loop
//! has no single entering block, or when the block that enters it does
//! not branch into the header directly.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{Inst, Module, Op, ValueId};

use crate::error::Result;
use crate::opt::util::{
    Cfg, collect_operands, definitions, dominates, dominator_children, dominator_preorder, dominators,
    is_movable,
};

/// Moves the invariant computations of every function.
pub(crate) fn run(module: &mut Module) -> Result<()> {
    let functions: Vec<ValueId> = module.functions().to_vec();
    for func in functions {
        hoist_function(module, func)?;
    }
    Ok(())
}

/// Moves the invariant computations of one function.
fn hoist_function(module: &mut Module, func: ValueId) -> Result<()> {
    let Some(body) = module
        .function(func)
        .and_then(|f| f.body.as_ref())
    else {
        return Ok(());
    };
    // The loops of the function.
    let mut loops: Vec<(usize, usize)> = Vec::new();
    for (index, block) in body.iter().enumerate() {
        for inst in &block.insts {
            if let Op::LoopMerge { cont, .. } = &inst.op {
                loops.push((index, cont.index()));
            }
        }
    }
    for (header, cont) in loops {
        hoist_loop(module, func, header, cont)?;
    }
    Ok(())
}

/// Moves the invariant computations of one loop.
fn hoist_loop(module: &mut Module, func: ValueId, header: usize, cont: usize) -> Result<()> {
    let Some(body) = module
        .function(func)
        .and_then(|f| f.body.as_ref())
    else {
        return Ok(());
    };
    let cfg = Cfg::build(body);
    let dom = dominators(&cfg);
    let children = dominator_children(&cfg, &dom);
    let order = dominator_preorder(&cfg, &children);
    let defs = definitions(body);

    // The blocks that read every value.
    let mut readers: BTreeMap<ValueId, Vec<usize>> = BTreeMap::new();
    for (block_index, block) in body.iter().enumerate() {
        for inst in &block.insts {
            let mut operands = Vec::new();
            collect_operands(&inst.op, &mut operands);
            for operand in operands {
                readers
                    .entry(operand)
                    .or_default()
                    .push(block_index);
            }
        }
    }

    // The blocks of the loop.
    let loop_blocks = natural_loop(&cfg, header, cont);

    // The block that enters the loop: the only predecessor of the
    // header that is not part of the loop and branches into it.
    let mut preheader = None;
    for &pred in cfg
        .preds
        .get(header)
        .map_or(&[][..], |list| list.as_slice())
    {
        if loop_blocks.contains(&pred) {
            continue;
        }
        let Some(last) = body
            .get(pred)
            .and_then(|block| block.insts.last())
        else {
            continue;
        };
        let Op::Branch { target } = last.op else {
            continue;
        };
        if target.index() != header {
            continue;
        }
        if preheader.is_some() {
            // The loop has more than one entering block.
            return Ok(());
        }
        preheader = Some(pred);
    }
    let Some(preheader) = preheader else {
        return Ok(());
    };

    // The invariant computations.
    let mut invariant_values: BTreeSet<ValueId> = BTreeSet::new();
    let mut invariant_instructions: BTreeSet<(usize, usize)> = BTreeSet::new();
    loop {
        let mut changed = false;
        for (block_index, block) in body.iter().enumerate() {
            if !loop_blocks.contains(&block_index) {
                continue;
            }
            for (inst_index, inst) in block.insts.iter().enumerate() {
                if invariant_instructions.contains(&(block_index, inst_index)) {
                    continue;
                }
                let Some(result) = inst.result else {
                    continue;
                };
                if invariant_values.contains(&result) {
                    continue;
                }
                if !is_movable(&inst.op) {
                    continue;
                }
                // The value must not be read outside the loop: the
                // block that enters the loop does not necessarily
                // dominate every reader.
                if readers
                    .get(&result)
                    .is_some_and(|blocks: &Vec<usize>| {
                        blocks
                            .iter()
                            .any(|block| !loop_blocks.contains(block))
                    })
                {
                    continue;
                }
                let mut operands = Vec::new();
                collect_operands(&inst.op, &mut operands);
                let usable = operands.iter().all(|operand| {
                    if invariant_values.contains(operand) {
                        return true;
                    }
                    match defs.get(operand) {
                        // A constant, an argument, or a global: always
                        // available.
                        None => true,
                        Some(&(def_block, _)) => {
                            !loop_blocks.contains(&def_block)
                                && (def_block == preheader || dominates(&dom, def_block, preheader))
                        }
                    }
                });
                if !usable {
                    continue;
                }
                invariant_instructions.insert((block_index, inst_index));
                invariant_values.insert(result);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if invariant_instructions.is_empty() {
        return Ok(());
    }

    // The order the loop runs the computations in: a block that
    // dominates another comes first.
    let mut positions: Vec<usize> = vec![usize::MAX; order.len()];
    for (position, block) in order.iter().enumerate() {
        if let Some(slot) = positions.get_mut(*block) {
            *slot = position;
        }
    }
    let mut ordered: Vec<(usize, usize)> = invariant_instructions.iter().copied().collect();
    ordered.sort_by_key(|(block, index)| {
        (
            positions
                .get(*block)
                .copied()
                .unwrap_or(usize::MAX),
            *index,
        )
    });

    // The move: the computations leave their block and join the block
    // that enters the loop.
    let Some(function) = module.function_mut(func) else {
        return Ok(());
    };
    let Some(body) = function.body.as_mut() else {
        return Ok(());
    };
    let mut moved: Vec<(usize, usize, Inst)> = Vec::with_capacity(ordered.len());
    for (block_index, block) in body.iter_mut().enumerate() {
        if !loop_blocks.contains(&block_index) {
            continue;
        }
        let old = core::mem::take(&mut block.insts);
        let mut kept: Vec<Inst> = Vec::with_capacity(old.len());
        for (inst_index, inst) in old.into_iter().enumerate() {
            if invariant_instructions.contains(&(block_index, inst_index)) {
                moved.push((
                    positions
                        .get(block_index)
                        .copied()
                        .unwrap_or(usize::MAX),
                    inst_index,
                    inst,
                ));
                continue;
            }
            kept.push(inst);
        }
        block.insts = kept;
    }
    if moved.is_empty() {
        return Ok(());
    }
    moved.sort_by_key(|(block, index, _)| (*block, *index));
    let moved: Vec<Inst> = moved
        .into_iter()
        .map(|(_, _, inst)| inst)
        .collect();
    let Some(preheader_block) = body.get_mut(preheader) else {
        return Ok(());
    };
    let mut old = core::mem::take(&mut preheader_block.insts);
    let terminator = old.pop();
    // The block starts with its merges, and a merge must stay at the
    // front.  The hoisted instructions join the block behind the
    // instructions it already holds: an operand of a hoisted
    // instruction may be defined anywhere in this block, and only its
    // end is behind all of them.
    let merges = old
        .iter()
        .position(|inst| !matches!(inst.op, Op::Phi { .. }))
        .unwrap_or(old.len());
    let mut fresh: Vec<Inst> = Vec::with_capacity(old.len() + moved.len() + 1);
    fresh.extend_from_slice(&old[..merges]);
    fresh.extend_from_slice(&old[merges..]);
    fresh.extend(moved);
    if let Some(terminator) = terminator {
        fresh.push(terminator);
    }
    preheader_block.insts = fresh;
    Ok(())
}

/// The blocks of the loop the back edge closes.
fn natural_loop(cfg: &Cfg, header: usize, cont: usize) -> BTreeSet<usize> {
    let mut blocks: BTreeSet<usize> = BTreeSet::new();
    blocks.insert(header);
    let mut stack: Vec<usize> = vec![cont];
    while let Some(block) = stack.pop() {
        if !blocks.insert(block) {
            continue;
        }
        let Some(preds) = cfg.preds.get(block) else {
            continue;
        };
        for &pred in preds {
            if pred != header && !blocks.contains(&pred) {
                stack.push(pred);
            }
        }
    }
    blocks
}
