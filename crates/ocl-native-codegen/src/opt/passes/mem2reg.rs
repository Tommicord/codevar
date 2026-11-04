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

//! Promotion of the function's stack slots into registers.
//!
//! The front end gives every local its own slot and reads and writes
//! it through pointers.  This pass turns those slots into values:
//!
//! 1. A slot is a candidate when its address never escapes — the only
//!    operations that mention the slot are loads from it and stores to
//!    it.  A slot whose address reaches a call, an access chain, or
//!    another pointer is left alone.
//! 2. The blocks that read or write the slot form the set of
//!    definitions.  The iterated dominance frontier of that set names
//!    the blocks where the value reaching them differs between the
//!    incoming edges; those blocks receive a merge.
//! 3. The dominator tree is walked once, carrying the value that
//!    reaches the current block.  Every load becomes a copy of the
//!    value that reaches it, and every store disappears.
//!
//! The pass refuses to promote a slot when the function's shape leaves
//! it undecided: a slot that an unreachable block mentions, a slot
//! whose definition is not in the entry block, or a slot whose merge
//! would not have one incoming per predecessor.  Such a slot keeps its
//! loads and stores, and the rest of the function is promoted as
//! usual.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{BasicBlock, ConstValue, Inst, Module, Op, TypeId, ValueId};

use crate::error::Result;
use crate::opt::util::{
    self, Cfg, block_id, collect_operands, dominator_children, dominators, replace_with_copy,
    strictly_dominates,
};

/// Promotes the stack slots of every function of `module`.
pub(crate) fn run(module: &mut Module) -> Result<()> {
    let functions: Vec<ValueId> = module.functions().to_vec();
    for func in functions {
        promote_function(module, func)?;
    }
    Ok(())
}

/// The value every load of a slot reads, by block and position.
type LoadValues = BTreeMap<(usize, usize), ValueId>;

/// The incoming values of every merge, by block.
type PhiIncomings = BTreeMap<usize, Vec<(ValueId, usize)>>;

/// The state one slot's promotion needs.
struct Promotion<'a> {
    /// The function the slot belongs to.
    func: ValueId,
    /// The slot.
    slot: &'a Slot,
    /// The control-flow graph of the function.
    cfg: &'a Cfg,
    /// The children of the dominator tree.
    children: &'a [Vec<usize>],
    /// The merge every block that needs one receives.
    phi_values: &'a BTreeMap<usize, ValueId>,
    /// The value the slot starts with.
    initial: ValueId,
}

/// One promotable stack slot.
struct Slot {
    /// The value of the slot pointer.
    value: ValueId,
    /// The type of the slot pointer.
    pointer: TypeId,
    /// The type the slot holds.
    pointee: TypeId,
    /// The value the slot starts with, when it has an initializer.
    init: Option<ValueId>,
}

/// What the analysis of a slot found.
struct Analysis {
    /// The blocks that need a merge.
    phi_blocks: BTreeSet<usize>,
    /// The loads of the slot, by block and position.
    loads: Vec<(usize, usize)>,
    /// The stores of the slot, by block and position.
    stores: Vec<(usize, usize)>,
}

/// Promotes the slots of one function.
fn promote_function(module: &mut Module, func: ValueId) -> Result<()> {
    let slots = {
        let Some(body) = module
            .function(func)
            .and_then(|f| f.body.as_ref())
        else {
            return Ok(());
        };
        collect_slots(module, body)
    };
    if slots.is_empty() {
        return Ok(());
    }
    let (cfg, dom, children) = {
        let Some(body) = module
            .function(func)
            .and_then(|f| f.body.as_ref())
        else {
            return Ok(());
        };
        let cfg = Cfg::build(body);
        let dom = dominators(&cfg);
        let children = dominator_children(&cfg, &dom);
        (cfg, dom, children)
    };
    for slot in slots {
        promote_slot(module, func, &slot, &cfg, &dom, &children)?;
    }
    Ok(())
}

/// Collects the slots of a function body.
fn collect_slots(module: &Module, body: &[BasicBlock]) -> Vec<Slot> {
    let mut slots = Vec::new();
    for block in body {
        for inst in &block.insts {
            let Op::Variable { init } = &inst.op else {
                continue;
            };
            let Some(result) = inst.result else {
                continue;
            };
            let Some(pointer) = inst.ty else {
                continue;
            };
            let Some((storage, pointee)) = module.pointer_parts(pointer) else {
                continue;
            };
            if storage != codevar_ocl_ir::ir::Storage::Function {
                continue;
            }
            slots.push(Slot {
                value: result,
                pointer,
                pointee,
                init: *init,
            });
        }
    }
    slots
}

/// Promotes one slot.
fn promote_slot(
    module: &mut Module,
    func: ValueId,
    slot: &Slot,
    cfg: &Cfg,
    dom: &[Vec<bool>],
    children: &[Vec<usize>],
) -> Result<()> {
    let Some(analysis) = analyze_slot(module, func, slot, cfg, dom) else {
        return Ok(());
    };
    if analysis.loads.is_empty() && analysis.stores.is_empty() {
        return Ok(());
    }

    // The values the promotion needs: one merge per block that needs
    // one, and the value the slot starts with.
    let undef = module
        .intern_const(slot.pointee, ConstValue::Undef)
        .map_err(|_| {
            util::failure(
                "promoting a stack slot",
                "the undefined value does not fit the slot",
            )
        })?;
    let undef_pointer = module
        .intern_const(slot.pointer, ConstValue::Undef)
        .map_err(|_| {
            util::failure(
                "promoting a stack slot",
                "the undefined pointer does not fit the slot",
            )
        })?;
    let initial = slot.init.unwrap_or(undef);
    let mut phi_values: BTreeMap<usize, ValueId> = BTreeMap::new();
    for block in &analysis.phi_blocks {
        let value = module.new_inst_value(slot.pointee);
        phi_values.insert(*block, value);
    }

    let promotion = Promotion {
        func,
        slot,
        cfg,
        children,
        phi_values: &phi_values,
        initial,
    };
    let Some((loads, incomings)) = rename_slot(module, &promotion, &analysis) else {
        return Ok(());
    };

    // Every merge must have one incoming per predecessor.
    for block in &analysis.phi_blocks {
        let count = incomings.get(block).map_or(0, |list| list.len());
        if count < 2 {
            return Ok(());
        }
    }

    apply_slot(module, &promotion, &analysis, &loads, &incomings, undef_pointer)?;
    Ok(())
}

/// Decides whether a slot can be promoted, and where the merges go.
fn analyze_slot(
    module: &Module,
    func: ValueId,
    slot: &Slot,
    cfg: &Cfg,
    dom: &[Vec<bool>],
) -> Option<Analysis> {
    let body = module.function(func)?.body.as_ref()?;
    let mut loads = Vec::new();
    let mut stores = Vec::new();
    let mut definition_block = None;
    for (block_index, block) in body.iter().enumerate() {
        for (inst_index, inst) in block.insts.iter().enumerate() {
            if let Some(result) = inst.result
                && result == slot.value
                && matches!(inst.op, Op::Variable { .. })
            {
                definition_block = Some(block_index);
            }
            let mut operands = Vec::new();
            collect_operands(&inst.op, &mut operands);
            if !operands.contains(&slot.value) {
                continue;
            }
            match &inst.op {
                Op::Load { ptr } if *ptr == slot.value => loads.push((block_index, inst_index)),
                Op::Store { ptr, .. } if *ptr == slot.value => stores.push((block_index, inst_index)),
                // The address of the slot reaches something else.
                _ => return None,
            }
        }
    }
    // The slot must start in the entry block: the value it starts with
    // is then available everywhere.
    if definition_block != Some(0) {
        return None;
    }
    // An unreachable block may mention the slot, and the pass cannot
    // say which value reaches it.
    for (block, _) in loads.iter().chain(stores.iter()) {
        if !cfg
            .reachable
            .get(*block)
            .copied()
            .unwrap_or(false)
        {
            return None;
        }
    }
    // A slot nobody reads keeps only its stores, and the pass removes
    // them: the slot is local, so the stores are dead.
    let def_blocks: BTreeSet<usize> = stores.iter().map(|(block, _)| *block).collect();
    let phi_blocks = iterated_frontier(cfg, dom, &def_blocks);
    Some(Analysis {
        phi_blocks,
        loads,
        stores,
    })
}

/// The iterated dominance frontier of the definition blocks.
fn iterated_frontier(cfg: &Cfg, dom: &[Vec<bool>], def_blocks: &BTreeSet<usize>) -> BTreeSet<usize> {
    let len = cfg.len();
    // The dominance frontier: the blocks a definition reaches that the
    // definition does not already cover.  A block belongs to the
    // frontier of an ancestor when the ancestor dominates one of its
    // predecessors but does not strictly dominate the block itself.
    let mut frontier = Vec::with_capacity(len);
    frontier.resize_with(len, Vec::new);
    for block in 0..len {
        if !cfg.reachable.get(block).copied().unwrap_or(false) {
            continue;
        }
        // The blocks that dominate one of the predecessors.
        let mut reaches = vec![false; len];
        for &pred in cfg
            .preds
            .get(block)
            .map_or(&[][..], |list| list.as_slice())
        {
            if !cfg.reachable.get(pred).copied().unwrap_or(false) {
                continue;
            }
            if let Some(row) = dom.get(pred) {
                for (flag, known) in reaches.iter_mut().zip(row.iter()) {
                    *flag |= *known;
                }
            }
        }
        for ancestor in 0..len {
            if !reaches.get(ancestor).copied().unwrap_or(false) {
                continue;
            }
            if strictly_dominates(dom, ancestor, block) {
                continue;
            }
            if let Some(list) = frontier.get_mut(ancestor) {
                list.push(block);
            }
        }
    }
    let mut phi_blocks = BTreeSet::new();
    let mut worklist: Vec<usize> = def_blocks.iter().copied().collect();
    while let Some(block) = worklist.pop() {
        for &next in frontier
            .get(block)
            .map_or(&[][..], |list| list.as_slice())
        {
            if phi_blocks.insert(next) {
                worklist.push(next);
            }
        }
    }
    phi_blocks
}

/// Walks the dominator tree, carrying the value that reaches the
/// current block.
fn rename_slot(
    module: &Module,
    promotion: &Promotion<'_>,
    analysis: &Analysis,
) -> Option<(LoadValues, PhiIncomings)> {
    let Promotion {
        func,
        slot,
        cfg,
        children,
        phi_values,
        initial,
    } = promotion;
    let body = module.function(*func)?.body.as_ref()?;
    let mut loads: LoadValues = BTreeMap::new();
    let mut incomings: PhiIncomings = BTreeMap::new();
    let mut stack: Vec<(usize, ValueId)> = Vec::new();
    stack.push((0usize, *initial));
    while let Some((block, inherited)) = stack.pop() {
        let mut current = inherited;
        if let Some(phi) = phi_values.get(&block) {
            current = *phi;
        }
        let block_def = body.get(block)?;
        for (index, inst) in block_def.insts.iter().enumerate() {
            match &inst.op {
                Op::Store { ptr, value } if *ptr == slot.value => current = *value,
                Op::Load { ptr } if *ptr == slot.value => {
                    loads.insert((block, index), current);
                }
                _ => {}
            }
        }
        for &successor in cfg
            .succs
            .get(block)
            .map_or(&[][..], |list| list.as_slice())
        {
            if phi_values.contains_key(&successor) {
                incomings
                    .entry(successor)
                    .or_default()
                    .push((current, block));
            }
        }
        for &child in children
            .get(block)
            .map_or(&[][..], |list| list.as_slice())
        {
            stack.push((child, current));
        }
    }
    // The analysis already found the loads; the walk must have found
    // them all.
    if loads.len() != analysis.loads.len() {
        return None;
    }
    Some((loads, incomings))
}

/// Writes the promotion into the function.
fn apply_slot(
    module: &mut Module,
    promotion: &Promotion<'_>,
    analysis: &Analysis,
    loads: &LoadValues,
    incomings: &PhiIncomings,
    undef_pointer: ValueId,
) -> Result<()> {
    let Promotion {
        func,
        slot,
        phi_values,
        ..
    } = promotion;
    let Some(function) = module.function_mut(*func) else {
        return Ok(());
    };
    let Some(body) = function.body.as_mut() else {
        return Ok(());
    };

    // The slot itself goes away; its value stays defined as a copy of
    // the undefined pointer.
    for block in body.iter_mut() {
        for inst in block.insts.iter_mut() {
            if inst.result == Some(slot.value) && matches!(inst.op, Op::Variable { .. }) {
                inst.op = Op::CopyObject {
                    operand: undef_pointer,
                };
            }
        }
    }

    // The blocks that change: those with a load, a store, or a merge.
    let mut changed: BTreeSet<usize> = BTreeSet::new();
    for (block, _) in &analysis.loads {
        changed.insert(*block);
    }
    for (block, _) in &analysis.stores {
        changed.insert(*block);
    }
    for block in analysis.phi_blocks.iter() {
        changed.insert(*block);
    }

    for block_index in changed {
        let Some(block) = body.get_mut(block_index) else {
            continue;
        };
        let old = core::mem::take(&mut block.insts);
        let mut fresh = Vec::with_capacity(old.len() + analysis.phi_blocks.len());
        // The merges come first: a block starts with them.
        if let Some(value) = phi_values.get(&block_index) {
            let mut pairs: Vec<(ValueId, usize)> = incomings
                .get(&block_index)
                .cloned()
                .unwrap_or_default();
            pairs.sort_by_key(|&(_, predecessor)| predecessor);
            let labelled: Vec<(ValueId, codevar_ocl_ir::ir::BlockId)> = pairs
                .into_iter()
                .map(|(value, predecessor)| (value, block_id(predecessor)))
                .collect();
            fresh.push(Inst::def(*value, slot.pointee, Op::Phi { incomings: labelled }));
        }
        for (index, mut inst) in old.into_iter().enumerate() {
            match &inst.op {
                Op::Store { ptr, .. } if *ptr == slot.value => continue,
                Op::Load { ptr } if *ptr == slot.value => {
                    let Some(value) = loads.get(&(block_index, index)) else {
                        continue;
                    };
                    replace_with_copy(&mut inst, *value);
                }
                _ => {}
            }
            fresh.push(inst);
        }
        block.insts = fresh;
    }
    Ok(())
}
