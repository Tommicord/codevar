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

//! Unrolling of the small counted loops.
//!
//! The pass understands one family of loops: a header that holds the
//! loop's merge and the induction variable, a condition that compares
//! the induction variable with a known bound, and a body that grows the
//! induction variable by a known step.
//!
//! ```text
//! header:  %i = OpPhi %int %start %pre, %next %back
//!          %cmp = OpSLessThan %bool %i %bound
//!          OpLoopMerge %merge %back
//!          OpBranchConditional %cmp %body %merge
//! body:    %next = OpIAdd %i %step
//!          ...
//!          OpBranch %back
//! back:    OpBranch %header
//! ```
//!
//! The header may also branch to a block that holds the condition, and
//! the body may branch to the block the back edge comes from; the
//! pass writes out the body either way.
//!
//! The induction variable starts at a known value, grows by a known
//! step, and is compared with a known bound, so the number of
//! iterations is known.  When that number is at most the limit, the
//! pass writes the iterations out one after another and removes the
//! loop; when it is larger, the pass writes out the first iterations
//! and leaves the loop to run the rest.
//!
//! The pass refuses to unroll a loop whose shape it does not
//! recognise, whose body holds a merge, whose iteration count is
//! unknown, and — for the complete unrolling — one whose values are
//! read after the loop, because the value the loop leaves behind would
//! change.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{BasicBlock, BinOp, BlockId, CmpOp, ConstValue, Inst, Module, Op, TypeId, ValueId};

use crate::error::Result;
use crate::opt::passes::cfg::remove_dead_blocks;
use crate::opt::util::{Cfg, block_id, collect_operands, definitions, int_shape, mask, sign_extend};

/// A loop the pass can unroll.
struct Counted {
    /// The block that holds the loop's merge.
    header: usize,
    /// The block that holds the condition.
    cond_block: usize,
    /// The block the loop runs.
    body: usize,
    /// The block the back edge comes from.
    cont: usize,
    /// The block the loop leaves to.
    merge: usize,
    /// The block that enters the loop.
    preheader: usize,
    /// The induction variable.
    iv: ValueId,
    /// The type of the induction variable.
    iv_type: TypeId,
    /// The width of the induction variable.
    bits: u8,
    /// The value the induction variable starts at.
    start: i128,
    /// The step the induction variable grows by.
    step: i128,
    /// The number of iterations the loop runs.
    trip: u64,
}

/// Duplicates the body of the small counted loops.
pub(crate) fn run(module: &mut Module, limit: u32) -> Result<()> {
    if limit == 0 {
        return Ok(());
    }
    let functions: Vec<ValueId> = module.functions().to_vec();
    for func in functions {
        unroll_function(module, func, limit)?;
    }
    Ok(())
}

/// Unrolls the counted loops of one function.
///
/// Every loop is unrolled once: the pass writes out the first
/// iterations and leaves the rest of them to the loop, so the size of
/// the code stays bounded by the limit.
fn unroll_function(module: &mut Module, func: ValueId, limit: u32) -> Result<()> {
    let mut done: BTreeSet<String> = BTreeSet::new();
    loop {
        let Some(body) = module
            .function(func)
            .and_then(|f| f.body.as_ref())
        else {
            return Ok(());
        };
        let cfg = Cfg::build(body);
        let defs = definitions(body);

        // The first loop the pass understands that it has not unrolled
        // yet.
        let mut candidate = None;
        for header in 0..body.len() {
            let Some(counted) = detect(module, body, &cfg, &defs, header) else {
                continue;
            };
            let Some(name) = body.get(header).map(|block| block.name.clone()) else {
                continue;
            };
            if done.contains(&name) {
                continue;
            }
            candidate = Some((counted, name));
            break;
        }
        let Some((counted, name)) = candidate else {
            return Ok(());
        };
        done.insert(name);
        if !unroll(module, func, &counted, limit)? {
            return Ok(());
        }
    }
}

/// Recognizes a counted loop.
fn detect(
    module: &Module,
    body: &[BasicBlock],
    cfg: &Cfg,
    defs: &BTreeMap<ValueId, (usize, usize)>,
    header: usize,
) -> Option<Counted> {
    let header_block = body.get(header)?;
    // The header holds the merge of the loop.
    let mut merge = None;
    let mut cont = None;
    for inst in &header_block.insts {
        if let Op::LoopMerge {
            merge: exit,
            cont: continue_block,
            ..
        } = &inst.op
        {
            merge = Some(exit.index());
            cont = Some(continue_block.index());
        }
    }
    let merge = merge?;
    let cont = cont?;

    // The block that holds the condition: either the header itself or
    // the block the header branches to.
    let condition_block = match header_block.insts.last().map(|inst| &inst.op) {
        Some(Op::BranchConditional { .. }) => header,
        Some(Op::Branch { target }) => target.index(),
        _ => return None,
    };
    let condition_def = body.get(condition_block)?;
    let mut condition = None;
    let mut then = None;
    let mut other = None;
    for inst in &condition_def.insts {
        if let Op::BranchConditional {
            cond,
            then: yes,
            other: no,
        } = &inst.op
        {
            condition = Some(*cond);
            then = Some(yes.index());
            other = Some(no.index());
        }
    }
    let condition = condition?;
    let then = then?;
    let other = other?;
    // The shape the pass understands: the false edge leaves the loop.
    if other != merge {
        return None;
    }
    // The body either is the continue block or branches to it.
    let loop_body = body.get(then)?;
    let body_last = loop_body.insts.last()?;
    let Op::Branch { target } = body_last.op else {
        return None;
    };
    if then == cont {
        // The body is the continue block: it branches back to the
        // header.
        if target.index() != header {
            return None;
        }
    } else {
        // The body branches to the continue block, which must hold
        // nothing but the branch back to the header: the copies skip
        // it.
        if target.index() != cont {
            return None;
        }
        let continue_block = body.get(cont)?;
        if continue_block.insts.len() != 1 {
            return None;
        }
        let Op::Branch { target } = continue_block.insts[0].op else {
            return None;
        };
        if target.index() != header {
            return None;
        }
    }
    // The body holds no merge of its own.
    if loop_body
        .insts
        .iter()
        .any(|inst| matches!(inst.op, Op::Phi { .. }))
    {
        return None;
    }
    // The header holds exactly one merge: the induction variable.
    let mut phis = loop_body_phis(header_block);
    if phis.len() != 1 {
        return None;
    }
    let phi = phis.remove(0);
    let (Some(iv), Some(iv_type)) = (phi.result, phi.ty) else {
        return None;
    };
    let Op::Phi { incomings } = &phi.op else {
        return None;
    };
    if incomings.len() != 2 {
        return None;
    }
    // The block that enters the loop.
    let preds = cfg.preds.get(header)?;
    // The header has two predecessors: the block that enters the loop
    // and the block the back edge comes from.
    let preheader = preds.iter().copied().find(|pred| *pred != cont)?;
    let last = body
        .get(preheader)
        .and_then(|block| block.insts.last())?;
    let Op::Branch { target } = last.op else {
        return None;
    };
    if target.index() != header {
        return None;
    }
    // The merge reads the value the loop starts with from the block
    // that enters it, and the value the loop produces from the back
    // edge.
    let start_entry = incomings
        .iter()
        .find(|(_, predecessor)| predecessor.index() == preheader)?;
    let next_entry = incomings
        .iter()
        .find(|(_, predecessor)| predecessor.index() != preheader)?;
    let (start_value, start_block) = (start_entry.0, start_entry.1.index());
    if start_block != preheader {
        return None;
    }
    let (start, signed) = int_value(module, start_value)?;
    let (bits, _) = int_shape(module, iv_type)?;
    if !signed {
        return None;
    }

    // The step: the induction variable grows by a known amount.
    let next = next_entry.0;
    let &(next_block, next_index) = defs.get(&next)?;
    if next_block != then {
        return None;
    }
    let next_inst = loop_body.insts.get(next_index)?;
    let Op::Binary {
        op: BinOp::IAdd,
        lhs,
        rhs,
    } = &next_inst.op
    else {
        return None;
    };
    let step_value = if resolve(defs, body, *lhs) == iv {
        *rhs
    } else if resolve(defs, body, *rhs) == iv {
        *lhs
    } else {
        return None;
    };
    let (step, _) = int_value(module, step_value)?;
    if step <= 0 {
        return None;
    }

    // The bound: the induction variable is compared with a constant.
    let &(cond_block, cond_index) = defs.get(&condition)?;
    let cond_inst = body.get(cond_block)?.insts.get(cond_index)?;
    let Op::Compare {
        op: comparison,
        lhs,
        rhs,
    } = &cond_inst.op
    else {
        return None;
    };
    if resolve(defs, body, *lhs) != iv {
        return None;
    }
    let (bound, _) = int_value(module, *rhs)?;
    let trip = match comparison {
        CmpOp::SLessThan => {
            if bound <= start {
                0
            } else {
                u64::try_from((bound - start + step - 1) / step).unwrap_or(u64::MAX)
            }
        }
        CmpOp::SLessThanEqual => {
            if bound < start {
                0
            } else {
                u64::try_from((bound - start) / step + 1).unwrap_or(u64::MAX)
            }
        }
        _ => return None,
    };
    Some(Counted {
        header,
        cond_block: condition_block,
        body: then,
        cont,
        merge,
        preheader,
        iv,
        iv_type,
        bits,
        start,
        step,
        trip,
    })
}

/// The value a chain of copies ends at.
///
/// The front end reads a local through its slot, so the value that
/// reaches the loop's arithmetic is a copy of the induction variable
/// rather than the variable itself.
fn resolve(defs: &BTreeMap<ValueId, (usize, usize)>, body: &[BasicBlock], value: ValueId) -> ValueId {
    let mut current = value;
    for _ in 0..64 {
        let Some(&(block, index)) = defs.get(&current) else {
            return current;
        };
        let Some(inst) = body
            .get(block)
            .and_then(|block| block.insts.get(index))
        else {
            return current;
        };
        let Op::CopyObject { operand } = inst.op else {
            return current;
        };
        current = operand;
    }
    current
}

/// The merges of a block.
fn loop_body_phis(block: &BasicBlock) -> Vec<&Inst> {
    block
        .insts
        .iter()
        .filter(|inst| matches!(inst.op, Op::Phi { .. }))
        .collect()
}

/// The value of an integer constant, read with the signedness of its
/// type.
fn int_value(module: &Module, value: ValueId) -> Option<(i128, bool)> {
    let Some(ConstValue::Int(raw)) = module.constant_value(value) else {
        return None;
    };
    let (bits, signed) = int_shape(module, module.type_of(value))?;
    let wide = if signed {
        i128::from(sign_extend(raw, bits))
    } else {
        i128::from(raw)
    };
    Some((wide, signed))
}

/// Unrolls one counted loop.
fn unroll(module: &mut Module, func: ValueId, counted: &Counted, limit: u32) -> Result<bool> {
    if counted.trip == 0 {
        return Ok(false);
    }
    let limit = u64::from(limit);
    let complete = counted.trip <= limit;
    let copies = if complete { counted.trip } else { limit };

    // The complete unrolling removes the loop, so the values the loop
    // produced must not be used anywhere else.
    if complete && !values_stay_inside(module, func, counted)? {
        return Ok(false);
    }

    // The copies of the body.
    let Some(source) = module
        .function(func)
        .and_then(|f| f.body.as_ref())
        .and_then(|body| body.get(counted.body))
        .cloned()
    else {
        return Ok(false);
    };
    let mut blocks: Vec<BlockId> = Vec::with_capacity(copies as usize);

    // The values of the body that read the induction variable.
    let Some(function_body) = module
        .function(func)
        .and_then(|function| function.body.as_ref())
    else {
        return Ok(false);
    };
    let defs = definitions(function_body);
    let mut readers: BTreeSet<ValueId> = BTreeSet::new();
    let Some(loop_body) = function_body.get(counted.body) else {
        return Ok(false);
    };
    for inst in &loop_body.insts {
        let mut operands = Vec::new();
        collect_operands(&inst.op, &mut operands);
        for operand in operands {
            if resolve(&defs, function_body, operand) == counted.iv {
                readers.insert(operand);
            }
        }
    }

    // The value the induction variable takes in each copy.
    let mut counters: Vec<ValueId> = Vec::with_capacity(copies as usize);
    for copy in 0..copies {
        let value = counted.start + i128::from(copy) * counted.step;
        let raw = mask(value as u64, counted.bits);
        let Ok(constant) = module.intern_const(counted.iv_type, ConstValue::Int(raw)) else {
            return Ok(false);
        };
        counters.push(constant);
    }
    // The value the loop starts at when it runs the rest of the
    // iterations.
    let remainder = if complete {
        None
    } else {
        let start = counted.start + i128::from(copies) * counted.step;
        let raw = mask(start as u64, counted.bits);
        let Ok(constant) = module.intern_const(counted.iv_type, ConstValue::Int(raw)) else {
            return Ok(false);
        };
        Some(constant)
    };

    // The copies of the loop's body.
    for _ in 0..copies {
        let name = module.uniquify_name("unroll");
        let Ok(block) = module.push_block(func, &name) else {
            return Ok(false);
        };
        blocks.push(block);
    }

    // The copies themselves.
    let last = copies as usize - 1;
    for (index, block) in blocks.iter().enumerate() {
        let target = if index == last {
            if complete {
                block_id(counted.merge)
            } else {
                block_id(counted.header)
            }
        } else {
            blocks[index + 1]
        };
        // The values this copy defines.
        let mut map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
        for inst in &source.insts {
            if let (Some(result), Some(ty)) = (inst.result, inst.ty) {
                let value = module.new_inst_value(ty);
                map.insert(result, value);
            }
        }
        let mut instructions: Vec<Inst> = Vec::with_capacity(source.insts.len());
        for (position, inst) in source.insts.iter().enumerate() {
            let mut inst = inst.clone();
            // The copy gets its own identity for every value.
            if let Some(result) = inst.result
                && let Some(copied) = map.get(&result)
            {
                inst.result = Some(*copied);
            }
            remap(&mut inst, &map, &readers, counters[index]);
            if position + 1 == source.insts.len() {
                inst.op = Op::Branch { target };
            }
            instructions.push(inst);
        }
        for inst in instructions {
            if module.emit(func, *block, inst).is_err() {
                return Ok(false);
            }
        }
    }

    // The block that enters the loop enters the first copy.
    let Some(function) = module.function_mut(func) else {
        return Ok(false);
    };
    let Some(body) = function.body.as_mut() else {
        return Ok(false);
    };
    let Some(preheader) = body.get_mut(counted.preheader) else {
        return Ok(false);
    };
    let Some(last_inst) = preheader.insts.last_mut() else {
        return Ok(false);
    };
    let Op::Branch { target } = &mut last_inst.op else {
        return Ok(false);
    };
    *target = blocks[0];

    if !complete {
        // The loop runs the rest of the iterations: its induction
        // variable starts where the copies leave it.
        let Some(constant) = remainder else {
            return Ok(false);
        };
        let Some(body) = function.body.as_mut() else {
            return Ok(false);
        };
        let Some(header) = body.get_mut(counted.header) else {
            return Ok(false);
        };
        for inst in header.insts.iter_mut() {
            let Op::Phi { incomings } = &mut inst.op else {
                continue;
            };
            for (value, predecessor) in incomings.iter_mut() {
                if predecessor.index() == counted.preheader {
                    *value = constant;
                    *predecessor = blocks[last];
                }
            }
        }
        return Ok(true);
    }

    // The complete unrolling removes the loop.  The continue block goes
    // with it when it is not the body: the copies branch into the
    // header directly.
    let mut dead: BTreeSet<usize> = BTreeSet::from([counted.header, counted.body]);
    if counted.cond_block != counted.header {
        dead.insert(counted.cond_block);
    }
    if counted.cont != counted.body {
        dead.insert(counted.cont);
    }
    remove_dead_blocks(module, func, &dead)?;
    Ok(true)
}

/// True when the values the loop produces are used nowhere else.
fn values_stay_inside(module: &Module, func: ValueId, counted: &Counted) -> Result<bool> {
    let Some(body) = module
        .function(func)
        .and_then(|f| f.body.as_ref())
    else {
        return Ok(false);
    };
    let mut loop_blocks: BTreeSet<usize> = BTreeSet::new();
    loop_blocks.insert(counted.header);
    loop_blocks.insert(counted.cond_block);
    loop_blocks.insert(counted.body);
    loop_blocks.insert(counted.cont);
    let mut defined: BTreeSet<ValueId> = BTreeSet::new();
    for index in &loop_blocks {
        let Some(block) = body.get(*index) else {
            return Ok(false);
        };
        for inst in &block.insts {
            if let Some(result) = inst.result {
                defined.insert(result);
            }
        }
    }
    for (index, block) in body.iter().enumerate() {
        for inst in &block.insts {
            let mut operands = Vec::new();
            collect_operands(&inst.op, &mut operands);
            for operand in operands {
                if !defined.contains(&operand) {
                    continue;
                }
                // The use must sit inside the loop as well.
                if !loop_blocks.contains(&index) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// Rewrites the operands of a copied instruction.
fn remap(inst: &mut Inst, map: &BTreeMap<ValueId, ValueId>, readers: &BTreeSet<ValueId>, counter: ValueId) {
    let value = |slot: &mut ValueId| {
        if readers.contains(slot) {
            *slot = counter;
            return;
        }
        if let Some(mapped) = map.get(slot) {
            *slot = *mapped;
        }
    };
    match &mut inst.op {
        Op::Variable { init } => {
            if let Some(slot) = init {
                value(slot);
            }
        }
        Op::Load { ptr } => value(ptr),
        Op::Store { ptr, value: stored } => {
            value(ptr);
            value(stored);
        }
        Op::AccessChain { base, indices } => {
            value(base);
            for index in indices.iter_mut() {
                value(index);
            }
        }
        Op::PtrAccessChain { base, indices } => {
            value(base);
            for index in indices.iter_mut() {
                value(index);
            }
        }
        Op::CopyObject { operand } => value(operand),
        Op::Unary { operand, .. } => value(operand),
        Op::Binary { lhs, rhs, .. } => {
            value(lhs);
            value(rhs);
        }
        Op::Compare { lhs, rhs, .. } => {
            value(lhs);
            value(rhs);
        }
        Op::Select { cond, a, b } => {
            value(cond);
            value(a);
            value(b);
        }
        Op::Convert { operand, .. } => value(operand),
        Op::Call { args, .. } => {
            for argument in args.iter_mut() {
                value(argument);
            }
        }
        Op::ExtInst { args, .. } => {
            for argument in args.iter_mut() {
                value(argument);
            }
        }
        Op::CompositeExtract { composite, .. } => value(composite),
        Op::CompositeConstruct { constituents } => {
            for constituent in constituents.iter_mut() {
                value(constituent);
            }
        }
        Op::Phi { .. } => {}
        Op::ControlBarrier { exec, mem, semantics } => {
            value(exec);
            value(mem);
            value(semantics);
        }
        Op::MemoryBarrier { mem, semantics } => {
            value(mem);
            value(semantics);
        }
        Op::SelectionMerge { .. } | Op::LoopMerge { .. } => {}
        Op::Branch { .. } => {}
        Op::BranchConditional { cond, .. } => value(cond),
        Op::Return => {}
        Op::ReturnValue { value: returned } => value(returned),
        Op::Unreachable => {}
    }
}
