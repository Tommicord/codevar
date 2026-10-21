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

//! Inlining of the called functions.
//!
//! The pass brings the body of a called function into its caller when
//! the callee is small enough:
//!
//! * The callee's blocks are copied into the caller.  Every value the
//!   callee defines gets a fresh identity in the caller, so the two
//!   bodies never share a value, and the callee's arguments bind to
//!   the arguments of the call.
//! * The block of the call is split: the part before the call branches
//!   into the copied body, and the part after the call moves into a
//!   continuation block the copied body branches to.
//! * The result of the call is the value the copied body returns.  A
//!   callee with several returns hands the continuation a merge, so
//!   every path that reaches the continuation agrees on the value.
//!
//! The pass refuses to inline a function that declares itself
//! imported, a function that calls its caller, a function without a
//! body, and a function whose body is larger than the threshold.  It
//! also stops after a bounded number of inlinings per caller, so a
//! cycle of calls cannot make it work forever.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{BasicBlock, BlockId, Inst, Module, Op, Type, ValueId};

use crate::error::Result;

/// The number of inlinings the pass performs for one caller.
const INLINE_BUDGET: u32 = 64;

/// Brings the called functions into their callers.
pub(crate) fn run(module: &mut Module, threshold: u32) -> Result<()> {
    let functions: Vec<ValueId> = module.functions().to_vec();
    for caller in functions {
        inline_calls(module, caller, threshold)?;
    }
    Ok(())
}

/// The number of instructions of a body.
fn instruction_count(body: &[BasicBlock]) -> u32 {
    let mut count = 0u32;
    for block in body {
        count = count.saturating_add(u32::try_from(block.insts.len()).unwrap_or(u32::MAX));
    }
    count
}

/// Inlines the calls of one caller.
fn inline_calls(module: &mut Module, caller: ValueId, threshold: u32) -> Result<()> {
    let mut budget = INLINE_BUDGET;
    while budget > 0 {
        budget -= 1;
        let Some(body) = module
            .function(caller)
            .and_then(|f| f.body.as_ref())
        else {
            return Ok(());
        };
        // The first call the pass can bring in.
        let mut target = None;
        for (block_index, block) in body.iter().enumerate() {
            for (inst_index, inst) in block.insts.iter().enumerate() {
                let Op::Call { callee, args } = &inst.op else {
                    continue;
                };
                if *callee == caller || !eligible(module, *callee, caller, args.len(), threshold) {
                    continue;
                }
                target = Some((block_index, inst_index, *callee));
                break;
            }
            if target.is_some() {
                break;
            }
        }
        let Some((block_index, inst_index, callee)) = target else {
            return Ok(());
        };
        if !inline_one(module, caller, callee, block_index, inst_index)? {
            return Ok(());
        }
    }
    Ok(())
}

/// True when the callee may be brought into the caller.
fn eligible(
    module: &Module,
    callee: ValueId,
    caller: ValueId,
    argument_count: usize,
    threshold: u32,
) -> bool {
    let Some(def) = module.function(callee) else {
        return false;
    };
    let Some(body) = def.body.as_ref() else {
        return false;
    };
    if def.args.len() != argument_count {
        return false;
    }
    if instruction_count(body) > threshold {
        return false;
    }
    // The callee must return, or the continuation would be a block
    // nobody reaches.
    let returns = body.iter().any(|block| {
        matches!(
            block.insts.last().map(|inst| &inst.op),
            Some(Op::Return | Op::ReturnValue { .. })
        )
    });
    if !returns {
        return false;
    }
    // A callee that calls its caller would not terminate.
    !body.iter().any(|block| {
        block
            .insts
            .iter()
            .any(|inst| matches!(&inst.op, Op::Call { callee, .. } if *callee == caller))
    })
}

/// Brings one call into its caller.
///
/// Returns `false` when the call cannot be brought in, in which case
/// the module is left as it was.
fn inline_one(
    module: &mut Module,
    caller: ValueId,
    callee: ValueId,
    block_index: usize,
    inst_index: usize,
) -> Result<bool> {
    // The call and the callee.
    let (call_args, call_result, call_type) = {
        let Some(body) = module
            .function(caller)
            .and_then(|f| f.body.as_ref())
        else {
            return Ok(false);
        };
        let Some(block) = body.get(block_index) else {
            return Ok(false);
        };
        let Some(inst) = block.insts.get(inst_index) else {
            return Ok(false);
        };
        let Op::Call { args, .. } = &inst.op else {
            return Ok(false);
        };
        (args.clone(), inst.result, inst.ty)
    };
    let (callee_args, callee_blocks, return_type) = {
        let Some(def) = module.function(callee) else {
            return Ok(false);
        };
        let Some(body) = def.body.as_ref() else {
            return Ok(false);
        };
        let Type::Function { ret, .. } = module.ty(def.sig) else {
            return Ok(false);
        };
        // The body is copied because the module cannot be borrowed
        // mutably while the callee is read.
        (def.args.clone(), body.clone(), *ret)
    };

    // The values the copy binds: the callee's arguments take the
    // caller's arguments, and the callee's results get a new identity.
    let mut values: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    for (callee_arg, call_arg) in callee_args.iter().zip(call_args.iter()) {
        values.insert(*callee_arg, *call_arg);
    }

    // The continuation block the copied body returns to.
    let continuation_name = module.uniquify_name("inline.cont");
    let continuation = match module.push_block(caller, &continuation_name) {
        Ok(block) => block,
        Err(_) => return Ok(false),
    };

    // The copies of the callee's blocks.
    let mut blocks: BTreeMap<usize, BlockId> = BTreeMap::new();
    for (index, block) in callee_blocks.iter().enumerate() {
        let name = module.uniquify_name(&block.name);
        let Ok(new_block) = module.push_block(caller, &name) else {
            return Ok(false);
        };
        blocks.insert(index, new_block);
    }

    // The copies of the callee's results.
    for block in &callee_blocks {
        for inst in &block.insts {
            if let (Some(result), Some(ty)) = (inst.result, inst.ty) {
                let value = module.new_inst_value(ty);
                values.insert(result, value);
            }
        }
    }

    // The merge the continuation needs when the callee returns from
    // several places.
    let mut returns: Vec<(ValueId, BlockId)> = Vec::new();
    for (index, block) in callee_blocks.iter().enumerate() {
        let Some(new_block) = blocks.get(&index).copied() else {
            return Ok(false);
        };
        for inst in &block.insts {
            let mut inst = inst.clone();
            // The copy gets a new identity, so the two bodies never
            // share a value.
            if let Some(result) = inst.result
                && let Some(copied) = values.get(&result)
            {
                inst.result = Some(*copied);
            }
            remap(&mut inst, &values, &blocks);
            match &mut inst.op {
                Op::Return => {
                    inst.op = Op::Branch { target: continuation };
                }
                Op::ReturnValue { value } => {
                    returns.push((*value, new_block));
                    inst.op = Op::Branch { target: continuation };
                }
                _ => {}
            }
            if module.emit(caller, new_block, inst).is_err() {
                return Ok(false);
            }
        }
    }

    // The merge of the several returns.
    let merged = if returns.len() > 1 {
        let value = module.new_inst_value(return_type);
        let incomings = returns.clone();
        Some((value, incomings))
    } else {
        None
    };

    // The caller's block: the call becomes a branch into the copy.
    let Some(function) = module.function_mut(caller) else {
        return Ok(false);
    };
    let Some(body) = function.body.as_mut() else {
        return Ok(false);
    };
    let Some(block) = body.get_mut(block_index) else {
        return Ok(false);
    };
    let old = core::mem::take(&mut block.insts);
    let mut fresh: Vec<Inst> = Vec::with_capacity(old.len());
    let mut after: Vec<Inst> = Vec::new();
    let Some(entry) = blocks.get(&0).copied() else {
        return Ok(false);
    };
    for (index, inst) in old.into_iter().enumerate() {
        if index == inst_index {
            fresh.push(Inst::none(Op::Branch { target: entry }));
            continue;
        }
        if index > inst_index {
            after.push(inst);
            continue;
        }
        fresh.push(inst);
    }
    block.insts = fresh;

    // The continuation: the instructions that follow the call, the
    // merge of the returns, and the value the call produced.
    let Some(continuation_block) = body.get_mut(continuation.index()) else {
        return Ok(false);
    };
    let mut fresh_continuation: Vec<Inst> = Vec::with_capacity(after.len() + 2);
    if let Some((value, incomings)) = &merged {
        fresh_continuation.push(Inst::def(
            *value,
            return_type,
            Op::Phi {
                incomings: incomings.clone(),
            },
        ));
    }
    if let Some(result) = call_result {
        let source = match &merged {
            Some((value, _)) => *value,
            None => match returns.first() {
                Some((value, _)) => *value,
                None => return Ok(false),
            },
        };
        let ty = call_type.unwrap_or(return_type);
        fresh_continuation.push(Inst::def(result, ty, Op::CopyObject { operand: source }));
    }
    let terminator = after.pop();
    fresh_continuation.extend(after);
    if let Some(terminator) = terminator {
        fresh_continuation.push(terminator);
    }
    continuation_block.insts = fresh_continuation;
    Ok(true)
}

/// Rewrites the operands of a copied instruction.
fn remap(inst: &mut Inst, values: &BTreeMap<ValueId, ValueId>, blocks: &BTreeMap<usize, BlockId>) {
    let value = |slot: &mut ValueId| {
        if let Some(mapped) = values.get(slot) {
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
        Op::Phi { incomings } => {
            for (incoming, predecessor) in incomings.iter_mut() {
                value(incoming);
                *predecessor = blocks
                    .get(&predecessor.index())
                    .copied()
                    .unwrap_or(*predecessor);
            }
        }
        Op::ControlBarrier { exec, mem, semantics } => {
            value(exec);
            value(mem);
            value(semantics);
        }
        Op::MemoryBarrier { mem, semantics } => {
            value(mem);
            value(semantics);
        }
        Op::SelectionMerge { target, .. } => {
            *target = blocks
                .get(&target.index())
                .copied()
                .unwrap_or(*target);
        }
        Op::LoopMerge { merge, cont, .. } => {
            *merge = blocks
                .get(&merge.index())
                .copied()
                .unwrap_or(*merge);
            *cont = blocks
                .get(&cont.index())
                .copied()
                .unwrap_or(*cont);
        }
        Op::Branch { target } => {
            *target = blocks
                .get(&target.index())
                .copied()
                .unwrap_or(*target);
        }
        Op::BranchConditional { then, other, .. } => {
            *then = blocks
                .get(&then.index())
                .copied()
                .unwrap_or(*then);
            *other = blocks
                .get(&other.index())
                .copied()
                .unwrap_or(*other);
        }
        Op::Return => {}
        Op::ReturnValue { value: returned } => value(returned),
        Op::Unreachable => {}
    }
}
