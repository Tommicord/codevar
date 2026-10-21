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

//! Straight-line vectorization: groups of independent scalar
//! operations over neighbouring values become one vector operation.
//!
//! The rewrite keeps the results of the group: every member's
//! instruction becomes a read of one lane of the new vector
//! operation.  The users of the group therefore keep working without
//! being rewritten, and the vector operation dominates exactly the
//! blocks the members dominated before.
//!
//! The members of a group do not have to sit next to one another —
//! four additions of four loads that follow one another are one
//! group — but the vector operation is placed where the operands of
//! every member are available.  When that place lies behind a
//! member, the member moves behind the vector operation, which is
//! only safe when nothing reads the member before that point.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{BasicBlock, BinOp, CmpOp, ConvOp, Inst, Module, Op, Type, TypeId, UnOp, ValueId};

use super::VectorConfig;

/// The opcode family a vectorizable instruction belongs to.
///
/// Two instructions share a vector only when their opcode, their
/// operand type, and their result type agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// A binary arithmetic operation.
    Binary(BinOp),
    /// A unary operation.
    Unary(UnOp),
    /// A comparison.
    Compare(CmpOp),
    /// A conversion.
    Convert(ConvOp),
}

/// A group of independent scalar instructions that becomes one vector
/// instruction.
struct Group {
    /// The indices of the members, in the order they appear.
    members: Vec<usize>,
    /// The index the vector operation is placed at.
    insert: usize,
    /// The opcode family the members share.
    shape: Shape,
    /// The type of the members' operands.
    operand_ty: TypeId,
    /// The type of the members' results.
    result_ty: TypeId,
}

/// Vectorizes the straight-line groups of every function in `module`.
pub(super) fn vectorize_straight_line(module: &mut Module, config: &VectorConfig) {
    let min = config.min_group_size as usize;
    let max = config.max_lanes as usize;
    if min < 2 || max < 2 {
        return;
    }
    for func in module.functions().to_vec() {
        let Some(mut blocks) = module
            .function_mut(func)
            .and_then(|f| f.body.take())
        else {
            continue;
        };
        for block in &mut blocks {
            vectorize_block(module, block, min, max);
        }
        if let Some(function) = module.function_mut(func) {
            function.body = Some(blocks);
        }
    }
}

/// Vectorizes the groups of one block.
fn vectorize_block(module: &mut Module, block: &mut BasicBlock, min: usize, max: usize) {
    loop {
        let definitions = definition_indices(block);
        let Some(group) = find_group(module, block, &definitions, min, max) else {
            break;
        };
        apply_group(module, block, &group);
    }
}

/// Maps every value the block defines to the index of its definition.
fn definition_indices(block: &BasicBlock) -> BTreeMap<ValueId, usize> {
    let mut definitions = BTreeMap::new();
    for (index, inst) in block.insts.iter().enumerate() {
        if let Some(result) = inst.result {
            definitions.insert(result, index);
        }
    }
    definitions
}

/// Finds the first group of the block that is worth a vector.
///
/// A group is a run of instructions that share one opcode and one
/// type, whose members are independent of one another, and whose
/// operands are all available where the vector operation goes.
fn find_group(
    module: &Module,
    block: &BasicBlock,
    definitions: &BTreeMap<ValueId, usize>,
    min: usize,
    max: usize,
) -> Option<Group> {
    let mut index = 0usize;
    while index < block.insts.len() {
        let Some(first) = candidate(module, &block.insts[index]) else {
            index += 1;
            continue;
        };
        // The run of candidates that share the opcode and the type.
        let mut members = alloc::vec![index];
        let mut scan = index + 1;
        while members.len() < max {
            let Some(inst) = block.insts.get(scan) else {
                break;
            };
            let Some(next) = candidate(module, inst) else {
                scan += 1;
                continue;
            };
            if next.shape != first.shape
                || next.operand_ty != first.operand_ty
                || next.result_ty != first.result_ty
            {
                break;
            }
            members.push(scan);
            scan += 1;
        }
        if members.len() < min {
            index = scan;
            continue;
        }
        if independent(block, definitions, &members)
            && let Some(insert) = insertion_point(block, definitions, &members)
        {
            return Some(Group {
                members,
                insert,
                shape: first.shape,
                operand_ty: first.operand_ty,
                result_ty: first.result_ty,
            });
        }
        index += 1;
    }
    None
}

/// True when no member of the group reads the result of another one.
fn independent(block: &BasicBlock, definitions: &BTreeMap<ValueId, usize>, members: &[usize]) -> bool {
    let mut operands = Vec::new();
    for &member in members {
        operands.clear();
        collect_operands(&block.insts[member].op, &mut operands);
        for &operand in operands.iter() {
            if let Some(&defined) = definitions.get(&operand)
                && members.contains(&defined)
            {
                return false;
            }
        }
    }
    true
}

/// The index the vector operation goes to.
///
/// The place must hold every operand of the group, and it must not
/// lie behind a use of a member's result, because the member would
/// then no longer dominate that use.
fn insertion_point(
    block: &BasicBlock,
    definitions: &BTreeMap<ValueId, usize>,
    members: &[usize],
) -> Option<usize> {
    let last = *members.last()?;
    let uses = first_uses(block, members);
    let mut operands = Vec::new();
    for position in *members.first()?..=last {
        let mut available = true;
        for &member in members {
            operands.clear();
            collect_operands(&block.insts[member].op, &mut operands);
            for &operand in operands.iter() {
                if let Some(&defined) = definitions.get(&operand)
                    && defined >= position
                {
                    available = false;
                }
            }
        }
        if !available {
            continue;
        }
        if uses.iter().any(|&read| read < position) {
            continue;
        }
        return Some(position);
    }
    None
}

/// The first instruction that reads each member's result.
fn first_uses(block: &BasicBlock, members: &[usize]) -> Vec<usize> {
    let mut uses = alloc::vec![usize::MAX; members.len()];
    let mut operands = Vec::new();
    for (index, inst) in block.insts.iter().enumerate() {
        operands.clear();
        collect_operands(&inst.op, &mut operands);
        for &operand in operands.iter() {
            for (lane, &member) in members.iter().enumerate() {
                if uses[lane] == usize::MAX && block.insts[member].result == Some(operand) {
                    uses[lane] = index;
                }
            }
        }
    }
    uses
}

/// The values an operation reads.
fn collect_operands(op: &Op, out: &mut Vec<ValueId>) {
    match op {
        Op::Variable { init } => {
            if let Some(init) = init {
                out.push(*init);
            }
        }
        Op::Load { ptr } => out.push(*ptr),
        Op::Store { ptr, value } => {
            out.push(*ptr);
            out.push(*value);
        }
        Op::AccessChain { base, indices } | Op::PtrAccessChain { base, indices } => {
            out.push(*base);
            out.extend_from_slice(indices);
        }
        Op::CopyObject { operand } => out.push(*operand),
        Op::Unary { operand, .. } => out.push(*operand),
        Op::Binary { lhs, rhs, .. } => {
            out.push(*lhs);
            out.push(*rhs);
        }
        Op::Compare { lhs, rhs, .. } => {
            out.push(*lhs);
            out.push(*rhs);
        }
        Op::Select { cond, a, b } => {
            out.push(*cond);
            out.push(*a);
            out.push(*b);
        }
        Op::Convert { operand, .. } => out.push(*operand),
        Op::Call { callee, args } => {
            out.push(*callee);
            out.extend_from_slice(args);
        }
        Op::ExtInst { args, .. } => out.extend_from_slice(args),
        Op::CompositeExtract { composite, .. } => out.push(*composite),
        Op::CompositeConstruct { constituents } => out.extend_from_slice(constituents),
        Op::Phi { incomings } => {
            for (value, _) in incomings {
                out.push(*value);
            }
        }
        Op::ControlBarrier { exec, mem, semantics } => {
            out.push(*exec);
            out.push(*mem);
            out.push(*semantics);
        }
        Op::MemoryBarrier { mem, semantics } => {
            out.push(*mem);
            out.push(*semantics);
        }
        Op::BranchConditional { cond, .. } => out.push(*cond),
        Op::ReturnValue { value } => out.push(*value),
        Op::SelectionMerge { .. }
        | Op::LoopMerge { .. }
        | Op::Branch { .. }
        | Op::Return
        | Op::Unreachable => {}
    }
}

/// The description of `inst` when it can take part in a group.
///
/// Only scalar operations qualify: a vector operation is already
/// vectorized.
fn candidate(module: &Module, inst: &Inst) -> Option<ShapeInfo> {
    let result_ty = inst.ty?;
    if is_vector(module, result_ty) {
        return None;
    }
    match &inst.op {
        Op::Binary { op, lhs, rhs } => {
            let lhs_ty = module.type_of(*lhs);
            if lhs_ty != module.type_of(*rhs) || lhs_ty != result_ty {
                return None;
            }
            Some(ShapeInfo {
                shape: Shape::Binary(*op),
                operand_ty: lhs_ty,
                result_ty,
            })
        }
        Op::Unary { op, operand } => {
            let operand_ty = module.type_of(*operand);
            if operand_ty != result_ty {
                return None;
            }
            Some(ShapeInfo {
                shape: Shape::Unary(*op),
                operand_ty,
                result_ty,
            })
        }
        Op::Compare { op, lhs, rhs } => {
            let operand_ty = module.type_of(*lhs);
            if operand_ty != module.type_of(*rhs) {
                return None;
            }
            if !matches!(module.ty(result_ty), Type::Bool) {
                return None;
            }
            Some(ShapeInfo {
                shape: Shape::Compare(*op),
                operand_ty,
                result_ty,
            })
        }
        Op::Convert { op, operand } => {
            let operand_ty = module.type_of(*operand);
            if is_vector(module, operand_ty) {
                return None;
            }
            Some(ShapeInfo {
                shape: Shape::Convert(*op),
                operand_ty,
                result_ty,
            })
        }
        _ => None,
    }
}

/// The description of a candidate instruction.
struct ShapeInfo {
    /// The opcode family.
    shape: Shape,
    /// The type of the operands.
    operand_ty: TypeId,
    /// The type of the result.
    result_ty: TypeId,
}

/// True when `ty` is a vector type.
fn is_vector(module: &Module, ty: TypeId) -> bool {
    matches!(module.ty(ty), Type::Vector { .. })
}

/// Rewrites the group into one vector operation.
fn apply_group(module: &mut Module, block: &mut BasicBlock, group: &Group) {
    let lanes = u32::try_from(group.members.len()).unwrap_or(u32::MAX);
    let (left, right) = member_operands(block, group);
    let mut new_insts: Vec<Inst> = Vec::with_capacity(3);
    let vector_ty = module.vector_ty(group.operand_ty, lanes);
    let result = match group.shape {
        Shape::Binary(op) => {
            let lhs = construct(module, &mut new_insts, vector_ty, left);
            let rhs = construct(module, &mut new_insts, vector_ty, right);
            let value = module.new_inst_value(vector_ty);
            new_insts.push(Inst::def(value, vector_ty, Op::Binary { op, lhs, rhs }));
            value
        }
        Shape::Unary(op) => {
            let operand = construct(module, &mut new_insts, vector_ty, left);
            let value = module.new_inst_value(vector_ty);
            new_insts.push(Inst::def(value, vector_ty, Op::Unary { op, operand }));
            value
        }
        Shape::Compare(op) => {
            let lhs = construct(module, &mut new_insts, vector_ty, left);
            let rhs = construct(module, &mut new_insts, vector_ty, right);
            let bool_ty = module.bool_ty();
            let result_ty = module.vector_ty(bool_ty, lanes);
            let value = module.new_inst_value(result_ty);
            new_insts.push(Inst::def(value, result_ty, Op::Compare { op, lhs, rhs }));
            value
        }
        Shape::Convert(op) => {
            let operand = construct(module, &mut new_insts, vector_ty, left);
            let result_ty = module.vector_ty(group.result_ty, lanes);
            let value = module.new_inst_value(result_ty);
            new_insts.push(Inst::def(value, result_ty, Op::Convert { op, operand }));
            value
        }
    };
    // The members that lie before the vector operation move behind it:
    // their result has to be read from the vector, which the vector
    // operation defines.
    let members: Vec<(Option<ValueId>, Option<TypeId>)> = group
        .members
        .iter()
        .map(|&member| (block.insts[member].result, block.insts[member].ty))
        .collect();
    let old = core::mem::take(&mut block.insts);
    let mut out: Vec<Inst> = Vec::with_capacity(old.len() + new_insts.len() + members.len());
    for (index, inst) in old.into_iter().enumerate() {
        if index == group.insert {
            out.append(&mut new_insts);
            // The members that moved read their lane right behind the
            // vector operation.
            for (lane, &member) in group.members.iter().enumerate() {
                if member < group.insert
                    && let Some(extract) = extract_inst(&members[lane], result, lane)
                {
                    out.push(extract);
                }
            }
        }
        let Some(lane) = group
            .members
            .iter()
            .position(|&member| member == index)
        else {
            out.push(inst);
            continue;
        };
        if index < group.insert {
            // The member moved behind the vector operation.
            continue;
        }
        if let Some(extract) = extract_inst(&members[lane], result, lane) {
            out.push(extract);
        }
    }
    block.insts = out;
}

/// The instruction that reads one lane of the vector.
fn extract_inst(member: &(Option<ValueId>, Option<TypeId>), result: ValueId, lane: usize) -> Option<Inst> {
    let value = member.0?;
    let ty = member.1?;
    Some(Inst::def(
        value,
        ty,
        Op::CompositeExtract {
            composite: result,
            indices: alloc::vec![u32::try_from(lane).unwrap_or(0)],
        },
    ))
}

/// The left and the right operands of every member of the group.
fn member_operands(block: &BasicBlock, group: &Group) -> (Vec<ValueId>, Vec<ValueId>) {
    let mut left = Vec::with_capacity(group.members.len());
    let mut right = Vec::with_capacity(group.members.len());
    for &member in &group.members {
        match &block.insts[member].op {
            Op::Binary { lhs, rhs, .. } => {
                left.push(*lhs);
                right.push(*rhs);
            }
            Op::Unary { operand, .. } => left.push(*operand),
            Op::Compare { lhs, rhs, .. } => {
                left.push(*lhs);
                right.push(*rhs);
            }
            Op::Convert { operand, .. } => left.push(*operand),
            _ => {}
        }
    }
    (left, right)
}

/// Builds the vector that holds the operands of the group.
fn construct(
    module: &mut Module,
    out: &mut Vec<Inst>,
    vector_ty: TypeId,
    constituents: Vec<ValueId>,
) -> ValueId {
    let value = module.new_inst_value(vector_ty);
    out.push(Inst::def(
        value,
        vector_ty,
        Op::CompositeConstruct { constituents },
    ));
    value
}
