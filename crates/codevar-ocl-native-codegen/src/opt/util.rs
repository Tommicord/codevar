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

//! Helpers the optimization passes share.
//!
//! Three facts about the intermediate representation shape every
//! helper in this module:
//!
//! * **The value arena is append-only.**  A value that an instruction
//!   defined cannot leave the arena, and the verifier rejects a value
//!   that no instruction defines any more.  A pass that removes a
//!   computation therefore has to keep its result defined: [`kill`]
//!   replaces the operation with a copy of the undefined value, which
//!   costs nothing and keeps the module verifiable.
//! * **Blocks are identified by their position.**  A block handle is an
//!   index into the function's block list, so a rewrite that adds or
//!   removes a block renumbers the handles around it.
//! * **Every value knows its type.**  The passes read the type arena
//!   rather than the value's definition to decide what an operation
//!   computes.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{BasicBlock, BlockId, ConstValue, Inst, Module, Op, Type, TypeId, ValueId};

use crate::error::CodegenError;

/// The block handle of the block at `index`.
pub(crate) fn block_id(index: usize) -> BlockId {
    BlockId(u32::try_from(index).unwrap_or(u32::MAX))
}

/// The number of copies the constant resolver follows before it gives
/// up; a chain longer than this is a cycle the pass must not trust.
const MAX_COPY_DEPTH: u32 = 64;

/// Builds the optimizer's failure.
pub(crate) fn failure(what: &'static str, detail: &str) -> CodegenError {
    CodegenError::Optimization {
        what,
        detail: String::from(detail),
    }
}

/// The control-flow graph of one function.
pub(crate) struct Cfg {
    /// The successors of every block, by block index.
    pub(crate) succs: Vec<Vec<usize>>,
    /// The predecessors of every block, by block index.
    pub(crate) preds: Vec<Vec<usize>>,
    /// Whether the block is reachable from the entry block.
    pub(crate) reachable: Vec<bool>,
}

impl Cfg {
    /// Builds the graph of `body`.
    ///
    /// A branch that names a block outside the function is ignored: the
    /// graph then describes a function the verifier would reject, and
    /// the passes that use it leave such a function alone.
    pub(crate) fn build(body: &[BasicBlock]) -> Self {
        let len = body.len();
        let mut succs = Vec::with_capacity(len);
        for block in body {
            let mut next = Vec::new();
            if let Some(last) = block.insts.last() {
                match &last.op {
                    Op::Branch { target } => push_target(&mut next, target.index(), len),
                    Op::BranchConditional { then, other, .. } => {
                        push_target(&mut next, then.index(), len);
                        push_target(&mut next, other.index(), len);
                    }
                    _ => {}
                }
            }
            succs.push(next);
        }
        let mut preds = Vec::with_capacity(len);
        preds.resize_with(len, Vec::new);
        for (block, next) in succs.iter().enumerate() {
            for &target in next {
                if let Some(list) = preds.get_mut(target) {
                    list.push(block);
                }
            }
        }
        let mut reachable = Vec::with_capacity(len);
        reachable.resize(len, false);
        if len > 0 {
            if let Some(flag) = reachable.first_mut() {
                *flag = true;
            }
            let mut queue = Vec::with_capacity(len);
            queue.push(0usize);
            while let Some(block) = queue.pop() {
                for &target in &succs[block] {
                    if let Some(flag) = reachable.get_mut(target)
                        && !*flag
                    {
                        *flag = true;
                        queue.push(target);
                    }
                }
            }
        }
        Self {
            succs,
            preds,
            reachable,
        }
    }

    /// The number of blocks in the graph.
    pub(crate) fn len(&self) -> usize {
        self.succs.len()
    }
}

/// Appends `target` to `next` when it names a block of the function.
fn push_target(next: &mut Vec<usize>, target: usize, len: usize) {
    if target < len {
        next.push(target);
    }
}

/// The dominator sets of every block: `dom[block][ancestor]` holds when
/// `ancestor` dominates `block`.
///
/// The computation matches the verifier's: the entry block dominates
/// only itself, an unreachable block dominates only itself, and every
/// other block is dominated by the intersection of its predecessors'
/// dominator sets.
pub(crate) fn dominators(cfg: &Cfg) -> Vec<Vec<bool>> {
    let len = cfg.len();
    let mut dom = Vec::with_capacity(len);
    for block in 0..len {
        let mut row = Vec::with_capacity(len);
        if cfg
            .reachable
            .get(block)
            .is_some_and(|reached| *reached)
        {
            row.resize(len, true);
        } else {
            row.resize(len, false);
            if let Some(flag) = row.get_mut(block) {
                *flag = true;
            }
        }
        dom.push(row);
    }
    if let Some(entry) = dom.first_mut() {
        for (block, flag) in entry.iter_mut().enumerate() {
            *flag = block == 0;
        }
    }
    loop {
        let mut changed = false;
        for block in 1..len {
            if !cfg
                .reachable
                .get(block)
                .is_some_and(|reached| *reached)
            {
                continue;
            }
            let mut next = Vec::with_capacity(len);
            next.resize(len, true);
            if let Some(list) = cfg.preds.get(block) {
                for &pred in list {
                    if let Some(row) = dom.get(pred) {
                        for (flag, &known) in next.iter_mut().zip(row.iter()) {
                            *flag &= known;
                        }
                    }
                }
            }
            if let Some(flag) = next.get_mut(block) {
                *flag = true;
            }
            if dom.get(block) != Some(&next) {
                if let Some(slot) = dom.get_mut(block) {
                    *slot = next;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    dom
}

/// True when `ancestor` dominates `block`.
pub(crate) fn dominates(dom: &[Vec<bool>], ancestor: usize, block: usize) -> bool {
    dom.get(block)
        .is_some_and(|set| set.get(ancestor).copied().unwrap_or(false))
}

/// True when `ancestor` dominates `block` and is not `block` itself.
pub(crate) fn strictly_dominates(dom: &[Vec<bool>], ancestor: usize, block: usize) -> bool {
    ancestor != block && dominates(dom, ancestor, block)
}

/// The children of every block in the dominator tree.
pub(crate) fn dominator_children(cfg: &Cfg, dom: &[Vec<bool>]) -> Vec<Vec<usize>> {
    let len = cfg.len();
    let mut children = Vec::with_capacity(len);
    children.resize_with(len, Vec::new);
    for block in 1..len {
        if !cfg
            .reachable
            .get(block)
            .is_some_and(|reached| *reached)
        {
            continue;
        }
        // The immediate dominator is the closest strict dominator: the
        // one that dominates the most blocks.
        let mut best = None;
        let mut best_depth = 0usize;
        for candidate in 0..len {
            if candidate == block || !strictly_dominates(dom, candidate, block) {
                continue;
            }
            let depth = dom[candidate]
                .iter()
                .filter(|flag| **flag)
                .count();
            if depth > best_depth {
                best = Some(candidate);
                best_depth = depth;
            }
        }
        if let Some(parent) = best
            && let Some(list) = children.get_mut(parent)
        {
            list.push(block);
        }
    }
    children
}

/// The blocks of the dominator tree in the order the tree is walked.
pub(crate) fn dominator_preorder(cfg: &Cfg, children: &[Vec<usize>]) -> Vec<usize> {
    let mut order = Vec::with_capacity(cfg.len());
    let mut stack = Vec::with_capacity(cfg.len());
    stack.push(0usize);
    while let Some(block) = stack.pop() {
        order.push(block);
        if let Some(list) = children.get(block) {
            for &child in list.iter().rev() {
                stack.push(child);
            }
        }
    }
    order
}

/// Where every instruction result is defined: the block and the
/// position of the instruction that defines it.
pub(crate) fn definitions(body: &[BasicBlock]) -> BTreeMap<ValueId, (usize, usize)> {
    let mut defs = BTreeMap::new();
    for (block_index, block) in body.iter().enumerate() {
        for (inst_index, inst) in block.insts.iter().enumerate() {
            if let Some(result) = inst.result {
                defs.insert(result, (block_index, inst_index));
            }
        }
    }
    defs
}

/// Appends the value operands of `op` to `out`.
pub(crate) fn collect_operands(op: &Op, out: &mut Vec<ValueId>) {
    match op {
        Op::Variable { init } => out.extend(init.iter().copied()),
        Op::Load { ptr } => out.push(*ptr),
        Op::Store { ptr, value } => {
            out.push(*ptr);
            out.push(*value);
        }
        Op::AccessChain { base, indices } => {
            out.push(*base);
            out.extend(indices.iter().copied());
        }
        Op::PtrAccessChain { base, indices } => {
            out.push(*base);
            out.extend(indices.iter().copied());
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
            out.extend(args.iter().copied());
        }
        Op::ExtInst { args, .. } => out.extend(args.iter().copied()),
        Op::CompositeExtract { composite, .. } => out.push(*composite),
        Op::CompositeConstruct { constituents } => out.extend(constituents.iter().copied()),
        Op::Phi { incomings } => out.extend(incomings.iter().map(|&(value, _)| value)),
        Op::ControlBarrier { exec, mem, semantics } => {
            out.push(*exec);
            out.push(*mem);
            out.push(*semantics);
        }
        Op::MemoryBarrier { mem, semantics } => {
            out.push(*mem);
            out.push(*semantics);
        }
        Op::SelectionMerge { .. } | Op::LoopMerge { .. } => {}
        Op::Branch { .. } => {}
        Op::BranchConditional { cond, .. } => out.push(*cond),
        Op::Return => {}
        Op::ReturnValue { value } => out.push(*value),
        Op::Unreachable => {}
    }
}

/// True when the operation reads or writes memory, calls out, or
/// synchronizes: such an instruction must survive every pass that
/// removes unused work.
pub(crate) fn has_side_effect(op: &Op) -> bool {
    matches!(
        op,
        Op::Store { .. }
            | Op::Call { .. }
            | Op::ExtInst { .. }
            | Op::ControlBarrier { .. }
            | Op::MemoryBarrier { .. }
    )
}

/// True when the operation computes a value without touching memory, so
/// it may be removed when nobody uses its result.
pub(crate) fn is_removable(op: &Op) -> bool {
    !has_side_effect(op)
        && matches!(
            op,
            Op::Variable { .. }
                | Op::Load { .. }
                | Op::AccessChain { .. }
                | Op::PtrAccessChain { .. }
                | Op::CopyObject { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Compare { .. }
                | Op::Select { .. }
                | Op::Convert { .. }
                | Op::CompositeExtract { .. }
                | Op::CompositeConstruct { .. }
                | Op::Phi { .. }
        )
}

/// True when the operation may be moved to another block without
/// changing what the program computes.
///
/// A load is excluded: the memory it reads may change between the two
/// positions, so the optimizer never hoists one.
pub(crate) fn is_movable(op: &Op) -> bool {
    matches!(
        op,
        Op::AccessChain { .. }
            | Op::PtrAccessChain { .. }
            | Op::CopyObject { .. }
            | Op::Unary { .. }
            | Op::Binary { .. }
            | Op::Compare { .. }
            | Op::Select { .. }
            | Op::Convert { .. }
            | Op::CompositeExtract { .. }
            | Op::CompositeConstruct { .. }
    )
}

/// Replaces the operation of `inst` with a copy of `value`.
///
/// The result keeps its identity: the value stays defined and its uses
/// stay valid, while the computation it performed disappears.  The
/// value arena is append-only, so the value a removed computation
/// defined cannot leave the module; copying the undefined value keeps
/// the module verifiable without keeping the computation.
pub(crate) fn replace_with_copy(inst: &mut Inst, value: ValueId) {
    inst.op = Op::CopyObject { operand: value };
}

/// The integer shape of a type: its width and its signedness.
pub(crate) fn int_shape(module: &Module, ty: TypeId) -> Option<(u8, bool)> {
    match module.ty(ty) {
        Type::Int { bits, signed } => Some((*bits, *signed)),
        _ => None,
    }
}

/// The width of a floating-point type.
pub(crate) fn float_width(module: &Module, ty: TypeId) -> Option<u16> {
    match module.ty(ty) {
        Type::Float { bits } => Some(*bits),
        _ => None,
    }
}

/// The mask of an integer of `bits` width.
pub(crate) fn mask(raw: u64, bits: u8) -> u64 {
    if bits >= 64 {
        raw
    } else {
        raw & ((1u64 << bits) - 1)
    }
}

/// The sign-extended value of an integer payload.
pub(crate) fn sign_extend(raw: u64, bits: u8) -> i64 {
    if bits == 0 {
        0
    } else if bits >= 64 {
        raw as i64
    } else {
        let shift = 64 - u32::from(bits);
        ((raw as i64) << shift) >> shift
    }
}

/// The unsigned value of an integer payload.
pub(crate) fn int_payload(value: ConstValue) -> Option<u64> {
    match value {
        ConstValue::Int(raw) => Some(raw),
        _ => None,
    }
}

/// The floating-point value of a constant payload.
pub(crate) fn float_payload(value: ConstValue) -> Option<f64> {
    match value {
        ConstValue::Float32(bits) => Some(f64::from(f32::from_bits(bits))),
        ConstValue::Float64(bits) => Some(f64::from_bits(bits)),
        _ => None,
    }
}

/// The constant payload of a floating-point value at `bits` width.
pub(crate) fn float_const(value: f64, bits: u16) -> Option<ConstValue> {
    match bits {
        32 => Some(ConstValue::from_f32_bits((value as f32).to_bits())),
        64 => Some(ConstValue::from_f64_bits(value.to_bits())),
        _ => None,
    }
}

/// Looks up the constant payload of a value.
pub(crate) trait ConstSource {
    /// The payload of `value`, when the pass knows one.
    fn payload(&self, value: ValueId) -> Option<ConstValue>;
}

/// Resolves the constant payload of a value, looking through the
/// copies the optimizer leaves behind.
///
/// The resolver needs the function's body because a copy is an
/// instruction, and the arena does not record which instruction
/// defines which value.
pub(crate) struct Resolver<'a> {
    /// The module the values belong to.
    module: &'a Module,
    /// The body of the function the values belong to.
    body: &'a [BasicBlock],
    /// Where every result is defined.
    defs: &'a BTreeMap<ValueId, (usize, usize)>,
}

impl<'a> Resolver<'a> {
    /// Builds a resolver over one function.
    pub(crate) fn new(
        module: &'a Module,
        body: &'a [BasicBlock],
        defs: &'a BTreeMap<ValueId, (usize, usize)>,
    ) -> Self {
        Self { module, body, defs }
    }

    /// The instruction that defines `value`, when there is one.
    pub(crate) fn defining(&self, value: ValueId) -> Option<&'a Inst> {
        let &(block, index) = self.defs.get(&value)?;
        self.body.get(block)?.insts.get(index)
    }
}

impl ConstSource for Resolver<'_> {
    fn payload(&self, value: ValueId) -> Option<ConstValue> {
        let mut current = value;
        for _ in 0..MAX_COPY_DEPTH {
            if let Some(payload) = self.module.constant_value(current) {
                return Some(payload);
            }
            let inst = self.defining(current)?;
            match inst.op {
                Op::CopyObject { operand } => current = operand,
                _ => return None,
            }
        }
        None
    }
}

/// The constant payload of `value` when the pass knows a value that is
/// neither undefined nor null.
pub(crate) fn known_payload<S: ConstSource>(source: &S, value: ValueId) -> Option<ConstValue> {
    match source.payload(value)? {
        ConstValue::Undef | ConstValue::Null => None,
        other => Some(other),
    }
}

/// The blocks a merge instruction names: the selection target, or the
/// loop's merge and continue blocks.
pub(crate) fn merge_targets(op: &Op, out: &mut Vec<usize>) {
    match op {
        Op::SelectionMerge { target, .. } => out.push(target.index()),
        Op::LoopMerge { merge, cont, .. } => {
            out.push(merge.index());
            out.push(cont.index());
        }
        _ => {}
    }
}
