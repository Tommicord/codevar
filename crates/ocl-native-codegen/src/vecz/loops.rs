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

//! Loop vectorization: a counted loop that walks a range with a unit
//! stride is executed a vector at a time.
//!
//! The rewrite turns the loop into two loops.  The first one walks
//! the range [`VectorConfig::max_lanes`] elements at a time and holds
//! the vectorized body; the second one is the original loop and
//! handles the elements that do not fill a whole vector.
//!
//! The vector loop enters the original loop through the block that
//! leaves the vector loop, so the original loop starts at the first
//! element the vector loop did not reach.  The induction variable of
//! the original loop therefore takes its starting value from the
//! vector loop, which is what tells a vectorized loop from a fresh
//! one when the vectorizer runs a second time.

use alloc::collections::BTreeMap;
use alloc::string::ToString;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{
    BasicBlock, BinOp, BlockId, CmpOp, ConstValue, Inst, Module, Op, Type, TypeId, ValueId,
};

use super::VectorConfig;

/// The shape of a counted loop the vectorizer recognizes.
struct LoopShape {
    /// The block that holds the merge instruction and the induction
    /// variable.
    header: usize,
    /// The block that holds the comparison; the header itself when
    /// the comparison sits next to the merge instruction.
    cond: usize,
    /// The body of the loop.
    body: usize,
    /// The block the loop leaves through.
    merge: usize,
    /// The block the loop continues in.
    cont: usize,
    /// The block the loop is entered from.
    preheader: usize,
    /// The induction variable.
    iv: ValueId,
    /// The value the induction variable starts from.
    init: ValueId,
    /// The bound the induction variable is compared against.
    bound: ValueId,
    /// The comparison the loop's exit depends on.
    cmp: CmpOp,
}

/// How a value of the loop body behaves across the lanes of a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// The value is the same in every lane.
    Invariant,
    /// The value is a vector, with one element per lane.
    Vector,
    /// The value is the induction variable plus a constant.
    Offset(u64),
    /// The value depends on the induction variable in a way a lane
    /// cannot express.
    LaneDependent,
}

/// Vectorizes the counted loops of every function in `module`.
pub(super) fn vectorize_loops(module: &mut Module, config: &VectorConfig) {
    let lanes = config.max_lanes;
    if lanes < 2 {
        return;
    }
    for func in module.functions().to_vec() {
        let mut rounds = 0usize;
        loop {
            rounds += 1;
            let Some(mut blocks) = module
                .function_mut(func)
                .and_then(|f| f.body.take())
            else {
                break;
            };
            let definitions = definition_sites(&blocks);
            let shape = find_counted_loop(module, &blocks, &definitions);
            let applied = match &shape {
                Some(shape) => apply_vectorization(module, &mut blocks, shape, lanes, &definitions),
                None => false,
            };
            if let Some(function) = module.function_mut(func) {
                function.body = Some(blocks);
            }
            if !applied {
                break;
            }
            if rounds > MAX_ROUNDS {
                break;
            }
        }
    }
}

/// The number of loops one function may hold before the rewrite gives
/// up; the rewrite adds four blocks per loop, so the bound only stops
/// a runaway.
const MAX_ROUNDS: usize = 1024;

/// Maps every value a function defines to the block and the index of
/// its definition.
fn definition_sites(blocks: &[BasicBlock]) -> BTreeMap<ValueId, (usize, usize)> {
    let mut sites = BTreeMap::new();
    for (block_index, block) in blocks.iter().enumerate() {
        for (index, inst) in block.insts.iter().enumerate() {
            if let Some(result) = inst.result {
                sites.insert(result, (block_index, index));
            }
        }
    }
    sites
}

/// The predecessors of every block.
fn predecessors(blocks: &[BasicBlock]) -> Vec<Vec<usize>> {
    let mut preds: Vec<Vec<usize>> = Vec::with_capacity(blocks.len());
    preds.resize_with(blocks.len(), Vec::new);
    for (index, block) in blocks.iter().enumerate() {
        let Some(last) = block.insts.last() else {
            continue;
        };
        match &last.op {
            Op::Branch { target } => {
                if let Some(list) = preds.get_mut(target.index()) {
                    list.push(index);
                }
            }
            Op::BranchConditional { then, other, .. } => {
                if let Some(list) = preds.get_mut(then.index()) {
                    list.push(index);
                }
                if let Some(list) = preds.get_mut(other.index()) {
                    list.push(index);
                }
            }
            _ => {}
        }
    }
    preds
}

/// Finds the first counted loop of the function.
fn find_counted_loop(
    module: &Module,
    blocks: &[BasicBlock],
    definitions: &BTreeMap<ValueId, (usize, usize)>,
) -> Option<LoopShape> {
    let preds = predecessors(blocks);
    for (header, block) in blocks.iter().enumerate() {
        if block.insts.len() < 3 {
            continue;
        }
        let last = block.insts.len() - 1;
        let Op::LoopMerge { merge, cont, .. } = &block.insts[last - 1].op else {
            continue;
        };
        let merge = merge.index();
        let cont = cont.index();
        if merge >= blocks.len() || cont >= blocks.len() || cont == header {
            continue;
        }
        let Op::Phi { incomings } = &block.insts[0].op else {
            continue;
        };
        if incomings.len() != 2 {
            continue;
        }
        let Some(iv) = block.insts[0].result else {
            continue;
        };
        let Some(preheader) = unique_predecessor(&preds[header], cont) else {
            continue;
        };
        let mut init = None;
        let mut carries = false;
        for (value, pred) in incomings {
            if pred.index() == preheader {
                init = Some(*value);
            } else if pred.index() == cont {
                carries = true;
            }
        }
        let Some(init) = init else {
            continue;
        };
        if !carries {
            continue;
        }
        // The comparison and the body: either the header holds the
        // comparison itself, or it branches to a block that does.
        let (cond, index, body) = match &block.insts[last].op {
            Op::BranchConditional {
                cond: value,
                then,
                other,
            } => {
                if other.index() != merge {
                    continue;
                }
                let Some(index) = comparison_index(block, *value, last) else {
                    continue;
                };
                (header, index, then.index())
            }
            Op::Branch { target } => {
                let cond_block = target.index();
                if cond_block >= blocks.len() {
                    continue;
                }
                let candidate = &blocks[cond_block];
                let Some(Op::BranchConditional {
                    cond: value,
                    then,
                    other,
                }) = candidate.insts.last().map(|inst| &inst.op)
                else {
                    continue;
                };
                if other.index() != merge {
                    continue;
                }
                let end = candidate.insts.len().saturating_sub(1);
                let Some(index) = comparison_index(candidate, *value, end) else {
                    continue;
                };
                (cond_block, index, then.index())
            }
            _ => continue,
        };
        if body >= blocks.len() || body == header || body == cont || body == merge {
            continue;
        }
        // The comparison must walk the induction variable against the
        // bound.
        let Some(cmp) = comparison_op(&blocks[cond], index) else {
            continue;
        };
        if cmp != CmpOp::SLessThan && cmp != CmpOp::ULessThan {
            continue;
        }
        if comparison_lhs(&blocks[cond], index) != Some(iv) {
            continue;
        }
        let Some(bound) = comparison_rhs(&blocks[cond], index) else {
            continue;
        };
        // The body must be a single block that falls into the continue
        // block and keeps nothing alive across iterations.
        if !body_is_vectorizable(&blocks[body], cont) {
            continue;
        }
        // The increment must step the induction variable by one.
        if !steps_by_one(module, &blocks[cont], iv) {
            continue;
        }
        // The induction variable must be an integer.
        if !matches!(module.ty(module.type_of(iv)), Type::Int { .. }) {
            continue;
        }
        // The induction variable must start from a value the
        // preheader computes, which is what tells a fresh loop from
        // the remainder of an already vectorized one.
        if !starts_in_preheader(module, init, preheader, definitions) {
            continue;
        }
        // The bound must be available outside the loop.
        let loop_blocks = [header, cond, body, cont, merge];
        if !outside_loop(bound, definitions, &loop_blocks) {
            continue;
        }
        return Some(LoopShape {
            header,
            cond,
            body,
            merge,
            cont,
            preheader,
            iv,
            init,
            bound,
            cmp,
        });
    }
    None
}

/// The index of the comparison that produced `cond`.
fn comparison_index(block: &BasicBlock, cond: ValueId, end: usize) -> Option<usize> {
    block.insts[..end]
        .iter()
        .position(|inst| inst.result == Some(cond))
}

/// The opcode of the comparison at `index`.
fn comparison_op(block: &BasicBlock, index: usize) -> Option<CmpOp> {
    match block.insts.get(index)?.op {
        Op::Compare { op, .. } => Some(op),
        _ => None,
    }
}

/// The left operand of the comparison at `index`.
fn comparison_lhs(block: &BasicBlock, index: usize) -> Option<ValueId> {
    match block.insts.get(index)?.op {
        Op::Compare { lhs, .. } => Some(lhs),
        _ => None,
    }
}

/// The right operand of the comparison at `index`.
fn comparison_rhs(block: &BasicBlock, index: usize) -> Option<ValueId> {
    match block.insts.get(index)?.op {
        Op::Compare { rhs, .. } => Some(rhs),
        _ => None,
    }
}

/// The predecessor of `block` that is not `other`.
fn unique_predecessor(preds: &[usize], other: usize) -> Option<usize> {
    let mut found = None;
    for &pred in preds {
        if pred == other {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(pred);
    }
    found
}

/// True when the body is a single block that falls into the continue
/// block and keeps nothing alive across iterations.
fn body_is_vectorizable(block: &BasicBlock, cont: usize) -> bool {
    let Some(last) = block.insts.last() else {
        return false;
    };
    if !matches!(last.op, Op::Branch { target } if target.index() == cont) {
        return false;
    }
    for inst in block.insts[..block.insts.len() - 1].iter() {
        if inst.op.is_terminator() {
            return false;
        }
        if matches!(
            inst.op,
            Op::Phi { .. } | Op::SelectionMerge { .. } | Op::LoopMerge { .. }
        ) {
            return false;
        }
    }
    true
}

/// True when the block steps the induction variable by one.
fn steps_by_one(module: &Module, block: &BasicBlock, iv: ValueId) -> bool {
    block.insts.iter().any(|inst| {
        matches!(
            &inst.op,
            Op::Binary { op: BinOp::IAdd, lhs, rhs }
                if *lhs == iv && module.constant_value(*rhs) == Some(ConstValue::Int(1))
        )
    })
}

/// True when the induction variable starts from a value the preheader
/// computes.
fn starts_in_preheader(
    module: &Module,
    init: ValueId,
    preheader: usize,
    definitions: &BTreeMap<ValueId, (usize, usize)>,
) -> bool {
    if module.constant_value(init).is_some() {
        return true;
    }
    matches!(definitions.get(&init), Some((block, _)) if *block == preheader)
}

/// True when the value is defined outside the loop.
fn outside_loop(
    value: ValueId,
    definitions: &BTreeMap<ValueId, (usize, usize)>,
    loop_blocks: &[usize],
) -> bool {
    match definitions.get(&value) {
        None => true,
        Some((block, _)) => !loop_blocks.contains(block),
    }
}

/// Rewrites the counted loop into a vector loop and the scalar
/// remainder loop.
///
/// Returns false when the body cannot be vectorized, which leaves the
/// function untouched.
fn apply_vectorization(
    module: &mut Module,
    blocks: &mut Vec<BasicBlock>,
    shape: &LoopShape,
    lanes: u32,
    definitions: &BTreeMap<ValueId, (usize, usize)>,
) -> bool {
    // The values the rewrite creates are given back when the body
    // cannot be vectorized, so that the module keeps no value that no
    // instruction defines.
    let mark = module.watermark();
    let iv_ty = module.type_of(shape.iv);
    let Ok(one) = module.intern_const(iv_ty, ConstValue::Int(1)) else {
        module.truncate_values(mark);
        return false;
    };
    let Ok(lane_count) = module.intern_const(iv_ty, ConstValue::Int(u64::from(lanes))) else {
        module.truncate_values(mark);
        return false;
    };

    // The vector loop's induction variable.  The body needs it, the
    // rest of the rewrite does not.
    let vi = module.new_inst_value(iv_ty);

    let body_len = blocks[shape.body].insts.len().saturating_sub(1);
    let body: Vec<Inst> = blocks[shape.body].insts[..body_len].to_vec();
    let loop_blocks = [shape.header, shape.cond, shape.body, shape.cont, shape.merge];
    let plan = VectorBody { vi, lanes, one };
    let Some(body_insts) = build_vector_body(module, &body, shape, &plan, definitions, &loop_blocks) else {
        module.truncate_values(mark);
        return false;
    };

    // The vector loop's induction variable and its guard: the range
    // must hold a whole vector, and the loop must not run past the
    // bound.
    let bool_ty = module.bool_ty();
    let vi_next = module.new_inst_value(iv_ty);
    let in_range = module.new_inst_value(bool_ty);
    let remaining = module.new_inst_value(iv_ty);
    let wide_enough = module.new_inst_value(bool_ty);
    let guard = module.new_inst_value(bool_ty);

    let base = blocks.len();
    let vhdr = BlockId(u32::try_from(base).unwrap_or(u32::MAX));
    let vbody = BlockId(u32::try_from(base + 1).unwrap_or(u32::MAX));
    let vcont = BlockId(u32::try_from(base + 2).unwrap_or(u32::MAX));
    let vexit = BlockId(u32::try_from(base + 3).unwrap_or(u32::MAX));
    let header = BlockId(u32::try_from(shape.header).unwrap_or(u32::MAX));
    let preheader = BlockId(u32::try_from(shape.preheader).unwrap_or(u32::MAX));

    let header_insts: Vec<Inst> = alloc::vec![
        Inst::def(
            vi,
            iv_ty,
            Op::Phi {
                incomings: alloc::vec![(shape.init, preheader), (vi_next, vcont)],
            },
        ),
        Inst::def(
            in_range,
            bool_ty,
            Op::Compare {
                op: shape.cmp,
                lhs: vi,
                rhs: shape.bound,
            },
        ),
        Inst::def(
            remaining,
            iv_ty,
            Op::Binary {
                op: BinOp::ISub,
                lhs: shape.bound,
                rhs: vi,
            },
        ),
        Inst::def(
            wide_enough,
            bool_ty,
            Op::Compare {
                op: greater_or_equal(shape.cmp),
                lhs: remaining,
                rhs: lane_count,
            },
        ),
        Inst::def(
            guard,
            bool_ty,
            Op::Binary {
                op: BinOp::LogicalAnd,
                lhs: in_range,
                rhs: wide_enough,
            },
        ),
        Inst::none(Op::LoopMerge {
            merge: vexit,
            cont: vcont,
            control: 0,
        }),
        Inst::none(Op::BranchConditional {
            cond: guard,
            then: vbody,
            other: vexit,
        }),
    ];

    let mut body_block = body_insts;
    body_block.push(Inst::none(Op::Branch { target: vcont }));

    let cont_block: Vec<Inst> = alloc::vec![
        Inst::def(
            vi_next,
            iv_ty,
            Op::Binary {
                op: BinOp::IAdd,
                lhs: vi,
                rhs: lane_count,
            },
        ),
        Inst::none(Op::Branch { target: vhdr }),
    ];

    let exit_block: Vec<Inst> = alloc::vec![Inst::none(Op::Branch { target: header })];

    blocks.push(BasicBlock {
        name: "loop.vector.header".to_string(),
        insts: header_insts,
    });
    blocks.push(BasicBlock {
        name: "loop.vector.body".to_string(),
        insts: body_block,
    });
    blocks.push(BasicBlock {
        name: "loop.vector.cont".to_string(),
        insts: cont_block,
    });
    blocks.push(BasicBlock {
        name: "loop.vector.exit".to_string(),
        insts: exit_block,
    });

    // The loop is now entered through the vector loop.
    if let Some(inst) = blocks[shape.preheader].insts.last_mut()
        && let Op::Branch { target } = &mut inst.op
        && target.index() == shape.header
    {
        *target = vhdr;
    }
    // The original loop starts at the first element the vector loop
    // did not reach.
    if let Some(inst) = blocks[shape.header].insts.first_mut()
        && let Op::Phi { incomings } = &mut inst.op
    {
        for (value, pred) in incomings.iter_mut() {
            if pred.index() == shape.preheader {
                *value = vi;
                *pred = vexit;
            }
        }
    }
    true
}

/// The comparison that holds when the range is wide enough.
fn greater_or_equal(cmp: CmpOp) -> CmpOp {
    match cmp {
        CmpOp::SLessThan => CmpOp::SGreaterThanEqual,
        _ => CmpOp::UGreaterThanEqual,
    }
}

/// The parameters of the vector body.
struct VectorBody {
    /// The induction variable of the vector loop.
    vi: ValueId,
    /// The number of lanes.
    lanes: u32,
    /// The constant one, of the induction variable's type.
    one: ValueId,
}

/// Builds the instructions of the vector loop's body.
///
/// The body is cloned with vector-typed values: the loads of the
/// original body become one load per lane, the operations between
/// them become one vector operation, and the stores become one store
/// per lane.
fn build_vector_body(
    module: &mut Module,
    body: &[Inst],
    shape: &LoopShape,
    plan: &VectorBody,
    definitions: &BTreeMap<ValueId, (usize, usize)>,
    loop_blocks: &[usize],
) -> Option<Vec<Inst>> {
    let mut builder = BodyBuilder {
        module,
        shape,
        vi: plan.vi,
        lanes: plan.lanes,
        one: plan.one,
        definitions,
        loop_blocks,
        out: Vec::with_capacity(body.len() * 2),
        map: BTreeMap::new(),
        kind: BTreeMap::new(),
        lane_pointers: BTreeMap::new(),
        chains: BTreeMap::new(),
        gathers: 0,
    };
    builder.build(body)
}

/// The state of the body rewrite.
struct BodyBuilder<'a> {
    /// The module the new values belong to.
    module: &'a mut Module,
    /// The loop the body belongs to.
    shape: &'a LoopShape,
    /// The induction variable of the vector loop.
    vi: ValueId,
    /// The number of lanes.
    lanes: u32,
    /// The constant one, of the induction variable's type.
    one: ValueId,
    /// Where every value of the function is defined.
    definitions: &'a BTreeMap<ValueId, (usize, usize)>,
    /// The blocks of the loop.
    loop_blocks: &'a [usize],
    /// The instructions of the vector body.
    out: Vec<Inst>,
    /// The value each body value maps to.
    map: BTreeMap<ValueId, ValueId>,
    /// How each body value behaves across the lanes.
    kind: BTreeMap<ValueId, Kind>,
    /// The pointers of a unit-stride access, one per lane.
    lane_pointers: BTreeMap<ValueId, Vec<ValueId>>,
    /// The indices of a unit-stride access, one per lane.
    chains: BTreeMap<ValueId, Vec<ValueId>>,
    /// The number of loads that were gathered.
    gathers: usize,
}

impl BodyBuilder<'_> {
    /// Rewrites the whole body.
    fn build(&mut self, body: &[Inst]) -> Option<Vec<Inst>> {
        self.map.insert(self.shape.iv, self.vi);
        self.kind.insert(self.shape.iv, Kind::Offset(0));
        for inst in body {
            self.step(inst)?;
        }
        if self.gathers == 0 {
            return None;
        }
        Some(core::mem::take(&mut self.out))
    }

    /// Rewrites one instruction of the body.
    fn step(&mut self, inst: &Inst) -> Option<()> {
        match &inst.op {
            Op::AccessChain { .. } | Op::PtrAccessChain { .. } => self.access(inst),
            Op::Load { .. } => self.load(inst),
            Op::Store { .. } => self.store(inst),
            Op::Binary { .. } => self.binary(inst),
            Op::Unary { .. } => self.unary(inst),
            Op::Compare { .. } => self.compare(inst),
            Op::Convert { .. } => self.convert(inst),
            Op::CopyObject { .. } => self.copy(inst),
            _ => None,
        }
    }

    /// How a value behaves across the lanes.
    fn classify(&self, value: ValueId) -> Kind {
        self.kind
            .get(&value)
            .copied()
            .unwrap_or(Kind::Invariant)
    }

    /// The value a body value maps to.
    fn mapped(&self, value: ValueId) -> ValueId {
        self.map.get(&value).copied().unwrap_or(value)
    }

    /// True when the value is available in the vector body.
    ///
    /// A value the body defines is mapped to its copy; a value from
    /// outside the loop is available everywhere the loop is; a value
    /// the loop defines elsewhere is not.
    fn available(&self, value: ValueId) -> bool {
        if self.map.contains_key(&value) {
            return true;
        }
        outside_loop(value, self.definitions, self.loop_blocks)
    }

    /// The offset an addition computes, when it adds a constant to
    /// the induction variable.
    fn offset_of(&self, lhs: ValueId, rhs: ValueId) -> Option<u64> {
        let left = self.kind.get(&lhs).copied();
        let right = self.kind.get(&rhs).copied();
        let (base, other) = match (left, right) {
            (Some(Kind::Offset(base)), None) => (base, rhs),
            (None, Some(Kind::Offset(base))) => (base, lhs),
            _ => return None,
        };
        let ConstValue::Int(raw) = self.module.constant_value(other)? else {
            return None;
        };
        Some(base.saturating_add(raw))
    }

    /// The indices of a unit-stride access, one per lane.
    fn chain(&mut self, index: ValueId) -> Option<Vec<ValueId>> {
        if let Some(existing) = self.chains.get(&index) {
            return Some(existing.clone());
        }
        let iv_ty = self.module.type_of(self.shape.iv);
        let mut lane_indices = Vec::with_capacity(self.lanes as usize);
        let mut current = self.mapped(index);
        lane_indices.push(current);
        for _ in 1..self.lanes {
            let next = self.module.new_inst_value(iv_ty);
            self.out.push(Inst::def(
                next,
                iv_ty,
                Op::Binary {
                    op: BinOp::IAdd,
                    lhs: current,
                    rhs: self.one,
                },
            ));
            current = next;
            lane_indices.push(current);
        }
        self.chains.insert(index, lane_indices.clone());
        Some(lane_indices)
    }

    /// Rewrites an access into the memory.
    fn access(&mut self, inst: &Inst) -> Option<()> {
        let (base, indices, is_pointer) = match &inst.op {
            Op::AccessChain { base, indices } => (*base, indices, false),
            Op::PtrAccessChain { base, indices } => (*base, indices, true),
            _ => return None,
        };
        if !self.available(base) || self.classify(base) != Kind::Invariant {
            return None;
        }
        let pointer_ty = inst.ty?;
        let first = *indices.first()?;
        let kind = self.classify(first);
        if matches!(kind, Kind::Offset(_)) {
            let lane_indices = self.chain(first)?;
            // The indices that walk into the aggregate must be the
            // same in every lane.
            for rest in &indices[1..] {
                if self.classify(*rest) != Kind::Invariant {
                    return None;
                }
            }
            let mut pointers = Vec::with_capacity(self.lanes as usize);
            for &lane_index in lane_indices.iter() {
                let mut new_indices = Vec::with_capacity(indices.len());
                new_indices.push(lane_index);
                for rest in &indices[1..] {
                    new_indices.push(self.mapped(*rest));
                }
                let pointer = self.module.new_inst_value(pointer_ty);
                let op = if is_pointer {
                    Op::PtrAccessChain {
                        base: self.mapped(base),
                        indices: new_indices,
                    }
                } else {
                    Op::AccessChain {
                        base: self.mapped(base),
                        indices: new_indices,
                    }
                };
                self.out.push(Inst::def(pointer, pointer_ty, op));
                pointers.push(pointer);
            }
            let result = inst.result?;
            self.lane_pointers
                .insert(result, pointers.clone());
            self.map.insert(result, pointers[0]);
            self.kind.insert(result, Kind::Invariant);
        } else if kind == Kind::Invariant {
            let mut new_indices = Vec::with_capacity(indices.len());
            for index in indices {
                if self.classify(*index) != Kind::Invariant {
                    return None;
                }
                new_indices.push(self.mapped(*index));
            }
            let pointer = self.module.new_inst_value(pointer_ty);
            let op = if is_pointer {
                Op::PtrAccessChain {
                    base: self.mapped(base),
                    indices: new_indices,
                }
            } else {
                Op::AccessChain {
                    base: self.mapped(base),
                    indices: new_indices,
                }
            };
            self.out.push(Inst::def(pointer, pointer_ty, op));
            let result = inst.result?;
            self.map.insert(result, pointer);
            self.kind.insert(result, Kind::Invariant);
        } else {
            return None;
        }
        Some(())
    }

    /// Rewrites a load: a load of a unit-stride access gathers one
    /// element per lane.
    fn load(&mut self, inst: &Inst) -> Option<()> {
        let Op::Load { ptr } = &inst.op else {
            return None;
        };
        let element_ty = inst.ty?;
        if matches!(self.module.ty(element_ty), Type::Vector { .. }) {
            // The memory already holds vectors: there is nothing to
            // widen.
            return None;
        }
        if let Some(pointers) = self.lane_pointers.get(ptr).cloned() {
            let vector_ty = self.module.vector_ty(element_ty, self.lanes);
            let mut lane_values = Vec::with_capacity(self.lanes as usize);
            for pointer in &pointers {
                let value = self.module.new_inst_value(element_ty);
                self.out
                    .push(Inst::def(value, element_ty, Op::Load { ptr: *pointer }));
                lane_values.push(value);
            }
            let vector = self.module.new_inst_value(vector_ty);
            self.out.push(Inst::def(
                vector,
                vector_ty,
                Op::CompositeConstruct {
                    constituents: lane_values,
                },
            ));
            let result = inst.result?;
            self.map.insert(result, vector);
            self.kind.insert(result, Kind::Vector);
            self.gathers += 1;
        } else {
            if !self.available(*ptr) {
                return None;
            }
            let value = self.module.new_inst_value(element_ty);
            self.out.push(Inst::def(
                value,
                element_ty,
                Op::Load {
                    ptr: self.mapped(*ptr),
                },
            ));
            let result = inst.result?;
            self.map.insert(result, value);
            self.kind.insert(result, Kind::Invariant);
        }
        Some(())
    }

    /// Rewrites a store: a store to a unit-stride access scatters one
    /// element per lane.
    fn store(&mut self, inst: &Inst) -> Option<()> {
        let Op::Store { ptr, value } = &inst.op else {
            return None;
        };
        let Some(pointers) = self.lane_pointers.get(ptr).cloned() else {
            // A store outside the unit-stride access would run a
            // different number of times.
            return None;
        };
        match self.classify(*value) {
            Kind::Vector => {
                let vector = self.mapped(*value);
                let Type::Vector { elem, .. } = self.module.ty(self.module.type_of(vector)) else {
                    return None;
                };
                let element = *elem;
                for (lane, pointer) in pointers.iter().enumerate() {
                    let element_value = self.module.new_inst_value(element);
                    self.out.push(Inst::def(
                        element_value,
                        element,
                        Op::CompositeExtract {
                            composite: vector,
                            indices: alloc::vec![u32::try_from(lane).unwrap_or(0)],
                        },
                    ));
                    self.out.push(Inst::none(Op::Store {
                        ptr: *pointer,
                        value: element_value,
                    }));
                }
            }
            Kind::Invariant => {
                let scalar = self.mapped(*value);
                for pointer in &pointers {
                    self.out.push(Inst::none(Op::Store {
                        ptr: *pointer,
                        value: scalar,
                    }));
                }
            }
            _ => return None,
        }
        Some(())
    }

    /// The element type of a vector operand.
    fn element_ty(&self, left: ValueId, right: ValueId) -> Option<TypeId> {
        let left_ty = self.module.type_of(self.mapped(left));
        if let Type::Vector { elem, .. } = self.module.ty(left_ty) {
            return Some(*elem);
        }
        let right_ty = self.module.type_of(self.mapped(right));
        if let Type::Vector { elem, .. } = self.module.ty(right_ty) {
            return Some(*elem);
        }
        None
    }

    /// The vector form of an operand: the vector itself, or a splat of
    /// a value that is the same in every lane.
    fn operand_vector(&mut self, value: ValueId, kind: Kind, vector_ty: TypeId) -> Option<ValueId> {
        match kind {
            Kind::Vector => Some(self.mapped(value)),
            Kind::Invariant => {
                let Type::Vector { elem, .. } = self.module.ty(vector_ty) else {
                    return None;
                };
                if self.module.type_of(self.mapped(value)) != *elem {
                    return None;
                }
                let mut constituents = Vec::with_capacity(self.lanes as usize);
                for _ in 0..self.lanes {
                    constituents.push(self.mapped(value));
                }
                let result = self.module.new_inst_value(vector_ty);
                self.out.push(Inst::def(
                    result,
                    vector_ty,
                    Op::CompositeConstruct { constituents },
                ));
                Some(result)
            }
            _ => None,
        }
    }

    /// Rewrites a binary operation.
    fn binary(&mut self, inst: &Inst) -> Option<()> {
        let Op::Binary { op, lhs, rhs } = &inst.op else {
            return None;
        };
        let result_ty = inst.ty?;
        // An addition of a constant to the induction variable keeps
        // the offset, and can still index the memory.
        if *op == BinOp::IAdd
            && let Some(offset) = self.offset_of(*lhs, *rhs)
        {
            let value = self.module.new_inst_value(result_ty);
            self.out.push(Inst::def(
                value,
                result_ty,
                Op::Binary {
                    op: *op,
                    lhs: self.mapped(*lhs),
                    rhs: self.mapped(*rhs),
                },
            ));
            let result = inst.result?;
            self.map.insert(result, value);
            self.kind.insert(result, Kind::Offset(offset));
            return Some(());
        }
        let left = self.classify(*lhs);
        let right = self.classify(*rhs);
        if left == Kind::Vector || right == Kind::Vector {
            let vector_ty = self
                .module
                .vector_ty(self.element_ty(*lhs, *rhs)?, self.lanes);
            let left_vector = self.operand_vector(*lhs, left, vector_ty)?;
            let right_vector = self.operand_vector(*rhs, right, vector_ty)?;
            let value = self.module.new_inst_value(vector_ty);
            self.out.push(Inst::def(
                value,
                vector_ty,
                Op::Binary {
                    op: *op,
                    lhs: left_vector,
                    rhs: right_vector,
                },
            ));
            let result = inst.result?;
            self.map.insert(result, value);
            self.kind.insert(result, Kind::Vector);
            return Some(());
        }
        let kind = if matches!(left, Kind::Offset(_) | Kind::LaneDependent)
            || matches!(right, Kind::Offset(_) | Kind::LaneDependent)
        {
            Kind::LaneDependent
        } else {
            Kind::Invariant
        };
        let value = self.module.new_inst_value(result_ty);
        self.out.push(Inst::def(
            value,
            result_ty,
            Op::Binary {
                op: *op,
                lhs: self.mapped(*lhs),
                rhs: self.mapped(*rhs),
            },
        ));
        let result = inst.result?;
        self.map.insert(result, value);
        self.kind.insert(result, kind);
        Some(())
    }

    /// Rewrites a unary operation.
    fn unary(&mut self, inst: &Inst) -> Option<()> {
        let Op::Unary { op, operand } = &inst.op else {
            return None;
        };
        let result_ty = inst.ty?;
        let kind = self.classify(*operand);
        if kind == Kind::Vector {
            let vector_ty = self
                .module
                .vector_ty(self.element_ty(*operand, *operand)?, self.lanes);
            let operand_vector = self.operand_vector(*operand, kind, vector_ty)?;
            let value = self.module.new_inst_value(vector_ty);
            self.out.push(Inst::def(
                value,
                vector_ty,
                Op::Unary {
                    op: *op,
                    operand: operand_vector,
                },
            ));
            let result = inst.result?;
            self.map.insert(result, value);
            self.kind.insert(result, Kind::Vector);
            return Some(());
        }
        if matches!(kind, Kind::Offset(_) | Kind::LaneDependent) {
            return None;
        }
        let value = self.module.new_inst_value(result_ty);
        self.out.push(Inst::def(
            value,
            result_ty,
            Op::Unary {
                op: *op,
                operand: self.mapped(*operand),
            },
        ));
        let result = inst.result?;
        self.map.insert(result, value);
        self.kind.insert(result, Kind::Invariant);
        Some(())
    }

    /// Rewrites a comparison.
    fn compare(&mut self, inst: &Inst) -> Option<()> {
        let Op::Compare { op, lhs, rhs } = &inst.op else {
            return None;
        };
        let left = self.classify(*lhs);
        let right = self.classify(*rhs);
        if left == Kind::Vector || right == Kind::Vector {
            let vector_ty = self
                .module
                .vector_ty(self.element_ty(*lhs, *rhs)?, self.lanes);
            let left_vector = self.operand_vector(*lhs, left, vector_ty)?;
            let right_vector = self.operand_vector(*rhs, right, vector_ty)?;
            let bool_ty = self.module.bool_ty();
            let result_ty = self.module.vector_ty(bool_ty, self.lanes);
            let value = self.module.new_inst_value(result_ty);
            self.out.push(Inst::def(
                value,
                result_ty,
                Op::Compare {
                    op: *op,
                    lhs: left_vector,
                    rhs: right_vector,
                },
            ));
            let result = inst.result?;
            self.map.insert(result, value);
            self.kind.insert(result, Kind::Vector);
            return Some(());
        }
        if left != Kind::Invariant || right != Kind::Invariant {
            return None;
        }
        let bool_ty = self.module.bool_ty();
        let value = self.module.new_inst_value(bool_ty);
        self.out.push(Inst::def(
            value,
            bool_ty,
            Op::Compare {
                op: *op,
                lhs: self.mapped(*lhs),
                rhs: self.mapped(*rhs),
            },
        ));
        let result = inst.result?;
        self.map.insert(result, value);
        self.kind.insert(result, Kind::Invariant);
        Some(())
    }

    /// Rewrites a conversion.
    fn convert(&mut self, inst: &Inst) -> Option<()> {
        let Op::Convert { op, operand } = &inst.op else {
            return None;
        };
        let result_ty = inst.ty?;
        let kind = self.classify(*operand);
        if kind == Kind::Vector {
            let operand_ty = self.module.type_of(self.mapped(*operand));
            let Type::Vector { elem, .. } = self.module.ty(operand_ty) else {
                return None;
            };
            let operand_vector_ty = self.module.vector_ty(*elem, self.lanes);
            let operand_vector = self.operand_vector(*operand, kind, operand_vector_ty)?;
            let vector_result_ty = self.module.vector_ty(result_ty, self.lanes);
            let value = self.module.new_inst_value(vector_result_ty);
            self.out.push(Inst::def(
                value,
                vector_result_ty,
                Op::Convert {
                    op: *op,
                    operand: operand_vector,
                },
            ));
            let result = inst.result?;
            self.map.insert(result, value);
            self.kind.insert(result, Kind::Vector);
            return Some(());
        }
        if kind != Kind::Invariant {
            return None;
        }
        let value = self.module.new_inst_value(result_ty);
        self.out.push(Inst::def(
            value,
            result_ty,
            Op::Convert {
                op: *op,
                operand: self.mapped(*operand),
            },
        ));
        let result = inst.result?;
        self.map.insert(result, value);
        self.kind.insert(result, Kind::Invariant);
        Some(())
    }

    /// Rewrites a copy.
    fn copy(&mut self, inst: &Inst) -> Option<()> {
        let Op::CopyObject { operand } = &inst.op else {
            return None;
        };
        let result_ty = inst.ty?;
        let kind = self.classify(*operand);
        let value = self.module.new_inst_value(result_ty);
        self.out.push(Inst::def(
            value,
            result_ty,
            Op::CopyObject {
                operand: self.mapped(*operand),
            },
        ));
        let result = inst.result?;
        self.map.insert(result, value);
        self.kind.insert(result, kind);
        Some(())
    }
}
