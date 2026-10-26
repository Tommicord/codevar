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

//! Peephole rewriting.
//!
//! The pass rewrites the operations whose simplest equivalent form is
//! shorter or cheaper:
//!
//! * `x + 0`, `x - 0`, `x * 1`, `x & x`, `x | x` and a negation of a
//!   negation collapse to the operand they already have;
//! * `x * 0`, `x ^ x` and `x & ~x` collapse to zero, and `x | ~x`
//!   collapses to the all-ones value of the type;
//! * a select on a known condition takes the arm the condition names;
//! * a comparison whose operands are known folds to a constant;
//! * a copy of a copy collapses to a copy of the original.
//!
//! The arithmetic rules apply to integers only: for a floating-point
//! operand `x + 0.0` is not `x` when `x` is a negative zero, and the
//! pass refuses to make that mistake.

use alloc::vec::Vec;

use codevar_ocl_ir::ir::{BinOp, ConstValue, Module, Op, TypeId, UnOp, ValueId};

use crate::error::Result;
use crate::opt::passes::fold::evaluate;
use crate::opt::util::{Resolver, definitions, int_shape, known_payload, mask};

/// What an instruction becomes.
enum Replacement {
    /// The instruction becomes a copy of an existing value.
    Value(ValueId),
    /// The instruction becomes a copy of a constant.
    Constant(TypeId, ConstValue),
}

/// Rewrites the operations of every function.
pub(crate) fn run(module: &mut Module) -> Result<()> {
    let functions: Vec<ValueId> = module.functions().to_vec();
    for func in functions {
        combine_function(module, func)?;
    }
    Ok(())
}

/// Rewrites the operations of one function.
fn combine_function(module: &mut Module, func: ValueId) -> Result<()> {
    let Some(body) = module
        .function(func)
        .and_then(|f| f.body.as_ref())
    else {
        return Ok(());
    };
    let defs = definitions(body);
    let resolver = Resolver::new(module, body, &defs);

    let mut rewrites: Vec<(usize, usize, Replacement)> = Vec::new();
    for (block_index, block) in body.iter().enumerate() {
        for (inst_index, inst) in block.insts.iter().enumerate() {
            let Some(ty) = inst.ty else {
                continue;
            };
            if let Some(replacement) = simplify(module, &resolver, &inst.op, ty) {
                rewrites.push((block_index, inst_index, replacement));
            }
        }
    }
    if rewrites.is_empty() {
        return Ok(());
    }

    // The constants the rewrites need.
    let mut ready: Vec<(usize, usize, ValueId)> = Vec::with_capacity(rewrites.len());
    for (block_index, inst_index, replacement) in rewrites {
        let value = match replacement {
            Replacement::Value(value) => value,
            Replacement::Constant(ty, constant) => {
                let Ok(value) = module.intern_const(ty, constant) else {
                    continue;
                };
                value
            }
        };
        ready.push((block_index, inst_index, value));
    }

    let Some(function) = module.function_mut(func) else {
        return Ok(());
    };
    let Some(body) = function.body.as_mut() else {
        return Ok(());
    };
    for (block_index, inst_index, value) in ready {
        let Some(block) = body.get_mut(block_index) else {
            continue;
        };
        let Some(inst) = block.insts.get_mut(inst_index) else {
            continue;
        };
        inst.op = Op::CopyObject { operand: value };
    }
    Ok(())
}

/// The simpler form of an operation, when there is one.
fn simplify(module: &Module, resolver: &Resolver<'_>, op: &Op, result_ty: TypeId) -> Option<Replacement> {
    match op {
        Op::Binary { op, lhs, rhs } => simplify_binary(module, resolver, *op, *lhs, *rhs, result_ty),
        Op::Unary { op, operand } => simplify_unary(resolver, *op, *operand),
        Op::Select { cond, a, b } => {
            let condition = known_payload(resolver, *cond)?;
            match condition {
                ConstValue::Bool(true) => Some(Replacement::Value(*a)),
                ConstValue::Bool(false) => Some(Replacement::Value(*b)),
                _ => None,
            }
        }
        Op::Compare { .. } => {
            let value = evaluate(module, resolver, op, result_ty)?;
            Some(Replacement::Constant(result_ty, value))
        }
        Op::CopyObject { operand } => {
            // A copy of a copy is a copy of the original.
            let inner = resolver.defining(*operand)?;
            let Op::CopyObject { operand } = inner.op else {
                return None;
            };
            Some(Replacement::Value(operand))
        }
        _ => None,
    }
}

/// The simpler form of a binary operation.
fn simplify_binary(
    module: &Module,
    resolver: &Resolver<'_>,
    op: BinOp,
    lhs: ValueId,
    rhs: ValueId,
    result_ty: TypeId,
) -> Option<Replacement> {
    // The rules that hold for a floating-point operand too.
    match op {
        BinOp::LogicalAnd => {
            if let Some(ConstValue::Bool(true)) = known_payload(resolver, rhs) {
                return Some(Replacement::Value(lhs));
            }
            if let Some(ConstValue::Bool(false)) = known_payload(resolver, rhs) {
                return Some(Replacement::Constant(result_ty, ConstValue::Bool(false)));
            }
            if let Some(ConstValue::Bool(true)) = known_payload(resolver, lhs) {
                return Some(Replacement::Value(rhs));
            }
            if let Some(ConstValue::Bool(false)) = known_payload(resolver, lhs) {
                return Some(Replacement::Constant(result_ty, ConstValue::Bool(false)));
            }
            return None;
        }
        BinOp::LogicalOr => {
            if let Some(ConstValue::Bool(true)) = known_payload(resolver, rhs) {
                return Some(Replacement::Constant(result_ty, ConstValue::Bool(true)));
            }
            if let Some(ConstValue::Bool(false)) = known_payload(resolver, rhs) {
                return Some(Replacement::Value(lhs));
            }
            if let Some(ConstValue::Bool(true)) = known_payload(resolver, lhs) {
                return Some(Replacement::Constant(result_ty, ConstValue::Bool(true)));
            }
            if let Some(ConstValue::Bool(false)) = known_payload(resolver, lhs) {
                return Some(Replacement::Value(rhs));
            }
            return None;
        }
        _ => {}
    }

    // The arithmetic rules need an integer type.
    let (bits, _) = int_shape(module, result_ty)?;

    let left = known_payload(resolver, lhs);
    let right = known_payload(resolver, rhs);
    let left_zero = matches!(left, Some(ConstValue::Int(0)));
    let right_zero = matches!(right, Some(ConstValue::Int(0)));
    let left_one = matches!(left, Some(ConstValue::Int(1)));
    let right_one = matches!(right, Some(ConstValue::Int(1)));

    match op {
        BinOp::IAdd => {
            if right_zero {
                return Some(Replacement::Value(lhs));
            }
            if left_zero {
                return Some(Replacement::Value(rhs));
            }
            None
        }
        BinOp::ISub => {
            if right_zero {
                return Some(Replacement::Value(lhs));
            }
            if lhs == rhs {
                return Some(Replacement::Constant(result_ty, ConstValue::Int(0)));
            }
            None
        }
        BinOp::IMul => {
            if right_one {
                return Some(Replacement::Value(lhs));
            }
            if left_one {
                return Some(Replacement::Value(rhs));
            }
            if right_zero || left_zero {
                return Some(Replacement::Constant(result_ty, ConstValue::Int(0)));
            }
            None
        }
        BinOp::BitwiseAnd => {
            if lhs == rhs {
                return Some(Replacement::Value(lhs));
            }
            if is_complement(resolver, rhs, lhs) || is_complement(resolver, lhs, rhs) {
                return Some(Replacement::Constant(result_ty, ConstValue::Int(0)));
            }
            None
        }
        BinOp::BitwiseOr => {
            if lhs == rhs {
                return Some(Replacement::Value(lhs));
            }
            if is_complement(resolver, rhs, lhs) || is_complement(resolver, lhs, rhs) {
                return Some(Replacement::Constant(
                    result_ty,
                    ConstValue::Int(mask(u64::MAX, bits)),
                ));
            }
            None
        }
        BinOp::BitwiseXor => {
            if lhs == rhs {
                return Some(Replacement::Constant(result_ty, ConstValue::Int(0)));
            }
            None
        }
        _ => None,
    }
}

/// The simpler form of a unary operation.
fn simplify_unary(resolver: &Resolver<'_>, op: UnOp, operand: ValueId) -> Option<Replacement> {
    let inner = resolver.defining(operand)?;
    let Op::Unary {
        op: inner_op,
        operand: inner_operand,
    } = inner.op
    else {
        return None;
    };
    let repeats = matches!(
        (op, inner_op),
        (UnOp::Not, UnOp::Not)
            | (UnOp::SNegate, UnOp::SNegate)
            | (UnOp::FNegate, UnOp::FNegate)
            | (UnOp::LogicalNot, UnOp::LogicalNot)
    );
    if repeats {
        Some(Replacement::Value(inner_operand))
    } else {
        None
    }
}

/// True when `value` is the complement of `other`.
fn is_complement(resolver: &Resolver<'_>, value: ValueId, other: ValueId) -> bool {
    let Some(inst) = resolver.defining(value) else {
        return false;
    };
    let Op::Unary {
        op: UnOp::Not,
        operand,
    } = inst.op
    else {
        return false;
    };
    operand == other
}
