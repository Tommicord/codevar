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

//! Constant evaluation.
//!
//! The evaluator computes the value of an operation whose operands the
//! caller already knows.  It is deliberately conservative:
//!
//! * an operation whose result the intermediate representation cannot
//!   represent as a constant — a division by zero, a shift past the
//!   width of the type, a conversion of a value that is not a number —
//!   is not folded at all;
//! * the width and the signedness of the operand type decide how the
//!   arithmetic behaves, so a 32-bit addition wraps at 32 bits and a
//!   signed division truncates toward zero;
//! * floating-point arithmetic is computed in the precision of the
//!   operand type, and a NaN result is canonicalized the way the
//!   intermediate representation stores it.

use alloc::vec::Vec;

use codevar_ocl_ir::ir::{BinOp, CmpOp, ConstValue, ConvOp, Module, Op, Type, TypeId, UnOp, ValueId};

use crate::error::Result;
use crate::opt::util::{
    ConstSource, Resolver, definitions, float_const, float_payload, float_width, int_payload, int_shape,
    known_payload, mask, sign_extend,
};

/// Folds the operations whose operands are all known.
pub(crate) fn run(module: &mut Module) -> Result<()> {
    let functions: Vec<ValueId> = module.functions().to_vec();
    for func in functions {
        fold_function(module, func)?;
    }
    Ok(())
}

/// Folds the operations of one function.
fn fold_function(module: &mut Module, func: ValueId) -> Result<()> {
    loop {
        // The folds the function's current shape allows.
        let Some(body) = module
            .function(func)
            .and_then(|f| f.body.as_ref())
        else {
            return Ok(());
        };
        let defs = definitions(body);
        let resolver = Resolver::new(module, body, &defs);
        let mut folds: Vec<(usize, usize, TypeId, ConstValue)> = Vec::new();
        for (block_index, block) in body.iter().enumerate() {
            for (inst_index, inst) in block.insts.iter().enumerate() {
                let (Some(_result), Some(ty)) = (inst.result, inst.ty) else {
                    continue;
                };
                if !matches!(
                    inst.op,
                    Op::Unary { .. }
                        | Op::Binary { .. }
                        | Op::Compare { .. }
                        | Op::Select { .. }
                        | Op::Convert { .. }
                ) {
                    continue;
                }
                let Some(value) = evaluate(module, &resolver, &inst.op, ty) else {
                    continue;
                };
                folds.push((block_index, inst_index, ty, value));
            }
        }
        if folds.is_empty() {
            return Ok(());
        }

        // The constants the folds need, interned before the function is
        // borrowed mutably.
        let mut constants: Vec<(usize, usize, ValueId)> = Vec::with_capacity(folds.len());
        for (block_index, inst_index, ty, value) in folds {
            let Ok(constant) = module.intern_const(ty, value) else {
                continue;
            };
            constants.push((block_index, inst_index, constant));
        }
        if constants.is_empty() {
            return Ok(());
        }

        // The rewrite itself: the operation becomes a copy of the
        // constant, so the value keeps its identity and its uses stay
        // valid.
        let Some(function) = module.function_mut(func) else {
            return Ok(());
        };
        let Some(body) = function.body.as_mut() else {
            return Ok(());
        };
        for (block_index, inst_index, constant) in constants {
            let Some(block) = body.get_mut(block_index) else {
                continue;
            };
            let Some(inst) = block.insts.get_mut(inst_index) else {
                continue;
            };
            inst.op = Op::CopyObject { operand: constant };
        }
    }
}

/// Computes the value of `op` when the caller knows every operand.
///
/// Returns `None` when the operation has no constant result the
/// intermediate representation can hold, or when an operand is not
/// known.
pub(crate) fn evaluate<S: ConstSource>(
    module: &Module,
    source: &S,
    op: &Op,
    result_ty: TypeId,
) -> Option<ConstValue> {
    match op {
        Op::Unary { op, operand } => {
            let value = known_payload(source, *operand)?;
            eval_unary(module, *op, value, module.type_of(*operand))
        }
        Op::Binary { op, lhs, rhs } => {
            let left = known_payload(source, *lhs)?;
            let right = known_payload(source, *rhs)?;
            eval_binary(module, *op, left, right, module.type_of(*lhs))
        }
        Op::Compare { op, lhs, rhs } => {
            let left = known_payload(source, *lhs)?;
            let right = known_payload(source, *rhs)?;
            eval_compare(module, *op, left, right, module.type_of(*lhs))
        }
        Op::Select { cond, a, b } => {
            let condition = known_payload(source, *cond)?;
            match condition {
                ConstValue::Bool(true) => source.payload(*a),
                ConstValue::Bool(false) => source.payload(*b),
                _ => None,
            }
        }
        Op::Convert { op, operand } => {
            let value = known_payload(source, *operand)?;
            eval_convert(module, *op, value, module.type_of(*operand), result_ty)
        }
        _ => None,
    }
}

/// Evaluates a unary operation.
fn eval_unary(module: &Module, op: UnOp, value: ConstValue, operand_ty: TypeId) -> Option<ConstValue> {
    match op {
        UnOp::Not => {
            let (bits, _) = int_shape(module, operand_ty)?;
            Some(ConstValue::Int(mask(!int_payload(value)?, bits)))
        }
        UnOp::SNegate => {
            let (bits, signed) = int_shape(module, operand_ty)?;
            if !signed {
                return None;
            }
            let raw = int_payload(value)?;
            Some(ConstValue::Int(mask(raw.wrapping_neg(), bits)))
        }
        UnOp::FNegate => {
            let bits = float_width(module, operand_ty)?;
            float_const(-float_payload(value)?, bits)
        }
        UnOp::LogicalNot => match value {
            ConstValue::Bool(bit) => Some(ConstValue::Bool(!bit)),
            _ => None,
        },
    }
}

/// Evaluates a binary operation.
fn eval_binary(
    module: &Module,
    op: BinOp,
    left: ConstValue,
    right: ConstValue,
    operand_ty: TypeId,
) -> Option<ConstValue> {
    match op {
        BinOp::IAdd
        | BinOp::ISub
        | BinOp::IMul
        | BinOp::BitwiseAnd
        | BinOp::BitwiseOr
        | BinOp::BitwiseXor
        | BinOp::ShiftLeftLogical
        | BinOp::ShiftRightArithmetic
        | BinOp::ShiftRightLogical
        | BinOp::SDiv
        | BinOp::UDiv
        | BinOp::SRem
        | BinOp::UMod => {
            let (bits, _) = int_shape(module, operand_ty)?;
            let a = int_payload(left)?;
            let b = int_payload(right)?;
            let raw = match op {
                BinOp::IAdd => a.wrapping_add(b),
                BinOp::ISub => a.wrapping_sub(b),
                BinOp::IMul => a.wrapping_mul(b),
                BinOp::BitwiseAnd => a & b,
                BinOp::BitwiseOr => a | b,
                BinOp::BitwiseXor => a ^ b,
                BinOp::ShiftLeftLogical => {
                    if b >= u64::from(bits) {
                        return None;
                    }
                    a << b
                }
                BinOp::ShiftRightLogical => {
                    if b >= u64::from(bits) {
                        return None;
                    }
                    a >> b
                }
                BinOp::ShiftRightArithmetic => {
                    if b >= u64::from(bits) {
                        return None;
                    }
                    mask((sign_extend(a, bits) >> b) as u64, bits)
                }
                BinOp::SDiv => {
                    let divisor = sign_extend(b, bits);
                    if divisor == 0 {
                        return None;
                    }
                    let dividend = sign_extend(a, bits);
                    let quotient = dividend.checked_div(divisor)?;
                    mask(quotient as u64, bits)
                }
                BinOp::SRem => {
                    let divisor = sign_extend(b, bits);
                    if divisor == 0 {
                        return None;
                    }
                    let dividend = sign_extend(a, bits);
                    let remainder = dividend.checked_rem(divisor)?;
                    mask(remainder as u64, bits)
                }
                BinOp::UDiv => {
                    if b == 0 {
                        return None;
                    }
                    a / b
                }
                BinOp::UMod => {
                    if b == 0 {
                        return None;
                    }
                    a % b
                }
                _ => return None,
            };
            Some(ConstValue::Int(mask(raw, bits)))
        }
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
            let bits = float_width(module, operand_ty)?;
            let a = float_payload(left)?;
            let b = float_payload(right)?;
            let value = match bits {
                32 => {
                    let (x, y) = (a as f32, b as f32);
                    match op {
                        BinOp::FAdd => f64::from(x + y),
                        BinOp::FSub => f64::from(x - y),
                        BinOp::FMul => f64::from(x * y),
                        BinOp::FDiv => f64::from(x / y),
                        BinOp::FRem => f64::from(libm::fmodf(x, y)),
                        _ => return None,
                    }
                }
                64 => match op {
                    BinOp::FAdd => a + b,
                    BinOp::FSub => a - b,
                    BinOp::FMul => a * b,
                    BinOp::FDiv => a / b,
                    BinOp::FRem => libm::fmod(a, b),
                    _ => return None,
                },
                _ => return None,
            };
            float_const(value, bits)
        }
        BinOp::LogicalAnd | BinOp::LogicalOr => {
            let a = match left {
                ConstValue::Bool(bit) => bit,
                _ => return None,
            };
            let b = match right {
                ConstValue::Bool(bit) => bit,
                _ => return None,
            };
            Some(ConstValue::Bool(if op == BinOp::LogicalAnd {
                a && b
            } else {
                a || b
            }))
        }
    }
}

/// Evaluates a comparison.
fn eval_compare(
    module: &Module,
    op: CmpOp,
    left: ConstValue,
    right: ConstValue,
    operand_ty: TypeId,
) -> Option<ConstValue> {
    match op {
        CmpOp::IEqual
        | CmpOp::INotEqual
        | CmpOp::SLessThan
        | CmpOp::SLessThanEqual
        | CmpOp::SGreaterThan
        | CmpOp::SGreaterThanEqual
        | CmpOp::ULessThan
        | CmpOp::ULessThanEqual
        | CmpOp::UGreaterThan
        | CmpOp::UGreaterThanEqual => {
            let (bits, _) = int_shape(module, operand_ty)?;
            let a = int_payload(left)?;
            let b = int_payload(right)?;
            let (signed_a, signed_b) = (sign_extend(a, bits), sign_extend(b, bits));
            let result = match op {
                CmpOp::IEqual => a == b,
                CmpOp::INotEqual => a != b,
                CmpOp::SLessThan => signed_a < signed_b,
                CmpOp::SLessThanEqual => signed_a <= signed_b,
                CmpOp::SGreaterThan => signed_a > signed_b,
                CmpOp::SGreaterThanEqual => signed_a >= signed_b,
                CmpOp::ULessThan => a < b,
                CmpOp::ULessThanEqual => a <= b,
                CmpOp::UGreaterThan => a > b,
                CmpOp::UGreaterThanEqual => a >= b,
                _ => return None,
            };
            Some(ConstValue::Bool(result))
        }
        CmpOp::FOrdEqual
        | CmpOp::FOrdNotEqual
        | CmpOp::FOrdLessThan
        | CmpOp::FOrdLessThanEqual
        | CmpOp::FOrdGreaterThan
        | CmpOp::FOrdGreaterThanEqual => {
            let a = float_payload(left)?;
            let b = float_payload(right)?;
            let result = match op {
                CmpOp::FOrdEqual => a == b,
                CmpOp::FOrdNotEqual => a != b,
                CmpOp::FOrdLessThan => a < b,
                CmpOp::FOrdLessThanEqual => a <= b,
                CmpOp::FOrdGreaterThan => a > b,
                CmpOp::FOrdGreaterThanEqual => a >= b,
                _ => return None,
            };
            Some(ConstValue::Bool(result))
        }
    }
}

/// Evaluates a conversion.
fn eval_convert(
    module: &Module,
    op: ConvOp,
    value: ConstValue,
    operand_ty: TypeId,
    result_ty: TypeId,
) -> Option<ConstValue> {
    match op {
        ConvOp::SConvert => {
            let (source_bits, _) = int_shape(module, operand_ty)?;
            let (target_bits, _) = int_shape(module, result_ty)?;
            let raw = int_payload(value)?;
            Some(ConstValue::Int(mask(
                sign_extend(raw, source_bits) as u64,
                target_bits,
            )))
        }
        ConvOp::UConvert => {
            let (target_bits, _) = int_shape(module, result_ty)?;
            let raw = int_payload(value)?;
            Some(ConstValue::Int(mask(raw, target_bits)))
        }
        ConvOp::FConvert => {
            let source_bits = float_width(module, operand_ty)?;
            let target_bits = float_width(module, result_ty)?;
            let raw = float_payload(value)?;
            let narrowed = if source_bits == 32 {
                f64::from(raw as f32)
            } else {
                raw
            };
            float_const(narrowed, target_bits)
        }
        ConvOp::ConvertFToS | ConvOp::ConvertFToU => {
            let (target_bits, target_signed) = int_shape(module, result_ty)?;
            let raw = float_payload(value)?;
            if !raw.is_finite() {
                return None;
            }
            let truncated = libm::trunc(raw);
            let wide = if target_signed {
                if truncated < i64::MIN as f64 || truncated > i64::MAX as f64 {
                    return None;
                }
                truncated as i64 as u64
            } else {
                let limit = if target_bits >= 64 {
                    f64::MAX
                } else {
                    (1u64 << target_bits) as f64
                };
                if truncated < 0.0 || truncated >= limit {
                    return None;
                }
                truncated as u64
            };
            Some(ConstValue::Int(mask(wide, target_bits)))
        }
        ConvOp::ConvertSToF | ConvOp::ConvertUToF => {
            let (source_bits, source_signed) = int_shape(module, operand_ty)?;
            let target_bits = float_width(module, result_ty)?;
            let raw = int_payload(value)?;
            let wide = if source_signed {
                sign_extend(raw, source_bits) as f64
            } else {
                raw as f64
            };
            float_const(wide, target_bits)
        }
        ConvOp::Bitcast => {
            let source = module.ty(operand_ty);
            let target = module.ty(result_ty);
            match (source, target) {
                (
                    Type::Int { bits: _, .. },
                    Type::Int {
                        bits: target_bits, ..
                    },
                ) => {
                    let raw = int_payload(value)?;
                    Some(ConstValue::Int(mask(raw, *target_bits)))
                }
                (Type::Int { bits, .. }, Type::Float { bits: target_bits }) => {
                    let raw = int_payload(value)?;
                    if u16::from(*bits) != *target_bits {
                        return None;
                    }
                    match *target_bits {
                        32 => Some(ConstValue::from_f32_bits(raw as u32)),
                        64 => Some(ConstValue::from_f64_bits(raw)),
                        _ => None,
                    }
                }
                (
                    Type::Float { bits, .. },
                    Type::Int {
                        bits: target_bits, ..
                    },
                ) => {
                    let raw = float_payload(value)?;
                    if *bits != u16::from(*target_bits) {
                        return None;
                    }
                    let bits = match *target_bits {
                        32 => u64::from((raw as f32).to_bits()),
                        64 => raw.to_bits(),
                        _ => return None,
                    };
                    Some(ConstValue::Int(mask(bits, *target_bits)))
                }
                (Type::Float { bits, .. }, Type::Float { bits: target_bits }) => {
                    if *bits != *target_bits {
                        return None;
                    }
                    Some(value)
                }
                _ => None,
            }
        }
    }
}
