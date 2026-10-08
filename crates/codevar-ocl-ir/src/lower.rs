//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! AST to IR lowering.
//!
//! This module converts the AST from [`codevar_ocl_parse`] together with
//! the semantic tables from [`codevar_ocl_sar`] into a [`Module`] from the
//! IR data model. The lowering follows LLVM's alloca-based pattern: every
//! local variable lives in a stack slot and reads and writes go through
//! `load`/`store`, so staged scalar optimizations (`mem2reg`, DCE, GVN,
//! inlining, LICM) have exactly the structure they expect.
//!
//! Structured control flow is emitted directly as SPIR-V-style
//! `OpSelectionMerge` / `OpLoopMerge` pairs, so the CFG produced here
//! already satisfies the structured-CFG rules that [`crate::verify`]
//! enforces.
//!
//! # Entry points
//!
//! * [`lower`] — analyzed AST plus semantic tables to [`Module`].
//! * [`lower_source`] — the whole pipeline: analyze, parse, then lower.
//!
//! Lowering mirrors the analyzer's typing rules exactly (literal suffix
//! defaults, unification order, swizzle sets, builtin signatures), so any
//! source that [`codevar_ocl_sar::analyze`] accepts without errors lowers
//! to IR that [`crate::verify::verify`] accepts.
//!
//! # Example
//!
//! ```
//! use codevar_ocl_ir::lower::lower_source;
//! use codevar_ocl_ir::verify::verify;
//!
//! let source = "#[kernel]\nfn zero(out: *mut int) {\n    *out = 0;\n}";
//! let module = lower_source(source).expect("lowering succeeds");
//! verify(&module).expect("IR verifies");
//! ```

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use codevar_ocl_lex::{Base, LiteralKind, TokenKind};
use codevar_ocl_parse::{
    BinaryOp, Block, Expr, ExprKind, FnItem, GenericParam, ItemKind, LetStmt, NodeId, Param, Pat, PatKind,
    Path, Program, Span, Stmt, StmtKind, Type, TypeKind, UnaryOp,
};
use codevar_ocl_sar::{
    AnalysisOutput, Builtin, BuiltinKind, BuiltinType, DeclKind, Diagnostic, Res, Scalar, Ty, coerce,
    lookup_builtin, lookup_builtin_fn,
};

use crate::ir::{
    BinOp, BlockId, BuildError, CmpOp, ConstValue, ConvOp, Decor, ExecutionModel, ExtSetId, Inst, Linkage,
    Module, Op, Storage, Target, Type as IrType, TypeId, UnOp, ValueId,
};
use crate::spirv::ops::ocl_opcode;

/// Error produced while lowering to IR.
///
/// Lowering is fail-fast: the first construct without an IR
/// representation aborts the whole compilation unit, and no partial
/// module is returned.
#[derive(Debug, Clone, PartialEq)]
pub enum LowerError {
    /// Semantic analysis reported at least one error, so lowering
    /// never started.
    AnalysisFailed {
        /// The error diagnostics, in source order.
        errors: Vec<Diagnostic>,
    },
    /// The source did not parse; the AST would be misleading.
    ParseFailed {
        /// Number of parser errors.
        count: usize,
    },
    /// A source construct has no IR representation.
    Unsupported {
        /// What kind of construct was rejected.
        what: &'static str,
        /// Where the construct occurs.
        span: Span,
    },
    /// A semantic type has no IR representation.
    UnsupportedType {
        /// The rejected type.
        ty: Ty,
        /// Where the type occurs.
        span: Span,
    },
    /// A value-returning function can reach the end of its body without
    /// returning.
    MissingReturn {
        /// Name of the offending function.
        name: String,
    },
    /// The IR builder rejected an otherwise valid construct.
    Build {
        /// What was being built when the builder failed.
        what: &'static str,
        /// The builder's own error.
        error: BuildError,
    },
}

impl core::fmt::Display for LowerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AnalysisFailed { errors } => {
                write!(f, "semantic analysis failed with {} error(s)", errors.len())
            }
            Self::ParseFailed { count } => write!(f, "parsing failed with {count} error(s)"),
            Self::Unsupported { what, span } => {
                write!(
                    f,
                    "{what} at {}..{}",
                    span.offset,
                    span.offset.saturating_add(span.len)
                )
            }
            Self::UnsupportedType { ty, span } => write!(
                f,
                "type `{ty}` has no IR representation at {}..{}",
                span.offset,
                span.offset.saturating_add(span.len)
            ),
            Self::MissingReturn { name } => write!(
                f,
                "function `{name}` can reach the end of its body without returning a value"
            ),
            Self::Build { what, error } => write!(f, "cannot build {what}: {error}"),
        }
    }
}

impl core::error::Error for LowerError {}

/// Strips the `r#` escape from a raw identifier, leaving other names alone.
fn strip_raw_ident(name: &str) -> &str {
    name.strip_prefix("r#").unwrap_or(name)
}

/// Replaces literal placeholders with their default types: an unsuffixed
/// integer literal becomes `int`, a float literal `float`.  The rule
/// recurses through arrays, tuples, pointers, and references so that
/// `let xs = [1, 2];` yields `[int; 2]`, exactly as the analyzer binds it.
fn default_literals(ty: &Ty) -> Ty {
    match ty {
        Ty::IntLit(_) => Ty::Scalar(Scalar::I32),
        Ty::FloatLit(_) => Ty::Scalar(Scalar::F32),
        Ty::Array { elem, len } => Ty::Array {
            elem: Box::new(default_literals(elem)),
            len: *len,
        },
        Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(default_literals).collect()),
        Ty::Ptr { mutable, inner } => Ty::Ptr {
            mutable: *mutable,
            inner: Box::new(default_literals(inner)),
        },
        Ty::Ref { mutable, inner } => Ty::Ref {
            mutable: *mutable,
            inner: Box::new(default_literals(inner)),
        },
        other => other.clone(),
    }
}

/// Evaluates an expression as a constant integer, `None` when it is not
/// constant arithmetic over integer literals.  Used for array lengths.
fn const_int(expr: &Expr) -> Option<i128> {
    match &expr.kind {
        ExprKind::Literal {
            text,
            kind:
                TokenKind::Literal {
                    kind:
                        LiteralKind::Int {
                            base,
                            empty_int: false,
                        },
                    suffix_start,
                },
        } => {
            let split = (*suffix_start as usize).min(text.len());
            let (body, _) = text.split_at(split);
            int_digits_value(body, *base)
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            expr: inner,
        } => const_int(inner)?.checked_neg(),
        ExprKind::Binary { op, lhs, rhs } => {
            let left = const_int(lhs)?;
            let right = const_int(rhs)?;
            match op {
                BinaryOp::Add => left.checked_add(right),
                BinaryOp::Sub => left.checked_sub(right),
                BinaryOp::Mul => left.checked_mul(right),
                BinaryOp::Div if right != 0 => left.checked_div(right),
                BinaryOp::Rem if right != 0 => left.checked_rem(right),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Parses the digits of an integer literal, underscores removed.
///
/// Returns `None` when the text is empty or does not fit in an `i128`,
/// which the caller reports as an out-of-range literal.
fn int_digits_value(text: &str, base: Base) -> Option<i128> {
    let digits = strip_radix_prefix(text, base);
    let mut cleaned = String::with_capacity(digits.len());
    cleaned.extend(
        digits
            .chars()
            .filter(|character| *character != '_'),
    );
    if cleaned.is_empty() {
        return None;
    }
    i128::from_str_radix(&cleaned, base as u32).ok()
}

/// Removes the `0x`/`0b`/`0o` prefix; decimal and C-style octal keep the
/// leading digit so `0755` parses as the octal constant it is.
fn strip_radix_prefix(text: &str, base: Base) -> &str {
    let has_o_prefix = text
        .get(1..2)
        .is_some_and(|letter| letter.eq_ignore_ascii_case("o"));
    match base {
        Base::Binary | Base::Hexadecimal => text.get(2..).unwrap_or(text),
        Base::Octal if has_o_prefix => text.get(2..).unwrap_or(text),
        _ => text,
    }
}

/// Parses a decimal float literal; hexadecimal floats return `None`
/// because `f64::from_str` does not implement the `0x1.8p3` grammar.
fn float_value(text: &str) -> Option<f64> {
    let mut cleaned = String::with_capacity(text.len());
    cleaned.extend(text.chars().filter(|character| *character != '_'));
    cleaned.parse::<f64>().ok()
}

/// Parses a hexadecimal float literal (`0x1.8p3`), underscores
/// removed; `None` when the text is not a well-formed hex float or
/// carries a non-hexadecimal digit.
///
/// The mantissa accumulates digit by digit and the binary exponent
/// scales the result.  Exponents past the f64 range saturate to
/// infinity or zero, which is what a C hexadecimal literal of that
/// magnitude denotes.
fn hex_float_value(text: &str) -> Option<f64> {
    let cleaned: String = text
        .chars()
        .filter(|character| *character != '_')
        .collect();
    let prefix = cleaned.get(..2)?;
    if !prefix.eq_ignore_ascii_case("0x") {
        return None;
    }
    let body = cleaned.get(2..)?;
    let (mantissa, exponent) = match body.find(['p', 'P']) {
        Some(position) => (
            body.get(..position)?,
            body.get(position + 1..)?.parse::<i32>().ok()?,
        ),
        None => (body, 0),
    };
    let (integral, fractional) = match mantissa.find('.') {
        Some(position) => (mantissa.get(..position)?, mantissa.get(position + 1..)?),
        None => (mantissa, ""),
    };
    if integral.is_empty() && fractional.is_empty() {
        return None;
    }
    let mut value = 0f64;
    for digit in integral.chars() {
        value = value * 16.0 + f64::from(digit.to_digit(16)?);
    }
    let mut scale = 1.0f64;
    for digit in fractional.chars() {
        scale /= 16.0;
        value += f64::from(digit.to_digit(16)?) * scale;
    }
    if exponent < -1075 || exponent > 1024 {
        let magnitude = if exponent > 1024 { f64::INFINITY } else { 0.0 };
        return Some(if value == 0.0 { 0.0 } else { value * magnitude });
    }
    let mut magnitude = 1.0f64;
    for _ in 0..exponent.unsigned_abs() {
        magnitude = if exponent > 0 {
            magnitude * 2.0
        } else {
            magnitude / 2.0
        };
    }
    Some(value * magnitude)
}

/// The constant payload of the float `value` reinterpreted at type
/// `ty`, `None` when `ty` holds none of the constant kinds.
///
/// A float narrows by truncating toward zero and reinterpreting the
/// bits, matching C's conversion; values beyond an integer's reach
/// saturate the way Rust's `as` cast does, since C leaves those
/// undefined.
fn float_payload(value: f64, ty: &IrType) -> Option<ConstValue> {
    match ty {
        IrType::Float { bits: 32 } => Some(ConstValue::from_f32_bits((value as f32).to_bits())),
        IrType::Float { bits: 64 } => Some(ConstValue::from_f64_bits(value.to_bits())),
        IrType::Int { bits, .. } => {
            let truncated = value.trunc() as i128;
            Some(ConstValue::Int(mask_bits(truncated as u64, *bits)))
        }
        IrType::Bool => Some(ConstValue::Bool(value != 0.0)),
        _ => None,
    }
}

/// Peels reference layers off `ty`, returning how many were removed and
/// the type underneath.
///
/// Array and pointer indexing use this: `p[i]` on a `&T` addresses
/// through the reference, while a `*T` base keeps its own indirection for
/// pointer arithmetic.
fn split_refs(ty: &Ty) -> (usize, Ty) {
    let mut layers = 0usize;
    let mut rest = ty.clone();
    loop {
        match rest {
            Ty::Ref { inner, .. } => {
                rest = *inner;
                layers += 1;
            }
            other => return (layers, other),
        }
    }
}

/// Peels every pointer and reference layer off `ty`, returning the count
/// and the innermost type.
///
/// Field reads and array iteration use this: reaching the aggregate means
/// loading through each indirection first.
fn split_indirection(ty: &Ty) -> (usize, Ty) {
    let mut layers = 0usize;
    let mut rest = ty.clone();
    loop {
        match rest {
            Ty::Ref { inner, .. } | Ty::Ptr { inner, .. } => {
                rest = *inner;
                layers += 1;
            }
            other => return (layers, other),
        }
    }
}

/// True when the item declares any type parameters, which lowering
/// rejects: monomorphization happens before IR construction.
fn has_type_parameters(generics: &[GenericParam]) -> bool {
    generics
        .iter()
        .any(|param| matches!(param, GenericParam::Type { .. }))
}

/// The IR binary opcode for arithmetic `op` at element kind, `None` when
/// the operator is not arithmetic (shifts, bitwise, and logical operators
/// take separate paths).
fn arith_binop(op: BinaryOp, float: bool, signed: bool) -> Option<BinOp> {
    Some(match (op, float, signed) {
        (BinaryOp::Add, false, _) => BinOp::IAdd,
        (BinaryOp::Add, true, _) => BinOp::FAdd,
        (BinaryOp::Sub, false, _) => BinOp::ISub,
        (BinaryOp::Sub, true, _) => BinOp::FSub,
        (BinaryOp::Mul, false, _) => BinOp::IMul,
        (BinaryOp::Mul, true, _) => BinOp::FMul,
        (BinaryOp::Div, true, _) => BinOp::FDiv,
        (BinaryOp::Div, false, true) => BinOp::SDiv,
        (BinaryOp::Div, false, false) => BinOp::UDiv,
        (BinaryOp::Rem, true, _) => BinOp::FRem,
        (BinaryOp::Rem, false, true) => BinOp::SRem,
        (BinaryOp::Rem, false, false) => BinOp::UMod,
        _ => return None,
    })
}

/// The IR comparison opcode for `op` at element kind, `None` when the
/// operator is not a comparison.
fn cmp_op(op: BinaryOp, float: bool, signed: bool) -> Option<CmpOp> {
    Some(match op {
        BinaryOp::Eq => {
            if float {
                CmpOp::FOrdEqual
            } else {
                CmpOp::IEqual
            }
        }
        BinaryOp::Ne => {
            if float {
                CmpOp::FOrdNotEqual
            } else {
                CmpOp::INotEqual
            }
        }
        BinaryOp::Lt => match (float, signed) {
            (true, _) => CmpOp::FOrdLessThan,
            (false, true) => CmpOp::SLessThan,
            (false, false) => CmpOp::ULessThan,
        },
        BinaryOp::Le => match (float, signed) {
            (true, _) => CmpOp::FOrdLessThanEqual,
            (false, true) => CmpOp::SLessThanEqual,
            (false, false) => CmpOp::ULessThanEqual,
        },
        BinaryOp::Gt => match (float, signed) {
            (true, _) => CmpOp::FOrdGreaterThan,
            (false, true) => CmpOp::SGreaterThan,
            (false, false) => CmpOp::UGreaterThan,
        },
        BinaryOp::Ge => match (float, signed) {
            (true, _) => CmpOp::FOrdGreaterThanEqual,
            (false, true) => CmpOp::SGreaterThanEqual,
            (false, false) => CmpOp::UGreaterThanEqual,
        },
        _ => return None,
    })
}

/// The `OpenCL.std` instruction number for `name` applied at scalar
/// `element`, `None` when the set defines no such instruction.
///
/// Numeric names carry a family prefix in the set: `abs` is `fabs` on
/// floats, `s_abs` on signed integers, `u_abs` on unsigned ones.  Names
/// that exist bare (`popcount`, `sign`) are looked up unprefixed, and
/// families without an integer form (`sign`) or a float form
/// (`popcount`) resolve to `None`.
fn ext_inst_opcode(name: &str, element: &IrType) -> Option<u32> {
    match element {
        IrType::Float { .. } => match name {
            "popcount" => None,
            "sign" => ocl_opcode("sign"),
            "abs" => ocl_opcode("fabs"),
            "min" | "max" | "clamp" => ocl_opcode(&format!("f{name}")),
            other => ocl_opcode(other),
        },
        IrType::Int { signed, .. } => match name {
            "sign" => None,
            "popcount" | "clz" | "ctz" | "rotate" => ocl_opcode(name),
            other => {
                let prefix = if *signed { 's' } else { 'u' };
                ocl_opcode(&format!("{prefix}_{other}"))
            }
        },
        _ => None,
    }
}

/// The swizzle alphabets the dialect accepts.
const SWIZZLE_SETS: [&str; 3] = ["xyzw", "rgba", "stpq"];

/// Lane indices a field access selects from a `lanes`-lane vector,
/// `None` when `field` is not a valid swizzle or lane index.
///
/// Mirrors the analyzer's rule: numeric fields index a lane directly,
/// swizzle letters come from one alphabet, and at most four lanes are
/// selected.
fn swizzle_lanes(field: &str, lanes: u8) -> Option<Vec<u32>> {
    if let Ok(index) = field.parse::<usize>() {
        if index < lanes as usize {
            return u32::try_from(index).ok().map(|lane| vec![lane]);
        }
        return None;
    }
    if field.is_empty() || field.chars().count() > 4 {
        return None;
    }
    let set = SWIZZLE_SETS.iter().find(|set| {
        field
            .chars()
            .all(|character| set.contains(character))
    })?;
    let mut out = Vec::with_capacity(field.len());
    for character in field.chars() {
        let position = set.find(character)?;
        if position >= lanes as usize {
            return None;
        }
        out.push(u32::try_from(position).ok()?);
    }
    Some(out)
}

/// True for an IR floating-point type, including vectors of floats.
fn ir_is_float(module: &Module, ty: TypeId) -> bool {
    match module.ty(ty) {
        IrType::Float { .. } => true,
        IrType::Vector { elem, .. } => matches!(module.ty(*elem), IrType::Float { .. }),
        _ => false,
    }
}

/// True for an IR boolean type, including vectors of booleans.
fn ir_is_bool(module: &Module, ty: TypeId) -> bool {
    match module.ty(ty) {
        IrType::Bool => true,
        IrType::Vector { elem, .. } => matches!(module.ty(*elem), IrType::Bool),
        _ => false,
    }
}

/// True for an IR integer type, including vectors of integers.
fn ir_is_int(module: &Module, ty: TypeId) -> bool {
    match module.ty(ty) {
        IrType::Int { .. } => true,
        IrType::Vector { elem, .. } => matches!(module.ty(*elem), IrType::Int { .. }),
        _ => false,
    }
}

/// `Some(signed)` when `ty` is an integer type, `None` otherwise.
fn ir_signedness(module: &Module, ty: TypeId) -> Option<bool> {
    match module.ty(ty) {
        IrType::Int { signed, .. } => Some(*signed),
        IrType::Vector { elem, .. } => match module.ty(*elem) {
            IrType::Int { signed, .. } => Some(*signed),
            _ => None,
        },
        _ => None,
    }
}

/// The scalar element type of `ty`; vectors resolve to their element.
fn ir_element(module: &Module, ty: TypeId) -> TypeId {
    match module.ty(ty) {
        IrType::Vector { elem, .. } => *elem,
        _ => ty,
    }
}

/// True when `value` is representable by an integer of `bits` width.
fn int_fits(value: i128, bits: u8, signed: bool) -> bool {
    if bits == 0 || bits > 64 {
        return false;
    }
    if signed {
        let limit = (1i128 << (bits - 1)) - 1;
        (-limit - 1..=limit).contains(&value)
    } else {
        value >= 0 && value < (1i128 << bits)
    }
}

/// Zero-extends or truncates `raw` to `bits`, the canonical integer
/// payload shape [`crate::ir::ConstValue::Int`] requires.
fn mask_bits(raw: u64, bits: u8) -> u64 {
    if bits >= 64 {
        raw
    } else {
        raw & ((1u64 << bits) - 1)
    }
}

/// Sign-extends the `bits`-wide payload `raw` to a full `i128`.
fn sign_extend(raw: u64, bits: u8) -> i128 {
    if bits == 0 || bits >= 64 {
        return raw as i128;
    }
    let shift = 128 - i32::from(bits);
    ((raw as i128) << shift) >> shift
}

/// Reinterprets the integer payload `raw` when it moves from an integer
/// of `from_bits`/`from_signed` to one of `to_bits` width.
fn reinterpret_int(raw: u64, from_bits: u8, from_signed: bool, to_bits: u8) -> u64 {
    let value = if from_signed {
        sign_extend(raw, from_bits)
    } else {
        i128::from(raw)
    };
    mask_bits(value as u64, to_bits)
}

/// The narrowest default integer scalar that holds `value`, in the order
/// `int`, `long`, `ulong`; `None` when no default type fits.
fn literal_int_scalar(value: i128) -> Option<Scalar> {
    [Scalar::I32, Scalar::I64, Scalar::U64]
        .into_iter()
        .find(|scalar| scalar.fits_int(value))
}

/// The type both operands of an operation materialize at.
///
/// Mirrors the analyzer's unification ([`codevar_ocl_sar::coerce`]) and
/// then normalizes literal placeholders the way code generation must:
/// every integer literal operand picks one width that holds them all, so
/// `3000000000 + 1` computes at `long` before the expression type
/// narrows it again.
///
/// Returns `None` when the operands are incompatible or a literal is out
/// of range for every default type.
fn unify_operand_types(left: &Ty, right: &Ty) -> Option<Ty> {
    let unified = if coerce(left, right) {
        right.clone()
    } else if coerce(right, left) {
        left.clone()
    } else {
        return None;
    };
    normalize_operand_ty(&unified, left, right)
}

/// Applies literal normalization to an already-unified operand type.
fn normalize_operand_ty(unified: &Ty, left: &Ty, right: &Ty) -> Option<Ty> {
    match unified {
        Ty::IntLit(_) => {
            let candidates = [Scalar::I32, Scalar::I64, Scalar::U64];
            let scalar = candidates.into_iter().find(|scalar| {
                [left, right].iter().all(|operand| match operand {
                    Ty::IntLit(value) => scalar.fits_int(*value),
                    _ => true,
                })
            })?;
            Some(Ty::Scalar(scalar))
        }
        Ty::FloatLit(_) => Some(Ty::Scalar(Scalar::F32)),
        other => Some(default_literals(other)),
    }
}

/// Lowers an already-analyzed program to an IR module.
///
/// # Errors
///
/// Returns [`LowerError::AnalysisFailed`] when analysis reported errors
/// (lowering never starts, since the tables would be incomplete), and a
/// construction error when a construct has no IR representation or the
/// IR builder rejects it.
pub fn lower(program: &Program, analyzed: &AnalysisOutput) -> Result<Module, LowerError> {
    if analyzed.has_errors() {
        return Err(LowerError::AnalysisFailed {
            errors: analyzed
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.is_error())
                .cloned()
                .collect(),
        });
    }
    let mut lowerer = Lowerer::new(analyzed);
    lowerer.run(program)?;
    Ok(lowerer.module)
}

/// Analyzes, parses, and lowers `source` in one call.
///
/// This is the pipeline entry point: it runs
/// [`codevar_ocl_sar::analyze`] and [`codevar_ocl_parse::parse`], then
/// [`lower`] on the results.
///
/// # Errors
///
/// Returns [`LowerError::AnalysisFailed`] when analysis reported errors,
/// [`LowerError::ParseFailed`] when the source did not parse, and
/// anything [`lower`] returns.
pub fn lower_source(source: &str) -> Result<Module, LowerError> {
    let analyzed = codevar_ocl_sar::analyze(source);
    if analyzed.has_errors() {
        return Err(LowerError::AnalysisFailed {
            errors: analyzed
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.is_error())
                .cloned()
                .collect(),
        });
    }
    let parsed = codevar_ocl_parse::parse(source);
    if !parsed.errors.is_empty() {
        return Err(LowerError::ParseFailed {
            count: parsed.errors.len(),
        });
    }
    lower(&parsed.program, &analyzed)
}

/// What lowering an expression produced.
///
/// The invariant: `Value` and `Void` are returned only while the current
/// block is [`BlockState::Open`], and `Dead` exactly when control flow
/// does not continue past the expression.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Lowered {
    /// A value at the expression's type.
    Value(ValueId),
    /// No value: the expression is unit-like (`void`, or a diverging
    /// expression in a value position that never produces one).
    Void,
    /// Control flow does not continue; the current block is terminated.
    Dead,
}

/// What the current block needs when control arrives at it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockState {
    /// The block is being filled; instructions may be appended.
    Open,
    /// The block already ends in a terminator.
    Terminated,
    /// The block exists but has no instructions yet — an arm no
    /// execution can reach, such as the `else` branch after a `return`.
    /// [`Lowerer::finish_current`] closes it with `OpUnreachable`.
    FreshDead,
}

/// How a loop's condition header ended.
#[derive(Debug, Clone, Copy, PartialEq)]
enum HeaderResult {
    /// The condition's evaluation terminated the header (`return`),
    /// so no body or exit edge exists yet.
    Dead,
    /// No condition: the header branches straight to the body.
    Uncond,
    /// A condition value; the header branches on it.
    Cond(ValueId),
}

/// How a loop's exit block becomes reachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExitReach {
    /// The condition header's false edge targets it.
    ViaCond,
    /// Nothing targets it; the block stays empty until a `break`
    /// arrives or the merge is closed with `OpUnreachable`.
    Untaken,
    /// A `break` targets it.
    Taken,
}

/// The blocks `break` and `continue` inside a loop jump to.
#[derive(Debug, Clone, Copy)]
struct LoopCtx {
    /// Continue (back-edge) block.
    cont: BlockId,
    /// Exit (break) block.
    exit: BlockId,
    /// How the exit block is reached.
    exit_reach: ExitReach,
}

/// What lowering remembers about a declared function.
#[derive(Debug, Clone)]
struct FnSig {
    /// The IR function value.
    ir: ValueId,
    /// Declared parameter types, in order.
    params: Vec<TypeId>,
    /// Declared return type.
    ret: TypeId,
}

/// Where a name binding lives.
#[derive(Debug, Clone, Copy)]
enum Binding {
    /// An immutable value that is already materialized — parameters and
    /// non-`mut` locals; reads are the value itself.
    Value(ValueId),
    /// A stack slot; reads and writes go through `load`/`store`.
    Slot(ValueId),
}

/// A writable location, or why there is none.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Place {
    /// A pointer to the location.
    Ptr(ValueId),
    /// The expression names something that cannot be written (an
    /// immutable binding, a field, a call result).
    NotPlace,
    /// Control flow does not reach the write.
    Dead,
}

/// A `for` loop's counter plumbing: the counter slot's pointer, its
/// IR type, the iterated element's IR type, the exclusive limit
/// value, the loop-comparison form, and (for aggregate iteration) the
/// iterated value's address for indexed element reads.  `None` means
/// an endpoint or the iterated value diverged.
type LoopSetup = Option<(ValueId, TypeId, TypeId, ValueId, CmpOp, Option<ValueId>)>;

/// The lowering context: module under construction plus semantic tables
/// and the state of the function currently being lowered.
struct Lowerer<'a> {
    /// The IR module being built.
    module: Module,
    /// Semantic tables from the analyzer.
    analysis: &'a AnalysisOutput,
    /// Type-alias names resolved to their targets.
    aliases: BTreeMap<String, Ty>,
    /// Declared functions, by name.
    signatures: BTreeMap<String, FnSig>,
    /// The `OpenCL.std` extended-instruction set, created on first use.
    ext_set: Option<ExtSetId>,
    /// Where each in-scope binding lives.
    slots: BTreeMap<NodeId, Binding>,
    /// Stack slots to splice at the top of the function's entry block.
    allocas: Vec<Inst>,
    /// Enclosing loops, innermost last.
    loops: Vec<LoopCtx>,
    /// The function being lowered.
    func: Option<ValueId>,
    /// Its name, for diagnostics.
    func_name: String,
    /// Its declared return type.
    ret: Ty,
    /// The block instructions are appended to.
    current: Option<BlockId>,
    /// Whether `current` still accepts instructions.
    state: BlockState,
}

impl<'a> Lowerer<'a> {
    /// Creates a lowering context for already-analyzed source.
    fn new(analysis: &'a AnalysisOutput) -> Self {
        Self {
            module: Module::new(Target::opencl()),
            analysis,
            aliases: BTreeMap::new(),
            signatures: BTreeMap::new(),
            ext_set: None,
            slots: BTreeMap::new(),
            allocas: Vec::new(),
            loops: Vec::new(),
            func: None,
            func_name: String::new(),
            ret: Ty::Void,
            current: None,
            state: BlockState::Terminated,
        }
    }

    /// True when the current block accepts more instructions.
    fn is_open(&self) -> bool {
        self.state == BlockState::Open
    }

    /// A construction failure blamed on `what` at an unspecified span —
    /// used where no source construct explains the problem.
    fn unsupported(&self, what: &'static str) -> LowerError {
        LowerError::Unsupported {
            what,
            span: Span::default(),
        }
    }

    /// Appends `inst` to the current block.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when control flow does not
    /// reach the block (or no function is being lowered), and a
    /// construction error when the module rejects the instruction.
    fn emit_raw(&mut self, inst: Inst) -> Result<(), LowerError> {
        if self.state != BlockState::Open {
            return Err(self.unsupported("an instruction in unreachable code"));
        }
        let (Some(func), Some(block)) = (self.func, self.current) else {
            return Err(self.unsupported("code outside a function"));
        };
        self.module
            .emit(func, block, inst)
            .map_err(|error| LowerError::Build {
                what: "an instruction",
                error,
            })
    }

    /// Appends a terminator to the current block and closes it.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the block is already
    /// closed, and a construction error when the module rejects it.
    fn term(&mut self, op: Op) -> Result<(), LowerError> {
        self.emit_raw(Inst::none(op))?;
        self.state = BlockState::Terminated;
        Ok(())
    }

    /// Appends an instruction that defines a result of type `ty`.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the block is closed, and
    /// a construction error when the module rejects it.
    fn def_op(&mut self, ty: TypeId, op: Op) -> Result<ValueId, LowerError> {
        let result = self.module.new_inst_value(ty);
        self.emit_raw(Inst::def(result, ty, op))?;
        Ok(result)
    }

    /// Appends a uniquely named block to the current function.
    ///
    /// # Errors
    ///
    /// Returns a construction error when no function body is open.
    fn push_block(&mut self, base: &str) -> Result<BlockId, LowerError> {
        let func = self
            .func
            .ok_or_else(|| self.unsupported("a block outside a function"))?;
        let name = self.module.uniquify_name(base);
        self.module
            .push_block(func, &name)
            .map_err(|error| LowerError::Build {
                what: "a basic block",
                error,
            })
    }

    /// Closes the current block if it never received instructions.
    ///
    /// # Errors
    ///
    /// Returns a construction error when the module rejects the closing
    /// `OpUnreachable`.
    fn finish_current(&mut self) -> Result<(), LowerError> {
        if self.state != BlockState::FreshDead {
            return Ok(());
        }
        let (Some(func), Some(block)) = (self.func, self.current) else {
            self.state = BlockState::Terminated;
            return Ok(());
        };
        self.module
            .emit(func, block, Inst::none(Op::Unreachable))
            .map_err(|error| LowerError::Build {
                what: "a block terminator",
                error,
            })?;
        self.state = BlockState::Terminated;
        Ok(())
    }

    /// Makes `block` the current block, closing the previous one first.
    ///
    /// `open` is `false` for a block whose instructions cannot be filled
    /// yet; it is closed with `OpUnreachable` when control moves on
    /// unless something branches into it meanwhile.
    ///
    /// # Errors
    ///
    /// Returns a construction error when closing the previous block
    /// fails.
    fn enter(&mut self, block: BlockId, open: bool) -> Result<(), LowerError> {
        self.finish_current()?;
        self.current = Some(block);
        self.state = if open {
            BlockState::Open
        } else {
            BlockState::FreshDead
        };
        Ok(())
    }

    /// Maps a semantic type to its IR type, applying literal defaults.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::UnsupportedType`] for types with no IR
    /// representation: `half` (OpenCL's `cl_khr_fp16` is not enabled),
    /// structs, tuples, slices, and the `void`/`!`/generic placeholders.
    fn map_ty(&mut self, ty: &Ty, span: Span) -> Result<TypeId, LowerError> {
        let ty = default_literals(ty);
        Ok(match &ty {
            Ty::Void => self.module.void_ty(),
            Ty::Scalar(scalar) => self.map_scalar(*scalar, span)?,
            Ty::Vector { elem, lanes } => {
                let element = self.map_scalar(*elem, span)?;
                self.module.vector_ty(element, u32::from(*lanes))
            }
            Ty::Ptr { inner, .. } => {
                let pointee = self.map_ty(inner, span)?;
                self.module
                    .ptr_ty(Storage::CrossWorkgroup, pointee)
            }
            Ty::Ref { inner, .. } => {
                let pointee = self.map_ty(inner, span)?;
                self.module.ptr_ty(Storage::Function, pointee)
            }
            Ty::Array { elem, len: Some(len) } => {
                let element = self.map_ty(elem, span)?;
                self.module.array_ty(element, *len)
            }
            Ty::IntLit(_) => self.module.int_ty(32, true),
            Ty::FloatLit(_) => self.module.float_ty(32),
            _ => return Err(LowerError::UnsupportedType { ty, span }),
        })
    }

    /// Maps a scalar to its IR type.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::UnsupportedType`] for `half`: 16-bit floats
    /// need an extension the module does not enable.
    fn map_scalar(&mut self, scalar: Scalar, span: Span) -> Result<TypeId, LowerError> {
        Ok(match scalar {
            Scalar::Bool => self.module.bool_ty(),
            Scalar::I8 => self.module.int_ty(8, true),
            Scalar::I16 => self.module.int_ty(16, true),
            Scalar::I32 => self.module.int_ty(32, true),
            Scalar::I64 => self.module.int_ty(64, true),
            Scalar::U8 => self.module.int_ty(8, false),
            Scalar::U16 => self.module.int_ty(16, false),
            Scalar::U32 => self.module.int_ty(32, false),
            Scalar::U64 => self.module.int_ty(64, false),
            Scalar::F32 => self.module.float_ty(32),
            Scalar::F64 => self.module.float_ty(64),
            Scalar::F16 => {
                return Err(LowerError::UnsupportedType {
                    ty: Ty::Scalar(Scalar::F16),
                    span,
                });
            }
        })
    }

    /// Resolves an AST type expression to a semantic type.
    ///
    /// The analyzer already resolved every type in checked source, so
    /// this rebuilds the same [`Ty`] from the tree: built-in names first,
    /// then declared aliases, then a struct instantiation.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for malformed types (an
    /// `Error` node, a multi-segment path, an invalid lane count, or an
    /// array length that is not a 32-bit constant).
    fn resolve_ast_ty(&self, ty: &Type) -> Result<Ty, LowerError> {
        Ok(match &ty.kind {
            TypeKind::Error => {
                return Err(LowerError::Unsupported {
                    what: "a malformed type",
                    span: ty.span,
                });
            }
            TypeKind::Ref { mutable, inner, .. } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.resolve_ast_ty(inner)?),
            },
            TypeKind::Ptr { mutable, inner } => Ty::Ptr {
                mutable: *mutable,
                inner: Box::new(self.resolve_ast_ty(inner)?),
            },
            TypeKind::Tuple(elems) => Ty::Tuple(
                elems
                    .iter()
                    .map(|elem| self.resolve_ast_ty(elem))
                    .collect::<Result<Vec<Ty>, LowerError>>()?,
            ),
            TypeKind::Slice(inner) => Ty::Array {
                elem: Box::new(self.resolve_ast_ty(inner)?),
                len: None,
            },
            TypeKind::Array { elem, len } => {
                let length = const_int(len).ok_or_else(|| LowerError::Unsupported {
                    what: "an array length that is not a constant integer",
                    span: len.span,
                })?;
                let length = u32::try_from(length).map_err(|_| LowerError::Unsupported {
                    what: "an array length that does not fit in 32 bits",
                    span: len.span,
                })?;
                Ty::Array {
                    elem: Box::new(self.resolve_ast_ty(elem)?),
                    len: Some(length),
                }
            }
            TypeKind::Path(path) => self.resolve_type_path(path, ty.span)?,
        })
    }

    /// Resolves a single-segment type path against built-ins, aliases,
    /// and declared structs.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the path is malformed.
    fn resolve_type_path(&self, path: &codevar_ocl_parse::TypePath, span: Span) -> Result<Ty, LowerError> {
        if path.segments.len() != 1 {
            return Err(LowerError::Unsupported {
                what: "a qualified type path",
                span,
            });
        }
        let Some(segment) = path.segments.last() else {
            return Err(LowerError::Unsupported {
                what: "an empty type path",
                span,
            });
        };
        let name = strip_raw_ident(&segment.name);
        match lookup_builtin(name) {
            Some(BuiltinType::Type(built)) => return Ok(built),
            Some(BuiltinType::BadLanes(_)) => {
                return Err(LowerError::Unsupported {
                    what: "a type with an invalid lane count",
                    span,
                });
            }
            None => {}
        }
        if let Some(target) = self.aliases.get(name) {
            return Ok(target.clone());
        }
        let args = path
            .args
            .iter()
            .map(|arg| match arg {
                codevar_ocl_parse::GenericArg::Type(inner) => self.resolve_ast_ty(inner),
                codevar_ocl_parse::GenericArg::Lifetime { .. } => Err(LowerError::Unsupported {
                    what: "a lifetime argument",
                    span,
                }),
            })
            .collect::<Result<Vec<Ty>, LowerError>>()?;
        Ok(Ty::Struct {
            name: name.to_string(),
            args,
        })
    }

    /// The semantic type the analyzer gave `expr`, or [`Ty::Error`].
    fn expr_ty(&self, expr: &Expr) -> Ty {
        self.analysis
            .types
            .get(expr.id)
            .cloned()
            .unwrap_or(Ty::Error)
    }

    /// Evaluates `expr` and hands back its value.
    ///
    /// # Errors
    ///
    /// Returns `None` when control flow does not reach past `expr`
    /// (lowering still ran it for its effects), and
    /// [`LowerError::Unsupported`] when `expr` produces no value where
    /// one is required, or when it fails to lower.
    fn value_of(&mut self, expr: &Expr) -> Result<Option<ValueId>, LowerError> {
        match self.lower_expr(expr)? {
            Lowered::Value(value) => Ok(Some(value)),
            Lowered::Void => Err(LowerError::Unsupported {
                what: "an expression that produces no value",
                span: expr.span,
            }),
            Lowered::Dead => Ok(None),
        }
    }

    /// Reads the value of a name binding, loading it from its slot.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the binding has no
    /// home (its scope has ended) or the load is malformed.
    fn binding_value(&mut self, binding: NodeId, span: Span) -> Result<Lowered, LowerError> {
        match self.slots.get(&binding).copied() {
            Some(Binding::Value(value)) => Ok(Lowered::Value(value)),
            Some(Binding::Slot(slot)) => {
                if !self.is_open() {
                    return Ok(Lowered::Dead);
                }
                Ok(Lowered::Value(self.load_ptr(slot)?))
            }
            None => Err(LowerError::Unsupported {
                what: "a binding outside its scope",
                span,
            }),
        }
    }

    /// Loads through `ptr`, producing the pointee value.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when `ptr` is not a pointer,
    /// and a construction error when the module rejects the load.
    fn load_ptr(&mut self, ptr: ValueId) -> Result<ValueId, LowerError> {
        let ptr_ty = self.module.type_of(ptr);
        let Some((_, pointee)) = self.module.pointer_parts(ptr_ty) else {
            return Err(self.unsupported("a load through a non-pointer"));
        };
        self.def_op(pointee, Op::Load { ptr })
    }

    /// Stores `value` through `ptr`, converting it to the pointee type.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when `ptr` is not a pointer
    /// or the value cannot be converted, and a construction error when
    /// the module rejects the store.
    fn store_ptr(&mut self, ptr: ValueId, value: ValueId, span: Span) -> Result<(), LowerError> {
        let ptr_ty = self.module.type_of(ptr);
        let Some((_, pointee)) = self.module.pointer_parts(ptr_ty) else {
            return Err(self.unsupported("a store through a non-pointer"));
        };
        let value = self.convert(value, pointee, span)?;
        self.emit_raw(Inst::none(Op::Store { ptr, value }))
    }

    /// Allocates a stack slot of type `ty` named `base`.
    ///
    /// Slots are collected in [`Lowerer::allocas`] and spliced into the
    /// function's entry block at the end, since SPIR-V requires every
    /// `OpVariable` to precede non-variable instructions there.
    ///
    /// # Errors
    ///
    /// Returns a construction error when the module rejects the value.
    fn alloc_slot(&mut self, ty: TypeId, base: &str) -> Result<ValueId, LowerError> {
        let ptr_ty = self.module.ptr_ty(Storage::Function, ty);
        let slot = self.module.new_inst_value(ptr_ty);
        let _ = self.module.name_value(slot, base);
        self.allocas
            .push(Inst::def(slot, ptr_ty, Op::Variable { init: None }));
        Ok(slot)
    }

    /// Allocates a stack slot of type `ty` initialized to null, for
    /// values that may never be written on some paths (the merge of an
    /// `if` without an `else`).
    ///
    /// # Errors
    ///
    /// Returns a construction error when the module rejects the
    /// initializer.
    fn alloc_slot_null(&mut self, ty: TypeId, base: &str) -> Result<ValueId, LowerError> {
        let init = self
            .module
            .intern_const(ty, ConstValue::Null)
            .map_err(|error| LowerError::Build {
                what: "a null slot initializer",
                error,
            })?;
        let ptr_ty = self.module.ptr_ty(Storage::Function, ty);
        let slot = self.module.new_inst_value(ptr_ty);
        let _ = self.module.name_value(slot, base);
        self.allocas
            .push(Inst::def(slot, ptr_ty, Op::Variable { init: Some(init) }));
        Ok(slot)
    }

    /// Materializes `value` in a fresh stack slot, for places that need
    /// an address but have none (an immutable array or vector parameter
    /// that is indexed or swizzled).
    ///
    /// # Errors
    ///
    /// Returns an error when the slot or the store cannot be built.
    fn spill_value(&mut self, value: ValueId, base: &str, span: Span) -> Result<ValueId, LowerError> {
        let ty = self.module.type_of(value);
        let slot = self.alloc_slot(ty, base)?;
        self.store_ptr(slot, value, span)?;
        Ok(slot)
    }

    /// A boolean constant.
    ///
    /// # Errors
    ///
    /// Returns a construction error when the module rejects it.
    fn const_bool(&mut self, value: bool) -> Result<ValueId, LowerError> {
        let ty = self.module.bool_ty();
        self.module
            .intern_const(ty, ConstValue::Bool(value))
            .map_err(|error| LowerError::Build {
                what: "a boolean constant",
                error,
            })
    }

    /// An integer constant of `ty`, range-checked against its width.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when `ty` is not an integer
    /// or `value` does not fit its width, and a construction error when
    /// the module rejects the payload.
    fn int_const(&mut self, value: i128, ty: TypeId, span: Span) -> Result<ValueId, LowerError> {
        let (bits, signed) = match self.module.ty(ty) {
            IrType::Int { bits, signed } => (*bits, *signed),
            _ => return Err(self.unsupported("an integer constant of a non-integer type")),
        };
        if !int_fits(value, bits, signed) {
            return Err(LowerError::Unsupported {
                what: "an integer constant out of range",
                span,
            });
        }
        let raw = mask_bits(value as u64, bits);
        self.module
            .intern_const(ty, ConstValue::Int(raw))
            .map_err(|error| LowerError::Build {
                what: "an integer constant",
                error,
            })
    }

    /// Lowers a path used as a value.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for qualified paths, names
    /// the analyzer did not resolve, and functions used as values (the
    /// analyzer reports those too, so they only appear when lowering is
    /// fed hand-built tables).
    fn path_value(&mut self, path: &Path, expr: &Expr) -> Result<Lowered, LowerError> {
        if path.segments.len() != 1 {
            return Err(LowerError::Unsupported {
                what: "a qualified path",
                span: path.span,
            });
        }
        if path.segments.is_empty() {
            return Err(LowerError::Unsupported {
                what: "an empty path",
                span: path.span,
            });
        }
        let Some(resolution) = self.analysis.resolutions.get(expr.id) else {
            return Err(LowerError::Unsupported {
                what: "an unresolved name",
                span: path.span,
            });
        };
        match resolution {
            Res::Local { binding } => self.binding_value(*binding, expr.span),
            Res::Function { .. } | Res::Builtin { .. } => Err(LowerError::Unsupported {
                what: "a function used as a value",
                span: path.span,
            }),
        }
    }

    /// Lowers the whole program in three passes: aliases, signatures,
    /// then bodies.
    ///
    /// The passes matter because functions may call each other in any
    /// order — every signature must exist before any body is lowered.
    /// Generic functions are skipped: they are monomorphized before IR
    /// construction, and the analyzer allows calls to them.
    ///
    /// # Errors
    ///
    /// Returns the first failure from any pass.
    fn run(&mut self, program: &Program) -> Result<(), LowerError> {
        self.declare_aliases(program)?;
        self.declare_functions(program)?;
        for item in &program.items {
            if let ItemKind::Fn(function) = &item.kind {
                if has_type_parameters(&function.generics) {
                    continue;
                }
                self.lower_function(function)?;
            }
        }
        Ok(())
    }

    /// Records every non-generic type alias's target.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when an aliased type is
    /// malformed.
    fn declare_aliases(&mut self, program: &Program) -> Result<(), LowerError> {
        for item in &program.items {
            let ItemKind::TypeAlias(alias) = &item.kind else {
                continue;
            };
            if has_type_parameters(&alias.generics) {
                continue;
            }
            let target = self.resolve_ast_ty(&alias.aliased)?;
            self.aliases
                .insert(strip_raw_ident(&alias.name).to_string(), target);
        }
        Ok(())
    }

    /// Declares every non-generic function and records kernels as entry
    /// points.
    ///
    /// # Errors
    ///
    /// Returns an error when a declared type has no IR representation or
    /// the module rejects the function.
    fn declare_functions(&mut self, program: &Program) -> Result<(), LowerError> {
        for item in &program.items {
            let ItemKind::Fn(function) = &item.kind else {
                continue;
            };
            if has_type_parameters(&function.generics) {
                continue;
            }
            let name = strip_raw_ident(&function.name).to_string();
            let ret = match &function.ret {
                Some(ty) => self.resolve_ast_ty(ty)?,
                None => Ty::Void,
            };
            let span = function.span;
            let ir_ret = self.map_ty(&ret, span)?;
            let mut ir_params = Vec::with_capacity(function.params.len());
            for param in &function.params {
                let ty = self.resolve_ast_ty(&param.ty)?;
                ir_params.push(self.map_ty(&ty, span)?);
            }
            let sig = self.module.fn_ty(ir_ret, ir_params.clone());
            let ir = self
                .module
                .add_function(&name, sig, Linkage::External)
                .map_err(|error| LowerError::Build {
                    what: "a function declaration",
                    error,
                })?;
            self.signatures.insert(
                name.clone(),
                FnSig {
                    ir,
                    params: ir_params,
                    ret: ir_ret,
                },
            );
            if self.is_kernel(&name) {
                self.module
                    .set_entry_point(ExecutionModel::Kernel, ir, &name)
                    .map_err(|error| LowerError::Build {
                        what: "an entry point",
                        error,
                    })?;
            }
        }
        Ok(())
    }

    /// True when the analyzer marked `name` as a `#[kernel]` function.
    fn is_kernel(&self, name: &str) -> bool {
        self.analysis
            .declarations
            .iter()
            .any(|declaration| {
                declaration.name == name
                    && matches!(&declaration.kind, DeclKind::Function { kernel: true, .. })
            })
    }

    /// Lowers one function's body.
    ///
    /// # Errors
    ///
    /// Returns an error when the body uses an unsupported construct, or
    /// when a value-returning function can fall off the end of its body
    /// ([`LowerError::MissingReturn`]).
    fn lower_function(&mut self, function: &FnItem) -> Result<(), LowerError> {
        let name = strip_raw_ident(&function.name);
        let signature = self
            .signatures
            .get(name)
            .cloned()
            .ok_or_else(|| LowerError::Unsupported {
                what: "a function without a declared signature",
                span: function.span,
            })?;
        let ret = match &function.ret {
            Some(ty) => self.resolve_ast_ty(ty)?,
            None => Ty::Void,
        };

        self.slots.clear();
        self.allocas.clear();
        self.loops.clear();
        self.func = Some(signature.ir);
        self.func_name = name.to_string();
        self.ret = ret.clone();
        self.current = None;
        self.state = BlockState::Terminated;

        self.module
            .begin_body(signature.ir)
            .map_err(|error| LowerError::Build {
                what: "a function body",
                error,
            })?;
        let entry = self.push_block("entry")?;
        self.enter(entry, true)?;

        for (index, param) in function.params.iter().enumerate() {
            self.bind_param(param, index, &signature)?;
        }

        match self.lower_block(&function.body)? {
            Lowered::Value(value) => {
                if self.is_open() {
                    if default_literals(&ret) == Ty::Void {
                        self.term(Op::Return)?;
                    } else {
                        let value = self.convert(value, signature.ret, function.span)?;
                        self.term(Op::ReturnValue { value })?;
                    }
                }
            }
            Lowered::Void => {
                if self.is_open() {
                    if default_literals(&ret) == Ty::Void {
                        self.term(Op::Return)?;
                    } else {
                        return Err(LowerError::MissingReturn {
                            name: self.func_name.clone(),
                        });
                    }
                }
            }
            Lowered::Dead => {}
        }
        self.finish_current()?;

        // SPIR-V requires every OpVariable to precede non-variable
        // instructions in the entry block, so the collected slots are
        // spliced in only now that the body is complete.
        let allocas = core::mem::take(&mut self.allocas);
        if !allocas.is_empty() {
            let body = self
                .module
                .function_mut(signature.ir)
                .and_then(|definition| definition.body.as_mut())
                .ok_or_else(|| LowerError::Unsupported {
                    what: "a function body",
                    span: function.span,
                })?;
            if let Some(block) = body.first_mut() {
                block.insts.splice(0..0, allocas);
            }
        }

        self.func = None;
        self.current = None;
        self.state = BlockState::Terminated;
        self.slots.clear();
        Ok(())
    }

    /// Binds one parameter: `mut` parameters get a stack slot the body
    /// stores into, plain parameters stay values.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed patterns, missing arguments, or a
    /// store the module rejects.
    fn bind_param(&mut self, param: &Param, index: usize, signature: &FnSig) -> Result<(), LowerError> {
        let argument = self
            .module
            .function(signature.ir)
            .and_then(|definition| definition.args.get(index))
            .copied()
            .ok_or_else(|| LowerError::Unsupported {
                what: "a parameter without an argument slot",
                span: param.span,
            })?;
        match &param.pat.kind {
            PatKind::Wild => Ok(()),
            PatKind::Error => Err(LowerError::Unsupported {
                what: "a malformed parameter pattern",
                span: param.pat.span,
            }),
            PatKind::Ident { name, mutable } => {
                let raw = strip_raw_ident(name);
                let _ = self.module.name_value(argument, raw);
                if *mutable {
                    let ty = signature
                        .params
                        .get(index)
                        .copied()
                        .ok_or_else(|| LowerError::Unsupported {
                            what: "a parameter without a declared type",
                            span: param.span,
                        })?;
                    let slot = self.alloc_slot(ty, raw)?;
                    self.store_ptr(slot, argument, param.span)?;
                    self.slots
                        .insert(param.pat.id, Binding::Slot(slot));
                } else {
                    self.slots
                        .insert(param.pat.id, Binding::Value(argument));
                }
                Ok(())
            }
        }
    }

    /// Lowers a block with its bindings scoped to the block.
    ///
    /// Bindings introduced inside are visible to the tail expression
    /// and gone once the block is done, matching the analyzer.
    ///
    /// # Errors
    ///
    /// Returns the first failure from any statement or the tail.
    fn lower_block(&mut self, block: &Block) -> Result<Lowered, LowerError> {
        let saved = self.slots.clone();
        let lowered = self.lower_block_inner(block);
        self.slots = saved;
        lowered
    }

    /// The unscoped half of [`Lowerer::lower_block`].
    fn lower_block_inner(&mut self, block: &Block) -> Result<Lowered, LowerError> {
        for stmt in &block.stmts {
            if !self.is_open() {
                return Ok(Lowered::Dead);
            }
            self.lower_stmt(stmt)?;
        }
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        match &block.tail {
            Some(tail) => self.lower_expr(tail),
            None => Ok(Lowered::Void),
        }
    }

    /// Lowers one statement.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed statements and whatever the
    /// statement's expression raises.
    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<(), LowerError> {
        match &stmt.kind {
            StmtKind::Let(let_stmt) => self.lower_let(let_stmt),
            StmtKind::Expr(expr) => {
                self.lower_expr(expr)?;
                Ok(())
            }
            StmtKind::Empty => Ok(()),
            StmtKind::Error => Err(LowerError::Unsupported {
                what: "a malformed statement",
                span: stmt.span,
            }),
        }
    }

    /// Lowers a `let` binding to a stack slot or a plain value.
    ///
    /// `mut` bindings become slots so they can be assigned and indexed;
    /// immutable bindings keep the value they were initialized with, so
    /// writing to them is rejected by [`Lowerer::place`].  The
    /// initializer is evaluated before the binding enters scope, so
    /// `let x = x;` reads the outer `x`.
    ///
    /// # Errors
    ///
    /// Returns an error when the binding has neither type nor
    /// initializer, its type has no IR representation, or its pattern is
    /// malformed.
    fn lower_let(&mut self, let_stmt: &LetStmt) -> Result<(), LowerError> {
        let span = let_stmt.span;
        let init = match (&let_stmt.ty, &let_stmt.init) {
            (None, None) => {
                return Err(LowerError::Unsupported {
                    what: "a `let` without a type or initializer",
                    span,
                });
            }
            (_, Some(expr)) => self.lower_expr(expr)?,
            (Some(_), None) => Lowered::Void,
        };
        match &let_stmt.pat.kind {
            PatKind::Wild => Ok(()),
            PatKind::Error => Err(LowerError::Unsupported {
                what: "a malformed pattern",
                span: let_stmt.pat.span,
            }),
            PatKind::Ident { name, mutable } => {
                let binding_ty = match &let_stmt.ty {
                    Some(ty) => self.resolve_ast_ty(ty)?,
                    None => {
                        let Some(expr) = &let_stmt.init else {
                            return Err(LowerError::Unsupported {
                                what: "a `let` without a type or initializer",
                                span,
                            });
                        };
                        let ty = self.expr_ty(expr);
                        if matches!(ty, Ty::Void) {
                            return Err(LowerError::Unsupported {
                                what: "a `let` bound to an expression with no value",
                                span,
                            });
                        }
                        default_literals(&ty)
                    }
                };
                let ir_ty = self.map_ty(&binding_ty, span)?;
                let raw = strip_raw_ident(name);
                match init {
                    Lowered::Value(value) if *mutable => {
                        let slot = self.alloc_slot(ir_ty, raw)?;
                        self.store_ptr(slot, value, span)?;
                        self.slots
                            .insert(let_stmt.pat.id, Binding::Slot(slot));
                    }
                    Lowered::Value(value) => {
                        self.slots
                            .insert(let_stmt.pat.id, Binding::Value(value));
                    }
                    Lowered::Void => {
                        // No value ever lands here (an `if` without an
                        // `else` whose arms diverge); a null slot keeps
                        // later reads defined.
                        let slot = self.alloc_slot_null(ir_ty, raw)?;
                        self.slots
                            .insert(let_stmt.pat.id, Binding::Slot(slot));
                    }
                    Lowered::Dead => {
                        let slot = self.alloc_slot(ir_ty, raw)?;
                        self.slots
                            .insert(let_stmt.pat.id, Binding::Slot(slot));
                    }
                }
                Ok(())
            }
        }
    }

    /// Lowers an expression to a value, a void result, or `Dead`.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for constructs without an IR
    /// representation, plus whatever the matching arm raises.
    fn lower_expr(&mut self, expr: &Expr) -> Result<Lowered, LowerError> {
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        match &expr.kind {
            ExprKind::Error => Err(LowerError::UnsupportedType {
                ty: Ty::Error,
                span: expr.span,
            }),
            ExprKind::Literal { text, kind } => self.eval_literal(expr, text, kind),
            ExprKind::Bool(value) => Ok(Lowered::Value(self.const_bool(*value)?)),
            ExprKind::Path(path) => self.path_value(path, expr),
            ExprKind::Unary { op, expr: inner } => self.lower_unary(expr, *op, inner),
            ExprKind::Binary { op, lhs, rhs } => self.lower_binary(expr, *op, lhs, rhs),
            ExprKind::Assign { op, lhs, rhs } => self.lower_assign(op.as_ref(), lhs, rhs),
            ExprKind::Call { callee, args } => self.lower_call(expr, callee, args),
            ExprKind::Index { expr: base, index } => self.lower_index(expr, base, index),
            ExprKind::Field { expr: base, name } => self.lower_field(expr, base, name),
            ExprKind::Try { .. } => Err(LowerError::Unsupported {
                what: "the `?` operator",
                span: expr.span,
            }),
            ExprKind::Cast { expr: inner, ty } => self.lower_cast(expr, inner, ty),
            ExprKind::Range { .. } => Err(LowerError::Unsupported {
                what: "a range used as a value",
                span: expr.span,
            }),
            ExprKind::Block(block) => self.lower_block(block),
            ExprKind::If {
                cond,
                then,
                else_branch,
            } => self.lower_if(expr, cond, then, else_branch.as_deref()),
            ExprKind::While { cond, body } => self.lower_while(cond, body),
            ExprKind::Loop { body } => self.lower_loop_body(body),
            ExprKind::For { pat, iter, body } => self.lower_for(expr, pat, iter, body),
            ExprKind::Return { expr: value } => self.lower_return(value.as_deref()),
            ExprKind::Break { expr: value } => self.lower_break(value.as_deref()),
            ExprKind::Continue => self.lower_continue(),
            ExprKind::Tuple(elems) => self.lower_tuple(expr, elems),
            ExprKind::Array(elems) => self.lower_array(expr, elems),
        }
    }

    /// Lowers a binary expression: short-circuit, comparison, then
    /// arithmetic, shift, and bitwise operators.
    ///
    /// # Errors
    ///
    /// Propagates the chosen arm's failures.
    fn lower_binary(
        &mut self,
        whole: &Expr,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<Lowered, LowerError> {
        match op {
            BinaryOp::And | BinaryOp::Or => self.short_circuit(whole, op, lhs, rhs),
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                self.lower_compare(whole, op, lhs, rhs)
            }
            _ => self.emit_binop(whole, op, lhs, rhs),
        }
    }

    /// Lowers a unary expression.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the operator has no IR
    /// representation for the operand's type, and propagates operand
    /// failures.
    fn lower_unary(&mut self, whole: &Expr, op: UnaryOp, inner: &Expr) -> Result<Lowered, LowerError> {
        match op {
            UnaryOp::Deref => {
                let Some(ptr) = self.value_of(inner)? else {
                    return Ok(Lowered::Dead);
                };
                if !self.is_open() {
                    return Ok(Lowered::Dead);
                }
                if self
                    .module
                    .pointer_parts(self.module.type_of(ptr))
                    .is_none()
                {
                    return Err(LowerError::Unsupported {
                        what: "a dereference of a non-pointer value",
                        span: whole.span,
                    });
                }
                Ok(Lowered::Value(self.load_ptr(ptr)?))
            }
            UnaryOp::AddrOf { .. } => match self.place(inner, true)? {
                Place::Ptr(ptr) => Ok(Lowered::Value(ptr)),
                Place::NotPlace => Err(LowerError::Unsupported {
                    what: "the address of a temporary value",
                    span: whole.span,
                }),
                Place::Dead => Ok(Lowered::Dead),
            },
            UnaryOp::Neg | UnaryOp::Not => {
                let Some(value) = self.value_of(inner)? else {
                    return Ok(Lowered::Dead);
                };
                if !self.is_open() {
                    return Ok(Lowered::Dead);
                }
                // The operation follows the operand's own type, not
                // the expression's: `-3000000000` computes at `long`
                // while the analyzer defaults the result to `int`,
                // and the IR requires operand and result to agree.
                let result_ty = self.module.type_of(value);
                let unop = match op {
                    UnaryOp::Neg if ir_is_float(&self.module, result_ty) => UnOp::FNegate,
                    UnaryOp::Neg
                        if ir_is_int(&self.module, result_ty)
                            && ir_signedness(&self.module, result_ty) == Some(false) =>
                    {
                        return Err(LowerError::Unsupported {
                            what: "negation of an unsigned integer",
                            span: whole.span,
                        });
                    }
                    UnaryOp::Neg if ir_is_int(&self.module, result_ty) => UnOp::SNegate,
                    UnaryOp::Neg => {
                        return Err(LowerError::Unsupported {
                            what: "negation of a non-numeric value",
                            span: whole.span,
                        });
                    }
                    UnaryOp::Not if ir_is_bool(&self.module, result_ty) => UnOp::LogicalNot,
                    UnaryOp::Not if ir_is_int(&self.module, result_ty) => UnOp::Not,
                    UnaryOp::Not => {
                        return Err(LowerError::Unsupported {
                            what: "bitwise negation of a non-integer value",
                            span: whole.span,
                        });
                    }
                    _ => {
                        return Err(LowerError::Unsupported {
                            what: "this unary operator",
                            span: whole.span,
                        });
                    }
                };
                Ok(Lowered::Value(self.def_op(
                    result_ty,
                    Op::Unary {
                        op: unop,
                        operand: value,
                    },
                )?))
            }
        }
    }

    /// Resolves `expr` to a writable address.
    ///
    /// Immutable bindings have no address: they stay values, so
    /// assignment to them is rejected here rather than stored into a
    /// stray copy.  With `allow_spill` a value is materialized in a
    /// temporary slot instead, which reads need but writes must not use.
    ///
    /// # Errors
    ///
    /// Returns [`Place::NotPlace`] when `expr` names something that
    /// cannot be written, or [`LowerError::Unsupported`] for malformed
    /// expressions.
    fn place(&mut self, expr: &Expr, allow_spill: bool) -> Result<Place, LowerError> {
        if !self.is_open() {
            return Ok(Place::Dead);
        }
        match &expr.kind {
            ExprKind::Path(path) => {
                if path.segments.len() != 1 {
                    return Err(LowerError::Unsupported {
                        what: "a qualified path",
                        span: path.span,
                    });
                }
                if path.segments.is_empty() {
                    return Err(LowerError::Unsupported {
                        what: "an empty path",
                        span: path.span,
                    });
                }
                let Some(resolution) = self.analysis.resolutions.get(expr.id) else {
                    return Err(LowerError::Unsupported {
                        what: "an unresolved name",
                        span: path.span,
                    });
                };
                let Res::Local { binding } = resolution else {
                    return Err(LowerError::Unsupported {
                        what: "assignment to a function",
                        span: expr.span,
                    });
                };
                match self.slots.get(binding).copied() {
                    Some(Binding::Slot(slot)) => Ok(Place::Ptr(slot)),
                    Some(Binding::Value(value)) if allow_spill => {
                        Ok(Place::Ptr(self.spill_value(value, "temp", expr.span)?))
                    }
                    Some(Binding::Value(_)) => Ok(Place::NotPlace),
                    None => Err(LowerError::Unsupported {
                        what: "a binding outside its scope",
                        span: expr.span,
                    }),
                }
            }
            ExprKind::Unary {
                op: UnaryOp::Deref,
                expr: inner,
            } => {
                let Some(ptr) = self.value_of(inner)? else {
                    return Ok(Place::Dead);
                };
                if self
                    .module
                    .pointer_parts(self.module.type_of(ptr))
                    .is_some()
                {
                    Ok(Place::Ptr(ptr))
                } else {
                    Err(LowerError::Unsupported {
                        what: "a dereference of a non-pointer value",
                        span: expr.span,
                    })
                }
            }
            ExprKind::Index { expr: base, index } => self.index_place(base, index, allow_spill, expr.span),
            ExprKind::Field { .. } => Err(LowerError::Unsupported {
                what: "assignment to a field or swizzle lane",
                span: expr.span,
            }),
            ExprKind::Error => Err(LowerError::UnsupportedType {
                ty: Ty::Error,
                span: expr.span,
            }),
            _ => {
                if allow_spill {
                    let Some(value) = self.value_of(expr)? else {
                        return Ok(Place::Dead);
                    };
                    Ok(Place::Ptr(self.spill_value(value, "temp", expr.span)?))
                } else {
                    Ok(Place::NotPlace)
                }
            }
        }
    }

    /// Resolves `base[index]` to an address.
    ///
    /// A `*T` base keeps its own indirection, so `p[i]` is pointer
    /// arithmetic ([`Op::PtrAccessChain`]) with the element offset as
    /// the only index.  An array, vector, or reference base addresses
    /// through the aggregate instead ([`Op::AccessChain`]); one
    /// remaining reference layer is loaded so the chain starts at the
    /// aggregate's pointer.
    ///
    /// # Errors
    ///
    /// Returns [`Place::NotPlace`] when `base` cannot be addressed, or
    /// [`LowerError::Unsupported`] when it cannot be indexed.
    fn index_place(
        &mut self,
        base: &Expr,
        index: &Expr,
        allow_spill: bool,
        span: Span,
    ) -> Result<Place, LowerError> {
        if !self.is_open() {
            return Ok(Place::Dead);
        }
        let base_ty = default_literals(&self.expr_ty(base));
        let (refs, rest) = split_refs(&base_ty);
        let address = match &rest {
            Ty::Ptr { .. } => {
                let Some(mut ptr) = self.value_of(base)? else {
                    return Ok(Place::Dead);
                };
                for _ in 0..refs {
                    if !self.is_open() {
                        return Ok(Place::Dead);
                    }
                    ptr = self.load_ptr(ptr)?;
                }
                if self
                    .module
                    .pointer_parts(self.module.type_of(ptr))
                    .is_none()
                {
                    return Err(LowerError::Unsupported {
                        what: "an index through a non-pointer value",
                        span,
                    });
                }
                ptr
            }
            Ty::Array { .. } | Ty::Vector { .. } => {
                let place = if refs == 0 {
                    self.place(base, allow_spill)?
                } else {
                    // The reference value is already a pointer: load the
                    // remaining layers so one pointer to the aggregate
                    // is left for the chain to walk.
                    let Some(mut ptr) = self.value_of(base)? else {
                        return Ok(Place::Dead);
                    };
                    for _ in 1..refs {
                        if !self.is_open() {
                            return Ok(Place::Dead);
                        }
                        ptr = self.load_ptr(ptr)?;
                    }
                    Place::Ptr(ptr)
                };
                match place {
                    Place::Ptr(ptr) => ptr,
                    Place::NotPlace => return Ok(Place::NotPlace),
                    Place::Dead => return Ok(Place::Dead),
                }
            }
            _ => {
                return Err(LowerError::Unsupported {
                    what: "an index into this value",
                    span,
                });
            }
        };
        if !self.is_open() {
            return Ok(Place::Dead);
        }
        let Some(index_value) = self.value_of(index)? else {
            return Ok(Place::Dead);
        };
        if !self.is_open() {
            return Ok(Place::Dead);
        }
        if matches!(rest, Ty::Ptr { .. }) {
            let result_ty = self.module.type_of(address);
            let value = self.def_op(
                result_ty,
                Op::PtrAccessChain {
                    base: address,
                    indices: vec![index_value],
                },
            )?;
            return Ok(Place::Ptr(value));
        }
        let ptr_ty = self.module.type_of(address);
        let (storage, pointee) =
            self.module
                .pointer_parts(ptr_ty)
                .ok_or_else(|| LowerError::Unsupported {
                    what: "an index into a non-pointer address",
                    span,
                })?;
        let element = match self.module.ty(pointee) {
            IrType::Array { elem, .. } | IrType::Vector { elem, .. } => *elem,
            _ => {
                return Err(LowerError::Unsupported {
                    what: "an index into this value",
                    span,
                });
            }
        };
        let result_ty = self.module.ptr_ty(storage, element);
        let value = self.def_op(
            result_ty,
            Op::AccessChain {
                base: address,
                indices: vec![index_value],
            },
        )?;
        Ok(Place::Ptr(value))
    }

    /// Lowers an index expression by reading through its address.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the value has no
    /// address, plus [`Lowerer::index_place`]'s failures.
    fn lower_index(&mut self, whole: &Expr, base: &Expr, index: &Expr) -> Result<Lowered, LowerError> {
        match self.index_place(base, index, true, whole.span)? {
            Place::Dead => Ok(Lowered::Dead),
            Place::NotPlace => Err(LowerError::Unsupported {
                what: "an index into a value with no address",
                span: whole.span,
            }),
            Place::Ptr(ptr) => {
                if !self.is_open() {
                    return Ok(Lowered::Dead);
                }
                Ok(Lowered::Value(self.load_ptr(ptr)?))
            }
        }
    }

    /// Lowers a field or swizzle read.
    ///
    /// The base is loaded through every pointer and reference layer the
    /// type wraps, then the named lanes are extracted from the vector.
    /// Struct and tuple fields have no IR representation yet.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::UnsupportedType`] when the aggregate is
    /// not a vector, and [`LowerError::Unsupported`] for an invalid
    /// swizzle.
    fn lower_field(&mut self, whole: &Expr, base: &Expr, name: &str) -> Result<Lowered, LowerError> {
        let span = whole.span;
        let base_ty = default_literals(&self.expr_ty(base));
        let (layers, rest) = split_indirection(&base_ty);
        let (elem, lanes) = match &rest {
            Ty::Vector { elem, lanes } => (*elem, *lanes),
            other => {
                let other = other.clone();
                return Err(LowerError::UnsupportedType { ty: other, span });
            }
        };
        let field = strip_raw_ident(name);
        let Some(indices) = swizzle_lanes(field, lanes) else {
            return Err(LowerError::Unsupported {
                what: "an invalid swizzle",
                span,
            });
        };
        let Some(mut value) = self.value_of(base)? else {
            return Ok(Lowered::Dead);
        };
        for _ in 0..layers {
            if !self.is_open() {
                return Ok(Lowered::Dead);
            }
            value = self.load_ptr(value)?;
        }
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let element_ty = self.map_scalar(elem, span)?;
        if indices.len() == 1 {
            let lane = indices.first().copied().unwrap_or(0);
            let out = self.def_op(
                element_ty,
                Op::CompositeExtract {
                    composite: value,
                    indices: vec![lane],
                },
            )?;
            return Ok(Lowered::Value(out));
        }
        let count = u8::try_from(indices.len()).unwrap_or(u8::MAX);
        let vector_ty = self.map_ty(&Ty::vector(elem, count), span)?;
        let mut constituents = Vec::with_capacity(indices.len());
        for &lane in &indices {
            let part = self.def_op(
                element_ty,
                Op::CompositeExtract {
                    composite: value,
                    indices: vec![lane],
                },
            )?;
            constituents.push(part);
        }
        Ok(Lowered::Value(
            self.def_op(vector_ty, Op::CompositeConstruct { constituents })?,
        ))
    }

    /// Lowers an assignment, plain or compound.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the target cannot be
    /// written — an immutable binding, a field, or a temporary.
    fn lower_assign(&mut self, op: Option<&BinaryOp>, lhs: &Expr, rhs: &Expr) -> Result<Lowered, LowerError> {
        let target = match self.place(lhs, false)? {
            Place::Ptr(ptr) => ptr,
            Place::NotPlace => {
                return Err(LowerError::Unsupported {
                    what: "assignment to an immutable binding",
                    span: lhs.span,
                });
            }
            Place::Dead => return Ok(Lowered::Dead),
        };
        let Some(value) = self.value_of(rhs)? else {
            return Ok(Lowered::Dead);
        };
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let value = match op {
            None => value,
            Some(op) => {
                let current = self.load_ptr(target)?;
                let lhs_ty = self.expr_ty(lhs);
                let rhs_ty = self.expr_ty(rhs);
                self.combine_values(*op, current, value, &lhs_ty, &rhs_ty, rhs.span)?
            }
        };
        self.store_ptr(target, value, rhs.span)?;
        Ok(Lowered::Void)
    }

    /// Converts a computed value to the type the analyzer gave the
    /// whole expression, which is what callers read it as.  Literal
    /// placeholders skip the conversion: their default applies at the
    /// use site, and the value may already be wider than the default.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::UnsupportedType`] when the expression type
    /// has no IR representation, and propagates conversion failures.
    fn finish_value(&mut self, whole: &Expr, value: ValueId) -> Result<Lowered, LowerError> {
        let ty = self.expr_ty(whole);
        if matches!(ty, Ty::IntLit(_) | Ty::FloatLit(_)) {
            // The value already sits at the operand-unified type,
            // which can be wider than the placeholder's default: in
            // `3000000000 + 1` the analyzer records `IntLit(1)` while
            // the sum computes at `long`.  Narrowing here would
            // truncate before any consumer could widen it back.
            return Ok(Lowered::Value(value));
        }
        let target = self.map_ty(&ty, whole.span)?;
        let value = self.convert(value, target, whole.span)?;
        Ok(Lowered::Value(value))
    }

    /// The type both operands of `op` materialize at, mirroring the
    /// analyzer: a shift takes the non-literal side's type, everything
    /// else unifies.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the operands are
    /// incompatible or an operand literal is out of range.
    fn operand_target(&self, op: BinaryOp, left: &Ty, right: &Ty, span: Span) -> Result<Ty, LowerError> {
        if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
            let unified = if matches!(left, Ty::IntLit(_)) && !matches!(right, Ty::IntLit(_)) {
                right.clone()
            } else {
                left.clone()
            };
            return normalize_operand_ty(&unified, left, right).ok_or_else(|| LowerError::Unsupported {
                what: "operands that cannot unify",
                span,
            });
        }
        unify_operand_types(left, right).ok_or_else(|| LowerError::Unsupported {
            what: "operands that cannot unify",
            span,
        })
    }

    /// Emits `op` over two evaluated operands at their unified type.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the operator has no IR
    /// form for the operand type, and propagates conversion failures.
    fn combine_values(
        &mut self,
        op: BinaryOp,
        left: ValueId,
        right: ValueId,
        left_ty: &Ty,
        right_ty: &Ty,
        span: Span,
    ) -> Result<ValueId, LowerError> {
        let target = self.operand_target(op, left_ty, right_ty, span)?;
        let ir_ty = self.map_ty(&target, span)?;
        let left = self.convert(left, ir_ty, span)?;
        let right = self.convert(right, ir_ty, span)?;
        let float = ir_is_float(&self.module, ir_ty);
        let boolean = ir_is_bool(&self.module, ir_ty);
        let signed = ir_signedness(&self.module, ir_ty).unwrap_or(true);
        match op {
            BinaryOp::And | BinaryOp::Or if boolean => {
                let op = if matches!(op, BinaryOp::And) {
                    BinOp::LogicalAnd
                } else {
                    BinOp::LogicalOr
                };
                self.def_op(
                    ir_ty,
                    Op::Binary {
                        op,
                        lhs: left,
                        rhs: right,
                    },
                )
            }
            BinaryOp::And | BinaryOp::Or => Err(LowerError::Unsupported {
                what: "logical operators on non-boolean values",
                span,
            }),
            BinaryOp::Shl | BinaryOp::Shr if ir_is_int(&self.module, ir_ty) => {
                let op = if matches!(op, BinaryOp::Shl) {
                    BinOp::ShiftLeftLogical
                } else if signed {
                    BinOp::ShiftRightArithmetic
                } else {
                    BinOp::ShiftRightLogical
                };
                self.def_op(
                    ir_ty,
                    Op::Binary {
                        op,
                        lhs: left,
                        rhs: right,
                    },
                )
            }
            BinaryOp::Shl | BinaryOp::Shr => Err(LowerError::Unsupported {
                what: "a shift of a non-integer value",
                span,
            }),
            BinaryOp::BitAnd | BinaryOp::BitXor | BinaryOp::BitOr if boolean => match op {
                BinaryOp::BitAnd => self.def_op(
                    ir_ty,
                    Op::Binary {
                        op: BinOp::LogicalAnd,
                        lhs: left,
                        rhs: right,
                    },
                ),
                BinaryOp::BitOr => self.def_op(
                    ir_ty,
                    Op::Binary {
                        op: BinOp::LogicalOr,
                        lhs: left,
                        rhs: right,
                    },
                ),
                _ => self.bool_xor(left, right, ir_ty),
            },
            BinaryOp::BitAnd | BinaryOp::BitXor | BinaryOp::BitOr if float => Err(LowerError::Unsupported {
                what: "a bitwise operation on floating-point values",
                span,
            }),
            BinaryOp::BitAnd | BinaryOp::BitXor | BinaryOp::BitOr => {
                let op = match op {
                    BinaryOp::BitAnd => BinOp::BitwiseAnd,
                    BinaryOp::BitOr => BinOp::BitwiseOr,
                    _ => BinOp::BitwiseXor,
                };
                self.def_op(
                    ir_ty,
                    Op::Binary {
                        op,
                        lhs: left,
                        rhs: right,
                    },
                )
            }
            other => {
                let op = arith_binop(other, float, signed).ok_or_else(|| LowerError::Unsupported {
                    what: "a binary operator without an IR form",
                    span,
                })?;
                self.def_op(
                    ir_ty,
                    Op::Binary {
                        op,
                        lhs: left,
                        rhs: right,
                    },
                )
            }
        }
    }

    /// Lowers arithmetic, shift, and bitwise operators and converts the
    /// result to the expression's own type.
    ///
    /// # Errors
    ///
    /// Propagates operand and [`Lowerer::combine_values`] failures.
    fn emit_binop(
        &mut self,
        whole: &Expr,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<Lowered, LowerError> {
        let left_ty = self.expr_ty(lhs);
        let right_ty = self.expr_ty(rhs);
        let Some(left) = self.value_of(lhs)? else {
            return Ok(Lowered::Dead);
        };
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let Some(right) = self.value_of(rhs)? else {
            return Ok(Lowered::Dead);
        };
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let value = self.combine_values(op, left, right, &left_ty, &right_ty, whole.span)?;
        self.finish_value(whole, value)
    }

    /// `left != right` for booleans, expanded without a `BoolXor`
    /// opcode: `(l && !r) || (!l && r)`.
    ///
    /// # Errors
    ///
    /// Returns a construction error when the module rejects an operand.
    fn bool_xor(&mut self, left: ValueId, right: ValueId, ty: TypeId) -> Result<ValueId, LowerError> {
        let not_right = self.def_op(
            ty,
            Op::Unary {
                op: UnOp::LogicalNot,
                operand: right,
            },
        )?;
        let not_left = self.def_op(
            ty,
            Op::Unary {
                op: UnOp::LogicalNot,
                operand: left,
            },
        )?;
        let first = self.def_op(
            ty,
            Op::Binary {
                op: BinOp::LogicalAnd,
                lhs: left,
                rhs: not_right,
            },
        )?;
        let second = self.def_op(
            ty,
            Op::Binary {
                op: BinOp::LogicalAnd,
                lhs: not_left,
                rhs: right,
            },
        )?;
        self.def_op(
            ty,
            Op::Binary {
                op: BinOp::LogicalOr,
                lhs: first,
                rhs: second,
            },
        )
    }

    /// Lowers a comparison, folding a vector mask down to the scalar
    /// `bool` the analyzer types every comparison as.
    ///
    /// Booleans compare through [`Lowerer::bool_xor`] because IR has no
    /// integer comparison on `bool`; ordering a `bool` has no sensible
    /// IR form either and is rejected.  A vector comparison produces a
    /// `boolN` mask, which folds with `all` (or `any` for `!=`).
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for pointer comparisons,
    /// composite operands, and ordered boolean comparisons.
    fn lower_compare(
        &mut self,
        whole: &Expr,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<Lowered, LowerError> {
        let span = whole.span;
        let left_ty = default_literals(&self.expr_ty(lhs));
        let right_ty = default_literals(&self.expr_ty(rhs));
        let target = unify_operand_types(&left_ty, &right_ty).ok_or_else(|| LowerError::Unsupported {
            what: "a comparison between incompatible types",
            span,
        })?;
        let (element, lanes) = match &target {
            Ty::Scalar(scalar) => (*scalar, None),
            Ty::Vector { elem, lanes } => (*elem, Some(*lanes)),
            Ty::Ptr { .. } => {
                return Err(LowerError::Unsupported {
                    what: "a pointer comparison",
                    span,
                });
            }
            _ => {
                return Err(LowerError::Unsupported {
                    what: "a comparison of composite values",
                    span,
                });
            }
        };
        let ir_ty = self.map_ty(&target, span)?;
        let Some(left) = self.value_of(lhs)? else {
            return Ok(Lowered::Dead);
        };
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let Some(right) = self.value_of(rhs)? else {
            return Ok(Lowered::Dead);
        };
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let left = self.convert(left, ir_ty, lhs.span)?;
        let right = self.convert(right, ir_ty, rhs.span)?;
        let bool_ty = self.module.bool_ty();
        let mask_ty = match lanes {
            Some(lanes) => self.module.vector_ty(bool_ty, u32::from(lanes)),
            None => bool_ty,
        };
        let mask = if matches!(element, Scalar::Bool) {
            match op {
                BinaryOp::Eq => {
                    let xor = self.bool_xor(left, right, ir_ty)?;
                    self.def_op(
                        ir_ty,
                        Op::Unary {
                            op: UnOp::LogicalNot,
                            operand: xor,
                        },
                    )?
                }
                BinaryOp::Ne => self.bool_xor(left, right, ir_ty)?,
                _ => {
                    return Err(LowerError::Unsupported {
                        what: "an ordered comparison of booleans",
                        span,
                    });
                }
            }
        } else {
            let float = matches!(element, Scalar::F16 | Scalar::F32 | Scalar::F64);
            let code = cmp_op(op, float, element.is_signed_int()).ok_or_else(|| LowerError::Unsupported {
                what: "a comparison this IR cannot express",
                span,
            })?;
            self.def_op(
                mask_ty,
                Op::Compare {
                    op: code,
                    lhs: left,
                    rhs: right,
                },
            )?
        };
        match lanes {
            Some(lanes) => {
                let any = matches!(op, BinaryOp::Ne);
                Ok(Lowered::Value(self.bool_vector_fold(mask, lanes, any, span)?))
            }
            None => Ok(Lowered::Value(mask)),
        }
    }

    /// Lowers `&&` and `||`.
    ///
    /// Scalar operands short-circuit through control flow; a vector
    /// mask cannot drive structured branches, so vector operands
    /// evaluate eagerly and the result folds back to a scalar `bool`.
    ///
    /// # Errors
    ///
    /// Propagates operand failures.
    fn short_circuit(
        &mut self,
        whole: &Expr,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<Lowered, LowerError> {
        let left_ty = default_literals(&self.expr_ty(lhs));
        let right_ty = default_literals(&self.expr_ty(rhs));
        let vector = matches!(left_ty, Ty::Vector { .. }) || matches!(right_ty, Ty::Vector { .. });
        if vector {
            let Some(left) = self.value_of(lhs)? else {
                return Ok(Lowered::Dead);
            };
            if !self.is_open() {
                return Ok(Lowered::Dead);
            }
            let Some(right) = self.value_of(rhs)? else {
                return Ok(Lowered::Dead);
            };
            if !self.is_open() {
                return Ok(Lowered::Dead);
            }
            let value = self.combine_values(op, left, right, &left_ty, &right_ty, whole.span)?;
            return self.finish_value(whole, value);
        }
        self.short_circuit_conditional(op, lhs, rhs)
    }

    /// Emits the short-circuit CFG for scalar `&&` and `||`.
    ///
    /// The left operand runs first and is spilled to a slot; a
    /// conditional branch decides whether the right operand runs at
    /// all, and both paths store the result before the merge loads it.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when an operand produces no
    /// value where one is required.
    fn short_circuit_conditional(
        &mut self,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<Lowered, LowerError> {
        let Some(left) = self.value_of(lhs)? else {
            return Ok(Lowered::Dead);
        };
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let bool_ty = self.module.bool_ty();
        let left = self.convert(left, bool_ty, lhs.span)?;
        let slot = self.alloc_slot(bool_ty, "logic")?;
        self.store_ptr(slot, left, lhs.span)?;

        let evaluate_block = self.push_block("logic.rhs")?;
        let short_block = self.push_block("logic.short")?;
        let merge_block = self.push_block("logic.merge")?;
        self.emit_raw(Inst::none(Op::SelectionMerge {
            target: merge_block,
            control: 0,
        }))?;
        let (then, other) = match op {
            BinaryOp::And => (evaluate_block, short_block),
            _ => (short_block, evaluate_block),
        };
        self.term(Op::BranchConditional {
            cond: left,
            then,
            other,
        })?;

        // Short-circuited path: the left operand already decided.
        self.enter(short_block, true)?;
        let short = self.const_bool(op == BinaryOp::Or)?;
        self.store_ptr(slot, short, lhs.span)?;
        self.term(Op::Branch { target: merge_block })?;

        // Right-operand path: store whatever it evaluates to.  A
        // diverging operand leaves its block terminated, and the merge
        // stays reachable through the short-circuit path alone.
        self.enter(evaluate_block, true)?;
        if let Some(right) = self.value_of(rhs)? {
            if self.is_open() {
                let right = self.convert(right, bool_ty, rhs.span)?;
                self.store_ptr(slot, right, rhs.span)?;
                self.term(Op::Branch { target: merge_block })?;
            }
        }
        self.enter(merge_block, true)?;
        Ok(Lowered::Value(self.load_ptr(slot)?))
    }

    /// Lowers an `if` expression: condition first, arms, then merge.
    ///
    /// The condition runs before any block exists, so a diverging
    /// condition leaves no structure behind.  A value-producing `if`
    /// stores each falling arm's result in a null-initialized slot, so
    /// an `if` without an `else` reads the implicit arm's default.
    ///
    /// # Errors
    ///
    /// Propagates the condition's and both arms' failures.
    fn lower_if(
        &mut self,
        whole: &Expr,
        cond: &Expr,
        then: &Block,
        else_branch: Option<&Expr>,
    ) -> Result<Lowered, LowerError> {
        let Some(condition) = self.value_of(cond)? else {
            return Ok(Lowered::Dead);
        };
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let boolean = self.module.bool_ty();
        let condition = self.convert(condition, boolean, cond.span)?;

        let slot = if self.expr_ty(whole).is_unit_like() {
            None
        } else {
            let result_ty = self.map_ty(&self.expr_ty(whole), whole.span)?;
            Some(self.alloc_slot_null(result_ty, "if")?)
        };

        let then_block = self.push_block("if.then")?;
        let else_block = match else_branch {
            Some(_) => Some(self.push_block("if.else")?),
            None => None,
        };
        let merge_block = self.push_block("if.merge")?;
        self.emit_raw(Inst::none(Op::SelectionMerge {
            target: merge_block,
            control: 0,
        }))?;
        self.term(Op::BranchConditional {
            cond: condition,
            then: then_block,
            other: else_block.unwrap_or(merge_block),
        })?;

        self.enter(then_block, true)?;
        let then_result = self.lower_block(then)?;
        let then_fell = self.is_open();
        if let Lowered::Value(value) = then_result
            && let Some(slot) = slot
        {
            self.store_ptr(slot, value, then.span)?;
        }
        if self.is_open() {
            self.term(Op::Branch { target: merge_block })?;
        }

        let else_fell = match (else_block, else_branch) {
            (Some(block), Some(expr)) => {
                self.enter(block, true)?;
                let result = self.lower_expr(expr)?;
                let fell = self.is_open();
                if let Lowered::Value(value) = result
                    && let Some(slot) = slot
                {
                    self.store_ptr(slot, value, expr.span)?;
                }
                if self.is_open() {
                    self.term(Op::Branch { target: merge_block })?;
                }
                fell
            }
            _ => true,
        };

        let merge_open = match else_branch {
            Some(_) => then_fell || else_fell,
            None => true,
        };
        self.enter(merge_block, merge_open)?;
        match slot {
            Some(slot) if self.is_open() => Ok(Lowered::Value(self.load_ptr(slot)?)),
            Some(_) => Ok(Lowered::Dead),
            None if self.is_open() => Ok(Lowered::Void),
            None => Ok(Lowered::Dead),
        }
    }

    /// Lowers a `while` loop: a condition re-evaluated on every
    /// iteration, then the body.
    ///
    /// # Errors
    ///
    /// Propagates the condition's and body's failures.
    fn lower_while(&mut self, cond: &Expr, body: &Block) -> Result<Lowered, LowerError> {
        let header = move |lowerer: &mut Self| {
            let Some(value) = lowerer.value_of(cond)? else {
                return Ok(HeaderResult::Dead);
            };
            if !lowerer.is_open() {
                return Ok(HeaderResult::Dead);
            }
            let boolean = lowerer.module.bool_ty();
            let value = lowerer.convert(value, boolean, cond.span)?;
            Ok(HeaderResult::Cond(value))
        };
        self.lower_loop(header, |_| Ok(()), body, |_| Ok(()))
    }

    /// Lowers a `loop`: an unconditional header re-entered from the
    /// back edge until a `break` leaves.
    ///
    /// # Errors
    ///
    /// Propagates the body's failures.
    fn lower_loop_body(&mut self, body: &Block) -> Result<Lowered, LowerError> {
        self.lower_loop(|_| Ok(HeaderResult::Uncond), |_| Ok(()), body, |_| Ok(()))
    }

    /// Emits a structured loop: header, condition chain, body,
    /// continue, and exit blocks.
    ///
    /// The header carries [`Op::LoopMerge`] and branches into the
    /// condition chain that `header` builds, so the continue block's
    /// back edge re-enters through the header and re-evaluates the
    /// condition each iteration.  `body_bind` runs at the top of the
    /// body (binding a `for` pattern), and `cont_fill` fills the
    /// continue block before its back edge.
    ///
    /// # Errors
    ///
    /// Propagates the header, binding, body, and continue failures.
    fn lower_loop<H, B, C>(
        &mut self,
        header: H,
        body_bind: B,
        body: &Block,
        cont_fill: C,
    ) -> Result<Lowered, LowerError>
    where
        H: FnOnce(&mut Self) -> Result<HeaderResult, LowerError>,
        B: FnOnce(&mut Self) -> Result<(), LowerError>,
        C: FnOnce(&mut Self) -> Result<(), LowerError>,
    {
        let header_block = self.push_block("loop.header")?;
        let cond_block = self.push_block("loop.cond")?;
        self.term(Op::Branch { target: header_block })?;
        self.enter(cond_block, true)?;
        let result = header(self)?;
        if matches!(result, HeaderResult::Dead) || !self.is_open() {
            // The condition diverged: close whatever it left open, then
            // strand the header so it terminates without a loop.
            if self.is_open() {
                self.term(Op::Branch { target: header_block })?;
            }
            self.enter(header_block, false)?;
            self.finish_current()?;
            return Ok(Lowered::Dead);
        }
        let cond_end = self
            .current
            .ok_or_else(|| self.unsupported("a loop condition block"))?;

        let body_block = self.push_block("loop.body")?;
        let cont_block = self.push_block("loop.cont")?;
        let exit_block = self.push_block("loop.exit")?;

        self.enter(header_block, true)?;
        self.emit_raw(Inst::none(Op::LoopMerge {
            merge: exit_block,
            cont: cont_block,
            control: 0,
        }))?;
        self.term(Op::Branch { target: cond_block })?;

        self.enter(cond_end, true)?;
        let exit_reach = match result {
            HeaderResult::Cond(condition) => {
                self.term(Op::BranchConditional {
                    cond: condition,
                    then: body_block,
                    other: exit_block,
                })?;
                ExitReach::ViaCond
            }
            HeaderResult::Uncond | HeaderResult::Dead => {
                self.term(Op::Branch { target: body_block })?;
                ExitReach::Untaken
            }
        };

        self.loops.push(LoopCtx {
            cont: cont_block,
            exit: exit_block,
            exit_reach,
        });
        self.enter(body_block, true)?;
        body_bind(self)?;
        self.lower_block(body)?;
        if self.is_open() {
            self.term(Op::Branch { target: cont_block })?;
        }
        self.enter(cont_block, true)?;
        cont_fill(self)?;
        if self.is_open() {
            self.term(Op::Branch { target: header_block })?;
        }
        self.finish_current()?;

        let Some(context) = self.loops.pop() else {
            return Err(self.unsupported("a loop without a context"));
        };
        self.enter(exit_block, context.exit_reach != ExitReach::Untaken)?;
        if self.is_open() {
            Ok(Lowered::Void)
        } else {
            Ok(Lowered::Dead)
        }
    }

    /// Lowers a `break`: branch to the enclosing loop's exit.
    ///
    /// An optional value is evaluated for its effects and discarded —
    /// loops in this dialect always type as `void`.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] outside a loop, and
    /// propagates the value's failures.
    fn lower_break(&mut self, value: Option<&Expr>) -> Result<Lowered, LowerError> {
        if let Some(expr) = value {
            let _ = self.value_of(expr)?;
            if !self.is_open() {
                return Ok(Lowered::Dead);
            }
        }
        let Some(context) = self.loops.last_mut() else {
            return Err(self.unsupported("a `break` outside of a loop"));
        };
        context.exit_reach = ExitReach::Taken;
        let exit = context.exit;
        self.term(Op::Branch { target: exit })?;
        Ok(Lowered::Dead)
    }

    /// Lowers a `continue`: branch to the enclosing loop's continue
    /// block.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] outside a loop.
    fn lower_continue(&mut self) -> Result<Lowered, LowerError> {
        let Some(context) = self.loops.last() else {
            return Err(self.unsupported("a `continue` outside of a loop"));
        };
        let cont = context.cont;
        self.term(Op::Branch { target: cont })?;
        Ok(Lowered::Dead)
    }

    /// Lowers a `return`: convert the value to the declared return
    /// type and terminate the block.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for a `return` whose value
    /// does not fit the function's signature, and propagates the
    /// value's and conversion's failures.
    fn lower_return(&mut self, value: Option<&Expr>) -> Result<Lowered, LowerError> {
        let returns_void = default_literals(&self.ret) == Ty::Void;
        match value {
            None if returns_void => {
                self.term(Op::Return)?;
                Ok(Lowered::Dead)
            }
            None => Err(self.unsupported("a `return` without a value")),
            Some(_) if returns_void => {
                Err(self.unsupported("a value returned from a function with no return type"))
            }
            Some(expr) => {
                let Some(value) = self.value_of(expr)? else {
                    return Ok(Lowered::Dead);
                };
                if !self.is_open() {
                    return Ok(Lowered::Dead);
                }
                let ret_ty = self.ret.clone();
                let target = self.map_ty(&ret_ty, expr.span)?;
                let value = self.convert(value, target, expr.span)?;
                self.term(Op::ReturnValue { value })?;
                Ok(Lowered::Dead)
            }
        }
    }

    /// Lowers a `for` loop with its pattern binding scoped to the
    /// loop, as the analyzer scopes it.
    ///
    /// # Errors
    ///
    /// Propagates the setup, binding, body, and continue failures.
    fn lower_for(
        &mut self,
        whole: &Expr,
        pat: &Pat,
        iter: &Expr,
        body: &Block,
    ) -> Result<Lowered, LowerError> {
        let saved = self.slots.clone();
        let lowered = self.lower_for_inner(whole, pat, iter, body);
        self.slots = saved;
        lowered
    }

    /// The unscoped half of [`Lowerer::lower_for`]: set up the counter,
    /// then run the shared loop with `for`-specific closures.
    fn lower_for_inner(
        &mut self,
        whole: &Expr,
        pat: &Pat,
        iter: &Expr,
        body: &Block,
    ) -> Result<Lowered, LowerError> {
        let span = whole.span;
        let setup = if let ExprKind::Range {
            start,
            end,
            inclusive,
        } = &iter.kind
        {
            self.range_loop_setup(span, start.as_deref(), end.as_deref(), *inclusive)?
        } else {
            self.aggregate_loop_setup(span, iter)?
        };
        let Some((counter, counter_ty, elem_ty, limit, comparison, element)) = setup else {
            return Ok(Lowered::Dead);
        };
        let header = move |lowerer: &mut Self| {
            let current = lowerer.load_ptr(counter)?;
            let boolean = lowerer.module.bool_ty();
            let condition = lowerer.def_op(
                boolean,
                Op::Compare {
                    op: comparison,
                    lhs: current,
                    rhs: limit,
                },
            )?;
            Ok(HeaderResult::Cond(condition))
        };
        let body_bind = move |lowerer: &mut Self| {
            let index = lowerer.load_ptr(counter)?;
            let value = match element {
                Some(base) => lowerer.element_at(base, index, span)?,
                None => index,
            };
            lowerer.bind_element(pat, value, elem_ty)
        };
        let cont_fill = move |lowerer: &mut Self| {
            let current = lowerer.load_ptr(counter)?;
            let step = lowerer.int_const(1, counter_ty, span)?;
            let next = lowerer.def_op(
                counter_ty,
                Op::Binary {
                    op: BinOp::IAdd,
                    lhs: current,
                    rhs: step,
                },
            )?;
            lowerer.store_ptr(counter, next, span)?;
            Ok(())
        };
        self.lower_loop(header, body_bind, body, cont_fill)
    }

    /// Computes a range loop's counter plumbing: counter slot, element
    /// type, end bound, comparison, and no aggregate address.
    ///
    /// The element type mirrors the analyzer's `for` rule — the
    /// endpoint the other coerces into, defaulted to a concrete
    /// literal type.  Endpoints evaluate before the loop starts, so
    /// their effects run once, as with Rust's `for` over a range.
    /// `None` overall means an endpoint diverged.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for an open-ended range or
    /// a non-integer element, and propagates evaluation failures.
    fn range_loop_setup(
        &mut self,
        span: Span,
        start: Option<&Expr>,
        end: Option<&Expr>,
        inclusive: bool,
    ) -> Result<LoopSetup, LowerError> {
        let Some(end_expr) = end else {
            return Err(LowerError::Unsupported {
                what: "an open-ended `for` range",
                span,
            });
        };
        let end_ty = self.expr_ty(end_expr);
        let elem = match start.map(|expr| self.expr_ty(expr)) {
            Some(start_ty) => {
                if coerce(&start_ty, &end_ty) {
                    end_ty
                } else {
                    start_ty
                }
            }
            None => end_ty,
        };
        let elem = default_literals(&elem);
        let Ty::Scalar(scalar) = &elem else {
            return Err(LowerError::Unsupported {
                what: "a `for` range over a non-scalar type",
                span,
            });
        };
        if !scalar.is_int() {
            return Err(LowerError::Unsupported {
                what: "a `for` range over a non-integer type",
                span,
            });
        }
        let elem_ty = self.map_ty(&elem, span)?;
        let low = match start {
            Some(expr) => {
                let Some(value) = self.value_of(expr)? else {
                    return Ok(None);
                };
                if !self.is_open() {
                    return Ok(None);
                }
                value
            }
            None => self.int_const(0, elem_ty, span)?,
        };
        let Some(high) = self.value_of(end_expr)? else {
            return Ok(None);
        };
        if !self.is_open() {
            return Ok(None);
        }
        let low = self.convert(low, elem_ty, span)?;
        let high = self.convert(high, elem_ty, span)?;
        let counter = self.alloc_slot(elem_ty, "for.index")?;
        self.store_ptr(counter, low, span)?;
        let comparison = cmp_op(
            if inclusive { BinaryOp::Le } else { BinaryOp::Lt },
            false,
            scalar.is_signed_int(),
        )
        .ok_or(LowerError::Unsupported {
            what: "a range comparison without an IR form",
            span,
        })?;
        Ok(Some((counter, elem_ty, elem_ty, high, comparison, None)))
    }

    /// Computes an aggregate loop's counter plumbing: the iterated
    /// value's address, element type, element count, and unsigned
    /// comparison.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for slices and non-aggregate
    /// iterated values, and propagates address and evaluation failures.
    fn aggregate_loop_setup(&mut self, span: Span, iter: &Expr) -> Result<LoopSetup, LowerError> {
        let iter_ty = default_literals(&self.expr_ty(iter));
        let (layers, rest) = split_indirection(&iter_ty);
        let (count, elem) = match &rest {
            Ty::Array { elem, len: Some(len) } => (*len, (**elem).clone()),
            Ty::Array { len: None, .. } => {
                return Err(LowerError::Unsupported {
                    what: "iteration over a slice",
                    span,
                });
            }
            Ty::Vector { elem, lanes } => (u32::from(*lanes), Ty::Scalar(*elem)),
            Ty::Error => {
                return Err(LowerError::UnsupportedType { ty: Ty::Error, span });
            }
            _ => {
                return Err(LowerError::Unsupported {
                    what: "iteration over this value",
                    span,
                });
            }
        };
        let elem_ty = self.map_ty(&elem, span)?;
        let element = if layers == 0 {
            match self.place(iter, true)? {
                Place::Ptr(ptr) => Some(ptr),
                Place::NotPlace => {
                    return Err(LowerError::Unsupported {
                        what: "the address of the iterated value",
                        span,
                    });
                }
                Place::Dead => return Ok(None),
            }
        } else {
            let Some(mut ptr) = self.value_of(iter)? else {
                return Ok(None);
            };
            if !self.is_open() {
                return Ok(None);
            }
            for _ in 1..layers {
                ptr = self.load_ptr(ptr)?;
            }
            Some(ptr)
        };
        let index_ty = self.module.int_ty(32, false);
        let counter = self.alloc_slot(index_ty, "for.index")?;
        let start = self.int_const(0, index_ty, span)?;
        self.store_ptr(counter, start, span)?;
        let limit = self.int_const(i128::from(count), index_ty, span)?;
        Ok(Some((
            counter,
            index_ty,
            elem_ty,
            limit,
            CmpOp::ULessThan,
            element,
        )))
    }

    /// Reads element `index` of the aggregate at `base`, walking one
    /// [`Op::AccessChain`] step and loading the result.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when `base` does not point
    /// at an array or vector, and propagates construction failures.
    fn element_at(&mut self, base: ValueId, index: ValueId, span: Span) -> Result<ValueId, LowerError> {
        let pointer = self.module.type_of(base);
        let (storage, pointee) = self
            .module
            .pointer_parts(pointer)
            .ok_or(LowerError::Unsupported {
                what: "an iteration over a non-pointer address",
                span,
            })?;
        let element = match self.module.ty(pointee) {
            IrType::Array { elem, .. } | IrType::Vector { elem, .. } => *elem,
            _ => {
                return Err(LowerError::Unsupported {
                    what: "iteration over this value",
                    span,
                });
            }
        };
        let lane = self.module.ptr_ty(storage, element);
        let address = self.def_op(
            lane,
            Op::AccessChain {
                base,
                indices: vec![index],
            },
        )?;
        self.load_ptr(address)
    }

    /// Binds a loop pattern to one element value: identifiers become
    /// plain values, `mut` identifiers become slots holding a copy, so
    /// writes never touch the loop counter itself.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for a malformed pattern, and
    /// propagates slot and store failures.
    fn bind_element(&mut self, pat: &Pat, value: ValueId, ty: TypeId) -> Result<(), LowerError> {
        match &pat.kind {
            PatKind::Wild => Ok(()),
            PatKind::Error => Err(LowerError::Unsupported {
                what: "a malformed pattern",
                span: pat.span,
            }),
            PatKind::Ident { name, mutable } => {
                let raw = strip_raw_ident(name);
                if *mutable {
                    let slot = self.alloc_slot(ty, raw)?;
                    self.store_ptr(slot, value, pat.span)?;
                    self.slots.insert(pat.id, Binding::Slot(slot));
                } else {
                    self.slots.insert(pat.id, Binding::Value(value));
                }
                Ok(())
            }
        }
    }

    /// Lowers a cast: evaluate the operand, then convert it to the
    /// annotated type.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::UnsupportedType`] when the target has no
    /// IR representation, and propagates the operand's and conversion's
    /// failures.
    fn lower_cast(&mut self, whole: &Expr, inner: &Expr, ty: &Type) -> Result<Lowered, LowerError> {
        let Some(value) = self.value_of(inner)? else {
            return Ok(Lowered::Dead);
        };
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let target = self.resolve_ast_ty(ty)?;
        let target = self.map_ty(&target, whole.span)?;
        let value = self.convert(value, target, whole.span)?;
        Ok(Lowered::Value(value))
    }

    /// Tuples have no IR representation, so a tuple expression reports
    /// its own type as unsupported at its span.
    ///
    /// # Errors
    ///
    /// Always returns [`LowerError::UnsupportedType`] with the tuple's
    /// semantic type.
    fn lower_tuple(&mut self, whole: &Expr, _elems: &[Expr]) -> Result<Lowered, LowerError> {
        Err(LowerError::UnsupportedType {
            ty: self.expr_ty(whole),
            span: whole.span,
        })
    }

    /// Lowers an array literal to an [`Op::CompositeConstruct`] of its
    /// converted elements.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::UnsupportedType`] when the literal has no
    /// sized array type, [`LowerError::Unsupported`] when its length
    /// disagrees with the type, and propagates element failures.
    fn lower_array(&mut self, whole: &Expr, elems: &[Expr]) -> Result<Lowered, LowerError> {
        let span = whole.span;
        let ty = default_literals(&self.expr_ty(whole));
        let Ty::Array { elem, len } = &ty else {
            return Err(LowerError::UnsupportedType { ty, span });
        };
        let Some(len) = len else {
            return Err(LowerError::UnsupportedType { ty, span });
        };
        let elem_ty = self.map_ty(elem, span)?;
        let mut constituents = Vec::with_capacity(elems.len());
        for element in elems {
            let Some(value) = self.value_of(element)? else {
                return Ok(Lowered::Dead);
            };
            if !self.is_open() {
                return Ok(Lowered::Dead);
            }
            constituents.push(self.convert(value, elem_ty, element.span)?);
        }
        if constituents.len() != *len as usize {
            return Err(LowerError::Unsupported {
                what: "an array literal that does not match its type",
                span,
            });
        }
        let array_ty = self.map_ty(&ty, span)?;
        let value = self.def_op(array_ty, Op::CompositeConstruct { constituents })?;
        Ok(Lowered::Value(value))
    }

    /// Lowers a call: resolve the callee, then dispatch to the user
    /// function or the built-in.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for callees that are not
    /// plain resolved paths, and propagates the chosen arm's failures.
    fn lower_call(&mut self, whole: &Expr, callee: &Expr, args: &[Expr]) -> Result<Lowered, LowerError> {
        let ExprKind::Path(path) = &callee.kind else {
            return Err(LowerError::Unsupported {
                what: "a non-function callee",
                span: callee.span,
            });
        };
        if path.segments.len() != 1 {
            return Err(LowerError::Unsupported {
                what: "a qualified path",
                span: callee.span,
            });
        }
        let Some(resolution) = self.analysis.resolutions.get(callee.id).cloned() else {
            return Err(LowerError::Unsupported {
                what: "an unresolved name",
                span: callee.span,
            });
        };
        match resolution {
            Res::Local { .. } => Err(LowerError::Unsupported {
                what: "a local used as a function",
                span: callee.span,
            }),
            Res::Function { name } => self.lower_user_call(whole, &name, args),
            Res::Builtin { name } => {
                let Some(builtin) = lookup_builtin_fn(&name) else {
                    return Err(LowerError::Unsupported {
                        what: "an unknown built-in",
                        span: callee.span,
                    });
                };
                self.lower_builtin(whole, builtin, args)
            }
        }
    }

    /// Lowers a call to a declared function: convert each argument to
    /// its parameter type, then [`Op::Call`].
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for a generic function (its
    /// signature was never declared) or an argument-count mismatch, and
    /// propagates argument failures.
    fn lower_user_call(&mut self, whole: &Expr, name: &str, args: &[Expr]) -> Result<Lowered, LowerError> {
        let span = whole.span;
        let signature = self
            .signatures
            .get(name)
            .cloned()
            .ok_or(LowerError::Unsupported {
                what: "a call to a generic function",
                span,
            })?;
        if args.len() != signature.params.len() {
            return Err(LowerError::Unsupported {
                what: "an argument count mismatch",
                span,
            });
        }
        let mut call_args = Vec::with_capacity(args.len());
        for (arg, expected) in args.iter().zip(&signature.params) {
            let Some(value) = self.value_of(arg)? else {
                return Ok(Lowered::Dead);
            };
            if !self.is_open() {
                return Ok(Lowered::Dead);
            }
            call_args.push(self.convert(value, *expected, arg.span)?);
        }
        if matches!(self.module.ty(signature.ret), IrType::Void) {
            self.emit_raw(Inst::none(Op::Call {
                callee: signature.ir,
                args: call_args,
            }))?;
            Ok(Lowered::Void)
        } else {
            let result = self.def_op(
                signature.ret,
                Op::Call {
                    callee: signature.ir,
                    args: call_args,
                },
            )?;
            self.finish_value(whole, result)
        }
    }

    /// Lowers a built-in call by its checked kind.
    ///
    /// # Errors
    ///
    /// Propagates the chosen arm's failures.
    fn lower_builtin(
        &mut self,
        whole: &Expr,
        builtin: &Builtin,
        args: &[Expr],
    ) -> Result<Lowered, LowerError> {
        match &builtin.kind {
            BuiltinKind::WorkItem => self.lower_work_item(whole, builtin, args),
            BuiltinKind::Fixed { .. } => self.lower_fixed_builtin(whole, builtin, args),
            BuiltinKind::Reduce { .. } => self.lower_reduce_builtin(whole, builtin, args),
            BuiltinKind::Numeric { .. } | BuiltinKind::Float { .. } => {
                self.lower_math_builtin(whole, builtin, args)
            }
        }
    }

    /// Lowers a work-item query to a call to its imported declaration.
    ///
    /// The OpenCL SPIR-V environment declares these with import
    /// linkage; the declaration is created once and reused.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the query has no
    /// dimension argument, and propagates evaluation and declaration
    /// failures.
    fn lower_work_item(
        &mut self,
        whole: &Expr,
        builtin: &Builtin,
        args: &[Expr],
    ) -> Result<Lowered, LowerError> {
        let span = whole.span;
        let Some(arg) = args.first() else {
            return Err(LowerError::Unsupported {
                what: "a work-item query without a dimension",
                span,
            });
        };
        let Some(dimension) = self.value_of(arg)? else {
            return Ok(Lowered::Dead);
        };
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let index_ty = self.module.int_ty(32, true);
        let dimension = self.convert(dimension, index_ty, arg.span)?;
        let callee = match self.module.find_function(builtin.name) {
            Some(declared) => declared,
            None => self.declare_work_item(builtin.name)?,
        };
        let size_ty = self.module.int_ty(64, false);
        let result = self.def_op(
            size_ty,
            Op::Call {
                callee,
                args: vec![dimension],
            },
        )?;
        self.finish_value(whole, result)
    }

    /// Declares a work-item query as an imported `size_t(int)`
    /// function with its required linkage decoration.
    ///
    /// # Errors
    ///
    /// Returns a construction error when the module rejects the
    /// declaration.
    fn declare_work_item(&mut self, name: &str) -> Result<ValueId, LowerError> {
        let index_ty = self.module.int_ty(32, true);
        let size_ty = self.module.int_ty(64, false);
        let signature = self.module.fn_ty(size_ty, vec![index_ty]);
        let declared = self
            .module
            .add_function(name, signature, Linkage::Import)
            .map_err(|error| LowerError::Build {
                what: "a work-item declaration",
                error,
            })?;
        self.module.decorate(
            declared,
            Decor::LinkageAttributes {
                name: name.to_string(),
                import: true,
            },
        );
        Ok(declared)
    }

    /// Lowers the fence family: `barrier` synchronizes the work-group
    /// at device scope, `sub_group_barrier` the sub-group, and the
    /// `mem_fence` functions order memory at device scope.  All use
    /// acquire-release semantics over cross-device and work-group
    /// memory.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for an unknown fixed
    /// built-in, and propagates argument and constant failures.
    fn lower_fixed_builtin(
        &mut self,
        whole: &Expr,
        builtin: &Builtin,
        args: &[Expr],
    ) -> Result<Lowered, LowerError> {
        let barrier = match builtin.name {
            "barrier" | "sub_group_barrier" => true,
            "mem_fence" | "read_mem_fence" | "write_mem_fence" => false,
            _ => {
                return Err(LowerError::Unsupported {
                    what: "a fixed-signature built-in",
                    span: whole.span,
                });
            }
        };
        for arg in args {
            let _ = self.value_of(arg)?;
            if !self.is_open() {
                return Ok(Lowered::Dead);
            }
        }
        let span = whole.span;
        let word_ty = self.module.int_ty(32, false);
        let device = self.int_const(1, word_ty, span)?;
        let semantics = self.int_const(776, word_ty, span)?;
        if barrier {
            let scope: i128 = if builtin.name == "sub_group_barrier" { 3 } else { 2 };
            let execution = self.int_const(scope, word_ty, span)?;
            self.emit_raw(Inst::none(Op::ControlBarrier {
                exec: execution,
                mem: device,
                semantics,
            }))?;
        } else {
            self.emit_raw(Inst::none(Op::MemoryBarrier {
                mem: device,
                semantics,
            }))?;
        }
        Ok(Lowered::Void)
    }

    /// Lowers `all` and `any`: a scalar mask passes through, and a
    /// vector mask folds lane by lane — `OpenCL.std` defines no
    /// reduction instruction.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for a reduction over
    /// non-boolean values, and propagates evaluation failures.
    fn lower_reduce_builtin(
        &mut self,
        whole: &Expr,
        builtin: &Builtin,
        args: &[Expr],
    ) -> Result<Lowered, LowerError> {
        let span = whole.span;
        let Some(arg) = args.first() else {
            return Err(LowerError::Unsupported {
                what: "a reduction without an argument",
                span,
            });
        };
        let Some(value) = self.value_of(arg)? else {
            return Ok(Lowered::Dead);
        };
        if !self.is_open() {
            return Ok(Lowered::Dead);
        }
        let value_ty = self.module.type_of(value);
        if ir_is_bool(&self.module, value_ty) {
            return Ok(Lowered::Value(value));
        }
        let lanes = match self.module.ty(value_ty) {
            IrType::Vector { elem, len } if ir_is_bool(&self.module, *elem) => {
                u8::try_from(*len).map_err(|_| LowerError::Unsupported {
                    what: "a reduction over an oversized vector",
                    span,
                })?
            }
            _ => {
                return Err(LowerError::Unsupported {
                    what: "a reduction of non-boolean values",
                    span,
                });
            }
        };
        let folded = self.bool_vector_fold(value, lanes, builtin.name == "any", span)?;
        Ok(Lowered::Value(folded))
    }

    /// Lowers a shape-preserving math built-in to an
    /// [`Op::ExtInst`] into `OpenCL.std`, converting every argument to
    /// the result type so scalars splat into vector shapes.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the instruction set
    /// has no opcode for the name at the result's element type, and
    /// propagates argument failures.
    fn lower_math_builtin(
        &mut self,
        whole: &Expr,
        builtin: &Builtin,
        args: &[Expr],
    ) -> Result<Lowered, LowerError> {
        let span = whole.span;
        let result_ty = self.map_ty(&self.expr_ty(whole), span)?;
        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            let Some(value) = self.value_of(arg)? else {
                return Ok(Lowered::Dead);
            };
            if !self.is_open() {
                return Ok(Lowered::Dead);
            }
            values.push(self.convert(value, result_ty, arg.span)?);
        }
        let element_id = match self.module.ty(result_ty) {
            IrType::Vector { elem, .. } => *elem,
            _ => result_ty,
        };
        let element = self.module.ty(element_id).clone();
        let opcode = ext_inst_opcode(builtin.name, &element).ok_or(LowerError::Unsupported {
            what: "a built-in without a SPIR-V opcode",
            span,
        })?;
        let set = self.ext_set();
        let result = self.def_op(
            result_ty,
            Op::ExtInst {
                set,
                inst: opcode,
                args: values,
            },
        )?;
        Ok(Lowered::Value(result))
    }

    /// The `OpenCL.std` extended-instruction set, created on first
    /// use.
    fn ext_set(&mut self) -> ExtSetId {
        match self.ext_set {
            Some(set) => set,
            None => {
                let set = self.module.add_ext_inst_set("OpenCL.std");
                self.ext_set = Some(set);
                set
            }
        }
    }

    /// Folds a boolean vector to a scalar with `&&` (`all`) or `||`
    /// (`any`), lane by lane.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for an empty vector, and
    /// propagates construction failures.
    fn bool_vector_fold(
        &mut self,
        value: ValueId,
        lanes: u8,
        any: bool,
        span: Span,
    ) -> Result<ValueId, LowerError> {
        let boolean = self.module.bool_ty();
        let mut folded: Option<ValueId> = None;
        for lane in 0..u32::from(lanes) {
            let component = self.def_op(
                boolean,
                Op::CompositeExtract {
                    composite: value,
                    indices: vec![lane],
                },
            )?;
            folded = Some(match folded {
                None => component,
                Some(previous) => self.def_op(
                    boolean,
                    Op::Binary {
                        op: if any { BinOp::LogicalOr } else { BinOp::LogicalAnd },
                        lhs: previous,
                        rhs: component,
                    },
                )?,
            });
        }
        folded.ok_or(LowerError::Unsupported {
            what: "a reduction over an empty vector",
            span,
        })
    }

    /// Evaluates a literal expression at its own type.
    ///
    /// The analyzer already parsed the value into the type it
    /// assigned: an unsuffixed integer or a character lands in
    /// [`Ty::IntLit`], an unsuffixed float in [`Ty::FloatLit`], and a
    /// suffix selects a concrete [`Ty::Scalar`].  Suffixed literals
    /// re-parse their digit body here.  An unsuffixed integer
    /// materializes at the narrowest default that holds it, so every
    /// consumer narrows through [`Lowerer::convert`] rather than
    /// truncating at the literal.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::UnsupportedType`] when the literal's type
    /// has no IR representation (a `half` suffix or a malformed
    /// analyzer table), and [`LowerError::Unsupported`] when the digit
    /// body does not parse or the value fits no default type.
    fn eval_literal(&mut self, whole: &Expr, text: &str, kind: &TokenKind) -> Result<Lowered, LowerError> {
        let span = whole.span;
        let ty = self.expr_ty(whole);
        if matches!(ty, Ty::Error) {
            return Err(LowerError::UnsupportedType { ty, span });
        }
        let TokenKind::Literal {
            kind: literal,
            suffix_start,
        } = kind
        else {
            return Err(LowerError::Unsupported {
                what: "a non-literal token",
                span,
            });
        };
        let split = (*suffix_start as usize).min(text.len());
        let (body, _) = text.split_at(split);
        match &ty {
            Ty::IntLit(value) => {
                let scalar = literal_int_scalar(*value).ok_or(LowerError::Unsupported {
                    what: "an integer literal out of range",
                    span,
                })?;
                let ir_ty = self.map_scalar(scalar, span)?;
                Ok(Lowered::Value(self.int_const(*value, ir_ty, span)?))
            }
            Ty::FloatLit(value) => {
                let value = match value {
                    Some(value) => *value,
                    None => hex_float_value(body).ok_or(LowerError::Unsupported {
                        what: "a malformed hexadecimal float literal",
                        span,
                    })?,
                };
                let ir_ty = self.module.float_ty(32);
                Ok(Lowered::Value(self.float_const(value, ir_ty, span)?))
            }
            Ty::Scalar(scalar) if scalar.is_int() => {
                let LiteralKind::Int { base, .. } = literal else {
                    return Err(LowerError::Unsupported {
                        what: "a malformed integer literal",
                        span,
                    });
                };
                let value = int_digits_value(body, *base).ok_or(LowerError::Unsupported {
                    what: "an integer literal out of range",
                    span,
                })?;
                let ir_ty = self.map_scalar(*scalar, span)?;
                Ok(Lowered::Value(self.int_const(value, ir_ty, span)?))
            }
            Ty::Scalar(scalar) if scalar.is_float() => {
                let value = match literal {
                    LiteralKind::Int { base, .. } => int_digits_value(body, *base).map(|value| value as f64),
                    LiteralKind::Float {
                        base: Base::Hexadecimal,
                        ..
                    } => hex_float_value(body),
                    LiteralKind::Float { .. } => float_value(body),
                    _ => None,
                }
                .ok_or(LowerError::Unsupported {
                    what: "a malformed float literal",
                    span,
                })?;
                let ir_ty = self.map_scalar(*scalar, span)?;
                Ok(Lowered::Value(self.float_const(value, ir_ty, span)?))
            }
            other => Err(LowerError::UnsupportedType {
                ty: other.clone(),
                span,
            }),
        }
    }

    /// A float constant of `ty`, canonicalizing NaN payloads.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when `ty` is not a
    /// floating-point type, and a construction error when the module
    /// rejects the payload.
    fn float_const(&mut self, value: f64, ty: TypeId, span: Span) -> Result<ValueId, LowerError> {
        let payload = match self.module.ty(ty) {
            IrType::Float { bits: 32 } => ConstValue::from_f32_bits((value as f32).to_bits()),
            IrType::Float { bits: 64 } => ConstValue::from_f64_bits(value.to_bits()),
            _ => {
                return Err(LowerError::Unsupported {
                    what: "a float constant of a non-floating-point type",
                    span,
                });
            }
        };
        self.module
            .intern_const(ty, payload)
            .map_err(|error| LowerError::Build {
                what: "a float constant",
                error,
            })
    }

    /// The lane count of `ty` when it is a vector, `None` for scalars
    /// and other aggregates.
    fn lanes_of(&self, ty: TypeId) -> Option<u32> {
        match self.module.ty(ty) {
            IrType::Vector { len, .. } => Some(*len),
            _ => None,
        }
    }

    /// Coerces `value` to `to`, the IR counterpart of the analyzer's
    /// implicit conversions wherever two types must agree: stores,
    /// arguments, returns, operands, and casts.
    ///
    /// The shape follows the IR's own vocabulary: `undef` and `null`
    /// re-materialize at any type, vectors convert lane by lane (a
    /// scalar operand splats, a boolean mask folds with
    /// [`Lowerer::bool_vector_fold`]), and scalars fold constants or
    /// use the conversion opcodes.  Integer constants wrap on
    /// narrowing, matching C's conversion rule.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] for conversions the IR
    /// cannot express — vector-to-scalar narrowing beyond a boolean
    /// mask, vectors of different lengths, and pointer casts that
    /// change the pointee — and propagates construction failures.
    fn convert(&mut self, value: ValueId, to: TypeId, span: Span) -> Result<ValueId, LowerError> {
        let from = self.module.type_of(value);
        if from == to {
            return Ok(value);
        }
        // Carrying no bits, `undef` and `null` take any type before
        // the lane rules below apply.
        if let Some(constant @ (ConstValue::Undef | ConstValue::Null)) = self.module.constant_value(value) {
            return self
                .module
                .intern_const(to, constant)
                .map_err(|error| LowerError::Build {
                    what: "a converted constant",
                    error,
                });
        }
        let from_lanes = self.lanes_of(from);
        let to_lanes = self.lanes_of(to);
        match (from_lanes, to_lanes) {
            (Some(lanes), Some(other)) if lanes == other => {
                let from_elem = ir_element(&self.module, from);
                let to_elem = ir_element(&self.module, to);
                let mut constituents = Vec::with_capacity(lanes as usize);
                for lane in 0..lanes {
                    let component = self.def_op(
                        from_elem,
                        Op::CompositeExtract {
                            composite: value,
                            indices: vec![lane],
                        },
                    )?;
                    constituents.push(self.convert(component, to_elem, span)?);
                }
                return Ok(self.def_op(to, Op::CompositeConstruct { constituents })?);
            }
            (Some(_), Some(_)) => {
                return Err(LowerError::Unsupported {
                    what: "a conversion between vectors of different lengths",
                    span,
                });
            }
            (Some(lanes), None) if ir_is_bool(&self.module, from) => {
                let lanes = u8::try_from(lanes).map_err(|_| LowerError::Unsupported {
                    what: "a boolean vector with more than 255 lanes",
                    span,
                })?;
                return Ok(self.bool_vector_fold(value, lanes, false, span)?);
            }
            (Some(_), None) => {
                return Err(LowerError::Unsupported {
                    what: "a conversion that narrows a vector to a scalar",
                    span,
                });
            }
            (None, Some(lanes)) => {
                let to_elem = ir_element(&self.module, to);
                let scalar = self.convert(value, to_elem, span)?;
                return Ok(self.def_op(
                    to,
                    Op::CompositeConstruct {
                        constituents: vec![scalar; lanes as usize],
                    },
                )?);
            }
            (None, None) => {}
        }
        if let Some(constant) = self.module.constant_value(value) {
            return self.convert_const(constant, from, to, span);
        }
        if ir_is_bool(&self.module, from) {
            let one;
            let zero;
            if ir_is_int(&self.module, to) {
                one = self.int_const(1, to, span)?;
                zero = self.int_const(0, to, span)?;
            } else if ir_is_float(&self.module, to) {
                one = self.float_const(1.0, to, span)?;
                zero = self.float_const(0.0, to, span)?;
            } else {
                return Err(LowerError::Unsupported {
                    what: "this type conversion",
                    span,
                });
            }
            return Ok(self.def_op(
                to,
                Op::Select {
                    cond: value,
                    a: one,
                    b: zero,
                },
            )?);
        }
        if ir_is_int(&self.module, from) && ir_is_int(&self.module, to) {
            let (from_bits, from_signed) = match self.module.ty(from) {
                IrType::Int { bits, signed } => (*bits, *signed),
                _ => {
                    return Err(LowerError::Unsupported {
                        what: "this type conversion",
                        span,
                    });
                }
            };
            let to_bits = match self.module.ty(to) {
                IrType::Int { bits, .. } => *bits,
                _ => {
                    return Err(LowerError::Unsupported {
                        what: "this type conversion",
                        span,
                    });
                }
            };
            let op = if from_bits == to_bits {
                ConvOp::Bitcast
            } else if from_signed {
                ConvOp::SConvert
            } else {
                ConvOp::UConvert
            };
            return Ok(self.def_op(to, Op::Convert { op, operand: value })?);
        }
        if ir_is_int(&self.module, from) && ir_is_float(&self.module, to) {
            let from_signed = match self.module.ty(from) {
                IrType::Int { signed, .. } => *signed,
                _ => {
                    return Err(LowerError::Unsupported {
                        what: "this type conversion",
                        span,
                    });
                }
            };
            let op = if from_signed {
                ConvOp::ConvertSToF
            } else {
                ConvOp::ConvertUToF
            };
            return Ok(self.def_op(to, Op::Convert { op, operand: value })?);
        }
        if ir_is_float(&self.module, from) && ir_is_int(&self.module, to) {
            let to_signed = match self.module.ty(to) {
                IrType::Int { signed, .. } => *signed,
                _ => {
                    return Err(LowerError::Unsupported {
                        what: "this type conversion",
                        span,
                    });
                }
            };
            let op = if to_signed {
                ConvOp::ConvertFToS
            } else {
                ConvOp::ConvertFToU
            };
            return Ok(self.def_op(to, Op::Convert { op, operand: value })?);
        }
        if ir_is_float(&self.module, from) && ir_is_float(&self.module, to) {
            return Ok(self.def_op(
                to,
                Op::Convert {
                    op: ConvOp::FConvert,
                    operand: value,
                },
            )?);
        }
        if ir_is_int(&self.module, from) && ir_is_bool(&self.module, to) {
            let zero = self.int_const(0, from, span)?;
            return Ok(self.def_op(
                to,
                Op::Compare {
                    op: CmpOp::INotEqual,
                    lhs: value,
                    rhs: zero,
                },
            )?);
        }
        if ir_is_float(&self.module, from) && ir_is_bool(&self.module, to) {
            let zero = self.float_const(0.0, from, span)?;
            let equals = self.module.bool_ty();
            let equals = self.def_op(
                equals,
                Op::Compare {
                    op: CmpOp::FOrdEqual,
                    lhs: value,
                    rhs: zero,
                },
            )?;
            return Ok(self.def_op(
                to,
                Op::Unary {
                    op: UnOp::LogicalNot,
                    operand: equals,
                },
            )?);
        }
        if self.module.pointer_parts(from).is_some() || self.module.pointer_parts(to).is_some() {
            return Err(LowerError::Unsupported {
                what: "a pointer cast that changes the pointee type",
                span,
            });
        }
        Err(LowerError::Unsupported {
            what: "this type conversion",
            span,
        })
    }

    /// Folds a constant conversion, applying C's rules to the
    /// payload: integers wrap on narrowing, floats truncate toward
    /// zero before the bits are reinterpreted, and booleans become
    /// 0/1.  `undef` and `null` pass through unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError::Unsupported`] when the target cannot
    /// hold the constant's kind, and a construction error when the
    /// module rejects the payload.
    fn convert_const(
        &mut self,
        constant: ConstValue,
        from: TypeId,
        to: TypeId,
        span: Span,
    ) -> Result<ValueId, LowerError> {
        let unsupported = || LowerError::Unsupported {
            what: "this type conversion",
            span,
        };
        let payload = match constant {
            ConstValue::Undef | ConstValue::Null => constant,
            ConstValue::Bool(flag) => match self.module.ty(to) {
                IrType::Bool => ConstValue::Bool(flag),
                IrType::Int { bits, .. } => ConstValue::Int(if flag { mask_bits(1, *bits) } else { 0 }),
                IrType::Float { bits: 32 } => {
                    ConstValue::from_f32_bits(if flag { 1.0f32.to_bits() } else { 0 })
                }
                IrType::Float { bits: 64 } => {
                    ConstValue::from_f64_bits(if flag { 1.0f64.to_bits() } else { 0 })
                }
                _ => return Err(unsupported()),
            },
            ConstValue::Int(raw) => {
                let (from_bits, from_signed) = match self.module.ty(from) {
                    IrType::Int { bits, signed } => (*bits, *signed),
                    _ => return Err(unsupported()),
                };
                let value = if from_signed {
                    sign_extend(raw, from_bits)
                } else {
                    i128::from(raw)
                };
                match self.module.ty(to) {
                    IrType::Int { bits, .. } => {
                        ConstValue::Int(reinterpret_int(raw, from_bits, from_signed, *bits))
                    }
                    IrType::Float { bits: 32 } => ConstValue::from_f32_bits((value as f64 as f32).to_bits()),
                    IrType::Float { bits: 64 } => ConstValue::from_f64_bits((value as f64).to_bits()),
                    IrType::Bool => ConstValue::Bool(value != 0),
                    _ => return Err(unsupported()),
                }
            }
            ConstValue::Float32(bits) => float_payload(f64::from(f32::from_bits(bits)), self.module.ty(to))
                .ok_or_else(|| unsupported())?,
            ConstValue::Float64(bits) => {
                float_payload(f64::from_bits(bits), self.module.ty(to)).ok_or_else(|| unsupported())?
            }
        };
        self.module
            .intern_const(to, payload)
            .map_err(|error| LowerError::Build {
                what: "a converted constant",
                error,
            })
    }

    // __NEXT_CHUNK
}
