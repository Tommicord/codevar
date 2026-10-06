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

//! The type system of the Codevar OpenCL dialect.
//!
//! Types are named with OpenCL spellings (`int`, `uint`, `float`, `int4`)
//! while also accepting the Rust-style aliases the parser tests use
//! (`i32`, `u8`, `f64`); [`Display`] always prints the canonical OpenCL
//! form. Three "placeholder" types carry the front end through inference:
//!
//! * [`Ty::IntLit`] and [`Ty::FloatLit`] are integer and float literals
//!   that have not been coerced yet; their values are kept so range checks
//!   can blame the literal rather than the expression.
//! * [`Ty::Error`] and [`Ty::Never`] coerce to everything, which stops one
//!   mistake from producing a cascade of follow-on errors.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// A scalar (single-lane) element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scalar {
    /// `bool`, one byte with a 0/1 value.
    Bool,
    /// `char` / `i8`, a signed 8-bit integer.
    I8,
    /// `short` / `i16`, a signed 16-bit integer.
    I16,
    /// `int` / `i32`, a signed 32-bit integer.
    I32,
    /// `long` / `i64`, a signed 64-bit integer.
    I64,
    /// `uchar` / `u8`, an unsigned 8-bit integer.
    U8,
    /// `ushort` / `u16`, an unsigned 16-bit integer.
    U16,
    /// `uint` / `u32`, an unsigned 32-bit integer.
    U32,
    /// `ulong` / `u64`, an unsigned 64-bit integer.
    U64,
    /// `half` / `f16`, a 16-bit float.
    F16,
    /// `float` / `f32`, a 32-bit float.
    F32,
    /// `double` / `f64`, a 64-bit float.
    F64,
}

impl Scalar {
    /// The canonical OpenCL spelling of this scalar.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "char",
            Self::I16 => "short",
            Self::I32 => "int",
            Self::I64 => "long",
            Self::U8 => "uchar",
            Self::U16 => "ushort",
            Self::U32 => "uint",
            Self::U64 => "ulong",
            Self::F16 => "half",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    /// True for the integer scalars (both signs, not `bool`).
    #[must_use]
    pub const fn is_int(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::U8 | Self::U16 | Self::U32 | Self::U64
        )
    }

    /// True for the floating-point scalars.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    /// True for signed integers (and signed floats are *not* signed ints).
    #[must_use]
    pub const fn is_signed_int(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// Width of the type in bits (`16` for `half`, `32` for `bool`).
    #[must_use]
    pub const fn bits(self) -> u16 {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 | Self::F16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 => 64,
        }
    }

    /// True when `value` is representable by this integer scalar.
    #[must_use]
    pub const fn fits_int(self, value: i128) -> bool {
        match self {
            Self::I8 => value >= i8::MIN as i128 && value <= i8::MAX as i128,
            Self::I16 => value >= i16::MIN as i128 && value <= i16::MAX as i128,
            Self::I32 => value >= i32::MIN as i128 && value <= i32::MAX as i128,
            Self::I64 => value >= i64::MIN as i128 && value <= i64::MAX as i128,
            Self::U8 => value >= 0 && value <= u8::MAX as i128,
            Self::U16 => value >= 0 && value <= u16::MAX as i128,
            Self::U32 => value >= 0 && value <= u32::MAX as i128,
            Self::U64 => value >= 0 && value <= u64::MAX as i128,
            _ => false,
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A type of the Codevar OpenCL dialect.
#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    /// The result of an already-reported error; coerces to anything.
    Error,
    /// The type of expressions that never produce a value (`return`).
    Never,
    /// The empty type: the result of a function that returns nothing.
    Void,
    /// A single-lane value.
    Scalar(Scalar),
    /// A multi-lane value: `int4`, `float3`, …
    Vector {
        /// Element type of each lane.
        elem: Scalar,
        /// Lane count; OpenCL defines 2, 3, 4, 8, and 16.
        lanes: u8,
    },
    /// A raw pointer: `*mut T` or `*const T`.
    Ptr {
        /// Whether the pointee may be written through the pointer.
        mutable: bool,
        /// Pointed-to type.
        inner: Box<Ty>,
    },
    /// A reference: `&mut T` or `&T`.
    Ref {
        /// Whether the referent may be written through the reference.
        mutable: bool,
        /// Referenced type.
        inner: Box<Ty>,
    },
    /// A fixed-size array `[T; N]`, or a slice `[T]` when `len` is `None`.
    Array {
        /// Element type.
        elem: Box<Ty>,
        /// Element count, or `None` for a slice.
        len: Option<u32>,
    },
    /// A tuple type `(A, B)`; the unit type is the empty tuple.
    Tuple(Vec<Ty>),
    /// An instantiation of a declared struct: `Particle`, `Pair<int>`.
    Struct {
        /// Declared struct name.
        name: String,
        /// Generic arguments, substituted at use sites.
        args: Vec<Ty>,
    },
    /// A declared generic parameter that has not been instantiated.
    Generic(String),
    /// An unsuffixed integer literal whose value has not been coerced.
    IntLit(i128),
    /// An unsuffixed float literal; `None` when the value could not be
    /// parsed (hexadecimal floats are accepted but not evaluated).
    FloatLit(Option<f64>),
}

impl Ty {
    /// Shorthand for [`Ty::Scalar`] with [`Scalar::Bool`].
    #[must_use]
    pub const fn bool() -> Self {
        Self::Scalar(Scalar::Bool)
    }

    /// Shorthand for the `void` type.
    #[must_use]
    pub const fn void() -> Self {
        Self::Void
    }

    /// True for the error placeholder.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error)
    }

    /// True for the never type.
    #[must_use]
    pub const fn is_never(&self) -> bool {
        matches!(self, Self::Never)
    }

    /// True for `void` and the never type (both mean "no value").
    #[must_use]
    pub const fn is_unit_like(&self) -> bool {
        matches!(self, Self::Void | Self::Never)
    }

    /// True for any integer scalar or integer literal.
    #[must_use]
    pub const fn is_int_like(&self) -> bool {
        matches!(self, Self::IntLit(_)) || matches!(self, Self::Scalar(s) if s.is_int())
    }

    /// True for any float scalar or float literal.
    #[must_use]
    pub const fn is_float_like(&self) -> bool {
        matches!(self, Self::FloatLit(_)) || matches!(self, Self::Scalar(s) if s.is_float())
    }

    /// True for anything arithmetic: integers, floats, and their literals.
    #[must_use]
    pub const fn is_numeric_like(&self) -> bool {
        self.is_int_like() || self.is_float_like()
    }

    /// The scalar inside a vector, or the type itself when scalar.
    #[must_use]
    pub fn elem(&self) -> Option<&Scalar> {
        match self {
            Self::Scalar(scalar) => Some(scalar),
            Self::Vector { elem, .. } => Some(elem),
            _ => None,
        }
    }

    /// The lane count of a vector.
    #[must_use]
    pub const fn lanes(&self) -> Option<u8> {
        match self {
            Self::Vector { lanes, .. } => Some(*lanes),
            Self::Scalar(_) => Some(1),
            _ => None,
        }
    }

    /// Builds a vector of `lanes` lanes over `elem`.
    #[must_use]
    pub const fn vector(elem: Scalar, lanes: u8) -> Self {
        if lanes == 1 {
            Self::Scalar(elem)
        } else {
            Self::Vector { elem, lanes }
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => f.write_str("{unknown}"),
            Self::Never => f.write_str("!"),
            Self::Void => f.write_str("void"),
            Self::Scalar(scalar) => scalar.fmt(f),
            Self::Vector { elem, lanes } => write!(f, "{elem}{lanes}"),
            Self::Ptr { mutable, inner } => {
                let kind = if *mutable { "mut" } else { "const" };
                write!(f, "*{kind} {inner}")
            }
            Self::Ref { mutable, inner } => {
                if *mutable {
                    write!(f, "&mut {inner}")
                } else {
                    write!(f, "&{inner}")
                }
            }
            Self::Array { elem, len: None } => write!(f, "[{elem}]"),
            Self::Array { elem, len: Some(len) } => write!(f, "[{elem}; {len}]"),
            Self::Tuple(elems) => {
                f.write_str("(")?;
                for (index, elem) in elems.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    elem.fmt(f)?;
                }
                if elems.len() == 1 {
                    f.write_str(",")?;
                }
                f.write_str(")")
            }
            Self::Struct { name, args } => {
                f.write_str(name)?;
                if args.is_empty() {
                    return Ok(());
                }
                f.write_str("<")?;
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    arg.fmt(f)?;
                }
                f.write_str(">")
            }
            Self::Generic(name) => f.write_str(name),
            Self::IntLit(_) => f.write_str("{integer}"),
            Self::FloatLit(_) => f.write_str("{float}"),
        }
    }
}

/// Resolves a built-in type name such as `int`, `i32`, `float4`, or `void`.
///
/// Names with a trailing lane count (`int4`, `float16`) resolve to
/// [`Ty::Vector`]. Lane counts OpenCL does not define are reported as
/// [`BuiltinType::BadLanes`] so the analyzer can blame the suffix.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::{BuiltinType, Scalar, Ty, lookup_builtin};
///
/// assert_eq!(lookup_builtin("int"), Some(BuiltinType::Type(Ty::Scalar(Scalar::I32))));
/// assert_eq!(lookup_builtin("i32"), Some(BuiltinType::Type(Ty::Scalar(Scalar::I32))));
/// assert!(matches!(lookup_builtin("float3"), Some(BuiltinType::Type(Ty::Vector { lanes: 3, .. }))));
/// assert!(matches!(lookup_builtin("int9"), Some(BuiltinType::BadLanes(_))));
/// assert_eq!(lookup_builtin("Widget"), None);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinType {
    /// The name resolved to a type.
    Type(Ty),
    /// The base type exists but the lane count does not.
    BadLanes(Scalar),
}

/// Resolves `name` against the built-in type table; `None` when unknown.
#[must_use]
pub fn lookup_builtin(name: &str) -> Option<BuiltinType> {
    // Whole-name lookup first: scalar names such as `i32`, `u8`, and `f64`
    // end in digits but are not vector types, so lane splitting must not
    // run before they get a chance to match.
    if let Some(scalar) = lookup_scalar(name) {
        return Some(BuiltinType::Type(Ty::Scalar(scalar)));
    }
    let (base, lanes) = split_lanes(name);
    if base == "void" {
        return if lanes.is_none() {
            Some(BuiltinType::Type(Ty::Void))
        } else {
            None
        };
    }
    let scalar = lookup_scalar(base)?;
    let Some(lanes) = lanes else {
        return Some(BuiltinType::Type(Ty::Scalar(scalar)));
    };
    if matches!(lanes, 2 | 3 | 4 | 8 | 16) {
        Some(BuiltinType::Type(Ty::Vector { elem: scalar, lanes }))
    } else {
        Some(BuiltinType::BadLanes(scalar))
    }
}

/// Splits a trailing lane count off a type name: `("int", Some(4))`.
fn split_lanes(name: &str) -> (&str, Option<u8>) {
    let digits = name
        .as_bytes()
        .iter()
        .rev()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digits == 0 || digits == name.len() {
        return (name, None);
    }
    let (base, suffix) = name.split_at(name.len() - digits);
    let lanes = suffix
        .parse::<u8>()
        .ok()
        .filter(|value| *value > 0);
    (base, lanes)
}

/// Resolves a scalar (single-lane) built-in name, lanes excluded.
fn lookup_scalar(name: &str) -> Option<Scalar> {
    match name {
        "bool" => Some(Scalar::Bool),
        "char" | "i8" => Some(Scalar::I8),
        "short" | "i16" => Some(Scalar::I16),
        "int" | "i32" => Some(Scalar::I32),
        "long" | "i64" => Some(Scalar::I64),
        "uchar" | "u8" => Some(Scalar::U8),
        "ushort" | "u16" => Some(Scalar::U16),
        "uint" | "u32" => Some(Scalar::U32),
        "ulong" | "u64" => Some(Scalar::U64),
        "isize" => Some(Scalar::I64),
        "usize" | "size_t" => Some(Scalar::U64),
        "ptrdiff_t" => Some(Scalar::I64),
        "half" | "f16" => Some(Scalar::F16),
        "float" | "f32" => Some(Scalar::F32),
        "double" | "f64" => Some(Scalar::F64),
        _ => None,
    }
}

/// True when `src` may be used where `dst` is expected.
///
/// The rules, in order:
///
/// * [`Ty::Error`], [`Ty::Never`], and [`Ty::Generic`] coerce both ways so
///   one bad expression never produces a cascade of errors.
/// * Literals coerce to any type that can hold their value.
/// * Integers widen to any other integer and convert to floats; floats
///   never implicitly convert back to integers.
/// * `bool` converts only to `bool`.
/// * A scalar splats into a vector of the same element; vectors match
///   lane-for-lane.
/// * Pointers widen from `*mut` to `*const`; other composites match
///   structurally.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::{Scalar, Ty, coerce};
///
/// let int = Ty::Scalar(Scalar::I32);
/// let float = Ty::Scalar(Scalar::F32);
/// assert!(coerce(&Ty::IntLit(1), &int));
/// assert!(coerce(&int, &float));
/// assert!(!coerce(&float, &int));
/// assert!(coerce(&Ty::IntLit(1), &Ty::vector(Scalar::I32, 4)));
/// ```
#[must_use]
pub fn coerce(src: &Ty, dst: &Ty) -> bool {
    if src.is_error() || dst.is_error() || src.is_never() {
        return true;
    }
    if matches!(src, Ty::Generic(_)) || matches!(dst, Ty::Generic(_)) {
        return true;
    }
    if src == dst {
        return true;
    }
    match (src, dst) {
        // Literals coerce to any literal of a compatible kind: two
        // integer literals in the same `if` or array never disagree.
        (Ty::IntLit(_), Ty::IntLit(_)) | (Ty::FloatLit(_), Ty::FloatLit(_)) => true,
        // An integer literal widens to a float literal (`1 + 1.5`).
        (Ty::IntLit(_), Ty::FloatLit(_)) => true,
        (Ty::IntLit(value), Ty::Scalar(scalar)) => literal_fits_int(*value, *scalar),
        (Ty::IntLit(value), Ty::Vector { elem, .. }) => literal_fits_int(*value, *elem),
        (Ty::FloatLit(value), Ty::Scalar(scalar)) => scalar.is_float() && float_fits(value, *scalar),
        (Ty::FloatLit(value), Ty::Vector { elem, .. }) => elem.is_float() && float_fits(value, *elem),
        (Ty::Scalar(src), Ty::Scalar(dst)) => scalar_coerce(*src, *dst),
        (Ty::Scalar(src), Ty::Vector { elem: dst, .. }) => scalar_coerce(*src, *dst),
        (Ty::Vector { elem: src, .. }, Ty::Vector { elem: dst, .. }) => scalar_coerce(*src, *dst),
        (
            Ty::Ptr {
                mutable: src_mut,
                inner: src_inner,
            },
            Ty::Ptr {
                mutable: dst_mut,
                inner: dst_inner,
            },
        ) => (!*dst_mut || *src_mut) && src_inner == dst_inner,
        _ => false,
    }
}

/// True when an integer literal value fits `scalar` (int or float).
fn literal_fits_int(value: i128, scalar: Scalar) -> bool {
    if scalar.is_int() {
        return scalar.fits_int(value);
    }
    if scalar.is_float() {
        let float = value as f64;
        return float.is_finite() && float.abs() <= scalar_max_magnitude(scalar);
    }
    false
}

/// Largest magnitude representable by a float scalar.
const fn scalar_max_magnitude(scalar: Scalar) -> f64 {
    match scalar {
        Scalar::F16 => 65_504.0,
        Scalar::F32 => f32::MAX as f64,
        Scalar::F64 => f64::MAX,
        _ => 0.0,
    }
}

/// True when an optional literal value is representable by `scalar`.
fn float_fits(value: &Option<f64>, scalar: Scalar) -> bool {
    match value {
        None => true,
        Some(value) => value.is_finite() && value.abs() <= scalar_max_magnitude(scalar),
    }
}

/// Scalar-to-scalar conversion without literals.
fn scalar_coerce(src: Scalar, dst: Scalar) -> bool {
    match (src, dst) {
        (a, b) if a == b => true,
        (src, dst) if src.is_int() && dst.is_int() => true,
        (src, dst) if src.is_int() && dst.is_float() => true,
        (src, dst) if src.is_float() && dst.is_float() => true,
        _ => false,
    }
}

/// Replaces generic parameters inside `ty` with the given arguments.
///
/// Parameters without a matching argument are left in place, so a type
/// whose generics were never constrained simply stays generic.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::{Scalar, Ty};
///
/// let pair = Ty::Tuple(vec![Ty::Generic(String::from("T")), Ty::Generic(String::from("T"))]);
/// let substituted = codevar_ocl_sar::substitute(&pair, &[String::from("T")], &[Ty::Scalar(Scalar::I32)]);
/// assert_eq!(substituted.to_string(), "(int, int)");
/// ```
#[must_use]
pub fn substitute(ty: &Ty, params: &[String], args: &[Ty]) -> Ty {
    match ty {
        Ty::Generic(name) => params
            .iter()
            .position(|param| param == name)
            .and_then(|index| args.get(index))
            .cloned()
            .unwrap_or_else(|| ty.clone()),
        Ty::Ptr { mutable, inner } => Ty::Ptr {
            mutable: *mutable,
            inner: Box::new(substitute(inner, params, args)),
        },
        Ty::Ref { mutable, inner } => Ty::Ref {
            mutable: *mutable,
            inner: Box::new(substitute(inner, params, args)),
        },
        Ty::Array { elem, len } => Ty::Array {
            elem: Box::new(substitute(elem, params, args)),
            len: *len,
        },
        Ty::Tuple(elems) => Ty::Tuple(
            elems
                .iter()
                .map(|elem| substitute(elem, params, args))
                .collect(),
        ),
        Ty::Struct { name, args: inner } => Ty::Struct {
            name: name.clone(),
            args: inner
                .iter()
                .map(|arg| substitute(arg, params, args))
                .collect(),
        },
        other => other.clone(),
    }
}

/// Matches an expected type against an actual one, collecting the generic
/// substitutions the match implies.
///
/// Returns `false` when the types cannot be unified, which the analyzer
/// reports as a type mismatch.
pub fn unify(expected: &Ty, actual: &Ty, substitutions: &mut Vec<(String, Ty)>) -> bool {
    if expected.is_error() || actual.is_error() || expected.is_never() || actual.is_never() {
        return true;
    }
    match (expected, actual) {
        (Ty::Generic(name), other) => {
            if let Some((_, seen)) = substitutions
                .iter()
                .find(|(param, _)| param == name)
            {
                return coerce(other, seen);
            }
            substitutions.push((name.clone(), other.clone()));
            true
        }
        (Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Array { .. }, _) => {
            coerce_matches(expected, actual, substitutions)
        }
        (Ty::Tuple(expected_elems), Ty::Tuple(actual_elems)) => {
            expected_elems.len() == actual_elems.len()
                && expected_elems
                    .iter()
                    .zip(actual_elems)
                    .all(|(expected, actual)| unify(expected, actual, substitutions))
        }
        (
            Ty::Struct { name, args },
            Ty::Struct {
                name: actual_name,
                args: actual_args,
            },
        ) => {
            name == actual_name
                && args.len() == actual_args.len()
                && args
                    .iter()
                    .zip(actual_args)
                    .all(|(expected, actual)| unify(expected, actual, substitutions))
        }
        _ => coerce_matches(expected, actual, substitutions),
    }
}

/// Structural match for the composite cases that [`unify`] delegates to.
fn coerce_matches(expected: &Ty, actual: &Ty, substitutions: &mut Vec<(String, Ty)>) -> bool {
    match (expected, actual) {
        (Ty::Ptr { inner, .. }, Ty::Ptr { inner: actual, .. })
        | (Ty::Ref { inner, .. }, Ty::Ref { inner: actual, .. })
        | (Ty::Array { elem: inner, .. }, Ty::Array { elem: actual, .. }) => {
            unify(inner, actual, substitutions)
        }
        _ => coerce(actual, expected),
    }
}
