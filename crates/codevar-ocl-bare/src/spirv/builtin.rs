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

//! Extended instructions and imported functions the interpreter knows.
//!
//! Two call surfaces reach this module from the loader and the
//! interpreter:
//!
//! * `OpExtInst` names an instruction of the imported `OpenCL.std` set;
//!   [`ext_inst_arity`] is the single table of which instructions are
//!   supported *and* how many arguments each takes, so an unsupported
//!   or malformed call is rejected while the binary is loaded rather
//!   than mid-kernel.
//! * A linkage `Import` declaration (`OpFunction` without a body plus a
//!   `LinkageAttributes` name) is dispatched by name at every
//!   `OpFunctionCall`; [`import_fn`] maps the `get_*` work-item
//!   queries — plain or Itanium-mangled — onto their built-in values.
//!
//! Instructions that need an out-pointer (`frexp`, `modf`, `fract`,
//! `lgamma_r`, `remquo`, `sincos`), the packed `vload*`/`vstore*`
//! family, `shuffle*` and `printf` are deliberately absent: the
//! embedded profile does not require them, and a module that uses one
//! fails to load with [`crate::Error::InvalidBinary`] instead of
//! running half a kernel.

use super::ops;

/// The exact argument count of `inst` when this driver implements the
/// `OpenCL.std` extended instruction, or `None` when the instruction is
/// rejected at load time.
pub(crate) fn ext_inst_arity(inst: u32) -> Option<u32> {
    Some(match inst {
        // One argument: transcendentals, rounding, integer helpers.
        ops::OCL_ACOS
        | ops::OCL_ACOSH
        | ops::OCL_ACOSPI
        | ops::OCL_ASIN
        | ops::OCL_ASINH
        | ops::OCL_ASINPI
        | ops::OCL_ATAN
        | ops::OCL_ATANH
        | ops::OCL_ATANPI
        | ops::OCL_CBRT
        | ops::OCL_CEIL
        | ops::OCL_COS
        | ops::OCL_COSH
        | ops::OCL_COSPI
        | ops::OCL_ERFC
        | ops::OCL_ERF
        | ops::OCL_EXP
        | ops::OCL_EXP2
        | ops::OCL_EXP10
        | ops::OCL_EXPM1
        | ops::OCL_FABS
        | ops::OCL_FLOOR
        | ops::OCL_ILOGB
        | ops::OCL_LGAMMA
        | ops::OCL_LOG
        | ops::OCL_LOG2
        | ops::OCL_LOG10
        | ops::OCL_LOG1P
        | ops::OCL_LOGB
        | ops::OCL_NAN
        | ops::OCL_RINT
        | ops::OCL_ROUND
        | ops::OCL_RSQRT
        | ops::OCL_SIN
        | ops::OCL_SINH
        | ops::OCL_SINPI
        | ops::OCL_SQRT
        | ops::OCL_TAN
        | ops::OCL_TANH
        | ops::OCL_TANPI
        | ops::OCL_TGAMMA
        | ops::OCL_TRUNC
        | ops::OCL_HALF_COS
        | ops::OCL_HALF_EXP
        | ops::OCL_HALF_EXP2
        | ops::OCL_HALF_EXP10
        | ops::OCL_HALF_LOG
        | ops::OCL_HALF_LOG2
        | ops::OCL_HALF_LOG10
        | ops::OCL_HALF_RECIP
        | ops::OCL_HALF_RSQRT
        | ops::OCL_HALF_SIN
        | ops::OCL_HALF_SQRT
        | ops::OCL_HALF_TAN
        | ops::OCL_NATIVE_COS
        | ops::OCL_NATIVE_EXP
        | ops::OCL_NATIVE_EXP2
        | ops::OCL_NATIVE_EXP10
        | ops::OCL_NATIVE_LOG
        | ops::OCL_NATIVE_LOG2
        | ops::OCL_NATIVE_LOG10
        | ops::OCL_NATIVE_RECIP
        | ops::OCL_NATIVE_RSQRT
        | ops::OCL_NATIVE_SIN
        | ops::OCL_NATIVE_SQRT
        | ops::OCL_NATIVE_TAN
        | ops::OCL_DEGREES
        | ops::OCL_RADIANS
        | ops::OCL_SIGN
        | ops::OCL_LENGTH
        | ops::OCL_NORMALIZE
        | ops::OCL_FAST_LENGTH
        | ops::OCL_FAST_NORMALIZE
        | ops::OCL_S_ABS
        | ops::OCL_CLZ
        | ops::OCL_CTZ
        | ops::OCL_POPCOUNT
        | ops::OCL_U_ABS
        | ops::OCL_PREFETCH => 1,

        // Two arguments: paired arithmetic, integer min/max/diff.
        ops::OCL_ATAN2
        | ops::OCL_ATAN2PI
        | ops::OCL_COPYSIGN
        | ops::OCL_FDIM
        | ops::OCL_FMAX
        | ops::OCL_FMIN
        | ops::OCL_FMOD
        | ops::OCL_HYPOT
        | ops::OCL_LDEXP
        | ops::OCL_MAXMAG
        | ops::OCL_MINMAG
        | ops::OCL_NEXTAFTER
        | ops::OCL_POW
        | ops::OCL_POWN
        | ops::OCL_POWR
        | ops::OCL_REMAINDER
        | ops::OCL_ROOTN
        | ops::OCL_HALF_DIVIDE
        | ops::OCL_HALF_POWR
        | ops::OCL_NATIVE_DIVIDE
        | ops::OCL_NATIVE_POWR
        | ops::OCL_F_MAX_COMMON
        | ops::OCL_F_MIN_COMMON
        | ops::OCL_STEP
        | ops::OCL_CROSS
        | ops::OCL_DISTANCE
        | ops::OCL_FAST_DISTANCE
        | ops::OCL_S_ABS_DIFF
        | ops::OCL_S_ADD_SAT
        | ops::OCL_U_ADD_SAT
        | ops::OCL_S_HADD
        | ops::OCL_U_HADD
        | ops::OCL_S_RHADD
        | ops::OCL_U_RHADD
        | ops::OCL_S_MAX
        | ops::OCL_U_MAX
        | ops::OCL_S_MIN
        | ops::OCL_U_MIN
        | ops::OCL_S_MUL_HI
        | ops::OCL_ROTATE
        | ops::OCL_S_SUB_SAT
        | ops::OCL_U_SUB_SAT
        | ops::OCL_U_UPSAMPLE
        | ops::OCL_S_UPSAMPLE
        | ops::OCL_S_MUL24
        | ops::OCL_U_MUL24
        | ops::OCL_U_ABS_DIFF
        | ops::OCL_U_MUL_HI => 2,

        // Three arguments: fused multiply-adds, clamps, selects.
        ops::OCL_FMA
        | ops::OCL_MAD
        | ops::OCL_F_CLAMP
        | ops::OCL_MIX
        | ops::OCL_SMOOTHSTEP
        | ops::OCL_S_CLAMP
        | ops::OCL_U_CLAMP
        | ops::OCL_S_MAD_HI
        | ops::OCL_U_MAD_SAT
        | ops::OCL_S_MAD_SAT
        | ops::OCL_S_MAD24
        | ops::OCL_U_MAD24
        | ops::OCL_U_MAD_HI
        | ops::OCL_BITSELECT
        | ops::OCL_SELECT => 3,

        _ => return None,
    })
}

/// Whether `inst` is an `OpenCL.std` instruction this driver executes.
#[must_use]
#[inline]
pub(crate) fn ext_inst_supported(inst: u32) -> bool {
    ext_inst_arity(inst).is_some()
}

/// A linkage-imported function the interpreter implements directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImportFn {
    /// `get_work_dim()` — takes no arguments.
    WorkDim,
    /// `get_global_id(dim)`.
    GlobalId,
    /// `get_local_id(dim)`.
    LocalId,
    /// `get_group_id(dim)`.
    GroupId,
    /// `get_global_offset(dim)`.
    GlobalOffset,
    /// `get_global_size(dim)`.
    GlobalSize,
    /// `get_local_size(dim)`.
    LocalSize,
    /// `get_num_groups(dim)`.
    NumGroups,
    /// `get_global_linear_id(dim)`.
    GlobalLinearId,
    /// `get_enqueued_local_size(dim)`.
    EnqueuedLocalSize,
}

impl ImportFn {
    /// The argument count every call site of this function must pass.
    #[must_use]
    #[inline]
    pub(crate) const fn argc(self) -> u32 {
        match self {
            Self::WorkDim => 0,
            Self::GlobalId
            | Self::LocalId
            | Self::GroupId
            | Self::GlobalOffset
            | Self::GlobalSize
            | Self::LocalSize
            | Self::NumGroups
            | Self::GlobalLinearId
            | Self::EnqueuedLocalSize => 1,
        }
    }
}

/// Resolves a linkage-import name to its interpreter implementation.
///
/// Accepts both the plain C name the workspace emits
/// (`"get_global_id"`) and the Itanium-mangled form other toolchains
/// use (`"_Z13get_global_idj"`); the mangled form is reduced to its
/// length-prefixed identifier before the lookup.
#[must_use]
pub(crate) fn import_fn(name: &str) -> Option<ImportFn> {
    match strip_itanium(name) {
        "get_work_dim" => Some(ImportFn::WorkDim),
        "get_global_id" => Some(ImportFn::GlobalId),
        "get_local_id" => Some(ImportFn::LocalId),
        "get_group_id" => Some(ImportFn::GroupId),
        "get_global_offset" => Some(ImportFn::GlobalOffset),
        "get_global_size" => Some(ImportFn::GlobalSize),
        "get_local_size" => Some(ImportFn::LocalSize),
        "get_num_groups" => Some(ImportFn::NumGroups),
        "get_global_linear_id" => Some(ImportFn::GlobalLinearId),
        "get_enqueued_local_size" => Some(ImportFn::EnqueuedLocalSize),
        _ => None,
    }
}

/// Reduces `"_Z13get_global_idj"` to `"get_global_id"`; every other
/// name is returned unchanged.
fn strip_itanium(name: &str) -> &str {
    let Some(rest) = name.strip_prefix("_Z") else {
        return name;
    };
    let digits = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits == 0 {
        return name;
    }
    let Ok(len) = rest[..digits].parse::<usize>() else {
        return name;
    };
    let ident = &rest[digits..];
    if ident.len() >= len && ident.is_char_boundary(len) {
        &ident[..len]
    } else {
        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_table_covers_the_implemented_set() {
        assert_eq!(ext_inst_arity(ops::OCL_SQRT), Some(1));
        assert_eq!(ext_inst_arity(ops::OCL_POW), Some(2));
        assert_eq!(ext_inst_arity(ops::OCL_FMA), Some(3));
        assert_eq!(ext_inst_arity(ops::OCL_PREFETCH), Some(1));
        // Rejected: out-pointers, packed memory, printf.
        assert_eq!(ext_inst_arity(ops::OCL_FREXP), None);
        assert_eq!(ext_inst_arity(ops::OCL_MODF), None);
        assert_eq!(ext_inst_arity(ops::OCL_PRINTF), None);
        assert_eq!(ext_inst_arity(ops::OCL_VLOADN), None);
        assert_eq!(ext_inst_arity(ops::OCL_SHUFFLE), None);
        // Out of range entirely.
        assert_eq!(ext_inst_arity(9999), None);
    }

    #[test]
    fn import_names_resolve_plain_and_mangled() {
        assert_eq!(import_fn("get_global_id"), Some(ImportFn::GlobalId));
        assert_eq!(
            import_fn("_Z13get_global_idj"),
            Some(ImportFn::GlobalId)
        );
        assert_eq!(import_fn("get_work_dim"), Some(ImportFn::WorkDim));
        assert_eq!(import_fn("printf"), None);
        assert_eq!(import_fn("_Zbad"), None);
        assert_eq!(ImportFn::WorkDim.argc(), 0);
        assert_eq!(ImportFn::GlobalId.argc(), 1);
    }
}
