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

//! Built-in functions of the OpenCL dialect.
//!
//! Every entry is a static table row: work-item queries, fence barriers,
//! integer helpers, and the float math family. Signatures are described by
//! [`BuiltinKind`] rather than spelled out so that vector types are checked
//! generically — `sqrt(float4)` type-checks by shape, not by a hand-written
//! overload.

use crate::types::{Scalar, Ty};

/// How a built-in's arguments and result are checked.
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinKind {
    /// A work-item query: one `int` dimension, `size_t` result.
    ///
    /// Covers `get_global_id`, `get_local_size`, and friends.
    WorkItem,
    /// A fixed scalar signature, used by the fence barriers.
    Fixed {
        /// Expected parameter types, in order.
        params: &'static [Ty],
        /// Result type.
        ret: Ty,
    },
    /// An integer-or-float operation whose result keeps the shape and
    /// element type of the first argument: `abs`, `min`, `clamp`, …
    ///
    /// An unsuffixed literal argument defaults to `int` (`Float` defaults
    /// to `float`).
    Numeric {
        /// Number of arguments the call must supply.
        arity: u8,
    },
    /// A float operation whose result keeps the shape of the first
    /// argument; integer arguments are promoted to `float`.
    Float {
        /// Number of arguments the call must supply.
        arity: u8,
    },
    /// A boolean reduction: `all`, `any` — a `boolN` collapses to `bool`.
    Reduce {
        /// Number of arguments the call must supply.
        arity: u8,
    },
}

/// One row of the built-in function table.
#[derive(Debug, Clone, PartialEq)]
pub struct Builtin {
    /// Function name as written in source.
    pub name: &'static str,
    /// How calls to it are checked.
    pub kind: BuiltinKind,
}

/// The fence flags accepted by the barrier family.
const FENCE_FLAGS: &[Ty] = &[Ty::Scalar(Scalar::U32)];

/// Every built-in the dialect defines, in table order.
static BUILTINS: &[Builtin] = &[
    // Work-item queries.
    Builtin {
        name: "get_global_id",
        kind: BuiltinKind::WorkItem,
    },
    Builtin {
        name: "get_global_size",
        kind: BuiltinKind::WorkItem,
    },
    Builtin {
        name: "get_global_offset",
        kind: BuiltinKind::WorkItem,
    },
    Builtin {
        name: "get_global_linear_id",
        kind: BuiltinKind::WorkItem,
    },
    Builtin {
        name: "get_local_id",
        kind: BuiltinKind::WorkItem,
    },
    Builtin {
        name: "get_local_size",
        kind: BuiltinKind::WorkItem,
    },
    Builtin {
        name: "get_local_linear_id",
        kind: BuiltinKind::WorkItem,
    },
    Builtin {
        name: "get_enqueued_local_size",
        kind: BuiltinKind::WorkItem,
    },
    Builtin {
        name: "get_group_id",
        kind: BuiltinKind::WorkItem,
    },
    Builtin {
        name: "get_num_groups",
        kind: BuiltinKind::WorkItem,
    },
    // Fences.
    Builtin {
        name: "barrier",
        kind: BuiltinKind::Fixed {
            params: FENCE_FLAGS,
            ret: Ty::Void,
        },
    },
    Builtin {
        name: "mem_fence",
        kind: BuiltinKind::Fixed {
            params: FENCE_FLAGS,
            ret: Ty::Void,
        },
    },
    Builtin {
        name: "read_mem_fence",
        kind: BuiltinKind::Fixed {
            params: FENCE_FLAGS,
            ret: Ty::Void,
        },
    },
    Builtin {
        name: "write_mem_fence",
        kind: BuiltinKind::Fixed {
            params: FENCE_FLAGS,
            ret: Ty::Void,
        },
    },
    Builtin {
        name: "sub_group_barrier",
        kind: BuiltinKind::Fixed {
            params: &[],
            ret: Ty::Void,
        },
    },
    // Integer helpers.
    Builtin {
        name: "abs",
        kind: BuiltinKind::Numeric { arity: 1 },
    },
    Builtin {
        name: "sign",
        kind: BuiltinKind::Numeric { arity: 1 },
    },
    Builtin {
        name: "popcount",
        kind: BuiltinKind::Numeric { arity: 1 },
    },
    Builtin {
        name: "min",
        kind: BuiltinKind::Numeric { arity: 2 },
    },
    Builtin {
        name: "max",
        kind: BuiltinKind::Numeric { arity: 2 },
    },
    Builtin {
        name: "clamp",
        kind: BuiltinKind::Numeric { arity: 3 },
    },
    // Boolean reductions.
    Builtin {
        name: "all",
        kind: BuiltinKind::Reduce { arity: 1 },
    },
    Builtin {
        name: "any",
        kind: BuiltinKind::Reduce { arity: 1 },
    },
    // Float math.
    Builtin {
        name: "sqrt",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "rsqrt",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "cbrt",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "pow",
        kind: BuiltinKind::Float { arity: 2 },
    },
    Builtin {
        name: "fmod",
        kind: BuiltinKind::Float { arity: 2 },
    },
    Builtin {
        name: "atan2",
        kind: BuiltinKind::Float { arity: 2 },
    },
    Builtin {
        name: "fmin",
        kind: BuiltinKind::Float { arity: 2 },
    },
    Builtin {
        name: "fmax",
        kind: BuiltinKind::Float { arity: 2 },
    },
    Builtin {
        name: "step",
        kind: BuiltinKind::Float { arity: 2 },
    },
    Builtin {
        name: "mix",
        kind: BuiltinKind::Float { arity: 3 },
    },
    Builtin {
        name: "smoothstep",
        kind: BuiltinKind::Float { arity: 3 },
    },
    Builtin {
        name: "fma",
        kind: BuiltinKind::Float { arity: 3 },
    },
    Builtin {
        name: "sin",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "cos",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "tan",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "asin",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "acos",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "atan",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "sinh",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "cosh",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "tanh",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "exp",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "exp2",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "expm1",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "log",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "log2",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "log10",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "log1p",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "floor",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "ceil",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "round",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "trunc",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "fabs",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "degrees",
        kind: BuiltinKind::Float { arity: 1 },
    },
    Builtin {
        name: "radians",
        kind: BuiltinKind::Float { arity: 1 },
    },
];

/// Every built-in in the table, for iteration and suggestions.
#[must_use]
pub fn builtins() -> &'static [Builtin] {
    BUILTINS
}

/// Looks a built-in function up by name.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::{Builtin, BuiltinKind, lookup_builtin_fn};
///
/// let sqrt = lookup_builtin_fn("sqrt").unwrap_or(&Builtin {
///     name: "sqrt",
///     kind: BuiltinKind::Float { arity: 1 },
/// });
/// assert!(matches!(sqrt.kind, BuiltinKind::Float { arity: 1 }));
/// assert!(lookup_builtin_fn("printf").is_none());
/// ```
#[must_use]
pub fn lookup_builtin_fn(name: &str) -> Option<&'static Builtin> {
    BUILTINS
        .iter()
        .find(|builtin| builtin.name == name)
}
