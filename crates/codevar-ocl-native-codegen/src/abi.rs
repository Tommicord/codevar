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

//! The facts the pipeline needs about the kernel ABI.
//!
//! The dialect reaches the work-item state and the kernel arguments
//! through the intermediate representation, not through machine
//! operations: a work-item query is a load of a decorated module
//! variable, and a kernel argument is a parameter of the entry
//! function.  This module turns those two shapes into the machine
//! operations that read the very same bytes, so the change is made
//! once, in the selection stage, and the back ends only ever see plain
//! loads.
//!
//! # The work-item state
//!
//! Six of the work-item queries are fields of the per-work-item state
//! published by [`codevar_ocl_abi::WorkItem`].  The remaining ones are
//! computed from those fields: the global size is the group count
//! times the local size, the enqueued local size is the local size,
//! and the linear identifiers are the mixed-radix sums of the
//! multi-dimensional ones.

use codevar_ocl_abi::WorkItemField;
use codevar_ocl_ir::spirv::ops::ocl_name;

use crate::mir::{ArgKind, MType};

/// Where a work-item query reads its value from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkItemSource {
    /// A field of the per-work-item state.
    Field(WorkItemField),
    /// The global size of the launch: the group count times the local
    /// size, component by component.
    GlobalSize,
    /// The enqueued local size of the launch, which is the local
    /// size.
    EnqueuedLocalSize,
    /// The linear global id of the work-item.
    GlobalLinearId,
    /// The linear local id of the work-item.
    LocalLinearId,
}

impl WorkItemSource {
    /// The field the source reads, when it reads one directly.
    #[must_use]
    pub const fn field(self) -> Option<WorkItemField> {
        match self {
            Self::Field(field) => Some(field),
            _ => None,
        }
    }
}

/// Maps a built-in variable name to the work-item state it comes from.
///
/// The names are the ones the intermediate representation uses as
/// built-in decorations, which follow the OpenCL environment's naming
/// of the work-item built-ins.
///
/// # Example
///
/// ```
/// use codevar_ocl_native_codegen::abi::{WorkItemSource, work_item_source};
/// use codevar_ocl_abi::WorkItemField;
///
/// let source = work_item_source("GlobalInvocationId").expect("a known built-in");
/// assert_eq!(source.field(), Some(WorkItemField::GlobalId));
/// assert!(work_item_source("NotABuiltIn").is_none());
/// ```
#[must_use]
pub fn work_item_source(builtin: &str) -> Option<WorkItemSource> {
    let source = match builtin {
        "GlobalInvocationId" => WorkItemSource::Field(WorkItemField::GlobalId),
        "LocalInvocationId" => WorkItemSource::Field(WorkItemField::LocalId),
        "WorkgroupId" => WorkItemSource::Field(WorkItemField::GroupId),
        "GlobalOffset" => WorkItemSource::Field(WorkItemField::GlobalOffset),
        "NumWorkgroups" => WorkItemSource::Field(WorkItemField::NumGroups),
        "WorkgroupSize" => WorkItemSource::Field(WorkItemField::LocalSize),
        "GlobalSize" => WorkItemSource::GlobalSize,
        "EnqueuedWorkgroupSize" => WorkItemSource::EnqueuedLocalSize,
        "GlobalLinearId" => WorkItemSource::GlobalLinearId,
        "LocalInvocationIndex" => WorkItemSource::LocalLinearId,
        _ => return None,
    };
    Some(source)
}

/// A mathematical operation the native back ends implement directly.
///
/// The list is short on purpose: every entry has a counterpart in the
/// baseline instruction set of each supported architecture, so a
/// compiled kernel never needs a helper routine and never needs an
/// absolute address.  The remaining mathematical built-ins are
/// reported as unsupported rather than approximated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathOp {
    /// The square root.
    Sqrt,
    /// The absolute value.
    Abs,
    /// The smaller of two values.
    Min,
    /// The larger of two values.
    Max,
    /// The positive difference of two values.
    Dim,
    /// The magnitude of one value with the sign of another.
    CopySign,
}

/// Maps an extended-instruction number to the mathematical operation it
/// stands for, when the native back ends implement it.
///
/// # Example
///
/// ```
/// use codevar_ocl_native_codegen::abi::{MathOp, math_op};
/// use codevar_ocl_ir::spirv::ops::ocl_opcode;
///
/// let opcode = ocl_opcode("sqrt").expect("the dialect knows the built-in");
/// assert_eq!(math_op(opcode), Some(MathOp::Sqrt));
/// ```
#[must_use]
pub fn math_op(opcode: u32) -> Option<MathOp> {
    let op = match ocl_name(opcode) {
        "sqrt" | "native_sqrt" | "half_sqrt" => MathOp::Sqrt,
        "fabs" => MathOp::Abs,
        "fmin" | "fmin_common" => MathOp::Min,
        "fmax" | "fmax_common" => MathOp::Max,
        "fdim" => MathOp::Dim,
        "copysign" => MathOp::CopySign,
        _ => return None,
    };
    Some(op)
}

/// The name of an extended instruction, for diagnostics.
#[must_use]
pub fn extended_instruction_name(opcode: u32) -> &'static str {
    ocl_name(opcode)
}

/// How a kernel argument is marshalled.
///
/// A buffer argument is a pointer stored in the argument slot; a
/// scalar argument is the value itself, stored in the first bytes of
/// the slot.
#[must_use]
pub fn argument_kind(ty: &MType) -> ArgKind {
    match ty {
        MType::Ptr { .. } => ArgKind::Pointer,
        other => ArgKind::Scalar {
            size: other.size_in_bytes().unwrap_or(4),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_ocl_ir::spirv::ops::ocl_opcode;

    #[test]
    fn every_work_item_field_is_reachable() {
        for field in WorkItemField::ALL {
            let builtin = match field {
                WorkItemField::GlobalId => "GlobalInvocationId",
                WorkItemField::LocalId => "LocalInvocationId",
                WorkItemField::GroupId => "WorkgroupId",
                WorkItemField::GlobalOffset => "GlobalOffset",
                WorkItemField::NumGroups => "NumWorkgroups",
                WorkItemField::LocalSize => "WorkgroupSize",
            };
            let source = work_item_source(builtin).expect("a known built-in");
            assert_eq!(source.field(), Some(field));
        }
    }

    #[test]
    fn pointer_arguments_are_pointers() {
        assert_eq!(argument_kind(&MType::Ptr { size: 4 }), ArgKind::Pointer);
    }

    #[test]
    fn scalar_arguments_carry_their_size() {
        assert_eq!(argument_kind(&MType::I32), ArgKind::Scalar { size: 4 });
        assert_eq!(argument_kind(&MType::F64), ArgKind::Scalar { size: 8 });
    }

    #[test]
    fn the_square_root_is_supported() {
        let opcode = ocl_opcode("sqrt").expect("the dialect knows the built-in");
        assert_eq!(math_op(opcode), Some(MathOp::Sqrt));
    }

    #[test]
    fn an_unsupported_built_in_has_no_operation() {
        let opcode = ocl_opcode("tanh").expect("the dialect knows the built-in");
        assert_eq!(math_op(opcode), None);
    }
}
