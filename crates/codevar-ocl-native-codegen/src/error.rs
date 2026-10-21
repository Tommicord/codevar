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

//! Failures of the native code generation pipeline.

use alloc::string::String;
use core::fmt;

use codevar_ocl_ir::verify::VerifyError;

/// Result alias for the pipeline.
pub type Result<T> = core::result::Result<T, CodegenError>;

/// Why a compilation unit could not be turned into machine code.
///
/// Every stage reports through this one type, so a caller learns which
/// stage refused the module and why without matching on the internals
/// of that stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodegenError {
    /// The intermediate representation did not verify.
    Verify(VerifyError),
    /// The module uses a construct the native back end cannot express.
    Unsupported {
        /// What kind of construct was rejected.
        what: &'static str,
        /// Where the construct occurs, in the module's own terms.
        detail: String,
    },
    /// The optimizer could not rewrite the module.
    Optimization {
        /// What the optimizer was doing.
        what: &'static str,
        /// Why it gave up.
        detail: String,
    },
    /// The selection stage could not turn the module into machine code.
    Selection {
        /// What the selection stage was doing.
        what: &'static str,
        /// Why it gave up.
        detail: String,
    },
    /// The register allocator could not assign registers.
    Allocation {
        /// What the allocator was doing.
        what: &'static str,
        /// Why it gave up.
        detail: String,
    },
    /// The target back end refused the machine code.
    Target {
        /// What the back end was doing.
        what: &'static str,
        /// Why it gave up.
        detail: String,
    },
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verify(error) => write!(f, "IR verification failed: {error}"),
            Self::Unsupported { what, detail } => {
                write!(f, "the native back end cannot express {what}: {detail}")
            }
            Self::Optimization { what, detail } => {
                write!(f, "the optimizer failed while {what}: {detail}")
            }
            Self::Selection { what, detail } => {
                write!(f, "instruction selection failed while {what}: {detail}")
            }
            Self::Allocation { what, detail } => {
                write!(f, "register allocation failed while {what}: {detail}")
            }
            Self::Target { what, detail } => {
                write!(f, "the target back end failed while {what}: {detail}")
            }
        }
    }
}

impl core::error::Error for CodegenError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Verify(error) => Some(error),
            _ => None,
        }
    }
}

impl From<VerifyError> for CodegenError {
    fn from(error: VerifyError) -> Self {
        Self::Verify(error)
    }
}
