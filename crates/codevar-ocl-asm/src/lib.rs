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

//! Assembler stage of the Codevar OpenCL compiler.
//!
//! Turns [`Module`] IR into an object file the driver can hand to a
//! driver or toolchain.  One backend exists today — [`spirv`] — with
//! room for more (the IR already carries the target environment).
//!
//! [`Module`]: codevar_ocl_ir::ir::Module

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

use codevar_ocl_ir::ir::Module;
use codevar_ocl_ir::verify::{VerifyError, verify};

/// The SPIR-V binary backend.
pub mod spirv;

/// Why a module could not be assembled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssembleError {
    /// The IR module failed [`verify`].
    Verify(VerifyError),
    /// The backend does not support a construct the module uses.
    Unsupported {
        /// What was rejected.
        what: &'static str,
    },
    /// A generated opcode table was missing an entry the backend needs.
    Table {
        /// The missing entry's name.
        entry: &'static str,
    },
    /// A single instruction would not fit SPIR-V's 16-bit word count.
    InstructionTooLarge {
        /// The opcode being encoded.
        opcode: u16,
    },
    /// A literal string exceeds SPIR-V's string limit.
    StringTooLong {
        /// The rejected length, in bytes.
        len: usize,
    },
    /// The module ran out of SPIR-V `<id>` space.
    IdExhausted,
}

impl fmt::Display for AssembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verify(error) => write!(f, "IR verification failed: {error}"),
            Self::Unsupported { what } => write!(f, "SPIR-V backend cannot encode {what}"),
            Self::Table { entry } => write!(f, "opcode table entry `{entry}` is missing"),
            Self::InstructionTooLarge { opcode } => {
                write!(f, "instruction {opcode} exceeds the SPIR-V word-count limit")
            }
            Self::StringTooLong { len } => {
                write!(f, "string of {len} bytes exceeds the SPIR-V string limit")
            }
            Self::IdExhausted => f.write_str("SPIR-V id space exhausted"),
        }
    }
}

impl core::error::Error for AssembleError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Verify(error) => Some(error),
            _ => None,
        }
    }
}

impl From<VerifyError> for AssembleError {
    fn from(error: VerifyError) -> Self {
        Self::Verify(error)
    }
}

/// Verifies `module` and assembles it into SPIR-V words.
///
/// # Errors
///
/// Returns [`AssembleError::Verify`] when the IR does not verify, and
/// the backend's error when the module uses an unsupported or malformed
/// construct.
pub fn assemble(module: &Module) -> Result<Vec<u32>, AssembleError> {
    verify(module)?;
    spirv::assemble(module)
}

/// Verifies `module` and assembles it into little-endian SPIR-V bytes.
///
/// # Errors
///
/// Same conditions as [`assemble`].
pub fn assemble_bytes(module: &Module) -> Result<Vec<u8>, AssembleError> {
    let words = assemble(module)?;
    Ok(spirv::to_bytes(&words))
}
