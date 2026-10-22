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

//! The target code API of the Codevar OpenCL compiler.
//!
//! The crate turns a dialect source into the machine code of one
//! kernel: the source is analyzed, parsed, lowered to the
//! intermediate representation, optimized, vectorized, and handed to
//! the back end of the target the program is running on.
//!
//! What comes back is the raw machine code of the kernel: bytes that
//! carry no absolute address, ready to be copied into executable
//! memory and entered at the code's entry point.  The kernel follows
//! the calling convention of [`codevar_ocl_abi`], so it takes the
//! per-work-item state as its single argument.
//!
//! # Example
//!
//! ```
//! use codevar_ocl_tc::compile_source;
//!
//! let source = "#[kernel]\nfn scale(out: *mut int, factor: int) {\n    let i = get_global_id(0);\n    out[i] = out[i] * factor;\n}\n";
//! match compile_source(source, "scale") {
//!     Ok(code) => assert!(!code.bytes().is_empty()),
//!     Err(error) => assert!(!error.to_string().is_empty()),
//! }
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use codevar_ocl_ir::ir::ValueId;
use codevar_ocl_native_codegen::error::CodegenError;
use codevar_ocl_native_codegen::pipeline::{CodegenOptions, compile_entry};
use codevar_ocl_native_codegen::target::NativeTarget;

use codevar_ocl_native_codegen_aarch64::AArch64Target;
use codevar_ocl_native_codegen_thumbv7m::ThumbV7mTarget;
use codevar_ocl_native_codegen_x86::X86Target;

/// The machine code of a compiled kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledCode {
    /// The bytes of the machine code.
    bytes: Vec<u8>,
    /// The offset of the entry point inside the code.
    entry_offset: u32,
    /// The number of machine instructions the code holds.
    instruction_count: u32,
}

impl CompiledCode {
    /// The bytes of the machine code.
    #[must_use]
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The offset of the entry point inside the code.
    #[must_use]
    #[inline]
    pub fn entry_offset(&self) -> u32 {
        self.entry_offset
    }

    /// The number of machine instructions the code holds.
    #[must_use]
    #[inline]
    pub fn instruction_count(&self) -> u32 {
        self.instruction_count
    }
}

/// The back end of the architecture the program is running on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostTarget {
    /// The x86-64 back end.
    X86(X86Target),
    /// The ARMv7-M back end.
    Thumb(ThumbV7mTarget),
    /// The AArch64 back end.
    AArch64(AArch64Target),
}

impl HostTarget {
    /// The back end as a target.
    #[must_use]
    pub fn as_target(&self) -> &dyn NativeTarget {
        match self {
            Self::X86(target) => target,
            Self::Thumb(target) => target,
            Self::AArch64(target) => target,
        }
    }
}

/// The back end of the architecture the program is running on.
///
/// # Errors
///
/// Returns an error when the architecture has no back end yet.
pub fn host_target() -> Result<HostTarget, Error> {
    #[cfg(target_arch = "x86_64")]
    {
        Ok(HostTarget::X86(X86Target::new()))
    }
    #[cfg(target_arch = "aarch64")]
    {
        Ok(HostTarget::AArch64(AArch64Target::new()))
    }
    #[cfg(target_arch = "arm")]
    {
        Ok(HostTarget::Thumb(ThumbV7mTarget::new()))
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
    {
        Err(Error::NoBackEnd {
            architecture: String::from(core::env!("CARGO_CFG_TARGET_ARCH")),
        })
    }
}

/// Why a kernel could not be compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The source did not analyze.
    Analysis {
        /// The diagnostics the analyzer produced.
        diagnostics: Vec<String>,
    },
    /// The source did not parse.
    Parse {
        /// Number of parse errors.
        count: usize,
    },
    /// The source could not be lowered to the intermediate
    /// representation.
    Lowering {
        /// What the lowering stage reported.
        detail: String,
    },
    /// The kernel the caller asked for does not exist.
    UnknownEntry {
        /// The name the caller asked for.
        name: String,
    },
    /// The architecture has no back end.
    NoBackEnd {
        /// The architecture the program is running on.
        architecture: String,
    },
    /// The machine code could not be produced.
    Codegen(CodegenError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Analysis { diagnostics } => {
                write!(f, "the source did not analyze:")?;
                for diagnostic in diagnostics {
                    write!(f, "\n  {diagnostic}")?;
                }
                Ok(())
            }
            Self::Parse { count } => write!(f, "the source has {count} parse errors"),
            Self::Lowering { detail } => write!(f, "lowering failed: {detail}"),
            Self::UnknownEntry { name } => write!(f, "no kernel named '{name}'"),
            Self::NoBackEnd { architecture } => {
                write!(
                    f,
                    "there is no native back end for the '{architecture}' architecture"
                )
            }
            Self::Codegen(error) => write!(f, "{error}"),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Codegen(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CodegenError> for Error {
    fn from(error: CodegenError) -> Self {
        Self::Codegen(error)
    }
}

/// Compiles the kernel `entry` of `source` for the host target.
///
/// # Errors
///
/// Returns an error when the source does not analyze, does not parse,
/// does not lower, names no such kernel, or when the machine code
/// cannot be produced.
pub fn compile_source(source: &str, entry: &str) -> Result<CompiledCode, Error> {
    let target = host_target()?;
    compile_source_for(target.as_target(), source, entry)
}

/// Compiles the kernel `entry` of `source` for `target`.
///
/// # Errors
///
/// Returns an error when the source does not analyze, does not parse,
/// does not lower, names no such kernel, or when the machine code
/// cannot be produced.
pub fn compile_source_for(
    target: &dyn NativeTarget,
    source: &str,
    entry: &str,
) -> Result<CompiledCode, Error> {
    let module = lower(source)?;
    compile_module_for(target, &module, entry)
}

/// Compiles the kernel `entry` of an already lowered `module` for
/// `target`.
///
/// Lowering once and compiling several entry points of the same module
/// is what a build script does: the source is analyzed and lowered a
/// single time, and every kernel of the file is then compiled from that
/// one module.
///
/// # Errors
///
/// Returns an error when the module names no such kernel, or when the
/// machine code cannot be produced.
pub fn compile_module_for(
    target: &dyn NativeTarget,
    module: &codevar_ocl_ir::ir::Module,
    entry: &str,
) -> Result<CompiledCode, Error> {
    let function = find_entry(module, entry)?;
    let options = CodegenOptions::default();
    let code = compile_entry(module, target, &options, function)?;
    Ok(CompiledCode {
        bytes: code.bytes().to_vec(),
        entry_offset: code.entry_offset(),
        instruction_count: code.instruction_count(),
    })
}

/// Lowers `source` to the intermediate representation.
///
/// # Errors
///
/// Returns an error when the source does not analyze, does not parse,
/// or does not lower.
pub fn lower(source: &str) -> Result<codevar_ocl_ir::ir::Module, Error> {
    match codevar_ocl_ir::lower::lower_source(source) {
        Ok(module) => Ok(module),
        Err(codevar_ocl_ir::lower::LowerError::AnalysisFailed { errors }) => {
            let diagnostics = errors
                .iter()
                .map(|diagnostic| diagnostic.message.clone())
                .collect();
            Err(Error::Analysis { diagnostics })
        }
        Err(codevar_ocl_ir::lower::LowerError::ParseFailed { count }) => Err(Error::Parse { count }),
        Err(error) => Err(Error::Lowering {
            detail: error.to_string(),
        }),
    }
}

/// The function of the entry point named `name`.
///
/// # Errors
///
/// Returns an error when the module declares no such entry point.
pub fn find_entry(module: &codevar_ocl_ir::ir::Module, name: &str) -> Result<ValueId, Error> {
    module
        .entry_points
        .iter()
        .find(|entry| entry.name == name)
        .map(|entry| entry.func)
        .ok_or_else(|| Error::UnknownEntry {
            name: String::from(name),
        })
}

/// The names of the kernels the source declares, in source order.
///
/// # Errors
///
/// Returns an error when the source does not analyze, does not parse,
/// or does not lower.
pub fn kernel_names(source: &str) -> Result<Vec<String>, Error> {
    let module = lower(source)?;
    let mut names = Vec::with_capacity(module.entry_points.len());
    for entry in &module.entry_points {
        names.push(entry.name.clone());
    }
    Ok(names)
}

/// A short description of the host target, for diagnostics.
#[must_use]
pub fn host_target_name() -> String {
    match host_target() {
        Ok(target) => target.as_target().name().to_string(),
        Err(error) => error.to_string(),
    }
}
