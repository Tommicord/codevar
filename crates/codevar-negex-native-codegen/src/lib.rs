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

//! Regex to SPIR-V and native machine code via the
//! [`codevar-ocl-native-codegen`] pipeline.
//!
//! This crate is the native-codegen back end of Negex: it takes the
//! target-independent IR produced by [`codevar-negex-codegen`] (via
//! [`lower_dfa`](codevar_negex_codegen::lower_dfa)), can emit that IR as
//! SPIR-V binary words following the builder pipeline of
//! [`codevar_ocl_ir::spirv`], and compiles a verifiable module to native
//! machine code for the host architecture (x86-64, aarch64, or Thumb-2)
//! through [`codevar-ocl-native-codegen`].
//!
//! The produced machine code follows the [`codevar_negex_abi`] contract:
//! a single `*mut MatchItem` argument, executed through an executable
//! page.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use codevar_logger::log_info;
use codevar_ocl_asm::AssembleError;
use codevar_ocl_asm::spirv::assemble as asm_assemble;
use codevar_ocl_ir::ir::Module;
use codevar_ocl_native_codegen::error::CodegenError as NativeCodegenError;
use codevar_ocl_native_codegen::opt::{OptConfig, Optimizations};
use codevar_ocl_native_codegen::pipeline::{CodegenOptions, compile_entry};
use codevar_ocl_native_codegen::target::NativeTarget;
use codevar_ocl_native_codegen_aarch64::AArch64Target;
use codevar_ocl_native_codegen_thumbv7m::ThumbV7mTarget;
use codevar_ocl_native_codegen_x86::X86Target;

/// The entry-point symbol name Negex compiles to.
pub const NEGEX_ENTRY: &str = "negex_match";

/// Result of compiling a regex IR module to native machine code.
#[derive(Debug, Clone)]
pub struct CompiledNative {
    /// The machine-code bytes.
    pub bytes: Vec<u8>,
    /// Offset of the entry point inside `bytes`.
    pub entry_offset: u32,
}

impl CompiledNative {
    /// The machine-code bytes.
    #[must_use]
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Offset of the entry point inside `bytes`.
    #[must_use]
    #[inline]
    pub fn entry_offset(&self) -> u32 {
        self.entry_offset
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
    #[inline]
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
pub fn host_target() -> Result<HostTarget, CodegenError> {
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
        Err(CodegenError::NoBackEnd {
            architecture: String::from(core::env!("CARGO_CFG_TARGET_ARCH")),
        })
    }
}

/// Compiles the `negex_match` function from `module` to native machine code
/// for `target`.
///
/// Unlike [`compile_module`], this does not require the function to be
/// declared as an entry point; it looks up the function by name.  This is
/// used by Negex because the match function follows the System V ABI
/// (a single `*mut MatchItem` argument), not the OpenCL kernel ABI.
///
/// # Errors
///
/// Returns an error when the function is not found, when the module does
/// not verify, when a pass cannot rewrite it, or when the target cannot
/// express it.
pub fn compile_negex_match(
    module: &Module,
    target: &dyn NativeTarget,
    options: &CodegenOptions,
) -> Result<CompiledNative, CodegenError> {
    let entry = module
        .find_function(NEGEX_ENTRY)
        .ok_or(CodegenError::NoEntryPoint)?;
    let code = compile_entry(module, target, options, entry)?;
    log_info!(
        "compiled negex_match to {} bytes, entry offset {}",
        code.bytes().len(),
        code.entry_offset()
    );
    Ok(CompiledNative {
        bytes: code.bytes().to_vec(),
        entry_offset: code.entry_offset(),
    })
}

/// The default optimization options used when the caller does not specify
/// any.
#[must_use]
#[inline]
pub fn default_options() -> CodegenOptions {
    CodegenOptions {
        opt: OptConfig {
            optimizations: Optimizations::O2,
            inline_threshold: 48,
            unroll_limit: 8,
        },
        vector: codevar_ocl_native_codegen::vecz::VectorConfig::default(),
    }
}

/// Emits SPIR-V binary words for `module` by delegating to the complete
/// SPIR-V assembler in [`codevar_ocl_asm::spirv`].
///
/// The emitted module follows the SPIR-V logical layout: capabilities,
/// extended-instruction imports, memory model, entry points, debug names,
/// annotations, types/constants/globals, and functions.
///
/// # Errors
///
/// Returns an error when the module declares no entry point, when a type
/// or value cannot be encoded, when ids run out, or when a construct is
/// not supported by the SPIR-V backend.
pub fn emit_spirv(module: &Module) -> Result<Vec<u32>, CodegenError> {
    asm_assemble(module).map_err(map_assemble_error)
}

/// Maps an `AssembleError` from the SPIR-V assembler to a `CodegenError`.
fn map_assemble_error(err: AssembleError) -> CodegenError {
    match err {
        AssembleError::Verify(_) => CodegenError::Unsupported {
            what: "SPIR-V assembly failed: IR verification failed",
        },
        AssembleError::Unsupported { what } => CodegenError::Unsupported { what },
        AssembleError::Table { .. } => CodegenError::Unsupported {
            what: "SPIR-V assembly failed: missing opcode table entry",
        },
        AssembleError::InstructionTooLarge { opcode } => CodegenError::InstructionTooLarge { opcode },
        AssembleError::StringTooLong { .. } => CodegenError::Unsupported {
            what: "SPIR-V assembly failed: string too long",
        },
        AssembleError::IdExhausted => CodegenError::IdExhausted,
    }
}

/// Why a regex could not be compiled to native code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodegenError {
    /// The module declares no entry point.
    NoEntryPoint,
    /// The architecture has no back end.
    NoBackEnd {
        /// The architecture the program is running on.
        architecture: String,
    },
    /// A SPIR-V instruction exceeds the 16-bit word count.
    InstructionTooLarge {
        /// The offending opcode.
        opcode: u16,
    },
    /// The SPIR-V id space is exhausted.
    IdExhausted,
    /// A handle is out of range.
    OutOfRange {
        /// What kind of handle is out of range.
        what: &'static str,
    },
    /// A construct cannot be encoded.
    Unsupported {
        /// What construct was rejected.
        what: &'static str,
    },
    /// The native code generator failed.
    Native(NativeCodegenError),
}

impl From<NativeCodegenError> for CodegenError {
    fn from(error: NativeCodegenError) -> Self {
        Self::Native(error)
    }
}

impl core::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoEntryPoint => f.write_str("the module declares no entry point"),
            Self::NoBackEnd { architecture } => write!(
                f,
                "there is no native back end for the '{architecture}' architecture"
            ),
            Self::InstructionTooLarge { opcode } => {
                write!(f, "SPIR-V instruction {opcode} is too large")
            }
            Self::IdExhausted => f.write_str("SPIR-V id space exhausted"),
            Self::OutOfRange { what } => write!(f, "{what} out of range"),
            Self::Unsupported { what } => write!(f, "{what} is not supported"),
            Self::Native(error) => write!(f, "{error}"),
        }
    }
}

impl core::error::Error for CodegenError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Native(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use codevar_negex_codegen::{Determinized, DfaState, lower_dfa};
    use codevar_ocl_ir::ir::{ExecutionModel, Inst, Linkage, Op, Target};
    use codevar_ocl_native_codegen::opt::Optimization;

    #[test]
    fn entry_constant_is_non_empty() {
        assert!(!NEGEX_ENTRY.is_empty());
    }

    #[test]
    fn host_target_selection_works_on_current_arch() {
        let target = host_target();
        assert!(
            target.is_ok(),
            "host target selection should succeed: {}",
            target
                .as_ref()
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default()
        );
        let target = target.expect("host target succeeds on this arch");
        assert!(!target.as_target().name().is_empty());
    }

    #[test]
    fn compile_module_produces_non_empty_bytes() {
        let mut module = Module::new(Target::opencl());
        let void = module.void_ty();
        let sig = module.fn_ty(void, Vec::new());
        let func = module
            .add_function(NEGEX_ENTRY, sig, Linkage::External)
            .expect("add function");
        module.begin_body(func).expect("begin body");
        let entry = module
            .push_block(func, "entry")
            .expect("push block");
        module
            .emit(func, entry, Inst::none(Op::Return))
            .expect("emit return");
        module
            .set_entry_point(ExecutionModel::Kernel, func, NEGEX_ENTRY)
            .expect("set entry");

        let target = host_target().expect("host target");
        let options = default_options();
        let compiled = compile_negex_match(&module, target.as_target(), &options);
        assert!(compiled.is_ok(), "compile should succeed");
        let compiled = compiled.expect("compile succeeded");
        assert!(!compiled.bytes.is_empty(), "bytes must be non-empty");
        assert!(
            compiled.entry_offset < compiled.bytes.len() as u32,
            "entry offset within bytes"
        );
    }

    #[test]
    fn emit_spirv_produces_words() {
        let dfa = simple_dfa();
        let module = lower_dfa(&dfa).expect("lowers");
        let words = emit_spirv(&module);
        assert!(
            words.is_ok(),
            "emit_spirv should succeed: {:?}",
            words.as_ref().err()
        );
        let words = words.expect("emit_spirv succeeded");
        assert!(!words.is_empty(), "SPIR-V words must be non-empty");
        // SPIR-V magic number (0x07230203 = "Spv\0")
        assert_eq!(words[0], 0x0723_0203);
    }

    #[test]
    fn codegen_error_displays() {
        let error = CodegenError::NoEntryPoint;
        assert!(!error.to_string().is_empty());
        let error = CodegenError::NoBackEnd {
            architecture: String::from("wasm32"),
        };
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn default_options_has_optimizations() {
        let options = default_options();
        assert!(
            options
                .opt
                .optimizations
                .has(Optimization::Mem2Reg),
            "default options enable Mem2Reg"
        );
        assert!(
            options
                .opt
                .optimizations
                .has(Optimization::DeadCode),
            "default options enable DeadCode"
        );
    }

    /// A tiny DFA: state 0 (start, non-accepting) transitions on class 0
    /// to state 1; state 1 is accepting with no transitions. This matches
    /// a single character of class 0 at the start of the input.
    fn simple_dfa() -> Determinized {
        Determinized {
            states: vec![
                DfaState {
                    accept: false,
                    transitions: vec![Some(1)],
                    repeat: None,
                },
                DfaState {
                    accept: true,
                    transitions: vec![],
                    repeat: None,
                },
            ],
            start: 0,
            matches_empty: false,
        }
    }
}
