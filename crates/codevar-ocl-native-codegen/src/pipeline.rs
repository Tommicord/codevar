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

//! The pipeline: from a verified module to machine code.
//!
//! [`compile_module`] runs the whole chain — verification,
//! optimization, vectorization, selection, register allocation, and
//! code emission — and hands back the bytes of the compiled kernel
//! together with the offset of its entry point.
//!
//! # The kernel ABI
//!
//! The compiled code is an `unsafe extern "C" fn(*mut WorkItem)`: the
//! entry point takes the per-work-item state and reads the kernel's
//! arguments from the block that state points at.  The code is
//! position-independent, so it can be copied into executable memory
//! and entered there.

use alloc::vec::Vec;

use codevar_ocl_ir::ir::{Module, ValueId};
use codevar_ocl_ir::verify::verify;

use crate::error::Result;
use crate::opt::{OptConfig, Optimization};
use crate::target::NativeTarget;
use crate::vecz::VectorConfig;

/// How the pipeline compiles a module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CodegenOptions {
    /// The optimizer's settings, including which optimizations run and
    /// whether the vectorizer does.
    pub opt: OptConfig,
    /// The vectorizer's settings.
    pub vector: VectorConfig,
}

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

/// Compiles `module` for `target`.
///
/// The entry function is the module's first entry point; a module
/// without one cannot be compiled.
///
/// # Errors
///
/// Returns an error when the module does not verify, when a pass
/// cannot rewrite it, or when the target cannot express it.
pub fn compile_module(
    module: &Module,
    target: &dyn NativeTarget,
    options: &CodegenOptions,
) -> Result<CompiledCode> {
    let entry = module
        .entry_points
        .first()
        .ok_or_else(|| crate::error::CodegenError::Selection {
            what: "choosing the entry point",
            detail: alloc::string::String::from("the module declares no entry point"),
        })?;
    compile_entry(module, target, options, entry.func)
}

/// Compiles the function `entry` of `module` for `target`.
///
/// # Errors
///
/// Returns an error when the module does not verify, when a pass
/// cannot rewrite it, or when the target cannot express it.
pub fn compile_entry(
    module: &Module,
    target: &dyn NativeTarget,
    options: &CodegenOptions,
    entry: ValueId,
) -> Result<CompiledCode> {
    verify(module)?;
    let mut module = module.clone();
    let optimizations = options.opt.optimizations;
    if !optimizations.is_empty() {
        crate::opt::optimize(&mut module, &options.opt)?;
        verify(&module)?;
    }
    if optimizations.has(Optimization::Vectorizable) {
        crate::vecz::vectorize(&mut module, &options.vector)?;
        verify(&module)?;
    }
    let selection = crate::isel::select_module(&module, entry)?;
    let mut machine = selection.module;
    crate::regalloc::allocate(&mut machine, target.registers())?;
    let image = target.lower_module(&machine, selection.entry)?;
    Ok(CompiledCode {
        bytes: image.bytes,
        entry_offset: image.entry_offset,
        instruction_count: image.instruction_count,
    })
}
