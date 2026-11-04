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

//! The ARMv7-M back end of the native code generator.
//!
//! The back end turns the machine IR into Thumb-2 machine code for
//! the Cortex-M family.  The integer registers are the thirteen
//! general-purpose registers the architecture has, the
//! floating-point registers are the single-precision registers of the
//! floating-point unit, and the code follows the procedure call
//! standard of the architecture.
//!
//! # The entry point
//!
//! The entry point of a compiled kernel is an
//! `unsafe extern "C" fn(*mut WorkItem)`, so the work-item pointer
//! arrives in the first argument register.
//!
//! # The Thumb state
//!
//! The code of the module is Thumb code, so it has to be entered in
//! the Thumb state.  A caller that jumps into the image has to set
//! the low bit of the address it jumps to; the address of the entry
//! point is the offset of the entry function inside the image.
//!
//! # The calling convention
//!
//! A function receives its arguments in the first four argument
//! registers and returns its result in the first one.  The registers
//! that survive a call are saved by the prologue of the function and
//! restored by its epilogue.  The frame of a function is reserved
//! with one subtraction of the stack pointer, so every slot of the
//! frame is at a known distance from the stack pointer, and the saved
//! registers sit above the frame.
//!
//! # The registers of a function
//!
//! The machine code that reaches the back end has already been
//! allocated: every operand and every destination names a physical
//! register, and the back end reads those registers as they are.  A
//! machine code that still names a virtual register never went
//! through the allocator, and the back end refuses it.
//!
//! The four argument registers are the scratch of the back end: the
//! allocator never hands them out, so the back end can use them for
//! the values it computes on the way without disturbing a live value.
//!
//! # Position independence
//!
//! The code is position-independent: no absolute address appears in
//! it.  A call to another function of the module is a relative
//! branch, and the back end refuses a call that reaches too far to
//! be encoded.
//!
//! # The floating-point unit
//!
//! The architecture has no mandatory floating-point hardware, so the
//! choice is explicit: [`ThumbV7mTarget::new`] describes a core with
//! the unit and [`ThumbV7mTarget::without_fp`] a core without it.
//! The registers of the unit are part of the register model of the
//! first and missing from the second.  The back end refuses the
//! floating-point operations of the machine IR either way; the
//! difference is the reason it gives.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use codevar_ocl_native_codegen::error::{CodegenError, Result};
use codevar_ocl_native_codegen::mir::{MirModule, SymId, SymbolKind};
use codevar_ocl_native_codegen::target::{CodeImage, NativeTarget, RegModel};

pub mod encode;
mod expand;
mod layout;
pub mod reg;

/// The ARMv7-M target.
///
/// The target describes the register file of the architecture and
/// turns a machine module into machine code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThumbV7mTarget {
    /// The register file of the target.
    registers: RegModel,
    /// Whether the core has the floating-point unit.
    has_fp: bool,
}

impl ThumbV7mTarget {
    /// Creates the target of a core with the floating-point unit.
    ///
    /// The registers of the unit are part of the register model, so
    /// the allocator can place the floating-point values of a module
    /// in them.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registers: register_model(),
            has_fp: true,
        }
    }

    /// Creates the target of a core without the floating-point unit.
    ///
    /// The floating-point registers are not part of the register
    /// model, and the floating-point operations of the machine IR are
    /// refused.
    #[must_use]
    pub fn without_fp() -> Self {
        Self {
            registers: register_model_with_fp(false),
            has_fp: false,
        }
    }

    /// Whether the core has the floating-point unit.
    #[must_use]
    pub fn has_fp(&self) -> bool {
        self.has_fp
    }
}

impl Default for ThumbV7mTarget {
    fn default() -> Self {
        Self::new()
    }
}

/// The register model of the architecture.
///
/// The integer registers are the thirteen general-purpose registers
/// the allocator may use; the argument registers belong to the
/// calling convention and are never handed out.  The floating-point
/// registers are the single-precision registers of the unit, and the
/// quadword registers are the vector registers of the file.
///
/// # Example
///
/// ```
/// use codevar_ocl_native_codegen::target::NativeTarget;
/// use codevar_ocl_native_codegen_thumbv7m::register_model;
///
/// let registers = register_model();
/// assert_eq!(registers.pointer_size, 4);
/// assert_eq!(registers.arg_regs.len(), 4);
/// assert_eq!(registers.return_regs.len(), 1);
/// ```
#[must_use]
pub fn register_model() -> RegModel {
    register_model_with_fp(true)
}

/// The register model of the architecture, with or without the
/// floating-point unit.
///
/// The registers of the unit are part of the model of a core that has
/// it, and missing from the model of a core that has not.
fn register_model_with_fp(with_fp: bool) -> RegModel {
    let mut int_regs = Vec::with_capacity(13);
    for number in 0..13u8 {
        int_regs.push(reg::gpr(number));
    }
    let mut float_regs = Vec::new();
    let mut vec_regs = Vec::new();
    if with_fp {
        for number in 0..32u8 {
            float_regs.push(reg::sreg(number));
        }
        for number in 0..8u8 {
            vec_regs.push(reg::qreg(number));
        }
    }
    let mut callee_saved = Vec::with_capacity(8);
    for number in 4..12u8 {
        callee_saved.push(reg::gpr(number));
    }
    let mut arg_regs = Vec::with_capacity(4);
    for number in 0..4u8 {
        arg_regs.push(reg::gpr(number));
    }
    RegModel {
        int_regs,
        float_regs,
        vec_regs,
        callee_saved,
        stack_pointer: reg::gpr(reg::SP),
        frame_pointer: None,
        arg_regs,
        return_regs: alloc::vec![reg::gpr(0)],
        pointer_size: 4,
    }
}

impl NativeTarget for ThumbV7mTarget {
    fn name(&self) -> &'static str {
        "thumbv7m"
    }

    fn registers(&self) -> &RegModel {
        &self.registers
    }

    fn lower_module(&self, module: &MirModule, entry: SymId) -> Result<CodeImage> {
        let entry_index = module
            .function_index(entry)
            .ok_or_else(|| CodegenError::Target {
                what: "lowering the module",
                detail: String::from("the entry symbol is not a function of the module"),
            })?;

        // The symbol of every function, so that a call can find the
        // function it names.
        let mut callees: Vec<Option<usize>> = vec![None; module.symbols.len()];
        for (index, symbol) in module.symbols.iter().enumerate() {
            if let SymbolKind::Function { index: function } = symbol.kind {
                callees[index] = Some(function);
            }
        }

        let mut functions = Vec::with_capacity(module.functions.len());
        for function in &module.functions {
            functions.push(expand::expand_function(function, &callees, self.has_fp)?);
        }

        let placed = layout::layout(&functions, entry_index)?;
        Ok(CodeImage {
            bytes: placed.image,
            entry_offset: placed.entry_offset,
            instruction_count: placed.instructions,
        })
    }
}

/// The ARMv7-M target, ready to use.
#[must_use]
pub fn target() -> ThumbV7mTarget {
    ThumbV7mTarget::new()
}
