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

//! The AArch64 back end of the native code generator.
//!
//! The back end turns the machine code into AArch64 machine code.
//! The integer registers are the thirty-one general-purpose
//! registers, the floating-point and the vector registers are the
//! thirty-two SIMD-and-floating-point registers, and the code
//! follows the procedure call standard of the architecture.
//!
//! The entry point of a compiled kernel is an
//! `unsafe extern "C" fn(*mut WorkItem)`, so the work-item pointer
//! arrives in the first argument register.
//!
//! # The register file
//!
//! | Numbers | Registers | Notes |
//! |---|---|---|
//! | `0` to `30` | `x0` to `x30` | `x30` is the link register |
//! | `31` | the stack pointer | also the zero register |
//! | `32` to `63` | `v0` to `v31` | the floating-point and the vector registers |
//!
//! The argument registers are `x0` to `x7` and the result of a
//! call comes back in `x0`; the floating-point arguments travel in
//! `v0` to `v7`.  None of them is ever given to a value: they
//! belong to the calling convention.  The registers that survive a
//! call are `x19` to `x28`, and the prologue of a function saves
//! the ones it uses.
//!
//! # The pools
//!
//! The back end publishes the pools the allocator draws its
//! registers from: `x9` to `x15` and `x19` to `x29` for the
//! integers, and `v8` to `v31` for the floating-point and the
//! vector values alike, because the architecture keeps them in one
//! file.  The argument registers are left out of the pools, and the
//! registers that survive a call are left in: the prologue saves
//! the ones a function uses.
//!
//! # The registers of a function
//!
//! The machine code that reaches the back end has already been
//! allocated: every operand and every destination names a physical
//! register, and the back end reads those registers as they are.  A
//! machine code that still names a virtual register never went
//! through the allocator, and the back end refuses it.
//!
//! # The frame
//!
//! The prologue of a function saves the callee-saved registers it
//! uses, then reserves the frame of the function with a single
//! subtraction of the stack pointer.  The frame base is the value of
//! the stack pointer right after that subtraction, and the saved
//! registers sit above the frame.  The epilogue restores them,
//! releases the frame and returns.  The slots the allocator spilled
//! into live in the same frame, and the operations that read and
//! write them address them from the frame base.
//!
//! # The code
//!
//! The code is position-independent: no address appears in it.  The
//! branches inside a function and the calls between the functions of
//! a module are relative, and every other instruction reaches its
//! data through a register.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod encode;
pub mod expand;
pub mod layout;
pub mod reg;

use alloc::vec;
use alloc::vec::Vec;

use codevar_ocl_native_codegen::error::Result;
use codevar_ocl_native_codegen::mir::{MirModule, PReg, SymId};
use codevar_ocl_native_codegen::target::{CodeImage, NativeTarget, RegModel};

/// The AArch64 target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AArch64Target {
    /// The register file of the target.
    registers: RegModel,
}

impl AArch64Target {
    /// Creates the target.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registers: register_model(),
        }
    }
}

impl Default for AArch64Target {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeTarget for AArch64Target {
    fn name(&self) -> &'static str {
        "aarch64"
    }

    fn registers(&self) -> &RegModel {
        &self.registers
    }

    fn lower_module(&self, module: &MirModule, entry: SymId) -> Result<CodeImage> {
        layout::lower(module, entry)
    }
}

/// The AArch64 target, ready to use.
#[must_use]
pub fn target() -> AArch64Target {
    AArch64Target::new()
}

/// The register file and the calling convention of the target.
///
/// The general-purpose registers that the allocator may use are
/// `x9` to `x15` and `x19` to `x29`: the argument registers and
/// the link register belong to the calling convention, and the two
/// registers the back end keeps for itself are never given to a
/// value.  The SIMD-and-floating-point registers that the allocator
/// may use are `v8` to `v31`; the first eight of them carry the
/// floating-point arguments of a call.
///
/// # Example
///
/// ```
/// use codevar_ocl_native_codegen::target::NativeTarget;
/// use codevar_ocl_native_codegen_aarch64::target;
///
/// let target = target();
/// let registers = target.registers();
/// assert_eq!(registers.pointer_size, 8);
/// assert_eq!(registers.arg_regs.len(), 8);
/// assert_eq!(registers.return_regs.len(), 1);
/// ```
#[must_use]
pub fn register_model() -> RegModel {
    let mut int_regs: Vec<PReg> = Vec::new();
    for index in 9u8..=15 {
        int_regs.push(reg::gpr(index));
    }
    for index in 19u8..=29 {
        int_regs.push(reg::gpr(index));
    }

    let mut float_regs: Vec<PReg> = Vec::new();
    let mut vec_regs: Vec<PReg> = Vec::new();
    for index in 8u8..=31 {
        float_regs.push(reg::simd(index));
        vec_regs.push(reg::simd(index));
    }

    let mut callee_saved: Vec<PReg> = Vec::new();
    for index in 19u8..=28 {
        callee_saved.push(reg::gpr(index));
    }

    let mut arg_regs: Vec<PReg> = Vec::new();
    for index in 0u8..=7 {
        arg_regs.push(reg::gpr(index));
    }

    RegModel {
        int_regs,
        float_regs,
        vec_regs,
        callee_saved,
        stack_pointer: reg::gpr(reg::SP),
        frame_pointer: None,
        arg_regs,
        return_regs: vec![reg::gpr(0)],
        pointer_size: 8,
    }
}
