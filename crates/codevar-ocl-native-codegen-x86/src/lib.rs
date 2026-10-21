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

//! The x86-64 back end of the native code generator.
//!
//! The back end turns the machine IR into System V x86-64 machine
//! code: the integer registers are the sixteen general-purpose
//! registers, the floating-point and vector registers are the sixteen
//! SSE registers, and the code follows the System V calling
//! convention.
//!
//! The entry point of a compiled kernel is an
//! `unsafe extern "C" fn(*mut WorkItem)`, so the work-item pointer
//! arrives in the first argument register.
//!
//! # The register file
//!
//! The sixteen general-purpose registers are numbered the way the
//! architecture's encoding numbers them, so the number of a register
//! is also its encoding field:
//!
//! | Number | Register | Number | Register |
//! |--------|----------|--------|----------|
//! | 0      | `rax`    | 8      | `r8`     |
//! | 1      | `rcx`    | 9      | `r9`     |
//! | 2      | `rdx`    | 10     | `r10`    |
//! | 3      | `rbx`    | 11     | `r11`    |
//! | 4      | `rsp`    | 12     | `r12`    |
//! | 5      | `rbp`    | 13     | `r13`    |
//! | 6      | `rsi`    | 14     | `r14`    |
//! | 7      | `rdi`    | 15     | `r15`    |
//!
//! The sixteen SSE registers follow them, so `xmm0` is 16 and `xmm15`
//! is 31.  A register number therefore tells whether the value it
//! holds is an integer or a floating-point one.
//!
//! The allocator is offered the registers `r10`, `r11`, `rbx`, `r12`,
//! `r13`, `r14`, and `r15` as integer registers, and `xmm0` through
//! `xmm13` as floating-point and vector registers.  The argument
//! registers, the accumulator, and the two last SSE registers belong
//! to the back end: the expander uses them as scratch space, and the
//! calling convention reserves them for the arguments and the result
//! of a call.
//!
//! # The registers of the machine IR
//!
//! The machine IR names the values of an operation with virtual
//! registers while the machine code is being built.  The allocator
//! gives every value a home before the back end sees the module, so
//! the destination of an instruction and the operands of an
//! instruction name the physical registers of the register model.
//!
//! # The frame
//!
//! The prologue pushes the callee-saved registers the function uses,
//! and reserves the frame with a single subtraction of a whole number
//! of sixteen-byte units.  The frame base is the stack pointer right
//! after that subtraction, so the slot of the frame at an offset `o`
//! is at `rsp + o`; the saved registers sit at `frame_base +
//! frame_size + 8 * index`, where the index is the position of the
//! register in the list the prologue pushes.  The epilogue releases
//! the frame, restores the saved registers, and returns.
//!
//! The number of saved registers is odd, so that the stack pointer is
//! aligned when the function calls: one more register is saved when
//! the function uses an even number of them.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use codevar_ocl_native_codegen::error::Result;
use codevar_ocl_native_codegen::mir::{MirModule, PReg, SymId};
use codevar_ocl_native_codegen::target::{CodeImage, NativeTarget, RegModel};

mod encode;
mod expand;
mod layout;
mod reg;

/// The x86-64 target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct X86Target {
    /// The register file of the target.
    registers: RegModel,
}

impl X86Target {
    /// Creates the target.
    #[must_use]
    pub fn new() -> Self {
        let mut int_regs: Vec<PReg> = Vec::with_capacity(7);
        for number in [
            reg::R10,
            reg::R11,
            reg::RBX,
            reg::R12,
            reg::R13,
            reg::R14,
            reg::R15,
        ] {
            int_regs.push(PReg::new(number));
        }
        let mut float_regs: Vec<PReg> = Vec::with_capacity(14);
        for number in 0..(reg::NUM_XMM - 2) {
            float_regs.push(PReg::new(reg::XMM_BASE + number));
        }
        let mut callee_saved: Vec<PReg> = Vec::with_capacity(6);
        for number in reg::CALLEE_SAVED {
            callee_saved.push(PReg::new(number));
        }
        let mut arg_regs: Vec<PReg> = Vec::with_capacity(6);
        for number in reg::ARG_REGS {
            arg_regs.push(PReg::new(number));
        }
        let return_regs: Vec<PReg> = alloc::vec![PReg::new(reg::RAX), PReg::new(reg::XMM0)];
        // The expander uses the argument registers and the accumulator
        // as scratch space, and the calling convention keeps them out
        // of the allocator's reach.
        debug_assert!(reg::SCRATCH_GPRS.contains(&reg::RAX));
        for number in reg::ARG_REGS {
            debug_assert!(reg::SCRATCH_GPRS.contains(&number));
        }
        Self {
            registers: RegModel {
                int_regs,
                // The vectors live in the same file as the floating-point
                // values, so the model lists the same registers twice.
                float_regs: float_regs.clone(),
                vec_regs: float_regs,
                callee_saved,
                stack_pointer: PReg::new(reg::RSP),
                frame_pointer: None,
                arg_regs,
                return_regs,
                pointer_size: 8,
            },
        }
    }
}

impl Default for X86Target {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeTarget for X86Target {
    fn name(&self) -> &'static str {
        "x86-64"
    }

    fn registers(&self) -> &RegModel {
        &self.registers
    }

    fn lower_module(&self, module: &MirModule, entry: SymId) -> Result<CodeImage> {
        let entry_index = module.function_index(entry).ok_or_else(|| {
            codevar_ocl_native_codegen::error::CodegenError::Target {
                what: "lowering the module",
                detail: String::from("the entry point does not name a function of the module"),
            }
        })?;
        if module.functions.is_empty() {
            return Err(codevar_ocl_native_codegen::error::CodegenError::Target {
                what: "lowering the module",
                detail: String::from("the module holds no function"),
            });
        }
        let mut expanded = Vec::with_capacity(module.functions.len());
        for function in &module.functions {
            expanded.push(expand::expand_function(module, function)?);
        }
        layout::lay_out(&expanded, entry_index)
    }
}

/// The x86-64 target, ready to use.
#[must_use]
pub fn target() -> X86Target {
    X86Target::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_register_model_holds_the_expected_registers() {
        let target = X86Target::new();
        let registers = target.registers();
        assert_eq!(registers.int_regs.len(), 7);
        assert_eq!(registers.float_regs.len(), 14);
        assert_eq!(registers.vec_regs, registers.float_regs);
        assert_eq!(registers.arg_regs.len(), 6);
        assert_eq!(registers.return_regs.len(), 2);
        assert_eq!(registers.pointer_size, 8);
        assert_eq!(registers.stack_pointer.number(), reg::RSP);
    }

    #[test]
    fn the_argument_registers_are_never_offered_to_the_allocator() {
        let target = X86Target::new();
        let registers = target.registers();
        for arg in &registers.arg_regs {
            assert!(!registers.int_regs.contains(arg));
        }
        for arg in &registers.return_regs {
            assert!(!registers.int_regs.contains(arg));
        }
    }

    #[test]
    fn the_scratch_registers_are_never_offered_to_the_allocator() {
        let target = X86Target::new();
        let registers = target.registers();
        for scratch in reg::SCRATCH_GPRS {
            let reg = PReg::new(scratch);
            assert!(!registers.int_regs.contains(&reg));
        }
        for scratch in reg::SCRATCH_XMMS {
            let reg = PReg::new(scratch);
            assert!(!registers.float_regs.contains(&reg));
            assert!(!registers.vec_regs.contains(&reg));
        }
    }
}
