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

//! x86-64 native codegen back end for the Negex regex engine.
//!
//! Implements [`codevar_ocl_native_codegen::target::NativeTarget`] for
//! x86-64: register model, calling convention, and machine-code lowering
//! for Negex-compiled IR modules.
//!
//! # The register model
//!
//! The integer registers are the sixteen general-purpose registers,
//! numbered the way the architecture's encoding numbers them: `rax` is
//! 0, `rcx` is 1, and so on up to `r15` at 15.  The callee-saved
//! registers are `rbx`, `rbp`, and `r12` through `r15`.  The stack
//! pointer is `rsp` and the argument registers follow the System V
//! ABI: `rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`.  The return register
//! is `rax`.  A pointer is eight bytes.
//!
//! # The lowering
//!
//! `lower_module` expands every machine operation of every function
//! into x86-64 instructions, lays the functions out at sixteen-byte
//! boundaries, resolves the relative displacements of the branches and
//! calls, and returns the position-independent bytes of the module.
//! The code follows the Negex ABI: the entry point takes a single
//! `*mut MatchItem` argument in `rdi` and returns its result in `rax`.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use codevar_logger::log_info;
use codevar_ocl_native_codegen::error::{CodegenError, Result};
use codevar_ocl_native_codegen::mir::{MirModule, PReg, SymId};
use codevar_ocl_native_codegen::target::{CodeImage, NativeTarget, RegModel, TargetCode};

mod encode;
mod expand;
mod layout;
mod reg;

/// The x86-64 target back end.
pub struct X86Target {
    /// The register file and calling convention of the target.
    registers: RegModel,
}

impl X86Target {
    /// Creates the x86-64 target.
    #[must_use]
    pub fn new() -> Self {
        let mut int_regs: Vec<PReg> = Vec::with_capacity(8);
        for number in [
            reg::R10,
            reg::R11,
            reg::RBX,
            reg::RBP,
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
        let return_regs: Vec<PReg> = alloc::vec![PReg::new(reg::RAX)];
        Self {
            registers: RegModel {
                int_regs,
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

    /// The register file and calling convention of the target.
    #[must_use]
    #[inline]
    pub fn registers(&self) -> &RegModel {
        &self.registers
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
        let entry_index = module
            .function_index(entry)
            .ok_or_else(|| CodegenError::Target {
                what: "lowering the module",
                detail: String::from("the entry point does not name a function of the module"),
            })?;
        if module.functions.is_empty() {
            return Err(CodegenError::Target {
                what: "lowering the module",
                detail: String::from("the module holds no function"),
            });
        }
        let mut expanded: Vec<expand::Expanded> = Vec::with_capacity(module.functions.len());
        for function in &module.functions {
            expanded.push(expand::expand_function(module, function, TargetCode::Baseline)?);
        }
        let image = layout::lay_out(&expanded, entry_index)?;
        log_info!(
            "lowered module to {} bytes, entry offset {}",
            image.bytes.len(),
            image.entry_offset
        );
        Ok(image)
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

    use codevar_ocl_native_codegen::mir::{
        MInst, MOp, MType, MirBlock, MirFunction, Operands, RegClass, VReg,
    };
    use codevar_ocl_native_codegen::target::NativeTarget;

    /// A tiny module: one function that returns its first argument.
    fn return_argument_module() -> (MirModule, SymId) {
        let mut module = MirModule::new();
        let entry = module.add_symbol(
            "negex_match",
            codevar_ocl_native_codegen::mir::SymbolKind::Function { index: 0 },
            true,
        );
        let mut function = MirFunction::new("negex_match");
        function.symbol = entry;
        function.is_entry = true;
        function.params.push(VReg::new(0));
        function.vreg_types.push(MType::I64);
        let mut block = MirBlock::new("entry");
        block.insts.push(MInst::def_physical(
            PReg::new(reg::RAX),
            MType::I64,
            MOp::Param { index: 0 },
            Operands::none(),
        ));
        block
            .insts
            .push(MInst::none(MType::Void, MOp::RetVoid, Operands::none()));
        function.push_block(block);
        module.functions.push(function);
        (module, entry)
    }

    #[test]
    fn target_name() {
        assert_eq!(X86Target::new().name(), "x86-64");
    }

    #[test]
    fn register_model_contents() {
        let target = X86Target::new();
        let registers = target.registers();
        assert_eq!(registers.pointer_size, 8);
        assert_eq!(registers.stack_pointer.number(), reg::RSP);
        assert_eq!(registers.arg_regs.len(), 6);
        assert_eq!(registers.return_regs.len(), 1);
        assert_eq!(registers.return_regs[0].number(), reg::RAX);
        assert_eq!(
            registers
                .arg_regs
                .iter()
                .map(|r| r.number())
                .collect::<Vec<_>>(),
            vec![reg::RDI, reg::RSI, reg::RDX, reg::RCX, reg::R8, reg::R9]
        );
        for callee in reg::CALLEE_SAVED {
            assert!(
                registers
                    .callee_saved
                    .iter()
                    .any(|r| r.number() == callee)
            );
        }
    }

    #[test]
    fn lower_module_produces_valid_bytes() {
        let target = X86Target::new();
        let (module, entry) = return_argument_module();
        let image = target
            .lower_module(&module, entry)
            .expect("lowering a trivial module succeeds");
        assert!(!image.bytes.is_empty(), "the image holds code");
        assert!(
            image.entry_offset < image.bytes.len() as u32,
            "the entry offset lies inside the image"
        );
        assert_eq!(
            image.instruction_count, 4,
            "parameter move, push/pop for alignment, and return"
        );
    }

    #[test]
    fn lower_module_rejects_unknown_entry() {
        let target = X86Target::new();
        let (module, _entry) = return_argument_module();
        let unknown = SymId::new(999);
        let result = target.lower_module(&module, unknown);
        assert!(result.is_err(), "an unknown entry point is rejected");
    }

    #[test]
    fn lower_module_rejects_empty_module() {
        let target = X86Target::new();
        let module = MirModule::new();
        let entry = SymId::new(0);
        let result = target.lower_module(&module, entry);
        assert!(result.is_err(), "an empty module is rejected");
    }

    mod reg_class {
        use super::*;

        #[test]
        fn integer_types_map_to_integer_registers() {
            assert_eq!(MType::I32.reg_class(), RegClass::Integer);
            assert_eq!(MType::I64.reg_class(), RegClass::Integer);
            assert!(MType::Ptr { size: 8 }.reg_class() == RegClass::Integer);
        }

        #[test]
        fn floating_types_map_to_float_registers() {
            assert_eq!(MType::F32.reg_class(), RegClass::Float);
            assert_eq!(MType::F64.reg_class(), RegClass::Float);
        }
    }
}
