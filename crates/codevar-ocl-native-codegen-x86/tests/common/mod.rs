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

//! The helpers the tests of the back end share.
//!
//! The machine IR names the values of an operation with virtual
//! registers, and after the register allocation those names are the
//! numbers of the physical registers that hold the values.  The
//! helpers here build a module that follows the convention: the
//! number of a virtual register is the number of the register in the
//! target's file.

#![allow(dead_code)]

use codevar_ocl_native_codegen::mir::{
    MInst, MOp, MType, MirBlock, MirFunction, MirModule, Operand, Operands, PReg, SymId,
};

/// The number of the accumulator.
pub const RAX: u16 = 0;
/// The number of the counter.
pub const RCX: u16 = 1;
/// The number of the data register.
pub const RDX: u16 = 2;
/// The number of the base register.
pub const RBX: u16 = 3;
/// The number of the stack pointer.
pub const RSP: u16 = 4;
/// The number of the frame pointer.
pub const RBP: u16 = 5;
/// The number of the first source index.
pub const RSI: u16 = 6;
/// The number of the first destination index.
pub const RDI: u16 = 7;
/// The number of the third extended register.
pub const R10: u16 = 10;
/// The number of the fourth extended register.
pub const R11: u16 = 11;
/// The number of the fifth extended register.
pub const R12: u16 = 12;
/// The number of the sixth extended register.
pub const R13: u16 = 13;
/// The number of the seventh extended register.
pub const R14: u16 = 14;
/// The number of the eighth extended register.
pub const R15: u16 = 15;
/// The number of the first SSE register.
pub const XMM0: u16 = 16;
/// The number of the second SSE register.
pub const XMM1: u16 = 17;
/// The number of the third SSE register.
pub const XMM3: u16 = 19;
/// The number of the third SSE register.
pub const XMM2: u16 = 18;
/// The number of the fifth SSE register.
pub const XMM4: u16 = 20;
/// The number of the sixth SSE register.
pub const XMM5: u16 = 21;

/// The register a value lives in, and the type of the value.
///
/// The register allocator has already given every value a home, so
/// the tests name the registers directly.  The helper remembers the
/// type of the value in the function's type table, as the allocator
/// does.
pub fn vreg(function: &mut MirFunction, register: u16, ty: MType) -> u16 {
    let index = register as usize;
    while function.vreg_types.len() <= index {
        function.vreg_types.push(MType::Void);
    }
    function.vreg_types[index] = ty;
    register
}

/// The operand that names a physical register.
pub fn reg(register: u16) -> Operand {
    Operand::Physical(PReg::new(register))
}

/// The operand that carries an immediate.
pub fn imm(value: i64) -> Operand {
    Operand::Immediate(value)
}

/// Builds an instruction that writes a register.
///
/// The type of the values the instruction reads is the type of the
/// value it writes, which is what the selection stage records for
/// most of the operations.
pub fn def(dest: u16, ty: MType, op: MOp, args: Operands) -> MInst {
    MInst::def_physical(PReg::new(dest), ty.clone(), op, args).with_operand_type(ty)
}

/// Builds an instruction that writes a register and names the type of
/// the values it reads.
pub fn def_with_operands(dest: u16, ty: MType, operand_ty: MType, op: MOp, args: Operands) -> MInst {
    MInst::def_physical(PReg::new(dest), ty, op, args).with_operand_type(operand_ty)
}

/// Builds an instruction that writes no register.
pub fn none(ty: MType, op: MOp, args: Operands) -> MInst {
    MInst::none(ty.clone(), op, args).with_operand_type(ty)
}

/// Builds an instruction that writes no register and names the type
/// of the values it reads.
pub fn none_with_operands(ty: MType, operand_ty: MType, op: MOp, args: Operands) -> MInst {
    MInst::none(ty, op, args).with_operand_type(operand_ty)
}

/// Creates a module with one function and returns it with the symbol
/// of the function.
pub fn module_with(function: MirFunction) -> (MirModule, SymId) {
    let mut module = MirModule::new();
    let symbol = module.add_symbol(
        &function.name,
        codevar_ocl_native_codegen::mir::SymbolKind::Function { index: 0 },
        false,
    );
    module.functions.push(function);
    (module, symbol)
}

/// Creates an empty block.
pub fn block(label: &str) -> MirBlock {
    MirBlock::new(label)
}
