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

//! The pieces of a machine module the tests build.

// Every test binary shares this module, and every binary uses a
// different part of it.
#![allow(dead_code)]

use codevar_ocl_native_codegen::error::CodegenError;
use codevar_ocl_native_codegen::mir::{
    MInst, MOp, MType, MirBlock, MirFunction, MirModule, Operand, Operands, SymId, SymbolKind, VReg,
};
use codevar_ocl_native_codegen::target::{CodeImage, NativeTarget};

use codevar_ocl_native_codegen_thumbv7m::{ThumbV7mTarget, register_model};

/// Allocates the registers of a module.
///
/// The machine code that reaches the back end has been through the
/// allocator, so the tests allocate their modules the way the
/// pipeline does before they hand them over.
pub fn allocate(module: &mut MirModule) {
    let model = register_model();
    if let Err(error) = codevar_ocl_native_codegen::regalloc::allocate(module, &model) {
        panic!("the module allocates: {error}");
    }
}

/// The code of a module.
///
/// The back end refuses a module it cannot express, and a test that
/// expects the code of a module does not expect a refusal.
pub fn lower(target: &ThumbV7mTarget, module: &MirModule, entry: SymId) -> CodeImage {
    match target.lower_module(module, entry) {
        Ok(image) => image,
        Err(error) => panic!("the module compiles: {error}"),
    }
}

/// The error the back end returns for a module it refuses.
pub fn refused(target: &ThumbV7mTarget, module: &MirModule, entry: SymId) -> CodegenError {
    match target.lower_module(module, entry) {
        Ok(_) => panic!("the module is refused"),
        Err(error) => error,
    }
}

/// A module under construction.
pub struct Builder {
    /// The module itself.
    pub module: MirModule,
    /// The function under construction.
    pub function: MirFunction,
}

impl Builder {
    /// Creates a builder of a module with one function.
    #[must_use]
    pub fn new(name: &str) -> Self {
        let module = MirModule::new();
        let function = MirFunction::new(name);
        Self { module, function }
    }

    /// Creates a virtual register of the given type.
    pub fn vreg(&mut self, ty: MType) -> VReg {
        self.function.new_vreg(ty)
    }

    /// Adds an instruction to the block under construction.
    pub fn push(&mut self, inst: MInst) {
        self.current_mut().insts.push(inst);
    }

    /// The block under construction.
    pub fn current_mut(&mut self) -> &mut MirBlock {
        if self.function.blocks.is_empty() {
            self.function.push_block(MirBlock::new("entry"));
        }
        self.function
            .blocks
            .last_mut()
            .unwrap_or_else(|| unreachable_block())
    }

    /// Adds a block and makes it the one under construction.
    pub fn block(&mut self, label: &str) {
        self.function.push_block(MirBlock::new(label));
    }

    /// Finishes the module, allocates its registers, and returns it
    /// with the symbol of the entry function.
    pub fn finish(self) -> (MirModule, SymId) {
        let (module, symbol) = self.finish_unallocated();
        let mut module = module;
        allocate(&mut module);
        (module, symbol)
    }
    /// Finishes the module without allocating its registers.
    ///
    /// The module is the one the back end refuses, because it still
    /// names virtual registers.
    pub fn finish_unallocated(mut self) -> (MirModule, SymId) {
        let symbol = self.module.add_symbol(
            &self.function.name.clone(),
            SymbolKind::Function { index: 0 },
            true,
        );
        self.function.symbol = symbol;
        self.function.is_entry = true;
        self.module.functions.push(self.function);
        (self.module, symbol)
    }
}

/// The block of a builder that has none.
fn unreachable_block() -> &'static mut MirBlock {
    unreachable!("a builder always has a block")
}

/// The operand of a virtual register.
#[must_use]
pub fn vreg_operand(reg: VReg) -> Operand {
    Operand::Virtual(reg)
}

/// The operand of an immediate.
#[must_use]
pub fn imm(value: i64) -> Operand {
    Operand::Immediate(value)
}

/// The operand list of two operands.
#[must_use]
pub fn two(first: Operand, second: Operand) -> Operands {
    Operands::two(first, second)
}

/// The operand list of one operand.
#[must_use]
pub fn one(first: Operand) -> Operands {
    Operands::one(first)
}

/// The operand list of three operands.
#[must_use]
pub fn three(first: Operand, second: Operand, third: Operand) -> Operands {
    Operands::three(first, second, third)
}

/// The instruction that defines a value.
#[must_use]
pub fn def(dest: VReg, ty: MType, op: MOp, args: Operands) -> MInst {
    MInst::def(dest, ty, op, args)
}

/// The instruction that has no result.
#[must_use]
pub fn none(ty: MType, op: MOp, args: Operands) -> MInst {
    MInst::none(ty, op, args)
}
