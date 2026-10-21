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

use codevar_ocl_native_codegen::mir::{
    MInst, MOp, MirBlock, MirFunction, MirModule, MType, Operand, Operands, SymbolKind, SymId,
    VReg,
};

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
        let mut module = MirModule::new();
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

    /// Finishes the module and returns it with the symbol of the
    /// entry function.
    pub fn finish(mut self) -> (MirModule, SymId) {
        let symbol = self
            .module
            .add_symbol(&self.function.name.clone(), SymbolKind::Function { index: 0 }, true);
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
