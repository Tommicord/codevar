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

//! A scratch test that generates a module with many operations.
//!
//! The test writes the image of the module and a list of the
//! functions it holds, so that a harness can call them one by one.

mod common;

use codevar_ocl_native_codegen::mir::{
    MInst, MOp, MirBlock, MirFunction, MirModule, MType, Operand, Operands, SymbolKind, SymId,
    VReg,
};
use codevar_ocl_native_codegen::target::NativeTarget;

use codevar_ocl_native_codegen_thumbv7m::ThumbV7mTarget;

/// The operations the module holds.
const OPERATIONS: &[(&str, MOp)] = &[
    ("add", MOp::Add),
    ("sub", MOp::Sub),
    ("mul", MOp::Mul),
    ("udiv", MOp::UDiv),
    ("urem", MOp::URem),
    ("sdiv", MOp::SDiv),
    ("srem", MOp::SRem),
    ("and", MOp::And),
    ("or", MOp::Or),
    ("xor", MOp::Xor),
    ("shl", MOp::Shl),
    ("lshr", MOp::LShr),
    ("ashr", MOp::AShr),
];

/// Builds a function that takes two parameters and applies the
/// operation to them.
fn binary(name: &str, op: MOp) -> MirFunction {
    let mut function = MirFunction::new(name);
    let left = function.new_vreg(MType::I32);
    let right = function.new_vreg(MType::I32);
    let result = function.new_vreg(MType::I32);
    let mut block = MirBlock::new("entry");
    block.insts.push(MInst::def(
        left,
        MType::I32,
        MOp::Param { index: 0 },
        Operands::one(Operand::Immediate(0)),
    ));
    block.insts.push(MInst::def(
        right,
        MType::I32,
        MOp::Param { index: 1 },
        Operands::one(Operand::Immediate(1)),
    ));
    block.insts.push(MInst::def(
        result,
        MType::I32,
        op,
        Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
    ));
    block.insts.push(MInst::none(
        MType::Void,
        MOp::Ret,
        Operands::one(Operand::Virtual(result)),
    ));
    function.push_block(block);
    function
}

/// Builds a function that compares its two parameters.
fn compare() -> MirFunction {
    let mut function = MirFunction::new("cmp");
    let left = function.new_vreg(MType::I32);
    let right = function.new_vreg(MType::I32);
    let result = function.new_vreg(MType::I32);
    let mut block = MirBlock::new("entry");
    block.insts.push(MInst::def(
        left,
        MType::I32,
        MOp::Param { index: 0 },
        Operands::one(Operand::Immediate(0)),
    ));
    block.insts.push(MInst::def(
        right,
        MType::I32,
        MOp::Param { index: 1 },
        Operands::one(Operand::Immediate(1)),
    ));
    block.insts.push(MInst::def(
        result,
        MType::I32,
        MOp::ICmp { pred: codevar_ocl_native_codegen::mir::IPred::Slt },
        Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
    ));
    block.insts.push(MInst::none(
        MType::Void,
        MOp::Ret,
        Operands::one(Operand::Virtual(result)),
    ));
    function.push_block(block);
    function
}

/// Builds a function that selects between its second and third
/// parameter.
fn select() -> MirFunction {
    let mut function = MirFunction::new("select");
    let cond = function.new_vreg(MType::I32);
    let yes = function.new_vreg(MType::I32);
    let no = function.new_vreg(MType::I32);
    let result = function.new_vreg(MType::I32);
    let mut block = MirBlock::new("entry");
    block.insts.push(MInst::def(
        cond,
        MType::I32,
        MOp::Param { index: 0 },
        Operands::one(Operand::Immediate(0)),
    ));
    block.insts.push(MInst::def(
        yes,
        MType::I32,
        MOp::Param { index: 1 },
        Operands::one(Operand::Immediate(1)),
    ));
    block.insts.push(MInst::def(
        no,
        MType::I32,
        MOp::Param { index: 2 },
        Operands::one(Operand::Immediate(2)),
    ));
    block.insts.push(MInst::def(
        result,
        MType::I32,
        MOp::Select,
        Operands::three(
            Operand::Virtual(cond),
            Operand::Virtual(yes),
            Operand::Virtual(no),
        ),
    ));
    block.insts.push(MInst::none(
        MType::Void,
        MOp::Ret,
        Operands::one(Operand::Virtual(result)),
    ));
    function.push_block(block);
    function
}

/// Builds a function that branches on its first parameter.
fn branch() -> MirFunction {
    let mut function = MirFunction::new("branch");
    let cond = function.new_vreg(MType::I32);
    let yes = function.new_vreg(MType::I32);
    let no = function.new_vreg(MType::I32);
    let mut block = MirBlock::new("entry");
    block.insts.push(MInst::def(
        cond,
        MType::I32,
        MOp::Param { index: 0 },
        Operands::one(Operand::Immediate(0)),
    ));
    block.insts.push(MInst::def(
        yes,
        MType::I32,
        MOp::Param { index: 1 },
        Operands::one(Operand::Immediate(1)),
    ));
    block.insts.push(MInst::def(
        no,
        MType::I32,
        MOp::Param { index: 2 },
        Operands::one(Operand::Immediate(2)),
    ));
    block.insts.push(MInst::none(
        MType::Void,
        MOp::CondBr,
        Operands::three(
            Operand::Virtual(cond),
            Operand::Immediate(1),
            Operand::Immediate(2),
        ),
    ));
    function.push_block(block);
    let mut taken = MirBlock::new("taken");
    taken.insts.push(MInst::none(
        MType::Void,
        MOp::Ret,
        Operands::one(Operand::Virtual(yes)),
    ));
    function.push_block(taken);
    let mut other = MirBlock::new("other");
    other.insts.push(MInst::none(
        MType::Void,
        MOp::Ret,
        Operands::one(Operand::Virtual(no)),
    ));
    function.push_block(other);
    function
}

/// Builds a function that stores a value in its frame and reads it
/// back.
fn memory() -> MirFunction {
    let mut function = MirFunction::new("memory");
    function.frame_size = 8;
    let value = function.new_vreg(MType::I32);
    let read = function.new_vreg(MType::I32);
    let mut block = MirBlock::new("entry");
    block.insts.push(MInst::def(
        value,
        MType::I32,
        MOp::Param { index: 0 },
        Operands::one(Operand::Immediate(0)),
    ));
    block.insts.push(MInst::none(
        MType::Void,
        MOp::SpillStore {
            offset: 0,
            size: 4,
        },
        Operands::one(Operand::Virtual(value)),
    ));
    block.insts.push(MInst::def(
        read,
        MType::I32,
        MOp::SpillLoad {
            offset: 0,
            size: 4,
        },
        Operands::none(),
    ));
    block.insts.push(MInst::none(
        MType::Void,
        MOp::Ret,
        Operands::one(Operand::Virtual(read)),
    ));
    function.push_block(block);
    function
}

#[test]
fn generate_the_module() {
    let target = ThumbV7mTarget::new();
    let mut functions: Vec<MirFunction> = OPERATIONS
        .iter()
        .map(|(name, op)| binary(name, op.clone()))
        .collect();
    functions.push(compare());
    functions.push(select());
    functions.push(branch());
    functions.push(memory());

    for function in &functions {
        let mut module = MirModule::new();
        let symbol = module.add_symbol("entry", SymbolKind::Function { index: 0 }, true);
        let mut function = function.clone();
        function.symbol = symbol;
        module.functions.push(function);
        let image = target
            .lower_module(&module, symbol)
            .expect("the module compiles");
        let path = format!("/tmp/opencode/thumbv7m-verify/f_{}.bin", functions
            .iter()
            .position(|f| f.name == module.functions[0].name)
            .unwrap_or(0));
        std::fs::write(&path, image.bytes()).expect("written");
        println!("{}: {} bytes", module.functions[0].name, image.bytes().len());
    }
}
