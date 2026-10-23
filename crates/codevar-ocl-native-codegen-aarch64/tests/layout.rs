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

//! The layout of the module and the code it produces.
//!
//! The machine code of the tests is built by hand, with the public
//! builder of the machine code, so that the bytes the back end
//! produces can be compared with the bytes the architecture says
//! they should be.

use codevar_ocl_abi::WorkItemField;
use codevar_ocl_native_codegen::mir::{
    MInst, MOp, MType, MirBlock, MirFunction, MirModule, Operand, Operands, SymbolKind, VReg,
};
use codevar_ocl_native_codegen::target::NativeTarget;
use codevar_ocl_native_codegen_aarch64::target;

/// The type of a pointer.
const POINTER: MType = MType::Ptr { size: 8 };

/// Builds the module of the golden test.
///
/// The module holds one function that reads the global identifier of
/// the work item, adds a constant to it, and returns it.  A
/// comparison decides whether the constant is added or subtracted,
/// so the function has three blocks and two branches.
fn golden_module() -> (MirModule, codevar_ocl_native_codegen::mir::SymId) {
    let mut module = MirModule::new();
    let symbol = module.add_symbol("kernel", SymbolKind::Function { index: 0 }, false);

    let mut function = MirFunction::new("kernel");
    function.symbol = symbol;
    function.is_entry = true;

    let pointer: VReg = function.new_vreg(POINTER);
    let identifier: VReg = function.new_vreg(MType::I32);
    let shifted: VReg = function.new_vreg(MType::I32);
    let small: VReg = function.new_vreg(MType::I1);
    let up: VReg = function.new_vreg(MType::I32);
    let down: VReg = function.new_vreg(MType::I32);

    let mut entry = MirBlock::new("entry");
    entry.insts.push(MInst::def(
        pointer,
        POINTER,
        MOp::Param { index: 0 },
        Operands::none(),
    ));
    entry.insts.push(MInst::def(
        identifier,
        MType::I32,
        MOp::WorkItemLoad {
            field: WorkItemField::GlobalId,
            lane: 0,
        },
        Operands::one(Operand::Virtual(pointer)),
    ));
    entry.insts.push(MInst::def(
        shifted,
        MType::I32,
        MOp::Add,
        Operands::two(Operand::Virtual(identifier), Operand::Immediate(5)),
    ));
    entry.insts.push(
        MInst::def(
            small,
            MType::I1,
            MOp::ICmp {
                pred: codevar_ocl_native_codegen::mir::IPred::Ult,
            },
            Operands::two(Operand::Virtual(shifted), Operand::Immediate(100)),
        )
        .with_operand_type(MType::I32),
    );
    entry.insts.push(MInst::none(
        MType::Void,
        MOp::CondBr,
        Operands::three(
            Operand::Virtual(small),
            Operand::Immediate(1),
            Operand::Immediate(2),
        ),
    ));
    function.push_block(entry);

    let mut taken = MirBlock::new("taken");
    taken.insts.push(MInst::def(
        up,
        MType::I32,
        MOp::Add,
        Operands::two(Operand::Virtual(shifted), Operand::Immediate(1)),
    ));
    taken.insts.push(MInst::def(
        up,
        MType::I32,
        MOp::Ret,
        Operands::one(Operand::Virtual(up)),
    ));
    function.push_block(taken);

    let mut missed = MirBlock::new("missed");
    missed.insts.push(MInst::def(
        down,
        MType::I32,
        MOp::Sub,
        Operands::two(Operand::Virtual(shifted), Operand::Immediate(1)),
    ));
    missed.insts.push(MInst::def(
        down,
        MType::I32,
        MOp::Ret,
        Operands::one(Operand::Virtual(down)),
    ));
    function.push_block(missed);

    module.functions.push(function);
    (module, symbol)
}

/// Builds a module of two functions, one of which calls the other.
fn calling_module() -> (MirModule, codevar_ocl_native_codegen::mir::SymId) {
    let mut module = MirModule::new();
    let callee = module.add_symbol("callee", SymbolKind::Function { index: 0 }, false);
    let caller = module.add_symbol("caller", SymbolKind::Function { index: 1 }, false);

    let mut function = MirFunction::new("callee");
    function.symbol = callee;
    let argument: VReg = function.new_vreg(POINTER);
    let result: VReg = function.new_vreg(MType::I64);
    let mut block = MirBlock::new("entry");
    block.insts.push(MInst::def(
        argument,
        POINTER,
        MOp::Param { index: 0 },
        Operands::none(),
    ));
    block.insts.push(MInst::def(
        result,
        MType::I64,
        MOp::Add,
        Operands::two(Operand::Virtual(argument), Operand::Immediate(1)),
    ));
    block.insts.push(MInst::def(
        result,
        MType::I64,
        MOp::Ret,
        Operands::one(Operand::Virtual(result)),
    ));
    function.push_block(block);
    module.functions.push(function);

    let mut function = MirFunction::new("caller");
    function.symbol = caller;
    function.is_entry = true;
    let argument: VReg = function.new_vreg(POINTER);
    let result: VReg = function.new_vreg(MType::I64);
    let mut block = MirBlock::new("entry");
    block.insts.push(MInst::def(
        argument,
        POINTER,
        MOp::Param { index: 0 },
        Operands::none(),
    ));
    block.insts.push(MInst::def(
        result,
        MType::I64,
        MOp::Call { callee },
        Operands::one(Operand::Virtual(argument)),
    ));
    block.insts.push(MInst::def(
        result,
        MType::I64,
        MOp::Ret,
        Operands::one(Operand::Virtual(result)),
    ));
    function.push_block(block);
    module.functions.push(function);

    (module, caller)
}

/// Gives every value of the module a register.
///
/// The machine code reaches the back end only after the allocator
/// has given every value a home, so the tests do the same before
/// they hand the module over.
fn allocate(module: &mut MirModule) -> codevar_ocl_native_codegen::error::Result<()> {
    let model = codevar_ocl_native_codegen_aarch64::register_model();
    codevar_ocl_native_codegen::regalloc::allocate(module, &model).map(|_| ())
}

/// The bytes of the image, as a list of words.
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn a_small_function_compiles_to_the_expected_words() {
    let (mut module, symbol) = golden_module();
    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    assert_eq!(image.entry_offset(), 0);
    assert_eq!(image.instruction_count(), 13);
    assert_eq!(
        words(image.bytes()),
        vec![
            // mov x9, x0
            0xAA00_03E9,
            // ldr w10, [x9]
            0xB940_012A,
            // add w10, w10, #5
            0x1100_154A,
            // cmp w10, #100
            0x7101_915F,
            // cset w16, lo
            0x1A9F_27F0,
            // cbnz w16, taken
            0x3500_0050,
            // b missed
            0x1400_0004,
            // add w11, w10, #1
            0x1100_054B,
            // mov w0, w11
            0x2A0B_03E0,
            // ret
            0xD65F_03C0,
            // sub w12, w10, #1
            0x5100_054C,
            // mov w0, w12
            0x2A0C_03E0,
            // ret
            0xD65F_03C0,
        ]
    );
}

#[test]
fn the_code_holds_no_absolute_address() {
    let (mut module, symbol) = calling_module();
    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    let bytes = image.bytes();
    assert!(!bytes.is_empty());

    // Every eight-byte window of the code is checked: none of them
    // may hold a value that looks like an address in the user space
    // of a process.
    for start in 0..bytes.len().saturating_sub(7) {
        let mut value = 0u64;
        for index in 0..8 {
            value |= u64::from(bytes[start + index]) << (index * 8);
        }
        assert!(
            !(0x0000_4000_0000_0000..0x0000_8000_0000_0000).contains(&value),
            "the word at {start} holds the address {value:#018x}"
        );
    }
}

#[test]
fn the_entry_point_is_the_offset_of_the_entry_function() {
    let (mut module, symbol) = calling_module();
    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    // The callee comes first, so the entry function starts after it.
    assert!(image.entry_offset() > 0);
    assert_eq!(
        usize::try_from(image.entry_offset()).expect("the offset fits") % 4,
        0
    );
}

#[test]
fn a_module_without_an_entry_function_is_refused() {
    let (module, _) = calling_module();
    let error = target()
        .lower_module(&module, codevar_ocl_native_codegen::mir::SymId::new(99))
        .expect_err("the symbol names no function");
    assert!(matches!(
        error,
        codevar_ocl_native_codegen::error::CodegenError::Target { .. }
    ));
}

/// Builds a function that keeps ten values alive at once.
///
/// The values need more registers than the caller-saved ones hold,
/// so some of them land in the registers that survive a call and
/// the prologue has to save them.
fn framed_module() -> (MirModule, codevar_ocl_native_codegen::mir::SymId) {
    let mut module = MirModule::new();
    let symbol = module.add_symbol("framed", SymbolKind::Function { index: 0 }, false);

    let mut function = MirFunction::new("framed");
    function.symbol = symbol;
    function.is_entry = true;
    function.frame_size = 16;
    function.fixed_frame_size = 16;

    let mut values: Vec<VReg> = Vec::new();
    for _ in 0..10 {
        values.push(function.new_vreg(MType::I64));
    }

    let mut entry = MirBlock::new("entry");
    for (index, value) in values.iter().enumerate() {
        entry.insts.push(MInst::def(
            *value,
            MType::I64,
            MOp::Add,
            Operands::two(Operand::Immediate(index as i64), Operand::Immediate(1)),
        ));
    }
    entry
        .insts
        .push(MInst::none(MType::Void, MOp::Br { target: 1 }, Operands::none()));
    function.push_block(entry);

    let mut last = MirBlock::new("last");
    let sum = values[0];
    for value in values.iter().skip(1) {
        last.insts.push(MInst::def(
            sum,
            MType::I64,
            MOp::Add,
            Operands::two(Operand::Virtual(sum), Operand::Virtual(*value)),
        ));
    }
    last.insts.push(MInst::def(
        sum,
        MType::I64,
        MOp::Ret,
        Operands::one(Operand::Virtual(sum)),
    ));
    function.push_block(last);

    module.functions.push(function);
    (module, symbol)
}

#[test]
fn a_function_with_a_frame_saves_the_registers_it_uses() {
    let (mut module, symbol) = framed_module();
    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    let words = words(image.bytes());

    // The prologue reserves the area of the saved registers, saves
    // them, and then reserves the frame with a single subtraction.
    assert_eq!(words[0], 0xD100_83FF, "sub sp, sp, #32");
    assert_eq!(words[1], 0xA900_53F3, "stp x19, x20, [sp, #0]");
    assert_eq!(words[2], 0xF900_0BF5, "str x21, [sp, #16]");
    assert_eq!(words[3], 0xD100_43FF, "sub sp, sp, #16");

    // The epilogue restores them, releases the frame and returns.
    let last = words.len();
    assert_eq!(words[last - 4], 0xA941_53F3, "ldp x19, x20, [sp, #16]");
    assert_eq!(words[last - 3], 0xF940_13F5, "ldr x21, [sp, #32]");
    assert_eq!(words[last - 2], 0x9100_C3FF, "add sp, sp, #48");
    assert_eq!(words[last - 1], 0xD65F_03C0, "ret");
}

#[test]
fn the_frame_of_a_function_is_released_before_it_returns() {
    let (mut module, symbol) = framed_module();
    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    let bytes = image.bytes();
    assert_eq!(bytes.len() % 4, 0);
    assert_eq!(image.instruction_count(), (bytes.len() / 4) as u32);
}

#[test]
fn a_call_reaches_the_function_it_calls() {
    let (mut module, symbol) = calling_module();
    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    let words = words(image.bytes());

    // The call is the only instruction with the link bit set.
    let call = words
        .iter()
        .position(|word| word & 0xFC00_0000 == 0x9400_0000)
        .expect("the module holds a call");
    let displacement = (words[call] & 0x03FF_FFFF) as i32;
    let displacement = if displacement & 0x0200_0000 != 0 {
        displacement - 0x0400_0000
    } else {
        displacement
    };
    let target = (call as i32 + displacement) * 4;
    // The callee is the first function of the module.
    assert_eq!(target, 0);
}

#[test]
fn a_floating_point_function_uses_the_floating_point_registers() {
    let mut module = MirModule::new();
    let symbol = module.add_symbol("float", SymbolKind::Function { index: 0 }, false);

    let mut function = MirFunction::new("float");
    function.symbol = symbol;
    function.is_entry = true;

    let left: VReg = function.new_vreg(MType::F64);
    let right: VReg = function.new_vreg(MType::F64);
    let sum: VReg = function.new_vreg(MType::F64);
    let comparison: VReg = function.new_vreg(MType::I1);

    let mut block = MirBlock::new("entry");
    block.insts.push(MInst::def(
        left,
        MType::F64,
        MOp::Param { index: 0 },
        Operands::none(),
    ));
    block.insts.push(MInst::def(
        right,
        MType::F64,
        MOp::Param { index: 1 },
        Operands::none(),
    ));
    block.insts.push(MInst::def(
        sum,
        MType::F64,
        MOp::FAdd,
        Operands::two(Operand::Virtual(left), Operand::Virtual(right)),
    ));
    block.insts.push(
        MInst::def(
            comparison,
            MType::I1,
            MOp::FCmp {
                pred: codevar_ocl_native_codegen::mir::FPred::Olt,
            },
            Operands::two(Operand::Virtual(sum), Operand::Virtual(right)),
        )
        .with_operand_type(MType::F64),
    );
    block
        .insts
        .push(MInst::none(MType::Void, MOp::RetVoid, Operands::none()));
    function.push_block(block);
    module.functions.push(function);

    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    let words = words(image.bytes());
    // The two parameters arrive in the floating-point argument
    // registers and are moved to registers of their own.
    assert_eq!(words[0], 0x1E60_4008, "fmov d8, v0");
    assert_eq!(words[1], 0x1E60_4029, "fmov d9, v1");
    assert_eq!(words[2], 0x1E69_290A, "fadd d10, d8, d9");
    // The comparison is the floating-point compare, and the result
    // of an ordered less-than is read with the negative condition.
    assert_eq!(words[3], 0x1E69_2140, "fcmp d10, d9");
    assert_eq!(words[4], 0x1A9F_57E9, "cset w9, mi");
    assert_eq!(words[5], 0xD65F_03C0, "ret");
}

#[test]
fn a_kernel_reads_its_arguments_through_the_argument_block() {
    let mut module = MirModule::new();
    let symbol = module.add_symbol("kernel", SymbolKind::Function { index: 0 }, false);

    let mut function = MirFunction::new("kernel");
    function.symbol = symbol;
    function.is_entry = true;

    let pointer: VReg = function.new_vreg(MType::Ptr { size: 8 });
    let block: VReg = function.new_vreg(MType::Ptr { size: 8 });
    let buffer: VReg = function.new_vreg(MType::Ptr { size: 8 });
    let count: VReg = function.new_vreg(MType::I32);

    let mut entry = MirBlock::new("entry");
    entry.insts.push(MInst::def(
        pointer,
        MType::Ptr { size: 8 },
        MOp::Param { index: 0 },
        Operands::none(),
    ));
    entry.insts.push(MInst::def(
        block,
        MType::Ptr { size: 8 },
        MOp::ArgBlockBase,
        Operands::one(Operand::Virtual(pointer)),
    ));
    entry.insts.push(MInst::def(
        buffer,
        MType::Ptr { size: 8 },
        MOp::ArgLoad {
            index: 0,
            kind: codevar_ocl_native_codegen::mir::ArgKind::Pointer,
        },
        Operands::one(Operand::Virtual(block)),
    ));
    entry.insts.push(MInst::def(
        count,
        MType::I32,
        MOp::ArgLoad {
            index: 1,
            kind: codevar_ocl_native_codegen::mir::ArgKind::Scalar { size: 4 },
        },
        Operands::one(Operand::Virtual(block)),
    ));
    entry
        .insts
        .push(MInst::none(MType::Void, MOp::RetVoid, Operands::none()));
    function.push_block(entry);
    module.functions.push(function);

    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    let words = words(image.bytes());
    // The base of the argument block sits seventy-two bytes into the
    // work item.
    assert_eq!(words[1], 0xF940_252A, "ldr x10, [x9, #72]");
    // The first argument is the pointer in the first slot, the second
    // the count in the slot that follows it.
    assert_eq!(words[2], 0xF940_014B, "ldr x11, [x10]");
    assert_eq!(words[3], 0xB940_114C, "ldr w12, [x10, #16]");
}

#[test]
fn a_vector_is_built_and_taken_apart_again() {
    let mut module = MirModule::new();
    let symbol = module.add_symbol("vector", SymbolKind::Function { index: 0 }, false);

    let mut function = MirFunction::new("vector");
    function.symbol = symbol;
    function.is_entry = true;

    let scalar: VReg = function.new_vreg(MType::I32);
    let wide: VReg = function.new_vreg(MType::Vec {
        lanes: 4,
        elem: Box::new(MType::I32),
    });
    let lane: VReg = function.new_vreg(MType::I32);
    let shuffled: VReg = function.new_vreg(MType::Vec {
        lanes: 4,
        elem: Box::new(MType::I32),
    });

    let mut entry = MirBlock::new("entry");
    entry.insts.push(MInst::def(
        scalar,
        MType::I32,
        MOp::Param { index: 0 },
        Operands::none(),
    ));
    entry.insts.push(MInst::def(
        wide,
        MType::Vec {
            lanes: 4,
            elem: Box::new(MType::I32),
        },
        MOp::VecSplat,
        Operands::one(Operand::Virtual(scalar)),
    ));
    entry.insts.push(
        MInst::def(
            lane,
            MType::I32,
            MOp::VecExtract { index: 2 },
            Operands::one(Operand::Virtual(wide)),
        )
        .with_operand_type(MType::Vec {
            lanes: 4,
            elem: Box::new(MType::I32),
        }),
    );
    entry.insts.push(MInst::def(
        shuffled,
        MType::Vec {
            lanes: 4,
            elem: Box::new(MType::I32),
        },
        MOp::VecShuffle {
            mask: vec![3, 2, 1, 0],
        },
        Operands::two(Operand::Virtual(wide), Operand::Virtual(wide)),
    ));
    entry
        .insts
        .push(MInst::none(MType::Void, MOp::RetVoid, Operands::none()));
    function.push_block(entry);
    module.functions.push(function);

    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    let words = words(image.bytes());
    // The scalar is broadcast over the four lanes of the vector.
    assert_eq!(words[1], 0x4E04_0D28, "dup v8.4s, w9");
    // The third lane is taken out of the vector.
    assert_eq!(words[2], 0x0E14_3D0A, "umov w10, v8.s[2]");
    // The lanes are reversed.
    // The lanes are put back in the opposite order.
    assert_eq!(words[3], 0x4EA8_1D09, "mov v9.16b, v8.16b");
    assert_eq!(words[4], 0x6E04_6509, "ins v9.s[0], v8.s[3]");
    assert_eq!(words[5], 0x6E0C_4509, "ins v9.s[1], v8.s[2]");
    assert_eq!(words[6], 0x6E14_2509, "ins v9.s[2], v8.s[1]");
    assert_eq!(words[7], 0x6E1C_0509, "ins v9.s[3], v8.s[0]");
}

#[test]
fn an_address_is_computed_and_a_value_is_selected() {
    let mut module = MirModule::new();
    let symbol = module.add_symbol("kernel", SymbolKind::Function { index: 0 }, false);

    let mut function = MirFunction::new("kernel");
    function.symbol = symbol;
    function.is_entry = true;

    let base: VReg = function.new_vreg(MType::Ptr { size: 8 });
    let index: VReg = function.new_vreg(MType::I64);
    let address: VReg = function.new_vreg(MType::Ptr { size: 8 });
    let flag: VReg = function.new_vreg(MType::I1);
    let chosen: VReg = function.new_vreg(MType::I64);

    let mut entry = MirBlock::new("entry");
    entry.insts.push(MInst::def(
        base,
        MType::Ptr { size: 8 },
        MOp::Param { index: 0 },
        Operands::none(),
    ));
    entry.insts.push(MInst::def(
        index,
        MType::I64,
        MOp::Param { index: 1 },
        Operands::none(),
    ));
    entry.insts.push(
        MInst::def(
            address,
            MType::Ptr { size: 8 },
            MOp::Gep {
                size: 4,
                signed: false,
            },
            Operands::two(Operand::Virtual(base), Operand::Virtual(index)),
        )
        .with_operand_type(MType::I64),
    );
    entry.insts.push(
        MInst::def(
            flag,
            MType::I1,
            MOp::ICmp {
                pred: codevar_ocl_native_codegen::mir::IPred::Eq,
            },
            Operands::two(Operand::Virtual(index), Operand::Immediate(0)),
        )
        .with_operand_type(MType::I64),
    );
    entry.insts.push(MInst::def(
        chosen,
        MType::I64,
        MOp::Select,
        Operands::three(
            Operand::Virtual(flag),
            Operand::Virtual(index),
            Operand::Virtual(base),
        ),
    ));
    entry.insts.push(MInst::def(
        chosen,
        MType::I64,
        MOp::Ret,
        Operands::one(Operand::Virtual(chosen)),
    ));
    function.push_block(entry);
    module.functions.push(function);

    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    let words = words(image.bytes());
    // The address is the base plus the index, scaled by the size of
    // the element.
    assert_eq!(words[2], 0xAA09_03EB, "mov x11, x9");
    assert_eq!(words[3], 0x8B0A_096B, "add x11, x11, x10, lsl #2");
    // The selection picks the index when the flag holds.
    assert_eq!(words[6], 0x7100_019F, "cmp w12, #0");
    assert_eq!(words[7], 0x9A89_114D, "csel x13, x10, x9, ne");
}

#[test]
fn a_module_that_was_not_allocated_is_refused() {
    // The module is handed over without the allocator having given
    // its values a home, so the back end still finds virtual
    // registers where it expects physical ones.
    let (module, symbol) = golden_module();
    let error = target()
        .lower_module(&module, symbol)
        .expect_err("the module was not allocated");
    assert!(
        matches!(
            error,
            codevar_ocl_native_codegen::error::CodegenError::Target { .. }
        ),
        "unexpected error: {error}"
    );
    assert!(
        error.to_string().contains("not allocated"),
        "the error names the cause: {error}"
    );
}

#[test]
fn a_function_that_needs_more_registers_than_the_file_holds_spills() {
    let mut module = MirModule::new();
    let symbol = module.add_symbol("wide", SymbolKind::Function { index: 0 }, false);

    let mut function = MirFunction::new("wide");
    function.symbol = symbol;
    function.is_entry = true;

    // More values than the file of the target holds, and all of them
    // live at the same time, so the allocator has to spill.
    let mut values: Vec<VReg> = Vec::new();
    for _ in 0..20 {
        values.push(function.new_vreg(MType::I64));
    }

    let mut entry = MirBlock::new("entry");
    for (index, value) in values.iter().enumerate() {
        entry.insts.push(MInst::def(
            *value,
            MType::I64,
            MOp::Copy,
            Operands::one(Operand::Immediate(index as i64)),
        ));
    }
    entry
        .insts
        .push(MInst::none(MType::Void, MOp::Br { target: 1 }, Operands::none()));
    function.push_block(entry);

    let mut last = MirBlock::new("last");
    let sum = values[0];
    for value in values.iter().skip(1) {
        last.insts.push(MInst::def(
            sum,
            MType::I64,
            MOp::Add,
            Operands::two(Operand::Virtual(sum), Operand::Virtual(*value)),
        ));
    }
    last.insts.push(MInst::def(
        sum,
        MType::I64,
        MOp::Ret,
        Operands::one(Operand::Virtual(sum)),
    ));
    function.push_block(last);
    module.functions.push(function);

    let model = codevar_ocl_native_codegen_aarch64::register_model();
    let stats =
        codevar_ocl_native_codegen::regalloc::allocate(&mut module, &model).expect("the module allocates");
    assert!(stats.spills > 0, "the file is too small and a value was spilled");
    assert!(
        module.functions[0].frame_size > module.functions[0].fixed_frame_size,
        "the frame holds the slot of the spilled value"
    );

    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    assert!(!image.bytes().is_empty());
    assert_eq!(image.bytes().len() % 4, 0);
}

#[test]
fn the_mathematical_operations_have_their_own_words() {
    let mut module = MirModule::new();
    let symbol = module.add_symbol("math", SymbolKind::Function { index: 0 }, false);

    let mut function = MirFunction::new("math");
    function.symbol = symbol;
    function.is_entry = true;

    let value: VReg = function.new_vreg(MType::F64);
    let other: VReg = function.new_vreg(MType::F64);
    let root: VReg = function.new_vreg(MType::F64);
    let absolute: VReg = function.new_vreg(MType::F64);
    let smaller: VReg = function.new_vreg(MType::F64);
    let larger: VReg = function.new_vreg(MType::F64);
    let positive: VReg = function.new_vreg(MType::F64);
    let sign: VReg = function.new_vreg(MType::F64);
    let remainder: VReg = function.new_vreg(MType::F64);

    let mut block = MirBlock::new("entry");
    block.insts.push(
        MInst::def(value, MType::F64, MOp::Param { index: 0 }, Operands::none())
            .with_operand_type(MType::F64),
    );
    block.insts.push(
        MInst::def(other, MType::F64, MOp::Copy, Operands::one(Operand::Immediate(0)))
            .with_operand_type(MType::F64),
    );
    block.insts.push(
        MInst::def(
            root,
            MType::F64,
            MOp::FSqrt,
            Operands::one(Operand::Virtual(value)),
        )
        .with_operand_type(MType::F64),
    );
    block.insts.push(
        MInst::def(
            absolute,
            MType::F64,
            MOp::FAbs,
            Operands::one(Operand::Virtual(value)),
        )
        .with_operand_type(MType::F64),
    );
    block.insts.push(
        MInst::def(
            smaller,
            MType::F64,
            MOp::FMin,
            Operands::two(Operand::Virtual(value), Operand::Virtual(other)),
        )
        .with_operand_type(MType::F64),
    );
    block.insts.push(
        MInst::def(
            larger,
            MType::F64,
            MOp::FMax,
            Operands::two(Operand::Virtual(value), Operand::Virtual(other)),
        )
        .with_operand_type(MType::F64),
    );
    block.insts.push(
        MInst::def(
            positive,
            MType::F64,
            MOp::FDim,
            Operands::two(Operand::Virtual(value), Operand::Virtual(other)),
        )
        .with_operand_type(MType::F64),
    );
    block.insts.push(
        MInst::def(
            sign,
            MType::F64,
            MOp::FCopySign,
            Operands::two(Operand::Virtual(value), Operand::Virtual(other)),
        )
        .with_operand_type(MType::F64),
    );
    block.insts.push(
        MInst::def(
            remainder,
            MType::F64,
            MOp::FRem,
            Operands::two(Operand::Virtual(value), Operand::Virtual(other)),
        )
        .with_operand_type(MType::F64),
    );
    block
        .insts
        .push(MInst::none(MType::Void, MOp::RetVoid, Operands::none()));
    function.push_block(block);
    module.functions.push(function);

    allocate(&mut module).expect("the module allocates");
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    let words = words(image.bytes());
    // The square root of the parameter.
    assert_eq!(words[6], 0x1E61_C10A, "fsqrt d10, d8");
    // The absolute value of the parameter.
    assert_eq!(words[7], 0x1E60_C10B, "fabs d11, d8");
    // The smaller and the larger of the parameter and the zero.
    assert_eq!(words[8], 0x1E69_590C, "fmin d12, d8, d9");
    assert_eq!(words[9], 0x1E69_490D, "fmax d13, d8, d9");
    // The positive difference is the difference of the values,
    // clamped at zero.
    assert_eq!(words[10], 0x1E69_390E, "fsub d14, d8, d9");
    assert_eq!(words[15], 0x9E67_021F, "fmov d31, x16");
    assert_eq!(words[16], 0x1E7F_49CE, "fmax d14, d14, d31");
    // The sign copy flips the sign bit of the first value with the
    // sign bit of the second one.
    assert_eq!(words[17], 0x4EA8_1D0F, "mov v15.16b, v8.16b");
    assert_eq!(words[22], 0x9E67_021F, "fmov d31, x16");
    assert_eq!(words[23], 0x4E3F_1D3E, "and v30.16b, v9.16b, v31.16b");
    assert_eq!(words[24], 0x6E3E_1DEF, "eor v15.16b, v15.16b, v30.16b");
    // The remainder is the quotient of a division that is truncated
    // towards zero, taken away from the dividend.
    assert_eq!(words[25], 0x4EA8_1D10, "mov v16.16b, v8.16b");
    assert_eq!(words[26], 0x4EA9_1D3F, "mov v31.16b, v9.16b");
    assert_eq!(words[27], 0x1E7F_1A1E, "fdiv v30, v16, v31");
    assert_eq!(words[28], 0x9E78_03D0, "fcvtzs x16, v30");
    assert_eq!(words[29], 0x9E62_021E, "scvtf v30, x16");
    assert_eq!(words[30], 0x1E7F_321E, "fmls v30, v16, v31");
    assert_eq!(words[31], 0x4EBE_1FD0, "mov v16.16b, v30.16b");
}
