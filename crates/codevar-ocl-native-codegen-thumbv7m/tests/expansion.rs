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

//! The expansion of whole modules.
//!
//! The tests here build the machine code by hand, allocate it the
//! way the pipeline does, and check the code the back end writes
//! for it.

mod common;

use codevar_ocl_native_codegen::mir::{MOp, MType, Operand, Operands, SymId};

use codevar_ocl_native_codegen_thumbv7m::ThumbV7mTarget;

/// The bytes of the code of a module, as a string of hex digits.
fn hex(bytes: &[u8]) -> String {
    let mut out = String::new();
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// A module whose function divides its two parameters.
fn dividing_module() -> (codevar_ocl_native_codegen::mir::MirModule, SymId) {
    let mut builder = common::Builder::new("divide");
    let left = builder.vreg(MType::I32);
    let right = builder.vreg(MType::I32);
    let quotient = builder.vreg(MType::I32);
    builder.push(common::def(
        left,
        MType::I32,
        MOp::Param { index: 0 },
        common::one(Operand::Immediate(0)),
    ));
    builder.push(common::def(
        right,
        MType::I32,
        MOp::Param { index: 1 },
        common::one(Operand::Immediate(0)),
    ));
    builder.push(common::def(
        quotient,
        MType::I32,
        MOp::UDiv,
        common::two(common::vreg_operand(left), common::vreg_operand(right)),
    ));
    builder.push(common::none(
        MType::Void,
        MOp::Ret,
        common::one(common::vreg_operand(quotient)),
    ));
    builder.finish()
}

/// A module whose function branches on its first parameter.
fn branching_module() -> (codevar_ocl_native_codegen::mir::MirModule, SymId) {
    let mut builder = common::Builder::new("branch");
    let condition = builder.vreg(MType::I32);
    let value = builder.vreg(MType::I32);
    builder.push(common::def(
        condition,
        MType::I32,
        MOp::Param { index: 0 },
        common::one(Operand::Immediate(0)),
    ));
    builder.push(common::def(
        value,
        MType::I32,
        MOp::Param { index: 1 },
        common::one(Operand::Immediate(0)),
    ));
    builder.push(common::none(
        MType::Void,
        MOp::CondBr,
        common::three(
            common::vreg_operand(condition),
            Operand::Immediate(1),
            Operand::Immediate(2),
        ),
    ));
    builder.block("taken");
    builder.push(common::none(
        MType::Void,
        MOp::Ret,
        common::one(common::vreg_operand(value)),
    ));
    builder.block("other");
    builder.push(common::none(
        MType::Void,
        MOp::Ret,
        common::one(common::vreg_operand(value)),
    ));
    builder.finish()
}

/// A module whose function reads the work item and one of its
/// arguments.
fn work_item_module() -> (codevar_ocl_native_codegen::mir::MirModule, SymId) {
    let mut builder = common::Builder::new("kernel");
    let item = builder.vreg(MType::Ptr { size: 0 });
    let base = builder.vreg(MType::Ptr { size: 0 });
    let argument = builder.vreg(MType::I32);
    builder.push(common::def(
        item,
        MType::Ptr { size: 0 },
        MOp::Param { index: 0 },
        common::one(Operand::Immediate(0)),
    ));
    builder.push(common::def(
        base,
        MType::Ptr { size: 0 },
        MOp::ArgBlockBase,
        common::one(common::vreg_operand(item)),
    ));
    builder.push(common::def(
        argument,
        MType::I32,
        MOp::ArgLoad {
            index: 0,
            kind: codevar_ocl_native_codegen::mir::ArgKind::Scalar { size: 4 },
        },
        common::one(common::vreg_operand(base)),
    ));
    builder.push(common::none(
        MType::Void,
        MOp::Ret,
        common::one(common::vreg_operand(argument)),
    ));
    builder.finish()
}

/// A module whose function selects between its second and third
/// parameter.
fn selecting_module() -> (codevar_ocl_native_codegen::mir::MirModule, SymId) {
    let mut builder = common::Builder::new("select");
    let condition = builder.vreg(MType::I32);
    let yes = builder.vreg(MType::I32);
    let no = builder.vreg(MType::I32);
    let result = builder.vreg(MType::I32);
    builder.push(common::def(
        condition,
        MType::I32,
        MOp::Param { index: 0 },
        common::one(Operand::Immediate(0)),
    ));
    builder.push(common::def(
        yes,
        MType::I32,
        MOp::Param { index: 1 },
        common::one(Operand::Immediate(0)),
    ));
    builder.push(common::def(
        no,
        MType::I32,
        MOp::Param { index: 2 },
        common::one(Operand::Immediate(0)),
    ));
    builder.push(common::def(
        result,
        MType::I32,
        MOp::Select,
        common::three(
            common::vreg_operand(condition),
            common::vreg_operand(yes),
            common::vreg_operand(no),
        ),
    ));
    builder.push(common::none(
        MType::Void,
        MOp::Ret,
        common::one(common::vreg_operand(result)),
    ));
    builder.finish()
}

/// A module whose function stores a constant through a pointer.
fn storing_module() -> (codevar_ocl_native_codegen::mir::MirModule, SymId) {
    let mut builder = common::Builder::new("store");
    let pointer = builder.vreg(MType::Ptr { size: 4 });
    builder.push(common::def(
        pointer,
        MType::Ptr { size: 4 },
        MOp::Param { index: 0 },
        common::one(Operand::Immediate(0)),
    ));
    builder.push(common::none(
        MType::Void,
        MOp::Store { size: 4 },
        common::two(common::vreg_operand(pointer), common::imm(0x1234)),
    ));
    builder.push(common::none(
        MType::Void,
        MOp::RetVoid,
        common::one(common::imm(0)),
    ));
    builder.finish()
}

#[test]
fn a_store_of_a_constant_keeps_the_address() {
    // The address and the value of a store are both constants or
    // registers, and the two need a register of their own.
    let (module, entry) = storing_module();
    let target = ThumbV7mTarget::new();
    let image = common::lower(&target, &module, entry);
    let bytes = image.bytes();
    // The address is the first argument, the value is the constant
    // in a register of its own.
    let code = hex(bytes);
    assert!(code.contains("41f23422"), "{code}");
    assert!(code.contains("2260"), "{code}");
}

#[test]
fn the_division_becomes_a_loop() {
    // The architecture has no division instruction, so the division
    // becomes a loop that shifts the dividend into the remainder.
    //
    // The expected bytes are the ones the architecture gives for:
    //
    //   b470       push  {r4, r5, r6}
    //   b081       sub   sp, #4
    //   4604       mov   r4, r0
    //   460d       mov   r5, r1
    //   4620       mov   r0, r4
    //   4629       mov   r1, r5
    //   2200       movs  r2, #0
    //   2320       movs  r3, #32
    //   1800       adds  r0, r0, r0
    //   4152       adcs  r2, r2
    //   428a       cmp   r2, r1
    //   f0c0 8002  bcc.w 0x1e
    //   1a52       subs  r2, r2, r1
    //   3001       adds  r0, #1
    //   3b01       subs  r3, #1
    //   f47f aff6  bne.w 0x10
    //   4606       mov   r6, r0
    //   4630       mov   r0, r6
    //   b001       add   sp, #4
    //   bc70       pop   {r4, r5, r6}
    //   4770       bx    lr
    let (module, entry) = dividing_module();
    let target = ThumbV7mTarget::new();
    let image = common::lower(&target, &module, entry);
    assert_eq!(
        hex(image.bytes()),
        "70b481b004460d462046294600222023001852418a42c0f00280521a0130013b7ff4f6af0646304601b070bc7047"
    );
    assert_eq!(image.instruction_count(), 20, "the division is a loop");
}

#[test]
fn the_branches_of_a_function_are_relative() {
    let (module, entry) = branching_module();
    let target = ThumbV7mTarget::new();
    let image = common::lower(&target, &module, entry);
    let bytes = image.bytes();
    // Every window of four bytes of the code is checked against the
    // range of the user-space addresses of a thirty-two bit machine:
    // a branch is the only thing that names a place, and it names it
    // as a distance.
    for window in bytes.windows(4) {
        let value = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
        assert!(
            !(0x0000_4000..0x0000_8000).contains(&value),
            "the code holds the address {value:#010x}"
        );
    }
    // The function has three blocks, so the conditional branch and
    // the branch that follows it are both in the code.
    let code = hex(bytes);
    assert!(code.contains("40f00280"), "{code}");
    assert!(code.contains("00f003b8"), "{code}");
    assert_eq!(image.instruction_count(), 12);
}

#[test]
fn the_work_item_is_read_at_the_published_offsets() {
    // The argument block base is the word at the offset the ABI
    // publishes, and the argument is at the first slot of the block.
    //
    // The expected bytes are the ones the architecture gives for:
    //
    //   b470       push  {r4, r5, r6}
    //   b081       sub   sp, #4
    //   4604       mov   r4, r0
    //   6ca5       ldr   r5, [r4, #72]
    //   682e       ldr   r6, [r5, #0]
    //   4630       mov   r0, r6
    //   b001       add   sp, #4
    //   bc70       pop   {r4, r5, r6}
    //   4770       bx    lr
    let (module, entry) = work_item_module();
    let target = ThumbV7mTarget::new();
    let image = common::lower(&target, &module, entry);
    assert_eq!(hex(image.bytes()), "70b481b00446a56c2e68304601b070bc7047");
}

#[test]
fn the_selection_becomes_a_conditional_block() {
    // The selection is a compare and a conditional block with a
    // "then" and an "else" part.  The condition of the block is odd,
    // so the two parts of the block swap places in the encoding.
    let (module, entry) = selecting_module();
    let target = ThumbV7mTarget::new();
    let image = common::lower(&target, &module, entry);
    let code = hex(image.bytes());
    // The head of the block is the "not equal" condition with the
    // mask of a block with a "then" and an "else" part.
    assert!(code.contains("14bf"), "{code}");
    // The two parts of the block are the two moves.
    assert!(code.contains("2f46"), "{code}");
    assert!(code.contains("4637"), "{code}");
}

#[test]
fn a_module_with_a_call_compiles() {
    // The module has two functions: the entry calls the other one and
    // returns its result.
    let mut module = codevar_ocl_native_codegen::mir::MirModule::new();
    let callee = module.add_symbol(
        "callee",
        codevar_ocl_native_codegen::mir::SymbolKind::Function { index: 1 },
        false,
    );
    let entry = module.add_symbol(
        "entry",
        codevar_ocl_native_codegen::mir::SymbolKind::Function { index: 0 },
        true,
    );

    let mut caller = codevar_ocl_native_codegen::mir::MirFunction::new("entry");
    caller.symbol = entry;
    let mut block = codevar_ocl_native_codegen::mir::MirBlock::new("entry");
    let result = caller.new_vreg(MType::I32);
    block
        .insts
        .push(codevar_ocl_native_codegen::mir::MInst::def(
            result,
            MType::I32,
            MOp::Call { callee },
            Operands::none(),
        ));
    block
        .insts
        .push(codevar_ocl_native_codegen::mir::MInst::none(
            MType::Void,
            MOp::Ret,
            Operands::one(Operand::Virtual(result)),
        ));
    caller.push_block(block);

    let mut callee_function = codevar_ocl_native_codegen::mir::MirFunction::new("callee");
    callee_function.symbol = callee;
    let mut callee_block = codevar_ocl_native_codegen::mir::MirBlock::new("entry");
    callee_block
        .insts
        .push(codevar_ocl_native_codegen::mir::MInst::none(
            MType::Void,
            MOp::RetVoid,
            Operands::none(),
        ));
    callee_function.push_block(callee_block);

    module.functions.push(caller);
    module.functions.push(callee_function);
    common::allocate(&mut module);

    let target = ThumbV7mTarget::new();
    let image = common::lower(&target, &module, entry);
    // The call is a relative branch, and the result comes back in the
    // first register.
    let code = hex(image.bytes());
    assert!(code.contains("00f003f8"), "{code}");
    assert_eq!(image.entry_offset(), 0);
}
