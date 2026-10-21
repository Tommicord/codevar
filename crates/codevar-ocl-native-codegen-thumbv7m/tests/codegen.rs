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

//! The code of whole modules.

mod common;

use codevar_ocl_native_codegen::error::CodegenError;
use codevar_ocl_native_codegen::mir::{
    MOp, MType, Operand, SymbolKind, SymId,
};
use codevar_ocl_native_codegen::target::NativeTarget;

use codevar_ocl_native_codegen_thumbv7m::ThumbV7mTarget;

/// The bytes of the code of a module, as a string of hex digits.
fn hex(bytes: &[u8]) -> String {
    let mut out = String::new();
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// A module that adds two numbers and returns the sum.
///
/// The function reads its two arguments from the argument registers,
/// adds them and returns the sum in the first one.
fn adding_module() -> (codevar_ocl_native_codegen::mir::MirModule, SymId) {
    let mut builder = common::Builder::new("add");
    let left = builder.vreg(MType::I32);
    let right = builder.vreg(MType::I32);
    let sum = builder.vreg(MType::I32);
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
        sum,
        MType::I32,
        MOp::Add,
        common::two(
            common::vreg_operand(left),
            common::vreg_operand(right),
        ),
    ));
    builder.push(common::none(
        MType::Void,
        MOp::Ret,
        common::one(common::vreg_operand(sum)),
    ));
    builder.finish()
}

#[test]
fn the_code_of_a_small_function_is_exact() {
    // The function moves its two arguments into the registers the
    // allocator gave them, adds them and returns the sum.
    //
    // The expected bytes are:
    //
    //   04 b5      push {lr}
    //   00 46      mov  r0, r0        ; the first argument
    //   01 46      mov  r1, r1        ; the second argument
    //   08 18      adds r0, r0, r1    ; the sum
    //   00 bd      pop  {pc}
    let (module, entry) = adding_module();
    let target = ThumbV7mTarget::new();
    let image = target
        .lower_module(&module, entry)
        .expect("the module compiles");
    assert_eq!(
        hex(image.bytes()),
        "04b5004601460818 00bd".replace(' ', "")
    );
    assert_eq!(image.entry_offset(), 0);
    assert_eq!(image.instruction_count(), 6);
}

#[test]
fn the_code_holds_no_absolute_address() {
    // Every window of four bytes of the code is checked against the
    // range of the user-space addresses of a thirty-two bit machine.
    // A branch is the only thing that names a place, and it names it
    // as a distance, so no window may hold an address.
    let (module, entry) = adding_module();
    let target = ThumbV7mTarget::new();
    let image = target
        .lower_module(&module, entry)
        .expect("the module compiles");
    let bytes = image.bytes();
    assert!(bytes.len() >= 4);
    for window in bytes.windows(4) {
        let value = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
        assert!(
            !(0x0000_4000..0x0000_8000).contains(&value),
            "the code holds the address {value:#010x}"
        );
    }
}

#[test]
fn the_floating_point_operations_are_refused_without_the_unit() {
    let mut builder = common::Builder::new("float");
    let left = builder.vreg(MType::F32);
    let right = builder.vreg(MType::F32);
    let sum = builder.vreg(MType::F32);
    builder.push(common::def(
        sum,
        MType::F32,
        MOp::FAdd,
        common::two(
            common::vreg_operand(left),
            common::vreg_operand(right),
        ),
    ));
    let (module, entry) = builder.finish();
    let target = ThumbV7mTarget::without_fp();
    let error = target
        .lower_module(&module, entry)
        .expect_err("the floating-point operation is refused");
    assert!(matches!(error, CodegenError::Unsupported { .. }));
    let detail = match error {
        CodegenError::Unsupported { detail, .. } => detail,
        _ => String::new(),
    };
    assert!(detail.contains("floating-point unit"), "{detail}");
}

#[test]
fn the_vector_operations_are_refused() {
    let mut builder = common::Builder::new("vector");
    let source = builder.vreg(MType::I32);
    let target = builder.vreg(MType::I32);
    builder.push(common::def(
        target,
        MType::I32,
        MOp::VecSplat,
        common::one(common::vreg_operand(source)),
    ));
    let (module, entry) = builder.finish();
    let target = ThumbV7mTarget::new();
    let error = target
        .lower_module(&module, entry)
        .expect_err("the vector operation is refused");
    assert!(matches!(error, CodegenError::Unsupported { .. }));
}

#[test]
fn a_call_to_a_function_of_the_module_is_relative() {
    // The module has two functions: the entry calls the other one.
    let mut module = codevar_ocl_native_codegen::mir::MirModule::new();
    let callee = module.add_symbol("callee", SymbolKind::Function { index: 1 }, false);
    let entry = module.add_symbol("entry", SymbolKind::Function { index: 0 }, true);

    let mut caller = codevar_ocl_native_codegen::mir::MirFunction::new("entry");
    caller.symbol = entry;
    let mut block = codevar_ocl_native_codegen::mir::MirBlock::new("entry");
    let result = caller.new_vreg(MType::I32);
    block.insts.push(codevar_ocl_native_codegen::mir::MInst::def(
        result,
        MType::I32,
        MOp::Call { callee },
        codevar_ocl_native_codegen::mir::Operands::none(),
    ));
    block
        .insts
        .push(codevar_ocl_native_codegen::mir::MInst::none(
            MType::Void,
            MOp::Ret,
            codevar_ocl_native_codegen::mir::Operands::one(
                Operand::Virtual(result),
            ),
        ));
    caller.push_block(block);

    let mut callee_function = codevar_ocl_native_codegen::mir::MirFunction::new("callee");
    callee_function.symbol = callee;
    let mut callee_block = codevar_ocl_native_codegen::mir::MirBlock::new("entry");
    let value = callee_function.new_vreg(MType::I32);
    callee_block
        .insts
        .push(codevar_ocl_native_codegen::mir::MInst::none(
            MType::Void,
            MOp::RetVoid,
            codevar_ocl_native_codegen::mir::Operands::none(),
        ));
    let _ = value;
    callee_function.push_block(callee_block);

    module.functions.push(caller);
    module.functions.push(callee_function);

    let target = ThumbV7mTarget::new();
    let image = target
        .lower_module(&module, entry)
        .expect("the module compiles");
    // The call is a relative branch, so the code holds no address.
    let bytes = image.bytes();
    for window in bytes.windows(4) {
        let value = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
        assert!(
            !(0x0000_4000..0x0000_8000).contains(&value),
            "the code holds the address {value:#010x}"
        );
    }
    // The entry point is the first function of the module.
    assert_eq!(image.entry_offset(), 0);
}

#[test]
fn the_register_model_names_the_registers_of_the_architecture() {
    let target = ThumbV7mTarget::new();
    let model = target.registers();
    assert_eq!(model.int_regs.len(), 13);
    assert_eq!(model.callee_saved.len(), 8);
    assert_eq!(model.arg_regs.len(), 4);
    assert_eq!(model.return_regs.len(), 1);
    assert_eq!(model.pointer_size, 4);
    assert_eq!(model.float_regs.len(), 32);
    assert_eq!(model.vec_regs.len(), 8);
    assert_eq!(model.stack_pointer.number(), 13);
}

#[test]
fn the_core_without_the_unit_has_no_floating_point_registers() {
    let target = ThumbV7mTarget::without_fp();
    let model = target.registers();
    assert!(model.float_regs.is_empty());
    assert!(model.vec_regs.is_empty());
    assert_eq!(model.int_regs.len(), 13);
}
