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

//! The layout of the compiled module.

// A test that cannot build its module, or that cannot map its page of
// executable memory, has nothing to say about the back end: the test
// itself fails, and the panic says why.
#![allow(clippy::expect_used, clippy::unwrap_used)]
mod common;

use codevar_ocl_native_codegen::mir::{MOp, MType, MirFunction, MirModule, Operands, SymbolKind};
use codevar_ocl_native_codegen::target::NativeTarget;
use codevar_ocl_native_codegen_x86::target;
use common::*;

/// The lowest address of the user space of a forty-eight-bit system.
const USER_SPACE_START: u64 = 0x0000_4000_0000_0000;
/// The first address above the user space of a forty-eight-bit
/// system.
const USER_SPACE_END: u64 = 0x0000_8000_0000_0000;

/// Builds a module of two functions: the first one adds one to its
/// argument, the second one calls it.
fn two_functions() -> (MirModule, codevar_ocl_native_codegen::mir::SymId) {
    let mut callee = MirFunction::new("callee");
    let mut entry = block("entry");
    let argument = vreg(&mut callee, RBX, MType::I64);
    let result = vreg(&mut callee, R10, MType::I64);
    entry.insts.push(def(
        argument,
        MType::I64,
        MOp::Param { index: 0 },
        Operands::none(),
    ));
    entry
        .insts
        .push(def(result, MType::I64, MOp::Add, Operands::two(reg(RBX), imm(1))));
    entry
        .insts
        .push(def(result, MType::I64, MOp::Ret, Operands::one(reg(R10))));
    callee.push_block(entry);
    callee.frame_size = 0;

    let mut caller = MirFunction::new("caller");
    let mut entry = block("entry");
    let argument = vreg(&mut caller, RBX, MType::I64);
    let result = vreg(&mut caller, R10, MType::I64);
    entry.insts.push(def(
        argument,
        MType::I64,
        MOp::Param { index: 0 },
        Operands::none(),
    ));
    entry.insts.push(def(
        result,
        MType::I64,
        MOp::Call {
            callee: callee.symbol,
        },
        Operands::one(reg(RBX)),
    ));
    entry
        .insts
        .push(def(result, MType::I64, MOp::Ret, Operands::one(reg(R10))));
    caller.push_block(entry);
    caller.frame_size = 0;

    let mut module = MirModule::new();
    let callee_symbol = module.add_symbol("callee", SymbolKind::Function { index: 0 }, false);
    let caller_symbol = module.add_symbol("caller", SymbolKind::Function { index: 1 }, false);
    callee.symbol = callee_symbol;
    caller.symbol = caller_symbol;
    module.functions.push(callee);
    module.functions.push(caller);
    (module, caller_symbol)
}

/// Whether the code holds an address of the user space.
fn holds_absolute_address(bytes: &[u8]) -> bool {
    if bytes.len() < 8 {
        return false;
    }
    bytes.windows(8).any(|window| {
        let value = u64::from_ne_bytes([
            window[0], window[1], window[2], window[3], window[4], window[5], window[6], window[7],
        ]);
        (USER_SPACE_START..USER_SPACE_END).contains(&value)
    })
}

#[test]
fn the_code_holds_no_absolute_address() {
    let (module, entry) = two_functions();
    let image = target()
        .lower_module(&module, entry)
        .expect("the module compiles");
    assert!(!image.bytes().is_empty());
    assert!(
        !holds_absolute_address(image.bytes()),
        "the code holds an address of the user space"
    );
}

#[test]
fn the_call_reaches_the_callee_relatively() {
    let (module, entry) = two_functions();
    let image = target()
        .lower_module(&module, entry)
        .expect("the module compiles");
    let bytes = image.bytes();
    // The callee holds no call, so the first call instruction of the
    // module is the one of the caller.
    let site = bytes
        .iter()
        .position(|byte| *byte == 0xE8)
        .expect("the module holds a call");
    let mut displacement = [0u8; 4];
    displacement.copy_from_slice(&bytes[site + 1..site + 5]);
    let displacement = i32::from_le_bytes(displacement);
    let reached = site as i64 + 5 + i64::from(displacement);
    assert_eq!(reached, 0, "the call does not reach the callee");
}

#[test]
fn the_entry_point_points_at_the_entry_function() {
    let (module, entry) = two_functions();
    let image = target()
        .lower_module(&module, entry)
        .expect("the module compiles");
    let offset = image.entry_offset() as usize;
    assert!(offset < image.bytes().len());
    assert!(image.instruction_count() > 0);
}

#[test]
fn a_branch_to_a_block_that_does_not_exist_is_refused() {
    let mut function = MirFunction::new("broken");
    let mut entry = block("entry");
    let result = vreg(&mut function, R10, MType::I64);
    entry
        .insts
        .push(none(MType::Void, MOp::Br { target: 7 }, Operands::none()));
    entry
        .insts
        .push(def(result, MType::I64, MOp::Ret, Operands::one(reg(R10))));
    function.push_block(entry);
    let (module, entry) = module_with(function);
    let error = target()
        .lower_module(&module, entry)
        .expect_err("the branch is refused");
    assert!(matches!(
        error,
        codevar_ocl_native_codegen::error::CodegenError::Target { .. }
    ));
}

#[test]
fn a_call_to_a_symbol_that_is_not_a_function_is_refused() {
    let mut function = MirFunction::new("broken");
    let mut entry = block("entry");
    let result = vreg(&mut function, R10, MType::I64);
    entry.insts.push(def(
        result,
        MType::I64,
        MOp::Call {
            callee: codevar_ocl_native_codegen::mir::SymId::new(9),
        },
        Operands::none(),
    ));
    function.push_block(entry);
    let (module, entry) = module_with(function);
    let error = target()
        .lower_module(&module, entry)
        .expect_err("the call is refused");
    assert!(matches!(
        error,
        codevar_ocl_native_codegen::error::CodegenError::Target { .. }
    ));
}

#[test]
fn a_floating_point_remainder_is_compiled() {
    let mut function = MirFunction::new("remainder");
    let mut entry = block("entry");
    let left = vreg(&mut function, XMM0, MType::F32);
    let right = vreg(&mut function, XMM1, MType::F32);
    let result = vreg(&mut function, XMM2, MType::F32);
    entry
        .insts
        .push(def(left, MType::F32, MOp::Copy, Operands::one(reg(XMM0))));
    entry
        .insts
        .push(def(right, MType::F32, MOp::Copy, Operands::one(reg(XMM1))));
    entry.insts.push(def(
        result,
        MType::F32,
        MOp::FRem,
        Operands::two(reg(XMM0), reg(XMM1)),
    ));
    function.push_block(entry);
    let (module, entry) = module_with(function);
    let image = target()
        .lower_module(&module, entry)
        .expect("the remainder compiles");
    assert!(!image.bytes().is_empty());
}
