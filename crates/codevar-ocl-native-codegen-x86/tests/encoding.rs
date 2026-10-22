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

//! The bytes of the mathematical operations.
//!
//! Every expectation of this file is the exact encoding of the
//! operation, byte for byte, in the order the bytes appear in memory.

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use codevar_ocl_native_codegen::mir::{MOp, MType, MirFunction, Operand, Operands};
use codevar_ocl_native_codegen::target::NativeTarget;
use codevar_ocl_native_codegen_x86::target;
use common::*;

/// The bits of one and a half, in single precision.
const ONE_AND_A_HALF: u64 = 0x3FC0_0000;

/// Builds a module whose only operation is `op` and returns the bytes
/// of the code.
fn bytes_of(ty: MType, op: MOp, args: Operands) -> Vec<u8> {
    let mut function = MirFunction::new("operation");
    let dest = vreg(&mut function, XMM0, ty.clone());
    let mut block = block("entry");
    block.insts.push(def(dest, ty, op, args));
    block
        .insts
        .push(none(MType::Void, MOp::RetVoid, Operands::none()));
    function.push_block(block);
    function.frame_size = 0;
    let (module, symbol) = module_with(function);
    let image = target()
        .lower_module(&module, symbol)
        .expect("the module compiles");
    image.bytes().to_vec()
}

/// The bytes of the operation, without the prologue and the epilogue.
fn body_of(ty: MType, op: MOp, args: Operands) -> Vec<u8> {
    let bytes = bytes_of(ty, op, args);
    // The prologue pushes one register and the epilogue pops it, so
    // the body lies between them.
    bytes[1..bytes.len() - 2].to_vec()
}

/// The bytes of a hexadecimal string.
fn hex(text: &str) -> Vec<u8> {
    let digits: Vec<char> = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let mut bytes = Vec::with_capacity(digits.len() / 2);
    let mut index = 0;
    while index < digits.len() {
        let high = digits[index]
            .to_digit(16)
            .expect("a hexadecimal digit") as u8;
        let low = digits[index + 1]
            .to_digit(16)
            .expect("a hexadecimal digit") as u8;
        bytes.push(high << 4 | low);
        index += 2;
    }
    bytes
}

#[test]
fn the_operations_have_their_own_bytes() {
    let cases: Vec<(MType, MOp, Operands, &str)> = vec![
        (
            MType::F32,
            MOp::FSqrt,
            Operands::one(reg(XMM1)),
            "F3 0F 10 C1 F3 0F 51 C0",
        ),
        (
            MType::F32,
            MOp::FAbs,
            Operands::one(reg(XMM1)),
            "48 C7 C0 FF FF FF 7F 66 4C 0F 6E F8 F3 0F 10 C1 41 0F 54 C7",
        ),
        (
            MType::F32,
            MOp::FMin,
            Operands::two(reg(XMM1), reg(XMM2)),
            "F3 0F 10 C1 F3 0F 5D C2",
        ),
        (
            MType::F32,
            MOp::FMax,
            Operands::two(reg(XMM1), reg(XMM2)),
            "F3 0F 10 C1 F3 0F 5F C2",
        ),
        (
            MType::F32,
            MOp::FDim,
            Operands::two(reg(XMM1), reg(XMM2)),
            "F3 0F 10 C1 F3 0F 5C C2 48 C7 C0 00 00 00 00 66 4C 0F 6E F0 F3 41 0F 5F C6",
        ),
        (
            MType::F32,
            MOp::FCopySign,
            Operands::two(reg(XMM1), reg(XMM2)),
            "F3 0F 10 C1 48 B8 00 00 00 80 00 00 00 00 66 4C 0F 6E F0 F3 44 0F 10 FA 45 0F 54 FE 44 0F 55 F0 41 0F 56 C7",
        ),
    ];
    for (ty, op, args, expected) in cases {
        let bytes = body_of(ty.clone(), op.clone(), args);
        assert_eq!(bytes, hex(expected), "unexpected bytes for {op:?}");
    }
}

#[test]
fn the_remainder_has_its_own_bytes() {
    let bytes = body_of(MType::F32, MOp::FRem, Operands::two(reg(XMM1), reg(XMM2)));
    assert_eq!(
        bytes,
        hex(
            "F3 44 0F 5A F9 66 4C 0F 7E F8 50 F3 44 0F 5A FA 66 4C 0F 7E F8 50 DD 04 24 DD 44 24 08 D9 F8 DD 1C 24 F3 44 0F 7E 34 24 F2 41 0F 5A C6 DD D8 48 83 C4 10"
        ),
        "unexpected bytes for the remainder"
    );
}

#[test]
fn the_square_root_of_a_double_has_its_own_bytes() {
    let bytes = body_of(MType::F64, MOp::FSqrt, Operands::one(reg(XMM1)));
    assert_eq!(
        bytes,
        hex("F2 0F 10 C1 F2 0F 51 C0"),
        "unexpected bytes for the square root"
    );
}

#[test]
fn a_float_immediate_is_moved_into_a_register() {
    let bytes = body_of(
        MType::F32,
        MOp::FSqrt,
        Operands::one(Operand::FloatImmediate(ONE_AND_A_HALF)),
    );
    assert_eq!(
        bytes,
        hex("48 C7 C0 00 00 C0 3F 66 4C 0F 6E F0 F3 41 0F 10 C6 F3 0F 51 C0"),
        "unexpected bytes for the square root"
    );
}
