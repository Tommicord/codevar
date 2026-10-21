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

//! The encodings of the instructions, byte for byte.
//!
//! Every expectation here is the encoding the architecture defines,
//! so a change in the encoder that breaks one of them is a change in
//! the code the machine runs.

use codevar_ocl_native_codegen_thumbv7m::encode;

/// The bytes of an encoding.
fn bytes_of(f: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut out = Vec::new();
    f(&mut out);
    out
}

#[test]
fn the_moves_match_the_architecture() {
    assert_eq!(bytes_of(|o| encode::movs_imm(o, 0, 1)), [0x01, 0x20]);
    assert_eq!(bytes_of(|o| encode::movs_imm(o, 7, 0xff)), [0xff, 0x27]);
    assert_eq!(bytes_of(|o| encode::mov_reg(o, 0, 1)), [0x08, 0x46]);
    assert_eq!(bytes_of(|o| encode::mov_reg(o, 1, 0)), [0x01, 0x46]);
    assert_eq!(
        bytes_of(|o| encode::mov_reg_wide(o, 4, 5)),
        [0x05, 0x04, 0x4f, 0xea]
    );
    assert_eq!(
        bytes_of(|o| encode::movw(o, 0, 0x1234)),
        [0x34, 0x20, 0x41, 0xf2]
    );
    assert_eq!(
        bytes_of(|o| encode::movt(o, 0, 0x5678)),
        [0x78, 0x60, 0xc5, 0xf2]
    );
}

#[test]
fn the_additions_match_the_architecture() {
    assert_eq!(bytes_of(|o| encode::adds_reg(o, 0, 1)), [0x08, 0x88]);
    assert_eq!(bytes_of(|o| encode::subs_reg(o, 0, 1)), [0x08, 0x1a]);
    assert_eq!(bytes_of(|o| encode::adds_imm3(o, 0, 1, 7)), [0xc8, 0x1d]);
    assert_eq!(bytes_of(|o| encode::adds_imm8(o, 0, 255)), [0xff, 0x30]);
    assert_eq!(bytes_of(|o| encode::subs_imm3(o, 0, 1, 7)), [0xc8, 0x1f]);
    assert_eq!(bytes_of(|o| encode::subs_imm8(o, 0, 255)), [0xff, 0x38]);
    assert_eq!(
        bytes_of(|o| encode::adds_reg_wide(o, 4, 5, 6)),
        [0x06, 0x04, 0x15, 0xeb]
    );
}

#[test]
fn the_logical_operations_match_the_architecture() {
    assert_eq!(bytes_of(|o| encode::ands_reg(o, 0, 1)), [0x08, 0x40]);
    assert_eq!(bytes_of(|o| encode::orrs_reg(o, 0, 1)), [0x08, 0x43]);
    assert_eq!(bytes_of(|o| encode::eors_reg(o, 0, 1)), [0x48, 0x40]);
    assert_eq!(bytes_of(|o| encode::mvns_reg(o, 0, 1)), [0xc8, 0x43]);
    assert_eq!(bytes_of(|o| encode::negs_reg(o, 0, 1)), [0x48, 0x42]);
    assert_eq!(bytes_of(|o| encode::cmp_reg(o, 0, 1)), [0x88, 0x42]);
    assert_eq!(bytes_of(|o| encode::cmp_imm(o, 0, 255)), [0xff, 0x28]);
    assert_eq!(bytes_of(|o| encode::adcs_reg(o, 4, 4)), [0xac, 0x41]);
}

#[test]
fn the_extensions_match_the_architecture() {
    assert_eq!(bytes_of(|o| encode::uxtb_reg(o, 0, 1)), [0xc8, 0xb2]);
    assert_eq!(bytes_of(|o| encode::uxth_reg(o, 0, 1)), [0x88, 0xb2]);
    assert_eq!(bytes_of(|o| encode::sxtb_reg(o, 0, 1)), [0x48, 0xb2]);
    assert_eq!(bytes_of(|o| encode::sxth_reg(o, 0, 1)), [0x08, 0xb2]);
}

#[test]
fn the_shifts_match_the_architecture() {
    assert_eq!(bytes_of(|o| encode::lsls_imm(o, 0, 1, 1)), [0x48, 0x00]);
    assert_eq!(bytes_of(|o| encode::lsrs_imm(o, 0, 1, 1)), [0x48, 0x08]);
    assert_eq!(bytes_of(|o| encode::asrs_imm(o, 0, 1, 1)), [0x48, 0x10]);
    assert_eq!(
        bytes_of(|o| encode::lsls_reg(o, 0, 1, 2)),
        [0x02, 0xf0, 0x01, 0xfa]
    );
    assert_eq!(
        bytes_of(|o| encode::lsrs_reg(o, 0, 1, 2)),
        [0x02, 0xf0, 0x21, 0xfa]
    );
    assert_eq!(
        bytes_of(|o| encode::asrs_reg(o, 0, 1, 2)),
        [0x02, 0xf0, 0x41, 0xfa]
    );
}

#[test]
fn the_multiplication_matches_the_architecture() {
    assert_eq!(
        bytes_of(|o| encode::muls_reg_wide(o, 4, 5, 6)),
        [0x06, 0xf4, 0x05, 0xfb]
    );
}

#[test]
fn the_loads_and_stores_match_the_architecture() {
    assert_eq!(bytes_of(|o| encode::ldr_imm(o, 0, 1, 1)), [0x48, 0x68]);
    assert_eq!(bytes_of(|o| encode::str_imm(o, 0, 1, 1)), [0x48, 0x60]);
    assert_eq!(bytes_of(|o| encode::ldrb_imm(o, 0, 1, 5)), [0x48, 0x79]);
    assert_of(&[0x48, 0x71], |o| encode::strb_imm(o, 0, 1, 5));
    assert_eq!(bytes_of(|o| encode::ldrh_imm(o, 0, 1, 5)), [0x48, 0x89]);
    assert_eq!(bytes_of(|o| encode::strh_imm(o, 0, 1, 5)), [0x48, 0x81]);
    assert_eq!(bytes_of(|o| encode::ldr_reg(o, 0, 1, 2)), [0x88, 0x58]);
    assert_eq!(bytes_of(|o| encode::str_reg(o, 0, 1, 2)), [0x88, 0x50]);
    assert_eq!(bytes_of(|o| encode::ldr_sp(o, 0, 1)), [0x01, 0x98]);
    assert_eq!(bytes_of(|o| encode::str_sp(o, 0, 2)), [0x02, 0x90]);
    assert_eq!(
        bytes_of(|o| encode::ldr_imm_wide(o, 0, 1, 240)),
        [0xf0, 0x00, 0xd1, 0xf8]
    );
}

#[test]
fn the_register_lists_match_the_architecture() {
    assert_eq!(bytes_of(|o| encode::push(o, 0x10, true)), [0x10, 0xb5]);
    assert_eq!(bytes_of(|o| encode::push(o, 0x10, false)), [0x10, 0xb4]);
    assert_eq!(bytes_of(|o| encode::pop(o, 0x10, true)), [0x10, 0xbd]);
    assert_eq!(bytes_of(|o| encode::pop(o, 0x10, false)), [0x10, 0xbc]);
    assert_eq!(bytes_of(|o| encode::push(o, 0, true)), [0x00, 0xb5]);
    assert_eq!(bytes_of(|o| encode::pop(o, 0, true)), [0x00, 0xbd]);
    assert_eq!(
        bytes_of(|o| encode::push(o, 0xff0, true)),
        [0xf0, 0xf4, 0x2d, 0xe9]
    );
}

#[test]
fn the_control_instructions_match_the_architecture() {
    assert_eq!(bytes_of(|o| encode::bx(o, 14)), [0x70, 0x47]);
    assert_eq!(bytes_of(|o| encode::undefined(o)), [0x00, 0xde]);
    assert_eq!(bytes_of(|o| encode::it(o, 0, encode::IT_ONE)), [0x08, 0xbf]);
    assert_eq!(
        bytes_of(|o| encode::it(o, 0, encode::IT_THEN_ELSE)),
        [0x0c, 0xbf]
    );
}

#[test]
fn the_branches_carry_their_displacement() {
    // A branch whose target is four bytes further reads the
    // displacement as one halfword.
    let bytes = encode::branch_bytes(4);
    assert_eq!(bytes[0] & 0x0f, 0x02);
    assert_eq!(bytes[3] & 0x07, 0x02);
    // A branch backwards keeps the sign of the displacement.
    let bytes = encode::branch_bytes(-4);
    assert_eq!(bytes[0] & 0x0f, 0x02);
    assert_eq!(bytes[3] & 0x07, 0x02);
    // A call has its own shape.
    let bytes = encode::branch_link_bytes(4);
    assert_eq!(bytes[2] & 0xf0, 0xf0);
    // A conditional branch names its condition in the first halfword.
    let bytes = encode::branch_cond_bytes(encode::COND_EQ, 4);
    assert_eq!((bytes[1] >> 4) & 0x0f, 0x0e);
}

/// Compares the bytes of an encoding with the expected ones.
fn assert_of(expected: &[u8], f: impl FnOnce(&mut Vec<u8>)) {
    assert_eq!(bytes_of(f), expected);
}
