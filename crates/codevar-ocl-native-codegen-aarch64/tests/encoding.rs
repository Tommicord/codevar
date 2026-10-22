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

//! The exact bytes of the instructions of the architecture.
//!
//! Every expectation of this file is the exact encoding of the
//! instruction, byte for byte, in the order the bytes appear in
//! memory.  The words are stored little-endian, so the byte that
//! holds the lowest part of the word comes first.

use codevar_ocl_native_codegen_aarch64::encode::{Cond, Direction, Encoder, Width};

/// Encodes one instruction and returns its bytes.
fn emit(encode: impl FnOnce(&mut Encoder<'_>)) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();
    {
        let mut encoder = Encoder::new(&mut bytes);
        encode(&mut encoder);
    }
    bytes
}

#[test]
fn the_control_instructions_have_their_own_bytes() {
    assert_eq!(emit(|encoder| encoder.ret()), vec![0xC0, 0x03, 0x5F, 0xD6]);
    assert_eq!(emit(|encoder| encoder.nop()), vec![0x1F, 0x20, 0x03, 0xD5]);
    assert_eq!(emit(|encoder| encoder.brk(0)), vec![0x00, 0x00, 0x20, 0xD4]);
    assert_eq!(emit(|encoder| encoder.brk(1)), vec![0x20, 0x00, 0x20, 0xD4]);
}

#[test]
fn the_branches_carry_their_displacement_in_words() {
    assert_eq!(emit(|encoder| encoder.b()), vec![0x00, 0x00, 0x00, 0x14]);
    assert_eq!(emit(|encoder| encoder.bl()), vec![0x00, 0x00, 0x00, 0x94]);
    assert_eq!(
        emit(|encoder| encoder.b_cond(Cond::Eq)),
        vec![0x00, 0x00, 0x00, 0x54]
    );
    assert_eq!(
        emit(|encoder| encoder.b_cond(Cond::Ne)),
        vec![0x01, 0x00, 0x00, 0x54]
    );
    assert_eq!(
        emit(|encoder| encoder.cbz(0, Width::Bits64)),
        vec![0x00, 0x00, 0x00, 0xB4]
    );
    assert_eq!(
        emit(|encoder| encoder.cbnz(0, Width::Bits64)),
        vec![0x00, 0x00, 0x00, 0xB5]
    );
    assert_eq!(
        emit(|encoder| encoder.cbz(5, Width::Bits32)),
        vec![0x05, 0x00, 0x00, 0x34]
    );
    assert_eq!(
        emit(|encoder| encoder.cbnz(5, Width::Bits32)),
        vec![0x05, 0x00, 0x00, 0x35]
    );
}

#[test]
fn the_data_processing_forms_have_their_own_bytes() {
    assert_eq!(
        emit(|encoder| encoder.add_reg(0, 0, 0, Width::Bits64)),
        vec![0x00, 0x00, 0x00, 0x8B]
    );
    assert_eq!(
        emit(|encoder| encoder.add_reg(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x00, 0x02, 0x8B]
    );
    assert_eq!(
        emit(|encoder| encoder.add_reg(0, 1, 2, Width::Bits32)),
        vec![0x20, 0x00, 0x02, 0x0B]
    );
    assert_eq!(
        emit(|encoder| encoder.sub_reg(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x00, 0x02, 0xCB]
    );
    assert_eq!(
        emit(|encoder| encoder.and_reg(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x00, 0x02, 0x8A]
    );
    assert_eq!(
        emit(|encoder| encoder.orr_reg(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x00, 0x02, 0xAA]
    );
    assert_eq!(
        emit(|encoder| encoder.eor_reg(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x00, 0x02, 0xCA]
    );
    assert_eq!(
        emit(|encoder| encoder.mul(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x7C, 0x02, 0x9B]
    );
    assert_eq!(
        emit(|encoder| encoder.mul(0, 1, 2, Width::Bits32)),
        vec![0x20, 0x7C, 0x02, 0x1B]
    );
    assert_eq!(
        emit(|encoder| encoder.msub(0, 1, 2, 3, Width::Bits64)),
        vec![0x20, 0x8C, 0x02, 0x9B]
    );
    assert_eq!(
        emit(|encoder| encoder.udiv(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x08, 0xC2, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.sdiv(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x0C, 0xC2, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.lsl_reg(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x20, 0xC2, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.lsr_reg(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x24, 0xC2, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.asr_reg(0, 1, 2, Width::Bits64)),
        vec![0x20, 0x28, 0xC2, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.neg_reg(0, 1, Width::Bits64)),
        vec![0xE0, 0x03, 0x01, 0xCB]
    );
    assert_eq!(
        emit(|encoder| encoder.mvn_reg(0, 1, Width::Bits64)),
        vec![0xE0, 0x03, 0x21, 0xAA]
    );
    assert_eq!(
        emit(|encoder| encoder.mov_reg(0, 1, Width::Bits64)),
        vec![0xE0, 0x03, 0x01, 0xAA]
    );
}

#[test]
fn the_immediate_forms_have_their_own_bytes() {
    assert_eq!(
        emit(|encoder| encoder.add_imm(0, 0, 1, Width::Bits64)),
        vec![0x00, 0x04, 0x00, 0x91]
    );
    assert_eq!(
        emit(|encoder| encoder.add_imm(0, 0, 1, Width::Bits32)),
        vec![0x00, 0x04, 0x00, 0x11]
    );
    assert_eq!(
        emit(|encoder| encoder.sub_imm(0, 0, 1, Width::Bits64)),
        vec![0x00, 0x04, 0x00, 0xD1]
    );
    assert_eq!(
        emit(|encoder| encoder.movz(0, 0, 0, Width::Bits64)),
        vec![0x00, 0x00, 0x80, 0xD2]
    );
    assert_eq!(
        emit(|encoder| encoder.movz(0, 0x1234, 0, Width::Bits64)),
        vec![0x80, 0x46, 0x82, 0xD2]
    );
    assert_eq!(
        emit(|encoder| encoder.movz(0, 0x1234, 1, Width::Bits64)),
        vec![0x80, 0x46, 0xA2, 0xD2]
    );
    assert_eq!(
        emit(|encoder| encoder.movk(0, 0x1234, 1, Width::Bits64)),
        vec![0x80, 0x46, 0xA2, 0xF2]
    );
    assert_eq!(
        emit(|encoder| encoder.movn(0, 0, 0, Width::Bits64)),
        vec![0x00, 0x00, 0x80, 0x92]
    );
    assert_eq!(
        emit(|encoder| encoder.cmp_imm(0, 1, Width::Bits64)),
        vec![0x1F, 0x04, 0x00, 0xF1]
    );
    assert_eq!(
        emit(|encoder| encoder.cmp_imm(0, 1, Width::Bits32)),
        vec![0x1F, 0x04, 0x00, 0x71]
    );
    assert_eq!(
        emit(|encoder| encoder.cmp_reg(0, 1, Width::Bits64)),
        vec![0x1F, 0x00, 0x01, 0xEB]
    );
}

#[test]
fn the_shifts_by_an_immediate_have_their_own_bytes() {
    assert_eq!(
        emit(|encoder| encoder.lsl_imm(0, 1, 3, Width::Bits64)),
        vec![0x20, 0xF0, 0x7D, 0xD3]
    );
    assert_eq!(
        emit(|encoder| encoder.lsr_imm(0, 1, 3, Width::Bits64)),
        vec![0x20, 0xFC, 0x43, 0xD3]
    );
    assert_eq!(
        emit(|encoder| encoder.asr_imm(0, 1, 3, Width::Bits64)),
        vec![0x20, 0xFC, 0x43, 0x93]
    );
    assert_eq!(
        emit(|encoder| encoder.lsl_imm(0, 1, 3, Width::Bits32)),
        vec![0x20, 0x70, 0x1D, 0x53]
    );
    assert_eq!(
        emit(|encoder| encoder.lsr_imm(0, 1, 3, Width::Bits32)),
        vec![0x20, 0x7C, 0x03, 0x53]
    );
    assert_eq!(
        emit(|encoder| encoder.asr_imm(0, 1, 3, Width::Bits32)),
        vec![0x20, 0x7C, 0x03, 0x13]
    );
}

#[test]
fn the_extensions_have_their_own_bytes() {
    assert_eq!(emit(|encoder| encoder.uxtw(0, 1)), vec![0x20, 0x7C, 0x40, 0xD3]);
    assert_eq!(emit(|encoder| encoder.sxtw(0, 1)), vec![0x20, 0x7C, 0x40, 0x93]);
    assert_eq!(emit(|encoder| encoder.uxtb(0, 1)), vec![0x20, 0x1C, 0x00, 0x53]);
    assert_eq!(emit(|encoder| encoder.uxth(0, 1)), vec![0x20, 0x3C, 0x00, 0x53]);
    assert_eq!(emit(|encoder| encoder.sxtb(0, 1)), vec![0x20, 0x1C, 0x40, 0x93]);
    assert_eq!(emit(|encoder| encoder.sxth(0, 1)), vec![0x20, 0x3C, 0x40, 0x93]);
    assert_eq!(emit(|encoder| encoder.sxtb32(0, 1)), vec![0x20, 0x1C, 0x00, 0x13]);
    assert_eq!(emit(|encoder| encoder.sxth32(0, 1)), vec![0x20, 0x3C, 0x00, 0x13]);
}

#[test]
fn the_conditional_selects_have_their_own_bytes() {
    assert_eq!(
        emit(|encoder| encoder.csel(0, 1, 2, Cond::Eq, Width::Bits64)),
        vec![0x20, 0x00, 0x82, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.csel(0, 1, 2, Cond::Gt, Width::Bits32)),
        vec![0x20, 0xC0, 0x82, 0x1A]
    );
    // The conditional set is the conditional increment of the zero
    // register, so the word carries the opposite condition.
    assert_eq!(
        emit(|encoder| encoder.cset(0, Cond::Eq, Width::Bits64)),
        vec![0xE0, 0x17, 0x9F, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.cset(0, Cond::Ne, Width::Bits32)),
        vec![0xE0, 0x07, 0x9F, 0x1A]
    );
}

#[test]
fn the_memory_accesses_have_their_own_bytes() {
    assert_eq!(
        emit(|encoder| encoder
            .load(0, 1, 8, 8)
            .expect("the access has an encoding")),
        vec![0x20, 0x04, 0x40, 0xF9]
    );
    assert_eq!(
        emit(|encoder| encoder
            .load(0, 1, 8, 4)
            .expect("the access has an encoding")),
        vec![0x20, 0x08, 0x40, 0xB9]
    );
    assert_eq!(
        emit(|encoder| encoder
            .store(0, 1, 8, 8)
            .expect("the access has an encoding")),
        vec![0x20, 0x04, 0x00, 0xF9]
    );
    assert_eq!(
        emit(|encoder| encoder
            .store(0, 1, 4, 4)
            .expect("the access has an encoding")),
        vec![0x20, 0x04, 0x00, 0xB9]
    );
    assert_eq!(
        emit(|encoder| encoder
            .load(0, 1, 4, 1)
            .expect("the access has an encoding")),
        vec![0x20, 0x10, 0x40, 0x39]
    );
    assert_eq!(
        emit(|encoder| encoder
            .load(0, 1, 4, 2)
            .expect("the access has an encoding")),
        vec![0x20, 0x08, 0x40, 0x79]
    );
    // The unscaled form reaches the offsets the scaled form cannot.
    assert_eq!(
        emit(|encoder| encoder
            .load(0, 1, 3, 8)
            .expect("the access has an encoding")),
        vec![0x20, 0x30, 0x40, 0xF8]
    );
    assert_eq!(
        emit(|encoder| encoder
            .store(0, 1, -8, 8)
            .expect("the access has an encoding")),
        vec![0x20, 0x80, 0x1F, 0xF8]
    );
}

#[test]
fn the_pair_accesses_have_their_own_bytes() {
    assert_eq!(
        emit(|encoder| encoder
            .pair(Direction::Load, 19, 20, 31, 16, 8)
            .expect("the access has an encoding")),
        vec![0xF3, 0x53, 0x41, 0xA9]
    );
    assert_eq!(
        emit(|encoder| encoder
            .pair(Direction::Store, 19, 20, 31, 16, 8)
            .expect("the access has an encoding")),
        vec![0xF3, 0x53, 0x01, 0xA9]
    );
    assert_eq!(
        emit(|encoder| encoder
            .pair(Direction::Store, 19, 20, 31, -16, 8)
            .expect("the access has an encoding")),
        vec![0xF3, 0x53, 0x3F, 0xA9]
    );
}

#[test]
fn the_floating_point_forms_have_their_own_bytes() {
    assert_eq!(
        emit(|encoder| encoder.fadd(0, 1, 2, false)),
        vec![0x20, 0x28, 0x62, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fadd(0, 1, 2, true)),
        vec![0x20, 0x28, 0x22, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fsub(0, 1, 2, false)),
        vec![0x20, 0x38, 0x62, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmul(0, 1, 2, false)),
        vec![0x20, 0x08, 0x62, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fdiv(0, 1, 2, false)),
        vec![0x20, 0x18, 0x62, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmax(0, 1, 2, false)),
        vec![0x20, 0x48, 0x62, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmin(0, 1, 2, false)),
        vec![0x20, 0x58, 0x62, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fabs(0, 1, false)),
        vec![0x20, 0xC0, 0x60, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fneg(0, 1, false)),
        vec![0x20, 0x40, 0x61, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmov_reg(0, 1, false)),
        vec![0x20, 0x40, 0x60, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcmp(0, 1, false)),
        vec![0x00, 0x20, 0x61, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcsel(0, 1, 2, Cond::Eq, false)),
        vec![0x20, 0x0C, 0x62, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.scvtf(0, 1, false)),
        vec![0x20, 0x00, 0x62, 0x9E]
    );
    assert_eq!(
        emit(|encoder| encoder.ucvtf(0, 1, false)),
        vec![0x20, 0x00, 0x63, 0x9E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcvtzs(0, 1, true, false)),
        vec![0x20, 0x00, 0x38, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcvtzu(0, 1, false, true)),
        vec![0x20, 0x00, 0x79, 0x9E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcvt_narrow(0, 1)),
        vec![0x20, 0x40, 0x62, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcvt_widen(0, 1)),
        vec![0x20, 0xC0, 0x22, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmov_from_gp(0, 1, true)),
        vec![0x20, 0x00, 0x27, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmov_to_gp(0, 1, true)),
        vec![0x20, 0x00, 0x26, 0x1E]
    );
}

#[test]
fn the_vector_forms_have_their_own_bytes() {
    assert_eq!(emit(|encoder| encoder.dup_4s(0, 1)), vec![0x20, 0x0C, 0x04, 0x4E]);
    assert_eq!(emit(|encoder| encoder.dup_2d(0, 1)), vec![0x20, 0x0C, 0x08, 0x4E]);
    assert_eq!(
        emit(|encoder| encoder.umov_32(0, 1, 2)),
        vec![0x20, 0x3C, 0x14, 0x0E]
    );
    assert_eq!(
        emit(|encoder| encoder.umov_64(0, 1, 1)),
        vec![0x20, 0x3C, 0x18, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.ins_32(0, 1, 1)),
        vec![0x20, 0x1C, 0x0C, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.ins_64(0, 1, 1)),
        vec![0x20, 0x1C, 0x18, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.ins_32_lane(0, 1, 0, 1)),
        vec![0x20, 0x24, 0x04, 0x6E]
    );
    assert_eq!(
        emit(|encoder| encoder.ins_64_lane(0, 1, 1, 0)),
        vec![0x20, 0x04, 0x18, 0x6E]
    );
    assert_eq!(
        emit(|encoder| encoder.orr_vec(0, 1, 2)),
        vec![0x20, 0x1C, 0xA2, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.mov_vec(0, 1)),
        vec![0x20, 0x1C, 0xA1, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.add_vec_32(0, 1, 2)),
        vec![0x20, 0x84, 0xA2, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.add_vec_64(0, 1, 2)),
        vec![0x20, 0x84, 0xE2, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.sub_vec_32(0, 1, 2)),
        vec![0x20, 0x84, 0xA2, 0x6E]
    );
    assert_eq!(
        emit(|encoder| encoder.mul_vec_32(0, 1, 2)),
        vec![0x20, 0x9C, 0xA2, 0x4E]
    );
}

/// The registers of an instruction reach the fields of its word.
///
/// The words above name the registers zero, one and two; these name
/// the registers three, four and five, so that a word that would
/// hide a field behind another one shows up here.
#[test]
fn the_registers_reach_their_own_fields() {
    assert_eq!(
        emit(|encoder| encoder.add_reg(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x00, 0x05, 0x8B]
    );
    assert_eq!(
        emit(|encoder| encoder.sub_reg(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x00, 0x05, 0xCB]
    );
    assert_eq!(
        emit(|encoder| encoder.and_reg(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x00, 0x05, 0x8A]
    );
    assert_eq!(
        emit(|encoder| encoder.orr_reg(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x00, 0x05, 0xAA]
    );
    assert_eq!(
        emit(|encoder| encoder.eor_reg(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x00, 0x05, 0xCA]
    );
    assert_eq!(
        emit(|encoder| encoder.mul(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x7C, 0x05, 0x9B]
    );
    assert_eq!(
        emit(|encoder| encoder.msub(3, 4, 5, 6, Width::Bits64)),
        vec![0x83, 0x98, 0x05, 0x9B]
    );
    assert_eq!(
        emit(|encoder| encoder.udiv(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x08, 0xC5, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.sdiv(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x0C, 0xC5, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.lsl_reg(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x20, 0xC5, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.lsr_reg(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x24, 0xC5, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.asr_reg(3, 4, 5, Width::Bits64)),
        vec![0x83, 0x28, 0xC5, 0x9A]
    );
    assert_eq!(
        emit(|encoder| encoder.neg_reg(3, 4, Width::Bits64)),
        vec![0xE3, 0x03, 0x04, 0xCB]
    );
    assert_eq!(
        emit(|encoder| encoder.mvn_reg(3, 4, Width::Bits64)),
        vec![0xE3, 0x03, 0x24, 0xAA]
    );
    assert_eq!(
        emit(|encoder| encoder.mov_reg(3, 4, Width::Bits64)),
        vec![0xE3, 0x03, 0x04, 0xAA]
    );
    assert_eq!(
        emit(|encoder| encoder.cmp_reg(3, 4, Width::Bits64)),
        vec![0x7F, 0x00, 0x04, 0xEB]
    );
    assert_eq!(
        emit(|encoder| encoder.csel(3, 4, 5, Cond::Ne, Width::Bits64)),
        vec![0x83, 0x10, 0x85, 0x9A]
    );
}

#[test]
fn the_floating_point_registers_reach_their_own_fields() {
    assert_eq!(
        emit(|encoder| encoder.fadd(3, 4, 5, false)),
        vec![0x83, 0x28, 0x65, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fadd(3, 4, 5, true)),
        vec![0x83, 0x28, 0x25, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fsub(3, 4, 5, false)),
        vec![0x83, 0x38, 0x65, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmul(3, 4, 5, false)),
        vec![0x83, 0x08, 0x65, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fdiv(3, 4, 5, false)),
        vec![0x83, 0x18, 0x65, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmax(3, 4, 5, false)),
        vec![0x83, 0x48, 0x65, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmin(3, 4, 5, false)),
        vec![0x83, 0x58, 0x65, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmov_reg(3, 4, false)),
        vec![0x83, 0x40, 0x60, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fabs(3, 4, false)),
        vec![0x83, 0xC0, 0x60, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fneg(3, 4, false)),
        vec![0x83, 0x40, 0x61, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fsqrt(3, 4, false)),
        vec![0x83, 0xC0, 0x61, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fsqrt(3, 4, true)),
        vec![0x83, 0xC0, 0x21, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmls(3, 4, 5, false)),
        vec![0x83, 0x30, 0x65, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmls(3, 4, 5, true)),
        vec![0x83, 0x30, 0x25, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcmp(3, 4, false)),
        vec![0x60, 0x20, 0x64, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcsel(3, 4, 5, Cond::Eq, false)),
        vec![0x83, 0x0C, 0x65, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcsel(3, 4, 5, Cond::Ne, true)),
        vec![0x83, 0x1C, 0x25, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.scvtf(3, 4, false)),
        vec![0x83, 0x00, 0x62, 0x9E]
    );
    assert_eq!(
        emit(|encoder| encoder.ucvtf(3, 4, false)),
        vec![0x83, 0x00, 0x63, 0x9E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcvtzs(3, 4, false, true)),
        vec![0x83, 0x00, 0x78, 0x9E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcvtzu(3, 4, false, true)),
        vec![0x83, 0x00, 0x79, 0x9E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcvt_narrow(3, 4)),
        vec![0x83, 0x40, 0x62, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fcvt_widen(3, 4)),
        vec![0x83, 0xC0, 0x22, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmov_from_gp(3, 4, true)),
        vec![0x83, 0x00, 0x27, 0x1E]
    );
    assert_eq!(
        emit(|encoder| encoder.fmov_to_gp(3, 4, true)),
        vec![0x83, 0x00, 0x26, 0x1E]
    );
}

#[test]
fn the_vector_registers_reach_their_own_fields() {
    assert_eq!(emit(|encoder| encoder.dup_4s(3, 4)), vec![0x83, 0x0C, 0x04, 0x4E]);
    assert_eq!(emit(|encoder| encoder.dup_2d(3, 4)), vec![0x83, 0x0C, 0x08, 0x4E]);
    assert_eq!(
        emit(|encoder| encoder.umov_32(3, 4, 0)),
        vec![0x83, 0x3C, 0x04, 0x0E]
    );
    assert_eq!(
        emit(|encoder| encoder.umov_64(3, 4, 0)),
        vec![0x83, 0x3C, 0x08, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.ins_32(3, 4, 0)),
        vec![0x83, 0x1C, 0x04, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.ins_64(3, 4, 0)),
        vec![0x83, 0x1C, 0x08, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.ins_32_lane(3, 4, 0, 1)),
        vec![0x83, 0x24, 0x04, 0x6E]
    );
    assert_eq!(
        emit(|encoder| encoder.orr_vec(3, 4, 5)),
        vec![0x83, 0x1C, 0xA5, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.and_vec(3, 4, 5)),
        vec![0x83, 0x1C, 0x25, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.eor_vec(3, 4, 5)),
        vec![0x83, 0x1C, 0x25, 0x6E]
    );
    assert_eq!(
        emit(|encoder| encoder.mov_vec(3, 4)),
        vec![0x83, 0x1C, 0xA4, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.add_vec_32(3, 4, 5)),
        vec![0x83, 0x84, 0xA5, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.add_vec_64(3, 4, 5)),
        vec![0x83, 0x84, 0xE5, 0x4E]
    );
    assert_eq!(
        emit(|encoder| encoder.sub_vec_32(3, 4, 5)),
        vec![0x83, 0x84, 0xA5, 0x6E]
    );
    assert_eq!(
        emit(|encoder| encoder.sub_vec_64(3, 4, 5)),
        vec![0x83, 0x84, 0xE5, 0x6E]
    );
    assert_eq!(
        emit(|encoder| encoder.mul_vec_32(3, 4, 5)),
        vec![0x83, 0x9C, 0xA5, 0x4E]
    );
}
