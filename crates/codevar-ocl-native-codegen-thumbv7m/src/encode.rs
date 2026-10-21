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

//! The byte-level encoder of the instruction set.
//!
//! Every function of this module writes the encoding of one
//! instruction into a buffer the caller owns.  The encodings are
//! the ones the architecture defines; the module never assembles
//! anything but the fixed bit patterns of the instruction set, and
//! it never looks at anything but the registers and the immediates
//! it is given.
//!
//! # The two widths
//!
//! The instruction set has a sixteen-bit encoding and a thirty-two
//! bit encoding.  The sixteen-bit one covers the common operations
//! on the low registers; the thirty-two bit one covers everything
//! else, in particular the operations that name a high register, an
//! operation on a shifted operand, or an immediate that does not fit
//! in eight bits.
//!
//! # The branch forms
//!
//! The branches carry their displacement in two halfwords, so they
//! are emitted as a placeholder and patched once the layout is
//! known.  The functions [`branch_bytes`], [`branch_link_bytes`] and
//! [`branch_cond_bytes`] compute the two halfwords of a branch with
//! a known displacement.

use alloc::vec::Vec;

/// The condition code of the "equal" comparison.
pub const COND_EQ: u8 = 0x0;

/// The condition code of the "not equal" comparison.
pub const COND_NE: u8 = 0x1;

/// The condition code of the "carry set" comparison.
pub const COND_CS: u8 = 0x2;

/// The condition code of the "carry clear" comparison.
pub const COND_CC: u8 = 0x3;

/// The condition code of the "minus" comparison.
pub const COND_MI: u8 = 0x4;

/// The condition code of the "plus" comparison.
pub const COND_PL: u8 = 0x5;

/// The condition code of the "overflow" comparison.
pub const COND_VS: u8 = 0x6;

/// The condition code of the "no overflow" comparison.
pub const COND_VC: u8 = 0x7;

/// The condition code of the "unsigned higher" comparison.
pub const COND_HI: u8 = 0x8;

/// The condition code of the "unsigned lower or same" comparison.
pub const COND_LS: u8 = 0x9;

/// The condition code of the "signed greater or equal" comparison.
pub const COND_GE: u8 = 0xa;

/// The condition code of the "signed less than" comparison.
pub const COND_LT: u8 = 0xb;

/// The condition code of the "signed greater than" comparison.
pub const COND_GT: u8 = 0xc;

/// The condition code of the "signed less or equal" comparison.
pub const COND_LE: u8 = 0xd;

/// The mask of a conditional execution block with one instruction.
pub const IT_ONE: u8 = 0x8;

/// The mask of a conditional execution block with a "then" and an
/// "else" part.
pub const IT_THEN_ELSE: u8 = 0xc;

/// The shift kinds of the thirty-two bit data-processing forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shift {
    /// Shift left.
    Left,
    /// Shift right, filling with zeros.
    Right,
    /// Shift right, filling with the sign.
    ArithmeticRight,
}

impl Shift {
    /// The encoding of the shift kind.
    #[must_use]
    #[inline]
    const fn code(self) -> u16 {
        match self {
            Self::Left => 0,
            Self::Right => 1,
            Self::ArithmeticRight => 2,
        }
    }
}

/// Writes a halfword into the buffer.
#[inline]
fn halfword(out: &mut Vec<u8>, value: u16) {
    out.push(u8::try_from(value & 0xff).unwrap_or(0));
    out.push(u8::try_from((value >> 8) & 0xff).unwrap_or(0));
}

/// Writes a word into the buffer.
///
/// The wide instructions keep their first halfword first, so the
/// high half of the value comes first in the buffer.
#[inline]
fn word(out: &mut Vec<u8>, value: u32) {
    out.push(u8::try_from((value >> 16) & 0xff).unwrap_or(0));
    out.push(u8::try_from((value >> 24) & 0xff).unwrap_or(0));
    out.push(u8::try_from(value & 0xff).unwrap_or(0));
    out.push(u8::try_from((value >> 8) & 0xff).unwrap_or(0));
}

/// `movs rd, #imm8`: move an eight-bit immediate into a register.
#[inline]
pub fn movs_imm(out: &mut Vec<u8>, rd: u8, imm: u8) {
    halfword(out, 0x2000 | u16::from(rd) << 8 | u16::from(imm));
}

/// `movw rd, #imm16`: move the low half of a sixteen-bit immediate
/// into a register.
#[inline]
pub fn movw(out: &mut Vec<u8>, rd: u8, imm: u16) {
    word(
        out,
        (0xf240 | u32::from(imm >> 12)) << 16
            | (u32::from((imm >> 8) & 0x7) << 12)
            | (u32::from(rd) << 8)
            | u32::from(imm & 0xff),
    );
}

/// `movt rd, #imm16`: move the high half of a sixteen-bit immediate
/// into a register.
#[inline]
pub fn movt(out: &mut Vec<u8>, rd: u8, imm: u16) {
    word(
        out,
        (0xf2c0 | u32::from(imm >> 12)) << 16
            | (u32::from((imm >> 8) & 0x7) << 12)
            | (u32::from(rd) << 8)
            | u32::from(imm & 0xff),
    );
}

/// `adds rd, rn, #imm3`: add a small immediate to a register.
#[inline]
pub fn adds_imm3(out: &mut Vec<u8>, rd: u8, rn: u8, imm: u8) {
    halfword(
        out,
        0x1c00 | u16::from(rn) << 6 | u16::from(rd) << 3 | u16::from(imm & 0x7),
    );
}

/// `adds rd, #imm8`: add an eight-bit immediate to a register.
#[inline]
pub fn adds_imm8(out: &mut Vec<u8>, rd: u8, imm: u8) {
    halfword(out, 0x3000 | u16::from(rd) << 8 | u16::from(imm));
}

/// `subs rd, rn, #imm3`: subtract a small immediate from a register.
#[inline]
pub fn subs_imm3(out: &mut Vec<u8>, rd: u8, rn: u8, imm: u8) {
    halfword(
        out,
        0x1e00 | u16::from(rn) << 6 | u16::from(rd) << 3 | u16::from(imm & 0x7),
    );
}

/// `subs rd, #imm8`: subtract an eight-bit immediate from a register.
#[inline]
pub fn subs_imm8(out: &mut Vec<u8>, rd: u8, imm: u8) {
    halfword(out, 0x3800 | u16::from(rd) << 8 | u16::from(imm));
}

/// `adds rd, rn, rm`: add two registers, in the sixteen-bit form.
///
/// The sixteen-bit form names three registers, so it only covers the
/// low ones.
#[inline]
pub fn adds_reg3(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    halfword(
        out,
        0x1800 | u16::from(rm) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `subs rd, rn, rm`: subtract two registers, in the sixteen-bit form.
#[inline]
pub fn subs_reg3(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    halfword(
        out,
        0x1a00 | u16::from(rm) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `ands rd, rm`: bitwise and of two registers.
///
/// The sixteen-bit form names the destination and the source only, so
/// it covers the low registers.
#[inline]
pub fn ands_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0x4000 | u16::from(rm) << 3 | u16::from(rd));
}

/// `orrs rd, rm`: bitwise or of two registers.
#[inline]
pub fn orrs_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0x4300 | u16::from(rm) << 3 | u16::from(rd));
}

/// `eors rd, rm`: bitwise exclusive or of two registers.
#[inline]
pub fn eors_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0x4040 | u16::from(rm) << 3 | u16::from(rd));
}

/// `mvns rd, rm`: the complement of a register.
#[inline]
pub fn mvns_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0x43c0 | u16::from(rm) << 3 | u16::from(rd));
}

/// `cmp rn, rm`: compare two registers.
#[inline]
pub fn cmp_reg(out: &mut Vec<u8>, rn: u8, rm: u8) {
    halfword(out, 0x4280 | u16::from(rm) << 3 | u16::from(rn));
}

/// `cmp rn, #imm8`: compare a register with an eight-bit immediate.
#[inline]
pub fn cmp_imm(out: &mut Vec<u8>, rn: u8, imm: u8) {
    halfword(out, 0x2800 | u16::from(rn) << 8 | u16::from(imm));
}

/// `negs rd, rm`: the negation of a register.
#[inline]
pub fn negs_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0x4240 | u16::from(rm) << 3 | u16::from(rd));
}

/// `mov rd, rm`: move a register.
///
/// The sixteen-bit form covers a low destination; the wide form
/// covers every register.
#[inline]
pub fn mov_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0x4600 | u16::from(rm) << 3 | u16::from(rd));
}

/// `mov.w rd, rm`: move a register, in the wide form.
#[inline]
pub fn mov_reg_wide(out: &mut Vec<u8>, rd: u8, rm: u8) {
    word(
        out,
        0xea4f_0000 | u32::from(rd) << 8 | u32::from(rm),
    );
}

/// `mvn.w rd, rm`: the complement of a register, in the wide form.
#[inline]
pub fn mvn_reg_wide(out: &mut Vec<u8>, rd: u8, rm: u8) {
    word(
        out,
        0xea6f_0000 | u32::from(rd) << 8 | u32::from(rm),
    );
}

/// `uxtb rd, rm`: the zero-extended low byte of a register.
#[inline]
pub fn uxtb_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0xb2c0 | u16::from(rm) << 3 | u16::from(rd));
}

/// `uxth rd, rm`: the zero-extended low halfword of a register.
#[inline]
pub fn uxth_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0xb288 | u16::from(rm) << 3 | u16::from(rd));
}

/// `sxtb rd, rm`: the sign-extended low byte of a register.
#[inline]
pub fn sxtb_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0xb248 | u16::from(rm) << 3 | u16::from(rd));
}

/// `sxth rd, rm`: the sign-extended low halfword of a register.
#[inline]
pub fn sxth_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0xb208 | u16::from(rm) << 3 | u16::from(rd));
}

/// `lsls rd, rm, #imm5`: shift a register left by a constant.
#[inline]
pub fn lsls_imm(out: &mut Vec<u8>, rd: u8, rm: u8, imm: u8) {
    halfword(
        out,
        u16::from(imm & 0x1f) << 6 | u16::from(rm) << 3 | u16::from(rd),
    );
}

/// `lsrs rd, rm, #imm5`: shift a register right by a constant.
#[inline]
pub fn lsrs_imm(out: &mut Vec<u8>, rd: u8, rm: u8, imm: u8) {
    halfword(
        out,
        0x0800 | u16::from(imm & 0x1f) << 6 | u16::from(rm) << 3 | u16::from(rd),
    );
}

/// `asrs rd, rm, #imm5`: shift a register right, filling with the
/// sign, by a constant.
#[inline]
pub fn asrs_imm(out: &mut Vec<u8>, rd: u8, rm: u8, imm: u8) {
    halfword(
        out,
        0x1000 | u16::from(imm & 0x1f) << 6 | u16::from(rm) << 3 | u16::from(rd),
    );
}

/// `lsl.w rd, rm, rs`: shift a register left by the amount in a
/// register.
#[inline]
pub fn lsls_reg(out: &mut Vec<u8>, rd: u8, rm: u8, rs: u8) {
    word(
        out,
        0xfa00_0000 | u32::from(rs) | 0xf000 | u32::from(rd) << 8 | u32::from(rm),
    );
}

/// `lsr.w rd, rm, rs`: shift a register right by the amount in a
/// register.
#[inline]
pub fn lsrs_reg(out: &mut Vec<u8>, rd: u8, rm: u8, rs: u8) {
    word(
        out,
        0xfa20_0000 | u32::from(rs) | 0xf000 | u32::from(rd) << 8 | u32::from(rm),
    );
}

/// `asr.w rd, rm, rs`: shift a register right, filling with the sign,
/// by the amount in a register.
#[inline]
pub fn asrs_reg(out: &mut Vec<u8>, rd: u8, rm: u8, rs: u8) {
    word(
        out,
        0xfa40_0000 | u32::from(rs) | 0xf000 | u32::from(rd) << 8 | u32::from(rm),
    );
}

/// `add.w rd, rn, rm`: add two registers, in the wide form.
#[inline]
pub fn adds_reg_wide(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    data_processing_wide(out, 0xeb00, rd, rn, rm, 0, Shift::Left);
}

/// `sub.w rd, rn, rm`: subtract two registers, in the wide form.
#[inline]
pub fn subs_reg_wide(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    data_processing_wide(out, 0xeba0, rd, rn, rm, 0, Shift::Left);
}

/// `and.w rd, rn, rm`: bitwise and of two registers, in the wide
/// form.
#[inline]
pub fn ands_reg_wide(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    data_processing_wide(out, 0xea00, rd, rn, rm, 0, Shift::Left);
}

/// `orr.w rd, rn, rm`: bitwise or of two registers, in the wide form.
#[inline]
pub fn orrs_reg_wide(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    data_processing_wide(out, 0xea40, rd, rn, rm, 0, Shift::Left);
}

/// `eor.w rd, rn, rm`: bitwise exclusive or of two registers, in the
/// wide form.
#[inline]
pub fn eors_reg_wide(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    data_processing_wide(out, 0xea80, rd, rn, rm, 0, Shift::Left);
}

/// Emits the wide data-processing form with a shifted operand.
fn data_processing_wide(
    out: &mut Vec<u8>,
    base: u16,
    rd: u8,
    rn: u8,
    rm: u8,
    shift: u8,
    kind: Shift,
) {
    let imm3 = u32::from((shift >> 2) & 0x7);
    let imm2 = u32::from(shift & 0x3);
    word(
        out,
        (u32::from(base) | u32::from(rn)) << 16
            | (imm3 << 12)
            | (u32::from(rd) << 8)
            | (imm2 << 6)
            | (u32::from(kind.code()) << 4)
            | u32::from(rm),
    );
}

/// `mul.w rd, rn, rm`: multiply two registers.
#[inline]
pub fn muls_reg_wide(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    word(
        out,
        (0xfb00 | u32::from(rn)) << 16 | 0xf000 | u32::from(rd) << 8 | u32::from(rm),
    );
}

/// `cmp.w rn, rm`: compare two registers, in the wide form.
#[inline]
pub fn cmp_reg_wide(out: &mut Vec<u8>, rn: u8, rm: u8) {
    word(out, (0xebb0 | u32::from(rn)) << 16 | 0x0f00 | u32::from(rm));
}

/// `rsb.w rd, rn, #0`: the negation of a register, in the wide form.
#[inline]
pub fn negs_reg_wide(out: &mut Vec<u8>, rd: u8, rn: u8) {
    word(
        out,
        (0xf1c0 | u32::from(rn)) << 16 | u32::from(rd) << 8,
    );
}

/// `add rd, sp, #imm`: the address of a stack slot.
///
/// The immediate is the distance from the stack pointer in bytes; it
/// reaches a kilobyte.
#[inline]
pub fn add_sp_imm(out: &mut Vec<u8>, rd: u8, imm: u16) {
    halfword(out, 0xa800 | u16::from(rd) | imm & 0x3fc);
}

/// `sub sp, #imm`: the subtraction of the frame from the stack
/// pointer.
///
/// The immediate is the distance in bytes; it is a multiple of four
/// and reaches a kilobyte.
#[inline]
pub fn sub_sp_imm(out: &mut Vec<u8>, imm: u32) {
    halfword(out, 0xb080 | u16::try_from((imm >> 2) & 0xff).unwrap_or(0));
}

/// `add sp, #imm`: the release of the frame.
///
/// The immediate is the distance in bytes; it is a multiple of four
/// and reaches a kilobyte.
#[inline]
pub fn add_sp_frame(out: &mut Vec<u8>, imm: u32) {
    halfword(out, 0xb000 | u16::try_from((imm >> 2) & 0xff).unwrap_or(0));
}

/// `add.w rd, rn, rm, lsl #shift`: add two registers with a shifted
/// operand.
#[inline]
pub fn adds_reg_shifted(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8, shift: u8) {
    data_processing_wide(out, 0xeb00, rd, rn, rm, shift, Shift::Left);
}

/// `adcs rd, rm`: the addition of two registers and the carry.
#[inline]
pub fn adcs_reg(out: &mut Vec<u8>, rd: u8, rm: u8) {
    halfword(out, 0x4140 | u16::from(rm) << 3 | u16::from(rd));
}

/// `ldr rd, [rn, #imm5*4]`: load a word.
#[inline]
pub fn ldr_imm(out: &mut Vec<u8>, rd: u8, rn: u8, imm5: u8) {
    halfword(
        out,
        0x6800 | u16::from(imm5 & 0x1f) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `str rd, [rn, #imm5*4]`: store a word.
#[inline]
pub fn str_imm(out: &mut Vec<u8>, rd: u8, rn: u8, imm5: u8) {
    halfword(
        out,
        0x6000 | u16::from(imm5 & 0x1f) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `ldrb rd, [rn, #imm5]`: load a byte.
#[inline]
pub fn ldrb_imm(out: &mut Vec<u8>, rd: u8, rn: u8, imm5: u8) {
    halfword(
        out,
        0x7800 | u16::from(imm5 & 0x1f) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `strb rd, [rn, #imm5]`: store a byte.
#[inline]
pub fn strb_imm(out: &mut Vec<u8>, rd: u8, rn: u8, imm5: u8) {
    halfword(
        out,
        0x7000 | u16::from(imm5 & 0x1f) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `ldrh rd, [rn, #imm5*2]`: load a halfword.
#[inline]
pub fn ldrh_imm(out: &mut Vec<u8>, rd: u8, rn: u8, imm5: u8) {
    halfword(
        out,
        0x8800 | u16::from(imm5 & 0x1f) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `strh rd, [rn, #imm5*2]`: store a halfword.
#[inline]
pub fn strh_imm(out: &mut Vec<u8>, rd: u8, rn: u8, imm5: u8) {
    halfword(
        out,
        0x8000 | u16::from(imm5 & 0x1f) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `ldr rd, [rn, rm]`: load a word at a computed address.
#[inline]
pub fn ldr_reg(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    halfword(
        out,
        0x5800 | u16::from(rm) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `str rd, [rn, rm]`: store a word at a computed address.
#[inline]
pub fn str_reg(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    halfword(
        out,
        0x5000 | u16::from(rm) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `ldrb rd, [rn, rm]`: load a byte at a computed address.
#[inline]
pub fn ldrb_reg(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    halfword(
        out,
        0x5c00 | u16::from(rm) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `strb rd, [rn, rm]`: store a byte at a computed address.
#[inline]
pub fn strb_reg(out: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    halfword(
        out,
        0x5400 | u16::from(rm) << 6 | u16::from(rn) << 3 | u16::from(rd),
    );
}

/// `ldr rd, [sp, #imm8*4]`: load a word from the stack.
#[inline]
pub fn ldr_sp(out: &mut Vec<u8>, rd: u8, imm8: u8) {
    halfword(out, 0x9800 | u16::from(rd) << 8 | u16::from(imm8));
}

/// `str rd, [sp, #imm8*4]`: store a word into the stack.
#[inline]
pub fn str_sp(out: &mut Vec<u8>, rd: u8, imm8: u8) {
    halfword(out, 0x9000 | u16::from(rd) << 8 | u16::from(imm8));
}

/// `ldr.w rd, [rn, #imm12]`: load a word at a large offset.
#[inline]
pub fn ldr_imm_wide(out: &mut Vec<u8>, rd: u8, rn: u8, imm12: u16) {
    word(
        out,
        (0xf8d0 | u32::from(rn)) << 16 | u32::from(rd) << 12 | u32::from(imm12),
    );
}

/// `str.w rd, [rn, #imm12]`: store a word at a large offset.
#[inline]
pub fn str_imm_wide(out: &mut Vec<u8>, rd: u8, rn: u8, imm12: u16) {
    word(
        out,
        (0xf8c0 | u32::from(rn)) << 16 | u32::from(rd) << 12 | u32::from(imm12),
    );
}

/// `ldrb.w rd, [rn, #imm12]`: load a byte at a large offset.
#[inline]
pub fn ldrb_imm_wide(out: &mut Vec<u8>, rd: u8, rn: u8, imm12: u16) {
    word(
        out,
        (0xf890 | u32::from(rn)) << 16 | u32::from(rd) << 12 | u32::from(imm12),
    );
}

/// `strb.w rd, [rn, #imm12]`: store a byte at a large offset.
#[inline]
pub fn strb_imm_wide(out: &mut Vec<u8>, rd: u8, rn: u8, imm12: u16) {
    word(
        out,
        (0xf880 | u32::from(rn)) << 16 | u32::from(rd) << 12 | u32::from(imm12),
    );
}

/// `ldrh.w rd, [rn, #imm12]`: load a halfword at a large offset.
#[inline]
pub fn ldrh_imm_wide(out: &mut Vec<u8>, rd: u8, rn: u8, imm12: u16) {
    word(
        out,
        (0xf8b0 | u32::from(rn)) << 16 | u32::from(rd) << 12 | u32::from(imm12),
    );
}

/// `strh.w rd, [rn, #imm12]`: store a halfword at a large offset.
#[inline]
pub fn strh_imm_wide(out: &mut Vec<u8>, rd: u8, rn: u8, imm12: u16) {
    word(
        out,
        (0xf8a0 | u32::from(rn)) << 16 | u32::from(rd) << 12 | u32::from(imm12),
    );
}

/// `ldr.w rd, [sp, #imm12]`: load a word from the stack at a large
/// offset.
#[inline]
pub fn ldr_sp_wide(out: &mut Vec<u8>, rd: u8, imm12: u16) {
    word(
        out,
        (0xf8d0 | u32::from(crate::reg::SP)) << 16 | u32::from(rd) << 12 | u32::from(imm12),
    );
}

/// `str.w rd, [sp, #imm12]`: store a word into the stack at a large
/// offset.
#[inline]
pub fn str_sp_wide(out: &mut Vec<u8>, rd: u8, imm12: u16) {
    word(
        out,
        (0xf8c0 | u32::from(crate::reg::SP)) << 16 | u32::from(rd) << 12 | u32::from(imm12),
    );
}



/// `push {registers, lr}`: save the callee-saved registers.
pub fn push(out: &mut Vec<u8>, registers: u16, link: bool) {
    let low = registers & 0xff;
    let high = (registers >> 8) & 0x0f;
    if high == 0 {
        halfword(out, 0xb400 | if link { 0x100 } else { 0 } | low);
    } else {
        word(
            out,
            0xe92d_0000
                | if link { 0x4000 } else { 0 }
                | u32::from(high) << 12
                | u32::from(low),
        );
    }
}

/// `pop {registers, pc}`: restore the callee-saved registers.
pub fn pop(out: &mut Vec<u8>, registers: u16, program_counter: bool) {
    let low = registers & 0xff;
    let high = (registers >> 8) & 0x0f;
    if high == 0 {
        halfword(out, 0xbc00 | if program_counter { 0x100 } else { 0 } | low);
    } else {
        word(
            out,
            0xe8bd_0000
                | if program_counter { 0x8000 } else { 0 }
                | u32::from(high) << 12
                | u32::from(low),
        );
    }
}

/// `bx rm`: branch to the address in a register.
#[inline]
pub fn bx(out: &mut Vec<u8>, rm: u8) {
    halfword(out, 0x4700 | u16::from(rm) << 3);
}

/// `it cond, mask`: the head of a conditional execution block.
#[inline]
pub fn it(out: &mut Vec<u8>, cond: u8, mask: u8) {
    halfword(out, 0xbf00 | u16::from(cond & 0xf) << 4 | u16::from(mask & 0xf));
}

/// `udf #0`: an instruction that is always undefined.
#[inline]
pub fn undefined(out: &mut Vec<u8>) {
    halfword(out, 0xde00);
}

/// The two halfwords of an unconditional branch with a displacement.
///
/// The displacement is the distance from the end of the branch to its
/// target, in bytes; it must be even.  The two high bits of the
/// displacement travel in the second halfword, inverted, so that the
/// sign of the displacement survives the encoding.
#[must_use]
pub fn branch_bytes(offset: i32) -> [u8; 4] {
    let sign = (offset >> 24) & 1;
    let imm10 = u32::try_from((offset >> 12) & 0x3ff).unwrap_or(0);
    let imm11 = u32::try_from((offset >> 1) & 0x7ff).unwrap_or(0);
    let first = 0xf000 | u32::try_from(sign).unwrap_or(0) << 10 | imm10;
    let j1 = (1 - ((offset >> 23) & 1)) ^ sign;
    let j2 = (1 - ((offset >> 22) & 1)) ^ sign;
    let second = 0xb000
        | u32::try_from(j1).unwrap_or(0) << 13
        | 1 << 12
        | u32::try_from(j2).unwrap_or(0) << 11
        | imm11;
    [
        u8::try_from(first & 0xff).unwrap_or(0),
        u8::try_from((first >> 8) & 0xff).unwrap_or(0),
        u8::try_from(second & 0xff).unwrap_or(0),
        u8::try_from((second >> 8) & 0xff).unwrap_or(0),
    ]
}

/// The two halfwords of a call with a displacement.
///
/// The displacement is the distance from the end of the call to its
/// target, in bytes; it must be even.
#[must_use]
pub fn branch_link_bytes(offset: i32) -> [u8; 4] {
    let sign = (offset >> 24) & 1;
    let imm10 = u32::try_from((offset >> 12) & 0x3ff).unwrap_or(0);
    let imm11 = u32::try_from((offset >> 1) & 0x7ff).unwrap_or(0);
    let first = 0xf000 | u32::try_from(sign).unwrap_or(0) << 10 | imm10;
    let j1 = (1 - ((offset >> 23) & 1)) ^ sign;
    let j2 = (1 - ((offset >> 22) & 1)) ^ sign;
    let second = 0xf000
        | u32::try_from(j1).unwrap_or(0) << 13
        | 1 << 12
        | u32::try_from(j2).unwrap_or(0) << 11
        | imm11;
    [
        u8::try_from(first & 0xff).unwrap_or(0),
        u8::try_from((first >> 8) & 0xff).unwrap_or(0),
        u8::try_from(second & 0xff).unwrap_or(0),
        u8::try_from((second >> 8) & 0xff).unwrap_or(0),
    ]
}

/// The two halfwords of a conditional branch with a displacement.
///
/// The displacement is the distance from the end of the branch to its
/// target, in bytes; it must be even.
#[must_use]
pub fn branch_cond_bytes(cond: u8, offset: i32) -> [u8; 4] {
    let sign = u32::try_from((offset >> 20) & 1).unwrap_or(0);
    let imm6 = u32::try_from((offset >> 12) & 0x3f).unwrap_or(0);
    let imm11 = u32::try_from((offset >> 1) & 0x7ff).unwrap_or(0);
    let first = 0xf000 | sign << 10 | u32::from(cond & 0xf) << 6 | imm6;
    let j1 = u32::try_from((offset >> 18) & 1).unwrap_or(0);
    let j2 = u32::try_from((offset >> 19) & 1).unwrap_or(0);
    let second = 0x8000 | j1 << 13 | j2 << 11 | imm11;
    [
        u8::try_from(first & 0xff).unwrap_or(0),
        u8::try_from((first >> 8) & 0xff).unwrap_or(0),
        u8::try_from(second & 0xff).unwrap_or(0),
        u8::try_from((second >> 8) & 0xff).unwrap_or(0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn bytes_of(f: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut out = Vec::new();
        f(&mut out);
        out
    }

    #[test]
    fn the_small_moves_match_the_architecture() {
        assert_eq!(bytes_of(|o| movs_imm(o, 0, 1)), vec![0x01, 0x20]);
        assert_eq!(bytes_of(|o| movs_imm(o, 7, 0xff)), vec![0xff, 0x27]);
        assert_eq!(bytes_of(|o| mov_reg(o, 0, 1)), vec![0x08, 0x46]);
        assert_eq!(bytes_of(|o| bx(o, 14)), vec![0x70, 0x47]);
    }

    #[test]
    fn the_register_lists_match_the_architecture() {
        assert_eq!(bytes_of(|o| push(o, 0x10, true)), vec![0x10, 0xb5]);
        assert_eq!(bytes_of(|o| pop(o, 0x10, true)), vec![0x10, 0xbd]);
        assert_eq!(bytes_of(|o| push(o, 0, true)), vec![0x00, 0xb5]);
        assert_eq!(bytes_of(|o| pop(o, 0, true)), vec![0x00, 0xbd]);
    }

    #[test]
    fn the_additions_match_the_architecture() {
        assert_eq!(bytes_of(|o| adds_reg(o, 0, 1)), vec![0x08, 0x88]);
        assert_eq!(bytes_of(|o| adds_imm3(o, 0, 1, 7)), vec![0xc8, 0x1d]);
        assert_eq!(bytes_of(|o| adds_imm8(o, 0, 255)), vec![0xff, 0x30]);
    }

    #[test]
    fn the_conditional_execution_head_matches_the_architecture() {
        assert_eq!(bytes_of(|o| it(o, COND_EQ, IT_ONE)), vec![0x08, 0xbf]);
        assert_eq!(
            bytes_of(|o| it(o, COND_EQ, IT_THEN_ELSE)),
            vec![0x0c, 0xbf]
        );
    }
}
