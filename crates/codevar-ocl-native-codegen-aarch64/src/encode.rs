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

//! The byte-level encoder of the AArch64 back end.
//!
//! Every instruction of the architecture is a fixed word of thirty-two
//! bits, stored little-endian.  The encoder turns an instruction into
//! that word and appends it to the buffer the caller owns, so the
//! layout stage can lay the words out, patch the branches and hand
//! the buffer on without copying it.
//!
//! # The shapes of the encodings
//!
//! The words fall into a handful of shapes, and each shape has its
//! own set of fixed bits:
//!
//! * **Data processing, register.**  A source register, an optional
//!   shift, a second source register and a destination.
//! * **Data processing, immediate.**  A source register, a
//!   twelve-bit immediate and a destination; the immediate may also
//!   be scaled by four thousand and ninety-six.
//! * **Logical, immediate.**  Not used: the back end materializes a
//!   logical immediate in a register instead, which keeps the encoder
//!   free of the bitmask encoding.
//! * **Conditional select.**  Two source registers, a condition and a
//!   destination.
//! * **Load and store.**  A base register, an offset and a
//!   destination; the offset is either scaled by the size of the
//!   access or a signed nine-bit value.
//! * **Load and store, pair.**  Two registers and an offset that is
//!   scaled by eight.
//! * **Branch.**  A displacement in words, either twenty-six bits
//!   wide for the unconditional forms or nineteen bits wide for the
//!   conditional ones.
//! * **Floating point and vector.**  The same shapes as the integer
//!   forms, with the width of the access in the type field.

use alloc::vec::Vec;

use codevar_ocl_native_codegen::error::{CodegenError, Result};

/// The width of an integer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    /// Thirty-two bits.
    Bits32,
    /// Sixty-four bits.
    Bits64,
}

impl Width {
    /// The width of a value of the given machine type.
    ///
    /// The narrow integers share the thirty-two-bit view: a machine
    /// value of eight or sixteen bits is held in a thirty-two-bit
    /// register, and the conversion that narrows it is a separate
    /// instruction.
    #[must_use]
    pub fn of_type(size: Option<u32>) -> Self {
        match size {
            Some(8) => Self::Bits64,
            _ => Self::Bits32,
        }
    }

    /// Whether the width is sixty-four bits.
    #[must_use]
    #[inline]
    pub const fn is_64(self) -> bool {
        matches!(self, Self::Bits64)
    }

    /// The number of bits the width stands for.
    #[must_use]
    #[inline]
    pub const fn bits(self) -> u32 {
        if self.is_64() { 64 } else { 32 }
    }
}

/// A condition of the architecture.
///
/// The codes are the ones the branch and select instructions carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Cond {
    /// Equal.
    Eq = 0,
    /// Not equal.
    Ne = 1,
    /// Carry set.
    Cs = 2,
    /// Carry clear.
    Cc = 3,
    /// Minus.
    Mi = 4,
    /// Plus.
    Pl = 5,
    /// Overflow.
    Vs = 6,
    /// No overflow.
    Vc = 7,
    /// Unsigned higher.
    Hi = 8,
    /// Unsigned lower or same.
    Ls = 9,
    /// Greater or equal.
    Ge = 10,
    /// Less than.
    Lt = 11,
    /// Greater than.
    Gt = 12,
    /// Less or equal.
    Le = 13,
    /// Always.
    Al = 14,
}

impl Cond {
    /// The condition that holds exactly when this one does not.
    #[must_use]
    #[inline]
    pub const fn inverted(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Cs => Self::Cc,
            Self::Cc => Self::Cs,
            Self::Mi => Self::Pl,
            Self::Pl => Self::Mi,
            Self::Vs => Self::Vc,
            Self::Vc => Self::Vs,
            Self::Hi => Self::Ls,
            Self::Ls => Self::Hi,
            Self::Ge => Self::Lt,
            Self::Lt => Self::Ge,
            Self::Gt => Self::Le,
            Self::Le => Self::Gt,
            Self::Al => Self::Al,
        }
    }

    /// The four-bit code of the condition.
    #[must_use]
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }
}

/// Whether an access reads memory or writes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// A load.
    Load,
    /// A store.
    Store,
}

/// The encoder of the back end.
///
/// The encoder appends the words it builds to the buffer the caller
/// owns; it never allocates a buffer of its own.
#[derive(Debug)]
pub struct Encoder<'a> {
    /// The buffer the words go to.
    out: &'a mut Vec<u8>,
    /// The number of words written so far.
    count: u32,
}

impl<'a> Encoder<'a> {
    /// Creates an encoder that writes into `out`.
    #[must_use]
    pub const fn new(out: &'a mut Vec<u8>) -> Self {
        Self { out, count: 0 }
    }

    /// The number of words written so far.
    #[must_use]
    #[inline]
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// The offset of the next word inside the buffer.
    #[must_use]
    #[inline]
    pub fn offset(&self) -> u32 {
        self.out.len() as u32
    }

    /// Appends a word.
    #[inline]
    pub fn word(&mut self, value: u32) {
        self.out.extend_from_slice(&value.to_le_bytes());
        self.count += 1;
    }

    // -- data processing, register ------------------------------------

    /// `add rd, rn, rm`.
    #[inline]
    pub fn add_reg(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0x8B00_0000 } else { 0x0B00_0000 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `add rd, rn, rm, lsl #shift`.
    ///
    /// The shift is a left shift, whose amount stands in the field
    /// of the instruction that holds the amount of a shift.
    #[inline]
    pub fn add_reg_lsl(&mut self, rd: u8, rn: u8, rm: u8, shift: u32, width: Width) {
        let base = if width.is_64() { 0x8B00_0000 } else { 0x0B00_0000 };
        self.word(base | (rm as u32) << 16 | (shift & 0x3F) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// `sub rd, rn, rm`.
    #[inline]
    pub fn sub_reg(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0xCB00_0000 } else { 0x4B00_0000 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `and rd, rn, rm`.
    #[inline]
    pub fn and_reg(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0x8A00_0000 } else { 0x0A00_0000 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `orr rd, rn, rm`.
    #[inline]
    pub fn orr_reg(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0xAA00_0000 } else { 0x2A00_0000 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `eor rd, rn, rm`.
    #[inline]
    pub fn eor_reg(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0xCA00_0000 } else { 0x4A00_0000 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `mov rd, rm`.
    #[inline]
    pub fn mov_reg(&mut self, rd: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0xAA00_0000 } else { 0x2A00_0000 };
        self.word(base | (rm as u32) << 16 | 31 << 5 | rd as u32);
    }

    /// `mvn rd, rm`.
    ///
    /// The complement is the logical or of the source and the
    /// opposite of it, so the word is the logical or with the
    /// complement of the first source switched on.
    #[inline]
    pub fn mvn_reg(&mut self, rd: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0xAA20_0000 } else { 0x2A20_0000 };
        self.word(base | (rm as u32) << 16 | 31 << 5 | rd as u32);
    }

    /// `neg rd, rm`.
    #[inline]
    pub fn neg_reg(&mut self, rd: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0xCB00_0000 } else { 0x4B00_0000 };
        self.word(base | (rm as u32) << 16 | 31 << 5 | rd as u32);
    }

    // -- data processing, immediate -----------------------------------

    /// `add rd, rn, #imm`.
    ///
    /// The immediate is the twelve-bit unsigned form.
    #[inline]
    pub fn add_imm(&mut self, rd: u8, rn: u8, imm: u16, width: Width) {
        let base = if width.is_64() { 0x9100_0000 } else { 0x1100_0000 };
        self.word(base | (imm as u32) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// `add rd, rn, #imm, lsl #12`.
    #[inline]
    pub fn add_imm_scaled(&mut self, rd: u8, rn: u8, imm: u16, width: Width) {
        let base = if width.is_64() { 0x9100_0000 } else { 0x1100_0000 };
        self.word(base | 1 << 22 | (imm as u32) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// `sub rd, rn, #imm`.
    #[inline]
    pub fn sub_imm(&mut self, rd: u8, rn: u8, imm: u16, width: Width) {
        let base = if width.is_64() { 0xD100_0000 } else { 0x5100_0000 };
        self.word(base | (imm as u32) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// `sub rd, rn, #imm, lsl #12`.
    #[inline]
    pub fn sub_imm_scaled(&mut self, rd: u8, rn: u8, imm: u16, width: Width) {
        let base = if width.is_64() { 0xD100_0000 } else { 0x5100_0000 };
        self.word(base | 1 << 22 | (imm as u32) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// `cmp rn, #imm`.
    #[inline]
    pub fn cmp_imm(&mut self, rn: u8, imm: u16, width: Width) {
        let base = if width.is_64() { 0xF100_0000 } else { 0x7100_0000 };
        self.word(base | (imm as u32) << 10 | (rn as u32) << 5 | 31);
    }

    /// `cmp rn, rm`.
    #[inline]
    pub fn cmp_reg(&mut self, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0xEB00_001F } else { 0x6B00_001F };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5);
    }

    // -- wide immediates ----------------------------------------------

    /// `movz rd, #imm, lsl #shift`.
    #[inline]
    pub fn movz(&mut self, rd: u8, imm: u16, shift: u32, width: Width) {
        let base = if width.is_64() { 0xD280_0000 } else { 0x5280_0000 };
        self.word(base | (shift & 0x3) << 21 | (imm as u32) << 5 | rd as u32);
    }

    /// `movn rd, #imm, lsl #shift`.
    #[inline]
    pub fn movn(&mut self, rd: u8, imm: u16, shift: u32, width: Width) {
        let base = if width.is_64() { 0x9280_0000 } else { 0x1280_0000 };
        self.word(base | (shift & 0x3) << 21 | (imm as u32) << 5 | rd as u32);
    }

    /// `movk rd, #imm, lsl #shift`.
    #[inline]
    pub fn movk(&mut self, rd: u8, imm: u16, shift: u32, width: Width) {
        let base = if width.is_64() { 0xF280_0000 } else { 0x7280_0000 };
        self.word(base | (shift & 0x3) << 21 | (imm as u32) << 5 | rd as u32);
    }

    // -- multiply and divide ------------------------------------------

    /// `mul rd, rn, rm`.
    #[inline]
    pub fn mul(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0x9B00_7C00 } else { 0x1B00_7C00 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `msub rd, rn, rm, ra`.
    #[inline]
    pub fn msub(&mut self, rd: u8, rn: u8, rm: u8, ra: u8, width: Width) {
        let base = if width.is_64() { 0x9B00_8000 } else { 0x1B00_8000 };
        self.word(base | (rm as u32) << 16 | (ra as u32) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// `udiv rd, rn, rm`.
    #[inline]
    pub fn udiv(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0x9AC0_0800 } else { 0x1AC0_0800 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `sdiv rd, rn, rm`.
    #[inline]
    pub fn sdiv(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0x9AC0_0C00 } else { 0x1AC0_0C00 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    // -- shifts --------------------------------------------------------

    /// `lsl rd, rn, rm`.
    #[inline]
    pub fn lsl_reg(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0x9AC0_2000 } else { 0x1AC0_2000 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `lsr rd, rn, rm`.
    #[inline]
    pub fn lsr_reg(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0x9AC0_2400 } else { 0x1AC0_2400 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `asr rd, rn, rm`.
    #[inline]
    pub fn asr_reg(&mut self, rd: u8, rn: u8, rm: u8, width: Width) {
        let base = if width.is_64() { 0x9AC0_2800 } else { 0x1AC0_2800 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `lsl rd, rn, #shift`.
    pub fn lsl_imm(&mut self, rd: u8, rn: u8, shift: u32, width: Width) {
        let bits = width.bits();
        let base = if width.is_64() { 0xD340_0000 } else { 0x5300_0000 };
        let immr = (bits - shift) & (bits - 1);
        let imms = (bits - 1 - shift) & (bits - 1);
        self.word(base | immr << 16 | imms << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// `lsr rd, rn, #shift`.
    pub fn lsr_imm(&mut self, rd: u8, rn: u8, shift: u32, width: Width) {
        let bits = width.bits();
        let base = if width.is_64() { 0xD340_0000 } else { 0x5300_0000 };
        self.word(base | (shift & (bits - 1)) << 16 | (bits - 1) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// `asr rd, rn, #shift`.
    pub fn asr_imm(&mut self, rd: u8, rn: u8, shift: u32, width: Width) {
        let bits = width.bits();
        let base = if width.is_64() { 0x9340_0000 } else { 0x1300_0000 };
        self.word(base | (shift & (bits - 1)) << 16 | (bits - 1) << 10 | (rn as u32) << 5 | rd as u32);
    }

    // -- extensions ----------------------------------------------------

    /// `uxtw rd, rn`.
    #[inline]
    pub fn uxtw(&mut self, rd: u8, rn: u8) {
        self.word(0xD340_7C00 | (rn as u32) << 5 | rd as u32);
    }

    /// `sxtw rd, rn`.
    #[inline]
    pub fn sxtw(&mut self, rd: u8, rn: u8) {
        self.word(0x9340_7C00 | (rn as u32) << 5 | rd as u32);
    }

    /// `uxtb rd, rn`.
    #[inline]
    pub fn uxtb(&mut self, rd: u8, rn: u8) {
        self.word(0x5300_1C00 | (rn as u32) << 5 | rd as u32);
    }

    /// `uxth rd, rn`.
    #[inline]
    pub fn uxth(&mut self, rd: u8, rn: u8) {
        self.word(0x5300_3C00 | (rn as u32) << 5 | rd as u32);
    }

    /// `sxtb rd, rn`, writing a sixty-four-bit result.
    #[inline]
    pub fn sxtb(&mut self, rd: u8, rn: u8) {
        self.word(0x9340_1C00 | (rn as u32) << 5 | rd as u32);
    }

    /// `sxtb rd, rn`, writing a thirty-two-bit result.
    #[inline]
    pub fn sxtb32(&mut self, rd: u8, rn: u8) {
        self.word(0x1300_1C00 | (rn as u32) << 5 | rd as u32);
    }

    /// `sxth rd, rn`, writing a sixty-four-bit result.
    #[inline]
    pub fn sxth(&mut self, rd: u8, rn: u8) {
        self.word(0x9340_3C00 | (rn as u32) << 5 | rd as u32);
    }

    /// `sxth rd, rn`, writing a thirty-two-bit result.
    #[inline]
    pub fn sxth32(&mut self, rd: u8, rn: u8) {
        self.word(0x1300_3C00 | (rn as u32) << 5 | rd as u32);
    }

    // -- conditional select --------------------------------------------

    /// `csel rd, rn, rm, cond`.
    #[inline]
    pub fn csel(&mut self, rd: u8, rn: u8, rm: u8, cond: Cond, width: Width) {
        let base = if width.is_64() { 0x9A80_0000 } else { 0x1A80_0000 };
        self.word(base | (rm as u32) << 16 | cond.code() << 12 | (rn as u32) << 5 | rd as u32);
    }

    /// `cset rd, cond`.
    ///
    /// The instruction is the conditional increment of the zero
    /// register, so the condition the word carries is the opposite
    /// of the one the instruction names.
    #[inline]
    pub fn cset(&mut self, rd: u8, cond: Cond, width: Width) {
        let base = if width.is_64() { 0x9A9F_07E0 } else { 0x1A9F_07E0 };
        self.word(base | cond.inverted().code() << 12 | rd as u32);
    }

    // -- branches ------------------------------------------------------

    /// `b` with a displacement of zero; the layout stage patches it.
    #[inline]
    pub fn b(&mut self) {
        self.word(0x1400_0000);
    }

    /// `bl` with a displacement of zero; the layout stage patches it.
    #[inline]
    pub fn bl(&mut self) {
        self.word(0x9400_0000);
    }

    /// `b.cond` with a displacement of zero; the layout stage patches
    /// it.
    #[inline]
    pub fn b_cond(&mut self, cond: Cond) {
        self.word(0x5400_0000 | cond.code());
    }

    /// `cbz rt, label` with a displacement of zero; the layout stage
    /// patches it.
    #[inline]
    pub fn cbz(&mut self, rt: u8, width: Width) {
        let base = if width.is_64() { 0xB400_0000 } else { 0x3400_0000 };
        self.word(base | rt as u32);
    }

    /// `cbnz rt, label` with a displacement of zero; the layout stage
    /// patches it.
    #[inline]
    pub fn cbnz(&mut self, rt: u8, width: Width) {
        let base = if width.is_64() { 0xB500_0000 } else { 0x3500_0000 };
        self.word(base | rt as u32);
    }

    /// `ret`.
    #[inline]
    pub fn ret(&mut self) {
        self.word(0xD65F_03C0);
    }

    /// `nop`.
    #[inline]
    pub fn nop(&mut self) {
        self.word(0xD503_201F);
    }

    /// `brk #imm`.
    #[inline]
    pub fn brk(&mut self, imm: u16) {
        self.word(0xD420_0000 | (imm as u32) << 5);
    }

    // -- memory --------------------------------------------------------

    /// A load or a store of the given size at the given offset.
    ///
    /// The offset is scaled by the size of the access when it can be,
    /// and is a signed nine-bit value otherwise.
    pub fn access(&mut self, direction: Direction, rt: u8, rn: u8, offset: i64, size: u32) -> Result<()> {
        if size != 1 && size != 2 && size != 4 && size != 8 {
            return Err(CodegenError::Target {
                what: "encoding a memory access",
                detail: alloc::format!("an access of {size} bytes has no encoding"),
            });
        }
        if offset >= 0 && offset % i64::from(size) == 0 {
            let scaled = offset / i64::from(size);
            if scaled <= 4095 {
                let base = match (direction, size) {
                    (Direction::Load, 8) => 0xF940_0000,
                    (Direction::Load, 4) => 0xB940_0000,
                    (Direction::Load, 2) => 0x7940_0000,
                    (Direction::Load, 1) => 0x3940_0000,
                    (Direction::Store, 8) => 0xF900_0000,
                    (Direction::Store, 4) => 0xB900_0000,
                    (Direction::Store, 2) => 0x7900_0000,
                    (Direction::Store, 1) => 0x3900_0000,
                    _ => 0,
                };
                self.word(base | (scaled as u32) << 10 | (rn as u32) << 5 | rt as u32);
                return Ok(());
            }
        }
        if (-256..=255).contains(&offset) {
            let base = match (direction, size) {
                (Direction::Load, 8) => 0xF840_0000,
                (Direction::Load, 4) => 0xB840_0000,
                (Direction::Load, 2) => 0x7840_0000,
                (Direction::Load, 1) => 0x3840_0000,
                (Direction::Store, 8) => 0xF800_0000,
                (Direction::Store, 4) => 0xB800_0000,
                (Direction::Store, 2) => 0x7800_0000,
                (Direction::Store, 1) => 0x3800_0000,
                _ => 0,
            };
            let imm = (offset as i32 as u32) & 0x1FF;
            self.word(base | imm << 12 | (rn as u32) << 5 | rt as u32);
            return Ok(());
        }
        Err(CodegenError::Target {
            what: "encoding a memory access",
            detail: alloc::format!("the offset {offset} is out of the range of the access forms"),
        })
    }

    /// A load of the given size.
    pub fn load(&mut self, rt: u8, rn: u8, offset: i64, size: u32) -> Result<()> {
        self.access(Direction::Load, rt, rn, offset, size)
    }

    /// A store of the given size.
    pub fn store(&mut self, rt: u8, rn: u8, offset: i64, size: u32) -> Result<()> {
        self.access(Direction::Store, rt, rn, offset, size)
    }

    /// A load at a register offset.
    ///
    /// The `index` parameter says whether the index register holds a
    /// thirty-two-bit value, which is sign-extended before the
    /// access.
    pub fn load_reg(&mut self, rt: u8, rn: u8, rm: u8, index32: bool, size: u32) {
        let (base, option, sign) = match (size, index32) {
            (8, false) => (0xF862_0000, 3u32, 0u32),
            (8, true) => (0xF862_0000, 6, 1),
            (4, false) => (0xB862_0000, 3, 0),
            (4, true) => (0xB862_0000, 6, 1),
            (2, false) => (0x7862_0000, 3, 0),
            (2, true) => (0x7862_0000, 6, 1),
            (_, false) => (0x3862_0000, 3, 0),
            (_, true) => (0x3862_0000, 6, 1),
        };
        self.word(
            base | (rm as u32) << 16 | option << 13 | sign << 12 | 0x800 | (rn as u32) << 5 | rt as u32,
        );
    }

    /// A store at a register offset.
    pub fn store_reg(&mut self, rt: u8, rn: u8, rm: u8, index32: bool, size: u32) {
        let (base, option, sign) = match (size, index32) {
            (8, false) => (0xF822_0000, 3u32, 0u32),
            (8, true) => (0xF822_0000, 6, 1),
            (4, false) => (0xB822_0000, 3, 0),
            (4, true) => (0xB822_0000, 6, 1),
            (2, false) => (0x7822_0000, 3, 0),
            (2, true) => (0x7822_0000, 6, 1),
            (_, false) => (0x3822_0000, 3, 0),
            (_, true) => (0x3822_0000, 6, 1),
        };
        self.word(
            base | (rm as u32) << 16 | option << 13 | sign << 12 | 0x800 | (rn as u32) << 5 | rt as u32,
        );
    }

    /// A load or a store of a pair of registers.
    ///
    /// The offset is signed and scaled by the size of the access.
    pub fn pair(
        &mut self,
        direction: Direction,
        rt: u8,
        rt2: u8,
        rn: u8,
        offset: i64,
        size: u32,
    ) -> Result<()> {
        let scale = i64::from(size);
        if offset % scale != 0 {
            return Err(CodegenError::Target {
                what: "encoding a pair access",
                detail: alloc::format!("the offset {offset} is not a multiple of the access size"),
            });
        }
        let words = offset / scale;
        if !(-64..=63).contains(&words) {
            return Err(CodegenError::Target {
                what: "encoding a pair access",
                detail: alloc::format!("the offset {offset} is out of the range of the pair forms"),
            });
        }
        let base = match (direction, size) {
            (Direction::Load, 8) => 0xA940_0000,
            (Direction::Store, 8) => 0xA900_0000,
            (Direction::Load, 4) => 0x2940_0000,
            (Direction::Store, 4) => 0x2900_0000,
            _ => {
                return Err(CodegenError::Target {
                    what: "encoding a pair access",
                    detail: alloc::format!("a pair access of {size} bytes has no encoding"),
                });
            }
        };
        let imm = (words as u32) & 0x7F;
        self.word(base | imm << 15 | (rt2 as u32) << 10 | (rn as u32) << 5 | rt as u32);
        Ok(())
    }

    // -- floating point -------------------------------------------------

    /// A floating-point operation with two sources.
    fn fp2(&mut self, base: u32, rd: u8, rn: u8, rm: u8, single: bool) {
        let base = if single { base & !0x0040_0000 } else { base };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `fadd`.
    #[inline]
    pub fn fadd(&mut self, rd: u8, rn: u8, rm: u8, single: bool) {
        self.fp2(0x1E60_2800, rd, rn, rm, single);
    }

    /// `fsub`.
    #[inline]
    pub fn fsub(&mut self, rd: u8, rn: u8, rm: u8, single: bool) {
        self.fp2(0x1E60_3800, rd, rn, rm, single);
    }

    /// `fmul`.
    #[inline]
    pub fn fmul(&mut self, rd: u8, rn: u8, rm: u8, single: bool) {
        self.fp2(0x1E60_0800, rd, rn, rm, single);
    }

    /// `fdiv`.
    #[inline]
    pub fn fdiv(&mut self, rd: u8, rn: u8, rm: u8, single: bool) {
        self.fp2(0x1E60_1800, rd, rn, rm, single);
    }

    /// `fmax`.
    #[inline]
    pub fn fmax(&mut self, rd: u8, rn: u8, rm: u8, single: bool) {
        self.fp2(0x1E60_4800, rd, rn, rm, single);
    }

    /// `fmin`.
    #[inline]
    pub fn fmin(&mut self, rd: u8, rn: u8, rm: u8, single: bool) {
        self.fp2(0x1E60_5800, rd, rn, rm, single);
    }

    /// A floating-point operation with one source.
    fn fp1(&mut self, base: u32, rd: u8, rn: u8, single: bool) {
        let base = if single { base & !0x0040_0000 } else { base };
        self.word(base | (rn as u32) << 5 | rd as u32);
    }

    /// `fmov rd, rn`.
    #[inline]
    pub fn fmov_reg(&mut self, rd: u8, rn: u8, single: bool) {
        self.fp1(0x1E60_4000, rd, rn, single);
    }

    /// `fabs rd, rn`.
    #[inline]
    pub fn fabs(&mut self, rd: u8, rn: u8, single: bool) {
        self.fp1(0x1E60_C000, rd, rn, single);
    }

    /// `fneg rd, rn`.
    #[inline]
    pub fn fneg(&mut self, rd: u8, rn: u8, single: bool) {
        self.fp1(0x1E61_4000, rd, rn, single);
    }

    /// `fsqrt rd, rn`.
    #[inline]
    pub fn fsqrt(&mut self, rd: u8, rn: u8, single: bool) {
        self.fp1(0x1E61_C000, rd, rn, single);
    }

    /// `fmls rd, rn, rm`.
    ///
    /// The destination holds the multiplier, so the instruction
    /// computes `rn - rd * rm` and writes the result into it.
    #[inline]
    pub fn fmls(&mut self, rd: u8, rn: u8, rm: u8, single: bool) {
        let base = if single { 0x1E20_3000 } else { 0x1E60_3000 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `fcmp rn, rm`.
    #[inline]
    pub fn fcmp(&mut self, rn: u8, rm: u8, single: bool) {
        let base = if single { 0x1E20_2000 } else { 0x1E60_2000 };
        self.word(base | (rm as u32) << 16 | (rn as u32) << 5);
    }

    /// `fcsel rd, rn, rm, cond`.
    #[inline]
    pub fn fcsel(&mut self, rd: u8, rn: u8, rm: u8, cond: Cond, single: bool) {
        let base = if single { 0x1E20_0C00 } else { 0x1E60_0C00 };
        self.word(base | (rm as u32) << 16 | cond.code() << 12 | (rn as u32) << 5 | rd as u32);
    }

    /// `scvtf rd, rn`.
    #[inline]
    pub fn scvtf(&mut self, rd: u8, rn: u8, single: bool) {
        let base = if single { 0x1E22_0000 } else { 0x9E62_0000 };
        self.word(base | (rn as u32) << 5 | rd as u32);
    }

    /// `ucvtf rd, rn`.
    #[inline]
    pub fn ucvtf(&mut self, rd: u8, rn: u8, single: bool) {
        let base = if single { 0x1E23_0000 } else { 0x9E63_0000 };
        self.word(base | (rn as u32) << 5 | rd as u32);
    }

    /// `fcvtzs rd, rn`.
    #[inline]
    pub fn fcvtzs(&mut self, rd: u8, rn: u8, single: bool, wide: bool) {
        let base = match (single, wide) {
            (true, false) => 0x1E38_0000,
            (true, true) => 0x9E38_0000,
            (false, false) => 0x1E78_0000,
            (false, true) => 0x9E78_0000,
        };
        self.word(base | (rn as u32) << 5 | rd as u32);
    }

    /// `fcvtzu rd, rn`.
    #[inline]
    pub fn fcvtzu(&mut self, rd: u8, rn: u8, single: bool, wide: bool) {
        let base = match (single, wide) {
            (true, false) => 0x1E39_0000,
            (true, true) => 0x9E39_0000,
            (false, false) => 0x1E79_0000,
            (false, true) => 0x9E79_0000,
        };
        self.word(base | (rn as u32) << 5 | rd as u32);
    }

    /// `fcvt rd, rn`, narrowing a double to a single.
    #[inline]
    pub fn fcvt_narrow(&mut self, rd: u8, rn: u8) {
        self.word(0x1E62_4000 | (rn as u32) << 5 | rd as u32);
    }

    /// `fcvt rd, rn`, widening a single to a double.
    #[inline]
    pub fn fcvt_widen(&mut self, rd: u8, rn: u8) {
        self.word(0x1E22_C000 | (rn as u32) << 5 | rd as u32);
    }

    /// `fmov rd, rn`, moving the bits of a general-purpose register
    /// into a floating-point register.
    #[inline]
    pub fn fmov_from_gp(&mut self, rd: u8, rn: u8, single: bool) {
        let base = match single {
            true => 0x1E27_0000,
            false => 0x9E67_0000,
        };
        self.word(base | (rn as u32) << 5 | rd as u32);
    }

    /// `fmov rd, rn`, moving the bits of a floating-point register
    /// into a general-purpose register.
    #[inline]
    pub fn fmov_to_gp(&mut self, rd: u8, rn: u8, single: bool) {
        let base = match single {
            true => 0x1E26_0000,
            false => 0x9E66_0000,
        };
        self.word(base | (rn as u32) << 5 | rd as u32);
    }

    // -- vector ---------------------------------------------------------

    /// `dup vd.4s, rn`.
    #[inline]
    pub fn dup_4s(&mut self, rd: u8, rn: u8) {
        self.word(0x4E04_0C00 | (rn as u32) << 5 | rd as u32);
    }

    /// `dup vd.2d, rn`.
    #[inline]
    pub fn dup_2d(&mut self, rd: u8, rn: u8) {
        self.word(0x4E08_0C00 | (rn as u32) << 5 | rd as u32);
    }

    /// `umov rd, vn.s[index]`.
    #[inline]
    pub fn umov_32(&mut self, rd: u8, rn: u8, index: u32) {
        let imm5 = (index << 3) | 0b100;
        self.word(0x0E00_3C00 | imm5 << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `umov rd, vn.d[index]`.
    #[inline]
    pub fn umov_64(&mut self, rd: u8, rn: u8, index: u32) {
        let imm5 = (index << 4) | 0b1000;
        self.word(0x4E00_3C00 | imm5 << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `ins vd.s[index], rn`.
    #[inline]
    pub fn ins_32(&mut self, rd: u8, rn: u8, index: u32) {
        let imm5 = (index << 3) | 0b100;
        self.word(0x4E00_1C00 | imm5 << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `ins vd.d[index], rn`.
    #[inline]
    pub fn ins_64(&mut self, rd: u8, rn: u8, index: u32) {
        let imm5 = (index << 4) | 0b1000;
        self.word(0x4E00_1C00 | imm5 << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `ins vd.s[index], vn.s[index]`.
    #[inline]
    pub fn ins_32_lane(&mut self, rd: u8, rn: u8, index: u32, source: u32) {
        let imm5 = (index << 3) | 0b100;
        let imm4 = source * 4;
        self.word(0x6E00_0400 | imm5 << 16 | imm4 << 11 | (rn as u32) << 5 | rd as u32);
    }

    /// `ins vd.d[index], vn.d[index]`.
    #[inline]
    pub fn ins_64_lane(&mut self, rd: u8, rn: u8, index: u32, source: u32) {
        let imm5 = (index << 4) | 0b1000;
        let imm4 = source * 8;
        self.word(0x6E00_0400 | imm5 << 16 | imm4 << 11 | (rn as u32) << 5 | rd as u32);
    }

    /// `orr vd.16b, vn.16b, vm.16b`.
    #[inline]
    pub fn orr_vec(&mut self, rd: u8, rn: u8, rm: u8) {
        self.word(0x4EA0_1C00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `mov vd.16b, vn.16b`.
    ///
    /// The move is the logical or of the source with itself.
    #[inline]
    pub fn mov_vec(&mut self, rd: u8, rn: u8) {
        self.word(0x4EA0_1C00 | (rn as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `and vd.16b, vn.16b, vm.16b`.
    #[inline]
    pub fn and_vec(&mut self, rd: u8, rn: u8, rm: u8) {
        self.word(0x4E20_1C00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `eor vd.16b, vn.16b, vm.16b`.
    #[inline]
    pub fn eor_vec(&mut self, rd: u8, rn: u8, rm: u8) {
        self.word(0x6E20_1C00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `add vd.4s, vn.4s, vm.4s`.
    #[inline]
    pub fn add_vec_32(&mut self, rd: u8, rn: u8, rm: u8) {
        self.word(0x4EA0_8400 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `add vd.2d, vn.2d, vm.2d`.
    #[inline]
    pub fn add_vec_64(&mut self, rd: u8, rn: u8, rm: u8) {
        self.word(0x4EE0_8400 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `sub vd.4s, vn.4s, vm.4s`.
    #[inline]
    pub fn sub_vec_32(&mut self, rd: u8, rn: u8, rm: u8) {
        self.word(0x6EA0_8400 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `sub vd.2d, vn.2d, vm.2d`.
    #[inline]
    pub fn sub_vec_64(&mut self, rd: u8, rn: u8, rm: u8) {
        self.word(0x6EE0_8400 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `mul vd.4s, vn.4s, vm.4s`.
    #[inline]
    pub fn mul_vec_32(&mut self, rd: u8, rn: u8, rm: u8) {
        self.word(0x4EA0_9C00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// `mul vd.2d, vn.2d, vm.2d`.
    #[inline]
    pub fn mul_vec_64(&mut self, rd: u8, rn: u8, rm: u8) {
        self.word(0x4EE0_9C00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
}
