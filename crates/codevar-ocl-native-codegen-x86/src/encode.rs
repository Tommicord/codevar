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

//! The byte-level instruction encoder.
//!
//! The encoder writes machine code into a buffer the caller owns, so
//! no allocation happens while code is produced.  Every method emits
//! exactly one instruction and knows the whole of the encoding rules
//! it needs: the extension prefix, the addressing mode, the immediate
//! form.
//!
//! The baseline is the scalar SSE instruction set: the arithmetic is
//! expressed with the scalar forms, the shuffles with the packed ones,
//! and nothing outside the baseline is used.
//!
//! # The order of the prefixes
//!
//! An instruction may carry three prefixes, and their order is fixed:
//! the legacy prefixes come first, the extension prefix next, and the
//! opcode last.  Every method of the encoder therefore emits the
//! legacy prefix, then the extension prefix, then the opcode, and
//! only then the mode byte and the displacement.

use alloc::vec::Vec;

use crate::reg;

/// The size of an operand.
///
/// The byte, word and double-word forms address the general-purpose
/// registers; the quad-word form is the natural width of a pointer;
/// the vector form is a whole SSE register.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Size {
    /// Eight bits.
    Byte,
    /// Sixteen bits.
    Word,
    /// Thirty-two bits.
    Dword,
    /// Sixty-four bits.
    Qword,
    /// A whole SSE register.
    Xmm,
}

impl Size {
    /// Whether the form widens the general-purpose registers.
    #[inline]
    pub(crate) const fn rex_w(self) -> bool {
        matches!(self, Self::Qword)
    }

    /// Whether the form narrows the general-purpose registers to
    /// sixteen bits.
    #[inline]
    pub(crate) const fn is_word(self) -> bool {
        matches!(self, Self::Word)
    }
}

/// The condition code of a branch, a conditional move, or a flag
/// materialization.
///
/// The numbering is the architecture's own: the low four bits of the
/// opcode select the condition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Cc {
    /// Equal.
    E,
    /// Not equal.
    Ne,
    /// Below, unsigned.
    B,
    /// Above or equal, unsigned.
    Ae,
    /// Below or equal, unsigned.
    Be,
    /// Above, unsigned.
    A,
    /// Parity odd.
    Np,
    /// Below, signed.
    L,
    /// Greater or equal, signed.
    Ge,
    /// Below or equal, signed.
    Le,
    /// Greater, signed.
    G,
}

impl Cc {
    /// The low four bits of the opcode that select the condition.
    #[inline]
    pub(crate) const fn bits(self) -> u8 {
        match self {
            Self::E => 4,
            Self::Ne => 5,
            Self::B => 2,
            Self::Ae => 3,
            Self::Be => 6,
            Self::A => 7,
            Self::Np => 11,
            Self::L => 12,
            Self::Ge => 13,
            Self::Le => 14,
            Self::G => 15,
        }
    }
}

/// The operation of an arithmetic instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Arith {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
}

impl Arith {
    /// The opcode that selects the operation.
    #[inline]
    const fn opcode(self) -> u8 {
        match self {
            Self::Add => 0x58,
            Self::Sub => 0x5C,
            Self::Mul => 0x59,
            Self::Div => 0x5E,
        }
    }
}

/// The encoder of the target's instructions.
///
/// The encoder owns no buffer of its own: it writes into the vector
/// the caller lends it, which keeps the production of a whole module
/// free of allocations once the buffer has grown to its final size.
pub(crate) struct Encoder<'a> {
    /// The buffer the instructions are written into.
    out: &'a mut Vec<u8>,
    /// The number of instructions emitted so far.
    count: u32,
}

impl<'a> Encoder<'a> {
    /// Creates an encoder that writes into `out`.
    pub(crate) fn new(out: &'a mut Vec<u8>) -> Self {
        Self { out, count: 0 }
    }

    /// The number of bytes emitted so far.
    pub(crate) fn position(&self) -> u32 {
        self.out.len() as u32
    }

    /// The number of instructions emitted so far.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Emits one byte.
    #[inline]
    fn byte(&mut self, value: u8) {
        self.out.push(value);
    }

    /// Emits the extension prefix, when one is needed.
    ///
    /// `r` and `b` are the extension bits of the register operands;
    /// `force` asks for the prefix even when no bit is set, which the
    /// byte forms need when the operand could be mistaken for one of
    /// the four high-byte registers.
    #[inline]
    fn rex(&mut self, w: bool, r: u8, b: u8, force: bool) {
        let prefix = 0x40 | (u8::from(w) << 3) | ((r & 1) << 2) | (b & 1);
        if prefix != 0x40 || force {
            self.byte(prefix);
        }
    }

    /// The encoding field and the extension bit of a register.
    #[inline]
    fn parts(reg: u16) -> (u8, u8) {
        if reg::is_xmm(reg) {
            let number = (reg - reg::XMM_BASE) as u8;
            (number & 7, (number >> 3) & 1)
        } else {
            ((reg & 7) as u8, ((reg >> 3) & 1) as u8)
        }
    }

    /// Whether the register needs the extension prefix to keep the
    /// byte form away from the high-byte registers.
    #[inline]
    fn byte_needs_rex(reg: u16) -> bool {
        let number = reg & 15;
        (4..=7).contains(&number)
    }

    /// Emits the extension prefix of an instruction that reads or
    /// writes memory.
    #[inline]
    fn rex_mem(&mut self, w: bool, reg: u16, base: u16, force: bool) {
        let reg_ext = Self::parts(reg).1;
        let base_ext = ((base >> 3) & 1) as u8;
        self.rex(w, reg_ext, base_ext, force);
    }

    /// Emits the mode byte of a register-to-register instruction.
    #[inline]
    fn modrm_rr(&mut self, reg: u8, rm: u8) {
        self.byte(0xC0 | ((reg & 7) << 3) | (rm & 7));
    }

    /// Emits the mode byte, the scale-index-base byte, and the
    /// displacement of an instruction that reads or writes memory.
    ///
    /// The base register is always a general-purpose one; no
    /// instruction of the back end needs an index register.
    fn modrm_mem(&mut self, reg: u8, base: u16, disp: i32) {
        let rm = (base & 7) as u8;
        let needs_sib = rm == 4;
        let (mode, disp_size) = if disp == 0 && rm != 5 {
            (0u8, 0u8)
        } else if (-128..=127).contains(&disp) {
            (1, 1)
        } else {
            (2, 4)
        };
        self.byte((mode << 6) | ((reg & 7) << 3) | rm);
        if needs_sib {
            // A scale of one, no index register, and the base register
            // itself.
            self.byte(0x24);
        }
        match disp_size {
            1 => self.byte(disp as i8 as u8),
            4 => {
                let bytes = (disp as u32).to_le_bytes();
                self.out.extend_from_slice(&bytes);
            }
            _ => {}
        }
    }

    /// Emits a move between two general-purpose registers.
    pub(crate) fn mov_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        if size.is_word() {
            self.byte(0x66);
        }
        self.rex(size.rex_w(), src_ext, dst_ext, false);
        self.byte(0x89);
        self.modrm_rr(src_field, dst_field);
    }

    /// Emits a move of an immediate into a general-purpose register.
    ///
    /// A sixty-four-bit register takes the immediate in the form that
    /// widens a thirty-two-bit value when the immediate is small
    /// enough, and the wide form otherwise.
    pub(crate) fn mov_ri(&mut self, size: Size, dst: u16, imm: i64) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        match size {
            Size::Qword => {
                if (i32::MIN as i64..=i32::MAX as i64).contains(&imm) {
                    self.rex(true, 0, dst_ext, false);
                    self.byte(0xC7);
                    self.modrm_rr(0, dst_field);
                    let bytes = (imm as i32).to_le_bytes();
                    self.out.extend_from_slice(&bytes);
                } else {
                    self.rex(true, 0, dst_ext, false);
                    self.byte(0xB8 + dst_field);
                    let bytes = imm.to_le_bytes();
                    self.out.extend_from_slice(&bytes);
                }
            }
            Size::Dword => {
                self.rex(false, 0, dst_ext, false);
                self.byte(0xB8 + dst_field);
                let bytes = (imm as i32).to_le_bytes();
                self.out.extend_from_slice(&bytes);
            }
            _ => {}
        }
    }

    /// Emits a load of memory into a general-purpose register.
    pub(crate) fn mov_rm(&mut self, size: Size, dst: u16, base: u16, disp: i32) {
        self.count += 1;
        let (dst_field, _) = Self::parts(dst);
        let force = size == Size::Byte && Self::byte_needs_rex(dst);
        if size.is_word() {
            self.byte(0x66);
        }
        self.rex_mem(size.rex_w(), dst, base, force);
        self.byte(0x8B);
        self.modrm_mem(dst_field, base, disp);
    }

    /// Emits a store of a general-purpose register into memory.
    pub(crate) fn mov_mr(&mut self, size: Size, base: u16, disp: i32, src: u16) {
        self.count += 1;
        let (src_field, _) = Self::parts(src);
        let force = size == Size::Byte && Self::byte_needs_rex(src);
        if size.is_word() {
            self.byte(0x66);
        }
        self.rex_mem(size.rex_w(), src, base, force);
        self.byte(0x89);
        self.modrm_mem(src_field, base, disp);
    }

    /// Emits a widening load of memory into a general-purpose register.
    pub(crate) fn movzx_rm(&mut self, src_size: Size, dst_size: Size, dst: u16, base: u16, disp: i32) {
        self.count += 1;
        let (dst_field, _) = Self::parts(dst);
        let force = src_size == Size::Byte && Self::byte_needs_rex(dst);
        self.rex_mem(dst_size.rex_w(), dst, base, force);
        self.byte(0x0F);
        self.byte(if src_size == Size::Byte { 0xB6 } else { 0xB7 });
        self.modrm_mem(dst_field, base, disp);
    }

    /// Emits a widening move between two general-purpose registers.
    pub(crate) fn movzx_rr(&mut self, src_size: Size, dst_size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        let (src_field, src_ext) = Self::parts(src);
        let force = src_size == Size::Byte && Self::byte_needs_rex(src);
        self.rex(dst_size.rex_w(), dst_ext, src_ext, force);
        self.byte(0x0F);
        self.byte(if src_size == Size::Byte { 0xB6 } else { 0xB7 });
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits a widening move between two general-purpose registers
    /// that keeps the sign.
    pub(crate) fn movsx_rr(&mut self, src_size: Size, dst_size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        let (src_field, src_ext) = Self::parts(src);
        let force = src_size == Size::Byte && Self::byte_needs_rex(src);
        self.rex(dst_size.rex_w(), dst_ext, src_ext, force);
        self.byte(0x0F);
        self.byte(if src_size == Size::Byte { 0xBE } else { 0xBF });
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits a move that widens a thirty-two-bit value to sixty-four
    /// bits, keeping the sign.
    pub(crate) fn movsxd_rr(&mut self, dst: u16, src: u16) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        let (src_field, src_ext) = Self::parts(src);
        self.rex(true, dst_ext, src_ext, false);
        self.byte(0x63);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits an arithmetic instruction between two general-purpose
    /// registers.
    ///
    /// `opcode` is the instruction's own byte; the destination is the
    /// register-to-register operand.
    fn alu_rr(&mut self, opcode: u8, size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        if size.is_word() {
            self.byte(0x66);
        }
        self.rex(size.rex_w(), src_ext, dst_ext, false);
        self.byte(opcode);
        self.modrm_rr(src_field, dst_field);
    }

    /// Emits an addition of two general-purpose registers.
    pub(crate) fn add_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.alu_rr(0x01, size, dst, src);
    }

    /// Emits a subtraction of two general-purpose registers.
    pub(crate) fn sub_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.alu_rr(0x29, size, dst, src);
    }

    /// Emits a bitwise and of two general-purpose registers.
    pub(crate) fn and_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.alu_rr(0x21, size, dst, src);
    }

    /// Emits a bitwise or of two general-purpose registers.
    pub(crate) fn or_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.alu_rr(0x09, size, dst, src);
    }

    /// Emits a bitwise exclusive or of two general-purpose registers.
    pub(crate) fn xor_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.alu_rr(0x31, size, dst, src);
    }

    /// Emits a comparison of two general-purpose registers.
    pub(crate) fn cmp_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.alu_rr(0x39, size, dst, src);
    }

    /// Emits a test of two general-purpose registers.
    pub(crate) fn test_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.alu_rr(0x85, size, dst, src);
    }

    /// Emits an arithmetic instruction of a register and an immediate.
    ///
    /// `digit` is the instruction's own digit: five for a subtraction,
    /// four for an and, and so on.
    fn alu_ri(&mut self, digit: u8, size: Size, dst: u16, imm: i64) {
        if size == Size::Qword && !(i32::MIN as i64..=i32::MAX as i64).contains(&imm) {
            // The immediate does not fit in the thirty-two-bit form of
            // the instruction, so it moves into the accumulator first
            // and the register form does the work.
            self.mov_ri(Size::Qword, reg::RAX, imm);
            let opcode = match digit {
                0 => 0x01,
                1 => 0x09,
                4 => 0x21,
                5 => 0x29,
                6 => 0x31,
                _ => 0x39,
            };
            self.alu_rr(opcode, size, dst, reg::RAX);
            return;
        }
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        self.rex(size.rex_w(), 0, dst_ext, false);
        if (-128..=127).contains(&imm) {
            self.byte(0x83);
            self.modrm_rr(digit, dst_field);
            self.byte(imm as i8 as u8);
        } else {
            self.byte(0x81);
            self.modrm_rr(digit, dst_field);
            let bytes = (imm as i32).to_le_bytes();
            self.out.extend_from_slice(&bytes);
        }
    }

    /// Emits an addition of an immediate to a register.
    pub(crate) fn add_ri(&mut self, size: Size, dst: u16, imm: i64) {
        self.alu_ri(0, size, dst, imm);
    }

    /// Emits a subtraction of an immediate from a register.
    pub(crate) fn sub_ri(&mut self, size: Size, dst: u16, imm: i64) {
        self.alu_ri(5, size, dst, imm);
    }

    /// Emits a bitwise and of an immediate with a register.
    pub(crate) fn and_ri(&mut self, size: Size, dst: u16, imm: i64) {
        self.alu_ri(4, size, dst, imm);
    }

    /// Emits a bitwise or of an immediate with a register.
    pub(crate) fn or_ri(&mut self, size: Size, dst: u16, imm: i64) {
        self.alu_ri(1, size, dst, imm);
    }

    /// Emits a bitwise exclusive or of an immediate with a register.
    pub(crate) fn xor_ri(&mut self, size: Size, dst: u16, imm: i64) {
        self.alu_ri(6, size, dst, imm);
    }

    /// Emits the comparison of a register and an immediate.
    pub(crate) fn cmp_ri(&mut self, size: Size, dst: u16, imm: i64) {
        self.alu_ri(7, size, dst, imm);
    }

    /// Emits a multiplication of two general-purpose registers.
    pub(crate) fn imul_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        let (src_field, src_ext) = Self::parts(src);
        self.rex(size.rex_w(), dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0xAF);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits a multiplication of a register and an immediate.
    pub(crate) fn imul_ri(&mut self, size: Size, dst: u16, src: u16, imm: i32) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        let (src_field, src_ext) = Self::parts(src);
        self.rex(size.rex_w(), dst_ext, src_ext, false);
        self.byte(0x69);
        self.modrm_rr(dst_field, src_field);
        let bytes = imm.to_le_bytes();
        self.out.extend_from_slice(&bytes);
    }

    /// Emits a division of a register pair by a register.
    ///
    /// The dividend is the accumulator widened into the data register;
    /// the quotient lands in the accumulator and the remainder in the
    /// data register.
    pub(crate) fn div_rr(&mut self, size: Size, divisor: u16, signed: bool) {
        self.count += 1;
        let (div_field, div_ext) = Self::parts(divisor);
        self.rex(size.rex_w(), 0, div_ext, false);
        self.byte(0xF7);
        self.modrm_rr(if signed { 7 } else { 6 }, div_field);
    }

    /// Emits the widening of the accumulator into the data register
    /// that precedes a signed division.
    pub(crate) fn cdq(&mut self, size: Size) {
        self.count += 1;
        self.rex(size.rex_w(), 0, 0, false);
        self.byte(0x99);
    }

    /// Emits a negation of a register.
    pub(crate) fn neg_rr(&mut self, size: Size, dst: u16) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        self.rex(size.rex_w(), 0, dst_ext, false);
        self.byte(0xF7);
        self.modrm_rr(3, dst_field);
    }

    /// Emits a bitwise complement of a register.
    pub(crate) fn not_rr(&mut self, size: Size, dst: u16) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        self.rex(size.rex_w(), 0, dst_ext, false);
        self.byte(0xF7);
        self.modrm_rr(2, dst_field);
    }

    /// Emits a shift of a register by a constant.
    ///
    /// `digit` is the instruction's own digit: four for a left shift,
    /// five for a logical right shift, and seven for an arithmetic
    /// one.
    pub(crate) fn shift_ri(&mut self, digit: u8, size: Size, dst: u16, amount: u8) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        self.rex(size.rex_w(), 0, dst_ext, false);
        self.byte(0xC1);
        self.modrm_rr(digit, dst_field);
        self.byte(amount);
    }

    /// Emits a shift of a register by the counter.
    pub(crate) fn shift_cl(&mut self, digit: u8, size: Size, dst: u16) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        self.rex(size.rex_w(), 0, dst_ext, false);
        self.byte(0xD3);
        self.modrm_rr(digit, dst_field);
    }

    /// Emits the materialization of a condition as a byte.
    pub(crate) fn setcc(&mut self, cc: Cc, dst: u16) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        self.rex(false, 0, dst_ext, Self::byte_needs_rex(dst));
        self.byte(0x0F);
        self.byte(0x90 + cc.bits());
        self.modrm_rr(dst_field, dst_field);
    }

    /// Emits a conditional move between two general-purpose registers.
    ///
    /// The instruction moves the source into the destination when the
    /// condition holds, and leaves the destination alone otherwise.
    pub(crate) fn cmovcc(&mut self, cc: Cc, size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (dst_field, dst_ext) = Self::parts(dst);
        let (src_field, src_ext) = Self::parts(src);
        self.rex(size.rex_w(), dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x40 + cc.bits());
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the computation of the address of memory.
    pub(crate) fn lea(&mut self, dst: u16, base: u16, disp: i32) {
        self.count += 1;
        let (dst_field, _) = Self::parts(dst);
        self.rex_mem(true, dst, base, false);
        self.byte(0x8D);
        self.modrm_mem(dst_field, base, disp);
    }

    /// Emits the push of a register.
    pub(crate) fn push(&mut self, reg: u16) {
        self.count += 1;
        let (field, ext) = Self::parts(reg);
        self.rex(false, 0, ext, false);
        self.byte(0x50 + field);
    }

    /// Emits the pop of a register.
    pub(crate) fn pop(&mut self, reg: u16) {
        self.count += 1;
        let (field, ext) = Self::parts(reg);
        self.rex(false, 0, ext, false);
        self.byte(0x58 + field);
    }

    /// Emits a call whose displacement the layout fills in.
    ///
    /// The method returns the offset of the displacement field.
    pub(crate) fn call_rel32(&mut self) -> u32 {
        self.count += 1;
        self.byte(0xE8);
        let site = self.position();
        let bytes = 0i32.to_le_bytes();
        self.out.extend_from_slice(&bytes);
        site
    }

    /// Emits an unconditional jump whose displacement the layout fills
    /// in.
    ///
    /// The method returns the offset of the displacement field.
    pub(crate) fn jmp_rel32(&mut self) -> u32 {
        self.count += 1;
        self.byte(0xE9);
        let site = self.position();
        let bytes = 0i32.to_le_bytes();
        self.out.extend_from_slice(&bytes);
        site
    }

    /// Emits a conditional jump whose displacement the layout fills in.
    ///
    /// The method returns the offset of the displacement field.
    pub(crate) fn jcc_rel32(&mut self, cc: Cc) -> u32 {
        self.count += 1;
        self.byte(0x0F);
        self.byte(0x80 + cc.bits());
        let site = self.position();
        let bytes = 0i32.to_le_bytes();
        self.out.extend_from_slice(&bytes);
        site
    }

    /// Emits a return from a function.
    pub(crate) fn ret(&mut self) {
        self.count += 1;
        self.byte(0xC3);
    }

    /// Emits an instruction that traps when it is reached.
    pub(crate) fn ud2(&mut self) {
        self.count += 1;
        self.byte(0x0F);
        self.byte(0x0B);
    }

    /// Emits the legacy prefix of a scalar floating-point operation.
    #[inline]
    fn scalar_prefix(&mut self, size: Size) {
        match size {
            Size::Dword => self.byte(0xF3),
            Size::Qword => self.byte(0xF2),
            _ => {}
        }
    }

    /// Emits a move between two SSE registers.
    pub(crate) fn mov_xmm_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.scalar_prefix(size);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x10);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits a load of memory into an SSE register.
    pub(crate) fn mov_xmm_rm(&mut self, size: Size, dst: u16, base: u16, disp: i32) {
        self.count += 1;
        let (dst_field, _) = Self::parts(dst);
        self.scalar_prefix(size);
        self.rex_mem(false, dst, base, false);
        self.byte(0x0F);
        self.byte(0x10);
        self.modrm_mem(dst_field, base, disp);
    }

    /// Emits a store of an SSE register into memory.
    pub(crate) fn mov_xmm_mr(&mut self, size: Size, base: u16, disp: i32, src: u16) {
        self.count += 1;
        let (src_field, _) = Self::parts(src);
        self.scalar_prefix(size);
        self.rex_mem(false, src, base, false);
        self.byte(0x0F);
        self.byte(0x11);
        self.modrm_mem(src_field, base, disp);
    }

    /// Emits an arithmetic instruction between two SSE registers.
    pub(crate) fn arith_xmm_rr(&mut self, arith: Arith, size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.scalar_prefix(size);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(arith.opcode());
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the square root of an SSE register.
    ///
    /// The machine IR has no operation that asks for a square root
    /// yet; the encoder knows the instruction all the same.
    #[allow(dead_code)]
    pub(crate) fn sqrt_xmm_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.scalar_prefix(size);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x51);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the comparison of two SSE registers.
    ///
    /// The flags the comparison leaves behind follow the ordered
    /// rules: the parity flag is set when either operand is not a
    /// number.
    pub(crate) fn ucomis_rr(&mut self, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x2E);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the conversion of a general-purpose register into an SSE
    /// register.
    pub(crate) fn cvtsi2_xmm(&mut self, to_double: bool, src_size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        if to_double {
            self.byte(0xF2);
        } else {
            self.byte(0xF3);
        }
        self.rex(src_size.rex_w(), dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x2A);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the conversion of an SSE register into a general-purpose
    /// register, truncating towards zero.
    pub(crate) fn cvtts_xmm2si(&mut self, to_double: bool, dst_size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        if to_double {
            self.byte(0xF2);
        } else {
            self.byte(0xF3);
        }
        self.rex(dst_size.rex_w(), dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x2C);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the conversion of an SSE register between the two
    /// floating-point widths.
    pub(crate) fn cvt_xmm2xmm(&mut self, to_double: bool, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        if to_double {
            self.byte(0xF3);
        } else {
            self.byte(0xF2);
        }
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x5A);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the move of the low half of a general-purpose register
    /// into an SSE register.
    pub(crate) fn movd_xmm_rm(&mut self, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.byte(0x66);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x6E);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the move of a general-purpose register into an SSE
    /// register.
    pub(crate) fn movq_xmm_rm(&mut self, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.byte(0x66);
        self.rex(true, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x6E);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the move of the low half of an SSE register into a
    /// general-purpose register.
    ///
    /// The instruction takes the vector register in the register field
    /// and the general-purpose one in the memory field, so the
    /// extension bits follow the fields.
    pub(crate) fn movd_rm_xmm(&mut self, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.byte(0x66);
        self.rex(false, src_ext, dst_ext, false);
        self.byte(0x0F);
        self.byte(0x7E);
        self.modrm_rr(src_field, dst_field);
    }

    /// Emits the move of an SSE register into a general-purpose
    /// register.
    pub(crate) fn movq_rm_xmm(&mut self, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.byte(0x66);
        self.rex(true, src_ext, dst_ext, false);
        self.byte(0x0F);
        self.byte(0x7E);
        self.modrm_rr(src_field, dst_field);
    }

    /// Emits the selection of the thirty-two-bit lanes of an SSE
    /// register.
    pub(crate) fn pshufd(&mut self, dst: u16, src: u16, selector: u8) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.byte(0x66);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x70);
        self.modrm_rr(dst_field, src_field);
        self.byte(selector);
    }

    /// Emits the selection of the sixteen-bit lanes of an SSE
    /// register.
    pub(crate) fn pshuflw(&mut self, dst: u16, src: u16, selector: u8) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.byte(0xF2);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x70);
        self.modrm_rr(dst_field, src_field);
        self.byte(selector);
    }

    /// Emits the interleaving of the low quad-words of two SSE
    /// registers.
    ///
    /// The baseline form of the instruction takes no immediate: it
    /// interleaves the low quad-word of the destination with the low
    /// quad-word of the source.
    pub(crate) fn punpcklqdq(&mut self, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.byte(0x66);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x6C);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the interleaving of the high quad-words of two SSE
    /// registers.
    pub(crate) fn punpckhqdq(&mut self, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.byte(0x66);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0x6D);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the extraction of a sixteen-bit lane into a
    /// general-purpose register.
    pub(crate) fn pextrw(&mut self, dst: u16, src: u16, lane: u8) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.byte(0x66);
        self.rex(false, src_ext, dst_ext, false);
        self.byte(0x0F);
        self.byte(0xC5);
        self.modrm_rr(src_field, dst_field);
        self.byte(lane);
    }

    /// Emits the insertion of a sixteen-bit lane from a
    /// general-purpose register.
    pub(crate) fn pinsrw(&mut self, dst: u16, src: u16, lane: u8) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.byte(0x66);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(0xC4);
        self.modrm_rr(dst_field, src_field);
        self.byte(lane);
    }

    /// Emits a bitwise operation between two SSE registers.
    fn logic_xmm_rr(&mut self, opcode: u8, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(opcode);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits the smaller of two SSE registers.
    ///
    /// The machine IR has no operation that asks for a minimum yet;
    /// the encoder knows the instruction all the same.
    #[allow(dead_code)]
    pub(crate) fn min_xmm_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.sse_rr(0x5D, size, dst, src);
    }

    /// Emits the larger of two SSE registers.
    ///
    /// The machine IR has no operation that asks for a maximum yet;
    /// the encoder knows the instruction all the same.
    #[allow(dead_code)]
    pub(crate) fn max_xmm_rr(&mut self, size: Size, dst: u16, src: u16) {
        self.sse_rr(0x5F, size, dst, src);
    }

    /// Emits a scalar floating-point instruction between two SSE
    /// registers.
    fn sse_rr(&mut self, opcode: u8, size: Size, dst: u16, src: u16) {
        self.count += 1;
        let (src_field, src_ext) = Self::parts(src);
        let (dst_field, dst_ext) = Self::parts(dst);
        self.scalar_prefix(size);
        self.rex(false, dst_ext, src_ext, false);
        self.byte(0x0F);
        self.byte(opcode);
        self.modrm_rr(dst_field, src_field);
    }

    /// Emits a bitwise and of two SSE registers.
    pub(crate) fn andps_rr(&mut self, dst: u16, src: u16) {
        self.logic_xmm_rr(0x54, dst, src);
    }

    /// Emits a bitwise and of the complement of a mask with an SSE
    /// register.
    pub(crate) fn andnps_rr(&mut self, dst: u16, src: u16) {
        self.logic_xmm_rr(0x55, dst, src);
    }

    /// Emits a bitwise or of two SSE registers.
    pub(crate) fn orps_rr(&mut self, dst: u16, src: u16) {
        self.logic_xmm_rr(0x56, dst, src);
    }

    /// Emits a bitwise exclusive or of two SSE registers.
    pub(crate) fn xorps_rr(&mut self, dst: u16, src: u16) {
        self.logic_xmm_rr(0x57, dst, src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn bytes_of(f: impl FnOnce(&mut Encoder<'_>)) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = Encoder::new(&mut out);
            f(&mut encoder);
        }
        out
    }

    #[test]
    fn a_move_between_registers_has_the_expected_bytes() {
        assert_eq!(
            bytes_of(|e| e.mov_rr(Size::Qword, reg::RAX, reg::RDI)),
            vec![0x48, 0x89, 0xF8]
        );
        assert_eq!(
            bytes_of(|e| e.mov_rr(Size::Dword, reg::RAX, reg::RDI)),
            vec![0x89, 0xF8]
        );
        assert_eq!(
            bytes_of(|e| e.mov_rr(Size::Qword, reg::RAX, reg::R15)),
            vec![0x4C, 0x89, 0xF8]
        );
    }

    #[test]
    fn an_addition_between_registers_has_the_expected_bytes() {
        assert_eq!(
            bytes_of(|e| e.add_rr(Size::Qword, reg::RAX, reg::RBX)),
            vec![0x48, 0x01, 0xD8]
        );
        assert_eq!(
            bytes_of(|e| e.add_rr(Size::Dword, reg::RAX, reg::RBX)),
            vec![0x01, 0xD8]
        );
    }

    #[test]
    fn a_subtraction_of_an_immediate_has_the_expected_bytes() {
        assert_eq!(
            bytes_of(|e| e.sub_ri(Size::Qword, reg::RSP, 0x20)),
            vec![0x48, 0x83, 0xEC, 0x20]
        );
        assert_eq!(
            bytes_of(|e| e.sub_ri(Size::Qword, reg::RSP, 0x100)),
            vec![0x48, 0x81, 0xEC, 0x00, 0x01, 0x00, 0x00]
        );
    }

    #[test]
    fn the_stack_helpers_have_the_expected_bytes() {
        assert_eq!(bytes_of(|e| e.push(reg::RBP)), vec![0x55]);
        assert_eq!(bytes_of(|e| e.push(reg::R12)), vec![0x41, 0x54]);
        assert_eq!(bytes_of(|e| e.pop(reg::R12)), vec![0x41, 0x5C]);
        assert_eq!(bytes_of(|e| e.ret()), vec![0xC3]);
    }

    #[test]
    fn a_move_of_an_immediate_has_the_expected_bytes() {
        assert_eq!(
            bytes_of(|e| e.mov_ri(Size::Dword, reg::RAX, 5)),
            vec![0xB8, 0x05, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            bytes_of(|e| e.mov_ri(Size::Qword, reg::RAX, -1)),
            vec![0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF]
        );
        assert_eq!(
            bytes_of(|e| e.mov_ri(Size::Qword, reg::RAX, 0x1_0000_0000)),
            vec![0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn the_memory_forms_use_the_shortest_displacement() {
        assert_eq!(
            bytes_of(|e| e.mov_rm(Size::Qword, reg::RAX, reg::RDI, 8)),
            vec![0x48, 0x8B, 0x47, 0x08]
        );
        assert_eq!(
            bytes_of(|e| e.mov_rm(Size::Qword, reg::RAX, reg::RDI, 0)),
            vec![0x48, 0x8B, 0x07]
        );
        assert_eq!(
            bytes_of(|e| e.mov_mr(Size::Qword, reg::RSP, 0x20, reg::RBX)),
            vec![0x48, 0x89, 0x5C, 0x24, 0x20]
        );
        assert_eq!(
            bytes_of(|e| e.mov_mr(Size::Qword, reg::RBP, 0, reg::RAX)),
            vec![0x48, 0x89, 0x45, 0x00]
        );
        assert_eq!(
            bytes_of(|e| e.mov_mr(Size::Word, reg::RSP, 0, reg::RAX)),
            vec![0x66, 0x89, 0x04, 0x24]
        );
    }

    #[test]
    fn the_widening_forms_have_the_expected_bytes() {
        assert_eq!(
            bytes_of(|e| e.movzx_rm(Size::Byte, Size::Dword, reg::RAX, reg::RDI, 0)),
            vec![0x0F, 0xB6, 0x07]
        );
        assert_eq!(
            bytes_of(|e| e.movzx_rm(Size::Word, Size::Dword, reg::RAX, reg::RDI, 0)),
            vec![0x0F, 0xB7, 0x07]
        );
        assert_eq!(
            bytes_of(|e| e.movsxd_rr(reg::RAX, reg::RDI)),
            vec![0x48, 0x63, 0xC7]
        );
    }

    #[test]
    fn the_division_forms_have_the_expected_bytes() {
        assert_eq!(
            bytes_of(|e| e.div_rr(Size::Qword, reg::RCX, true)),
            vec![0x48, 0xF7, 0xF9]
        );
        assert_eq!(
            bytes_of(|e| e.div_rr(Size::Qword, reg::RCX, false)),
            vec![0x48, 0xF7, 0xF1]
        );
        assert_eq!(bytes_of(|e| e.cdq(Size::Dword)), vec![0x99]);
        assert_eq!(bytes_of(|e| e.cdq(Size::Qword)), vec![0x48, 0x99]);
    }

    #[test]
    fn the_condition_forms_have_the_expected_bytes() {
        assert_eq!(bytes_of(|e| e.setcc(Cc::Ne, reg::RAX)), vec![0x0F, 0x95, 0xC0]);
        assert_eq!(bytes_of(|e| e.setcc(Cc::Ne, reg::RDX)), vec![0x0F, 0x95, 0xD2]);
        assert_eq!(
            bytes_of(|e| e.cmovcc(Cc::Ne, Size::Qword, reg::RAX, reg::RBX)),
            vec![0x48, 0x0F, 0x45, 0xC3]
        );
        assert_eq!(
            bytes_of(|e| e.test_rr(Size::Qword, reg::RAX, reg::RAX)),
            vec![0x48, 0x85, 0xC0]
        );
    }

    #[test]
    fn the_address_form_has_the_expected_bytes() {
        assert_eq!(
            bytes_of(|e| e.lea(reg::RAX, reg::RSP, 16)),
            vec![0x48, 0x8D, 0x44, 0x24, 0x10]
        );
        assert_eq!(
            bytes_of(|e| e.lea(reg::RAX, reg::RSP, 0)),
            vec![0x48, 0x8D, 0x04, 0x24]
        );
    }

    #[test]
    fn the_branch_forms_leave_room_for_the_displacement() {
        assert_eq!(
            bytes_of(|e| {
                e.jmp_rel32();
            })
            .len(),
            5
        );
        assert_eq!(
            bytes_of(|e| {
                e.jcc_rel32(Cc::Ne);
            }),
            vec![0x0F, 0x85, 0, 0, 0, 0]
        );
        assert_eq!(
            bytes_of(|e| {
                e.call_rel32();
            }),
            vec![0xE8, 0, 0, 0, 0]
        );
    }

    #[test]
    fn the_vector_forms_have_the_expected_bytes() {
        assert_eq!(
            bytes_of(|e| e.mov_xmm_rm(Size::Dword, reg::XMM0, reg::RDI, 0)),
            vec![0xF3, 0x0F, 0x10, 0x07]
        );
        assert_eq!(
            bytes_of(|e| e.arith_xmm_rr(Arith::Add, Size::Qword, reg::XMM1, reg::XMM0)),
            vec![0xF2, 0x0F, 0x58, 0xC8]
        );
        assert_eq!(
            bytes_of(|e| e.pshufd(reg::XMM0, reg::XMM1, 0)),
            vec![0x66, 0x0F, 0x70, 0xC1, 0x00]
        );
        assert_eq!(
            bytes_of(|e| e.movq_xmm_rm(reg::XMM0, reg::RAX)),
            vec![0x66, 0x48, 0x0F, 0x6E, 0xC0]
        );
        assert_eq!(
            bytes_of(|e| e.punpcklqdq(reg::XMM0, reg::XMM0)),
            vec![0x66, 0x0F, 0x6C, 0xC0]
        );
        assert_eq!(
            bytes_of(|e| e.pinsrw(reg::XMM0, reg::RAX, 2)),
            vec![0x66, 0x0F, 0xC4, 0xC0, 0x02]
        );
    }
}
