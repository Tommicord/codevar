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

//! The register file of the architecture.
//!
//! The back end numbers the registers of the machine so that the
//! pipeline can talk about them with a single number:
//!
//! | numbers | registers |
//! |---------|-----------|
//! | 0..=15  | the general-purpose registers `r0`..`r15` |
//! | 16..=47 | the single-precision registers `s0`..`s31` |
//! | 48..=63 | the double-precision registers `d0`..`d15` |
//! | 64..=71 | the quadword registers `q0`..`q7` |
//!
//! The general-purpose registers keep their architectural numbers:
//! `r13` is the stack pointer, `r14` the link register and `r15` the
//! program counter.  The floating-point registers are three views of
//! one file: `d0` is `s0` and `s1`, and `q0` is `d0` and `d1`.
//!
//! The single-precision and double-precision views are *views*: a
//! value of one class lives in the file of the other, so the
//! allocator must never hand out the same number twice across
//! classes.  The back end therefore keeps the single-precision
//! registers for floating-point values and the quadword registers
//! for vector values, and refuses the operations that would need the
//! two views of one register at the same time.

use alloc::string::String;

use codevar_ocl_native_codegen::mir::PReg;

/// The number of the stack pointer.
pub const SP: u8 = 13;

/// The number of the link register.
pub const LR: u8 = 14;

/// The number of the program counter.
pub const PC: u8 = 15;

/// The number of general-purpose registers, the program counter
/// included.
pub const NUM_GPR: u8 = 16;

/// The number of the first single-precision register.
pub const S_BASE: u16 = 16;

/// The number of the first double-precision register.
pub const D_BASE: u16 = 48;

/// The number of the first quadword register.
pub const Q_BASE: u16 = 64;

/// The number of single-precision registers.
pub const NUM_S: u16 = 32;

/// The number of double-precision registers.
pub const NUM_D: u16 = 16;

/// The number of quadword registers.
pub const NUM_Q: u16 = 8;

/// The general-purpose register of the given architectural number.
#[must_use]
#[inline]
pub const fn gpr(number: u8) -> PReg {
    PReg::new(number as u16)
}

/// The single-precision register of the given number.
#[must_use]
#[inline]
pub const fn sreg(number: u8) -> PReg {
    PReg::new(S_BASE + number as u16)
}

/// The double-precision register of the given number.
#[must_use]
#[inline]
pub const fn dreg(number: u8) -> PReg {
    PReg::new(D_BASE + number as u16)
}

/// The quadword register of the given number.
#[must_use]
#[inline]
pub const fn qreg(number: u8) -> PReg {
    PReg::new(Q_BASE + number as u16)
}

/// Whether the register is one of the general-purpose registers.
#[must_use]
#[inline]
pub const fn is_gpr(reg: PReg) -> bool {
    reg.number() < NUM_GPR as u16
}

/// Whether the register is one of the single-precision registers.
#[must_use]
#[inline]
pub const fn is_s(reg: PReg) -> bool {
    reg.number() >= S_BASE && reg.number() < S_BASE + NUM_S
}

/// Whether the register is one of the double-precision registers.
#[must_use]
#[inline]
pub const fn is_d(reg: PReg) -> bool {
    reg.number() >= D_BASE && reg.number() < D_BASE + NUM_D
}

/// Whether the register is one of the quadword registers.
#[must_use]
#[inline]
pub const fn is_q(reg: PReg) -> bool {
    reg.number() >= Q_BASE && reg.number() < Q_BASE + NUM_Q
}

/// The architectural number of the general-purpose register, when the
/// register is one.
#[must_use]
#[inline]
pub const fn gpr_number(reg: PReg) -> Option<u8> {
    if is_gpr(reg) {
        Some(reg.number() as u8)
    } else {
        None
    }
}

/// The number of the single-precision register, when the register is
/// one.
#[must_use]
#[inline]
pub const fn s_number(reg: PReg) -> Option<u8> {
    if is_s(reg) {
        Some((reg.number() - S_BASE) as u8)
    } else {
        None
    }
}

/// The number of the double-precision register, when the register is
/// one.
#[must_use]
#[inline]
pub const fn d_number(reg: PReg) -> Option<u8> {
    if is_d(reg) {
        Some((reg.number() - D_BASE) as u8)
    } else {
        None
    }
}

/// The number of the quadword register, when the register is one.
#[must_use]
#[inline]
pub const fn q_number(reg: PReg) -> Option<u8> {
    if is_q(reg) {
        Some((reg.number() - Q_BASE) as u8)
    } else {
        None
    }
}

/// Writes the name of the register into `out`, for diagnostics.
///
/// The name is the architectural one: `r0`..`r15`, `s0`..`s31`,
/// `d0`..`d15` or `q0`..`q7`.
pub fn write_name(reg: PReg, out: &mut String) {
    if let Some(number) = gpr_number(reg) {
        out.push('r');
        push_number(out, number as u32);
    } else if let Some(number) = s_number(reg) {
        out.push('s');
        push_number(out, number as u32);
    } else if let Some(number) = d_number(reg) {
        out.push('d');
        push_number(out, number as u32);
    } else if let Some(number) = q_number(reg) {
        out.push('q');
        push_number(out, number as u32);
    } else {
        out.push_str("reg");
        push_number(out, u32::from(reg.number()));
    }
}

/// The name of the register, for diagnostics.
#[must_use]
pub fn name(reg: PReg) -> String {
    let mut out = String::new();
    write_name(reg, &mut out);
    out
}

/// Writes a decimal number into `out`.
fn push_number(out: &mut String, value: u32) {
    let mut digits = [0u8; 10];
    let mut index = digits.len();
    let mut rest = value;
    loop {
        index -= 1;
        digits[index] = b'0' + u8::try_from(rest % 10).unwrap_or(0);
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    for digit in &digits[index..] {
        out.push(char::from(*digit));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_views_of_the_file_have_distinct_numbers() {
        assert_eq!(gpr(0).number(), 0);
        assert_eq!(sreg(0).number(), 16);
        assert_eq!(dreg(0).number(), 48);
        assert_eq!(qreg(0).number(), 64);
        assert!(!is_gpr(sreg(0)));
        assert!(is_s(sreg(31)));
        assert!(is_d(dreg(15)));
        assert!(is_q(qreg(7)));
    }

    #[test]
    fn the_names_follow_the_architecture() {
        assert_eq!(name(gpr(13)), "r13");
        assert_eq!(name(sreg(31)), "s31");
        assert_eq!(name(dreg(0)), "d0");
        assert_eq!(name(qreg(7)), "q7");
    }
}
