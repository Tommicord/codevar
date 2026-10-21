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

//! The register file of the AArch64 back end.
//!
//! The architecture has two register files.  The general-purpose
//! file holds thirty-one registers, `x0` to `x30`; the number
//! thirty-one is the stack pointer when an instruction uses it as
//! the base of an access and the zero register otherwise, so the
//! back end never hands it out.  The second file holds the
//! thirty-two SIMD-and-floating-point registers, `v0` to `v31`;
//! every value of a floating-point or vector type lives in one of
//! them, and the width of the access decides whether the
//! instruction reads the single-precision, the double-precision or
//! the full-width view of the register.
//!
//! # The numbering of the physical registers
//!
//! The back end numbers the physical registers of the machine in
//! one space, so that a physical register always names the same
//! architectural register:
//!
//! | Number | Register |
//! |---|---|
//! | `0` to `30` | `x0` to `x30` |
//! | `31` | the stack pointer and the zero register |
//! | `32` to `63` | `v0` to `v31` |
//!
//! The two ranges are disjoint, which lets the back end tell the
//! class of an operand from its number alone: a number below
//! thirty-two is a general-purpose register, a number above it is a
//! SIMD-and-floating-point register.  That matters for the calls,
//! whose arguments are told apart by their class alone.

use alloc::string::String;

use codevar_ocl_native_codegen::mir::PReg;

/// The number the stack pointer and the zero register share.
pub const SP: u8 = 31;

/// The physical number of the first SIMD-and-floating-point
/// register.
pub const SIMD_BASE: u16 = 32;

/// The number of SIMD-and-floating-point registers.
pub const SIMD_COUNT: u8 = 32;

/// The first general-purpose register scratch of the back end.
///
/// The scratch holds a value that has no home yet — a wide
/// immediate, the quotient of a division — while the instruction
/// that consumes it runs.  It is never handed to the allocator, so
/// no live value can be clobbered by it.
pub const SCRATCH_A: u8 = 16;

/// The second general-purpose register scratch of the back end.
pub const SCRATCH_B: u8 = 17;

/// The SIMD-and-floating-point register scratch of the back end.
pub const SIMD_SCRATCH: u8 = 31;

/// The physical number of a general-purpose register.
#[must_use]
#[inline]
pub const fn gpr(index: u8) -> PReg {
    PReg::new(index as u16)
}

/// The physical number of a SIMD-and-floating-point register.
#[must_use]
#[inline]
pub const fn simd(index: u8) -> PReg {
    PReg::new(SIMD_BASE + index as u16)
}

/// Whether the physical register names a SIMD-and-floating-point
/// register.
#[must_use]
#[inline]
pub const fn is_simd(reg: PReg) -> bool {
    reg.number() >= SIMD_BASE
}

/// The architectural number of a general-purpose register.
#[must_use]
#[inline]
pub const fn gpr_index(reg: PReg) -> Option<u8> {
    let number = reg.number();
    if number < SIMD_BASE {
        Some(number as u8)
    } else {
        None
    }
}

/// The architectural number of a SIMD-and-floating-point register.
#[must_use]
#[inline]
pub const fn simd_index(reg: PReg) -> Option<u8> {
    let number = reg.number();
    if number >= SIMD_BASE && number < SIMD_BASE + SIMD_COUNT as u16 {
        Some((number - SIMD_BASE) as u8)
    } else {
        None
    }
}

/// A human-readable name of a physical register, for diagnostics.
#[must_use]
pub fn describe(reg: PReg) -> String {
    match gpr_index(reg) {
        Some(SP) => String::from("sp"),
        Some(index) => {
            let mut name = String::from("x");
            let mut digits = [0u8; 2];
            let mut count = 0usize;
            let mut value = index;
            loop {
                digits[count] = b'0' + (value % 10);
                count += 1;
                value /= 10;
                if value == 0 {
                    break;
                }
            }
            for index in (0..count).rev() {
                name.push(digits[index] as char);
            }
            name
        }
        None => {
            let index = simd_index(reg).unwrap_or(0);
            let mut name = String::from("v");
            let mut digits = [0u8; 2];
            let mut count = 0usize;
            let mut value = index;
            loop {
                digits[count] = b'0' + (value % 10);
                count += 1;
                value /= 10;
                if value == 0 {
                    break;
                }
            }
            for position in (0..count).rev() {
                name.push(digits[position] as char);
            }
            name
        }
    }
}
