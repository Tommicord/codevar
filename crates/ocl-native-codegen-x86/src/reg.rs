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

//! The register file of the target.
//!
//! The general-purpose registers are numbered the way the
//! architecture's encoding numbers them, so the low three bits of a
//! number are the register's encoding field and the fourth bit is the
//! extension bit that selects the second group of eight.  The SSE
//! registers follow the general-purpose ones, starting at
//! [`XMM_BASE`], so a register number alone tells whether the value it
//! holds is an integer or a floating-point one.
//!
//! The numbering is the one the whole back end uses: the register
//! model, the encoder, the expander, and the layout all speak it.

/// The number of the first SSE register.
///
/// The sixteen general-purpose registers occupy the numbers below this
/// one, and the sixteen SSE registers the numbers from it upwards.
pub(crate) const XMM_BASE: u16 = 16;

/// The number of general-purpose registers.
pub(crate) const NUM_GPR: u16 = 16;

/// The number of SSE registers.
pub(crate) const NUM_XMM: u16 = 16;

/// The number of registers in the file.
pub(crate) const TOTAL_REGS: u16 = 32;

/// The accumulator.
pub(crate) const RAX: u16 = 0;
/// The counter.
pub(crate) const RCX: u16 = 1;
/// The data register.
pub(crate) const RDX: u16 = 2;
/// The base register.
pub(crate) const RBX: u16 = 3;
/// The stack pointer.
pub(crate) const RSP: u16 = 4;
/// The frame pointer.
pub(crate) const RBP: u16 = 5;
/// The source index.
pub(crate) const RSI: u16 = 6;
/// The destination index.
pub(crate) const RDI: u16 = 7;
/// The first extended register.
pub(crate) const R8: u16 = 8;
/// The second extended register.
pub(crate) const R9: u16 = 9;
/// The third extended register.
pub(crate) const R10: u16 = 10;
/// The fourth extended register.
pub(crate) const R11: u16 = 11;
/// The fifth extended register.
pub(crate) const R12: u16 = 12;
/// The sixth extended register.
pub(crate) const R13: u16 = 13;
/// The seventh extended register.
pub(crate) const R14: u16 = 14;
/// The eighth extended register.
pub(crate) const R15: u16 = 15;

/// The first SSE register.
pub(crate) const XMM0: u16 = XMM_BASE;
/// The second SSE register.
pub(crate) const XMM1: u16 = XMM_BASE + 1;
/// The second-to-last SSE register.
pub(crate) const XMM14: u16 = XMM_BASE + 14;
/// The last SSE register.
pub(crate) const XMM15: u16 = XMM_BASE + 15;

/// The registers integer arguments arrive in, in argument order.
pub(crate) const ARG_REGS: [u16; 6] = [RDI, RSI, RDX, RCX, R8, R9];

/// The registers floating-point arguments arrive in, in argument order.
pub(crate) const FLOAT_ARG_REGS: [u16; 8] = [
    XMM0,
    XMM0 + 1,
    XMM0 + 2,
    XMM0 + 3,
    XMM0 + 4,
    XMM0 + 5,
    XMM0 + 6,
    XMM0 + 7,
];

/// The registers whose value survives a call.
pub(crate) const CALLEE_SAVED: [u16; 6] = [RBX, RBP, R12, R13, R14, R15];

/// The registers the expander may use as scratch.
///
/// None of them is ever handed out to the allocator, so a value the
/// expander writes into one of them cannot be live.  The argument
/// registers are in the set because the calling convention reserves
/// them for the arguments of a call, and the accumulator because the
/// calling convention reserves it for the result of a call.
pub(crate) const SCRATCH_GPRS: [u16; 7] = [RAX, RCX, RDX, RDI, RSI, R8, R9];

/// The registers that keep the stack aligned when every callee-saved
/// register is already saved.
///
/// They are only saved when the function needs one more of them, and
/// saving a register the function does not use changes nothing.
pub(crate) const PADDING_REGS: [u16; 2] = [R10, R11];

/// The SSE registers the expander may use as scratch.
///
/// They are the last two of the file, and the register model does not
/// offer them to the allocator.
pub(crate) const SCRATCH_XMMS: [u16; 2] = [XMM14, XMM15];

/// Whether the register holds a floating-point or a vector value.
#[inline]
pub(crate) const fn is_xmm(reg: u16) -> bool {
    reg >= XMM_BASE
}

/// Whether the number names a register of the file.
#[inline]
pub(crate) const fn is_valid(reg: u16) -> bool {
    reg < TOTAL_REGS
}

/// The name of a general-purpose register.
#[inline]
pub(crate) const fn gpr_name(reg: u16) -> &'static str {
    match reg {
        RAX => "rax",
        RCX => "rcx",
        RDX => "rdx",
        RBX => "rbx",
        RSP => "rsp",
        RBP => "rbp",
        RSI => "rsi",
        RDI => "rdi",
        R8 => "r8",
        R9 => "r9",
        R10 => "r10",
        R11 => "r11",
        R12 => "r12",
        R13 => "r13",
        R14 => "r14",
        R15 => "r15",
        _ => "?",
    }
}

/// The name of an SSE register.
#[inline]
pub(crate) const fn xmm_name(reg: u16) -> &'static str {
    match reg {
        XMM0 => "xmm0",
        XMM1 => "xmm1",
        XMM14 => "xmm14",
        XMM15 => "xmm15",
        _ => "xmm",
    }
}

/// The name of a register, for diagnostics.
#[inline]
pub(crate) const fn name(reg: u16) -> &'static str {
    if is_xmm(reg) { xmm_name(reg) } else { gpr_name(reg) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_general_purpose_registers_are_numbered_in_encoding_order() {
        assert_eq!(gpr_name(RAX), "rax");
        assert_eq!(gpr_name(RBX), "rbx");
        assert_eq!(gpr_name(RSP), "rsp");
        assert_eq!(gpr_name(RBP), "rbp");
        assert_eq!(gpr_name(R15), "r15");
    }

    #[test]
    fn the_sse_registers_follow_the_general_purpose_ones() {
        assert!(is_xmm(XMM0));
        assert!(is_xmm(XMM15));
        assert!(!is_xmm(R15));
        assert_eq!(TOTAL_REGS, NUM_GPR + NUM_XMM);
    }

    #[test]
    fn the_scratch_registers_are_not_offered_to_the_allocator() {
        for reg in SCRATCH_GPRS {
            assert!(!CALLEE_SAVED.contains(&reg));
        }
        for reg in SCRATCH_XMMS {
            assert!(is_xmm(reg));
        }
    }
}
