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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Portable stack unwinding
//!
//! Captures machine registers with inline assembly, then walks frames using
//! platform unwind tables (`.eh_frame` DWARF CFI on ELF/Mach-O, WinDBG-style
//! `Rtl*` APIs on Windows) with a frame-pointer fallback. No external unwind
//! library is linked.

use core::ffi::c_void;
#[cfg(all(unix, not(target_arch = "wasm32")))]
use libc;

/// Hard cap on the number of frames reported by [`trace`] and
/// [`capture_frames`].
pub const MAX_FRAMES: usize = 256;

/// Page size constant for mincore checks.
#[cfg(all(unix, not(target_arch = "wasm32")))]
const PAGE_SIZE: usize = 4096;

/// Check whether `addr` points to resident, readable memory using `mincore`.
/// This avoids segmentation faults when probing possibly-unmapped addresses during
/// stack unwinding without unsafe file I/O.
#[cfg(all(unix, not(target_arch = "wasm32")))]
#[inline]
fn is_readable(addr: usize) -> bool {
    let page = addr & !(PAGE_SIZE - 1);
    let mut vec = [0u8; 1];
    unsafe {
        // Safety: `page` is page-aligned and `vec` is one valid byte;
        // `mincore` only writes the queried residency bytes and retains no
        // pointer. The pointer type differs per libc (`*mut u8` on Linux,
        // `*mut c_char` on the BSDs/macOS), hence the inferred cast.
        libc::mincore(page as *mut c_void, PAGE_SIZE, vec.as_mut_ptr().cast()) == 0
            && (vec[0] & 1) != 0
    }
}

/// Minimal `MEMORY_BASIC_INFORMATION` as filled in by `VirtualQuery`.
///
/// `#[repr(C)]` layout matches the Win32 structure on both 32- and 64-bit
/// Windows: the optional `PartitionId` field introduced on newer SDKs
/// occupies the padding that follows `allocation_protect` here.
#[cfg(all(windows, not(target_vendor = "uwp")))]
#[repr(C)]
struct MemoryBasicInformation {
    base_address: *mut c_void,
    allocation_base: *mut c_void,
    allocation_protect: u32,
    region_size: usize,
    state: u32,
    protect: u32,
    kind: u32,
}

#[cfg(all(windows, not(target_vendor = "uwp")))]
windows_link::link!(
    "kernel32.dll" "system"
    fn VirtualQuery(
        lp_address: *const c_void,
        lp_buffer: *mut MemoryBasicInformation,
        dw_length: usize,
    ) -> usize
);

/// Check whether `addr` lies in committed, readable memory by querying the
/// VM region that contains it with `VirtualQuery`. Avoids the segmentation
/// fault `IsBadReadPtr` is documented to still raise under race conditions.
#[cfg(all(windows, not(target_vendor = "uwp")))]
#[inline]
fn is_readable(addr: usize) -> bool {
    /// `MEM_COMMIT`: the region holds committed pages.
    const MEM_COMMIT: u32 = 0x1000;
    /// `PAGE_NOACCESS`: no access of any kind is granted.
    const PAGE_NOACCESS: u32 = 0x01;
    /// `PAGE_GUARD`: access raises `EXCEPTION_GUARD_PAGE`.
    const PAGE_GUARD: u32 = 0x100;

    let mut info = MemoryBasicInformation {
        base_address: core::ptr::null_mut(),
        allocation_base: core::ptr::null_mut(),
        allocation_protect: 0,
        region_size: 0,
        state: 0,
        protect: 0,
        kind: 0,
    };
    // Safety: `info` is a valid, correctly-sized out-buffer and `addr` is a
    // plain address; `VirtualQuery` neither retains nor dereferences
    // `lpAddress` itself and returns the number of bytes written.
    let written = unsafe {
        VirtualQuery(
            addr as *const c_void,
            &raw mut info,
            core::mem::size_of::<MemoryBasicInformation>(),
        )
    };
    written != 0
        && info.state == MEM_COMMIT
        && (info.protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0
}

/// Fallback for targets without a residency probe (e.g. `wasm32`, UWP):
/// conservatively reject every address so probing never faults.
#[cfg(not(any(
    all(unix, not(target_arch = "wasm32")),
    all(windows, not(target_vendor = "uwp")),
)))]
#[inline]
fn is_readable(_addr: usize) -> bool {
    false
}

/// A single stack frame yielded by [`trace`].
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    ip: usize,
    sp: usize,
    module_base: Option<usize>,
}

impl Frame {
    /// Instruction pointer (return address) of this frame.
    #[inline]
    pub fn ip(&self) -> usize {
        self.ip
    }

    /// Stack pointer of this frame.
    #[inline]
    pub fn sp(&self) -> usize {
        self.sp
    }

    /// Load address of the module containing [`Frame::ip`], when known.
    #[inline]
    pub fn module_base_address(&self) -> Option<usize> {
        self.module_base
    }

    /// Best-effort symbol start address; defaults to the instruction pointer
    /// until symbolication is available.
    #[inline]
    pub fn symbol_address(&self) -> *mut c_void {
        self.ip as *mut c_void
    }

    /// Constructs a new [`Frame`] for testing and pretty-printing purposes.
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn new(ip: usize, sp: usize, module_base: Option<usize>) -> Self {
        Self {
            ip,
            sp,
            module_base,
        }
    }
}

/// Walks the current thread's call stack, invoking `cb` for each frame from
/// the most recent call outward.
///
/// Return `false` from `cb` to stop the walk. This function never panics and
/// silently stops when unwind information is missing or corrupt. The first
/// frames may lie inside `trace` itself.
pub fn trace(cb: &mut dyn FnMut(&Frame) -> bool) {
    cfg_if::cfg_if! {
        if #[cfg(all(
            windows,
            not(target_vendor = "uwp"),
            any(target_arch = "x86_64", target_arch = "aarch64"),
        ))] {
            windows::trace_inner(cb);
        } else if #[cfg(all(
            unix,
            not(target_vendor = "apple"),
            target_pointer_width = "64",
        ))] {
            elf::trace_inner(cb);
        } else if #[cfg(all(unix, target_vendor = "apple", target_pointer_width = "64"))] {
            apple::trace_inner(cb);
        } else if #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))] {
            fp::trace_inner(cb);
        } else {
            let _ = cb;
        }
    }
}

/// Captures up to `out.len()` stack frames into `out`, most recent first.
///
/// Returns the number of frames written. Performs **no heap allocation**,
/// making it suitable for `no_std` and async-signal-safe contexts (such as
/// inside a signal handler). The walk semantics match [`trace`]: it never
/// panics and silently stops when unwind information is missing or corrupt.
///
/// # Performance
///
/// Stack-only; the cost is one unwind walk bounded by
/// `min(out.len(), MAX_FRAMES)`.
pub fn capture_frames(out: &mut [Frame]) -> usize {
    let limit = out.len().min(MAX_FRAMES);
    let mut count = 0usize;
    trace(&mut |frame| {
        if count >= limit {
            return false;
        }
        if let Some(slot) = out.get_mut(count) {
            *slot = *frame;
        }
        count += 1;
        count < limit
    });
    count
}

/// Shared machine-register snapshot used by CFI and frame-pointer stepping.
///
/// `gpr` is indexed by DWARF register numbers. `ip` is the current program
/// counter (not necessarily stored in `gpr`).
#[derive(Clone, Copy)]
struct Regs {
    gpr: [usize; 64],
    ip: usize,
}

impl Regs {
    const fn new() -> Self {
        Self {
            gpr: [0; 64],
            ip: 0,
        }
    }
}

/// Live unwind cursor: register file plus stack pointer and stop flag.
struct UnwindState {
    regs: Regs,
    sp: usize,
    stop: bool,
}

impl UnwindState {
    const fn new() -> Self {
        Self {
            regs: Regs::new(),
            sp: 0,
            stop: false,
        }
    }
}

/// Reads a native word from `addr`, rejecting the null page and misalignment.
///
/// # Safety
///
/// `addr` must be a readable address of at least `size_of::<usize>()` bytes
/// when it passes the low-address and alignment checks; otherwise this returns
/// `None`. Callers must only point at mapped stack, unwind tables, or other
/// known-mapped process memory — a wild pointer that passes these checks can
/// still fault.
#[inline]
unsafe fn read_word(addr: usize) -> Option<usize> {
    if addr < 4096
        || !addr.is_multiple_of(core::mem::size_of::<usize>())
        || !is_readable(addr)
    {
        return None;
    }
    let value = unsafe { core::ptr::read_unaligned(addr as *const usize) };
    Some(value)
}

/// Number of frames from nested helpers used by the unit tests.
mod capture {
    use super::{Regs, UnwindState};

    /// Captures callee-saved registers and the current instruction pointer.
    ///
    /// Returns `None` on architectures without an implementation.
    pub(super) fn current() -> Option<UnwindState> {
        let mut state = UnwindState::new();
        capture_arch(&mut state.regs)?;
        state.sp = state.regs.gpr[sp_index()];
        Some(state)
    }

    /// DWARF register index of the stack pointer for this architecture.
    #[cfg(target_arch = "x86_64")]
    const fn sp_index() -> usize {
        7
    }

    /// DWARF register index of the stack pointer for this architecture.
    #[cfg(target_arch = "aarch64")]
    const fn sp_index() -> usize {
        31
    }

    /// DWARF register index of the stack pointer for this architecture.
    #[cfg(target_arch = "x86")]
    const fn sp_index() -> usize {
        4
    }

    /// DWARF register index of the stack pointer for this architecture.
    #[cfg(target_arch = "arm")]
    const fn sp_index() -> usize {
        13
    }

    /// DWARF register index of the stack pointer for architectures without
    /// a register-capture implementation. [`capture_arch`] returns `None`
    /// for them, so [`current`] bails out before this index is ever read.
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "x86",
        target_arch = "arm",
    )))]
    const fn sp_index() -> usize {
        0
    }

    #[cfg(target_arch = "x86_64")]
    fn capture_arch(regs: &mut Regs) -> Option<()> {
        let ip: usize;
        let sp: usize;
        let gpr = regs.gpr.as_mut_ptr();
        // Safety: all outputs are stack locals; no memory is written through
        // pointers other than `gpr` (64 × usize, in bounds). `nostack` is
        // valid because the block only reads registers and PC.
        unsafe {
            core::arch::asm!(
            "lea {ip}, [rip + 0]",
            "mov {sp}, rsp",
            "mov [{rbp_o}], rbp",
            "mov [{rbx_o}], rbx",
            "mov [{r12_o}], r12",
            "mov [{r13_o}], r13",
            "mov [{r14_o}], r14",
            "mov [{r15_o}], r15",
            ip = out(reg) ip,
            sp = out(reg) sp,
            rbp_o = in(reg) gpr.add(6),
            rbx_o = in(reg) gpr.add(3),
            r12_o = in(reg) gpr.add(12),
            r13_o = in(reg) gpr.add(13),
            r14_o = in(reg) gpr.add(14),
            r15_o = in(reg) gpr.add(15),
            options(nostack),
            );
        }
        regs.gpr[7] = sp;
        regs.gpr[16] = ip;
        regs.ip = ip;
        Some(())
    }

    #[cfg(target_arch = "aarch64")]
    fn capture_arch(regs: &mut Regs) -> Option<()> {
        let ip: usize;
        let sp: usize;
        let gpr = regs.gpr.as_mut_ptr();
        // Safety: outputs are locals; `gpr` points at the 64-word register
        // file and all indexed stores stay in bounds. No stack adjustment.
        unsafe {
            core::arch::asm!(
            "adr {ip}, 1f",
            "b 2f",
            "1:",
            "2:",
            "mov {sp}, sp",
            "mov [{x29_o}], x29",
            "mov [{x30_o}], x30",
            "mov [{x19_o}], x19",
            "mov [{x20_o}], x20",
            "mov [{x21_o}], x21",
            "mov [{x22_o}], x22",
            "mov [{x23_o}], x23",
            "mov [{x24_o}], x24",
            "mov [{x25_o}], x25",
            "mov [{x26_o}], x26",
            "mov [{x27_o}], x27",
            "mov [{x28_o}], x28",
            ip = out(reg) ip,
            sp = out(reg) sp,
            x29_o = in(reg) gpr.add(29),
            x30_o = in(reg) gpr.add(30),
            x19_o = in(reg) gpr.add(19),
            x20_o = in(reg) gpr.add(20),
            x21_o = in(reg) gpr.add(21),
            x22_o = in(reg) gpr.add(22),
            x23_o = in(reg) gpr.add(23),
            x24_o = in(reg) gpr.add(24),
            x25_o = in(reg) gpr.add(25),
            x26_o = in(reg) gpr.add(26),
            x27_o = in(reg) gpr.add(27),
            x28_o = in(reg) gpr.add(28),
            options(nostack),
            );
        }
        regs.gpr[31] = sp;
        regs.ip = ip;
        Some(())
    }

    #[cfg(target_arch = "x86")]
    fn capture_arch(regs: &mut Regs) -> Option<()> {
        let ip: usize;
        let sp: usize;
        let gpr = regs.gpr.as_mut_ptr();
        // Safety: locals only; `gpr` indexes into the register file. The
        // `call` pushes a return address (stack is written) so `nostack` is
        // intentionally omitted.
        unsafe {
            core::arch::asm!(
            "mov {sp}, esp",
            "mov [{ebp_o}], ebp",
            "mov [{ebx_o}], ebx",
            "mov [{esi_o}], esi",
            "mov [{edi_o}], edi",
            "call 1f",
            "1:",
            "pop {ip}",
            sp = out(reg) sp,
            ip = out(reg) ip,
            ebp_o = in(reg) gpr.add(5),
            ebx_o = in(reg) gpr.add(3),
            esi_o = in(reg) gpr.add(6),
            edi_o = in(reg) gpr.add(7),
            );
        }
        regs.gpr[4] = sp;
        regs.gpr[8] = ip;
        regs.ip = ip;
        Some(())
    }

    #[cfg(target_arch = "arm")]
    fn capture_arch(regs: &mut Regs) -> Option<()> {
        let ip: usize;
        let sp: usize;
        let gpr = regs.gpr.as_mut_ptr();
        // Safety: locals only; `gpr` points at the register file with all
        // stores in bounds. `adr` does not touch the stack.
        unsafe {
            core::arch::asm!(
            "adr {ip}, 1f",
            "1:",
            "mov {sp}, sp",
            "mov [{fp_o}], r11",
            "mov [{lr_o}], r14",
            "mov [{r4_o}], r4",
            "mov [{r5_o}], r5",
            "mov [{r6_o}], r6",
            "mov [{r7_o}], r7",
            "mov [{r8_o}], r8",
            "mov [{r9_o}], r9",
            "mov [{r10_o}], r10",
            ip = out(reg) ip,
            sp = out(reg) sp,
            fp_o = in(reg) gpr.add(11),
            lr_o = in(reg) gpr.add(14),
            r4_o = in(reg) gpr.add(4),
            r5_o = in(reg) gpr.add(5),
            r6_o = in(reg) gpr.add(6),
            r7_o = in(reg) gpr.add(7),
            r8_o = in(reg) gpr.add(8),
            r9_o = in(reg) gpr.add(9),
            r10_o = in(reg) gpr.add(10),
            options(nostack),
            );
        }
        regs.gpr[13] = sp;
        regs.ip = ip;
        Some(())
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "x86",
        target_arch = "arm",
    )))]
    fn capture_arch(_regs: &mut Regs) -> Option<()> {
        None
    }
}

/// Frame-pointer chain walking (x86_64 / aarch64).
mod fp {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    use super::{Frame, read_word};

    /// DWARF register index of the frame pointer.
    #[cfg(target_arch = "x86_64")]
    const FP: usize = 6;

    /// DWARF register index of the frame pointer.
    #[cfg(target_arch = "aarch64")]
    const FP: usize = 29;

    /// Steps one frame using `[fp]` / `[fp + 8]`.
    ///
    /// Returns `false` when the chain looks corrupt or ends.
    pub(super) fn step(state: &mut super::UnwindState) -> bool {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let fp = state.regs.gpr[FP];
            let sp = state.sp;
            if fp == 0 || !fp.is_multiple_of(16) {
                return false;
            }
            if fp < sp {
                return false;
            }
            let Some(next_fp) = (unsafe { read_word(fp) }) else {
                return false;
            };
            let Some(next_ip) = (unsafe { read_word(fp.wrapping_add(8)) }) else {
                return false;
            };
            let next_sp = fp.wrapping_add(16);
            if next_ip == 0 || next_sp <= sp {
                return false;
            }
            if next_fp != 0 && next_fp <= fp {
                return false;
            }
            state.regs.gpr[FP] = next_fp;
            state.regs.ip = next_ip;
            state.sp = next_sp;
            true
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = state;
            false
        }
    }

    /// Frame-pointer-only walk used when no unwind tables are available.
    ///
    /// Mirrors the availability of [`super::walk`] (referenced from
    /// [`super::trace`] only on these architectures).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[allow(dead_code)]
    pub(super) fn trace_inner(cb: &mut dyn FnMut(&Frame) -> bool) {
        super::walk(cb, None);
    }
}

/// DWARF `.eh_frame` CFI interpreter shared by ELF and Mach-O backends.
///
/// Only reachable from the 64-bit Unix backends (`elf` / `apple`) at run
/// time; on other targets it is compiled solely for the unit tests, so the
/// dead-code lint is silenced there.
#[cfg_attr(not(all(unix, target_pointer_width = "64")), allow(dead_code))]
mod cfi {
    use super::{Regs, UnwindState, read_word};

    /// Maximum depth of `DW_CFA_remember_state` stacks.
    const STATE_STACK_LIMIT: usize = 32;

    /// Maximum depth of the DWARF expression evaluation stack.
    const EXPR_STACK_LIMIT: usize = 64;

    /// Maximum bytes accepted for a single unwind entry length.
    const MAX_ENTRY_LEN: usize = 1 << 28;

    #[derive(Clone, Copy)]
    enum RegRule {
        SameValue,
        Undefined,
        Offset(i64),
        ValOffset(i64),
        Register(usize),
        Expr(usize, usize),
        ValExpr(usize, usize),
    }

    #[derive(Clone, Copy)]
    enum CfaRule {
        RegOff(usize, i64),
        Expr(usize, usize),
        Undefined,
    }

    #[derive(Clone, Copy)]
    struct CfState {
        cfa: CfaRule,
        rules: [RegRule; 64],
    }

    impl CfState {
        const fn new() -> Self {
            Self {
                cfa: CfaRule::Undefined,
                rules: [RegRule::SameValue; 64],
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Cie {
        code_factor: u64,
        data_factor: i64,
        ret_reg: usize,
        fde_enc: u8,
        lsda_enc: u8,
        is_signal: bool,
        address_size: u8,
        version: u8,
        /// Byte offset of the CIE's initial-instruction stream inside `eh_frame`.
        insts_off: usize,
        insts_end: usize,
    }

    /// Cursor over a mapped `.eh_frame` (or Mach-O equivalent) region.
    struct SliceReader<'a> {
        data: &'a [u8],
        pos: usize,
        /// Virtual address of `data[0]`.
        base: usize,
        /// Base for `DW_EH_PE_datarel` (usually `.eh_frame_hdr`).
        datarel_base: usize,
    }

    impl<'a> SliceReader<'a> {
        fn new(data: &'a [u8], base: usize, datarel_base: usize) -> Self {
            Self {
                data,
                pos: 0,
                base,
                datarel_base,
            }
        }

        fn remaining(&self) -> usize {
            self.data.len().saturating_sub(self.pos)
        }

        fn addr(&self) -> usize {
            self.base.wrapping_add(self.pos)
        }

        fn byte(&mut self) -> Option<u8> {
            let b = *self.data.get(self.pos)?;
            self.pos += 1;
            Some(b)
        }

        fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
            let end = self.pos.checked_add(n)?;
            let s = self.data.get(self.pos..end)?;
            self.pos = end;
            Some(s)
        }

        fn u8(&mut self) -> Option<u8> {
            self.byte()
        }

        fn u16(&mut self) -> Option<u16> {
            let b = self.bytes(2)?;
            Some(u16::from_le_bytes([b[0], b[1]]))
        }

        fn u32(&mut self) -> Option<u32> {
            let b = self.bytes(4)?;
            Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }

        fn u64(&mut self) -> Option<u64> {
            let b = self.bytes(8)?;
            Some(u64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }

        fn usize_native(&mut self, width: u8) -> Option<usize> {
            match width {
                8 => self.u64().map(|v| v as usize),
                4 => self.u32().map(|v| v as usize),
                2 => self.u16().map(|v| v as usize),
                1 => self.u8().map(|v| v as usize),
                _ => None,
            }
        }

        fn uleb(&mut self) -> Option<u64> {
            let mut result: u64 = 0;
            let mut shift = 0u32;
            loop {
                let b = self.u8()?;
                result |= u64::from(b & 0x7f) << shift;
                if b & 0x80 == 0 {
                    break;
                }
                shift += 7;
                if shift >= 64 {
                    return None;
                }
            }
            Some(result)
        }

        fn sleb(&mut self) -> Option<i64> {
            let mut result: i64 = 0;
            let mut shift = 0u32;
            let mut b;
            loop {
                b = self.u8()?;
                result |= i64::from(b & 0x7f) << shift;
                shift += 7;
                if b & 0x80 == 0 {
                    break;
                }
                if shift >= 64 {
                    return None;
                }
            }
            if shift < 64 && b & 0x40 != 0 {
                result |= -1i64 << shift;
            }
            Some(result)
        }

        fn cstr(&mut self) -> Option<&'a [u8]> {
            let start = self.pos;
            loop {
                let b = self.u8()?;
                if b == 0 {
                    return self.data.get(start..self.pos - 1);
                }
            }
        }

        /// Reads a value with `enc`, applying pointer application and
        /// indirection. `field_addr` is the address of the encoded field.
        fn read_encoded(&mut self, enc: u8, field_addr: usize) -> Option<usize> {
            if enc == 0xff {
                return None;
            }
            let raw = self.read_encoded_raw(enc)?;
            let mut value = match enc & 0x70 {
                0x00 => raw,
                0x10 => raw.wrapping_add(field_addr),
                0x20 => raw,
                0x30 => raw.wrapping_add(self.datarel_base),
                0x40 => raw,
                0x50 => raw,
                _ => return None,
            };
            if enc & 0x80 != 0 {
                value = unsafe { read_word(value) }?;
            }
            Some(value)
        }

        /// Reads the numeric payload without applying pcrel/datarel.
        fn read_encoded_raw(&mut self, enc: u8) -> Option<usize> {
            match enc & 0x0f {
                0x00 => self.usize_native(core::mem::size_of::<usize>() as u8),
                0x01 => self.uleb().map(|v| v as usize),
                0x02 => self.u16().map(|v| v as usize),
                0x03 => self.u32().map(|v| v as usize),
                0x04 => self.u64().map(|v| v as usize),
                0x09 => self.sleb().map(|v| v as usize),
                0x0a => self.u16().map(|v| v as i16 as usize),
                0x0b => self.u32().map(|v| v as i32 as usize),
                0x0c => self.u64().map(|v| v as i64 as usize),
                _ => None,
            }
        }
    }

    /// Result of locating an FDE for a program counter.
    struct Fde<'a> {
        cie: Cie,
        initial_location: usize,
        address_range: usize,
        /// Initial instructions of the FDE (after header/augmentation).
        insts: SliceReader<'a>,
    }

    impl Fde<'_> {
        fn contains(&self, ip: usize) -> bool {
            if self.address_range == 0 {
                return false;
            }
            let start = self.initial_location;
            let end = start.wrapping_add(self.address_range);
            if end < start {
                return ip >= start;
            }
            ip >= start && ip < end
        }
    }

    /// Module unwind-table view: addresses of `.eh_frame` / `.eh_frame_hdr`.
    #[derive(Clone, Copy)]
    pub(super) struct UnwindTables {
        pub eh_frame: (usize, usize),
        pub eh_frame_hdr: Option<(usize, usize)>,
        pub datarel_base: usize,
    }

    #[allow(unused_assignments)]
    fn parse_cie(reader: &mut SliceReader<'_>, entry_end: usize) -> Option<Cie> {
        let version = reader.u8()?;
        if version != 1 && version != 3 && version != 4 {
            return None;
        }
        let aug = reader.cstr()?;
        let mut address_size = core::mem::size_of::<usize>() as u8;
        if version >= 4 {
            address_size = reader.u8()?;
            let _segment_size = reader.u8()?;
        }
        let code_factor = reader.uleb()?;
        let data_factor = reader.sleb()?;
        // `.eh_frame` uses ULEB128 for the return-address register even at
        // version 1 (values < 128 are identical to a single byte).
        let ret_reg = reader.uleb()? as usize;

        let mut fde_enc: u8 = 0x00;
        let mut lsda_enc: u8 = 0xff;
        let mut personality_enc: u8 = 0xff;
        let mut is_signal = false;

        if aug.first() == Some(&b'z') {
            let aug_len = reader.uleb()? as usize;
            let aug_start = reader.pos;
            let aug_end = aug_start.checked_add(aug_len)?;
            if aug_end > entry_end {
                return None;
            }
            for &c in &aug[1..] {
                match c {
                    b'S' => is_signal = true,
                    b'R' => fde_enc = reader.u8()?,
                    b'L' => lsda_enc = reader.u8()?,
                    b'P' => {
                        personality_enc = reader.u8()?;
                        let field = reader.addr();
                        reader.read_encoded(personality_enc, field)?;
                    }
                    _ => {}
                }
                if reader.pos > aug_end {
                    return None;
                }
            }
            reader.pos = aug_end;
        } else {
            for &c in aug {
                match c {
                    b'S' => is_signal = true,
                    b'R' => fde_enc = reader.u8()?,
                    b'L' => lsda_enc = reader.u8()?,
                    b'P' => {
                        personality_enc = reader.u8()?;
                        let field = reader.addr();
                        reader.read_encoded(personality_enc, field)?;
                    }
                    _ => {}
                }
            }
        }
        let insts_off = reader.pos;
        let insts_end = entry_end.min(reader.data.len());
        if insts_off > insts_end {
            return None;
        }
        Some(Cie {
            code_factor,
            data_factor,
            ret_reg,
            fde_enc,
            lsda_enc,
            is_signal,
            address_size,
            version,
            insts_off,
            insts_end,
        })
    }

    fn parse_fde<'a>(
        reader: &mut SliceReader<'a>,
        entry_end: usize,
        cie: Cie,
    ) -> Option<Fde<'a>> {
        let loc_field = reader.addr();
        let initial_location = reader.read_encoded(cie.fde_enc, loc_field)?;
        let address_range = reader.read_encoded_raw(cie.fde_enc)?;

        if cie.lsda_enc != 0xff {
            let has_z = {
                // Augmentation data is present whenever `z` was set; LSDA
                // encoding non-omit implies either `z` or a bare `L`.
                true
            };
            if has_z {
                let aug_len = reader.uleb()? as usize;
                let aug_start = reader.pos;
                let aug_end = aug_start.checked_add(aug_len)?;
                if aug_end > entry_end {
                    return None;
                }
                if cie.lsda_enc != 0xff {
                    let field = reader.addr();
                    reader.read_encoded(cie.lsda_enc, field)?;
                }
                reader.pos = aug_end;
            }
        } else {
            // `z` augmentation without LSDA: still consume the length prefix.
            // Detect `z` via a conservative uleb only when the CIE personality
            // or other z-only fields imply it. GCC/Clang always emit `z`.
            // Try: if remaining bytes look like a plausible aug length that
            // lands inside the entry, skip it.
            let save = reader.pos;
            if let Some(aug_len) = reader.uleb() {
                let aug_end = save.wrapping_add(aug_len as usize);
                if aug_len as usize <= entry_end.saturating_sub(save)
                    && aug_end <= entry_end
                    && aug_end >= save
                {
                    // Require that the CIE had `z` by checking version/aug is
                    // not enough; FDEs with `zR` always have this uleb. FDEs
                    // without `z` would mis-parse — detect by seeing whether
                    // instructions after a skipped region still fit.
                    reader.pos = aug_end;
                } else {
                    reader.pos = save;
                }
            } else {
                reader.pos = save;
            }
            let _ = cie.version;
        }

        let insts_off = reader.pos;
        let insts_end = entry_end.min(reader.data.len());
        if insts_off > insts_end {
            return None;
        }
        let mut insts = SliceReader::new(
            &reader.data[insts_off..insts_end],
            reader.base + insts_off,
            reader.datarel_base,
        );
        insts.pos = 0;
        Some(Fde {
            cie,
            initial_location,
            address_range,
            insts,
        })
    }

    /// Parses a CIE at `cie_off` (start of the entry's id/length body) and an
    /// FDE header at `fde_off`, both relative to `eh_frame`.
    fn entry_offsets(entry_start: usize, data: &[u8]) -> Option<(usize, usize, bool)> {
        if entry_start + 4 > data.len() {
            return None;
        }
        let mut r = SliceReader::new(data, 0, 0);
        r.pos = entry_start;
        let first = r.u32()? as usize;
        if first == 0xffff_ffff {
            let len = r.u64()? as usize;
            if len > MAX_ENTRY_LEN || len == 0 {
                return None;
            }
            let body = r.pos;
            let end = body.checked_add(len)?;
            if end > data.len() {
                return None;
            }
            let id = r.u64()?;
            Some((body, end, id == 0))
        } else {
            if first == 0 || first > MAX_ENTRY_LEN {
                return None;
            }
            let body = entry_start + 4;
            let end = body.checked_add(first)?;
            if end > data.len() {
                return None;
            }
            let mut idr = SliceReader::new(data, 0, 0);
            idr.pos = body;
            let id = idr.u32()? as usize;
            Some((body, end, id == 0))
        }
    }

    /// Walks `.eh_frame` linearly looking for an FDE covering `ip`.
    fn scan_eh_frame<'a>(
        data: &'a [u8],
        base: usize,
        datarel_base: usize,
        ip: usize,
    ) -> Option<Fde<'a>> {
        let mut off = 0usize;
        while off + 4 <= data.len() {
            if data[off..off + 4] == [0, 0, 0, 0] {
                break;
            }
            let Some((body, end, is_cie)) = entry_offsets(off, data) else {
                break;
            };
            if end <= off {
                break;
            }
            if !is_cie {
                let mut idr = SliceReader::new(data, base, datarel_base);
                idr.pos = body;
                // `body` is the address of the CIE-pointer field.
                let cie_ptr_field = body;
                let cie_ptr = idr.u32()? as usize;
                if cie_ptr > cie_ptr_field {
                    off = end;
                    continue;
                }
                let cie_start = cie_ptr_field - cie_ptr;
                // CIE body starts after its length/id header.
                let Some((_cie_body, cie_end, true)) = entry_offsets(cie_start, data)
                else {
                    off = end;
                    continue;
                };
                let mut cr = SliceReader::new(data, base, datarel_base);
                // Skip length field (4 or 12 bytes) plus CIE id (4 or 8).
                let id_size =
                    if data.get(cie_start..cie_start + 4) == Some(&[0xff; 4][..]) {
                        20
                    } else {
                        8
                    };
                cr.pos = cie_start + id_size;
                let cie = parse_cie(&mut cr, cie_end)?;
                let mut fr = SliceReader::new(data, base, datarel_base);
                fr.pos = body + 4;
                let fde = parse_fde(&mut fr, end, cie)?;
                if fde.contains(ip) {
                    return Some(fde);
                }
            }
            off = end;
        }
        None
    }

    /// Binary-searches `.eh_frame_hdr` for an FDE address covering `ip`.
    fn hdr_find_fde<'a>(
        tables: &UnwindTables,
        ip: usize,
        frame: &'a [u8],
    ) -> Option<Fde<'a>> {
        let (hdr_addr, hdr_len) = tables.eh_frame_hdr?;
        if hdr_len < 4 || hdr_addr < 4096 {
            return None;
        }
        let hdr = unsafe { core::slice::from_raw_parts(hdr_addr as *const u8, hdr_len) };
        if hdr[0] != 1 {
            return None;
        }
        let eh_frame_ptr_enc = hdr[1];
        let fde_count_enc = hdr[2];
        let table_enc = hdr[3];
        let mut r = SliceReader::new(&hdr[4..], hdr_addr + 4, hdr_addr);
        let field = r.addr();
        let _eh_frame_ptr = r.read_encoded(eh_frame_ptr_enc, field)?;
        let count_field = r.addr();
        let count = r.read_encoded(fde_count_enc, count_field)?;
        let count = count.min(1 << 24);
        let entry_size = fixed_encoded_size(table_enc)?;
        let table_bytes = count.checked_mul(entry_size.checked_mul(2)?)?;
        if r.remaining() < table_bytes {
            return None;
        }
        let table = &hdr[4 + r.pos..4 + r.pos + table_bytes];

        // Binary search for the greatest initial_location <= ip.
        let mut lo = 0usize;
        let mut hi = count;
        let mut best = None;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let mut er = SliceReader::new(table, hdr_addr + 4 + r.pos, hdr_addr);
            er.pos = mid * entry_size * 2;
            let loc_field = er.addr();
            let loc = er.read_encoded(table_enc, loc_field)?;
            if loc <= ip {
                best = Some(mid);
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let idx = best?;
        let mut er = SliceReader::new(table, hdr_addr + 4 + r.pos, hdr_addr);
        er.pos = idx * entry_size * 2;
        let loc_field = er.addr();
        let _loc = er.read_encoded(table_enc, loc_field)?;
        let fde_field = er.addr();
        let fde_addr = er.read_encoded(table_enc, fde_field)?;

        // Map absolute FDE address into `frame`.
        let frame_addr = tables.eh_frame.0;
        if fde_addr < frame_addr {
            return None;
        }
        let off = fde_addr - frame_addr;
        if off >= frame.len() {
            return None;
        }
        let (body, end, is_cie) = entry_offsets(off, frame)?;
        if is_cie {
            return None;
        }
        let mut idr = SliceReader::new(frame, frame_addr, tables.datarel_base);
        idr.pos = body;
        // `body` is the address of the CIE-pointer field (entry start + 4).
        let cie_ptr_field = body;
        let cie_ptr = idr.u32()? as usize;
        if cie_ptr > cie_ptr_field {
            return None;
        }
        let cie_start = cie_ptr_field - cie_ptr;
        let Some((_, cie_end, true)) = entry_offsets(cie_start, frame) else {
            return None;
        };
        let id_size = if frame.get(cie_start..cie_start + 4) == Some(&[0xff; 4][..]) {
            20
        } else {
            8
        };
        let mut cr = SliceReader::new(frame, frame_addr, tables.datarel_base);
        cr.pos = cie_start + id_size;
        let cie = parse_cie(&mut cr, cie_end)?;
        let mut fr = SliceReader::new(frame, frame_addr, tables.datarel_base);
        fr.pos = body + 4;
        let fde = parse_fde(&mut fr, end, cie)?;
        if fde.contains(ip) { Some(fde) } else { None }
    }

    fn fixed_encoded_size(enc: u8) -> Option<usize> {
        match enc & 0x0f {
            0x02 => Some(2),
            0x03 => Some(4),
            0x04 => Some(8),
            0x0a => Some(2),
            0x0b => Some(4),
            0x0c => Some(8),
            0x00 => Some(core::mem::size_of::<usize>()),
            _ => None,
        }
    }

    fn find_fde<'a>(
        tables: &UnwindTables,
        frame: &'a [u8],
        ip: usize,
    ) -> Option<Fde<'a>> {
        if let Some(fde) = hdr_find_fde(tables, ip, frame) {
            return Some(fde);
        }
        scan_eh_frame(frame, tables.eh_frame.0, tables.datarel_base, ip)
    }

    fn eval_expr(
        data: &[u8],
        base: usize,
        regs: &[usize; 64],
        cfa: usize,
        address_size: u8,
    ) -> Option<usize> {
        let mut r = SliceReader::new(data, base, 0);
        let mut stack = [0usize; EXPR_STACK_LIMIT];
        let mut sp = 0usize;
        let push = |stack: &mut [usize; EXPR_STACK_LIMIT],
                    sp: &mut usize,
                    v: usize|
         -> Option<()> {
            if *sp >= EXPR_STACK_LIMIT {
                return None;
            }
            stack[*sp] = v;
            *sp += 1;
            Some(())
        };
        while r.remaining() > 0 {
            let op = r.u8()?;
            match op {
                0x03 => {
                    let v = r.usize_native(address_size)?;
                    push(&mut stack, &mut sp, v)?;
                }
                0x06 => {
                    let a = *stack.get(sp.checked_sub(1)?)?;
                    let v = unsafe { read_word(a) }?;
                    stack[sp - 1] = v;
                }
                0x08 => {
                    let v = r.u8()? as usize;
                    push(&mut stack, &mut sp, v)?;
                }
                0x09 => {
                    let v = r.u8()? as i8 as usize;
                    push(&mut stack, &mut sp, v)?;
                }
                0x0a => {
                    let v = r.u16()? as usize;
                    push(&mut stack, &mut sp, v)?;
                }
                0x0b => {
                    let v = r.u16()? as i16 as usize;
                    push(&mut stack, &mut sp, v)?;
                }
                0x0c => {
                    let v = r.u32()? as usize;
                    push(&mut stack, &mut sp, v)?;
                }
                0x0d => {
                    let v = r.u32()? as i32 as usize;
                    push(&mut stack, &mut sp, v)?;
                }
                0x0e => {
                    let v = r.u64()? as usize;
                    push(&mut stack, &mut sp, v)?;
                }
                0x0f => {
                    let v = r.u64()? as i64 as usize;
                    push(&mut stack, &mut sp, v)?;
                }
                0x10 => {
                    let v = r.uleb()? as usize;
                    push(&mut stack, &mut sp, v)?;
                }
                0x11 => {
                    let v = r.sleb()? as usize;
                    push(&mut stack, &mut sp, v)?;
                }
                0x12 => {
                    let v = *stack.get(sp.checked_sub(1)?)?;
                    push(&mut stack, &mut sp, v)?;
                }
                0x13 => {
                    sp = sp.checked_sub(1)?;
                }
                0x14 => {
                    let v = *stack.get(sp.checked_sub(1)?)?;
                    push(&mut stack, &mut sp, v)?;
                }
                0x15 => {
                    let off = r.u8()? as usize;
                    let v = *stack.get(sp.checked_sub(off + 1)?)?;
                    push(&mut stack, &mut sp, v)?;
                }
                0x16 => {
                    if sp < 2 {
                        return None;
                    }
                    stack.swap(sp - 1, sp - 2);
                }
                0x19 => {
                    let v = *stack.get(sp.checked_sub(1)?)?;
                    stack[sp - 1] = (v as isize).unsigned_abs();
                }
                0x1a => {
                    if sp < 2 {
                        return None;
                    }
                    stack[sp - 2] &= stack[sp - 1];
                    sp -= 1;
                }
                0x1b => {
                    if sp < 2 {
                        return None;
                    }
                    let b = stack[sp - 1] as i64;
                    if b == 0 {
                        return None;
                    }
                    stack[sp - 2] = ((stack[sp - 2] as i64) / b) as usize;
                    sp -= 1;
                }
                0x1c => {
                    if sp < 2 {
                        return None;
                    }
                    stack[sp - 2] = stack[sp - 2].wrapping_sub(stack[sp - 1]);
                    sp -= 1;
                }
                0x1d => {
                    if sp < 2 {
                        return None;
                    }
                    let b = stack[sp - 1] as i64;
                    if b == 0 {
                        return None;
                    }
                    stack[sp - 2] = ((stack[sp - 2] as i64) % b) as usize;
                    sp -= 1;
                }
                0x1e => {
                    if sp < 2 {
                        return None;
                    }
                    stack[sp - 2] = stack[sp - 2].wrapping_mul(stack[sp - 1]);
                    sp -= 1;
                }
                0x1f => {
                    let v = *stack.get(sp.checked_sub(1)?)?;
                    stack[sp - 1] = (v as isize).wrapping_neg() as usize;
                }
                0x20 => {
                    let v = *stack.get(sp.checked_sub(1)?)?;
                    stack[sp - 1] = !v;
                }
                0x21 => {
                    if sp < 2 {
                        return None;
                    }
                    stack[sp - 2] |= stack[sp - 1];
                    sp -= 1;
                }
                0x22 => {
                    if sp < 2 {
                        return None;
                    }
                    stack[sp - 2] = stack[sp - 2].wrapping_add(stack[sp - 1]);
                    sp -= 1;
                }
                0x23 => {
                    let c = r.uleb()? as usize;
                    let v = *stack.get(sp.checked_sub(1)?)?;
                    stack[sp - 1] = v.wrapping_add(c);
                }
                0x24 => {
                    if sp < 2 {
                        return None;
                    }
                    stack[sp - 2] = stack[sp - 2].wrapping_shl(stack[sp - 1] as u32);
                    sp -= 1;
                }
                0x25 => {
                    if sp < 2 {
                        return None;
                    }
                    stack[sp - 2] = stack[sp - 2].wrapping_shr(stack[sp - 1] as u32);
                    sp -= 1;
                }
                0x26 => {
                    if sp < 2 {
                        return None;
                    }
                    stack[sp - 2] = ((stack[sp - 2] as isize)
                        .wrapping_shr(stack[sp - 1] as u32))
                        as usize;
                    sp -= 1;
                }
                0x27 => {
                    if sp < 2 {
                        return None;
                    }
                    stack[sp - 2] ^= stack[sp - 1];
                    sp -= 1;
                }
                0x28 => {
                    let off = r.u16()? as i16;
                    let cond = *stack.get(sp.checked_sub(1)?)? != 0;
                    sp -= 1;
                    if cond {
                        let target = (r.pos as isize + isize::from(off)) as usize;
                        if target > data.len() {
                            return None;
                        }
                        r.pos = target;
                    }
                }
                0x29 => {
                    cmp_pop(&mut stack, &mut sp, |a, b| a == b)?;
                }
                0x2a => {
                    cmp_pop(&mut stack, &mut sp, |a, b| (a as i64) >= (b as i64))?;
                }
                0x2b => {
                    cmp_pop(&mut stack, &mut sp, |a, b| (a as i64) > (b as i64))?;
                }
                0x2c => {
                    cmp_pop(&mut stack, &mut sp, |a, b| (a as i64) <= (b as i64))?;
                }
                0x2d => {
                    cmp_pop(&mut stack, &mut sp, |a, b| (a as i64) < (b as i64))?;
                }
                0x2e => {
                    cmp_pop(&mut stack, &mut sp, |a, b| a != b)?;
                }
                0x2f => {
                    let off = r.u16()? as i16;
                    let target = (r.pos as isize + isize::from(off)) as usize;
                    if target > data.len() {
                        return None;
                    }
                    r.pos = target;
                }
                0x30..=0x4f => {
                    push(&mut stack, &mut sp, usize::from(op - 0x30))?;
                }
                0x50..=0x6f => {
                    let reg = usize::from(op - 0x50);
                    push(&mut stack, &mut sp, *regs.get(reg)?)?;
                }
                0x70..=0x8f => {
                    let reg = usize::from(op - 0x70);
                    let off = r.sleb()?;
                    let v = (*regs.get(reg)? as i64).wrapping_add(off);
                    push(&mut stack, &mut sp, v as usize)?;
                }
                0x90 => {
                    let reg = r.uleb()? as usize;
                    push(&mut stack, &mut sp, *regs.get(reg)?)?;
                }
                0x91 => {
                    let off = r.sleb()?;
                    let v = (cfa as i64).wrapping_add(off);
                    push(&mut stack, &mut sp, v as usize)?;
                }
                0x92 => {
                    let reg = r.uleb()? as usize;
                    let off = r.sleb()?;
                    let v = (*regs.get(reg)? as i64).wrapping_add(off);
                    push(&mut stack, &mut sp, v as usize)?;
                }
                0x94 => {
                    let size = usize::from(r.u8()?);
                    let a = *stack.get(sp.checked_sub(1)?)?;
                    if size == 0 {
                        stack[sp - 1] = 0;
                    } else if a < 4096 {
                        return None;
                    } else {
                        let bytes = unsafe {
                            core::slice::from_raw_parts(a as *const u8, size.min(8))
                        };
                        let mut v = 0usize;
                        for (i, &b) in bytes.iter().enumerate() {
                            v |= usize::from(b) << (8 * i);
                        }
                        stack[sp - 1] = v;
                    }
                }
                0x96 => {}
                0x9c => {
                    push(&mut stack, &mut sp, cfa)?;
                }
                _ => return None,
            }
        }
        if sp == 0 {
            return None;
        }
        Some(stack[sp - 1])
    }

    fn cmp_pop(
        stack: &mut [usize; EXPR_STACK_LIMIT],
        sp: &mut usize,
        f: impl FnOnce(usize, usize) -> bool,
    ) -> Option<()> {
        if *sp < 2 {
            return None;
        }
        let a = stack[*sp - 2];
        let b = stack[*sp - 1];
        stack[*sp - 2] = usize::from(f(a, b));
        *sp -= 1;
        Some(())
    }

    /// Executes a CFA instruction stream against `state`.
    ///
    /// When `tracking` is true, location advances are checked against `target`
    /// (the frame's IP). Returns the final location.
    #[allow(clippy::too_many_arguments)]
    fn exec_cfa(
        reader: &mut SliceReader<'_>,
        state: &mut CfState,
        base_state: &CfState,
        state_stack: &mut [Option<CfState>; STATE_STACK_LIMIT],
        stack_len: &mut usize,
        code_factor: u64,
        data_factor: i64,
        address_size: u8,
        mut loc: usize,
        target: usize,
        tracking: bool,
    ) -> Option<usize> {
        while reader.remaining() > 0 {
            if tracking && loc > target {
                break;
            }
            let op = reader.u8()?;
            let short_hi = op & 0xc0;
            let short_reg = usize::from(op & 0x3f);
            match short_hi {
                0x40 => {
                    loc = loc.wrapping_add(short_reg.wrapping_mul(code_factor as usize))
                }
                0x80 => {
                    let off = reader.uleb()? as i64;
                    if short_reg >= 64 {
                        return None;
                    }
                    state.rules[short_reg] =
                        RegRule::Offset(off.wrapping_mul(data_factor));
                }
                0xc0 => {
                    if short_reg >= 64 {
                        return None;
                    }
                    state.rules[short_reg] = base_state.rules[short_reg];
                }
                _ => match op {
                    0x00 => {}
                    0x01 => {
                        loc = reader.usize_native(address_size)?;
                    }
                    0x02 => {
                        loc = loc.wrapping_add(
                            usize::from(reader.u8()?) * code_factor as usize,
                        );
                    }
                    0x03 => {
                        loc = loc.wrapping_add(
                            usize::from(reader.u16()?) * code_factor as usize,
                        );
                    }
                    0x04 => {
                        loc = loc.wrapping_add(
                            (reader.u32()? as usize).wrapping_mul(code_factor as usize),
                        );
                    }
                    0x05 => {
                        let reg = reader.uleb()? as usize;
                        let off = reader.uleb()? as i64;
                        if reg >= 64 {
                            return None;
                        }
                        state.rules[reg] = RegRule::Offset(off.wrapping_mul(data_factor));
                    }
                    0x06 => {
                        let reg = reader.uleb()? as usize;
                        if reg >= 64 {
                            return None;
                        }
                        state.rules[reg] = base_state.rules[reg];
                    }
                    0x07 => {
                        let reg = reader.uleb()? as usize;
                        if reg >= 64 {
                            return None;
                        }
                        state.rules[reg] = RegRule::Undefined;
                    }
                    0x08 => {
                        let reg = reader.uleb()? as usize;
                        if reg >= 64 {
                            return None;
                        }
                        state.rules[reg] = RegRule::SameValue;
                    }
                    0x09 => {
                        let reg = reader.uleb()? as usize;
                        let src = reader.uleb()? as usize;
                        if reg >= 64 || src >= 64 {
                            return None;
                        }
                        state.rules[reg] = RegRule::Register(src);
                    }
                    0x0a => {
                        if *stack_len >= STATE_STACK_LIMIT {
                            return None;
                        }
                        state_stack[*stack_len] = Some(*state);
                        *stack_len += 1;
                    }
                    0x0b => {
                        if *stack_len == 0 {
                            return None;
                        }
                        *stack_len -= 1;
                        if let Some(s) = state_stack[*stack_len] {
                            *state = s;
                        }
                    }
                    0x0c => {
                        let reg = reader.uleb()? as usize;
                        let off = reader.uleb()? as i64;
                        if reg >= 64 {
                            return None;
                        }
                        state.cfa = CfaRule::RegOff(reg, off);
                    }
                    0x0d => {
                        let reg = reader.uleb()? as usize;
                        if reg >= 64 {
                            return None;
                        }
                        let off = match state.cfa {
                            CfaRule::RegOff(_, o) => o,
                            _ => 0,
                        };
                        state.cfa = CfaRule::RegOff(reg, off);
                    }
                    0x0e => {
                        let off = reader.uleb()? as i64;
                        let reg = match state.cfa {
                            CfaRule::RegOff(r, _) => r,
                            _ => return None,
                        };
                        state.cfa = CfaRule::RegOff(reg, off);
                    }
                    0x0f => {
                        let len = reader.uleb()? as usize;
                        let start = reader.pos;
                        let end = start.checked_add(len)?;
                        if end > reader.data.len() {
                            return None;
                        }
                        state.cfa = CfaRule::Expr(start, end);
                        reader.pos = end;
                    }
                    0x10 => {
                        let reg = reader.uleb()? as usize;
                        let len = reader.uleb()? as usize;
                        if reg >= 64 {
                            return None;
                        }
                        let start = reader.pos;
                        let end = start.checked_add(len)?;
                        if end > reader.data.len() {
                            return None;
                        }
                        state.rules[reg] = RegRule::Expr(start, end);
                        reader.pos = end;
                    }
                    0x11 => {
                        let reg = reader.uleb()? as usize;
                        let off = reader.sleb()?;
                        if reg >= 64 {
                            return None;
                        }
                        state.rules[reg] = RegRule::Offset(off.wrapping_mul(data_factor));
                    }
                    0x12 => {
                        let reg = reader.uleb()? as usize;
                        let off = reader.sleb()?;
                        if reg >= 64 {
                            return None;
                        }
                        state.cfa = CfaRule::RegOff(reg, off.wrapping_mul(data_factor));
                    }
                    0x13 => {
                        let off = reader.sleb()?;
                        let reg = match state.cfa {
                            CfaRule::RegOff(r, _) => r,
                            _ => return None,
                        };
                        state.cfa = CfaRule::RegOff(reg, off.wrapping_mul(data_factor));
                    }
                    0x14 => {
                        let reg = reader.uleb()? as usize;
                        let off = reader.uleb()? as i64;
                        if reg >= 64 {
                            return None;
                        }
                        state.rules[reg] =
                            RegRule::ValOffset(off.wrapping_mul(data_factor));
                    }
                    0x15 => {
                        let reg = reader.uleb()? as usize;
                        let off = reader.sleb()?;
                        if reg >= 64 {
                            return None;
                        }
                        state.rules[reg] =
                            RegRule::ValOffset(off.wrapping_mul(data_factor));
                    }
                    0x16 => {
                        let reg = reader.uleb()? as usize;
                        let len = reader.uleb()? as usize;
                        if reg >= 64 {
                            return None;
                        }
                        let start = reader.pos;
                        let end = start.checked_add(len)?;
                        if end > reader.data.len() {
                            return None;
                        }
                        state.rules[reg] = RegRule::ValExpr(start, end);
                        reader.pos = end;
                    }
                    0x2d => {
                        // SPARC window save / AArch64 negate RA: no-op for
                        // backtraces.
                    }
                    0x2e => {
                        let _args_size = reader.uleb()?;
                    }
                    _ => return None,
                },
            }
        }
        Some(loc)
    }

    /// Applies computed rules to produce the caller register file.
    fn apply_rules(
        state: &CfState,
        regs: &Regs,
        cie: Cie,
        frame_data: &[u8],
        frame_base: usize,
    ) -> Option<UnwindState> {
        let old = *regs;
        let cfa = match state.cfa {
            CfaRule::RegOff(reg, off) => {
                let v = *old.gpr.get(reg)?;
                (v as i64).wrapping_add(off) as usize
            }
            CfaRule::Expr(start, end) => {
                let end = end.min(frame_data.len());
                if start > end {
                    return None;
                }
                let slice = frame_data.get(start..end)?;
                eval_expr(slice, frame_base + start, &old.gpr, 0, cie.address_size)?
            }
            CfaRule::Undefined => return None,
        };

        // Prefer a sane stack pointer if the CFA rule is nonsense.
        if cfa < 4096 {
            return None;
        }

        let mut new_regs = old;
        for (i, rule) in state.rules.iter().enumerate() {
            let value = match *rule {
                RegRule::SameValue => old.gpr[i],
                RegRule::Undefined => 0,
                RegRule::Offset(off) => {
                    let addr = (cfa as i64).wrapping_add(off) as usize;
                    unsafe { read_word(addr) }?
                }
                RegRule::ValOffset(off) => (cfa as i64).wrapping_add(off) as usize,
                RegRule::Register(r) => *old.gpr.get(r)?,
                RegRule::Expr(start, end) => {
                    let end = end.min(frame_data.len());
                    if start > end {
                        return None;
                    }
                    let slice = frame_data.get(start..end)?;
                    let addr = eval_expr(
                        slice,
                        frame_base + start,
                        &old.gpr,
                        cfa,
                        cie.address_size,
                    )?;
                    unsafe { read_word(addr) }?
                }
                RegRule::ValExpr(start, end) => {
                    let end = end.min(frame_data.len());
                    if start > end {
                        return None;
                    }
                    let slice = frame_data.get(start..end)?;
                    eval_expr(slice, frame_base + start, &old.gpr, cfa, cie.address_size)?
                }
            };
            new_regs.gpr[i] = value;
        }

        new_regs.gpr[ret_sp_index()] = cfa;
        let next_ip = *new_regs.gpr.get(cie.ret_reg)?;
        if next_ip == 0 {
            return None;
        }
        let mut out = UnwindState::new();
        out.regs = new_regs;
        out.regs.ip = next_ip;
        out.sp = cfa;
        Some(out)
    }

    fn ret_sp_index() -> usize {
        if cfg!(target_arch = "aarch64") {
            31
        } else if cfg!(target_arch = "x86_64") {
            7
        } else {
            0
        }
    }

    /// Attempts one CFI step for `ip` using `tables` over `frame`.
    ///
    /// Returns `Ok(state)` on success, `Err(true)` when stepping must stop
    /// (signal frame / end), `Err(false)` when CFI is unavailable.
    pub(super) fn step(
        state: &mut UnwindState,
        tables: &UnwindTables,
        frame: &[u8],
    ) -> Result<Option<UnwindState>, bool> {
        let ip = state.regs.ip;
        let Some(fde) = find_fde(tables, frame, ip) else {
            return Err(false);
        };

        let mut cf = CfState::new();
        let mut base = CfState::new();
        let mut stack_buf = [None; STATE_STACK_LIMIT];
        let mut stack_len = 0usize;

        // CIE initial instructions.
        let cie_insts = frame
            .get(fde.cie.insts_off..fde.cie.insts_end)
            .unwrap_or(&[]);
        if cie_insts.is_empty() && fde.cie.insts_off < fde.cie.insts_end {
            return Err(false);
        }
        let mut cie_reader = SliceReader::new(
            cie_insts,
            tables.eh_frame.0.wrapping_add(fde.cie.insts_off),
            tables.datarel_base,
        );
        if exec_cfa(
            &mut cie_reader,
            &mut cf,
            &base,
            &mut stack_buf,
            &mut stack_len,
            fde.cie.code_factor,
            fde.cie.data_factor,
            fde.cie.address_size,
            0,
            0,
            false,
        )
        .is_none()
        {
            return Err(false);
        }
        base = cf;
        stack_len = 0;
        stack_buf = [None; STATE_STACK_LIMIT];

        // FDE instructions up to `ip`.
        let mut fde_reader = fde.insts;
        if exec_cfa(
            &mut fde_reader,
            &mut cf,
            &base,
            &mut stack_buf,
            &mut stack_len,
            fde.cie.code_factor,
            fde.cie.data_factor,
            fde.cie.address_size,
            fde.initial_location,
            ip,
            true,
        )
        .is_none()
        {
            return Err(false);
        }

        let mut next = apply_rules(&cf, &state.regs, fde.cie, frame, tables.eh_frame.0)
            .ok_or(true)?;
        if fde.cie.is_signal {
            next.stop = true;
        }
        Ok(Some(next))
    }
}

/// ELF (Linux/BSD) backend via `dl_iterate_phdr` + `.eh_frame`.
#[cfg(all(unix, not(target_vendor = "apple"), target_pointer_width = "64"))]
mod elf {
    use super::{Frame, UnwindState, cfi, fp};

    const PT_LOAD: u32 = 1;
    const PT_GNU_EH_FRAME: u32 = 0x6474_e550;
    const PF_X: u32 = 1;

    #[repr(C)]
    struct Phdr {
        p_type: u32,
        p_flags: u32,
        p_offset: u64,
        p_vaddr: u64,
        p_paddr: u64,
        p_filesz: u64,
        p_memsz: u64,
        p_align: u64,
    }

    #[repr(C)]
    struct DlPhdrInfo {
        dlpi_addr: usize,
        dlpi_name: *const i8,
        dlpi_phdr: *const Phdr,
        dlpi_phnum: u16,
    }

    unsafe extern "C" {
        fn dl_iterate_phdr(
            callback: extern "C" fn(
                *const DlPhdrInfo,
                usize,
                *mut core::ffi::c_void,
            ) -> i32,
            data: *mut core::ffi::c_void,
        ) -> i32;
    }

    #[derive(Clone, Copy, Default)]
    pub(super) struct Found {
        pub(super) base: Option<usize>,
        pub(super) eh_frame: Option<(usize, usize)>,
        pub(super) eh_frame_hdr: Option<(usize, usize)>,
        pub(super) datarel_base: usize,
    }

    struct Search {
        ip: usize,
        found: Found,
    }

    fn load_range(info: &DlPhdrInfo, ph: &Phdr) -> (usize, usize) {
        let start = info.dlpi_addr.wrapping_add(ph.p_vaddr as usize);
        let end = start.wrapping_add(ph.p_memsz as usize);
        (start, end)
    }

    extern "C" fn callback(
        info: *const DlPhdrInfo,
        _size: usize,
        data: *mut core::ffi::c_void,
    ) -> i32 {
        if info.is_null() || data.is_null() {
            return 0;
        }
        let info = unsafe { &*info };
        let search = unsafe { &mut *(data as *mut Search) };
        if search.found.base.is_some() {
            return 1;
        }
        if info.dlpi_phdr.is_null() || info.dlpi_phnum == 0 {
            return 0;
        }
        let phdrs = unsafe {
            core::slice::from_raw_parts(info.dlpi_phdr, usize::from(info.dlpi_phnum))
        };

        let mut min_start = usize::MAX;
        let mut in_text = false;
        let mut eh_hdr: Option<(usize, usize)> = None;
        for ph in phdrs {
            match ph.p_type {
                PT_LOAD => {
                    let (start, end) = load_range(info, ph);
                    if start < min_start {
                        min_start = start;
                    }
                    if ph.p_flags & PF_X != 0 && search.ip >= start && search.ip < end {
                        in_text = true;
                    }
                }
                PT_GNU_EH_FRAME => {
                    let start = info.dlpi_addr.wrapping_add(ph.p_vaddr as usize);
                    let len = ph.p_memsz as usize;
                    if len > 0 && len < (1 << 24) {
                        eh_hdr = Some((start, len));
                    }
                }
                _ => {}
            }
        }
        if !in_text {
            return 0;
        }

        let base = if min_start == usize::MAX {
            info.dlpi_addr
        } else {
            min_start
        };

        // Resolve `.eh_frame` from the GNU eh_frame_hdr pointer.
        let mut eh_frame = None;
        let mut datarel_base = 0;
        if let Some((hdr, hdr_len)) = eh_hdr {
            datarel_base = hdr;
            if let Some((ef, ef_len)) = parse_hdr_for_eh_frame(hdr, hdr_len, phdrs, info)
            {
                eh_frame = Some((ef, ef_len));
            }
        }
        if eh_frame.is_none() {
            // Fall back to section headers on the main image.
            if let Some(t) = sections_eh_frame(info.dlpi_addr, phdrs) {
                eh_frame = t.0;
                if eh_frame.is_some() {
                    if datarel_base == 0 {
                        datarel_base = t.1.unwrap_or(0);
                    }
                    eh_hdr = eh_hdr.or(t.2);
                }
            }
        }

        search.found = Found {
            base: Some(base),
            eh_frame,
            eh_frame_hdr: eh_hdr,
            datarel_base,
        };
        1
    }

    fn parse_hdr_for_eh_frame(
        hdr: usize,
        hdr_len: usize,
        phdrs: &[Phdr],
        info: &DlPhdrInfo,
    ) -> Option<(usize, usize)> {
        if hdr < 4096 || hdr_len < 4 {
            return None;
        }
        let bytes = unsafe { core::slice::from_raw_parts(hdr as *const u8, hdr_len) };
        if bytes[0] != 1 {
            return None;
        }
        let enc = bytes[1];
        // eh_frame_ptr at offset 4.
        let field = hdr + 4;
        let value = decode_at(&bytes[4..], enc, field, hdr)?;
        // Bound the section by the containing PT_LOAD.
        for ph in phdrs {
            if ph.p_type != PT_LOAD {
                continue;
            }
            let (start, end) = load_range(info, ph);
            if value >= start && value < end {
                return Some((value, end - value));
            }
        }
        // Unknown bounds: cap at 16 MiB.
        Some((value, 1 << 24))
    }

    fn decode_at(data: &[u8], enc: u8, field: usize, datarel: usize) -> Option<usize> {
        if enc == 0xff || data.is_empty() {
            return None;
        }
        let mut pos = 0usize;
        let raw = match enc & 0x0f {
            0x00 => {
                let n = core::mem::size_of::<usize>();
                if data.len() < n {
                    return None;
                }
                let mut v = 0usize;
                for (i, &byte) in data.iter().enumerate().take(n) {
                    v |= usize::from(byte) << (8 * i);
                }
                pos = n;
                v
            }
            0x01 => {
                let mut v = 0u64;
                let mut shift = 0;
                loop {
                    let b = *data.get(pos)?;
                    pos += 1;
                    v |= u64::from(b & 0x7f) << shift;
                    if b & 0x80 == 0 {
                        break;
                    }
                    shift += 7;
                    if shift >= 64 {
                        return None;
                    }
                }
                v as usize
            }
            0x02 => {
                if data.len() < 2 {
                    return None;
                }
                pos = 2;
                u16::from_le_bytes([data[0], data[1]]) as usize
            }
            0x03 => {
                if data.len() < 4 {
                    return None;
                }
                pos = 4;
                u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize
            }
            0x04 => {
                if data.len() < 8 {
                    return None;
                }
                pos = 8;
                u64::from_le_bytes(data[..8].try_into().ok()?) as usize
            }
            0x0b => {
                if data.len() < 4 {
                    return None;
                }
                pos = 4;
                i32::from_le_bytes(data[..4].try_into().ok()?) as usize
            }
            0x0c => {
                if data.len() < 8 {
                    return None;
                }
                pos = 8;
                i64::from_le_bytes(data[..8].try_into().ok()?) as usize
            }
            _ => return None,
        };
        let _ = pos;
        let value = match enc & 0x70 {
            0x00 => raw,
            0x10 => raw.wrapping_add(field),
            0x30 => raw.wrapping_add(datarel),
            _ => raw,
        };
        Some(value)
    }

    /// Section-header lookup result: `.eh_frame`, `.eh_frame_hdr` address,
    /// and `.eh_frame_hdr` range.
    type SectionsEhFrame = (
        Option<(usize, usize)>,
        Option<usize>,
        Option<(usize, usize)>,
    );

    /// Parses ELF section headers at load bias `bias` looking for `.eh_frame*`.
    fn sections_eh_frame(bias: usize, phdrs: &[Phdr]) -> Option<SectionsEhFrame> {
        // Locate the segment holding the ELF header (usually p_offset == 0).
        let mut ehdr_addr = 0usize;
        let mut found_ehdr = false;
        for ph in phdrs {
            if ph.p_type == PT_LOAD && ph.p_offset == 0 {
                ehdr_addr = bias.wrapping_add(ph.p_vaddr as usize);
                found_ehdr = true;
                break;
            }
        }
        if !found_ehdr || ehdr_addr < 4096 {
            return None;
        }
        let ehdr = unsafe { core::slice::from_raw_parts(ehdr_addr as *const u8, 64) };
        if ehdr.len() < 64 || ehdr[0..4] != [0x7f, b'E', b'L', b'F'] {
            return None;
        }
        let shoff = u64::from_le_bytes(ehdr[40..48].try_into().ok()?) as usize;
        let shentsize = u16::from_le_bytes(ehdr[58..60].try_into().ok()?) as usize;
        let shnum = u16::from_le_bytes(ehdr[60..62].try_into().ok()?) as usize;
        let shstrndx = u16::from_le_bytes(ehdr[62..64].try_into().ok()?) as usize;
        if shentsize < 64 || shnum == 0 || shnum > 4096 {
            return None;
        }
        let shdrs_addr = ehdr_addr.checked_add(shoff)?;
        let sh_size = shentsize.checked_mul(shnum)?;
        let shdrs =
            unsafe { core::slice::from_raw_parts(shdrs_addr as *const u8, sh_size) };
        let get = |i: usize| -> Option<&[u8]> {
            let off = i.checked_mul(shentsize)?;
            shdrs.get(off..off + 64)
        };
        let strtab = get(shstrndx)?;
        // ELF64 Shdr: sh_addr at 16, sh_size at 32.
        let str_addr = bias
            .wrapping_add(u64::from_le_bytes(strtab[16..24].try_into().ok()?) as usize);
        let str_size = u64::from_le_bytes(strtab[32..40].try_into().ok()?) as usize;
        if str_addr < 4096 || str_size > (1 << 24) {
            return None;
        }
        let strtab_bytes =
            unsafe { core::slice::from_raw_parts(str_addr as *const u8, str_size) };

        let mut eh_frame = None;
        let mut hdr = None;
        for i in 0..shnum {
            let sh = get(i)?;
            let name_off = u32::from_le_bytes(sh[0..4].try_into().ok()?) as usize;
            let sh_addr = bias
                .wrapping_add(u64::from_le_bytes(sh[16..24].try_into().ok()?) as usize);
            let sh_size = u64::from_le_bytes(sh[32..40].try_into().ok()?) as usize;
            let name = cstr_at(strtab_bytes, name_off)?;
            if name == b".eh_frame" && sh_size > 0 && sh_size < (1 << 28) {
                eh_frame = Some((sh_addr, sh_size));
            } else if name == b".eh_frame_hdr" && sh_size > 0 {
                hdr = Some((sh_addr, sh_size));
            }
        }
        eh_frame?;
        Some((eh_frame, hdr.map(|(a, _)| a), hdr))
    }

    fn cstr_at(data: &[u8], off: usize) -> Option<&[u8]> {
        let rest = data.get(off..)?;
        let end = rest.iter().position(|&b| b == 0)?;
        Some(&rest[..end])
    }

    pub(super) fn find(ip: usize) -> Found {
        let mut search = Search {
            ip,
            found: Found::default(),
        };
        unsafe {
            dl_iterate_phdr(
                callback,
                &mut search as *mut Search as *mut core::ffi::c_void,
            );
        }
        search.found
    }

    pub(super) fn module_base(ip: usize) -> Option<usize> {
        find(ip).base
    }

    pub(super) fn trace_inner(cb: &mut dyn FnMut(&Frame) -> bool) {
        super::walk(cb, Some(|state: &mut UnwindState| cfi_or_fp(state)));
    }

    fn cfi_or_fp(state: &mut UnwindState) -> bool {
        let ip = state.regs.ip;
        let found = find(ip);
        if let Some((ef, elen)) = found.eh_frame
            && ef >= 4096
            && elen > 0
            && elen < (1 << 28)
        {
            let frame = unsafe { core::slice::from_raw_parts(ef as *const u8, elen) };
            let tables = cfi::UnwindTables {
                eh_frame: (ef, elen),
                eh_frame_hdr: found.eh_frame_hdr,
                datarel_base: found.datarel_base,
            };
            match cfi::step(state, &tables, frame) {
                Ok(Some(next)) => {
                    *state = next;
                    return true;
                }
                Ok(None) | Err(true) => return false,
                Err(false) => {}
            }
        }
        fp::step(state)
    }
}

/// Mach-O (Apple) backend: `.eh_frame` from image segments + FP fallback.
#[cfg(all(unix, target_vendor = "apple", target_pointer_width = "64"))]
mod apple {
    use super::{Frame, UnwindState, cfi, fp};

    const LC_SEGMENT_64: u32 = 0x19;

    #[repr(C)]
    struct MachHeader64 {
        magic: u32,
        cputype: u32,
        cpusubtype: u32,
        filetype: u32,
        ncmds: u32,
        sizeofcmds: u32,
        flags: u32,
        reserved: u32,
    }

    #[repr(C)]
    struct SegmentCommand64 {
        cmd: u32,
        cmdsize: u32,
        segname: [u8; 16],
        vmaddr: u64,
        vmsize: u64,
        fileoff: u64,
        filesize: u64,
        maxprot: u32,
        initprot: u32,
        nsects: u32,
        flags: u32,
    }

    #[repr(C)]
    struct Section64 {
        sectname: [u8; 16],
        segname: [u8; 16],
        addr: u64,
        size: u64,
        offset: u32,
        align: u32,
        reloff: u32,
        nreloc: u32,
        flags: u32,
        reserved1: u32,
        reserved2: u32,
        reserved3: u32,
    }

    unsafe extern "C" {
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_header(index: u32) -> *const MachHeader64;
        fn _dyld_get_image_vmaddr_slide(index: u32) -> isize;
    }

    pub(super) struct Found {
        eh_frame: Option<(usize, usize)>,
        eh_frame_hdr: Option<(usize, usize)>,
        base: Option<usize>,
        datarel_base: usize,
    }

    fn scan_image(header: *const MachHeader64, slide: isize) -> Found {
        let mut found = Found {
            eh_frame: None,
            eh_frame_hdr: None,
            base: None,
            datarel_base: 0,
        };
        if header.is_null() {
            return found;
        }
        let mh = unsafe { &*header };
        if mh.magic != 0xfeed_facf || mh.ncmds > 4096 {
            return found;
        }
        let mut cmd_addr =
            (header as usize).wrapping_add(core::mem::size_of::<MachHeader64>());
        for _ in 0..mh.ncmds {
            let cmd = unsafe { core::ptr::read_unaligned(cmd_addr as *const u32) };
            let cmdsize = unsafe {
                core::ptr::read_unaligned(cmd_addr.wrapping_add(4) as *const u32)
            };
            if cmdsize < 8 || cmdsize as usize > (1 << 24) {
                break;
            }
            if cmd == LC_SEGMENT_64 {
                let seg = unsafe { &*(cmd_addr as *const SegmentCommand64) };
                let vmaddr = (seg.vmaddr as isize + slide) as usize;
                if found.base.is_none() || vmaddr < found.base.unwrap_or(usize::MAX) {
                    found.base = Some(vmaddr);
                }
                if seg.nsects > 0 && seg.nsects < 1024 {
                    let sects_addr =
                        cmd_addr.wrapping_add(core::mem::size_of::<SegmentCommand64>());
                    for s in 0..seg.nsects {
                        let sect = unsafe {
                            &*(sects_addr.wrapping_add(
                                s as usize * core::mem::size_of::<Section64>(),
                            ) as *const Section64)
                        };
                        let name = sect.sectname.split(|&b| b == 0).next().unwrap_or(b"");
                        let addr = (sect.addr as isize + slide) as usize;
                        let size = sect.size as usize;
                        if size == 0 || size > (1 << 28) {
                            continue;
                        }
                        if name == b"__eh_frame" {
                            found.eh_frame = Some((addr, size));
                        } else if name == b"__eh_frame_hdr" {
                            found.eh_frame_hdr = Some((addr, size));
                            found.datarel_base = addr;
                        }
                    }
                }
            }
            cmd_addr = cmd_addr.wrapping_add(cmdsize as usize);
        }
        found
    }

    fn find(_ip: usize) -> Found {
        let count = unsafe { _dyld_image_count() };
        for i in 0..count.min(4096) {
            let header = unsafe { _dyld_get_image_header(i) };
            let slide = unsafe { _dyld_get_image_vmaddr_slide(i) };
            let f = scan_image(header, slide);
            // Without phdr text ranges, accept the first image that has
            // unwind info when we cannot test containment precisely; refine
            // by checking FDE later. Prefer images with eh_frame.
            if f.eh_frame.is_some() {
                return f;
            }
        }
        Found {
            eh_frame: None,
            eh_frame_hdr: None,
            base: None,
            datarel_base: 0,
        }
    }

    pub(super) fn module_base(ip: usize) -> Option<usize> {
        find(ip).base
    }

    pub(super) fn trace_inner(cb: &mut dyn FnMut(&Frame) -> bool) {
        super::walk(
            cb,
            Some(|state: &mut UnwindState| {
                let found = find(state.regs.ip);
                if let Some((ef, elen)) = found.eh_frame {
                    let frame =
                        unsafe { core::slice::from_raw_parts(ef as *const u8, elen) };
                    let tables = cfi::UnwindTables {
                        eh_frame: (ef, elen),
                        eh_frame_hdr: found.eh_frame_hdr,
                        datarel_base: found.datarel_base,
                    };
                    match cfi::step(state, &tables, frame) {
                        Ok(Some(next)) => {
                            *state = next;
                            return true;
                        }
                        Err(true) | Ok(None) => return false,
                        Err(false) => {}
                    }
                }
                fp::step(state)
            }),
        );
    }
}

/// Windows x64/ARM64 backend using `RtlCaptureContext` / `RtlVirtualUnwind`.
#[cfg(all(
    windows,
    not(target_vendor = "uwp"),
    any(target_arch = "x86_64", target_arch = "aarch64"),
))]
mod windows {
    use super::{Frame, MAX_FRAMES, UnwindState, fp};
    use core::ffi::c_void;
    use core::ptr;

    // Parameter names are snake_case for the lints; they do not affect the
    // ABI of the imported symbols.
    windows_link::link!("kernel32.dll" "system" fn RtlCaptureContext(context_record: *mut c_void) -> ());
    windows_link::link!(
        "kernel32.dll" "system"
        fn RtlLookupFunctionEntry(
            control_pc: usize,
            image_base: *mut usize,
            history_table: *mut c_void,
        ) -> *mut RuntimeFunction
    );
    windows_link::link!(
        "kernel32.dll" "system"
        fn RtlVirtualUnwind(
            handler_type: u32,
            control_pc: usize,
            image_base: usize,
            function_entry: *mut RuntimeFunction,
            context_record: *mut c_void,
            handler_data: *mut *mut c_void,
            establisher_frame: *mut usize,
            context_pointers: *mut c_void,
        ) -> ()
    );

    #[repr(C)]
    struct RuntimeFunction {
        begin_address: u32,
        end_address: u32,
        unwind_data: u32,
    }

    /// Opaque, oversized, 16-byte-aligned stand-in for `CONTEXT`.
    #[repr(C, align(16))]
    struct ContextBuf([u8; 2048]);

    impl ContextBuf {
        const fn new() -> Self {
            Self([0; 2048])
        }

        fn zero(&mut self) {
            for b in self.0.iter_mut() {
                *b = 0;
            }
        }

        #[cfg(target_arch = "x86_64")]
        fn ip(&self) -> usize {
            read_u64(&self.0, 0xf8) as usize
        }

        #[cfg(target_arch = "x86_64")]
        fn set_ip(&mut self, v: usize) {
            write_u64(&mut self.0, 0xf8, v as u64);
        }

        #[cfg(target_arch = "x86_64")]
        fn sp(&self) -> usize {
            read_u64(&self.0, 0x98) as usize
        }

        #[cfg(target_arch = "x86_64")]
        fn fp(&self) -> usize {
            read_u64(&self.0, 0xa0) as usize
        }

        #[cfg(target_arch = "aarch64")]
        fn ip(&self) -> usize {
            read_u64(&self.0, 0x108) as usize
        }

        #[cfg(target_arch = "aarch64")]
        fn set_ip(&mut self, v: usize) {
            write_u64(&mut self.0, 0x108, v as u64);
        }

        #[cfg(target_arch = "aarch64")]
        fn sp(&self) -> usize {
            read_u64(&self.0, 0x100) as usize
        }

        #[cfg(target_arch = "aarch64")]
        fn fp(&self) -> usize {
            read_u64(&self.0, 0xf0) as usize
        }

        fn as_ptr(&mut self) -> *mut c_void {
            self.0.as_mut_ptr().cast()
        }
    }

    fn read_u64(buf: &[u8], off: usize) -> u64 {
        let b = match buf.get(off..off + 8) {
            Some(b) => b,
            None => return 0,
        };
        u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }

    fn write_u64(buf: &mut [u8; 2048], off: usize, v: u64) {
        if off + 8 > buf.len() {
            return;
        }
        let bytes = v.to_le_bytes();
        buf[off..off + 8].copy_from_slice(&bytes);
    }

    pub(super) fn trace_inner(cb: &mut dyn FnMut(&Frame) -> bool) {
        let mut ctx = ContextBuf::new();
        ctx.zero();
        unsafe { RtlCaptureContext(ctx.as_ptr()) };

        let mut prev_ip = ctx.ip();
        let mut prev_sp = ctx.sp();
        for _ in 0..MAX_FRAMES {
            let ip = ctx.ip();
            let sp = ctx.sp();
            if ip == 0 {
                break;
            }
            let mut base = 0usize;
            let entry = unsafe { RtlLookupFunctionEntry(ip, &mut base, ptr::null_mut()) };
            let module_base = if base != 0 { Some(base) } else { None };
            let frame = Frame {
                ip,
                sp,
                module_base,
            };
            if !cb(&frame) {
                return;
            }
            if entry.is_null() {
                // Leaf / no pdata: try a frame-pointer step on a shadow state.
                let mut st = UnwindState::new();
                st.regs.ip = ip;
                st.sp = sp;
                #[cfg(target_arch = "x86_64")]
                {
                    st.regs.gpr[6] = ctx.fp();
                    st.regs.gpr[7] = sp;
                }
                #[cfg(target_arch = "aarch64")]
                {
                    st.regs.gpr[29] = ctx.fp();
                    st.regs.gpr[31] = sp;
                }
                if !fp::step(&mut st) {
                    break;
                }
                ctx.set_ip(st.regs.ip);
                write_u64(&mut ctx.0, sp_offset(), st.sp as u64);
                #[cfg(target_arch = "x86_64")]
                write_u64(&mut ctx.0, 0xa0, st.regs.gpr[6] as u64);
                #[cfg(target_arch = "aarch64")]
                write_u64(&mut ctx.0, 0xf0, st.regs.gpr[29] as u64);
            } else {
                let mut handler_data: *mut c_void = ptr::null_mut();
                let mut establisher = 0usize;
                unsafe {
                    RtlVirtualUnwind(
                        0,
                        ip,
                        base,
                        entry,
                        ctx.as_ptr(),
                        &raw mut handler_data,
                        &raw mut establisher,
                        ptr::null_mut(),
                    );
                }
            }
            let next_ip = ctx.ip();
            let next_sp = ctx.sp();
            if next_ip == 0 || (next_ip == prev_ip && next_sp == prev_sp) {
                break;
            }
            prev_ip = next_ip;
            prev_sp = next_sp;
        }
    }

    #[cfg(target_arch = "x86_64")]
    const fn sp_offset() -> usize {
        0x98
    }

    #[cfg(target_arch = "aarch64")]
    const fn sp_offset() -> usize {
        0x100
    }
}

/// Shared frame walk driver.
///
/// `step_fn` is `None` on backends without CFI (pure frame-pointer walk).
#[cfg(any(
    all(unix, not(target_vendor = "apple"), target_pointer_width = "64"),
    all(unix, target_vendor = "apple", target_pointer_width = "64"),
    all(
        windows,
        not(target_vendor = "uwp"),
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    any(target_arch = "x86_64", target_arch = "aarch64"),
))]
#[cfg_attr(
    all(
        windows,
        not(target_vendor = "uwp"),
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    allow(dead_code)
)]
fn walk(
    cb: &mut dyn FnMut(&Frame) -> bool,
    step_fn: Option<fn(&mut UnwindState) -> bool>,
) {
    use cfg_if::cfg_if;

    cfg_if! {
        if #[cfg(all(unix, not(target_vendor = "apple"), target_pointer_width = "64"))] {
            let module_base = elf::module_base;
        } else if #[cfg(all(unix, target_vendor = "apple", target_pointer_width = "64"))] {
            let module_base = apple::module_base;
        } else {
            let module_base = |_: usize| None::<usize>;
        }
    }

    let Some(mut state) = capture::current() else {
        return;
    };

    for _ in 0..MAX_FRAMES {
        if state.regs.ip == 0 {
            break;
        }
        let frame = Frame {
            ip: state.regs.ip,
            sp: state.sp,
            module_base: module_base(state.regs.ip),
        };
        if !cb(&frame) {
            return;
        }
        if state.stop {
            break;
        }
        let stepped = match step_fn {
            Some(f) => f(&mut state),
            None => fp::step(&mut state),
        };
        if !stepped {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small stack-only frame collector used by nesting tests (no heap).
    struct FrameList {
        frames: [Frame; 64],
        len: usize,
    }

    impl FrameList {
        const fn new() -> Self {
            Self {
                frames: [Frame::new(0, 0, None); 64],
                len: 0,
            }
        }

        fn as_slice(&self) -> &[Frame] {
            &self.frames[..self.len]
        }
    }

    #[inline(never)]
    fn level_c(out: &mut FrameList) {
        collect(out);
    }

    #[inline(never)]
    fn level_b(out: &mut FrameList) {
        level_c(out);
    }

    #[inline(never)]
    fn level_a(out: &mut FrameList) {
        level_b(out);
    }

    #[inline(never)]
    fn collect(out: &mut FrameList) {
        out.len = capture_frames(&mut out.frames);
    }

    /// Maximum encoded size of a synthetic `.eh_frame` produced by
    /// [`synthetic_eh_frame`] (CIE ~18 B + FDE ~25 B on 64-bit; 128 is safe).
    const SYNTH_EH_FRAME_CAP: usize = 128;

    /// Builds a minimal synthetic `.eh_frame` with one CIE and one FDE
    /// covering `[fde_start, fde_start + range)` into `out`.
    ///
    /// CIE uses `ret_reg = 16`, `data_factor = -8`, `def_cfa rsp, 16`, and
    /// `offset r16, 1` (return address at `CFA - 8`).
    ///
    /// Returns the number of bytes written.
    fn synthetic_eh_frame(
        out: &mut [u8; SYNTH_EH_FRAME_CAP],
        fde_start: usize,
        range: usize,
    ) -> usize {
        // CIE body after length field:
        //   id=0, version=1, aug="", code_factor=1, data_factor=-8, ret_reg=16
        //   DW_CFA_def_cfa r7, 16; DW_CFA_offset r16, 1
        let cie_payload: &[u8] = &[
            0x00, 0x00, 0x00, 0x00, // CIE id
            0x01, // version
            0x00, // augmentation ""
            0x01, // code_factor = 1
            0x78, // data_factor = -8 (SLEB)
            0x10, // ret_reg = 16 (ULEB)
            0x0c, 0x07, 0x10, // DW_CFA_def_cfa r7, 16
            0x90, 0x01, // DW_CFA_offset r16, uleb 1
        ];
        let cie_len = cie_payload.len() as u32;
        let cie_total = 4 + cie_payload.len();
        // FDE: cie_pointer = distance from pointer field (cie_total + 4) back
        // to CIE start (0).
        let cie_ptr = (cie_total + 4) as u32;
        let fde_payload_len = 4 + core::mem::size_of::<usize>() * 2 + 1;
        let fde_len = fde_payload_len as u32;
        let mut pos = 0usize;

        let chunks: [&[u8]; 7] = [
            &cie_len.to_le_bytes(),
            cie_payload,
            &fde_len.to_le_bytes(),
            &cie_ptr.to_le_bytes(),
            &fde_start.to_le_bytes(),
            &range.to_le_bytes(),
            &[0x00], // DW_CFA_nop
        ];
        for chunk in chunks {
            let Some(end) = pos.checked_add(chunk.len()) else {
                return 0;
            };
            if end > out.len() {
                return 0;
            }
            out[pos..end].copy_from_slice(chunk);
            pos = end;
        }
        pos
    }

    #[test]
    fn captures_nested_frames() {
        let mut list = FrameList::new();
        level_a(&mut list);
        let frames = list.as_slice();
        assert!(
            frames.len() >= 3,
            "expected >= 3 frames, got {}",
            frames.len()
        );
        assert!(frames.iter().all(|f| f.ip() != 0));
    }

    #[test]
    fn stack_pointers_increase_outward() {
        let mut list = FrameList::new();
        level_a(&mut list);
        let frames = list.as_slice();
        assert!(
            frames.len() >= 2,
            "expected >= 2 frames, got {}",
            frames.len()
        );
        assert!(
            frames.windows(2).all(|w| w[1].sp() > w[0].sp()),
            "SP must increase toward callers"
        );
    }

    #[test]
    fn capture_frames_empty_slice_returns_zero() {
        let mut out: [Frame; 0] = [];
        assert_eq!(capture_frames(&mut out), 0);
    }

    #[test]
    fn capture_frames_respects_output_len() {
        let mut out = [Frame::new(0, 0, None); 3];
        let n = capture_frames(&mut out);
        assert_eq!(n, 3, "must fill the whole slice before stopping");
        assert!(out[..n].iter().all(|f| f.ip() != 0));
    }

    #[test]
    fn capture_frames_writes_most_recent_first() {
        let mut out = [Frame::new(0, 0, None); 16];
        let n = capture_frames(&mut out);
        assert!(n >= 2);
        assert!(
            out[..n].windows(2).all(|w| w[1].sp() >= w[0].sp()),
            "frames must be ordered most-recent-first (nondecreasing SP)"
        );
    }

    #[test]
    fn callback_false_stops_immediately() {
        let mut n = 0usize;
        trace(&mut |_| {
            n += 1;
            false
        });
        assert_eq!(n, 1);
    }

    #[test]
    fn symbol_address_is_non_null_when_traced() {
        let mut ok = false;
        trace(&mut |f| {
            ok = !f.symbol_address().is_null();
            false
        });
        assert!(ok);
    }

    #[test]
    fn trace_respects_frame_cap() {
        let mut n = 0usize;
        trace(&mut |_| {
            n += 1;
            true
        });
        assert!(n <= MAX_FRAMES, "n={n} > MAX_FRAMES");
        assert!(n >= 1, "expected at least the first frame");
    }

    #[test]
    fn read_word_rejects_null_page_and_misalignment() {
        // Safety: only addresses rejected by the low/misalign checks, plus a
        // mapped stack local.
        unsafe {
            assert_eq!(read_word(0), None);
            assert_eq!(read_word(8), None);
            assert_eq!(read_word(4095), None);
            assert_eq!(read_word(4097), None);
            assert_eq!(read_word(4103), None);
            let word = 0x0123_4567_89ab_cdefu64;
            let addr = (&raw const word).addr();
            assert_eq!(addr % core::mem::size_of::<usize>(), 0);
            assert_eq!(read_word(addr), Some(word as usize));
        }
    }

    #[test]
    fn capture_yields_sane_ip_and_sp() {
        let Some(state) = capture::current() else {
            return;
        };
        assert_ne!(state.regs.ip, 0);
        assert!(state.sp >= 4096, "sp should be a real stack address");
    }

    #[cfg(all(unix, not(target_vendor = "apple"), target_pointer_width = "64"))]
    #[test]
    fn module_base_none_for_invalid_ip() {
        assert_eq!(elf::module_base(0), None);
        assert_eq!(elf::module_base(1), None);
        assert_eq!(elf::module_base(usize::MAX), None);
        assert_eq!(elf::module_base(0xdead_beef_dead_beef), None);
    }

    #[cfg(all(unix, not(target_vendor = "apple"), target_pointer_width = "64"))]
    #[test]
    fn module_base_some_for_live_ip() {
        // The function's own address is a live IP inside this binary; using
        // it (rather than inline assembly reading the PC) keeps the test
        // portable across architectures.
        #[inline(never)]
        fn ip_here() -> usize {
            // Cast via a pointer: the `function_casts_as_integer` lint asks
            // for the intermediate pointer conversion even though the value
            // is identical.
            ip_here as *const () as usize
        }
        let ip = ip_here();
        let base = elf::module_base(ip);
        assert!(
            base.is_some(),
            "code in this binary must have a module base"
        );
        assert!(base.is_some_and(|b| b < ip));
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    mod fp_edge {
        use super::super::UnwindState;
        use super::super::fp;
        #[cfg(target_arch = "x86_64")]
        const FP: usize = 6;
        #[cfg(target_arch = "aarch64")]
        const FP: usize = 29;

        fn state_with(fp_val: usize, sp: usize) -> UnwindState {
            let mut st = UnwindState::new();
            st.regs.gpr[FP] = fp_val;
            st.regs.ip = 0x1000;
            st.sp = sp;
            st
        }

        #[test]
        fn rejects_zero_fp() {
            assert!(!fp::step(&mut state_with(0, 0x7fff_0000)));
        }

        #[test]
        fn rejects_misaligned_fp() {
            assert!(!fp::step(&mut state_with(0x1008, 0x1000)));
            assert!(!fp::step(&mut state_with(0x1010 & !0xf | 8, 0x1000)));
        }

        #[test]
        fn rejects_fp_below_sp() {
            let fp_val = 0x4000usize;
            assert!(!fp::step(&mut state_with(fp_val, fp_val + 16)));
        }

        #[test]
        fn advances_through_mapped_frame() {
            #[repr(C, align(16))]
            struct Slot {
                next_fp: usize,
                next_ip: usize,
            }
            let slot = Slot {
                next_fp: 0,
                next_ip: 0x1234_0000,
            };
            let fp_val = (&raw const slot).addr();
            assert_eq!(fp_val % 16, 0);
            let mut st = state_with(fp_val, fp_val.wrapping_sub(16));
            assert!(fp::step(&mut st));
            assert_eq!(st.regs.ip, 0x1234_0000);
            assert_eq!(st.sp, fp_val + 16);
            assert_eq!(st.regs.gpr[FP], 0);
            // Next step hits fp == 0 and stops.
            assert!(!fp::step(&mut st));
        }

        #[test]
        fn rejects_zero_next_ip() {
            #[repr(C, align(16))]
            struct Slot {
                next_fp: usize,
                next_ip: usize,
            }
            let slot = Slot {
                next_fp: 0,
                next_ip: 0,
            };
            let fp_val = (&raw const slot).addr();
            let mut st = state_with(fp_val, fp_val.wrapping_sub(16));
            assert!(!fp::step(&mut st));
        }

        #[test]
        fn rejects_non_increasing_next_fp() {
            #[repr(C, align(16))]
            struct Slot {
                next_fp: usize,
                next_ip: usize,
            }
            let mut slot = Slot {
                next_fp: 0,
                next_ip: 0x1234_0000,
            };
            let fp_val = (&raw mut slot).addr();
            slot.next_fp = fp_val;
            let mut st = state_with(fp_val, fp_val.wrapping_sub(16));
            // Read the written field so the store is not dead.
            assert_eq!(slot.next_fp, fp_val);
            assert!(!fp::step(&mut st));
        }
    }

    mod cfi_edge {
        use super::super::UnwindState;
        use super::super::cfi::{self, UnwindTables};
        use super::{SYNTH_EH_FRAME_CAP, synthetic_eh_frame};

        fn tables_for(frame: &[u8]) -> UnwindTables {
            UnwindTables {
                eh_frame: (frame.as_ptr() as usize, frame.len()),
                eh_frame_hdr: None,
                datarel_base: 0,
            }
        }

        fn assert_err_false(r: Result<Option<UnwindState>, bool>, ctx: &str) {
            match r {
                Err(false) => {}
                Err(true) => panic!("{ctx}: unexpected Err(true)"),
                Ok(None) => panic!("{ctx}: unexpected Ok(None)"),
                Ok(Some(_)) => panic!("{ctx}: unexpected Ok(Some)"),
            }
        }

        #[test]
        fn empty_frame_rejects_invalid_ip() {
            let mut st = UnwindState::new();
            st.regs.ip = 0;
            st.sp = 0x7fff_0000;
            let empty: [u8; 0] = [];
            assert_err_false(
                cfi::step(&mut st, &tables_for(&empty), &empty),
                "empty frame",
            );
        }

        #[test]
        fn garbage_eh_frame_does_not_panic() {
            let mut st = UnwindState::new();
            st.regs.ip = 0xdead_beef;
            st.sp = 0x7fff_0000;
            let pattern_a = [0x00u8; 64];
            let pattern_b = [0xffu8; 64];
            let mut pattern_c = [0u8; 64];
            for (i, b) in pattern_c.iter_mut().enumerate() {
                *b = [0xaa, 0x55, 0x01, 0x7f][i & 3];
            }
            for pattern in [
                pattern_a.as_slice(),
                pattern_b.as_slice(),
                pattern_c.as_slice(),
            ] {
                match cfi::step(&mut st, &tables_for(pattern), pattern) {
                    Err(_) | Ok(None) => {}
                    Ok(Some(_)) => panic!("garbage must not produce a step"),
                }
            }
        }

        #[test]
        fn unknown_ip_yields_no_fde() {
            let mut st = UnwindState::new();
            let fde_start = 0x1000_0000usize;
            let mut frame = [0u8; SYNTH_EH_FRAME_CAP];
            let flen = synthetic_eh_frame(&mut frame, fde_start, 0x100);
            let frame = &frame[..flen];
            let tables = tables_for(frame);
            st.regs.ip = fde_start + 0x200;
            assert_err_false(cfi::step(&mut st, &tables, frame), "past range");
            st.regs.ip = fde_start.wrapping_sub(1);
            assert_err_false(cfi::step(&mut st, &tables, frame), "before range");
            st.regs.ip = 0;
            assert_err_false(cfi::step(&mut st, &tables, frame), "ip=0");
        }

        #[test]
        fn synthetic_fde_steps_successfully() {
            let mut st = UnwindState::new();
            let fde_start = 0x1000_0000usize;
            let range = 0x100usize;
            let mut frame = [0u8; SYNTH_EH_FRAME_CAP];
            let flen = synthetic_eh_frame(&mut frame, fde_start, range);
            let frame = &frame[..flen];
            let tables = tables_for(frame);
            // Stack-resident fake stack image. CFA = rsp + 16 and RA is at
            // CFA - 8 = rsp + 8, so with rsp = base the RA lives at base + 8
            // (offset 16 from `rsp_var` below: base = rsp_var + 8).
            #[repr(C, align(16))]
            struct FakeStack {
                _keep: [usize; 2],
                ra_slot: usize,
                pad: [usize; 13],
            }
            let mut stack = FakeStack {
                _keep: [0; 2],
                ra_slot: 0x1000_0180,
                pad: [0; 13],
            };
            let base = (&raw mut stack).addr();
            st.regs.gpr[7] = base + 8;
            st.regs.ip = fde_start + 16;
            st.sp = base + 8;
            match cfi::step(&mut st, &tables, frame) {
                Ok(Some(next)) => {
                    assert_eq!(next.regs.ip, 0x1000_0180);
                }
                Ok(None) => panic!("synthetic step returned Ok(None)"),
                Err(false) => panic!("synthetic FDE not found"),
                Err(true) => panic!("synthetic step stopped unexpectedly"),
            }
        }

        #[test]
        fn out_of_range_ret_reg_is_rejected() {
            let mut frame = [0u8; SYNTH_EH_FRAME_CAP];
            let flen = synthetic_eh_frame(&mut frame, 0x1000_0000, 0x100);
            let frame = &mut frame[..flen];
            // CIE ret_reg ULEB sits at offset 12 in a short CIE.
            assert_eq!(frame[12], 0x10);
            frame[12] = 0x40; // 64 — first invalid index
            let mut st = UnwindState::new();
            st.regs.ip = 0x1000_0010;
            st.regs.gpr[7] = 0x2_0000;
            st.sp = 0x2_0000;
            let tables = tables_for(frame);
            if let Ok(Some(_)) = cfi::step(&mut st, &tables, frame) {
                panic!("OOB ret_reg must not produce a step");
            }
        }

        #[test]
        fn cie_pointer_must_resolve_to_cie_start() {
            let mut st = UnwindState::new();
            let fde_start = 0x2000_0000usize;
            let mut frame = [0u8; SYNTH_EH_FRAME_CAP];
            let flen = synthetic_eh_frame(&mut frame, fde_start, 0x40);
            let frame = &frame[..flen];
            let tables = tables_for(frame);
            let frame_ptr = frame.as_ptr() as usize;
            st.regs.gpr[7] = frame_ptr;
            st.regs.ip = fde_start + 8;
            st.sp = frame_ptr;
            match cfi::step(&mut st, &tables, frame) {
                Ok(Some(_)) => {}
                Err(false) => panic!("CIE at offset 0 must be found via cie_pointer"),
                Err(true) => panic!("unexpected stop"),
                Ok(None) => panic!("unexpected Ok(None)"),
            }
        }

        #[cfg(all(unix, not(target_vendor = "apple"), target_pointer_width = "64"))]
        #[test]
        fn real_eh_frame_steps_for_live_ip() {
            use super::super::capture;
            use super::super::elf;
            let Some(mut st) = capture::current() else {
                return;
            };
            let found = elf::find(st.regs.ip);
            let Some((ef, elen)) = found.eh_frame else {
                return;
            };
            assert!(ef >= 4096 && elen > 0 && elen < (1 << 28));
            // Safety: ef/elen come from the loader and bound a mapped region.
            let frame = unsafe { core::slice::from_raw_parts(ef as *const u8, elen) };
            let tables = UnwindTables {
                eh_frame: (ef, elen),
                eh_frame_hdr: found.eh_frame_hdr,
                datarel_base: found.datarel_base,
            };
            match cfi::step(&mut st, &tables, frame) {
                Ok(Some(next)) => {
                    assert_ne!(next.regs.ip, 0);
                    assert!(next.sp >= 4096);
                }
                Err(false) => panic!("live IP must have an FDE"),
                Err(true) => panic!("live IP step stopped unexpectedly"),
                Ok(None) => panic!("live IP apply_rules failed"),
            }
        }

        #[cfg(all(unix, not(target_vendor = "apple"), target_pointer_width = "64"))]
        #[test]
        fn real_eh_frame_rejects_corrupt_ip() {
            use super::super::capture;
            use super::super::elf;
            let Some(mut st) = capture::current() else {
                return;
            };
            let found = elf::find(st.regs.ip);
            let Some((ef, elen)) = found.eh_frame else {
                return;
            };
            let frame = unsafe { core::slice::from_raw_parts(ef as *const u8, elen) };
            let tables = UnwindTables {
                eh_frame: (ef, elen),
                eh_frame_hdr: found.eh_frame_hdr,
                datarel_base: found.datarel_base,
            };
            for bad in [0usize, 1, 0xdead, usize::MAX, 0xdead_beef_dead_beef] {
                st.regs.ip = bad;
                assert_err_false(
                    cfi::step(&mut st, &tables, frame),
                    "corrupt ip must miss FDE",
                );
            }
        }
    }

    /// `walk` is only compiled for these targets (see its own `#[cfg]`);
    /// the tests that drive it are gated identically so targets such as
    /// `wasm32` and 32-bit Unix still build.
    #[cfg(any(
        all(unix, not(target_vendor = "apple"), target_pointer_width = "64"),
        all(unix, target_vendor = "apple", target_pointer_width = "64"),
        all(
            windows,
            not(target_vendor = "uwp"),
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        any(target_arch = "x86_64", target_arch = "aarch64"),
    ))]
    mod walk_tests {
        use super::*;

        #[test]
        fn walk_terminates_after_capture_with_forced_stop() {
            // `walk` always captures first; force stop via callback false
            // after one frame and ensure no further frames are produced even
            // if the captured IP is valid.
            let mut n = 0usize;
            walk(
                &mut |_| {
                    n += 1;
                    false
                },
                None,
            );
            assert_eq!(n, 1);
        }

        #[test]
        fn walk_uses_step_fn_when_provided() {
            // A step_fn that always fails yields exactly one frame.
            let mut n = 0usize;
            walk(
                &mut |_| {
                    n += 1;
                    true
                },
                Some(|_| false),
            );
            assert_eq!(n, 1);
        }
    }

    #[cfg(all(unix, not(target_vendor = "apple"), target_pointer_width = "64"))]
    #[test]
    fn debug_walk_detail() {
        #[inline(never)]
        fn inner() {
            let mut list = FrameList::new();
            collect(&mut list);
            let frames = list.as_slice();
            std::eprintln!("frames={}", frames.len());
            for (i, f) in frames.iter().enumerate() {
                std::eprintln!(
                    "  [{i}] ip={:#x} sp={:#x} base={:?}",
                    f.ip(),
                    f.sp(),
                    elf::module_base(f.ip())
                );
            }
        }
        inner();
        // also direct CFI steps
        let Some(mut st) = capture::current() else {
            return;
        };
        let found = elf::find(st.regs.ip);
        std::eprintln!("eh_frame={:?}", found.eh_frame);
        if let Some((ef, elen)) = found.eh_frame {
            let frame = unsafe { core::slice::from_raw_parts(ef as *const u8, elen) };
            let tables = cfi::UnwindTables {
                eh_frame: (ef, elen),
                eh_frame_hdr: found.eh_frame_hdr,
                datarel_base: found.datarel_base,
            };
            {
                let ip = st.regs.ip;
                {
                    // re-parse via step internals isn't exposed; print encodings from hdr
                    if let Some((ha, hl)) = tables.eh_frame_hdr {
                        let hdr = unsafe {
                            core::slice::from_raw_parts(ha as *const u8, hl.min(64))
                        };
                        std::eprintln!(
                            "hdr bytes[0..16]={:02x?}",
                            &hdr[..16.min(hdr.len())]
                        );
                    }
                    std::eprintln!(
                        "frame addr={:#x} len={} datarel={:#x}",
                        tables.eh_frame.0,
                        frame.len(),
                        tables.datarel_base
                    );
                    std::eprintln!("ip={:#x} in_module={:?}", ip, elf::module_base(ip));
                }
            }
            for step in 0..6 {
                let ip = st.regs.ip;
                let fp = st.regs.gpr[6];
                let sp = st.sp;
                match cfi::step(&mut st, &tables, frame) {
                    Ok(Some(next)) => {
                        std::eprintln!(
                            "cfi step {step}: {ip:#x}->{:#x} sp {sp:#x}->{:#x} fp {fp:#x}->{:#x} next_base={:?}",
                            next.regs.ip,
                            next.sp,
                            next.regs.gpr[6],
                            elf::module_base(next.regs.ip)
                        );
                        st = next;
                    }
                    Err(e) => {
                        std::eprintln!(
                            "cfi step {step}: Err({e}) at ip={ip:#x} sp={sp:#x} fp={fp:#x}"
                        );
                        // try FP from here
                        if fp::step(&mut st) {
                            std::eprintln!("  fp fallback ok ip={:#x}", st.regs.ip);
                        } else {
                            std::eprintln!("  fp fallback fail");
                            break;
                        }
                    }
                    Ok(None) => {
                        std::eprintln!("cfi step {step}: Ok(None) at ip={ip:#x}");
                        break;
                    }
                }
            }
        }
        // FP steps from capture
        let Some(mut st) = capture::current() else {
            return;
        };
        std::eprintln!(
            "FP start ip={:#x} fp={:#x} sp={:#x}",
            st.regs.ip,
            st.regs.gpr[6],
            st.sp
        );
        for step in 0..8 {
            if fp::step(&mut st) {
                std::eprintln!(
                    "fp step {step}: ip={:#x} fp={:#x} sp={:#x}",
                    st.regs.ip,
                    st.regs.gpr[6],
                    st.sp
                );
            } else {
                std::eprintln!("fp step {step}: stop");
                break;
            }
        }
    }

    #[test]
    fn deep_recursion_stays_within_cap() {
        #[inline(never)]
        fn rec(depth: usize, out: &mut FrameList) {
            if depth == 0 {
                collect(out);
            } else {
                rec(depth - 1, out);
            }
        }
        let mut list = FrameList::new();
        rec(16, &mut list);
        let frames = list.as_slice();
        assert!(!frames.is_empty());
        assert!(frames.len() <= MAX_FRAMES);
        assert!(frames.iter().all(|f| f.ip() != 0));
    }
}
