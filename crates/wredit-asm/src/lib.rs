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

//! Architecture-specific optimized memory and cache primitives.
//!
//! Every supported target has a hand-written assembly implementation
//! (under `src/arch/<arch>/`), compiled by [`build.rs`] through the
//! `cc` crate and selected here at compile time. Targets without an
//! assembly implementation fall back to `core::ptr` copies.
//!
//! Where the ISA offers several levels (x86_64: SSE2 / AVX2 / AVX-512;
//! aarch64: NEON / SVE), the per-architecture module performs *run-time*
//! dispatch through [`codevar_base::cpuid`] so a single binary runs on
//! the widest instruction set the current CPU supports without faulting
//! on older hardware.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "x86")]
pub mod i686;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(all(target_arch = "arm", not(target_feature = "thumb-mode")))]
pub mod armv7;

#[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))]
pub mod thumbv7m;

#[cfg(target_arch = "riscv64")]
pub mod riscv64;

#[cfg(target_arch = "riscv32")]
pub mod riscv32;

/// Cache-line size in bytes, used as the prefetch stride and for
/// buffer-padding decisions. 64 bytes covers the L1 cache line on
/// x86_64 and the common ARM/RISC-V cores, matching the stride of the
/// assembly `prefetch_range` routines.
pub const CACHE_LINE_SIZE: usize = 64;

/// Prefetch hint: load into all cache levels (temporal, expect reuse).
pub const PREFETCH_T0: u8 = 0;

/// Prefetch hint: load into L2 and beyond.
pub const PREFETCH_T1: u8 = 1;

/// Prefetch hint: load into L3 and beyond.
pub const PREFETCH_T2: u8 = 2;

/// Prefetch hint: non-temporal, do not pollute the cache (streaming).
pub const PREFETCH_NTA: u8 = 3;

#[cfg(target_arch = "aarch64")]
use self::aarch64 as imp;
#[cfg(all(target_arch = "arm", not(target_feature = "thumb-mode")))]
use self::armv7 as imp;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "x86",
    target_arch = "aarch64",
    target_arch = "riscv64",
    target_arch = "riscv32",
    target_arch = "arm",
)))]
use self::fallback as imp;
#[cfg(target_arch = "x86")]
use self::i686 as imp;
#[cfg(target_arch = "riscv32")]
use self::riscv32 as imp;
#[cfg(target_arch = "riscv64")]
use self::riscv64 as imp;
#[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))]
use self::thumbv7m as imp;
/// The implementation module for the current target.
#[cfg(target_arch = "x86_64")]
use self::x86_64 as imp;

/// Pure-Rust fallback for targets without an assembly implementation
/// (for example `wasm32`).
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "x86",
    target_arch = "aarch64",
    target_arch = "riscv64",
    target_arch = "riscv32",
    target_arch = "arm",
)))]
mod fallback {
    /// Copies `len` bytes from `src` to `dst`; the regions must not
    /// overlap.
    ///
    /// # Safety
    ///
    /// Both ranges must be valid for `len` bytes and must not overlap.
    #[inline]
    pub unsafe fn memcpy(dst: *mut u8, src: *const u8, len: usize) {
        // Safety: guaranteed by the caller.
        unsafe { core::ptr::copy_nonoverlapping(src, dst, len) };
    }

    /// Copies `len` bytes from `src` to `dst`, handling overlap.
    ///
    /// # Safety
    ///
    /// Both ranges must be valid for `len` bytes; overlap is allowed.
    #[inline]
    pub unsafe fn memmove(dst: *mut u8, src: *const u8, len: usize) {
        // Safety: guaranteed by the caller.
        unsafe { core::ptr::copy(src, dst, len) };
    }

    /// Returns a pointer to the first occurrence of `needle`, or null.
    ///
    /// # Safety
    ///
    /// `haystack` must be valid for reads of `len` bytes.
    #[inline]
    pub unsafe fn memchr(haystack: *const u8, needle: u8, len: usize) -> *const u8 {
        if len == 0 {
            return core::ptr::null();
        }
        // Safety: guaranteed by the caller.
        let slice = unsafe { core::slice::from_raw_parts(haystack, len) };
        match slice.iter().position(|&b| b == needle) {
            // Safety: `i < len`.
            Some(i) => unsafe { haystack.add(i) },
            None => core::ptr::null(),
        }
    }

    /// Prefetch is a no-op without an assembly implementation.
    ///
    /// # Safety
    ///
    /// Prefetch never faults.
    #[inline]
    pub unsafe fn prefetch(_addr: *const u8, _hint: u8) {}

    /// Prefetch is a no-op without an assembly implementation.
    ///
    /// # Safety
    ///
    /// Prefetch never faults.
    #[inline]
    pub unsafe fn prefetch_range(_addr: *const u8, _len: usize) {}
}

/// Copies `len` bytes from `src` to `dst`. The regions must not
/// overlap; use [`memmove`] when they might.
///
/// # Safety
///
/// `src` must be valid for reads of `len` bytes and `dst` for writes of
/// `len` bytes, with no overlap.
#[inline]
pub unsafe fn memcpy(dst: *mut u8, src: *const u8, len: usize) {
    // Safety: forwarded to the caller.
    unsafe { imp::memcpy(dst, src, len) };
}

/// Copies `len` bytes from `src` to `dst`, correctly handling overlap.
///
/// # Safety
///
/// `src` must be valid for reads of `len` bytes and `dst` for writes of
/// `len` bytes; the regions may overlap.
#[inline]
pub unsafe fn memmove(dst: *mut u8, src: *const u8, len: usize) {
    // Safety: forwarded to the caller.
    unsafe { imp::memmove(dst, src, len) };
}

/// Returns a pointer to the first occurrence of `needle` in
/// `haystack[..len]`, or `None` when the byte does not appear.
///
/// # Safety
///
/// `haystack` must be valid for reads of `len` bytes.
#[inline]
pub unsafe fn memchr(haystack: *const u8, needle: u8, len: usize) -> Option<*const u8> {
    if len == 0 || haystack.is_null() {
        return None;
    }
    // Safety: the caller guarantees `haystack` is valid for `len` bytes.
    let found = unsafe { imp::memchr(haystack, needle, len) };
    if found.is_null() { None } else { Some(found) }
}

/// Issues a prefetch for `addr` with the given hint (one of
/// [`PREFETCH_T0`], [`PREFETCH_T1`], [`PREFETCH_T2`], [`PREFETCH_NTA`]).
///
/// # Safety
///
/// `addr` is only a hint and may point anywhere, even unmapped memory:
/// the underlying instruction never faults.
#[inline]
pub unsafe fn prefetch(addr: *const u8, hint: u8) {
    // Safety: forwarded to the caller.
    unsafe { imp::prefetch(addr, hint) };
}

/// Prefetches every cache line across `[addr, addr + len)`.
///
/// # Safety
///
/// `addr` is only a hint and may point anywhere; the underlying
/// instruction never faults.
#[inline]
pub unsafe fn prefetch_range(addr: *const u8, len: usize) {
    // Safety: forwarded to the caller.
    unsafe { imp::prefetch_range(addr, len) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memcpy_and_memmove_agree_with_scalar() {
        let src: [u8; 64] = core::array::from_fn(|i| i as u8);
        let mut dst = [0u8; 64];

        unsafe { memcpy(dst.as_mut_ptr(), src.as_ptr(), src.len()) };
        assert_eq!(dst, src);

        // Overlapping shift right by 4 elements.
        let mut overlap: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        unsafe { memmove(overlap.as_mut_ptr().add(4), overlap.as_ptr(), 12) };
        assert_eq!(overlap, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        // Overlapping shift left by 4 elements.
        let mut overlap2: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        unsafe { memmove(overlap2.as_mut_ptr(), overlap2.as_ptr().add(4), 12) };
        assert_eq!(
            overlap2,
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16]
        );
    }

    #[test]
    fn memchr_finds_and_misses() {
        let haystack = b"hello world";
        let expected = haystack.as_ptr() as usize + 4;
        // Safety: `haystack` is a valid slice for its whole length.
        let result = unsafe { memchr(haystack.as_ptr(), b'o', haystack.len()) };
        assert_eq!(result.map(|p| p as usize), Some(expected));

        assert_eq!(unsafe { memchr(haystack.as_ptr(), b'z', haystack.len()) }, None);
        assert_eq!(unsafe { memchr(haystack.as_ptr(), b'h', 0) }, None);

        // Needle in the final partial chunk exercises the vector tail.
        let mut buf = [0x11u8; 300];
        buf[297] = 0x77;
        // Safety: `buf` is a valid slice for its whole length.
        let found = unsafe { memchr(buf.as_ptr(), 0x77, buf.len()) };
        assert_eq!(found, Some(unsafe { buf.as_ptr().add(297) }));
    }

    #[test]
    fn prefetch_is_a_noop_without_crash() {
        let data = [0u8; 4096];
        unsafe {
            prefetch(data.as_ptr(), PREFETCH_T0);
            prefetch(data.as_ptr(), PREFETCH_T1);
            prefetch(data.as_ptr(), PREFETCH_T2);
            prefetch(data.as_ptr(), PREFETCH_NTA);
            prefetch_range(data.as_ptr(), data.len());
        }
    }
}
