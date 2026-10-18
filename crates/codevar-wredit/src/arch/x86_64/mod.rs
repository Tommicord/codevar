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

//! x86_64 assembly bindings with run-time CPU dispatch.
//!
//! The `memmove`/`memcpy`/`memchr` sources are assembled once per
//! instruction-set level (SSE2, AVX2, AVX-512) by `build.rs`, and this
//! module selects the widest level the running CPU supports through
//! [`codevar_base::cpuid`]. The level is cached in an atomic so the hot
//! path costs a single relaxed load.
//!
//! The prefetch helpers use only the SSE-era `prefetch` instruction and
//! therefore have a single, unconditionally available implementation.

use core::sync::atomic::{AtomicU8, Ordering};

use codevar_base::cpuid::{Feature, has};

// ---------------------------------------------------------------------
// Raw assembly entry points (one set per ISA level).
// ---------------------------------------------------------------------
unsafe extern "C" {
    fn codevar_wredit_memmove_sse2(dst: *mut u8, src: *const u8, len: usize);
    fn codevar_wredit_memmove_avx2(dst: *mut u8, src: *const u8, len: usize);
    fn codevar_wredit_memmove_avx512(dst: *mut u8, src: *const u8, len: usize);

    fn codevar_wredit_memcpy_sse2(dst: *mut u8, src: *const u8, len: usize);
    fn codevar_wredit_memcpy_avx2(dst: *mut u8, src: *const u8, len: usize);
    fn codevar_wredit_memcpy_avx512(dst: *mut u8, src: *const u8, len: usize);

    fn codevar_wredit_memchr_sse2(haystack: *const u8, needle: u8, len: usize) -> *const u8;
    fn codevar_wredit_memchr_avx2(haystack: *const u8, needle: u8, len: usize) -> *const u8;
    fn codevar_wredit_memchr_avx512(haystack: *const u8, needle: u8, len: usize) -> *const u8;

    fn codevar_wredit_prefetch_t0(addr: *const u8);
    fn codevar_wredit_prefetch_t1(addr: *const u8);
    fn codevar_wredit_prefetch_t2(addr: *const u8);
    fn codevar_wredit_prefetch_nta(addr: *const u8);
    fn codevar_wredit_prefetch_range(addr: *const u8, len: usize);
}

/// Sentinel meaning "not probed yet".
const LEVEL_UNINIT: u8 = u8::MAX;
/// SSE2 baseline (always available on x86_64).
const LEVEL_SSE2: u8 = 0;
/// AVX2.
const LEVEL_AVX2: u8 = 1;
/// AVX-512 (F + BW + VL).
const LEVEL_AVX512: u8 = 2;

/// Cached ISA level, probed once on first use.
static LEVEL: AtomicU8 = AtomicU8::new(LEVEL_UNINIT);

/// Probes the CPU for the widest supported ISA level. Cold path: runs
/// only once per process.
#[cold]
fn detect_level() -> u8 {
    if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx512Vl) {
        LEVEL_AVX512
    } else if has(Feature::Avx2) && has(Feature::Avx) {
        LEVEL_AVX2
    } else {
        LEVEL_SSE2
    }
}

/// Returns the cached ISA level, initializing it on first use.
#[inline]
fn level() -> u8 {
    let cached = LEVEL.load(Ordering::Relaxed);
    if cached != LEVEL_UNINIT {
        return cached;
    }
    let detected = detect_level();
    LEVEL.store(detected, Ordering::Relaxed);
    detected
}

/// Copies `len` bytes from `src` to `dst`. The regions must not overlap;
/// use [`memmove`] when they might.
///
/// # Safety
///
/// `src` must be valid for reads of `len` bytes and `dst` for writes of
/// `len` bytes, with no overlap.
#[inline]
pub unsafe fn memcpy(dst: *mut u8, src: *const u8, len: usize) {
    if len >= super::CACHE_LINE_SIZE {
        // Safety: prefetch never faults.
        unsafe { codevar_wredit_prefetch_range(dst, len) };
    }
    // Safety: guaranteed by the caller; the ISA level matches the CPU.
    unsafe {
        match level() {
            LEVEL_AVX512 => codevar_wredit_memcpy_avx512(dst, src, len),
            LEVEL_AVX2 => codevar_wredit_memcpy_avx2(dst, src, len),
            _ => codevar_wredit_memcpy_sse2(dst, src, len),
        }
    }
}

/// Copies `len` bytes from `src` to `dst`, correctly handling overlap
/// by copying forward when `dst < src` and backward when `dst > src`.
///
/// # Safety
///
/// `src` must be valid for reads of `len` bytes and `dst` for writes of
/// `len` bytes; the regions may overlap.
#[inline]
pub unsafe fn memmove(dst: *mut u8, src: *const u8, len: usize) {
    if len >= super::CACHE_LINE_SIZE {
        // Safety: prefetch never faults.
        unsafe {
            codevar_wredit_prefetch_range(src, len);
            codevar_wredit_prefetch_range(dst, len);
        }
    }
    // Safety: guaranteed by the caller; the ISA level matches the CPU.
    unsafe {
        match level() {
            LEVEL_AVX512 => codevar_wredit_memmove_avx512(dst, src, len),
            LEVEL_AVX2 => codevar_wredit_memmove_avx2(dst, src, len),
            _ => codevar_wredit_memmove_sse2(dst, src, len),
        }
    }
}

/// Returns a pointer to the first occurrence of `needle`, or a null
/// pointer when it does not appear in `haystack[..len]`.
///
/// # Safety
///
/// `haystack` must be valid for reads of `len` bytes.
#[inline]
pub unsafe fn memchr(haystack: *const u8, needle: u8, len: usize) -> *const u8 {
    if len >= super::CACHE_LINE_SIZE {
        // Safety: prefetch never faults.
        unsafe { codevar_wredit_prefetch_range(haystack, len) };
    }
    // Safety: guaranteed by the caller; the ISA level matches the CPU.
    unsafe {
        match level() {
            LEVEL_AVX512 => codevar_wredit_memchr_avx512(haystack, needle, len),
            LEVEL_AVX2 => codevar_wredit_memchr_avx2(haystack, needle, len),
            _ => codevar_wredit_memchr_sse2(haystack, needle, len),
        }
    }
}

/// Prefetch the cache line containing `addr` with the given hint.
///
/// # Safety
///
/// `addr` is only a hint and may point anywhere (even unmapped memory);
/// the instruction never faults.
#[inline]
pub unsafe fn prefetch(addr: *const u8, hint: u8) {
    // Safety: prefetch never faults.
    unsafe {
        match hint {
            super::PREFETCH_T1 => codevar_wredit_prefetch_t1(addr),
            super::PREFETCH_T2 => codevar_wredit_prefetch_t2(addr),
            super::PREFETCH_NTA => codevar_wredit_prefetch_nta(addr),
            _ => codevar_wredit_prefetch_t0(addr),
        }
    }
}

/// Prefetch every 64-byte cache line across `[addr, addr + len)`.
///
/// # Safety
///
/// `addr` is only a prefetch hint; the instruction never faults.
#[inline]
pub unsafe fn prefetch_range(addr: *const u8, len: usize) {
    // Safety: prefetch never faults.
    unsafe { codevar_wredit_prefetch_range(addr, len) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memcpy_and_memmove_match_source() {
        let src: [u8; 256] = core::array::from_fn(|i| (i % 251) as u8);
        let mut dst = [0u8; 256];
        unsafe { memcpy(dst.as_mut_ptr(), src.as_ptr(), src.len()) };
        assert_eq!(dst, src);

        // Forward overlap (dst < src): shift right by 17.
        let mut buf = [0u8; 256];
        buf.copy_from_slice(&src);
        unsafe { memmove(buf.as_mut_ptr().add(17), buf.as_ptr(), 200) };
        for i in 0..200 {
            assert_eq!(buf[17 + i], src[i], "forward byte {i}");
        }

        // Backward overlap (dst > src): shift left by 17.
        let mut buf2 = [0u8; 256];
        buf2.copy_from_slice(&src);
        unsafe { memmove(buf2.as_mut_ptr(), buf2.as_ptr().add(17), 200) };
        for i in 0..200 {
            assert_eq!(buf2[i], src[17 + i], "backward byte {i}");
        }
    }

    #[test]
    fn memchr_matches_scalar_scan() {
        let hay = b"hello";
        let found = unsafe { memchr(hay.as_ptr(), b'o', hay.len()) };
        assert_eq!(found as usize - hay.as_ptr() as usize, 4);
        assert!(unsafe { memchr(hay.as_ptr(), b'z', hay.len()) }.is_null());
        assert!(unsafe { memchr(hay.as_ptr(), b'h', 0) }.is_null());

        // Needle at the end exercises the vector tail.
        let mut buf = [0x11u8; 300];
        buf[298] = 0x77;
        let found = unsafe { memchr(buf.as_ptr(), 0x77, buf.len()) };
        assert_eq!(found as usize - buf.as_ptr() as usize, 298);
    }

    #[test]
    fn prefetch_helpers_do_not_crash() {
        let data = [0u8; 4096];
        unsafe {
            prefetch(data.as_ptr(), super::super::PREFETCH_T0);
            prefetch(data.as_ptr(), super::super::PREFETCH_T1);
            prefetch(data.as_ptr(), super::super::PREFETCH_T2);
            prefetch(data.as_ptr(), super::super::PREFETCH_NTA);
            prefetch_range(data.as_ptr(), data.len());
        }
    }
}
