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
    fn codevar_wredit_memchr_sse42(haystack: *const u8, needle: u8, len: usize) -> *const u8;

    // Spill-free packed-word newline kernels (`wordscan.S`), one set
    // per ISA level.
    fn codevar_wredit_wc_count_sse2(raw: *const u32, len: usize) -> usize;
    fn codevar_wredit_wc_count_avx2(raw: *const u32, len: usize) -> usize;
    fn codevar_wredit_wc_count_avx512(raw: *const u32, len: usize) -> usize;

    fn codevar_wredit_wc_find_nth_sse2(raw: *const u32, len: usize, n: usize) -> usize;
    fn codevar_wredit_wc_find_nth_avx2(raw: *const u32, len: usize, n: usize) -> usize;
    fn codevar_wredit_wc_find_nth_avx512(raw: *const u32, len: usize, n: usize) -> usize;

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
/// SSE4.2 (PCMPESTRI string-search instruction).
const LEVEL_SSE42: u8 = 1;
/// AVX2.
const LEVEL_AVX2: u8 = 2;
/// AVX-512 (F + BW + VL).
const LEVEL_AVX512: u8 = 3;

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
    } else if has(Feature::Sse42) {
        LEVEL_SSE42
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
    // The copy loops stream on their own; prefetching every cache line
    // up front costs an instruction per line and starves the load/store
    // ports, so it is deliberately left off here.
    //
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
    // Safety: guaranteed by the caller; the ISA level matches the CPU.
    unsafe {
        match level() {
            LEVEL_AVX512 => codevar_wredit_memchr_avx512(haystack, needle, len),
            LEVEL_AVX2 => codevar_wredit_memchr_avx2(haystack, needle, len),
            LEVEL_SSE42 => codevar_wredit_memchr_sse42(haystack, needle, len),
            _ => codevar_wredit_memchr_sse2(haystack, needle, len),
        }
    }
}

/// Counts packed newline words in `raw[..len]`.
///
/// # Safety
///
/// `raw` must be valid for reads of `len` `u32` words, or null when
/// `len == 0`.
#[inline]
pub(crate) unsafe fn newline_count(raw: *const u32, len: usize) -> usize {
    // Safety: guaranteed by the caller; the selected kernel matches
    // the probed ISA level and never reads past `len`.
    unsafe {
        match level() {
            LEVEL_AVX512 => codevar_wredit_wc_count_avx512(raw, len),
            LEVEL_AVX2 => codevar_wredit_wc_count_avx2(raw, len),
            _ => codevar_wredit_wc_count_sse2(raw, len),
        }
    }
}

/// Finds the `n`-th packed newline (one-based) in `raw[..len]`.
///
/// # Safety
///
/// `raw` must be valid for reads of `len` words; `n > 0`.
#[inline]
pub(crate) unsafe fn newline_nth(raw: *const u32, len: usize, n: usize) -> Option<usize> {
    // Safety: see `newline_count`.
    let found = unsafe {
        match level() {
            LEVEL_AVX512 => codevar_wredit_wc_find_nth_avx512(raw, len, n),
            LEVEL_AVX2 => codevar_wredit_wc_find_nth_avx2(raw, len, n),
            _ => codevar_wredit_wc_find_nth_sse2(raw, len, n),
        }
    };
    (found != usize::MAX).then_some(found)
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
    fn memchr_sse42_matches_scalar_scan() {
        if !has(Feature::Sse42) {
            return; // PCMPESTRI requires an SSE4.2-capable CPU.
        }
        // Scalar reference: first index of `needle` in `hay[..len]`, or
        // `len` when absent (mirrors the NULL return as "past the end").
        let scalar = |hay: &[u8], needle: u8, len: usize| -> usize {
            hay[..len]
                .iter()
                .position(|&b| b == needle)
                .unwrap_or(len)
        };

        // Covers: start, middle, chunk boundary (16), end of a chunk,
        // just past a chunk boundary, multi-chunk wide loop, absent, and
        // a needle spanning the 64-byte wide-loop boundary.
        let mut hay = [0x41u8; 200];
        hay[0] = 0x01;
        hay[16] = 0x02;
        hay[31] = 0x03;
        hay[48] = 0x04;
        hay[63] = 0x05;
        hay[64] = 0x06;
        hay[127] = 0x07;
        hay[128] = 0x08;
        hay[199] = 0x09;

        for (needle, want) in [
            (0x01, 0),
            (0x02, 16),
            (0x03, 31),
            (0x04, 48),
            (0x05, 63),
            (0x06, 64),
            (0x07, 127),
            (0x08, 128),
            (0x09, 199),
            (0x00, 200), // absent: "past the end"
        ] {
            let got = unsafe {
                let p = codevar_wredit_memchr_sse42(hay.as_ptr(), needle, hay.len());
                if p.is_null() {
                    hay.len()
                } else {
                    p as usize - hay.as_ptr() as usize
                }
            };
            assert_eq!(got, want, "needle 0x{needle:02x}");
            assert_eq!(got, scalar(&hay, needle, hay.len()), "needle 0x{needle:02x}");
        }

        // Short lengths that never reach the wide loop.
        for len in 0..17 {
            for &needle in &[0x01, 0x41] {
                let got = unsafe {
                    let p = codevar_wredit_memchr_sse42(hay.as_ptr(), needle, len);
                    if p.is_null() {
                        len
                    } else {
                        p as usize - hay.as_ptr() as usize
                    }
                };
                assert_eq!(got, scalar(&hay, needle, len), "len {len} needle 0x{needle:02x}");
            }
        }
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
