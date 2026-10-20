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

//! AArch64 assembly bindings.
//!
//! Advanced SIMD (NEON) is mandatory in ARMv8-A, so the baseline kernels
//! need no runtime check. SVE is optional: an SVE copy kernel is also
//! linked and selected through [`codevar_base::cpuid`] when the CPU
//! reports it. `memchr` has no SVE path (NEON is already sufficient).

use core::sync::atomic::{AtomicU8, Ordering};

use codevar_base::cpuid::{Feature, has};

unsafe extern "C" {
    fn codevar_wredit_memmove(dst: *mut u8, src: *const u8, len: usize);
    fn codevar_wredit_memmove_sve(dst: *mut u8, src: *const u8, len: usize);
    fn codevar_wredit_memcpy(dst: *mut u8, src: *const u8, len: usize);
    fn codevar_wredit_memcpy_sve(dst: *mut u8, src: *const u8, len: usize);
    fn codevar_wredit_memchr(haystack: *const u8, needle: u8, len: usize) -> *const u8;

    fn codevar_wredit_prefetch_t0(addr: *const u8);
    fn codevar_wredit_prefetch_t1(addr: *const u8);
    fn codevar_wredit_prefetch_t2(addr: *const u8);
    fn codevar_wredit_prefetch_nta(addr: *const u8);
    fn codevar_wredit_prefetch_range(addr: *const u8, len: usize);
}

/// Sentinel meaning "not probed yet".
const UNINIT: u8 = u8::MAX;

/// Cached SVE availability, probed once on first use.
static HAS_SVE: AtomicU8 = AtomicU8::new(UNINIT);

/// Returns `true` when SVE is usable, caching the probe result.
#[inline]
fn sve_enabled() -> bool {
    let cached = HAS_SVE.load(Ordering::Relaxed);
    if cached != UNINIT {
        return cached != 0;
    }
    let enabled = u8::from(has(Feature::Sve));
    HAS_SVE.store(enabled, Ordering::Relaxed);
    enabled != 0
}

/// Copies `len` bytes from `src` to `dst`. The regions must not overlap.
///
/// # Safety
///
/// Both ranges must be valid for `len` bytes and must not overlap.
#[inline]
pub unsafe fn memcpy(dst: *mut u8, src: *const u8, len: usize) {
    // The copy loops stream on their own; prefetching every cache line
    // up front costs an instruction per line and starves the load/store
    // ports, so it is deliberately left off here.
    //
    // Safety: guaranteed by the caller; the ISA path matches the CPU.
    unsafe {
        if sve_enabled() {
            codevar_wredit_memcpy_sve(dst, src, len);
        } else {
            codevar_wredit_memcpy(dst, src, len);
        }
    }
}

/// Copies `len` bytes from `src` to `dst`, correctly handling overlap.
///
/// # Safety
///
/// Both ranges must be valid for `len` bytes; overlap is allowed.
#[inline]
pub unsafe fn memmove(dst: *mut u8, src: *const u8, len: usize) {
    // Safety: guaranteed by the caller; the ISA path matches the CPU.
    unsafe {
        if sve_enabled() {
            codevar_wredit_memmove_sve(dst, src, len);
        } else {
            codevar_wredit_memmove(dst, src, len);
        }
    }
}

/// Returns a pointer to the first occurrence of `needle`, or null.
///
/// # Safety
///
/// `haystack` must be valid for reads of `len` bytes.
#[inline]
pub unsafe fn memchr(haystack: *const u8, needle: u8, len: usize) -> *const u8 {
    // Safety: guaranteed by the caller.
    unsafe { codevar_wredit_memchr(haystack, needle, len) }
}

/// Prefetch the cache line containing `addr` with the given hint.
///
/// # Safety
///
/// Prefetch never faults, so `addr` may point anywhere.
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
/// Prefetch never faults.
#[inline]
pub unsafe fn prefetch_range(addr: *const u8, len: usize) {
    // Safety: prefetch never faults.
    unsafe { codevar_wredit_prefetch_range(addr, len) };
}
