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

//! i686 (32-bit x86) assembly bindings.
//!
//! Unlike x86_64, the i686 baseline does not guarantee SSE2, so
//! `build.rs` only enables the SIMD path (the `CODEVAR_HAVE_SSE2`
//! preprocessor branch) when the target features include `sse2`; the
//! same sources then provide a pure-integer fallback.

unsafe extern "C" {
    fn codevar_wredit_memmove(dst: *mut u8, src: *const u8, len: usize);
    fn codevar_wredit_memcpy(dst: *mut u8, src: *const u8, len: usize);
    fn codevar_wredit_memchr(haystack: *const u8, needle: u8, len: usize) -> *const u8;
    fn codevar_wredit_prefetch_t0(addr: *const u8);
    fn codevar_wredit_prefetch_t1(addr: *const u8);
    fn codevar_wredit_prefetch_t2(addr: *const u8);
    fn codevar_wredit_prefetch_nta(addr: *const u8);
    fn codevar_wredit_prefetch_range(addr: *const u8, len: usize);
}

/// Copies `len` bytes from `src` to `dst`. The regions must not overlap.
///
/// # Safety
///
/// Both ranges must be valid for `len` bytes and must not overlap.
#[inline]
pub unsafe fn memcpy(dst: *mut u8, src: *const u8, len: usize) {
    // Safety: guaranteed by the caller.
    unsafe { codevar_wredit_memcpy(dst, src, len) }
}

/// Copies `len` bytes from `src` to `dst`, handling overlap.
///
/// # Safety
///
/// Both ranges must be valid for `len` bytes; overlap is allowed.
#[inline]
pub unsafe fn memmove(dst: *mut u8, src: *const u8, len: usize) {
    // Safety: guaranteed by the caller.
    unsafe { codevar_wredit_memmove(dst, src, len) }
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

/// Prefetch every cache line across `[addr, addr + len)`.
///
/// # Safety
///
/// Prefetch never faults.
#[inline]
pub unsafe fn prefetch_range(addr: *const u8, len: usize) {
    // Safety: prefetch never faults.
    unsafe { codevar_wredit_prefetch_range(addr, len) };
}
