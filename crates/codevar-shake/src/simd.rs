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

//! SIMD backends for the batched Keccak-f[1600] permutation.
//!
//! Each backend permutes `k` independent states laid out
//! position-major (`state[p * k + b]`), holding one vector per
//! permutation position so that ρ, π, θ, χ and ι become pure vector
//! operations with fixed rotation amounts — the same math as the
//! scalar reference in [`crate::keccak`], bit for bit. Backends are
//! selected at run time via `basic_cpuid`; when none applies, the
//! scalar lockstep loop is used, which shares the identical layout.

/// Dispatches the batched permutation to the best available backend
/// for `lanes` (`1..=8`; `state.len()` must be `25 * lanes`).
///
/// The result is bit-identical to [`crate::keccak::keccakf_batch`]
/// for every lane count.
pub fn keccakf_batch(state: &mut [u64], lanes: usize) {
    if dispatch(state, lanes) {
        return;
    }
    crate::keccak::keccakf_batch(state, lanes);
}

/// Runs a SIMD backend when one matches `lanes`; returns `false` when
/// the caller must fall back to the scalar loop.
fn dispatch(state: &mut [u64], lanes: usize) -> bool {
    cfg_if::cfg_if! {
        if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
            x86::run(state, lanes)
        } else if #[cfg(target_arch = "aarch64")] {
            aarch64::run(state, lanes)
        } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
            wasm::run(state, lanes)
        } else {
            let _ = (state, lanes);
            false
        }
    }
}

/// x86 / x86_64 backends: SSE2 for two lanes, SSSE3 and SSE4.1 tiers
/// for two lanes, AVX2 for four and eight.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    use crate::keccak::{RHO_OFFSETS, ROUND_CONSTANTS, STATE_LANES};

    /// Returns `true` when a SIMD backend handled the call.
    ///
    /// Dispatch lattice (widest batch first), mirroring XKCP's
    /// recommended x86-64 runtime selection: AVX2 for 4- and 8-lane
    /// batches, then for 2-lane batches the widest 128-bit tier the
    /// hardware provides — SSE4.1, then SSSE3, then SSE2.
    pub(super) fn run(state: &mut [u64], lanes: usize) -> bool {
        if state.len() != STATE_LANES * lanes {
            return false;
        }
        match lanes {
            8 if avx2_available() => {
                // SAFETY: `avx2_available` proves AVX2; each call
                // covers one 4-lane half of the 8-lane batch.
                unsafe {
                    keccakf_avx2(state, 8, 0);
                    keccakf_avx2(state, 8, 4);
                }
                true
            }
            4 if avx2_available() => {
                // SAFETY: `avx2_available` proves the AVX2 target
                // feature; `state.len()` is exactly 25 * 4 lanes.
                unsafe { keccakf_avx2(state, 4, 0) };
                true
            }
            2 => run_two_lane(state),
            _ => false,
        }
    }

    /// Selects the widest 128-bit tier the hardware supports for a
    /// two-lane batch: SSE4.1, then SSSE3, then SSE2.
    fn run_two_lane(state: &mut [u64]) -> bool {
        if sse41_available() {
            // SAFETY: `sse41_available` proves the SSE4.1 target
            // feature; `state.len()` is exactly 25 * 2 lanes.
            unsafe { keccakf_sse41(state) };
            true
        } else if ssse3_available() {
            // SAFETY: `ssse3_available` proves the SSSE3 target
            // feature; `state.len()` is exactly 25 * 2 lanes.
            unsafe { keccakf_ssse3(state) };
            true
        } else if sse2_available() {
            // SAFETY: `sse2_available` proves the SSE2 target feature
            // (always true on x86_64); `state.len()` is 25 * 2 lanes.
            unsafe { keccakf_sse2(state) };
            true
        } else {
            false
        }
    }

    #[inline]
    fn sse2_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            true
        }
        #[cfg(target_arch = "x86")]
        {
            codevar_base::cpuid::has(codevar_base::cpuid::Feature::Sse2)
        }
    }

    #[inline]
    fn ssse3_available() -> bool {
        codevar_base::cpuid::has(codevar_base::cpuid::Feature::Ssse3)
    }

    #[inline]
    fn sse41_available() -> bool {
        codevar_base::cpuid::has(codevar_base::cpuid::Feature::Sse41)
    }

    #[inline]
    fn avx2_available() -> bool {
        codevar_base::cpuid::has(codevar_base::cpuid::Feature::Avx2)
    }

    /// Two-lane permutation compiled under the SSSE3 tier.
    ///
    /// Keccak-f[1600]'s 64-bit-lane ρ/π/χ steps need no instruction
    /// beyond SSE2, so this tier runs the same lockstep kernel as
    /// [`keccakf_sse2`]; it exists so the dispatch lattice tracks the
    /// real hardware tier (as XKCP's x86-64 selector does) and so the
    /// compiler may re-optimize surrounding code with SSSE3 enabled.
    ///
    /// # Safety
    ///
    /// Requires the SSSE3 target feature (which implies SSE2 at run
    /// time) and `state.len() == 50`.
    #[target_feature(enable = "ssse3")]
    unsafe fn keccakf_ssse3(state: &mut [u64]) {
        // SAFETY: SSSE3 implies SSE2 on any CPU that reports it, so
        // the SSE2 kernel's precondition holds; the caller guarantees
        // `state.len() == 25 * 2`.
        unsafe { keccakf_sse2(state) }
    }

    /// Two-lane permutation compiled under the SSE4.1 tier.
    ///
    /// Like [`keccakf_ssse3`], this tier runs the shared SSE2-quality
    /// lockstep kernel: Keccak-f's 64-bit-lane rotation and chi steps
    /// use no SSE4.1 instruction, but selecting the tier keeps the
    /// hardware lattice accurate and lets the compiler re-optimize
    /// surrounding code with SSE4.1 enabled.
    ///
    /// # Safety
    ///
    /// Requires the SSE4.1 target feature (which implies SSE2 at run
    /// time) and `state.len() == 50`.
    #[target_feature(enable = "sse4.1")]
    unsafe fn keccakf_sse41(state: &mut [u64]) {
        // SAFETY: SSE4.1 implies SSE2 on any CPU that reports it, so
        // the SSE2 kernel's precondition holds; the caller guarantees
        // `state.len() == 25 * 2`.
        unsafe { keccakf_sse2(state) }
    }

    /// One two-lane batched permutation over `state` (SSE2).
    ///
    /// # Safety
    ///
    /// Requires the SSE2 target feature and `state.len() == 50`.
    #[target_feature(enable = "sse2")]
    unsafe fn keccakf_sse2(state: &mut [u64]) {
        // SAFETY: the `#[target_feature]` attribute proves SSE2; the
        // pointer arithmetic below stays inside `state` because the
        // caller guarantees `state.len() == STATE_LANES * 2`.
        unsafe {
            let mut a: [__m128i; STATE_LANES] =
                core::array::from_fn(|p| _mm_loadu_si128(state.as_ptr().add(p * 2) as *const __m128i));
            let mut b: [__m128i; STATE_LANES] = [_mm_setzero_si128(); STATE_LANES];
            for &rc in &ROUND_CONSTANTS {
                keccakf_round_sse2(&mut a, &mut b, rc);
            }
            for (p, &value) in a.iter().enumerate() {
                _mm_storeu_si128(state.as_mut_ptr().add(p * 2) as *mut __m128i, value);
            }
        }
    }

    /// One two-lane Keccak-f round (SSE2).
    ///
    /// # Safety
    ///
    /// Requires the SSE2 target feature.
    #[target_feature(enable = "sse2")]
    unsafe fn keccakf_round_sse2(a: &mut [__m128i; STATE_LANES], b: &mut [__m128i; STATE_LANES], rc: u64) {
        // SAFETY: the `#[target_feature]` attribute proves SSE2; both
        // arrays have exactly STATE_LANES elements and every index is
        // in `0..25` by construction of the loops below.
        unsafe {
            // θ.
            let mut c = [_mm_setzero_si128(); 5];
            for (x, slot) in c.iter_mut().enumerate() {
                *slot = _mm_xor_si128(
                    _mm_xor_si128(a[x], a[x + 5]),
                    _mm_xor_si128(_mm_xor_si128(a[x + 10], a[x + 15]), a[x + 20]),
                );
            }
            for x in 0..5 {
                let d = _mm_xor_si128(c[(x + 4) % 5], rotl128(c[(x + 1) % 5], 1));
                for y in 0..5 {
                    a[x + 5 * y] = _mm_xor_si128(a[x + 5 * y], d);
                }
            }
            // ρ and π.
            for y in 0..5 {
                for x in 0..5 {
                    let offset = RHO_OFFSETS[x + 5 * y];
                    b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl128(a[x + 5 * y], offset);
                }
            }
            // χ: A = B ^ ((~B[x+1]) & B[x+2]).
            for y in 0..5 {
                for x in 0..5 {
                    a[x + 5 * y] = _mm_xor_si128(
                        b[x + 5 * y],
                        _mm_andnot_si128(b[(x + 1) % 5 + 5 * y], b[(x + 2) % 5 + 5 * y]),
                    );
                }
            }
            // ι.
            a[0] = _mm_xor_si128(a[0], _mm_set1_epi64x(rc as i64));
        }
    }

    /// One four-lane batched permutation over one 4-lane half of
    /// `state` (AVX2). `lanes` is the full batch stride and `offset`
    /// the half's start inside each position, so an 8-lane batch is
    /// two calls with `offset` 0 and 4.
    ///
    /// # Safety
    ///
    /// Requires the AVX2 target feature;
    /// `state.len() == STATE_LANES * lanes` with `lanes` in `{4, 8}`
    /// and `offset` in `{0, 4}` with `offset + 4 <= lanes`.
    #[target_feature(enable = "avx2")]
    unsafe fn keccakf_avx2(state: &mut [u64], lanes: usize, offset: usize) {
        debug_assert!(matches!(lanes, 4 | 8));
        debug_assert!(offset + 4 <= lanes);
        // SAFETY: the `#[target_feature]` attribute proves AVX2; the
        // pointer arithmetic stays inside `state` because the caller
        // guarantees the documented length and stride.
        unsafe {
            let mut a: [__m256i; STATE_LANES] = core::array::from_fn(|p| {
                _mm256_loadu_si256(state.as_ptr().add(p * lanes + offset) as *const __m256i)
            });
            let mut b: [__m256i; STATE_LANES] = [_mm256_setzero_si256(); STATE_LANES];
            for &rc in &ROUND_CONSTANTS {
                keccakf_round_avx2(&mut a, &mut b, rc);
            }
            for (p, &value) in a.iter().enumerate() {
                _mm256_storeu_si256(state.as_mut_ptr().add(p * lanes + offset) as *mut __m256i, value);
            }
        }
    }

    /// One four-lane Keccak-f round (AVX2).
    ///
    /// # Safety
    ///
    /// Requires the AVX2 target feature.
    #[target_feature(enable = "avx2")]
    unsafe fn keccakf_round_avx2(a: &mut [__m256i; STATE_LANES], b: &mut [__m256i; STATE_LANES], rc: u64) {
        // SAFETY: the `#[target_feature]` attribute proves AVX2; both
        // arrays have exactly STATE_LANES elements and every index is
        // in `0..25` by construction of the loops below.
        unsafe {
            // θ.
            let mut c = [_mm256_setzero_si256(); 5];
            for (x, slot) in c.iter_mut().enumerate() {
                *slot = _mm256_xor_si256(
                    _mm256_xor_si256(a[x], a[x + 5]),
                    _mm256_xor_si256(_mm256_xor_si256(a[x + 10], a[x + 15]), a[x + 20]),
                );
            }
            for x in 0..5 {
                let d = _mm256_xor_si256(c[(x + 4) % 5], rotl256(c[(x + 1) % 5], 1));
                for y in 0..5 {
                    a[x + 5 * y] = _mm256_xor_si256(a[x + 5 * y], d);
                }
            }
            // ρ and π.
            for y in 0..5 {
                for x in 0..5 {
                    let rho = RHO_OFFSETS[x + 5 * y];
                    b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl256(a[x + 5 * y], rho);
                }
            }
            // χ.
            for y in 0..5 {
                for x in 0..5 {
                    a[x + 5 * y] = _mm256_xor_si256(
                        b[x + 5 * y],
                        _mm256_andnot_si256(b[(x + 1) % 5 + 5 * y], b[(x + 2) % 5 + 5 * y]),
                    );
                }
            }
            // ι.
            a[0] = _mm256_xor_si256(a[0], _mm256_set1_epi64x(rc as i64));
        }
    }

    /// Rotate every 64-bit lane of `value` left by `k`.
    ///
    /// Uses the variable-count shift forms (`_mm_sll_epi64` /
    /// `_mm_srl_epi64`) because the immediate-count forms require a
    /// compile-time constant, while the ρ offset is only known at run
    /// time.
    ///
    /// # Safety
    ///
    /// Requires the SSE2 target feature; `k` must be in `0..=63`.
    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn rotl128(value: __m128i, k: u32) -> __m128i {
        // SAFETY: the `#[target_feature]` attribute proves SSE2; the
        // `k == 0` guard leaves both counts in `1..=63`, and the
        // variable-count shifts read the count from a register.
        if k == 0 {
            return value;
        }
        let left = _mm_sll_epi64(value, _mm_cvtsi32_si128(k as i32));
        let right = _mm_srl_epi64(value, _mm_cvtsi32_si128(64 - k as i32));
        _mm_or_si128(left, right)
    }

    /// Rotate every 64-bit lane of `value` left by `k`.
    ///
    /// # Safety
    ///
    /// Requires the AVX2 target feature; `k` must be in `0..=63`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn rotl256(value: __m256i, k: u32) -> __m256i {
        // SAFETY: the `#[target_feature]` attribute proves AVX2; the
        // `k == 0` guard leaves both counts in `1..=63`, and the
        // variable-count shifts read the count from a register.
        if k == 0 {
            return value;
        }
        let left = _mm256_sll_epi64(value, _mm_cvtsi32_si128(k as i32));
        let right = _mm256_srl_epi64(value, _mm_cvtsi32_si128(64 - k as i32));
        _mm256_or_si256(left, right)
    }
}

/// aarch64 NEON backend for two-lane batches.
#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use core::arch::aarch64::{
        int64x2_t, uint64x2_t, vbicq_u64, vdupq_n_s64, vdupq_n_u64, veorq_u64, vld1q_u64, vorrq_u64,
        vshlq_u64, vst1q_u64,
    };

    use crate::keccak::{RHO_OFFSETS, ROUND_CONSTANTS, STATE_LANES};

    /// Returns `true` when the NEON backend handled the call.
    pub(super) fn run(state: &mut [u64], lanes: usize) -> bool {
        if lanes != 2 || state.len() != STATE_LANES * 2 {
            return false;
        }
        if !codevar_base::cpuid::has(codevar_base::cpuid::Feature::Neon) {
            return false;
        }
        // SAFETY: the probe above confirmed Advanced SIMD;
        // `state.len()` is exactly 25 * 2 lanes.
        unsafe { keccakf_neon(state) };
        true
    }

    /// One two-lane batched permutation (NEON).
    ///
    /// # Safety
    ///
    /// Requires the NEON target feature and `state.len() == 50`.
    #[target_feature(enable = "neon")]
    unsafe fn keccakf_neon(state: &mut [u64]) {
        // SAFETY: the `#[target_feature]` attribute proves NEON; the
        // pointer arithmetic stays inside `state` because the caller
        // guarantees `state.len() == STATE_LANES * 2`.
        unsafe {
            let mut a: [uint64x2_t; STATE_LANES] =
                core::array::from_fn(|p| vld1q_u64(state.as_ptr().add(p * 2)));
            let mut b: [uint64x2_t; STATE_LANES] = core::array::from_fn(|_| vdupq_n_u64(0));
            for &rc in &ROUND_CONSTANTS {
                // θ.
                let mut c: [uint64x2_t; 5] = core::array::from_fn(|_| vdupq_n_u64(0));
                for (x, slot) in c.iter_mut().enumerate() {
                    *slot = veorq_u64(
                        veorq_u64(a[x], a[x + 5]),
                        veorq_u64(veorq_u64(a[x + 10], a[x + 15]), a[x + 20]),
                    );
                }
                for x in 0..5 {
                    let d = veorq_u64(c[(x + 4) % 5], rotl64x2(c[(x + 1) % 5], 1));
                    for y in 0..5 {
                        a[x + 5 * y] = veorq_u64(a[x + 5 * y], d);
                    }
                }
                // ρ and π.
                for y in 0..5 {
                    for x in 0..5 {
                        let offset = RHO_OFFSETS[x + 5 * y];
                        b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64x2(a[x + 5 * y], offset);
                    }
                }
                // χ: A = B ^ (B[x+2] & ~B[x+1]).
                for y in 0..5 {
                    for x in 0..5 {
                        a[x + 5 * y] = veorq_u64(
                            b[x + 5 * y],
                            vbicq_u64(b[(x + 2) % 5 + 5 * y], b[(x + 1) % 5 + 5 * y]),
                        );
                    }
                }
                // ι.
                a[0] = veorq_u64(a[0], vdupq_n_u64(rc));
            }
            for (p, &value) in a.iter().enumerate() {
                vst1q_u64(state.as_mut_ptr().add(p * 2), value);
            }
        }
    }

    /// Rotate both 64-bit lanes left by `k`.
    ///
    /// Uses the variable-shift intrinsic: a negative shift amount is a
    /// right shift, so `rotl(v, k) = v << k | v >> (64 - k)` becomes
    /// `vshl(v, k) | vshl(v, k - 64)`.
    ///
    /// # Safety
    ///
    /// Requires the NEON target feature; `k` must be in `0..=63`.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn rotl64x2(value: uint64x2_t, k: u32) -> uint64x2_t {
        // SAFETY: the `#[target_feature]` attribute proves NEON; the
        // `k == 0` guard leaves `k - 64` in `-64..=-1`, in range.
        unsafe {
            if k == 0 {
                return value;
            }
            let left: int64x2_t = vdupq_n_s64(k as i64);
            let right: int64x2_t = vdupq_n_s64(k as i64 - 64);
            vorrq_u64(vshlq_u64(value, left), vshlq_u64(value, right))
        }
    }
}

/// Wasm SIMD128 backend for two-lane batches.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use crate::keccak::{RHO_OFFSETS, ROUND_CONSTANTS, STATE_LANES};
    use core::arch::wasm32::{
        u64x2_shl, u64x2_shr, u64x2_splat, v128, v128_andnot, v128_load, v128_or, v128_store, v128_xor,
        v128_zero,
    };

    /// Returns `true` when the Wasm backend handled the call.
    pub(super) fn run(state: &mut [u64], lanes: usize) -> bool {
        if lanes != 2 || state.len() != STATE_LANES * 2 {
            return false;
        }
        // SAFETY: the module is compiled only with `simd128` enabled;
        // `state.len()` is exactly 25 * 2 lanes.
        unsafe { keccakf_wasm(state) };
        true
    }

    /// One two-lane batched permutation (Wasm SIMD128).
    ///
    /// # Safety
    ///
    /// Requires the `simd128` target feature and `state.len() == 50`.
    #[target_feature(enable = "simd128")]
    unsafe fn keccakf_wasm(state: &mut [u64]) {
        // SAFETY: the `#[target_feature]` attribute proves `simd128`;
        // the pointer arithmetic stays inside `state` because the
        // caller guarantees `state.len() == STATE_LANES * 2`.
        unsafe {
            let mut a: [v128; STATE_LANES] =
                core::array::from_fn(|p| v128_load(state.as_ptr().add(p * 2) as *const v128));
            let mut b: [v128; STATE_LANES] = [v128_zero(); STATE_LANES];
            for &rc in &ROUND_CONSTANTS {
                // θ.
                let mut c = [v128_zero(); 5];
                for (x, slot) in c.iter_mut().enumerate() {
                    *slot = v128_xor(
                        v128_xor(a[x], a[x + 5]),
                        v128_xor(v128_xor(a[x + 10], a[x + 15]), a[x + 20]),
                    );
                }
                for x in 0..5 {
                    let d = v128_xor(c[(x + 4) % 5], rotl64x2(c[(x + 1) % 5], 1));
                    for y in 0..5 {
                        a[x + 5 * y] = v128_xor(a[x + 5 * y], d);
                    }
                }
                // ρ and π.
                for y in 0..5 {
                    for x in 0..5 {
                        let offset = RHO_OFFSETS[x + 5 * y];
                        b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64x2(a[x + 5 * y], offset);
                    }
                }
                // χ: A = B ^ (B[x+2] & ~B[x+1]).
                for y in 0..5 {
                    for x in 0..5 {
                        a[x + 5 * y] = v128_xor(
                            b[x + 5 * y],
                            v128_andnot(b[(x + 2) % 5 + 5 * y], b[(x + 1) % 5 + 5 * y]),
                        );
                    }
                }
                // ι.
                a[0] = v128_xor(a[0], u64x2_splat(rc));
            }
            for (p, &value) in a.iter().enumerate() {
                v128_store(state.as_mut_ptr().add(p * 2) as *mut v128, value);
            }
        }
    }

    /// Rotate both 64-bit lanes left by `k`.
    ///
    /// # Safety
    ///
    /// Requires the `simd128` target feature; `k` must be in `0..=63`.
    #[target_feature(enable = "simd128")]
    #[inline]
    unsafe fn rotl64x2(value: v128, k: u32) -> v128 {
        // SAFETY: the `#[target_feature]` attribute proves `simd128`;
        // the `k == 0` guard leaves `64 - k` in `1..=64`, and Wasm
        // shift intrinsics mask the count to 6 bits.
        unsafe {
            if k == 0 {
                return value;
            }
            v128_or(u64x2_shl(value, k), u64x2_shr(value, 64 - k))
        }
    }
}
