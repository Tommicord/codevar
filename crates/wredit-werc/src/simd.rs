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

//! Runtime-dispatched kernels for stage 1 and the mixer of stage 2.
//!
//! The order of a plane's bytes *is* the format: the model reads the planes
//! as a document, so a kernel that leaves them in a permuted element order is
//! not "faster but different", it is wrong. That rules out the
//! cheapest-looking deinterleave. The three-level `punpck` tree is 12 uops
//! for all four planes of 16 elements, but it pairs neighbouring registers
//! before it separates lanes, so each plane comes out in the order
//! `[0, 4, 8, 12, 2, 6, 10, 14, 1, 5, ...]`; putting that back needs a
//! lane-crossing byte permute per plane. And extracting four plane bytes on
//! SSE2 takes two `packus` stages, which costs about what the scalar byte
//! stores already do. So machines without AVX2 keep the scalar loop, and the
//! tiers below are the ones that pay.
//!
//! | tier | split | join |
//! |------|-------|------|
//! | AVX-512F + AVX512BW | shift, then `vpmovdb` truncates 16 dwords to the chunk's 16 plane bytes | `vpmovzxbd` widens 16 plane bytes to dwords, shift, OR |
//! | AVX2 | one in-lane `vpshufb` picks bytes `k, 4+k, 8+k, 12+k` of each 4-element lane, `vpermd` puts the two groups side by side | `vpmovzxbd` on 8 bytes, shift, OR |
//! | everything else | scalar | scalar |
//!
//! Both directions work one plane at a time, which keeps each kernel a single
//! loop with no per-element branch on which planes are present;
//! [`crate::plane`] calls them at most four times per stream. The shift is a
//! compile-time constant per plane because `vpsrld`/`vpslld` take an
//! immediate, so each kernel is monomorphised once per plane rather than
//! building a count vector in the loop.

/// Ask the memory system to bring one cache line in before it is read.
///
/// Stage 2's per-bit cost is dominated by random reads into tables larger
/// than L2, and the eight context slots, mixer rows and match pointer are
/// independent addresses: issuing all the hints before the first dependent
/// load lets the misses overlap instead of serializing them. A hint has no
/// architectural effect - a useless one costs cycles and nothing else - so
/// every tier still produces the identical stream.
///
/// # Safety
///
/// - `base.add(offset)` must stay inside the allocation `base` points into.
///   Forming the pointer is the contract; the prefetch itself never
///   dereferences it.
#[inline]
pub(crate) unsafe fn touch(base: *const u8, offset: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_MM_HINT_T0, _mm_prefetch};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        // SAFETY: the caller bounds `offset` inside the allocation, and
        // `_mm_prefetch` only issues the hint.
        unsafe { _mm_prefetch(base.add(offset).cast::<i8>(), _MM_HINT_T0) };
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = (base, offset);
}

/// The three first-layer mixer dot products for one bit.
///
/// Every mixer input is a stretched 12-bit probability, bounded by the
/// stretch table to +-1065 with a bias of 256, so a `vpmuldq` product is
/// exact whenever the weight itself fits an `i32`. The kernel checks that
/// on the rows it already loaded and reports `None` when it does not hold,
/// which leaves the caller's scalar loop in charge. `wrapping_add` is
/// associative mod 2^64, so the vector summation tree computes the same
/// dot the scalar chain does, lane order notwithstanding.
#[inline]
pub(crate) fn mixer_dots(
    rows: [&[i64]; crate::model::NMX],
    st: &[i32; crate::model::NW],
) -> Option<[i64; crate::model::NMX]> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};
        if has(Feature::Avx2) {
            // SAFETY: the probe above confirmed AVX2 and the OS-enabled YMM
            // state; each kernel load stays inside the 10 weights or 10
            // inputs of the slices handed in.
            return unsafe { x86::mixer_dots_avx2(rows, st) };
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = (rows, st);
    None
}

/// Gather byte `shift / 8` of every element into `plane`.
///
/// `plane` and `elements` are the same length. Every tier writes the same
/// bytes at the same offsets as the scalar loop in [`crate::plane`], which is
/// what makes a stream encoded on one machine decode on another.
#[inline]
pub(crate) fn write_plane(elements: &[u32], plane: &mut [u8], shift: u32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};
        if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx2) {
            // SAFETY: the probes above confirmed AVX-512F/BW - and with them
            // the OS-enabled ZMM state - plus AVX2. The kernel reads only
            // `elements` and writes only `plane`, each inside its own length.
            unsafe { x86::write_plane_avx512(elements, plane, shift) };
            return;
        }
        if has(Feature::Avx2) {
            // SAFETY: the probe above confirmed AVX2 and the YMM state; the
            // kernel's loads and stores stay inside `elements` and `plane`.
            unsafe { x86::write_plane_avx2(elements, plane, shift) };
            return;
        }
    }
    crate::plane::write_shifted(elements, plane, shift);
}

/// Put every byte of `plane` back into the bits of `out` it came from.
///
/// `plane` and `out` are the same length. With `assign` the kernel overwrites
/// `out` instead of reading it first, which is what plane 0 needs; the other
/// planes only OR, because their bits are disjoint.
#[inline]
pub(crate) fn read_plane(plane: &[u8], out: &mut [u32], shift: u32, assign: bool) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};
        if has(Feature::Avx512F) && has(Feature::Avx512Bw) && has(Feature::Avx2) {
            // SAFETY: as for `write_plane_avx512`; the 16-byte read is
            // `_mm_loadu_si128`, which cannot run past `plane`.
            unsafe { x86::read_plane_avx512(plane, out, shift, assign) };
            return;
        }
        if has(Feature::Avx2) {
            // SAFETY: the probe above confirmed AVX2; `_mm_loadl_epi64`
            // reads exactly 8 bytes, so the last chunk stays inside `plane`.
            unsafe { x86::read_plane_avx2(plane, out, shift, assign) };
            return;
        }
    }
    if assign {
        crate::plane::assign_shifted(plane, out, shift);
    } else {
        crate::plane::read_shifted(plane, out, shift);
    }
}

/// AVX-512 and AVX2 kernels for the two directions of stage 1.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use crate::plane;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Elements settled per chunk by the 512-bit kernels: one `__m512i` of
    /// dwords, one 16-byte plane store.
    const CHUNK_512: usize = 16;

    /// Elements settled per chunk by the 256-bit kernels.
    const CHUNK_256: usize = 8;

    /// `vpshufb` table that gathers the plane bytes of the four elements in a
    /// 16-byte lane into positions 0..3.
    ///
    /// No shift is needed: byte `plane` of element `j` already sits at lane
    /// offset `4 * j + plane`, and `vpshufb` indices are lane-local, so the
    /// same 16 bytes serve both halves of a `__m256i`.
    pub(super) const fn gather_table(plane: usize) -> [i8; 32] {
        let mut table = [-1i8; 32];
        let mut j = 0;
        while j < 4 {
            let byte = (4 * j + plane) as i8;
            table[j] = byte;
            table[16 + j] = byte;
            j += 1;
        }
        table
    }

    /// # Safety
    ///
    /// - `plane.len() == elements.len()`
    /// - the executing CPU supports AVX-512F and AVX-512BW, and the OS has
    ///   enabled the ZMM state.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn write_plane_avx512(elements: &[u32], plane: &mut [u8], shift: u32) {
        match shift {
            // SAFETY: the invariants are the caller's, and the buffer
            // lengths are unchanged by the split.
            0 => unsafe { write_chunks_avx512::<0>(elements, plane) },
            8 => unsafe { write_chunks_avx512::<8>(elements, plane) },
            16 => unsafe { write_chunks_avx512::<16>(elements, plane) },
            24 => unsafe { write_chunks_avx512::<24>(elements, plane) },
            other => plane::write_shifted(elements, plane, other),
        }
    }

    /// # Safety
    ///
    /// - `plane.len() == out.len()`
    /// - the executing CPU supports AVX-512F and AVX-512BW, and the OS has
    ///   enabled the ZMM state.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn read_plane_avx512(plane: &[u8], out: &mut [u32], shift: u32, assign: bool) {
        match shift {
            // SAFETY: as for `write_plane_avx512`.
            0 => unsafe { read_chunks_avx512::<0>(plane, out, assign) },
            8 => unsafe { read_chunks_avx512::<8>(plane, out, assign) },
            16 => unsafe { read_chunks_avx512::<16>(plane, out, assign) },
            24 => unsafe { read_chunks_avx512::<24>(plane, out, assign) },
            other => {
                if assign {
                    plane::assign_shifted(plane, out, other);
                } else {
                    plane::read_shifted(plane, out, other);
                }
            }
        }
    }

    /// # Safety
    ///
    /// - `plane.len() == elements.len()`
    /// - the executing CPU supports AVX2, and the OS has enabled the YMM
    ///   state.
    #[target_feature(enable = "avx2")]
    pub unsafe fn write_plane_avx2(elements: &[u32], plane: &mut [u8], shift: u32) {
        debug_assert_eq!(plane.len(), elements.len());
        let table = gather_table(shift as usize / 8);
        // SAFETY: reinterpreting a `[i8; 32]` as a vector is defined for
        // every element type, and `table` outlives the loop it feeds.
        let shuffle = unsafe { _mm256_loadu_si256(table.as_ptr().cast::<__m256i>()) };
        // `vpshufb` is lane-local, so the four plane bytes of each lane land
        // in that lane's dword 0 - offsets 0 and 4 of the dword stream.
        let compact = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
        let chunks = elements.len() / CHUNK_256;
        for c in 0..chunks {
            let i = c * CHUNK_256;
            // SAFETY: `i + 8 <= elements.len()` by construction of `chunks`.
            let words = unsafe { _mm256_loadu_si256(elements.as_ptr().add(i).cast::<__m256i>()) };
            let grouped = _mm256_shuffle_epi8(words, shuffle);
            // `vpermd` moves whole dwords, which is what puts the two
            // 4-byte groups adjacent; `vpermq` at qword granularity would
            // leave each group alone in its own 8-byte slot.
            let packed = _mm256_permutevar8x32_epi32(grouped, compact);
            // SAFETY: an 8-byte store at `i`, and `i + 8 <= plane.len()`.
            unsafe {
                _mm_storel_epi64(
                    plane.as_mut_ptr().add(i).cast::<__m128i>(),
                    _mm256_castsi256_si128(packed),
                )
            };
        }
        let done = chunks * CHUNK_256;
        plane::write_shifted(&elements[done..], &mut plane[done..], shift);
    }

    /// # Safety
    ///
    /// - `plane.len() == out.len()`
    /// - the executing CPU supports AVX2, and the OS has enabled the YMM
    ///   state.
    #[target_feature(enable = "avx2")]
    pub unsafe fn read_plane_avx2(plane: &[u8], out: &mut [u32], shift: u32, assign: bool) {
        match shift {
            // SAFETY: the invariants are the caller's; `assign` only decides
            // whether the accumulator is read, never how far the loop runs.
            0 => unsafe { read_chunks_avx2::<0>(plane, out, assign) },
            8 => unsafe { read_chunks_avx2::<8>(plane, out, assign) },
            16 => unsafe { read_chunks_avx2::<16>(plane, out, assign) },
            24 => unsafe { read_chunks_avx2::<24>(plane, out, assign) },
            other => {
                if assign {
                    plane::assign_shifted(plane, out, other);
                } else {
                    plane::read_shifted(plane, out, other);
                }
            }
        }
    }

    /// One plane of every 16-element chunk, via `vpmovdb`.
    ///
    /// # Safety
    ///
    /// - `plane.len() == elements.len()`
    /// - the executing CPU supports AVX-512F and AVX-512BW.
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn write_chunks_avx512<const SHIFT: u32>(elements: &[u32], plane: &mut [u8]) {
        let chunks = elements.len() / CHUNK_512;
        for c in 0..chunks {
            let i = c * CHUNK_512;
            // SAFETY: `i + 16 <= elements.len()`, so the 64-byte read is in
            // bounds.
            let words = unsafe { _mm512_loadu_si512(elements.as_ptr().add(i).cast::<__m512i>()) };
            let bytes = _mm512_cvtepi32_epi8(_mm512_srli_epi32::<SHIFT>(words));
            // SAFETY: `i + 16 == i + CHUNK_512 <= plane.len()`.
            unsafe { _mm_storeu_si128(plane.as_mut_ptr().add(i).cast::<__m128i>(), bytes) };
        }
        let done = chunks * CHUNK_512;
        plane::write_shifted(&elements[done..], &mut plane[done..], SHIFT);
    }

    /// Widen one plane of every chunk and fold it back into `out`.
    ///
    /// # Safety
    ///
    /// - `plane.len() == out.len()`
    /// - the executing CPU supports AVX-512F and AVX-512BW.
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn read_chunks_avx512<const SHIFT: u32>(plane: &[u8], out: &mut [u32], assign: bool) {
        let chunks = plane.len() / CHUNK_512;
        let zero = _mm512_setzero_si512();
        for c in 0..chunks {
            let i = c * CHUNK_512;
            // SAFETY: `i + 16 <= plane.len()`.
            let bytes = unsafe { _mm_loadu_si128(plane.as_ptr().add(i).cast::<__m128i>()) };
            let widened = _mm512_slli_epi32::<SHIFT>(_mm512_cvtepi8_epi32(bytes));
            // SAFETY: `i + 16 <= out.len()`; the read is skipped when the
            // caller asked for an overwrite.
            let acc = unsafe {
                if assign {
                    zero
                } else {
                    _mm512_loadu_si512(out.as_ptr().add(i).cast::<__m512i>())
                }
            };
            // SAFETY: the store covers the same 16 words as the read.
            unsafe {
                _mm512_storeu_si512(
                    out.as_mut_ptr().add(i).cast::<__m512i>(),
                    _mm512_or_si512(acc, widened),
                )
            };
        }
        let done = chunks * CHUNK_512;
        if assign {
            plane::assign_shifted(&plane[done..], &mut out[done..], SHIFT);
        } else {
            plane::read_shifted(&plane[done..], &mut out[done..], SHIFT);
        }
    }

    /// Widen one plane of every 8-element chunk and fold it back into `out`.
    ///
    /// # Safety
    ///
    /// - `plane.len() == out.len()`
    /// - the executing CPU supports AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn read_chunks_avx2<const SHIFT: i32>(plane: &[u8], out: &mut [u32], assign: bool) {
        debug_assert_eq!(plane.len(), out.len());
        let chunks = plane.len() / CHUNK_256;
        let zero = _mm256_setzero_si256();
        for c in 0..chunks {
            let i = c * CHUNK_256;
            // SAFETY: reads exactly 8 bytes at `i`, and `i + 8 <=
            // plane.len()`.
            let bytes = unsafe { _mm_loadl_epi64(plane.as_ptr().add(i).cast::<__m128i>()) };
            let widened = _mm256_slli_epi32::<SHIFT>(_mm256_cvtepu8_epi32(bytes));
            // SAFETY: `i + 8 <= out.len()`.
            let acc = unsafe {
                if assign {
                    zero
                } else {
                    _mm256_loadu_si256(out.as_ptr().add(i).cast::<__m256i>())
                }
            };
            // SAFETY: the store covers the same 8 words as the read.
            unsafe {
                _mm256_storeu_si256(
                    out.as_mut_ptr().add(i).cast::<__m256i>(),
                    _mm256_or_si256(acc, widened),
                )
            };
        }
        let done = chunks * CHUNK_256;
        if assign {
            plane::assign_shifted(&plane[done..], &mut out[done..], SHIFT as u32);
        } else {
            plane::read_shifted(&plane[done..], &mut out[done..], SHIFT as u32);
        }
    }

    /// Number of mixer weights: one per input plus the bias.
    const NW: usize = crate::model::NW;

    /// Number of first-layer mixers.
    const NMX: usize = crate::model::NMX;

    /// One past the largest weight a `vpmuldq` product reproduces exactly.
    const FIT_HI: i64 = i32::MAX as i64 + 1;

    /// One below the smallest weight a `vpmuldq` product reproduces exactly.
    const FIT_LO: i64 = i32::MIN as i64 - 1;

    /// The three first-layer dots, with the range check folded into the
    /// loads the products need anyway.
    ///
    /// # Safety
    ///
    /// - the executing CPU supports AVX2 and the OS has enabled the YMM
    ///   state;
    /// - each row holds exactly `NW` weights and `st` exactly `NW` inputs.
    #[target_feature(enable = "avx2,sse4.1")]
    pub unsafe fn mixer_dots_avx2(rows: [&[i64]; NMX], st: &[i32; NW]) -> Option<[i64; NMX]> {
        // SAFETY: the caller guarantees the row lengths and the AVX2 state;
        // every load below covers at most the 80 bytes of one row and the
        // 40 bytes of `st`.
        unsafe {
            let s0 = _mm256_cvtepi32_epi64(_mm_loadu_si128(st.as_ptr().cast()));
            let s1 = _mm256_cvtepi32_epi64(_mm_loadu_si128(st.as_ptr().add(4).cast()));
            let s2 = _mm_cvtepi32_epi64(_mm_loadl_epi64(st.as_ptr().add(8).cast()));
            let hi = _mm256_set1_epi64x(FIT_HI);
            let lo = _mm256_set1_epi64x(FIT_LO);
            // All-ones while every lane still fits an `i32`: 64-bit
            // min/max are AVX-512-only instructions, so the fit check is a
            // compare-and-mask fold over the row loads the products use
            // anyway. Zero padding in the extended pair lanes is
            // trivially in range.
            let mut fits = _mm256_set1_epi64x(-1);
            let mut dots = [0i64; NMX];
            for (j, row) in rows.iter().enumerate() {
                let base = row.as_ptr().cast::<__m256i>();
                let w0 = _mm256_loadu_si256(base);
                let w1 = _mm256_loadu_si256(base.add(1));
                let wx = _mm_loadu_si128(row.as_ptr().add(8).cast::<__m128i>());
                let w2 = _mm256_inserti128_si256(_mm256_setzero_si256(), wx, 1);
                for w in [w0, w1, w2] {
                    let inr = _mm256_and_si256(_mm256_cmpgt_epi64(hi, w), _mm256_cmpgt_epi64(w, lo));
                    fits = _mm256_and_si256(fits, inr);
                }
                let p = _mm256_add_epi64(_mm256_mul_epi32(w0, s0), _mm256_mul_epi32(w1, s1));
                let px = _mm256_inserti128_si256(_mm256_setzero_si256(), _mm_mul_epi32(wx, s2), 0);
                let t = _mm256_add_epi64(p, px);
                let fold = _mm_add_epi64(_mm256_castsi256_si128(t), _mm256_extracti128_si256::<1>(t));
                let pair = _mm_add_epi64(fold, _mm_unpackhi_epi64(fold, fold));
                dots[j] = _mm_extract_epi64::<0>(pair);
            }
            if _mm256_movemask_pd(_mm256_castsi256_pd(fits)) == 0b1111 {
                Some(dots)
            } else {
                None
            }
        }
    }
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod tests {
    //! Each tier is checked against the scalar reference only when this host
    //! reports the feature, so a green run on an AVX2-only machine says
    //! nothing about the AVX-512 kernels - a tier that skips is not a tier
    //! that passed.
    use super::mixer_dots;
    use super::x86::{
        gather_table, read_plane_avx2, read_plane_avx512, write_plane_avx2, write_plane_avx512,
    };
    use crate::model::{NMX, NW};
    use crate::plane::{assign_shifted, read_shifted, write_shifted};
    use codevar_base::cpuid::{Feature, has};

    /// A xorshift32 sweep: every plane byte is exercised, and 1000 is not a
    /// multiple of either chunk size, so the scalar tails run too.
    fn elements(n: usize) -> Vec<u32> {
        let mut out = vec![0u32; n];
        let mut x = 0x2545_F491u32;
        for slot in &mut out {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *slot = x;
        }
        out
    }

    fn assert_write_matches_scalar(
        n: usize,
        kernel: unsafe fn(&[u32], &mut [u8], u32),
        label: &str,
        probe: impl Fn() -> bool,
    ) {
        if !probe() {
            return;
        }
        let data = elements(n);
        for shift in [0, 8, 16, 24] {
            let mut fast = vec![0u8; n];
            // SAFETY: the probe above ran the same CPUID check the dispatch
            // uses, and both buffers are `n` long.
            unsafe { kernel(&data, &mut fast, shift) };
            let mut reference = vec![0u8; n];
            write_shifted(&data, &mut reference, shift);
            assert_eq!(fast, reference, "{label} write differs at shift {shift}");
        }
    }

    fn assert_read_matches_scalar(
        n: usize,
        kernel: unsafe fn(&[u8], &mut [u32], u32, bool),
        label: &str,
        probe: impl Fn() -> bool,
    ) {
        if !probe() {
            return;
        }
        let plane_bytes: Vec<u8> = elements(n).iter().map(|&e| e as u8).collect();
        let seed = elements(n);
        for shift in [0, 8, 16, 24] {
            for assign in [true, false] {
                let mut fast = seed.clone();
                // SAFETY: probed as above; `fast` holds `n` words, the length
                // the kernel reads from `plane_bytes`.
                unsafe { kernel(&plane_bytes, &mut fast, shift, assign) };
                let mut reference = seed.clone();
                if assign {
                    assign_shifted(&plane_bytes, &mut reference, shift);
                } else {
                    read_shifted(&plane_bytes, &mut reference, shift);
                }
                assert_eq!(fast, reference, "{label} read differs at {shift}/{assign}");
            }
        }
    }

    #[test]
    fn gather_table_selects_one_lane_per_plane() {
        for plane in 0..4 {
            let table = gather_table(plane);
            for lane in 0..2 {
                for element in 0..4 {
                    assert_eq!(table[lane * 16 + element], (4 * element + plane) as i8);
                }
            }
            assert_eq!(table[4..16], [-1i8; 12]);
            assert_eq!(table[20..32], [-1i8; 12]);
        }
    }

    #[test]
    fn avx512_write_matches_the_scalar_reference() {
        assert_write_matches_scalar(1000, write_plane_avx512, "avx512", || {
            has(Feature::Avx512F) && has(Feature::Avx512Bw)
        });
    }

    #[test]
    fn avx512_read_matches_the_scalar_reference() {
        assert_read_matches_scalar(1000, read_plane_avx512, "avx512", || {
            has(Feature::Avx512F) && has(Feature::Avx512Bw)
        });
    }

    #[test]
    fn avx2_write_matches_the_scalar_reference() {
        assert_write_matches_scalar(1000, write_plane_avx2, "avx2", || has(Feature::Avx2));
    }

    #[test]
    fn avx2_read_matches_the_scalar_reference() {
        assert_read_matches_scalar(1000, read_plane_avx2, "avx2", || has(Feature::Avx2));
    }

    /// Mixer rows and inputs from a xorshift32 sweep, all weights well
    /// inside the `i32` range the vector products reproduce exactly.
    fn mixer_fixture() -> ([[i64; NW]; NMX], [i32; NW]) {
        let mut rows = [[0i64; NW]; NMX];
        let mut x = 0x2545_F491u32;
        for row in rows.iter_mut() {
            for w in row.iter_mut() {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                *w = ((x as i32) >> 4) as i64;
            }
        }
        let mut st = [0i32; NW];
        for s in st.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *s = (x as i32 & 0x7FF) - 1024;
        }
        (rows, st)
    }

    /// The wrapping scalar dots, independent of `Model`, as the reference.
    fn scalar_dots(rows: &[[i64; NW]; NMX], st: &[i32; NW]) -> [i64; NMX] {
        let mut out = [0i64; NMX];
        for (j, row) in rows.iter().enumerate() {
            for (&w, &s) in row.iter().zip(st) {
                out[j] = out[j].wrapping_add(w.wrapping_mul(i64::from(s)));
            }
        }
        out
    }

    #[test]
    fn avx2_mixer_dots_match_the_wrapping_scalar_chain() {
        if !has(Feature::Avx2) {
            return;
        }
        let (rows, st) = mixer_fixture();
        let dots = mixer_dots([&rows[0][..], &rows[1][..], &rows[2][..]], &st).unwrap_or([0i64; NMX]);
        assert_eq!(dots, scalar_dots(&rows, &st), "the vector dots diverged");
    }

    #[test]
    fn weights_too_wide_for_the_products_refuse_the_vector_path() {
        let (mut rows, st) = mixer_fixture();
        rows[1][7] = i64::MAX;
        assert!(
            mixer_dots([&rows[0][..], &rows[1][..], &rows[2][..]], &st).is_none(),
            "an out-of-range weight must fall back to the scalar chain"
        );
    }
}
