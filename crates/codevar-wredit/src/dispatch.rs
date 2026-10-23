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

//! Backend selection between the CPU/SIMD tiers and the GPU kernels.
//!
//! The engine has two ways to run a bulk scan: the runtime-dispatched
//! SIMD kernels in [`crate::simd`] (CPU), and the OpenCL kernels in
//! [`crate::hot`] (GPU). The GPU wins on huge documents but pays a
//! fixed per-dispatch cost — a buffer upload and a read-back, plus a
//! program rebuild whenever the process switches kernels — so on small
//! inputs it is strictly slower than the CPU tier.
//!
//! [`Heuristic`] encodes that crossover, [`BulkBackend`] is the tier
//! contract (the GPU tier answers [`None`] when it declines, the SIMD
//! tier always answers), and [`Accelerator`] applies the heuristic and
//! falls through from the GPU to the SIMD tier. [`BulkOps`] is the
//! engine-facing surface: callers depend on it without knowing which
//! tier ran.
//!
//! # Performance
//!
//! The defaults in [`Heuristic::DEFAULT`] are deliberately
//! conservative: they keep the CPU/SIMD tier for everything below a few
//! mebibytes, so enabling the dispatcher can never make an existing
//! small edit slower. Tune them with `cargo bench -p codevar-wredit
//! --bench hot` on the target machine.

use alloc::vec;
use alloc::vec::Vec;

use crate::hot;
use crate::simd;

/// Which execution tier should service a bulk operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The parallel OpenCL kernels.
    Gpu,
    /// The runtime-dispatched SIMD kernels.
    Cpu,
}

/// The input sizes at which the GPU overtakes the CPU/SIMD tier.
///
/// Two thresholds because the two families of kernels take different
/// unit sizes: the byte kernels index UTF-8/transaction buffers, the
/// word kernels index the engine's `u32` code points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heuristic {
    /// Minimum byte length before the GPU is considered.
    pub gpu_min_bytes: usize,
    /// Minimum word length before the GPU is considered.
    pub gpu_min_words: usize,
}

impl Heuristic {
    /// Conservative defaults: the GPU is only considered from a few
    /// mebibytes upward, so small edits always stay on the CPU tier.
    pub const DEFAULT: Self = Self {
        gpu_min_bytes: 4 << 20,
        gpu_min_words: 1 << 20,
    };

    /// The tier to use for a byte-oriented operation of `len` bytes.
    #[must_use]
    pub const fn backend_for_bytes(&self, len: usize) -> Backend {
        if len >= self.gpu_min_bytes {
            Backend::Gpu
        } else {
            Backend::Cpu
        }
    }

    /// The tier to use for a word-oriented operation of `len` words.
    #[must_use]
    pub const fn backend_for_words(&self, len: usize) -> Backend {
        if len >= self.gpu_min_words {
            Backend::Gpu
        } else {
            Backend::Cpu
        }
    }
}

impl Default for Heuristic {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A tier that can service the engine's bulk operations.
///
/// Every method answers [`None`] to mean "this tier declines; run the
/// other one". The SIMD tier never declines, so the dispatcher always
/// has a result; the GPU tier declines when there is no device, when a
/// buffer is empty, or when an input is past a kernel's 32-bit limits.
pub trait BulkBackend {
    /// Counts the UTF-8 code points in `bytes`.
    fn count_code_points(&self, bytes: &[u8]) -> Option<u64> {
        let _ = bytes;
        None
    }

    /// Position-weighted checksum of `bytes`.
    fn checksum(&self, bytes: &[u8]) -> Option<u32> {
        let _ = bytes;
        None
    }

    /// Counts line feeds in a plain `u32` code-point buffer.
    fn count_newlines(&self, words: &[u32]) -> Option<usize> {
        let _ = words;
        None
    }

    /// Counts occurrences of `needle` in a `u32` buffer.
    fn count_value(&self, words: &[u32], needle: u32) -> Option<usize> {
        let _ = (words, needle);
        None
    }

    /// Locates the first `needle` in a `u32` buffer.
    ///
    /// The outer [`Option`] reports whether this tier handled the query;
    /// the inner one is the match. [`None`] as the inner value therefore
    /// means "handled, no match", not "unavailable".
    fn find_first(&self, words: &[u32], needle: u32) -> Option<Option<usize>> {
        let _ = (words, needle);
        None
    }

    /// Marks which positions of `words` start a `needle` match.
    ///
    /// `flags[i]` is `1` when `words[i..i + needle.len()] == needle` and
    /// `0` otherwise; the result has the same length as `words`.
    fn find_matches(&self, words: &[u32], needle: &[u32]) -> Option<Vec<u32>> {
        let _ = (words, needle);
        None
    }

    /// Compacts `words` by dropping the gap `[gap_start, gap_start +
    /// gap_len)`; the input holds `live_len + gap_len` elements and the
    /// result holds the `live_len` survivors.
    fn gap_compact(&self, words: &[u32], gap_start: usize, gap_len: usize) -> Option<Vec<u32>> {
        let _ = (words, gap_start, gap_len);
        None
    }
}

/// The CPU/SIMD tier: runtime-dispatched vector kernels, always able to
/// answer.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdBackend;

/// The GPU tier: the [`crate::hot`] OpenCL host wrappers, which decline
/// whenever a pipeline cannot be built or an input is out of range.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuBackend;

impl BulkBackend for SimdBackend {
    fn count_code_points(&self, bytes: &[u8]) -> Option<u64> {
        Some(simd::count_utf8_units(bytes))
    }

    fn checksum(&self, bytes: &[u8]) -> Option<u32> {
        Some(simd::linear_checksum(bytes))
    }

    fn count_newlines(&self, words: &[u32]) -> Option<usize> {
        Some(
            words
                .iter()
                .filter(|&&word| word == u32::from(b'\n'))
                .count(),
        )
    }

    fn count_value(&self, words: &[u32], needle: u32) -> Option<usize> {
        Some(simd::count_value_u32(words, needle) as usize)
    }

    fn find_first(&self, words: &[u32], needle: u32) -> Option<Option<usize>> {
        Some(simd::find_first_u32(words, needle))
    }

    fn find_matches(&self, words: &[u32], needle: &[u32]) -> Option<Vec<u32>> {
        Some(simd_find_matches(words, needle))
    }

    fn gap_compact(&self, words: &[u32], gap_start: usize, gap_len: usize) -> Option<Vec<u32>> {
        if gap_len == 0 || gap_start > words.len() {
            return None;
        }
        let live_len = words.len().saturating_sub(gap_len);
        let start = gap_start.min(live_len);
        let mut out = Vec::with_capacity(live_len);
        out.extend_from_slice(&words[..start]);
        out.extend_from_slice(&words[start + gap_len..live_len + gap_len]);
        Some(out)
    }
}

impl BulkBackend for GpuBackend {
    fn count_code_points(&self, bytes: &[u8]) -> Option<u64> {
        hot::try_char_count(bytes)
    }

    fn checksum(&self, bytes: &[u8]) -> Option<u32> {
        hot::try_block_checksum(bytes)
    }

    fn count_newlines(&self, words: &[u32]) -> Option<usize> {
        hot::try_count_newlines(words)
    }

    fn count_value(&self, words: &[u32], needle: u32) -> Option<usize> {
        hot::try_count_u32(words, needle)
    }

    fn find_first(&self, words: &[u32], needle: u32) -> Option<Option<usize>> {
        hot::try_find_first_u32(words, needle)
    }

    fn find_matches(&self, words: &[u32], needle: &[u32]) -> Option<Vec<u32>> {
        if words.is_empty() || needle.is_empty() {
            return None;
        }
        // Safety: both slices are valid for their whole lengths.
        unsafe { hot::find_matches(words.as_ptr(), words.len(), needle.as_ptr(), needle.len()) }
    }

    fn gap_compact(&self, words: &[u32], gap_start: usize, gap_len: usize) -> Option<Vec<u32>> {
        if gap_len == 0 || gap_start > words.len() {
            return None;
        }
        let live_len = words.len().checked_sub(gap_len)?;
        let mut out = vec![0u32; live_len];
        // Safety: `words` is valid for its whole length; `out` holds
        // exactly `live_len` elements and `gap_compact` writes no more.
        let handled =
            unsafe { hot::gap_compact(words.as_ptr(), live_len, gap_start, gap_len, out.as_mut_ptr()) };
        handled.map(|()| out)
    }
}

/// CPU/SIMD match flags for [`BulkBackend::find_matches`].
///
/// Only positions whose leading word equals `needle[0]` can start a
/// match, so the scan uses [`simd::find_first_u32`] to jump between
/// candidates instead of comparing every position: the vector kernel
/// rejects the long runs that do not contain the first word.
fn simd_find_matches(words: &[u32], needle: &[u32]) -> Vec<u32> {
    let mut flags = vec![0u32; words.len()];
    if needle.is_empty() || needle.len() > words.len() {
        return flags;
    }
    let last_start = words.len() - needle.len();
    let first = needle[0];
    let mut from = 0usize;
    while from <= last_start {
        let Some(offset) = simd::find_first_u32(&words[from..=last_start], first) else {
            break;
        };
        let candidate = from + offset;
        if words[candidate..candidate + needle.len()] == *needle {
            flags[candidate] = 1;
        }
        from = candidate + 1;
    }
    flags
}

/// The bulk operations the editing engine runs over documents.
///
/// This is the seam between the edit layer and the accelerated tiers:
/// implementors decide which backend runs, callers only see the result.
pub trait BulkOps {
    /// Counts the UTF-8 code points in `bytes`.
    fn count_code_points(&self, bytes: &[u8]) -> u64;

    /// Position-weighted checksum of `bytes`.
    fn checksum(&self, bytes: &[u8]) -> u32;

    /// Counts line feeds in a plain `u32` code-point buffer.
    fn count_newlines(&self, words: &[u32]) -> usize;

    /// Counts occurrences of `needle` in a `u32` buffer.
    fn count_value(&self, words: &[u32], needle: u32) -> usize;

    /// Locates the first `needle` in a `u32` buffer.
    fn find_first(&self, words: &[u32], needle: u32) -> Option<usize>;

    /// Marks which positions of `words` start a `needle` match.
    fn find_matches(&self, words: &[u32], needle: &[u32]) -> Vec<u32>;

    /// Compacts `words` by dropping the gap `[gap_start, gap_start +
    /// gap_len)`.
    fn gap_compact(&self, words: &[u32], gap_start: usize, gap_len: usize) -> Vec<u32>;
}

/// Runs [`BulkOps`] over the GPU and SIMD tiers, picking one per call.
#[derive(Debug, Clone, Copy)]
pub struct Accelerator {
    heuristic: Heuristic,
    gpu: GpuBackend,
    cpu: SimdBackend,
}

impl Accelerator {
    /// Builds an accelerator with an explicit heuristic.
    #[must_use]
    pub const fn new(heuristic: Heuristic) -> Self {
        Self {
            heuristic,
            gpu: GpuBackend,
            cpu: SimdBackend,
        }
    }

    /// Builds an accelerator with the conservative defaults.
    #[must_use]
    pub const fn with_defaults() -> Self {
        Self::new(Heuristic::DEFAULT)
    }

    /// The policy in force.
    #[must_use]
    pub const fn heuristic(&self) -> Heuristic {
        self.heuristic
    }

    /// `true` when a byte operation of `len` bytes is routed to the GPU.
    #[must_use]
    pub const fn prefers_gpu_for_bytes(&self, len: usize) -> bool {
        matches!(self.heuristic.backend_for_bytes(len), Backend::Gpu)
    }

    /// `true` when a word operation of `len` words is routed to the GPU.
    #[must_use]
    pub const fn prefers_gpu_for_words(&self, len: usize) -> bool {
        matches!(self.heuristic.backend_for_words(len), Backend::Gpu)
    }
}

impl BulkOps for Accelerator {
    fn count_code_points(&self, bytes: &[u8]) -> u64 {
        if self.prefers_gpu_for_bytes(bytes.len())
            && let Some(count) = self.gpu.count_code_points(bytes)
        {
            return count;
        }
        // The SIMD tier always answers; the zero is an unreachable net.
        self.cpu.count_code_points(bytes).unwrap_or(0)
    }

    fn checksum(&self, bytes: &[u8]) -> u32 {
        if self.prefers_gpu_for_bytes(bytes.len())
            && let Some(sum) = self.gpu.checksum(bytes)
        {
            return sum;
        }
        self.cpu.checksum(bytes).unwrap_or(0)
    }

    fn count_newlines(&self, words: &[u32]) -> usize {
        if self.prefers_gpu_for_words(words.len())
            && let Some(count) = self.gpu.count_newlines(words)
        {
            return count;
        }
        self.cpu.count_newlines(words).unwrap_or(0)
    }

    fn count_value(&self, words: &[u32], needle: u32) -> usize {
        if self.prefers_gpu_for_words(words.len())
            && let Some(count) = self.gpu.count_value(words, needle)
        {
            return count;
        }
        self.cpu.count_value(words, needle).unwrap_or(0)
    }

    fn find_first(&self, words: &[u32], needle: u32) -> Option<usize> {
        if self.prefers_gpu_for_words(words.len())
            && let Some(found) = self.gpu.find_first(words, needle)
        {
            return found;
        }
        self.cpu.find_first(words, needle).flatten()
    }

    fn find_matches(&self, words: &[u32], needle: &[u32]) -> Vec<u32> {
        if self.prefers_gpu_for_words(words.len())
            && let Some(flags) = self.gpu.find_matches(words, needle)
        {
            return flags;
        }
        self.cpu
            .find_matches(words, needle)
            .unwrap_or_else(|| vec![0u32; words.len()])
    }

    fn gap_compact(&self, words: &[u32], gap_start: usize, gap_len: usize) -> Vec<u32> {
        if self.prefers_gpu_for_words(words.len())
            && let Some(out) = self.gpu.gap_compact(words, gap_start, gap_len)
        {
            return out;
        }
        self.cpu
            .gap_compact(words, gap_start, gap_len)
            .unwrap_or_default()
    }
}

impl Default for Accelerator {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::{Accelerator, Backend, BulkBackend, BulkOps, Heuristic, SimdBackend};

    /// A heuristic that always takes the GPU branch, so the tests
    /// exercise the fall-through path regardless of the device.
    const ALL_GPU: Heuristic = Heuristic {
        gpu_min_bytes: 0,
        gpu_min_words: 0,
    };

    fn words(len: usize) -> alloc::vec::Vec<u32> {
        let mut out = alloc::vec![u32::from(b'x'); len];
        if len > 0 {
            out[0] = u32::from(b'\n');
        }
        if len > 1 {
            out[len - 1] = 7;
        }
        out
    }

    #[test]
    fn heuristic_selects_by_threshold() {
        let heuristic = Heuristic {
            gpu_min_bytes: 10,
            gpu_min_words: 20,
        };
        assert_eq!(heuristic.backend_for_bytes(9), Backend::Cpu);
        assert_eq!(heuristic.backend_for_bytes(10), Backend::Gpu);
        assert_eq!(heuristic.backend_for_words(19), Backend::Cpu);
        assert_eq!(heuristic.backend_for_words(20), Backend::Gpu);
        // The defaults keep small edits on the CPU tier.
        let defaults = Heuristic::DEFAULT;
        assert_eq!(defaults.backend_for_bytes(1024), Backend::Cpu);
        assert_eq!(defaults.backend_for_words(1024), Backend::Cpu);
    }

    /// The dispatcher must return exactly what the SIMD tier returns,
    /// whichever tier actually ran.
    #[test]
    fn accelerator_matches_the_simd_tier() {
        let cpu = SimdBackend;
        for len in [0usize, 1, 63, 64, 65, 300] {
            let buffer = words(len);
            for accelerator in [Accelerator::with_defaults(), Accelerator::new(ALL_GPU)] {
                assert_eq!(
                    accelerator.find_first(&buffer, 7),
                    cpu.find_first(&buffer, 7).flatten(),
                    "len {len}"
                );
                assert_eq!(
                    accelerator.count_value(&buffer, u32::from(b'x')),
                    cpu.count_value(&buffer, u32::from(b'x'))
                        .unwrap_or(0),
                    "len {len}"
                );
                assert_eq!(
                    accelerator.count_newlines(&buffer),
                    cpu.count_newlines(&buffer).unwrap_or(0),
                    "len {len}"
                );
            }
        }
    }

    #[test]
    fn accelerator_matches_the_simd_tier_for_bytes() {
        let cpu = SimdBackend;
        for len in [0usize, 1, 63, 64, 65, 400] {
            let mut bytes = alloc::vec![b'a'; len];
            if len >= 2 {
                bytes[len - 1] = 0xC3;
                bytes[len - 2] = 0xA9;
            }
            for accelerator in [Accelerator::with_defaults(), Accelerator::new(ALL_GPU)] {
                assert_eq!(
                    accelerator.count_code_points(&bytes),
                    cpu.count_code_points(&bytes).unwrap_or(0),
                    "len {len}"
                );
                assert_eq!(
                    accelerator.checksum(&bytes),
                    cpu.checksum(&bytes).unwrap_or(0),
                    "len {len}"
                );
            }
        }
    }

    #[test]
    fn gap_compact_drops_the_gap_for_both_tiers() {
        // Four live elements after dropping `[1, 4)` out of a seven-slot
        // buffer.
        let buffer = [10u32, 20, 30, 40, 50, 60, 70];
        let expected = alloc::vec![10u32, 50, 60, 70];
        let cpu = SimdBackend;
        assert_eq!(cpu.gap_compact(&buffer, 1, 3), Some(expected.clone()));
        for accelerator in [Accelerator::with_defaults(), Accelerator::new(ALL_GPU)] {
            assert_eq!(accelerator.gap_compact(&buffer, 1, 3), expected, "len 7");
        }
        // A zero-length gap is left to the caller (empty result).
        assert_eq!(
            Accelerator::with_defaults().gap_compact(&buffer, 0, 0),
            alloc::vec![]
        );
    }

    /// Match flags must agree with a scalar oracle for a length-one
    /// needle and for overlapping multi-word needles.
    #[test]
    fn find_matches_flags_match_a_scalar_oracle() {
        let haystack = [1u32, 2, 1, 2, 1, 2, 1];
        let oracle = |needle: &[u32]| -> alloc::vec::Vec<u32> {
            let mut flags = alloc::vec![0u32; haystack.len()];
            if needle.is_empty() || needle.len() > haystack.len() {
                return flags;
            }
            for i in 0..=haystack.len() - needle.len() {
                if haystack[i..i + needle.len()] == *needle {
                    flags[i] = 1;
                }
            }
            flags
        };
        for needle in [
            alloc::vec![1u32],
            alloc::vec![1, 2, 1],
            alloc::vec![2, 1],
            alloc::vec![9],
        ] {
            let expected = oracle(&needle);
            for accelerator in [Accelerator::with_defaults(), Accelerator::new(ALL_GPU)] {
                assert_eq!(
                    accelerator.find_matches(&haystack, &needle),
                    expected,
                    "needle {needle:?}"
                );
            }
        }
    }
}
