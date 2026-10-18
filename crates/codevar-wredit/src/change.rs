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

//! Change distance and blending of similar edits.
//!
//! The engine records edits as changes against the block store. Two
//! versions of the same change — for example the same line rewritten
//! by two collaborators, or an edit that only shifts indentation —
//! are not conflicting edits: they are one edit seen twice, and the
//! engine should *blend* them instead of treating them as a conflict.
//!
//! The decision rests on a capped Levenshtein distance: how many
//! single-code-point edits separate the two changes, computed only up
//! to a caller-chosen cap. [`changes_are_similar`] can fold whitespace
//! first ([`ChangeFlags::FOLD_WHITESPACE`]), so changes that differ
//! merely in spacing, indentation, or line breaks collapse to the
//! same shape and blend for free, while genuinely different edits
//! stay apart.
//!
//! The distance itself is computed by the CPU reference
//! ([`levenshtein_capped`]) or, when the caller sets
//! [`ChangeFlags::HOT`], by the GPU kernel behind
//! [`crate::hot::try_levenshtein`]; both paths are required to agree
//! on every input. [`ChangeFlags`] picks that backend and the
//! behavior properties the way `RasterFlags` configures the
//! rasterizer: named constants combined with `|`.

use alloc::vec;
use alloc::vec::Vec;

/// Code points [`fold_whitespace`] treats as whitespace: space, tab,
/// carriage return, and line feed.
const WHITESPACE: [u32; 4] = [0x20, 0x09, 0x0D, 0x0A];

/// Collapses every run of whitespace in `codes` into a single space.
///
/// Two changes that differ only in the amount or kind of whitespace
/// (indentation width, tabs versus spaces, trailing blanks, line
/// breaks) fold to the same sequence, which is what lets
/// [`changes_are_similar`] blend whitespace-only variations.
///
/// # Examples
///
/// ```
/// use codevar_wredit::change::fold_whitespace;
///
/// let indented = [' ' as u32, ' ' as u32, 'x' as u32];
/// let tabs = ['\t' as u32, 'x' as u32];
/// assert_eq!(fold_whitespace(&indented), fold_whitespace(&tabs));
/// ```
pub fn fold_whitespace(codes: &[u32]) -> Vec<u32> {
    let mut folded = Vec::with_capacity(codes.len());
    let mut in_run = false;
    for &code in codes {
        if WHITESPACE.contains(&code) {
            if !in_run {
                folded.push(0x20);
                in_run = true;
            }
        } else {
            folded.push(code);
            in_run = false;
        }
    }
    folded
}

/// Capped Levenshtein distance between two code-point sequences.
///
/// Returns `min(distance, max_dist + 1)`: the exact distance while it
/// does not exceed `max_dist`, and `max_dist + 1` otherwise. The
/// computation is banded — only cells with `|row - col| <= max_dist`
/// are evaluated — so the cost is `O(a.len() * (2 * max_dist + 1))`
/// and a small cap over long sequences stays cheap. This is the
/// reference the GPU path ([`crate::hot::try_levenshtein`]) must
/// match exactly, cell for cell.
///
/// # Examples
///
/// ```
/// use codevar_wredit::change::levenshtein_capped;
///
/// let kitten = [1u32, 2, 3, 4, 5, 6];
/// let sitting = [1u32, 7, 3, 4, 5, 8, 9];
/// assert_eq!(levenshtein_capped(&kitten, &sitting, 10), 3);
/// assert_eq!(levenshtein_capped(&kitten, &sitting, 1), 2);
/// ```
pub fn levenshtein_capped(a: &[u32], b: &[u32], max_dist: u32) -> u32 {
    let longest = a.len().max(b.len());
    let cap = usize::try_from(max_dist)
        .unwrap_or(usize::MAX)
        .min(longest);
    let cap_u32 = u32::try_from(cap).unwrap_or(u32::MAX);
    let sat = cap_u32.saturating_add(1);
    let b_len = b.len();
    // The empty cases never touch the rows below.
    if a.is_empty() {
        return u32::try_from(b_len).unwrap_or(u32::MAX).min(sat);
    }
    if b.is_empty() {
        return u32::try_from(a.len())
            .unwrap_or(u32::MAX)
            .min(sat);
    }
    let mut prev = vec![0u32; b_len + 1];
    let mut cur = vec![0u32; b_len + 1];
    // Row 0: `j` insertions, saturated. Cells past the band stay
    // untouched; the guards below never read them.
    for (j, cell) in prev
        .iter_mut()
        .enumerate()
        .take(b_len.min(cap) + 1)
    {
        *cell = u32::try_from(j).unwrap_or(sat);
    }
    let mut far = false;
    let mut r = 1usize;
    while r <= a.len() && !far {
        // Band of the current and the previous row; cells outside act
        // as `sat`, exactly like the kernel's unwritten cells.
        let lo = r.saturating_sub(cap);
        let hi = r.saturating_add(cap).min(b_len);
        let p_lo = (r - 1).saturating_sub(cap);
        let p_hi = (r - 1).saturating_add(cap).min(b_len);
        let mut row_min = sat;
        if lo == 0 {
            // D(r, 0) = r insertions, saturated.
            let first = u32::try_from(r).unwrap_or(sat).min(sat);
            cur[0] = first;
            row_min = row_min.min(first);
        }
        let mut col = lo.max(1);
        while col <= hi {
            let cost = if a[r - 1] != b[col - 1] { 1 } else { 0 };
            // Each term saturates at the cap before the minimum, so
            // out-of-band reads (guarded to `sat`) can never leak a
            // stale value into a within-cap result.
            let sub = if col > p_lo {
                prev[col - 1].saturating_add(cost).min(sat)
            } else {
                sat
            };
            let del = if col <= p_hi {
                prev[col].saturating_add(1).min(sat)
            } else {
                sat
            };
            let ins = if col > lo {
                cur[col - 1].saturating_add(1).min(sat)
            } else {
                sat
            };
            let value = sub.min(del).min(ins);
            cur[col] = value;
            row_min = row_min.min(value);
            col += 1;
        }
        // The row minimum never decreases across rows: once every
        // in-band cell exceeds the cap, no later row can recover.
        if row_min > cap_u32 {
            far = true;
        }
        core::mem::swap(&mut prev, &mut cur);
        r += 1;
    }
    if far || a.len().abs_diff(b.len()) > cap {
        return sat;
    }
    prev[b_len]
}

/// Backend selection and behavior properties for the change
/// distance operations.
///
/// A transparent bit set in the shape of `RasterFlags`: named
/// constants that callers combine with [`core::ops::BitOr`] and test
/// with [`ChangeFlags::contains`]. [`ChangeFlags::DEFAULT`] (all
/// bits clear) selects the portable CPU reference on raw code
/// points, so a caller that wants no special behavior passes it.
///
/// # Examples
///
/// ```
/// use codevar_wredit::change::ChangeFlags;
///
/// let flags = ChangeFlags::HOT | ChangeFlags::FOLD_WHITESPACE;
/// assert!(flags.contains(ChangeFlags::HOT));
/// assert!(!ChangeFlags::DEFAULT.contains(ChangeFlags::HOT));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct ChangeFlags(pub u32);

impl ChangeFlags {
    /// CPU reference on the raw code points: no GPU, no folding.
    pub const DEFAULT: Self = Self(0x0);
    /// Compute the distance on the GPU through
    /// [`crate::hot::try_levenshtein`], silently falling back to
    /// [`levenshtein_capped`] whenever no device, driver, or
    /// SPIR-V-capable implementation is available (or the input
    /// exceeds the kernel's limits). Both paths agree on every
    /// input, so the flag only selects where the work runs.
    pub const HOT: Self = Self(0x1);
    /// Fold runs of whitespace to a single space
    /// ([`fold_whitespace`]) before comparing, so changes that differ
    /// only in indentation, spacing, tabs versus spaces, or line
    /// breaks collapse to the same shape. Honored by
    /// [`changes_are_similar`]; [`change_distance`] treats its input
    /// as already prepared.
    pub const FOLD_WHITESPACE: Self = Self(0x2);

    /// The raw flag bits.
    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// `true` when every bit of `other` is set in `self`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl core::ops::BitOr for ChangeFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for ChangeFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl core::fmt::Debug for ChangeFlags {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ChangeFlags({:#04X})", self.0)
    }
}

/// Distance between two changes on the backend selected by `flags`.
///
/// With [`ChangeFlags::HOT`] the distance comes from
/// [`crate::hot::try_levenshtein`], falling back to
/// [`levenshtein_capped`] whenever the GPU path answers [`None`]
/// (no device, or an input past the kernel's limits); otherwise the
/// CPU reference runs directly. Both paths return the same value on
/// every input, so the flags never change the result — only where
/// it is computed.
///
/// # Examples
///
/// ```
/// use codevar_wredit::change::{ChangeFlags, change_distance};
///
/// let before = [1u32, 2, 3];
/// let after = [1u32, 2, 4];
/// assert_eq!(change_distance(&before, &after, 8, ChangeFlags::DEFAULT), 1);
/// // The same answer with the GPU asked for (it may fall back).
/// assert_eq!(
///     change_distance(&before, &after, 8, ChangeFlags::HOT),
///     change_distance(&before, &after, 8, ChangeFlags::DEFAULT),
/// );
/// ```
pub fn change_distance(a: &[u32], b: &[u32], max_dist: u32, flags: ChangeFlags) -> u32 {
    if flags.contains(ChangeFlags::HOT)
        && let Some(distance) = crate::hot::try_levenshtein(a, b, max_dist)
    {
        return distance;
    }
    levenshtein_capped(a, b, max_dist)
}

/// Decides whether two changes are similar enough to blend.
///
/// The folded (`ChangeFlags::FOLD_WHITESPACE`) or raw sequences then
/// need a distance of at most `max_dist` — pass a small cap (even
/// `0`) to accept only whitespace-level differences, or a larger one
/// to also blend small real edits such as an occasional renamed
/// identifier.
///
/// [`ChangeFlags::FOLD_WHITESPACE`] is what makes whitespace-only
/// variations blend: without it, indentation and line-break changes
/// count as real edits. [`ChangeFlags::HOT`] selects the distance
/// backend exactly as in [`change_distance`] and never changes the
/// verdict.
///
/// # Examples
///
/// ```
/// use codevar_wredit::change::{ChangeFlags, changes_are_similar};
///
/// // Same line, different indentation: blends with a cap of 0 …
/// let a = [' ' as u32, ' ' as u32, 'x' as u32];
/// let b = ['\t' as u32, 'x' as u32];
/// assert!(changes_are_similar(&a, &b, 0, ChangeFlags::FOLD_WHITESPACE));
/// // … but only when the whitespace folding is asked for.
/// assert!(!changes_are_similar(&a, &b, 0, ChangeFlags::DEFAULT));
/// // A genuinely different edit never blends at cap 0.
/// let other = ['y' as u32, 'z' as u32];
/// assert!(!changes_are_similar(&a, &other, 0, ChangeFlags::FOLD_WHITESPACE));
/// ```
pub fn changes_are_similar(a: &[u32], b: &[u32], max_dist: u32, flags: ChangeFlags) -> bool {
    let distance = if flags.contains(ChangeFlags::FOLD_WHITESPACE) {
        let folded_a = fold_whitespace(a);
        let folded_b = fold_whitespace(b);
        change_distance(&folded_a, &folded_b, max_dist, flags)
    } else {
        change_distance(a, b, max_dist, flags)
    };
    distance <= max_dist
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: classic full-matrix Wagner-Fischer,
    /// used to check the banded computation against unlimited `max_dist`.
    fn reference(a: &[u32], b: &[u32]) -> u32 {
        let mut prev: Vec<u32> = (0..=b.len() as u32).collect();
        let mut cur = vec![0u32; b.len() + 1];
        for r in 1..=a.len() {
            cur[0] = r as u32;
            for c in 1..=b.len() {
                let cost = if a[r - 1] != b[c - 1] { 1 } else { 0 };
                cur[c] = (prev[c - 1] + cost)
                    .min(prev[c] + 1)
                    .min(cur[c - 1] + 1);
            }
            core::mem::swap(&mut prev, &mut cur);
        }
        prev[b.len()]
    }

    #[test]
    fn capped_distance_matches_the_full_reference() {
        let a: Vec<u32> = (0..40u32).map(|i| (i * 37 % 26) + 0x61).collect();
        let b: Vec<u32> = (0..40u32).map(|i| (i * 41 % 26) + 0x61).collect();
        let full = reference(&a, &b);
        // An uncapped band must reproduce the reference exactly.
        assert_eq!(levenshtein_capped(&a, &b, 200), full);
        // Distances beyond the cap collapse to `cap + 1`.
        if full > 3 {
            assert_eq!(levenshtein_capped(&a, &b, 3), 4);
        }
        assert_eq!(levenshtein_capped(&a, &b, full), full);
    }

    #[test]
    fn small_edits_are_counted_exactly() {
        let kitten = [1u32, 2, 3, 4, 5, 6];
        let sitting = [1u32, 7, 3, 4, 5, 8, 9];
        assert_eq!(levenshtein_capped(&kitten, &sitting, 10), 3);
        assert_eq!(levenshtein_capped(&sitting, &kitten, 10), 3);
        // One past the cap: saturated, not exact.
        assert_eq!(levenshtein_capped(&kitten, &sitting, 2), 3);
        assert_eq!(levenshtein_capped(&kitten, &sitting, 0), 1);
    }

    #[test]
    fn empty_inputs_return_the_other_length() {
        assert_eq!(levenshtein_capped(&[], &[], 5), 0);
        assert_eq!(levenshtein_capped(&[], &[1, 2, 3], 5), 3);
        assert_eq!(levenshtein_capped(&[1, 2, 3], &[], 5), 3);
        // Capped: the answer saturates at `max_dist + 1`.
        assert_eq!(levenshtein_capped(&[], &[1, 2, 3], 1), 2);
        assert_eq!(levenshtein_capped(&[1, 2, 3], &[], 0), 1);
    }

    #[test]
    fn identical_sequences_cost_zero() {
        let same: Vec<u32> = (0..64u32).map(|i| i % 7).collect();
        assert_eq!(levenshtein_capped(&same, &same, 0), 0);
        assert_eq!(change_distance(&same, &same, 0, ChangeFlags::DEFAULT), 0);
    }

    #[test]
    fn flags_combine_and_test_bits_like_raster_flags() {
        let both = ChangeFlags::HOT | ChangeFlags::FOLD_WHITESPACE;
        assert_eq!(both.bits(), 0x3);
        assert!(both.contains(ChangeFlags::HOT));
        assert!(both.contains(ChangeFlags::FOLD_WHITESPACE));
        assert!(!ChangeFlags::DEFAULT.contains(ChangeFlags::HOT));
        assert!(!ChangeFlags::HOT.contains(ChangeFlags::FOLD_WHITESPACE));
        let mut accumulated = ChangeFlags::HOT;
        accumulated |= ChangeFlags::FOLD_WHITESPACE;
        assert_eq!(accumulated, both);
        assert_eq!(format!("{:?}", ChangeFlags::DEFAULT), "ChangeFlags(0x00)");
    }

    #[test]
    fn whitespace_folding_collapses_runs_to_one_space() {
        let spaces = [0x20u32, 0x20, 0x20, b'a' as u32];
        let tabs = [0x09u32, 0x09, b'a' as u32];
        let one = [0x20u32, b'a' as u32];
        assert_eq!(fold_whitespace(&spaces), one);
        assert_eq!(fold_whitespace(&tabs), one);
        // A run spanning different whitespace kinds folds as one run.
        assert_eq!(fold_whitespace(&[0x20, 0x0A, b'a' as u32]), one);
        // Non-whitespace stays untouched, including non-ASCII.
        assert_eq!(fold_whitespace(&[0xE9, 0x20, 0xE8]), [0xE9, 0x20, 0xE8]);
        assert_eq!(fold_whitespace(&[]), Vec::<u32>::new());
    }

    #[test]
    fn whitespace_only_changes_blend_but_real_edits_do_not() {
        let indented: Vec<u32> = [' ' as u32, ' ' as u32, ' ' as u32, 'x' as u32].to_vec();
        let tabbed: Vec<u32> = ['\t' as u32, 'x' as u32].to_vec();
        // Folded: the indentation variants collapse and blend.
        assert!(changes_are_similar(
            &indented,
            &tabbed,
            0,
            ChangeFlags::FOLD_WHITESPACE
        ));
        // Unfolded (`DEFAULT`): the same pair is two real edits apart.
        assert!(!changes_are_similar(&indented, &tabbed, 0, ChangeFlags::DEFAULT));
        let different: Vec<u32> = ['y' as u32, 'z' as u32].to_vec();
        assert!(!changes_are_similar(
            &indented,
            &different,
            0,
            ChangeFlags::FOLD_WHITESPACE
        ));
        // With a generous cap even a one-character edit blends.
        assert!(changes_are_similar(
            &indented,
            &different,
            4,
            ChangeFlags::FOLD_WHITESPACE
        ));
        // The backend never changes the verdict.
        assert_eq!(
            changes_are_similar(
                &indented,
                &tabbed,
                0,
                ChangeFlags::FOLD_WHITESPACE | ChangeFlags::HOT
            ),
            changes_are_similar(&indented, &tabbed, 0, ChangeFlags::FOLD_WHITESPACE),
        );
    }

    /// When a device is usable the GPU must agree with the CPU
    /// reference on every input; when it is not, `try_levenshtein`
    /// answers [`None`] and the fallback carries the result. Both
    /// outcomes are valid here.
    #[test]
    fn gpu_distance_agrees_with_the_cpu_reference() {
        let cases: [(Vec<u32>, Vec<u32>, u32); 6] = [
            (vec![], vec![], 3),
            (vec![1, 2, 3], vec![], 3),
            (vec![], vec![1, 2, 3], 3),
            (vec![7, 7, 7], vec![7, 8, 7, 7], 5),
            (
                (0..200u32).map(|i| i % 13).collect(),
                (0..200u32).map(|i| (i * 5) % 13).collect(),
                12,
            ),
            (
                (0..512u32).map(|i| i % 31).collect(),
                (0..512u32).map(|i| (i + 1) % 31).collect(),
                4096,
            ),
        ];
        for (a, b, cap) in cases {
            let cpu = levenshtein_capped(&a, &b, cap);
            // When a device handled the pair, it must agree with the
            // CPU reference; with no device `try_levenshtein` is `None`.
            if let Some(gpu) = crate::hot::try_levenshtein(&a, &b, cap) {
                assert_eq!(gpu, cpu, "GPU/CPU mismatch for cap {cap}");
            }
            // The hot-aware wrapper must never disagree either.
            assert_eq!(change_distance(&a, &b, cap, ChangeFlags::HOT), cpu);
        }
    }

    /// Early exit and band guards on sequences whose distance sits
    /// right at the diagonal-gap boundary.
    #[test]
    fn wide_diagonal_gap_saturates() {
        let short = [1u32, 2];
        let long: Vec<u32> = (0..50u32).collect();
        // The lengths differ by more than the cap: saturated at once.
        assert_eq!(levenshtein_capped(&short, &long, 5), 6);
        assert_eq!(levenshtein_capped(&long, &short, 5), 6);
        // But an uncapped run finds the true distance.
        let full = reference(&short, &long);
        assert_eq!(levenshtein_capped(&short, &long, 100), full);
    }
}
