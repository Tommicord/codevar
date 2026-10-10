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

//! Constant-time primitives built on all-ones bit masks.
//!
//! Every comparison here returns a *mask* (`0` or `u64::MAX`, never an
//! intermediate value) computed with fixed-sequence wrapping integer
//! arithmetic — no branches, no lookup tables, and no indexing that
//! depends on the operands. [`select`] then chooses between two values
//! with pure AND/OR/XOR, so callers such as the Mergen merge algorithm
//! can order entries, resolve conflicts, and pick winners in constant
//! time with respect to the *values* involved (lengths and counts are
//! public and may still branch).
//!
//! A mask `m` is `0` when the predicate is false and `u64::MAX` when
//! it is true, so `select(m, a, b)` is `a` for a true predicate and
//! `b` for a false one.
//!
//! ## Examples
//!
//! ```
//! use codevar_shake::ct;
//!
//! // min/max without branching on the values.
//! let (a, b) = (7u64, 11u64);
//! let less = ct::lt_mask(a, b);
//! assert_eq!(ct::min(a, b), 7);
//! assert_eq!(ct::max(a, b), 11);
//! assert_eq!(ct::select(less, a, b), a);
//! ```

/// All-ones mask (`u64::MAX`) when `bit` is 1, zero when `bit` is 0.
///
/// Only the least significant bit of `bit` is inspected.
#[inline]
#[must_use]
pub fn mask_from_bit(bit: u64) -> u64 {
    (bit & 1).wrapping_neg()
}

/// Chooses `a` when the mask is all-ones, `b` when it is zero.
///
/// For any mask other than those two canonical values the result is a
/// bitwise mix of `a` and `b`; callers should always obtain the mask
/// from a predicate function in this module.
#[inline]
#[must_use]
pub fn select(mask: u64, a: u64, b: u64) -> u64 {
    (a & mask) | (b & !mask)
}

/// Chooses `a` when the mask is all-ones, `b` when it is zero.
///
/// See [`select`] for the mask contract.
#[inline]
#[must_use]
pub fn select_u8(mask: u64, a: u8, b: u8) -> u8 {
    select(mask, u64::from(a), u64::from(b)) as u8
}

/// Swaps `a` and `b` when the mask is all-ones, leaves them when it
/// is zero. Branchless XOR swap; see [`select`] for the mask
/// contract.
#[inline]
pub fn swap(mask: u64, a: &mut u64, b: &mut u64) {
    let delta = (*a ^ *b) & mask;
    *a ^= delta;
    *b ^= delta;
}

/// All-ones mask when `x` is zero, zero otherwise.
///
/// Computed as `(((x | -x) >> 63) & 1 ^ 1).wrapping_neg()`:
/// `x | -x` has its sign bit set for every nonzero `x` (two's
/// complement, wrapping negation), so the inverted sign bit is 1 only
/// when `x == 0`.
#[inline]
#[must_use]
pub fn is_zero_mask(x: u64) -> u64 {
    ((((x | x.wrapping_neg()) >> 63) & 1) ^ 1).wrapping_neg()
}

/// All-ones mask when `a == b`, zero otherwise.
#[inline]
#[must_use]
pub fn eq_mask(a: u64, b: u64) -> u64 {
    is_zero_mask(a ^ b)
}

/// All-ones mask when `a != b`, zero otherwise.
#[inline]
#[must_use]
pub fn ne_mask(a: u64, b: u64) -> u64 {
    !eq_mask(a, b)
}

/// All-ones mask when `a < b` (unsigned), zero otherwise.
///
/// Unsigned order equals signed order once the sign bit of both
/// operands is flipped, so this flips both and applies Hacker's
/// Delight signed less-than: with `diff = x - y` (wrapping),
/// `t = (x ^ y) & (diff ^ x)`, the sign bit of `diff ^ t` is set
/// exactly when `x <s y`.
#[inline]
#[must_use]
pub fn lt_mask(a: u64, b: u64) -> u64 {
    const TOP: u64 = 1 << 63;
    let x = a ^ TOP;
    let y = b ^ TOP;
    let diff = x.wrapping_sub(y);
    let t = (x ^ y) & (diff ^ x);
    ((diff ^ t) >> 63).wrapping_neg()
}

/// All-ones mask when `a > b` (unsigned), zero otherwise.
#[inline]
#[must_use]
pub fn gt_mask(a: u64, b: u64) -> u64 {
    lt_mask(b, a)
}

/// All-ones mask when `a <= b` (unsigned), zero otherwise.
#[inline]
#[must_use]
pub fn le_mask(a: u64, b: u64) -> u64 {
    !gt_mask(a, b)
}

/// All-ones mask when `a >= b` (unsigned), zero otherwise.
#[inline]
#[must_use]
pub fn ge_mask(a: u64, b: u64) -> u64 {
    !lt_mask(a, b)
}

/// The smaller of `a` and `b` (unsigned), selected branchlessly.
#[inline]
#[must_use]
pub fn min(a: u64, b: u64) -> u64 {
    select(lt_mask(a, b), a, b)
}

/// The larger of `a` and `b` (unsigned), selected branchlessly.
#[inline]
#[must_use]
pub fn max(a: u64, b: u64) -> u64 {
    select(lt_mask(a, b), b, a)
}

/// All-ones mask when the first `out.len()` bytes of `a` and `b` are
/// equal, zero otherwise.
///
/// Compares the common prefix with one branchless accumulation; any
/// length difference beyond that prefix makes the result not-equal.
/// Buffer *lengths* are public, so the length branch is allowed.
#[inline]
#[must_use]
pub fn eq_bytes_mask(a: &[u8], b: &[u8]) -> u64 {
    let shared = a.len().min(b.len());
    let mut acc = 0u64;
    for index in 0..shared {
        acc |= u64::from(a[index] ^ b[index]);
    }
    if a.len() != b.len() {
        return 0;
    }
    is_zero_mask(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift-style PRNG so the tests need no
    /// dependency and stay reproducible.
    fn next(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    #[test]
    fn mask_from_bit_is_canonical() {
        assert_eq!(mask_from_bit(0), 0);
        assert_eq!(mask_from_bit(1), u64::MAX);
        assert_eq!(mask_from_bit(2), 0);
        assert_eq!(mask_from_bit(0xDEAD_BEEF), u64::MAX);
    }

    #[test]
    fn select_picks_per_mask() {
        assert_eq!(select(u64::MAX, 1, 2), 1);
        assert_eq!(select(0, 1, 2), 2);
    }

    #[test]
    fn select_u8_picks_per_mask() {
        assert_eq!(select_u8(u64::MAX, 0xAB, 0xCD), 0xAB);
        assert_eq!(select_u8(0, 0xAB, 0xCD), 0xCD);
    }

    #[test]
    fn swap_swaps_only_under_all_ones() {
        let mut a = 5u64;
        let mut b = 9u64;
        swap(u64::MAX, &mut a, &mut b);
        assert_eq!((a, b), (9, 5));
        swap(0, &mut a, &mut b);
        assert_eq!((a, b), (9, 5));
    }

    #[test]
    fn predicates_match_reference_over_edge_cases_and_stream() {
        let edge = [
            0u64,
            1,
            2,
            u64::MAX,
            u64::MAX - 1,
            1 << 63,
            (1 << 63) - 1,
            0x8000_0000_0000_0000,
            0x7FFF_FFFF_FFFF_FFFF,
        ];
        let mut pairs: Vec<(u64, u64)> = Vec::new();
        for &x in &edge {
            for &y in &edge {
                pairs.push((x, y));
            }
        }
        let mut state = 0x243F_6A88_85A3_08D3u64;
        for _ in 0..512 {
            pairs.push((next(&mut state), next(&mut state)));
        }
        for (a, b) in pairs {
            assert_eq!(is_zero_mask(a) != 0, a == 0, "is_zero({a})");
            assert_eq!(eq_mask(a, b) != 0, a == b, "eq({a}, {b})");
            assert_eq!(ne_mask(a, b) != 0, a != b, "ne({a}, {b})");
            assert_eq!(lt_mask(a, b) != 0, a < b, "lt({a}, {b})");
            assert_eq!(gt_mask(a, b) != 0, a > b, "gt({a}, {b})");
            assert_eq!(le_mask(a, b) != 0, a <= b, "le({a}, {b})");
            assert_eq!(ge_mask(a, b) != 0, a >= b, "ge({a}, {b})");
            assert_eq!(min(a, b), a.min(b), "min({a}, {b})");
            assert_eq!(max(a, b), a.max(b), "max({a}, {b})");
            assert_eq!(select(lt_mask(a, b), a, b), a.min(b));
        }
    }

    #[test]
    fn eq_bytes_mask_matches_slices() {
        assert_ne!(eq_bytes_mask(b"abc", b"abc"), 0);
        assert_eq!(eq_bytes_mask(b"abc", b"abd"), 0);
        assert_eq!(eq_bytes_mask(b"abc", b"ab"), 0);
        assert_eq!(eq_bytes_mask(b"ab", b"abc"), 0);
        assert_ne!(eq_bytes_mask(b"", b""), 0);
        let big_a = vec![0u8; 4096];
        let mut big_b = vec![0u8; 4096];
        assert_ne!(eq_bytes_mask(&big_a, &big_b), 0);
        big_b[4095] = 1;
        assert_eq!(eq_bytes_mask(&big_a, &big_b), 0);
    }
}
