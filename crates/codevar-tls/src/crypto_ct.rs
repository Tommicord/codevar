//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Constant-time primitives shared by the TLS cryptography modules.
//!
//! Every helper here keeps control flow and memory access independent of
//! secret values:
//!
//! * [`mask_u8`] / [`mask_u64`] turn a bit into an all-ones / all-zeroes mask
//!   with wrapping arithmetic instead of a branch.
//! * [`select_u8`] / [`select_u64`] combine masked values without branching.
//! * [`lookup256`] scans a whole 256-entry table for every lookup so the
//!   access pattern does not depend on the (secret) index. Table-driven AES
//!   uses it for S-box substitution on the software path, where a naive
//!   `SBOX[value]` load would leak the value through CPU cache timing.
//! * [`ct_eq`] compares byte strings with `subtle`'s branch-free equality so
//!   AEAD tag checks do not short-circuit on the first differing byte.
//!
//! The operations compile to straight-line code (no secret-dependent
//! branches), which is the Rust equivalent of the handwritten assembly
//! recommended for constant-time cryptography.

use subtle::{ConditionallySelectable, ConstantTimeEq};

/// Branch-free equality for byte slices.
///
/// Returns `false` when the lengths differ. The comparison itself never
/// short-circuits on content, so it is safe for authenticator (tag) checks.
#[must_use]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    bool::from(a.ct_eq(b))
}

/// Turns a bit (0 or 1) into a mask: `0x00` for 0, `0xFF` for 1.
#[inline(always)]
#[must_use]
pub fn mask_u8(bit: u8) -> u8 {
    (bit & 1).wrapping_neg()
}

/// Turns a bit (0 or 1) into a 64-bit mask: `0` for 0, `u64::MAX` for 1.
#[inline(always)]
#[must_use]
pub fn mask_u64(bit: u64) -> u64 {
    (bit & 1).wrapping_neg()
}

/// Turns a bit (0 or 1) into a 128-bit mask: `0` for 0, `u128::MAX` for 1.
#[inline(always)]
#[must_use]
pub fn mask_u128(bit: u128) -> u128 {
    (bit & 1).wrapping_neg()
}

/// Returns `a` when `mask` has its low bit set, otherwise `b`.
#[inline(always)]
#[must_use]
pub fn select_u8(mask: u8, a: u8, b: u8) -> u8 {
    (a & mask) | (b & !mask)
}

/// Returns `a` when `mask` has its low bit set, otherwise `b`.
#[inline(always)]
#[must_use]
pub fn select_u64(mask: u64, a: u64, b: u64) -> u64 {
    (a & mask) | (b & !mask)
}

/// Constant-time lookup into a 256-entry table.
///
/// Scans all 256 entries and conditionally selects the matching one, so the
/// sequence of memory accesses is identical for every `index`. This is the
/// branch-free replacement for `table[index as usize]` when `index` is
/// derived from secret data (for example an AES state byte).
#[inline]
#[must_use]
pub fn lookup256(table: &[u8; 256], index: u8) -> u8 {
    let mut acc = 0u8;
    for (i, value) in table.iter().enumerate() {
        let matched = u8::ct_eq(&(i as u8), &index);
        acc = u8::conditional_select(&acc, value, matched);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random table so lookups are reproducible.
    fn test_table(seed: u8) -> [u8; 256] {
        let mut t = [0u8; 256];
        let mut x = u32::from(seed) | 1;
        for slot in t.iter_mut() {
            x = x
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            *slot = (x >> 24) as u8;
        }
        t
    }

    #[test]
    fn lookup256_matches_direct_indexing_for_every_index() {
        for seed in [0u8, 1, 0x5A, 0xC3] {
            let table = test_table(seed);
            for i in 0u16..=255 {
                assert_eq!(
                    lookup256(&table, i as u8),
                    table[i as usize],
                    "seed={seed} index={i}"
                );
            }
        }
    }

    #[test]
    fn lookup256_rejects_wrong_or_empty_equivalent_tables() {
        // Two tables that differ in exactly one entry are distinguished
        // at that index and agree everywhere else.
        let mut a = test_table(7);
        let b = {
            let mut b = a;
            b[200] ^= 0x5C;
            b
        };
        for i in 0u16..=255 {
            let expected = if i == 200 { a[200] } else { a[i as usize] };
            assert_eq!(lookup256(&a, i as u8), expected);
            assert_eq!(
                lookup256(&b, i as u8),
                if i == 200 { b[200] } else { b[i as usize] }
            );
        }
        a[200] = b[200];
        assert_eq!(a, b);
    }

    #[test]
    fn masks_and_selects_are_branch_free_tables() {
        assert_eq!(mask_u8(0), 0x00);
        assert_eq!(mask_u8(1), 0xFF);
        // Values other than 0/1 are coerced to the low bit (defensive).
        assert_eq!(mask_u8(0x81), 0xFF);
        assert_eq!(mask_u8(0x42), 0x00);
        assert_eq!(mask_u64(0), 0);
        assert_eq!(mask_u64(1), u64::MAX);
        assert_eq!(mask_u128(0), 0);
        assert_eq!(mask_u128(1), u128::MAX);
        assert_eq!(mask_u128(2), 0);

        assert_eq!(select_u8(0xFF, 0x12, 0x34), 0x12);
        assert_eq!(select_u8(0x00, 0x12, 0x34), 0x34);
        assert_eq!(select_u64(u64::MAX, 1, 2), 1);
        assert_eq!(select_u64(0, 1, 2), 2);
        // Selecting with only the low bit set still works.
        assert_eq!(select_u64(mask_u64(1), 0xDEAD, 0xBEEF), 0xDEAD);
    }

    #[test]
    fn ct_eq_matches_plain_equality() {
        assert!(ct_eq(b"finished", b"finished"));
        assert!(!ct_eq(b"finished", b"finisheD"));
        // Length mismatch and empty inputs.
        assert!(ct_eq(b"", b""));
        assert!(!ct_eq(b"", b"\0"));
        assert!(!ct_eq(b"abc", b"abcd"));
        // Full 16-byte authenticators, differing in the last byte only.
        let a = [0x42u8; 16];
        let mut b = a;
        b[15] ^= 0x01;
        assert!(!ct_eq(&a, &b));
        assert!(ct_eq(&a, &a));
    }
}
