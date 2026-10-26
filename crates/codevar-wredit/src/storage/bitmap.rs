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

//! The free-space bitmap of the engine storage.
//!
//! Disk file systems (ext2 and friends) track free blocks and free
//! inodes with a bitmap laid out at a fixed location, so allocation
//! is a single scan for the first clear bit and freeing is a single
//! bit store. [`Bitmap`] is exactly that: one bit per address, bit 0
//! of byte 0 the lowest address, `1` meaning *used*.

use crate::error::{WreditError, WreditResult};

use super::block::BLOCK_SIZE;

/// A one-block allocation bitmap: `BLOCK_SIZE * 8` bits, `1` = used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bitmap {
    /// One bit per address; byte `i` covers addresses `8*i..8*i + 8`.
    bits: [u8; BLOCK_SIZE],
}

impl Bitmap {
    /// An all-clear bitmap: every address is free.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bits: [0; BLOCK_SIZE],
        }
    }

    /// `true` when the bit for `index` is set (the address is used).
    #[must_use]
    pub fn is_used(&self, index: u32) -> bool {
        let (byte, mask) = split(index);
        self.bits
            .get(byte)
            .is_some_and(|byte| byte & mask != 0)
    }

    /// Marks `index` used.
    pub fn set_used(&mut self, index: u32) {
        let (byte, mask) = split(index);
        if let Some(byte) = self.bits.get_mut(byte) {
            *byte |= mask;
        }
    }

    /// Marks `index` free.
    pub fn set_free(&mut self, index: u32) {
        let (byte, mask) = split(index);
        if let Some(byte) = self.bits.get_mut(byte) {
            *byte &= !mask;
        }
    }

    /// Lowest free index below `limit`, if any.
    #[must_use]
    pub fn find_free(&self, limit: u32) -> Option<u32> {
        let whole = (limit / 8) as usize;
        for (position, &byte) in self.bits.iter().enumerate().take(whole) {
            if byte != 0xFF {
                // `!byte` has 1s where `byte` has clear bits, so its
                // lowest set bit is the lowest clear bit of `byte`.
                let bit = (!byte).trailing_zeros();
                let index = (position as u32) * 8 + bit;
                if index < limit {
                    return Some(index);
                }
                return None;
            }
        }
        // Tail bits above the last whole byte.
        let mut index = whole as u32 * 8;
        while index < limit && self.is_used(index) {
            index += 1;
        }
        (index < limit).then_some(index)
    }

    /// Number of set bits below `limit`.
    #[must_use]
    pub fn count_used(&self, limit: u32) -> u32 {
        let whole = (limit / 8) as usize;
        let mut count: u32 = 0;
        for &byte in &self.bits[..whole] {
            count = count.saturating_add(byte.count_ones());
        }
        let mut index = whole as u32 * 8;
        while index < limit {
            if self.is_used(index) {
                count += 1;
            }
            index += 1;
        }
        count
    }

    /// Number of clear bits below `limit`.
    #[must_use]
    pub fn count_free(&self, limit: u32) -> u32 {
        limit.saturating_sub(self.count_used(limit))
    }

    /// Copies a raw block into the bitmap.
    ///
    /// # Errors
    ///
    /// [`WreditError::InvalidOperation`] when `bytes` is not exactly
    /// one block long.
    pub fn from_bytes(bytes: &[u8]) -> WreditResult<Self> {
        if bytes.len() != BLOCK_SIZE {
            return Err(WreditError::InvalidOperation {
                what: "a bitmap occupies exactly one block",
            });
        }
        let mut bits = [0u8; BLOCK_SIZE];
        bits.copy_from_slice(bytes);
        Ok(Self { bits })
    }

    /// The raw block backing this bitmap, ready for the device.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8] {
        &self.bits
    }
}

impl Default for Bitmap {
    fn default() -> Self {
        Self::new()
    }
}

/// Byte position and mask of `index`.
#[inline]
fn split(index: u32) -> (usize, u8) {
    let byte = (index / 8) as usize;
    let mask = 1u8 << (index % 8);
    (byte, mask)
}

#[cfg(test)]
mod tests {
    use super::Bitmap;

    /// Justification: `unwrap` in tests is permitted by the project
    /// lint configuration (`allow-unwrap-in-tests`).

    #[test]
    fn set_clear_and_test_round_trip() {
        let mut bitmap = Bitmap::new();
        assert!(!bitmap.is_used(0));
        bitmap.set_used(0);
        bitmap.set_used(9);
        bitmap.set_used(32767);
        assert!(bitmap.is_used(0));
        assert!(bitmap.is_used(9));
        assert!(bitmap.is_used(32767));
        assert!(!bitmap.is_used(1));
        bitmap.set_free(9);
        assert!(!bitmap.is_used(9));
    }

    #[test]
    fn find_free_returns_the_lowest_clear_bit() {
        let mut bitmap = Bitmap::new();
        assert_eq!(bitmap.find_free(64), Some(0));
        bitmap.set_used(0);
        assert_eq!(bitmap.find_free(64), Some(1));
        bitmap.set_used(1);
        bitmap.set_used(2);
        assert_eq!(bitmap.find_free(64), Some(3));
        // Across a byte boundary.
        for index in 0..8 {
            bitmap.set_used(index);
        }
        assert_eq!(bitmap.find_free(64), Some(8));
    }

    #[test]
    fn find_free_respects_the_limit() {
        let mut bitmap = Bitmap::new();
        for index in 0..16 {
            bitmap.set_used(index);
        }
        assert_eq!(bitmap.find_free(16), None);
        assert_eq!(bitmap.find_free(17), Some(16));
        assert_eq!(bitmap.find_free(0), None);
    }

    #[test]
    fn counts_match_the_bits_set() {
        let mut bitmap = Bitmap::new();
        bitmap.set_used(3);
        bitmap.set_used(100);
        bitmap.set_used(200);
        assert_eq!(bitmap.count_used(256), 3);
        assert_eq!(bitmap.count_free(256), 253);
        assert_eq!(bitmap.count_used(100), 1);
    }

    #[test]
    fn raw_round_trip() {
        let mut bitmap = Bitmap::new();
        bitmap.set_used(5);
        let bytes = bitmap.as_bytes();
        let restored = Bitmap::from_bytes(bytes).unwrap();
        assert_eq!(restored, bitmap);
        assert!(Bitmap::from_bytes(&[0u8; 16]).is_err());
    }
}
