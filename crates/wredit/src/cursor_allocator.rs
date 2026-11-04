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

//! Bitmap cursor allocator.
//!
//! Allocates cursor ids from a fixed bitmap, the same way a disk file
//! system allocates inodes from its inode bitmap. The bitmap is a
//! single fixed-size array on the stack — no heap allocation — and
//! supports at most [`MAX_CURSORS`] (0x2000 = 8192) cursors, exactly
//! one bit per id.
//!
//! Bit `i` of byte `i / 8` being set means cursor id `i` is in use.
//! Allocation scans for the lowest clear bit; freeing clears it.

use crate::error::{WreditError, WreditResult};

/// Maximum number of cursors.
pub const MAX_CURSORS: usize = 0x2000;

/// Number of bytes the bitmap occupies.
const BITMAP_BYTES: usize = MAX_CURSORS / 8;

/// Allocates and frees cursor ids from a fixed bitmap.
///
/// # Examples
///
/// ```
/// use codevar_wredit::cursor_allocator::CursorAllocator;
///
/// let mut alloc = CursorAllocator::new();
/// let a = alloc.alloc().unwrap();
/// let b = alloc.alloc().unwrap();
/// assert_ne!(a, b);
/// assert!(alloc.is_used(a));
/// assert!(alloc.is_used(b));
/// alloc.free(a);
/// assert!(!alloc.is_used(a));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CursorAllocator {
    /// One bit per cursor id; bit `i % 8` of byte `i / 8`.
    bits: [u8; BITMAP_BYTES],
    /// Number of ids currently allocated.
    count: usize,
}

impl Default for CursorAllocator {
    fn default() -> Self {
        Self {
            bits: [0; BITMAP_BYTES],
            count: 0,
        }
    }
}

impl CursorAllocator {
    /// An empty allocator: no cursors in use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bits: [0; BITMAP_BYTES],
            count: 0,
        }
    }

    /// Maximum number of cursors this allocator can hand out.
    #[must_use]
    pub const fn capacity() -> usize {
        MAX_CURSORS
    }

    /// Number of cursors currently allocated.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// `true` when no cursor id is allocated.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// `true` when every cursor id is allocated.
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.count == MAX_CURSORS
    }

    /// Allocates the lowest free cursor id.
    ///
    /// # Errors
    ///
    /// [`WreditError::DiskFull`] when every cursor id is in use.
    pub fn alloc(&mut self) -> WreditResult<usize> {
        if self.is_full() {
            return Err(WreditError::DiskFull { free: 0 });
        }
        let id = self
            .find_free()
            .ok_or(WreditError::InvalidOperation {
                what: "allocator reported free cursors but none could be found",
            })?;
        self.set_used(id);
        Ok(id)
    }

    /// Frees `cursor_id`. Returns `false` when the id was not in use.
    pub fn free(&mut self, cursor_id: usize) -> bool {
        if cursor_id >= MAX_CURSORS {
            return false;
        }
        let (byte, mask) = split(cursor_id);
        if self.bits[byte] & mask != 0 {
            self.bits[byte] &= !mask;
            self.count -= 1;
            true
        } else {
            false
        }
    }

    /// `true` when `cursor_id` is currently allocated.
    #[must_use]
    pub fn is_used(&self, cursor_id: usize) -> bool {
        if cursor_id >= MAX_CURSORS {
            return false;
        }
        let (byte, mask) = split(cursor_id);
        self.bits[byte] & mask != 0
    }

    /// Marks every cursor id as free.
    pub fn clear(&mut self) {
        self.bits = [0; BITMAP_BYTES];
        self.count = 0;
    }

    /// Lowest free cursor id, if any.
    fn find_free(&self) -> Option<usize> {
        for (position, &byte) in self.bits.iter().enumerate() {
            if byte != 0xFF {
                // `!byte` has 1s where `byte` has clear bits, so its
                // lowest set bit is the lowest clear bit of `byte`.
                let bit = (!byte).trailing_zeros() as usize;
                let id = position * 8 + bit;
                if id < MAX_CURSORS {
                    return Some(id);
                }
            }
        }
        None
    }

    /// Marks `cursor_id` used. Caller must ensure the id is free.
    fn set_used(&mut self, cursor_id: usize) {
        let (byte, mask) = split(cursor_id);
        self.bits[byte] |= mask;
        self.count += 1;
    }
}

/// Byte position and mask of `cursor_id`.
#[inline]
fn split(cursor_id: usize) -> (usize, u8) {
    let byte = cursor_id / 8;
    let mask = 1u8 << (cursor_id % 8);
    (byte, mask)
}

#[cfg(test)]
mod tests {
    use super::{CursorAllocator, MAX_CURSORS};

    /// Justification: `unwrap` in tests is permitted by the project
    /// lint configuration (`allow-unwrap-in-tests`).

    #[test]
    fn alloc_returns_the_lowest_free_id() {
        let mut alloc = CursorAllocator::new();
        assert_eq!(alloc.alloc().unwrap(), 0);
        assert_eq!(alloc.alloc().unwrap(), 1);
        alloc.free(0);
        assert_eq!(alloc.alloc().unwrap(), 0);
    }

    #[test]
    fn free_clears_the_bit() {
        let mut alloc = CursorAllocator::new();
        alloc.alloc().unwrap();
        assert!(alloc.is_used(0));
        assert!(alloc.free(0));
        assert!(!alloc.is_used(0));
        assert!(!alloc.free(0)); // already free
    }

    #[test]
    fn out_of_range_ids_are_never_used() {
        let mut alloc = CursorAllocator::new();
        assert!(!alloc.is_used(MAX_CURSORS));
        assert!(!alloc.free(MAX_CURSORS));
    }

    #[test]
    fn alloc_exhausts_at_max_cursors() {
        let mut alloc = CursorAllocator::new();
        for _ in 0..MAX_CURSORS {
            assert!(alloc.alloc().is_ok());
        }
        assert!(alloc.is_full());
        assert!(matches!(
            alloc.alloc(),
            Err(crate::error::WreditError::DiskFull { free: 0 })
        ));
        // Freeing one makes room again.
        alloc.free(0);
        assert!(alloc.alloc().is_ok());
    }

    #[test]
    fn bitmap_stays_near_one_kib() {
        // 8192 bits = 1024 bytes plus a small count field; the whole
        // allocator must fit on the stack and never touch the heap.
        assert!(core::mem::size_of::<CursorAllocator>() <= 2048);
    }
}
