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

//! Packed element format and the sparse aligned heap buffer of the
//! edit engine.
//!
//! A Wredit document is an array of packed 32-bit *raw* elements.
//! Bits `0..28` hold the code point ([`CHAR_MASK`]) and bits `28..32`
//! hold an element *type* ([`TYPE_MASK`], shifted by [`SHIFT_TYPE`]).
//! Two types are meaningful to the flush layer: [`T_COMMON`] for an
//! ordinary committed character and [`T_MERGE`] for an element that
//! overwrites the raw element that follows it when the staged run is
//! committed (see [`crate::edit::flush`]).
//!
//! [`WritableAlignedPtr`] is the sparse growable heap buffer backing
//! both the raw document and the shared staging arena. It starts
//! empty (no allocation at all), grows geometrically through
//! `alloc::alloc`, and uses checked arithmetic everywhere so a
//! hostile capacity request can never wrap.

use alloc::alloc::{Layout, alloc, dealloc};
use core::mem::align_of;
use core::ptr::NonNull;

use codevar_logger::log_error;

use crate::error::{WreditError, WreditResult};

/// Shift of the element type field inside a packed raw element.
pub const SHIFT_TYPE: u32 = 28;

/// Mask of the element type field (already shifted).
///
/// [`make_packed`] shifts a type value by [`SHIFT_TYPE`] and masks it
/// with this constant, so only the low 4 bits of a type survive.
pub const TYPE_MASK: u32 = 0xF;

/// Mask of the character field inside a packed raw element.
pub const CHAR_MASK: u32 = 0xFF_FFFF;

/// Element type: an ordinary committed character.
pub const T_COMMON: u32 = 0xE;

/// Element type: on flush, overwrites the raw element that follows
/// the insertion point instead of inserting a new element.
pub const T_MERGE: u32 = 0xD;

/// Unknown (unset) document encoding.
pub const ENC_UNKNOWN: u16 = 0x0;
/// UTF-8 document encoding.
pub const ENC_UTF8: u16 = 0x1;
/// UTF-16 little-endian document encoding.
pub const ENC_UTF16LE: u16 = 0x2;
/// UTF-16 big-endian document encoding.
pub const ENC_UTF16BE: u16 = 0x3;
/// Single-byte document encoding (Latin-1 style).
pub const ENC_SINGLE_BYTE: u16 = 0x4;

/// Capacity of the document name in bytes (without the terminator).
pub const NAME_LEN: usize = 0xFF;

/// Packs a character and an element type into one raw element.
///
/// Both inputs are truncated: the character keeps its low 24 bits,
/// the type its low 4 bits.
///
/// # Examples
///
/// ```
/// use codevar_wredit::edit::{T_COMMON, make_packed, packed_char, packed_type};
///
/// let element = make_packed(b'a' as u32, T_COMMON);
/// assert_eq!(packed_char(element), b'a' as u32);
/// assert_eq!(packed_type(element), T_COMMON);
/// ```
#[must_use]
#[inline]
pub const fn make_packed(character: u32, element_type: u32) -> u32 {
    ((element_type & TYPE_MASK) << SHIFT_TYPE) | (character & CHAR_MASK)
}

/// Extracts the character field of a packed raw element.
#[must_use]
#[inline]
pub const fn packed_char(element: u32) -> u32 {
    element & CHAR_MASK
}

/// Extracts the element type of a packed raw element.
#[must_use]
#[inline]
pub const fn packed_type(element: u32) -> u32 {
    (element >> SHIFT_TYPE) & TYPE_MASK
}

/// Standard gap sizes offered to document layouts.
///
/// The value doubles as the `GAP_SIZE` const generic of the engine's
/// gap structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritableGapSize {
    /// A 128 element gap: smallest footprint, most frequent flushes.
    Gap128,
    /// A 256 element gap: the default for interactive editing.
    Gap256,
    /// A 512 element gap.
    Gap512,
    /// A 1024 element gap: batch/paste oriented.
    Gap1024,
}

impl WritableGapSize {
    /// The gap size in elements.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        match self {
            Self::Gap128 => 128,
            Self::Gap256 => 256,
            Self::Gap512 => 512,
            Self::Gap1024 => 1024,
        }
    }
}

impl From<WritableGapSize> for usize {
    fn from(size: WritableGapSize) -> usize {
        size.as_usize()
    }
}

/// Smallest number of elements allocated on the first growth.
const MIN_CAPACITY: usize = 32;

/// A sparse, growable, alignment-honouring heap buffer of `T`.
///
/// The buffer starts *empty*: no memory is allocated until
/// [`Self::ensure_capacity`] is called, and capacity grows
/// geometrically (doubling, with a floor of [`MIN_CAPACITY`]) so
/// appends stay amortized O(1). Only the first [`Self::count`]
/// elements are initialized; writing element `i` requires
/// `i < count` or a preceding [`Self::set_count`] that covers it.
///
/// The allocation is released on drop through `alloc::dealloc`,
/// which in a Codevar process resolves to the launcher-installed
/// TLSF allocator.
pub struct WritableAlignedPtr<T: Copy, const ALIGN: usize> {
    /// Start of the heap allocation; `None` while the buffer is empty.
    buff: Option<NonNull<T>>,
    /// Number of *initialized* elements, always `<= allocated`.
    count: usize,
    /// Number of elements the allocation can hold.
    allocated: usize,
}

impl<T: Copy, const ALIGN: usize> WritableAlignedPtr<T, ALIGN> {
    /// Creates an empty buffer: no heap allocation happens here.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buff: None,
            count: 0,
            allocated: 0,
        }
    }

    /// Read-only pointer to element `0`, or a dangling (never
    /// dereferenceable) pointer while the buffer is empty.
    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        match self.buff {
            Some(ptr) => ptr.as_ptr(),
            None => core::ptr::NonNull::dangling().as_ptr(),
        }
    }

    /// Mutable pointer to element `0`, or a dangling (never
    /// dereferenceable) pointer while the buffer is empty.
    #[must_use]
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        match self.buff {
            Some(ptr) => ptr.as_ptr(),
            None => core::ptr::NonNull::dangling().as_ptr(),
        }
    }

    /// Number of initialized elements.
    #[must_use]
    #[inline]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Number of elements the buffer can hold without growing.
    #[must_use]
    #[inline]
    pub const fn allocated(&self) -> usize {
        self.allocated
    }

    /// Sets the number of initialized elements.
    ///
    /// Clamps `count` to the allocated capacity so the buffer can
    /// never claim to hold more elements than it allocated; the
    /// clamped value is returned.
    #[inline]
    pub fn set_count(&mut self, count: usize) -> usize {
        self.count = count.min(self.allocated);
        self.count
    }

    /// A shared reference to the initialized elements.
    #[must_use]
    #[inline]
    pub fn slice(&self) -> &[T] {
        // Safety: `buff` (when present) points to `allocated`
        // initialized-or-readable elements and `count <= allocated`,
        // so `..count` is fully initialized and in bounds.
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.count) }
    }

    /// A mutable reference to the initialized elements.
    #[must_use]
    #[inline]
    pub fn slice_mut(&mut self) -> &mut [T] {
        let count = self.count;
        // Safety: as in `slice`, plus `&mut self` guarantees exclusive
        // access to the whole buffer.
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), count) }
    }

    /// Grows the buffer so it can hold at least `capacity` elements.
    ///
    /// Growth is geometric: the new capacity is the largest of
    /// `capacity`, twice the current capacity and [`MIN_CAPACITY`],
    /// all computed with checked arithmetic. Reallocating copies the
    /// initialized prefix and frees the old allocation.
    ///
    /// # Errors
    ///
    /// * [`WreditError::LimitExceeded`] when `capacity` cannot be
    ///   expressed as a memory layout (overflow).
    /// * [`WreditError::CapacityExhausted`] when the allocator
    ///   refuses the request; the buffer is left untouched.
    pub fn ensure_capacity(&mut self, capacity: usize) -> WreditResult<()> {
        if capacity <= self.allocated {
            return Ok(());
        }
        let doubled = self.allocated.saturating_mul(2);
        let new_capacity = capacity.max(doubled).max(MIN_CAPACITY);
        let new_buff = ensure_merge_buffer::<T>(new_capacity, ALIGN.max(align_of::<T>()))?;
        // Copy the initialized prefix into the fresh allocation.
        // Safety: `new_buff` holds `new_capacity >= count` elements of
        // `T`; both regions are in bounds and non-overlapping.
        let byte_len = self
            .count
            .saturating_mul(core::mem::size_of::<T>());
        if byte_len != 0 {
            unsafe {
                crate::arch::memcpy(
                    new_buff.as_ptr().cast::<u8>(),
                    self.as_ptr().cast::<u8>(),
                    byte_len,
                );
            }
        }
        if let Some(old) = self.buff {
            // Safety: `old` was allocated by `ensure_merge_buffer`
            // with exactly this layout, and nothing aliases it now.
            unsafe {
                let layout = match Layout::array::<T>(self.allocated) {
                    Ok(layout) => layout,
                    Err(_) => {
                        // Layout succeeded at allocation time for the
                        // same element count, so this cannot fail; if
                        // it somehow does, leak rather than free with
                        // a wrong layout.
                        log_error!("writable buffer layout mismatch while shrinking; leaking old allocation");
                        self.buff = Some(new_buff);
                        self.allocated = new_capacity;
                        return Ok(());
                    }
                };
                dealloc(old.as_ptr().cast::<u8>(), layout);
            }
        }
        self.buff = Some(new_buff);
        self.allocated = new_capacity;
        Ok(())
    }

    /// Marks `count` elements as initialized without growing.
    ///
    /// The caller guarantees that elements `0..count` have been fully
    /// written; the count is still clamped to the capacity.
    #[inline]
    pub fn init_count(&mut self, count: usize) {
        let _ = self.set_count(count);
    }
}

impl<T: Copy, const ALIGN: usize> Default for WritableAlignedPtr<T, ALIGN> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const ALIGN: usize> Drop for WritableAlignedPtr<T, ALIGN> {
    fn drop(&mut self) {
        if let Some(buff) = self.buff {
            // Safety: `buff` was allocated by `ensure_merge_buffer`
            // with `Layout::array::<T>(self.allocated)` and this
            // aligned layout; nothing else owns it.
            if let Ok(layout) = Layout::array::<T>(self.allocated) {
                unsafe {
                    dealloc(buff.as_ptr().cast::<u8>(), layout);
                }
            }
        }
    }
}

/// Allocates a fresh uninitialized heap array of `capacity`
/// elements aligned to `align`.
///
/// This is the single allocation funnel of the edit engine. Unlike
/// `alloc::alloc::handle_alloc_error`, a failed allocation is *not*
/// fatal: it is logged through `codevar_logger` and an error is
/// returned so the operation can report
/// [`WreditError::CapacityExhausted`] and leave the document intact.
///
/// # Errors
///
/// [`WreditError::CapacityExhausted`] when the layout overflows or
/// the global allocator refuses the request.
pub fn ensure_merge_buffer<T: Copy>(capacity: usize, align: usize) -> WreditResult<NonNull<T>> {
    if capacity == 0 {
        return Err(WreditError::InvalidGapSize { size: 0 });
    }
    let layout = Layout::array::<T>(capacity)
        .and_then(|l| l.align_to(align.max(align_of::<T>()).max(1)))
        .map_err(|_| {
            log_error!("writable buffer layout overflow");
            WreditError::LimitExceeded {
                what: "writable buffer layout",
            }
        })?;
    // SAFETY: layout has non-zero size (capacity > 0 and T is not a
    // zero-sized type in engine instantiations); align is a power of
    // two guaranteed by `Layout`.
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        log_error!("writable buffer allocation failed");
        return Err(WreditError::CapacityExhausted {
            requested: capacity,
            available: 0,
        });
    }
    NonNull::new(ptr.cast::<T>()).ok_or(WreditError::CapacityExhausted {
        requested: capacity,
        available: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_helpers_roundtrip_and_truncate() {
        let element = make_packed(0x1F_600, T_MERGE);
        assert_eq!(packed_char(element), 0x1F_600);
        assert_eq!(packed_type(element), T_MERGE);
        // High bits of both fields are truncated, never shifted into
        // each other: only the low 24 bits of the character survive.
        let truncated = make_packed(0xFF_FFFF + 7, 0xF_F + T_COMMON);
        assert_eq!(packed_char(truncated), 6);
        assert_eq!(packed_type(truncated), (0xF_F + T_COMMON) & TYPE_MASK);
    }

    #[test]
    fn gap_sizes_match_their_names() {
        assert_eq!(WritableGapSize::Gap128.as_usize(), 128);
        assert_eq!(WritableGapSize::Gap256.as_usize(), 256);
        assert_eq!(WritableGapSize::Gap512.as_usize(), 512);
        assert_eq!(WritableGapSize::Gap1024.as_usize(), 1024);
        assert_eq!(usize::from(WritableGapSize::Gap1024), 1024);
    }

    #[test]
    fn aligned_ptr_starts_empty_and_grows_geometrically() {
        let mut buffer: WritableAlignedPtr<u32, 64> = WritableAlignedPtr::new();
        assert_eq!(buffer.allocated(), 0);
        assert_eq!(buffer.count(), 0);
        assert!(buffer.slice().is_empty());

        buffer.ensure_capacity(1).unwrap();
        assert!(buffer.allocated() >= 1);
        let after_first = buffer.allocated();

        buffer.ensure_capacity(after_first + 1).unwrap();
        assert!(buffer.allocated() >= (after_first + 1).max(after_first * 2));
    }

    #[test]
    fn aligned_ptr_preserves_initialized_prefix_across_growth() {
        let mut buffer: WritableAlignedPtr<u32, 64> = WritableAlignedPtr::new();
        buffer.ensure_capacity(4).unwrap();
        buffer.set_count(4);
        buffer.slice_mut().copy_from_slice(&[1, 2, 3, 4]);
        buffer.ensure_capacity(1024).unwrap();
        assert_eq!(buffer.slice(), &[1, 2, 3, 4]);
        buffer.set_count(1024);
        assert_eq!(buffer.slice().len(), 1024);
    }

    #[test]
    fn set_count_is_clamped_to_capacity() {
        let mut buffer: WritableAlignedPtr<u32, 64> = WritableAlignedPtr::new();
        assert_eq!(buffer.set_count(10), 0);
        buffer.ensure_capacity(8).unwrap();
        // The first growth floors at `MIN_CAPACITY`, so a count
        // inside the allocation is accepted unchanged...
        assert_eq!(buffer.set_count(10), 10);
        // ...and one beyond it is clamped to the allocation.
        assert_eq!(buffer.set_count(64), buffer.allocated());
    }

    #[test]
    fn overflow_capacity_requests_are_rejected_not_wrapped() {
        let mut buffer: WritableAlignedPtr<u64, 64> = WritableAlignedPtr::new();
        let result = buffer.ensure_capacity(usize::MAX / 4);
        assert!(matches!(result, Err(WreditError::LimitExceeded { .. })));
        // The buffer is unchanged after the failure.
        assert_eq!(buffer.allocated(), 0);
        assert_eq!(buffer.count(), 0);
    }

    #[test]
    fn zero_sized_requests_are_rejected() {
        assert!(matches!(
            ensure_merge_buffer::<u32>(0, 64),
            Err(WreditError::InvalidGapSize { size: 0 })
        ));
    }

    #[test]
    fn size_of_element_is_the_packed_word() {
        assert_eq!(core::mem::size_of::<u32>(), 4);
    }
}
