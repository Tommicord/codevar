//! Copyright 2026 Codevar
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

use std::alloc::{self, Layout};
use std::ptr::NonNull;

/// `enum WritableGapSize`
///
/// Defines the size for all gaps in a writable data structure
///
/// Is used to define the size of gap space of all cursor, must be fixed
/// size and aligned to 16-bytes to avoid cache misses
pub enum WritableGapSize {
    /// Gap size of 128 bytes (small)
    Gap128 = 128,

    /// Gap size of 256 bytes (medium)
    Gap256 = 256,

    /// Gap size of 512 bytes (larger)
    Gap512 = 512,

    /// Gap size of 1024 bytes (largest)
    Gap1024 = 1024,
}

impl WritableGapSize {
    /// Returns the gap size as a usize.
    pub const fn as_usize(self) -> usize {
        self as usize
    }
}

/// `pub struct WritableAlignedPtr<T, const ALIGN: usize>`
///
/// Aligned pointer for writable data structures.
///
/// This struct provides aligned memory allocation for performance-critical operations
/// ARM architectures require alignment to powers of 2 for optimal performance
///
/// # Type Parameters
///
/// * `T` - The type of elements stored in the aligned array
/// * `ALIGN` - The alignment requirement (default: 64 bytes)
///
/// # Safety
///
/// This struct uses unsafe blocks for raw memory manipulation to achieve maximum
/// performance. All safety invariants must be maintained by the caller.
///
/// # Example
///
/// ```
/// use codevar_core::edit::WritableAlignedPtr;
///
/// // Create new pointer of i32 with 64-byte alignment
/// let mut ptr: WritableAlignedPtr<i32, 64> = WritableAlignedPtr::new();
/// unsafe {
///     ptr.malloc(1024);
///     // Use the pointer here
///     // Rest of code...
///
///     // Free the memory
///     ptr.free();
/// }
/// ```
#[repr(C)]
pub struct WritableAlignedPtr<T, const ALIGN: usize = 64> {
    /// Aligned array pointer
    arr: Option<NonNull<T>>,
    /// Total allocated capacity in number of elements
    allocated: usize,
    /// Current count of elements in use
    count: usize,
}

impl<T, const ALIGN: usize> Default for WritableAlignedPtr<T, ALIGN> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(E0133)]
impl<T, const ALIGN: usize> WritableAlignedPtr<T, ALIGN> {
    /// Creates a new aligned pointer with no allocated memory
    ///
    /// # Safety
    ///
    /// The alignment must be a power of 2

    pub const fn new() -> Self {
        Self {
            arr: None,
            allocated: 0,
            count: 0,
        }
    }

    /// Allocates aligned memory for the specified offset
    ///
    /// This method ensures the allocation is properly aligned to the specified
    /// alignment value. If the current capacity is insufficient, it reallocates
    /// with a growth strategy of 1.5x the current size plus the new offset
    ///
    /// # Arguments
    ///
    /// * `offset` - The number of elements to allocate space for
    ///
    /// # Safety
    ///
    /// The caller must ensure that the alignment is a power of 2 and that
    /// the allocation succeeds. If allocation fails, the pointer is set to null
    #[allow(E0133)]

    pub unsafe fn malloc(&mut self, offset: usize) {
        let align = ALIGN;
        let aligned_offset = if offset % align != 0 {
            (offset / align).saturating_add(1).saturating_mul(align)
        } else {
            offset
        };

        if self.count.saturating_add(offset) > self.allocated {
            let new_size = self
                .count
                .saturating_add(self.count >> 1)
                .saturating_add(aligned_offset);
            let byte_size = new_size.saturating_mul(std::mem::size_of::<T>());
            let layout =
                Layout::from_size_align(byte_size, align).unwrap_or_else(|_| unsafe {
                    Layout::from_size_align_unchecked(byte_size, 64)
                });
            let new_ptr = alloc::alloc(layout);
            let new_ptr = NonNull::new(new_ptr as *mut T);

            if let Some(new_arr) = new_ptr {
                if let Some(old_arr) = self.arr {
                    std::ptr::copy_nonoverlapping(
                        old_arr.as_ptr(),
                        new_arr.as_ptr(),
                        self.count,
                    );
                    let old_layout = Layout::from_size_align_unchecked(
                        self.allocated.saturating_mul(std::mem::size_of::<T>()),
                        align,
                    );
                    alloc::dealloc(old_arr.as_ptr() as *mut u8, old_layout);
                }
                self.arr = Some(new_arr);
                self.allocated = new_size;
            } else {
                // Allocation failed, set pointer to null
                self.arr = None;
            }
        }
    }

    /// Frees the allocated aligned memory.
    ///
    /// # Safety
    ///
    /// The pointer must have been allocated with the same alignment value

    pub unsafe fn free(&mut self) {
        if let Some(arr) = self.arr {
            let layout = Layout::from_size_align_unchecked(
                self.allocated * std::mem::size_of::<T>(),
                ALIGN,
            );
            alloc::dealloc(arr.as_ptr() as *mut u8, layout);
            self.arr = None;
        }
        self.count = 0;
        self.allocated = 0;
    }

    /// Returns a reference to the element at the specified index, if within bounds
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to access
    ///
    /// # Returns
    ///
    /// * `Some(&T)` if the index is within bounds
    /// * `None` if the index is out of bounds
    ///
    /// # Safety
    ///
    /// The pointer must be valid and the index must be within bounds.

    pub unsafe fn get(&self, index: usize) -> Option<&T> {
        if index >= self.count {
            return None;
        }
        self.arr.map(|arr| &*arr.as_ptr().add(index))
    }

    /// Returns a mutable reference to the element at the specified index, if within bounds
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to access
    ///
    /// # Returns
    ///
    /// * `Some(&mut T)` if the index is within bounds
    /// * `None` if the index is out of bounds
    ///
    /// # Safety
    ///
    /// The pointer must be valid and the index must be within bounds

    pub unsafe fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.count {
            return None;
        }
        self.arr.map(|arr| &mut *arr.as_ptr().add(index))
    }

    /// Returns the element at the specified index without bounds checking
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to access
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds and the pointer is valid

    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        let arr = self.arr.unwrap_or_else(|| {
            // Return a dangling pointer if no memory is allocated
            // This is a safety violation but prevents panic
            NonNull::dangling()
        });
        &*arr.as_ptr().add(index)
    }

    /// Returns a mutable reference to the element at the specified index without bounds checking
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to access
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds and the pointer is valid

    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        let arr = self.arr.unwrap_or_else(|| {
            // Return a dangling pointer if no memory is allocated
            // This is a safety violation but prevents panic
            std::ptr::NonNull::dangling()
        });
        &mut *arr.as_ptr().add(index)
    }

    /// Returns the raw pointer to the array.
    ///
    /// # Safety
    ///
    /// The pointer may be null if no memory has been allocated

    pub fn as_ptr(&self) -> *const T {
        self.arr.map_or(std::ptr::null(), |arr| arr.as_ptr())
    }

    /// Returns the mutable raw pointer to the array.
    ///
    /// # Safety
    ///
    /// The pointer may be null if no memory has been allocated

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.arr.map_or(std::ptr::null_mut(), |arr| arr.as_ptr())
    }

    /// Returns the number of elements currently in use.

    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the total allocated capacity in number of elements

    pub fn allocated(&self) -> usize {
        self.allocated
    }

    /// Sets the count of elements in use.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the new count does not exceed the allocated capacity

    pub unsafe fn set_count(&mut self, count: usize) {
        self.count = count;
    }

    /// Checks if the pointer has valid allocated memory
    ///
    /// # Returns
    ///
    /// * `true` if the pointer is non-null and count > 0
    /// * `false` otherwise

    pub fn is_valid(&self) -> bool {
        self.arr.is_some() && self.count > 0
    }
}

impl<T, const ALIGN: usize> Drop for WritableAlignedPtr<T, ALIGN> {
    fn drop(&mut self) {
        unsafe {
            self.free();
        }
    }
}

impl<T, const ALIGN: usize> std::ops::Index<usize> for WritableAlignedPtr<T, ALIGN> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe {
            debug_assert!(index < self.count, "Index out of bounds");
            self.get_unchecked(index)
        }
    }
}

impl<T, const ALIGN: usize> std::ops::IndexMut<usize> for WritableAlignedPtr<T, ALIGN> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe {
            debug_assert!(index < self.count, "Index out of bounds");
            self.get_unchecked_mut(index)
        }
    }
}
