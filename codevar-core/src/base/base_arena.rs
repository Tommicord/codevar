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
use std::cell::UnsafeCell;
use std::ptr::NonNull;

/// Arena allocator for fast allocation of objects with a fixed lifetime.
///
/// # Safety
///
/// The arena allocator uses unsafe blocks for raw pointer manipulation.
/// All allocated objects must be dropped before the arena is dropped.
#[repr(C)]
pub struct Allocator {
    /// Pointer to the start of the arena memory
    ptr: NonNull<u8>,
    /// Current position in the arena
    pos: UnsafeCell<usize>,
    /// Total capacity of the arena
    capacity: usize,
}

impl Allocator {
    /// Creates a new arena with the specified capacity.
    ///
    /// # Safety
    ///
    /// The capacity must be non-zero and the allocation must succeed.

    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Arena capacity must be non-zero");

        let layout = Layout::from_size_align(capacity, 8).unwrap_or_else(|_| {
            Layout::from_size_align(capacity.next_power_of_two(), 8).unwrap_or_else(
                |_| {
                    alloc::handle_alloc_error(Layout::new::<u8>());
                },
            )
        });
        let ptr = unsafe { alloc::alloc(layout) };

        let ptr = NonNull::new(ptr).unwrap_or_else(|| {
            alloc::handle_alloc_error(layout);
            unreachable!()
        });

        Self {
            ptr,
            pos: UnsafeCell::new(0),
            capacity,
        }
    }

    /// Allocates space for a value of type T and returns a pointer to it.
    ///
    /// # Safety
    ///
    /// The caller must ensure the arena has enough capacity and that the
    /// pointer is used correctly.

    pub unsafe fn allocate<T>(&self) -> *mut T {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        let current_pos = *self.pos.get();
        let aligned_pos = (current_pos + align - 1) & !(align - 1);

        assert!(aligned_pos + size <= self.capacity, "Arena out of capacity");
        let ptr = self.ptr.as_ptr().add(aligned_pos) as *mut T;
        *self.pos.get() = aligned_pos + size;

        ptr
    }

    /// Allocates and initializes a value in the arena.
    ///
    /// # Safety
    ///
    /// The arena must have enough capacity.

    pub fn alloc<T>(&self, value: T) -> &mut T {
        unsafe {
            let ptr = self.allocate::<T>();
            ptr.write(value);
            &mut *ptr
        }
    }

    /// Pushes a value to the arena and returns a reference to it.
    ///
    /// # Safety
    ///
    /// The arena must have enough capacity.

    pub fn push<T>(&self, value: T) -> &mut T {
        self.alloc(value)
    }

    /// Returns the current position in the arena.

    pub fn position(&self) -> usize {
        unsafe { *self.pos.get() }
    }

    /// Returns the total capacity of the arena.

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Resets the arena to its initial state, clearing all allocations.
    ///
    /// # Safety
    ///
    /// All previously allocated objects must have been dropped before calling this.

    pub unsafe fn reset(&self) {
        *self.pos.get() = 0;
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        // SAFETY: We own the memory and it was allocated with alloc::alloc
        unsafe {
            alloc::dealloc(
                self.ptr.as_ptr(),
                Layout::from_size_align_unchecked(self.capacity, 8),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Allocator;

    fn arena_creation() {
        let arena = Allocator::new(1024);
        assert_eq!(arena.capacity(), 1024);
        assert_eq!(arena.position(), 0);
    }

    fn arena_alloc() {
        let arena = Allocator::new(1024);
        let value = arena.alloc(42u32);
        assert_eq!(*value, 42);
    }

    fn arena_push() {
        let arena = Allocator::new(1024);
        let value = arena.push(100i32);
        assert_eq!(*value, 100);
    }

    fn arena_multiple_allocs() {
        let arena = Allocator::new(1024);
        arena.alloc(1u8);
        arena.alloc(2u16);
        arena.alloc(3u32);
        assert!(arena.position() > 0);
    }

    fn arena_reset() {
        let arena = Allocator::new(1024);
        arena.alloc(42u32);
        unsafe {
            arena.reset();
        }
        assert_eq!(arena.position(), 0);
    }
}
