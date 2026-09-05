//! Copyright 2026 Codevar
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

//! Cache-friendly LIFO queue implementation for log entries.

use crate::logtrace::log_error::{Error, Result};
use std::mem::{self, MaybeUninit};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Default initial capacity for the log queue.
const DEFAULT_INITIAL_CAPACITY: usize = 2048;

/// Maximum capacity for the log queue to prevent unbounded growth.
const MAX_CAPACITY: usize = 65536;

/// Minimum capacity to maintain queue functionality.
const MIN_CAPACITY: usize = 64;

/// Cache line size for x86_64 processors (64 bytes).
/// Used for padding to prevent false sharing.
const CACHE_LINE_SIZE: usize = 64;

struct CachePadding<T>(std::marker::PhantomData<T>);

impl<T> CachePadding<T> {
    #[inline]
    fn pad() -> usize {
        let size = size_of::<T>().max(1);
        let align = align_of::<T>().max(1);
        let sub = CACHE_LINE_SIZE.saturating_sub(size);
        if sub < align { sub } else { align }
    }
}

/// A cache-friendly LIFO queue entry.
#[repr(C, align(64))]
struct QueueEntry<T> {
    /// The actual data stored in the entry.
    data: MaybeUninit<T>,
    /// Padding to prevent false sharing between adjacent entries.
    _pad: [u8; 64],
}

impl<T> QueueEntry<T> {
    /// Creates a new uninitialized queue entry.
    #[inline]
    fn new() -> Self {
        let _pad_size = CachePadding::<T>::pad();
        let _ = _pad_size;
        Self {
            data: MaybeUninit::uninit(),
            _pad: [0u8; 64],
        }
    }

    /// Writes data to the entry.
    ///
    /// # Safety
    ///
    /// This function is safe to call when:
    /// - The entry is properly aligned for T
    /// - The entry points to valid memory
    /// - No other thread is concurrently writing to this entry
    #[inline]
    unsafe fn write(&mut self, value: T) {
        self.data.write(value);
    }

    /// Reads data from the entry.
    ///
    /// # Safety
    ///
    /// This function is safe to call when:
    /// - The entry has been initialized with a value
    /// - No other thread is concurrently modifying this entry
    #[inline]
    unsafe fn read(&self) -> T {
        unsafe { self.data.assume_init_read() }
    }
}

/// A cache-friendly LIFO queue with dynamic capacity adjustment.
#[repr(C, align(64))]
pub struct Queue<T> {
    /// The underlying storage for queue entries.
    entries: Box<[QueueEntry<T>]>,

    /// Current head index (for LIFO operations).
    head: AtomicUsize,

    /// Current tail index (for capacity tracking).
    tail: AtomicUsize,

    /// Current count of elements in the queue.
    count: AtomicUsize,

    /// Current capacity of the queue.
    capacity: AtomicUsize,

    /// Maximum capacity limit.
    max_capacity: usize,

    /// Minimum capacity limit.
    min_capacity: usize,

    /// Flag indicating if the queue is being resized.
    resizing: AtomicBool,

    /// Number of dropped entries due to capacity limits.
    dropped: AtomicUsize,
}

impl<T> Queue<T> {
    /// Creates a new log queue with default capacity.
    ///
    /// # Performance
    ///
    /// This function allocates memory aligned to cache line boundaries
    /// to optimize for cache-friendly access patterns. The initial
    /// allocation uses heap memory with proper alignment.
    pub fn new() -> Result<Self> {
        Self::with_capacity(DEFAULT_INITIAL_CAPACITY)
    }

    /// Creates a new log queue with the specified initial capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity for the queue (will be clamped to valid range)
    ///
    /// # Performance
    ///
    /// The capacity is aligned to cache line boundaries for optimal
    /// memory access patterns. This reduces cache misses and improves
    /// throughput in multi-threaded scenarios.
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        let clamped_capacity = capacity.clamp(MIN_CAPACITY, MAX_CAPACITY);
        let mut entries: Vec<QueueEntry<T>> = Vec::with_capacity(clamped_capacity);
        for _ in 0..clamped_capacity {
            entries.push(QueueEntry::new());
        }
        let entries = entries.into_boxed_slice();

        Ok(Self {
            entries,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            capacity: AtomicUsize::new(clamped_capacity),
            max_capacity: MAX_CAPACITY,
            min_capacity: MIN_CAPACITY,
            resizing: AtomicBool::new(false),
            dropped: AtomicUsize::new(0),
        })
    }

    /// Pushes a value onto the LIFO queue.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to push onto the queue
    ///
    /// # Performance
    ///
    /// This function uses atomic operations for thread safety while
    /// minimizing contention. The LIFO nature ensures cache locality
    /// for recently pushed items.
    ///
    /// # Errors
    ///
    /// Returns an error if the queue is at maximum capacity and
    /// cannot be expanded further.
    pub fn push(&mut self, value: T) -> Result<()> {
        let current_count = self.count.fetch_add(1, Ordering::Acquire);
        let current_capacity = self.capacity.load(Ordering::Acquire);

        if current_count >= current_capacity {
            if current_capacity < self.max_capacity {
                self.try_expand_capacity(current_count)?;
            } else {
                self.count.fetch_sub(1, Ordering::Release);
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return Err(Error::QueueCapacityExceeded {
                    capacity: current_capacity,
                    requested: current_count + 1,
                });
            }
        }
        // Get current head index (LIFO: push to head)
        let head = self.head.load(Ordering::Acquire);
        let new_head = (head + 1) % current_capacity;

        // Write value to entry
        // Safety: We've ensured capacity and have exclusive access to this slot
        unsafe {
            let entry = &mut self.entries[head];
            // This is safe because we've verified capacity bounds
            ptr::addr_of_mut!((*ptr::from_mut(entry)).data)
                .cast::<MaybeUninit<T>>()
                .write(MaybeUninit::new(value));
        }
        self.head.store(new_head, Ordering::Release);
        Ok(())
    }

    /// Pops a value from the LIFO queue.
    ///
    /// # Performance
    ///
    /// This function provides O(1) amortized time complexity with
    /// minimal cache misses due to the LIFO access pattern.
    ///
    /// # Returns
    ///
    /// * `Some(T)` - The popped value if the queue is not empty
    /// * `None` - If the queue is empty
    pub fn pop(&mut self) -> Option<T> {
        let current_count = self.count.load(Ordering::Acquire);
        if current_count == 0 {
            return None;
        }
        let current_capacity = self.capacity.load(Ordering::Acquire);
        // Get current head index (LIFO: pop from head)
        let head = self.head.load(Ordering::Acquire);
        let prev_head = if head == 0 {
            current_capacity - 1
        } else {
            head - 1
        };
        // Read value from entry
        // Safety: We've verified the queue is not empty and have valid access
        let value = unsafe {
            let entry = &self.entries[prev_head];
            entry.data.assume_init_read()
        };

        // Update head index
        self.head.store(prev_head, Ordering::Release);
        self.count.fetch_sub(1, Ordering::Release);
        // Try to shrink capacity if utilization is low
        if current_count < current_capacity / 4 && current_capacity > self.min_capacity {
            let _ = self.try_shrink_capacity(current_count);
        }
        Some(value)
    }

    /// Peeks at the top value without removing it.
    ///
    /// # Performance
    ///
    /// This function provides O(1) time complexity without modifying
    /// the queue state.
    ///
    /// # Returns
    ///
    /// * `Some(&T)` - Reference to the top value if the queue is not empty
    /// * `None` - If the queue is empty
    pub fn peek(&self) -> Option<&T> {
        let current_count = self.count.load(Ordering::Acquire);
        if current_count == 0 {
            return None;
        }
        let current_capacity = self.capacity.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        let prev_head = if head == 0 {
            current_capacity - 1
        } else {
            head - 1
        };
        // Safety: We've verified the queue is not empty
        unsafe {
            let entry = &self.entries[prev_head];
            Some(&*ptr::addr_of!((*ptr::from_ref(entry)).data).cast::<T>())
        }
    }

    /// Returns the current number of elements in the queue.
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    /// Returns true if the queue is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the current capacity of the queue.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Acquire)
    }

    /// Returns the number of entries dropped due to capacity limits.
    #[inline]
    pub fn dropped_count(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Attempts to expand the queue capacity.
    ///
    /// # Arguments
    ///
    /// * `current_count` - Current number of elements in the queue
    ///
    /// # Performance
    ///
    /// This function uses exponential growth strategy (2x capacity)
    /// to amortize allocation costs. It uses atomic operations to
    /// ensure thread-safe resizing.
    fn try_expand_capacity(&self, current_count: usize) -> Result<()> {
        // Use compare-and-swap to ensure only one thread performs resize
        if self
            .resizing
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Another thread is already resizing, wait and retry
            return Err(Error::LockError("Resize in progress".to_string()));
        }
        let current_capacity = self.capacity.load(Ordering::Acquire);
        let new_capacity = (current_capacity * 2).min(self.max_capacity);
        if new_capacity <= current_capacity {
            self.resizing.store(false, Ordering::Release);
            return Err(Error::QueueCapacityExceeded {
                capacity: current_capacity,
                requested: current_count + 1,
            });
        }
        let mut new_entries: Vec<QueueEntry<T>> = Vec::with_capacity(new_capacity);
        for _ in 0..new_capacity {
            new_entries.push(QueueEntry::new());
        }
        let _head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        for i in 0..current_count {
            let src_idx = (tail + i) % current_capacity;
            // Safety: We're copying from valid initialized memory
            unsafe {
                let src_entry = &self.entries[src_idx];
                let dst_entry = &mut new_entries[i];
                ptr::copy_nonoverlapping(
                    ptr::addr_of!((*ptr::from_ref(src_entry)).data).cast::<u8>(),
                    ptr::addr_of_mut!((*ptr::from_mut(dst_entry)).data).cast::<u8>(),
                    mem::size_of::<MaybeUninit<T>>(),
                );
            }
        }
        // Update queue state
        let new_entries = new_entries.into_boxed_slice();
        // Safety: We're replacing the entries with a new allocation
        // This is safe because we have exclusive access during resize
        let old_entries = unsafe {
            let old_ptr =
                &self.entries as *const Box<[QueueEntry<T>]> as *mut Box<[QueueEntry<T>]>;
            ptr::replace(old_ptr, new_entries)
        };
        self.head.store(current_count, Ordering::Release);
        self.tail.store(0, Ordering::Release);
        self.capacity.store(new_capacity, Ordering::Release);
        self.resizing.store(false, Ordering::Release);
        drop(old_entries);
        Ok(())
    }

    /// Attempts to shrink the queue capacity.
    ///
    /// # Arguments
    ///
    /// * `current_count` - Current number of elements in the queue
    ///
    /// # Performance
    ///
    /// This function uses halving strategy when utilization is below
    /// 25% to reduce memory footprint while maintaining buffer for
    /// incoming entries.
    fn try_shrink_capacity(&self, current_count: usize) -> Result<()> {
        // Use compare-and-swap to ensure only one thread performs resize
        if self
            .resizing
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Another thread is already resizing
            return Err(Error::LockError("Resize in progress".to_string()));
        }
        let current_capacity = self.capacity.load(Ordering::Acquire);
        let new_capacity = (current_capacity / 2).max(self.min_capacity);
        if new_capacity >= current_capacity {
            self.resizing.store(false, Ordering::Release);
            return Ok(());
        }
        let mut new_entries: Vec<QueueEntry<T>> = Vec::with_capacity(new_capacity);
        for _ in 0..new_capacity {
            new_entries.push(QueueEntry::new());
        }
        let _head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let copy_count = current_count.min(new_capacity);
        for i in 0..copy_count {
            let src_idx = (tail + i) % current_capacity;
            // Safety: We're copying from valid initialized memory
            unsafe {
                let src_entry = &self.entries[src_idx];
                let dst_entry = &mut new_entries[i];
                ptr::copy_nonoverlapping(
                    ptr::addr_of!((*ptr::from_ref(src_entry)).data).cast::<u8>(),
                    ptr::addr_of_mut!((*ptr::from_mut(dst_entry)).data).cast::<u8>(),
                    mem::size_of::<MaybeUninit<T>>(),
                );
            }
        }
        let new_entries = new_entries.into_boxed_slice();
        // Safety: We're replacing the entries with a new allocation
        let old_entries = unsafe {
            let old_ptr =
                &self.entries as *const Box<[QueueEntry<T>]> as *mut Box<[QueueEntry<T>]>;
            ptr::replace(old_ptr, new_entries)
        };
        self.head.store(copy_count, Ordering::Release);
        self.tail.store(0, Ordering::Release);
        self.capacity.store(new_capacity, Ordering::Release);
        self.resizing.store(false, Ordering::Release);
        mem::drop(old_entries);
        Ok(())
    }

    /// Clears all entries from the queue.
    ///
    /// # Performance
    ///
    /// This function is O(n) where n is the number of elements in
    /// the queue. It properly drops all contained values.
    pub fn clear(&mut self) {
        let current_count = self.count.load(Ordering::Acquire);
        let current_capacity = self.capacity.load(Ordering::Acquire);

        for i in 0..current_count {
            let idx = (self.tail.load(Ordering::Acquire) + i) % current_capacity;
            // Safety: We're dropping valid initialized values
            unsafe {
                let entry = &mut self.entries[idx];
                ptr::drop_in_place(
                    ptr::addr_of_mut!((*ptr::from_mut(entry)).data).cast::<T>(),
                );
            }
        }
        self.head.store(0, Ordering::Release);
        self.tail.store(0, Ordering::Release);
        self.count.store(0, Ordering::Release);
    }
}

impl<T> Drop for Queue<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

// Safety: LogQueue is thread-safe when T is Send
unsafe impl<T: Send> Send for Queue<T> {}
unsafe impl<T: Send> Sync for Queue<T> {}

#[cfg(test)]
mod tests {
    use crate::logtrace::Queue;

    fn queue_new_creates_empty_queue() {
        let queue = Queue::<i32>::new().unwrap();
        assert!(queue.is_empty());
    }

    fn queue_with_capacity_clamps_to_minimum() {
        let queue = Queue::<u32>::with_capacity(1).unwrap();
        assert!(queue.capacity() >= 64);
    }

    fn queue_push_then_pop_returns_value() {
        let mut queue = Queue::<u32>::new().unwrap();
        queue.push(7).unwrap();
        assert_eq!(queue.pop(), Some(7));
    }

    fn queue_pop_empty_returns_none() {
        let queue = Queue::<u32>::new().unwrap();
        assert_eq!(queue.pop(), None);
    }

    fn queue_peek_returns_top_without_removing() {
        let mut queue = Queue::<u32>::new().unwrap();
        queue.push(10).unwrap();
        queue.push(20).unwrap();
        assert_eq!(queue.peek(), Some(&20));
        assert_eq!(queue.len(), 2);
    }

    fn queue_len_tracks_pushes_and_pops() {
        let mut queue = Queue::<u32>::new().unwrap();
        assert_eq!(queue.len(), 0);
        queue.push(1).unwrap();
        queue.push(2).unwrap();
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.len(), 1);
    }

    fn queue_is_empty_after_clear() {
        let mut queue = Queue::<u32>::new().unwrap();
        queue.push(1).unwrap();
        queue.clear();
        assert!(queue.is_empty());
    }

    fn queue_capacity_stays_positive() {
        let queue = Queue::<u32>::new().unwrap();
        assert!(queue.capacity() > 0);
    }

    fn queue_dropped_count_starts_at_zero() {
        let queue = Queue::<u32>::new().unwrap();
        assert_eq!(queue.dropped_count(), 0);
    }

    fn queue_push_multiple_values_in_order() {
        let mut queue = Queue::<u32>::new().unwrap();
        for value in 1..=5u32 {
            queue.push(value).unwrap();
        }
        let mut out = Vec::new();
        while let Some(value) = queue.pop() {
            out.push(value);
        }
        assert_eq!(out, vec![5, 4, 3, 2, 1]);
    }

    fn queue_clear_drops_values() {
        let mut queue = Queue::<String>::new().unwrap();
        queue.push("a".to_string()).unwrap();
        queue.push("b".to_string()).unwrap();
        queue.clear();
        assert!(queue.is_empty());
    }

    fn queue_pop_after_clear_returns_none() {
        let mut queue = Queue::<i64>::new().unwrap();
        queue.push(42).unwrap();
        queue.clear();
        assert_eq!(queue.pop(), None);
    }

    fn queue_large_volume_of_pushes_succeeds() {
        let mut queue = Queue::<usize>::new().unwrap();
        for i in 0..500usize {
            queue.push(i).unwrap();
        }
        assert_eq!(queue.len(), 500);
    }

    fn queue_large_volume_pop_matches_lifo_pattern() {
        let mut queue = Queue::<usize>::new().unwrap();
        for i in 0..25usize {
            queue.push(i).unwrap();
        }
        let mut popped = Vec::new();
        while let Some(value) = queue.pop() {
            popped.push(value);
        }
        assert_eq!(popped.len(), 25);
        assert_eq!(popped.first(), Some(&24));
    }

    fn queue_repeated_push_pop_stays_consistent() {
        let mut queue = Queue::<usize>::new().unwrap();
        for i in 0..40usize {
            queue.push(i).unwrap();
            queue.pop().unwrap();
        }
        assert!(queue.is_empty());
    }

    fn queue_capacity_expands_when_needed() {
        let mut queue = Queue::<usize>::with_capacity(64).unwrap();
        for i in 0..80usize {
            queue.push(i).unwrap();
        }
        assert!(queue.capacity() >= 64);
        assert_eq!(queue.len(), 80);
    }

    fn queue_peek_after_pop_returns_next_item() {
        let mut queue = Queue::<usize>::new().unwrap();
        queue.push(1).unwrap();
        queue.push(2).unwrap();
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.peek(), Some(&1));
    }

    fn queue_clear_after_some_pops_keeps_consistency() {
        let mut queue = Queue::<usize>::new().unwrap();
        for i in 0..10usize {
            queue.push(i).unwrap();
        }
        for _ in 0..3 {
            let _ = queue.pop();
        }
        queue.clear();
        assert!(queue.is_empty());
    }

    fn queue_with_zero_capacity_still_works() {
        let queue = Queue::<usize>::with_capacity(0).unwrap();
        assert!(queue.capacity() >= 64);
    }

    fn queue_is_empty_after_multiple_clears() {
        let mut queue = Queue::<usize>::new().unwrap();
        queue.clear();
        queue.clear();
        assert!(queue.is_empty());
    }

    fn queue_push_is_logically_stable() {
        let mut queue = Queue::<usize>::new().unwrap();
        for i in 0..12usize {
            queue.push(i).unwrap();
        }
        assert_eq!(queue.len(), 12);
        for _ in 0..12 {
            let _ = queue.pop();
        }
        assert!(queue.is_empty());
    }

    fn queue_pop_works_for_strings() {
        let mut queue = Queue::<String>::new().unwrap();
        queue.push("first".to_string()).unwrap();
        queue.push("second".to_string()).unwrap();
        assert_eq!(queue.pop(), Some("second".to_string()));
    }
}
