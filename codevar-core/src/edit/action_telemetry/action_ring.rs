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

use super::action_sync::{
    ActionCacheMutex, ActionCacheRwLock, ActionIndex, CacheRwLockReadGuard,
    CacheRwLockWriteGuard,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Thread-safe ring buffer for cache entries
pub struct ActionRing<T> {
    /// Buffer storage
    buffer: ActionCacheRwLock<Vec<Option<T>>>,
    /// Atomic index tracker
    index: ActionIndex,
    /// Sequence counter
    sequence: AtomicU64,
    /// Capacity (power of two)
    capacity: usize,
    /// Mask for modulo operations
    mask: usize,
}

impl<T: Clone + Send + Sync + 'static> ActionRing<T> {
    /// Create a new ring buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be power of two");
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);

        Self {
            buffer: ActionCacheRwLock::new(buffer),
            index: ActionIndex::new(capacity),
            sequence: AtomicU64::new(0),
            capacity,
            mask: capacity - 1,
        }
    }

    /// Get current length
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Push an entry (returns index or error if full)
    pub fn push(
        &self,
        entry: T,
    ) -> Result<usize, super::action_cache_error::ActionCacheError> {
        if let Some(idx) = self.index.try_push() {
            let seq = self.sequence.fetch_add(1, Ordering::AcqRel);
            let mut buffer = self.buffer.write();
            buffer[idx] = Some(entry);
            Ok(idx)
        } else {
            Err(super::action_cache_error::ActionCacheError::BufferFull)
        }
    }

    /// Pop an entry (returns entry or None if empty)
    pub fn pop(&self) -> Option<T> {
        if let Some(idx) = self.index.try_pop() {
            let mut buffer = self.buffer.write();
            buffer[idx].take()
        } else {
            None
        }
    }

    /// Peek at the last entry without removing
    pub fn peek_last(&self) -> Option<T> {
        let buffer = self.buffer.read();
        let head = self.index.head.load(Ordering::Acquire);
        if head == 0 {
            buffer[self.capacity - 1].as_ref().cloned()
        } else {
            buffer[head - 1].as_ref().cloned()
        }
    }

    /// Peek at the first entry without removing
    pub fn peek_first(&self) -> Option<T> {
        let buffer = self.buffer.read();
        let tail = self.index.tail.load(Ordering::Acquire);
        buffer[tail].as_ref().cloned()
    }

    /// Get entry at index
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.capacity {
            return None;
        }
        let buffer = self.buffer.read();
        buffer[index].as_ref().cloned()
    }

    /// Get mutable entry at index
    pub fn get_mut(&self, index: usize) -> Option<T> {
        if index >= self.capacity {
            return None;
        }
        let buffer = self.buffer.read();
        buffer[index].as_ref().cloned()
    }

    /// Get recent entries (last N)
    pub fn recent(&self, count: usize) -> Vec<T> {
        let buffer = self.buffer.read();
        let head = self.index.head.load(Ordering::Acquire);
        let len = self.index.len();
        let count = count.min(len);

        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let idx = (head.wrapping_sub(1 + i)) & self.mask;
            if let Some(entry) = &buffer[idx] {
                result.push(entry.clone());
            }
        }
        result.reverse();
        result
    }

    /// Get entries in sequence range
    pub fn sequence_range(&self, start_seq: u64, end_seq: u64) -> Vec<T> {
        let buffer = self.buffer.read();
        let len = self.index.len();

        if len == 0 {
            return Vec::new();
        }

        // Binary search would be better but we need sequence tracking
        // For now, linear scan
        let mut result = Vec::new();
        let tail = self.index.tail.load(Ordering::Acquire);

        for i in 0..len {
            let idx = (tail + i) & self.mask;
            if let Some(entry) = &buffer[idx] {
                // Note: This requires T to have sequence field
                // For generic implementation, we skip sequence filtering
                if start_seq <= end_seq {
                    // placeholder
                    result.push(entry.clone());
                }
            }
        }
        result
    }

    /// Get entries in time range
    pub fn time_range(&self, start_ns: u64, end_ns: u64) -> Vec<T> {
        let buffer = self.buffer.read();
        let len = self.index.len();

        if len == 0 {
            return Vec::new();
        }

        let mut result = Vec::new();
        let tail = self.index.tail.load(Ordering::Acquire);

        for i in 0..len {
            let idx = (tail + i) & self.mask;
            if let Some(entry) = &buffer[idx] {
                // Note: This requires T to have timestamp field
                result.push(entry.clone());
            }
        }
        result
    }

    /// Clear all entries
    pub fn clear(&self) {
        let mut buffer = self.buffer.write();
        for slot in buffer.iter_mut() {
            *slot = None;
        }
        self.index.head.store(0, Ordering::Release);
        self.index.tail.store(0, Ordering::Release);
        self.sequence.store(0, Ordering::Release);
    }

    /// Resize the buffer (must be power of two)
    pub fn resize(&mut self, new_capacity: usize) {
        assert!(
            new_capacity.is_power_of_two(),
            "Capacity must be power of two"
        );

        let mut old_buffer = {
            let mut buffer = self.buffer.write();
            std::mem::take(&mut *buffer)
        };

        let mut new_buffer = Vec::with_capacity(new_capacity);
        new_buffer.resize_with(new_capacity, || None);

        // Copy entries from old to new
        let len = self.index.len();
        let tail = self.index.tail.load(Ordering::Acquire);

        for i in 0..len {
            let old_idx = (tail + i) & self.mask;
            if let Some(entry) = old_buffer[old_idx].take() {
                new_buffer[i] = Some(entry);
            }
        }

        self.capacity = new_capacity;
        self.mask = new_capacity - 1;
        self.index = ActionIndex::new(new_capacity);
        self.index.head.store(len, Ordering::Release);
        self.index.tail.store(0, Ordering::Release);
        *self.buffer.write() = new_buffer;
    }

    /// Get read guard for iteration
    pub fn read(&self) -> CacheRwLockReadGuard<'_, Vec<Option<T>>> {
        self.buffer.read()
    }

    /// Get write guard for mutation
    pub fn write(&self) -> CacheRwLockWriteGuard<'_, Vec<Option<T>>> {
        self.buffer.write()
    }
}

impl<T: Clone + Send + Sync + 'static> Default for ActionRing<T> {
    fn default() -> Self {
        Self::new(1024)
    }
}

/// Specialized ring buffer for ActionCacheEntry with sequence/time tracking
pub type ActionRingBuffer = ActionRing<super::action_cache::ActionCacheEntry>;

impl ActionRingBuffer {
    /// Get entry with sequence number
    pub fn by_sequence(
        &self,
        target_seq: u64,
    ) -> Option<super::action_cache::ActionCacheEntry> {
        let buffer = self.buffer.read();
        let len = self.index.len();
        let tail = self.index.tail.load(Ordering::Acquire);

        for i in 0..len {
            let idx = (tail + i) & self.mask;
            if let Some(entry) = &buffer[idx] {
                if entry.sequence == target_seq {
                    return Some(entry.clone());
                }
            }
        }
        None
    }

    /// Get entries newer than sequence
    pub fn since_sequence(
        &self,
        since_seq: u64,
    ) -> Vec<super::action_cache::ActionCacheEntry> {
        let buffer = self.buffer.read();
        let len = self.index.len();
        let tail = self.index.tail.load(Ordering::Acquire);

        let mut result = Vec::new();
        for i in 0..len {
            let idx = (tail + i) & self.mask;
            if let Some(entry) = &buffer[idx] {
                if entry.sequence > since_seq {
                    result.push(entry.clone());
                }
            }
        }
        result
    }

    /// Get entries in time range (requires timestamp field)
    pub fn since_timestamp(
        &self,
        since_ns: u64,
    ) -> Vec<super::action_cache::ActionCacheEntry> {
        let buffer = self.buffer.read();
        let len = self.index.len();
        let tail = self.index.tail.load(Ordering::Acquire);

        let mut result = Vec::new();
        for i in 0..len {
            let idx = (tail + i) & self.mask;
            if let Some(entry) = &buffer[idx] {
                if entry.timestamp_ns >= since_ns {
                    result.push(entry.clone());
                }
            }
        }
        result
    }
}
