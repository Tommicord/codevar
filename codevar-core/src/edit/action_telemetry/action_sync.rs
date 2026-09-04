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

//! Synchronization Primitives for Action Cache
//!
//! Provides optimized synchronization primitives for high-concurrency scenarios.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::Duration;

/// Read-write lock optimized for read-heavy workloads
pub struct ActionCacheRwLock<T> {
    inner: RwLock<T>,
}

impl<T> ActionCacheRwLock<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: RwLock::new(value),
        }
    }

    pub fn read(&self) -> CacheRwLockReadGuard<'_, T> {
        CacheRwLockReadGuard {
            guard: self
                .inner
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        }
    }

    pub fn write(&self) -> CacheRwLockWriteGuard<'_, T> {
        CacheRwLockWriteGuard {
            guard: self
                .inner
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        }
    }

    pub fn try_read(&self) -> Option<CacheRwLockReadGuard<'_, T>> {
        self.inner
            .try_read()
            .ok()
            .map(|guard| CacheRwLockReadGuard { guard })
    }

    pub fn try_write(&self) -> Option<CacheRwLockWriteGuard<'_, T>> {
        self.inner
            .try_write()
            .ok()
            .map(|guard| CacheRwLockWriteGuard { guard })
    }
}

pub struct CacheRwLockReadGuard<'a, T> {
    guard: std::sync::RwLockReadGuard<'a, T>,
}

impl<'a, T> Deref for CacheRwLockReadGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

pub struct CacheRwLockWriteGuard<'a, T> {
    guard: std::sync::RwLockWriteGuard<'a, T>,
}

impl<'a, T> Deref for CacheRwLockWriteGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<'a, T> DerefMut for CacheRwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

/// Mutex optimized for low-contention scenarios
pub type ActionCacheMutex<T> = Mutex<T>;

/// Atomic U64 with extended operations
pub trait ActionAtomicExt {
    /// Fetch and add with saturation
    fn fetch_add_saturating(&self, val: u64) -> u64;
    /// Fetch and subtract with saturation
    fn fetch_sub_saturating(&self, val: u64) -> u64;
    /// Compare and swap with saturation
    fn compare_exchange_saturating(&self, current: u64, new: u64) -> Result<u64, u64>;
    /// Load with acquire ordering
    fn load_acquire(&self) -> u64;
    /// Store with release ordering
    fn store_release(&self, val: u64);
}

impl ActionAtomicExt for AtomicU64 {
    fn fetch_add_saturating(&self, val: u64) -> u64 {
        let mut current = self.load(Ordering::Relaxed);
        loop {
            let new = current.saturating_add(val);
            match self.compare_exchange_weak(
                current,
                new,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return current,
                Err(x) => current = x,
            }
        }
    }

    fn fetch_sub_saturating(&self, val: u64) -> u64 {
        let mut current = self.load(Ordering::Relaxed);
        loop {
            let new = current.saturating_sub(val);
            match self.compare_exchange_weak(
                current,
                new,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return current,
                Err(x) => current = x,
            }
        }
    }

    fn compare_exchange_saturating(&self, current: u64, new: u64) -> Result<u64, u64> {
        self.compare_exchange(current, new, Ordering::AcqRel, Ordering::Relaxed)
    }

    fn load_acquire(&self) -> u64 {
        self.load(Ordering::Acquire)
    }

    fn store_release(&self, val: u64) {
        self.store(val, Ordering::Release);
    }
}

/// Spinlock for very short critical sections
pub struct Spinlock {
    locked: AtomicBool,
}

impl Spinlock {
    pub const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    pub fn lock(&self) -> ActionSpinlockGuard<'_> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        ActionSpinlockGuard { lock: self }
    }

    pub fn try_lock(&self) -> Option<ActionSpinlockGuard<'_>> {
        if self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(ActionSpinlockGuard { lock: self })
        } else {
            None
        }
    }
}

pub struct ActionSpinlockGuard<'a> {
    lock: &'a Spinlock,
}

impl<'a> Drop for ActionSpinlockGuard<'a> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

/// Seqlock for read-heavy, write-rare scenarios
pub struct ActionSeqLock<T> {
    seq: AtomicUsize,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for ActionSeqLock<T> {}
unsafe impl<T: Send> Sync for ActionSeqLock<T> {}

impl<T> ActionSeqLock<T> {
    pub fn new(data: T) -> Self {
        Self {
            seq: AtomicUsize::new(0),
            data: UnsafeCell::new(data),
        }
    }

    pub fn read(&self) -> T
    where
        T: Copy,
    {
        loop {
            let seq = self.seq.load(Ordering::Acquire);
            if seq & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let data = unsafe { *self.data.get() };
            if self.seq.load(Ordering::Acquire) == seq {
                return data;
            }
        }
    }

    pub fn write(&self) -> ActionSeqLockWriteGuard<'_, T> {
        let seq = self.seq.fetch_add(1, Ordering::AcqRel) + 1;
        ActionSeqLockWriteGuard { lock: self, seq }
    }
}

pub struct ActionSeqLockWriteGuard<'a, T> {
    lock: &'a ActionSeqLock<T>,
    seq: usize,
}

impl<'a, T> Drop for ActionSeqLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.seq.store(self.seq + 1, Ordering::Release);
    }
}

impl<'a, T> Deref for ActionSeqLockWriteGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> DerefMut for ActionSeqLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

/// Wait-free ring buffer index tracker
pub struct ActionIndex {
    pub head: AtomicUsize,
    pub tail: AtomicUsize,
    mask: usize,
}

impl ActionIndex {
    pub fn new(capacity: usize) -> Self {
        let capacity = next_power_of_two_usize(capacity.max(2));
        Self {
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            mask: capacity - 1,
        }
    }

    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (head.wrapping_sub(tail)) & self.mask
    }

    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    pub fn is_full(&self) -> bool {
        ((self.head.load(Ordering::Acquire) + 1) & self.mask)
            == self.tail.load(Ordering::Acquire)
    }

    pub fn try_push(&self) -> Option<usize> {
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            if ((head + 1) & self.mask) == tail {
                return None;
            }
            match self.head.compare_exchange_weak(
                head,
                (head + 1) & self.mask,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(head),
                Err(h) => head = h,
            }
        }
    }

    pub fn try_pop(&self) -> Option<usize> {
        let mut tail = self.tail.load(Ordering::Acquire);
        loop {
            let head = self.head.load(Ordering::Acquire);
            if tail == head {
                return None;
            }
            match self.tail.compare_exchange_weak(
                tail,
                (tail + 1) & self.mask,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(tail),
                Err(t) => tail = t,
            }
        }
    }
}

fn next_power_of_two_usize(value: usize) -> usize {
    if value == 0 {
        return 1;
    }
    let mut n = value - 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n + 1
}
