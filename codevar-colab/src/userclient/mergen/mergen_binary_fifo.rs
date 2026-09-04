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

//! Cache-friendly Binary FIFO for Mergen character ordering.
//!
//! This module provides a cache-optimized binary FIFO structure that maintains
//! characters in merge order based on their timestamps and user IDs. The data
//! structure is designed to minimize CPU cache misses

use crate::userclient::mergen::mergen_hash::compare_hashes;
use std::cmp::Ordering;

/// Entry in the Binary FIFO for merged characters.
///
/// Designed for cache-friendly layout with minimal padding and
/// optimal alignment for SIMD operations.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FifoEntry {
    /// The character itself (4 bytes for alignment)
    pub character: u32,
    /// 8-byte hash for ordering (timestamp + user ID)
    pub hash: u64,
    /// Change count (number of times this character was modified)
    pub change_count: u32,
    /// Original position in source text
    pub source_position: u32,
}

impl FifoEntry {
    /// Creates a new FIFO entry.
    #[inline]
    pub fn new(
        character: u32,
        hash: u64,
        change_count: u32,
        source_position: u32,
    ) -> Self {
        Self {
            character,
            hash,
            change_count,
            source_position,
        }
    }

    /// Returns the character as a Rust char.
    #[inline]
    pub fn as_char(&self) -> char {
        unsafe { char::from_u32_unchecked(self.character) }
    }
}

impl Default for FifoEntry {
    fn default() -> Self {
        Self {
            character: 0,
            hash: 0,
            change_count: 0,
            source_position: 0,
        }
    }
}

/// Cache-friendly Binary FIFO using Structure of Arrays (SoA) layout.
///
/// This layout provides better cache locality by keeping related data
/// in contiguous memory, reducing cache misses during sorting and
/// insertion operations.
#[derive(Clone, Debug)]
pub struct Fifo {
    /// Characters stored contiguously (4 bytes each)
    pub(crate) characters: Vec<u32>,
    /// Hashes stored contiguously (8 bytes each)
    pub(crate) hashes: Vec<u64>,
    /// Change counts stored contiguously (4 bytes each)
    pub(crate) change_counts: Vec<u32>,
    /// Source positions stored contiguously (4 bytes each)
    pub(crate) source_positions: Vec<u32>,
    /// Current number of entries
    len: usize,
    /// Pre-allocated capacity
    capacity: usize,
}

pub const DEFAULT_CAPACITY: usize = 4096;

impl Fifo {
    /// Creates a new FIFO with default capacity.
    ///
    /// # Returns
    ///
    /// New Fifo instance
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Creates a new FIFO with specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity for the FIFO
    ///
    /// # Returns
    ///
    /// New Fifo instance
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            characters: Vec::with_capacity(capacity),
            hashes: Vec::with_capacity(capacity),
            change_counts: Vec::with_capacity(capacity),
            source_positions: Vec::with_capacity(capacity),
            len: 0,
            capacity,
        }
    }

    /// Returns the current number of entries in the FIFO.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the FIFO is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity of the FIFO.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clears all entries from the FIFO without deallocating memory.
    pub fn clear(&mut self) {
        self.characters.clear();
        self.hashes.clear();
        self.change_counts.clear();
        self.source_positions.clear();
        self.len = 0;
    }

    /// Inserts a single entry maintaining FIFO order based on hash.
    ///
    /// Uses binary search for optimal insertion performance (O(log n)).
    ///
    /// # Arguments
    ///
    /// * `entry` - The entry to insert
    pub fn insert(&mut self, entry: FifoEntry) {
        if self.len >= self.capacity {
            self.expand_capacity();
        }
        let pos = self.hashes[..self.len]
            .binary_search_by(|&hash| compare_hashes(hash, entry.hash))
            .unwrap_or_else(|pos| pos);
        self.characters.insert(pos, entry.character);
        self.hashes.insert(pos, entry.hash);
        self.change_counts.insert(pos, entry.change_count);
        self.source_positions.insert(pos, entry.source_position);
        self.len += 1;
    }

    /// Batch inserts multiple entries efficiently.
    ///
    /// Sorts the new entries and merges them with existing data in a single pass,
    /// which is more efficient than multiple individual insertions.
    ///
    /// # Arguments
    ///
    /// * `entries` - Slice of entries to insert
    pub fn batch_insert(&mut self, entries: &[FifoEntry]) {
        if entries.is_empty() {
            return;
        }
        let required_capacity = self.len + entries.len();
        if required_capacity > self.capacity {
            self.expand_to_capacity(required_capacity);
        }
        let mut sorted_entries = entries.to_vec();
        sorted_entries.sort_by(|a, b| compare_hashes(a.hash, b.hash));
        self.merge_sorted(&sorted_entries);
    }

    /// Merges sorted entries with existing sorted entries.
    ///
    /// This is O(n + m) where n is current size and m is new entries size,
    /// much more efficient than individual insertions.
    fn merge_sorted(&mut self, new_entries: &[FifoEntry]) {
        let mut result_chars = Vec::with_capacity(self.len + new_entries.len());
        let mut result_hashes = Vec::with_capacity(self.len + new_entries.len());
        let mut result_counts = Vec::with_capacity(self.len + new_entries.len());
        let mut result_positions = Vec::with_capacity(self.len + new_entries.len());

        let mut i = 0;
        let mut j = 0;

        while i < self.len && j < new_entries.len() {
            let existing_hash = self.hashes[i];
            let new_hash = new_entries[j].hash;

            match compare_hashes(existing_hash, new_hash) {
                Ordering::Less => {
                    result_chars.push(self.characters[i]);
                    result_hashes.push(existing_hash);
                    result_counts.push(self.change_counts[i]);
                    result_positions.push(self.source_positions[i]);
                    i += 1;
                }
                Ordering::Greater => {
                    result_chars.push(new_entries[j].character);
                    result_hashes.push(new_hash);
                    result_counts.push(new_entries[j].change_count);
                    result_positions.push(new_entries[j].source_position);
                    j += 1;
                }
                Ordering::Equal => {
                    if self.change_counts[i] >= new_entries[j].change_count {
                        result_chars.push(self.characters[i]);
                        result_hashes.push(existing_hash);
                        result_counts.push(self.change_counts[i]);
                        result_positions.push(self.source_positions[i]);
                    } else {
                        result_chars.push(new_entries[j].character);
                        result_hashes.push(new_hash);
                        result_counts.push(new_entries[j].change_count);
                        result_positions.push(new_entries[j].source_position);
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < self.len {
            result_chars.push(self.characters[i]);
            result_hashes.push(self.hashes[i]);
            result_counts.push(self.change_counts[i]);
            result_positions.push(self.source_positions[i]);
            i += 1;
        }
        while j < new_entries.len() {
            result_chars.push(new_entries[j].character);
            result_hashes.push(new_entries[j].hash);
            result_counts.push(new_entries[j].change_count);
            result_positions.push(new_entries[j].source_position);
            j += 1;
        }
        self.characters = result_chars;
        self.hashes = result_hashes;
        self.change_counts = result_counts;
        self.source_positions = result_positions;
        self.len = self.characters.len();
    }

    /// Expands capacity by 2x when needed.
    fn expand_capacity(&mut self) {
        let new_capacity = self.capacity.saturating_mul(2).max(1);
        self.expand_to_capacity(new_capacity);
    }

    /// Expands to specific capacity.
    fn expand_to_capacity(&mut self, new_capacity: usize) {
        self.characters.reserve(new_capacity - self.capacity);
        self.hashes.reserve(new_capacity - self.capacity);
        self.change_counts.reserve(new_capacity - self.capacity);
        self.source_positions.reserve(new_capacity - self.capacity);
        self.capacity = new_capacity;
    }

    /// Extracts the ordered character sequence as a String.
    ///
    /// # Returns
    ///
    /// String containing all characters in merge order
    pub fn extract_ordered(&self) -> String {
        self.characters[..self.len]
            .iter()
            .map(|&c| unsafe { char::from_u32_unchecked(c) })
            .collect()
    }
}

impl Default for Fifo {
    fn default() -> Self {
        Self::new()
    }
}
