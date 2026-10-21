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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The shuffle table: a move-to-front list of the blocks the packer has
//! already emitted.
//!
//! The table is what makes a repeated block cheap to reach. A block is
//! looked up by its signature; the *rank* of the hit is the distance of
//! the reference, so the rank doubles as the code. Because the most
//! recently used block is almost always the one the next block refers
//! to, the move-to-front order keeps the common case at rank zero.
//!
//! The table only exists while packing: the decoder reconstructs the
//! same references from the stream alone.

use alloc::vec::Vec;

/// One block the packer has already emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShuffleEntry {
    /// The signature of the block.
    pub signature: u64,
    /// Where the block starts in the packed output.
    pub offset: usize,
    /// How many bytes the block holds.
    pub length: usize,
}

/// A move-to-front list of recently emitted blocks.
#[derive(Debug, Clone)]
pub struct ShuffleTable {
    entries: Vec<ShuffleEntry>,
    capacity: usize,
}

impl ShuffleTable {
    /// Creates a table that keeps at most `capacity` blocks.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    /// The rank of the block with `signature`, if the table holds one.
    ///
    /// The rank is the position in the move-to-front order, so a rank of
    /// zero means the block was the last one used.
    #[must_use]
    pub fn rank_of(&self, signature: u64) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.signature == signature)
    }

    /// Moves the entry at `rank` to the front.
    pub fn promote(&mut self, rank: usize) {
        if rank == 0 || rank >= self.entries.len() {
            return;
        }
        let entry = self.entries.remove(rank);
        self.entries.insert(0, entry);
    }

    /// Records a block that has just been emitted.
    pub fn insert(&mut self, signature: u64, offset: usize, length: usize) {
        self.entries.insert(
            0,
            ShuffleEntry {
                signature,
                offset,
                length,
            },
        );
        if self.entries.len() > self.capacity {
            self.entries.truncate(self.capacity);
        }
    }

    /// The entry at `rank`, if the rank is inside the table.
    #[must_use]
    pub fn get(&self, rank: usize) -> Option<ShuffleEntry> {
        self.entries.get(rank).copied()
    }

    /// How many blocks the table holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_entry_lands_at_rank_zero() {
        let mut table = ShuffleTable::new(4);
        table.insert(1, 0, 10);
        table.insert(2, 10, 10);
        assert_eq!(table.rank_of(2), Some(0));
        assert_eq!(table.rank_of(1), Some(1));
        assert_eq!(table.get(0).map(|entry| entry.offset), Some(10));
    }

    #[test]
    fn promoting_moves_an_entry_to_the_front() {
        let mut table = ShuffleTable::new(4);
        table.insert(1, 0, 10);
        table.insert(2, 10, 10);
        table.insert(3, 20, 10);
        assert_eq!(table.rank_of(1), Some(2));
        table.promote(2);
        assert_eq!(table.rank_of(1), Some(0));
        assert_eq!(table.rank_of(3), Some(1));
    }

    #[test]
    fn the_table_never_grows_past_its_capacity() {
        let mut table = ShuffleTable::new(3);
        for index in 0..10u64 {
            table.insert(index, index as usize * 10, 10);
        }
        assert_eq!(table.len(), 3);
        assert_eq!(table.rank_of(9), Some(0));
        assert_eq!(table.rank_of(0), None);
    }
}
