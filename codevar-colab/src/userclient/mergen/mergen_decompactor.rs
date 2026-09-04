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

//! Decompression and compaction utilities for Mergen data structures.
//!
//! This module provides utilities for compressing and decompressing Mergen
//! data structures to reduce memory usage and improve cache efficiency.

use crate::userclient::mergen::mergen_binary_fifo::{Fifo, FifoEntry};
use crate::userclient::mergen::mergen_blockchain::BlockChainUnit;
use std::collections::HashMap;

/// Default block size for compaction operations.
const DEFAULT_BLOCK_SIZE: usize = 4096;

/// Compaction strategy for reducing memory footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// No compaction
    None,
    /// Remove duplicate blocks
    Deduplicate,
    /// Compress similar blocks
    CompressSimilar,
    /// Full compaction
    Full,
}

impl Default for CompactionStrategy {
    fn default() -> Self {
        CompactionStrategy::Deduplicate
    }
}

/// Block compactor for reducing memory usage.
pub struct BlockCompactor {
    /// Strategy for compaction
    strategy: CompactionStrategy,
    /// Block size for compaction operations
    block_size: usize,
}

impl BlockCompactor {
    /// Creates a new block compactor with the specified strategy.
    pub fn new(strategy: CompactionStrategy, block_size: usize) -> Self {
        Self {
            strategy,
            block_size,
        }
    }

    /// Creates a compactor with default settings.
    pub fn default() -> Self {
        Self::new(CompactionStrategy::default(), DEFAULT_BLOCK_SIZE)
    }

    /// Compacts a vector of blocks based on the configured strategy.
    pub fn compact_blocks(&self, blocks: &[BlockChainUnit]) -> Vec<BlockChainUnit> {
        match self.strategy {
            CompactionStrategy::None => blocks.to_vec(),
            CompactionStrategy::Deduplicate => self.deduplicate_blocks(blocks),
            CompactionStrategy::CompressSimilar => self.compress_similar_blocks(blocks),
            CompactionStrategy::Full => self.full_compaction(blocks),
        }
    }

    /// Removes duplicate blocks based on content hash.
    fn deduplicate_blocks(&self, blocks: &[BlockChainUnit]) -> Vec<BlockChainUnit> {
        let mut seen = HashMap::new();
        let mut result = Vec::new();

        for block in blocks {
            let key = self.block_key(block);
            if !seen.contains_key(&key) {
                seen.insert(key, ());
                result.push(block.clone());
            }
        }

        result
    }

    /// Compresses similar blocks by merging adjacent blocks with same content.
    fn compress_similar_blocks(&self, blocks: &[BlockChainUnit]) -> Vec<BlockChainUnit> {
        if blocks.is_empty() {
            return Vec::new();
        }
        let mut result = vec![blocks[0].clone()];
        for block in blocks.iter().skip(1) {
            let last = result.last_mut().unwrap();
            if last.data == block.data && last.offset + last.data.len() == block.offset {
                last.data.push_str(&block.data);
                last.width += block.width;
                last.height += block.height;
                last.change_count = last.change_count.max(block.change_count);
            } else {
                result.push(block.clone());
            }
        }

        result
    }

    /// Performs full compaction combining all strategies.
    fn full_compaction(&self, blocks: &[BlockChainUnit]) -> Vec<BlockChainUnit> {
        let deduplicated = self.deduplicate_blocks(blocks);
        self.compress_similar_blocks(&deduplicated)
    }

    /// Creates a unique key for a block for deduplication.
    fn block_key(&self, block: &BlockChainUnit) -> (String, u64, u32) {
        (block.data.clone(), block.user_id, block.change_count)
    }

    /// Returns the current compaction strategy.
    pub fn strategy(&self) -> CompactionStrategy {
        self.strategy
    }

    /// Sets the compaction strategy.
    pub fn set_strategy(&mut self, strategy: CompactionStrategy) {
        self.strategy = strategy;
    }
}

impl Default for BlockCompactor {
    fn default() -> Self {
        Self::default()
    }
}

/// FIFO compactor for reducing memory usage in FIFO structures.
pub struct FifoCompactor {
    /// Strategy for compaction
    strategy: CompactionStrategy,
}

impl FifoCompactor {
    /// Creates a new FIFO compactor with the specified strategy.
    pub fn new(strategy: CompactionStrategy) -> Self {
        Self { strategy }
    }

    /// Creates a compactor with default settings.
    pub fn default() -> Self {
        Self::new(CompactionStrategy::default())
    }

    /// Compacts a FIFO structure based on the configured strategy.
    pub fn compact_fifo(&self, fifo: &Fifo) -> Fifo {
        match self.strategy {
            CompactionStrategy::None => {
                let mut result = Fifo::with_capacity(fifo.capacity());
                for i in 0..fifo.len() {
                    let entry = FifoEntry::new(
                        fifo.characters[i],
                        fifo.hashes[i],
                        fifo.change_counts[i],
                        fifo.source_positions[i],
                    );
                    result.insert(entry);
                }
                result
            }
            CompactionStrategy::Deduplicate => self.deduplicate_fifo(fifo),
            CompactionStrategy::CompressSimilar => self.compress_similar_fifo(fifo),
            CompactionStrategy::Full => self.full_compaction_fifo(fifo),
        }
    }

    /// Removes duplicate entries from the FIFO.
    fn deduplicate_fifo(&self, fifo: &Fifo) -> Fifo {
        let mut seen = HashMap::new();
        let mut result = Fifo::with_capacity(fifo.capacity());

        for i in 0..fifo.len() {
            let key = (fifo.characters[i], fifo.hashes[i]);
            if !seen.contains_key(&key) {
                seen.insert(key, ());
                let entry = FifoEntry::new(
                    fifo.characters[i],
                    fifo.hashes[i],
                    fifo.change_counts[i],
                    fifo.source_positions[i],
                );
                result.insert(entry);
            }
        }

        result
    }

    /// Compresses similar consecutive entries in the FIFO.
    fn compress_similar_fifo(&self, fifo: &Fifo) -> Fifo {
        if fifo.is_empty() {
            return Fifo::new();
        }

        let mut result = Fifo::with_capacity(fifo.capacity());
        let mut last_entry = FifoEntry::new(
            fifo.characters[0],
            fifo.hashes[0],
            fifo.change_counts[0],
            fifo.source_positions[0],
        );

        for i in 1..fifo.len() {
            let current = FifoEntry::new(
                fifo.characters[i],
                fifo.hashes[i],
                fifo.change_counts[i],
                fifo.source_positions[i],
            );

            if last_entry.character == current.character {
                last_entry.change_count =
                    last_entry.change_count.max(current.change_count);
            } else {
                result.insert(last_entry);
                last_entry = current;
            }
        }

        result.insert(last_entry);
        result
    }

    /// Performs full compaction combining all strategies.
    fn full_compaction_fifo(&self, fifo: &Fifo) -> Fifo {
        let deduplicated = self.deduplicate_fifo(fifo);
        self.compress_similar_fifo(&deduplicated)
    }

    /// Returns the current compaction strategy.
    pub fn strategy(&self) -> CompactionStrategy {
        self.strategy
    }

    /// Sets the compaction strategy.
    pub fn set_strategy(&mut self, strategy: CompactionStrategy) {
        self.strategy = strategy;
    }
}

impl Default for FifoCompactor {
    fn default() -> Self {
        Self::default()
    }
}

/// Statistics about compaction operations.
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    /// Original size in bytes
    pub original_size: usize,
    /// Compacted size in bytes
    pub compacted_size: usize,
    /// Number of blocks removed
    pub blocks_removed: usize,
    /// Compression ratio (original / compacted)
    pub compression_ratio: f32,
}

impl CompactionStats {
    /// Creates new compaction statistics.
    pub fn new(
        original_size: usize,
        compacted_size: usize,
        blocks_removed: usize,
    ) -> Self {
        let compression_ratio = if compacted_size > 0 {
            (original_size as f32) / (compacted_size as f32)
        } else {
            0.0
        };

        Self {
            original_size,
            compacted_size,
            blocks_removed,
            compression_ratio,
        }
    }

    /// Returns the space saved in bytes.
    pub fn space_saved(&self) -> usize {
        self.original_size.saturating_sub(self.compacted_size)
    }
}

/// Memory-efficient decompressor for Mergen data structures.
pub struct Decompressor {
    /// Block size for decompression operations
    block_size: usize,
}

impl Decompressor {
    /// Creates a new decompressor with the specified block size.
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }

    /// Creates a decompressor with default settings.
    pub fn default() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE)
    }

    /// Decompresses compacted blocks back to their original form.
    pub fn decompress_blocks(&self, blocks: &[BlockChainUnit]) -> Vec<BlockChainUnit> {
        let mut result = Vec::new();

        for block in blocks {
            if block.data.len() > self.block_size {
                for (i, chunk) in
                    block.data.as_bytes().chunks(self.block_size).enumerate()
                {
                    let content = String::from_utf8_lossy(chunk).into_owned();
                    let sub_block = BlockChainUnit::new(
                        block.offset + (i * self.block_size),
                        block.width,
                        1,
                        content,
                        block.timestamp,
                        block.user_id,
                    );
                    result.push(sub_block);
                }
            } else {
                result.push(block.clone());
            }
        }

        result
    }

    /// Returns the block size used for decompression.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::default()
    }
}
