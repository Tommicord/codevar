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

//! Block-chain based merge ordering for the Mergen text resolution algorithm.

use crate::userclient::mergen::mergen_binary_fifo::{Fifo, FifoEntry};
use crate::userclient::mergen::mergen_hash::{compare_hashes, generate_mergen_hash};

/// A single editable block in the Mergen block chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockChainUnit {
    /// Offset position in the document.
    pub offset: usize,
    /// Width of the block in columns.
    pub width: u32,
    /// Height of the block in rows.
    pub height: u32,
    /// Text content contained in the block.
    pub data: String,
    /// Timestamp of the edit that produced the block.
    pub timestamp: u64,
    /// User identifier that produced the block.
    pub user_id: u64,
    /// Number of times the block changed after its initial creation.
    pub change_count: u32,
}

impl BlockChainUnit {
    /// Creates a new block from editable text.
    #[inline]
    pub fn new(
        offset: usize,
        width: u32,
        height: u32,
        data: impl Into<String>,
        timestamp: u64,
        user_id: u64,
    ) -> Self {
        Self {
            offset,
            width,
            height,
            data: data.into(),
            timestamp,
            user_id,
            change_count: 0,
        }
    }

    /// Computes the merge ordering hash for this block.
    #[inline]
    pub fn ordering_hash(&self) -> u64 {
        generate_mergen_hash(self.timestamp, self.user_id)
    }
}

/// Ordered collection of text blocks merged using the Mergen algorithm.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockChain {
    blocks: Vec<BlockChainUnit>,
}

impl BlockChain {
    /// Creates an empty block chain.
    #[inline]
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Creates a chain from a text buffer by splitting it into 16-character chunks.
    pub fn from_text<T: AsRef<str>>(text: T) -> Self {
        let text = text.as_ref();
        let mut chain = Self::new();
        const CHUNK_SIZE: usize = 16;

        for (index, chunk) in text.as_bytes().chunks(CHUNK_SIZE).enumerate() {
            let content = String::from_utf8_lossy(chunk).into_owned();
            chain.push_block(BlockChainUnit::new(
                index,
                content.chars().count() as u32,
                1,
                content,
                (index + 1) as u64,
                1,
            ));
        }

        chain
    }

    /// Adds a block while preserving the total ordering by timestamp and user hash.
    pub fn push_block(&mut self, block: BlockChainUnit) {
        self.blocks.push(block);
        self.blocks.sort_by(|left, right| {
            compare_hashes(left.ordering_hash(), right.ordering_hash())
        });
    }

    /// Adds multiple blocks while preserving ordering.
    pub fn push_blocks(&mut self, blocks: &[BlockChainUnit]) {
        for block in blocks {
            self.push_block(block.clone());
        }
    }

    /// Gets blocks by their position in the document.
    pub fn get_blocks_at_position(&self, offset: usize) -> Vec<&BlockChainUnit> {
        self.blocks
            .iter()
            .filter(|block| block.offset == offset)
            .collect()
    }

    /// Gets all blocks from a specific user.
    pub fn get_blocks_by_user(&self, user_id: u64) -> Vec<&BlockChainUnit> {
        self.blocks
            .iter()
            .filter(|block| block.user_id == user_id)
            .collect()
    }

    /// Returns the number of blocks in the chain.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns an iterator over the blocks.
    pub fn iter(&self) -> impl Iterator<Item = &BlockChainUnit> {
        self.blocks.iter()
    }

    /// Returns a mutable iterator over the blocks.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut BlockChainUnit> {
        self.blocks.iter_mut()
    }

    /// Resolves the chain into a final merged text buffer.
    pub fn resolve(&self) -> String {
        let mut ordered = self.blocks.clone();
        ordered.sort_by(|left, right| {
            compare_hashes(left.ordering_hash(), right.ordering_hash())
        });

        ordered
            .into_iter()
            .flat_map(|block| block.data.chars().collect::<Vec<_>>())
            .collect()
    }

    /// Resolves the chain into a cache-friendly FIFO ordered by the merge hash.
    pub fn resolve_fifo(&self) -> Fifo {
        let mut ordered = self.blocks.clone();
        ordered.sort_by(|left, right| {
            compare_hashes(left.ordering_hash(), right.ordering_hash())
        });
        let mut fifo = Fifo::new();
        let mut entries = Vec::new();

        for (block_index, block) in ordered.iter().enumerate() {
            for (char_index, ch) in block.data.chars().enumerate() {
                let hash = generate_mergen_hash(
                    block.timestamp.wrapping_add(char_index as u64),
                    block.user_id.wrapping_add(char_index as u64),
                );
                entries.push(FifoEntry::new(
                    u32::from(ch),
                    hash,
                    block.change_count,
                    (block_index as u32)
                        .wrapping_mul(1024)
                        .wrapping_add(char_index as u32),
                ));
            }
        }

        entries.sort_by(|left, right| compare_hashes(left.hash, right.hash));
        for entry in entries {
            fifo.insert(entry);
        }

        fifo
    }

    /// Returns the current number of blocks in the chain.
    #[inline]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Returns true when no blocks are tracked.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}
