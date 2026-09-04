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

//! Conflict detection and handling for the Mergen algorithm.
//!
//! This module provides the core conflict detection system that identifies
//! when multiple users have edited the same text blocks and provides
//! mechanisms for resolving these conflicts deterministically.

use crate::userclient::mergen::mergen_blockchain::BlockChainUnit;
use crate::userclient::mergen::mergen_hash::compare_hashes;
use std::collections::HashMap;

/// Represents a conflict between multiple edits to the same block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockConflict {
    /// The position of the conflicted block in the document
    pub position: usize,
    /// All conflicting blocks from different users
    pub conflicting_blocks: Vec<BlockChainUnit>,
    /// The winning block based on merge ordering
    pub resolved_block: Option<BlockChainUnit>,
}

impl BlockConflict {
    /// Creates a new conflict from conflicting blocks.
    pub fn new(position: usize, conflicting_blocks: Vec<BlockChainUnit>) -> Self {
        Self {
            position,
            conflicting_blocks,
            resolved_block: None,
        }
    }

    /// Resolves the conflict using the Mergen ordering algorithm.
    pub fn resolve(&mut self) {
        if self.conflicting_blocks.is_empty() {
            return;
        }

        let mut sorted_blocks = self.conflicting_blocks.clone();
        sorted_blocks
            .sort_by(|a, b| compare_hashes(a.ordering_hash(), b.ordering_hash()));

        self.resolved_block = Some(sorted_blocks[0].clone());
    }

    /// Returns true if this conflict has been resolved.
    pub fn is_resolved(&self) -> bool {
        self.resolved_block.is_some()
    }

    /// Returns the number of conflicting users.
    pub fn conflict_count(&self) -> usize {
        self.conflicting_blocks.len()
    }
}

/// Conflict detector for identifying overlapping edits.
pub struct ConflictDetector {
    /// Block size for conflict detection (must be multiple of 8 or 16)
    block_size: usize,
}

impl ConflictDetector {
    /// Creates a new conflict detector with the specified block size.
    pub fn new(block_size: usize) -> Self {
        assert!(
            block_size == 8 || block_size == 16,
            "Block size must be 8 or 16 for optimal GPU/CPU processing"
        );
        Self { block_size }
    }

    /// Creates a conflict detector with the default block size (16).
    pub fn default() -> Self {
        Self::new(16)
    }

    /// Detects conflicts between multiple block chains.
    pub fn detect_conflicts(&self, chains: &[Vec<BlockChainUnit>]) -> Vec<BlockConflict> {
        let mut position_map: HashMap<usize, Vec<BlockChainUnit>> = HashMap::new();

        for chain in chains {
            for block in chain {
                position_map
                    .entry(block.offset)
                    .or_insert_with(Vec::new)
                    .push(block.clone());
            }
        }
        position_map
            .into_iter()
            .filter(|(_, blocks)| blocks.len() > 1)
            .map(|(position, blocks)| BlockConflict::new(position, blocks))
            .collect()
    }

    /// Detects conflicts within a single chain (for detecting internal inconsistencies).
    pub fn detect_internal_conflicts(
        &self,
        chain: &[BlockChainUnit],
    ) -> Vec<BlockConflict> {
        let mut position_map: HashMap<usize, Vec<BlockChainUnit>> = HashMap::new();

        for block in chain {
            position_map
                .entry(block.offset)
                .or_insert_with(Vec::new)
                .push(block.clone());
        }
        position_map
            .into_iter()
            .filter(|(_, blocks)| blocks.len() > 1)
            .map(|(position, blocks)| BlockConflict::new(position, blocks))
            .collect()
    }

    /// Resolves all detected conflicts using the Mergen ordering.
    pub fn resolve_all_conflicts(&self, conflicts: &mut [BlockConflict]) {
        for conflict in conflicts {
            conflict.resolve();
        }
    }

    /// Returns the block size used for conflict detection.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

impl Default for ConflictDetector {
    fn default() -> Self {
        Self::new(16)
    }
}

/// Conflict resolution result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictResolution {
    /// No conflict detected
    NoConflict,
    /// Conflict resolved with the winning block
    Resolved(BlockChainUnit),
    /// Conflict could not be resolved (requires manual intervention)
    Unresolved(BlockConflict),
}

/// Conflict resolver that applies different resolution strategies.
pub struct ConflictResolver {
    detector: ConflictDetector,
}

impl ConflictResolver {
    /// Creates a new conflict resolver.
    pub fn new(block_size: usize) -> Self {
        Self {
            detector: ConflictDetector::new(block_size),
        }
    }

    /// Creates a conflict resolver with default settings.
    pub fn default() -> Self {
        Self {
            detector: ConflictDetector::default(),
        }
    }

    /// Resolves conflicts between multiple chains using timestamp-based ordering.
    pub fn resolve_chains(
        &self,
        chains: &[Vec<BlockChainUnit>],
    ) -> Vec<ConflictResolution> {
        let conflicts = self.detector.detect_conflicts(chains);
        let mut results = Vec::new();

        if conflicts.is_empty() {
            return vec![ConflictResolution::NoConflict];
        }
        for mut conflict in conflicts {
            conflict.resolve();
            if let Some(resolved) = conflict.resolved_block {
                results.push(ConflictResolution::Resolved(resolved));
            } else {
                results.push(ConflictResolution::Unresolved(conflict));
            }
        }
        results
    }

    /// Returns the conflict detector used by this resolver.
    pub fn detector(&self) -> &ConflictDetector {
        &self.detector
    }
}

impl Default for ConflictResolver {
    fn default() -> Self {
        Self::default()
    }
}
