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

//! Conflict resolution strategies for the Mergen algorithm.
//!
//! This module provides pluggable strategies for resolving conflicts when
//! multiple users edit the same text blocks. Different strategies can be
//! used depending on the use case (e.g., real-time collaboration vs. batch merging).

use crate::userclient::mergen::mergen_blockchain::BlockChainUnit;
use crate::userclient::mergen::mergen_conflict::BlockConflict;
use crate::userclient::mergen::mergen_hash::compare_hashes;
use bsize::{BSize, ByteSize};

/// Default maximum number of conflicts to resolve before giving up.
const DEFAULT_MAX_CONFLICTS: usize = 1024;

/// Block size constants for optimal GPU/CPU processing.
pub mod block_sizes {
    /// Minimum block size for optimal processing
    pub const MIN_BLOCK_SIZE: usize = 8;
    /// Maximum block size for optimal processing
    pub const MAX_BLOCK_SIZE: usize = 16;
    /// Default block size
    pub const DEFAULT_BLOCK_SIZE: usize = 16;
}

/// Strategy for resolving merge conflicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveStrategy {
    /// Resolve by timestamp (earliest wins, deterministic)
    TimestampOrder,
    /// Resolve by user ID (lower ID wins, deterministic)
    UserIdOrder,
    /// Resolve by change count (fewer changes wins)
    ChangeCountOrder,
    /// Resolve by block position (earlier in document wins)
    PositionOrder,
    /// Resolve by content length (shorter wins)
    ContentLengthOrder,
    /// Resolve by content similarity (most similar to existing content wins)
    ContentSimilarity,
    /// Resolve by most recent activity (latest timestamp wins)
    MostRecent,
    /// Custom resolution using a provided callback
    Custom(fn(&[BlockChainUnit]) -> Option<BlockChainUnit>),
}

impl ResolveStrategy {
    /// Resolves a conflict using this strategy.
    pub fn resolve(&self, conflict: &BlockConflict) -> Option<BlockChainUnit> {
        if conflict.conflicting_blocks.is_empty() {
            return None;
        }

        match self {
            ResolveStrategy::TimestampOrder => conflict
                .conflicting_blocks
                .iter()
                .min_by(|a, b| compare_hashes(a.ordering_hash(), b.ordering_hash()))
                .cloned(),
            ResolveStrategy::UserIdOrder => conflict
                .conflicting_blocks
                .iter()
                .min_by_key(|b| b.user_id)
                .cloned(),
            ResolveStrategy::ChangeCountOrder => conflict
                .conflicting_blocks
                .iter()
                .min_by_key(|b| b.change_count)
                .cloned(),
            ResolveStrategy::PositionOrder => conflict
                .conflicting_blocks
                .iter()
                .min_by_key(|b| b.offset)
                .cloned(),
            ResolveStrategy::ContentLengthOrder => conflict
                .conflicting_blocks
                .iter()
                .min_by_key(|b| b.data.len())
                .cloned(),
            ResolveStrategy::ContentSimilarity => self.resolve_by_similarity(conflict),
            ResolveStrategy::MostRecent => conflict
                .conflicting_blocks
                .iter()
                .max_by_key(|b| b.timestamp)
                .cloned(),
            ResolveStrategy::Custom(resolver) => resolver(&conflict.conflicting_blocks),
        }
    }

    /// Resolves conflict by content similarity using Levenshtein distance.
    fn resolve_by_similarity(&self, conflict: &BlockConflict) -> Option<BlockChainUnit> {
        if conflict.conflicting_blocks.len() < 2 {
            return conflict.conflicting_blocks.first().cloned();
        }
        let first = &conflict.conflicting_blocks[0];
        let mut best_block = first.clone();
        let mut best_similarity = Self::approximate_similarity(&first.data, &first.data);

        for block in conflict.conflicting_blocks.iter().skip(1) {
            let similarity = Self::approximate_similarity(&first.data, &block.data);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_block = block.clone();
            }
        }
        Some(best_block)
    }

    /// Calculates similarity between two strings (0.0 to 1.0).
    fn approximate_similarity(a: &str, b: &str) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let max_len = a.len().max(b.len());
        let distance = Self::levenshtein_distance(a, b);
        1.0 - (distance as f32 / max_len as f32)
    }

    /// Computes Levenshtein distance between two strings.
    fn levenshtein_distance(a: &str, b: &str) -> usize {
        let a_chars = a.chars().collect::<Vec<char>>();
        let b_chars = b.chars().collect::<Vec<char>>();
        let m = a_chars.len();
        let n = b_chars.len();
        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }
        let mut previous = vec![0; n + 1];
        let mut current = vec![0; n + 1];

        for (i, val) in previous.iter_mut().enumerate() {
            *val = i;
        }
        for (i, &a_char) in a_chars.iter().enumerate() {
            current[0] = i + 1;
            for (j, &b_char) in b_chars.iter().enumerate() {
                let cost = if a_char == b_char { 0 } else { 1 };
                current[j + 1] =
                    [current[j] + 1, previous[j + 1] + 1, previous[j] + cost]
                        .into_iter()
                        .min()
                        .unwrap_or_default();
            }
            std::mem::swap(&mut previous, &mut current);
        }
        previous[n]
    }

    /// Returns the default strategy for the Mergen algorithm.
    pub fn default() -> Self {
        ResolveStrategy::TimestampOrder
    }
}

impl Default for ResolveStrategy {
    fn default() -> Self {
        Self::default()
    }
}

/// Configuration for conflict resolution.
#[derive(Debug, Clone)]
pub struct ResolveConfig {
    /// The primary resolution strategy
    pub strategy: ResolveStrategy,
    /// Whether to fallback to secondary strategies if primary fails
    pub enable_fallback: bool,
    /// Maximum number of conflicts to resolve before giving up
    pub max_conflicts: usize,
}

impl ResolveConfig {
    /// Creates a new resolution config with the specified strategy.
    pub fn new(strategy: ResolveStrategy) -> Self {
        Self {
            strategy,
            enable_fallback: true,
            max_conflicts: DEFAULT_MAX_CONFLICTS,
        }
    }

    /// Creates a config with unlimited conflict resolution.
    pub fn unlimited(strategy: ResolveStrategy) -> Self {
        Self {
            strategy,
            enable_fallback: true,
            max_conflicts: usize::MAX,
        }
    }

    /// Sets whether fallback strategies are enabled.
    pub fn with_fallback(mut self, enable: bool) -> Self {
        self.enable_fallback = enable;
        self
    }

    /// Sets the maximum number of conflicts to resolve.
    pub fn with_max_conflicts(mut self, max: usize) -> Self {
        self.max_conflicts = max;
        self
    }
}

impl Default for ResolveConfig {
    fn default() -> Self {
        Self::new(ResolveStrategy::default())
    }
}

/// Strategic conflict resolver that applies configured strategies.
pub struct StrategyResolver {
    config: ResolveConfig,
    fallback_strategies: Vec<ResolveStrategy>,
}

impl StrategyResolver {
    /// Creates a new strategic resolver with the given configuration.
    pub fn new(config: ResolveConfig) -> Self {
        let fallback_strategies = if config.enable_fallback {
            vec![
                ResolveStrategy::UserIdOrder,
                ResolveStrategy::ChangeCountOrder,
                ResolveStrategy::PositionOrder,
                ResolveStrategy::ContentSimilarity,
            ]
        } else {
            Vec::new()
        };

        Self {
            config,
            fallback_strategies,
        }
    }

    /// Resolves a single conflict using the configured strategy.
    pub fn resolve_conflict(&self, conflict: &BlockConflict) -> Option<BlockChainUnit> {
        let mut result = self.config.strategy.resolve(conflict);
        if result.is_none() && self.config.enable_fallback {
            for strategy in &self.fallback_strategies {
                result = strategy.resolve(conflict);
                if result.is_some() {
                    break;
                }
            }
        }
        result
    }

    /// Resolves multiple conflicts.
    pub fn resolve_conflicts(
        &self,
        conflicts: &[BlockConflict],
    ) -> Vec<Option<BlockChainUnit>> {
        conflicts
            .iter()
            .take(self.config.max_conflicts)
            .map(|conflict| self.resolve_conflict(conflict))
            .collect()
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &ResolveConfig {
        &self.config
    }

    /// Updates the configuration.
    pub fn set_config(&mut self, config: ResolveConfig) {
        self.fallback_strategies = if config.enable_fallback {
            vec![
                ResolveStrategy::UserIdOrder,
                ResolveStrategy::ChangeCountOrder,
                ResolveStrategy::PositionOrder,
                ResolveStrategy::ContentSimilarity,
            ]
        } else {
            Vec::new()
        };
        self.config = config;
    }

    /// Returns the fallback strategies used by this resolver.
    pub fn fallback_strategies(&self) -> &[ResolveStrategy] {
        &self.fallback_strategies
    }
}

impl Default for StrategyResolver {
    fn default() -> Self {
        Self::new(ResolveConfig::default())
    }
}

/// Hybrid resolver that combines multiple strategies for complex scenarios.
pub struct HyResolver {
    primary: StrategyResolver,
    secondary: Option<StrategyResolver>,
}

impl HyResolver {
    /// Creates a new hybrid resolver with primary and optional secondary resolvers.
    pub fn new(primary: StrategyResolver, secondary: Option<StrategyResolver>) -> Self {
        Self { primary, secondary }
    }

    /// Creates a hybrid resolver with default primary resolver.
    pub fn default() -> Self {
        Self::new(StrategyResolver::default(), None)
    }

    /// Resolves a conflict using the primary resolver, falling back to secondary if needed.
    pub fn resolve_conflict(&self, conflict: &BlockConflict) -> Option<BlockChainUnit> {
        let result = self.primary.resolve_conflict(conflict);

        if result.is_none() {
            if let Some(secondary) = &self.secondary {
                return secondary.resolve_conflict(conflict);
            }
        }
        result
    }

    /// Sets the secondary resolver.
    pub fn set_secondary(&mut self, secondary: StrategyResolver) {
        self.secondary = Some(secondary);
    }

    /// Returns the primary resolver.
    pub fn primary(&self) -> &StrategyResolver {
        &self.primary
    }

    /// Returns the secondary resolver if configured.
    pub fn secondary(&self) -> Option<&StrategyResolver> {
        self.secondary.as_ref()
    }
}

impl Default for HyResolver {
    fn default() -> Self {
        Self::default()
    }
}
