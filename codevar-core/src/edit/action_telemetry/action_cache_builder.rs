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

//! Action Cache Builder

use super::action_cache::ActionCache;
use super::action_cache_config::ActionCacheConfig;
use super::action_cache_error::{ActionCacheError, ActionCacheResult};

/// Builder for ActionCache with fluent API
pub struct ActionCacheBuilder {
    config: ActionCacheConfig,
    initial_capacity: Option<usize>,
    eviction_policy: Option<
        Box<
            dyn super::cache_eviction_policy::EvictionPolicy<
                    super::action_cache::ActionCacheEntry,
                >,
        >,
    >,
}

impl ActionCacheBuilder {
    /// Create a new builder with defaults
    pub fn new() -> Self {
        Self {
            config: ActionCacheConfig::default(),
            initial_capacity: None,
            eviction_policy: None,
        }
    }

    /// Set initial capacity
    pub fn initial_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = Some(capacity);
        self
    }

    /// Set minimum capacity
    pub fn min_capacity(mut self, capacity: usize) -> Self {
        self.config.min_capacity = capacity;
        self
    }

    /// Set maximum capacity
    pub fn max_capacity(mut self, capacity: usize) -> Self {
        self.config.max_capacity = capacity;
        self
    }

    /// Set grow threshold in basis points (0-10000)
    pub fn grow_threshold_bps(mut self, threshold: u64) -> Self {
        self.config.grow_threshold_bps = threshold.clamp(0, 10000);
        self
    }

    /// Set shrink threshold in basis points (0-10000)
    pub fn shrink_threshold_bps(mut self, threshold: u64) -> Self {
        self.config.shrink_threshold_bps = threshold.clamp(0, 10000);
        self
    }

    /// Set shrink cooldown in nanoseconds
    pub fn shrink_cooldown_ns(mut self, cooldown: u64) -> Self {
        self.config.shrink_cooldown_ns = cooldown;
        self
    }

    /// Enable/disable adaptive sizing
    pub fn adaptive_sizing(mut self, enabled: bool) -> Self {
        self.config.adaptive_sizing = enabled;
        self
    }

    /// Set maximum entry age in nanoseconds
    pub fn max_entry_age_ns(mut self, age: u64) -> Self {
        self.config.max_entry_age_ns = age;
        self
    }

    /// Enable/disable compression
    pub fn enable_compression(mut self, enabled: bool) -> Self {
        self.config.enable_compression = enabled;
        self
    }

    /// Set compression threshold
    pub fn compression_threshold(mut self, threshold: usize) -> Self {
        self.config.compression_threshold = threshold;
        self
    }

    /// Enable/disable persistent stats
    pub fn persist_stats(mut self, enabled: bool) -> Self {
        self.config.persist_stats = enabled;
        self
    }

    /// Use low memory preset
    pub fn low_memory(mut self) -> Self {
        self.config = ActionCacheConfig::low_memory();
        self
    }

    /// Use high throughput preset
    pub fn high_throughput(mut self) -> Self {
        self.config = ActionCacheConfig::high_throughput();
        self
    }

    /// Use WASM optimized preset
    pub fn wasm_optimized(mut self) -> Self {
        self.config = ActionCacheConfig::wasm_optimized();
        self
    }

    /// Set custom eviction policy
    pub fn eviction_policy(
        mut self,
        policy: Box<
            dyn super::cache_eviction_policy::EvictionPolicy<
                    super::action_cache::ActionCacheEntry,
                >,
        >,
    ) -> Self {
        self.eviction_policy = Some(policy);
        self
    }

    /// Build the ActionCache
    pub fn build(self) -> ActionCacheResult<ActionCache> {
        self.config
            .validate()
            .map_err(|_| ActionCacheError::ConfigValidationFailed)?;

        let capacity = self
            .initial_capacity
            .unwrap_or(self.config.initial_capacity)
            .clamp(self.config.min_capacity, self.config.max_capacity);
        let mut cache = ActionCache::with_config(self.config, capacity);

        if let Some(policy) = self.eviction_policy {
            *cache
                .eviction_policy
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = policy;
        }
        Ok(cache)
    }
}

impl Default for ActionCacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}
