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

//! Action cache configuration using integer thresholds.

/// Basis points for percentage calculations (10000 = 100%)
const BASIS_POINTS: u32 = 10000;

/// Configuration for [`super::action_cache::ActionCache`].
#[derive(Debug, Clone)]
pub struct ActionCacheConfig {
    /// Initial capacity of the ring buffer (power of two).
    pub initial_capacity: usize,
    /// Minimum capacity (never shrink below this).
    pub min_capacity: usize,
    /// Maximum capacity (never grow above this).
    pub max_capacity: usize,
    /// Utilization threshold to grow, in basis points.
    pub grow_threshold_bps: u64,
    /// Utilization threshold to shrink, in basis points.
    pub shrink_threshold_bps: u64,
    /// Cooldown before shrink, in nanoseconds.
    pub shrink_cooldown_ns: u64,
    /// Enable adaptive resizing based on telemetry.
    pub adaptive_sizing: bool,
    /// Maximum age of entries before considered stale, in nanoseconds.
    pub max_entry_age_ns: u64,
    /// Enable compression of cache entries.
    pub enable_compression: bool,
    /// Compression threshold in bytes.
    pub compression_threshold: usize,
    /// Persist statistics to disk.
    pub persist_stats: bool,
}

impl ActionCacheConfig {
    /// Creates a default action cache configuration.
    pub fn new() -> Self {
        let initial_capacity = 1024;
        let min_capacity = 256;
        let max_capacity = 8192;
        Self {
            initial_capacity,
            min_capacity,
            max_capacity,
            grow_threshold_bps: 7_500,
            shrink_threshold_bps: 2_000,
            shrink_cooldown_ns: 300_000_000_000,
            adaptive_sizing: true,
            max_entry_age_ns: 3_600_000_000_000,
            enable_compression: false,
            compression_threshold: 512,
            persist_stats: false,
        }
    }

    /// Creates a low-memory configuration.
    pub fn low_memory() -> Self {
        Self {
            initial_capacity: 256,
            min_capacity: 64,
            max_capacity: 1024,
            grow_threshold_bps: 8_000,
            shrink_threshold_bps: 1_500,
            shrink_cooldown_ns: 600_000_000_000,
            adaptive_sizing: true,
            max_entry_age_ns: 1_800_000_000_000,
            enable_compression: true,
            compression_threshold: 256,
            persist_stats: false,
        }
    }

    /// Creates a high-throughput configuration.
    pub fn high_throughput() -> Self {
        Self {
            initial_capacity: 4096,
            min_capacity: 1024,
            max_capacity: 16384,
            grow_threshold_bps: 7_000,
            shrink_threshold_bps: 2_500,
            shrink_cooldown_ns: 200_000_000_000,
            adaptive_sizing: true,
            max_entry_age_ns: 7_200_000_000_000,
            enable_compression: false,
            compression_threshold: 1024,
            persist_stats: true,
        }
    }

    /// Creates a WASM-optimized configuration.
    pub fn wasm_optimized() -> Self {
        Self {
            initial_capacity: 512,
            min_capacity: 128,
            max_capacity: 4096,
            grow_threshold_bps: 7_500,
            shrink_threshold_bps: 2_000,
            shrink_cooldown_ns: 400_000_000_000,
            adaptive_sizing: true,
            max_entry_age_ns: 3_600_000_000_000,
            enable_compression: true,
            compression_threshold: 128,
            persist_stats: false,
        }
    }

    /// Validates capacity and threshold relationships.
    pub fn validate(&self) -> Result<(), String> {
        if self.initial_capacity == 0 || !self.initial_capacity.is_power_of_two() {
            return Err("initial_capacity must be a power of two".to_string());
        }
        if self.min_capacity == 0 || !self.min_capacity.is_power_of_two() {
            return Err("min_capacity must be a power of two".to_string());
        }
        if self.max_capacity < self.min_capacity {
            return Err("max_capacity must be >= min_capacity".to_string());
        }
        if self.initial_capacity < self.min_capacity
            || self.initial_capacity > self.max_capacity
        {
            return Err("initial_capacity must be between min and max".to_string());
        }
        if self.grow_threshold_bps > BASIS_POINTS as u64 {
            return Err("grow_threshold_bps must be <= 10000".to_string());
        }
        if self.shrink_threshold_bps >= self.grow_threshold_bps {
            return Err("shrink_threshold_bps must be < grow_threshold_bps".to_string());
        }
        Ok(())
    }
}

impl Default for ActionCacheConfig {
    fn default() -> Self {
        Self::new()
    }
}
