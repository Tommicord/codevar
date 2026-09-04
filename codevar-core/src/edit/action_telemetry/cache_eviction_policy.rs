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

//! Cache Eviction Policies
//!
//! Provides different eviction strategies for the action cache with
//! dynamic threshold calculation based on user activity patterns.

use super::action_cache::ActionCacheEntry;
use super::action_eviction_error::{EvictionError, EvictionResult};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;

/// Trait for eviction policies
pub trait EvictionPolicy<T>: Send + Sync {
    /// Select victim indices for eviction
    fn select_victims(
        &self,
        buffer: &[Option<T>],
        count: usize,
    ) -> EvictionResult<Vec<usize>>;

    /// Called when an entry is accessed
    fn on_access(&mut self, index: usize, entry: &T) -> EvictionResult<()>;

    /// Called when an entry is inserted
    fn on_insert(&mut self, index: usize, entry: &T) -> EvictionResult<()>;

    /// Called when an entry is removed
    fn on_remove(&mut self, index: usize) -> EvictionResult<()>;

    /// Update dynamic thresholds based on telemetry
    fn update_thresholds(&mut self, telemetry: &EvictionTelemetry) -> EvictionResult<()>;

    /// Get policy name
    fn name(&self) -> &'static str;
}

/// Telemetry data for dynamic threshold calculation
#[derive(Debug, Clone, Copy, Default)]
pub struct EvictionTelemetry {
    /// Current cache utilization (0-10000 = 0.00%-100.00%)
    pub utilization_basis_points: u64,
    /// Actions per second (scaled by 1000)
    pub actions_per_second_milli: u64,
    /// Navigation ratio (0-10000 = 0.00%-100.00%)
    pub navigation_ratio_basis_points: u64,
    /// Visible key ratio (0-10000 = 0.00%-100.00%)
    pub visible_key_ratio_basis_points: u64,
    /// Average access frequency per entry
    pub avg_access_frequency: u64,
    /// Cache size in entries
    pub cache_size: usize,
    /// Time since last eviction (nanoseconds)
    pub time_since_last_eviction_ns: u64,
    /// Burst mode active
    pub burst_mode: bool,
}

/// Default eviction threshold in basis points (75%)
const DEFAULT_THRESHOLD_BPS: u64 = 7500;

/// Dynamic threshold calculator using integer arithmetic
pub struct ThresholdCapabilityGetter {
    /// Base eviction threshold (basis points)
    base_threshold_bps: u64,
    /// Minimum threshold (basis points)
    min_threshold_bps: u64,
    /// Maximum threshold (basis points)
    max_threshold_bps: u64,
    /// Velocity sensitivity (0-10000)
    velocity_sensitivity_bps: u64,
    /// Navigation sensitivity (0-10000)
    navigation_sensitivity_bps: u64,
    /// Frequency sensitivity (0-10000)
    frequency_sensitivity_bps: u64,
    /// Burst mode multiplier (10000 = 1.0x)
    burst_multiplier: u64,
}

impl ThresholdCapabilityGetter {
    pub fn new() -> Self {
        Self {
            base_threshold_bps: 7500,         // 75%
            min_threshold_bps: 5000,          // 50%
            max_threshold_bps: 9500,          // 95%
            velocity_sensitivity_bps: 2000,   // 20%
            navigation_sensitivity_bps: 1500, // 15%
            frequency_sensitivity_bps: 1000,  // 10%
            burst_multiplier: 15000,          // 1.5x
        }
    }

    /// Calculate dynamic eviction threshold based on telemetry
    /// Returns threshold in basis points (0-10000)
    pub fn threshold(&self, telemetry: &EvictionTelemetry) -> u64 {
        let mut threshold = self.base_threshold_bps;

        // Adjust for utilization pressure
        if telemetry.utilization_basis_points > 8000 {
            threshold = threshold.saturating_add(1000); // +10%
        } else if telemetry.utilization_basis_points < 3000 {
            threshold = threshold.saturating_sub(500); // -5%
        }

        // Adjust for typing velocity (higher velocity = keep more = higher threshold)
        let velocity_factor =
            (telemetry.actions_per_second_milli * self.velocity_sensitivity_bps) / 10000;
        threshold = threshold.saturating_add(velocity_factor);
        // Adjust for navigation-heavy usage (more navigation = lower threshold, evict more)
        if telemetry.navigation_ratio_basis_points > 7000 {
            let nav_factor = (telemetry.navigation_ratio_basis_points
                * self.navigation_sensitivity_bps)
                / 10000;
            threshold = threshold.saturating_sub(nav_factor / 10); // Reduce threshold
        }

        // Adjust for access frequency (high frequency = keep more)
        let freq_factor =
            (telemetry.avg_access_frequency * self.frequency_sensitivity_bps) / 10000;
        threshold = threshold.saturating_add(freq_factor / 100);

        // Burst mode: increase threshold to retain more during bursts
        if telemetry.burst_mode {
            threshold = (threshold * self.burst_multiplier) / 10000;
        }

        // Clamp to bounds
        threshold.clamp(self.min_threshold_bps, self.max_threshold_bps)
    }

    /// Calculate eviction batch size based on cache size and pressure
    pub fn batch_size(
        &self,
        telemetry: &EvictionTelemetry,
        min_batch: usize,
        max_batch: usize,
    ) -> usize {
        let cache_size = telemetry.cache_size;
        let pressure = telemetry.utilization_basis_points;
        let base_batch = (cache_size / 100).max(min_batch).min(max_batch);

        // Scale by pressure (higher pressure = larger batch)
        let pressure_factor = (pressure * 2) / 10000; // 0-2x
        let batch = base_batch.saturating_mul(1 + pressure_factor as usize);
        batch.clamp(min_batch, max_batch)
    }

    /// Calculate score for an entry (higher = more valuable = keep)
    /// Uses integer arithmetic: score = (frequency * 10000) / (age_seconds + 1)
    pub fn entry_score(&self, entry: &ActionCacheEntry, current_time_ns: u64) -> u64 {
        let age_ns = current_time_ns.saturating_sub(entry.timestamp_ns);
        let age_seconds = (age_ns / 1_000_000_000).max(1); // At least 1 second

        let frequency = entry.access_count as u64;
        // Score = (frequency * 10000) / age_seconds
        // Higher score = more valuable = keep longer
        (frequency * 10000) / age_seconds
    }
}

impl Default for ThresholdCapabilityGetter {
    fn default() -> Self {
        Self::new()
    }
}

/// LRU (Least Recently Used) eviction policy
pub struct LruEviction {
    /// Access order (most recent at front)
    access_order: Vec<usize>,
    /// Index to position in access_order
    index_map: HashMap<usize, usize>,
    /// Dynamic threshold calculator
    threshold_calc: ThresholdCapabilityGetter,
    /// Current eviction threshold (basis points)
    current_threshold_bps: u64,
}

impl LruEviction {
    pub fn new() -> Self {
        Self {
            access_order: Vec::new(),
            index_map: HashMap::new(),
            threshold_calc: ThresholdCapabilityGetter::new(),
            current_threshold_bps: 7500,
        }
    }

    fn move_to_front(&mut self, index: usize) {
        if let Some(&pos) = self.index_map.get(&index) {
            if pos > 0 {
                self.access_order.remove(pos);
                self.access_order.insert(0, index);
                for (i, &idx) in self.access_order.iter().enumerate() {
                    self.index_map.insert(idx, i);
                }
            }
        } else {
            self.access_order.insert(0, index);
            self.index_map.insert(index, 0);
        }
    }
}

impl Default for LruEviction {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy<ActionCacheEntry> for LruEviction {
    fn select_victims(
        &self,
        buffer: &[Option<ActionCacheEntry>],
        count: usize,
    ) -> EvictionResult<Vec<usize>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut victims = Vec::with_capacity(count);
        for &idx in self.access_order.iter().rev().take(count) {
            if idx < buffer.len() && buffer[idx].is_some() {
                victims.push(idx);
            }
        }
        if victims.is_empty() && !buffer.is_empty() {
            // Fallback: find any valid entry from the end
            for idx in (0..buffer.len()).rev() {
                if buffer[idx].is_some() && victims.len() < count {
                    victims.push(idx);
                }
            }
        }
        Ok(victims)
    }

    fn on_access(
        &mut self,
        index: usize,
        _entry: &ActionCacheEntry,
    ) -> EvictionResult<()> {
        self.move_to_front(index);
        Ok(())
    }

    fn on_insert(
        &mut self,
        index: usize,
        _entry: &ActionCacheEntry,
    ) -> EvictionResult<()> {
        self.move_to_front(index);
        Ok(())
    }

    fn on_remove(&mut self, index: usize) -> EvictionResult<()> {
        if let Some(&pos) = self.index_map.get(&index) {
            self.access_order.remove(pos);
            self.index_map.remove(&index);
            for (i, &idx) in self.access_order.iter().enumerate().skip(pos) {
                self.index_map.insert(idx, i);
            }
        }
        Ok(())
    }

    fn update_thresholds(&mut self, telemetry: &EvictionTelemetry) -> EvictionResult<()> {
        self.current_threshold_bps = self.threshold_calc.threshold(telemetry);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "LRU"
    }
}

/// Frequency-weighted eviction policy with dynamic thresholds
/// Evicts entries with lowest computed score (frequency / age)
pub struct FrequencyWeightedEviction {
    /// Computed scores for each entry (higher = keep)
    scores: RwLock<HashMap<usize, u64>>,
    /// Last computation time
    last_computed: RwLock<Instant>,
    /// Recompute interval
    recompute_interval: std::time::Duration,
    /// Dynamic threshold calculator
    threshold_capability: ThresholdCapabilityGetter,
    /// Current eviction threshold (basis points)
    current_threshold_bps: RwLock<u64>,
    /// Minimum score to keep (entries below this are eviction candidates)
    min_keep_score: RwLock<u64>,
}

impl FrequencyWeightedEviction {
    pub fn new() -> Self {
        Self {
            scores: RwLock::new(HashMap::new()),
            last_computed: RwLock::new(Instant::now()),
            recompute_interval: std::time::Duration::from_secs(10),
            threshold_capability: ThresholdCapabilityGetter::new(),
            current_threshold_bps: RwLock::new(7500),
            min_keep_score: RwLock::new(100),
        }
    }

    fn recompute_scores(
        &self,
        buffer: &[Option<ActionCacheEntry>],
    ) -> EvictionResult<()> {
        let now = Instant::now();
        let mut last_computed = self.last_computed.write().map_err(|e| {
            EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
        })?;
        if now.duration_since(*last_computed) < self.recompute_interval {
            return Ok(());
        }
        *last_computed = now;
        drop(last_computed);

        let mut scores = self.scores.write().map_err(|e| {
            EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
        })?;
        scores.clear();
        let current_time = current_timestamp_ns();

        for (idx, entry_opt) in buffer.iter().enumerate() {
            if let Some(entry) = entry_opt {
                let score = self.threshold_capability.entry_score(entry, current_time);
                scores.insert(idx, score);
            }
        }

        Ok(())
    }

    /// Get current threshold
    pub fn current_threshold(&self) -> u64 {
        match self.current_threshold_bps.read() {
            Ok(guard) => *guard,
            Err(_) => DEFAULT_THRESHOLD_BPS,
        }
    }
}

impl Default for FrequencyWeightedEviction {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy<ActionCacheEntry> for FrequencyWeightedEviction {
    fn select_victims(
        &self,
        buffer: &[Option<ActionCacheEntry>],
        count: usize,
    ) -> EvictionResult<Vec<usize>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        self.recompute_scores(buffer)?;

        let scores = self.scores.read().map_err(|e| {
            EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
        })?;
        let mut scored: Vec<_> = scores
            .iter()
            .filter(|(idx, _)| **idx < buffer.len() && buffer[**idx].is_some())
            .map(|(idx, score)| (*idx, *score))
            .collect();
        drop(scores);

        // Sort by score ascending (lowest first = evict first)
        scored.sort_by(|a, b| a.1.cmp(&b.1));
        let victims = scored.into_iter().take(count).map(|(idx, _)| idx).collect();

        Ok(victims)
    }

    fn on_access(
        &mut self,
        index: usize,
        entry: &ActionCacheEntry,
    ) -> EvictionResult<()> {
        let current_time = current_timestamp_ns();
        let score = self.threshold_capability.entry_score(entry, current_time);
        self.scores
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .insert(index, score);
        Ok(())
    }

    fn on_insert(
        &mut self,
        index: usize,
        entry: &ActionCacheEntry,
    ) -> EvictionResult<()> {
        self.on_access(index, entry)
    }

    fn on_remove(&mut self, index: usize) -> EvictionResult<()> {
        self.scores
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .remove(&index);
        Ok(())
    }

    fn update_thresholds(&mut self, telemetry: &EvictionTelemetry) -> EvictionResult<()> {
        let threshold = self.threshold_capability.threshold(telemetry);
        *self.current_threshold_bps.write().map_err(|e| {
            EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
        })? = threshold;
        // Adjust min_keep_score based on threshold
        *self.min_keep_score.write().map_err(|e| {
            EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
        })? = (threshold * 100) / 10000; // Rough mapping
        Ok(())
    }

    fn name(&self) -> &'static str {
        "FrequencyWeighted"
    }
}

/// Adaptive eviction policy that dynamically switches strategies
pub struct AdaptiveEviction {
    lru: RwLock<LruEviction>,
    frequency: RwLock<FrequencyWeightedEviction>,
    current_strategy: RwLock<EvictionStrategy>,
    threshold_calc: ThresholdCapabilityGetter,
    last_strategy_switch: RwLock<Instant>,
    strategy_switch_cooldown: std::time::Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvictionStrategy {
    /// LRU (Least-Recently Used) Strategy
    Lru,
    /// Frequency weighted strategy
    Fw,
}

impl AdaptiveEviction {
    pub fn new() -> Self {
        Self {
            lru: RwLock::new(LruEviction::new()),
            frequency: RwLock::new(FrequencyWeightedEviction::new()),
            current_strategy: RwLock::new(EvictionStrategy::Fw),
            threshold_calc: ThresholdCapabilityGetter::new(),
            last_strategy_switch: RwLock::new(Instant::now()),
            strategy_switch_cooldown: std::time::Duration::from_secs(30),
        }
    }

    fn should_switch_strategy(&self, telemetry: &EvictionTelemetry) -> bool {
        let last_switch = match self.last_strategy_switch.read() {
            Ok(guard) => guard,
            Err(_) => return false,
        };
        if last_switch.elapsed() < self.strategy_switch_cooldown {
            return false;
        }
        drop(last_switch);

        // Switch to LRU if navigation-heavy (temporal locality matters more)
        // Switch to FrequencyWeighted if typing-heavy (frequency matters more)
        let current_strategy = match self.current_strategy.read() {
            Ok(guard) => guard,
            Err(_) => return false,
        };
        match *current_strategy {
            EvictionStrategy::Fw => telemetry.navigation_ratio_basis_points > 8000,
            EvictionStrategy::Lru => {
                telemetry.visible_key_ratio_basis_points > 7000
                    && telemetry.actions_per_second_milli > 5000
            }
        }
    }

    fn switch_strategy(&self, telemetry: &EvictionTelemetry) -> EvictionResult<()> {
        let mut current_strategy = self.current_strategy.write().map_err(|e| {
            EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
        })?;
        *current_strategy = match *current_strategy {
            EvictionStrategy::Fw => EvictionStrategy::Lru,
            EvictionStrategy::Lru => EvictionStrategy::Fw,
        };
        drop(current_strategy);

        *self.last_strategy_switch.write().map_err(|e| {
            EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
        })? = Instant::now();
        let _ = self
            .lru
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .update_thresholds(telemetry);
        let _ = self
            .frequency
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .update_thresholds(telemetry);
        Ok(())
    }
}

impl Default for AdaptiveEviction {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy<ActionCacheEntry> for AdaptiveEviction {
    fn select_victims(
        &self,
        buffer: &[Option<ActionCacheEntry>],
        count: usize,
    ) -> EvictionResult<Vec<usize>> {
        if self.should_switch_strategy(&EvictionTelemetry::default()) {
            // Note: We can't actually switch here due to const reference
            // The switch will happen on next update_thresholds call
        }
        let current_strategy = self.current_strategy.read().map_err(|e| {
            EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
        })?;
        match *current_strategy {
            EvictionStrategy::Lru => self
                .lru
                .read()
                .map_err(|e| {
                    EvictionError::TelemetryError(format!(
                        "Failed to acquire lock: {}",
                        e
                    ))
                })?
                .select_victims(buffer, count),
            EvictionStrategy::Fw => self
                .frequency
                .read()
                .map_err(|e| {
                    EvictionError::TelemetryError(format!(
                        "Failed to acquire lock: {}",
                        e
                    ))
                })?
                .select_victims(buffer, count),
        }
    }

    fn on_access(
        &mut self,
        index: usize,
        entry: &ActionCacheEntry,
    ) -> EvictionResult<()> {
        self.lru
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .on_access(index, entry)?;
        self.frequency
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .on_access(index, entry)?;
        Ok(())
    }

    fn on_insert(
        &mut self,
        index: usize,
        entry: &ActionCacheEntry,
    ) -> EvictionResult<()> {
        self.lru
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .on_insert(index, entry)?;
        self.frequency
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .on_insert(index, entry)?;
        Ok(())
    }

    fn on_remove(&mut self, index: usize) -> EvictionResult<()> {
        self.lru
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .on_remove(index)?;
        self.frequency
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .on_remove(index)?;
        Ok(())
    }

    fn update_thresholds(&mut self, telemetry: &EvictionTelemetry) -> EvictionResult<()> {
        if self.should_switch_strategy(telemetry) {
            self.switch_strategy(telemetry)?;
        }
        self.lru
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .update_thresholds(telemetry)?;
        self.frequency
            .write()
            .map_err(|e| {
                EvictionError::TelemetryError(format!("Failed to acquire lock: {}", e))
            })?
            .update_thresholds(telemetry)?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Adaptive"
    }
}

fn current_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
