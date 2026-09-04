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

//! Action cache statistics using integer ratios.

/// Basis points for percentage calculations (10000 = 100%)
const BASIS_POINTS: u32 = 10000;

/// Snapshot of action-cache counters.
#[derive(Debug, Clone, Default)]
pub struct ActionCacheStats {
    /// Total actions ever pushed.
    pub total_pushed: u64,
    /// Current number of entries.
    pub current_size: usize,
    /// Current capacity.
    pub capacity: usize,
    /// Number of composite actions.
    pub composite_actions: u64,
    /// Number of insert actions (visible keys).
    pub insert_actions: u64,
    /// Number of deletion actions.
    pub deletion_actions: u64,
    /// Number of navigation actions.
    pub navigation_actions: u64,
    /// Number of special-key actions.
    pub special_actions: u64,
    /// Number of evictions performed.
    pub evictions: u64,
    /// Uptime in nanoseconds.
    pub uptime_ns: u64,
}

impl ActionCacheStats {
    /// Empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Utilization in basis points.
    pub fn utilization_bps(&self) -> u64 {
        if self.capacity == 0 {
            0
        } else {
            (self.current_size as u64).saturating_mul(BASIS_POINTS as u64)
                / self.capacity as u64
        }
    }

    /// Visible-key ratio in basis points.
    pub fn visible_key_ratio_bps(&self) -> u64 {
        ratio_bps(self.insert_actions, self.total_pushed)
    }

    /// Navigation ratio in basis points.
    pub fn navigation_ratio_bps(&self) -> u64 {
        ratio_bps(self.navigation_actions, self.total_pushed)
    }

    /// Milli-actions per second.
    pub fn milli_actions_per_second(&self) -> u64 {
        if self.uptime_ns == 0 {
            0
        } else {
            self.total_pushed
                .saturating_mul(1_000)
                .saturating_mul(1_000_000_000)
                / self.uptime_ns
        }
    }
}

fn ratio_bps(part: u64, total: u64) -> u64 {
    if total == 0 {
        0
    } else {
        part.saturating_mul(BASIS_POINTS as u64) / total
    }
}
