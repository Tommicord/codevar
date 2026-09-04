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

//! Action Cache with Adaptive Ring Buffer
//!
//! Provides a high-performance, thread-safe ring buffer for storing
//! packed actions with dynamic resizing based on telemetry.

use super::action_cache_config::ActionCacheConfig;
use super::action_cache_error::{ActionCacheError, ActionCacheResult};
use super::action_cache_stats::ActionCacheStats;
use super::action_format::{ActionFlag, PackedAction};
use super::action_ring::ActionRing;
use super::action_sync::{ActionCacheRwLock, CacheRwLockReadGuard};
use super::cache_eviction_policy::{EvictionPolicy, FrequencyWeightedEviction};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LockResult, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

/// Basis points for percentage calculations (10000 = 100%)
const BASIS_POINTS: u32 = 10000;

/// Helper function to handle mutex poisoning
fn handle_mutex_poison<T>(result: LockResult<MutexGuard<'_, T>>) -> MutexGuard<'_, T> {
    result.unwrap_or_else(|e| e.into_inner())
}
use crate::edit::action_telemetry::ActionCacheMutex;
use std::collections::HashMap;
use wasm_bindgen::UnwrapThrowExt;

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct ActionCacheEntry {
    /// The packed action
    pub action: PackedAction,
    /// Timestamp when action was added (nanoseconds since epoch)
    pub timestamp_ns: u64,
    /// Logical sequence number
    pub sequence: u64,
    /// File ID this action belongs to
    pub file_id: u64,
    /// Access count for frequency-weighted eviction
    pub access_count: u32,
    /// Last access timestamp
    pub last_access_ns: u64,
}

impl ActionCacheEntry {
    pub fn new(action: PackedAction, file_id: u64, sequence: u64) -> Self {
        let now_ns = current_timestamp_ns();
        Self {
            action,
            timestamp_ns: now_ns,
            sequence,
            file_id,
            access_count: 0,
            last_access_ns: now_ns,
        }
    }

    pub fn mark_accessed(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
        self.last_access_ns = current_timestamp_ns();
    }
}

/// High-performance action cache with adaptive sizing
pub struct ActionCache {
    /// Ring buffer storage
    buffer: ActionRing<ActionCacheEntry>,
    /// Configuration
    config: ActionCacheRwLock<ActionCacheConfig>,
    /// Statistics
    stats: ActionCacheMutex<ActionCacheStats>,
    /// Eviction policy
    pub eviction_policy: ActionCacheMutex<Box<dyn EvictionPolicy<ActionCacheEntry>>>,
    /// File ID to index mapping for fast lookup
    file_indices: ActionCacheRwLock<HashMap<u64, Vec<usize>>>,
    /// Global sequence counter
    sequence_counter: AtomicU64,
    /// Total actions ever added
    total_actions: AtomicU64,
    /// Cache creation time
    created_at: Instant,
    /// Last resize time
    last_resize: ActionCacheMutex<Instant>,
    /// Resize in progress flag
    resizing: ActionCacheMutex<bool>,
}

impl ActionCache {
    /// Create a new action cache with default configuration
    pub fn new() -> Self {
        let config = ActionCacheConfig::default();
        let capacity = config.initial_capacity;
        Self::with_config(config, capacity)
    }

    /// Create with custom configuration
    pub fn with_config(config: ActionCacheConfig, initial_capacity: usize) -> Self {
        let capacity = initial_capacity
            .max(config.min_capacity)
            .min(config.max_capacity);
        let buffer = ActionRing::new(capacity);
        let now = Instant::now();

        Self {
            buffer,
            config: ActionCacheRwLock::new(config),
            stats: ActionCacheMutex::new(ActionCacheStats::new()),
            eviction_policy: ActionCacheMutex::new(Box::new(
                FrequencyWeightedEviction::new(),
            )),
            file_indices: ActionCacheRwLock::new(HashMap::new()),
            sequence_counter: AtomicU64::new(0),
            total_actions: AtomicU64::new(0),
            created_at: now,
            last_resize: ActionCacheMutex::new(now),
            resizing: ActionCacheMutex::new(false),
        }
    }

    /// Create with builder pattern
    pub fn builder() -> super::action_cache_builder::ActionCacheBuilder {
        super::action_cache_builder::ActionCacheBuilder::new()
    }

    /// Push an action to the cache
    pub fn push(&self, action: PackedAction, file_id: u64) -> ActionCacheResult<()> {
        self.push_with_count(action, file_id, 1)
    }

    /// Push an action with a specific count
    pub fn push_with_count(
        &self,
        mut action: PackedAction,
        file_id: u64,
        count: u32,
    ) -> ActionCacheResult<()> {
        action.set_count(count);
        let sequence = self.sequence_counter.fetch_add(1, Ordering::AcqRel);
        let entry = ActionCacheEntry::new(action, file_id, sequence);

        // Try to push to ring buffer
        let result = self.buffer.push(entry);

        match result {
            Ok(index) => {
                self.update_file_index(file_id, index, true);
                self.update_stats_on_push(action);
                self.total_actions.fetch_add(1, Ordering::AcqRel);
                self.maybe_resize();
                Ok(())
            }
            Err(ActionCacheError::BufferFull) => {
                // Try eviction
                self.evict_and_push(action, file_id, sequence)
            }
            Err(e) => Err(e),
        }
    }

    /// Push from UserActionEvent
    pub fn push_event(
        &self,
        event: &super::super::wredit_observer::UserActionEvent,
        file_id: u64,
        cursor_id: usize,
    ) -> ActionCacheResult<()> {
        let action = PackedAction::from_event(event, cursor_id);
        self.push(action, file_id)
    }

    /// Try to merge with previous action if same type
    pub fn push_mergeable(
        &self,
        action: PackedAction,
        file_id: u64,
    ) -> ActionCacheResult<bool> {
        if let Some(last) = self.peek_last() {
            if last.action.action_key() == action.action_key()
                && last.action.flag() == ActionFlag::Normal
                && action.flag() == ActionFlag::Normal
                && last.file_id == file_id
            {
                // Same action type, try to increment count
                if last.action.count() < PackedAction::MAX_COUNT {
                    // We need to modify the last entry, this requires special handling
                    // For now, just push as new entry with composite flag
                    let mut composite = action;
                    composite.set_flag(ActionFlag::Composite);
                    self.push(composite, file_id)?;
                    return Ok(true);
                }
            }
        }
        self.push(action, file_id)?;
        Ok(false)
    }

    /// Evict entries and push new one
    fn evict_and_push(
        &self,
        action: PackedAction,
        file_id: u64,
        sequence: u64,
    ) -> ActionCacheResult<()> {
        let mut policy = self
            .eviction_policy
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let mut buffer = self.buffer.write();
        let mut indices = self.file_indices.write();
        let candidates = policy
            .select_victims(&buffer, 1)
            .map_err(|_| ActionCacheError::NotAvailableVictims)?;
        for &victim_idx in &candidates {
            if let Some(Some(entry)) = buffer.get(victim_idx) {
                if let Some(vec) = indices.get_mut(&entry.file_id) {
                    vec.retain(|&i| i != victim_idx);
                    if vec.is_empty() {
                        indices.remove(&entry.file_id);
                    }
                }
            }
        }
        drop(policy);
        drop(indices);
        drop(buffer);
        self.push_with_count(action, file_id, action.count())
    }

    /// Update file index mapping
    fn update_file_index(&self, file_id: u64, index: usize, add: bool) {
        let mut indices = self.file_indices.write();
        if add {
            indices.entry(file_id).or_default().push(index);
        } else {
            if let Some(mut vec) = indices.get_mut(&file_id) {
                vec.retain(|&i| i != index);
                if vec.is_empty() {
                    indices.remove(&file_id);
                }
            }
        }
    }

    /// Update statistics on push
    fn update_stats_on_push(&self, action: PackedAction) {
        let mut stats = self.stats.lock().unwrap_or_else(|err| err.into_inner());
        stats.total_pushed += 1;
        stats.current_size = self.buffer.len();
        if action.flag() == ActionFlag::Composite {
            stats.composite_actions += 1;
        }
        if action.action_key().is_navigation() {
            stats.navigation_actions += 1;
        } else if action.action_key().is_insert() {
            stats.insert_actions += 1;
        } else if action.action_key().is_deletion() {
            stats.deletion_actions += 1;
        }
    }

    /// Check if resize is needed
    fn maybe_resize(&self) {
        let config = self.config.read();
        let current_len = self.buffer.len();
        let capacity = self.buffer.capacity();
        // Check if we should grow
        if current_len >= capacity && capacity < config.max_capacity {
            let utilization = current_len as u64 * BASIS_POINTS as u64 / capacity as u64;
            if utilization > config.grow_threshold_bps {
                drop(config);
                self.resize(capacity * 2);
                return;
            }
        }
        // Check if we should shrink
        if (current_len < (capacity >> 2)) && capacity > config.min_capacity {
            let time_since_resize = self
                .last_resize
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .elapsed();
            if time_since_resize.as_secs() > config.shrink_cooldown_ns {
                drop(config);
                self.resize(capacity >> 1);
            }
        }
    }

    /// Resize the ring buffer
    fn resize(&self, new_capacity: usize) {
        let mut resizing = self.resizing.lock().unwrap_or_else(|err| err.into_inner());
        if *resizing {
            return;
        }
        *resizing = true;
        drop(resizing);

        let config = self.config.read();
        let new_capacity = new_capacity.clamp(config.min_capacity, config.max_capacity);
        drop(config);

        let mut buffer = self.buffer.write();
        buffer.resize(new_capacity, None);
        drop(buffer);

        *self.last_resize.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
        let mut resizing = self.resizing.lock().unwrap_or_else(|e| e.into_inner());
        *resizing = false;
    }

    /// Peek at the last entry without removing
    pub fn peek_last(&self) -> Option<ActionCacheEntry> {
        self.buffer.peek_last()
    }

    /// Peek at the first entry
    pub fn peek_first(&self) -> Option<ActionCacheEntry> {
        self.buffer.peek_first()
    }

    /// Get entry at index
    pub fn get(&self, index: usize) -> Option<ActionCacheEntry> {
        self.buffer.get(index).map(|mut e| {
            e.mark_accessed();
            e
        })
    }

    /// Get entries for a specific file
    pub fn file_actions(&self, file_id: u64) -> Vec<ActionCacheEntry> {
        let indices = self.file_indices.read();
        if let Some(indexes) = indices.get(&file_id) {
            let buffer = self.buffer.read();
            indexes
                .iter()
                .filter_map(|&i| buffer.get(i).and_then(|opt| opt.as_ref().cloned()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get recent actions (last N)
    pub fn recent(&self, count: usize) -> Vec<ActionCacheEntry> {
        self.buffer.recent(count)
    }

    /// Get actions in sequence range
    pub fn sequence_range(&self, start_seq: u64, end_seq: u64) -> Vec<ActionCacheEntry> {
        self.buffer.sequence_range(start_seq, end_seq)
    }

    /// Get actions in time range
    pub fn time_range(&self, start_ns: u64, end_ns: u64) -> Vec<ActionCacheEntry> {
        self.buffer.time_range(start_ns, end_ns)
    }

    /// Get current statistics
    pub fn stats(&self) -> ActionCacheStats {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.current_size = self.buffer.len();
        stats.capacity = self.buffer.capacity();
        stats.uptime_ns = self.created_at.elapsed().as_nanos() as u64;
        stats.clone()
    }

    /// Get configuration
    pub fn config(&self) -> ActionCacheConfig {
        self.config.read().clone()
    }

    /// Update configuration
    pub fn set_config(&self, config: ActionCacheConfig) {
        *self.config.write() = config;
    }

    /// Clear all entries
    pub fn clear(&self) {
        self.buffer.clear();
        self.file_indices.write().clear();
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.current_size = 0;
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Get total actions ever pushed
    pub fn total_actions(&self) -> u64 {
        self.total_actions.load(Ordering::Acquire)
    }

    /// Iterate over all entries (oldest first)
    pub fn iter(&self) -> ActionCacheIter<'_> {
        ActionCacheIter {
            buffer: self.buffer.read(),
            index: 0,
        }
    }

    /// Iterate over entries for a specific file
    pub fn iter_file(&self, file_id: u64) -> FileActionIter<'_> {
        let indices = self.file_indices.read();
        let indexes = indices.get(&file_id).cloned().unwrap_or_default();
        drop(indices);
        FileActionIter {
            buffer: self.buffer.read(),
            indexes,
            pos: 0,
        }
    }
}

/// Iterator over all cache entries
pub struct ActionCacheIter<'a> {
    buffer: CacheRwLockReadGuard<'a, Vec<Option<ActionCacheEntry>>>,
    index: usize,
}

impl<'a> Iterator for ActionCacheIter<'a> {
    type Item = ActionCacheEntry;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.buffer.len() {
            if let Some(entry) = &self.buffer[self.index] {
                self.index += 1;
                return Some(entry.clone());
            }
            self.index += 1;
        }
        None
    }
}

/// Iterator over file-specific entries
pub struct FileActionIter<'a> {
    buffer: CacheRwLockReadGuard<'a, Vec<Option<ActionCacheEntry>>>,
    indexes: Vec<usize>,
    pos: usize,
}

impl<'a> Iterator for FileActionIter<'a> {
    type Item = ActionCacheEntry;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.indexes.len() {
            let idx = self.indexes[self.pos];
            self.pos += 1;
            if let Some(entry) = &self.buffer[idx] {
                return Some(entry.clone());
            }
        }
        None
    }
}

impl Default for ActionCache {
    fn default() -> Self {
        Self::new()
    }
}

fn current_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
