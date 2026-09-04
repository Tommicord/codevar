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

//! Block-level caching system for Mergen algorithm.
//!
//! This module provides a specialized caching system designed for the Mergen
//! algorithm's block-based text processing. The cache operates at the block level
//! (8×8 or 16×16 character blocks) to match the algorithm's parallel processing
//! architecture.

use crate::userclient::mergen::mergen_hash::{
    compute_content_hash, generate_mergen_hash,
};
use bsize::{BSize, ByteSize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

pub const MAX_BLOCKS: usize = 524288;
pub const MIN_CACHE_SIZE: BSize = BSize::mib(16);
pub const MAX_CACHE_SIZE: BSize = BSize::mib(128);

/// Block-level cache key for Mergen algorithm.
///
/// Combines block content hash with user metadata to uniquely identify
/// a specific block merge scenario. This aligns with Mergen's block-based
/// processing approach.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct BlockCacheKey {
    /// Hash of the block content (using xxHash64)
    pub content_hash: u64,
    /// Combined user hash for all participants in this block merge
    pub user_hash: u64,
    /// Block position in the document (line, column)
    pub position: (u32, u32),
    /// Block dimensions (width, height)
    pub dimensions: (u16, u16),
}

impl BlockCacheKey {
    /// Creates a new block cache key.
    #[inline]
    pub fn new(
        content_hash: u64,
        user_hash: u64,
        position: (u32, u32),
        dimensions: (u16, u16),
    ) -> Self {
        Self {
            content_hash,
            user_hash,
            position,
            dimensions,
        }
    }

    /// Creates a cache key from block data and user information.
    pub fn from_block_data(
        block_content: &str,
        user_ids: &[u64],
        position: (u32, u32),
        dimensions: (u16, u16),
    ) -> Self {
        let content_hash = compute_content_hash(block_content);
        let user_hash = user_ids.iter().fold(0u64, |acc, &id| acc.wrapping_add(id));
        Self::new(content_hash, user_hash, position, dimensions)
    }
}

/// Cached block data with merge metadata.
///
/// Represents a block that has been processed and cached, including
/// the merged result and metadata for cache management.
#[derive(Debug, Clone)]
pub struct CachedBlock {
    /// The merged block content
    pub merged_content: String,
    /// Mergen hash for ordering (timestamp + user ID)
    pub mergen_hash: u64,
    /// Timestamp when this block was cached
    pub cached_at: u64,
    /// Number of times this block has been accessed
    pub access_count: u32,
    /// Flag indicating if this block has been modified since caching
    pub dirty: bool,
    /// Original content hash for validation
    pub original_hash: u64,
}

impl CachedBlock {
    /// Creates a new cached block.
    pub fn new(merged_content: String, mergen_hash: u64, original_hash: u64) -> Self {
        let cached_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            merged_content,
            mergen_hash,
            cached_at,
            access_count: 1,
            dirty: false,
            original_hash,
        }
    }

    /// Records access to this cached block.
    #[inline]
    pub fn record_access(&mut self) {
        self.access_count += 1;
    }

    /// Marks this block as dirty (modified since caching).
    #[inline]
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Returns the age of this cache entry in seconds.
    pub fn age_seconds(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.cached_at)
    }

    /// Validates if the cached block is still valid.
    pub fn is_valid(&self, current_hash: u64) -> bool {
        !self.dirty && self.original_hash == current_hash
    }
}

/// Block cache entry with LRU tracking.
struct CacheEntry {
    /// The cached block data
    block: CachedBlock,
    /// Last access timestamp for LRU eviction
    last_access: u64,
}

/// Block-level LRU cache for Mergen algorithm.
///
/// Optimized for block-level caching with efficient LRU eviction
/// and dirty block tracking.
struct BlockLruCache {
    /// Internal storage using HashMap for O(1) lookups
    storage: HashMap<BlockCacheKey, CacheEntry>,
    /// Access order tracking for LRU eviction (using Vec for simplicity)
    access_order: Vec<BlockCacheKey>,
    /// Maximum number of blocks to cache
    max_blocks: usize,
    /// Maximum total size in bytes
    max_size_bytes: usize,
    /// Current total size in bytes
    current_size_bytes: usize,
}

impl BlockLruCache {
    /// Creates a new block LRU cache.
    pub fn new(max_blocks: usize, max_size_bytes: usize) -> Self {
        Self {
            storage: HashMap::with_capacity(max_blocks),
            access_order: Vec::with_capacity(max_blocks),
            max_blocks,
            max_size_bytes,
            current_size_bytes: 0,
        }
    }

    /// Attempts to get a cached block.
    pub fn get(&mut self, key: BlockCacheKey) -> Option<&CachedBlock> {
        if let Some(entry) = self.storage.get_mut(&key) {
            entry.block.record_access();
            entry.last_access = current_timestamp();

            // Move to end of access order (most recently used)
            if let Some(pos) = self.access_order.iter().position(|&k| k == key) {
                self.access_order.remove(pos);
                self.access_order.push(key);
            }

            Some(&entry.block)
        } else {
            None
        }
    }

    /// Inserts a new block into the cache.
    pub fn put(&mut self, key: BlockCacheKey, block: CachedBlock) {
        let block_size = block.merged_content.len();

        if let Some(existing) = self.storage.remove(&key) {
            self.current_size_bytes -= existing.block.merged_content.len();
            if let Some(pos) = self.access_order.iter().position(|&k| k == key) {
                self.access_order.remove(pos);
            }
        }
        while (self.storage.len() >= self.max_blocks
            || self.current_size_bytes + block_size > self.max_size_bytes)
            && !self.access_order.is_empty()
        {
            self.evict_lru();
        }
        let entry = CacheEntry {
            block,
            last_access: current_timestamp(),
        };
        self.current_size_bytes += block_size;
        self.storage.insert(key, entry);
        self.access_order.push(key);
    }

    /// Evicts the least recently used block.
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self.access_order.first() {
            if let Some(removed) = self.storage.remove(lru_key) {
                self.current_size_bytes -= removed.block.merged_content.len();
            }
            self.access_order.remove(0);
        }
    }

    /// Marks a block as dirty if it exists in cache.
    pub fn mark_dirty(&mut self, key: BlockCacheKey) {
        if let Some(entry) = self.storage.get_mut(&key) {
            entry.block.mark_dirty();
        }
    }

    /// Removes dirty blocks from cache.
    pub fn remove_dirty(&mut self) {
        let dirty_keys: Vec<_> = self
            .storage
            .iter()
            .filter(|(_, entry)| entry.block.dirty)
            .map(|(key, _)| *key)
            .collect();

        for key in dirty_keys {
            if let Some(removed) = self.storage.remove(&key) {
                self.current_size_bytes -= removed.block.merged_content.len();
                if let Some(pos) = self.access_order.iter().position(|&k| k == key) {
                    self.access_order.remove(pos);
                }
            }
        }
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        self.storage.clear();
        self.access_order.clear();
        self.current_size_bytes = 0;
    }

    /// Returns the current number of cached blocks.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns the current memory usage in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        self.current_size_bytes
    }
}

/// Mergen block cache system.
///
/// Provides block-level caching optimized for the Mergen algorithm's
/// parallel processing architecture.
pub struct BlockCache {
    /// Block-level LRU cache
    block_cache: BlockLruCache,
    /// Maximum total memory for caching
    max_total_memory: usize,
}

impl BlockCache {
    /// Creates a new Mergen block cache with optimal settings.
    pub fn new() -> Self {
        let available_memory = Self::detect_available_memory();
        let cache_memory = available_memory.clamp(MIN_CACHE_SIZE, MAX_CACHE_SIZE);
        Self {
            block_cache: BlockLruCache::new(MAX_BLOCKS, cache_memory.bytes()),
            max_total_memory: cache_memory.bytes(),
        }
    }

    /// Creates a disabled cache (no caching).
    pub fn disabled() -> Self {
        Self {
            block_cache: BlockLruCache::new(0, 0),
            max_total_memory: 0,
        }
    }

    /// Detects available system memory.
    ///
    /// Attempts to detect actual available memory using platform-specific methods,
    /// falling back to conservative defaults if detection fails.
    fn detect_available_memory() -> BSize {
        #[cfg(target_os = "linux")]
        {
            Self::detect_memory_linux()
        }

        #[cfg(target_os = "macos")]
        {
            Self::detect_memory_macos()
        }

        #[cfg(target_os = "windows")]
        {
            Self::detect_memory_windows()
        }

        #[cfg(not(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "windows"
        )))]
        {
            Self::default_memory_estimate()
        }
    }

    /// Detects available memory on Linux systems.
    #[cfg(target_os = "linux")]
    fn detect_memory_linux() -> BSize {
        // Try to read from /proc/meminfo
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return BSize::kb(kb); // Convert KB to bytes
                        }
                    }
                }
            }
        }

        // Fallback to MemTotal if MemAvailable not available
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return BSize::kb(kb >> 1); // Use 50% of total memory
                        }
                    }
                }
            }
        }
        Self::default_memory_estimate()
    }

    /// Detects available memory on macOS systems.
    #[cfg(target_os = "macos")]
    fn detect_memory_macos() -> BSize {
        // Use sysctl to get memory info
        use std::process::Command;

        if let Ok(output) = Command::new("sysctl").args(["hw.memsize"]).output() {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Some(bytes_str) = output_str.split_whitespace().nth(1) {
                    if let Ok(total_bytes) = bytes_str.trim().parse::<u64>() {
                        return BSize::b(total_bytes >> 1); // Use 50% of total memory
                    }
                }
            }
        }
        Self::default_memory_estimate()
    }

    /// Detects available memory on Windows systems.
    #[cfg(target_os = "windows")]
    fn detect_memory_windows() -> BSize {
        // Use GlobalMemoryStatusEx via Windows API would be ideal,
        // but for cross-platform compatibility we'll use a fallback
        // In a real implementation, you'd use the winapi crate
        Self::default_memory_estimate()
    }

    /// Provides a conservative default memory estimate when detection fails.
    fn default_memory_estimate() -> BSize {
        BSize::mib(256)
    }

    /// Attempts to retrieve a cached block.
    pub fn get_block(&mut self, key: BlockCacheKey, current_hash: u64) -> Option<String> {
        if let Some(cached) = self.block_cache.get(key) {
            return if cached.is_valid(current_hash) {
                Some(cached.merged_content.clone())
            } else {
                self.block_cache.mark_dirty(key);
                None
            };
        }
        None
    }

    /// Caches a processed block.
    pub fn cache_block(
        &mut self,
        key: BlockCacheKey,
        merged_content: String,
        timestamp: u64,
        user_id: u64,
        original_hash: u64,
    ) {
        if self.max_total_memory == 0 {
            return;
        }
        let mergen_hash = generate_mergen_hash(timestamp, user_id);
        let cached_block = CachedBlock::new(merged_content, mergen_hash, original_hash);
        self.block_cache.put(key, cached_block);
    }

    /// Marks a block as dirty (content changed).
    pub fn mark_block_dirty(&mut self, key: BlockCacheKey) {
        self.block_cache.mark_dirty(key);
    }

    /// Removes all dirty blocks from cache.
    pub fn cleanup_dirty_blocks(&mut self) {
        let before_len = self.block_cache.len();
        self.block_cache.remove_dirty();
        let _evicted = before_len - self.block_cache.len();
    }

    /// Clears all cached blocks.
    pub fn clear(&mut self) {
        self.block_cache.clear();
    }

    /// Returns current memory usage.
    pub fn memory_usage(&self) -> usize {
        self.block_cache.memory_usage_bytes()
    }

    /// Returns total cache capacity.
    pub fn capacity(&self) -> usize {
        self.max_total_memory
    }

    /// Returns the number of cached blocks.
    pub fn block_count(&self) -> usize {
        self.block_cache.len()
    }
}

impl Default for BlockCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns current timestamp in seconds.
#[inline]
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
