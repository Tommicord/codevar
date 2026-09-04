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

//! # Action Telemetry and Caching Module
//!
//! This module provides high-performance action caching and telemetry for the text editor,
//! enabling efficient tracking, storage, and analysis of user editing actions across multiple files.
//!
//! ## Architecture
//!
//! The action telemetry system is built around several key components:
//!
//! - **Action Format**: Bit-packed 8-byte action format for efficient storage and transmission
//! - **Ring Buffer Cache**: Adaptive ring buffer with dynamic sizing and configurable eviction policies
//! - **Telemetry Collection**: Real-time collection of user activity metrics and patterns
//! - **Thread Safety**: Multi-file observation support with lock-based synchronization
//! - **Error Handling**: Comprehensive error types for cache operations and eviction failures
//!
//! ## Key Design Decisions
//!
//! - **Bit-packed Format**: Actions are stored in a compact 8-byte format to minimize memory usage
//!   and optimize cache locality. Each action contains action type, cursor position, character data,
//!   and flags in a tightly packed structure.
//!
//! - **Ring Buffer Strategy**: Uses a ring buffer instead of a traditional cache to provide O(1)
//!   insertion and eviction operations, with configurable capacity and eviction policies.
//!
//! - **Eviction Policies**: Supports multiple eviction strategies (LRU, frequency-weighted) to
//!   optimize cache hit rates for different usage patterns.
//!
//! - **Thread Safety**: Provides both mutex and RwLock variants for different concurrency patterns,
//!   allowing multiple readers or single writer access as needed.
//!
//! ## Safety Considerations
//!
//! This module uses unsafe code primarily for:
//! - Bit manipulation and packing operations
//! - Memory-efficient data structures
//! - Performance-critical paths in cache operations

pub mod action_cache;
pub mod action_cache_builder;
pub mod action_cache_config;
pub mod action_cache_error;
pub mod action_cache_stats;
pub mod action_eviction_error;
pub mod action_format;
pub mod action_ring;
pub mod action_sync;
pub mod action_telemetry_config;
pub mod cache_eviction_policy;

pub use action_cache::{ActionCache, ActionCacheEntry};
pub use action_cache_builder::ActionCacheBuilder;
pub use action_cache_config::ActionCacheConfig;
pub use action_cache_error::{ActionCacheError, ActionCacheResult};
pub use action_cache_stats::ActionCacheStats;
pub use action_eviction_error::{EvictionError, EvictionResult};
pub use action_format::{ActionExtraInfo, ActionFlag, ActionKey, PackedAction};
pub use action_ring::ActionRing;
pub use action_sync::{
    ActionAtomicExt, ActionCacheMutex, ActionCacheRwLock, CacheRwLockReadGuard,
    CacheRwLockWriteGuard,
};
pub use action_telemetry_config::TelemetryConfig;
pub use cache_eviction_policy::{EvictionPolicy, FrequencyWeightedEviction, LruEviction};
