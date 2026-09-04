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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! # Mergen Algorithm for Conflict-Free Text Resolution
//!
//! This module implements the Mergen algorithm, a sophisticated conflict-free text resolution
//! system designed for real-time collaborative editing. It provides deterministic conflict
//! resolution without operational transformation, using blockchain-based change tracking and
//! hash-based conflict detection.
//!
//! ## Architecture Overview
//!
//! The Mergen system is built around several core components:
//!
//! - **Blockchain-based Change Tracking**: Each edit is stored as a block in a blockchain structure
//! - **Deterministic Ordering**: Hash-based ordering ensures consistent conflict resolution across all clients
//! - **Binary FIFO Encoding**: Efficient serialization of edit operations for network transmission
//! - **Conflict Detection**: Real-time conflict detection with multiple resolution strategies
//! - **Caching System**: Performance optimization through intelligent block caching
//!
//! ## Key Design Principles
//!
//! - **Conflict-Free by Design**: Uses operational transformation principles to avoid conflicts
//! - **Deterministic Resolution**: Same inputs always produce the same output across all clients
//! - **Network Efficiency**: Binary encoding minimizes bandwidth usage
//! - **Scalability**: Handles hundreds of concurrent users with minimal performance degradation
//!
//! ## Blockchain Architecture
//!
//! The core data structure is a blockchain where each block represents an edit operation:
//!
//! - **Block Structure**: Contains user ID, timestamp, content hash, and edit data
//! - **Hash Chaining**: Each block references the previous block's hash for integrity
//! - **Deterministic Ordering**: Blocks are ordered by hash to ensure consistent merge order
//! - **Conflict Detection**: Hash mismatches indicate conflicts that need resolution
//!
//! ## Safety and Performance
//!
//! This module uses unsafe code for:
//!
//! - **SIMD Operations**: Vectorized binary data processing for performance
//! - **Memory Management**: Direct memory operations for binary encoding/decoding
//! - **Hash Computation**: Optimized cryptographic hash operations
//!
//! All unsafe operations maintain strict invariants for memory safety and data integrity.
//!
//! ## Module Organization
//!
//! ### Core Algorithm
//! - `mergen_blockchain`: Blockchain-based change tracking and block management
//! - `mergen_client`: Main client interface for merge operations
//! - `mergen_conflict`: Conflict detection and resolution logic
//! - `mergen_resolve_strategy`: Pluggable conflict resolution strategies
//!
//! ### Encoding/Decoding
//! - `mergen_binary_fifo`: Binary FIFO queue for network transmission
//! - `mergen_fifo_encode`: Binary encoding of edit operations
//! - `mergen_fifo_decode`: Binary decoding of edit operations
//! - `mergen_binary_resolve`: Binary data conflict resolution
//!
//! ### Performance Optimization
//! - `mergen_cache`: Block caching system for performance
//! - `mergen_decompactor`: Data compression for storage efficiency
//! - `mergen_hash`: Hash computation utilities
//!
//! ## Conflict Resolution Strategies
//!
//! The system supports multiple resolution strategies:
//!
//! - **Timestamp Ordering**: Prefer edits with earlier timestamps
//! - **User ID Ordering**: Prefer edits from lower user IDs
//! - **Content Similarity**: Choose edits with higher content similarity
//! - **Most Recent**: Prefer the most recent edit in the timeline
//! - **Custom**: User-defined resolution functions

pub mod mergen_binary_fifo;
pub mod mergen_binary_resolve;
pub mod mergen_blockchain;
pub mod mergen_cache;
pub mod mergen_client;
pub mod mergen_conflict;
pub mod mergen_decompactor;
pub mod mergen_fifo_decode;
pub mod mergen_fifo_encode;
pub mod mergen_hash;
pub mod mergen_resolve_strategy;
