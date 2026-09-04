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

//! # User View Module
//!
//! This module provides comprehensive user view management for the collaborative editing system,
//! including efficient serialization, compression, and storage of user editing state. It handles
//! the complete user session state including file systems, cursors, history, and actions.
//!
//! ## Architecture Overview
//!
//! The user view system is designed to efficiently manage and serialize complete user editing sessions:
//!
//! - **Binary Serialization**: Efficient binary encoding of user state for network transmission
//! - **Compression**: LZMA compression for reducing bandwidth and storage requirements
//! - **Caching**: Multi-level caching for frequently accessed user data
//! - **File System Management**: Virtual file system tracking per user session
//! - **Action Tracking**: Comprehensive user action logging and replay
//!
//! ## Key Design Principles
//!
//! - **Efficient Serialization**: Binary format optimized for size and parsing speed
//! - **Incremental Updates**: Support for partial updates to minimize bandwidth
//! - **Compression**: LZMA compression for maximum space efficiency
//! - **Type Safety**: Strong typing for all user view components
//! - **Performance**: Optimized for real-time collaborative editing scenarios
//!
//! ## Serialization Architecture
//!
//! The user view uses a multi-section binary format:
//!
//! - **Header**: Magic number, version, and section metadata
//! - **User Section**: User metadata and session information
//! - **File System Section**: Virtual file system state and file handles
//! - **Cursor Section**: Cursor positions and associated metadata
//! - **History Section**: Undo/redo history and transaction units
//! - **Action Section**: User action log for replay and analysis
//!
//! ## Safety and Performance
//!
//! This module uses unsafe code for:
//!
//! - **Binary Serialization**: Direct memory operations for efficient encoding/decoding
//! - **Compression**: Low-level LZMA compression operations
//! - **Memory Management**: Custom allocators for performance optimization
//! - **Bit Manipulation**: Efficient data packing and unpacking
//!
//! All unsafe operations maintain strict invariants for memory safety and data integrity.
//!
//! ## Module Organization
//!
//! ### Core User View
//! - `view_user`: Main user view structure and session management
//! - `view_user_actions`: User action tracking and logging
//! - `view_user_opened_fs`: Opened file system tracking per user
//!
//! ### Serialization
//! - `view_packing_encoder`: Binary encoding of user view data
//! - `view_packing_decoder`: Binary decoding of user view data
//! - `view_packing_builder`: Builder pattern for constructing user views
//! - `view_packing_cache`: Caching system for serialized views
//!
//! ### Compression
//! - `view_compression`: LZMA compression and decompression
//! - `view_fs_packing`: File system specific serialization
//! - `view_history_packing`: History data serialization
//! - `view_txu_packing`: Transaction unit serialization
//!
//! ### File System Management
//! - `view_filesystem`: Virtual file system implementation
//! - `view_file_table`: File handle and metadata management
//! - `view_dir_table`: Directory structure management
//! - `view_filesystem_type_handler`: File type handling and conversion
//!
//! ### Cursor and History
//! - `view_writable_cursor`: Cursor position and state management
//! - `view_writable_cursor_array`: Multi-cursor array management
//! - `view_writable_history`: History tracking and undo/redo
//! - `view_writable_history_txus`: Transaction unit management
//!
//! ### Action System
//! - `view_action_cache`: Action caching for performance
//! - `view_action_packing`: Action serialization
//! - `view_action_queue`: Action queue management
//!
//! ## Data Flow
//!
//! 1. **User Session Creation**: Initialize user view with default state
//! 2. **Edit Operations**: Track all user actions in the action queue
//! 3. **Serialization**: Convert user view to binary format with compression
//! 4. **Network Transmission**: Send serialized view to server/other clients
//! 5. **Deserialization**: Reconstruct user view from binary data
//! 6. **State Application**: Apply received state to local editing session
//!
//! ## Performance Considerations
//!
//! - **Incremental Updates**: Only send changed sections to minimize bandwidth
//! - **Compression Levels**: Configurable compression for different network conditions
//! - **Caching Strategy**: Cache frequently accessed user views
//! - **Memory Pooling**: Reuse memory buffers for serialization operations

mod view_action_cache;
mod view_action_packing;
mod view_action_queue;
mod view_compression;
mod view_dir_properties;
mod view_dir_table;
mod view_dir_table_packing;
mod view_file_order_table;
mod view_file_table;
mod view_filesystem;
mod view_filesystem_type_handler;
mod view_fs_packing;
mod view_fs_queue;
mod view_fs_table_packing;
mod view_history_packing;
mod view_packing_builder;
mod view_packing_cache;
mod view_packing_decoder;
mod view_packing_encoder;
mod view_txu_packing;
mod view_user;
mod view_user_actions;
mod view_user_opened_fs;
mod view_user_opened_fs_packing;
mod view_user_opened_fs_tracking;
mod view_writable_cursor;
mod view_writable_cursor_array;
mod view_writable_history;
mod view_writable_history_txus;
mod view_writable_properties;

pub use view_action_cache::{ActionCache, PackedAction, encode_action_key};
pub use view_action_queue::ActionQueue;
pub use view_compression::{
    LzmaCompressor, LzmaDecompressor, LzmaError, LzmaParams, LzmaResult,
};
pub use view_packing_builder::{UserViewBuilder, UserViewTokenizer};
pub use view_packing_decoder::BytecodeDecoder;
pub use view_packing_encoder::{
    BytecodeEncoder, BytecodeHeader, SectionHeader, SectionType,
};
pub use view_user::UserView;
