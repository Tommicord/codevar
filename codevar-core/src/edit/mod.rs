//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! # Edit (Wredit engine) Module
//!
//! This module provides the core text editing functionality, implementing
//! a high-performance gap-buffer based text editing system optimized for WASM deployment.
//!
//! ## Architecture Overview
//!
//! The editing system is built around several key architectural components:
//!
//! - **Gap Buffer System**: Efficient text storage using gap buffers with cursor-based editing
//! - **Shared Gap Allocation**: Multi-cursor support through shared gap suballocation
//! - **Text Encoding Support**: Multiple text encodings (UTF-8, UTF-16, UTF-32, ISO-8859-1)
//! - **History Management**: Undo/redo functionality with transaction-based change tracking
//! - **Event System**: Observable pattern for action tracking and telemetry
//! - **File System**: Virtual file system with handle-based file management
//!
//! ## Gap Buffer Architecture
//!
//! The core editing system uses gap buffers, which provide O(1) insertion and deletion at cursor
//! positions. Key characteristics:
//!
//! - **Cursor Management**: Each cursor maintains its position and associated gap
//! - **Shared Gaps**: When cursors are in proximity, they share gap space for efficiency
//! - **Gap Flushing**: Automatic gap buffer management when gaps become full
//! - **Multi-Encoding**: Supports different text encodings through type parameters
//!
//! ## Safety and Performance
//!
//! This module extensively uses unsafe code for performance-critical operations:
//!
//! - **Raw Pointer Operations**: Direct memory manipulation for gap buffer operations
//! - **Bit Manipulation**: Efficient character encoding and decoding
//! - **Memory Alignment**: Cache-optimized data structures for performance
//! - **SIMD Operations**: Vectorized operations for bulk text processing
//!
//! All unsafe operations are carefully documented with their invariants and safety requirements.
//!
//! ## Module Organization
//!
//! ### Core Editing
//! - `wredit_base`: Base types and aligned memory management
//! - `wredit_base_writable`: Main writable buffer implementation
//! - `wredit_base_shared_gap`: Shared gap allocation for multi-cursor support
//! - `wredit_cursor`: Cursor management and positioning
//!
//! ### Text Operations
//! - `wredit_textedit_trait`: Trait definitions for text editing operations
//! - `wredit_textedit_put`: Character insertion operations
//! - `wredit_textedit_delete`: Character deletion operations
//! - `wredit_textedit_replace`: Text replacement operations
//! - `wredit_textedit_gap`: Gap buffer management
//!
//! ### History and Undo
//! - `wredit_history`: History tracking with undo/redo support
//! - `wredit_history_txu`: Transaction-based change units
//! - `wredit_history_txu_ppbuff`: Protocol buffer serialization for history
//! - `wredit_history_compress`: FcWare compression for large cold TXUs
//! - `wredit_memory_guard`: Periodic memory reclaim for opened files
//!
//! ### Encoding and I/O
//! - `wredit_encode`: Text encoding and decoding
//! - `wredit_filesystem`: Virtual file system
//! - `wredit_file_table`: File handle management
//! - `wredit_flush`: Buffer flushing operations
//!
//! ### Event System
//! - `wredit_event`: Event handling and action processing
//! - `wredit_observer`: Observer pattern for action tracking
//! - `action_telemetry`: Action caching and telemetry collection
//!
//! ### Utilities
//! - `wredit_writable_trait`: Writable trait definitions
//! - `regexp`: Regular expression pattern matching
//! - `wredit_keyboard_comm`: Keyboard communication interface

pub mod action_telemetry;
pub mod regexp;
pub mod wredit_base;
pub mod wredit_base_bridge;
pub mod wredit_base_shared_gap;
pub mod wredit_base_writable;
pub mod wredit_cursor;
pub mod wredit_encode;
pub mod wredit_event;
pub mod wredit_file_table;
pub mod wredit_filesystem;
pub mod wredit_flush;
pub mod wredit_history;
pub mod wredit_history_compress;
pub mod wredit_history_txu;
pub mod wredit_history_txu_ppbuff;
pub mod wredit_keyboard_comm;
pub mod wredit_memory_guard;
pub mod wredit_observer;
pub mod wredit_textedit_buffer_query;
pub mod wredit_textedit_cursor;
pub mod wredit_textedit_delete;
pub mod wredit_textedit_gap;
pub mod wredit_textedit_history;
pub mod wredit_textedit_overwrite;
pub mod wredit_textedit_put;
pub mod wredit_textedit_replace;
pub mod wredit_textedit_selection;
pub mod wredit_textedit_trait;
pub mod wredit_writable_trait;

pub use wredit_base::{WritableAlignedPtr, WritableGapSize};
pub use wredit_base_bridge::{ChannelEventBridge, EventBridge, NullEventBridge};
pub use wredit_base_shared_gap::{SharedGap, SharedGapCursorRegion};
pub use wredit_base_writable::{
    BaseWritable, StreamWritable, WritableEvent, WritableEventType, WritableGap,
    WritableGapPtr, WritableGaps,
};
pub use wredit_cursor::{BidiIndex, Cursor};
pub use wredit_encode::{Encoder, EncodingType};
pub use wredit_event::{WritableAction, WritableEventHandler};
pub use wredit_file_table::{OpenedFile, OpenedFileTable, WritableFileHandle};
pub use wredit_filesystem::{
    FileHandle, FileSystemError, FileSystemHandler, FileSystemResult,
};
pub use wredit_flush::{flush_all, flush_gap, flush_stream_write_scalar};
pub use wredit_history::{History, MAX_SIZE};
pub use wredit_history_compress::{
    HISTORY_TOTAL_COMPRESS_THRESHOLD, KEEP_RECENT_UNCOMPRESSED,
    TXU_BLOB_COMPRESS_THRESHOLD, TXU_BLOB_COMPRESS_THRESHOLD_AGGRESSIVE,
};
pub use wredit_history_txu::{
    HistoryTXU, HistoryTXUDelta, HistoryTXUDeltaHash, HistoryTXUDeltaType, TXU_HASH_SIZE,
};
pub use wredit_history_txu_ppbuff::{ChangeType, KeyValue, PpbuffBuilder};
pub use wredit_keyboard_comm::WritableKeyboardComm;
pub use wredit_memory_guard::{
    edit_memory_monitor, maybe_reclaim_opened_files, reclaim_history,
    reclaim_opened_files_now,
};
pub use wredit_observer::{
    UserActionEvent, UserActionObserver, UserActionObserverRegistry, UserActionType,
};
pub use wredit_textedit_trait::{
    TextEditableCursor, TextEditableDelete, TextEditableGap, TextEditableHistory,
    TextEditableOverwrite, TextEditablePut, TextEditableQuery, TextEditableReplace,
    TextEditableSelection,
};
pub use wredit_writable_trait::Writable;
