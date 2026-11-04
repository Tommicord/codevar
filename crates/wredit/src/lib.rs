//! Copyright 2026 Codevar Project
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

//! The Wredit text-editing engine core.
//!
//! This module tree implements a multi-cursor, gap-buffer based text
//! engine. Documents are stored as packed 32-bit *raw* elements in a
//! sparsely grown heap buffer ([`base::WritableAlignedPtr`]); edits
//! are staged either in a private per-cursor gap or in a shared gap
//! suballocated from a fixed descriptor table
//! ([`shared_gap::SharedGapTable`]) and committed into the raw
//! buffer by the flush layer ([`flush_writable`]).

#![cfg_attr(not(test), no_std)]
extern crate alloc;

pub mod base;
mod codebook;
mod core_op_dispatcher;
pub mod cursor;
pub mod cursor_allocator;
pub mod error;
pub mod flush_writable;
pub mod fs_id_compress;
pub mod shared_gap;
pub mod simd;
pub mod staging_gap;
pub mod storage_bind;
pub mod traits;
pub mod writable;
mod writable_scan;

pub mod fs;
pub mod fs_bitmap;
pub mod fs_block;
pub mod fs_writable;

pub use fs::{DirEntry, DiskFs, FileType, Inode, SuperBlock};
pub use fs_block::{BLOCK_SHIFT, BLOCK_SIZE, BlockDevice, MAX_BLOCKS, MemoryDisk};
pub use fs_writable::WritableFile;

pub use base::{
    CHAR_MASK, COMMON_MASK, ENC_SINGLE_BYTE, ENC_UNKNOWN, ENC_UTF8, ENC_UTF16BE, ENC_UTF16LE, MERGE_MASK,
    NAME_LEN, SHIFT_TYPE, TYPE_MASK, WritableAlignedPtr, WritableGapSize, make_packed, packed_char,
    packed_type,
};
pub use cursor::{BidiIndex, Cursor};
pub use shared_gap::{SharedGapSlot, SharedGapTable};
pub use traits::{
    TextEditableBufferQuery, TextEditableCursor, TextEditableDelete, TextEditableGap, TextEditableOverwrite,
    TextEditablePut, TextEditableReplace, TextEditableSelection, Writable,
};
pub use writable::{
    BaseWritable, EventBridge, MAX_CURSORS, MAX_GAPS, SHARED_GAP_SLOTS, StreamWritable, WritableEvent,
    WritableEventType, WritableGap, WritableGapPtr, WritableGaps,
};
