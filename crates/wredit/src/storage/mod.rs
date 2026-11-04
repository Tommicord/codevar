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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Block-storage layer: how the engine stores files and directories.
//!
//! Real disks store information as a fixed array of blocks plus a few
//! metadata structures laid out at fixed locations:
//!
//! - a **superblock** describing the geometry (block count, inode
//!   count, where the bitmaps and inode table live),
//! - a **block bitmap** tracking which blocks are free,
pub mod bitmap;
pub mod block;
pub mod fs;
pub mod writable;

pub use block::{BLOCK_SHIFT, BLOCK_SIZE, BlockDevice, MAX_BLOCKS, MemoryDisk};
pub use fs::{DirEntry, DiskFs, FileType, Inode, SuperBlock};
pub use writable::WritableFile;
