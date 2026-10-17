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

//! Filesystem layout for the engine storage.
//!
//! Mirrors how real disk file systems (ext2 and friends) lay out a
//! volume: a **superblock** at a fixed location describing the
//! geometry, a **block bitmap** and an **inode bitmap** tracking free
//! resources, an **inode table** holding file metadata and block
//! pointers, and **data blocks** holding file contents and directory
//! entries. Everything is addressed by block index; the block device
//! below guarantees whole-block reads and writes, so the filesystem
//! never sees a short or torn transfer.
//!
//! All multi-byte integers are little-endian, and every structure is
//! serialized to a fixed byte layout rather than relying on `repr(C)`
//! padding, so the on-disk image is stable and portable.
//!
//! # Layout (block numbers are 0-based)
//!
//! ```text
//! block 0 : superblock
//! block 1 : block bitmap       (one bit per block, 1 = used)
//! block 2 : inode bitmap       (one bit per inode, 1 = used)
//! block 3 : inode table start
//! ...     : inode table        (INODE_SIZE bytes per inode)
//! ...     : data blocks        (files and directories)
//! ```
//!
//! # Heap usage
//!
//! [`DiskFs`] keeps the superblock and both bitmaps in fixed arrays;
//! block and indirect-pointer buffers live on the stack. Nothing in
//! this module allocates on the heap. Path resolution uses a fixed
//! stack buffer for the parent-directory stack.

use crate::error::{WreditError, WreditResult};

use super::bitmap::Bitmap;
use super::block::{BLOCK_SIZE, BlockDevice};

/// Magic number identifying a Wredit engine storage image.
const SUPER_MAGIC: u32 = 0xEF53;

/// On-disk inode size, in bytes.
const INODE_SIZE: usize = 0x80;

/// Direct block pointers per inode.
const DIRECT_COUNT: usize = 0x1B;

/// Max name length in a directory entry (bytes).
const NAME_MAX: usize = 0x100;

/// Directory entry fixed size, in bytes.
const DIRENT_SIZE: usize = 0x20;

/// Inodes per block.
const INODES_PER_BLOCK: usize = BLOCK_SIZE / INODE_SIZE; // 32

/// Directory entries per block.
const DIRENTS_PER_BLOCK: usize = BLOCK_SIZE / DIRENT_SIZE; // 128

/// Pointers in one indirect block.
const INDIRECT_PTRS: usize = BLOCK_SIZE / 4; // 1024

/// Maximum directory nesting tracked for `..` resolution.
const DEPTH_MAX: usize = 64;

/// File type encoded in a directory entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FileType {
    Unknown = 0,
    Regular = 1,
    Directory = 2,
}

/// The superblock: fixed geometry of the volume, stored at block 0.
///
/// Serialized little-endian to one block. Fields not listed here are
/// reserved and zeroed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuperBlock {
    pub block_count: u32,
    pub inode_count: u32,
    pub block_bitmap_block: u32,
    pub inode_bitmap_block: u32,
    pub inode_table_block: u32,
    pub inode_table_blocks: u32,
    pub root_inode: u32,
    pub first_free_block: u32,
    pub first_free_inode: u32,
    pub state: u32,
}

impl SuperBlock {
    const MAGIC_OFFSET: usize = 0x00;
    const BLOCK_COUNT_OFFSET: usize = 0x04;
    const INODE_COUNT_OFFSET: usize = 0x08;
    const BB_BLOCK_OFFSET: usize = 0x0C;
    const IB_BLOCK_OFFSET: usize = 0x10;
    const IT_BLOCK_OFFSET: usize = 0x14;
    const IT_BLOCKS_OFFSET: usize = 0x18;
    const ROOT_OFFSET: usize = 0x1C;
    const FFB_OFFSET: usize = 0x20;
    const FFI_OFFSET: usize = 0x24;
    const STATE_OFFSET: usize = 0x28;

    /// Serializes the superblock to one block of bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; BLOCK_SIZE] {
        let mut out = [0u8; BLOCK_SIZE];
        out[Self::MAGIC_OFFSET..][..4].copy_from_slice(&SUPER_MAGIC.to_le_bytes());
        out[Self::BLOCK_COUNT_OFFSET..][..4].copy_from_slice(&self.block_count.to_le_bytes());
        out[Self::INODE_COUNT_OFFSET..][..4].copy_from_slice(&self.inode_count.to_le_bytes());
        out[Self::BB_BLOCK_OFFSET..][..4].copy_from_slice(&self.block_bitmap_block.to_le_bytes());
        out[Self::IB_BLOCK_OFFSET..][..4].copy_from_slice(&self.inode_bitmap_block.to_le_bytes());
        out[Self::IT_BLOCK_OFFSET..][..4].copy_from_slice(&self.inode_table_block.to_le_bytes());
        out[Self::IT_BLOCKS_OFFSET..][..4].copy_from_slice(&self.inode_table_blocks.to_le_bytes());
        out[Self::ROOT_OFFSET..][..4].copy_from_slice(&self.root_inode.to_le_bytes());
        out[Self::FFB_OFFSET..][..4].copy_from_slice(&self.first_free_block.to_le_bytes());
        out[Self::FFI_OFFSET..][..4].copy_from_slice(&self.first_free_inode.to_le_bytes());
        out[Self::STATE_OFFSET..][..4].copy_from_slice(&self.state.to_le_bytes());
        out
    }

    /// Deserializes a block into a superblock, validating the magic.
    pub fn from_bytes(bytes: &[u8]) -> WreditResult<Self> {
        if bytes.len() != BLOCK_SIZE {
            return Err(WreditError::InvalidOperation {
                what: "a superblock occupies exactly one block",
            });
        }
        let magic = u32::from_le_bytes(
            bytes[Self::MAGIC_OFFSET..][..4]
                .try_into()
                .unwrap_or([0; 4]),
        );
        if magic != SUPER_MAGIC {
            return Err(WreditError::CorruptImage {
                what: "bad superblock magic",
            });
        }
        Ok(Self {
            block_count: read_u32(bytes, Self::BLOCK_COUNT_OFFSET),
            inode_count: read_u32(bytes, Self::INODE_COUNT_OFFSET),
            block_bitmap_block: read_u32(bytes, Self::BB_BLOCK_OFFSET),
            inode_bitmap_block: read_u32(bytes, Self::IB_BLOCK_OFFSET),
            inode_table_block: read_u32(bytes, Self::IT_BLOCK_OFFSET),
            inode_table_blocks: read_u32(bytes, Self::IT_BLOCKS_OFFSET),
            root_inode: read_u32(bytes, Self::ROOT_OFFSET),
            first_free_block: read_u32(bytes, Self::FFB_OFFSET),
            first_free_inode: read_u32(bytes, Self::FFI_OFFSET),
            state: read_u32(bytes, Self::STATE_OFFSET),
        })
    }
}

#[inline]
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..][..4].try_into().unwrap_or([0; 4]))
}

/// An inode: file metadata plus block pointers, stored in the inode
/// table. Serialized little-endian to [`INODE_SIZE`] bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inode {
    pub mode: u16,
    pub size: u32,
    pub block_count: u32,
    pub direct: [u32; DIRECT_COUNT],
    pub indirect: u32,
    pub atime: u32,
    pub mtime: u32,
    pub ctime: u32,
}

const MODE_OFFSET: usize = 0x00;
const SIZE_OFFSET: usize = 0x02;
const BLOCK_COUNT_OFFSET: usize = 0x06;
const DIRECT_OFFSET: usize = 0x0A;
const INDIRECT_OFFSET: usize = 0x36;
const ATIME_OFFSET: usize = 0x3A;
const MTIME_OFFSET: usize = 0x3E;
const CTIME_OFFSET: usize = 0x42;

impl Inode {
    /// `true` when this inode represents a directory.
    #[must_use]
    pub const fn is_directory(&self) -> bool {
        self.mode & 0x4000 != 0
    }

    /// `true` when this inode represents a regular file.
    #[must_use]
    pub const fn is_regular(&self) -> bool {
        self.mode & 0x8000 != 0
    }

    #[must_use]
    pub const fn directory() -> Self {
        Self::with_mode(0x4000)
    }

    #[must_use]
    pub const fn regular() -> Self {
        Self::with_mode(0x8000)
    }

    const fn with_mode(mode: u16) -> Self {
        Self {
            mode,
            size: 0,
            block_count: 0,
            direct: [0; DIRECT_COUNT],
            indirect: 0,
            atime: 0,
            mtime: 0,
            ctime: 0,
        }
    }

    /// Serializes this inode to [`INODE_SIZE`] bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; INODE_SIZE] {
        let mut out = [0u8; INODE_SIZE];
        out[MODE_OFFSET..][..2].copy_from_slice(&self.mode.to_le_bytes());
        out[SIZE_OFFSET..][..4].copy_from_slice(&self.size.to_le_bytes());
        out[BLOCK_COUNT_OFFSET..][..4].copy_from_slice(&self.block_count.to_le_bytes());
        for (i, &block) in self.direct.iter().enumerate() {
            let base = DIRECT_OFFSET + i * 4;
            out[base..][..4].copy_from_slice(&block.to_le_bytes());
        }
        out[INDIRECT_OFFSET..][..4].copy_from_slice(&self.indirect.to_le_bytes());
        out[ATIME_OFFSET..][..4].copy_from_slice(&self.atime.to_le_bytes());
        out[MTIME_OFFSET..][..4].copy_from_slice(&self.mtime.to_le_bytes());
        out[CTIME_OFFSET..][..4].copy_from_slice(&self.ctime.to_le_bytes());
        out
    }

    /// Deserializes [`INODE_SIZE`] bytes into an inode.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut direct = [0u32; DIRECT_COUNT];
        for (i, slot) in direct.iter_mut().enumerate() {
            let base = DIRECT_OFFSET + i * 4;
            *slot = read_u32(bytes, base);
        }
        Self {
            mode: read_u16(bytes, MODE_OFFSET),
            size: read_u32(bytes, SIZE_OFFSET),
            block_count: read_u32(bytes, BLOCK_COUNT_OFFSET),
            direct,
            indirect: read_u32(bytes, INDIRECT_OFFSET),
            atime: read_u32(bytes, ATIME_OFFSET),
            mtime: read_u32(bytes, MTIME_OFFSET),
            ctime: read_u32(bytes, CTIME_OFFSET),
        }
    }
}

#[inline]
fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..][..2].try_into().unwrap_or([0; 2]))
}

/// One directory entry: fixed [`DIRENT_SIZE`] bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirEntry {
    pub inode: u32,
    pub rec_len: u16,
    pub name_len: u8,
    pub file_type: u8,
    pub name: [u8; NAME_MAX],
}

impl DirEntry {
    /// An empty (unused) entry.
    #[must_use]
    pub const fn unused() -> Self {
        Self {
            inode: 0,
            rec_len: DIRENT_SIZE as u16,
            name_len: 0,
            file_type: FileType::Unknown as u8,
            name: [0; NAME_MAX],
        }
    }

    /// Serializes this entry to [`DIRENT_SIZE`] bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; DIRENT_SIZE] {
        let mut out = [0u8; DIRENT_SIZE];
        out[0..][..4].copy_from_slice(&self.inode.to_le_bytes());
        out[4..][..2].copy_from_slice(&self.rec_len.to_le_bytes());
        out[6] = self.name_len;
        out[7] = self.file_type;
        out[8..][..NAME_MAX].copy_from_slice(&self.name);
        out
    }

    /// Deserializes [`DIRENT_SIZE`] bytes into an entry.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            inode: read_u32(bytes, 0),
            rec_len: read_u16(bytes, 4),
            name_len: bytes[6],
            file_type: bytes[7],
            name: bytes[8..][..NAME_MAX]
                .try_into()
                .unwrap_or([0; NAME_MAX]),
        }
    }

    /// The name as a `&str`, if it is valid UTF-8.
    #[must_use]
    pub fn name_str(&self) -> Option<&str> {
        core::str::from_utf8(&self.name[..self.name_len as usize]).ok()
    }
}

/// A filesystem on top of a block device.
///
/// The superblock and both bitmaps are cached in fixed arrays, so the
/// struct itself is heap-free. Inodes and data blocks are read from
/// and written to the device on demand.
///
/// # Examples
///
/// ```
/// use codevar_wredit::error::WreditResult;
/// use codevar_wredit::storage::block::MemoryDisk;
/// use codevar_wredit::storage::fs::{DiskFs, Inode};
///
/// fn create_file() -> WreditResult<()> {
///     let disk = MemoryDisk::new(256)?;
///     let mut fs = DiskFs::format(disk, 256, 128)?;
///     let inode = fs.create(2, "hello.txt", 0x8000)?;
///     fs.write_at(inode, 0, b"world")?;
///     let mut buf = [0u8; 5];
///     assert_eq!(fs.read_at(inode, 0, &mut buf)?, 5);
///     assert_eq!(&buf, b"world");
///     Ok(())
/// }
///
/// assert!(create_file().is_ok());
/// ```
pub struct DiskFs<D: BlockDevice> {
    device: D,
    sb: SuperBlock,
    block_bitmap: Bitmap,
    inode_bitmap: Bitmap,
    sb_dirty: bool,
    bb_dirty: bool,
    ib_dirty: bool,
}

impl<D: BlockDevice> DiskFs<D> {
    /// Formats `device` as an empty Wredit volume, taking ownership of
    /// the device.
    ///
    /// # Errors
    ///
    /// [`WreditError::DiskFull`] when the device is too small to hold
    /// the metadata structures.
    /// [`WreditError::DeviceIo`] when a metadata block cannot be
    /// written.
    pub fn format(device: D, block_count: u64, inode_count: u64) -> WreditResult<Self> {
        let block_count = u32::try_from(block_count)
            .map_err(|_| WreditError::LimitExceeded { what: "block count" })?
            as u64;
        let inode_count = u32::try_from(inode_count)
            .map_err(|_| WreditError::LimitExceeded { what: "inode count" })?
            as u64;
        if block_count == 0 || block_count > (BLOCK_SIZE * 8) as u64 {
            return Err(WreditError::BlockOutOfRange {
                index: block_count,
                count: (BLOCK_SIZE * 8) as u64,
            });
        }
        if inode_count == 0 || inode_count > (BLOCK_SIZE * 8) as u64 {
            return Err(WreditError::BlockOutOfRange {
                index: inode_count,
                count: (BLOCK_SIZE * 8) as u64,
            });
        }
        // block 0: superblock, 1: block bitmap, 2: inode bitmap,
        // 3..: inode table, then data blocks.
        let inode_table_blocks = ((inode_count as usize)
            .saturating_mul(INODE_SIZE)
            .saturating_add(BLOCK_SIZE - 1))
            / BLOCK_SIZE;
        let metadata_blocks = 3u64.saturating_add(inode_table_blocks as u64);
        if block_count <= metadata_blocks {
            return Err(WreditError::DiskFull { free: 0 });
        }

        let mut block_bitmap = Bitmap::new();
        let mut inode_bitmap = Bitmap::new();
        // Mark the metadata blocks used.
        for b in 0..metadata_blocks {
            block_bitmap.set_used(b as u32);
        }

        let root_block = metadata_blocks; // first data block holds the root directory
        block_bitmap.set_used(root_block as u32);

        let sb = SuperBlock {
            block_count: block_count as u32,
            inode_count: inode_count as u32,
            block_bitmap_block: 1,
            inode_bitmap_block: 2,
            inode_table_block: 3,
            inode_table_blocks: inode_table_blocks as u32,
            root_inode: 2, // like ext2: inode 1 reserved, root is 2
            first_free_block: (root_block as u32) + 1,
            first_free_inode: 3, // inode 2 is root
            state: 1,            // clean
        };

        // Root directory inode.
        let mut root = Inode::directory();
        root.direct[0] = root_block as u32;
        root.block_count = 1;
        root.size = (2 * DIRENT_SIZE) as u32; // "." and ".."
        // Inodes 0 and 1 are reserved (like ext2); inode 2 is root.
        inode_bitmap.set_used(0);
        inode_bitmap.set_used(1);
        inode_bitmap.set_used(2);

        let mut fs = Self {
            device,
            sb,
            block_bitmap,
            inode_bitmap,
            sb_dirty: true,
            bb_dirty: true,
            ib_dirty: true,
        };
        // Write the root directory block with "." and "..".
        let mut dir_block = [0u8; BLOCK_SIZE];
        let mut dot = DirEntry::unused();
        dot.inode = 2;
        dot.rec_len = DIRENT_SIZE as u16;
        dot.name_len = 1;
        dot.file_type = FileType::Directory as u8;
        dot.name[0] = b'.';
        let mut dotdot = DirEntry::unused();
        dotdot.inode = 2;
        dotdot.rec_len = DIRENT_SIZE as u16;
        dotdot.name_len = 2;
        dotdot.file_type = FileType::Directory as u8;
        dotdot.name[0] = b'.';
        dotdot.name[1] = b'.';
        dir_block[0..DIRENT_SIZE].copy_from_slice(&dot.to_bytes());
        dir_block[DIRENT_SIZE..2 * DIRENT_SIZE].copy_from_slice(&dotdot.to_bytes());
        fs.device.write_block(root_block, &dir_block)?;
        // Write the root inode into the inode table (inode 2).
        fs.write_inode_raw(2, &root)?;
        // Flush all metadata.
        fs.flush()?;
        Ok(fs)
    }

    /// Opens an existing Wredit volume from `device`, taking ownership.
    ///
    /// # Errors
    ///
    /// [`WreditError::CorruptImage`] when the superblock magic is bad
    /// or the geometry is inconsistent.
    /// [`WreditError::DeviceIo`] when a metadata block cannot be read.
    pub fn open(device: D) -> WreditResult<Self> {
        let mut sb_block = [0u8; BLOCK_SIZE];
        device.read_block(0, &mut sb_block)?;
        let sb = SuperBlock::from_bytes(&sb_block)?;
        if sb.block_count == 0
            || sb.block_count > (BLOCK_SIZE * 8) as u32
            || sb.inode_count == 0
            || sb.inode_count > (BLOCK_SIZE * 8) as u32
        {
            return Err(WreditError::CorruptImage {
                what: "bad superblock geometry",
            });
        }
        let mut bb_block = [0u8; BLOCK_SIZE];
        device.read_block(sb.block_bitmap_block.into(), &mut bb_block)?;
        let block_bitmap = Bitmap::from_bytes(&bb_block)?;
        let mut ib_block = [0u8; BLOCK_SIZE];
        device.read_block(sb.inode_bitmap_block.into(), &mut ib_block)?;
        let inode_bitmap = Bitmap::from_bytes(&ib_block)?;
        Ok(Self {
            device,
            sb,
            block_bitmap,
            inode_bitmap,
            sb_dirty: false,
            bb_dirty: false,
            ib_dirty: false,
        })
    }

    /// Writes all dirty metadata (superblock and bitmaps) to the device.
    pub fn flush(&mut self) -> WreditResult<()> {
        if self.sb_dirty {
            self.device.write_block(0, &self.sb.to_bytes())?;
            self.sb_dirty = false;
        }
        if self.bb_dirty {
            self.device.write_block(
                self.sb.block_bitmap_block.into(),
                &self
                    .block_bitmap
                    .as_bytes()
                    .try_into()
                    .unwrap_or([0; BLOCK_SIZE]),
            )?;
            self.bb_dirty = false;
        }
        if self.ib_dirty {
            self.device.write_block(
                self.sb.inode_bitmap_block.into(),
                &self
                    .inode_bitmap
                    .as_bytes()
                    .try_into()
                    .unwrap_or([0; BLOCK_SIZE]),
            )?;
            self.ib_dirty = false;
        }
        Ok(())
    }

    /// Consumes the filesystem, flushing metadata, and returns the
    /// underlying device. Useful for re-opening an image in tests.
    pub fn into_device(mut self) -> D {
        let _ = self.flush();
        let this = core::mem::ManuallyDrop::new(self);
        // SAFETY: `ManuallyDrop` prevents `DiskFs::drop` from running,
        // so the device's heap buffer is moved out exactly once.
        unsafe { core::ptr::read(&this.device) }
    }

    /// Allocates a free data block.
    pub fn alloc_block(&mut self) -> WreditResult<u32> {
        let block = self
            .block_bitmap
            .find_free(self.sb.block_count)
            .ok_or(WreditError::DiskFull {
                free: self.block_bitmap.count_free(self.sb.block_count) as u64,
            })?;
        self.block_bitmap.set_used(block);
        self.bb_dirty = true;
        if block + 1 > self.sb.first_free_block {
            self.sb.first_free_block = block + 1;
            self.sb_dirty = true;
        }
        Ok(block)
    }

    /// Frees a data block.
    pub fn free_block(&mut self, block: u32) -> WreditResult<()> {
        if block >= self.sb.block_count {
            return Err(WreditError::BlockOutOfRange {
                index: block as u64,
                count: self.sb.block_count as u64,
            });
        }
        self.block_bitmap.set_free(block);
        self.bb_dirty = true;
        if block < self.sb.first_free_block {
            self.sb.first_free_block = block;
            self.sb_dirty = true;
        }
        Ok(())
    }

    /// Allocates a free inode.
    pub fn alloc_inode(&mut self) -> WreditResult<u32> {
        let inode = self
            .inode_bitmap
            .find_free(self.sb.inode_count)
            .ok_or(WreditError::DiskFull {
                free: self.inode_bitmap.count_free(self.sb.inode_count) as u64,
            })?;
        self.inode_bitmap.set_used(inode);
        self.ib_dirty = true;
        if inode + 1 > self.sb.first_free_inode {
            self.sb.first_free_inode = inode + 1;
            self.sb_dirty = true;
        }
        Ok(inode)
    }

    /// Frees an inode.
    pub fn free_inode(&mut self, inode: u32) -> WreditResult<()> {
        if inode == 0 || inode > self.sb.inode_count {
            return Err(WreditError::InvalidAddress {
                address: inode as u64,
            });
        }
        self.inode_bitmap.set_free(inode);
        self.ib_dirty = true;
        if inode < self.sb.first_free_inode {
            self.sb.first_free_inode = inode;
            self.sb_dirty = true;
        }
        Ok(())
    }

    /// Reads an inode from the table.
    pub fn read_inode(&self, inode: u32) -> WreditResult<Inode> {
        if inode == 0 || inode > self.sb.inode_count {
            return Err(WreditError::NotFound { what: "inode" });
        }
        let index = (inode - 1) as usize;
        let block = self.sb.inode_table_block + (index / INODES_PER_BLOCK) as u32;
        let offset = (index % INODES_PER_BLOCK) * INODE_SIZE;
        let mut block_bytes = [0u8; BLOCK_SIZE];
        self.device
            .read_block(block.into(), &mut block_bytes)?;
        let inode_bytes = &block_bytes[offset..offset + INODE_SIZE];
        Ok(Inode::from_bytes(inode_bytes))
    }

    /// Writes an inode to the table.
    pub fn write_inode(&mut self, inode: u32, value: &Inode) -> WreditResult<()> {
        if inode == 0 || inode > self.sb.inode_count {
            return Err(WreditError::InvalidAddress {
                address: inode as u64,
            });
        }
        self.write_inode_raw(inode, value)
    }

    fn write_inode_raw(&mut self, inode: u32, value: &Inode) -> WreditResult<()> {
        let index = (inode - 1) as usize;
        let block = self.sb.inode_table_block + (index / INODES_PER_BLOCK) as u32;
        let offset = (index % INODES_PER_BLOCK) * INODE_SIZE;
        let mut block_bytes = [0u8; BLOCK_SIZE];
        self.device
            .read_block(block.into(), &mut block_bytes)?;
        block_bytes[offset..offset + INODE_SIZE].copy_from_slice(&value.to_bytes());
        self.device
            .write_block(block.into(), &block_bytes)?;
        Ok(())
    }

    /// Resolves logical block `logical` of `ino` to a physical block,
    /// allocating blocks (and the indirect block) when `allocate` is
    /// true.
    fn block_for(&mut self, ino: &mut Inode, logical: u32, allocate: bool) -> WreditResult<u32> {
        if logical < DIRECT_COUNT as u32 {
            let slot = logical as usize;
            if ino.direct[slot] != 0 {
                return Ok(ino.direct[slot]);
            }
            if !allocate {
                return Err(WreditError::BlockNotFound {
                    address: logical as u64,
                });
            }
            let block = self.alloc_block()?;
            ino.direct[slot] = block;
            ino.block_count = ino.block_count.saturating_add(1);
            return Ok(block);
        }
        // Indirect.
        let idx = (logical - DIRECT_COUNT as u32) as usize;
        if idx >= INDIRECT_PTRS {
            return Err(WreditError::LimitExceeded {
                what: "inode block pointers",
            });
        }
        if ino.indirect == 0 {
            if !allocate {
                return Err(WreditError::BlockNotFound {
                    address: logical as u64,
                });
            }
            let indirect_block = self.alloc_block()?;
            ino.indirect = indirect_block;
            ino.block_count = ino.block_count.saturating_add(1);
            let mut block_bytes = [0u8; BLOCK_SIZE];
            self.device
                .read_block(indirect_block.into(), &mut block_bytes)?;
            let data = self.alloc_block()?;
            block_bytes[idx * 4..][..4].copy_from_slice(&data.to_le_bytes());
            self.device
                .write_block(indirect_block.into(), &block_bytes)?;
            return Ok(data);
        }
        let mut block_bytes = [0u8; BLOCK_SIZE];
        self.device
            .read_block(ino.indirect.into(), &mut block_bytes)?;
        let data = read_u32(&block_bytes, idx * 4);
        if data != 0 {
            return Ok(data);
        }
        if !allocate {
            return Err(WreditError::BlockNotFound {
                address: logical as u64,
            });
        }
        let block = self.alloc_block()?;
        block_bytes[idx * 4..][..4].copy_from_slice(&block.to_le_bytes());
        self.device
            .write_block(ino.indirect.into(), &block_bytes)?;
        Ok(block)
    }

    /// Reads up to `buf.len()` bytes from `inode` at `offset`.
    pub fn read_at(&mut self, inode: u32, offset: u64, buf: &mut [u8]) -> WreditResult<usize> {
        let mut ino = self.read_inode(inode)?;
        if offset >= ino.size as u64 || buf.is_empty() {
            return Ok(0);
        }
        let want = buf
            .len()
            .min((ino.size as u64).saturating_sub(offset) as usize);
        let mut read = 0usize;
        let mut block_buf = [0u8; BLOCK_SIZE];
        while read < want {
            let logical = ((offset as usize + read) / BLOCK_SIZE) as u32;
            let block = self.block_for(&mut ino, logical, false)?;
            self.device
                .read_block(block.into(), &mut block_buf)?;
            let block_start = (logical as usize) * BLOCK_SIZE;
            let skip = (offset as usize + read) - block_start;
            let take = want.min(BLOCK_SIZE - skip).min(want - read);
            buf[read..read + take].copy_from_slice(&block_buf[skip..skip + take]);
            read += take;
        }
        Ok(read)
    }

    /// Writes `buf` to `inode` at `offset`, allocating blocks as
    /// needed. Returns the number of bytes written.
    pub fn write_at(&mut self, inode: u32, offset: u64, buf: &[u8]) -> WreditResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut ino = self.read_inode(inode)?;
        let mut written = 0usize;
        let mut block_buf = [0u8; BLOCK_SIZE];
        while written < buf.len() {
            let logical = ((offset as usize + written) / BLOCK_SIZE) as u32;
            let block = self.block_for(&mut ino, logical, true)?;
            let block_start = (logical as usize) * BLOCK_SIZE;
            let skip = (offset as usize + written) - block_start;
            let take = (buf.len() - written).min(BLOCK_SIZE - skip);
            if skip == 0 && take == BLOCK_SIZE {
                // Full-block write: no need to read first.
                self.device
                    .write_block(block.into(), &buf[written..written + take])?;
            } else {
                self.device
                    .read_block(block.into(), &mut block_buf)?;
                block_buf[skip..skip + take].copy_from_slice(&buf[written..written + take]);
                self.device
                    .write_block(block.into(), &block_buf)?;
            }
            written += take;
        }
        let new_size = (offset as usize + written) as u32;
        if new_size > ino.size {
            ino.size = new_size;
        }
        // Recompute block_count from highest allocated logical block.
        let last_logical = ((offset as usize + written - 1) / BLOCK_SIZE) as u32;
        let mut blocks_needed = last_logical + 1;
        if blocks_needed > DIRECT_COUNT as u32 + INDIRECT_PTRS as u32 {
            blocks_needed = DIRECT_COUNT as u32 + INDIRECT_PTRS as u32;
        }
        ino.block_count = blocks_needed;
        self.write_inode(inode, &ino)?;
        Ok(written)
    }

    /// Reads a directory block's entries and finds `name`, returning
    /// its inode.
    fn find_dirent(&mut self, dir_inode: u32, name: &str) -> WreditResult<u32> {
        let ino = self.read_inode(dir_inode)?;
        if !ino.is_directory() {
            return Err(WreditError::NotADirectory {
                what: "path component",
            });
        }
        let name_bytes = name.as_bytes();
        if name_bytes.len() > NAME_MAX {
            return Err(WreditError::NameTooLong {
                len: name_bytes.len(),
            });
        }
        let mut block_buf = [0u8; BLOCK_SIZE];
        let blocks = (ino.size as usize).div_ceil(BLOCK_SIZE);
        for b in 0..blocks {
            let logical = b as u32;
            let block = match self.block_for_direct(&ino, logical) {
                Some(block) => block,
                None => continue,
            };
            self.device
                .read_block(block.into(), &mut block_buf)?;
            for e in 0..DIRENTS_PER_BLOCK {
                let entry = DirEntry::from_bytes(&block_buf[e * DIRENT_SIZE..][..DIRENT_SIZE]);
                if entry.inode == 0 {
                    continue;
                }
                if entry.name_len as usize == name_bytes.len()
                    && entry.name[..name_bytes.len()] == name_bytes[..]
                {
                    return Ok(entry.inode);
                }
            }
        }
        Err(WreditError::NotFound {
            what: "directory entry",
        })
    }

    /// Resolves a logical block to a physical block without allocating.
    fn block_for_direct(&self, ino: &Inode, logical: u32) -> Option<u32> {
        if logical < DIRECT_COUNT as u32 {
            let block = ino.direct[logical as usize];
            return (block != 0).then_some(block);
        }
        if ino.indirect == 0 {
            return None;
        }
        let idx = (logical - DIRECT_COUNT as u32) as usize;
        if idx >= INDIRECT_PTRS {
            return None;
        }
        let mut block_bytes = [0u8; BLOCK_SIZE];
        self.device
            .read_block(ino.indirect.into(), &mut block_bytes)
            .ok()?;
        let block = read_u32(&block_bytes, idx * 4);
        (block != 0).then_some(block)
    }

    /// Records that `ino` now uses physical `block` at logical slot
    /// `logical` (used when appending fresh blocks).
    fn block_for_direct_alloc(&mut self, ino: &mut Inode, logical: u32, block: u32) -> WreditResult<()> {
        if logical < DIRECT_COUNT as u32 {
            ino.direct[logical as usize] = block;
            ino.block_count = ino.block_count.saturating_add(1);
            return Ok(());
        }
        if ino.indirect == 0 {
            return Err(WreditError::InvalidOperation {
                what: "no indirect block to append into",
            });
        }
        let idx = (logical - DIRECT_COUNT as u32) as usize;
        let mut block_bytes = [0u8; BLOCK_SIZE];
        self.device
            .read_block(ino.indirect.into(), &mut block_bytes)?;
        block_bytes[idx * 4..][..4].copy_from_slice(&block.to_le_bytes());
        self.device
            .write_block(ino.indirect.into(), &block_bytes)?;
        Ok(())
    }

    /// Appends a directory entry `name -> inode` to `parent`.
    fn add_dirent(&mut self, parent: u32, name: &str, inode: u32, file_type: FileType) -> WreditResult<()> {
        let mut ino = self.read_inode(parent)?;
        if !ino.is_directory() {
            return Err(WreditError::NotADirectory {
                what: "parent directory",
            });
        }
        let name_bytes = name.as_bytes();
        if name_bytes.len() > NAME_MAX {
            return Err(WreditError::NameTooLong {
                len: name_bytes.len(),
            });
        }
        // Find a free slot or append a new block.
        let mut block_buf = [0u8; BLOCK_SIZE];
        let blocks = (ino.size as usize).div_ceil(BLOCK_SIZE);
        let mut free_entry_index: Option<(u32, usize)> = None;
        for b in 0..blocks {
            let logical = b as u32;
            let block = match self.block_for_direct(&ino, logical) {
                Some(block) => block,
                None => {
                    // Allocate a fresh block.
                    let block = self.alloc_block()?;
                    self.block_for_direct_alloc(&mut ino, logical, block)?;
                    block
                }
            };
            self.device
                .read_block(block.into(), &mut block_buf)?;
            for e in 0..DIRENTS_PER_BLOCK {
                let entry = DirEntry::from_bytes(&block_buf[e * DIRENT_SIZE..][..DIRENT_SIZE]);
                if entry.inode == 0 {
                    free_entry_index = Some((block, e));
                    break;
                }
            }
            if free_entry_index.is_some() {
                break;
            }
        }
        let (block, entry_index) = match free_entry_index {
            Some(v) => v,
            None => {
                let block = self.alloc_block()?;
                self.block_for_direct_alloc(&mut ino, blocks as u32, block)?;
                ino.size = (blocks as u32 + 1) * BLOCK_SIZE as u32;
                ino.block_count = blocks as u32 + 1;
                (block, 0)
            }
        };
        self.device
            .read_block(block.into(), &mut block_buf)?;
        let mut entry = DirEntry::unused();
        entry.inode = inode;
        entry.rec_len = DIRENT_SIZE as u16;
        entry.name_len = name_bytes.len() as u8;
        entry.file_type = file_type as u8;
        entry.name[..name_bytes.len()].copy_from_slice(name_bytes);
        block_buf[entry_index * DIRENT_SIZE..][..DIRENT_SIZE].copy_from_slice(&entry.to_bytes());
        self.device
            .write_block(block.into(), &block_buf)?;
        self.write_inode(parent, &ino)?;
        Ok(())
    }

    /// Resolves `path` starting at `start_inode`, returning the target
    /// inode.
    ///
    /// `.` and `..` are handled; `..` walks the parent stack. An empty
    /// path or `/` resolves to `start_inode`.
    pub fn lookup(&mut self, path: &str, start_inode: u32) -> WreditResult<u32> {
        let mut current = start_inode;
        let mut dir_stack: [u32; DEPTH_MAX] = [0; DEPTH_MAX];
        let mut depth = 0usize;
        let mut rest = path;
        loop {
            rest = rest.trim_start_matches('/');
            if rest.is_empty() {
                break;
            }
            let (comp, after) = match rest.find('/') {
                Some(pos) => (&rest[..pos], &rest[pos + 1..]),
                None => (rest, ""),
            };
            if comp == "." {
                rest = after;
                continue;
            }
            if comp == ".." {
                if depth > 0 {
                    depth -= 1;
                    current = dir_stack[depth];
                }
                rest = after;
                continue;
            }
            let next = self.find_dirent(current, comp)?;
            let next_ino = self.read_inode(next)?;
            if after.is_empty() {
                current = next;
                break;
            }
            if !next_ino.is_directory() {
                return Err(WreditError::NotADirectory {
                    what: "path component",
                });
            }
            if depth >= DEPTH_MAX {
                return Err(WreditError::LimitExceeded {
                    what: "directory nesting",
                });
            }
            dir_stack[depth] = current;
            depth += 1;
            current = next;
            rest = after;
        }
        Ok(current)
    }

    /// Creates `name` inside `parent` with `mode`, returning its new
    /// inode.
    pub fn create(&mut self, parent: u32, name: &str, mode: u16) -> WreditResult<u32> {
        // Reject names that already exist.
        if self.find_dirent(parent, name).is_ok() {
            return Err(WreditError::AlreadyExists { what: "name" });
        }
        let inode = self.alloc_inode()?;
        let new_ino = if mode & 0x4000 != 0 {
            Inode::directory()
        } else {
            Inode::regular()
        };
        self.write_inode(inode, &new_ino)?;
        let file_type = if new_ino.is_directory() {
            FileType::Directory
        } else {
            FileType::Regular
        };
        self.add_dirent(parent, name, inode, file_type)?;
        Ok(inode)
    }

    /// Makes `name` a directory inside `parent`.
    pub fn mkdir(&mut self, parent: u32, name: &str) -> WreditResult<u32> {
        self.create(parent, name, 0x4000)
    }

    /// Removes the directory entry `name` from `parent` without
    /// freeing the inode or its blocks.
    pub fn unlink(&mut self, parent: u32, name: &str) -> WreditResult<()> {
        let target = self.find_dirent(parent, name)?;
        let ino = self.read_inode(parent)?;
        let name_bytes = name.as_bytes();
        let mut block_buf = [0u8; BLOCK_SIZE];
        let blocks = (ino.size as usize).div_ceil(BLOCK_SIZE);
        for b in 0..blocks {
            let logical = b as u32;
            let block = match self.block_for_direct(&ino, logical) {
                Some(block) => block,
                None => continue,
            };
            self.device
                .read_block(block.into(), &mut block_buf)?;
            for e in 0..DIRENTS_PER_BLOCK {
                let off = e * DIRENT_SIZE;
                let entry = DirEntry::from_bytes(&block_buf[off..][..DIRENT_SIZE]);
                if entry.inode == target
                    && entry.name_len as usize == name_bytes.len()
                    && entry.name[..name_bytes.len()] == name_bytes[..]
                {
                    let mut blank = DirEntry::unused();
                    blank.rec_len = DIRENT_SIZE as u16;
                    block_buf[off..][..DIRENT_SIZE].copy_from_slice(&blank.to_bytes());
                    self.device
                        .write_block(block.into(), &block_buf)?;
                    return Ok(());
                }
            }
        }
        Err(WreditError::NotFound {
            what: "directory entry",
        })
    }

    /// The number of blocks the device holds.
    #[must_use]
    pub fn block_count(&self) -> u32 {
        self.sb.block_count
    }

    /// The root directory inode.
    #[must_use]
    pub const fn root_inode(&self) -> u32 {
        self.sb.root_inode
    }
}

impl<D: BlockDevice> Drop for DiskFs<D> {
    fn drop(&mut self) {
        // Best-effort flush of metadata on drop; ignore errors since
        // destructors must not panic.
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::{BLOCK_SIZE, DiskFs, NAME_MAX};
    use crate::error::WreditError;
    use crate::storage::block::MemoryDisk;

    /// Justification: `unwrap` in tests is permitted by the project
    /// lint configuration (`allow-unwrap-in-tests`).

    #[test]
    fn format_and_round_trip_a_file() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        assert_eq!(fs.root_inode(), 2);
        let inode = fs.create(2, "hello.txt", 0x8000).unwrap();
        assert_eq!(fs.write_at(inode, 0, b"world").unwrap(), 5);
        let mut buf = [0u8; 5];
        assert_eq!(fs.read_at(inode, 0, &mut buf).unwrap(), 5);
        assert_eq!(&buf, b"world");
    }

    #[test]
    fn directories_and_lookup() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        fs.mkdir(2, "src").unwrap();
        let src = fs.lookup("src", 2).unwrap();
        assert!(fs.read_inode(src).unwrap().is_directory());
        fs.create(src, "main.k", 0x8000).unwrap();
        let main_k = fs.lookup("src/main.k", 2).unwrap();
        fs.write_at(main_k, 0, b"fn main() {}").unwrap();
        let mut buf = [0u8; 12];
        assert_eq!(fs.read_at(main_k, 0, &mut buf).unwrap(), 12);
        assert_eq!(&buf, b"fn main() {}");
        // `..` from src walks back to root.
        assert_eq!(fs.lookup("src/..", 2).unwrap(), 2);
    }

    #[test]
    fn duplicate_and_missing_names() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        fs.create(2, "a.txt", 0x8000).unwrap();
        assert!(matches!(
            fs.create(2, "a.txt", 0x8000),
            Err(WreditError::AlreadyExists { .. })
        ));
        assert!(matches!(
            fs.lookup("nope.txt", 2),
            Err(WreditError::NotFound { .. })
        ));
    }

    #[test]
    fn writing_past_a_block_allocates_and_reads_back() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let inode = fs.create(2, "big.txt", 0x8000).unwrap();
        let data = [0x42u8; BLOCK_SIZE + 16];
        assert_eq!(fs.write_at(inode, 0, &data).unwrap(), data.len());
        let mut buf = [0u8; BLOCK_SIZE + 16];
        assert_eq!(fs.read_at(inode, 0, &mut buf).unwrap(), data.len());
        assert_eq!(&buf, &data);
    }

    #[test]
    fn reopening_an_image_preserves_data() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let inode = fs.create(2, "persist.txt", 0x8000).unwrap();
        fs.write_at(inode, 0, b"durable").unwrap();
        let disk = fs.into_device();
        let mut fs2 = DiskFs::open(disk).unwrap();
        assert_eq!(fs2.root_inode(), 2);
        let inode = fs2.lookup("persist.txt", 2).unwrap();
        let mut buf = [0u8; 7];
        assert_eq!(fs2.read_at(inode, 0, &mut buf).unwrap(), 7);
        assert_eq!(&buf, b"durable");
    }

    #[test]
    fn unlink_removes_the_entry() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        fs.create(2, "tmp.txt", 0x8000).unwrap();
        fs.unlink(2, "tmp.txt").unwrap();
        assert!(matches!(
            fs.lookup("tmp.txt", 2),
            Err(WreditError::NotFound { .. })
        ));
    }

    #[test]
    fn name_too_long_is_rejected() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        let long = "a".repeat(NAME_MAX + 1);
        assert!(matches!(
            fs.create(2, &long, 0x8000),
            Err(WreditError::NameTooLong { .. })
        ));
    }
}
