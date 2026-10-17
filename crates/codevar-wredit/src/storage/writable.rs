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

//! Writable (file) layer over the block filesystem.
//!
//! [`WritableFile`] gives the engine a byte-oriented view of a stored
//! regular file. Writes accumulate into a fixed-size stack cache of
//! `STAGING` bytes — the engine's write-back buffer — which is flushed to
//! the underlying [`DiskFs`] blocks when it fills, on an explicit
//! [`WritableFile::flush`], or on [`WritableFile::sync_all`]. Because the
//! cache lives inline (`[u8; STAGING]`), a writable file never allocates
//! on the heap.

use super::block::BlockDevice;
use super::fs::DiskFs;
use crate::error::{WreditError, WreditResult};

/// A buffered, seekable view of one regular file.
///
/// The write cache is a stack array of `STAGING` bytes. Full cache lines
/// are written straight through to the file's blocks; a trailing partial
/// line is held until [`Self::flush`].
///
/// # Type parameters
///
/// * `D`       - the underlying block device
/// * `STAGING` - write-cache size in bytes; must be greater than zero
///
/// # Examples
///
/// ```
/// use codevar_wredit::storage::block::MemoryDisk;
/// use codevar_wredit::storage::fs::DiskFs;
/// use codevar_wredit::storage::writable::WritableFile;
///
/// fn write_file() -> codevar_wredit::error::WreditResult<()> {
///     let disk = MemoryDisk::new(256)?;
///     let mut fs = DiskFs::format(disk, 256, 128)?;
///     let inode = fs.create(2, "notes.txt", 0x8000)?;
///     let mut file = WritableFile::<MemoryDisk, 64>::new(&mut fs, inode)?;
///     file.write(b"hello")?;
///     file.write(b" world")?;
///     file.flush()?;
///     let mut buf = [0u8; 11];
///     file.set_position(0)?;
///     assert_eq!(file.read(&mut buf)?, 11);
///     assert_eq!(&buf, b"hello world");
///     Ok(())
/// }
///
/// assert!(write_file().is_ok());
/// ```
pub struct WritableFile<'a, D: BlockDevice, const STAGING: usize> {
    /// The filesystem the file's blocks live in.
    fs: &'a mut DiskFs<D>,
    /// Inode of the open file.
    inode: u32,
    /// Logical byte offset of the next read or write.
    position: u64,
    /// Stack write cache of unwritten bytes.
    stage: [u8; STAGING],
    /// Number of valid bytes currently in `stage`.
    stage_len: usize,
}

impl<'a, D: BlockDevice, const STAGING: usize> WritableFile<'a, D, STAGING> {
    /// Opens `inode` for buffered writing.
    ///
    /// # Errors
    ///
    /// * [`WreditError::InvalidGapSize`] when `STAGING` is zero.
    /// * [`WreditError::IsADirectory`] when `inode` is a directory.
    /// * Propagates any device I/O error from the inode read.
    pub fn new(fs: &'a mut DiskFs<D>, inode: u32) -> WreditResult<Self> {
        if STAGING == 0 {
            return Err(WreditError::InvalidGapSize { size: 0 });
        }
        let ino = fs.read_inode(inode)?;
        if !ino.is_regular() {
            return Err(WreditError::IsADirectory {
                what: "writable file target",
            });
        }
        Ok(Self {
            fs,
            inode,
            position: 0,
            stage: [0; STAGING],
            stage_len: 0,
        })
    }

    /// Inode backing this writable file.
    #[must_use]
    pub const fn inode(&self) -> u32 {
        self.inode
    }

    /// Current logical byte position.
    #[must_use]
    pub const fn position(&self) -> u64 {
        self.position
    }

    /// Moves the logical position to `offset`.
    ///
    /// Pending cached bytes are flushed first so the cache and position
    /// stay consistent.
    ///
    /// # Errors
    ///
    /// Propagates a device I/O error from the flush.
    pub fn set_position(&mut self, offset: u64) -> WreditResult<()> {
        self.flush()?;
        self.position = offset;
        Ok(())
    }

    /// Appends `buf` to the file at the current position.
    ///
    /// Bytes are staged in the stack cache; complete `STAGING`-sized runs
    /// are written through immediately.
    ///
    /// # Returns
    ///
    /// The number of bytes accepted, always `buf.len()`.
    ///
    /// # Errors
    ///
    /// Propagates a device I/O error from any write-through.
    pub fn write(&mut self, buf: &[u8]) -> WreditResult<usize> {
        let mut written = 0usize;
        while written < buf.len() {
            let space = STAGING - self.stage_len;
            let take = (buf.len() - written).min(space);
            self.stage[self.stage_len..self.stage_len + take]
                .copy_from_slice(&buf[written..written + take]);
            self.stage_len += take;
            self.position = self.position.saturating_add(take as u64);
            written += take;
            if self.stage_len == STAGING {
                self.flush_cache()?;
            }
        }
        Ok(written)
    }

    /// Reads up to `buf.len()` bytes starting at the current position.
    ///
    /// Any cached bytes at the current position are flushed first so reads
    /// observe the most recent writes.
    ///
    /// # Returns
    ///
    /// The number of bytes read, which may be less than `buf.len()` at the
    /// end of the file.
    ///
    /// # Errors
    ///
    /// Propagates a device I/O error from the underlying read.
    pub fn read(&mut self, buf: &mut [u8]) -> WreditResult<usize> {
        self.flush()?;
        let size = u64::from(self.fs.read_inode(self.inode)?.size);
        let available = size.saturating_sub(self.position) as usize;
        let want = buf.len().min(available);
        self.fs.read_at(self.inode, self.position, &mut buf[..want])?;
        self.position = self.position.saturating_add(want as u64);
        Ok(want)
    }

    /// Writes any staged bytes through to the file's blocks.
    ///
    /// # Errors
    ///
    /// Propagates a device I/O error from the write.
    pub fn flush(&mut self) -> WreditResult<()> {
        if self.stage_len == 0 {
            return Ok(());
        }
        self.flush_cache()
    }

    /// Flushes staged bytes and forces the filesystem metadata to disk.
    ///
    /// # Errors
    ///
    /// Propagates any device I/O error from the cache or metadata flush.
    pub fn sync_all(&mut self) -> WreditResult<()> {
        self.flush()?;
        self.fs.flush()
    }

    /// Writes the whole cache to the file at `position - stage_len`.
    fn flush_cache(&mut self) -> WreditResult<()> {
        let len = self.stage_len;
        if len == 0 {
            return Ok(());
        }
        let start = self.position.saturating_sub(len as u64);
        self.fs.write_at(self.inode, start, &self.stage[..len])?;
        self.stage[..len].fill(0);
        self.stage_len = 0;
        Ok(())
    }
}

impl<'a, D: BlockDevice, const STAGING: usize> Drop for WritableFile<'a, D, STAGING> {
    /// Best-effort flush so staged bytes are not silently dropped.
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::WritableFile;
    use crate::error::WreditResult;
    use crate::storage::block::MemoryDisk;
    use crate::storage::fs::DiskFs;

    /// Justification: `unwrap` in tests is permitted by the project
    /// lint configuration (`allow-unwrap-in-tests`).

    #[test]
    fn write_then_flush_persists_to_blocks() -> WreditResult<()> {
        let disk = MemoryDisk::new(256)?;
        let mut fs = DiskFs::format(disk, 256, 128)?;
        let inode = fs.create(2, "notes.txt", 0x8000)?;
        {
            let mut file = WritableFile::<MemoryDisk, 64>::new(&mut fs, inode)?;
            assert_eq!(file.write(b"hello world")?, 11);
            file.flush()?;
        }
        let mut buf = [0u8; 11];
        assert_eq!(fs.read_at(inode, 0, &mut buf)?, 11);
        assert_eq!(&buf, b"hello world");
        Ok(())
    }

    #[test]
    fn staging_cache_flushes_when_full() -> WreditResult<()> {
        let disk = MemoryDisk::new(256)?;
        let mut fs = DiskFs::format(disk, 256, 128)?;
        let inode = fs.create(2, "chunked.txt", 0x8000)?;
        let payload = [0x41u8; 20];
        {
            let mut file = WritableFile::<MemoryDisk, 8>::new(&mut fs, inode)?;
            assert_eq!(file.write(&payload)?, 20);
            file.sync_all()?;
            let mut readback = [0u8; 20];
            file.set_position(0)?;
            assert_eq!(file.read(&mut readback)?, 20);
            assert_eq!(&readback, &payload);
        }
        Ok(())
    }

    #[test]
    fn directory_target_is_rejected() {
        let disk = MemoryDisk::new(256).unwrap();
        let mut fs = DiskFs::format(disk, 256, 128).unwrap();
        fs.mkdir(2, "src").unwrap();
        let dir = fs.lookup("src", 2).unwrap();
        assert!(matches!(
            WritableFile::<MemoryDisk, 8>::new(&mut fs, dir),
            Err(crate::error::WreditError::IsADirectory { .. })
        ));
    }
}
