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

//! The block device the engine storage sits on.
//!
//! Real disks expose a fixed-size block interface; the engine mirrors
//! it with the [`BlockDevice`] trait: reads and writes always transfer
//! exactly one [`BLOCK_SIZE`]-byte block, and out-of-range indices are
//! rejected with [`WreditError::BlockOutOfRange`] — the filesystem
//! layer above never sees a short or torn transfer.
//!
//! [`MemoryDisk`] is the reference implementation: a flat, zeroed
//! array of blocks that lets the whole storage stack run in RAM (and
//! in `no_std` environments) without touching the host filesystem.

use alloc::vec::Vec;

use crate::error::{WreditError, WreditResult};

/// Bytes per block — the allocation unit of the whole storage layer.
///
/// Matches the default block size of mainstream disk file systems
/// (ext, NTFS on most volumes), which keeps metadata overhead small
/// and lets SIMD kernels operate on whole blocks.
pub const BLOCK_SIZE: usize = 4096;

/// `log2(`[`BLOCK_SIZE`]`)` — block/byte conversions shift by this.
pub const BLOCK_SHIFT: u32 = 12;

/// Largest device the bitmaps can address: one bit per block.
pub const MAX_BLOCKS: u64 = (BLOCK_SIZE * 8) as u64;

/// A single fixed-size block.
pub type Block = [u8; BLOCK_SIZE];

/// The block interface of a storage device.
///
/// Implementations must treat `index` as authoritative: an index past
/// [`BlockDevice::block_count`] is [`WreditError::BlockOutOfRange`],
/// never a wrap or a silent clamp.
pub trait BlockDevice {
    /// Number of blocks the device holds.
    fn block_count(&self) -> u64;

    /// Copies block `index` into `out`, which must be exactly one
    /// block long.
    ///
    /// # Errors
    ///
    /// * [`WreditError::BlockOutOfRange`] when `index` is past the end
    /// * [`WreditError::InvalidOperation`] when `out` is not exactly
    ///   [`BLOCK_SIZE`] bytes
    /// * [`WreditError::DeviceIo`] on a device-side failure
    fn read_block(&self, index: u64, out: &mut [u8]) -> WreditResult<()>;

    /// Writes `data` (exactly one block) to block `index`.
    ///
    /// # Errors
    ///
    /// Same set as [`BlockDevice::read_block`].
    fn write_block(&mut self, index: u64, data: &[u8]) -> WreditResult<()>;
}

/// An in-RAM block device: the engine's "platters" as one flat array.
///
/// The whole filesystem stack ([`crate::storage::fs::DiskFs`]) is
/// written against the [`BlockDevice`] trait, so the same image can
/// later be persisted to (or loaded from) a real file or a WASM
/// persistent store without touching the filesystem code.
///
/// # Examples
///
/// ```
/// use codevar_wredit::error::WreditResult;
/// use codevar_wredit::storage::block::{BlockDevice, MemoryDisk, BLOCK_SIZE};
///
/// fn inspect() -> WreditResult<u64> {
///     let disk = MemoryDisk::new(64)?;
///     let mut block = [0u8; BLOCK_SIZE];
///     disk.read_block(0, &mut block)?;
///     Ok(disk.block_count())
/// }
///
/// assert_eq!(inspect(), Ok(64));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDisk {
    /// Flat storage: `block_count() * BLOCK_SIZE` zeroed bytes.
    blocks: Vec<u8>,
}

impl MemoryDisk {
    /// Creates a zeroed device of `block_count` blocks.
    ///
    /// # Errors
    ///
    /// * [`WreditError::BlockOutOfRange`] when `block_count` is zero
    ///   or exceeds [`MAX_BLOCKS`]
    /// * [`WreditError::DiskFull`] when the host cannot back the
    ///   allocation
    pub fn new(block_count: u64) -> WreditResult<Self> {
        if block_count == 0 || block_count > MAX_BLOCKS {
            return Err(WreditError::BlockOutOfRange {
                index: block_count,
                count: MAX_BLOCKS,
            });
        }
        let byte_len = block_count
            .checked_mul(BLOCK_SIZE as u64)
            .and_then(|total| usize::try_from(total).ok())
            .ok_or(WreditError::LimitExceeded {
                what: "memory disk size",
            })?;
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(byte_len)
            .map_err(|_| WreditError::DiskFull { free: 0 })?;
        blocks.resize(byte_len, 0);
        Ok(Self { blocks })
    }

    /// Number of blocks the device holds.
    #[must_use]
    pub fn block_count(&self) -> u64 {
        (self.blocks.len() / BLOCK_SIZE) as u64
    }

    /// Byte offset of `index`, or [`WreditError::BlockOutOfRange`].
    fn byte_offset(&self, index: u64) -> WreditResult<usize> {
        let count = self.block_count();
        if index >= count {
            return Err(WreditError::BlockOutOfRange { index, count });
        }
        usize::try_from(index)
            .ok()
            .and_then(|i| i.checked_mul(BLOCK_SIZE))
            .ok_or(WreditError::InvalidAddress { address: index })
    }
}

impl BlockDevice for MemoryDisk {
    fn block_count(&self) -> u64 {
        MemoryDisk::block_count(self)
    }

    fn read_block(&self, index: u64, out: &mut [u8]) -> WreditResult<()> {
        check_transfer(out.len())?;
        let start = self.byte_offset(index)?;
        out.copy_from_slice(&self.blocks[start..start + BLOCK_SIZE]);
        Ok(())
    }

    fn write_block(&mut self, index: u64, data: &[u8]) -> WreditResult<()> {
        check_transfer(data.len())?;
        let start = self.byte_offset(index)?;
        self.blocks[start..start + BLOCK_SIZE].copy_from_slice(data);
        Ok(())
    }
}

/// Rejects transfers that are not exactly one block long.
fn check_transfer(len: usize) -> WreditResult<()> {
    if len != BLOCK_SIZE {
        return Err(WreditError::InvalidOperation {
            what: "block transfers must be exactly one block",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BLOCK_SIZE, BlockDevice, MemoryDisk};

    /// Justification: `unwrap` in tests is permitted by the project
    /// lint configuration (`allow-unwrap-in-tests`).

    #[test]
    fn new_device_is_zeroed_and_reports_its_geometry() {
        let disk = MemoryDisk::new(8).unwrap();
        assert_eq!(disk.block_count(), 8);
        let mut block = [0u8; BLOCK_SIZE];
        disk.read_block(7, &mut block).unwrap();
        assert_eq!(block, [0u8; BLOCK_SIZE]);
    }

    #[test]
    fn writes_round_trip_through_the_device() {
        let mut disk = MemoryDisk::new(4).unwrap();
        let mut block = [0u8; BLOCK_SIZE];
        block[0] = 0xAB;
        block[BLOCK_SIZE - 1] = 0xCD;
        disk.write_block(2, &block).unwrap();
        let mut read_back = [0u8; BLOCK_SIZE];
        disk.read_block(2, &mut read_back).unwrap();
        assert_eq!(read_back, block);
        // Neighbouring blocks stay untouched.
        disk.read_block(3, &mut read_back).unwrap();
        assert_eq!(read_back, [0u8; BLOCK_SIZE]);
    }

    #[test]
    fn out_of_range_and_malformed_transfers_are_rejected() {
        let mut disk = MemoryDisk::new(2).unwrap();
        let block = [0u8; BLOCK_SIZE];
        assert!(matches!(
            disk.read_block(2, &mut [0u8; BLOCK_SIZE]),
            Err(crate::error::WreditError::BlockOutOfRange { index: 2, count: 2 })
        ));
        assert!(matches!(
            disk.write_block(9, &block),
            Err(crate::error::WreditError::BlockOutOfRange { index: 9, count: 2 })
        ));
        assert!(matches!(
            disk.read_block(0, &mut [0u8; 16]),
            Err(crate::error::WreditError::InvalidOperation { .. })
        ));
    }

    #[test]
    fn absurd_geometries_are_rejected_before_allocating() {
        assert!(MemoryDisk::new(0).is_err());
        assert!(MemoryDisk::new(u64::MAX).is_err());
    }
}
