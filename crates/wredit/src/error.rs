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

//! Error types for the Wredit text editing engine.
//!
//! Every fallible operation in this crate reports [`Error`];
//! the crate never panics in production code paths.

use core::fmt;

/// Result alias used throughout the Wredit engine.
pub type Result<T> = core::result::Result<T, Error>;

/// Everything that can go wrong inside the Wredit engine.
///
/// Variants are grouped by the layer that raises them: position and
/// cursor validation ([`Error::InvalidPosition`],
/// [`Error::InvalidCursor`]), the staging gap layer
/// ([`Error::InvalidGapSize`], [`Error::StagingFull`]),
/// the block storage layer ([`Error::InvalidAddress`],
/// [`Error::BlockNotFound`], [`Error::CapacityExhausted`],
/// [`Error::BlockOutOfRange`], [`Error::DiskFull`],
/// [`Error::DeviceIo`]), the namespace layer
/// ([`Error::NotFound`], [`Error::AlreadyExists`],
/// [`Error::NotADirectory`], [`Error::IsADirectory`],
/// [`Error::DirectoryNotEmpty`], [`Error::InvalidPath`],
/// [`Error::NameTooLong`]), image validation
/// ([`Error::CorruptImage`]), and encoding
/// ([`Error::Encoding`]).
///
/// The GPU (HOT) path deliberately has **no** error variants: an
/// unavailable or failing device is not an error condition, the
/// operation simply runs on the CPU/SIMD path instead
/// ([`crate::hot`] falls back internally).
///
/// # Examples
///
/// ```
/// use codevar_wredit::error::{Error, Result};
///
/// fn checked(position: u64, len: u64) -> Result<u64> {
///     if position > len {
///         return Err(Error::InvalidPosition { position, len });
///     }
///     Ok(position)
/// }
///
/// assert_eq!(checked(3, 5), Ok(3));
/// assert!(matches!(
///     checked(7, 5),
///     Err(Error::InvalidPosition { position: 7, len: 5 })
/// ));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A cursor or byte position lies outside the current document.
    InvalidPosition {
        /// The rejected position, in code points.
        position: u64,
        /// The current document length in code points.
        len: u64,
    },
    /// A half-open range is malformed (`start > end`) or escapes the
    /// document.
    OutOfRange {
        /// Range start (inclusive).
        start: u64,
        /// Range end (exclusive).
        end: u64,
    },
    /// The cursor id does not exist in the cursor pool.
    InvalidCursor {
        /// The rejected cursor id.
        cursor: usize,
    },
    /// The buddy allocator has no free block of the requested order.
    CapacityExhausted {
        /// Number of 4 KiB blocks the caller asked for.
        requested: usize,
        /// Number of blocks the pool still holds free.
        available: usize,
    },
    /// A virtual address is malformed (for example unaligned where
    /// block alignment is required).
    InvalidAddress {
        /// The rejected virtual address.
        address: u64,
    },
    /// The block table has no entry for the requested address.
    BlockNotFound {
        /// The virtual address that failed to resolve.
        address: u64,
    },
    /// The textlike encoder or decoder rejected the input.
    Encoding,
    /// The byte stream is not valid in the selected encoding.
    MalformedInput {
        /// Encoding name (`"UTF-8"`, `"windows-1252"`, ...).
        encoding: &'static str,
        /// Byte offset of the first sequence that failed validation.
        offset: usize,
    },
    /// A scalar value cannot be represented in the selected encoding.
    UnmappableCharacter {
        /// The scalar that has no mapping.
        character: char,
    },
    /// A requested range exceeds what the operation supports (for
    /// example a length that no longer fits the kernel's `int`
    /// parameters).
    LimitExceeded {
        /// Name of the limit that was hit.
        what: &'static str,
    },
    /// An operation is not valid for the current state of the engine.
    InvalidOperation {
        /// Short description of the violated precondition.
        what: &'static str,
    },
    /// A staging gap was configured with a zero (or otherwise unusable)
    /// capacity.
    InvalidGapSize {
        /// The rejected staging capacity, in elements.
        size: usize,
    },
    /// Every staging region is full and the staging area cannot accept
    /// the write; the caller must flush or relocate the gap first.
    StagingFull {
        /// Elements currently staged across all regions.
        staged: usize,
        /// Total staging capacity in elements.
        capacity: usize,
    },
    /// A block device was asked for a block outside its geometry.
    BlockOutOfRange {
        /// The rejected block index.
        index: u64,
        /// Number of blocks the device holds.
        count: u64,
    },
    /// The block store has no free block (or inode) left.
    DiskFull {
        /// Number of blocks (or inodes) still free.
        free: u64,
    },
    /// A stored image failed its structural checks (bad magic, version,
    /// geometry, or a broken root directory).
    CorruptImage {
        /// Short description of the failed check.
        what: &'static str,
    },
    /// The requested path does not exist in the engine storage.
    NotFound {
        /// Short description of what was missing.
        what: &'static str,
    },
    /// The requested name is already present in the parent directory.
    AlreadyExists {
        /// Short description of what already exists.
        what: &'static str,
    },
    /// A directory name exceeds the on-disk name limit.
    NameTooLong {
        /// The rejected name length in bytes.
        len: usize,
    },
    /// The underlying block device failed a read or write.
    DeviceIo {
        /// Short description of the failed transfer.
        what: &'static str,
    },
    /// A path component that must be a directory is a regular file.
    NotADirectory {
        /// Short description of the offending component.
        what: &'static str,
    },
    /// An operation requires a regular file but the target is a
    /// directory.
    IsADirectory {
        /// Short description of the offending target.
        what: &'static str,
    },
    /// A directory still holds entries and cannot be removed.
    DirectoryNotEmpty {
        /// Short description of the offending directory.
        what: &'static str,
    },
    /// A path is syntactically invalid (empty component, missing
    /// separator, or a component that is not valid UTF-8 on disk).
    InvalidPath {
        /// Short description of the defect.
        what: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPosition { position, len } => {
                write!(f, "position {position} is outside the document (length {len})")
            }
            Self::OutOfRange { start, end } => {
                write!(f, "range {start}..{end} is invalid")
            }
            Self::InvalidCursor { cursor } => write!(f, "cursor {cursor} does not exist"),
            Self::CapacityExhausted { requested, available } => write!(
                f,
                "block allocator exhausted: requested {requested} blocks, {available} free"
            ),
            Self::InvalidAddress { address } => write!(f, "invalid virtual address {address:#x}"),
            Self::BlockNotFound { address } => {
                write!(f, "no block mapped at virtual address {address:#x}")
            }
            Self::Encoding => f.write_str("text encoding failed"),
            Self::MalformedInput { encoding, offset } => {
                write!(f, "malformed {encoding} input at byte offset {offset}")
            }
            Self::UnmappableCharacter { character } => {
                write!(f, "character {character:?} has no mapping in the target encoding")
            }
            Self::LimitExceeded { what } => write!(f, "limit exceeded: {what}"),
            Self::InvalidOperation { what } => write!(f, "invalid operation: {what}"),
            Self::InvalidGapSize { size } => {
                write!(f, "staging gap size {size} cannot hold a region")
            }
            Self::StagingFull { staged, capacity } => {
                write!(f, "staging area full: {staged} of {capacity} elements staged")
            }
            Self::BlockOutOfRange { index, count } => {
                write!(f, "block {index} is outside the device ({count} blocks)")
            }
            Self::DiskFull { free } => write!(f, "the block store is full ({free} free)"),
            Self::CorruptImage { what } => write!(f, "corrupt storage image: {what}"),
            Self::NotFound { what } => write!(f, "not found: {what}"),
            Self::AlreadyExists { what } => write!(f, "already exists: {what}"),
            Self::NameTooLong { len } => write!(f, "name of {len} bytes exceeds the limit"),
            Self::DeviceIo { what } => write!(f, "block device I/O failed: {what}"),
            Self::NotADirectory { what } => write!(f, "not a directory: {what}"),
            Self::IsADirectory { what } => write!(f, "is a directory: {what}"),
            Self::DirectoryNotEmpty { what } => write!(f, "directory not empty: {what}"),
            Self::InvalidPath { what } => write!(f, "invalid path: {what}"),
        }
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_every_variant() {
        let samples = [
            (
                Error::InvalidPosition { position: 9, len: 4 },
                "position 9 is outside the document (length 4)",
            ),
            (
                Error::OutOfRange { start: 5, end: 2 },
                "range 5..2 is invalid",
            ),
            (
                Error::InvalidCursor { cursor: 3 },
                "cursor 3 does not exist",
            ),
            (
                Error::CapacityExhausted {
                    requested: 8,
                    available: 2,
                },
                "block allocator exhausted: requested 8 blocks, 2 free",
            ),
            (
                Error::InvalidAddress { address: 0x1234 },
                "invalid virtual address 0x1234",
            ),
            (
                Error::BlockNotFound { address: 0x1000 },
                "no block mapped at virtual address 0x1000",
            ),
            (Error::Encoding, "text encoding failed"),
            (
                Error::MalformedInput {
                    encoding: "UTF-8",
                    offset: 7,
                },
                "malformed UTF-8 input at byte offset 7",
            ),
            (
                Error::UnmappableCharacter { character: '☃' },
                "character '☃' has no mapping in the target encoding",
            ),
            (
                Error::LimitExceeded { what: "kernel args" },
                "limit exceeded: kernel args",
            ),
            (
                Error::InvalidOperation { what: "empty group" },
                "invalid operation: empty group",
            ),
            (
                Error::InvalidGapSize { size: 0 },
                "staging gap size 0 cannot hold a region",
            ),
            (
                Error::StagingFull {
                    staged: 256,
                    capacity: 256,
                },
                "staging area full: 256 of 256 elements staged",
            ),
            (
                Error::BlockOutOfRange { index: 40, count: 32 },
                "block 40 is outside the device (32 blocks)",
            ),
            (
                Error::DiskFull { free: 0 },
                "the block store is full (0 free)",
            ),
            (
                Error::CorruptImage { what: "bad magic" },
                "corrupt storage image: bad magic",
            ),
            (Error::NotFound { what: "file" }, "not found: file"),
            (
                Error::AlreadyExists { what: "directory" },
                "already exists: directory",
            ),
            (
                Error::NameTooLong { len: 60 },
                "name of 60 bytes exceeds the limit",
            ),
            (
                Error::DeviceIo {
                    what: "writing block 7",
                },
                "block device I/O failed: writing block 7",
            ),
            (Error::NotADirectory { what: "a/b" }, "not a directory: a/b"),
            (Error::IsADirectory { what: "/src" }, "is a directory: /src"),
            (
                Error::DirectoryNotEmpty { what: "/src" },
                "directory not empty: /src",
            ),
            (
                Error::InvalidPath {
                    what: "empty component",
                },
                "invalid path: empty component",
            ),
        ];
        for (error, expected) in samples {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn is_std_error() {
        let err: &dyn core::error::Error = &Error::Encoding;
        assert!(err.source().is_none());
    }
}
