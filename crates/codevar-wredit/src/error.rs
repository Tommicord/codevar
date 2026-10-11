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
//! Every fallible operation in this crate reports [`WreditError`];
//! the crate never panics in production code paths.

use core::fmt;

/// Result alias used throughout the Wredit engine.
pub type WreditResult<T> = Result<T, WreditError>;

/// Everything that can go wrong inside the Wredit engine.
///
/// Variants are grouped by the layer that raises them: position and
/// cursor validation ([`WreditError::InvalidPosition`],
/// [`WreditError::InvalidCursor`]), the block storage layer
/// ([`WreditError::InvalidAddress`], [`WreditError::BlockNotFound`],
/// [`WreditError::CapacityExhausted`]), encoding
/// ([`WreditError::Encoding`]), and the optional OpenCL backend
/// ([`WreditError::GpuUnavailable`], [`WreditError::GpuFailed`]).
///
/// # Examples
///
/// ```
/// use codevar_wredit::error::{WreditError, WreditResult};
///
/// fn checked(position: u64, len: u64) -> WreditResult<u64> {
///     if position > len {
///         return Err(WreditError::InvalidPosition { position, len });
///     }
///     Ok(position)
/// }
///
/// assert_eq!(checked(3, 5), Ok(3));
/// assert!(matches!(
///     checked(7, 5),
///     Err(WreditError::InvalidPosition { position: 7, len: 5 })
/// ));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WreditError {
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
    /// A requested range exceeds what the operation supports (for
    /// example a length that no longer fits the kernel's `int`
    /// parameters).
    LimitExceeded {
        /// Name of the limit that was hit.
        what: &'static str,
    },
    /// The OpenCL backend could not be initialised on this machine.
    ///
    /// Callers are expected to fall back to the CPU path rather than
    /// surface this to users; it only appears when a caller explicitly
    /// demanded the HOT path ([`crate::hot`]) without fallback.
    GpuUnavailable,
    /// An OpenCL call failed after the pipeline was built.
    GpuFailed {
        /// Short description of the failed step.
        what: &'static str,
    },
    /// An operation is not valid for the current state of the engine.
    InvalidOperation {
        /// Short description of the violated precondition.
        what: &'static str,
    },
}

impl fmt::Display for WreditError {
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
            Self::LimitExceeded { what } => write!(f, "limit exceeded: {what}"),
            Self::GpuUnavailable => f.write_str("no usable OpenCL device is available"),
            Self::GpuFailed { what } => write!(f, "OpenCL operation failed: {what}"),
            Self::InvalidOperation { what } => write!(f, "invalid operation: {what}"),
        }
    }
}

impl core::error::Error for WreditError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_every_variant() {
        let samples = [
            (
                WreditError::InvalidPosition { position: 9, len: 4 },
                "position 9 is outside the document (length 4)",
            ),
            (
                WreditError::OutOfRange { start: 5, end: 2 },
                "range 5..2 is invalid",
            ),
            (
                WreditError::InvalidCursor { cursor: 3 },
                "cursor 3 does not exist",
            ),
            (
                WreditError::CapacityExhausted {
                    requested: 8,
                    available: 2,
                },
                "block allocator exhausted: requested 8 blocks, 2 free",
            ),
            (
                WreditError::InvalidAddress { address: 0x1234 },
                "invalid virtual address 0x1234",
            ),
            (
                WreditError::BlockNotFound { address: 0x1000 },
                "no block mapped at virtual address 0x1000",
            ),
            (WreditError::Encoding, "text encoding failed"),
            (
                WreditError::LimitExceeded { what: "kernel args" },
                "limit exceeded: kernel args",
            ),
            (
                WreditError::GpuUnavailable,
                "no usable OpenCL device is available",
            ),
            (
                WreditError::GpuFailed { what: "enqueue" },
                "OpenCL operation failed: enqueue",
            ),
            (
                WreditError::InvalidOperation { what: "empty group" },
                "invalid operation: empty group",
            ),
        ];
        for (error, expected) in samples {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn is_std_error() {
        let err: &dyn core::error::Error = &WreditError::Encoding;
        assert!(err.source().is_none());
    }
}
