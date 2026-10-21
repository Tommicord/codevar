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

//! Error type for the Packlow codec.

use alloc::string::String;
use core::fmt;

/// The result of a Packlow operation.
pub type PackResult<T> = Result<T, PackError>;

/// Everything that can go wrong while packing or unpacking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    /// The packed stream ended before the decoder expected it to.
    ///
    /// The decoder reached the end of the input while a token still had
    /// to be read, which means the stream is truncated.
    Truncated {
        /// How many bytes were still missing when the stream ran out.
        missing: usize,
    },
    /// The packed stream contains a token the decoder does not know.
    UnknownToken {
        /// The raw token value.
        token: u8,
    },
    /// A back reference points before the start of the output.
    BadDistance {
        /// The distance the stream asked for.
        distance: u32,
        /// How many bytes were decoded so far.
        produced: usize,
    },
    /// A length field exceeds the configured maximum.
    LengthOverflow {
        /// The length the stream asked for.
        length: usize,
    },
    /// The decoded bytes do not match the checksum the header carries.
    ///
    /// The stream was truncated or corrupted on the way in.
    ChecksumMismatch {
        /// The checksum the header declared.
        expected: u64,
        /// The checksum of the bytes the decoder produced.
        actual: u64,
    },
    /// The GPU path failed and no CPU fallback was requested.
    GpuUnavailable {
        /// A human readable description of the failure.
        detail: String,
    },
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { missing } => {
                write!(f, "the packed stream is truncated, {missing} bytes are missing")
            }
            Self::UnknownToken { token } => write!(f, "unknown packed token {token:#04x}"),
            Self::BadDistance { distance, produced } => write!(
                f,
                "back reference {distance} points before the {produced} bytes decoded so far"
            ),
            Self::LengthOverflow { length } => {
                write!(f, "length {length} exceeds the maximum the codec allows")
            }
            Self::ChecksumMismatch { expected, actual } => write!(
                f,
                "the decoded bytes do not match the header: expected {expected:#018x}, got {actual:#018x}"
            ),
            Self::GpuUnavailable { detail } => write!(f, "the GPU path is unavailable: {detail}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_render_readable_messages() {
        assert!(
            PackError::Truncated { missing: 4 }
                .to_string()
                .contains("truncated")
        );
        assert!(
            PackError::UnknownToken { token: 9 }
                .to_string()
                .contains("unknown")
        );
        assert!(
            PackError::BadDistance {
                distance: 7,
                produced: 2
            }
            .to_string()
            .contains("before")
        );
        assert!(
            PackError::LengthOverflow { length: 9 }
                .to_string()
                .contains("maximum")
        );
        assert!(
            PackError::ChecksumMismatch {
                expected: 1,
                actual: 2
            }
            .to_string()
            .contains("header")
        );
        assert!(
            PackError::GpuUnavailable {
                detail: "no device".into()
            }
            .to_string()
            .contains("no device")
        );
    }
}
