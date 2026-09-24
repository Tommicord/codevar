//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

use core::fmt;

/// FcWare compression error.
#[derive(Debug, Eq, PartialEq)]
pub enum CompressorError {
    /// Frame magic or layout is not recognized.
    InvalidFrame,
    /// Frame ended before a complete record was read.
    TruncatedFrame,
    /// Unexpected token marker in a byte stream.
    InvalidToken(InvalidToken),
    /// LZ / substring match references are out of range.
    InvalidMatch,
    /// Decoded length does not match the frame length field.
    LengthMismatch(LengthMismatch),
    /// Input exceeds frame size limits.
    InputTooLarge,
    /// Invalid BMP / dictionary code point.
    InvalidCodePoint(InvalidCodePoint),
    /// Dictionary or table index is out of range.
    InvalidIndex,
    /// Duplicate-byte table index is out of range.
    InvalidDuplicateIndex,
    /// Control / padding byte is illegal.
    InvalidControl(InvalidControl),
    /// Caller-provided output buffer is too small.
    OutputTooSmall,
}

/// Invalid control or padding byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidControl(pub u8);

impl fmt::Display for InvalidControl {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid FcWare control byte {:#04x}", self.0)
    }
}

impl core::error::Error for InvalidControl {}

/// Unexpected stream token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidToken(pub u8);

impl fmt::Display for InvalidToken {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid FcWare token {:#04x}", self.0)
    }
}

impl core::error::Error for InvalidToken {}

/// Expected vs actual decoded length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthMismatch {
    /// Length declared by the frame header.
    pub expected: usize,
    /// Length produced by the decoder.
    pub actual: usize,
}

impl LengthMismatch {
    /// Creates a length-mismatch error payload.
    #[inline]
    #[must_use]
    pub const fn new(expected: usize, actual: usize) -> Self {
        Self { expected, actual }
    }
}

impl fmt::Display for LengthMismatch {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FcWare length mismatch: expected {}, got {}",
            self.expected, self.actual
        )
    }
}

impl core::error::Error for LengthMismatch {}

/// Invalid code-point value in a value codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidCodePoint(pub u8);

impl fmt::Display for InvalidCodePoint {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid BMP code point {:#04x}", self.0)
    }
}

impl core::error::Error for InvalidCodePoint {}

impl CompressorError {
    /// Builds [`CompressorError::LengthMismatch`].
    #[inline]
    #[must_use]
    pub const fn length_mismatch(expected: usize, actual: usize) -> Self {
        Self::LengthMismatch(LengthMismatch::new(expected, actual))
    }

    /// Builds [`CompressorError::InvalidToken`].
    #[inline]
    #[must_use]
    pub const fn invalid_token(token: u8) -> Self {
        Self::InvalidToken(InvalidToken(token))
    }

    /// Builds [`CompressorError::InvalidControl`].
    #[inline]
    #[must_use]
    pub const fn invalid_control(control: u8) -> Self {
        Self::InvalidControl(InvalidControl(control))
    }
}

impl fmt::Display for CompressorError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFrame => f.write_str("invalid FcWare frame"),
            Self::TruncatedFrame => f.write_str("truncated FcWare frame"),
            Self::InvalidToken(err) => err.fmt(f),
            Self::InvalidMatch => f.write_str("invalid FcWare match"),
            Self::LengthMismatch(err) => err.fmt(f),
            Self::InputTooLarge => f.write_str("input exceeds FcWare frame limits"),
            Self::InvalidCodePoint(err) => err.fmt(f),
            Self::InvalidIndex => f.write_str("invalid FcWare dictionary index"),
            Self::InvalidDuplicateIndex => {
                f.write_str("invalid FcWare duplicate-byte index")
            }
            Self::InvalidControl(err) => err.fmt(f),
            Self::OutputTooSmall => f.write_str("output buffer too small"),
        }
    }
}

impl From<InvalidToken> for CompressorError {
    #[inline]
    fn from(err: InvalidToken) -> Self {
        Self::InvalidToken(err)
    }
}

impl From<LengthMismatch> for CompressorError {
    #[inline]
    fn from(err: LengthMismatch) -> Self {
        Self::LengthMismatch(err)
    }
}

impl From<InvalidCodePoint> for CompressorError {
    #[inline]
    fn from(err: InvalidCodePoint) -> Self {
        Self::InvalidCodePoint(err)
    }
}

impl From<InvalidControl> for CompressorError {
    #[inline]
    fn from(err: InvalidControl) -> Self {
        Self::InvalidControl(err)
    }
}

impl core::error::Error for CompressorError {}

/// Result alias for FcWare operations.
pub type CompressorResult<T> = core::result::Result<T, CompressorError>;
