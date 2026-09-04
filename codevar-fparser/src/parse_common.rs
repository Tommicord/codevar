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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Shared types for format and encoding parsers.

/// Result type for parser operations.
pub type ParserResult<T> = Result<T, ParserError>;

/// Errors that can occur during parsing operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParserError {
    /// Invalid file / encoding header.
    InvalidHeader(String),
    /// Unsupported file type or encoding.
    UnsupportedFileType(String),
    /// Insufficient data for parsing.
    InsufficientData,
    /// Structural / tokenization error.
    TokenizationError(String),
    /// Decode error (invalid sequences, truncated multi-byte, etc.).
    DecodeError(String),
    /// I/O error.
    IoError(String),
}

impl core::fmt::Display for ParserError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidHeader(msg) => write!(f, "Invalid header: {msg}"),
            Self::UnsupportedFileType(msg) => write!(f, "Unsupported file type: {msg}"),
            Self::InsufficientData => write!(f, "Insufficient data for parsing"),
            Self::TokenizationError(msg) => write!(f, "Tokenization error: {msg}"),
            Self::DecodeError(msg) => write!(f, "Decode error: {msg}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for ParserError {}

/// Kind of a structural or lexical segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentKind {
    /// File / container signature bytes.
    Signature,
    /// Format header (IHDR, SOF, PE COFF, ELF e_ident, …).
    Header,
    /// Named chunk / section / marker payload.
    Chunk,
    /// Table (symbol, relocation, color, …).
    Table,
    /// Compressed or entropy-coded payload.
    Payload,
    /// Decoded text rune / grapheme cluster span (byte range in source).
    TextRun,
    /// Trailer / end marker.
    Trailer,
    /// Unclassified remainder.
    Other,
}

/// A byte span produced by a parser for other modules to consume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// Segment classification.
    pub kind: SegmentKind,
    /// Start offset in the source buffer.
    pub start: usize,
    /// Length in bytes.
    pub len: usize,
    /// Optional subtype tag (FourCC, marker id, section id, code page, …).
    pub tag: u32,
}

impl Segment {
    /// Creates a segment.
    #[must_use]
    pub const fn new(kind: SegmentKind, start: usize, len: usize, tag: u32) -> Self {
        Self {
            kind,
            start,
            len,
            tag,
        }
    }

    /// Exclusive end offset.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.start + self.len
    }

    /// Returns the slice for this segment from `data`.
    #[must_use]
    pub fn slice<'a>(&self, data: &'a [u8]) -> Option<&'a [u8]> {
        data.get(self.start..self.end())
    }
}

/// Lightweight header validation outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationInfo {
    /// Whether the header is structurally acceptable.
    pub valid: bool,
    /// Short diagnostic message.
    pub message: String,
    /// Optional primary dimension / size field.
    pub width: Option<u32>,
    /// Optional secondary dimension field.
    pub height: Option<u32>,
}

impl ValidationInfo {
    /// Successful validation without dimensions.
    #[must_use]
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            valid: true,
            message: message.into(),
            width: None,
            height: None,
        }
    }

    /// Successful validation with width / height.
    #[must_use]
    pub fn ok_dims(message: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            valid: true,
            message: message.into(),
            width: Some(width),
            height: Some(height),
        }
    }

    /// Failed validation.
    #[must_use]
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            valid: false,
            message: message.into(),
            width: None,
            height: None,
        }
    }
}

/// Common interface implemented by each format / encoding parser module.
pub trait FormatParser {
    /// Human-readable parser name.
    fn name() -> &'static str;

    /// Returns `true` when `data` matches this format's magic / signature.
    fn detect(data: &[u8]) -> bool;

    /// Validates the structural header without fully decoding the payload.
    ///
    /// # Errors
    ///
    /// Returns [`ParserError`] when the buffer is too short or structurally invalid.
    fn validate(data: &[u8]) -> ParserResult<ValidationInfo>;

    /// Splits `data` into smaller segments for other modules.
    ///
    /// # Errors
    ///
    /// Returns [`ParserError`] on truncated or malformed structure.
    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>>;
}

/// Common interface for UTF encoding parsers.
pub trait UtfParser: FormatParser {
    /// Returns `true` if `data` begins with this encoding's BOM.
    fn has_bom(data: &[u8]) -> bool;

    /// Decodes `data` into Unicode scalar values, skipping an optional BOM.
    ///
    /// # Errors
    ///
    /// Returns [`ParserError::DecodeError`] on an invalid encoding sequence.
    fn decode(data: &[u8]) -> ParserResult<Vec<char>>;

    /// Validates that `data` (after optional BOM) is well-formed for this encoding.
    ///
    /// # Errors
    ///
    /// Returns [`ParserError::DecodeError`] when a sequence is invalid.
    fn validate_bytes(data: &[u8]) -> ParserResult<ValidationInfo>;
}

/// Reads a big-endian `u16`, or `None` if out of bounds.
#[must_use]
pub fn read_u16_be(data: &[u8], at: usize) -> Option<u16> {
    let b = data.get(at..at + 2)?;
    Some(u16::from_be_bytes([b[0], b[1]]))
}

/// Reads a little-endian `u16`, or `None` if out of bounds.
#[must_use]
pub fn read_u16_le(data: &[u8], at: usize) -> Option<u16> {
    let b = data.get(at..at + 2)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

/// Reads a big-endian `u32`, or `None` if out of bounds.
#[must_use]
pub fn read_u32_be(data: &[u8], at: usize) -> Option<u32> {
    let b = data.get(at..at + 4)?;
    Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// Reads a little-endian `u32`, or `None` if out of bounds.
#[must_use]
pub fn read_u32_le(data: &[u8], at: usize) -> Option<u32> {
    let b = data.get(at..at + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// FourCC tag from four ASCII bytes.
#[must_use]
pub const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    u32::from_be_bytes([a, b, c, d])
}
