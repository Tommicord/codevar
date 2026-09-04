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

//! UTF-32 text decoder and segmenter.
//!
//! Validates UTF-32 structure and emits per-code-point byte spans
//! without depending on a full Unicode library.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, UtfParser,
    ValidationInfo,
};

/// UTF-32 big-endian byte-order mark.
pub const UTF32_BE_BOM: [u8; 4] = [0x00, 0x00, 0xFE, 0xFF];

/// UTF-32 little-endian byte-order mark.
pub const UTF32_LE_BOM: [u8; 4] = [0xFF, 0xFE, 0x00, 0x00];

/// Endianness for UTF-32 decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    /// Big-endian (BE).
    Big,
    /// Little-endian (LE).
    Little,
}

/// UTF-32 encoding parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct Utf32Parser;

impl Utf32Parser {
    /// Creates a new UTF-32 parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detects endianness from BOM if present.
    #[must_use]
    pub fn detect_endianness(data: &[u8]) -> Option<Endianness> {
        if data.starts_with(&UTF32_BE_BOM) {
            Some(Endianness::Big)
        } else if data.starts_with(&UTF32_LE_BOM) {
            Some(Endianness::Little)
        } else {
            None
        }
    }

    /// Decodes `data` into Unicode scalar values with explicit endianness.
    ///
    /// # Errors
    ///
    /// Returns [`ParserError::DecodeError`] on an invalid UTF-32 sequence.
    pub fn decode_with_endianness(
        data: &[u8],
        endianness: Endianness,
    ) -> ParserResult<Vec<char>> {
        decode_utf32(data, endianness)
    }
}

impl UtfParser for Utf32Parser {
    fn has_bom(data: &[u8]) -> bool {
        Self::detect_endianness(data).is_some()
    }

    fn decode(data: &[u8]) -> ParserResult<Vec<char>> {
        let (endianness, start) = if let Some(end) = Self::detect_endianness(data) {
            (end, 4)
        } else {
            // Default to big-endian when no BOM
            (Endianness::Big, 0)
        };
        decode_utf32(&data[start..], endianness)
    }

    fn validate_bytes(data: &[u8]) -> ParserResult<ValidationInfo> {
        let (endianness, start) = if let Some(end) = Self::detect_endianness(data) {
            (end, 4)
        } else {
            (Endianness::Big, 0)
        };
        validate_utf32(&data[start..], endianness)?;
        if Self::has_bom(data) {
            Ok(ValidationInfo::ok("valid UTF-32 with BOM"))
        } else {
            Ok(ValidationInfo::ok("valid UTF-32"))
        }
    }
}

impl FormatParser for Utf32Parser {
    fn name() -> &'static str {
        "utf-32"
    }

    fn detect(data: &[u8]) -> bool {
        if Self::has_bom(data) {
            return true;
        }
        // Check if data could be valid UTF-32 (multiple of 4 and valid sequences)
        if data.len() % 4 != 0 {
            return false;
        }
        // Try both endiannesses
        validate_utf32(data, Endianness::Big).is_ok()
            || validate_utf32(data, Endianness::Little).is_ok()
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        Self::validate_bytes(data)
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
        let mut out = Vec::new();
        let (endianness, start) = if let Some(end) = Self::detect_endianness(data) {
            out.push(Segment::new(
                SegmentKind::Signature,
                0,
                4,
                if end == Endianness::Big {
                    0x0000FEFF
                } else {
                    0xFFFE0000
                },
            ));
            (end, 4)
        } else {
            (Endianness::Big, 0)
        };

        let mut i = start;
        while i < data.len() {
            let _cp = decode_one(data, i, endianness)?;
            out.push(Segment::new(SegmentKind::TextRun, i, 4, 0));
            i += 4;
        }
        Ok(out)
    }
}

fn validate_utf32(data: &[u8], endianness: Endianness) -> ParserResult<()> {
    let mut i = 0;
    while i + 4 <= data.len() {
        let cp = decode_one(data, i, endianness)?;
        // Validate code point is a valid Unicode scalar
        if !is_valid_scalar(cp) {
            return Err(ParserError::DecodeError(format!(
                "invalid Unicode scalar value 0x{cp:08X} at {i}"
            )));
        }
        i += 4;
    }
    if i < data.len() {
        return Err(ParserError::DecodeError(
            "truncated UTF-32 sequence".to_string(),
        ));
    }
    Ok(())
}

fn decode_utf32(data: &[u8], endianness: Endianness) -> ParserResult<Vec<char>> {
    let mut chars = Vec::new();
    let mut i = 0;
    while i + 4 <= data.len() {
        let cp = decode_one(data, i, endianness)?;
        if let Some(ch) = char::from_u32(cp) {
            chars.push(ch);
        } else {
            return Err(ParserError::DecodeError(format!(
                "invalid Unicode scalar at {i}"
            )));
        }
        i += 4;
    }
    if i < data.len() {
        return Err(ParserError::DecodeError(
            "truncated UTF-32 sequence".to_string(),
        ));
    }
    Ok(chars)
}

/// Decodes one UTF-32 code point at `at`. Returns the code point value.
fn decode_one(data: &[u8], at: usize, endianness: Endianness) -> ParserResult<u32> {
    if at + 4 > data.len() {
        return Err(ParserError::InsufficientData);
    }
    let bytes = [data[at], data[at + 1], data[at + 2], data[at + 3]];
    Ok(match endianness {
        Endianness::Big => u32::from_be_bytes(bytes),
        Endianness::Little => u32::from_le_bytes(bytes),
    })
}

/// Checks if a code point is a valid Unicode scalar value.
const fn is_valid_scalar(cp: u32) -> bool {
    // Unicode scalar values: 0x0000..=0xD7FF, 0xE000..=0x10FFFF
    // Surrogate range: 0xD800..=0xDFFF (invalid)
    // Maximum: 0x10FFFF
    cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF)
}
