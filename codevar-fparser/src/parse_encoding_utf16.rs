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

//! UTF-16 text decoder and segmenter.
//!
//! Validates UTF-16 structure (RFC 2781) and emits per-code-point byte spans
//! (or coarser runs) without depending on a full Unicode library.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, UtfParser,
    ValidationInfo,
};

/// UTF-16 big-endian byte-order mark.
pub const UTF16_BE_BOM: [u8; 2] = [0xFE, 0xFF];

/// UTF-16 little-endian byte-order mark.
pub const UTF16_LE_BOM: [u8; 2] = [0xFF, 0xFE];

/// Endianness for UTF-16 decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    /// Big-endian (BE).
    Big,
    /// Little-endian (LE).
    Little,
}

/// UTF-16 encoding parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct Utf16Parser;

impl Utf16Parser {
    /// Creates a new UTF-16 parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detects endianness from BOM if present.
    #[must_use]
    pub fn detect_endianness(data: &[u8]) -> Option<Endianness> {
        if data.starts_with(&UTF16_BE_BOM) {
            Some(Endianness::Big)
        } else if data.starts_with(&UTF16_LE_BOM) {
            Some(Endianness::Little)
        } else {
            None
        }
    }

    /// Decodes `data` into Unicode scalar values with explicit endianness.
    ///
    /// # Errors
    ///
    /// Returns [`ParserError::DecodeError`] on an invalid UTF-16 sequence.
    pub fn decode_with_endianness(
        data: &[u8],
        endianness: Endianness,
    ) -> ParserResult<Vec<char>> {
        decode_utf16(data, endianness)
    }
}

impl UtfParser for Utf16Parser {
    fn has_bom(data: &[u8]) -> bool {
        Self::detect_endianness(data).is_some()
    }

    fn decode(data: &[u8]) -> ParserResult<Vec<char>> {
        let (endianness, start) = if let Some(end) = Self::detect_endianness(data) {
            (end, 2)
        } else {
            // Default to big-endian when no BOM (per RFC 2781)
            (Endianness::Big, 0)
        };
        decode_utf16(&data[start..], endianness)
    }

    fn validate_bytes(data: &[u8]) -> ParserResult<ValidationInfo> {
        let (endianness, start) = if let Some(end) = Self::detect_endianness(data) {
            (end, 2)
        } else {
            (Endianness::Big, 0)
        };
        validate_utf16(&data[start..], endianness)?;
        if Self::has_bom(data) {
            Ok(ValidationInfo::ok("valid UTF-16 with BOM"))
        } else {
            Ok(ValidationInfo::ok("valid UTF-16"))
        }
    }
}

impl FormatParser for Utf16Parser {
    fn name() -> &'static str {
        "utf-16"
    }

    fn detect(data: &[u8]) -> bool {
        if Self::has_bom(data) {
            return true;
        }
        // Check if data could be valid UTF-16 (even length and valid sequences)
        if data.len() % 2 != 0 {
            return false;
        }
        // Try both endiannesses
        validate_utf16(data, Endianness::Big).is_ok()
            || validate_utf16(data, Endianness::Little).is_ok()
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
                2,
                if end == Endianness::Big {
                    0xFEFF
                } else {
                    0xFFFE
                },
            ));
            (end, 2)
        } else {
            (Endianness::Big, 0)
        };

        let mut i = start;
        while i < data.len() {
            let (code_units, _code_point) = decode_one(data, i, endianness)?;
            let byte_len = code_units * 2;
            out.push(Segment::new(SegmentKind::TextRun, i, byte_len, 0));
            i += byte_len;
        }
        Ok(out)
    }
}

fn validate_utf16(data: &[u8], endianness: Endianness) -> ParserResult<()> {
    let mut i = 0;
    while i + 2 <= data.len() {
        let (code_units, _) = decode_one(data, i, endianness)?;
        i += code_units * 2;
    }
    if i < data.len() {
        return Err(ParserError::DecodeError(
            "truncated UTF-16 sequence".to_string(),
        ));
    }
    Ok(())
}

fn decode_utf16(data: &[u8], endianness: Endianness) -> ParserResult<Vec<char>> {
    let mut chars = Vec::new();
    let mut i = 0;
    while i + 2 <= data.len() {
        let (code_units, cp) = decode_one(data, i, endianness)?;
        if let Some(ch) = char::from_u32(cp) {
            chars.push(ch);
        } else {
            return Err(ParserError::DecodeError(format!(
                "invalid Unicode scalar at {i}"
            )));
        }
        i += code_units * 2;
    }
    if i < data.len() {
        return Err(ParserError::DecodeError(
            "truncated UTF-16 sequence".to_string(),
        ));
    }
    Ok(chars)
}

/// Decodes one UTF-16 code point at `at`. Returns `(code_units, code_point)`.
fn decode_one(
    data: &[u8],
    at: usize,
    endianness: Endianness,
) -> ParserResult<(usize, u32)> {
    let unit1 = read_u16(data, at, endianness)?;

    // High surrogate range: 0xD800..=0xDBFF
    // Low surrogate range: 0xDC00..=0xDFFF
    if unit1 >= 0xD800 && unit1 <= 0xDBFF {
        // Surrogate pair - need second unit
        if at + 4 > data.len() {
            return Err(ParserError::DecodeError(format!(
                "truncated UTF-16 surrogate pair at {at}"
            )));
        }
        let unit2 = read_u16(data, at + 2, endianness)?;

        if !(unit2 >= 0xDC00 && unit2 <= 0xDFFF) {
            return Err(ParserError::DecodeError(format!(
                "invalid UTF-16 surrogate pair at {at}"
            )));
        }

        // Compute code point from surrogate pair
        let high = u32::from(unit1) - 0xD800;
        let low = u32::from(unit2) - 0xDC00;
        let cp = (high << 10) + low + 0x10000;
        Ok((2, cp))
    } else if unit1 >= 0xDC00 && unit1 <= 0xDFFF {
        return Err(ParserError::DecodeError(format!(
            "lone low surrogate at {at}"
        )));
    } else {
        // Single code unit (BMP character)
        Ok((1, u32::from(unit1)))
    }
}

fn read_u16(data: &[u8], at: usize, endianness: Endianness) -> ParserResult<u16> {
    if at + 2 > data.len() {
        return Err(ParserError::InsufficientData);
    }
    let bytes = [data[at], data[at + 1]];
    Ok(match endianness {
        Endianness::Big => u16::from_be_bytes(bytes),
        Endianness::Little => u16::from_le_bytes(bytes),
    })
}
