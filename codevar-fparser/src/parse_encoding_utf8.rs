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

//! UTF-8 text decoder and segmenter.
//!
//! Validates UTF-8 structure (RFC 3629) and emits per-code-point byte spans
//! (or coarser runs) without depending on a full Unicode library.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, UtfParser,
    ValidationInfo,
};

/// UTF-8 byte-order mark.
pub const UTF8_BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

/// UTF-8 encoding parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct Utf8Parser;

impl Utf8Parser {
    /// Creates a new UTF-8 parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl UtfParser for Utf8Parser {
    fn has_bom(data: &[u8]) -> bool {
        data.starts_with(&UTF8_BOM)
    }

    fn decode(data: &[u8]) -> ParserResult<Vec<char>> {
        let start = if Self::has_bom(data) { 3 } else { 0 };
        decode_utf8(&data[start..])
    }

    fn validate_bytes(data: &[u8]) -> ParserResult<ValidationInfo> {
        let start = if Self::has_bom(data) { 3 } else { 0 };
        validate_utf8(&data[start..])?;
        if Self::has_bom(data) {
            Ok(ValidationInfo::ok("valid UTF-8 with BOM"))
        } else {
            Ok(ValidationInfo::ok("valid UTF-8"))
        }
    }
}

impl FormatParser for Utf8Parser {
    fn name() -> &'static str {
        "utf-8"
    }

    fn detect(data: &[u8]) -> bool {
        if Self::has_bom(data) {
            return true;
        }
        // Prefer explicit BOM; otherwise accept only when fully valid and
        // not claimed by a binary magic elsewhere (caller decides order).
        !data.is_empty() && validate_utf8(data).is_ok() && looks_textual(data)
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        Self::validate_bytes(data)
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
        let mut out = Vec::new();
        let mut i = 0usize;
        if Self::has_bom(data) {
            out.push(Segment::new(SegmentKind::Signature, 0, 3, 0xEF_BB_BF));
            i = 3;
        }
        while i < data.len() {
            let (ch_len, _) = decode_one(data, i)?;
            out.push(Segment::new(SegmentKind::TextRun, i, ch_len, 0));
            i += ch_len;
        }
        Ok(out)
    }
}

fn looks_textual(data: &[u8]) -> bool {
    let sample = if data.len() > 256 { &data[..256] } else { data };
    !sample.contains(&0)
}

fn validate_utf8(data: &[u8]) -> ParserResult<()> {
    let mut i = 0;
    while i < data.len() {
        let (len, _) = decode_one(data, i)?;
        i += len;
    }
    Ok(())
}

fn decode_utf8(data: &[u8]) -> ParserResult<Vec<char>> {
    let mut chars = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let (len, cp) = decode_one(data, i)?;
        if let Some(ch) = char::from_u32(cp) {
            chars.push(ch);
        } else {
            return Err(ParserError::DecodeError(format!(
                "invalid Unicode scalar at {i}"
            )));
        }
        i += len;
    }
    Ok(chars)
}

/// Decodes one UTF-8 code point at `at`. Returns `(byte_len, code_point)`.
fn decode_one(data: &[u8], at: usize) -> ParserResult<(usize, u32)> {
    let b0 = *data.get(at).ok_or(ParserError::InsufficientData)?;
    match b0 {
        0x00..=0x7F => Ok((1, u32::from(b0))),
        0xC2..=0xDF => {
            let b1 = next(data, at + 1)?;
            if !is_cont(b1) {
                return Err(bad(at, "bad 2-byte UTF-8 continuation"));
            }
            let cp = (u32::from(b0 & 0x1F) << 6) | u32::from(b1 & 0x3F);
            Ok((2, cp))
        }
        0xE0..=0xEF => {
            let b1 = next(data, at + 1)?;
            let b2 = next(data, at + 2)?;
            if !is_cont(b1) || !is_cont(b2) {
                return Err(bad(at, "bad 3-byte UTF-8 continuation"));
            }
            // Overlong / surrogate rejection (RFC 3629).
            if b0 == 0xE0 && b1 < 0xA0 {
                return Err(bad(at, "overlong 3-byte UTF-8"));
            }
            if b0 == 0xED && b1 >= 0xA0 {
                return Err(bad(at, "UTF-8 surrogate"));
            }
            let cp = (u32::from(b0 & 0x0F) << 12)
                | (u32::from(b1 & 0x3F) << 6)
                | u32::from(b2 & 0x3F);
            Ok((3, cp))
        }
        0xF0..=0xF4 => {
            let b1 = next(data, at + 1)?;
            let b2 = next(data, at + 2)?;
            let b3 = next(data, at + 3)?;
            if !is_cont(b1) || !is_cont(b2) || !is_cont(b3) {
                return Err(bad(at, "bad 4-byte UTF-8 continuation"));
            }
            if b0 == 0xF0 && b1 < 0x90 {
                return Err(bad(at, "overlong 4-byte UTF-8"));
            }
            if b0 == 0xF4 && b1 > 0x8F {
                return Err(bad(at, "UTF-8 above U+10FFFF"));
            }
            let cp = (u32::from(b0 & 0x07) << 18)
                | (u32::from(b1 & 0x3F) << 12)
                | (u32::from(b2 & 0x3F) << 6)
                | u32::from(b3 & 0x3F);
            Ok((4, cp))
        }
        _ => Err(bad(at, "invalid UTF-8 lead byte")),
    }
}

const fn is_cont(b: u8) -> bool {
    (b & 0xC0) == 0x80
}

fn next(data: &[u8], at: usize) -> ParserResult<u8> {
    data.get(at).copied().ok_or(ParserError::InsufficientData)
}

fn bad(at: usize, msg: &str) -> ParserError {
    ParserError::DecodeError(format!("{msg} at {at}"))
}
