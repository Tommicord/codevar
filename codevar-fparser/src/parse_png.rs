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

//! PNG structural parser (signature + chunks).
//!
//! Follows picoPNG `readPngHeader` / libpng chunk framing: 8-byte signature,
//! then length(4) + type(4) + data + CRC(4). Does not inflate IDAT.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, ValidationInfo,
    fourcc, read_u32_be,
};

/// PNG signature (picoPNG / libpng).
pub const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// PNG format parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct PngParser;

impl PngParser {
    /// Creates a PNG parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl FormatParser for PngParser {
    fn name() -> &'static str {
        "png"
    }

    fn detect(data: &[u8]) -> bool {
        data.len() >= 8 && data[..8] == PNG_SIG
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        if data.len() < 24 {
            return Err(ParserError::InsufficientData);
        }
        if !Self::detect(data) {
            return Ok(ValidationInfo::err("bad PNG signature"));
        }
        if &data[12..16] != b"IHDR" {
            return Ok(ValidationInfo::err("first chunk is not IHDR"));
        }
        let width = read_u32_be(data, 16).ok_or(ParserError::InsufficientData)?;
        let height = read_u32_be(data, 20).ok_or(ParserError::InsufficientData)?;
        if width == 0 || height == 0 {
            return Ok(ValidationInfo::err("PNG IHDR has zero dimension"));
        }
        Ok(ValidationInfo::ok_dims("valid PNG IHDR", width, height))
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
        if !Self::detect(data) {
            return Err(ParserError::InvalidHeader("bad PNG signature".into()));
        }
        let mut out = vec![Segment::new(SegmentKind::Signature, 0, 8, 0)];
        let mut i = 8usize;
        while i + 12 <= data.len() {
            let length =
                read_u32_be(data, i).ok_or(ParserError::InsufficientData)? as usize;
            let type_tag =
                read_u32_be(data, i + 4).ok_or(ParserError::InsufficientData)?;
            let data_end = (i + 8)
                .checked_add(length)
                .ok_or_else(|| ParserError::TokenizationError("chunk overflow".into()))?;
            let chunk_end = data_end
                .checked_add(4)
                .ok_or_else(|| ParserError::TokenizationError("crc overflow".into()))?;
            if chunk_end > data.len() {
                return Err(ParserError::InsufficientData);
            }
            let kind = if type_tag == fourcc(b'I', b'H', b'D', b'R') {
                SegmentKind::Header
            } else if type_tag == fourcc(b'I', b'D', b'A', b'T') {
                SegmentKind::Payload
            } else if type_tag == fourcc(b'I', b'E', b'N', b'D') {
                SegmentKind::Trailer
            } else {
                SegmentKind::Chunk
            };
            out.push(Segment::new(kind, i, chunk_end - i, type_tag));
            i = chunk_end;
            if type_tag == fourcc(b'I', b'E', b'N', b'D') {
                break;
            }
        }
        Ok(out)
    }
}
