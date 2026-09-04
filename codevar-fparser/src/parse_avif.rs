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

//! AVIF structural parser (ISO BMFF / HEIF `ftyp` brand).
//!
//! Detects the File Type Box (`ftyp`) with major/compatible brand `avif` or
//! `avis` and walks top-level boxes (size + type) without decoding tiles.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, ValidationInfo,
    fourcc, read_u32_be,
};

const BRAND_AVIF: u32 = fourcc(b'a', b'v', b'i', b'f');
const BRAND_AVIS: u32 = fourcc(b'a', b'v', b'i', b's');
const BOX_FTYP: u32 = fourcc(b'f', b't', b'y', b'p');

/// AVIF format parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct AvifParser;

impl AvifParser {
    /// Creates an AVIF parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Reads brands from an `ftyp` box starting at offset 0.
    #[must_use]
    pub fn ftyp_has_avif_brand(data: &[u8]) -> bool {
        if data.len() < 12 {
            return false;
        }
        let Some(size) = read_u32_be(data, 0) else {
            return false;
        };
        let Some(typ) = read_u32_be(data, 4) else {
            return false;
        };
        if typ != BOX_FTYP {
            return false;
        }
        let end = if size == 0 {
            data.len()
        } else if size == 1 {
            // 64-bit largesize — treat remaining buffer as box for detection.
            data.len()
        } else {
            usize::try_from(size).unwrap_or(0).min(data.len())
        };
        if end < 16 {
            return false;
        }
        let major = read_u32_be(data, 8).unwrap_or(0);
        if major == BRAND_AVIF || major == BRAND_AVIS {
            return true;
        }
        // Compatible brands start at offset 16.
        let mut i = 16usize;
        while i + 4 <= end {
            let brand = read_u32_be(data, i).unwrap_or(0);
            if brand == BRAND_AVIF || brand == BRAND_AVIS {
                return true;
            }
            i += 4;
        }
        false
    }
}

impl FormatParser for AvifParser {
    fn name() -> &'static str {
        "avif"
    }

    fn detect(data: &[u8]) -> bool {
        Self::ftyp_has_avif_brand(data)
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        if data.len() < 12 {
            return Err(ParserError::InsufficientData);
        }
        let typ = read_u32_be(data, 4).ok_or(ParserError::InsufficientData)?;
        if typ != BOX_FTYP {
            return Ok(ValidationInfo::err("missing ftyp box"));
        }
        if !Self::ftyp_has_avif_brand(data) {
            return Ok(ValidationInfo::err("ftyp without avif/avis brand"));
        }
        Ok(ValidationInfo::ok("valid AVIF ftyp brand"))
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
        let info = Self::validate(data)?;
        if !info.valid {
            return Err(ParserError::InvalidHeader(info.message));
        }
        let mut out = Vec::new();
        let mut i = 0usize;
        while i + 8 <= data.len() {
            let size32 = read_u32_be(data, i).ok_or(ParserError::InsufficientData)?;
            let typ = read_u32_be(data, i + 4).ok_or(ParserError::InsufficientData)?;
            let (header_len, total) = match size32 {
                0 => (8usize, data.len() - i),
                1 => {
                    if i + 16 > data.len() {
                        return Err(ParserError::InsufficientData);
                    }
                    let size64 = {
                        let b = &data[i + 8..i + 16];
                        u64::from_be_bytes([
                            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                        ])
                    };
                    let total = usize::try_from(size64).map_err(|_| {
                        ParserError::TokenizationError("box size too large".into())
                    })?;
                    (16usize, total)
                }
                n => {
                    let total = usize::try_from(n).map_err(|_| {
                        ParserError::TokenizationError("box size too large".into())
                    })?;
                    (8usize, total)
                }
            };
            if total < header_len || i + total > data.len() {
                return Err(ParserError::InsufficientData);
            }
            let kind = if typ == BOX_FTYP {
                SegmentKind::Header
            } else if typ == fourcc(b'm', b'd', b'a', b't') {
                SegmentKind::Payload
            } else {
                SegmentKind::Chunk
            };
            out.push(Segment::new(kind, i, total, typ));
            i += total;
            if size32 == 0 {
                break;
            }
        }
        Ok(out)
    }
}
