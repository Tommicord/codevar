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

//! JPEG structural parser (header + marker segments).
//!
//! Inspired by picojpeg (`locateSOIMarker`, `JPEG_MARKER`) — validates SOI /
//! marker framing and emits segments without decoding MCU / IDCT data.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, ValidationInfo,
    read_u16_be,
};

const M_SOI: u8 = 0xD8;
const M_EOI: u8 = 0xD9;
const M_SOS: u8 = 0xDA;
const M_SOF0: u8 = 0xC0;
const M_SOF2: u8 = 0xC2;

/// JPEG format parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct JpegParser;

impl JpegParser {
    /// Creates a JPEG parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl FormatParser for JpegParser {
    fn name() -> &'static str {
        "jpeg"
    }

    fn detect(data: &[u8]) -> bool {
        data.len() >= 2 && data[0] == 0xFF && data[1] == M_SOI
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        if !Self::detect(data) {
            return Ok(ValidationInfo::err("missing JPEG SOI"));
        }
        if data.len() < 4 {
            return Err(ParserError::InsufficientData);
        }
        if data[2] != 0xFF {
            return Ok(ValidationInfo::err("byte after SOI is not 0xFF"));
        }
        if let Some((w, h)) = find_sof_dims(data) {
            Ok(ValidationInfo::ok_dims("valid JPEG SOI / SOF", w, h))
        } else {
            Ok(ValidationInfo::ok("valid JPEG SOI"))
        }
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
        if !Self::detect(data) {
            return Err(ParserError::InvalidHeader("missing JPEG SOI".into()));
        }
        let mut out = Vec::new();
        out.push(Segment::new(SegmentKind::Signature, 0, 2, u32::from(M_SOI)));
        let mut i = 2usize;
        while i < data.len() {
            i = skip_to_marker(data, i);
            if i >= data.len() {
                break;
            }
            let marker_start = i;
            i = skip_ff_fill(data, i);
            let Some(&marker) = data.get(i) else {
                return Err(ParserError::TokenizationError(
                    "truncated JPEG marker".into(),
                ));
            };
            i += 1;
            if is_standalone(marker) {
                out.push(Segment::new(
                    SegmentKind::Chunk,
                    marker_start,
                    i - marker_start,
                    u32::from(marker),
                ));
                if marker == M_EOI {
                    break;
                }
                continue;
            }
            let seg_len =
                read_u16_be(data, i).ok_or(ParserError::InsufficientData)? as usize;
            if seg_len < 2 {
                return Err(ParserError::TokenizationError(
                    "invalid JPEG segment length".into(),
                ));
            }
            let seg_end = i
                .checked_add(seg_len)
                .ok_or_else(|| ParserError::TokenizationError("overflow".into()))?;
            if seg_end > data.len() {
                return Err(ParserError::InsufficientData);
            }
            let kind = if marker == M_SOF0 || marker == M_SOF2 {
                SegmentKind::Header
            } else {
                SegmentKind::Chunk
            };
            out.push(Segment::new(
                kind,
                marker_start,
                seg_end - marker_start,
                u32::from(marker),
            ));
            i = seg_end;
            if marker == M_SOS {
                i = emit_scan_payload(data, i, &mut out);
            }
        }
        Ok(out)
    }
}

fn is_standalone(marker: u8) -> bool {
    marker == M_SOI
        || marker == M_EOI
        || marker == 0x01
        || (0xD0..=0xD7).contains(&marker)
}

fn skip_to_marker(data: &[u8], mut i: usize) -> usize {
    while i < data.len() && data[i] != 0xFF {
        i += 1;
    }
    i
}

fn skip_ff_fill(data: &[u8], mut i: usize) -> usize {
    while i < data.len() && data[i] == 0xFF {
        i += 1;
    }
    i
}

fn emit_scan_payload(data: &[u8], mut i: usize, out: &mut Vec<Segment>) -> usize {
    let scan_start = i;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] != 0x00 {
            let next = data[i + 1];
            if next == M_EOI || (0xD0..=0xD7).contains(&next) {
                break;
            }
        }
        i += 1;
    }
    if i > scan_start {
        out.push(Segment::new(
            SegmentKind::Payload,
            scan_start,
            i - scan_start,
            0,
        ));
    }
    i
}

fn find_sof_dims(data: &[u8]) -> Option<(u32, u32)> {
    let mut i = 2usize;
    while i + 3 < data.len() {
        i = skip_to_marker(data, i);
        if i >= data.len() {
            break;
        }
        i = skip_ff_fill(data, i);
        let marker = *data.get(i)?;
        i += 1;
        if is_standalone(marker) {
            if marker == M_EOI {
                break;
            }
            continue;
        }
        let len = read_u16_be(data, i)? as usize;
        if len < 2 {
            break;
        }
        let end = i.checked_add(len)?;
        if end > data.len() {
            break;
        }
        if (marker == M_SOF0 || marker == M_SOF2) && len >= 7 {
            let h = u32::from(read_u16_be(data, i + 3)?);
            let w = u32::from(read_u16_be(data, i + 5)?);
            return Some((w, h));
        }
        if marker == M_SOS {
            break;
        }
        i = end;
    }
    None
}
