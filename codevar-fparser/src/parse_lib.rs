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

//! UNIX / MSVC static library (`.lib` / `.a`) archive parser.
//!
//! Recognizes the ar archive magic `!<arch>\n` and emits member headers
//! (60-byte headers) as segments. Does not decompress nested objects.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, ValidationInfo,
};

/// Archive magic used by GNU `ar` and MSVC `.lib`.
pub const AR_MAGIC: &[u8] = b"!<arch>\n";

/// Static library / ar archive parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct LibParser;

impl LibParser {
    /// Creates a LIB/ar parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl FormatParser for LibParser {
    fn name() -> &'static str {
        "lib"
    }

    fn detect(data: &[u8]) -> bool {
        data.starts_with(AR_MAGIC)
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        if !Self::detect(data) {
            return Ok(ValidationInfo::err("missing ar / LIB magic"));
        }
        if data.len() < AR_MAGIC.len() + 60 && data.len() != AR_MAGIC.len() {
            // Empty archive is valid; otherwise expect at least one header.
            if data.len() > AR_MAGIC.len() {
                return Err(ParserError::InsufficientData);
            }
        }
        Ok(ValidationInfo::ok("valid ar / LIB archive signature"))
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
        if !Self::detect(data) {
            return Err(ParserError::InvalidHeader("missing ar magic".into()));
        }
        let mut out = vec![Segment::new(SegmentKind::Signature, 0, AR_MAGIC.len(), 0)];
        let mut i = AR_MAGIC.len();
        while i + 60 <= data.len() {
            let header_start = i;
            // Size field is ASCII decimal at offset 48, length 10, within header.
            let size_field = data
                .get(i + 48..i + 58)
                .ok_or(ParserError::InsufficientData)?;
            let size = parse_ar_size(size_field)?;
            let member_start = i + 60;
            let member_end = member_start.checked_add(size).ok_or_else(|| {
                ParserError::TokenizationError("member overflow".into())
            })?;
            // Members are 2-byte aligned; padding may follow.
            let aligned_end = member_end + (member_end % 2);
            if member_end > data.len() {
                return Err(ParserError::InsufficientData);
            }
            out.push(Segment::new(SegmentKind::Header, header_start, 60, 0));
            if size > 0 {
                out.push(Segment::new(SegmentKind::Payload, member_start, size, 0));
            }
            i = aligned_end.min(data.len());
            if i < aligned_end {
                break;
            }
        }
        Ok(out)
    }
}

fn parse_ar_size(field: &[u8]) -> ParserResult<usize> {
    let s = field
        .iter()
        .take_while(|&&b| b != b' ')
        .copied()
        .collect::<Vec<u8>>();
    let text = core::str::from_utf8(&s)
        .map_err(|_| ParserError::TokenizationError("non-UTF8 ar size field".into()))?;
    text.parse::<usize>()
        .map_err(|_| ParserError::TokenizationError(format!("bad ar size '{text}'")))
}
