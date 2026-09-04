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

//! US-ASCII (IANA `US-ASCII` / Windows code page 20127) decoder.
//!
//! Accepts only bytes in `0x00..=0x7F`.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, ValidationInfo,
};

/// US-ASCII text parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct UsAsciiParser;

impl UsAsciiParser {
    /// Creates a US-ASCII parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Decodes bytes to chars; fails on any high bit.
    ///
    /// # Errors
    ///
    /// Returns [`ParserError::DecodeError`] if any byte is `>= 0x80`.
    pub fn decode(data: &[u8]) -> ParserResult<Vec<char>> {
        let mut out = Vec::with_capacity(data.len());
        for (i, &b) in data.iter().enumerate() {
            if b >= 0x80 {
                return Err(ParserError::DecodeError(format!(
                    "non-ASCII byte 0x{b:02X} at {i}"
                )));
            }
            out.push(char::from(b));
        }
        Ok(out)
    }
}

impl FormatParser for UsAsciiParser {
    fn name() -> &'static str {
        "us-ascii"
    }

    fn detect(data: &[u8]) -> bool {
        !data.is_empty() && data.iter().all(|&b| b < 0x80)
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        Self::decode(data)?;
        Ok(ValidationInfo::ok("valid US-ASCII"))
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
        Self::decode(data)?;
        Ok(data
            .iter()
            .enumerate()
            .map(|(i, &b)| Segment::new(SegmentKind::TextRun, i, 1, u32::from(b)))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_ascii_rejects_high() {
        assert!(UsAsciiParser::decode(b"hello").is_ok());
        assert!(UsAsciiParser::decode(&[0x80]).is_err());
    }
}
