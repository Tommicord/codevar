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

//! ISO-8859-1 (Latin-1) decoder.
//!
//! Maps each byte `b` to `U+0000..=U+00FF` (identity). Distinct from
//! Windows-1252, which remaps `0x80..=0x9F`.

use crate::parse_common::{
    FormatParser, ParserResult, Segment, SegmentKind, ValidationInfo,
};

/// ISO-8859-1 text parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct Iso88591Parser;

impl Iso88591Parser {
    /// Creates an ISO-8859-1 parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Decodes every byte as a Latin-1 scalar.
    #[must_use]
    pub fn decode(data: &[u8]) -> Vec<char> {
        data.iter().map(|&b| char::from(b)).collect()
    }
}

impl FormatParser for Iso88591Parser {
    fn name() -> &'static str {
        "iso-8859-1"
    }

    fn detect(data: &[u8]) -> bool {
        // Any byte sequence is valid Latin-1; detection is never exclusive.
        !data.is_empty()
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        let _ = data;
        Ok(ValidationInfo::ok("valid ISO-8859-1"))
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
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
    fn maps_high_bytes() {
        let chars = Iso88591Parser::decode(&[0xA9]);
        assert_eq!(chars, vec!['©']);
    }
}
