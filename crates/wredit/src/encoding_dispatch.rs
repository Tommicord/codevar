//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Encoding registry and I/O-boundary codec entry points.
//!
//! The edit buffer stores one Unicode scalar value per packed element
//! (see [`crate::base`]); encoding is a property of the byte stream at
//! the document boundary.  This module is the single place that maps a
//! stable [`crate::base`] tag to a [`DocumentEncoding`] and performs the
//! byte ↔ scalar conversion.
//!
//! The previous design used a dense function-pointer table with one row
//! per encoding.  That table could not represent UTF-32 (absent from
//! the WHATWG Encoding Standard) or the 28 selectable single-byte
//! encodings, and it clamped every unknown tag to windows-1252.  The
//! registry below keeps tag resolution explicit and rejects unknown
//! tags instead of guessing.

use crate::encoding::DocumentEncoding;

/// Resolves a stable document-encoding tag.
///
/// Returns `None` for [`ENC_UNKNOWN`] and for tags outside the
/// documented ranges.  Callers must choose an explicit fallback rather
/// than relying on a silent clamp.
#[must_use]
#[inline]
pub fn encoding_for_tag(tag: u16) -> Option<DocumentEncoding> {
    DocumentEncoding::from_tag(tag)
}

/// Resolves a WHATWG encoding label such as `"utf-8"` or
/// `"windows-1252"`.
#[must_use]
#[inline]
pub fn encoding_for_label(label: &str) -> Option<DocumentEncoding> {
    DocumentEncoding::for_label(label)
}

/// Identifies an encoding from a leading byte-order mark.
#[must_use]
#[inline]
pub fn encoding_for_bom(bytes: &[u8]) -> Option<DocumentEncoding> {
    DocumentEncoding::sniff_bom(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{
        ENC_SINGLE_BYTE, ENC_SINGLE_BYTE_COUNT, ENC_SINGLE_BYTE_LAST, ENC_UNKNOWN, ENC_UTF8, ENC_UTF16BE,
        ENC_UTF16LE, ENC_UTF32BE, ENC_UTF32LE,
    };
    use crate::encoding::SINGLE_BYTE_COUNT;

    #[test]
    fn registry_resolves_every_documented_tag() {
        assert_eq!(encoding_for_tag(ENC_UTF8), Some(DocumentEncoding::Utf8));
        assert_eq!(encoding_for_tag(ENC_UTF16LE), Some(DocumentEncoding::Utf16Le));
        assert_eq!(encoding_for_tag(ENC_UTF16BE), Some(DocumentEncoding::Utf16Be));
        assert_eq!(encoding_for_tag(ENC_UTF32LE), Some(DocumentEncoding::Utf32Le));
        assert_eq!(encoding_for_tag(ENC_UTF32BE), Some(DocumentEncoding::Utf32Be));
        for index in 0..ENC_SINGLE_BYTE_COUNT {
            assert_eq!(
                encoding_for_tag(ENC_SINGLE_BYTE + index),
                Some(DocumentEncoding::SingleByte(index as u8))
            );
        }
        assert_eq!(
            encoding_for_tag(ENC_SINGLE_BYTE_LAST),
            Some(DocumentEncoding::SingleByte(SINGLE_BYTE_COUNT - 1))
        );
    }

    #[test]
    fn registry_rejects_unknown_and_out_of_range_tags() {
        assert_eq!(encoding_for_tag(ENC_UNKNOWN), None);
        assert_eq!(encoding_for_tag(0x6), None);
        assert_eq!(encoding_for_tag(0xF), None);
        assert_eq!(encoding_for_tag(ENC_SINGLE_BYTE_LAST + 1), None);
        assert_eq!(encoding_for_tag(u16::MAX), None);
    }

    #[test]
    fn registry_resolves_labels_and_boms() {
        assert_eq!(
            encoding_for_label("windows-1252"),
            Some(DocumentEncoding::SingleByte(20))
        );
        assert_eq!(encoding_for_label("no-such-encoding"), None);
        assert_eq!(
            encoding_for_bom(&[0xFF, 0xFE, 0x00, 0x00]),
            Some(DocumentEncoding::Utf32Le)
        );
        assert_eq!(encoding_for_bom(b"plain"), None);
    }
}
