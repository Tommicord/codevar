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

//! Document encodings understood by the Wredit engine.
//!
//! A Wredit document stores one Unicode scalar value per packed raw
//! element (see [`crate::base`]).  Encoding is therefore a property of
//! the byte stream at the I/O boundary, not of the edit buffer: UTF-8,
//! UTF-16 and UTF-32 documents all decode into the same scalar stream
//! and encode back out through [`DocumentEncoding`].
//!
//! This module owns the encoding registry.  The codec functions live
//! beside the edit-buffer integration in [`crate::encoding_dispatch`]
//! so the registry stays a pure, allocation-free table.

use codevar_textlike_encode::encoding::{Encoding, UTF_8, UTF_16BE, UTF_16LE};
use codevar_textlike_encode::legacy::SINGLE_BYTE_ENCODINGS;

use crate::base::{
    ENC_SINGLE_BYTE, ENC_SINGLE_BYTE_LAST, ENC_UTF8, ENC_UTF16BE, ENC_UTF16LE, ENC_UTF32BE, ENC_UTF32LE,
};

/// Number of selectable single-byte encodings.
///
/// The value matches the length of
/// [`codevar_textlike_encode::legacy::SINGLE_BYTE_ENCODINGS`] and the
/// width of the [`crate::base::ENC_SINGLE_BYTE`] tag range.
pub const SINGLE_BYTE_COUNT: u8 = 28;

/// A document encoding supported by the engine.
///
/// The Unicode variants describe how bytes are decoded into scalar
/// values and encoded back out.  The single-byte variant selects one
/// of the tables exported by `codevar-textlike-encode`; the index is
/// stable and is exactly the [`SINGLE_BYTE_ENCODINGS`] order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DocumentEncoding {
    /// UTF-8, the default for new documents.
    Utf8,
    /// UTF-16 little-endian.
    Utf16Le,
    /// UTF-16 big-endian.
    Utf16Be,
    /// UTF-32 little-endian (UTF-32 is not part of the WHATWG
    /// Encoding Standard, so this codec is implemented by Wredit).
    Utf32Le,
    /// UTF-32 big-endian.
    Utf32Be,
    /// One of the WHATWG single-byte encodings, by index into
    /// [`SINGLE_BYTE_ENCODINGS`].
    SingleByte(u8),
}

impl DocumentEncoding {
    /// The stable on-disk/API tag for this encoding.
    #[must_use]
    #[inline]
    pub const fn tag(self) -> u16 {
        match self {
            Self::Utf8 => ENC_UTF8,
            Self::Utf16Le => ENC_UTF16LE,
            Self::Utf16Be => ENC_UTF16BE,
            Self::Utf32Le => ENC_UTF32LE,
            Self::Utf32Be => ENC_UTF32BE,
            Self::SingleByte(index) => ENC_SINGLE_BYTE + index as u16,
        }
    }

    /// Resolves a tag produced by [`Self::tag`].
    ///
    /// Returns `None` for [`crate::base::ENC_UNKNOWN`] and for every
    /// tag outside the documented ranges; callers must not silently
    /// clamp an unknown encoding to a default.
    #[must_use]
    #[inline]
    pub const fn from_tag(tag: u16) -> Option<Self> {
        match tag {
            ENC_UTF8 => Some(Self::Utf8),
            ENC_UTF16LE => Some(Self::Utf16Le),
            ENC_UTF16BE => Some(Self::Utf16Be),
            ENC_UTF32LE => Some(Self::Utf32Le),
            ENC_UTF32BE => Some(Self::Utf32Be),
            tag if tag >= ENC_SINGLE_BYTE && tag <= ENC_SINGLE_BYTE_LAST => {
                Some(Self::SingleByte((tag - ENC_SINGLE_BYTE) as u8))
            }
            _ => None,
        }
    }

    /// `true` when this variant selects a valid single-byte table.
    #[must_use]
    #[inline]
    pub const fn is_valid(self) -> bool {
        match self {
            Self::SingleByte(index) => index < SINGLE_BYTE_COUNT,
            _ => true,
        }
    }

    /// `true` for the three Unicode transfer formats.
    #[must_use]
    #[inline]
    pub const fn is_unicode(self) -> bool {
        !matches!(self, Self::SingleByte(_))
    }

    /// `true` for the UTF-32 variants, whose codecs are Wredit-owned.
    #[must_use]
    #[inline]
    pub const fn is_utf32(self) -> bool {
        matches!(self, Self::Utf32Le | Self::Utf32Be)
    }

    /// Human-readable encoding name.
    ///
    /// Returns `"single-byte"` for an out-of-range index rather than
    /// panicking; [`Self::is_valid`] distinguishes that case.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Utf8 => UTF_8.name,
            Self::Utf16Le => UTF_16LE.name,
            Self::Utf16Be => UTF_16BE.name,
            Self::Utf32Le => "UTF-32LE",
            Self::Utf32Be => "UTF-32BE",
            Self::SingleByte(index) => SINGLE_BYTE_ENCODINGS
                .get(index as usize)
                .map_or("single-byte", |encoding| encoding.name),
        }
    }

    /// The `codevar-textlike-encode` metadata for the encodings that
    /// crate implements.
    ///
    /// UTF-32 is absent from the WHATWG Encoding Standard and therefore
    /// returns `None`; use [`Self::is_utf32`] to select the Wredit
    /// codec.
    #[must_use]
    pub fn textlike(self) -> Option<&'static Encoding> {
        match self {
            Self::Utf8 => Some(&UTF_8),
            Self::Utf16Le => Some(&UTF_16LE),
            Self::Utf16Be => Some(&UTF_16BE),
            Self::Utf32Le | Self::Utf32Be => None,
            Self::SingleByte(index) => SINGLE_BYTE_ENCODINGS.get(index as usize).copied(),
        }
    }

    /// Maps a `codevar-textlike-encode` encoding back to a document
    /// encoding, or `None` when the crate has no matching variant
    /// (for example UTF-32, which the standard does not define).
    #[must_use]
    pub fn from_textlike(encoding: &'static Encoding) -> Option<Self> {
        if core::ptr::eq(encoding, &UTF_8) {
            return Some(Self::Utf8);
        }
        if core::ptr::eq(encoding, &UTF_16LE) {
            return Some(Self::Utf16Le);
        }
        if core::ptr::eq(encoding, &UTF_16BE) {
            return Some(Self::Utf16Be);
        }
        SINGLE_BYTE_ENCODINGS
            .iter()
            .position(|candidate| core::ptr::eq(*candidate, encoding))
            .and_then(|index| u8::try_from(index).ok())
            .map(Self::SingleByte)
    }

    /// Resolves a WHATWG encoding label such as `"utf-8"` or
    /// `"windows-1252"`.
    ///
    /// Labels that name encodings this workspace does not implement
    /// return `None`; they are not silently mapped to a fallback.
    #[must_use]
    pub fn for_label(label: &str) -> Option<Self> {
        codevar_textlike_encode::legacy::for_label(label).and_then(Self::from_textlike)
    }

    /// The byte-order mark emitted by this encoding, or an empty slice
    /// for single-byte encodings.
    #[must_use]
    #[inline]
    pub const fn bom(self) -> &'static [u8] {
        match self {
            Self::Utf8 => &[0xEF, 0xBB, 0xBF],
            Self::Utf16Le => &[0xFF, 0xFE],
            Self::Utf16Be => &[0xFE, 0xFF],
            Self::Utf32Le => &[0xFF, 0xFE, 0x00, 0x00],
            Self::Utf32Be => &[0x00, 0x00, 0xFE, 0xFF],
            Self::SingleByte(_) => &[],
        }
    }

    /// Removes the encoding's own byte-order mark from the head of a
    /// byte stream.
    ///
    /// A BOM belonging to a different encoding is left untouched; use
    /// [`Self::sniff_bom`] first when the encoding is not yet known.
    #[must_use]
    pub fn strip_bom(self, bytes: &[u8]) -> &[u8] {
        let bom = self.bom();
        if !bom.is_empty() && bytes.starts_with(bom) {
            &bytes[bom.len()..]
        } else {
            bytes
        }
    }

    /// Identifies the encoding from a leading byte-order mark.
    ///
    /// UTF-32 marks are tested before their UTF-16 prefixes: the
    /// little-endian `FF FE 00 00` stream also starts with the UTF-16LE
    /// mark `FF FE`, so testing UTF-16 first would misclassify it.
    #[must_use]
    pub fn sniff_bom(bytes: &[u8]) -> Option<Self> {
        if bytes.starts_with(&[0x00, 0x00, 0xFE, 0xFF]) {
            Some(Self::Utf32Be)
        } else if bytes.starts_with(&[0xFF, 0xFE, 0x00, 0x00]) {
            Some(Self::Utf32Le)
        } else if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
            Some(Self::Utf8)
        } else if bytes.starts_with(&[0xFE, 0xFF]) {
            Some(Self::Utf16Be)
        } else if bytes.starts_with(&[0xFF, 0xFE]) {
            Some(Self::Utf16Le)
        } else {
            None
        }
    }
}

impl Default for DocumentEncoding {
    /// New documents default to UTF-8.
    #[inline]
    fn default() -> Self {
        Self::Utf8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{ENC_SINGLE_BYTE, ENC_SINGLE_BYTE_COUNT, ENC_SINGLE_BYTE_LAST, ENC_UNKNOWN};

    #[test]
    fn unicode_tags_round_trip() {
        for encoding in [
            DocumentEncoding::Utf8,
            DocumentEncoding::Utf16Le,
            DocumentEncoding::Utf16Be,
            DocumentEncoding::Utf32Le,
            DocumentEncoding::Utf32Be,
        ] {
            assert!(encoding.is_valid());
            assert_eq!(DocumentEncoding::from_tag(encoding.tag()), Some(encoding));
        }
    }

    #[test]
    fn every_single_byte_tag_round_trips() {
        for index in 0..ENC_SINGLE_BYTE_COUNT {
            let encoding = DocumentEncoding::SingleByte(index as u8);
            assert!(encoding.is_valid());
            assert_eq!(encoding.tag(), ENC_SINGLE_BYTE + index);
            assert_eq!(
                DocumentEncoding::from_tag(encoding.tag()),
                Some(DocumentEncoding::SingleByte(index as u8))
            );
            assert!(encoding.textlike().is_some());
            assert!(!encoding.is_unicode());
        }
        assert_eq!(
            DocumentEncoding::from_tag(ENC_SINGLE_BYTE_LAST),
            Some(DocumentEncoding::SingleByte((ENC_SINGLE_BYTE_COUNT - 1) as u8))
        );
    }

    #[test]
    fn unknown_and_out_of_range_tags_resolve_to_none() {
        assert_eq!(DocumentEncoding::from_tag(ENC_UNKNOWN), None);
        assert_eq!(DocumentEncoding::from_tag(0x6), None);
        assert_eq!(DocumentEncoding::from_tag(0xF), None);
        assert_eq!(DocumentEncoding::from_tag(ENC_SINGLE_BYTE_LAST + 1), None);
        assert_eq!(DocumentEncoding::from_tag(u16::MAX), None);
    }

    #[test]
    fn invalid_single_byte_index_is_reported_not_clamped() {
        let encoding = DocumentEncoding::SingleByte(SINGLE_BYTE_COUNT);
        assert!(!encoding.is_valid());
        assert_eq!(encoding.name(), "single-byte");
        assert!(encoding.textlike().is_none());
    }

    #[test]
    fn utf8_and_utf16_map_to_textlike_metadata() {
        assert!(core::ptr::eq(
            DocumentEncoding::Utf8
                .textlike()
                .unwrap_or(&UTF_8),
            &UTF_8
        ));
        assert!(DocumentEncoding::Utf32Le.textlike().is_none());
        assert!(DocumentEncoding::Utf32Be.textlike().is_none());
        assert_eq!(
            DocumentEncoding::from_textlike(&UTF_8),
            Some(DocumentEncoding::Utf8)
        );
        assert_eq!(
            DocumentEncoding::from_textlike(&UTF_16LE),
            Some(DocumentEncoding::Utf16Le)
        );
        assert_eq!(
            DocumentEncoding::from_textlike(&UTF_16BE),
            Some(DocumentEncoding::Utf16Be)
        );
    }

    #[test]
    fn label_lookup_selects_the_expected_encoding() {
        assert_eq!(DocumentEncoding::for_label("utf-8"), Some(DocumentEncoding::Utf8));
        assert_eq!(
            DocumentEncoding::for_label("UTF-16LE"),
            Some(DocumentEncoding::Utf16Le)
        );
        assert_eq!(
            DocumentEncoding::for_label("windows-1252"),
            Some(DocumentEncoding::SingleByte(20))
        );
        assert_eq!(DocumentEncoding::for_label("not-a-real-encoding"), None);
    }

    #[test]
    fn bom_sniffing_prefers_utf32_over_utf16_prefixes() {
        assert_eq!(
            DocumentEncoding::sniff_bom(&[0xFF, 0xFE, 0x00, 0x00, 0x41]),
            Some(DocumentEncoding::Utf32Le)
        );
        assert_eq!(
            DocumentEncoding::sniff_bom(&[0x00, 0x00, 0xFE, 0xFF, 0x41]),
            Some(DocumentEncoding::Utf32Be)
        );
        assert_eq!(
            DocumentEncoding::sniff_bom(&[0xEF, 0xBB, 0xBF, 0x41]),
            Some(DocumentEncoding::Utf8)
        );
        assert_eq!(
            DocumentEncoding::sniff_bom(&[0xFE, 0xFF, 0x00, 0x41]),
            Some(DocumentEncoding::Utf16Be)
        );
        assert_eq!(
            DocumentEncoding::sniff_bom(&[0xFF, 0xFE, 0x41, 0x00]),
            Some(DocumentEncoding::Utf16Le)
        );
        assert_eq!(DocumentEncoding::sniff_bom(b"plain"), None);
    }

    #[test]
    fn strip_bom_only_removes_the_matching_mark() {
        let utf8 = [0xEF, 0xBB, 0xBF, b'a'];
        assert_eq!(DocumentEncoding::Utf8.strip_bom(&utf8), b"a");
        assert_eq!(DocumentEncoding::Utf16Le.strip_bom(&utf8), &utf8);
        assert_eq!(DocumentEncoding::Utf8.strip_bom(b"abc"), b"abc");
    }

    #[test]
    fn default_is_utf8() {
        assert_eq!(DocumentEncoding::default(), DocumentEncoding::Utf8);
    }
}
