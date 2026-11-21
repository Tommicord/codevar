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

use alloc::string::String;
use alloc::vec::Vec;

use codevar_logger::log_error;
use codevar_textlike_encode::encoding::{CoderResult, DecoderResult, EncoderResult};

use crate::encoding::{DocumentEncoding, is_valid_scalar};
use crate::error::{Error, Result};

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

/// Policy applied when a byte stream contains a malformed sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodePolicy {
    /// Reject the first malformed sequence.
    Strict,
    /// Substitute U+FFFD and continue decoding.
    Replace,
}

/// Policy applied when a scalar value has no mapping in the target
/// encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodePolicy {
    /// Reject the first unmappable scalar.
    Strict,
    /// Write an ASCII question mark (`?`) instead.
    Replace,
}

/// Size of the stack buffer used to drain the streaming decoders.
const DECODE_CHUNK: usize = 0x1000;

/// Decodes `bytes` from `encoding` into Unicode scalar values.
///
/// A leading BOM belonging to `encoding` is removed. `policy` controls
/// whether malformed input is rejected or replaced with U+FFFD.
///
/// # Errors
///
/// * [`Error::Encoding`] for an invalid [`DocumentEncoding`] or a
///   decoder state failure.
/// * [`Error::MalformedInput`] under [`DecodePolicy::Strict`].
pub fn decode(bytes: &[u8], encoding: DocumentEncoding, policy: DecodePolicy) -> Result<Vec<u32>> {
    if !encoding.is_valid() {
        return Err(Error::Encoding);
    }
    let bytes = encoding.strip_bom(bytes);
    if encoding.is_utf32() {
        decode_utf32(bytes, encoding, policy)
    } else {
        decode_textlike(bytes, encoding, policy)
    }
}

/// Decodes `bytes` into a `String`.
///
/// # Errors
///
/// Same as [`decode`].
pub fn decode_to_string(bytes: &[u8], encoding: DocumentEncoding, policy: DecodePolicy) -> Result<String> {
    let scalars = decode(bytes, encoding, policy)?;
    let mut text = String::with_capacity(scalars.len());
    for value in scalars {
        // SAFETY: `decode` only yields Unicode scalar values: the
        // UTF-32 path validates every word and the textlike path takes
        // its scalars from a checked `&str`, so the unchecked
        // conversion cannot see a surrogate or an out-of-range code
        // point.
        text.push(unsafe { char::from_u32_unchecked(value) });
    }
    Ok(text)
}

/// Encodes Unicode scalar values into `encoding`.
///
/// No byte-order mark is emitted; use [`encode_with_bom`] when the
/// caller wants one.
///
/// # Errors
///
/// * [`Error::Encoding`] for an invalid [`DocumentEncoding`] or a
///   scalar outside the Unicode range.
/// * [`Error::UnmappableCharacter`] under
///   [`EncodePolicy::Strict`] when the target cannot represent a
///   scalar.
pub fn encode(scalars: &[u32], encoding: DocumentEncoding, policy: EncodePolicy) -> Result<Vec<u8>> {
    if !encoding.is_valid() {
        return Err(Error::Encoding);
    }
    match encoding {
        DocumentEncoding::Utf8 => encode_utf8(scalars),
        DocumentEncoding::Utf16Le => encode_utf16(scalars, true),
        DocumentEncoding::Utf16Be => encode_utf16(scalars, false),
        DocumentEncoding::Utf32Le => encode_utf32(scalars, true),
        DocumentEncoding::Utf32Be => encode_utf32(scalars, false),
        DocumentEncoding::SingleByte(_) => encode_single_byte(scalars, encoding, policy),
    }
}

/// Encodes a `&str` into `encoding`.
///
/// # Errors
///
/// Same as [`encode`].
pub fn encode_from_str(text: &str, encoding: DocumentEncoding, policy: EncodePolicy) -> Result<Vec<u8>> {
    let scalars: Vec<u32> = text.chars().map(u32::from).collect();
    encode(&scalars, encoding, policy)
}

/// Encodes scalar values and prefixes the encoding's byte-order mark,
/// if it has one.
///
/// # Errors
///
/// Same as [`encode`].
pub fn encode_with_bom(scalars: &[u32], encoding: DocumentEncoding, policy: EncodePolicy) -> Result<Vec<u8>> {
    let bom = encoding.bom();
    let mut bytes = Vec::with_capacity(
        bom.len()
            .saturating_add(scalars.len().saturating_mul(4)),
    );
    bytes.extend_from_slice(bom);
    bytes.extend_from_slice(&encode(scalars, encoding, policy)?);
    Ok(bytes)
}

/// Decode status of one streaming iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeStatus {
    Done,
    Again,
    Malformed,
}

fn decode_textlike(bytes: &[u8], encoding: DocumentEncoding, policy: DecodePolicy) -> Result<Vec<u32>> {
    let Some(metadata) = encoding.textlike() else {
        return Err(Error::Encoding);
    };
    let mut decoder = metadata.new_decoder_without_bom_handling();
    let mut utf8 = Vec::with_capacity(bytes.len().saturating_mul(3));
    let mut chunk = [0u8; DECODE_CHUNK];
    let mut offset = 0usize;
    loop {
        let (status, read, written) = match policy {
            DecodePolicy::Strict => {
                let (result, read, written) =
                    decoder.decode_to_utf8_without_replacement(&bytes[offset..], &mut chunk, true);
                let status = match result {
                    DecoderResult::InputEmpty => DecodeStatus::Done,
                    DecoderResult::OutputFull => DecodeStatus::Again,
                    DecoderResult::Malformed(_, _) => DecodeStatus::Malformed,
                };
                (status, read, written)
            }
            DecodePolicy::Replace => {
                let (result, read, written, _) = decoder.decode_to_utf8(&bytes[offset..], &mut chunk, true);
                let status = match result {
                    CoderResult::InputEmpty => DecodeStatus::Done,
                    CoderResult::OutputFull => DecodeStatus::Again,
                };
                (status, read, written)
            }
        };
        utf8.extend_from_slice(&chunk[..written]);
        offset = offset.saturating_add(read);
        match status {
            DecodeStatus::Done => break,
            DecodeStatus::Again => {
                if read == 0 && written == 0 {
                    log_error!("wredit: decoder made no progress at byte {offset}");
                    return Err(Error::Encoding);
                }
            }
            DecodeStatus::Malformed => {
                return Err(Error::MalformedInput {
                    encoding: encoding.name(),
                    offset,
                });
            }
        }
    }
    let text = core::str::from_utf8(&utf8).map_err(|_| Error::Encoding)?;
    Ok(text.chars().map(u32::from).collect())
}

fn decode_utf32(bytes: &[u8], encoding: DocumentEncoding, policy: DecodePolicy) -> Result<Vec<u32>> {
    let little_endian = matches!(encoding, DocumentEncoding::Utf32Le);
    let mut scalars = Vec::with_capacity(bytes.len() / 4);
    let mut offset = 0usize;
    while offset + 4 <= bytes.len() {
        let word = if little_endian {
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        } else {
            u32::from_be_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        };
        if is_valid_scalar(word) {
            // Validating the code point directly avoids building a
            // `char` only to convert it back to `u32`.
            scalars.push(word);
        } else {
            match policy {
                DecodePolicy::Strict => {
                    return Err(Error::MalformedInput {
                        encoding: encoding.name(),
                        offset,
                    });
                }
                DecodePolicy::Replace => scalars.push(u32::from(char::REPLACEMENT_CHARACTER)),
            }
        }
        offset += 4;
    }
    if offset < bytes.len() {
        match policy {
            DecodePolicy::Strict => {
                return Err(Error::MalformedInput {
                    encoding: encoding.name(),
                    offset,
                });
            }
            DecodePolicy::Replace => scalars.push(u32::from(char::REPLACEMENT_CHARACTER)),
        }
    }
    Ok(scalars)
}

fn encode_utf8(scalars: &[u32]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(scalars.len().saturating_mul(3));
    let mut buffer = [0u8; 4];
    for &value in scalars {
        let character = scalar_to_char(value)?;
        bytes.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
    }
    Ok(bytes)
}

fn encode_utf16(scalars: &[u32], little_endian: bool) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(scalars.len().saturating_mul(2));
    let mut buffer = [0u16; 2];
    for &value in scalars {
        let character = scalar_to_char(value)?;
        for unit in character.encode_utf16(&mut buffer) {
            let encoded = if little_endian {
                unit.to_le_bytes()
            } else {
                unit.to_be_bytes()
            };
            bytes.extend_from_slice(&encoded);
        }
    }
    Ok(bytes)
}

fn encode_utf32(scalars: &[u32], little_endian: bool) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(scalars.len().saturating_mul(4));
    for &value in scalars {
        if !is_valid_scalar(value) {
            return Err(Error::Encoding);
        }
        // The scalar itself carries the UTF-32 payload, so the validated
        // `u32` is written directly instead of round-tripping through
        // `char`.
        let encoded = if little_endian {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        };
        bytes.extend_from_slice(&encoded);
    }
    Ok(bytes)
}

fn encode_single_byte(scalars: &[u32], encoding: DocumentEncoding, policy: EncodePolicy) -> Result<Vec<u8>> {
    let Some(metadata) = encoding.textlike() else {
        return Err(Error::Encoding);
    };
    let mut encoder = metadata.new_encoder();
    let mut bytes = Vec::with_capacity(scalars.len());
    let mut utf8_buffer = [0u8; 4];
    let mut byte_buffer = [0u8; 4];
    for &value in scalars {
        let character = scalar_to_char(value)?;
        let source = character.encode_utf8(&mut utf8_buffer);
        match encoder.encode_from_utf8_without_replacement(source, &mut byte_buffer, true) {
            (EncoderResult::InputEmpty, read, 1) if read == source.len() => bytes.push(byte_buffer[0]),
            (EncoderResult::Unmappable(_), _, _) | (EncoderResult::InputEmpty, _, 0) => match policy {
                EncodePolicy::Strict => {
                    return Err(Error::UnmappableCharacter { character });
                }
                EncodePolicy::Replace => bytes.push(b'?'),
            },
            (EncoderResult::OutputFull, _, _) => {
                return Err(Error::LimitExceeded {
                    what: "single-byte encode buffer",
                });
            }
            _ => return Err(Error::Encoding),
        }
    }
    Ok(bytes)
}

#[inline]
fn scalar_to_char(value: u32) -> Result<char> {
    char::from_u32(value).ok_or(Error::Encoding)
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

    #[test]
    fn utf8_round_trips_astral_characters() {
        let text = "aé€😀";
        let scalars: Vec<u32> = text.chars().map(u32::from).collect();
        let bytes = encode(&scalars, DocumentEncoding::Utf8, EncodePolicy::Strict).unwrap();
        assert_eq!(bytes, text.as_bytes());
        assert_eq!(
            decode(&bytes, DocumentEncoding::Utf8, DecodePolicy::Strict).unwrap(),
            scalars
        );
    }

    #[test]
    fn utf16_round_trips_both_endians() {
        let text = "aé€😀";
        let scalars: Vec<u32> = text.chars().map(u32::from).collect();
        for encoding in [DocumentEncoding::Utf16Le, DocumentEncoding::Utf16Be] {
            let bytes = encode(&scalars, encoding, EncodePolicy::Strict).unwrap();
            assert_eq!(decode(&bytes, encoding, DecodePolicy::Strict).unwrap(), scalars);
        }
    }

    #[test]
    fn utf32_round_trips_both_endians_and_rejects_partial_units() {
        let text = "aé€😀";
        let scalars: Vec<u32> = text.chars().map(u32::from).collect();
        for encoding in [DocumentEncoding::Utf32Le, DocumentEncoding::Utf32Be] {
            let bytes = encode(&scalars, encoding, EncodePolicy::Strict).unwrap();
            assert_eq!(bytes.len(), scalars.len() * 4);
            assert_eq!(decode(&bytes, encoding, DecodePolicy::Strict).unwrap(), scalars);

            let mut partial = bytes.clone();
            partial.push(0x41);
            assert!(matches!(
                decode(&partial, encoding, DecodePolicy::Strict),
                Err(Error::MalformedInput { .. })
            ));
            assert_eq!(
                decode(&partial, encoding, DecodePolicy::Replace)
                    .unwrap()
                    .last()
                    .copied(),
                Some(u32::from(char::REPLACEMENT_CHARACTER))
            );
        }
    }

    #[test]
    fn utf32_rejects_surrogate_and_out_of_range_scalars() {
        for word in [0xD800u32, 0xDFFF, 0x11_0000, u32::MAX] {
            let bytes = word.to_le_bytes();
            assert!(matches!(
                decode(&bytes, DocumentEncoding::Utf32Le, DecodePolicy::Strict),
                Err(Error::MalformedInput { .. })
            ));
        }
        assert!(matches!(
            encode(&[0xD800], DocumentEncoding::Utf8, EncodePolicy::Strict),
            Err(Error::Encoding)
        ));
    }

    #[test]
    fn utf8_strict_and_replacement_policies_differ() {
        let malformed = [0x61, 0xFF, 0x62];
        assert!(matches!(
            decode(&malformed, DocumentEncoding::Utf8, DecodePolicy::Strict),
            Err(Error::MalformedInput { .. })
        ));
        assert_eq!(
            decode(&malformed, DocumentEncoding::Utf8, DecodePolicy::Replace).unwrap(),
            vec![
                u32::from('a'),
                u32::from(char::REPLACEMENT_CHARACTER),
                u32::from('b')
            ]
        );
    }

    #[test]
    fn decoding_strips_a_matching_bom() {
        let scalars = [u32::from('a')];
        for encoding in [
            DocumentEncoding::Utf8,
            DocumentEncoding::Utf16Le,
            DocumentEncoding::Utf16Be,
            DocumentEncoding::Utf32Le,
            DocumentEncoding::Utf32Be,
        ] {
            let bytes = encode_with_bom(&scalars, encoding, EncodePolicy::Strict).unwrap();
            assert!(bytes.starts_with(encoding.bom()));
            assert_eq!(decode(&bytes, encoding, DecodePolicy::Strict).unwrap(), scalars);
        }
    }

    #[test]
    fn windows1252_round_trips_and_replaces_unmappable_scalars() {
        let encoding = DocumentEncoding::SingleByte(20);
        let euro = u32::from('€');
        let bytes = encode(&[euro], encoding, EncodePolicy::Strict).unwrap();
        assert_eq!(bytes, vec![0x80]);
        assert_eq!(
            decode(&bytes, encoding, DecodePolicy::Strict).unwrap(),
            vec![euro]
        );

        let snowman = u32::from('☃');
        assert!(matches!(
            encode(&[snowman], encoding, EncodePolicy::Strict),
            Err(Error::UnmappableCharacter { character: '☃' })
        ));
        assert_eq!(
            encode(&[snowman], encoding, EncodePolicy::Replace).unwrap(),
            vec![b'?']
        );
    }

    #[test]
    fn streaming_decoder_handles_inputs_larger_than_one_chunk() {
        let text: String = core::iter::repeat_n('a', DECODE_CHUNK * 2 + 17).collect();
        let scalars: Vec<u32> = text.chars().map(u32::from).collect();
        let bytes = encode_from_str(&text, DocumentEncoding::Utf8, EncodePolicy::Strict).unwrap();
        assert_eq!(
            decode(&bytes, DocumentEncoding::Utf8, DecodePolicy::Strict).unwrap(),
            scalars
        );
        assert_eq!(
            decode_to_string(&bytes, DocumentEncoding::Utf8, DecodePolicy::Strict).unwrap(),
            text
        );
    }

    #[test]
    fn empty_input_round_trips() {
        for encoding in [
            DocumentEncoding::Utf8,
            DocumentEncoding::Utf16Le,
            DocumentEncoding::Utf16Be,
            DocumentEncoding::Utf32Le,
            DocumentEncoding::Utf32Be,
            DocumentEncoding::SingleByte(20),
        ] {
            assert!(
                decode(&[], encoding, DecodePolicy::Strict)
                    .unwrap()
                    .is_empty()
            );
            assert!(
                encode(&[], encoding, EncodePolicy::Strict)
                    .unwrap()
                    .is_empty()
            );
        }
    }
}
