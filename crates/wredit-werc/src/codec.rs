// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

//! The driver: one bit loop that serves both directions.
//!
//! [`encode`] and [`decode`] differ in three places only - which side knows
//! the bit, whether a byte is appended to the output, and whether the coder
//! runs at all - so they share one private loop rather than each carrying a copy. A
//! model or coder change lands in both directions at once, which is the only
//! reason a stream encoded on one build decodes on another.
//!
//! A pass is a list of blocks. A priming block is a dictionary: the model
//! predicts and updates on every bit of it, but the coder never sees it, and
//! the byte history is dropped afterwards - the decoder cannot confirm a
//! match into text it does not have, so a match model that could reach the
//! dictionary would predict bytes no one can verify.

use alloc::vec;
use alloc::vec::Vec;

use crate::coder::{Decoder, Encoder};
use crate::error::{WercError, WercResult};
use crate::frame::{HEADER_SIZE, Header, Params};
use crate::model::Model;
use crate::plane;

/// Which side of the stream a pass is on.
enum Side<'a> {
    /// The bits are known before the coder sees them.
    Encode(Encoder),
    /// The bits come out of the payload.
    Decode(Decoder<'a>),
}

/// The encode side, which borrows nothing.
fn encoding() -> Side<'static> {
    Side::Encode(Encoder::new())
}

/// The decode side, reading `payload`.
fn decoding(payload: &[u8]) -> Side<'_> {
    Side::Decode(Decoder::new(payload))
}

impl Side<'_> {
    /// Code one bit with probability `p` and return the bit that was coded.
    ///
    /// A priming block hands the caller's bit straight back: the model learns
    /// from it, and nothing is written to or read from the stream.
    #[inline]
    fn code(&mut self, p: u32, bit: u32, priming: bool) -> u32 {
        if priming {
            return bit;
        }
        match self {
            Self::Encode(coder) => {
                coder.encode_bit(p, bit != 0);
                bit
            }
            Self::Decode(coder) => u32::from(coder.decode_bit(p)),
        }
    }

    /// The coded payload, with the interval flushed, or nothing on a decode.
    fn finish(self) -> Vec<u8> {
        match self {
            Self::Encode(mut coder) => {
                coder.finish();
                coder.into_bytes()
            }
            Self::Decode(_) => Vec::new(),
        }
    }
}

/// A run of bytes inside a pass.
struct Block<'a> {
    /// The bytes to code; empty whenever the coder produces them.
    source: &'a [u8],
    /// Bytes this block codes.
    len: usize,
    /// Fold into the model without touching the stream.
    priming: bool,
}

impl<'a> Block<'a> {
    /// A dictionary block: known bytes, no coder.
    fn priming(source: &'a [u8]) -> Self {
        Self {
            source,
            len: source.len(),
            priming: true,
        }
    }

    /// The block the stream carries.
    fn data(source: &'a [u8]) -> Self {
        Self {
            source,
            len: source.len(),
            priming: false,
        }
    }

    /// The block the decoder has to produce.
    fn to_decode(len: usize) -> Self {
        Self {
            source: &[],
            len,
            priming: false,
        }
    }
}

/// Code or decode `blocks`, appending the bytes a decoder produced to `out`.
///
/// `out` stays empty on the encode side, where the bytes are already known.
/// The numbers are the same on both sides because the model is asked for a
/// probability *before* the bit is known, so both train it identically.
fn run(model: &mut Model, blocks: &[Block<'_>], side: &mut Side<'_>, out: &mut Vec<u8>) {
    for block in blocks {
        // The bits fed to the model are known whenever the bytes are known,
        // which is true of a priming block even on the decode side.
        let from_input = matches!(side, Side::Encode(_)) || block.priming;
        for i in 0..block.len {
            let byte_in = block.source.get(i).copied().unwrap_or_default();
            let mut c0 = 1u32;
            let mut byte_out = 0u8;
            for nbits in 0..8u32 {
                let p = model.predict(c0, nbits).cast_unsigned();
                let bit = side.code(p, (u32::from(byte_in) >> (7 - nbits)) & 1, block.priming);
                model.update(bit);
                c0 = (c0 << 1) | bit;
                byte_out = (byte_out << 1) | bit as u8;
            }
            if !from_input {
                out.push(byte_out);
            }
            model.byte_update(if from_input { byte_in } else { byte_out });
        }
        if block.priming {
            model.reset_history();
        }
    }
}

/// The dictionary as the plane bytes both sides prime on.
///
/// Derived from the same elements on both sides, so it costs nothing on the
/// wire. A dictionary that cannot be laid out is an error rather than a
/// silently empty priming pass: the two sides would then train different
/// models.
fn dictionary_planes(dictionary: &[u32]) -> WercResult<Vec<u8>> {
    if dictionary.is_empty() {
        return Ok(Vec::new());
    }
    let flags = plane::presence(dictionary);
    let body_len =
        plane::body_len(dictionary.len(), flags).ok_or(WercError::Unsupported { field: "dictionary" })?;
    let mut planes = vec![0u8; body_len];
    let written = plane::split(dictionary, flags, &mut planes)?;
    planes.truncate(written);
    Ok(planes)
}

/// Compress `elements` into a complete frame.
///
/// `dictionary` is text both sides already have - the engine's boilerplate,
/// or the document's own prefix. It is not transmitted, and the model has
/// trained on it before the first coded byte, which is what lets a short
/// document reach a usable ratio at all.
///
/// # Errors
///
/// [`WercError::Unsupported`] if the parameters build no model, or if the
/// document or its coded payload is too large for the header's `u32` fields;
/// whatever stage 1 reports for a malformed plane layout.
///
/// # Examples
///
/// ```
/// use codevar_wredit_werc::codec::encode;
/// use codevar_wredit_werc::element::{COMMON_MASK, pack};
/// use codevar_wredit_werc::frame::{Header, Params, HEADER_SIZE};
///
/// let elements: Vec<u32> = "fn main() {}"
///     .bytes()
///     .map(|b| pack(u32::from(b), COMMON_MASK))
///     .collect();
/// let blob = encode(&Params::MINIMAL, &[], &elements).unwrap_or_default();
/// assert!(blob.len() > HEADER_SIZE);
/// let header = Header::parse(&blob).unwrap_or(Header::new(Params::MINIMAL, 0, 0, 0));
/// assert_eq!(header.count as usize, elements.len());
/// ```
pub fn encode(params: &Params, dictionary: &[u32], elements: &[u32]) -> WercResult<Vec<u8>> {
    let count = u32::try_from(elements.len()).map_err(|_| WercError::Unsupported { field: "count" })?;
    let flags = plane::presence(elements);
    let body_len = plane::body_len(elements.len(), flags).ok_or(WercError::Unsupported { field: "count" })?;
    let mut planes = vec![0u8; body_len];
    let written = plane::split(elements, flags, &mut planes)?;
    planes.truncate(written);

    let dict = dictionary_planes(dictionary)?;
    let blocks = [Block::priming(&dict), Block::data(&planes)];
    let mut model = params.model()?;
    let mut side = encoding();
    let mut discarded = Vec::new();
    run(&mut model, &blocks, &mut side, &mut discarded);
    let payload = side.finish();

    let payload_len =
        u32::try_from(payload.len()).map_err(|_| WercError::Unsupported { field: "payload" })?;
    let header = Header::new(*params, flags, count, payload_len);
    let mut head = [0u8; HEADER_SIZE];
    header.write_to(&mut head)?;
    let mut out = Vec::with_capacity(HEADER_SIZE + payload.len());
    out.extend_from_slice(&head);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Restore the elements a frame holds.
///
/// The model parameters come from the frame rather than from the caller, so a
/// stream cannot be decoded against a model that did not produce it, and
/// `dictionary` must be the same array the encoder primed with, in the same
/// order.
///
/// # Errors
///
/// [`WercError::NotWezc`] if the magic is foreign,
/// [`WercError::InputTooShort`] if the frame is shorter than its header
/// claims, [`WercError::OutputTooSmall`] if `elements` cannot hold the
/// document's count, [`WercError::Unsupported`] if a declared parameter
/// builds no model, and whatever stage 1 reports for a malformed plane
/// layout. Bytes past the document's count in `elements` are left alone, so a
/// caller may pass a larger buffer than it needs.
///
/// # Examples
///
/// ```
/// use codevar_wredit_werc::codec::{decode, encode};
/// use codevar_wredit_werc::element::{COMMON_MASK, pack};
/// use codevar_wredit_werc::frame::Params;
///
/// let elements: Vec<u32> = "hello world"
///     .bytes()
///     .map(|b| pack(u32::from(b), COMMON_MASK))
///     .collect();
/// let blob = encode(&Params::MINIMAL, &[], &elements).unwrap_or_default();
/// let mut back = vec![0u32; elements.len()];
/// let n = decode(&blob, &[], &mut back).unwrap_or(0);
/// assert_eq!(&back[..n], elements.as_slice());
/// ```
pub fn decode(blob: &[u8], dictionary: &[u32], elements: &mut [u32]) -> WercResult<usize> {
    let header = Header::parse(blob)?;
    let body_len = match header.body_len() {
        Some(len) => len,
        None => return Err(WercError::Unsupported { field: "count" }),
    };
    let needed = HEADER_SIZE + header.payload_len as usize;
    if blob.len() < needed {
        return Err(WercError::InputTooShort {
            needed,
            available: blob.len(),
        });
    }
    let count = header.count as usize;
    if elements.len() < count {
        return Err(WercError::OutputTooSmall {
            needed: count,
            available: elements.len(),
        });
    }

    let dict = dictionary_planes(dictionary)?;
    let payload = &blob[HEADER_SIZE..needed];
    let blocks = [Block::priming(&dict), Block::to_decode(body_len)];
    let mut model = header.params.model()?;
    let mut side = decoding(payload);
    let mut planes = Vec::with_capacity(body_len);
    run(&mut model, &blocks, &mut side, &mut planes);
    plane::join(header.flags, &planes, &mut elements[..count])
}

#[cfg(test)]
mod tests {
    use super::{Block, Side, dictionary_planes, encoding, run};
    use crate::codec::{decode, encode};
    use crate::element::{COMMON_MASK, pack};
    use crate::error::WercError;
    use crate::frame::{HEADER_SIZE, Header, MAGIC, Params};
    use crate::plane;

    fn elements_of(text: &str) -> Vec<u32> {
        text.bytes()
            .map(|b| pack(u32::from(b), COMMON_MASK))
            .collect()
    }

    /// Code plane bytes directly, at the parameters the prototype reference
    /// vectors were produced with.
    fn coded(planes: &[u8], dictionary: &[u8]) -> Vec<u8> {
        let mut model = Params::MINIMAL
            .model()
            .unwrap_or_else(|_| unreachable!());
        let blocks = [Block::priming(dictionary), Block::data(planes)];
        let mut side = encoding();
        let mut out = Vec::new();
        run(&mut model, &blocks, &mut side, &mut out);
        side.finish()
    }

    const PLANES: &[u8] = b"fn main() {\n    let x = 1;\n}\n";

    #[test]
    fn the_payload_matches_the_python_reference() {
        // 29 plane bytes at (8, 4, 11), produced by tools/wezc.py. This pins
        // the whole driver against the prototype, not only the stages.
        #[rustfmt::skip]
        const GOLDEN: &[u8] = &[
            0x8A, 0x5B, 0xB6, 0x4B, 0x3C, 0xC3, 0xE9, 0xBE, 0x1B, 0x73, 0xF2, 0x4B, 0xCD, 0xF7,
            0xDE, 0x8D, 0x78, 0xA2, 0xE7, 0x65, 0x05, 0xCE, 0x30, 0xA5, 0xF2, 0x4F, 0xB6, 0xE3,
            0x9B, 0x71,
        ];
        assert_eq!(coded(PLANES, &[]), GOLDEN);
    }

    #[test]
    fn a_primed_pass_matches_the_python_reference() {
        // The same body primed with 19 dictionary bytes. The reference does
        // the predict/update loop on the dictionary and then drops the byte
        // history, which is what this checks the driver reproduces.
        #[rustfmt::skip]
        const GOLDEN: &[u8] = &[
            0x48, 0xA9, 0xD9, 0xCC, 0x25, 0xBC, 0x96, 0x47, 0xA0, 0x9E, 0x08, 0x36, 0x77, 0x46,
            0x21, 0x59, 0xA0, 0x62, 0x96, 0x41, 0xAC, 0x96, 0xDF, 0xB0, 0xB3,
        ];
        assert_eq!(coded(PLANES, b"    let value = 1;\n"), GOLDEN);
    }

    #[test]
    fn a_frame_round_trips_the_elements() {
        let source = elements_of("fn main() {\n    let x = 1 + 2;\n}\n");
        let blob = encode(&Params::MINIMAL, &[], &source).unwrap_or_default();
        let mut back = vec![0u32; source.len()];
        assert_eq!(decode(&blob, &[], &mut back).ok(), Some(source.len()));
        assert_eq!(&back[..source.len()], source.as_slice());
    }

    #[test]
    fn the_frame_declares_the_document_it_carries() {
        let source = elements_of("hello world");
        let blob = encode(&Params::MINIMAL, &[], &source).unwrap_or_default();
        let header = Header::parse(&blob).unwrap_or(Header::new(Params::MINIMAL, 0, 0, 0));
        assert_eq!(header.count as usize, source.len());
        assert_eq!(header.flags, plane::presence(&source));
        assert_eq!(
            header.payload_len as usize + HEADER_SIZE,
            blob.len(),
            "the frame is exactly its header and its payload"
        );
        assert_eq!(&blob[..MAGIC.len()], &MAGIC);
    }

    #[test]
    fn a_repetitive_document_gets_smaller() {
        let source = elements_of("let value = compute(a, b);\n").repeat(40);
        let blob = encode(&Params::MINIMAL, &[], &source).unwrap_or_default();
        assert!(
            blob.len() < source.len(),
            "{} element bytes in, {} out",
            source.len(),
            blob.len()
        );
    }

    #[test]
    fn a_dictionary_shrinks_the_stream_without_entering_it() {
        let dictionary = elements_of("let value = compute(a, b);\nlet other = compute(c, d);\n");
        let source = elements_of("let value = compute(a, b);\n");
        let with = encode(&Params::MINIMAL, &dictionary, &source).unwrap_or_default();
        let without = encode(&Params::MINIMAL, &[], &source).unwrap_or_default();
        assert!(
            with.len() < without.len(),
            "primed {} bytes, unprimed {}",
            with.len(),
            without.len()
        );
        // The dictionary is not in the payload: without it the model starts
        // cold and the same bytes come out as a different document.
        let mut back = vec![0u32; source.len()];
        assert_eq!(decode(&with, &[], &mut back).ok(), Some(source.len()));
        assert_ne!(&back[..source.len()], source.as_slice());
        assert_eq!(decode(&with, &dictionary, &mut back).ok(), Some(source.len()));
        assert_eq!(&back[..source.len()], source.as_slice());
    }

    #[test]
    fn an_empty_document_is_a_legal_frame() {
        let blob = encode(&Params::MINIMAL, &[], &[]).unwrap_or_default();
        let header = Header::parse(&blob).unwrap_or(Header::new(Params::MINIMAL, 0, 1, 1));
        assert_eq!((header.count, header.body_len()), (0, Some(0)));
        let mut back = [0u32; 1];
        assert_eq!(decode(&blob, &[], &mut back).ok(), Some(0));
    }

    #[test]
    fn wide_codepoints_keep_their_planes() {
        let source: Vec<u32> = "ab京\u{1F680}z"
            .chars()
            .map(|c| pack(c as u32, COMMON_MASK))
            .collect();
        let blob = encode(&Params::MINIMAL, &[], &source).unwrap_or_default();
        assert_eq!(
            Header::parse(&blob).map(|h| h.flags),
            Ok(plane::presence(&source))
        );
        let mut back = vec![0u32; source.len()];
        assert_eq!(decode(&blob, &[], &mut back).ok(), Some(source.len()));
        assert_eq!(&back[..source.len()], source.as_slice());
    }

    #[test]
    fn a_short_output_reports_the_count_it_needed() {
        let source = elements_of("four chars");
        let blob = encode(&Params::MINIMAL, &[], &source).unwrap_or_default();
        let mut back = vec![0u32; source.len() - 1];
        assert_eq!(
            decode(&blob, &[], &mut back),
            Err(WercError::OutputTooSmall {
                needed: source.len(),
                available: source.len() - 1
            })
        );
    }

    #[test]
    fn a_frame_shorter_than_it_claims_is_refused() {
        let source = elements_of("a document with some bytes in it");
        let mut blob = encode(&Params::MINIMAL, &[], &source).unwrap_or_default();
        let header = Header::parse(&blob).unwrap_or(Header::new(Params::MINIMAL, 0, 0, 0));
        blob.truncate(HEADER_SIZE + header.payload_len as usize - 1);
        assert_eq!(
            decode(&blob, &[], &mut vec![0u32; source.len()]),
            Err(WercError::InputTooShort {
                needed: blob.len() + 1,
                available: blob.len()
            })
        );
    }

    #[test]
    fn a_parameter_that_builds_no_model_stops_the_decode() {
        let source = elements_of("text");
        let mut blob = encode(&Params::MINIMAL, &[], &source).unwrap_or_default();
        blob[5] = 40; // mem_bits past the largest table this build sizes
        assert_eq!(
            decode(&blob, &[], &mut vec![0u32; source.len()]),
            Err(WercError::Unsupported { field: "mem_bits" })
        );
    }

    #[test]
    fn the_dictionary_is_the_same_planes_on_both_sides() {
        let dictionary = elements_of("shared boilerplate");
        let flags = plane::presence(&dictionary);
        assert_eq!(
            dictionary_planes(&dictionary)
                .unwrap_or_default()
                .len(),
            plane::body_len(dictionary.len(), flags).unwrap_or_default()
        );
        assert!(
            dictionary_planes(&[])
                .unwrap_or_default()
                .is_empty()
        );
    }

    #[test]
    fn a_decode_appends_planes_and_an_encode_does_not() {
        // The single loop must write each decoded byte exactly once, which is
        // what the frame's derived body length then agrees with.
        let mut out = Vec::new();
        let mut model = Params::MINIMAL
            .model()
            .unwrap_or_else(|_| unreachable!());
        let blocks = [Block::priming(b"ab"), Block::to_decode(3)];
        let mut side = Side::Decode(crate::coder::Decoder::new(&[]));
        run(&mut model, &blocks, &mut side, &mut out);
        assert_eq!(out.len(), 3, "only the coded block is appended");
    }
}
