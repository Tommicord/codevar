//! Copyright 2026 Codevar Project
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

//! FcWare public compression API.
//!
//! Frame codecs share a three-byte magic prefix. Use [`compress`] /
//! [`decompress`] for byte payloads and [`compress_values`] /
//! [`decompress_values`] for [`u16`] streams.

use crate::compression_bitward::{bitward_decode, bitward_encode};
use crate::compression_delta::{delta_decode, delta_encode};
use crate::compression_dysu::{
    dynamic_substring_decode, dynamic_substring_encode, substring_decode, substring_encode,
};
use crate::compression_error::{CompressorError, CompressorResult};
use crate::compression_frame::FrameKind;
use crate::compression_huffman::{huffman_decode, huffman_encode};
use crate::compression_lzmatch::lz_match_decode;
use crate::compression_stream::{
    HASH_TABLE_SIZE, HISTORY_LIMIT, LzWorkspace, NO_POSITION, compress_block_into,
};
use crate::compression_valmap::{valmap_decode_frame, valmap_encode_frame};
use alloc::vec;
use alloc::vec::Vec;

use crate::compression_bits::BitLaneReader;
pub use crate::compression_bits::{BIT_LANES, BitLaneReader as BitReader};
pub use crate::compression_error::{
    CompressorResult as CompressionResult, InvalidCodePoint, InvalidControl, InvalidToken, LengthMismatch,
};
pub use crate::compression_frame::FrameKind as Frame;
pub use crate::compression_stream::{LzWorkspace as StreamWorkspace, StreamingEncoder};

/// Default block size for [`Codec::Dysu`].
pub const DEFAULT_DYSU_BLOCK: usize = 4096;

/// Byte-oriented FcWare codec selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Codec {
    /// LZ match / literal framing (`LM\x01`).
    LzMatch,
    /// Canonical Huffman (`HF\x01`).
    Huffman,
    /// Substring matches (`SX\x01`).
    Substring,
    /// Blocked dynamic substring matches (`DX\x01`).
    Dysu {
        /// Independent block size in bytes.
        block_size: usize,
    },
    /// Delta / run residuals (`DL\x01`).
    Delta,
}

/// Value-oriented ([`u16`]) FcWare codec selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueCodec {
    /// Bitward packed u16 stream (`BW\x01`).
    Bitward,
    /// Predictive dictionary coding (`DI\x01`).
    Dictionary,
}

impl Codec {
    /// Returns the frame kind written by this codec.
    #[inline]
    #[must_use]
    pub const fn frame_kind(self) -> FrameKind {
        match self {
            Self::LzMatch => FrameKind::LzMatch,
            Self::Huffman => FrameKind::Huffman,
            Self::Substring => FrameKind::Substring,
            Self::Dysu { .. } => FrameKind::Dysu,
            Self::Delta => FrameKind::Delta,
        }
    }

    /// Dysu codec using [`DEFAULT_DYSU_BLOCK`].
    #[inline]
    #[must_use]
    pub const fn dysu_default() -> Self {
        Self::Dysu {
            block_size: DEFAULT_DYSU_BLOCK,
        }
    }
}

impl ValueCodec {
    /// Returns the frame kind written by this value codec.
    #[inline]
    #[must_use]
    pub const fn frame_kind(self) -> FrameKind {
        match self {
            Self::Bitward => FrameKind::Bitward,
            Self::Dictionary => FrameKind::Dictionary,
        }
    }
}

/// Compresses `input` with the selected byte codec into a self-describing frame.
///
/// # Errors
///
/// Returns an [`CompressorError`] when the input exceeds frame limits or a codec fails.
#[inline]
pub fn compress(input: &[u8], codec: Codec) -> CompressorResult<Vec<u8>> {
    match codec {
        Codec::LzMatch => lz_match_encode(input),
        Codec::Huffman => huffman_encode(input),
        Codec::Substring => substring_encode(input),
        Codec::Dysu { block_size } => dynamic_substring_encode(input, block_size),
        Codec::Delta => delta_encode(input),
    }
}

/// Decompresses an FcWare byte frame, detecting the codec from the magic prefix.
///
/// # Errors
///
/// Returns [`CompressorError::InvalidFrame`] for unknown or value-oriented magics, or the
/// underlying codec error for truncated / corrupt payloads.
#[inline]
pub fn decompress(frame: &[u8]) -> CompressorResult<Vec<u8>> {
    let kind = FrameKind::from_magic(frame).ok_or(CompressorError::InvalidFrame)?;
    match kind {
        FrameKind::LzMatch => lz_match_decode(frame),
        FrameKind::Huffman => huffman_decode(frame),
        FrameKind::Substring => substring_decode(frame),
        FrameKind::Dysu => dynamic_substring_decode(frame),
        FrameKind::Delta => delta_decode(frame),
        FrameKind::Bitward | FrameKind::Dictionary => Err(CompressorError::InvalidFrame),
    }
}

/// Compresses [`u16`] values with a value codec.
///
/// # Errors
///
/// Returns an [`CompressorError`] when encoding fails or input is too large.
#[inline]
pub fn compress_values(values: &[u16], codec: ValueCodec) -> CompressorResult<Vec<u8>> {
    match codec {
        ValueCodec::Bitward => bitward_encode(values),
        ValueCodec::Dictionary => valmap_encode_frame(values),
    }
}

/// Decompresses a Bitward or Dictionary frame into [`u16`] values.
///
/// # Errors
///
/// Returns [`CompressorError::InvalidFrame`] for non-value frames, or a decode error.
#[inline]
pub fn decompress_values(frame: &[u8]) -> CompressorResult<Vec<u16>> {
    let kind = FrameKind::from_magic(frame).ok_or(CompressorError::InvalidFrame)?;
    match kind {
        FrameKind::Bitward => bitward_decode(frame),
        FrameKind::Dictionary => valmap_decode_frame(frame),
        _ => Err(CompressorError::InvalidFrame),
    }
}

/// Detects the frame kind from a magic prefix without decoding.
#[inline]
#[must_use]
pub fn detect_frame(frame: &[u8]) -> Option<FrameKind> {
    FrameKind::from_magic(frame)
}

/// Encodes `input` as an LZ-match frame using a temporary workspace.
pub fn lz_match_encode(input: &[u8]) -> CompressorResult<Vec<u8>> {
    let mut heads = vec![NO_POSITION; HASH_TABLE_SIZE];
    let mut previous = vec![NO_POSITION; HISTORY_LIMIT + 1];
    let mut workspace = LzWorkspace::new(&mut heads, &mut previous)?;
    let capacity = lz_match_bound(input.len());
    let mut output = vec![0u8; capacity];
    let written = compress_block_into(input, &mut output, &mut workspace)?;
    output.truncate(written);
    Ok(output)
}

/// Worst-case LZ-match frame size for `input_len` source bytes.
#[inline]
#[must_use]
pub const fn lz_match_bound(input_len: usize) -> usize {
    // header (7) + literal records (1 + 2 + payload) per u16::MAX chunk
    let chunks = input_len / (u16::MAX as usize) + 1;
    7 + input_len + chunks * 3
}

/// Creates a bit-lane reader over a Huffman-style MSB-first payload.
#[inline]
#[must_use]
pub fn bit_reader(data: &[u8], available_bits: usize) -> BitLaneReader<'_> {
    BitLaneReader::new(data, available_bits)
}

#[cfg(test)]
mod tests {
    use super::{
        Codec, DEFAULT_DYSU_BLOCK, ValueCodec, bit_reader, compress, compress_values, decompress,
        decompress_values, detect_frame, lz_match_bound, lz_match_encode,
    };
    use crate::CompressorError;
    use crate::compression_frame::FrameKind;
    use alloc::vec;
    use alloc::vec::Vec;

    fn roundtrip_bytes(input: &[u8], codec: Codec) {
        let frame = compress(input, codec).expect("compress");
        assert_eq!(detect_frame(&frame), Some(codec.frame_kind()));
        let decoded = decompress(&frame).expect("decompress");
        assert_eq!(decoded, input);
    }

    fn roundtrip_values(values: &[u16], codec: ValueCodec) {
        let frame = compress_values(values, codec).expect("compress_values");
        assert_eq!(detect_frame(&frame), Some(codec.frame_kind()));
        let decoded = decompress_values(&frame).expect("decompress_values");
        assert_eq!(decoded, values);
    }

    #[test]
    fn empty_roundtrips_all_byte_codecs() {
        for codec in [
            Codec::LzMatch,
            Codec::Huffman,
            Codec::Substring,
            Codec::dysu_default(),
            Codec::Delta,
        ] {
            roundtrip_bytes(b"", codec);
        }
    }

    #[test]
    fn compressible_and_random_roundtrips() {
        let repeated = b"abcabcabcabcabcabcabcabcabcabc";
        let mixed: Vec<u8> = (0..256)
            .map(|v| v as u8)
            .cycle()
            .take(512)
            .collect();
        for codec in [
            Codec::LzMatch,
            Codec::Huffman,
            Codec::Substring,
            Codec::Dysu { block_size: 64 },
            Codec::Delta,
        ] {
            roundtrip_bytes(repeated, codec);
            roundtrip_bytes(&mixed, codec);
        }
    }

    #[test]
    fn value_codec_roundtrips() {
        let values = [0u16, 1, 1, 1, 0x0101, 0x00ff, 0xff00, 42, 42, 7];
        roundtrip_values(&values, ValueCodec::Bitward);
        roundtrip_values(&values, ValueCodec::Dictionary);
        roundtrip_values(&[], ValueCodec::Bitward);
        roundtrip_values(&[], ValueCodec::Dictionary);
    }

    #[test]
    fn decompress_rejects_value_frames_and_garbage() {
        let bitward = compress_values(&[1, 2, 3], ValueCodec::Bitward).expect("bw");
        assert_eq!(decompress(&bitward), Err(CompressorError::InvalidFrame));
        assert_eq!(decompress(b"XX\x01"), Err(CompressorError::InvalidFrame));
        assert_eq!(decompress(b""), Err(CompressorError::InvalidFrame));
        assert_eq!(decompress_values(b"LM\x01"), Err(CompressorError::InvalidFrame));
    }

    #[test]
    fn lz_match_bound_covers_encode() {
        let input = vec![0xABu8; 1000];
        let frame = lz_match_encode(&input).expect("encode");
        assert!(frame.len() <= lz_match_bound(input.len()));
        assert_eq!(decompress(&frame).expect("decode"), input);
    }

    #[test]
    fn dysu_default_block_constant() {
        assert_eq!(DEFAULT_DYSU_BLOCK, 4096);
        assert_eq!(
            Codec::dysu_default(),
            Codec::Dysu {
                block_size: DEFAULT_DYSU_BLOCK
            }
        );
    }

    #[test]
    fn bit_reader_smoke() {
        let data = [0b1010_0000];
        let mut reader = bit_reader(&data, 4);
        assert_eq!(reader.next_bit().expect("b0"), 1);
        assert_eq!(reader.next_bit().expect("b1"), 0);
        assert_eq!(reader.next_bit().expect("b2"), 1);
        assert_eq!(reader.next_bit().expect("b3"), 0);
        assert!(reader.next_bit().is_err());
    }

    #[test]
    fn frame_kind_mapping() {
        assert_eq!(Codec::LzMatch.frame_kind(), FrameKind::LzMatch);
        assert_eq!(ValueCodec::Bitward.frame_kind(), FrameKind::Bitward);
    }
}
