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

//! Base64 (RFC 4648) helpers for WebSocket handshake material.
//!
//! Encoding and decoding are delegated to the shared SIMD-powered
//! codec in [`codevar_base::basic_base64`]; this module only adapts
//! its error type to [`WsError::Decode`](crate::ws_error::WsError).
//! Only the standard alphabet with `=` padding is supported, which is
//! sufficient for `Sec-WebSocket-Key` (16 random bytes → 24 chars) and
//! `Sec-WebSocket-Accept` (20-byte SHA-1 → 28 chars).

use crate::ws_error::{WsError, WsResult};
use codevar_base::basic_base64;

/// Encodes `input` as a Base64 string with standard padding.
#[must_use]
pub fn encode(input: &[u8]) -> String {
    basic_base64::encode(input)
}

/// Decodes a standard Base64 string into bytes.
///
/// Accepts optional whitespace; rejects non-canonical padding and alphabet
/// characters outside the standard Base64 set.
///
/// # Errors
///
/// Maps [`basic_base64::Base64Error`](codevar_base::basic_base64::Base64Error)
/// onto [`WsError::Decode`] with the original message text.
pub fn decode(input: &str) -> WsResult<Vec<u8>> {
    basic_base64::decode(input).map_err(|error| WsError::decode(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_rfc4648_test_vectors() {
        let vectors: [(&[u8], &str); 8] = [
            (b"", ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
            (b"Man", "TWFu"),
        ];
        for (input, expected) in vectors {
            assert_eq!(encode(input), expected, "input {input:?}");
        }
    }

    #[test]
    fn decodes_rfc4648_test_vectors() {
        let vectors: [(&str, &[u8]); 8] = [
            ("", b""),
            ("Zg==", b"f"),
            ("Zm8=", b"fo"),
            ("Zm9v", b"foo"),
            ("Zm9vYg==", b"foob"),
            ("Zm9vYmE=", b"fooba"),
            ("Zm9vYmFy", b"foobar"),
            ("TWFu", b"Man"),
        ];
        for (input, expected) in vectors {
            assert_eq!(decode(input).unwrap(), expected, "input {input:?}");
        }
    }

    #[test]
    fn round_trips_every_byte_value() {
        let data: Vec<u8> = (0u16..=0xFF).map(|i| u8::try_from(i).unwrap()).collect();
        let encoded = encode(&data);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn round_trips_input_lengths_zero_to_eight() {
        for len in 0..=8usize {
            let input: Vec<u8> = (0..len)
                .map(|i| u8::try_from((i * 31 + 7) % 256).unwrap())
                .collect();
            let encoded = encode(&input);
            assert_eq!(encoded.len(), len.div_ceil(3) * 4, "len {len}");
            let decoded = decode(&encoded).unwrap();
            assert_eq!(decoded, input, "len {len}");
        }
    }

    #[test]
    fn encode_length_follows_ceil_formula() {
        for n in 0..=16usize {
            let input = vec![0xABu8; n];
            assert_eq!(encode(&input).len(), n.div_ceil(3) * 4, "n {n}");
        }
    }

    #[test]
    fn encode_uses_standard_alphabet_with_padding() {
        let all_ones = [0xFFu8; 3];
        assert_eq!(encode(&all_ones), "////");

        let plus_slash = [0xFB, 0xF0, 0x00];
        let s = encode(&plus_slash);
        assert!(s.contains('+') || s.contains('/'), "encoded: {s}");

        assert_eq!(encode(&[0x00]), "AA==");
        assert_eq!(encode(&[0x00, 0x00]), "AAA=");
        assert_eq!(encode(&[0x00, 0x00, 0x00]), "AAAA");
    }

    #[test]
    fn decode_accepts_ascii_whitespace_anywhere() {
        assert_eq!(decode("Zg==\n").unwrap(), vec![b'f']);
        assert_eq!(decode(" Zg== ").unwrap(), vec![b'f']);
        assert_eq!(decode("Zg\t==\r\n").unwrap(), vec![b'f']);
        assert_eq!(decode(" Zm 8 = ").unwrap(), b"fo".to_vec());
        assert_eq!(decode("   ").unwrap(), Vec::<u8>::new());
        assert_eq!(decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn decode_rejects_lengths_not_multiple_of_four() {
        for n in 0..8usize {
            if n % 4 == 0 {
                continue;
            }
            let input = "A".repeat(n);
            let err = decode(&input).unwrap_err();
            match err {
                WsError::Decode(msg) => {
                    assert!(msg.contains("multiple of 4"), "message: {msg}");
                }
                other => panic!("expected Decode error, got {other:?}"),
            }
        }
        assert!(decode("Zg").is_err());
        assert!(decode("Zg=").is_err());
        assert!(decode("Zg===").is_err());
    }

    #[test]
    fn decode_rejects_invalid_alphabet_characters() {
        for input in ["****", "-_-_", "!!!!", "Zg=!", "Zgé"] {
            let err = decode(input).unwrap_err();
            match err {
                WsError::Decode(msg) => {
                    assert!(msg.contains("invalid base64"), "input {input:?}: {msg}");
                }
                other => panic!("expected Decode for {input:?}, got {other:?}"),
            }
        }
        // URL-safe alphabet is intentionally unsupported: "-w==" decodes
        // byte 0xFB in RFC 4648 §5, but '-' is rejected here.
        assert_eq!(encode(&[0xFB]), "+w==");
        assert!(decode("-w==").is_err());
    }

    #[test]
    fn decode_rejects_malformed_padding_placement() {
        assert!(matches!(decode("AA=A"), Err(WsError::Decode(_))));
        assert!(matches!(decode("Zg=A"), Err(WsError::Decode(_))));
        assert!(matches!(decode("A==="), Err(WsError::Decode(_))));
        assert!(matches!(decode("=AAA"), Err(WsError::Decode(_))));
        assert!(matches!(decode("===="), Err(WsError::Decode(_))));
    }

    #[test]
    fn decode_handles_canonical_zero_padding_vectors() {
        assert_eq!(decode("AA==").unwrap(), vec![0u8]);
        assert_eq!(decode("AAA=").unwrap(), vec![0u8, 0u8]);
        assert_eq!(decode("AAAA").unwrap(), vec![0u8, 0u8, 0u8]);
        assert_eq!(decode("Zg==").unwrap(), vec![b'f']);
        assert_eq!(decode("TWFu").unwrap(), b"Man".to_vec());
    }

    #[test]
    fn decode_keeps_structural_padding_but_ignores_trailing_bits() {
        // Documents current behavior: non-zero trailing bits in the final
        // sextet are not rejected as non-canonical.
        assert_eq!(decode("AB==").unwrap(), vec![0u8]);
        // Z = 25 = 0b011001; only the top 4 bits feed the output byte,
        // so the result is 0b0000_0001 = 1 (trailing bits are ignored).
        assert_eq!(decode("AZ==").unwrap(), vec![1u8]);
    }

    #[test]
    fn one_mib_round_trip() {
        let data: Vec<u8> = (0..1024 * 1024)
            .map(|i| u8::try_from(i % 256).unwrap())
            .collect();
        let encoded = encode(&data);
        assert_eq!(encoded.len(), data.len().div_ceil(3) * 4);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn many_small_round_trips_reuse_buffers() {
        for i in 0..1000usize {
            let input: Vec<u8> = (0..(i % 17))
                .map(|j| u8::try_from((j * 7 + i) % 256).unwrap())
                .collect();
            let encoded = encode(&input);
            let decoded = decode(&encoded).unwrap();
            assert_eq!(decoded, input, "iteration {i}");
        }
    }

    #[test]
    fn encode_output_never_contains_whitespace_or_newlines() {
        let data: Vec<u8> = (0u16..=255).map(|i| u8::try_from(i).unwrap()).collect();
        let encoded = encode(&data);
        assert!(!encoded.chars().any(char::is_whitespace));
        assert!(
            encoded.chars().all(|c| {
                c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='
            })
        );
    }
}
