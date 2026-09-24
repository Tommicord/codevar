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

//! Incremental UTF-8 validation for WebSocket text payloads.
//!
//! RFC 6455 §5.6 permits individual text *frames* to contain partial UTF-8
//! sequences, but the reassembled *message* MUST be valid UTF-8. Close
//! frame reasons (RFC 6455 §5.5.1) must also be valid UTF-8. Invalid UTF-8
//! fails the connection with status code 1007 (RFC 6455 §8.1).

use crate::ws_error::{WsError, WsResult};

/// Streaming UTF-8 validator that accepts incomplete trailing sequences
/// across fragment boundaries.
///
/// # Example
///
/// ```
/// use codevar_wsocket::ws_utf8::Utf8Validator;
///
/// let mut v = Utf8Validator::new();
/// // Split multi-byte character across two fragments: U+00E9 (é) = C3 A9
/// if let Err(_) = v.feed(&[0xC3]) {
///     return;
/// }
/// if let Err(_) = v.feed(&[0xA9]) {
///     return;
/// }
/// if let Err(_) = v.finish() {
///     return;
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct Utf8Validator {
    /// Bytes of an incomplete multi-byte sequence carried across feeds.
    pending: [u8; 4],
    /// Number of valid bytes currently in `pending`.
    pending_len: u8,
    /// Expected total length of the multi-byte sequence being assembled.
    needed: u8,
}

impl Utf8Validator {
    /// Creates a fresh validator with no pending state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: [0; 4],
            pending_len: 0,
            needed: 0,
        }
    }

    /// Feeds the next fragment of a text message.
    ///
    /// Returns [`WsError::InvalidUtf8`] on a well-formedness failure.
    /// Incomplete sequences at the end of `bytes` are retained for the
    /// next call.
    pub fn feed(&mut self, bytes: &[u8]) -> WsResult<()> {
        let mut i = 0;
        while i < bytes.len() {
            if self.pending_len > 0 {
                let b = bytes[i];
                if !is_continuation(b) {
                    return Err(WsError::InvalidUtf8);
                }
                self.pending[self.pending_len as usize] = b;
                self.pending_len += 1;
                i += 1;
                if self.pending_len == self.needed {
                    validate_complete_sequence(&self.pending[..self.needed as usize])?;
                    self.pending_len = 0;
                    self.needed = 0;
                }
                continue;
            }

            let b = bytes[i];
            let width = utf8_width(b)?;
            if width == 1 {
                i += 1;
                continue;
            }
            let remaining = bytes.len() - i;
            if remaining >= width {
                validate_complete_sequence(&bytes[i..i + width])?;
                i += width;
            } else {
                self.pending[..remaining].copy_from_slice(&bytes[i..]);
                self.pending_len = remaining as u8;
                self.needed = width as u8;
                break;
            }
        }
        Ok(())
    }

    /// Completes validation; fails if a multi-byte sequence is still open.
    pub fn finish(&self) -> WsResult<()> {
        if self.pending_len == 0 {
            Ok(())
        } else {
            Err(WsError::InvalidUtf8)
        }
    }

    /// Resets to the empty state.
    pub fn reset(&mut self) {
        self.pending_len = 0;
        self.needed = 0;
    }
}

/// Validates that `bytes` is a complete, well-formed UTF-8 string.
///
/// Uses the table-driven validator from `codevar-textlike-encode`; the
/// streaming [`Utf8Validator`] remains for fragment boundaries where a
/// multi-byte sequence may be split across feeds.
pub fn validate_utf8(bytes: &[u8]) -> WsResult<()> {
    if codevar_textlike_encode::encoding_utf8::utf8_valid_up_to(bytes) == bytes.len() {
        Ok(())
    } else {
        Err(WsError::InvalidUtf8)
    }
}

fn utf8_width(lead: u8) -> WsResult<usize> {
    match lead {
        0x00..=0x7F => Ok(1),
        0xC2..=0xDF => Ok(2),
        0xE0..=0xEF => Ok(3),
        0xF0..=0xF4 => Ok(4),
        // Overlong / invalid lead bytes (C0, C1, F5–FF) and continuation
        // bytes used as leads are rejected.
        _ => Err(WsError::InvalidUtf8),
    }
}

const fn is_continuation(b: u8) -> bool {
    (b & 0xC0) == 0x80
}

fn validate_complete_sequence(seq: &[u8]) -> WsResult<()> {
    match seq {
        [b] if *b <= 0x7F => Ok(()),
        [a, b] if (0xC2..=0xDF).contains(a) && is_continuation(*b) => Ok(()),
        [0xE0, b, c] if (0xA0..=0xBF).contains(b) && is_continuation(*c) => Ok(()),
        [0xED, b, c] if (0x80..=0x9F).contains(b) && is_continuation(*c) => Ok(()),
        [a, b, c] if (0xE1..=0xEC).contains(a) || *a == 0xEE || *a == 0xEF => {
            if is_continuation(*b) && is_continuation(*c) {
                Ok(())
            } else {
                Err(WsError::InvalidUtf8)
            }
        }
        [0xF0, b, c, d]
            if (0x90..=0xBF).contains(b)
                && is_continuation(*c)
                && is_continuation(*d) =>
        {
            Ok(())
        }
        [a, b, c, d]
            if (0xF1..=0xF3).contains(a)
                && is_continuation(*b)
                && is_continuation(*c)
                && is_continuation(*d) =>
        {
            Ok(())
        }
        [0xF4, b, c, d]
            if (0x80..=0x8F).contains(b)
                && is_continuation(*c)
                && is_continuation(*d) =>
        {
            Ok(())
        }
        _ => Err(WsError::InvalidUtf8),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPLIT_SAMPLE: &str = "héllo wörld — € 🙂 end";

    const VALID_SEQUENCES: &[&[u8]] = &[
        &[0x00],
        &[0x7F],
        &[0xC2, 0x80],
        &[0xC3, 0xA9],
        &[0xDF, 0xBF],
        &[0xE0, 0xA0, 0x80],
        &[0xE1, 0x80, 0x80],
        &[0xED, 0x9F, 0xBF],
        &[0xEE, 0x80, 0x80],
        &[0xEF, 0xBF, 0xBD],
        &[0xF0, 0x90, 0x80, 0x80],
        &[0xF0, 0x9F, 0x99, 0x82],
        &[0xF4, 0x8F, 0xBF, 0xBF],
    ];

    const INVALID_SEQUENCES: &[&[u8]] = &[
        &[0x80],
        &[0xBF],
        &[0xC0, 0x80],
        &[0xC1, 0xBF],
        &[0xE0, 0x80, 0x80],
        &[0xE0, 0x9F, 0xBF],
        &[0xED, 0xA0, 0x80],
        &[0xED, 0xBF, 0xBF],
        &[0xF0, 0x80, 0x80, 0x80],
        &[0xF4, 0x90, 0x80, 0x80],
        &[0xF5, 0x80, 0x80, 0x80],
        &[0xFE],
        &[0xFF],
        &[0xC3],
        &[0xE2, 0x82],
        &[0xF0, 0x9F],
        &[0x41, 0xFF],
        &[0xE2, 0x41, 0x80],
        &[0xE2, 0x82, 0x41],
        &[0xF0, 0x9F, 0x41, 0x80],
    ];

    #[test]
    fn accepts_empty_input() {
        let mut v = Utf8Validator::new();
        v.feed(&[]).unwrap();
        v.feed(&[]).unwrap();
        v.finish().unwrap();
        validate_utf8(&[]).unwrap();
    }

    #[test]
    fn accepts_full_ascii_range() {
        let ascii: Vec<u8> = (0u16..=0x7F).map(|i| u8::try_from(i).unwrap()).collect();
        validate_utf8(&ascii).unwrap();

        let mut v = Utf8Validator::new();
        v.feed(&ascii).unwrap();
        v.finish().unwrap();
    }

    #[test]
    fn accepts_boundary_valid_sequences() {
        for seq in VALID_SEQUENCES {
            validate_utf8(seq).unwrap_or_else(|e| panic!("rejected {seq:02x?}: {e}"));
        }
    }

    #[test]
    fn rejects_boundary_invalid_sequences() {
        for seq in INVALID_SEQUENCES {
            assert!(
                validate_utf8(seq).is_err(),
                "accepted invalid UTF-8: {seq:02x?}"
            );
        }
    }

    #[test]
    fn rejects_lone_surrogates_cesu8_style() {
        for seq in [
            [0xEDu8, 0xA0, 0x80].as_slice(),
            &[0xED, 0xAD, 0xBF][..],
            &[0xED, 0xB0, 0x80][..],
            &[0xED, 0xBF, 0xBF][..],
        ] {
            assert!(matches!(validate_utf8(seq), Err(WsError::InvalidUtf8)));
        }
    }

    #[test]
    fn rejects_overlong_encodings() {
        for seq in [
            [0xC0u8, 0x80].as_slice(),
            &[0xC1, 0xBF][..],
            &[0xE0, 0x80, 0x80][..],
            &[0xE0, 0x9F, 0xBF][..],
            &[0xF0, 0x80, 0x80, 0x80][..],
            &[0xF0, 0x8F, 0xBF, 0xBF][..],
        ] {
            assert!(matches!(validate_utf8(seq), Err(WsError::InvalidUtf8)));
        }
    }

    #[test]
    fn rejects_truncated_sequences_at_finish() {
        for seq in [&[0xC3][..], &[0xE2, 0x82][..], &[0xF0, 0x9F, 0x99][..]] {
            let mut v = Utf8Validator::new();
            v.feed(seq).unwrap();
            assert!(matches!(v.finish(), Err(WsError::InvalidUtf8)));
        }
    }

    #[test]
    fn split_code_points_accepted_at_every_byte_boundary() {
        let bytes = SPLIT_SAMPLE.as_bytes();
        assert!(bytes.len() > 12);
        for split in 0..=bytes.len() {
            let mut v = Utf8Validator::new();
            v.feed(&bytes[..split])
                .unwrap_or_else(|e| panic!("first half at {split}: {e}"));
            v.feed(&bytes[split..])
                .unwrap_or_else(|e| panic!("second half at {split}: {e}"));
            v.finish()
                .unwrap_or_else(|e| panic!("finish at split {split}: {e}"));
        }
    }

    #[test]
    fn byte_by_byte_feeding_preserves_validity() {
        let mut v = Utf8Validator::new();
        for b in SPLIT_SAMPLE.as_bytes() {
            v.feed(&[*b]).unwrap();
        }
        v.finish().unwrap();
    }

    #[test]
    fn chunked_feeding_matches_one_shot_validation() {
        let bytes = SPLIT_SAMPLE.as_bytes();
        for chunk in [1usize, 2, 3, 4, 5, 7] {
            let mut v = Utf8Validator::new();
            for piece in bytes.chunks(chunk) {
                v.feed(piece)
                    .unwrap_or_else(|e| panic!("chunk {chunk}: {e}"));
            }
            v.finish().unwrap_or_else(|e| panic!("chunk {chunk}: {e}"));
        }
        validate_utf8(bytes).unwrap();
    }

    #[test]
    fn finish_fails_while_multi_byte_sequence_pending() {
        let mut v = Utf8Validator::new();
        v.feed(&[0xC3]).unwrap();
        assert!(v.finish().is_err());

        v.feed(&[0xA9]).unwrap();
        v.finish().unwrap();
    }

    #[test]
    fn reset_clears_pending_state() {
        let mut v = Utf8Validator::new();
        v.feed(&[0xC3]).unwrap();
        assert!(v.finish().is_err());
        v.reset();
        v.finish().unwrap();
        v.feed(b"fresh").unwrap();
        v.finish().unwrap();
    }

    #[test]
    fn rejects_non_continuation_byte_after_partial_sequence() {
        let mut v = Utf8Validator::new();
        v.feed(&[0xC3]).unwrap();
        assert!(matches!(v.feed(&[0x41]), Err(WsError::InvalidUtf8)));

        let mut v3 = Utf8Validator::new();
        v3.feed(&[0xE2, 0x82]).unwrap();
        assert!(matches!(v3.feed(&[0x20]), Err(WsError::InvalidUtf8)));
    }

    #[test]
    fn rejects_continuation_byte_without_pending_lead() {
        let mut v = Utf8Validator::new();
        assert!(matches!(v.feed(&[0x80]), Err(WsError::InvalidUtf8)));

        let mut v2 = Utf8Validator::new();
        v2.feed(&[0x41]).unwrap();
        assert!(matches!(v2.feed(&[0xBF]), Err(WsError::InvalidUtf8)));
        v2.finish().unwrap();
    }

    #[test]
    fn invalid_byte_fails_feed_even_after_valid_prefix() {
        let mut v = Utf8Validator::new();
        assert!(matches!(
            v.feed(&[0x41, 0xFF, 0x42]),
            Err(WsError::InvalidUtf8)
        ));
    }

    #[test]
    fn pending_bytes_survive_across_feed_calls_until_complete() {
        let mut v = Utf8Validator::new();
        v.feed(&[0xF0]).unwrap();
        v.feed(&[0x9F]).unwrap();
        v.feed(&[0x99]).unwrap();
        v.finish().unwrap_err();
        v.feed(&[0x82]).unwrap();
        v.finish().unwrap();
    }

    #[test]
    fn default_and_new_are_equivalent_empty_validators() {
        let default = Utf8Validator::default();
        let fresh = Utf8Validator::new();
        assert_eq!(format!("{default:?}"), format!("{fresh:?}"));
        default.finish().unwrap();
        fresh.finish().unwrap();
    }

    #[test]
    fn validator_clone_inherits_pending_state() {
        let mut v = Utf8Validator::new();
        v.feed(&[0xC3]).unwrap();
        let mut clone = v.clone();
        assert!(clone.finish().is_err());
        clone.feed(&[0xA9]).unwrap();
        clone.finish().unwrap();
        assert!(v.finish().is_err());
    }
}
