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

//! Incremental UTF-8 validation for WebSocket text payloads.
//!
//! RFC 6455 §5.6 permits individual text *frames* to contain partial UTF-8
//! sequences, but the reassembled *message* MUST be valid UTF-8. Close
//! frame reasons (RFC 6455 §5.5.1) must also be valid UTF-8. Invalid UTF-8
//! fails the connection with status code 1007 (RFC 6455 §8.1).

use crate::network::ws_error::{WsError, WsResult};

/// Streaming UTF-8 validator that accepts incomplete trailing sequences
/// across fragment boundaries.
///
/// # Example
///
/// ```
/// use codevar_colab::network::ws_utf8::Utf8Validator;
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
pub fn validate_utf8(bytes: &[u8]) -> WsResult<()> {
    let mut v = Utf8Validator::new();
    v.feed(bytes)?;
    v.finish()
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
