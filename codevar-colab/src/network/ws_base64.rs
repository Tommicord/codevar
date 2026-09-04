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

//! Minimal Base64 (RFC 4648) codec for WebSocket handshake material.
//!
//! Only the standard alphabet with `=` padding is supported. This is
//! sufficient for `Sec-WebSocket-Key` (16 random bytes → 24 chars) and
//! `Sec-WebSocket-Accept` (20-byte SHA-1 → 28 chars).

use crate::network::ws_error::{WsError, WsResult};

const ENCODE_TABLE: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes `input` as a Base64 string with standard padding.
#[must_use]
pub fn encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(encoded_len(input.len()));
    let mut i = 0;
    while i + 3 <= input.len() {
        let n = (u32::from(input[i]) << 16)
            | (u32::from(input[i + 1]) << 8)
            | u32::from(input[i + 2]);
        out.push(ENCODE_TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(ENCODE_TABLE[((n >> 12) & 0x3f) as usize] as char);
        out.push(ENCODE_TABLE[((n >> 6) & 0x3f) as usize] as char);
        out.push(ENCODE_TABLE[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = u32::from(input[i]) << 16;
        out.push(ENCODE_TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(ENCODE_TABLE[((n >> 12) & 0x3f) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = (u32::from(input[i]) << 16) | (u32::from(input[i + 1]) << 8);
        out.push(ENCODE_TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(ENCODE_TABLE[((n >> 12) & 0x3f) as usize] as char);
        out.push(ENCODE_TABLE[((n >> 6) & 0x3f) as usize] as char);
        out.push('=');
    }
    out
}

/// Decodes a standard Base64 string into bytes.
///
/// Accepts optional whitespace; rejects non-canonical padding and alphabet
/// characters outside the standard Base64 set.
pub fn decode(input: &str) -> WsResult<Vec<u8>> {
    let cleaned: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if cleaned.len() % 4 != 0 {
        return Err(WsError::decode("base64 length must be a multiple of 4"));
    }
    let mut out = Vec::with_capacity(cleaned.len() / 4 * 3);
    let mut i = 0;
    while i < cleaned.len() {
        let a = decode_char(cleaned[i])?;
        let b = decode_char(cleaned[i + 1])?;
        let (c, pad_c) = decode_char_or_pad(cleaned[i + 2])?;
        let (d, pad_d) = decode_char_or_pad(cleaned[i + 3])?;
        if pad_c && !pad_d {
            return Err(WsError::decode("invalid base64 padding"));
        }
        if pad_c && a == 0xff {
            return Err(WsError::decode("invalid base64"));
        }
        let n = (u32::from(a) << 18)
            | (u32::from(b) << 12)
            | (u32::from(c) << 6)
            | u32::from(d);
        out.push(((n >> 16) & 0xff) as u8);
        if !pad_c {
            out.push(((n >> 8) & 0xff) as u8);
        }
        if !pad_d {
            out.push((n & 0xff) as u8);
        }
        i += 4;
    }
    Ok(out)
}

fn encoded_len(n: usize) -> usize {
    n.div_ceil(3) * 4
}

fn decode_char(c: u8) -> WsResult<u8> {
    match c {
        b'A'..=b'Z' => Ok(c - b'A'),
        b'a'..=b'z' => Ok(c - b'a' + 26),
        b'0'..=b'9' => Ok(c - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(WsError::decode(format!(
            "invalid base64 character 0x{c:02x}"
        ))),
    }
}

fn decode_char_or_pad(c: u8) -> WsResult<(u8, bool)> {
    if c == b'=' {
        Ok((0, true))
    } else {
        Ok((decode_char(c)?, false))
    }
}
