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

//! WebSocket frame encode / decode (RFC 6455 §5).
//!
//! # Wire format
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-------+-+-------------+-------------------------------+
//! |F|R|R|R| opcode|M| Payload len |    Extended payload length    |
//! |I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
//! |N|V|V|V|       |S|             |   (if payload len==126/127)   |
//! | |1|2|3|       |K|             |                               |
//! +-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
//! |     Extended payload length continued, if payload len == 127  |
//! + - - - - - - - - - - - - - - - +-------------------------------+
//! |                               |Masking-key, if MASK set to 1  |
//! +-------------------------------+-------------------------------+
//! | Masking-key (continued)       |          Payload Data         |
//! +-------------------------------- - - - - - - - - - - - - - - - +
//! ```
//!
//! Length encoding uses the shortest form (RFC 6455 §5.2): values 0–125
//! are inline, 126 introduces a 16-bit length, and 127 a 64-bit length
//! whose most-significant bit MUST be 0.

use crate::network::ws_error::{WsError, WsResult};
use crate::network::ws_ids::{MAX_CONTROL_PAYLOAD, Role, WsCloseCode, WsOpcode};

/// Parsed frame header (without payload bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WsFrameHeader {
    /// FIN bit — final fragment of the message.
    pub fin: bool,
    /// RSV1 bit — MUST be 0 unless an extension is negotiated.
    pub rsv1: bool,
    /// RSV2 bit — MUST be 0 unless an extension is negotiated.
    pub rsv2: bool,
    /// RSV3 bit — MUST be 0 unless an extension is negotiated.
    pub rsv3: bool,
    /// Frame opcode.
    pub opcode: WsOpcode,
    /// Whether a masking key follows the length field.
    pub masked: bool,
    /// Payload length in bytes (Extension data + Application data).
    pub payload_len: u64,
    /// Masking key when `masked` is true.
    pub mask_key: Option<[u8; 4]>,
    /// Total header size in bytes (2 + extended length + optional mask).
    pub header_len: usize,
}

impl WsFrameHeader {
    /// Attempts to parse a frame header from the start of `buf`.
    ///
    /// Returns `Ok(None)` when `buf` does not yet contain a complete header.
    pub fn parse(buf: &[u8], max_frame_size: usize) -> WsResult<Option<Self>> {
        if buf.len() < 2 {
            return Ok(None);
        }
        let b0 = buf[0];
        let b1 = buf[1];
        let fin = (b0 & 0x80) != 0;
        let rsv1 = (b0 & 0x40) != 0;
        let rsv2 = (b0 & 0x20) != 0;
        let rsv3 = (b0 & 0x10) != 0;
        let opcode = WsOpcode::from_u8(b0 & 0x0f)?;
        let masked = (b1 & 0x80) != 0;
        let len7 = b1 & 0x7f;

        let (payload_len, len_bytes) = match len7 {
            0..=125 => (u64::from(len7), 0usize),
            126 => {
                if buf.len() < 4 {
                    return Ok(None);
                }
                let len = u64::from(u16::from_be_bytes([buf[2], buf[3]]));
                // Minimal-length encoding required (RFC 6455 §5.2).
                if len <= 125 {
                    return Err(WsError::protocol(
                        WsCloseCode::ProtocolError,
                        "non-minimal 16-bit payload length",
                    ));
                }
                (len, 2)
            }
            127 => {
                if buf.len() < 10 {
                    return Ok(None);
                }
                let len = u64::from_be_bytes([
                    buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8], buf[9],
                ]);
                // MSB must be 0 (RFC 6455 §5.2).
                if len & 0x8000_0000_0000_0000 != 0 {
                    return Err(WsError::protocol(
                        WsCloseCode::ProtocolError,
                        "64-bit payload length has MSB set",
                    ));
                }
                if len <= 0xffff {
                    return Err(WsError::protocol(
                        WsCloseCode::ProtocolError,
                        "non-minimal 64-bit payload length",
                    ));
                }
                (len, 8)
            }
            _ => unreachable!("len7 is 7 bits"),
        };

        if payload_len > max_frame_size as u64 {
            return Err(WsError::MessageTooBig {
                size: payload_len as usize,
                limit: max_frame_size,
            });
        }

        if opcode.is_control() {
            if !fin {
                return Err(WsError::protocol(
                    WsCloseCode::ProtocolError,
                    "control frames must not be fragmented",
                ));
            }
            if payload_len > MAX_CONTROL_PAYLOAD as u64 {
                return Err(WsError::protocol(
                    WsCloseCode::ProtocolError,
                    "control frame payload exceeds 125 bytes",
                ));
            }
        }

        let mask_bytes = if masked { 4usize } else { 0 };
        let header_len = 2 + len_bytes + mask_bytes;
        if buf.len() < header_len {
            return Ok(None);
        }

        let mask_key = if masked {
            let start = 2 + len_bytes;
            Some([buf[start], buf[start + 1], buf[start + 2], buf[start + 3]])
        } else {
            None
        };

        Ok(Some(Self {
            fin,
            rsv1,
            rsv2,
            rsv3,
            opcode,
            masked,
            payload_len,
            mask_key,
            header_len,
        }))
    }

    /// Total size of this frame on the wire (header + payload).
    #[must_use]
    pub fn frame_len(&self) -> Option<usize> {
        (self.payload_len as usize).checked_add(self.header_len)
    }

    /// Rejects non-zero RSV bits when no extensions are negotiated.
    pub fn check_rsv_clear(&self) -> WsResult<()> {
        if self.rsv1 || self.rsv2 || self.rsv3 {
            return Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                "RSV bits must be 0 without negotiated extensions",
            ));
        }
        Ok(())
    }

    /// Enforces the masking rules for `role` (RFC 6455 §5.1).
    pub fn check_masking(&self, role: Role) -> WsResult<()> {
        let expected = role.expects_inbound_masked();
        if self.masked != expected {
            let msg = if expected {
                "server received an unmasked frame"
            } else {
                "client received a masked frame"
            };
            return Err(WsError::protocol(WsCloseCode::ProtocolError, msg));
        }
        Ok(())
    }
}

/// A fully decoded (and unmasked) WebSocket frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WsFrame {
    /// Frame header metadata.
    pub header: WsFrameHeader,
    /// Unmasked payload bytes.
    pub payload: Vec<u8>,
}

impl WsFrame {
    /// Creates a data or control frame ready for encoding.
    #[must_use]
    pub fn new(fin: bool, opcode: WsOpcode, payload: Vec<u8>) -> Self {
        Self {
            header: WsFrameHeader {
                fin,
                rsv1: false,
                rsv2: false,
                rsv3: false,
                opcode,
                masked: false,
                payload_len: payload.len() as u64,
                mask_key: None,
                header_len: 0,
            },
            payload,
        }
    }

    /// Convenience constructor for a final text frame.
    #[must_use]
    pub fn text(payload: impl Into<Vec<u8>>) -> Self {
        Self::new(true, WsOpcode::Text, payload.into())
    }

    /// Convenience constructor for a final binary frame.
    #[must_use]
    pub fn binary(payload: impl Into<Vec<u8>>) -> Self {
        Self::new(true, WsOpcode::Binary, payload.into())
    }

    /// Convenience constructor for a Close frame with optional code/reason.
    pub fn close(code: Option<WsCloseCode>, reason: &str) -> WsResult<Self> {
        let payload = encode_close_payload(code, reason)?;
        Ok(Self::new(true, WsOpcode::Close, payload))
    }

    /// Convenience constructor for a Ping frame.
    pub fn ping(payload: impl Into<Vec<u8>>) -> WsResult<Self> {
        let payload = payload.into();
        if payload.len() > MAX_CONTROL_PAYLOAD {
            return Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                "ping payload exceeds 125 bytes",
            ));
        }
        Ok(Self::new(true, WsOpcode::Ping, payload))
    }

    /// Convenience constructor for a Pong frame.
    pub fn pong(payload: impl Into<Vec<u8>>) -> WsResult<Self> {
        let payload = payload.into();
        if payload.len() > MAX_CONTROL_PAYLOAD {
            return Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                "pong payload exceeds 125 bytes",
            ));
        }
        Ok(Self::new(true, WsOpcode::Pong, payload))
    }

    /// Encodes this frame onto `out`, applying masking when `mask_key` is set.
    ///
    /// Clients MUST supply a fresh random masking key for every frame
    /// (RFC 6455 §5.3). Servers MUST pass `None`.
    pub fn encode(&self, out: &mut Vec<u8>, mask_key: Option<[u8; 4]>) -> WsResult<()> {
        let len = self.payload.len();
        let mut b0 = self.header.opcode.as_u8();
        if self.header.fin {
            b0 |= 0x80;
        }
        if self.header.rsv1 {
            b0 |= 0x40;
        }
        if self.header.rsv2 {
            b0 |= 0x20;
        }
        if self.header.rsv3 {
            b0 |= 0x10;
        }
        out.push(b0);

        let mask_bit: u8 = if mask_key.is_some() { 0x80 } else { 0 };
        if len <= 125 {
            out.push(mask_bit | (len as u8));
        } else if len <= 0xffff {
            out.push(mask_bit | 126);
            out.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            out.push(mask_bit | 127);
            out.extend_from_slice(&(len as u64).to_be_bytes());
        }

        if let Some(key) = mask_key {
            out.extend_from_slice(&key);
            let start = out.len();
            out.extend_from_slice(&self.payload);
            apply_mask(&mut out[start..], key);
        } else {
            out.extend_from_slice(&self.payload);
        }
        Ok(())
    }
}

/// Applies the XOR masking transform in place (RFC 6455 §5.3).
///
/// The same function both masks and unmasks.
#[inline]
pub fn apply_mask(data: &mut [u8], key: [u8; 4]) {
    for (i, byte) in data.iter_mut().enumerate() {
        *byte ^= key[i % 4];
    }
}

/// Generates a cryptographically random 32-bit masking key.
pub fn random_mask_key() -> WsResult<[u8; 4]> {
    let mut key = [0u8; 4];
    getrandom::getrandom(&mut key).map_err(|_| WsError::RandomFailed)?;
    Ok(key)
}

/// Encodes an optional Close status code and UTF-8 reason into a payload.
pub fn encode_close_payload(
    code: Option<WsCloseCode>,
    reason: &str,
) -> WsResult<Vec<u8>> {
    match code {
        None => {
            if reason.is_empty() {
                Ok(Vec::new())
            } else {
                Err(WsError::protocol(
                    WsCloseCode::ProtocolError,
                    "Close reason requires a status code",
                ))
            }
        }
        Some(c) => {
            if !c.is_sendable() {
                return Err(WsError::protocol(
                    WsCloseCode::ProtocolError,
                    format!("close code {} must not be sent on the wire", c.as_u16()),
                ));
            }
            let reason_bytes = reason.as_bytes();
            if reason_bytes.len() + 2 > MAX_CONTROL_PAYLOAD {
                return Err(WsError::protocol(
                    WsCloseCode::ProtocolError,
                    "Close reason exceeds control frame limit",
                ));
            }
            let mut payload = Vec::with_capacity(2 + reason_bytes.len());
            payload.extend_from_slice(&c.as_u16().to_be_bytes());
            payload.extend_from_slice(reason_bytes);
            Ok(payload)
        }
    }
}

/// Decodes a Close frame payload into `(code, reason)`.
///
/// An empty payload yields `(NoStatusReceived, "")`. A single-byte payload
/// is a protocol error. The reason string is validated as UTF-8.
pub fn decode_close_payload(payload: &[u8]) -> WsResult<(WsCloseCode, String)> {
    if payload.is_empty() {
        return Ok((WsCloseCode::NoStatusReceived, String::new()));
    }
    if payload.len() == 1 {
        return Err(WsError::protocol(
            WsCloseCode::ProtocolError,
            "Close payload must be empty or at least 2 bytes",
        ));
    }
    let code = WsCloseCode::from_u16(u16::from_be_bytes([payload[0], payload[1]]))?;
    let reason_bytes = &payload[2..];
    crate::network::ws_utf8::validate_utf8(reason_bytes)?;
    let reason =
        String::from_utf8(reason_bytes.to_vec()).map_err(|_| WsError::InvalidUtf8)?;
    Ok((code, reason))
}

/// Tries to parse one complete frame from the front of `buf`.
///
/// On success returns `(frame, bytes_consumed)`. Returns `Ok(None)` when
/// more bytes are needed. The payload is unmasked before return.
pub fn try_parse_frame(
    buf: &[u8],
    role: Role,
    max_frame_size: usize,
) -> WsResult<Option<(WsFrame, usize)>> {
    let Some(header) = WsFrameHeader::parse(buf, max_frame_size)? else {
        return Ok(None);
    };
    header.check_rsv_clear()?;
    header.check_masking(role)?;
    let Some(total) = header.frame_len() else {
        return Err(WsError::MessageTooBig {
            size: usize::MAX,
            limit: max_frame_size,
        });
    };
    if buf.len() < total {
        return Ok(None);
    }
    let mut payload = buf[header.header_len..total].to_vec();
    if let Some(key) = header.mask_key {
        apply_mask(&mut payload, key);
    }
    let consumed = total;
    Ok(Some((WsFrame { header, payload }, consumed)))
}
