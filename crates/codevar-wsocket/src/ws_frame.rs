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

use crate::ws_error::{WsError, WsResult};
use crate::ws_ids::{MAX_CONTROL_PAYLOAD, Role, WsCloseCode, WsOpcode};

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
            // `len7` is `b1 & 0x7f`, so 127 is the only remaining value.
            _ => {
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
    crate::ws_utf8::validate_utf8(reason_bytes)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ws_ids::DEFAULT_MAX_FRAME_SIZE;

    fn parse_header(buf: &[u8]) -> WsResult<Option<WsFrameHeader>> {
        WsFrameHeader::parse(buf, DEFAULT_MAX_FRAME_SIZE)
    }

    fn parse_client(buf: &[u8]) -> WsResult<Option<(WsFrame, usize)>> {
        try_parse_frame(buf, Role::Client, DEFAULT_MAX_FRAME_SIZE)
    }

    fn parse_server(buf: &[u8]) -> WsResult<Option<(WsFrame, usize)>> {
        try_parse_frame(buf, Role::Server, DEFAULT_MAX_FRAME_SIZE)
    }

    fn encode_unmasked(frame: &WsFrame) -> Vec<u8> {
        let mut out = Vec::new();
        frame.encode(&mut out, None).expect("encode");
        out
    }

    fn make_header(payload_len: u64, header_len: usize) -> WsFrameHeader {
        WsFrameHeader {
            fin: true,
            rsv1: false,
            rsv2: false,
            rsv3: false,
            opcode: WsOpcode::Binary,
            masked: false,
            payload_len,
            mask_key: None,
            header_len,
        }
    }

    #[test]
    fn header_parse_waits_for_incomplete_base_header() {
        assert!(parse_header(&[]).unwrap().is_none());
        assert!(parse_header(&[0x81]).unwrap().is_none());
    }

    #[test]
    fn header_parse_extracts_flags_opcode_and_length() {
        let mut buf = vec![0xC1, 0x80 | 10, 1, 2, 3, 4];
        buf.extend_from_slice(&[0u8; 10]);
        let header = parse_header(&buf).unwrap().unwrap();
        assert!(header.fin);
        assert!(header.rsv1);
        assert!(!header.rsv2);
        assert!(!header.rsv3);
        assert_eq!(header.opcode, WsOpcode::Text);
        assert!(header.masked);
        assert_eq!(header.payload_len, 10);
        assert_eq!(header.mask_key, Some([1, 2, 3, 4]));
        assert_eq!(header.header_len, 6);
    }

    #[test]
    fn header_parse_accepts_all_defined_opcodes() {
        for v in [0u8, 0x1, 0x2, 0x8, 0x9, 0xA] {
            let buf = [0x80 | v, 0x00];
            let header = parse_header(&buf).unwrap().unwrap();
            assert_eq!(header.opcode.as_u8(), v, "opcode {v:#x}");
        }
    }

    #[test]
    fn header_parse_rejects_reserved_opcodes() {
        for v in [3u8, 4, 5, 6, 7, 0xB, 0xC, 0xD, 0xE, 0xF] {
            let buf = [0x80 | v, 0x00];
            let err = parse_header(&buf).unwrap_err();
            assert!(
                matches!(
                    err,
                    WsError::Protocol {
                        close_code: WsCloseCode::ProtocolError,
                        ..
                    }
                ),
                "opcode {v:#x}: {err:?}"
            );
        }
    }

    #[test]
    fn header_parse_waits_for_truncated_masked_header() {
        let full = [0x81, 0x85, 9, 8, 7, 6];
        for n in 0..full.len() {
            assert!(
                parse_header(&full[..n]).unwrap().is_none(),
                "truncated at {n}"
            );
        }
        let header = parse_header(&full).unwrap().unwrap();
        assert_eq!(header.mask_key, Some([9, 8, 7, 6]));
        assert_eq!(header.header_len, 6);
    }

    #[test]
    fn header_parse_enforces_minimal_16bit_length_encoding() {
        let mut full = vec![0x81, 126];
        full.extend_from_slice(&126u16.to_be_bytes());
        for n in 2..full.len() {
            assert!(
                parse_header(&full[..n]).unwrap().is_none(),
                "truncated at {n}"
            );
        }
        let header = parse_header(&full).unwrap().unwrap();
        assert_eq!(header.payload_len, 126);
        assert_eq!(header.header_len, 4);

        let non_minimal = [0x81, 126, 0x00, 0x7D];
        assert!(matches!(
            parse_header(&non_minimal).unwrap_err(),
            WsError::Protocol { .. }
        ));
        let zero_via_16bit = [0x81, 126, 0x00, 0x00];
        assert!(matches!(
            parse_header(&zero_via_16bit).unwrap_err(),
            WsError::Protocol { .. }
        ));
    }

    #[test]
    fn header_parse_enforces_64bit_length_rules() {
        let mut full = vec![0x81, 127];
        full.extend_from_slice(&0x1_0000u64.to_be_bytes());
        for n in 2..full.len() {
            assert!(
                parse_header(&full[..n]).unwrap().is_none(),
                "truncated at {n}"
            );
        }
        let header = parse_header(&full).unwrap().unwrap();
        assert_eq!(header.payload_len, 0x1_0000);
        assert_eq!(header.header_len, 10);

        let mut non_minimal = vec![0x81, 127];
        non_minimal.extend_from_slice(&0xFFFFu64.to_be_bytes());
        assert!(matches!(
            parse_header(&non_minimal).unwrap_err(),
            WsError::Protocol { .. }
        ));

        let mut msb_set = vec![0x81, 127];
        msb_set.extend_from_slice(&0x8000_0000_0000_0001u64.to_be_bytes());
        assert!(matches!(
            parse_header(&msb_set).unwrap_err(),
            WsError::Protocol { .. }
        ));
    }

    #[test]
    fn header_parse_rejects_oversize_frames_before_payload_arrives() {
        let cases = [
            u64::try_from(DEFAULT_MAX_FRAME_SIZE).unwrap() + 1,
            u64::from(u32::MAX),
            0x7FFF_FFFF_FFFF_FFFF,
        ];
        for declared in cases {
            let mut buf = vec![0x82, 127];
            buf.extend_from_slice(&declared.to_be_bytes());
            let err = parse_header(&buf).unwrap_err();
            match err {
                WsError::MessageTooBig { size, limit } => {
                    assert!(size > limit);
                    assert_eq!(limit, DEFAULT_MAX_FRAME_SIZE);
                }
                other => panic!("expected MessageTooBig for {declared}, got {other:?}"),
            }
        }
    }

    #[test]
    fn header_parse_rejects_fragmented_control_frames() {
        for b0 in [0x08u8, 0x09, 0x0A] {
            let buf = [b0, 0x00];
            let err = parse_header(&buf).unwrap_err();
            assert!(
                matches!(
                    err,
                    WsError::Protocol {
                        close_code: WsCloseCode::ProtocolError,
                        ..
                    }
                ),
                "b0 {b0:#x}: {err:?}"
            );
        }
    }

    #[test]
    fn header_parse_rejects_control_frames_over_125_bytes() {
        let ping_126 = [0x89, 126, 0x00, 0x7E];
        assert!(matches!(
            parse_header(&ping_126).unwrap_err(),
            WsError::Protocol { .. }
        ));

        let mut pong_64k = vec![0x8A, 127];
        pong_64k.extend_from_slice(&0x1_0000u64.to_be_bytes());
        assert!(matches!(
            parse_header(&pong_64k).unwrap_err(),
            WsError::Protocol { .. }
        ));

        let max_control = [0x89, 125];
        assert!(parse_header(&max_control).unwrap().is_some());
    }

    #[test]
    fn header_frame_len_reports_overflow_as_none() {
        let max_len = make_header(u64::try_from(usize::MAX).unwrap(), 2);
        assert!(max_len.frame_len().is_none());

        let boundary = make_header(u64::try_from(usize::MAX - 2).unwrap(), 2);
        assert_eq!(boundary.frame_len(), Some(usize::MAX));

        let small = make_header(5, 6);
        assert_eq!(small.frame_len(), Some(11));
    }

    #[test]
    fn check_rsv_clear_rejects_each_reserved_bit() {
        let base = make_header(0, 2);
        assert!(base.check_rsv_clear().is_ok());

        let mut rsv1 = make_header(0, 2);
        rsv1.rsv1 = true;
        assert!(rsv1.check_rsv_clear().is_err());

        let mut rsv2 = make_header(0, 2);
        rsv2.rsv2 = true;
        assert!(rsv2.check_rsv_clear().is_err());

        let mut rsv3 = make_header(0, 2);
        rsv3.rsv3 = true;
        assert!(rsv3.check_rsv_clear().is_err());
    }

    #[test]
    fn check_masking_enforces_role_expectations() {
        let mut unmasked = make_header(0, 2);
        unmasked.masked = false;
        assert!(unmasked.check_masking(Role::Client).is_ok());
        assert!(matches!(
            unmasked.check_masking(Role::Server).unwrap_err(),
            WsError::Protocol { .. }
        ));

        let mut masked = make_header(0, 6);
        masked.masked = true;
        masked.mask_key = Some([1, 2, 3, 4]);
        assert!(masked.check_masking(Role::Server).is_ok());
        assert!(matches!(
            masked.check_masking(Role::Client).unwrap_err(),
            WsError::Protocol { .. }
        ));
    }

    #[test]
    fn try_parse_frame_rejects_encoded_reserved_bits() {
        for flag in [0x40u8, 0x20, 0x10] {
            let mut frame = WsFrame::text("hi");
            match flag {
                0x40 => frame.header.rsv1 = true,
                0x20 => frame.header.rsv2 = true,
                _ => frame.header.rsv3 = true,
            }
            let encoded = encode_unmasked(&frame);
            let err = parse_client(&encoded).unwrap_err();
            assert!(
                matches!(
                    err,
                    WsError::Protocol {
                        close_code: WsCloseCode::ProtocolError,
                        ..
                    }
                ),
                "flag {flag:#x}: {err:?}"
            );
        }
    }

    #[test]
    fn apply_mask_is_involution_across_odd_lengths() {
        let mut data = b"the quick brown fox".to_vec();
        let original = data.clone();
        apply_mask(&mut data, [1, 2, 3, 4]);
        assert_ne!(data, original);
        apply_mask(&mut data, [1, 2, 3, 4]);
        assert_eq!(data, original);

        let mut zeroed = vec![10u8, 20, 30, 40, 50];
        apply_mask(&mut zeroed, [0, 0, 0, 0]);
        assert_eq!(zeroed, vec![10, 20, 30, 40, 50]);

        let mut empty: Vec<u8> = Vec::new();
        apply_mask(&mut empty, [0xAA, 0x55, 0xF0, 0x0F]);
        assert!(empty.is_empty());

        for len in [1usize, 3, 5, 7, 8, 9] {
            let mut bytes: Vec<u8> = (0..len)
                .map(|i| u8::try_from(i * 17 % 256).unwrap())
                .collect();
            let orig = bytes.clone();
            apply_mask(&mut bytes, [0xAA, 0x55, 0xF0, 0x0F]);
            assert_ne!(bytes, orig, "len {len}");
            apply_mask(&mut bytes, [0xAA, 0x55, 0xF0, 0x0F]);
            assert_eq!(bytes, orig, "len {len}");
        }
    }

    #[test]
    fn random_mask_key_yields_distinct_usable_keys() {
        let mut keys = Vec::with_capacity(64);
        for _ in 0..64 {
            keys.push(random_mask_key().unwrap());
        }
        assert!(keys.iter().any(|k| *k != keys[0]));

        let mut data = vec![7u8; 33];
        let original = data.clone();
        apply_mask(&mut data, keys[0]);
        assert_ne!(data, original);
        apply_mask(&mut data, keys[0]);
        assert_eq!(data, original);
    }

    #[test]
    fn try_parse_frame_round_trips_every_opcode() {
        let cases: [(WsOpcode, Vec<u8>); 6] = [
            (WsOpcode::Continuation, b"part".to_vec()),
            (WsOpcode::Text, b"hello".to_vec()),
            (WsOpcode::Binary, vec![0, 1, 2, 255]),
            (WsOpcode::Close, Vec::new()),
            (WsOpcode::Ping, vec![0xAB; 125]),
            (WsOpcode::Pong, vec![0xCD; 1]),
        ];
        for (opcode, payload) in cases {
            let frame = WsFrame::new(true, opcode, payload.clone());
            let encoded = encode_unmasked(&frame);
            let (parsed, consumed) = parse_client(&encoded).unwrap().unwrap();
            assert_eq!(consumed, encoded.len(), "opcode {opcode}");
            assert_eq!(parsed.header.opcode, opcode);
            assert!(parsed.header.fin);
            assert_eq!(parsed.payload, payload, "opcode {opcode}");
        }
    }

    #[test]
    fn try_parse_frame_round_trips_length_boundaries() {
        for len in [0usize, 1, 125, 126, 65535, 65536] {
            let payload: Vec<u8> =
                (0..len).map(|i| u8::try_from(i % 256).unwrap()).collect();
            let frame = WsFrame::new(true, WsOpcode::Binary, payload.clone());
            let encoded = encode_unmasked(&frame);

            let marker = match len {
                0..=125 => u8::try_from(len).unwrap(),
                126..=65535 => 126,
                _ => 127,
            };
            assert_eq!(encoded[1] & 0x7F, marker, "len {len}");

            let expected_header = match len {
                0..=125 => 2usize,
                126..=65535 => 4,
                _ => 10,
            };
            let (parsed, consumed) = parse_client(&encoded).unwrap().unwrap();
            assert_eq!(consumed, encoded.len(), "len {len}");
            assert_eq!(parsed.header.header_len, expected_header, "len {len}");
            assert_eq!(
                parsed.header.payload_len,
                u64::try_from(len).unwrap(),
                "len {len}"
            );
            assert_eq!(parsed.payload, payload, "len {len}");
        }
    }

    #[test]
    fn try_parse_frame_round_trips_masked_payload_as_server() {
        let payload = b"masked payload bytes".to_vec();
        let frame = WsFrame::new(true, WsOpcode::Binary, payload.clone());
        let key = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut out = Vec::new();
        frame.encode(&mut out, Some(key)).unwrap();

        let wire_payload_start = out.len() - payload.len();
        assert_ne!(&out[wire_payload_start..], &payload[..]);

        for n in 0..out.len() {
            assert!(
                parse_server(&out[..n]).unwrap().is_none(),
                "masked frame truncated at {n}"
            );
        }

        let (parsed, consumed) = parse_server(&out).unwrap().unwrap();
        assert_eq!(consumed, out.len());
        assert_eq!(parsed.payload, payload);
        assert!(parsed.header.masked);
        assert_eq!(parsed.header.mask_key, Some(key));
    }

    #[test]
    fn try_parse_frame_rejects_masking_violations() {
        let frame = WsFrame::text("payload");
        let unmasked = encode_unmasked(&frame);
        let err = parse_server(&unmasked).unwrap_err();
        let msg = match err {
            WsError::Protocol { message, .. } => message,
            other => format!("unexpected variant: {other:?}"),
        };
        assert!(msg.contains("unmasked"), "message: {msg}");

        let mut masked_out = Vec::new();
        frame.encode(&mut masked_out, Some([1, 2, 3, 4])).unwrap();
        let err = parse_client(&masked_out).unwrap_err();
        let msg = match err {
            WsError::Protocol { message, .. } => message,
            other => format!("unexpected variant: {other:?}"),
        };
        assert!(msg.contains("masked frame"), "message: {msg}");
    }

    #[test]
    fn try_parse_frame_waits_until_full_payload_arrives() {
        let frame = WsFrame::text("0123456789");
        let encoded = encode_unmasked(&frame);
        let total = encoded.len();
        for n in 0..total {
            assert!(
                parse_client(&encoded[..n]).unwrap().is_none(),
                "unmasked frame truncated at {n}"
            );
        }
        let (parsed, consumed) = parse_client(&encoded).unwrap().unwrap();
        assert_eq!(consumed, total);
        assert_eq!(parsed.payload, frame.payload);
    }

    #[test]
    fn try_parse_frame_consumes_exactly_one_frame() {
        let first = WsFrame::text("first");
        let second = WsFrame::binary(vec![1, 2, 3]);
        let mut buf = encode_unmasked(&first);
        let first_len = buf.len();
        buf.extend_from_slice(&encode_unmasked(&second));

        let (parsed, consumed) = parse_client(&buf).unwrap().unwrap();
        assert_eq!(consumed, first_len);
        assert_eq!(parsed.payload, first.payload);

        let (parsed2, consumed2) = parse_client(&buf[consumed..]).unwrap().unwrap();
        assert_eq!(consumed2, buf.len() - first_len);
        assert_eq!(parsed2.payload, second.payload);
        assert_eq!(parsed2.header.opcode, WsOpcode::Binary);
    }

    #[test]
    fn constructors_build_expected_frames() {
        let text = WsFrame::text("hi");
        assert!(text.header.fin);
        assert_eq!(text.header.opcode, WsOpcode::Text);
        assert_eq!(text.payload, b"hi");
        assert_eq!(text.header.payload_len, 2);

        let binary = WsFrame::binary(vec![9u8]);
        assert_eq!(binary.header.opcode, WsOpcode::Binary);

        let close = WsFrame::close(Some(WsCloseCode::Normal), "bye").unwrap();
        assert_eq!(close.header.opcode, WsOpcode::Close);
        assert_eq!(close.payload, [0x03, 0xE8, b'b', b'y', b'e']);

        let ping = WsFrame::ping(b"ping").unwrap();
        assert_eq!(ping.header.opcode, WsOpcode::Ping);
        assert!(ping.header.fin);

        let pong = WsFrame::pong(Vec::new()).unwrap();
        assert_eq!(pong.header.opcode, WsOpcode::Pong);
        assert!(pong.payload.is_empty());

        let partial = WsFrame::new(false, WsOpcode::Text, b"a".to_vec());
        assert!(!partial.header.fin);
    }

    #[test]
    fn ping_pong_and_close_constructors_enforce_limits() {
        assert!(WsFrame::ping(vec![0u8; 125]).is_ok());
        assert!(matches!(
            WsFrame::ping(vec![0u8; 126]).unwrap_err(),
            WsError::Protocol { .. }
        ));
        assert!(WsFrame::pong(vec![0u8; 125]).is_ok());
        assert!(matches!(
            WsFrame::pong(vec![0u8; 126]).unwrap_err(),
            WsError::Protocol { .. }
        ));

        assert!(WsFrame::close(None, "").is_ok());
        assert!(WsFrame::close(None, "reason needs code").is_err());
        assert!(WsFrame::close(Some(WsCloseCode::Normal), "bye").is_ok());
        for reserved in [
            WsCloseCode::NoStatusReceived,
            WsCloseCode::Abnormal,
            WsCloseCode::TlsHandshake,
        ] {
            let err = WsFrame::close(Some(reserved), "").unwrap_err();
            assert!(
                matches!(err, WsError::Protocol { .. }),
                "code {reserved}: {err:?}"
            );
        }
    }

    #[test]
    fn close_payload_encode_decode_round_trips() {
        let payload = encode_close_payload(Some(WsCloseCode::Normal), "bye").unwrap();
        assert_eq!(payload, [0x03, 0xE8, b'b', b'y', b'e']);
        let (code, reason) = decode_close_payload(&payload).unwrap();
        assert_eq!(code, WsCloseCode::Normal);
        assert_eq!(reason, "bye");

        let empty = encode_close_payload(None, "").unwrap();
        assert!(empty.is_empty());
        let (code, reason) = decode_close_payload(&empty).unwrap();
        assert_eq!(code, WsCloseCode::NoStatusReceived);
        assert!(reason.is_empty());

        let app = encode_close_payload(Some(WsCloseCode::Other(3000)), "app").unwrap();
        assert_eq!(app, [0x0B, 0xB8, b'a', b'p', b'p']);
        let (code, reason) = decode_close_payload(&app).unwrap();
        assert_eq!(code, WsCloseCode::Other(3000));
        assert_eq!(reason, "app");

        let max_reason = "r".repeat(123);
        let full =
            encode_close_payload(Some(WsCloseCode::GoingAway), &max_reason).unwrap();
        assert_eq!(full.len(), 125);
        let (code, reason) = decode_close_payload(&full).unwrap();
        assert_eq!(code, WsCloseCode::GoingAway);
        assert_eq!(reason, max_reason);
    }

    #[test]
    fn encode_close_payload_rejects_invalid_inputs() {
        assert!(encode_close_payload(None, "reason without code").is_err());
        for code in [
            WsCloseCode::NoStatusReceived,
            WsCloseCode::Abnormal,
            WsCloseCode::TlsHandshake,
        ] {
            let err = encode_close_payload(Some(code), "").unwrap_err();
            assert!(
                matches!(
                    err,
                    WsError::Protocol {
                        close_code: WsCloseCode::ProtocolError,
                        ..
                    }
                ),
                "code {code}: {err:?}"
            );
        }
        let too_long = "r".repeat(124);
        assert!(encode_close_payload(Some(WsCloseCode::Normal), &too_long).is_err());
    }

    #[test]
    fn decode_close_payload_rejects_truncated_and_reserved_codes() {
        assert!(decode_close_payload(&[0x00]).is_err());
        for bytes in [
            [0x00u8, 0x00].as_slice(),
            &[0x03, 0xE7][..],
            &[0x03, 0xED][..],
            &[0x03, 0xEE][..],
            &[0x03, 0xF7][..],
            &[0xFF, 0xFF][..],
        ] {
            assert!(
                decode_close_payload(bytes).is_err(),
                "payload {:02x?}",
                bytes
            );
        }

        let (code, reason) = decode_close_payload(&[0x03, 0xE8]).unwrap();
        assert_eq!(code, WsCloseCode::Normal);
        assert!(reason.is_empty());

        let mut utf8_reason = vec![0x03, 0xE8];
        utf8_reason.extend_from_slice("héllo".as_bytes());
        let (code, reason) = decode_close_payload(&utf8_reason).unwrap();
        assert_eq!(code, WsCloseCode::Normal);
        assert_eq!(reason, "héllo");
    }

    #[test]
    fn decode_close_payload_rejects_invalid_utf8_reason() {
        assert!(matches!(
            decode_close_payload(&[0x03, 0xE8, 0xFF]),
            Err(WsError::InvalidUtf8)
        ));
        assert!(matches!(
            decode_close_payload(&[0x03, 0xE8, 0xC3]),
            Err(WsError::InvalidUtf8)
        ));
        assert!(matches!(
            decode_close_payload(&[0x03, 0xE8, 0xC0, 0x80]),
            Err(WsError::InvalidUtf8)
        ));
    }

    #[test]
    fn one_mib_payload_round_trips_masked_and_unmasked() {
        let len = 1024 * 1024;
        let payload: Vec<u8> = (0..len).map(|i| u8::try_from(i % 251).unwrap()).collect();
        let frame = WsFrame::new(true, WsOpcode::Binary, payload.clone());

        let unmasked = encode_unmasked(&frame);
        let (parsed, consumed) = parse_client(&unmasked).unwrap().unwrap();
        assert_eq!(consumed, unmasked.len());
        assert_eq!(parsed.payload, payload);

        let mut masked_out = Vec::new();
        frame.encode(&mut masked_out, Some([9, 8, 7, 6])).unwrap();
        let (parsed, consumed) = parse_server(&masked_out).unwrap().unwrap();
        assert_eq!(consumed, masked_out.len());
        assert_eq!(parsed.payload, payload);
    }

    #[test]
    fn repeated_small_round_trips_reuse_output_buffer() {
        let mut out = Vec::with_capacity(128);
        for i in 0..500 {
            out.clear();
            let text = format!("chunk-{i}");
            let frame = WsFrame::text(text.as_bytes().to_vec());
            frame.encode(&mut out, None).unwrap();
            let (parsed, consumed) = parse_client(&out).unwrap().unwrap();
            assert_eq!(consumed, out.len(), "iteration {i}");
            assert_eq!(parsed.payload, text.as_bytes(), "iteration {i}");
        }
    }
}
