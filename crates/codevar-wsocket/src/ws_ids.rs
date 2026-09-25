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

//! Numeric identifiers, opcodes, close codes, and protocol constants.
//!
//! Values follow RFC 6455 §5.2 (opcodes), §7.4 (status codes), and §11.3
//! (handshake header semantics).

use crate::ws_error::{WsError, WsResult};
use bsize::BSize;
use std::fmt;

/// WebSocket protocol version advertised in `Sec-WebSocket-Version` (RFC 6455).
pub const VERSION: u8 = 13;

/// GUID concatenated with `Sec-WebSocket-Key` when computing `Sec-WebSocket-Accept`
/// (RFC 6455 §1.3 / §4.2.2).
pub const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Default maximum reassembled message size (16 MiB).
pub const DEFAULT_MAX_MESSAGE_SIZE: usize = BSize::mib(16).bytes();

/// Default maximum single-frame payload size (16 MiB).
pub const DEFAULT_MAX_FRAME_SIZE: usize = BSize::mib(16).bytes();

/// Maximum payload length allowed for a control frame (RFC 6455 §5.5).
pub const MAX_CONTROL_PAYLOAD: usize = 125;

/// Endpoint role determining masking rules (RFC 6455 §5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Client: MUST mask all outbound frames; MUST reject masked inbound frames.
    Client,
    /// Server: MUST NOT mask outbound frames; MUST reject unmasked inbound frames.
    Server,
}

impl Role {
    /// Returns `true` when this role must mask outbound frames.
    #[must_use]
    pub const fn must_mask_outbound(self) -> bool {
        matches!(self, Self::Client)
    }

    /// Returns `true` when an inbound frame with `masked == frame_masked` is legal.
    #[must_use]
    pub const fn expects_inbound_masked(self) -> bool {
        matches!(self, Self::Server)
    }
}

/// Frame opcodes (RFC 6455 §5.2).
///
/// Control opcodes have the high bit of the 4-bit opcode field set (`0x8`–`0xF`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WsOpcode {
    /// Continuation frame of a fragmented message (`%x0`).
    Continuation = 0x0,
    /// Text data frame (`%x1`); payload MUST be UTF-8 when complete.
    Text = 0x1,
    /// Binary data frame (`%x2`).
    Binary = 0x2,
    /// Connection close control frame (`%x8`).
    Close = 0x8,
    /// Ping control frame (`%x9`).
    Ping = 0x9,
    /// Pong control frame (`%xA`).
    Pong = 0xA,
}

impl WsOpcode {
    /// Parses an opcode nibble. Reserved values are rejected.
    pub fn from_u8(v: u8) -> WsResult<Self> {
        match v {
            0x0 => Ok(Self::Continuation),
            0x1 => Ok(Self::Text),
            0x2 => Ok(Self::Binary),
            0x8 => Ok(Self::Close),
            0x9 => Ok(Self::Ping),
            0xA => Ok(Self::Pong),
            0x3..=0x7 => Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                format!("reserved non-control opcode 0x{v:x}"),
            )),
            0xB..=0xF => Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                format!("reserved control opcode 0x{v:x}"),
            )),
            other => Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                format!("invalid opcode 0x{other:x}"),
            )),
        }
    }

    /// Wire value (low 4 bits).
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns `true` for control opcodes (`Close`, `Ping`, `Pong`).
    #[must_use]
    pub const fn is_control(self) -> bool {
        matches!(self, Self::Close | Self::Ping | Self::Pong)
    }

    /// Returns `true` for data opcodes (`Text`, `Binary`).
    #[must_use]
    pub const fn is_data(self) -> bool {
        matches!(self, Self::Text | Self::Binary)
    }
}

impl fmt::Display for WsOpcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Continuation => write!(f, "continuation"),
            Self::Text => write!(f, "text"),
            Self::Binary => write!(f, "binary"),
            Self::Close => write!(f, "close"),
            Self::Ping => write!(f, "ping"),
            Self::Pong => write!(f, "pong"),
        }
    }
}

/// WebSocket close status codes (RFC 6455 §7.4.1).
///
/// Codes 1005 and 1006 are reserved for applications and MUST NOT appear
/// on the wire in a Close frame body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WsCloseCode {
    /// 1000 — normal closure.
    Normal,
    /// 1001 — endpoint is going away.
    GoingAway,
    /// 1002 — protocol error.
    ProtocolError,
    /// 1003 — unsupported data type.
    UnsupportedData,
    /// 1005 — no status code was present (MUST NOT be sent on the wire).
    NoStatusReceived,
    /// 1006 — abnormal closure (MUST NOT be sent on the wire).
    Abnormal,
    /// 1007 — invalid payload data (e.g. non-UTF-8 text).
    InvalidPayloadData,
    /// 1008 — policy violation.
    PolicyViolation,
    /// 1009 — message too big.
    MessageTooBig,
    /// 1010 — mandatory extension missing (client only).
    MandatoryExtension,
    /// 1011 — unexpected server condition.
    InternalError,
    /// 1015 — TLS handshake failure (MUST NOT be sent on the wire).
    TlsHandshake,
    /// Application-defined or IANA-registered code in an allowed range.
    Other(u16),
}

impl WsCloseCode {
    /// Parses a 16-bit status code from a Close frame body.
    ///
    /// Rejects codes that MUST NOT appear on the wire (1005, 1006, 1015)
    /// and codes in the forbidden ranges 0–999.
    pub fn from_u16(code: u16) -> WsResult<Self> {
        match code {
            1000 => Ok(Self::Normal),
            1001 => Ok(Self::GoingAway),
            1002 => Ok(Self::ProtocolError),
            1003 => Ok(Self::UnsupportedData),
            1005 | 1006 | 1015 => Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                format!("close code {code} must not be sent on the wire"),
            )),
            1007 => Ok(Self::InvalidPayloadData),
            1008 => Ok(Self::PolicyViolation),
            1009 => Ok(Self::MessageTooBig),
            1010 => Ok(Self::MandatoryExtension),
            1011 => Ok(Self::InternalError),
            1004 => Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                "close code 1004 is reserved",
            )),
            0..=999 => Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                format!("close code {code} is forbidden (below 1000)"),
            )),
            // 1012–1014 and 1016–2999 are registered / reserved by IANA; accept
            // as opaque application codes. 3000–4999 are for libraries/apps.
            1012..=1014 | 1016..=4999 => Ok(Self::Other(code)),
            other => Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                format!("close code {other} is out of range"),
            )),
        }
    }

    /// Wire value suitable for a Close frame body.
    ///
    /// # Panics
    ///
    /// This method does not panic; reserved codes (`NoStatusReceived`,
    /// `Abnormal`, `TlsHandshake`) still return their numeric values so
    /// callers that misuse them can be caught by [`is_sendable`].
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        match self {
            Self::Normal => 1000,
            Self::GoingAway => 1001,
            Self::ProtocolError => 1002,
            Self::UnsupportedData => 1003,
            Self::NoStatusReceived => 1005,
            Self::Abnormal => 1006,
            Self::InvalidPayloadData => 1007,
            Self::PolicyViolation => 1008,
            Self::MessageTooBig => 1009,
            Self::MandatoryExtension => 1010,
            Self::InternalError => 1011,
            Self::TlsHandshake => 1015,
            Self::Other(c) => c,
        }
    }

    /// Returns `true` when this code may legally appear in a Close frame body.
    #[must_use]
    pub const fn is_sendable(self) -> bool {
        !matches!(self, Self::NoStatusReceived | Self::Abnormal | Self::TlsHandshake)
    }
}

impl fmt::Display for WsCloseCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.as_u16(), self.label())
    }
}

impl WsCloseCode {
    /// Short English label for logging / diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal closure",
            Self::GoingAway => "going away",
            Self::ProtocolError => "protocol error",
            Self::UnsupportedData => "unsupported data",
            Self::NoStatusReceived => "no status received",
            Self::Abnormal => "abnormal closure",
            Self::InvalidPayloadData => "invalid payload data",
            Self::PolicyViolation => "policy violation",
            Self::MessageTooBig => "message too big",
            Self::MandatoryExtension => "mandatory extension",
            Self::InternalError => "internal error",
            Self::TlsHandshake => "TLS handshake",
            Self::Other(_) => "other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn protocol_constants_have_expected_values() {
        assert_eq!(VERSION, 13);
        assert_eq!(GUID, "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
        assert_eq!(GUID.len(), 36);
        assert_eq!(MAX_CONTROL_PAYLOAD, 125);
        assert_eq!(DEFAULT_MAX_FRAME_SIZE, 16 * 1024 * 1024);
        assert_eq!(DEFAULT_MAX_MESSAGE_SIZE, 16 * 1024 * 1024);
        assert_eq!(DEFAULT_MAX_FRAME_SIZE, DEFAULT_MAX_MESSAGE_SIZE);
    }

    #[test]
    fn role_masking_expectations_are_symmetric() {
        assert!(Role::Client.must_mask_outbound());
        assert!(!Role::Client.expects_inbound_masked());
        assert!(!Role::Server.must_mask_outbound());
        assert!(Role::Server.expects_inbound_masked());
    }

    #[test]
    fn role_derives_are_consistent() {
        assert_eq!(Role::Client, Role::Client);
        assert_ne!(Role::Client, Role::Server);
        let mut set = HashSet::new();
        set.insert(Role::Client);
        set.insert(Role::Server);
        set.insert(Role::Client);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn opcode_from_u8_accepts_defined_values() {
        let expected = [
            (0x0u8, WsOpcode::Continuation),
            (0x1, WsOpcode::Text),
            (0x2, WsOpcode::Binary),
            (0x8, WsOpcode::Close),
            (0x9, WsOpcode::Ping),
            (0xA, WsOpcode::Pong),
        ];
        for (raw, opcode) in expected {
            assert_eq!(WsOpcode::from_u8(raw).unwrap(), opcode, "raw {raw:#x}");
            assert_eq!(opcode.as_u8(), raw);
        }
    }

    #[test]
    fn opcode_from_u8_rejects_reserved_and_out_of_range_values() {
        for raw in [
            0x3u8, 0x4, 0x5, 0x6, 0x7, 0xB, 0xC, 0xD, 0xE, 0xF, 0x10, 0x7F, 0xFF,
        ] {
            let err = WsOpcode::from_u8(raw).unwrap_err();
            assert!(
                matches!(
                    err,
                    WsError::Protocol {
                        close_code: WsCloseCode::ProtocolError,
                        ..
                    }
                ),
                "raw {raw:#x}: {err:?}"
            );
        }
    }

    #[test]
    fn opcode_rejection_messages_distinguish_reserved_ranges() {
        let non_control = match WsOpcode::from_u8(0x3).unwrap_err() {
            WsError::Protocol { message, .. } => message,
            other => format!("unexpected variant: {other:?}"),
        };
        assert!(non_control.contains("reserved non-control"), "{non_control}");

        let control = match WsOpcode::from_u8(0xB).unwrap_err() {
            WsError::Protocol { message, .. } => message,
            other => format!("unexpected variant: {other:?}"),
        };
        assert!(control.contains("reserved control"), "{control}");

        let out_of_range = match WsOpcode::from_u8(0x40).unwrap_err() {
            WsError::Protocol { message, .. } => message,
            other => format!("unexpected variant: {other:?}"),
        };
        assert!(out_of_range.contains("invalid opcode"), "{out_of_range}");
    }

    #[test]
    fn opcode_control_and_data_classification() {
        for opcode in [WsOpcode::Close, WsOpcode::Ping, WsOpcode::Pong] {
            assert!(opcode.is_control(), "{opcode}");
            assert!(!opcode.is_data(), "{opcode}");
        }
        for opcode in [WsOpcode::Text, WsOpcode::Binary] {
            assert!(opcode.is_data(), "{opcode}");
            assert!(!opcode.is_control(), "{opcode}");
        }
        assert!(!WsOpcode::Continuation.is_control());
        assert!(!WsOpcode::Continuation.is_data());
    }

    #[test]
    fn opcode_display_and_debug_strings() {
        let expected = [
            (WsOpcode::Continuation, "continuation", "Continuation"),
            (WsOpcode::Text, "text", "Text"),
            (WsOpcode::Binary, "binary", "Binary"),
            (WsOpcode::Close, "close", "Close"),
            (WsOpcode::Ping, "ping", "Ping"),
            (WsOpcode::Pong, "pong", "Pong"),
        ];
        for (opcode, name, debug_name) in expected {
            assert_eq!(opcode.to_string(), name);
            assert_eq!(format!("{opcode:?}"), debug_name);
        }
    }

    #[test]
    fn opcode_variants_hash_distinctly() {
        let opcodes = [
            WsOpcode::Continuation,
            WsOpcode::Text,
            WsOpcode::Binary,
            WsOpcode::Close,
            WsOpcode::Ping,
            WsOpcode::Pong,
        ];
        let set: HashSet<WsOpcode> = opcodes.into_iter().collect();
        assert_eq!(set.len(), 6);
        assert!(set.contains(&WsOpcode::Pong));
    }

    #[test]
    fn close_code_from_u16_accepts_valid_codes() {
        let named = [
            (1000u16, WsCloseCode::Normal),
            (1001, WsCloseCode::GoingAway),
            (1002, WsCloseCode::ProtocolError),
            (1003, WsCloseCode::UnsupportedData),
            (1007, WsCloseCode::InvalidPayloadData),
            (1008, WsCloseCode::PolicyViolation),
            (1009, WsCloseCode::MessageTooBig),
            (1010, WsCloseCode::MandatoryExtension),
            (1011, WsCloseCode::InternalError),
        ];
        for (raw, code) in named {
            assert_eq!(WsCloseCode::from_u16(raw).unwrap(), code, "raw {raw}");
        }

        for raw in [1012u16, 1013, 1014, 1016, 3000, 4999] {
            assert_eq!(
                WsCloseCode::from_u16(raw).unwrap(),
                WsCloseCode::Other(raw),
                "raw {raw}"
            );
        }
    }

    #[test]
    fn close_code_from_u16_rejects_reserved_and_forbidden_codes() {
        for raw in [0u16, 1, 999, 1004, 1005, 1006, 1015, 5000, 65535] {
            let err = WsCloseCode::from_u16(raw).unwrap_err();
            assert!(
                matches!(
                    err,
                    WsError::Protocol {
                        close_code: WsCloseCode::ProtocolError,
                        ..
                    }
                ),
                "raw {raw}: {err:?}"
            );
        }
    }

    #[test]
    fn close_code_rejection_messages_carry_the_offending_value() {
        for (raw, needle) in [
            (500u16, "500"),
            (1004u16, "reserved"),
            (1005u16, "1005"),
            (5000u16, "5000"),
        ] {
            let msg = match WsCloseCode::from_u16(raw).unwrap_err() {
                WsError::Protocol { message, .. } => message,
                other => format!("unexpected variant: {other:?}"),
            };
            assert!(msg.contains(needle), "raw {raw}: {msg}");
        }
    }

    #[test]
    fn close_code_as_u16_round_trips_parseable_values() {
        for raw in [
            1000u16, 1001, 1002, 1003, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1016, 3000, 4999,
        ] {
            let code = WsCloseCode::from_u16(raw).unwrap();
            assert_eq!(code.as_u16(), raw, "raw {raw}");
        }
    }

    #[test]
    fn close_code_as_u16_exposes_reserved_variants() {
        assert_eq!(WsCloseCode::NoStatusReceived.as_u16(), 1005);
        assert_eq!(WsCloseCode::Abnormal.as_u16(), 1006);
        assert_eq!(WsCloseCode::TlsHandshake.as_u16(), 1015);
        assert_eq!(WsCloseCode::Other(0).as_u16(), 0);
        assert_eq!(WsCloseCode::Other(65535).as_u16(), 65535);
    }

    #[test]
    fn close_code_is_sendable_matrix() {
        assert!(!WsCloseCode::NoStatusReceived.is_sendable());
        assert!(!WsCloseCode::Abnormal.is_sendable());
        assert!(!WsCloseCode::TlsHandshake.is_sendable());
        for code in [
            WsCloseCode::Normal,
            WsCloseCode::GoingAway,
            WsCloseCode::ProtocolError,
            WsCloseCode::UnsupportedData,
            WsCloseCode::InvalidPayloadData,
            WsCloseCode::PolicyViolation,
            WsCloseCode::MessageTooBig,
            WsCloseCode::MandatoryExtension,
            WsCloseCode::InternalError,
            WsCloseCode::Other(3000),
        ] {
            assert!(code.is_sendable(), "{code}");
        }
    }

    #[test]
    fn close_code_display_formats_numeric_value_and_label() {
        assert_eq!(WsCloseCode::Normal.to_string(), "1000 (normal closure)");
        assert_eq!(WsCloseCode::ProtocolError.to_string(), "1002 (protocol error)");
        assert_eq!(WsCloseCode::Abnormal.to_string(), "1006 (abnormal closure)");
        assert_eq!(WsCloseCode::Other(3000).to_string(), "3000 (other)");
    }

    #[test]
    fn close_code_labels_cover_every_variant() {
        let expected = [
            (WsCloseCode::Normal, "normal closure"),
            (WsCloseCode::GoingAway, "going away"),
            (WsCloseCode::ProtocolError, "protocol error"),
            (WsCloseCode::UnsupportedData, "unsupported data"),
            (WsCloseCode::NoStatusReceived, "no status received"),
            (WsCloseCode::Abnormal, "abnormal closure"),
            (WsCloseCode::InvalidPayloadData, "invalid payload data"),
            (WsCloseCode::PolicyViolation, "policy violation"),
            (WsCloseCode::MessageTooBig, "message too big"),
            (WsCloseCode::MandatoryExtension, "mandatory extension"),
            (WsCloseCode::InternalError, "internal error"),
            (WsCloseCode::TlsHandshake, "TLS handshake"),
            (WsCloseCode::Other(1012), "other"),
        ];
        for (code, label) in expected {
            assert_eq!(code.label(), label, "{code}");
        }
    }

    #[test]
    fn close_code_debug_strings_are_variant_names() {
        assert_eq!(format!("{:?}", WsCloseCode::Normal), "Normal");
        assert_eq!(format!("{:?}", WsCloseCode::TlsHandshake), "TlsHandshake");
        assert_eq!(format!("{:?}", WsCloseCode::Other(3000)), "Other(3000)");
    }

    #[test]
    fn close_code_variants_hash_distinctly() {
        let codes = [
            WsCloseCode::Normal,
            WsCloseCode::GoingAway,
            WsCloseCode::ProtocolError,
            WsCloseCode::UnsupportedData,
            WsCloseCode::NoStatusReceived,
            WsCloseCode::Abnormal,
            WsCloseCode::InvalidPayloadData,
            WsCloseCode::PolicyViolation,
            WsCloseCode::MessageTooBig,
            WsCloseCode::MandatoryExtension,
            WsCloseCode::InternalError,
            WsCloseCode::TlsHandshake,
            WsCloseCode::Other(1012),
            WsCloseCode::Other(3000),
        ];
        let set: HashSet<WsCloseCode> = codes.into_iter().collect();
        assert_eq!(set.len(), codes.len());
        assert!(set.contains(&WsCloseCode::Other(1012)));
        assert!(!set.contains(&WsCloseCode::Other(1013)));
    }
}
