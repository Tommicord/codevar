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

//! Numeric identifiers, opcodes, close codes, and protocol constants.
//!
//! Values follow RFC 6455 §5.2 (opcodes), §7.4 (status codes), and §11.3
//! (handshake header semantics).

use crate::network::ws_error::{WsError, WsResult};
use std::fmt;

/// WebSocket protocol version advertised in `Sec-WebSocket-Version` (RFC 6455).
pub const VERSION: u8 = 13;

/// GUID concatenated with `Sec-WebSocket-Key` when computing `Sec-WebSocket-Accept`
/// (RFC 6455 §1.3 / §4.2.2).
pub const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Default maximum reassembled message size (16 MiB).
pub const DEFAULT_MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Default maximum single-frame payload size (16 MiB).
pub const DEFAULT_MAX_FRAME_SIZE: usize = 16 * 1024 * 1024;

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
        !matches!(
            self,
            Self::NoStatusReceived | Self::Abnormal | Self::TlsHandshake
        )
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
