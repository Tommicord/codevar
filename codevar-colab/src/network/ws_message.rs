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

//! Application-facing WebSocket messages.
//!
//! A [`WsMessage`] is the unit delivered to (and accepted from) the application
//! after frame reassembly. Control frames that the stack handles internally
//! (automatic Pong replies, Close echo) are still surfaced so callers can
//! observe keepalives and peer-initiated shutdowns.

use crate::network::ws_ids::WsCloseCode;

/// A complete WebSocket message after reassembly (RFC 6455 §6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsMessage {
    /// UTF-8 text message (`opcode` Text, possibly fragmented).
    Text(String),
    /// Opaque binary message (`opcode` Binary, possibly fragmented).
    Binary(Vec<u8>),
    /// Ping control frame. The stack automatically replies with a matching
    /// Pong unless the connection is already closing.
    Ping(Vec<u8>),
    /// Pong control frame (solicited or unsolicited heartbeat).
    Pong(Vec<u8>),
    /// Close control frame with status code and UTF-8 reason.
    Close {
        /// Status code from the peer (or [`WsCloseCode::NoStatusReceived`]).
        code: WsCloseCode,
        /// UTF-8 close reason (may be empty).
        reason: String,
    },
}

impl WsMessage {
    /// Returns `true` for text or binary data messages.
    #[must_use]
    pub const fn is_data(&self) -> bool {
        matches!(self, Self::Text(_) | Self::Binary(_))
    }

    /// Returns `true` for Ping / Pong / Close control messages.
    #[must_use]
    pub const fn is_control(&self) -> bool {
        matches!(self, Self::Ping(_) | Self::Pong(_) | Self::Close { .. })
    }

    /// Creates a text message from a string.
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Creates a binary message from bytes.
    #[must_use]
    pub fn binary(b: impl Into<Vec<u8>>) -> Self {
        Self::Binary(b.into())
    }

    /// Creates a Close message.
    #[must_use]
    pub fn close(code: WsCloseCode, reason: impl Into<String>) -> Self {
        Self::Close {
            code,
            reason: reason.into(),
        }
    }
}
