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

//! WebSocket error types.
//!
//! Errors map to RFC 6455 failure modes: handshake rejection, framing
//! protocol violations, UTF-8 failures, and size limits. Fatal framing
//! errors carry an optional [`CloseCode`](crate::network::ws_ids::WsCloseCode)
//! so the connection layer can emit a Close frame before tearing down.

use crate::network::ws_ids::WsCloseCode;
use std::fmt;

/// Result alias for WebSocket operations.
pub type WsResult<T> = Result<T, WsError>;

/// Errors produced by the WebSocket stack.
///
/// # Categories
///
/// - **Handshake** — malformed or rejected HTTP Upgrade exchange
/// - **Protocol** — framing / masking / fragmentation violations (RFC 6455 §5)
/// - **Payload** — UTF-8 or size-limit failures
/// - **Lifecycle** — operations attempted in an invalid connection state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsError {
    /// Peer closed the connection cleanly (Close handshake completed).
    Closed,
    /// Would block waiting for more input bytes.
    WouldBlock,
    /// Opening handshake is not finished yet.
    HandshakeNotComplete,
    /// Failed to obtain cryptographically secure random bytes.
    RandomFailed,
    /// HTTP Upgrade handshake failed or was rejected.
    Handshake(String),
    /// Incoming bytes could not be decoded as a WebSocket frame or HTTP message.
    Decode(String),
    /// Framing or state-machine protocol violation (RFC 6455 §5 / §7).
    Protocol {
        /// Human-readable description.
        message: String,
        /// Close status code that should be sent when failing the connection.
        close_code: WsCloseCode,
    },
    /// Text message or Close reason is not valid UTF-8 (RFC 6455 §8.1).
    InvalidUtf8,
    /// Frame or reassembled message exceeds the configured size limit.
    MessageTooBig {
        /// Observed size in bytes.
        size: usize,
        /// Configured maximum in bytes.
        limit: usize,
    },
    /// Unsupported feature (extension, reserved opcode, wrong version).
    Unsupported(String),
    /// Operation is illegal in the current connection state.
    InvalidState(String),
    /// Internal invariant failure.
    Internal(String),
    /// Underlying I/O failure (stream adapter).
    Io(String),
}

impl WsError {
    /// Returns the Close status code that should be sent when failing the
    /// connection, if this error maps to a wire-visible failure.
    #[must_use]
    pub fn close_code(&self) -> Option<WsCloseCode> {
        match self {
            Self::Protocol { close_code, .. } => Some(*close_code),
            Self::InvalidUtf8 => Some(WsCloseCode::InvalidPayloadData),
            Self::MessageTooBig { .. } => Some(WsCloseCode::MessageTooBig),
            Self::Unsupported(_) => Some(WsCloseCode::ProtocolError),
            Self::Decode(_) => Some(WsCloseCode::ProtocolError),
            Self::Closed
            | Self::WouldBlock
            | Self::HandshakeNotComplete
            | Self::RandomFailed
            | Self::Handshake(_)
            | Self::InvalidState(_)
            | Self::Internal(_)
            | Self::Io(_) => None,
        }
    }

    /// Creates a decode error with a context message.
    #[must_use]
    pub fn decode(msg: impl Into<String>) -> Self {
        Self::Decode(msg.into())
    }

    /// Creates a handshake error.
    #[must_use]
    pub fn handshake(msg: impl Into<String>) -> Self {
        Self::Handshake(msg.into())
    }

    /// Creates a protocol error with the given close code.
    #[must_use]
    pub fn protocol(close_code: WsCloseCode, msg: impl Into<String>) -> Self {
        Self::Protocol {
            message: msg.into(),
            close_code,
        }
    }

    /// Creates an invalid-state error.
    #[must_use]
    pub fn invalid_state(msg: impl Into<String>) -> Self {
        Self::InvalidState(msg.into())
    }
}

impl fmt::Display for WsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => write!(f, "WebSocket connection closed"),
            Self::WouldBlock => write!(f, "WebSocket operation would block"),
            Self::HandshakeNotComplete => write!(f, "WebSocket handshake not complete"),
            Self::RandomFailed => write!(f, "failed to generate secure random bytes"),
            Self::Handshake(m) => write!(f, "WebSocket handshake error: {m}"),
            Self::Decode(m) => write!(f, "WebSocket decode error: {m}"),
            Self::Protocol {
                message,
                close_code,
            } => write!(f, "WebSocket protocol error ({close_code}): {message}"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 in WebSocket text payload"),
            Self::MessageTooBig { size, limit } => {
                write!(f, "WebSocket message too big: {size} > {limit}")
            }
            Self::Unsupported(m) => write!(f, "unsupported WebSocket feature: {m}"),
            Self::InvalidState(m) => write!(f, "invalid WebSocket state: {m}"),
            Self::Internal(m) => write!(f, "internal WebSocket error: {m}"),
            Self::Io(m) => write!(f, "WebSocket I/O error: {m}"),
        }
    }
}

impl std::error::Error for WsError {}

impl From<getrandom::Error> for WsError {
    fn from(_: getrandom::Error) -> Self {
        Self::RandomFailed
    }
}

impl From<std::io::Error> for WsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
