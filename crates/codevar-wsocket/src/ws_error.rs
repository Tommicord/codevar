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

//! WebSocket error types.
//!
//! Errors map to RFC 6455 failure modes: handshake rejection, framing
//! protocol violations, UTF-8 failures, and size limits. Fatal framing
//! errors carry an optional [`CloseCode`](crate::ws_ids::WsCloseCode)
//! so the connection layer can emit a Close frame before tearing down.

use crate::ws_ids::WsCloseCode;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ws_ids::WsCloseCode;

    #[test]
    fn display_texts_for_unit_variants() {
        assert_eq!(WsError::Closed.to_string(), "WebSocket connection closed");
        assert_eq!(
            WsError::WouldBlock.to_string(),
            "WebSocket operation would block"
        );
        assert_eq!(
            WsError::HandshakeNotComplete.to_string(),
            "WebSocket handshake not complete"
        );
        assert_eq!(
            WsError::RandomFailed.to_string(),
            "failed to generate secure random bytes"
        );
        assert_eq!(
            WsError::InvalidUtf8.to_string(),
            "invalid UTF-8 in WebSocket text payload"
        );
    }

    #[test]
    fn display_texts_embed_string_payloads() {
        assert_eq!(
            WsError::Handshake("bad upgrade".into()).to_string(),
            "WebSocket handshake error: bad upgrade"
        );
        assert_eq!(
            WsError::decode("frame truncated").to_string(),
            "WebSocket decode error: frame truncated"
        );
        assert_eq!(
            WsError::Unsupported("permessage-deflate".into()).to_string(),
            "unsupported WebSocket feature: permessage-deflate"
        );
        assert_eq!(
            WsError::invalid_state("already closed").to_string(),
            "invalid WebSocket state: already closed"
        );
        assert_eq!(
            WsError::Internal("invariant violated".into()).to_string(),
            "internal WebSocket error: invariant violated"
        );
        assert_eq!(
            WsError::Io("broken pipe".into()).to_string(),
            "WebSocket I/O error: broken pipe"
        );
    }

    #[test]
    fn display_text_for_protocol_and_size_errors() {
        let protocol = WsError::protocol(WsCloseCode::ProtocolError, "boom");
        assert_eq!(
            protocol.to_string(),
            "WebSocket protocol error (1002 (protocol error)): boom"
        );

        let utf8_code = WsError::protocol(WsCloseCode::InvalidPayloadData, "bad text");
        assert_eq!(
            utf8_code.to_string(),
            "WebSocket protocol error (1007 (invalid payload data)): bad text"
        );

        let too_big = WsError::MessageTooBig { size: 10, limit: 5 };
        assert_eq!(too_big.to_string(), "WebSocket message too big: 10 > 5");
    }

    #[test]
    fn close_code_classification_covers_every_variant() {
        assert_eq!(
            WsError::protocol(WsCloseCode::GoingAway, "m").close_code(),
            Some(WsCloseCode::GoingAway)
        );
        assert_eq!(
            WsError::InvalidUtf8.close_code(),
            Some(WsCloseCode::InvalidPayloadData)
        );
        assert_eq!(
            WsError::MessageTooBig { size: 1, limit: 0 }.close_code(),
            Some(WsCloseCode::MessageTooBig)
        );
        assert_eq!(
            WsError::Unsupported("x".into()).close_code(),
            Some(WsCloseCode::ProtocolError)
        );
        assert_eq!(
            WsError::decode("x").close_code(),
            Some(WsCloseCode::ProtocolError)
        );

        for non_wire in [
            WsError::Closed,
            WsError::WouldBlock,
            WsError::HandshakeNotComplete,
            WsError::RandomFailed,
            WsError::Handshake("h".into()),
            WsError::invalid_state("s"),
            WsError::Internal("i".into()),
            WsError::Io("io".into()),
        ] {
            assert_eq!(non_wire.close_code(), None, "{non_wire:?}");
        }
    }

    #[test]
    fn constructors_build_expected_variants() {
        assert_eq!(WsError::decode("m"), WsError::Decode(String::from("m")));
        assert_eq!(
            WsError::handshake("m"),
            WsError::Handshake(String::from("m"))
        );
        assert_eq!(
            WsError::invalid_state("m"),
            WsError::InvalidState(String::from("m"))
        );
        assert_eq!(
            WsError::protocol(WsCloseCode::InternalError, "m"),
            WsError::Protocol {
                message: String::from("m"),
                close_code: WsCloseCode::InternalError,
            }
        );
    }

    #[test]
    fn equality_matches_variants_and_payloads() {
        assert_eq!(WsError::Closed, WsError::Closed);
        assert_eq!(WsError::InvalidUtf8, WsError::InvalidUtf8);
        assert_eq!(WsError::decode("a"), WsError::decode("a"));
        assert_ne!(WsError::decode("a"), WsError::decode("b"));
        assert_ne!(WsError::decode("a"), WsError::handshake("a"));
        assert_ne!(
            WsError::protocol(WsCloseCode::Normal, "m"),
            WsError::protocol(WsCloseCode::GoingAway, "m")
        );
        assert_ne!(
            WsError::MessageTooBig { size: 1, limit: 2 },
            WsError::MessageTooBig { size: 2, limit: 1 }
        );
    }

    #[test]
    fn variant_matching_via_matches_macro() {
        assert!(matches!(WsError::WouldBlock, WsError::WouldBlock));
        assert!(matches!(
            WsError::protocol(WsCloseCode::Normal, "m"),
            WsError::Protocol { .. }
        ));
        assert!(matches!(
            WsError::MessageTooBig { size: 1, limit: 0 },
            WsError::MessageTooBig { .. }
        ));
        assert!(!matches!(WsError::Closed, WsError::Protocol { .. }));
        assert!(!matches!(WsError::InvalidUtf8, WsError::Decode(_)));
    }

    #[test]
    fn clone_preserves_variant_and_payload() {
        let original = WsError::protocol(WsCloseCode::PolicyViolation, "detail");
        assert_eq!(original.clone(), original);

        let too_big = WsError::MessageTooBig { size: 9, limit: 8 };
        assert_eq!(too_big.clone(), too_big);

        let decoded = WsError::decode("bytes");
        assert_eq!(decoded.clone(), decoded);
    }

    #[test]
    fn error_trait_object_reports_no_source_chain() {
        let errors = [
            WsError::Closed,
            WsError::decode("m"),
            WsError::protocol(WsCloseCode::Normal, "m"),
            WsError::InvalidUtf8,
            WsError::MessageTooBig { size: 2, limit: 1 },
            WsError::Io("io".into()),
        ];
        for err in errors {
            let boxed: Box<dyn std::error::Error> = Box::new(err.clone());
            assert!(boxed.source().is_none(), "source for {err:?}");
            assert_eq!(boxed.to_string(), err.to_string());
        }
    }

    #[test]
    fn from_io_error_wraps_message() {
        let io_err = std::io::Error::other("disk on fire");
        let ws_err = WsError::from(io_err);
        assert!(matches!(ws_err, WsError::Io(_)));
        if let WsError::Io(msg) = &ws_err {
            assert!(msg.contains("disk on fire"), "message: {msg}");
        }
        assert_eq!(ws_err.close_code(), None);

        let kind_only =
            WsError::from(std::io::Error::from(std::io::ErrorKind::ConnectionReset));
        assert!(matches!(kind_only, WsError::Io(_)));
    }

    #[test]
    fn result_alias_supports_question_mark_propagation() {
        fn inner() -> WsResult<u8> {
            Ok(7)
        }
        fn outer() -> WsResult<u8> {
            let v = inner()?;
            Ok(v + 1)
        }
        assert_eq!(outer().unwrap(), 8);

        fn failing() -> WsResult<u8> {
            Err(WsError::protocol(WsCloseCode::Normal, "nope"))
        }
        let err = failing().unwrap_err();
        assert_eq!(
            err.close_code(),
            Some(WsCloseCode::Normal),
            "propagated error: {err}"
        );
    }

    #[test]
    fn random_failed_maps_to_no_close_code() {
        let err = WsError::RandomFailed;
        assert_eq!(err.close_code(), None);
        assert_eq!(err.to_string(), "failed to generate secure random bytes");
    }
}
