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

//! Application-facing WebSocket messages.
//!
//! A [`WsMessage`] is the unit delivered to (and accepted from) the application
//! after frame reassembly. Control frames that the stack handles internally
//! (automatic Pong replies, Close echo) are still surfaced so callers can
//! observe keepalives and peer-initiated shutdowns.

use crate::ws_ids::WsCloseCode;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ws_ids::WsCloseCode;

    #[test]
    fn is_data_is_true_for_text_and_binary_only() {
        assert!(WsMessage::Text(String::from("hi")).is_data());
        assert!(WsMessage::Binary(vec![1, 2, 3]).is_data());
        assert!(WsMessage::text("hi").is_data());
        assert!(WsMessage::binary(vec![0u8]).is_data());

        assert!(!WsMessage::Ping(vec![]).is_data());
        assert!(!WsMessage::Pong(vec![]).is_data());
        assert!(!WsMessage::close(WsCloseCode::Normal, "").is_data());
    }

    #[test]
    fn is_control_is_true_for_ping_pong_and_close() {
        assert!(WsMessage::Ping(vec![1u8]).is_control());
        assert!(WsMessage::Pong(vec![1u8]).is_control());
        assert!(WsMessage::close(WsCloseCode::GoingAway, "bye").is_control());

        assert!(!WsMessage::text("data").is_control());
        assert!(!WsMessage::binary(vec![0u8]).is_control());
    }

    #[test]
    fn every_variant_is_exactly_one_of_data_or_control() {
        let messages = [
            WsMessage::text("a"),
            WsMessage::binary(vec![0]),
            WsMessage::Ping(vec![]),
            WsMessage::Pong(vec![]),
            WsMessage::close(WsCloseCode::Normal, ""),
        ];
        for message in messages {
            assert!(
                message.is_data() != message.is_control(),
                "classification mismatch for {message:?}"
            );
        }
    }

    #[test]
    fn text_constructor_handles_empty_and_unicode() {
        assert_eq!(WsMessage::text(""), WsMessage::Text(String::new()));
        assert_eq!(WsMessage::text("héllo"), WsMessage::Text(String::from("héllo")));

        let owned = String::from("owned");
        let msg = WsMessage::text(owned.clone());
        assert_eq!(msg, WsMessage::Text(owned));

        assert_eq!(WsMessage::Text(String::from("🙂")), WsMessage::text("🙂"));
    }

    #[test]
    fn binary_constructor_preserves_bytes() {
        assert_eq!(WsMessage::binary(vec![]), WsMessage::Binary(Vec::new()));
        let bytes: Vec<u8> = (0u16..=255).map(|i| u8::try_from(i).unwrap()).collect();
        let msg = WsMessage::binary(bytes.clone());
        assert!(msg.is_data());
        assert_eq!(msg, WsMessage::Binary(bytes));
    }

    #[test]
    fn close_constructor_carries_code_and_reason() {
        let msg = WsMessage::close(WsCloseCode::PolicyViolation, "nope");
        assert_eq!(
            msg,
            WsMessage::Close {
                code: WsCloseCode::PolicyViolation,
                reason: String::from("nope"),
            }
        );

        let empty = WsMessage::close(WsCloseCode::Normal, "");
        assert_eq!(
            empty,
            WsMessage::Close {
                code: WsCloseCode::Normal,
                reason: String::new(),
            }
        );
    }

    #[test]
    fn equality_distinguishes_variants_and_payloads() {
        assert_eq!(WsMessage::text("x"), WsMessage::text("x"));
        assert_ne!(WsMessage::text("x"), WsMessage::text("y"));
        assert_ne!(WsMessage::text("1"), WsMessage::binary(vec![b'1']));
        assert_ne!(WsMessage::Ping(vec![1]), WsMessage::Pong(vec![1]));
        assert_ne!(
            WsMessage::close(WsCloseCode::Normal, ""),
            WsMessage::close(WsCloseCode::GoingAway, "")
        );
        assert_ne!(
            WsMessage::close(WsCloseCode::Normal, "a"),
            WsMessage::close(WsCloseCode::Normal, "b")
        );
    }

    #[test]
    fn clone_preserves_variant_and_content() {
        let original = WsMessage::close(WsCloseCode::InternalError, "oops");
        assert_eq!(original.clone(), original);

        let binary = WsMessage::binary(vec![9, 8, 7]);
        assert_eq!(binary.clone(), binary);
    }

    #[test]
    fn debug_format_identifies_variants() {
        assert!(format!("{:?}", WsMessage::text("t")).starts_with("Text"));
        assert!(format!("{:?}", WsMessage::binary(vec![])).starts_with("Binary"));
        assert!(format!("{:?}", WsMessage::Ping(vec![])).starts_with("Ping"));
        assert!(format!("{:?}", WsMessage::Pong(vec![])).starts_with("Pong"));

        let close_dbg = format!("{:?}", WsMessage::close(WsCloseCode::Normal, "r"));
        assert!(close_dbg.starts_with("Close"));
        assert!(close_dbg.contains("code: Normal"));
        assert!(close_dbg.contains("reason: \"r\""));
    }
}
