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

//! Error types shared by the Wayland client and server ports.

use alloc::string::String;
use core::fmt;

/// Error reported by the peer through `wl_display.error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WlProtocolError {
    /// Protocol error code, interpreted through `enum wl_display_error`.
    pub code: u32,
    /// Id of the object the error is attributed to.
    pub object_id: u32,
    /// Interface of the offending object, empty when unknown.
    pub interface: &'static str,
    /// Human readable description sent by the peer.
    pub message: String,
}

impl WlProtocolError {
    /// Creates a protocol error payload.
    #[inline]
    #[must_use]
    pub fn new(
        code: u32,
        object_id: u32,
        interface: &'static str,
        message: String,
    ) -> Self {
        Self {
            code,
            object_id,
            interface,
            message,
        }
    }
}

impl fmt::Display for WlProtocolError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "protocol error {} on {}#{}: {}",
            self.code, self.interface, self.object_id, self.message
        )
    }
}

impl core::error::Error for WlProtocolError {}

/// Error produced by Wayland client and server operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WlError {
    /// A non-blocking operation could not make progress.
    WouldBlock,
    /// The peer closed the connection or the socket failed.
    Disconnected,
    /// Transport level failure described by the backend.
    Io(String),
    /// The peer reported a protocol error via `wl_display.error`.
    Protocol(WlProtocolError),
    /// An encoded message exceeds the maximum wire message size.
    MessageTooBig(usize),
    /// The object id does not exist or has already been destroyed.
    InvalidObject(u32),
    /// The object received a request it does not implement.
    InvalidMethod {
        /// Interface of the target object.
        interface: &'static str,
        /// Opcode of the unmatched request.
        opcode: u32,
    },
    /// A marshalled argument does not match the message signature.
    InvalidArgument(String),
    /// The operation is not valid in the current state.
    InvalidState(String),
    /// The object map cannot allocate another id.
    TooManyObjects,
    /// The requested feature is not provided by this implementation.
    Unsupported(String),
}

impl WlError {
    /// Builds [`WlError::Io`] from a description.
    #[inline]
    #[must_use]
    pub fn io(message: impl Into<String>) -> Self {
        Self::Io(message.into())
    }

    /// Builds [`WlError::InvalidArgument`] from a description.
    #[inline]
    #[must_use]
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    /// Builds [`WlError::InvalidState`] from a description.
    #[inline]
    #[must_use]
    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::InvalidState(message.into())
    }

    /// Builds [`WlError::Unsupported`] from a description.
    #[inline]
    #[must_use]
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    /// Builds [`WlError::Protocol`] from its payload.
    #[inline]
    #[must_use]
    pub const fn protocol(error: WlProtocolError) -> Self {
        Self::Protocol(error)
    }

    /// Returns `true` when the peer reported a protocol error.
    #[inline]
    #[must_use]
    pub const fn is_protocol(&self) -> bool {
        matches!(self, Self::Protocol(_))
    }

    /// Returns `true` when no progress can be made without waiting.
    #[inline]
    #[must_use]
    pub const fn is_would_block(&self) -> bool {
        matches!(self, Self::WouldBlock)
    }
}

impl fmt::Display for WlError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WouldBlock => f.write_str("operation would block"),
            Self::Disconnected => f.write_str("connection closed"),
            Self::Io(message) => write!(f, "transport error: {message}"),
            Self::Protocol(error) => error.fmt(f),
            Self::MessageTooBig(size) => {
                write!(f, "message of {size} bytes exceeds the maximum size")
            }
            Self::InvalidObject(id) => write!(f, "invalid object id {id}"),
            Self::InvalidMethod { interface, opcode } => {
                write!(f, "invalid method {opcode} on {interface}")
            }
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::InvalidState(message) => write!(f, "invalid state: {message}"),
            Self::TooManyObjects => f.write_str("object map is full"),
            Self::Unsupported(message) => write!(f, "unsupported: {message}"),
        }
    }
}

impl core::error::Error for WlError {}

/// Result alias for Wayland operations.
pub type WlResult<T> = core::result::Result<T, WlError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_error_display_contains_context() {
        let error = WlProtocolError::new(
            1,
            0xff00_0001,
            "wl_registry",
            String::from("unknown global"),
        );
        let err = WlError::protocol(error);
        assert!(err.is_protocol());
        assert_eq!(
            err.to_string(),
            "protocol error 1 on wl_registry#4278190081: unknown global"
        );
    }

    #[test]
    fn would_block_predicate() {
        assert!(WlError::WouldBlock.is_would_block());
        assert!(!WlError::Disconnected.is_would_block());
        assert!(!WlError::WouldBlock.is_protocol());
    }
}
