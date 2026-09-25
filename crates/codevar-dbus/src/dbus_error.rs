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

//! Error types shared by the D-Bus client implementation.

use alloc::string::String;
use core::fmt;

/// Error produced by D-Bus operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbusError {
    /// A non-blocking operation could not make progress.
    WouldBlock,
    /// The bus or the peer closed the connection.
    Disconnected,
    /// A system call failed with `errno`.
    Io {
        /// Name of the failing system call, e.g. `connect`.
        op: &'static str,
        /// Value of `errno` reported by the kernel.
        errno: i32,
    },
    /// A bus name, object path or member violates the specification.
    InvalidName(String),
    /// A type signature is malformed or unsupported.
    InvalidSignature(String),
    /// A message violates the wire format.
    InvalidMessage(String),
    /// An encoded message exceeds the maximum wire message size.
    MessageTooBig(usize),
    /// A bus address cannot be parsed or resolved.
    InvalidAddress(String),
    /// The authentication handshake failed.
    Auth(String),
    /// The bus answered a method call with an error message.
    Remote {
        /// D-Bus error name, e.g. `org.freedesktop.DBus.Error.ServiceUnknown`.
        name: String,
        /// Human readable description sent with the error.
        message: String,
    },
    /// The reply to a method call did not arrive in time.
    Timeout,
    /// The operation is not valid in the current state.
    InvalidState(String),
    /// The requested feature is not provided by this implementation.
    Unsupported(String),
}

impl DbusError {
    /// Builds [`DbusError::Io`] from a system call name and `errno`.
    #[inline]
    #[must_use]
    pub const fn io(op: &'static str, errno: i32) -> Self {
        Self::Io { op, errno }
    }

    /// Builds [`DbusError::InvalidName`] from a description.
    #[inline]
    #[must_use]
    pub fn invalid_name(message: impl Into<String>) -> Self {
        Self::InvalidName(message.into())
    }

    /// Builds [`DbusError::InvalidSignature`] from a description.
    #[inline]
    #[must_use]
    pub fn invalid_signature(message: impl Into<String>) -> Self {
        Self::InvalidSignature(message.into())
    }

    /// Builds [`DbusError::InvalidMessage`] from a description.
    #[inline]
    #[must_use]
    pub fn invalid_message(message: impl Into<String>) -> Self {
        Self::InvalidMessage(message.into())
    }

    /// Builds [`DbusError::InvalidAddress`] from a description.
    #[inline]
    #[must_use]
    pub fn invalid_address(message: impl Into<String>) -> Self {
        Self::InvalidAddress(message.into())
    }

    /// Builds [`DbusError::Auth`] from a description.
    #[inline]
    #[must_use]
    pub fn auth(message: impl Into<String>) -> Self {
        Self::Auth(message.into())
    }

    /// Builds [`DbusError::InvalidState`] from a description.
    #[inline]
    #[must_use]
    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::InvalidState(message.into())
    }

    /// Builds [`DbusError::Unsupported`] from a description.
    #[inline]
    #[must_use]
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    /// Builds [`DbusError::Remote`] from an error name and description.
    #[inline]
    #[must_use]
    pub fn remote(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Remote {
            name: name.into(),
            message: message.into(),
        }
    }

    /// Returns `true` when no progress can be made without waiting.
    #[inline]
    #[must_use]
    pub const fn is_would_block(&self) -> bool {
        matches!(self, Self::WouldBlock)
    }

    /// Returns `true` when the connection is gone.
    #[inline]
    #[must_use]
    pub const fn is_disconnected(&self) -> bool {
        matches!(self, Self::Disconnected)
    }

    /// Returns `true` when the bus reported a remote error.
    #[inline]
    #[must_use]
    pub const fn is_remote(&self) -> bool {
        matches!(self, Self::Remote { .. })
    }

    /// Returns `true` when the handshake failed.
    #[inline]
    #[must_use]
    pub const fn is_auth(&self) -> bool {
        matches!(self, Self::Auth(_))
    }
}

impl fmt::Display for DbusError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WouldBlock => f.write_str("operation would block"),
            Self::Disconnected => f.write_str("connection closed"),
            Self::Io { op, errno } => write!(f, "`{op}` failed with errno {errno}"),
            Self::InvalidName(message) => write!(f, "invalid name: {message}"),
            Self::InvalidSignature(message) => write!(f, "invalid signature: {message}"),
            Self::InvalidMessage(message) => write!(f, "invalid message: {message}"),
            Self::MessageTooBig(size) => {
                write!(f, "message of {size} bytes exceeds the maximum size")
            }
            Self::InvalidAddress(message) => write!(f, "invalid address: {message}"),
            Self::Auth(message) => write!(f, "authentication failed: {message}"),
            Self::Remote { name, message } => write!(f, "remote error {name}: {message}"),
            Self::Timeout => f.write_str("method call timed out"),
            Self::InvalidState(message) => write!(f, "invalid state: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported: {message}"),
        }
    }
}

impl core::error::Error for DbusError {}

/// Result alias for D-Bus operations.
pub type DbusResult<T> = core::result::Result<T, DbusError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_display_contains_call_and_errno() {
        let err = DbusError::io("connect", 111);
        assert_eq!(err.to_string(), "`connect` failed with errno 111");
    }

    #[test]
    fn remote_error_display_contains_name_and_message() {
        let err = DbusError::remote(
            "org.freedesktop.DBus.Error.ServiceUnknown",
            "no such service",
        );
        assert!(err.is_remote());
        assert_eq!(
            err.to_string(),
            "remote error org.freedesktop.DBus.Error.ServiceUnknown: no such service"
        );
    }

    #[test]
    fn predicates_distinguish_error_kinds() {
        assert!(DbusError::WouldBlock.is_would_block());
        assert!(DbusError::Disconnected.is_disconnected());
        assert!(!DbusError::Timeout.is_would_block());
        assert!(!DbusError::Timeout.is_disconnected());
        assert!(!DbusError::Timeout.is_remote());
    }
}
