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

//! TLS error types.

use crate::network::tls_alert::AlertDescription;
use std::fmt;

/// Result alias for TLS operations.
pub type TlsResult<T> = Result<T, TlsError>;

/// Errors produced by the TLS stack.
///
/// Fatal handshake or record failures carry an [`AlertDescription`] so the
/// peer can be notified with a matching TLS alert (RFC 8446 §6 / RFC 5246 §7.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TlsError {
    /// Peer closed the connection cleanly (`close_notify`).
    Closed,
    /// Would block waiting for more input or output buffer space.
    WouldBlock,
    /// Handshake is not finished yet.
    HandshakeNotComplete,
    /// Failed to obtain cryptographically secure random bytes.
    RandomFailed,
    /// Incoming bytes could not be decoded as a TLS structure.
    Decode(String),
    /// Peer sent a TLS alert.
    PeerAlert {
        /// Alert severity.
        level: crate::network::tls_alert::AlertLevel,
        /// Alert description.
        description: AlertDescription,
    },
    /// Local condition that must be reported as a fatal alert.
    Alert(AlertDescription),
    /// Cryptographic verification failed (MAC, Finished, signature, etc.).
    Crypto(String),
    /// Certificate path or identity validation failed.
    Certificate(String),
    /// Unsupported or mismatched protocol parameter.
    Unsupported(String),
    /// Internal invariant failure without a dedicated alert mapping.
    Internal(String),
    /// Underlying I/O failure.
    Io(String),
}

impl TlsError {
    /// Returns the fatal alert description that should be sent to the peer, if any.
    #[must_use]
    pub fn alert_description(&self) -> Option<AlertDescription> {
        match self {
            Self::Alert(d) | Self::PeerAlert { description: d, .. } => Some(*d),
            Self::Decode(_) => Some(AlertDescription::DecodeError),
            Self::Crypto(_) => Some(AlertDescription::DecryptError),
            Self::Certificate(_) => Some(AlertDescription::BadCertificate),
            Self::Unsupported(_) => Some(AlertDescription::IllegalParameter),
            Self::Closed => Some(AlertDescription::CloseNotify),
            Self::WouldBlock
            | Self::HandshakeNotComplete
            | Self::RandomFailed
            | Self::Internal(_)
            | Self::Io(_) => None,
        }
    }

    /// Creates a decode error with a static context message.
    #[must_use]
    pub fn decode(msg: impl Into<String>) -> Self {
        Self::Decode(msg.into())
    }

    /// Creates a crypto error.
    #[must_use]
    pub fn crypto(msg: impl Into<String>) -> Self {
        Self::Crypto(msg.into())
    }

    /// Creates a certificate error.
    #[must_use]
    pub fn certificate(msg: impl Into<String>) -> Self {
        Self::Certificate(msg.into())
    }
}

impl fmt::Display for TlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => write!(f, "TLS connection closed"),
            Self::WouldBlock => write!(f, "TLS operation would block"),
            Self::HandshakeNotComplete => write!(f, "TLS handshake not complete"),
            Self::RandomFailed => write!(f, "failed to generate secure random bytes"),
            Self::Decode(m) => write!(f, "TLS decode error: {m}"),
            Self::PeerAlert { level, description } => {
                write!(f, "peer alert ({level:?}): {description:?}")
            }
            Self::Alert(d) => write!(f, "local TLS alert: {d:?}"),
            Self::Crypto(m) => write!(f, "TLS crypto error: {m}"),
            Self::Certificate(m) => write!(f, "TLS certificate error: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported TLS feature: {m}"),
            Self::Internal(m) => write!(f, "internal TLS error: {m}"),
            Self::Io(m) => write!(f, "TLS I/O error: {m}"),
        }
    }
}

impl std::error::Error for TlsError {}

impl From<getrandom::Error> for TlsError {
    fn from(_: getrandom::Error) -> Self {
        Self::RandomFailed
    }
}
