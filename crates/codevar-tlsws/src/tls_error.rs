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

//! TLS error types.

use crate::tls_alert::AlertDescription;
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
        level: crate::tls_alert::AlertLevel,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tls_alert::{AlertDescription, AlertLevel};
    use std::error::Error as StdError;

    #[test]
    fn display_text_is_exact_for_unit_variants() {
        assert_eq!(TlsError::Closed.to_string(), "TLS connection closed");
        assert_eq!(
            TlsError::WouldBlock.to_string(),
            "TLS operation would block"
        );
        assert_eq!(
            TlsError::HandshakeNotComplete.to_string(),
            "TLS handshake not complete"
        );
        assert_eq!(
            TlsError::RandomFailed.to_string(),
            "failed to generate secure random bytes"
        );
    }

    #[test]
    fn display_text_is_exact_for_string_variants() {
        assert_eq!(
            TlsError::decode("bad len").to_string(),
            "TLS decode error: bad len"
        );
        assert_eq!(
            TlsError::crypto("mac mismatch").to_string(),
            "TLS crypto error: mac mismatch"
        );
        assert_eq!(
            TlsError::certificate("expired").to_string(),
            "TLS certificate error: expired"
        );
        assert_eq!(
            TlsError::Unsupported("suite".into()).to_string(),
            "unsupported TLS feature: suite"
        );
        assert_eq!(
            TlsError::Internal("invariant".into()).to_string(),
            "internal TLS error: invariant"
        );
        assert_eq!(
            TlsError::Io("reset".into()).to_string(),
            "TLS I/O error: reset"
        );
    }

    #[test]
    fn display_text_is_exact_for_alert_variants() {
        assert_eq!(
            TlsError::Alert(AlertDescription::HandshakeFailure).to_string(),
            "local TLS alert: HandshakeFailure"
        );
        let peer = TlsError::PeerAlert {
            level: AlertLevel::Warning,
            description: AlertDescription::CloseNotify,
        };
        assert_eq!(peer.to_string(), "peer alert (Warning): CloseNotify");
        let fatal = TlsError::PeerAlert {
            level: AlertLevel::Fatal,
            description: AlertDescription::DecodeError,
        };
        assert_eq!(fatal.to_string(), "peer alert (Fatal): DecodeError");
    }

    #[test]
    fn alert_description_mappings() {
        assert_eq!(
            TlsError::Alert(AlertDescription::InternalError).alert_description(),
            Some(AlertDescription::InternalError)
        );
        let peer = TlsError::PeerAlert {
            level: AlertLevel::Fatal,
            description: AlertDescription::ProtocolVersion,
        };
        assert_eq!(
            peer.alert_description(),
            Some(AlertDescription::ProtocolVersion)
        );
        assert_eq!(
            TlsError::decode("x").alert_description(),
            Some(AlertDescription::DecodeError)
        );
        assert_eq!(
            TlsError::crypto("x").alert_description(),
            Some(AlertDescription::DecryptError)
        );
        assert_eq!(
            TlsError::certificate("x").alert_description(),
            Some(AlertDescription::BadCertificate)
        );
        assert_eq!(
            TlsError::Unsupported("x".into()).alert_description(),
            Some(AlertDescription::IllegalParameter)
        );
        assert_eq!(
            TlsError::Closed.alert_description(),
            Some(AlertDescription::CloseNotify)
        );
    }

    #[test]
    fn non_alertable_variants_map_to_none() {
        for err in [
            TlsError::WouldBlock,
            TlsError::HandshakeNotComplete,
            TlsError::RandomFailed,
            TlsError::Internal("x".into()),
            TlsError::Io("x".into()),
        ] {
            assert_eq!(err.alert_description(), None, "err={err:?}");
        }
    }

    #[test]
    fn constructors_produce_expected_variants() {
        assert_eq!(TlsError::decode("m"), TlsError::Decode("m".into()));
        assert_eq!(TlsError::crypto("m"), TlsError::Crypto("m".into()));
        assert_eq!(
            TlsError::certificate("m"),
            TlsError::Certificate("m".into())
        );
    }

    #[test]
    fn partial_eq_distinguishes_variants_and_payloads() {
        assert_eq!(TlsError::decode("a"), TlsError::Decode("a".into()));
        assert_ne!(TlsError::decode("a"), TlsError::decode("b"));
        assert_ne!(
            TlsError::Alert(AlertDescription::DecodeError),
            TlsError::Alert(AlertDescription::DecryptError)
        );
        assert_ne!(TlsError::Closed, TlsError::WouldBlock);
        assert_ne!(
            TlsError::Closed.alert_description().is_some(),
            TlsError::WouldBlock.alert_description().is_some()
        );
        let cloned = TlsError::crypto("same").clone();
        assert_eq!(cloned, TlsError::crypto("same"));
    }

    #[test]
    fn error_source_chain_is_none() {
        let err: Box<dyn StdError> = Box::new(TlsError::decode("deep"));
        assert!(err.source().is_none());
        assert!(TlsError::Closed.source().is_none());
    }

    #[test]
    fn getrandom_error_converts_to_random_failed() {
        let converted = TlsError::from(getrandom::Error::UNSUPPORTED);
        assert_eq!(converted, TlsError::RandomFailed);
        assert_eq!(converted.alert_description(), None);
    }
}
