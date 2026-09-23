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

//! TLS alert protocol (RFC 8446 §6, RFC 5246 §7.2).

use crate::tls_error::{TlsError, TlsResult};

/// Alert severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AlertLevel {
    /// Warning-level alert (TLS 1.2); TLS 1.3 treats all alerts as fatal except
    /// `close_notify` and `user_canceled`.
    Warning = 1,
    /// Fatal alert; the connection must be closed.
    Fatal = 2,
}

impl AlertLevel {
    /// Parses an alert level byte.
    pub fn from_u8(v: u8) -> TlsResult<Self> {
        match v {
            1 => Ok(Self::Warning),
            2 => Ok(Self::Fatal),
            _ => Err(TlsError::decode(format!("unknown alert level {v}"))),
        }
    }
}

/// Alert description codes shared by TLS 1.2 and TLS 1.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AlertDescription {
    /// Clean closure notification.
    CloseNotify = 0,
    /// Inappropriate message.
    UnexpectedMessage = 10,
    /// Record MAC / AEAD failure.
    BadRecordMac = 20,
    /// Record overflow.
    RecordOverflow = 22,
    /// Decompression failure (legacy; not used with null compression).
    DecompressionFailure = 30,
    /// Handshake failure.
    HandshakeFailure = 40,
    /// Bad certificate.
    BadCertificate = 42,
    /// Unsupported certificate.
    UnsupportedCertificate = 43,
    /// Certificate revoked.
    CertificateRevoked = 44,
    /// Certificate expired.
    CertificateExpired = 45,
    /// Other certificate problem.
    CertificateUnknown = 46,
    /// Illegal parameter.
    IllegalParameter = 47,
    /// Unknown certificate authority.
    UnknownCa = 48,
    /// Access denied.
    AccessDenied = 49,
    /// Decode error.
    DecodeError = 50,
    /// Decrypt error (Finished / signature).
    DecryptError = 51,
    /// Protocol version not supported.
    ProtocolVersion = 70,
    /// Insufficient security.
    InsufficientSecurity = 71,
    /// Internal error.
    InternalError = 80,
    /// Inappropriate fallback (RFC 7507).
    InappropriateFallback = 86,
    /// User canceled.
    UserCanceled = 90,
    /// Missing extension (TLS 1.3).
    MissingExtension = 109,
    /// Unsupported extension.
    UnsupportedExtension = 110,
    /// Unrecognized name (SNI).
    UnrecognizedName = 112,
    /// Bad certificate status response.
    BadCertificateStatusResponse = 113,
    /// Unknown PSK identity.
    UnknownPskIdentity = 115,
    /// Certificate required.
    CertificateRequired = 116,
    /// No application protocol (ALPN).
    NoApplicationProtocol = 120,
}

impl AlertDescription {
    /// Parses an alert description byte.
    pub fn from_u8(v: u8) -> TlsResult<Self> {
        let desc = match v {
            0 => Self::CloseNotify,
            10 => Self::UnexpectedMessage,
            20 => Self::BadRecordMac,
            22 => Self::RecordOverflow,
            30 => Self::DecompressionFailure,
            40 => Self::HandshakeFailure,
            42 => Self::BadCertificate,
            43 => Self::UnsupportedCertificate,
            44 => Self::CertificateRevoked,
            45 => Self::CertificateExpired,
            46 => Self::CertificateUnknown,
            47 => Self::IllegalParameter,
            48 => Self::UnknownCa,
            49 => Self::AccessDenied,
            50 => Self::DecodeError,
            51 => Self::DecryptError,
            70 => Self::ProtocolVersion,
            71 => Self::InsufficientSecurity,
            80 => Self::InternalError,
            86 => Self::InappropriateFallback,
            90 => Self::UserCanceled,
            109 => Self::MissingExtension,
            110 => Self::UnsupportedExtension,
            112 => Self::UnrecognizedName,
            113 => Self::BadCertificateStatusResponse,
            115 => Self::UnknownPskIdentity,
            116 => Self::CertificateRequired,
            120 => Self::NoApplicationProtocol,
            other => {
                return Err(TlsError::decode(format!(
                    "unknown alert description {other}"
                )));
            }
        };
        Ok(desc)
    }

    /// Wire encoding.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Two-byte alert payload: level || description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Alert {
    /// Severity.
    pub level: AlertLevel,
    /// Description.
    pub description: AlertDescription,
}

impl Alert {
    /// Builds a fatal alert.
    #[must_use]
    pub const fn fatal(description: AlertDescription) -> Self {
        Self {
            level: AlertLevel::Fatal,
            description,
        }
    }

    /// Builds a warning alert (used for `close_notify`).
    #[must_use]
    pub const fn warning(description: AlertDescription) -> Self {
        Self {
            level: AlertLevel::Warning,
            description,
        }
    }

    /// Encodes the alert payload.
    #[must_use]
    pub fn encode(self) -> [u8; 2] {
        [self.level as u8, self.description.as_u8()]
    }

    /// Decodes an alert payload.
    pub fn decode(data: &[u8]) -> TlsResult<Self> {
        if data.len() != 2 {
            return Err(TlsError::decode("alert must be exactly 2 bytes"));
        }
        Ok(Self {
            level: AlertLevel::from_u8(data[0])?,
            description: AlertDescription::from_u8(data[1])?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_descriptions() -> [AlertDescription; 28] {
        [
            AlertDescription::CloseNotify,
            AlertDescription::UnexpectedMessage,
            AlertDescription::BadRecordMac,
            AlertDescription::RecordOverflow,
            AlertDescription::DecompressionFailure,
            AlertDescription::HandshakeFailure,
            AlertDescription::BadCertificate,
            AlertDescription::UnsupportedCertificate,
            AlertDescription::CertificateRevoked,
            AlertDescription::CertificateExpired,
            AlertDescription::CertificateUnknown,
            AlertDescription::IllegalParameter,
            AlertDescription::UnknownCa,
            AlertDescription::AccessDenied,
            AlertDescription::DecodeError,
            AlertDescription::DecryptError,
            AlertDescription::ProtocolVersion,
            AlertDescription::InsufficientSecurity,
            AlertDescription::InternalError,
            AlertDescription::InappropriateFallback,
            AlertDescription::UserCanceled,
            AlertDescription::MissingExtension,
            AlertDescription::UnsupportedExtension,
            AlertDescription::UnrecognizedName,
            AlertDescription::BadCertificateStatusResponse,
            AlertDescription::UnknownPskIdentity,
            AlertDescription::CertificateRequired,
            AlertDescription::NoApplicationProtocol,
        ]
    }

    #[test]
    fn alert_level_parses_valid_bytes() {
        assert_eq!(AlertLevel::from_u8(1).unwrap(), AlertLevel::Warning);
        assert_eq!(AlertLevel::from_u8(2).unwrap(), AlertLevel::Fatal);
        assert_eq!(AlertLevel::Warning as u8, 1);
        assert_eq!(AlertLevel::Fatal as u8, 2);
    }

    #[test]
    fn alert_level_rejects_invalid_bytes() {
        for bad in [0u8, 3, 4, 127, 128, 255] {
            let err = AlertLevel::from_u8(bad).unwrap_err();
            assert!(matches!(err, TlsError::Decode(_)), "bad={bad}");
            assert!(err.to_string().contains(&format!("alert level {bad}")));
        }
    }

    #[test]
    fn alert_description_round_trips_every_variant() {
        let mut seen = std::collections::HashSet::new();
        for desc in all_descriptions() {
            assert_eq!(AlertDescription::from_u8(desc.as_u8()).unwrap(), desc);
            assert!(seen.insert(desc.as_u8()), "dup for {desc:?}");
        }
        assert_eq!(AlertDescription::CloseNotify.as_u8(), 0);
        assert_eq!(AlertDescription::NoApplicationProtocol.as_u8(), 120);
    }

    #[test]
    fn alert_description_rejects_invalid_bytes() {
        for bad in [1u8, 9, 11, 21, 41, 69, 85, 108, 121, 127, 255] {
            let err = AlertDescription::from_u8(bad).unwrap_err();
            assert!(matches!(err, TlsError::Decode(_)), "bad={bad}");
            assert!(
                err.to_string()
                    .contains(&format!("alert description {bad}"))
            );
        }
    }

    #[test]
    fn fatal_and_warning_constructors_set_level() {
        let fatal = Alert::fatal(AlertDescription::HandshakeFailure);
        assert_eq!(fatal.level, AlertLevel::Fatal);
        assert_eq!(fatal.description, AlertDescription::HandshakeFailure);

        let warning = Alert::warning(AlertDescription::CloseNotify);
        assert_eq!(warning.level, AlertLevel::Warning);
        assert_eq!(warning.description, AlertDescription::CloseNotify);
    }

    #[test]
    fn alert_encode_produces_level_then_description() {
        assert_eq!(
            Alert::fatal(AlertDescription::DecodeError).encode(),
            [2, 50]
        );
        assert_eq!(
            Alert::warning(AlertDescription::CloseNotify).encode(),
            [1, 0]
        );
        assert_eq!(
            Alert::fatal(AlertDescription::NoApplicationProtocol).encode(),
            [2, 120]
        );
    }

    #[test]
    fn alert_round_trips_through_encode_decode() {
        for desc in all_descriptions() {
            for level in [AlertLevel::Warning, AlertLevel::Fatal] {
                let original = Alert {
                    level,
                    description: desc,
                };
                let wire = original.encode();
                let decoded = Alert::decode(&wire).unwrap();
                assert_eq!(decoded, original);
                assert_eq!(decoded.encode(), wire);
            }
        }
    }

    #[test]
    fn alert_decode_rejects_wrong_lengths() {
        for len in [0usize, 1, 3, 4] {
            let data = vec![1u8; len];
            let err = Alert::decode(&data).unwrap_err();
            assert!(matches!(err, TlsError::Decode(_)), "len={len}");
            assert!(
                err.to_string().contains("alert must be exactly 2 bytes"),
                "len={len}"
            );
        }
    }

    #[test]
    fn alert_decode_rejects_invalid_level_and_description() {
        let mut bad_level = Alert::fatal(AlertDescription::CloseNotify).encode();
        bad_level[0] = 0;
        assert!(Alert::decode(&bad_level).is_err());

        let mut bad_desc = Alert::fatal(AlertDescription::CloseNotify).encode();
        bad_desc[1] = 3;
        assert!(Alert::decode(&bad_desc).is_err());

        let both_bad = [0u8, 0];
        assert!(Alert::decode(&both_bad).is_err());
    }

    #[test]
    fn truncated_single_byte_alert_is_rejected() {
        let err = Alert::decode(&[2]).unwrap_err();
        assert!(matches!(err, TlsError::Decode(_)));
        assert!(err.to_string().contains("exactly 2 bytes"));
    }
}
