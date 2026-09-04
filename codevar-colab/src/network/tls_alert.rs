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

//! TLS alert protocol (RFC 8446 §6, RFC 5246 §7.2).

use crate::network::tls_error::{TlsError, TlsResult};

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
