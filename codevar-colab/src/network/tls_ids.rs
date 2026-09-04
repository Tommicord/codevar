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

//! Numeric identifiers for versions, cipher suites, groups, and signatures.

use crate::network::tls_error::{TlsError, TlsResult};

/// TLS protocol version wire values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u16)]
pub enum ProtocolVersion {
    /// TLS 1.0 (0x0301) — rejected by this stack.
    Tls10 = 0x0301,
    /// TLS 1.1 (0x0302) — rejected by this stack.
    Tls11 = 0x0302,
    /// TLS 1.2 (RFC 5246).
    Tls12 = 0x0303,
    /// TLS 1.3 (RFC 8446).
    Tls13 = 0x0304,
}

impl ProtocolVersion {
    /// Parses a version from two big-endian bytes.
    pub fn from_u16(v: u16) -> TlsResult<Self> {
        match v {
            0x0301 => Ok(Self::Tls10),
            0x0302 => Ok(Self::Tls11),
            0x0303 => Ok(Self::Tls12),
            0x0304 => Ok(Self::Tls13),
            other => Err(TlsError::Unsupported(format!(
                "unknown protocol version 0x{other:04x}"
            ))),
        }
    }

    /// Wire value.
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }

    /// Encodes as two big-endian bytes.
    #[must_use]
    pub const fn to_be_bytes(self) -> [u8; 2] {
        self.as_u16().to_be_bytes()
    }
}

/// Content types for the TLS record layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContentType {
    /// Change cipher spec (TLS 1.2; ignored as middlebox CCS in TLS 1.3).
    ChangeCipherSpec = 20,
    /// Alert.
    Alert = 21,
    /// Handshake.
    Handshake = 22,
    /// Application data (also TLS 1.3 outer ciphertext type).
    ApplicationData = 23,
}

impl ContentType {
    /// Parses a content type byte.
    pub fn from_u8(v: u8) -> TlsResult<Self> {
        match v {
            20 => Ok(Self::ChangeCipherSpec),
            21 => Ok(Self::Alert),
            22 => Ok(Self::Handshake),
            23 => Ok(Self::ApplicationData),
            other => Err(TlsError::decode(format!("unknown content type {other}"))),
        }
    }

    /// Wire value.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Handshake message types (RFC 8446 §4 / RFC 5246 §7.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HandshakeType {
    /// HelloRequest (TLS 1.2 only).
    HelloRequest = 0,
    /// ClientHello.
    ClientHello = 1,
    /// ServerHello / HelloRetryRequest.
    ServerHello = 2,
    /// NewSessionTicket.
    NewSessionTicket = 4,
    /// EndOfEarlyData (TLS 1.3).
    EndOfEarlyData = 5,
    /// EncryptedExtensions (TLS 1.3).
    EncryptedExtensions = 8,
    /// Certificate.
    Certificate = 11,
    /// ServerKeyExchange (TLS 1.2).
    ServerKeyExchange = 12,
    /// CertificateRequest.
    CertificateRequest = 13,
    /// ServerHelloDone (TLS 1.2).
    ServerHelloDone = 14,
    /// CertificateVerify.
    CertificateVerify = 15,
    /// ClientKeyExchange (TLS 1.2).
    ClientKeyExchange = 16,
    /// Finished.
    Finished = 20,
    /// KeyUpdate (TLS 1.3).
    KeyUpdate = 24,
    /// Message hash (synthetic HRR transcript, TLS 1.3 §4.4.1).
    MessageHash = 254,
}

impl HandshakeType {
    /// Parses a handshake type byte.
    pub fn from_u8(v: u8) -> TlsResult<Self> {
        match v {
            0 => Ok(Self::HelloRequest),
            1 => Ok(Self::ClientHello),
            2 => Ok(Self::ServerHello),
            4 => Ok(Self::NewSessionTicket),
            5 => Ok(Self::EndOfEarlyData),
            8 => Ok(Self::EncryptedExtensions),
            11 => Ok(Self::Certificate),
            12 => Ok(Self::ServerKeyExchange),
            13 => Ok(Self::CertificateRequest),
            14 => Ok(Self::ServerHelloDone),
            15 => Ok(Self::CertificateVerify),
            16 => Ok(Self::ClientKeyExchange),
            20 => Ok(Self::Finished),
            24 => Ok(Self::KeyUpdate),
            254 => Ok(Self::MessageHash),
            other => Err(TlsError::decode(format!("unknown handshake type {other}"))),
        }
    }

    /// Wire value.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Extension types (RFC 8446 §4.2, RFC 5246 / related RFCs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum ExtensionType {
    /// Server name indication (RFC 6066).
    ServerName = 0,
    /// Supported groups / elliptic curves.
    SupportedGroups = 10,
    /// EC point formats (TLS 1.2).
    EcPointFormats = 11,
    /// Signature algorithms.
    SignatureAlgorithms = 13,
    /// Application-layer protocol negotiation (RFC 7301).
    ApplicationLayerProtocolNegotiation = 16,
    /// Status request (OCSP).
    StatusRequest = 5,
    /// Signed certificate timestamp.
    SignedCertificateTimestamp = 18,
    /// Extended master secret (RFC 7627).
    ExtendedMasterSecret = 23,
    /// Session ticket (RFC 5077).
    SessionTicket = 35,
    /// Pre-shared key (TLS 1.3).
    PreSharedKey = 41,
    /// Early data (TLS 1.3).
    EarlyData = 42,
    /// Supported versions (TLS 1.3).
    SupportedVersions = 43,
    /// Cookie (TLS 1.3).
    Cookie = 44,
    /// PSK key exchange modes (TLS 1.3).
    PskKeyExchangeModes = 45,
    /// Certificate authorities.
    CertificateAuthorities = 47,
    /// OID filters.
    OidFilters = 48,
    /// Post-handshake auth.
    PostHandshakeAuth = 49,
    /// Signature algorithms for certificates.
    SignatureAlgorithmsCert = 50,
    /// Key share (TLS 1.3).
    KeyShare = 51,
    /// Renegotiation info (RFC 5746).
    RenegotiationInfo = 0xff01,
}

impl ExtensionType {
    /// Parses an extension type.
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0 => Some(Self::ServerName),
            5 => Some(Self::StatusRequest),
            10 => Some(Self::SupportedGroups),
            11 => Some(Self::EcPointFormats),
            13 => Some(Self::SignatureAlgorithms),
            16 => Some(Self::ApplicationLayerProtocolNegotiation),
            18 => Some(Self::SignedCertificateTimestamp),
            23 => Some(Self::ExtendedMasterSecret),
            35 => Some(Self::SessionTicket),
            41 => Some(Self::PreSharedKey),
            42 => Some(Self::EarlyData),
            43 => Some(Self::SupportedVersions),
            44 => Some(Self::Cookie),
            45 => Some(Self::PskKeyExchangeModes),
            47 => Some(Self::CertificateAuthorities),
            48 => Some(Self::OidFilters),
            49 => Some(Self::PostHandshakeAuth),
            50 => Some(Self::SignatureAlgorithmsCert),
            51 => Some(Self::KeyShare),
            0xff01 => Some(Self::RenegotiationInfo),
            _ => None,
        }
    }

    /// Wire value.
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }
}

/// AEAD + HKDF-hash cipher suites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum CipherSuite {
    /// TLS 1.3 AES-128-GCM with SHA-256.
    TlsAes128GcmSha256 = 0x1301,
    /// TLS 1.3 AES-256-GCM with SHA-384.
    TlsAes256GcmSha384 = 0x1302,
    /// TLS 1.3 ChaCha20-Poly1305 with SHA-256.
    TlsChacha20Poly1305Sha256 = 0x1303,
    /// TLS 1.2 ECDHE-ECDSA AES-128-GCM SHA-256.
    TlsEcdheEcdsaWithAes128GcmSha256 = 0xc02b,
    /// TLS 1.2 ECDHE-ECDSA AES-256-GCM SHA-384.
    TlsEcdheEcdsaWithAes256GcmSha384 = 0xc02c,
    /// TLS 1.2 ECDHE-RSA AES-128-GCM SHA-256.
    TlsEcdheRsaWithAes128GcmSha256 = 0xc02f,
    /// TLS 1.2 ECDHE-RSA AES-256-GCM SHA-384.
    TlsEcdheRsaWithAes256GcmSha384 = 0xc030,
    /// TLS 1.2 ECDHE-RSA ChaCha20-Poly1305 SHA-256.
    TlsEcdheRsaWithChacha20Poly1305Sha256 = 0xcca8,
    /// TLS 1.2 ECDHE-ECDSA ChaCha20-Poly1305 SHA-256.
    TlsEcdheEcdsaWithChacha20Poly1305Sha256 = 0xcca9,
}

impl CipherSuite {
    /// Parses a cipher suite code point.
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0x1301 => Some(Self::TlsAes128GcmSha256),
            0x1302 => Some(Self::TlsAes256GcmSha384),
            0x1303 => Some(Self::TlsChacha20Poly1305Sha256),
            0xc02b => Some(Self::TlsEcdheEcdsaWithAes128GcmSha256),
            0xc02c => Some(Self::TlsEcdheEcdsaWithAes256GcmSha384),
            0xc02f => Some(Self::TlsEcdheRsaWithAes128GcmSha256),
            0xc030 => Some(Self::TlsEcdheRsaWithAes256GcmSha384),
            0xcca8 => Some(Self::TlsEcdheRsaWithChacha20Poly1305Sha256),
            0xcca9 => Some(Self::TlsEcdheEcdsaWithChacha20Poly1305Sha256),
            _ => None,
        }
    }

    /// Wire value.
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }

    /// Whether this is a TLS 1.3-only suite.
    #[must_use]
    pub const fn is_tls13(self) -> bool {
        matches!(
            self,
            Self::TlsAes128GcmSha256
                | Self::TlsAes256GcmSha384
                | Self::TlsChacha20Poly1305Sha256
        )
    }

    /// Whether this is a TLS 1.2 suite.
    #[must_use]
    pub const fn is_tls12(self) -> bool {
        !self.is_tls13()
    }

    /// Hash algorithm used by the suite (HKDF / PRF).
    #[must_use]
    pub const fn hash_algorithm(self) -> HashAlgorithm {
        match self {
            Self::TlsAes256GcmSha384
            | Self::TlsEcdheEcdsaWithAes256GcmSha384
            | Self::TlsEcdheRsaWithAes256GcmSha384 => HashAlgorithm::Sha384,
            _ => HashAlgorithm::Sha256,
        }
    }

    /// AEAD algorithm for the suite.
    #[must_use]
    pub const fn aead_algorithm(self) -> AeadAlgorithm {
        match self {
            Self::TlsAes128GcmSha256
            | Self::TlsEcdheEcdsaWithAes128GcmSha256
            | Self::TlsEcdheRsaWithAes128GcmSha256 => AeadAlgorithm::Aes128Gcm,
            Self::TlsAes256GcmSha384
            | Self::TlsEcdheEcdsaWithAes256GcmSha384
            | Self::TlsEcdheRsaWithAes256GcmSha384 => AeadAlgorithm::Aes256Gcm,
            Self::TlsChacha20Poly1305Sha256
            | Self::TlsEcdheRsaWithChacha20Poly1305Sha256
            | Self::TlsEcdheEcdsaWithChacha20Poly1305Sha256 => {
                AeadAlgorithm::ChaCha20Poly1305
            }
        }
    }

    /// Whether the TLS 1.2 suite authenticates with an ECDSA certificate.
    #[must_use]
    pub const fn tls12_uses_ecdsa(self) -> bool {
        matches!(
            self,
            Self::TlsEcdheEcdsaWithAes128GcmSha256
                | Self::TlsEcdheEcdsaWithAes256GcmSha384
                | Self::TlsEcdheEcdsaWithChacha20Poly1305Sha256
        )
    }

    /// Whether the TLS 1.2 suite authenticates with an RSA certificate.
    #[must_use]
    pub const fn tls12_uses_rsa(self) -> bool {
        matches!(
            self,
            Self::TlsEcdheRsaWithAes128GcmSha256
                | Self::TlsEcdheRsaWithAes256GcmSha384
                | Self::TlsEcdheRsaWithChacha20Poly1305Sha256
        )
    }

    /// Default preference order offered by clients.
    ///
    /// Order is device-aware: AES-256-GCM first when AES-NI / ARM crypto is
    /// available, otherwise ChaCha20-Poly1305 first (mobile / WASM / software).
    #[must_use]
    pub fn default_offered() -> &'static [CipherSuite] {
        crate::network::tls_crypto_device::preferred_cipher_suites()
    }
}

/// Hash algorithm used by cipher suites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    /// SHA-256 (32-byte output).
    Sha256,
    /// SHA-384 (48-byte output).
    Sha384,
}

impl HashAlgorithm {
    /// Digest output length in bytes.
    #[must_use]
    pub const fn output_len(self) -> usize {
        match self {
            Self::Sha256 => 32,
            Self::Sha384 => 48,
        }
    }
}

/// AEAD algorithms used by supported cipher suites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AeadAlgorithm {
    /// AES-128-GCM (16-byte key, 12-byte IV, 16-byte tag).
    Aes128Gcm,
    /// AES-256-GCM (32-byte key, 12-byte IV, 16-byte tag).
    Aes256Gcm,
    /// ChaCha20-Poly1305 (32-byte key, 12-byte IV, 16-byte tag).
    ChaCha20Poly1305,
}

impl AeadAlgorithm {
    /// Trafic key length in bytes.
    #[must_use]
    pub const fn key_len(self) -> usize {
        match self {
            Self::Aes128Gcm => 16,
            Self::Aes256Gcm | Self::ChaCha20Poly1305 => 32,
        }
    }

    /// Nonce / IV length in bytes (always 12 for these AEADs).
    #[must_use]
    pub const fn iv_len(self) -> usize {
        12
    }

    /// Authentication tag length in bytes.
    #[must_use]
    pub const fn tag_len(self) -> usize {
        16
    }

    /// TLS 1.2 fixed IV length for GCM (RFC 5288); ChaCha20 uses full 12-byte IV.
    #[must_use]
    pub const fn tls12_fixed_iv_len(self) -> usize {
        match self {
            Self::Aes128Gcm | Self::Aes256Gcm => 4,
            Self::ChaCha20Poly1305 => 12,
        }
    }

    /// TLS 1.2 explicit nonce length placed in the record (8 for GCM, 0 for ChaCha20).
    #[must_use]
    pub const fn tls12_explicit_nonce_len(self) -> usize {
        match self {
            Self::Aes128Gcm | Self::Aes256Gcm => 8,
            Self::ChaCha20Poly1305 => 0,
        }
    }
}

/// Named groups / curves (RFC 8446 §4.2.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum NamedGroup {
    /// secp256r1 (NIST P-256).
    Secp256r1 = 0x0017,
    /// secp384r1 (NIST P-384).
    Secp384r1 = 0x0018,
    /// X25519.
    X25519 = 0x001d,
}

impl NamedGroup {
    /// Parses a named group.
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0x0017 => Some(Self::Secp256r1),
            0x0018 => Some(Self::Secp384r1),
            0x001d => Some(Self::X25519),
            _ => None,
        }
    }

    /// Wire value.
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }

    /// Default preference order.
    #[must_use]
    pub fn default_offered() -> &'static [NamedGroup] {
        &DEFAULT_GROUPS
    }
}

const DEFAULT_GROUPS: [NamedGroup; 3] = [
    NamedGroup::X25519,
    NamedGroup::Secp256r1,
    NamedGroup::Secp384r1,
];

/// Signature schemes (RFC 8446 §4.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum SignatureScheme {
    /// RSASSA-PKCS1-v1_5 with SHA-256 (TLS 1.2 certificates / signatures).
    RsaPkcs1Sha256 = 0x0401,
    /// RSASSA-PKCS1-v1_5 with SHA-384.
    RsaPkcs1Sha384 = 0x0501,
    /// ECDSA P-256 with SHA-256.
    EcdsaSecp256r1Sha256 = 0x0403,
    /// ECDSA P-384 with SHA-384.
    EcdsaSecp384r1Sha384 = 0x0503,
    /// RSASSA-PSS with SHA-256 (rsaEncryption OID).
    RsaPssRsaeSha256 = 0x0804,
    /// RSASSA-PSS with SHA-384 (rsaEncryption OID).
    RsaPssRsaeSha384 = 0x0805,
    /// Ed25519.
    Ed25519 = 0x0807,
}

impl SignatureScheme {
    /// Parses a signature scheme.
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0x0401 => Some(Self::RsaPkcs1Sha256),
            0x0501 => Some(Self::RsaPkcs1Sha384),
            0x0403 => Some(Self::EcdsaSecp256r1Sha256),
            0x0503 => Some(Self::EcdsaSecp384r1Sha384),
            0x0804 => Some(Self::RsaPssRsaeSha256),
            0x0805 => Some(Self::RsaPssRsaeSha384),
            0x0807 => Some(Self::Ed25519),
            _ => None,
        }
    }

    /// Wire value.
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }

    /// Whether this scheme may be used for TLS 1.3 CertificateVerify.
    #[must_use]
    pub const fn allowed_in_tls13_cert_verify(self) -> bool {
        !matches!(self, Self::RsaPkcs1Sha256 | Self::RsaPkcs1Sha384)
    }

    /// Default preference order.
    #[must_use]
    pub fn default_offered() -> &'static [SignatureScheme] {
        &DEFAULT_SIGNATURE_SCHEMES
    }
}

const DEFAULT_SIGNATURE_SCHEMES: [SignatureScheme; 7] = [
    SignatureScheme::EcdsaSecp256r1Sha256,
    SignatureScheme::EcdsaSecp384r1Sha384,
    SignatureScheme::Ed25519,
    SignatureScheme::RsaPssRsaeSha256,
    SignatureScheme::RsaPssRsaeSha384,
    SignatureScheme::RsaPkcs1Sha256,
    SignatureScheme::RsaPkcs1Sha384,
];

/// PSK key exchange modes (TLS 1.3 §4.2.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PskKeyExchangeMode {
    /// PSK-only key exchange.
    PskKe = 0,
    /// PSK with (EC)DHE.
    PskDheKe = 1,
}

impl PskKeyExchangeMode {
    /// Parses a PSK KE mode.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::PskKe),
            1 => Some(Self::PskDheKe),
            _ => None,
        }
    }
}

/// KeyUpdate request values (TLS 1.3 §4.6.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KeyUpdateRequest {
    /// Update keys but do not request a reciprocal update.
    UpdateNotRequested = 0,
    /// Update keys and request the peer to update as well.
    UpdateRequested = 1,
}

impl KeyUpdateRequest {
    /// Parses a KeyUpdate request.
    pub fn from_u8(v: u8) -> TlsResult<Self> {
        match v {
            0 => Ok(Self::UpdateNotRequested),
            1 => Ok(Self::UpdateRequested),
            other => Err(TlsError::decode(format!(
                "invalid KeyUpdate request {other}"
            ))),
        }
    }
}

/// TLS 1.3 HelloRetryRequest random value: SHA-256("HelloRetryRequest").
pub const HELLO_RETRY_REQUEST_RANDOM: [u8; 32] = [
    0xCF, 0x21, 0xAD, 0x74, 0xE5, 0x9A, 0x61, 0x11, 0xBE, 0x1D, 0x8C, 0x02, 0x1E, 0x65,
    0xB8, 0x91, 0xC2, 0xA2, 0x11, 0x16, 0x7A, 0xBB, 0x8C, 0x5E, 0x07, 0x9E, 0x09, 0xE2,
    0xC8, 0xA8, 0x33, 0x9C,
];

/// Downgrade sentinel placed in ServerHello.random when a TLS 1.3 server
/// negotiates TLS 1.2 (`DOWNGRD` || 0x01).
pub const DOWNGRADE_TLS12_SENTINEL: [u8; 8] =
    [0x44, 0x4F, 0x57, 0x4E, 0x47, 0x52, 0x44, 0x01];

/// Downgrade sentinel for TLS 1.1 or below (`DOWNGRD` || 0x00).
pub const DOWNGRADE_TLS11_SENTINEL: [u8; 8] =
    [0x44, 0x4F, 0x57, 0x4E, 0x47, 0x52, 0x44, 0x00];

/// Maximum plaintext fragment length (2^14).
pub const MAX_FRAGMENT_LENGTH: usize = 16384;

/// Maximum TLS 1.3 ciphertext expansion (AEAD tag + inner content type + padding bound).
pub const MAX_TLS13_CIPHERTEXT_OVERHEAD: usize = 256;

/// Maximum ciphertext record payload accepted by the record layer.
pub const MAX_CIPHERTEXT_LENGTH: usize =
    MAX_FRAGMENT_LENGTH + MAX_TLS13_CIPHERTEXT_OVERHEAD;
