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

pub mod aead;
pub mod alert;
pub mod cert;
pub mod client;
pub mod codec;
pub mod connection;
pub mod connection_conf;
pub mod crypto_aes_gcm;
pub mod crypto_chacha20poly1305;
pub mod crypto_ct;
pub mod crypto_device;
pub mod crypto_hash;
pub mod crypto_random;
pub mod error;
pub mod extensions;
pub mod handshake;
pub mod hkdf;
pub mod ids;
pub mod key_schedule;
pub mod kx;
pub mod prf;
pub mod record;
pub mod server;
pub mod sign;
pub mod stream;
pub mod transcript;

pub use alert::{Alert, AlertDescription, AlertLevel};
pub use cert::{CertVerifier, LeafKeyKind, ParsedCert, RootCertStore, ServerName, parse_pem_certs};
pub use client::TlsClientConnection;
pub use connection::{ConnectionState, IoState};
pub use connection_conf::{
    CertifiedKey, ClientConfig, ClientConfigBuilder, ServerConfig, ServerConfigBuilder,
};
pub use error::{TlsError, TlsResult};
pub use ids::{
    AeadAlgorithm, CipherSuite, ContentType, ExtensionType, HandshakeType, HashAlgorithm, KeyUpdateRequest,
    NamedGroup, ProtocolVersion, PskKeyExchangeMode, SignatureScheme,
};
pub use server::TlsServerConnection;
pub use stream::{TlsSession, TlsStream};
