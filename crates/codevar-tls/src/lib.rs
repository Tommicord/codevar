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

pub mod tls_aead;
pub mod tls_alert;
pub mod tls_cert;
pub mod tls_client;
pub mod tls_codec;
pub mod tls_connection;
pub mod tls_connection_conf;
pub mod tls_crypto_aes_gcm;
pub mod tls_crypto_chacha20poly1305;
pub mod tls_crypto_device;
pub mod tls_crypto_hash;
pub mod tls_crypto_random;
pub mod tls_error;
pub mod tls_extensions;
pub mod tls_handshake;
pub mod tls_hkdf;
pub mod tls_ids;
pub mod tls_key_schedule;
pub mod tls_kx;
pub mod tls_prf;
pub mod tls_record;
pub mod tls_server;
pub mod tls_sign;
pub mod tls_stream;
pub mod tls_transcript;

pub use tls_alert::{Alert, AlertDescription, AlertLevel};
pub use tls_cert::{CertVerifier, LeafKeyKind, ParsedCert, RootCertStore, ServerName, parse_pem_certs};
pub use tls_client::TlsClientConnection;
pub use tls_connection::{ConnectionState, IoState};
pub use tls_connection_conf::{
    CertifiedKey, ClientConfig, ClientConfigBuilder, ServerConfig, ServerConfigBuilder,
};
pub use tls_error::{TlsError, TlsResult};
pub use tls_ids::{
    AeadAlgorithm, CipherSuite, ContentType, ExtensionType, HandshakeType, HashAlgorithm, KeyUpdateRequest,
    NamedGroup, ProtocolVersion, PskKeyExchangeMode, SignatureScheme,
};
pub use tls_server::TlsServerConnection;
pub use tls_stream::{TlsSession, TlsStream};
