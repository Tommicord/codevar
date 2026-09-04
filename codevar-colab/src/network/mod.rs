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
pub mod ws_base64;
pub mod ws_client;
pub mod ws_connection;
pub mod ws_error;
pub mod ws_frame;
pub mod ws_handshake;
pub mod ws_ids;
pub mod ws_message;
pub mod ws_server;
pub mod ws_stream;
pub mod ws_utf8;

pub use crate::network::ws_client::ClientConnection as WsClientConnection;
pub use crate::network::ws_connection::{
    ConnectionConfig as WsConnectionConfig, ConnectionState as WsConnectionState,
    IoState as WsIoState,
};
pub use crate::network::ws_error::{WsError, WsResult};
pub use crate::network::ws_frame::{WsFrame, WsFrameHeader};
pub use crate::network::ws_handshake::{
    HandshakeRequest, WsClientHandshake, WsHandshakeResponse, WsServerHandshake,
    accept_key_from_nonce, compute_accept_key, generate_key_nonce,
};
pub use crate::network::ws_ids::{
    DEFAULT_MAX_FRAME_SIZE, DEFAULT_MAX_MESSAGE_SIZE, GUID, Role, VERSION, WsCloseCode,
    WsOpcode,
};
pub use crate::network::ws_message::WsMessage;
pub use crate::network::ws_server::ServerConnection as WsServerConnection;
pub use crate::network::ws_stream::WebSocketStream as WsWebSocketStream;

pub use tls_alert::{Alert, AlertDescription, AlertLevel};
pub use tls_cert::{
    CertVerifier, LeafKeyKind, ParsedCert, RootCertStore, ServerName, parse_pem_certs,
};
pub use tls_client::TlsClientConnection;
pub use tls_connection::{ConnectionState, IoState};
pub use tls_connection_conf::{
    CertifiedKey, ClientConfig, ClientConfigBuilder, ServerConfig, ServerConfigBuilder,
};
pub use tls_error::{TlsError, TlsResult};
pub use tls_ids::{
    AeadAlgorithm, CipherSuite, ContentType, ExtensionType, HandshakeType, HashAlgorithm,
    KeyUpdateRequest, NamedGroup, ProtocolVersion, PskKeyExchangeMode, SignatureScheme,
};
pub use tls_server::TlsServerConnection;
pub use tls_stream::{TlsSession, TlsStream};
