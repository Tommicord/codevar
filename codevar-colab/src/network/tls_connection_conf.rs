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

//! TLS client and server configuration.

use crate::network::tls_cert::{CertVerifier, RootCertStore, parse_pem_certs};
use crate::network::tls_error::{TlsError, TlsResult};
use crate::network::tls_ids::{
    CipherSuite, NamedGroup, ProtocolVersion, SignatureScheme,
};
use crate::network::tls_sign::PrivateKey;
use std::sync::Arc;

/// Shared client configuration.
#[derive(Clone)]
pub struct ClientConfig {
    /// Enabled protocol versions (preference order).
    pub versions: Vec<ProtocolVersion>,
    /// Offered cipher suites (preference order).
    pub cipher_suites: Vec<CipherSuite>,
    /// Offered named groups.
    pub named_groups: Vec<NamedGroup>,
    /// Offered signature algorithms.
    pub signature_schemes: Vec<SignatureScheme>,
    /// ALPN protocols (preference order).
    pub alpn_protocols: Vec<Vec<u8>>,
    /// Certificate verifier.
    pub verifier: Arc<CertVerifier>,
    /// Whether to require Extended Master Secret for TLS 1.2.
    pub require_ems: bool,
}

impl ClientConfig {
    /// Builds a client config trusting the provided root store.
    #[must_use]
    pub fn builder() -> ClientConfigBuilder {
        ClientConfigBuilder::default()
    }

    /// Convenience: insecure verifier (no certificate checks). For tests only.
    #[must_use]
    pub fn dangerous_insecure() -> Arc<Self> {
        Arc::new(Self {
            versions: vec![ProtocolVersion::Tls13, ProtocolVersion::Tls12],
            cipher_suites: CipherSuite::default_offered().to_vec(),
            named_groups: NamedGroup::default_offered().to_vec(),
            signature_schemes: SignatureScheme::default_offered().to_vec(),
            alpn_protocols: Vec::new(),
            verifier: Arc::new(CertVerifier::dangerous_insecure()),
            require_ems: true,
        })
    }
}

/// Builder for [`ClientConfig`].
#[derive(Default)]
pub struct ClientConfigBuilder {
    roots: RootCertStore,
    alpn: Vec<Vec<u8>>,
    versions: Option<Vec<ProtocolVersion>>,
    skip_hostname: bool,
    insecure: bool,
}

impl ClientConfigBuilder {
    /// Adds PEM trust anchors.
    pub fn with_root_pem(mut self, pem: &[u8]) -> TlsResult<Self> {
        self.roots.add_pem(pem)?;
        Ok(self)
    }

    /// Replaces the root store.
    #[must_use]
    pub fn with_root_store(mut self, roots: RootCertStore) -> Self {
        self.roots = roots;
        self
    }

    /// Sets ALPN protocols.
    #[must_use]
    pub fn with_alpn(mut self, protocols: Vec<Vec<u8>>) -> Self {
        self.alpn = protocols;
        self
    }

    /// Restricts enabled versions.
    #[must_use]
    pub fn with_versions(mut self, versions: Vec<ProtocolVersion>) -> Self {
        self.versions = Some(versions);
        self
    }

    /// Skips hostname verification.
    #[must_use]
    pub fn dangerous_skip_hostname(mut self) -> Self {
        self.skip_hostname = true;
        self
    }

    /// Disables all certificate validation (tests only).
    #[must_use]
    pub fn dangerous_insecure(mut self) -> Self {
        self.insecure = true;
        self
    }

    /// Builds the config.
    pub fn build(self) -> TlsResult<ClientConfig> {
        let verifier = if self.insecure {
            CertVerifier::dangerous_insecure()
        } else {
            let mut v = CertVerifier::new(self.roots);
            if self.skip_hostname {
                v = v.with_skip_hostname();
            }
            v
        };
        Ok(ClientConfig {
            versions: self
                .versions
                .unwrap_or_else(|| vec![ProtocolVersion::Tls13, ProtocolVersion::Tls12]),
            cipher_suites: CipherSuite::default_offered().to_vec(),
            named_groups: NamedGroup::default_offered().to_vec(),
            signature_schemes: SignatureScheme::default_offered().to_vec(),
            alpn_protocols: self.alpn,
            verifier: Arc::new(verifier),
            require_ems: true,
        })
    }
}

/// A single server certificate chain with private key.
#[derive(Clone)]
pub struct CertifiedKey {
    /// DER certificate chain (leaf first).
    pub cert_chain: Vec<Vec<u8>>,
    /// Private key matching the leaf.
    pub key: Arc<PrivateKey>,
}

impl CertifiedKey {
    /// Loads a PEM certificate chain and PEM private key.
    pub fn from_pem(cert_pem: &[u8], key_pem: &[u8]) -> TlsResult<Self> {
        let cert_chain = parse_pem_certs(cert_pem)?;
        let key = PrivateKey::from_pem(key_pem)?;
        Ok(Self {
            cert_chain,
            key: Arc::new(key),
        })
    }
}

/// Shared server configuration.
#[derive(Clone)]
pub struct ServerConfig {
    /// Enabled protocol versions.
    pub versions: Vec<ProtocolVersion>,
    /// Supported cipher suites.
    pub cipher_suites: Vec<CipherSuite>,
    /// Supported named groups.
    pub named_groups: Vec<NamedGroup>,
    /// Supported signature schemes.
    pub signature_schemes: Vec<SignatureScheme>,
    /// ALPN protocols the server can select.
    pub alpn_protocols: Vec<Vec<u8>>,
    /// Server certificate / key.
    pub certified_key: CertifiedKey,
    /// Whether TLS 1.2 requires Extended Master Secret.
    pub require_ems: bool,
}

impl ServerConfig {
    /// Builds a server config from a certified key.
    pub fn builder(certified_key: CertifiedKey) -> ServerConfigBuilder {
        ServerConfigBuilder {
            certified_key,
            alpn: Vec::new(),
            versions: None,
        }
    }

    /// Convenience constructor.
    pub fn new(certified_key: CertifiedKey) -> Arc<Self> {
        Arc::new(Self {
            versions: vec![ProtocolVersion::Tls13, ProtocolVersion::Tls12],
            cipher_suites: CipherSuite::default_offered().to_vec(),
            named_groups: NamedGroup::default_offered().to_vec(),
            signature_schemes: SignatureScheme::default_offered().to_vec(),
            alpn_protocols: Vec::new(),
            certified_key,
            require_ems: true,
        })
    }
}

/// Builder for [`ServerConfig`].
pub struct ServerConfigBuilder {
    certified_key: CertifiedKey,
    alpn: Vec<Vec<u8>>,
    versions: Option<Vec<ProtocolVersion>>,
}

impl ServerConfigBuilder {
    /// Sets ALPN protocols.
    #[must_use]
    pub fn with_alpn(mut self, protocols: Vec<Vec<u8>>) -> Self {
        self.alpn = protocols;
        self
    }

    /// Restricts enabled versions.
    #[must_use]
    pub fn with_versions(mut self, versions: Vec<ProtocolVersion>) -> Self {
        self.versions = Some(versions);
        self
    }

    /// Builds the config.
    #[must_use]
    pub fn build(self) -> ServerConfig {
        ServerConfig {
            versions: self
                .versions
                .unwrap_or_else(|| vec![ProtocolVersion::Tls13, ProtocolVersion::Tls12]),
            cipher_suites: CipherSuite::default_offered().to_vec(),
            named_groups: NamedGroup::default_offered().to_vec(),
            signature_schemes: SignatureScheme::default_offered().to_vec(),
            alpn_protocols: self.alpn,
            certified_key: self.certified_key,
            require_ems: true,
        }
    }
}

/// Selects the mutually preferred protocol version.
pub fn select_version(
    offered: &[ProtocolVersion],
    supported: &[ProtocolVersion],
) -> TlsResult<ProtocolVersion> {
    for v in supported {
        if offered.contains(v) {
            return Ok(*v);
        }
    }
    Err(TlsError::Alert(
        crate::network::tls_alert::AlertDescription::ProtocolVersion,
    ))
}

/// Selects the first mutually supported cipher suite from `preference`.
pub fn select_cipher_suite(
    preference: &[CipherSuite],
    offered_codes: &[u16],
    version: ProtocolVersion,
) -> TlsResult<CipherSuite> {
    for suite in preference {
        let ok_version = match version {
            ProtocolVersion::Tls13 => suite.is_tls13(),
            ProtocolVersion::Tls12 => suite.is_tls12(),
            _ => false,
        };
        if ok_version && offered_codes.contains(&suite.as_u16()) {
            return Ok(*suite);
        }
    }
    Err(TlsError::Alert(
        crate::network::tls_alert::AlertDescription::HandshakeFailure,
    ))
}

/// Selects a mutually supported named group.
pub fn select_group(
    preference: &[NamedGroup],
    offered: &[NamedGroup],
) -> TlsResult<NamedGroup> {
    for g in preference {
        if offered.contains(g) {
            return Ok(*g);
        }
    }
    Err(TlsError::Alert(
        crate::network::tls_alert::AlertDescription::HandshakeFailure,
    ))
}

/// Selects ALPN protocol (server preference order).
pub fn select_alpn(server: &[Vec<u8>], client: &[Vec<u8>]) -> Option<Vec<u8>> {
    if server.is_empty() || client.is_empty() {
        return None;
    }
    for s in server {
        if client.iter().any(|c| c == s) {
            return Some(s.clone());
        }
    }
    None
}
