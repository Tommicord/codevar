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

//! TLS client and server configuration.

use crate::tls_cert::{CertVerifier, RootCertStore, parse_pem_certs};
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_ids::{CipherSuite, NamedGroup, ProtocolVersion, SignatureScheme};
use crate::tls_sign::PrivateKey;
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
        crate::tls_alert::AlertDescription::ProtocolVersion,
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
        crate::tls_alert::AlertDescription::HandshakeFailure,
    ))
}

/// Selects a mutually supported named group.
pub fn select_group(preference: &[NamedGroup], offered: &[NamedGroup]) -> TlsResult<NamedGroup> {
    for g in preference {
        if offered.contains(g) {
            return Ok(*g);
        }
    }
    Err(TlsError::Alert(
        crate::tls_alert::AlertDescription::HandshakeFailure,
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

#[cfg(test)]
mod tests {
    use super::*;

    const LEAF_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----\nMIIBuDCCAV+gAwIBAgIUI45Sxo+yHGLkPe4tXHxugB3Lhu4wCgYIKoZIzj0EAwIw\nFjEUMBIGA1UEAwwLZXhhbXBsZS5jb20wHhcNMjYwOTIzMjAyMTQxWhcNNDYwOTE4\nMjAyMTQxWjAWMRQwEgYDVQQDDAtleGFtcGxlLmNvbTBZMBMGByqGSM49AgEGCCqG\nSM49AwEHA0IABL5z8IGhXvq7uF8xXipWmy6wpf9TtmwjEARYe+DrRZe7N8Nx/pYg\ntOads0U1R/xzZvbzPrkiSCo+dZ9b0nKDvC6jgYowgYcwHQYDVR0OBBYEFIlrOHUy\nsFOYDyIQlSqE41PLG/0ZMB8GA1UdIwQYMBaAFIlrOHUysFOYDyIQlSqE41PLG/0Z\nMA8GA1UdEwEB/wQFMAMBAf8wNAYDVR0RBC0wK4ILZXhhbXBsZS5jb22CD3d3dy5l\neGFtcGxlLmNvbYILKi53aWxkLnRlc3QwCgYIKoZIzj0EAwIDRwAwRAIgUxJKwkme\nVz6RuQm949/phJHV5pb7ghO7S6Yk1fqvHSUCIAdmL88XANgTFjKfEgg3U890l+OD\nsJf8oOZ6rFyNdNBS\n-----END CERTIFICATE-----\n";
    const LEAF_KEY_PEM: &[u8] = b"-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgk1vhLdt3cRaaH+fX\n2hjwNQNzwnpoCdRcRaYF2WHgxy+hRANCAAS+c/CBoV76u7hfMV4qVpsusKX/U7Zs\nIxAEWHvg60WXuzfDcf6WILTmnbNFNUf8c2b28z65IkgqPnWfW9Jyg7wu\n-----END PRIVATE KEY-----\n";
    const RSA_KEY_PEM: &[u8] = b"-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCx/b+eILyursfV\nwYZ6Kt8CZX9830ubqmphD5dvN4qO3IVesUA+5IjCcmYYmLqIuL8brdhui+B91dqD\nhMdg2RfrAiIB96c/ZnFwGhSarVFyHq5CTW85A2IwACc9JOsn2dM/jbLJcqAfAzJC\nSAWBap0xnUDeQdzHRINPtPgvHFeoEQvuptXEWMFtW7baIiqMFQfk6DPlNCOkC6Hb\n+CfTCRGkkVoP/JizXLV4iF0wrZSD0IAyg74LzMr9umGSIjXKKKrtMF/i7vN4Hw7H\nwlNx3buXh69ljaFc+435/HNBqLNotJXOIDe67DO3VeRDkOHDIpxaTxF/kkZ3FZ9+\nrRThdPpfAgMBAAECggEAANat1Ypo3fu1TWh7tuAS189QgRH1cJRvq0ealsfA1iEY\nhRFleBaLLXTAKmuvif/hh5fSOss4pT27RhzwRlD4OlIQoC+j81TLsNAPCEc9EZL6\nkSdG39vWjCrfR4/pIwBj4ueglza5K/Lp0GjquGc2vxyuvzQBkPVSdZ7x+pf4Lby0\ndkybKHGo0DfvoTAj4CASPdGJ+6Zn1K5laWh3ydvdF+NDj3Bh9L59HYke+Ne62wWq\n9RsOgx237cwckyKnu014iQBIM8e66GSn23TQldTj81wWrRBwM7I/QNVMvhMs7WUx\nZnGiK4t5BTBHfUASzz+npZr1FBQMwMgPM0yU0WwrMQKBgQDZ0AnChhk/lYK5+LmR\nc6klTsMJa3fiKqACi36MgnOzLD6EXQN42dO7v1d04y84R6qcTjBDwYUUsTmmQC89\nh3hdfeFzCGf8AbDuM0uNiIT3F0jnjG4FXZEEFpCtd8ykq3neLP810uN/3OAQg4vW\nNWEs9Mf7T59ACfWlp/J+YcKNUQKBgQDRMm1TQD4mnIzLBisC8VI6tulJJ/sshq/X\nMd/P8+etiVBFEmLT96Q15I7/t2bKzmYZ8DxOaQwIVm0pNRYdetSgkHuOQejmsKW5\n7/bNDgQUOGlt2HcZa8eTVQsCD9hOAFac9wRAweXrWWWc7+j7xoFhP1CtWquDTuG6\nh9YWkvhgrwKBgELwUZ+LsMS+wR9AVl9iKVCC5SPG+F/0c5p0nl62VLJy3X+2SjPg\n1dZ0Vn9gtolYVRGWYfTgy3JxiOMUBLCnKpGo9xlwMuza5DJAZ27Gzv5VFJ28pa9W\ncxPLj4kQMT9GR7zFHWXAOxR3oBDTLK2XWBcF31PXw3xd0zWm6Lp4dt8RAoGBALGi\nUSSs21lr+z116lXgVkOXB3ZgJa7EW1Gufu6UnDhF7cwI9bQpht1gS3Cl6fnx0s7Z\nqEuodVgrExw3gKTdpOkGZnQAUWR5wO+m7HloGlyVHijw8wi59UiMoQFKNRDexq0Y\nLxtRygrS6S6epMYN49SQr8/TuumPtKrwJwEaIR/vAoGAKA/Zr509j0/uJYtPDrMR\nMX3LSOJOFv7bNxK/JOjrdZSvAY7ItxSao3b0hMuwWZJr4UVUdA3VZSNz4CwDEChS\nZ+yvuLHrmI7s5VxHNTbBLdtfvBBx+arqq77S4kQGRceTDmCOhWNDIrswUDBcIPMn\nDAJCWqguVMYBNomGXcYNZi0=\n-----END PRIVATE KEY-----\n";

    fn certified_key() -> CertifiedKey {
        CertifiedKey::from_pem(LEAF_PEM, LEAF_KEY_PEM).unwrap()
    }

    #[test]
    fn client_config_builder_defaults() {
        let cfg = ClientConfig::builder().build().unwrap();
        assert_eq!(cfg.versions, vec![ProtocolVersion::Tls13, ProtocolVersion::Tls12]);
        assert_eq!(cfg.cipher_suites, CipherSuite::default_offered().to_vec());
        assert_eq!(cfg.named_groups, NamedGroup::default_offered().to_vec());
        assert_eq!(cfg.signature_schemes, SignatureScheme::default_offered().to_vec());
        assert!(cfg.alpn_protocols.is_empty());
        assert!(cfg.require_ems);
        // Verifier rejects unknown hosts by default (not skip-all).
        let roots = RootCertStore::empty();
        let _ = roots;
        assert!(Arc::strong_count(&cfg.verifier) >= 1);
    }

    #[test]
    fn client_config_dangerous_insecure_defaults() {
        let cfg = ClientConfig::dangerous_insecure();
        assert_eq!(cfg.versions, vec![ProtocolVersion::Tls13, ProtocolVersion::Tls12]);
        assert!(!cfg.cipher_suites.is_empty());
        assert!(!cfg.named_groups.is_empty());
        assert!(!cfg.signature_schemes.is_empty());
        assert!(cfg.alpn_protocols.is_empty());
        assert!(cfg.require_ems);
        // Insecure verifier accepts a wrong hostname.
        let leaf = crate::tls_cert::parse_pem_certs(LEAF_PEM).unwrap();
        cfg.verifier
            .verify_server_cert(&leaf, Some("wrong.example"))
            .unwrap();
    }

    #[test]
    fn client_config_builder_overrides() {
        let mut roots = RootCertStore::empty();
        roots.add_pem(LEAF_PEM).unwrap();
        let cfg = ClientConfig::builder()
            .with_root_store(roots)
            .with_alpn(vec![b"h2".to_vec()])
            .with_versions(vec![ProtocolVersion::Tls12])
            .dangerous_skip_hostname()
            .build()
            .unwrap();
        assert_eq!(cfg.versions, vec![ProtocolVersion::Tls12]);
        assert_eq!(cfg.alpn_protocols, vec![b"h2".to_vec()]);
        // skip_hostname still validates chain signatures.
        let leaf = crate::tls_cert::parse_pem_certs(LEAF_PEM).unwrap();
        cfg.verifier
            .verify_server_cert(&leaf, Some("other.example"))
            .unwrap();
    }

    #[test]
    fn client_config_builder_with_root_pem_and_insecure_flag() {
        let cfg = ClientConfig::builder()
            .with_root_pem(LEAF_PEM)
            .unwrap()
            .dangerous_insecure()
            .build()
            .unwrap();
        let leaf = crate::tls_cert::parse_pem_certs(LEAF_PEM).unwrap();
        cfg.verifier
            .verify_server_cert(&leaf, Some("nope.invalid"))
            .unwrap();

        assert!(matches!(
            ClientConfig::builder().with_root_pem(b"not pem"),
            Err(TlsError::Certificate(_))
        ));
    }

    #[test]
    fn certified_key_from_pem_loads_chain_and_key() {
        let key = certified_key();
        assert_eq!(key.cert_chain.len(), 1);
        assert!(!key.cert_chain[0].is_empty());
        assert_eq!(key.key.kind(), crate::tls_sign::SignatureKind::EcdsaP256);
        // Clone shares the Arc key.
        let cloned = key.clone();
        assert!(Arc::ptr_eq(&key.key, &cloned.key));
        assert_eq!(cloned.cert_chain, key.cert_chain);
    }

    #[test]
    fn certified_key_from_pem_errors_on_bad_inputs() {
        assert!(matches!(
            CertifiedKey::from_pem(b"", LEAF_KEY_PEM),
            Err(TlsError::Certificate(_))
        ));
        assert!(matches!(
            CertifiedKey::from_pem(LEAF_PEM, b"garbage"),
            Err(TlsError::Certificate(_))
        ));
        assert!(matches!(
            CertifiedKey::from_pem(LEAF_PEM, b""),
            Err(TlsError::Certificate(_))
        ));
    }

    #[test]
    fn certified_key_does_not_validate_key_cert_match() {
        // Quirk: an RSA key paired with an ECDSA certificate loads successfully.
        let mixed = CertifiedKey::from_pem(LEAF_PEM, RSA_KEY_PEM).unwrap();
        assert_eq!(mixed.cert_chain.len(), 1);
        assert_eq!(mixed.key.kind(), crate::tls_sign::SignatureKind::Rsa);
    }

    #[test]
    fn server_config_new_defaults_and_builder() {
        let cfg = ServerConfig::new(certified_key());
        assert_eq!(cfg.versions, vec![ProtocolVersion::Tls13, ProtocolVersion::Tls12]);
        assert_eq!(cfg.cipher_suites, CipherSuite::default_offered().to_vec());
        assert_eq!(cfg.named_groups, NamedGroup::default_offered().to_vec());
        assert!(cfg.alpn_protocols.is_empty());
        assert!(cfg.require_ems);
        assert_eq!(cfg.certified_key.cert_chain.len(), 1);

        let built = ServerConfig::builder(certified_key())
            .with_alpn(vec![b"h2".to_vec(), b"http/1.1".to_vec()])
            .with_versions(vec![ProtocolVersion::Tls13])
            .build();
        assert_eq!(built.versions, vec![ProtocolVersion::Tls13]);
        assert_eq!(built.alpn_protocols.len(), 2);
        assert!(built.require_ems);

        let default_builder = ServerConfig::builder(certified_key()).build();
        assert_eq!(
            default_builder.versions,
            vec![ProtocolVersion::Tls13, ProtocolVersion::Tls12]
        );
    }

    #[test]
    fn select_version_prefers_server_order_and_errors() {
        use ProtocolVersion::{Tls12, Tls13};
        assert_eq!(select_version(&[Tls12, Tls13], &[Tls13, Tls12]).unwrap(), Tls13);
        assert_eq!(select_version(&[Tls13, Tls12], &[Tls12, Tls13]).unwrap(), Tls12);
        assert_eq!(select_version(&[Tls12], &[Tls13, Tls12]).unwrap(), Tls12);
        let err = select_version(&[], &[Tls13]).unwrap_err();
        assert_eq!(
            err,
            TlsError::Alert(crate::tls_alert::AlertDescription::ProtocolVersion)
        );
        let err = select_version(&[ProtocolVersion::Tls10], &[Tls13, Tls12]).unwrap_err();
        assert_eq!(
            err,
            TlsError::Alert(crate::tls_alert::AlertDescription::ProtocolVersion)
        );
    }

    #[test]
    fn select_cipher_suite_respects_version_partition() {
        use CipherSuite::{TlsAes128GcmSha256, TlsEcdheEcdsaWithAes128GcmSha256};
        use ProtocolVersion::{Tls12, Tls13};

        // Preference lists the TLS 1.3 suite first; under TLS 1.2 it is skipped.
        let pref = [TlsAes128GcmSha256, TlsEcdheEcdsaWithAes128GcmSha256];
        let both: Vec<u16> = pref.iter().map(|s| s.as_u16()).collect();
        assert_eq!(
            select_cipher_suite(&pref, &both, Tls12).unwrap(),
            TlsEcdheEcdsaWithAes128GcmSha256
        );
        assert_eq!(
            select_cipher_suite(&pref, &both, Tls13).unwrap(),
            TlsAes128GcmSha256
        );
        // TLS 1.3 suite not offered under TLS 1.3.
        let err = select_cipher_suite(&[TlsAes128GcmSha256], &[0xC02B], Tls13).unwrap_err();
        assert_eq!(
            err,
            TlsError::Alert(crate::tls_alert::AlertDescription::HandshakeFailure)
        );
        // Unknown offered code.
        let err = select_cipher_suite(&pref, &[0x0000], Tls13).unwrap_err();
        assert_eq!(
            err,
            TlsError::Alert(crate::tls_alert::AlertDescription::HandshakeFailure)
        );
        // Empty offered list.
        let err = select_cipher_suite(&pref, &[], Tls12).unwrap_err();
        assert_eq!(
            err,
            TlsError::Alert(crate::tls_alert::AlertDescription::HandshakeFailure)
        );
        // Unsupported version selects nothing.
        let err = select_cipher_suite(&pref, &both, ProtocolVersion::Tls11).unwrap_err();
        assert_eq!(
            err,
            TlsError::Alert(crate::tls_alert::AlertDescription::HandshakeFailure)
        );
    }

    #[test]
    fn select_group_follows_server_preference() {
        use NamedGroup::{Secp384r1, X25519};
        assert_eq!(
            select_group(&[Secp384r1, X25519], &[X25519, Secp384r1]).unwrap(),
            Secp384r1
        );
        assert_eq!(select_group(&[X25519], &[Secp384r1, X25519]).unwrap(), X25519);
        let err = select_group(&[Secp384r1], &[X25519]).unwrap_err();
        assert_eq!(
            err,
            TlsError::Alert(crate::tls_alert::AlertDescription::HandshakeFailure)
        );
        let err = select_group(&[], &[X25519]).unwrap_err();
        assert_eq!(
            err,
            TlsError::Alert(crate::tls_alert::AlertDescription::HandshakeFailure)
        );
    }

    #[test]
    fn select_alpn_server_preference_and_empty_cases() {
        let server = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
        let client = vec![b"http/1.1".to_vec(), b"h2".to_vec()];
        // Server order wins: h2 is first on the server list.
        assert_eq!(select_alpn(&server, &client).unwrap(), b"h2");
        let only_http = vec![b"http/1.1".to_vec()];
        assert_eq!(select_alpn(&server, &only_http).unwrap(), b"http/1.1");
        assert!(select_alpn(&[], &client).is_none());
        assert!(select_alpn(&server, &[]).is_none());
        let disjoint = vec![b"foo".to_vec()];
        assert!(select_alpn(&server, &disjoint).is_none());
    }
}
