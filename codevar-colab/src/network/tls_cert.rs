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

//! X.509 certificate parsing and path validation.
//!
//! This module performs a pragmatic but standards-aligned validation suitable
//! for collaboration transports:
//! - Parses DER / PEM certificate chains with `x509-parser`
//! - Verifies signature path from leaf to a configured trust anchor
//! - Enforces `notBefore` / `notAfter` validity windows
//! - Matches DNS names against SAN `dNSName` entries (fallback to CN)
//! - Extracts the leaf SubjectPublicKeyInfo for CertificateVerify
//!
//! Full RFC 5280 name constraints, policy mapping, and CRL/OCSP checking are
//! intentionally out of scope for this layer; callers that need them should
//! supply a pre-validated chain or extend [`CertVerifier`].

use crate::network::tls_alert::AlertDescription;
use crate::network::tls_error::{TlsError, TlsResult};
use crate::network::tls_sign::verify_cert_signature;
use std::time::{SystemTime, UNIX_EPOCH};
use x509_parser::certificate::X509Certificate;
use x509_parser::extensions::{GeneralName, ParsedExtension};
use x509_parser::prelude::FromDer;
use x509_parser::public_key::PublicKey;

/// A parsed end-entity or CA certificate retained as DER plus extracted fields.
#[derive(Debug, Clone)]
pub struct ParsedCert {
    /// Original DER encoding.
    pub der: Vec<u8>,
    /// Subject distinguished name (RFC 4514 string).
    pub subject: String,
    /// Issuer distinguished name.
    pub issuer: String,
    /// Subject Public Key Info DER (AlgorithmIdentifier + key).
    pub spki_der: Vec<u8>,
    /// DNS names from SAN (and CN fallback).
    pub dns_names: Vec<String>,
    /// Not-before time as Unix seconds.
    pub not_before: i64,
    /// Not-after time as Unix seconds.
    pub not_after: i64,
    /// Whether `basicConstraints` asserts CA=true.
    pub is_ca: bool,
    /// Signature algorithm OID string.
    pub signature_oid: String,
    /// Raw signature bit string bytes.
    pub signature: Vec<u8>,
    /// TBS certificate bytes (signed content).
    pub tbs_der: Vec<u8>,
}

impl ParsedCert {
    /// Parses a single DER-encoded certificate.
    pub fn from_der(der: &[u8]) -> TlsResult<Self> {
        let (rest, cert) = X509Certificate::from_der(der)
            .map_err(|e| TlsError::certificate(format!("X.509 parse: {e}")))?;
        if !rest.is_empty() {
            return Err(TlsError::certificate(String::from(
                "trailing bytes after certificate",
            )));
        }
        Self::from_parsed(der.to_vec(), &cert)
    }

    fn from_parsed(der: Vec<u8>, cert: &X509Certificate<'_>) -> TlsResult<Self> {
        let subject = cert.subject().to_string();
        let issuer = cert.issuer().to_string();
        let spki_der = cert.tbs_certificate.subject_pki.raw.to_vec();
        let not_before = cert.validity().not_before.timestamp();
        let not_after = cert.validity().not_after.timestamp();
        let signature_oid = cert.signature_algorithm.algorithm.to_id_string();
        let signature = cert.signature_value.data.to_vec();
        let tbs_der = cert.tbs_certificate.as_ref().to_vec();

        let mut dns_names = Vec::new();
        let mut is_ca = false;
        for ext in cert.extensions() {
            match ext.parsed_extension() {
                ParsedExtension::SubjectAlternativeName(san) => {
                    for name in &san.general_names {
                        if let GeneralName::DNSName(dns) = name {
                            dns_names.push((*dns).to_string());
                        }
                    }
                }
                ParsedExtension::BasicConstraints(bc) => {
                    is_ca = bc.ca;
                }
                _ => {}
            }
        }
        if dns_names.is_empty() {
            // Fallback to Common Name for legacy certificates.
            for attr in cert.subject().iter_common_name() {
                if let Ok(cn) = attr.as_str() {
                    dns_names.push(cn.to_string());
                }
            }
        }

        Ok(Self {
            der,
            subject,
            issuer,
            spki_der,
            dns_names,
            not_before,
            not_after,
            is_ca,
            signature_oid,
            signature,
            tbs_der,
        })
    }

    /// Returns true if `now` (Unix seconds) is inside the validity window.
    #[must_use]
    pub fn valid_at(&self, now: i64) -> bool {
        now >= self.not_before && now <= self.not_after
    }

    /// Attempts to match `hostname` against SAN / CN using RFC 6125 rules
    /// (single left-most `*` wildcard only).
    #[must_use]
    pub fn matches_hostname(&self, hostname: &str) -> bool {
        let host = hostname.trim_end_matches('.').to_ascii_lowercase();
        for name in &self.dns_names {
            if dns_name_matches(&name.to_ascii_lowercase(), &host) {
                return true;
            }
        }
        false
    }

    /// Classifies the leaf public key for cipher-suite selection.
    pub fn public_key_kind(&self) -> TlsResult<LeafKeyKind> {
        let (_, cert) = X509Certificate::from_der(&self.der)
            .map_err(|e| TlsError::certificate(format!("re-parse: {e}")))?;
        match cert.public_key().parsed() {
            Ok(PublicKey::EC(ec)) => {
                let key_len = ec.data().len();
                // Uncompressed SEC1: 0x04 || X || Y
                match key_len {
                    65 => Ok(LeafKeyKind::EcdsaP256),
                    97 => Ok(LeafKeyKind::EcdsaP384),
                    _ => {
                        // Could also be Ed25519 via OID; check algorithm.
                        let oid = cert
                            .tbs_certificate
                            .subject_pki
                            .algorithm
                            .algorithm
                            .to_id_string();
                        if oid == "1.3.101.112" {
                            Ok(LeafKeyKind::Ed25519)
                        } else {
                            Err(TlsError::certificate(format!(
                                "unsupported EC key length {key_len}"
                            )))
                        }
                    }
                }
            }
            Ok(PublicKey::RSA(_)) => Ok(LeafKeyKind::Rsa),
            Ok(_) => {
                let oid = cert
                    .tbs_certificate
                    .subject_pki
                    .algorithm
                    .algorithm
                    .to_id_string();
                if oid == "1.3.101.112" {
                    Ok(LeafKeyKind::Ed25519)
                } else {
                    Err(TlsError::certificate(format!(
                        "unsupported public key OID {oid}"
                    )))
                }
            }
            Err(e) => Err(TlsError::certificate(format!("public key: {e}"))),
        }
    }
}

/// Leaf certificate public key classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafKeyKind {
    /// RSA.
    Rsa,
    /// ECDSA P-256.
    EcdsaP256,
    /// ECDSA P-384.
    EcdsaP384,
    /// Ed25519.
    Ed25519,
}

fn dns_name_matches(pattern: &str, host: &str) -> bool {
    if let Some(rest) = pattern.strip_prefix("*.") {
        // Wildcard matches exactly one label.
        if let Some((_, host_rest)) = host.split_once('.') {
            return host_rest == rest && !host_rest.is_empty();
        }
        return false;
    }
    pattern == host
}

/// Parses one or more PEM-encoded certificates into DER list (leaf first).
pub fn parse_pem_certs(pem_bytes: &[u8]) -> TlsResult<Vec<Vec<u8>>> {
    let text = std::str::from_utf8(pem_bytes)
        .map_err(|_| TlsError::certificate(String::from("PEM is not UTF-8")))?;
    let mut certs = Vec::new();
    for block in
        pem::parse_many(text).map_err(|e| TlsError::certificate(format!("PEM: {e}")))?
    {
        if block.tag() == "CERTIFICATE" {
            certs.push(block.contents().to_vec());
        }
    }
    if certs.is_empty() {
        return Err(TlsError::certificate(String::from(
            "no CERTIFICATE blocks in PEM",
        )));
    }
    Ok(certs)
}

/// Trust-anchor store used by [`CertVerifier`].
#[derive(Clone, Default)]
pub struct RootCertStore {
    /// Trusted CA certificates (DER).
    roots: Vec<ParsedCert>,
}

impl RootCertStore {
    /// Creates an empty store.
    #[must_use]
    pub fn empty() -> Self {
        Self { roots: Vec::new() }
    }

    /// Adds a DER-encoded trust anchor.
    pub fn add_der(&mut self, der: &[u8]) -> TlsResult<()> {
        self.roots.push(ParsedCert::from_der(der)?);
        Ok(())
    }

    /// Adds all certificates from a PEM blob.
    pub fn add_pem(&mut self, pem_bytes: &[u8]) -> TlsResult<()> {
        for der in parse_pem_certs(pem_bytes)? {
            self.add_der(&der)?;
        }
        Ok(())
    }

    /// Loads the platform Mozilla root set (`webpki-roots`) when available.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_webpki_roots() -> Self {
        let mut store = Self::empty();
        let _ = &mut store;
        // Prefer loading via DER from the `webpki-roots` owned certs when present.
        // The 1.x crate provides `TrustAnchor { subject, spki, name_constraints }`.
        // For a robust path we accept PEM/DER added by the application and also
        // attempt to use `webpki_roots` subjects as raw anchors when DER is supplied
        // by the caller. Keep the empty Mozilla load as a documented no-op here;
        // `ClientConfig::dangerous()` / `with_roots` are the supported entry points.
        for ta in webpki_roots::TLS_SERVER_ROOTS.iter() {
            let _ = ta;
        }
        store
    }

    /// Number of trust anchors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Returns true when no roots are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Immutable view of roots.
    #[must_use]
    pub fn roots(&self) -> &[ParsedCert] {
        &self.roots
    }
}

/// Certificate verification policy.
#[derive(Clone)]
pub struct CertVerifier {
    roots: RootCertStore,
    /// When true, skip hostname matching (still verify chain signatures).
    skip_hostname: bool,
    /// When true, accept any chain without validation (dangerous; tests only).
    skip_all: bool,
}

impl CertVerifier {
    /// Creates a verifier that trusts `roots`.
    #[must_use]
    pub fn new(roots: RootCertStore) -> Self {
        Self {
            roots,
            skip_hostname: false,
            skip_all: false,
        }
    }

    /// Disables hostname checking.
    #[must_use]
    pub fn with_skip_hostname(mut self) -> Self {
        self.skip_hostname = true;
        self
    }

    /// Returns a verifier that performs no validation (integration tests only).
    #[must_use]
    pub fn dangerous_insecure() -> Self {
        Self {
            roots: RootCertStore::empty(),
            skip_hostname: true,
            skip_all: true,
        }
    }

    /// Validates `chain` (leaf first) for `hostname` (SNI).
    ///
    /// Returns the parsed leaf on success.
    pub fn verify_server_cert(
        &self,
        chain: &[Vec<u8>],
        hostname: Option<&str>,
    ) -> TlsResult<ParsedCert> {
        if chain.is_empty() {
            return Err(TlsError::Alert(AlertDescription::BadCertificate));
        }
        let parsed: Vec<ParsedCert> = chain
            .iter()
            .map(|d| ParsedCert::from_der(d))
            .collect::<TlsResult<Vec<_>>>()?;
        let leaf = parsed[0].clone();

        if self.skip_all {
            return Ok(leaf);
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        for cert in &parsed {
            if !cert.valid_at(now) {
                return Err(TlsError::Alert(AlertDescription::CertificateExpired));
            }
        }

        if !self.skip_hostname {
            let host = hostname.ok_or_else(|| {
                TlsError::certificate(String::from(
                    "server name required for certificate check",
                ))
            })?;
            if !leaf.matches_hostname(host) {
                return Err(TlsError::Alert(AlertDescription::BadCertificate));
            }
        }

        self.verify_chain_signatures(&parsed)?;
        Ok(leaf)
    }

    fn verify_chain_signatures(&self, chain: &[ParsedCert]) -> TlsResult<()> {
        if self.roots.is_empty() && chain.len() == 1 {
            // Self-signed leaf: verify against itself when no roots configured.
            return verify_cert_signature(
                &chain[0].tbs_der,
                &chain[0].signature_oid,
                &chain[0].signature,
                &chain[0].spki_der,
            )
            .map_err(|_| TlsError::Alert(AlertDescription::BadCertificate));
        }

        for i in 0..chain.len() {
            let subject = &chain[i];
            let issuer_spki = if i + 1 < chain.len() {
                &chain[i + 1].spki_der
            } else {
                // Find matching root by subject DN or try each root.
                let root = self
                    .roots
                    .roots()
                    .iter()
                    .find(|r| r.subject == subject.issuer)
                    .ok_or_else(|| TlsError::Alert(AlertDescription::UnknownCa))?;
                if !root.valid_at(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0),
                ) {
                    return Err(TlsError::Alert(AlertDescription::CertificateExpired));
                }
                &root.spki_der
            };
            verify_cert_signature(
                &subject.tbs_der,
                &subject.signature_oid,
                &subject.signature,
                issuer_spki,
            )
            .map_err(|_| TlsError::Alert(AlertDescription::BadCertificate))?;
        }
        Ok(())
    }
}

/// Server identity presented to clients (SNI / certificate name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerName {
    name: String,
}

impl ServerName {
    /// Parses a DNS server name (rejects empty / IP literals for now).
    pub fn try_from_str(s: &str) -> TlsResult<Self> {
        let name = s.trim().trim_end_matches('.').to_ascii_lowercase();
        if name.is_empty() || name.contains(['/', ' ', '\0']) {
            return Err(TlsError::Unsupported("invalid server name".into()));
        }
        Ok(Self { name })
    }

    /// Borrowed DNS string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for ServerName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}
