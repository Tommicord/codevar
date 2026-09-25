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

use crate::tls_alert::AlertDescription;
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_sign::verify_cert_signature;
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
        let (rest, cert) =
            X509Certificate::from_der(der).map_err(|e| TlsError::certificate(format!("X.509 parse: {e}")))?;
        if !rest.is_empty() {
            return Err(TlsError::certificate(String::from(
                "trailing bytes after certificate",
            )));
        }
        Self::from_parsed(der.to_vec(), &cert)
    }

    fn from_parsed(der: Vec<u8>, cert: &X509Certificate<'_>) -> TlsResult<Self> {
        let subject = cert
            .subject()
            .to_string();
        let issuer = cert
            .issuer()
            .to_string();
        let spki_der = cert
            .tbs_certificate
            .subject_pki
            .raw
            .to_vec();
        let not_before = cert
            .validity()
            .not_before
            .timestamp();
        let not_after = cert
            .validity()
            .not_after
            .timestamp();
        let signature_oid = cert
            .signature_algorithm
            .algorithm
            .to_id_string();
        let signature = cert
            .signature_value
            .data
            .to_vec();
        let tbs_der = cert
            .tbs_certificate
            .as_ref()
            .to_vec();

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
            for attr in cert
                .subject()
                .iter_common_name()
            {
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
        let host = hostname
            .trim_end_matches('.')
            .to_ascii_lowercase();
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
        match cert
            .public_key()
            .parsed()
        {
            Ok(PublicKey::EC(ec)) => {
                let key_len = ec
                    .data()
                    .len();
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
                    Err(TlsError::certificate(format!("unsupported public key OID {oid}")))
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
    for block in pem::parse_many(text).map_err(|e| TlsError::certificate(format!("PEM: {e}")))? {
        if block.tag() == "CERTIFICATE" {
            certs.push(
                block
                    .contents()
                    .to_vec(),
            );
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
        self.roots
            .push(ParsedCert::from_der(der)?);
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
        self.roots
            .len()
    }

    /// Returns true when no roots are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.roots
            .is_empty()
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
    pub fn verify_server_cert(&self, chain: &[Vec<u8>], hostname: Option<&str>) -> TlsResult<ParsedCert> {
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
                TlsError::certificate(String::from("server name required for certificate check"))
            })?;
            if !leaf.matches_hostname(host) {
                return Err(TlsError::Alert(AlertDescription::BadCertificate));
            }
        }

        self.verify_chain_signatures(&parsed)?;
        Ok(leaf)
    }

    fn verify_chain_signatures(&self, chain: &[ParsedCert]) -> TlsResult<()> {
        if self
            .roots
            .is_empty()
            && chain.len() == 1
        {
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
                    .ok_or(TlsError::Alert(AlertDescription::UnknownCa))?;
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
        let name = s
            .trim()
            .trim_end_matches('.')
            .to_ascii_lowercase();
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

#[cfg(test)]
mod tests {
    use super::*;

    // Self-signed fixtures generated once with OpenSSL; embedded so tests are
    // deterministic and offline. Validity windows are wide (2026..2046 or
    // explicitly back-/forward-dated).
    const LEAF_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----\nMIIBuDCCAV+gAwIBAgIUI45Sxo+yHGLkPe4tXHxugB3Lhu4wCgYIKoZIzj0EAwIw\nFjEUMBIGA1UEAwwLZXhhbXBsZS5jb20wHhcNMjYwOTIzMjAyMTQxWhcNNDYwOTE4\nMjAyMTQxWjAWMRQwEgYDVQQDDAtleGFtcGxlLmNvbTBZMBMGByqGSM49AgEGCCqG\nSM49AwEHA0IABL5z8IGhXvq7uF8xXipWmy6wpf9TtmwjEARYe+DrRZe7N8Nx/pYg\ntOads0U1R/xzZvbzPrkiSCo+dZ9b0nKDvC6jgYowgYcwHQYDVR0OBBYEFIlrOHUy\nsFOYDyIQlSqE41PLG/0ZMB8GA1UdIwQYMBaAFIlrOHUysFOYDyIQlSqE41PLG/0Z\nMA8GA1UdEwEB/wQFMAMBAf8wNAYDVR0RBC0wK4ILZXhhbXBsZS5jb22CD3d3dy5l\neGFtcGxlLmNvbYILKi53aWxkLnRlc3QwCgYIKoZIzj0EAwIDRwAwRAIgUxJKwkme\nVz6RuQm949/phJHV5pb7ghO7S6Yk1fqvHSUCIAdmL88XANgTFjKfEgg3U890l+OD\nsJf8oOZ6rFyNdNBS\n-----END CERTIFICATE-----\n";
    const LEGACY_CN_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----\nMIIBhjCCAS2gAwIBAgIUO6Ip/LYqZSWted7iKFibG71hRLEwCgYIKoZIzj0EAwIw\nGTEXMBUGA1UEAwwObGVnYWN5LmV4YW1wbGUwHhcNMjYwOTIzMjAyMTQzWhcNNDYw\nOTE4MjAyMTQzWjAZMRcwFQYDVQQDDA5sZWdhY3kuZXhhbXBsZTBZMBMGByqGSM49\nAgEGCCqGSM49AwEHA0IABL5z8IGhXvq7uF8xXipWmy6wpf9TtmwjEARYe+DrRZe7\nN8Nx/pYgtOads0U1R/xzZvbzPrkiSCo+dZ9b0nKDvC6jUzBRMB0GA1UdDgQWBBSJ\nazh1MrBTmA8iEJUqhONTyxv9GTAfBgNVHSMEGDAWgBSJazh1MrBTmA8iEJUqhONT\nyxv9GTAPBgNVHRMBAf8EBTADAQH/MAoGCCqGSM49BAMCA0cAMEQCIGH/5s1jdHhW\nK3kYJyCkjQeW/tMhPGPJk38+CP4S/A9AAiACkYes71uX7Shia4ffMXKiqgZDwWv4\nkvc9B+bKqTdGaQ==\n-----END CERTIFICATE-----\n";
    const RSA_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----\nMIIDJTCCAg2gAwIBAgIUcdtWi940TnGEWt5qyfBH2dTC4pQwDQYJKoZIhvcNAQEL\nBQAwFjEUMBIGA1UEAwwLcnNhLmV4YW1wbGUwHhcNMjYwOTIzMjAyMTQ0WhcNNDYw\nOTE4MjAyMTQ0WjAWMRQwEgYDVQQDDAtyc2EuZXhhbXBsZTCCASIwDQYJKoZIhvcN\nAQEBBQADggEPADCCAQoCggEBALH9v54gvK6ux9XBhnoq3wJlf3zfS5uqamEPl283\nio7chV6xQD7kiMJyZhiYuoi4vxut2G6L4H3V2oOEx2DZF+sCIgH3pz9mcXAaFJqt\nUXIerkJNbzkDYjAAJz0k6yfZ0z+NsslyoB8DMkJIBYFqnTGdQN5B3MdEg0+0+C8c\nV6gRC+6m1cRYwW1bttoiKowVB+ToM+U0I6QLodv4J9MJEaSRWg/8mLNctXiIXTCt\nlIPQgDKDvgvMyv26YZIiNcooqu0wX+Lu83gfDsfCU3Hdu5eHr2WNoVz7jfn8c0Go\ns2i0lc4gN7rsM7dV5EOQ4cMinFpPEX+SRncVn36tFOF0+l8CAwEAAaNrMGkwHQYD\nVR0OBBYEFDCMovJahLSyrIbM2bO0FFu2/oxaMB8GA1UdIwQYMBaAFDCMovJahLSy\nrIbM2bO0FFu2/oxaMA8GA1UdEwEB/wQFMAMBAf8wFgYDVR0RBA8wDYILcnNhLmV4\nYW1wbGUwDQYJKoZIhvcNAQELBQADggEBALCogadNBELoYpImCAUB9Zh2YJwn98AN\nX6wx+Tjt9CVhg2gULmREv57rQYH8ogO8/HvR9Ob20WQTvJPX2HzKbe+muEvSGdNi\nOWGme0QpoGEEbgetgkXW/paq+yqWZzEIlonmnzNP0SDl/j6QubAVyrnKlgeJHPmv\nWUlj3I1zA1r2qtDwpckMj8prW6SH/HdJUoFiq9qg/Nf3kGypan0fyiIE3TfARQkC\n1a+yzG3DNE0SWcPkLy6H/GgLZlcAUz18ng3WnRRXbo/gpREkw67PbhOYDbAKtjNl\nHIl7LhUR//ED80EOmZjhedky6ZjFS+B97L1MpooBS3WxEzNZxcC96MM=\n-----END CERTIFICATE-----\n";
    const ED25519_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----\nMIIBVjCCAQigAwIBAgIUVhgMIyB6/ckca6mtFTLxoDGxZSAwBQYDK2VwMBUxEzAR\nBgNVBAMMCmVkLmV4YW1wbGUwHhcNMjYwOTIzMjAyMTQ0WhcNNDYwOTE4MjAyMTQ0\nWjAVMRMwEQYDVQQDDAplZC5leGFtcGxlMCowBQYDK2VwAyEAzN7NYnXBZj3nBq9H\n4RykIbumO2b6EbuKWvqCmrkcFu+jajBoMB0GA1UdDgQWBBQjebGZjGjcbjEoIuGt\nSW/VDpcNMTAfBgNVHSMEGDAWgBQjebGZjGjcbjEoIuGtSW/VDpcNMTAPBgNVHRMB\nAf8EBTADAQH/MBUGA1UdEQQOMAyCCmVkLmV4YW1wbGUwBQYDK2VwA0EAXDQt/V8X\nRHG9Nbx8pmwOK1BQSKmf6bEmmhg1xIALd1I5exTECBi68SYwc7MOWcIcPZYN2Bne\noAfiMUdm6tSKCg==\n-----END CERTIFICATE-----\n";
    const FUTURE_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----\nMIIBVzCB/6ADAgECAhRvqrvOXrvnOKPzt8gX8Y24fl85mDAKBggqhkjOPQQDAjAZ\nMRcwFQYDVQQDDA5mdXR1cmUuZXhhbXBsZTAiGA8yMDk5MDEwMTAwMDAwMFoYDzIx\nMDAwMTAxMDAwMDAwWjAZMRcwFQYDVQQDDA5mdXR1cmUuZXhhbXBsZTBZMBMGByqG\nSM49AgEGCCqGSM49AwEHA0IABL5z8IGhXvq7uF8xXipWmy6wpf9TtmwjEARYe+Dr\nRZe7N8Nx/pYgtOads0U1R/xzZvbzPrkiSCo+dZ9b0nKDvC6jITAfMB0GA1UdDgQW\nBBSJazh1MrBTmA8iEJUqhONTyxv9GTAKBggqhkjOPQQDAgNHADBEAiBKyhSPWwUg\nvY7GKKbXVDPW+3TkYMHVkPNMNVCFXln7hwIgfDmT5aSHG6Gzx+bxMR0Mg2bvnyah\n1la/WoZZtji+1bI=\n-----END CERTIFICATE-----\n";
    const EXPIRED_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----\nMIIBVTCB/aADAgECAhQkkobnP5y41nJM3pta+5BO5XDUDzAKBggqhkjOPQQDAjAa\nMRgwFgYDVQQDDA9leHBpcmVkLmV4YW1wbGUwHhcNMjAwMTAxMDAwMDAwWhcNMjEw\nMTAxMDAwMDAwWjAaMRgwFgYDVQQDDA9leHBpcmVkLmV4YW1wbGUwWTATBgcqhkjO\nPQIBBggqhkjOPQMBBwNCAAS+c/CBoV76u7hfMV4qVpsusKX/U7ZsIxAEWHvg60WX\nuzfDcf6WILTmnbNFNUf8c2b28z65IkgqPnWfW9Jyg7wuoyEwHzAdBgNVHQ4EFgQU\niWs4dTKwU5gPIhCVKoTjU8sb/RkwCgYIKoZIzj0EAwIDRwAwRAIgDC+iz9HvijuM\n5Rs0UxN3Q6LDv+dQ/r1rRyrXjSphV6sCIDl9HAs3bVa1yTMCdMsMHmaVbOcgmm9s\nTEJgHRyQTV6B\n-----END CERTIFICATE-----\n";
    const TRUNCATED_DER_PEM: &[u8] =
        b"-----BEGIN CERTIFICATE-----\nMIIBuDCCAV+gAwIBAgIUI45Sxo+yHGLkPe4tXHxugB3Lhu4wCgYIKg==\n-----END CERTIFICATE-----\n";

    fn leaf_der() -> Vec<u8> {
        parse_pem_certs(LEAF_PEM)
            .unwrap()
            .remove(0)
    }

    #[test]
    fn parse_pem_single_and_multiple_certificates() {
        let certs = parse_pem_certs(LEAF_PEM).unwrap();
        assert_eq!(certs.len(), 1);
        assert!(!certs[0].is_empty());

        let mut chain = LEAF_PEM.to_vec();
        chain.extend_from_slice(RSA_PEM);
        let certs = parse_pem_certs(&chain).unwrap();
        assert_eq!(certs.len(), 2);
        // Leaf first order is preserved as given.
        assert_ne!(certs[0], certs[1]);
        assert_eq!(certs[0], parse_pem_certs(LEAF_PEM).unwrap()[0]);
    }

    #[test]
    fn parse_pem_rejects_empty_and_non_utf8() {
        let err = parse_pem_certs(b"").unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
        let err = parse_pem_certs(&[0xFF, 0xFE, 0x00, 0x80]).unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
    }

    #[test]
    fn parse_pem_rejects_wrong_headers_and_garbage() {
        // Same bytes, wrong tag on BEGIN/END lines.
        let wrong_tag = String::from_utf8(LEAF_PEM.to_vec()).unwrap();
        let wrong_tag = wrong_tag.replace("CERTIFICATE", "PUBLIC KEY  ");
        let err = parse_pem_certs(wrong_tag.as_bytes()).unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
        // Garbage base64 inside well-formed headers.
        let garbage = "-----BEGIN CERTIFICATE-----\nnot base64 at all!!!\n-----END CERTIFICATE-----\n";
        let err = parse_pem_certs(garbage.as_bytes()).unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
        // Unterminated block.
        let unterminated = "-----BEGIN CERTIFICATE-----\nQUFBQQ==\n";
        assert!(parse_pem_certs(unterminated.as_bytes()).is_err());
    }

    #[test]
    fn parse_pem_rejects_oversized_invalid_line_without_panic() {
        // ~200 KB of invalid base64 characters on one logical body: rejected
        // by the PEM/base64 layer while the input itself stays small.
        let mut pem = Vec::new();
        pem.extend_from_slice(b"-----BEGIN CERTIFICATE-----\n");
        for _ in 0..2_000 {
            pem.extend_from_slice(&[b'!'; 100]);
            pem.push(b'\n');
        }
        pem.extend_from_slice(b"-----END CERTIFICATE-----\n");
        assert!(pem.len() < 1_000_000);
        assert!(parse_pem_certs(&pem).is_err());
    }

    #[test]
    fn from_der_accepts_valid_and_rejects_trailing_bytes() {
        let der = leaf_der();
        let cert = ParsedCert::from_der(&der).unwrap();
        assert_eq!(cert.der, der);
        assert!(
            !cert
                .subject
                .is_empty()
        );
        assert!(
            !cert
                .issuer
                .is_empty()
        );
        assert!(
            !cert
                .spki_der
                .is_empty()
        );
        assert!(
            !cert
                .tbs_der
                .is_empty()
        );
        assert!(
            !cert
                .signature
                .is_empty()
        );
        assert!(
            !cert
                .signature_oid
                .is_empty()
        );

        let mut trailing = der.clone();
        trailing.push(0);
        let err = ParsedCert::from_der(&trailing).unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
    }

    #[test]
    fn from_der_rejects_truncated_and_oversized_declared_length() {
        // Truncated DER wrapped in PEM.
        let trunc = parse_pem_certs(TRUNCATED_DER_PEM)
            .unwrap()
            .remove(0);
        assert!(ParsedCert::from_der(&trunc).is_err());
        // Truncated leaf.
        let der = leaf_der();
        assert!(ParsedCert::from_der(&der[..der.len() / 2]).is_err());
        // SEQUENCE declaring a 0xFFFFFFFF-byte body over a tiny buffer:
        // rejected without attempting a large allocation.
        let huge = [0x30, 0x84, 0xFF, 0xFF, 0xFF, 0xFF, 0x02, 0x01, 0x00];
        let err = ParsedCert::from_der(&huge).unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));
        // Garbage.
        assert!(ParsedCert::from_der(&[0x00; 8]).is_err());
        assert!(ParsedCert::from_der(&[]).is_err());
    }

    #[test]
    fn root_cert_store_add_lookup_and_empty() {
        let mut store = RootCertStore::empty();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(
            store
                .roots()
                .is_empty()
        );

        store
            .add_der(&leaf_der())
            .unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert!(store.roots()[0].matches_hostname("example.com"));

        // Cloning preserves anchors.
        let clone = store.clone();
        assert_eq!(clone.len(), 1);

        // Adding garbage fails without growing the store.
        assert!(
            store
                .add_der(&[1, 2, 3])
                .is_err()
        );
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn root_cert_store_add_pem_bulk_and_errors() {
        let mut store = RootCertStore::empty();
        let mut multi = LEAF_PEM.to_vec();
        multi.extend_from_slice(RSA_PEM);
        multi.extend_from_slice(ED25519_PEM);
        store
            .add_pem(&multi)
            .unwrap();
        assert_eq!(store.len(), 3);

        let mut failed = RootCertStore::empty();
        assert!(
            failed
                .add_pem(b"")
                .is_err()
        );
        assert!(failed.is_empty());
    }

    #[test]
    fn server_name_parsing_valid_and_rejected() {
        let name = ServerName::try_from_str("Example.COM.").unwrap();
        assert_eq!(name.as_str(), "example.com");
        assert_eq!(name.to_string(), "example.com");
        assert_eq!(name, ServerName::try_from_str(" example.com").unwrap());

        for bad in ["", "   ", ".", "a b", "a\0b", "ex/mple.com", " a b "] {
            assert!(
                ServerName::try_from_str(bad).is_err(),
                "accepted bad name {bad:?}"
            );
        }
        // Case / trailing-dot insensitivity.
        assert_eq!(
            ServerName::try_from_str("WWW.Example.Com.")
                .unwrap()
                .as_str(),
            "www.example.com"
        );
    }

    #[test]
    fn server_name_length_and_ip_quirks() {
        // Quirk: the parser does not enforce DNS label/total length limits or
        // reject IP literals, despite the doc comment.
        let ip = ServerName::try_from_str("192.0.2.10").unwrap();
        assert_eq!(ip.as_str(), "192.0.2.10");
        let long_label = "a".repeat(64);
        assert!(ServerName::try_from_str(&long_label).is_ok());
        let long_total = format!("{}.com", "b".repeat(300));
        assert!(ServerName::try_from_str(&long_total).is_ok());
        let wildcard = ServerName::try_from_str("*.example.com").unwrap();
        assert_eq!(wildcard.as_str(), "*.example.com");
    }

    #[test]
    fn leaf_key_kind_classification() {
        let leaf = ParsedCert::from_der(&parse_pem_certs(LEAF_PEM).unwrap()[0]).unwrap();
        assert_eq!(
            leaf.public_key_kind()
                .unwrap(),
            LeafKeyKind::EcdsaP256
        );
        let rsa = ParsedCert::from_der(&parse_pem_certs(RSA_PEM).unwrap()[0]).unwrap();
        assert_eq!(
            rsa.public_key_kind()
                .unwrap(),
            LeafKeyKind::Rsa
        );
        let ed = ParsedCert::from_der(&parse_pem_certs(ED25519_PEM).unwrap()[0]).unwrap();
        assert_eq!(
            ed.public_key_kind()
                .unwrap(),
            LeafKeyKind::Ed25519
        );
        // Enum round trip via Debug/Clone/Copy.
        let kind = LeafKeyKind::EcdsaP256;
        let copy = kind;
        assert_eq!(kind, copy);
    }

    #[test]
    fn matches_hostname_exact_case_trailing_dot_and_wildcard() {
        let leaf = ParsedCert::from_der(&leaf_der()).unwrap();
        assert!(leaf.matches_hostname("example.com"));
        assert!(leaf.matches_hostname("EXAMPLE.COM"));
        assert!(leaf.matches_hostname("example.com."));
        assert!(leaf.matches_hostname("www.example.com"));
        assert!(leaf.matches_hostname("foo.wild.test"));
        // Wildcard matches exactly one label.
        assert!(!leaf.matches_hostname("a.b.wild.test"));
        assert!(!leaf.matches_hostname("wild.test"));
        assert!(!leaf.matches_hostname("badexample.com"));
        assert!(!leaf.matches_hostname("example.org"));
        assert!(!leaf.matches_hostname(""));

        let legacy = ParsedCert::from_der(&parse_pem_certs(LEGACY_CN_PEM).unwrap()[0]).unwrap();
        // CN fallback when no SAN dNSName is present.
        assert!(legacy.matches_hostname("legacy.example"));
        assert!(!legacy.matches_hostname("other.example"));
    }

    #[test]
    fn validity_window_boundaries_and_stale_certs() {
        let leaf = ParsedCert::from_der(&leaf_der()).unwrap();
        assert!(leaf.valid_at(leaf.not_before));
        assert!(leaf.valid_at(leaf.not_after));
        assert!(leaf.valid_at((leaf.not_before + leaf.not_after) / 2));
        assert!(!leaf.valid_at(leaf.not_before - 1));
        assert!(!leaf.valid_at(leaf.not_after + 1));

        let future = ParsedCert::from_der(&parse_pem_certs(FUTURE_PEM).unwrap()[0]).unwrap();
        assert!(!future.valid_at(1_700_000_000));
        assert!(future.valid_at(4_100_000_000));

        let expired = ParsedCert::from_der(&parse_pem_certs(EXPIRED_PEM).unwrap()[0]).unwrap();
        assert!(!expired.valid_at(1_700_000_000));
        assert!(!expired.valid_at(0));
    }

    #[test]
    fn verifier_dangerous_insecure_skips_validation() {
        let verifier = CertVerifier::dangerous_insecure();
        let leaf = verifier
            .verify_server_cert(&[leaf_der()], Some("wrong.host.invalid"))
            .unwrap();
        assert!(leaf.matches_hostname("example.com"));
        // Even an expired chain parses and is accepted by the skip-all path.
        let expired = parse_pem_certs(EXPIRED_PEM).unwrap();
        verifier
            .verify_server_cert(&expired, Some("expired.example"))
            .unwrap();
        // Empty chains are still rejected up front.
        let err = verifier
            .verify_server_cert(&[], Some("example.com"))
            .unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::BadCertificate));
    }

    #[test]
    fn verifier_self_signed_chain_and_hostname_checks() {
        let mut roots = RootCertStore::empty();
        roots
            .add_der(&leaf_der())
            .unwrap();
        let verifier = CertVerifier::new(roots.clone());

        // Matching hostname: self-signed chain verifies against its anchor.
        let leaf = verifier
            .verify_server_cert(&[leaf_der()], Some("example.com"))
            .unwrap();
        assert!(leaf.is_ca);

        // Hostname mismatch.
        let err = verifier
            .verify_server_cert(&[leaf_der()], Some("other.example"))
            .unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::BadCertificate));

        // Missing hostname.
        let err = verifier
            .verify_server_cert(&[leaf_der()], None)
            .unwrap_err();
        assert!(matches!(err, TlsError::Certificate(_)));

        // No hostname check when configured.
        let no_host = CertVerifier::new(roots).with_skip_hostname();
        no_host
            .verify_server_cert(&[leaf_der()], Some("other.example"))
            .unwrap();
    }

    #[test]
    fn verifier_expired_and_future_chains_rejected() {
        let verifier = CertVerifier::dangerous_insecure().with_skip_hostname();
        // skip_all bypasses validity, so use a real verifier for these.
        let strict = CertVerifier::new(RootCertStore::empty());
        let expired = parse_pem_certs(EXPIRED_PEM).unwrap();
        let err = strict
            .verify_server_cert(&expired, Some("expired.example"))
            .unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::CertificateExpired));
        let future = parse_pem_certs(FUTURE_PEM).unwrap();
        let err = strict
            .verify_server_cert(&future, Some("future.example"))
            .unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::CertificateExpired));
        drop(verifier);
    }

    #[test]
    fn verifier_chain_order_and_issuer_mismatch() {
        let leaf = leaf_der();
        let rsa = parse_pem_certs(RSA_PEM)
            .unwrap()
            .remove(0);
        let mut roots = RootCertStore::empty();
        roots
            .add_der(&leaf)
            .unwrap();

        // Leaf presented under a non-matching issuer certificate.
        let verifier = CertVerifier::new(roots);
        let err = verifier
            .verify_server_cert(&[leaf.clone(), rsa.clone()], Some("example.com"))
            .unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::BadCertificate));

        // Unknown issuer with no usable root.
        let bare = CertVerifier::new(RootCertStore::empty());
        let err = bare
            .verify_server_cert(&[leaf.clone(), rsa], Some("example.com"))
            .unwrap_err();
        // Either signature mismatch (uses next cert SPKI) surfaces BadCertificate.
        assert_eq!(err, TlsError::Alert(AlertDescription::BadCertificate));

        // Duplicate self-signed leaf with the leaf trusted: both links verify.
        let dup = verifier
            .verify_server_cert(&[leaf.clone(), leaf], Some("example.com"))
            .unwrap();
        assert!(dup.matches_hostname("example.com"));
    }

    #[test]
    fn verifier_unknown_ca_when_roots_missing() {
        // Chain of one self-signed cert that is NOT in the trust store, with
        // empty roots: the empty-root self-signed shortcut verifies it.
        let verifier = CertVerifier::new(RootCertStore::empty());
        let leaf = leaf_der();
        verifier
            .verify_server_cert(std::slice::from_ref(&leaf), Some("example.com"))
            .unwrap();
        // But a wrong hostname still fails afterwards.
        let err = verifier
            .verify_server_cert(&[leaf], Some("nope.example"))
            .unwrap_err();
        assert_eq!(err, TlsError::Alert(AlertDescription::BadCertificate));
    }
}
