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

//! TLS certificate / ServerName / config selection unit tests.

use codevar_colab::network::tls_connection_conf::{
    select_alpn, select_cipher_suite, select_group, select_version,
};
use codevar_colab::network::{
    CipherSuite, NamedGroup, ProtocolVersion, ServerName, parse_pem_certs,
};
use rcgen::{CertificateParams, KeyPair};

#[test]
fn server_name_parsing() {
    assert!(ServerName::try_from_str("example.com").is_ok());
    assert!(ServerName::try_from_str("localhost").is_ok());
    assert!(ServerName::try_from_str("").is_err());
}

#[test]
fn parse_pem_certs_roundtrip() {
    let key_pair = KeyPair::generate().expect("keygen");
    let params = CertificateParams::new(vec!["localhost".into()]).expect("params");
    let cert = params.self_signed(&key_pair).expect("self-signed");
    let pem = cert.pem();
    let certs = parse_pem_certs(pem.as_bytes()).expect("parse");
    assert_eq!(certs.len(), 1);
    assert!(
        parse_pem_certs(b"not pem").is_err()
            || parse_pem_certs(b"not pem").unwrap().is_empty()
    );
}

#[test]
fn select_version_prefers_supported_order() {
    let offered = [ProtocolVersion::Tls12, ProtocolVersion::Tls13];
    let supported = [ProtocolVersion::Tls13, ProtocolVersion::Tls12];
    assert_eq!(
        select_version(&offered, &supported).unwrap(),
        ProtocolVersion::Tls13
    );
    assert!(
        select_version(&[ProtocolVersion::Tls13], &[ProtocolVersion::Tls12]).is_err()
    );
}

#[test]
fn select_cipher_suite_and_group() {
    let preference = [
        CipherSuite::TlsAes256GcmSha384,
        CipherSuite::TlsAes128GcmSha256,
    ];
    let offered_codes = [
        CipherSuite::TlsAes128GcmSha256.as_u16(),
        CipherSuite::TlsAes256GcmSha384.as_u16(),
    ];
    assert_eq!(
        select_cipher_suite(&preference, &offered_codes, ProtocolVersion::Tls13).unwrap(),
        CipherSuite::TlsAes256GcmSha384
    );

    // TLS 1.2 suite must not be selected for TLS 1.3.
    let tls12_only = [CipherSuite::TlsEcdheRsaWithAes128GcmSha256.as_u16()];
    assert!(
        select_cipher_suite(&preference, &tls12_only, ProtocolVersion::Tls13).is_err()
    );

    let pref_groups = [NamedGroup::X25519, NamedGroup::Secp256r1];
    let offered_groups = [NamedGroup::Secp256r1, NamedGroup::X25519];
    assert_eq!(
        select_group(&pref_groups, &offered_groups).unwrap(),
        NamedGroup::X25519
    );
    assert!(select_group(&[NamedGroup::X25519], &[NamedGroup::Secp384r1]).is_err());
}

#[test]
fn select_alpn_server_preference() {
    let server = [b"h2".to_vec(), b"http/1.1".to_vec()];
    let client = [b"http/1.1".to_vec(), b"h2".to_vec()];
    assert_eq!(select_alpn(&server, &client), Some(b"h2".to_vec()));
    assert_eq!(select_alpn(&server, &[b"spdy/3".to_vec()]), None);
    assert_eq!(select_alpn(&[], &client), None);
}
