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

//! TLS alert and identifier unit tests.

use codevar_colab::network::{
    Alert, AlertDescription, AlertLevel, CipherSuite, ContentType, ExtensionType,
    HandshakeType, NamedGroup, ProtocolVersion, SignatureScheme, TlsError,
};

#[test]
fn alert_encode_decode_roundtrip() {
    let a = Alert::fatal(AlertDescription::HandshakeFailure);
    let bytes = a.encode();
    assert_eq!(bytes, [2, 40]);
    let decoded = Alert::decode(&bytes).unwrap();
    assert_eq!(decoded, a);

    let w = Alert::warning(AlertDescription::CloseNotify);
    assert_eq!(w.encode(), [1, 0]);
    assert_eq!(Alert::decode(&w.encode()).unwrap(), w);
}

#[test]
fn alert_rejects_bad_payloads() {
    assert!(Alert::decode(&[]).is_err());
    assert!(Alert::decode(&[1]).is_err());
    assert!(Alert::decode(&[1, 0, 0]).is_err());
    assert!(AlertLevel::from_u8(0).is_err());
    assert!(AlertDescription::from_u8(1).is_err());
}

#[test]
fn protocol_version_roundtrip() {
    for v in [
        ProtocolVersion::Tls10,
        ProtocolVersion::Tls11,
        ProtocolVersion::Tls12,
        ProtocolVersion::Tls13,
    ] {
        assert_eq!(ProtocolVersion::from_u16(v.as_u16()).unwrap(), v);
        assert_eq!(v.to_be_bytes(), v.as_u16().to_be_bytes());
    }
    assert!(ProtocolVersion::from_u16(0x0305).is_err());
}

#[test]
fn content_and_handshake_types() {
    for (v, t) in [
        (20, ContentType::ChangeCipherSpec),
        (21, ContentType::Alert),
        (22, ContentType::Handshake),
        (23, ContentType::ApplicationData),
    ] {
        assert_eq!(ContentType::from_u8(v).unwrap(), t);
        assert_eq!(t.as_u8(), v);
    }
    assert!(ContentType::from_u8(99).is_err());

    assert_eq!(HandshakeType::ClientHello as u8, 1);
    assert_eq!(HandshakeType::ServerHello as u8, 2);
    assert_eq!(HandshakeType::Finished as u8, 20);
}

#[test]
fn cipher_suite_lookup() {
    let suite = CipherSuite::TlsAes128GcmSha256;
    assert_eq!(CipherSuite::from_u16(suite.as_u16()), Some(suite));
    assert_eq!(CipherSuite::from_u16(0x0000), None);
}

#[test]
fn named_group_and_signature_scheme() {
    assert_eq!(
        NamedGroup::from_u16(NamedGroup::X25519.as_u16()),
        Some(NamedGroup::X25519)
    );
    assert_eq!(
        SignatureScheme::from_u16(SignatureScheme::EcdsaSecp256r1Sha256.as_u16()),
        Some(SignatureScheme::EcdsaSecp256r1Sha256)
    );
}

#[test]
fn extension_type_known_values() {
    assert_eq!(ExtensionType::ServerName as u16, 0);
    assert_eq!(ExtensionType::SupportedVersions as u16, 43);
    assert_eq!(ExtensionType::KeyShare as u16, 51);
}

#[test]
fn tls_error_alert_mapping() {
    assert_eq!(
        TlsError::decode("x").alert_description(),
        Some(AlertDescription::DecodeError)
    );
    assert_eq!(
        TlsError::crypto("x").alert_description(),
        Some(AlertDescription::DecryptError)
    );
    assert!(TlsError::WouldBlock.alert_description().is_none());
    assert!(TlsError::Closed.to_string().contains("closed"));
}
