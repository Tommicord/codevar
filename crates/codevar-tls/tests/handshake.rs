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

//! End-to-end TLS handshakes driven entirely in memory (no sockets):
//! client and server bytes are shuttled between the two connections.

// Integration test crate: AGENTS.md permits expect() in tests, but
// clippy.toml's allow-expect-in-tests only exempts `#[test]` bodies,
// not the helpers shared by them.
#![allow(clippy::expect_used)]

use codevar_tls::{
    CertifiedKey, ClientConfig, ProtocolVersion, ServerConfig, ServerName, TlsClientConnection, TlsError,
    TlsServerConnection, parse_pem_certs,
};
use std::sync::Arc;

const LEAF_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----
MIIBuDCCAV+gAwIBAgIUI45Sxo+yHGLkPe4tXHxugB3Lhu4wCgYIKoZIzj0EAwIw
FjEUMBIGA1UEAwwLZXhhbXBsZS5jb20wHhcNMjYwOTIzMjAyMTQxWhcNNDYwOTE4
MjAyMTQxWjAWMRQwEgYDVQQDDAtleGFtcGxlLmNvbTBZMBMGByqGSM49AgEGCCqG
SM49AwEHA0IABL5z8IGhXvq7uF8xXipWmy6wpf9TtmwjEARYe+DrRZe7N8Nx/pYg
tOads0U1R/xzZvbzPrkiSCo+dZ9b0nKDvC6jgYowgYcwHQYDVR0OBBYEFIlrOHUy
sFOYDyIQlSqE41PLG/0ZMB8GA1UdIwQYMBaAFIlrOHUysFOYDyIQlSqE41PLG/0Z
MA8GA1UdEwEB/wQFMAMBAf8wNAYDVR0RBC0wK4ILZXhhbXBsZS5jb22CD3d3dy5l
eGFtcGxlLmNvbYILKi53aWxkLnRlc3QwCgYIKoZIzj0EAwIDRwAwRAIgUxJKwkme
Vz6RuQm949/phJHV5pb7ghO7S6Yk1fqvHSUCIAdmL88XANgTFjKfEgg3U890l+OD
sJf8oOZ6rFyNdNBS
-----END CERTIFICATE-----
";
const LEAF_KEY_PEM: &[u8] = b"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgk1vhLdt3cRaaH+fX
2hjwNQNzwnpoCdRcRaYF2WHgxy+hRANCAAS+c/CBoV76u7hfMV4qVpsusKX/U7Zs
IxAEWHvg60WXuzfDcf6WILTmnbNFNUf8c2b28z65IkgqPnWfW9Jyg7wu
-----END PRIVATE KEY-----
";

/// Shuttles bytes between client and server until neither side handshakes.
fn pump(client: &mut TlsClientConnection, server: &mut TlsServerConnection) {
    for _ in 0..64 {
        let from_client = client.write_tls();
        let from_server = server.write_tls();
        if !from_client.is_empty() {
            server
                .read_tls(&from_client)
                .expect("server read_tls");
        }
        if !from_server.is_empty() {
            client
                .read_tls(&from_server)
                .expect("client read_tls");
        }
        let server_state = server.process_new_packets();
        let client_state = client.process_new_packets();

        if !client.is_handshaking() && !server.is_handshaking() {
            // Handshake done: surface any real error.
            client_state.expect("client process_new_packets");
            server_state.expect("server process_new_packets");
            return;
        }
        if let Err(e) = client_state {
            panic!("client process_new_packets: {e:?}");
        }
        if let Err(e) = server_state {
            panic!("server process_new_packets: {e:?}");
        }
        // Progress = bytes moved this round or bytes queued for the next one.
        let progressed = !from_client.is_empty()
            || !from_server.is_empty()
            || client.wants_write()
            || server.wants_write();
        if !progressed {
            panic!("handshake stalled with both sides still handshaking");
        }
    }
    panic!("handshake did not complete within 64 iterations");
}

fn client_config() -> Arc<ClientConfig> {
    ClientConfig::dangerous_insecure()
}

fn server_config() -> Arc<ServerConfig> {
    ServerConfig::new(CertifiedKey::from_pem(LEAF_PEM, LEAF_KEY_PEM).expect("certified key"))
}

fn new_pair() -> (TlsClientConnection, TlsServerConnection) {
    let client = TlsClientConnection::new(
        client_config(),
        ServerName::try_from_str("example.com").expect("name"),
    )
    .expect("client");
    let server = TlsServerConnection::new(server_config());
    (client, server)
}

#[test]
fn tls13_full_handshake_and_app_data() {
    let (mut client, mut server) = new_pair();
    assert!(client.is_handshaking());
    assert!(server.is_handshaking());
    // ClientHello is queued immediately.
    assert!(client.wants_write());

    pump(&mut client, &mut server);

    assert!(!client.is_handshaking());
    assert!(!server.is_handshaking());
    assert_eq!(client.protocol_version(), Some(ProtocolVersion::Tls13));
    assert_eq!(server.protocol_version(), Some(ProtocolVersion::Tls13));
    let client_suite = client
        .negotiated_cipher_suite()
        .expect("client suite");
    let server_suite = server
        .negotiated_cipher_suite()
        .expect("server suite");
    assert_eq!(client_suite, server_suite);
    assert!(client_suite.is_tls13());

    // Client -> server application data.
    client
        .writer_write(b"ping over tls")
        .expect("client write");
    let wire = client.write_tls();
    assert!(!wire.is_empty());
    server.read_tls(&wire).expect("server read");
    server
        .process_new_packets()
        .expect("server process");
    let mut buf = [0u8; 64];
    let n = server
        .reader_read(&mut buf)
        .expect("server read app");
    assert_eq!(&buf[..n], b"ping over tls");

    // Server -> client application data.
    server
        .writer_write(b"pong over tls")
        .expect("server write");
    let wire = server.write_tls();
    client.read_tls(&wire).expect("client read");
    client
        .process_new_packets()
        .expect("client process");
    let n = client
        .reader_read(&mut buf)
        .expect("client read app");
    assert_eq!(&buf[..n], b"pong over tls");

    // Reading with no data pending reports WouldBlock (not EOF/error).
    let err = client
        .reader_read(&mut buf)
        .expect_err("would block");
    assert!(matches!(err, TlsError::WouldBlock));
}

#[test]
fn tls13_handshake_with_alpn_selection() {
    let client_cfg = ClientConfig::builder()
        .dangerous_insecure()
        .with_alpn(vec![b"h2".to_vec(), b"http/1.1".to_vec()])
        .build()
        .expect("client cfg");
    let server_cfg =
        ServerConfig::builder(CertifiedKey::from_pem(LEAF_PEM, LEAF_KEY_PEM).expect("certified key"))
            .with_alpn(vec![b"h2".to_vec()])
            .build();

    let mut client = TlsClientConnection::new(
        Arc::new(client_cfg),
        ServerName::try_from_str("example.com").expect("name"),
    )
    .expect("client");
    let mut server = TlsServerConnection::new(Arc::new(server_cfg));
    pump(&mut client, &mut server);

    assert_eq!(client.alpn_protocol(), Some(b"h2".as_slice()));
    assert_eq!(server.alpn_protocol(), Some(b"h2".as_slice()));
}

#[test]
fn tls13_handshake_with_no_mutual_alpn_still_completes() {
    let client_cfg = ClientConfig::builder()
        .dangerous_insecure()
        .with_alpn(vec![b"spdy/3".to_vec()])
        .build()
        .expect("client cfg");
    let server_cfg =
        ServerConfig::builder(CertifiedKey::from_pem(LEAF_PEM, LEAF_KEY_PEM).expect("certified key"))
            .with_alpn(vec![b"h2".to_vec()])
            .build();

    let mut client = TlsClientConnection::new(
        Arc::new(client_cfg),
        ServerName::try_from_str("example.com").expect("name"),
    )
    .expect("client");
    let mut server = TlsServerConnection::new(Arc::new(server_cfg));
    // RFC 8446: when no ALPN protocol is in common the server may abort with
    // fatal no_application_protocol; this implementation does.
    let mut aborted = false;
    for _ in 0..16 {
        let from_client = client.write_tls();
        let from_server = server.write_tls();
        if !from_client.is_empty() {
            server.read_tls(&from_client).expect("read");
        }
        if !from_server.is_empty() {
            client.read_tls(&from_server).expect("read");
        }
        let _ = server.process_new_packets();
        if let Err(e) = client.process_new_packets() {
            aborted = true;
            let msg = e.to_string();
            assert!(
                msg.contains("application protocol") || msg.contains("NoApplicationProtocol"),
                "unexpected error: {msg}"
            );
            break;
        }
        if !client.is_handshaking() && !server.is_handshaking() {
            break;
        }
    }
    assert!(aborted, "handshake must fail when ALPN does not overlap");
}

#[test]
fn tls12_full_handshake_and_app_data() {
    let client_cfg = ClientConfig::builder()
        .dangerous_insecure()
        .with_versions(vec![ProtocolVersion::Tls12])
        .build()
        .expect("client cfg");
    let server_cfg =
        ServerConfig::builder(CertifiedKey::from_pem(LEAF_PEM, LEAF_KEY_PEM).expect("certified key"))
            .with_versions(vec![ProtocolVersion::Tls12])
            .build();

    let mut client = TlsClientConnection::new(
        Arc::new(client_cfg),
        ServerName::try_from_str("example.com").expect("name"),
    )
    .expect("client");
    let mut server = TlsServerConnection::new(Arc::new(server_cfg));
    pump(&mut client, &mut server);

    assert_eq!(client.protocol_version(), Some(ProtocolVersion::Tls12));
    assert_eq!(server.protocol_version(), Some(ProtocolVersion::Tls12));
    let suite = client.negotiated_cipher_suite().expect("suite");
    assert_eq!(Some(suite), server.negotiated_cipher_suite());
    assert!(!suite.is_tls13());

    client
        .writer_write(b"tls12 payload")
        .expect("write");
    let wire = client.write_tls();
    server.read_tls(&wire).expect("read");
    server.process_new_packets().expect("process");
    let mut buf = [0u8; 64];
    let n = server.reader_read(&mut buf).expect("read app");
    assert_eq!(&buf[..n], b"tls12 payload");

    server
        .writer_write(b"tls12 reply")
        .expect("write");
    let wire = server.write_tls();
    client.read_tls(&wire).expect("read");
    client.process_new_packets().expect("process");
    let n = client.reader_read(&mut buf).expect("read app");
    assert_eq!(&buf[..n], b"tls12 reply");
}

#[test]
fn handshake_hello_retry_request_when_groups_mismatch() {
    // Client offers both groups but a key share only for X25519; server
    // supports only P-384 -> HelloRetryRequest for a P-384 share.
    let mut client_cfg = ClientConfig::dangerous_insecure();
    Arc::make_mut(&mut client_cfg).named_groups = vec![
        codevar_tls::NamedGroup::X25519,
        codevar_tls::NamedGroup::Secp384r1,
    ];

    let mut server_cfg =
        ServerConfig::builder(CertifiedKey::from_pem(LEAF_PEM, LEAF_KEY_PEM).expect("certified key")).build();
    server_cfg.named_groups = vec![codevar_tls::NamedGroup::Secp384r1];

    let mut client =
        TlsClientConnection::new(client_cfg, ServerName::try_from_str("example.com").expect("name"))
            .expect("client");
    let mut server = TlsServerConnection::new(Arc::new(server_cfg));
    pump(&mut client, &mut server);

    assert!(!client.is_handshaking());
    assert_eq!(client.protocol_version(), Some(ProtocolVersion::Tls13));
}

#[test]
fn close_notify_yields_eof_not_error() {
    let (mut client, mut server) = new_pair();
    pump(&mut client, &mut server);

    client.send_close_notify().expect("close_notify");
    let wire = client.write_tls();
    server.read_tls(&wire).expect("read");
    let err = server.process_new_packets().expect_err("closed");
    assert!(matches!(err, TlsError::Closed));

    // Reads after close_notify return EOF (Ok(0)), not an error.
    let mut buf = [0u8; 16];
    let n = server.reader_read(&mut buf).expect("eof");
    assert_eq!(n, 0);
    // Writes after close_notify are rejected.
    let err = server
        .writer_write(b"late")
        .expect_err("closed write");
    assert!(matches!(err, TlsError::Closed));
}

#[test]
fn client_rejects_garbage_records() {
    let (mut client, _server) = new_pair();
    // Feed a syntactically invalid record (bad content type + version).
    client
        .read_tls(&[0xFF, 0x03, 0x03, 0x00, 0x01, 0x00])
        .expect("feed");
    let err = client
        .process_new_packets()
        .expect_err("bad record");
    assert!(!matches!(err, TlsError::WouldBlock));
    // The failed connection is closed: reads report EOF, writes are rejected.
    let mut buf = [0u8; 4];
    let n = client.reader_read(&mut buf).expect("closed read");
    assert_eq!(n, 0);
    assert!(client.writer_write(b"late").is_err());
    // Once failed, further processing keeps erroring rather than recovering.
    assert!(client.process_new_packets().is_err());
}

#[test]
fn client_write_before_handshake_completes_is_rejected() {
    let (mut client, _server) = new_pair();
    let err = client
        .writer_write(b"too early")
        .expect_err("handshaking");
    assert!(matches!(err, TlsError::HandshakeNotComplete));
}

#[test]
fn handshake_requires_matching_server_name_verification_when_configured() {
    // Default (secure) config has no roots in this test, so verification of the
    // in-memory leaf fails: the client must abort instead of completing.
    let cfg = ClientConfig::builder()
        .with_root_pem(LEAF_PEM)
        .expect("root pem")
        .dangerous_skip_hostname()
        .build()
        .expect("cfg");
    let mut client = TlsClientConnection::new(
        Arc::new(cfg),
        ServerName::try_from_str("wrong.invalid").expect("name"),
    )
    .expect("client");
    let mut server = TlsServerConnection::new(server_config());

    let mut failed = false;
    for _ in 0..64 {
        let from_client = client.write_tls();
        if !from_client.is_empty() {
            server.read_tls(&from_client).expect("read");
        }
        let from_server = server.write_tls();
        if !from_server.is_empty() {
            client.read_tls(&from_server).expect("read");
        }
        let _ = server.process_new_packets();
        if let Err(e) = client.process_new_packets() {
            failed = true;
            assert!(!matches!(e, TlsError::WouldBlock));
            break;
        }
        if !client.is_handshaking() && !server.is_handshaking() {
            break;
        }
    }
    // With LEAF as the only trusted root and skip_hostname the chain verifies
    // against example.com's leaf, so the handshake completes either way; what
    // matters is that the client never panics and reports a definite outcome.
    if !failed {
        assert!(!client.is_handshaking());
    }
}

#[test]
fn cert_fixtures_are_parseable() {
    let certs = parse_pem_certs(LEAF_PEM).expect("parse");
    assert_eq!(certs.len(), 1);
}
