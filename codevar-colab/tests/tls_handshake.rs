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

//! End-to-end TLS handshake tests for the src TLS stack.

use codevar_colab::network::{
    CertifiedKey, ClientConfig, ProtocolVersion, ServerConfig, ServerName,
    TlsClientConnection, TlsServerConnection,
};
use rcgen::{CertificateParams, KeyPair};
use std::sync::Arc;

fn generate_self_signed() -> (Vec<u8>, Vec<u8>) {
    let key_pair = KeyPair::generate().expect("keygen");
    let mut params = CertificateParams::new(vec!["localhost".into()]).expect("params");
    params.is_ca = rcgen::IsCa::NoCa;
    let cert = params.self_signed(&key_pair).expect("self-signed");
    (
        cert.pem().into_bytes(),
        key_pair.serialize_pem().into_bytes(),
    )
}

fn drive_handshake(client: &mut TlsClientConnection, server: &mut TlsServerConnection) {
    for _ in 0..64 {
        if !client.is_handshaking() && !server.is_handshaking() {
            break;
        }
        let c2s = client.write_tls();
        if !c2s.is_empty() {
            server.read_tls(&c2s).expect("server read_tls");
            server.process_new_packets().expect("server process");
        }
        let s2c = server.write_tls();
        if !s2c.is_empty() {
            client.read_tls(&s2c).expect("client read_tls");
            client.process_new_packets().expect("client process");
        }
        if c2s.is_empty() && s2c.is_empty() {
            client.process_new_packets().expect("client drain");
            server.process_new_packets().expect("server drain");
            let more_c = client.write_tls();
            let more_s = server.write_tls();
            if more_c.is_empty() && more_s.is_empty() {
                if client.is_handshaking() || server.is_handshaking() {
                    panic!("handshake stalled");
                }
                break;
            }
            if !more_c.is_empty() {
                server.read_tls(&more_c).expect("server read more");
                server.process_new_packets().expect("server process more");
            }
            if !more_s.is_empty() {
                client.read_tls(&more_s).expect("client read more");
                client.process_new_packets().expect("client process more");
            }
        }
    }
    assert!(
        !client.is_handshaking() && !server.is_handshaking(),
        "handshake did not complete"
    );
}

fn exchange_appdata(client: &mut TlsClientConnection, server: &mut TlsServerConnection) {
    let msg = b"hello collaboration";
    client.writer_write(msg).expect("client write");
    let c2s = client.write_tls();
    server.read_tls(&c2s).expect("server read");
    server.process_new_packets().expect("server process app");
    let mut buf = [0u8; 64];
    let n = server.reader_read(&mut buf).expect("server app read");
    assert_eq!(&buf[..n], msg);

    let reply = b"ack";
    server.writer_write(reply).expect("server write");
    let s2c = server.write_tls();
    client.read_tls(&s2c).expect("client read");
    client.process_new_packets().expect("client process app");
    let n = client.reader_read(&mut buf).expect("client app read");
    assert_eq!(&buf[..n], reply);
}

#[test]
fn tls13_handshake_and_appdata() {
    let (cert_pem, key_pem) = generate_self_signed();
    let certified = CertifiedKey::from_pem(&cert_pem, &key_pem).expect("certified key");
    let server_cfg = Arc::new(
        ServerConfig::builder(certified)
            .with_versions(vec![ProtocolVersion::Tls13])
            .build(),
    );
    let client_cfg = Arc::new(
        ClientConfig::builder()
            .dangerous_insecure()
            .with_versions(vec![ProtocolVersion::Tls13])
            .build()
            .expect("client"),
    );

    let mut client = TlsClientConnection::new(
        client_cfg,
        ServerName::try_from_str("localhost").expect("sni"),
    )
    .expect("client conn");
    let mut server = TlsServerConnection::new(server_cfg);

    drive_handshake(&mut client, &mut server);
    assert_eq!(client.protocol_version(), Some(ProtocolVersion::Tls13));
    assert_eq!(server.protocol_version(), Some(ProtocolVersion::Tls13));
    exchange_appdata(&mut client, &mut server);
}

#[test]
fn tls12_handshake_and_appdata() {
    let (cert_pem, key_pem) = generate_self_signed();
    let certified = CertifiedKey::from_pem(&cert_pem, &key_pem).expect("certified key");
    let server_cfg = Arc::new(
        ServerConfig::builder(certified)
            .with_versions(vec![ProtocolVersion::Tls12])
            .build(),
    );
    let client_cfg = Arc::new(
        ClientConfig::builder()
            .dangerous_insecure()
            .with_versions(vec![ProtocolVersion::Tls12])
            .build()
            .expect("client"),
    );

    let mut client = TlsClientConnection::new(
        client_cfg,
        ServerName::try_from_str("localhost").expect("sni"),
    )
    .expect("client conn");
    let mut server = TlsServerConnection::new(server_cfg);

    drive_handshake(&mut client, &mut server);
    assert_eq!(client.protocol_version(), Some(ProtocolVersion::Tls12));
    assert_eq!(server.protocol_version(), Some(ProtocolVersion::Tls12));
    exchange_appdata(&mut client, &mut server);
}

#[test]
fn prefer_tls13_when_both_offered() {
    let (cert_pem, key_pem) = generate_self_signed();
    let certified = CertifiedKey::from_pem(&cert_pem, &key_pem).expect("certified key");
    let server_cfg = ServerConfig::new(certified);
    let client_cfg = ClientConfig::dangerous_insecure();

    let mut client = TlsClientConnection::new(
        client_cfg,
        ServerName::try_from_str("localhost").expect("sni"),
    )
    .expect("client");
    let mut server = TlsServerConnection::new(server_cfg);
    drive_handshake(&mut client, &mut server);
    assert_eq!(client.protocol_version(), Some(ProtocolVersion::Tls13));
}

#[test]
fn alpn_negotiation() {
    let (cert_pem, key_pem) = generate_self_signed();
    let certified = CertifiedKey::from_pem(&cert_pem, &key_pem).expect("certified key");
    let server_cfg = Arc::new(
        ServerConfig::builder(certified)
            .with_alpn(vec![b"h2".to_vec(), b"http/1.1".to_vec()])
            .with_versions(vec![ProtocolVersion::Tls13])
            .build(),
    );
    let client_cfg = Arc::new(
        ClientConfig::builder()
            .dangerous_insecure()
            .with_alpn(vec![b"http/1.1".to_vec(), b"h2".to_vec()])
            .with_versions(vec![ProtocolVersion::Tls13])
            .build()
            .expect("client"),
    );

    let mut client = TlsClientConnection::new(
        client_cfg,
        ServerName::try_from_str("localhost").expect("sni"),
    )
    .expect("client");
    let mut server = TlsServerConnection::new(server_cfg);
    drive_handshake(&mut client, &mut server);

    // Server preference order wins: h2 before http/1.1 among client's offers.
    assert_eq!(client.alpn_protocol(), Some(b"h2".as_slice()));
    assert_eq!(server.alpn_protocol(), Some(b"h2".as_slice()));
}

#[test]
fn close_notify_and_empty_appdata() {
    let (cert_pem, key_pem) = generate_self_signed();
    let certified = CertifiedKey::from_pem(&cert_pem, &key_pem).expect("certified key");
    let server_cfg = Arc::new(
        ServerConfig::builder(certified)
            .with_versions(vec![ProtocolVersion::Tls13])
            .build(),
    );
    let client_cfg = Arc::new(
        ClientConfig::builder()
            .dangerous_insecure()
            .with_versions(vec![ProtocolVersion::Tls13])
            .build()
            .expect("client"),
    );

    let mut client = TlsClientConnection::new(
        client_cfg,
        ServerName::try_from_str("localhost").expect("sni"),
    )
    .expect("client");
    let mut server = TlsServerConnection::new(server_cfg);
    drive_handshake(&mut client, &mut server);

    // Empty write should succeed (no-op or empty record depending on impl).
    client.writer_write(b"").expect("empty write");

    let large = vec![0x5au8; 16_000];
    client.writer_write(&large).expect("large write");
    let c2s = client.write_tls();
    server.read_tls(&c2s).expect("server read");
    server.process_new_packets().expect("server process");
    let mut buf = vec![0u8; 20_000];
    let n = server.reader_read(&mut buf).expect("read large");
    assert_eq!(&buf[..n], large.as_slice());

    client.send_close_notify().expect("close_notify");
    let c2s = client.write_tls();
    server.read_tls(&c2s).expect("feed close");
    let err = server.process_new_packets();
    // Peer close_notify surfaces as Closed (or succeeds then Closed on read).
    match err {
        Ok(_) => {
            let mut tmp = [0u8; 1];
            let read_err = server.reader_read(&mut tmp);
            assert!(read_err.is_err());
        }
        Err(e) => {
            assert!(
                e.to_string().contains("closed") || e.to_string().contains("Closed"),
                "unexpected error: {e}"
            );
        }
    }
}

#[test]
fn bidirectional_multiple_messages() {
    let (cert_pem, key_pem) = generate_self_signed();
    let certified = CertifiedKey::from_pem(&cert_pem, &key_pem).expect("certified key");
    let server_cfg = Arc::new(
        ServerConfig::builder(certified)
            .with_versions(vec![ProtocolVersion::Tls13])
            .build(),
    );
    let client_cfg = Arc::new(
        ClientConfig::builder()
            .dangerous_insecure()
            .with_versions(vec![ProtocolVersion::Tls13])
            .build()
            .expect("client"),
    );

    let mut client = TlsClientConnection::new(
        client_cfg,
        ServerName::try_from_str("localhost").expect("sni"),
    )
    .expect("client");
    let mut server = TlsServerConnection::new(server_cfg);
    drive_handshake(&mut client, &mut server);

    for i in 0..5u8 {
        let msg = [i; 32];
        client.writer_write(&msg).unwrap();
        let wire = client.write_tls();
        server.read_tls(&wire).unwrap();
        server.process_new_packets().unwrap();
        let mut buf = [0u8; 32];
        assert_eq!(server.reader_read(&mut buf).unwrap(), 32);
        assert_eq!(buf, msg);

        server.writer_write(&[i.wrapping_add(1); 8]).unwrap();
        let wire = server.write_tls();
        client.read_tls(&wire).unwrap();
        client.process_new_packets().unwrap();
        let mut buf = [0u8; 8];
        assert_eq!(client.reader_read(&mut buf).unwrap(), 8);
        assert_eq!(buf, [i.wrapping_add(1); 8]);
    }
}

#[test]
fn version_mismatch_tls13_client_tls12_only_server() {
    let (cert_pem, key_pem) = generate_self_signed();
    let certified = CertifiedKey::from_pem(&cert_pem, &key_pem).expect("certified key");
    let server_cfg = Arc::new(
        ServerConfig::builder(certified)
            .with_versions(vec![ProtocolVersion::Tls12])
            .build(),
    );
    let client_cfg = Arc::new(
        ClientConfig::builder()
            .dangerous_insecure()
            .with_versions(vec![ProtocolVersion::Tls13])
            .build()
            .expect("client"),
    );

    let mut client = TlsClientConnection::new(
        client_cfg,
        ServerName::try_from_str("localhost").expect("sni"),
    )
    .expect("client");
    let mut server = TlsServerConnection::new(server_cfg);

    let mut failed = false;
    for _ in 0..32 {
        let c2s = client.write_tls();
        if !c2s.is_empty() {
            let _ = server.read_tls(&c2s);
            if server.process_new_packets().is_err() {
                failed = true;
                break;
            }
        }
        let s2c = server.write_tls();
        if !s2c.is_empty() {
            let _ = client.read_tls(&s2c);
            if client.process_new_packets().is_err() {
                failed = true;
                break;
            }
        }
        if c2s.is_empty() && s2c.is_empty() {
            break;
        }
    }
    assert!(
        failed || client.is_handshaking() || server.is_handshaking(),
        "expected handshake failure on version mismatch"
    );
}
