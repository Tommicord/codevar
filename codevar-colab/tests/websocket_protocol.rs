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

//! End-to-end WebSocket protocol tests (RFC 6455) including edge cases.

use codevar_colab::network::{
    WsClientConnection, WsCloseCode, WsConnectionConfig, WsError, WsFrame, WsMessage,
    WsOpcode, WsServerConnection,
};

fn drive_handshake(client: &mut WsClientConnection, server: &mut WsServerConnection) {
    for _ in 0..32 {
        if client.is_open() && server.is_open() {
            break;
        }
        let c2s = client.take_write();
        if !c2s.is_empty() {
            server.feed(&c2s).expect("server feed");
            server.process().expect("server process");
        }
        let s2c = server.take_write();
        if !s2c.is_empty() {
            client.feed(&s2c).expect("client feed");
            client.process().expect("client process");
        }
        if c2s.is_empty() && s2c.is_empty() {
            client.process().expect("client drain");
            server.process().expect("server drain");
            if client.is_handshaking() || server.is_handshaking() {
                panic!("handshake stalled");
            }
            break;
        }
    }
    assert!(
        client.is_open() && server.is_open(),
        "handshake did not complete"
    );
}

fn pump(client: &mut WsClientConnection, server: &mut WsServerConnection) {
    for _ in 0..16 {
        let c2s = client.take_write();
        if !c2s.is_empty() {
            server.feed(&c2s).expect("server feed");
            server.process().expect("server process");
        }
        let s2c = server.take_write();
        if !s2c.is_empty() {
            client.feed(&s2c).expect("client feed");
            client.process().expect("client process");
        }
        if c2s.is_empty() && s2c.is_empty() {
            break;
        }
    }
}

fn open_pair() -> (WsClientConnection, WsServerConnection) {
    let mut client = WsClientConnection::connect("/", "localhost", None).expect("client");
    let mut server = WsServerConnection::accept(None).expect("server");
    drive_handshake(&mut client, &mut server);
    (client, server)
}

#[test]
fn handshake_and_text_exchange() {
    let mut client =
        WsClientConnection::connect("/chat", "example.com", Some(vec!["chat".into()]))
            .expect("client");
    let mut server =
        WsServerConnection::accept(Some(vec!["chat".into(), "superchat".into()]))
            .expect("server");

    drive_handshake(&mut client, &mut server);
    assert_eq!(client.protocol(), Some("chat"));
    assert_eq!(server.protocol(), Some("chat"));
    assert_eq!(server.path(), Some("/chat"));

    client.send_text("hello").expect("send text");
    pump(&mut client, &mut server);
    match server.read_message().expect("read") {
        Some(WsMessage::Text(t)) => assert_eq!(t, "hello"),
        other => panic!("unexpected {other:?}"),
    }

    server.send_binary(b"\x00\x01\xff").expect("send binary");
    pump(&mut client, &mut server);
    match client.read_message().expect("read") {
        Some(WsMessage::Binary(b)) => assert_eq!(b, b"\x00\x01\xff"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn ping_pong_and_close() {
    let (mut client, mut server) = open_pair();

    client.send_ping(b"probe").expect("ping");
    pump(&mut client, &mut server);
    assert_eq!(
        server.read_message().expect("ping msg"),
        Some(WsMessage::Ping(b"probe".to_vec()))
    );
    pump(&mut client, &mut server);
    assert_eq!(
        client.read_message().expect("pong msg"),
        Some(WsMessage::Pong(b"probe".to_vec()))
    );

    client.close(WsCloseCode::Normal, "bye").expect("close");
    pump(&mut client, &mut server);
    match server.read_message().expect("close msg") {
        Some(WsMessage::Close { code, reason }) => {
            assert_eq!(code, WsCloseCode::Normal);
            assert_eq!(reason, "bye");
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(server.is_closed());
    pump(&mut client, &mut server);
    assert!(client.is_closed());
}

#[test]
fn fragmented_message() {
    let (mut client, mut server) = open_pair();

    let f1 = WsFrame::new(false, WsOpcode::Text, b"Hel".to_vec());
    let f2 = WsFrame::new(false, WsOpcode::Continuation, b"l".to_vec());
    let f3 = WsFrame::new(true, WsOpcode::Continuation, b"o".to_vec());
    client.send_frame(&f1).expect("f1");
    client.send_frame(&f2).expect("f2");
    client.send_frame(&f3).expect("f3");
    pump(&mut client, &mut server);
    assert_eq!(
        server.read_message().expect("msg"),
        Some(WsMessage::Text("Hello".into()))
    );
}

#[test]
fn ping_interleaved_in_fragmented_message() {
    let (mut client, mut server) = open_pair();

    client
        .send_frame(&WsFrame::new(false, WsOpcode::Text, b"ab".to_vec()))
        .unwrap();
    client.send_ping(b"keep").unwrap();
    client
        .send_frame(&WsFrame::new(true, WsOpcode::Continuation, b"cd".to_vec()))
        .unwrap();
    pump(&mut client, &mut server);

    assert_eq!(
        server.read_message().unwrap(),
        Some(WsMessage::Ping(b"keep".to_vec()))
    );
    assert_eq!(
        server.read_message().unwrap(),
        Some(WsMessage::Text("abcd".into()))
    );
    // Auto-pong delivered to client.
    pump(&mut client, &mut server);
    assert_eq!(
        client.read_message().unwrap(),
        Some(WsMessage::Pong(b"keep".to_vec()))
    );
}

#[test]
fn reject_unmasked_from_client() {
    let (mut _client, mut server) = open_pair();
    let unmasked = [0x81u8, 0x05, b'H', b'e', b'l', b'l', b'o'];
    server.feed(&unmasked).expect("feed");
    let err = server.process().expect_err("must reject unmasked");
    assert!(err.close_code().is_some());
}

#[test]
fn reject_masked_from_server() {
    let (mut client, mut _server) = open_pair();
    let masked = [
        0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d, 0x7f, 0x9f, 0x4d, 0x51, 0x58,
    ];
    client.feed(&masked).unwrap();
    let err = client.process().expect_err("must reject masked");
    assert!(err.close_code().is_some());
}

#[test]
fn invalid_utf8_text_fails_connection() {
    let (mut client, mut server) = open_pair();
    // Send a text frame with invalid UTF-8 payload (masked by client path).
    client
        .send_frame(&WsFrame::new(true, WsOpcode::Text, vec![0xff, 0xfe]))
        .unwrap();
    let wire = client.take_write();
    server.feed(&wire).unwrap();
    let err = server.process().expect_err("invalid utf8");
    assert!(
        matches!(err, WsError::InvalidUtf8)
            || err.close_code() == Some(WsCloseCode::InvalidPayloadData)
    );
}

#[test]
fn fragmented_invalid_utf8_fails_on_finish() {
    let (mut client, mut server) = open_pair();
    // Split invalid sequence: lead C3 then invalid continuation 0x20
    client
        .send_frame(&WsFrame::new(false, WsOpcode::Text, vec![0xc3]))
        .unwrap();
    client
        .send_frame(&WsFrame::new(true, WsOpcode::Continuation, vec![0x20]))
        .unwrap();
    let wire = client.take_write();
    server.feed(&wire).unwrap();
    assert!(server.process().is_err());
}

#[test]
fn message_size_limit() {
    let config = WsConnectionConfig {
        max_frame_size: 64,
        max_message_size: 32,
        auto_pong: true,
        auto_close_reply: true,
    };
    let mut client = WsClientConnection::connect_with_config(
        "/",
        "localhost",
        vec![],
        None,
        WsConnectionConfig::default(),
    )
    .unwrap();
    let mut server = WsServerConnection::accept_with_config(vec![], config).unwrap();
    drive_handshake(&mut client, &mut server);

    let big = vec![b'x'; 40];
    client.send_binary(&big).unwrap();
    let wire = client.take_write();
    server.feed(&wire).unwrap();
    let err = server.process().expect_err("too big");
    assert!(matches!(err, WsError::MessageTooBig { .. }));
}

#[test]
fn cannot_send_data_after_close() {
    let (mut client, mut server) = open_pair();
    client.close(WsCloseCode::Normal, "").unwrap();
    assert!(client.send_text("nope").is_err());
    assert!(client.send_binary(b"nope").is_err());
    pump(&mut client, &mut server);
}

#[test]
fn simultaneous_close() {
    let (mut client, mut server) = open_pair();
    client.close(WsCloseCode::Normal, "c").unwrap();
    server.close(WsCloseCode::GoingAway, "s").unwrap();
    pump(&mut client, &mut server);
    assert!(client.is_closed());
    assert!(server.is_closed());
}

#[test]
fn empty_close_payload() {
    let (mut client, mut server) = open_pair();
    client
        .send_frame(&WsFrame::close(None, "").unwrap())
        .unwrap();
    // Mark local close manually via close API is cleaner, but Frame::close(None)
    // still sends a Close frame, connection tracking uses close().
    // Use the public close with Normal instead for state, and separately test
    // peer empty close by injecting.
    let wire = client.take_write();
    server.feed(&wire).unwrap();
    server.process().unwrap();
    match server.read_message().unwrap() {
        Some(WsMessage::Close { code, reason }) => {
            assert_eq!(code, WsCloseCode::NoStatusReceived);
            assert!(reason.is_empty());
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn close_with_empty_reason_code_only() {
    let (mut client, mut server) = open_pair();
    client.close(WsCloseCode::PolicyViolation, "").unwrap();
    pump(&mut client, &mut server);
    match server.read_message().unwrap() {
        Some(WsMessage::Close { code, reason }) => {
            assert_eq!(code, WsCloseCode::PolicyViolation);
            assert!(reason.is_empty());
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn large_binary_and_unicode_text() {
    let (mut client, mut server) = open_pair();
    let bin = (0..10_000u32).map(|i| (i % 256) as u8).collect::<Vec<_>>();
    client.send_binary(&bin).unwrap();
    pump(&mut client, &mut server);
    assert_eq!(server.read_message().unwrap(), Some(WsMessage::Binary(bin)));

    let text = "协作 editing 🚀".repeat(100);
    server.send_text(&text).unwrap();
    pump(&mut client, &mut server);
    assert_eq!(client.read_message().unwrap(), Some(WsMessage::Text(text)));
}

#[test]
fn incremental_feed_of_handshake_and_frame() {
    let mut client = WsClientConnection::connect("/", "h", None).unwrap();
    let mut server = WsServerConnection::accept(None).unwrap();
    let req = client.take_write();
    // Feed request one byte at a time.
    for chunk in req.chunks(1) {
        server.feed(chunk).unwrap();
        let _ = server.process();
    }
    assert!(server.is_open());
    let resp = server.take_write();
    for chunk in resp.chunks(3) {
        client.feed(chunk).unwrap();
        let _ = client.process();
    }
    assert!(client.is_open());

    client.send_text("x").unwrap();
    let framed = client.take_write();
    for chunk in framed.chunks(2) {
        server.feed(chunk).unwrap();
        let _ = server.process();
    }
    assert_eq!(
        server.read_message().unwrap(),
        Some(WsMessage::Text("x".into()))
    );
}

#[test]
fn unexpected_continuation_fails() {
    let (mut client, mut server) = open_pair();
    client
        .send_frame(&WsFrame::new(true, WsOpcode::Continuation, b"x".to_vec()))
        .unwrap();
    let wire = client.take_write();
    server.feed(&wire).unwrap();
    assert!(server.process().is_err());
}

#[test]
fn new_data_during_fragmentation_fails() {
    let (mut client, mut server) = open_pair();
    client
        .send_frame(&WsFrame::new(false, WsOpcode::Text, b"a".to_vec()))
        .unwrap();
    client
        .send_frame(&WsFrame::new(true, WsOpcode::Binary, b"b".to_vec()))
        .unwrap();
    let wire = client.take_write();
    server.feed(&wire).unwrap();
    assert!(server.process().is_err());
}

#[test]
fn rfc_accept_key_via_public_api() {
    use codevar_colab::network::compute_accept_key;
    assert_eq!(
        compute_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
        "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    );
}

#[test]
fn message_helpers() {
    assert!(WsMessage::text("a").is_data());
    assert!(WsMessage::binary(vec![1]).is_data());
    assert!(WsMessage::Ping(vec![]).is_control());
    assert!(WsMessage::close(WsCloseCode::Normal, "").is_control());
}

#[test]
fn close_code_wire_rules() {
    assert!(WsCloseCode::from_u16(1000).is_ok());
    assert!(WsCloseCode::from_u16(1005).is_err());
    assert!(WsCloseCode::from_u16(1006).is_err());
    assert!(WsCloseCode::from_u16(1015).is_err());
    assert!(WsCloseCode::from_u16(3000).is_ok());
    assert!(!WsCloseCode::Abnormal.is_sendable());
    assert!(WsCloseCode::Normal.is_sendable());
}
