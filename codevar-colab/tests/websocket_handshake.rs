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

//! WebSocket opening-handshake unit tests (RFC 6455 §4).

use codevar_colab::network::ws_handshake::{try_parse_request, try_parse_response};
use codevar_colab::network::{
    VERSION, WsClientHandshake, WsServerHandshake, accept_key_from_nonce,
    compute_accept_key, generate_key_nonce,
};

#[test]
fn rfc6455_accept_key_example() {
    assert_eq!(
        compute_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
        "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    );
}

#[test]
fn accept_key_from_nonce_matches_compute() {
    let nonce = generate_key_nonce().expect("nonce");
    let from_nonce = accept_key_from_nonce(&nonce);
    let b64 = codevar_colab::network::ws_base64::encode(&nonce);
    assert_eq!(from_nonce, compute_accept_key(&b64));
}

#[test]
fn client_request_contains_required_headers() {
    let hs =
        WsClientHandshake::new("/chat", "server.example.com", vec!["chat".into()], None)
            .expect("hs");
    let bytes = hs.encode_request();
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(text.starts_with("GET /chat HTTP/1.1\r\n"));
    assert!(text.contains("Host: server.example.com\r\n"));
    assert!(text.contains("Upgrade: websocket\r\n"));
    assert!(text.contains("Connection: Upgrade\r\n"));
    assert!(text.contains(&format!("Sec-WebSocket-Key: {}\r\n", hs.key_b64)));
    assert!(text.contains(&format!("Sec-WebSocket-Version: {VERSION}\r\n")));
    assert!(text.contains("Sec-WebSocket-Protocol: chat\r\n"));
    assert!(text.ends_with("\r\n\r\n"));

    let (req, n) = try_parse_request(&bytes).unwrap().unwrap();
    assert_eq!(n, bytes.len());
    assert_eq!(req.path, "/chat");
    assert_eq!(req.host, "server.example.com");
    assert_eq!(req.key, hs.key_b64);
    assert_eq!(req.protocols, vec!["chat"]);
}

#[test]
fn client_request_with_origin() {
    let hs = WsClientHandshake::new("/", "ex.com", vec![], Some("https://ex.com".into()))
        .unwrap();
    let bytes = hs.encode_request();
    let (req, _) = try_parse_request(&bytes).unwrap().unwrap();
    assert_eq!(req.origin.as_deref(), Some("https://ex.com"));
}

#[test]
fn client_rejects_invalid_path() {
    assert!(WsClientHandshake::new("chat", "h", vec![], None).is_err());
    assert!(WsClientHandshake::new("", "h", vec![], None).is_err());
}

#[test]
fn partial_request_returns_none() {
    let partial = b"GET / HTTP/1.1\r\nHost: x\r\n";
    assert!(try_parse_request(partial).unwrap().is_none());
}

#[test]
fn reject_non_get_and_http10() {
    let bad_method = b"POST / HTTP/1.1\r\nHost: x\r\n\r\n";
    assert!(try_parse_request(bad_method).is_err());
    let bad_ver = b"GET / HTTP/1.0\r\nHost: x\r\n\r\n";
    assert!(try_parse_request(bad_ver).is_err());
}

#[test]
fn server_accept_and_client_validate_roundtrip() {
    let mut client =
        WsClientHandshake::new("/ws", "example.com", vec!["a".into(), "b".into()], None)
            .unwrap();
    let req_bytes = client.encode_request();
    let (req, _) = try_parse_request(&req_bytes).unwrap().unwrap();

    let mut server = WsServerHandshake::new(vec!["b".into(), "c".into()]);
    server.accept_request(req).unwrap();
    // Client preference order: first mutually supported is "b".
    assert_eq!(server.selected_protocol.as_deref(), Some("b"));

    let resp_bytes = server.encode_response().unwrap();
    let (resp, _) = try_parse_response(&resp_bytes).unwrap().unwrap();
    assert_eq!(resp.status, 101);
    client.validate_response(&resp).unwrap();
    assert_eq!(client.selected_protocol.as_deref(), Some("b"));
}

#[test]
fn server_rejects_wrong_version() {
    let mut client = WsClientHandshake::new("/", "h", vec![], None).unwrap();
    let mut req_bytes = client.encode_request();
    // Corrupt version header in the raw request for the parser path via
    // a handcrafted request instead.
    let bad = format!(
        "GET / HTTP/1.1\r\nHost: h\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
         Sec-WebSocket-Key: {}\r\nSec-WebSocket-Version: 8\r\n\r\n",
        client.key_b64
    );
    let (req, _) = try_parse_request(bad.as_bytes()).unwrap().unwrap();
    let mut server = WsServerHandshake::new(vec![]);
    let err = server.accept_request(req).unwrap_err();
    assert!(err.to_string().contains("Version"));
    let _ = &mut client; // silence
}

#[test]
fn server_rejects_bad_key_length() {
    let bad =
        b"GET / HTTP/1.1\r\nHost: h\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
Sec-WebSocket-Key: dG9vLXNob3J0\r\nSec-WebSocket-Version: 13\r\n\r\n";
    let (req, _) = try_parse_request(bad).unwrap().unwrap();
    let mut server = WsServerHandshake::new(vec![]);
    assert!(server.accept_request(req).is_err());
}

#[test]
fn server_rejects_missing_upgrade() {
    let bad = b"GET / HTTP/1.1\r\nHost: h\r\nConnection: Upgrade\r\n\
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";
    let (req, _) = try_parse_request(bad).unwrap().unwrap();
    let mut server = WsServerHandshake::new(vec![]);
    assert!(server.accept_request(req).is_err());
}

#[test]
fn client_rejects_wrong_accept_and_status() {
    let mut client = WsClientHandshake::new("/", "h", vec![], None).unwrap();
    let mut resp = codevar_colab::network::WsHandshakeResponse {
        status: 200,
        accept: client.expected_accept.clone(),
        protocol: None,
        headers: std::collections::HashMap::from([
            ("upgrade".into(), "websocket".into()),
            ("connection".into(), "Upgrade".into()),
            (
                "sec-websocket-accept".into(),
                client.expected_accept.clone(),
            ),
        ]),
    };
    assert!(client.validate_response(&resp).is_err());

    resp.status = 101;
    resp.accept = "not-the-right-accept============".into();
    resp.headers
        .insert("sec-websocket-accept".into(), resp.accept.clone());
    assert!(client.validate_response(&resp).is_err());
}

#[test]
fn client_rejects_unoffered_subprotocol() {
    let mut client = WsClientHandshake::new("/", "h", vec!["chat".into()], None).unwrap();
    let resp = codevar_colab::network::WsHandshakeResponse {
        status: 101,
        accept: client.expected_accept.clone(),
        protocol: Some("other".into()),
        headers: std::collections::HashMap::from([
            ("upgrade".into(), "websocket".into()),
            ("connection".into(), "Upgrade".into()),
            (
                "sec-websocket-accept".into(),
                client.expected_accept.clone(),
            ),
            ("sec-websocket-protocol".into(), "other".into()),
        ]),
    };
    assert!(client.validate_response(&resp).is_err());
}

#[test]
fn client_rejects_unoffered_extensions() {
    let mut client = WsClientHandshake::new("/", "h", vec![], None).unwrap();
    let resp = codevar_colab::network::WsHandshakeResponse {
        status: 101,
        accept: client.expected_accept.clone(),
        protocol: None,
        headers: std::collections::HashMap::from([
            ("upgrade".into(), "websocket".into()),
            ("connection".into(), "Upgrade".into()),
            (
                "sec-websocket-accept".into(),
                client.expected_accept.clone(),
            ),
            (
                "sec-websocket-extensions".into(),
                "permessage-deflate".into(),
            ),
        ]),
    };
    assert!(client.validate_response(&resp).is_err());
}

#[test]
fn version_rejection_response_format() {
    let bytes = WsServerHandshake::encode_version_rejection();
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.starts_with("HTTP/1.1 400 Bad Request\r\n"));
    assert!(text.contains(&format!("Sec-WebSocket-Version: {VERSION}\r\n")));
}

#[test]
fn duplicate_headers_are_combined() {
    let raw = b"GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\nUpgrade: websocket\r\n\
Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
Sec-WebSocket-Version: 13\r\n\r\n";
    let (req, _) = try_parse_request(raw).unwrap().unwrap();
    assert_eq!(req.host, "a, b");
}

#[test]
fn connection_header_token_case_insensitive() {
    let raw = b"GET / HTTP/1.1\r\nHost: h\r\nUpgrade: WebSocket\r\n\
Connection: keep-alive, Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
Sec-WebSocket-Version: 13\r\n\r\n";
    let (req, _) = try_parse_request(raw).unwrap().unwrap();
    let mut server = WsServerHandshake::new(vec![]);
    server.accept_request(req).unwrap();
}
