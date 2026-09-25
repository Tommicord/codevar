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

//! HTTP/1.1 opening handshake (RFC 6455 §4).
//!
//! # Client requirements (RFC 6455 §4.1)
//!
//! The client sends a GET request with:
//!
//! - `Upgrade: websocket`
//! - `Connection: Upgrade`
//! - `Sec-WebSocket-Key` — Base64 of 16 random bytes
//! - `Sec-WebSocket-Version: 13`
//! - `Host` — authority of the target
//! - optional `Sec-WebSocket-Protocol`, `Origin`, `Sec-WebSocket-Extensions`
//!
//! The server response MUST be `101 Switching Protocols` with a matching
//! `Sec-WebSocket-Accept` derived as:
//!
//! ```text
//! base64( SHA1( Sec-WebSocket-Key || GUID ) )
//! ```
//!
//! where `GUID` is [`GUID`](crate::ws_ids::GUID).

use crate::ws_base64;
use crate::ws_error::{WsError, WsResult};
use crate::ws_ids::{GUID, VERSION};
use sha1::{Digest, Sha1};
use std::collections::HashMap;

/// Generates 16 cryptographically random bytes for `Sec-WebSocket-Key`.
pub fn generate_key_nonce() -> WsResult<[u8; 16]> {
    let mut nonce = [0u8; 16];
    getrandom::getrandom(&mut nonce).map_err(|_| WsError::RandomFailed)?;
    Ok(nonce)
}

/// Computes `Sec-WebSocket-Accept` from the raw 16-byte key nonce.
#[must_use]
pub fn accept_key_from_nonce(nonce: &[u8; 16]) -> String {
    compute_accept_key(&ws_base64::encode(nonce))
}

/// Computes `Sec-WebSocket-Accept` from the Base64 `Sec-WebSocket-Key` value.
///
/// Per RFC 6455 §4.2.2: `base64( SHA-1( key || GUID ) )`.
#[must_use]
pub fn compute_accept_key(sec_websocket_key: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(sec_websocket_key.as_bytes());
    hasher.update(GUID.as_bytes());
    let digest = hasher.finalize();
    ws_base64::encode(&digest)
}

/// Parsed client opening-handshake request.
#[derive(Debug, Clone)]
pub struct HandshakeRequest {
    /// Request-target path (e.g. `"/chat"`).
    pub path: String,
    /// `Host` header value.
    pub host: String,
    /// Raw Base64 `Sec-WebSocket-Key`.
    pub key: String,
    /// Requested subprotocols, in preference order.
    pub protocols: Vec<String>,
    /// Optional `Origin` header.
    pub origin: Option<String>,
    /// All headers (lower-cased names) for extension inspection.
    pub headers: HashMap<String, String>,
}

/// Parsed server opening-handshake response.
#[derive(Debug, Clone)]
pub struct WsHandshakeResponse {
    /// HTTP status code (must be 101 for success).
    pub status: u16,
    /// `Sec-WebSocket-Accept` value.
    pub accept: String,
    /// Selected subprotocol, if any.
    pub protocol: Option<String>,
    /// All headers (lower-cased names).
    pub headers: HashMap<String, String>,
}

/// Client-side handshake builder / validator.
#[derive(Debug, Clone)]
pub struct WsClientHandshake {
    /// Request-target path.
    pub path: String,
    /// `Host` header.
    pub host: String,
    /// Raw 16-byte nonce (before Base64).
    pub nonce: [u8; 16],
    /// Base64-encoded key sent on the wire.
    pub key_b64: String,
    /// Expected `Sec-WebSocket-Accept`.
    pub expected_accept: String,
    /// Offered subprotocols.
    pub protocols: Vec<String>,
    /// Optional `Origin`.
    pub origin: Option<String>,
    /// Selected subprotocol after a successful response.
    pub selected_protocol: Option<String>,
}

impl WsClientHandshake {
    /// Builds a client handshake for `path` on `host`.
    ///
    /// `protocols` are offered via `Sec-WebSocket-Protocol` (comma-separated).
    pub fn new(
        path: impl Into<String>,
        host: impl Into<String>,
        protocols: Vec<String>,
        origin: Option<String>,
    ) -> WsResult<Self> {
        let path = path.into();
        let host = host.into();
        if path.is_empty() || !path.starts_with('/') {
            return Err(WsError::handshake(
                "request path must be non-empty and begin with '/'",
            ));
        }
        let nonce = generate_key_nonce()?;
        let key_b64 = ws_base64::encode(&nonce);
        let expected_accept = compute_accept_key(&key_b64);
        Ok(Self {
            path,
            host,
            nonce,
            key_b64,
            expected_accept,
            protocols,
            origin,
            selected_protocol: None,
        })
    }

    /// Serializes the HTTP Upgrade request.
    #[must_use]
    pub fn encode_request(&self) -> Vec<u8> {
        let mut req = String::new();
        req.push_str(&format!("GET {} HTTP/1.1\r\n", self.path));
        req.push_str(&format!("Host: {}\r\n", self.host));
        req.push_str("Upgrade: websocket\r\n");
        req.push_str("Connection: Upgrade\r\n");
        req.push_str(&format!("Sec-WebSocket-Key: {}\r\n", self.key_b64));
        req.push_str(&format!("Sec-WebSocket-Version: {VERSION}\r\n"));
        if let Some(ref origin) = self.origin {
            req.push_str(&format!("Origin: {origin}\r\n"));
        }
        if !self
            .protocols
            .is_empty()
        {
            req.push_str("Sec-WebSocket-Protocol: ");
            req.push_str(
                &self
                    .protocols
                    .join(", "),
            );
            req.push_str("\r\n");
        }
        req.push_str("\r\n");
        req.into_bytes()
    }

    /// Validates a server response against this handshake.
    pub fn validate_response(&mut self, response: &WsHandshakeResponse) -> WsResult<()> {
        if response.status != 101 {
            return Err(WsError::handshake(format!(
                "expected 101 Switching Protocols, got {}",
                response.status
            )));
        }
        let upgrade = response
            .headers
            .get("upgrade")
            .map(|s| s.as_str())
            .unwrap_or("");
        if !eq_ignore_ascii_case(upgrade, "websocket") {
            return Err(WsError::handshake("response missing Upgrade: websocket"));
        }
        let connection = response
            .headers
            .get("connection")
            .map(|s| s.as_str())
            .unwrap_or("");
        if !header_token_contains(connection, "Upgrade") {
            return Err(WsError::handshake("response missing Connection: Upgrade"));
        }
        if !eq_ignore_ascii_case(&response.accept, &self.expected_accept) {
            return Err(WsError::handshake("Sec-WebSocket-Accept mismatch"));
        }
        if let Some(ref proto) = response.protocol {
            if !self
                .protocols
                .iter()
                .any(|p| p == proto)
            {
                return Err(WsError::handshake(format!(
                    "server selected unoffered subprotocol '{proto}'"
                )));
            }
            self.selected_protocol = Some(proto.clone());
        } else if response
            .headers
            .contains_key("sec-websocket-protocol")
        {
            return Err(WsError::handshake("empty Sec-WebSocket-Protocol in response"));
        }
        // Reject unnegotiated extensions (we offer none).
        if let Some(ext) = response
            .headers
            .get("sec-websocket-extensions")
            && !ext
                .trim()
                .is_empty()
        {
            return Err(WsError::handshake(format!(
                "server selected unoffered extension '{ext}'"
            )));
        }
        Ok(())
    }
}

/// Server-side handshake acceptor.
#[derive(Debug, Clone)]
pub struct WsServerHandshake {
    /// Subprotocols the server is willing to select (preference order).
    pub supported_protocols: Vec<String>,
    /// Parsed request, filled after a successful `accept_request`.
    pub request: Option<HandshakeRequest>,
    /// Selected subprotocol.
    pub selected_protocol: Option<String>,
}

impl WsServerHandshake {
    /// Creates a server handshake with optional supported subprotocols.
    #[must_use]
    pub fn new(supported_protocols: Vec<String>) -> Self {
        Self {
            supported_protocols,
            request: None,
            selected_protocol: None,
        }
    }

    /// Validates a client request and prepares the 101 response parameters.
    pub fn accept_request(&mut self, request: HandshakeRequest) -> WsResult<()> {
        // Version
        let version = request
            .headers
            .get("sec-websocket-version")
            .map(String::as_str)
            .unwrap_or("");
        if version != VERSION.to_string() {
            return Err(WsError::handshake(format!(
                "unsupported Sec-WebSocket-Version '{version}' (need {VERSION})"
            )));
        }
        // Upgrade
        let upgrade = request
            .headers
            .get("upgrade")
            .map(String::as_str)
            .unwrap_or("");
        if !eq_ignore_ascii_case(upgrade, "websocket") {
            return Err(WsError::handshake("missing Upgrade: websocket"));
        }
        let connection = request
            .headers
            .get("connection")
            .map(String::as_str)
            .unwrap_or("");
        if !header_token_contains(connection, "Upgrade") {
            return Err(WsError::handshake("missing Connection: Upgrade"));
        }
        if request
            .key
            .is_empty()
        {
            return Err(WsError::handshake("missing Sec-WebSocket-Key"));
        }
        // Key must decode to 16 bytes (RFC 6455 §4.1).
        let decoded = ws_base64::decode(&request.key)?;
        if decoded.len() != 16 {
            return Err(WsError::handshake("Sec-WebSocket-Key must decode to 16 bytes"));
        }
        // Select first mutually supported subprotocol.
        self.selected_protocol = request
            .protocols
            .iter()
            .find(|p| {
                self.supported_protocols
                    .iter()
                    .any(|s| s == *p)
            })
            .cloned();
        self.request = Some(request);
        Ok(())
    }

    /// Serializes the `101 Switching Protocols` response.
    pub fn encode_response(&self) -> WsResult<Vec<u8>> {
        let request = self
            .request
            .as_ref()
            .ok_or_else(|| WsError::invalid_state("no accepted request"))?;
        let accept = compute_accept_key(&request.key);
        let mut resp = String::new();
        resp.push_str("HTTP/1.1 101 Switching Protocols\r\n");
        resp.push_str("Upgrade: websocket\r\n");
        resp.push_str("Connection: Upgrade\r\n");
        resp.push_str(&format!("Sec-WebSocket-Accept: {accept}\r\n"));
        if let Some(ref proto) = self.selected_protocol {
            resp.push_str(&format!("Sec-WebSocket-Protocol: {proto}\r\n"));
        }
        resp.push_str("\r\n");
        Ok(resp.into_bytes())
    }

    /// Builds a `400 Bad Request` response advertising supported versions.
    #[must_use]
    pub fn encode_version_rejection() -> Vec<u8> {
        let mut resp = String::new();
        resp.push_str("HTTP/1.1 400 Bad Request\r\n");
        resp.push_str(&format!("Sec-WebSocket-Version: {VERSION}\r\n"));
        resp.push_str("Content-Length: 0\r\n");
        resp.push_str("\r\n");
        resp.into_bytes()
    }
}

/// Parses an HTTP request from `buf`.
///
/// Returns `Ok(None)` if the full header block (ending in `\r\n\r\n`) is
/// not yet available. On success returns `(request, bytes_consumed)`.
pub fn try_parse_request(buf: &[u8]) -> WsResult<Option<(HandshakeRequest, usize)>> {
    let Some(header_end) = find_header_end(buf) else {
        return Ok(None);
    };
    let header_block = std::str::from_utf8(&buf[..header_end])
        .map_err(|_| WsError::handshake("request headers are not valid UTF-8"))?;
    let mut lines = header_block.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| WsError::handshake("empty request"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| WsError::handshake("malformed request line"))?;
    let path = parts
        .next()
        .ok_or_else(|| WsError::handshake("malformed request line"))?;
    let version = parts
        .next()
        .ok_or_else(|| WsError::handshake("malformed request line"))?;
    if method != "GET" {
        return Err(WsError::handshake(format!(
            "WebSocket upgrade requires GET, got '{method}'"
        )));
    }
    if version != "HTTP/1.1" {
        return Err(WsError::handshake(format!(
            "WebSocket upgrade requires HTTP/1.1, got '{version}'"
        )));
    }
    let headers = parse_headers(lines)?;
    let host = headers
        .get("host")
        .cloned()
        .ok_or_else(|| WsError::handshake("missing Host header"))?;
    let key = headers
        .get("sec-websocket-key")
        .cloned()
        .unwrap_or_default();
    let protocols = headers
        .get("sec-websocket-protocol")
        .map(|v| split_tokens(v))
        .unwrap_or_default();
    let origin = headers
        .get("origin")
        .cloned();
    let consumed = header_end + 4; // include CRLF CRLF
    Ok(Some((
        HandshakeRequest {
            path: path.to_string(),
            host,
            key,
            protocols,
            origin,
            headers,
        },
        consumed,
    )))
}

/// Parses an HTTP response from `buf`.
///
/// Returns `Ok(None)` until the full header block is available.
pub fn try_parse_response(buf: &[u8]) -> WsResult<Option<(WsHandshakeResponse, usize)>> {
    let Some(header_end) = find_header_end(buf) else {
        return Ok(None);
    };
    let header_block = std::str::from_utf8(&buf[..header_end])
        .map_err(|_| WsError::handshake("response headers are not valid UTF-8"))?;
    let mut lines = header_block.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| WsError::handshake("empty response"))?;
    let mut parts = status_line.split_whitespace();
    let _http = parts
        .next()
        .ok_or_else(|| WsError::handshake("malformed status line"))?;
    let status: u16 = parts
        .next()
        .ok_or_else(|| WsError::handshake("malformed status line"))?
        .parse()
        .map_err(|_| WsError::handshake("invalid HTTP status code"))?;
    let headers = parse_headers(lines)?;
    let accept = headers
        .get("sec-websocket-accept")
        .cloned()
        .unwrap_or_default();
    let protocol = headers
        .get("sec-websocket-protocol")
        .cloned();
    let consumed = header_end + 4;
    Ok(Some((
        WsHandshakeResponse {
            status,
            accept,
            protocol,
            headers,
        },
        consumed,
    )))
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4)
        .position(|w| w == b"\r\n\r\n")
}

fn parse_headers<'a, I>(lines: I) -> WsResult<HashMap<String, String>>
where
    I: Iterator<Item = &'a str>,
{
    let mut headers = HashMap::new();
    for line in lines {
        if line.is_empty() {
            break;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| WsError::handshake(format!("malformed header line: {line}")))?;
        let name = name
            .trim()
            .to_ascii_lowercase();
        let value = value
            .trim()
            .to_string();
        // Combine duplicate headers with comma (RFC 7230 §3.2.2), except
        // Sec-WebSocket-Extensions / Protocol which may appear multiple times;
        // joining with comma preserves list semantics.
        headers
            .entry(name)
            .and_modify(|existing: &mut String| {
                existing.push_str(", ");
                existing.push_str(&value);
            })
            .or_insert(value);
    }
    Ok(headers)
}

fn split_tokens(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| {
            s.trim()
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

fn eq_ignore_ascii_case(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

fn header_token_contains(header: &str, token: &str) -> bool {
    header
        .split(',')
        .any(|t| {
            t.trim()
                .eq_ignore_ascii_case(token)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ws_base64;

    #[test]
    fn rfc6455_accept_key_test_vector() {
        assert_eq!(
            compute_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn protocol_constants_match_rfc() {
        assert_eq!(GUID, "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
        assert_eq!(VERSION, 13);
    }

    #[test]
    fn accept_key_from_nonce_matches_rfc_vector() {
        let nonce = *b"the sample nonce";
        assert_eq!(nonce.len(), 16);
        assert_eq!(ws_base64::encode(&nonce), "dGhlIHNhbXBsZSBub25jZQ==");
        assert_eq!(accept_key_from_nonce(&nonce), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
        let manual = compute_accept_key(&ws_base64::encode(&nonce));
        assert_eq!(accept_key_from_nonce(&nonce), manual);
    }

    #[test]
    fn generate_key_nonce_is_unique_and_well_formed() {
        let a = generate_key_nonce().expect("rand a");
        let b = generate_key_nonce().expect("rand b");
        assert_ne!(a, b);
        let encoded = ws_base64::encode(&a);
        assert_eq!(encoded.len(), 24);
        assert!(encoded.ends_with("=="));
        for ch in encoded
            .chars()
            .filter(|c| *c != '=')
        {
            assert!(
                ch.is_ascii_alphanumeric() || ch == '+' || ch == '/',
                "unexpected base64 character {ch}"
            );
        }
        assert_eq!(ws_base64::decode(&encoded).expect("decode"), a.to_vec());
        assert_ne!(accept_key_from_nonce(&a), accept_key_from_nonce(&b));
    }

    #[test]
    fn client_handshake_validates_path_and_derives_expectations() {
        for bad in ["", "relative", "http://x/y"] {
            let err = WsClientHandshake::new(bad, "host", vec![], None).expect_err("bad path accepted");
            assert!(matches!(err, WsError::Handshake(_)));
        }
        let hs = WsClientHandshake::new(
            "/chat",
            "example.com",
            vec!["chat".to_string(), "superchat".to_string()],
            Some("https://origin.example".to_string()),
        )
        .expect("valid handshake");
        assert_eq!(hs.path, "/chat");
        assert_eq!(hs.host, "example.com");
        assert_eq!(
            hs.key_b64
                .len(),
            24
        );
        assert_eq!(
            ws_base64::decode(&hs.key_b64)
                .expect("key")
                .len(),
            16
        );
        assert_eq!(hs.expected_accept, compute_accept_key(&hs.key_b64));
        assert!(
            hs.selected_protocol
                .is_none()
        );
    }

    #[test]
    fn encode_request_contains_required_headers() {
        let hs = WsClientHandshake::new(
            "/chat",
            "example.com",
            vec!["chat".to_string(), "superchat".to_string()],
            Some("https://origin.example".to_string()),
        )
        .expect("valid handshake");
        let req = hs.encode_request();
        let text = std::str::from_utf8(&req).expect("utf8");
        assert!(text.starts_with("GET /chat HTTP/1.1\r\n"));
        assert!(text.contains("Host: example.com\r\n"));
        assert!(text.contains("Upgrade: websocket\r\n"));
        assert!(text.contains("Connection: Upgrade\r\n"));
        assert!(text.contains(&format!("Sec-WebSocket-Key: {}\r\n", hs.key_b64)));
        assert!(text.contains("Sec-WebSocket-Version: 13\r\n"));
        assert!(text.contains("Origin: https://origin.example\r\n"));
        assert!(text.contains("Sec-WebSocket-Protocol: chat, superchat\r\n"));
        assert!(text.ends_with("\r\n\r\n"));

        let plain = WsClientHandshake::new("/", "h", vec![], None).expect("valid handshake");
        let plain_text = String::from_utf8(plain.encode_request()).expect("utf8");
        assert!(!plain_text.contains("Sec-WebSocket-Protocol"));
        assert!(!plain_text.contains("Origin"));
    }

    #[test]
    fn request_response_round_trip() {
        let mut client = WsClientHandshake::new("/chat", "example.com", vec!["chat".to_string()], None)
            .expect("valid handshake");
        let bytes = client.encode_request();
        let (req, consumed) = try_parse_request(&bytes)
            .expect("parse")
            .expect("complete");
        assert_eq!(consumed, bytes.len());
        assert_eq!(req.path, "/chat");
        assert_eq!(req.host, "example.com");
        assert_eq!(req.key, client.key_b64);
        assert_eq!(req.protocols, vec!["chat".to_string()]);
        assert!(
            req.origin
                .is_none()
        );
        assert_eq!(
            req.headers
                .get("upgrade")
                .map(String::as_str),
            Some("websocket")
        );
        assert_eq!(
            req.headers
                .get("sec-websocket-version")
                .map(String::as_str),
            Some("13")
        );

        let mut server = WsServerHandshake::new(vec!["chat".to_string()]);
        server
            .accept_request(req)
            .expect("accept");
        assert_eq!(
            server
                .selected_protocol
                .as_deref(),
            Some("chat")
        );
        let resp_bytes = server
            .encode_response()
            .expect("encode");

        let (resp, consumed2) = try_parse_response(&resp_bytes)
            .expect("parse")
            .expect("complete");
        assert_eq!(consumed2, resp_bytes.len());
        assert_eq!(resp.status, 101);
        assert_eq!(resp.accept, compute_accept_key(&client.key_b64));
        assert_eq!(
            resp.protocol
                .as_deref(),
            Some("chat")
        );
        client
            .validate_response(&resp)
            .expect("validate");
        assert_eq!(
            client
                .selected_protocol
                .as_deref(),
            Some("chat")
        );
    }

    #[test]
    fn incomplete_request_returns_none_until_header_terminator() {
        let hs = WsClientHandshake::new("/", "h", vec![], None).expect("handshake");
        let bytes = hs.encode_request();
        for i in 0..bytes.len() {
            let parsed = try_parse_request(&bytes[..i]).expect("prefix parses");
            assert!(parsed.is_none(), "prefix of {i} bytes parsed early");
        }
        assert!(
            try_parse_request(&bytes)
                .expect("full")
                .is_some()
        );

        // Trailing bytes after the header block are not consumed.
        let mut with_tail = bytes.clone();
        with_tail.extend_from_slice(b"EXTRA");
        let (req, consumed) = try_parse_request(&with_tail)
            .expect("parse")
            .expect("complete");
        assert_eq!(consumed, bytes.len());
        assert_eq!(req.path, "/");
    }

    #[test]
    fn incomplete_response_returns_none_until_header_terminator() {
        let full = b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\n";
        for i in 0..full.len() {
            let parsed = try_parse_response(&full[..i]).expect("prefix parses");
            assert!(parsed.is_none(), "prefix of {i} bytes parsed early");
        }
        let (resp, consumed) = try_parse_response(full)
            .expect("parse")
            .expect("complete");
        assert_eq!(consumed, full.len());
        assert_eq!(resp.status, 101);
        assert_eq!(
            resp.headers
                .get("upgrade")
                .map(String::as_str),
            Some("websocket")
        );
    }

    #[test]
    fn request_header_edge_cases() {
        // Case-insensitive names, whitespace around name/value, empty value.
        let raw = b"GET / HTTP/1.1\r\nHoSt: Example.COM\r\n  X-Empty:   \r\n\r\n";
        let (req, _) = try_parse_request(raw)
            .expect("parse")
            .expect("complete");
        assert_eq!(req.host, "Example.COM");
        assert_eq!(
            req.headers
                .get("x-empty")
                .map(String::as_str),
            Some("")
        );
        assert!(
            req.key
                .is_empty()
        );
        assert!(
            req.protocols
                .is_empty()
        );

        // Duplicate protocol headers are comma-joined (list semantics).
        let raw2 = b"GET / HTTP/1.1\r\nHost: h\r\n\
                    Sec-WebSocket-Protocol: chat\r\n\
                    Sec-WebSocket-Protocol: super\r\n\r\n";
        let (req2, _) = try_parse_request(raw2)
            .expect("parse")
            .expect("complete");
        assert_eq!(req2.protocols, vec!["chat".to_string(), "super".to_string()]);

        // Empty protocol tokens are dropped.
        let raw3 = b"GET / HTTP/1.1\r\nHost: h\r\nSec-WebSocket-Protocol: a,, b ,\r\n\r\n";
        let (req3, _) = try_parse_request(raw3)
            .expect("parse")
            .expect("complete");
        assert_eq!(req3.protocols, vec!["a".to_string(), "b".to_string()]);

        // Origin header is surfaced.
        let raw4 = b"GET / HTTP/1.1\r\nHost: h\r\nOrigin: http://o\r\n\r\n";
        let (req4, _) = try_parse_request(raw4)
            .expect("parse")
            .expect("complete");
        assert_eq!(
            req4.origin
                .as_deref(),
            Some("http://o")
        );
    }

    #[test]
    fn malformed_requests_are_rejected() {
        // Missing Host header.
        let missing_host = b"GET / HTTP/1.1\r\n\r\n";
        assert!(matches!(
            try_parse_request(missing_host),
            Err(WsError::Handshake(_))
        ));

        // Whitespace-folded header lines have no colon -> rejected as malformed.
        let folded = b"GET / HTTP/1.1\r\nHost: h\r\nX-Long: part1\r\n  part2\r\n\r\n";
        assert!(matches!(try_parse_request(folded), Err(WsError::Handshake(_))));

        // Non-GET method.
        let post = b"POST / HTTP/1.1\r\nHost: h\r\n\r\n";
        assert!(matches!(try_parse_request(post), Err(WsError::Handshake(_))));

        // Wrong HTTP version.
        let v10 = b"GET / HTTP/1.0\r\nHost: h\r\n\r\n";
        assert!(matches!(try_parse_request(v10), Err(WsError::Handshake(_))));

        // Request line with too few tokens.
        let short = b"GET\r\nHost: h\r\n\r\n";
        assert!(matches!(try_parse_request(short), Err(WsError::Handshake(_))));

        // Header block is not valid UTF-8.
        let bad_utf8 = b"GET / HTTP/1.1\r\nHost: \xff\xfe\r\n\r\n";
        assert!(matches!(try_parse_request(bad_utf8), Err(WsError::Handshake(_))));
    }

    #[test]
    fn response_status_line_variants() {
        let ok = b"HTTP/1.1 101 Switching Protocols\r\n\r\n";
        let (r, _) = try_parse_response(ok)
            .expect("parse")
            .expect("complete");
        assert_eq!(r.status, 101);
        assert!(
            r.accept
                .is_empty()
        );
        assert!(
            r.protocol
                .is_none()
        );

        let bad = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
        let (r2, _) = try_parse_response(bad)
            .expect("parse")
            .expect("complete");
        assert_eq!(r2.status, 400);

        // Non-numeric status token.
        let bogus = b"HTTP/1.1 xyz\r\n\r\n";
        assert!(matches!(try_parse_response(bogus), Err(WsError::Handshake(_))));

        // Not an HTTP status line at all (no status token).
        let garbage = b"GARBAGE\r\n\r\n";
        assert!(matches!(try_parse_response(garbage), Err(WsError::Handshake(_))));

        // Missing status token entirely.
        let short = b"HTTP/1.1\r\n\r\n";
        assert!(matches!(try_parse_response(short), Err(WsError::Handshake(_))));

        // Non-UTF-8 header block.
        let bad_utf8 = b"HTTP/1.1 101\r\nX: \xff\r\n\r\n";
        assert!(matches!(try_parse_response(bad_utf8), Err(WsError::Handshake(_))));

        // Accept and subprotocol are extracted.
        let with_keys = b"HTTP/1.1 101\r\n\
                         Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\
                         Sec-WebSocket-Protocol: chat\r\n\r\n";
        let (r3, _) = try_parse_response(with_keys)
            .expect("parse")
            .expect("complete");
        assert_eq!(
            r3.accept
                .as_str(),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
        assert_eq!(
            r3.protocol
                .as_deref(),
            Some("chat")
        );
    }

    fn base_response(accept: &str) -> WsHandshakeResponse {
        let mut headers = HashMap::new();
        headers.insert("upgrade".to_string(), "websocket".to_string());
        headers.insert("connection".to_string(), "Upgrade".to_string());
        WsHandshakeResponse {
            status: 101,
            accept: accept.to_string(),
            protocol: None,
            headers,
        }
    }

    #[test]
    fn validate_response_rejects_bad_status_upgrade_and_accept() {
        let mut hs = WsClientHandshake::new("/", "h", vec!["chat".to_string()], None).expect("handshake");
        let good_accept = hs
            .expected_accept
            .clone();

        let mut resp = base_response(&good_accept);
        resp.status = 400;
        assert!(matches!(hs.validate_response(&resp), Err(WsError::Handshake(_))));

        let mut resp = base_response(&good_accept);
        resp.headers
            .remove("upgrade");
        assert!(matches!(hs.validate_response(&resp), Err(WsError::Handshake(_))));

        let mut resp = base_response(&good_accept);
        resp.headers
            .insert("upgrade".to_string(), "h2c".to_string());
        assert!(matches!(hs.validate_response(&resp), Err(WsError::Handshake(_))));

        let mut resp = base_response(&good_accept);
        resp.headers
            .insert("connection".to_string(), "keep-alive".to_string());
        assert!(matches!(hs.validate_response(&resp), Err(WsError::Handshake(_))));

        // Connection may be a token list, case-insensitively.
        let mut resp = base_response(&good_accept);
        resp.headers
            .insert("connection".to_string(), "keep-alive, Upgrade".to_string());
        hs.validate_response(&resp)
            .expect("token list accepted");

        let resp = base_response("AAAAAAAAAAAAAAAAAAAAAAAAAAAA=");
        assert!(matches!(hs.validate_response(&resp), Err(WsError::Handshake(_))));

        // Correct everything with matching accept: success path.
        let resp = base_response(&good_accept);
        hs.validate_response(&resp)
            .expect("valid response");
        assert!(
            hs.selected_protocol
                .is_none()
        );
    }

    #[test]
    fn validate_response_subprotocol_and_extension_rules() {
        let mut hs =
            WsClientHandshake::new("/", "h", vec!["chat".to_string(), "superchat".to_string()], None)
                .expect("handshake");
        let accept = hs
            .expected_accept
            .clone();

        // Offered subprotocol is selected.
        let mut resp = base_response(&accept);
        resp.protocol = Some("superchat".to_string());
        hs.validate_response(&resp)
            .expect("offered protocol");
        assert_eq!(
            hs.selected_protocol
                .as_deref(),
            Some("superchat")
        );

        // Unoffered subprotocol is rejected.
        let mut resp = base_response(&accept);
        resp.protocol = Some("evil".to_string());
        assert!(matches!(hs.validate_response(&resp), Err(WsError::Handshake(_))));

        // Empty Sec-WebSocket-Protocol header with no selected protocol fails.
        let mut resp = base_response(&accept);
        resp.headers
            .insert("sec-websocket-protocol".to_string(), String::new());
        assert!(matches!(hs.validate_response(&resp), Err(WsError::Handshake(_))));

        // Unoffered extension is rejected; empty extension header is fine.
        let mut resp = base_response(&accept);
        resp.headers
            .insert(
                "sec-websocket-extensions".to_string(),
                "permessage-deflate".to_string(),
            );
        assert!(matches!(hs.validate_response(&resp), Err(WsError::Handshake(_))));

        let mut resp = base_response(&accept);
        resp.headers
            .insert("sec-websocket-extensions".to_string(), String::new());
        hs.validate_response(&resp)
            .expect("empty extension");
    }

    fn base_request() -> HandshakeRequest {
        let mut headers = HashMap::new();
        headers.insert("sec-websocket-version".to_string(), "13".to_string());
        headers.insert("upgrade".to_string(), "websocket".to_string());
        headers.insert("connection".to_string(), "Upgrade".to_string());
        HandshakeRequest {
            path: "/".to_string(),
            host: "h".to_string(),
            key: "dGhlIHNhbXBsZSBub25jZQ==".to_string(),
            protocols: vec![],
            origin: None,
            headers,
        }
    }

    #[test]
    fn server_accept_request_validation() {
        let mut hs = WsServerHandshake::new(vec![]);
        hs.accept_request(base_request())
            .expect("valid request");
        assert!(
            hs.request
                .is_some()
        );
        assert!(
            hs.selected_protocol
                .is_none()
        );

        // Wrong version.
        let mut req = base_request();
        req.headers
            .insert("sec-websocket-version".to_string(), "12".to_string());
        assert!(matches!(hs.accept_request(req), Err(WsError::Handshake(_))));

        // Missing version header.
        let mut req = base_request();
        req.headers
            .remove("sec-websocket-version");
        assert!(matches!(hs.accept_request(req), Err(WsError::Handshake(_))));

        // Upgrade header missing or wrong value (case-insensitive match).
        let mut req = base_request();
        req.headers
            .remove("upgrade");
        assert!(matches!(hs.accept_request(req), Err(WsError::Handshake(_))));
        let mut req = base_request();
        req.headers
            .insert("upgrade".to_string(), "h2c".to_string());
        assert!(matches!(hs.accept_request(req), Err(WsError::Handshake(_))));
        let mut req = base_request();
        req.headers
            .insert("upgrade".to_string(), "WebSocket".to_string());
        hs.accept_request(req)
            .expect("case-insensitive upgrade");

        // Connection must contain the Upgrade token.
        let mut req = base_request();
        req.headers
            .insert("connection".to_string(), "keep-alive".to_string());
        assert!(matches!(hs.accept_request(req), Err(WsError::Handshake(_))));

        // Missing key.
        let mut req = base_request();
        req.key = String::new();
        assert!(matches!(hs.accept_request(req), Err(WsError::Handshake(_))));

        // Key that does not decode to 16 bytes ("12345678" -> 8 bytes).
        let mut req = base_request();
        req.key = "MTIzNDU2Nzg=".to_string();
        assert!(matches!(hs.accept_request(req), Err(WsError::Handshake(_))));

        // Malformed Base64 key surfaces a decode error.
        let mut req = base_request();
        req.key = "****".to_string();
        assert!(matches!(hs.accept_request(req), Err(WsError::Decode(_))));
    }

    #[test]
    fn server_selects_first_mutual_subprotocol_in_client_order() {
        let mut hs = WsServerHandshake::new(vec!["a".to_string(), "b".to_string()]);
        let mut req = base_request();
        req.protocols = vec!["b".to_string(), "a".to_string()];
        hs.accept_request(req)
            .expect("accept");
        assert_eq!(
            hs.selected_protocol
                .as_deref(),
            Some("b")
        );

        // No mutual protocol -> none selected.
        let mut req = base_request();
        req.protocols = vec!["c".to_string()];
        hs.accept_request(req)
            .expect("accept");
        assert!(
            hs.selected_protocol
                .is_none()
        );

        // Client offers nothing -> none selected.
        hs.accept_request(base_request())
            .expect("accept");
        assert!(
            hs.selected_protocol
                .is_none()
        );
    }

    #[test]
    fn encode_response_requires_accepted_request() {
        let hs = WsServerHandshake::new(vec![]);
        assert!(matches!(hs.encode_response(), Err(WsError::InvalidState(_))));
    }

    #[test]
    fn version_rejection_advertises_supported_version() {
        let bytes = WsServerHandshake::encode_version_rejection();
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.starts_with("HTTP/1.1 400 Bad Request\r\n"));
        assert!(text.contains("Sec-WebSocket-Version: 13\r\n"));
        assert!(text.contains("Content-Length: 0\r\n"));
        assert!(text.ends_with("\r\n\r\n"));
    }

    #[test]
    fn oversize_header_value_is_handled_without_failure() {
        let mut raw = Vec::new();
        raw.extend_from_slice(b"GET / HTTP/1.1\r\nHost: h\r\nX-Big: ");
        raw.extend(std::iter::repeat_n(b'A', 1024 * 1024));
        raw.extend_from_slice(b"\r\n\r\n");
        let (req, consumed) = try_parse_request(&raw)
            .expect("parse")
            .expect("complete");
        assert_eq!(consumed, raw.len());
        assert_eq!(
            req.headers
                .get("x-big")
                .map(String::len),
            Some(1024 * 1024)
        );
        // Without the terminator the parser only reports "need more bytes".
        let partial = &raw[..raw.len() - 1];
        assert!(
            try_parse_request(partial)
                .expect("prefix")
                .is_none()
        );
    }

    #[test]
    fn crlf_injection_in_path_breaks_the_request_line() {
        let mut hs = WsClientHandshake::new("/", "h", vec![], None).expect("handshake");
        hs.path = "/x\r\nInjected: yes".to_string();
        let bytes = hs.encode_request();
        assert!(matches!(try_parse_request(&bytes), Err(WsError::Handshake(_))));
    }

    #[test]
    fn crlf_injection_in_origin_splits_into_a_separate_header() {
        let hs =
            WsClientHandshake::new("/", "h", vec![], Some("o\r\nX-Evil: 1".to_string())).expect("handshake");
        let bytes = hs.encode_request();
        let (req, _) = try_parse_request(&bytes)
            .expect("parse")
            .expect("complete");
        // The injected line is parsed as its own header, and Origin is truncated.
        assert_eq!(
            req.headers
                .get("x-evil")
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(
            req.origin
                .as_deref(),
            Some("o")
        );
    }

    #[test]
    fn header_injection_in_host_is_rejected_as_malformed_header() {
        let mut hs = WsClientHandshake::new("/", "h", vec![], None).expect("handshake");
        hs.host = "h\r\nGET /evil HTTP/1.1".to_string();
        let bytes = hs.encode_request();
        // The injected line has no colon, so it is rejected as malformed.
        assert!(matches!(try_parse_request(&bytes), Err(WsError::Handshake(_))));
    }
}
