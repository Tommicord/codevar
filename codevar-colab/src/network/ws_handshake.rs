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
//! where `GUID` is [`GUID`](crate::network::ws_ids::GUID).

use crate::network::ws_base64;
use crate::network::ws_error::{WsError, WsResult};
use crate::network::ws_ids::{GUID, VERSION};
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
        if !self.protocols.is_empty() {
            req.push_str("Sec-WebSocket-Protocol: ");
            req.push_str(&self.protocols.join(", "));
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
            if !self.protocols.iter().any(|p| p == proto) {
                return Err(WsError::handshake(format!(
                    "server selected unoffered subprotocol '{proto}'"
                )));
            }
            self.selected_protocol = Some(proto.clone());
        } else if response.headers.contains_key("sec-websocket-protocol") {
            return Err(WsError::handshake(
                "empty Sec-WebSocket-Protocol in response",
            ));
        }
        // Reject unnegotiated extensions (we offer none).
        if let Some(ext) = response.headers.get("sec-websocket-extensions") {
            if !ext.trim().is_empty() {
                return Err(WsError::handshake(format!(
                    "server selected unoffered extension '{ext}'"
                )));
            }
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
        if request.key.is_empty() {
            return Err(WsError::handshake("missing Sec-WebSocket-Key"));
        }
        // Key must decode to 16 bytes (RFC 6455 §4.1).
        let decoded = ws_base64::decode(&request.key)?;
        if decoded.len() != 16 {
            return Err(WsError::handshake(
                "Sec-WebSocket-Key must decode to 16 bytes",
            ));
        }
        // Select first mutually supported subprotocol.
        self.selected_protocol = request
            .protocols
            .iter()
            .find(|p| self.supported_protocols.iter().any(|s| s == *p))
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
    let origin = headers.get("origin").cloned();
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
    let protocol = headers.get("sec-websocket-protocol").cloned();
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
    buf.windows(4).position(|w| w == b"\r\n\r\n")
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
        let (name, value) = line.split_once(':').ok_or_else(|| {
            WsError::handshake(format!("malformed header line: {line}"))
        })?;
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim().to_string();
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
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn eq_ignore_ascii_case(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

fn header_token_contains(header: &str, token: &str) -> bool {
    header
        .split(',')
        .any(|t| t.trim().eq_ignore_ascii_case(token))
}
