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

//! WebSocket server connection (RFC 6455 §4.2 / §5).
//!
//! [`ServerConnection`] waits for a client HTTP Upgrade request, validates
//! it, queues a `101 Switching Protocols` response, then exchanges framed
//! messages. Outbound frames are never masked (RFC 6455 §5.1).

use crate::ws_connection::{CommonState, ConnectionConfig, ConnectionState, IoState};
use crate::ws_error::{WsError, WsResult};
use crate::ws_frame::WsFrame;
use crate::ws_handshake::{WsServerHandshake, try_parse_request};
use crate::ws_ids::{Role, WsCloseCode};
use crate::ws_message::WsMessage;

/// WebSocket server connection.
///
/// # Lifecycle
///
/// 1. [`ServerConnection::accept`] creates a listening connection.
/// 2. Feed the client's Upgrade request and call [`Self::process`].
/// 3. Send [`Self::take_write`] (the 101 response) to the client.
/// 4. Exchange messages once [`Self::is_open`] is `true`.
pub struct ServerConnection {
    common: CommonState,
    handshake: WsServerHandshake,
}

impl ServerConnection {
    /// Creates a server connection that will accept an incoming Upgrade.
    ///
    /// `protocols` lists subprotocols the server is willing to select, in
    /// preference order. The first client-offered protocol that appears in
    /// this list is chosen.
    pub fn accept(protocols: Option<Vec<String>>) -> WsResult<Self> {
        Self::accept_with_config(
            protocols.unwrap_or_default(),
            ConnectionConfig::default(),
        )
    }

    /// Like [`Self::accept`], with full configuration control.
    pub fn accept_with_config(
        protocols: Vec<String>,
        config: ConnectionConfig,
    ) -> WsResult<Self> {
        Ok(Self {
            common: CommonState::new(Role::Server, config),
            handshake: WsServerHandshake::new(protocols),
        })
    }

    /// Request-target path from the client's Upgrade request, if accepted.
    #[must_use]
    pub fn path(&self) -> Option<&str> {
        self.handshake.request.as_ref().map(|r| r.path.as_str())
    }

    /// Negotiated subprotocol after a successful handshake.
    #[must_use]
    pub fn protocol(&self) -> Option<&str> {
        self.common.protocol.as_deref()
    }

    /// Current connection lifecycle state.
    #[must_use]
    pub fn state(&self) -> ConnectionState {
        self.common.state
    }

    /// Returns `true` when the opening handshake has completed.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.common.is_open()
    }

    /// Returns `true` while waiting for / processing the Upgrade request.
    #[must_use]
    pub fn is_handshaking(&self) -> bool {
        self.common.is_connecting()
    }

    /// Returns `true` when the connection is fully closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.common.is_closed()
    }

    /// Returns `true` when outbound bytes are pending.
    #[must_use]
    pub fn wants_write(&self) -> bool {
        self.common.wants_write()
    }

    /// Returns `true` when more network input may be useful.
    #[must_use]
    pub fn wants_read(&self) -> bool {
        self.common.wants_read()
    }

    /// Feeds transport bytes received from the peer.
    pub fn feed(&mut self, data: &[u8]) -> WsResult<()> {
        if self.common.is_closed() && data.is_empty() {
            return Ok(());
        }
        if self.common.is_closed() {
            return Err(WsError::Closed);
        }
        self.common.feed(data);
        Ok(())
    }

    /// Drains all buffered outbound transport bytes.
    pub fn take_write(&mut self) -> Vec<u8> {
        self.common.take_write()
    }

    /// Processes buffered input: completes the handshake and/or parses frames.
    pub fn process(&mut self) -> WsResult<IoState> {
        if self.common.is_connecting() {
            self.process_handshake()?;
        }
        if self.common.is_open() || self.common.state == ConnectionState::Closing {
            return self.common.process_frames();
        }
        Ok(IoState {
            pending_rx: self.common.rx_buf().len(),
            pending_tx: self.common.pending_tx_len(),
            pending_messages: self.common.pending_messages(),
        })
    }

    fn process_handshake(&mut self) -> WsResult<()> {
        match try_parse_request(self.common.rx_buf())? {
            Some((request, consumed)) => {
                self.common.consume_rx(consumed);
                match self.handshake.accept_request(request) {
                    Ok(()) => {
                        let response = self.handshake.encode_response()?;
                        self.common.queue_raw(&response);
                        self.common
                            .mark_open(self.handshake.selected_protocol.clone());
                        Ok(())
                    }
                    Err(e) => {
                        // Advertise supported version on version mismatch.
                        if e.to_string().contains("Sec-WebSocket-Version") {
                            self.common
                                .queue_raw(&WsServerHandshake::encode_version_rejection());
                        }
                        self.common.state = ConnectionState::Closed;
                        Err(e)
                    }
                }
            }
            None => Ok(()),
        }
    }

    /// Sends a UTF-8 text message.
    pub fn send_text(&mut self, text: &str) -> WsResult<()> {
        self.common.send_text(text)
    }

    /// Sends a binary message.
    pub fn send_binary(&mut self, data: &[u8]) -> WsResult<()> {
        self.common.send_binary(data)
    }

    /// Sends a Ping control frame.
    pub fn send_ping(&mut self, payload: &[u8]) -> WsResult<()> {
        self.common.send_ping(payload)
    }

    /// Sends a Pong control frame.
    pub fn send_pong(&mut self, payload: &[u8]) -> WsResult<()> {
        self.common.send_pong(payload)
    }

    /// Sends an arbitrary frame (advanced use; prefer the typed senders).
    pub fn send_frame(&mut self, frame: &WsFrame) -> WsResult<()> {
        self.common.send_frame(frame)
    }

    /// Starts the closing handshake.
    pub fn close(&mut self, code: WsCloseCode, reason: &str) -> WsResult<()> {
        self.common.close(code, reason)
    }

    /// Pops the next complete message, if any.
    pub fn read_message(&mut self) -> WsResult<Option<WsMessage>> {
        if let Some(ref e) = self.common.error {
            return Err(e.clone());
        }
        Ok(self.common.read_message())
    }

    /// Peer close code from the first received Close frame.
    #[must_use]
    pub fn peer_close_code(&self) -> Option<WsCloseCode> {
        self.common.peer_close_code
    }

    /// Peer close reason from the first received Close frame.
    #[must_use]
    pub fn peer_close_reason(&self) -> Option<&str> {
        self.common.peer_close_reason.as_deref()
    }

    /// Immutable access to shared state (for diagnostics).
    #[must_use]
    pub fn common(&self) -> &CommonState {
        &self.common
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ws_frame::try_parse_frame;
    use crate::ws_handshake::{WsClientHandshake, try_parse_response};
    use crate::ws_ids::DEFAULT_MAX_FRAME_SIZE;

    const VALID_KEY: &str = "dGhlIHNhbXBsZSBub25jZQ==";

    fn upgrade_request(path: &str, version: &str, key: &str, extra: &str) -> Vec<u8> {
        format!(
            "GET {path} HTTP/1.1\r\n\
             Host: example.com\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: {key}\r\n\
             Sec-WebSocket-Version: {version}\r\n\
             {extra}\r\n"
        )
        .into_bytes()
    }

    fn masked(frame: &WsFrame, key: [u8; 4]) -> Vec<u8> {
        let mut out = Vec::new();
        frame.encode(&mut out, Some(key)).expect("encode");
        out
    }

    #[test]
    fn accept_starts_handshaking_with_empty_buffers() {
        let mut server = ServerConnection::accept(None).expect("accept");
        assert!(server.is_handshaking());
        assert!(!server.is_open());
        assert!(!server.is_closed());
        assert!(!server.wants_write());
        assert!(server.wants_read());
        assert!(server.path().is_none());
        assert!(server.protocol().is_none());

        let io = server.process().expect("process");
        assert_eq!(io.pending_rx, 0);
        assert_eq!(io.pending_tx, 0);
        assert!(server.is_handshaking());
    }

    #[test]
    fn upgrade_request_completes_handshake_and_queues_101() {
        let mut server = ServerConnection::accept(None).expect("accept");
        let request = upgrade_request("/chat", "13", VALID_KEY, "");
        server.feed(&request).expect("feed");
        server.process().expect("process");
        assert!(server.is_open());
        assert!(!server.is_handshaking());
        assert_eq!(server.path(), Some("/chat"));
        assert!(server.wants_write());

        let response = server.take_write();
        let text = String::from_utf8(response.clone()).expect("utf8");
        assert!(text.starts_with("HTTP/1.1 101 Switching Protocols\r\n"));
        assert!(text.contains("Upgrade: websocket\r\n"));
        assert!(text.contains("Connection: Upgrade\r\n"));
        assert!(text.contains(&format!(
            "Sec-WebSocket-Accept: {}\r\n",
            crate::ws_handshake::compute_accept_key(VALID_KEY)
        )));
        assert!(text.ends_with("\r\n\r\n"));

        // The response validates as a client would.
        let (resp, consumed) = try_parse_response(&response)
            .expect("parse")
            .expect("complete");
        assert_eq!(consumed, response.len());
        let mut client = WsClientHandshake::new("/chat", "example.com", vec![], None)
            .expect("handshake");
        // Rebuild with the same key the request used.
        client.key_b64 = VALID_KEY.to_string();
        client.expected_accept = crate::ws_handshake::compute_accept_key(VALID_KEY);
        client.validate_response(&resp).expect("validate");
    }

    #[test]
    fn handshake_byte_by_byte_matches_all_at_once() {
        let request = upgrade_request("/", "13", VALID_KEY, "");

        let mut incremental = ServerConnection::accept(None).expect("accept");
        for (i, b) in request.iter().enumerate() {
            incremental.feed(&[*b]).expect("feed");
            incremental.process().expect("process");
            if i + 1 < request.len() {
                assert!(incremental.is_handshaking(), "opened early at byte {i}");
            }
        }
        assert!(incremental.is_open());

        let mut at_once = ServerConnection::accept(None).expect("accept");
        at_once.feed(&request).expect("feed");
        at_once.process().expect("process");
        assert!(at_once.is_open());
        assert_eq!(at_once.state(), incremental.state());
        assert_eq!(at_once.path(), incremental.path());
    }

    #[test]
    fn pipelined_masked_frame_after_request_is_delivered() {
        let mut server = ServerConnection::accept(None).expect("accept");
        let mut payload = upgrade_request("/", "13", VALID_KEY, "");
        payload.extend(masked(&WsFrame::text(b"early"), [9, 9, 9, 9]));
        server.feed(&payload).expect("feed");
        server.process().expect("process");
        assert!(server.is_open());
        match server.read_message().expect("read") {
            Some(WsMessage::Text(t)) => assert_eq!(t, "early"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn truncated_request_keeps_handshake_pending() {
        let mut server = ServerConnection::accept(None).expect("accept");
        let request = upgrade_request("/", "13", VALID_KEY, "");
        let split = request.len() / 2;
        server.feed(&request[..split]).expect("feed");
        let io = server.process().expect("process");
        assert_eq!(io.pending_rx, split);
        assert!(server.is_handshaking());
        assert!(!server.wants_write());

        server.feed(&request[split..]).expect("feed");
        server.process().expect("process");
        assert!(server.is_open());
    }

    #[test]
    fn wrong_version_gets_400_rejection_and_closes() {
        let mut server = ServerConnection::accept(None).expect("accept");
        let request = upgrade_request("/", "12", VALID_KEY, "");
        server.feed(&request).expect("feed");
        let err = server.process().expect_err("bad version");
        assert!(err.to_string().contains("Sec-WebSocket-Version"));
        assert!(server.is_closed());

        let rejection = server.take_write();
        let text = String::from_utf8(rejection).expect("utf8");
        assert!(text.starts_with("HTTP/1.1 400 Bad Request\r\n"));
        assert!(text.contains("Sec-WebSocket-Version: 13\r\n"));
        assert!(text.contains("Content-Length: 0\r\n"));

        server.feed(&[]).expect("empty feed");
        assert!(matches!(server.feed(&[1u8]), Err(WsError::Closed)));
    }

    #[test]
    fn request_parse_errors_propagate_without_closing() {
        // Non-GET method is rejected at parse time; the state stays
        // connecting (mirrors the client-side parse-error quirk).
        let mut server = ServerConnection::accept(None).expect("accept");
        let post = b"POST / HTTP/1.1\r\nHost: h\r\n\r\n".to_vec();
        server.feed(&post).expect("feed");
        let err = server.process().expect_err("non-get");
        assert!(matches!(err, WsError::Handshake(_)));
        assert!(server.is_handshaking());
        assert!(!server.wants_write());

        // Garbage request line behaves the same way.
        let mut server2 = ServerConnection::accept(None).expect("accept");
        server2.feed(b"GARBAGE\r\n\r\n").expect("feed");
        assert!(matches!(server2.process(), Err(WsError::Handshake(_))));
        assert!(server2.is_handshaking());
    }

    #[test]
    fn missing_or_short_key_rejects_without_rejection_response() {
        // Missing key.
        let mut server = ServerConnection::accept(None).expect("accept");
        let no_key = format!(
            "GET / HTTP/1.1\r\n\
             Host: example.com\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Version: 13\r\n\r\n"
        );
        server.feed(no_key.as_bytes()).expect("feed");
        let err = server.process().expect_err("missing key");
        assert!(err.to_string().contains("Sec-WebSocket-Key"));
        assert!(server.is_closed());
        assert!(!server.wants_write());

        // Key that decodes to 8 bytes instead of 16.
        let mut server2 = ServerConnection::accept(None).expect("accept");
        let short_key = upgrade_request("/", "13", "MTIzNDU2Nzg=", "");
        server2.feed(&short_key).expect("feed");
        let err2 = server2.process().expect_err("short key");
        assert!(err2.to_string().contains("16 bytes"));
        assert!(server2.is_closed());
        assert!(!server2.wants_write());

        // Malformed Base64 key surfaces a decode error and closes.
        let mut server3 = ServerConnection::accept(None).expect("accept");
        let bad_key = upgrade_request("/", "13", "****", "");
        server3.feed(&bad_key).expect("feed");
        let err3 = server3.process().expect_err("bad key");
        assert!(matches!(err3, WsError::Decode(_)));
        assert!(server3.is_closed());
        assert!(!server3.wants_write());
    }

    #[test]
    fn protocol_negotiation_selects_first_client_offered_mutual() {
        let mut server =
            ServerConnection::accept(Some(vec!["a".to_string(), "b".to_string()]))
                .expect("accept");
        let request =
            upgrade_request("/", "13", VALID_KEY, "Sec-WebSocket-Protocol: b, a\r\n");
        server.feed(&request).expect("feed");
        server.process().expect("process");
        assert!(server.is_open());
        assert_eq!(server.protocol(), Some("b"));

        let response = String::from_utf8(server.take_write()).expect("utf8");
        assert!(response.contains("Sec-WebSocket-Protocol: b\r\n"));

        // No mutual protocol -> none selected, no protocol header.
        let mut server2 =
            ServerConnection::accept(Some(vec!["z".to_string()])).expect("accept");
        let request2 =
            upgrade_request("/", "13", VALID_KEY, "Sec-WebSocket-Protocol: a\r\n");
        server2.feed(&request2).expect("feed");
        server2.process().expect("process");
        assert_eq!(server2.protocol(), None);
        let response2 = String::from_utf8(server2.take_write()).expect("utf8");
        assert!(!response2.contains("Sec-WebSocket-Protocol"));
    }

    #[test]
    fn masked_text_from_client_is_delivered_and_server_replies_unmasked() {
        let mut server = ServerConnection::accept(None).expect("accept");
        let request = upgrade_request("/", "13", VALID_KEY, "");
        server.feed(&request).expect("feed");
        server.process().expect("process");
        assert!(server.is_open());
        server.take_write(); // drain the 101 handshake response

        server
            .feed(&masked(&WsFrame::text(b"hi"), [1, 2, 3, 4]))
            .expect("feed");
        server.process().expect("process");
        match server.read_message().expect("read") {
            Some(WsMessage::Text(t)) => assert_eq!(t, "hi"),
            other => panic!("unexpected {other:?}"),
        }

        server.send_text("yo").expect("send");
        let wire = server.take_write();
        assert_eq!(wire[0], 0x81);
        assert_eq!(wire[1] & 0x80, 0, "server frames must be unmasked");
        let (frame, _) = try_parse_frame(&wire, Role::Client, DEFAULT_MAX_FRAME_SIZE)
            .expect("parse")
            .expect("complete");
        assert_eq!(frame.payload, b"yo");
    }

    #[test]
    fn unmasked_frame_from_client_is_rejected() {
        let mut server = ServerConnection::accept(None).expect("accept");
        let request = upgrade_request("/", "13", VALID_KEY, "");
        server.feed(&request).expect("feed");
        server.process().expect("process");

        let mut unmasked_frame = Vec::new();
        WsFrame::text(b"x")
            .encode(&mut unmasked_frame, None)
            .expect("encode");
        server.feed(&unmasked_frame).expect("feed");
        let err = server.process().expect_err("unmasked");
        assert!(matches!(err, WsError::Protocol { .. }));
        assert!(server.is_closed());
        let read_err = server.read_message().expect_err("sticky");
        assert!(matches!(read_err, WsError::Protocol { .. }));
    }

    #[test]
    fn ping_from_client_gets_unmasked_pong() {
        let mut server = ServerConnection::accept(None).expect("accept");
        let request = upgrade_request("/", "13", VALID_KEY, "");
        server.feed(&request).expect("feed");
        server.process().expect("process");
        server.take_write(); // drain the 101 handshake response

        let ping = WsFrame::ping(b"tick").expect("ping frame");
        server.feed(&masked(&ping, [5, 5, 5, 5])).expect("feed");
        server.process().expect("process");
        assert!(matches!(
            server.read_message().expect("read"),
            Some(WsMessage::Ping(p)) if p == b"tick"
        ));

        let tx = server.take_write();
        assert_eq!(tx[0], 0x8A, "pong opcode");
        assert_eq!(tx[1] & 0x80, 0, "server pong must be unmasked");
        let (frame, _) = try_parse_frame(&tx, Role::Client, DEFAULT_MAX_FRAME_SIZE)
            .expect("parse")
            .expect("complete");
        assert_eq!(frame.payload, b"tick");
    }

    #[test]
    fn close_handshake_replies_unmasked_and_surfaces_message() {
        let mut server = ServerConnection::accept(None).expect("accept");
        let request = upgrade_request("/", "13", VALID_KEY, "");
        server.feed(&request).expect("feed");
        server.process().expect("process");
        server.take_write();

        let close =
            WsFrame::close(Some(WsCloseCode::Normal), "done").expect("close frame");
        server.feed(&masked(&close, [7, 7, 7, 7])).expect("feed");
        server.process().expect("process");
        match server.read_message().expect("read") {
            Some(WsMessage::Close { code, reason }) => {
                assert_eq!(code, WsCloseCode::Normal);
                assert_eq!(reason, "done");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(server.is_closed());
        assert_eq!(server.peer_close_code(), Some(WsCloseCode::Normal));
        assert_eq!(server.peer_close_reason(), Some("done"));

        let tx = server.take_write();
        assert_eq!(tx[0] & 0x0F, 0x8, "close reply opcode");
        assert_eq!(tx[1] & 0x80, 0, "close reply must be unmasked");
        assert!(matches!(server.feed(&[1u8]), Err(WsError::Closed)));
    }

    #[test]
    fn send_before_open_is_rejected() {
        let mut server = ServerConnection::accept(None).expect("accept");
        assert!(matches!(
            server.send_text("x"),
            Err(WsError::InvalidState(_))
        ));
        assert!(matches!(
            server.send_binary(b"x"),
            Err(WsError::InvalidState(_))
        ));
        assert!(matches!(
            server.send_ping(b"x"),
            Err(WsError::InvalidState(_))
        ));
        assert!(matches!(
            server.send_pong(b"x"),
            Err(WsError::InvalidState(_))
        ));
    }

    #[test]
    fn accept_with_config_enforces_frame_limits() {
        let config = ConnectionConfig {
            max_frame_size: 512,
            max_message_size: 512,
            ..ConnectionConfig::default()
        };
        let mut server =
            ServerConnection::accept_with_config(vec![], config).expect("accept");
        let request = upgrade_request("/", "13", VALID_KEY, "");
        server.feed(&request).expect("feed");
        server.process().expect("process");
        assert!(server.is_open());

        // Feed only a 64-bit length header claiming 1 MiB (> 512 limit).
        server
            .feed(&[0x82u8, 0x7F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00])
            .expect("feed");
        let err = server.process().expect_err("too big");
        assert_eq!(
            err,
            WsError::MessageTooBig {
                size: 1024 * 1024,
                limit: 512,
            }
        );
        assert!(server.is_closed());
    }
}
