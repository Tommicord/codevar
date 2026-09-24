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

//! WebSocket client connection (RFC 6455 §4.1 / §5).
//!
//! [`ClientConnection`] is I/O-agnostic: feed response bytes with [`Self::feed`],
//! drain request / frame bytes with [`Self::take_write`], and advance protocol
//! state with [`Self::process`].

use crate::ws_connection::{CommonState, ConnectionConfig, ConnectionState, IoState};
use crate::ws_error::{WsError, WsResult};
use crate::ws_frame::WsFrame;
use crate::ws_handshake::{WsClientHandshake, try_parse_response};
use crate::ws_ids::{Role, WsCloseCode};
use crate::ws_message::WsMessage;

/// WebSocket client connection.
///
/// # Lifecycle
///
/// 1. [`ClientConnection::connect`] queues the HTTP Upgrade request.
/// 2. Send [`Self::take_write`] bytes to the server.
/// 3. Feed the server's HTTP response and call [`Self::process`] until
///    [`Self::is_open`] returns `true`.
/// 4. Exchange messages with [`Self::send_text`] / [`Self::send_binary`] /
///    [`Self::read_message`].
/// 5. Shut down with [`Self::close`].
pub struct ClientConnection {
    common: CommonState,
    handshake: WsClientHandshake,
}

impl ClientConnection {
    /// Starts a client connection to `path` on `host`.
    ///
    /// `protocols` are offered via `Sec-WebSocket-Protocol`. Pass `None`
    /// to omit the header.
    ///
    /// The Upgrade request is queued immediately and available via
    /// [`Self::take_write`].
    pub fn connect(
        path: impl Into<String>,
        host: impl Into<String>,
        protocols: Option<Vec<String>>,
    ) -> WsResult<Self> {
        Self::connect_with_config(
            path,
            host,
            protocols.unwrap_or_default(),
            None,
            ConnectionConfig::default(),
        )
    }

    /// Like [`Self::connect`], with full configuration control.
    pub fn connect_with_config(
        path: impl Into<String>,
        host: impl Into<String>,
        protocols: Vec<String>,
        origin: Option<String>,
        config: ConnectionConfig,
    ) -> WsResult<Self> {
        let handshake = WsClientHandshake::new(path, host, protocols, origin)?;
        let mut common = CommonState::new(Role::Client, config);
        let request = handshake.encode_request();
        common.queue_raw(&request);
        Ok(Self { common, handshake })
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

    /// Returns `true` while the handshake is in progress.
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
        match try_parse_response(self.common.rx_buf())? {
            Some((response, consumed)) => {
                self.common.consume_rx(consumed);
                self.handshake
                    .validate_response(&response)
                    .inspect_err(|_| {
                        self.common.state = ConnectionState::Closed;
                    })?;
                self.common
                    .mark_open(self.handshake.selected_protocol.clone());
                Ok(())
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
    use crate::ws_handshake::{WsServerHandshake, try_parse_request};
    use crate::ws_ids::DEFAULT_MAX_FRAME_SIZE;
    use crate::ws_server::ServerConnection;

    fn raw_response(status_line: &str, headers: &[(&str, &str)]) -> Vec<u8> {
        let mut s = format!("{status_line}\r\n");
        for (name, value) in headers {
            s.push_str(&format!("{name}: {value}\r\n"));
        }
        s.push_str("\r\n");
        s.into_bytes()
    }

    /// Builds a valid 101 response for `client` by running a real
    /// server-side handshake against the queued Upgrade request.
    fn server_accept_response(client: &mut ClientConnection) -> Vec<u8> {
        let request = client.take_write();
        let (req, _) = try_parse_request(&request)
            .expect("parse")
            .expect("complete");
        let mut hs = WsServerHandshake::new(vec![]);
        hs.accept_request(req).expect("accept");
        hs.encode_response().expect("encode")
    }

    fn unmasked(frame: &WsFrame) -> Vec<u8> {
        let mut out = Vec::new();
        frame.encode(&mut out, None).expect("encode");
        out
    }

    #[test]
    fn connect_queues_upgrade_request_and_starts_handshaking() {
        let mut client =
            ClientConnection::connect("/chat", "example.com", None).expect("connect");
        assert!(client.is_handshaking());
        assert!(!client.is_open());
        assert!(!client.is_closed());
        assert!(client.wants_write());
        assert!(client.wants_read());
        assert_eq!(client.state(), ConnectionState::Connecting);
        assert!(client.protocol().is_none());

        let wire = client.take_write();
        let text = String::from_utf8(wire.clone()).expect("utf8");
        assert!(text.starts_with("GET /chat HTTP/1.1\r\n"));
        assert!(text.contains("Host: example.com\r\n"));
        assert!(text.contains("Upgrade: websocket\r\n"));
        assert!(text.contains("Sec-WebSocket-Version: 13\r\n"));
        assert!(text.ends_with("\r\n\r\n"));
        assert!(!client.wants_write());

        // Processing with no response bytes keeps the handshake pending.
        let io = client.process().expect("process");
        assert_eq!(io.pending_rx, 0);
        assert!(client.is_handshaking());
    }

    #[test]
    fn connect_rejects_invalid_path() {
        for bad in ["", "no-slash", "http://host/path"] {
            let result = ClientConnection::connect(bad, "h", None);
            assert!(
                matches!(result, Err(WsError::Handshake(_))),
                "bad path {bad} accepted"
            );
        }
    }

    #[test]
    fn handshake_completes_via_server_connection() {
        let mut client = ClientConnection::connect(
            "/chat",
            "example.com",
            Some(vec!["chat".to_string()]),
        )
        .expect("connect");
        let mut server =
            ServerConnection::accept(Some(vec!["chat".to_string()])).expect("accept");

        let request = client.take_write();
        server.feed(&request).expect("feed");
        server.process().expect("server process");
        assert!(server.is_open());

        let response = server.take_write();
        client.feed(&response).expect("feed");
        client.process().expect("client process");
        assert!(client.is_open());
        assert!(!client.is_handshaking());
        assert_eq!(client.protocol(), Some("chat"));
        assert_eq!(server.protocol(), Some("chat"));
        assert_eq!(server.path(), Some("/chat"));
        assert!(!client.wants_write());
        assert_eq!(client.state(), ConnectionState::Open);
    }

    #[test]
    fn handshake_byte_by_byte_matches_all_at_once() {
        let mut incremental = ClientConnection::connect("/", "h", None).expect("connect");
        let response = server_accept_response(&mut incremental);
        for (i, b) in response.iter().enumerate() {
            incremental.feed(&[*b]).expect("feed");
            incremental.process().expect("process");
            if i + 1 < response.len() {
                assert!(incremental.is_handshaking(), "opened early at byte {i}");
            }
        }
        assert!(incremental.is_open());

        let mut at_once = ClientConnection::connect("/", "h", None).expect("connect");
        let response2 = server_accept_response(&mut at_once);
        assert_eq!(response.len(), response2.len());
        at_once.feed(&response2).expect("feed");
        at_once.process().expect("process");
        assert!(at_once.is_open());
        assert_eq!(at_once.state(), incremental.state());
    }

    #[test]
    fn truncated_response_keeps_handshake_pending() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let response = server_accept_response(&mut client);
        let split = response.len() / 2;
        client.feed(&response[..split]).expect("feed");
        let io = client.process().expect("process");
        assert_eq!(io.pending_rx, split);
        assert!(client.is_handshaking());

        client.feed(&response[split..]).expect("feed");
        client.process().expect("process");
        assert!(client.is_open());
        assert_eq!(client.state(), ConnectionState::Open);
    }

    #[test]
    fn pipelined_frame_after_response_is_delivered_in_one_process() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let mut payload = server_accept_response(&mut client);
        payload.extend(unmasked(&WsFrame::text(b"early")));
        client.feed(&payload).expect("feed");
        client.process().expect("process");
        assert!(client.is_open());
        match client.read_message().expect("read") {
            Some(WsMessage::Text(t)) => assert_eq!(t, "early"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn handshake_rejects_bad_status() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let resp = raw_response("HTTP/1.1 400 Bad Request", &[("Content-Length", "0")]);
        client.feed(&resp).expect("feed");
        let err = client.process().expect_err("bad status");
        assert!(matches!(err, WsError::Handshake(_)));
        assert!(client.is_closed());
        // Empty feed after close is tolerated; non-empty is rejected.
        client.feed(&[]).expect("empty feed");
        assert!(matches!(client.feed(&[0u8]), Err(WsError::Closed)));
        let io = client.process().expect("closed process");
        assert_eq!(io.pending_rx, 0);
    }

    #[test]
    fn handshake_rejects_wrong_accept_key() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let response = server_accept_response(&mut client);
        let response = String::from_utf8(response).expect("utf8");
        let key = "Sec-WebSocket-Accept: ";
        let start = response.find(key).expect("accept header") + key.len();
        let original = response.as_bytes()[start];
        let tampered = if original == b'A' { b'B' } else { b'A' };
        let mut bytes = response.into_bytes();
        bytes[start] = tampered;
        client.feed(&bytes).expect("feed");
        let err = client.process().expect_err("accept mismatch");
        assert!(matches!(err, WsError::Handshake(_)));
        assert!(client.is_closed());
    }

    #[test]
    fn handshake_rejects_bad_upgrade_headers() {
        // Wrong upgrade value.
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let resp = raw_response(
            "HTTP/1.1 101 Switching Protocols",
            &[
                ("Upgrade", "h2c"),
                ("Connection", "Upgrade"),
                ("Sec-WebSocket-Accept", "AAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
            ],
        );
        client.feed(&resp).expect("feed");
        assert!(matches!(client.process(), Err(WsError::Handshake(_))));
        assert!(client.is_closed());

        // Missing Upgrade header.
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let resp = raw_response(
            "HTTP/1.1 101 Switching Protocols",
            &[
                ("Connection", "Upgrade"),
                ("Sec-WebSocket-Accept", "AAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
            ],
        );
        client.feed(&resp).expect("feed");
        assert!(matches!(client.process(), Err(WsError::Handshake(_))));

        // Connection without the Upgrade token.
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let resp = raw_response(
            "HTTP/1.1 101 Switching Protocols",
            &[
                ("Upgrade", "websocket"),
                ("Connection", "keep-alive"),
                ("Sec-WebSocket-Accept", "AAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
            ],
        );
        client.feed(&resp).expect("feed");
        assert!(matches!(client.process(), Err(WsError::Handshake(_))));
    }

    #[test]
    fn garbage_status_line_fails_but_leaves_connecting_state() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        client.feed(b"NOT-HTTP-AT-ALL\r\n\r\n").expect("feed");
        let err = client.process().expect_err("garbage");
        assert!(matches!(err, WsError::Handshake(_)));
        // Quirk: parse-level failures propagate without closing the state.
        assert!(client.is_handshaking());
        // Retrying yields the same error deterministically.
        assert!(matches!(client.process(), Err(WsError::Handshake(_))));
    }

    #[test]
    fn send_before_open_is_rejected() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        assert!(matches!(
            client.send_text("x"),
            Err(WsError::InvalidState(_))
        ));
        assert!(matches!(
            client.send_binary(b"x"),
            Err(WsError::InvalidState(_))
        ));
        assert!(matches!(
            client.send_ping(b"x"),
            Err(WsError::InvalidState(_))
        ));
        assert!(matches!(
            client.send_pong(b"x"),
            Err(WsError::InvalidState(_))
        ));
    }

    #[test]
    fn outbound_client_frames_are_masked() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let response = server_accept_response(&mut client);
        client.feed(&response).expect("feed");
        client.process().expect("process");
        assert!(client.is_open());

        client.send_text("hello").expect("send");
        let wire = client.take_write();
        assert_eq!(wire[0], 0x81);
        assert_ne!(wire[1] & 0x80, 0);
        let (frame, consumed) =
            try_parse_frame(&wire, Role::Server, DEFAULT_MAX_FRAME_SIZE)
                .expect("parse")
                .expect("complete");
        assert_eq!(consumed, wire.len());
        assert_eq!(frame.payload, b"hello");
    }

    #[test]
    fn inbound_text_binary_and_controls_are_delivered() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let response = server_accept_response(&mut client);
        client.feed(&response).expect("feed");
        client.process().expect("process");
        assert!(client.is_open());

        let mut input = unmasked(&WsFrame::text(b"hi"));
        input.extend(unmasked(&WsFrame::binary(vec![1, 2, 3])));
        let ping = WsFrame::ping(b"p").expect("ping frame");
        input.extend(unmasked(&ping));
        let pong = WsFrame::pong(b"q").expect("pong frame");
        input.extend(unmasked(&pong));
        client.feed(&input).expect("feed");
        client.process().expect("process");

        assert!(matches!(
            client.read_message().expect("read"),
            Some(WsMessage::Text(t)) if t == "hi"
        ));
        assert!(matches!(
            client.read_message().expect("read"),
            Some(WsMessage::Binary(b)) if b == [1, 2, 3]
        ));
        assert!(matches!(
            client.read_message().expect("read"),
            Some(WsMessage::Ping(p)) if p == b"p"
        ));
        assert!(matches!(
            client.read_message().expect("read"),
            Some(WsMessage::Pong(p)) if p == b"q"
        ));
        assert!(client.read_message().expect("read").is_none());
        // The auto-pong for the server ping is queued.
        let pong = client.take_write();
        assert_eq!(pong[0], 0x8A);
    }

    #[test]
    fn inbound_masked_frame_fails_protocol_and_sticks() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let response = server_accept_response(&mut client);
        client.feed(&response).expect("feed");
        client.process().expect("process");

        let mut masked = Vec::new();
        WsFrame::text(b"x")
            .encode(&mut masked, Some([1, 2, 3, 4]))
            .expect("encode");
        client.feed(&masked).expect("feed");
        let err = client.process().expect_err("masked inbound");
        assert!(matches!(err, WsError::Protocol { .. }));
        assert!(client.is_closed());

        // The error is sticky for readers.
        let read_err = client.read_message().expect_err("sticky");
        assert!(matches!(read_err, WsError::Protocol { .. }));

        // A fail-close frame (masked, client role) was queued.
        let tx = client.take_write();
        assert_eq!(tx[0] & 0x0F, 0x8);
        assert_ne!(tx[1] & 0x80, 0);
    }

    #[test]
    fn invalid_utf8_text_frame_fails_connection() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let response = server_accept_response(&mut client);
        client.feed(&response).expect("feed");
        client.process().expect("process");

        client
            .feed(&unmasked(&WsFrame::text(vec![0xC0, 0x80])))
            .expect("feed");
        let err = client.process().expect_err("bad utf8");
        assert_eq!(err, WsError::InvalidUtf8);
        assert!(client.is_closed());
    }

    #[test]
    fn close_handshake_and_peer_close_accessors() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        let response = server_accept_response(&mut client);
        client.feed(&response).expect("feed");
        client.process().expect("process");
        assert!(client.is_open());

        client
            .close(WsCloseCode::Normal, "done")
            .expect("local close");
        assert_eq!(client.state(), ConnectionState::Closing);
        let tx = client.take_write();
        assert_eq!(tx[0] & 0x0F, 0x8); // close opcode

        let close =
            WsFrame::close(Some(WsCloseCode::GoingAway), "bye").expect("close frame");
        client.feed(&unmasked(&close)).expect("feed");
        client.process().expect("peer close");
        assert!(client.is_closed());
        assert_eq!(client.peer_close_code(), Some(WsCloseCode::GoingAway));
        assert_eq!(client.peer_close_reason(), Some("bye"));
        match client.read_message().expect("read") {
            Some(WsMessage::Close { code, reason }) => {
                assert_eq!(code, WsCloseCode::GoingAway);
                assert_eq!(reason, "bye");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(client.read_message().expect("read").is_none());
        assert!(!client.wants_read());
    }

    #[test]
    fn close_before_handshake_closes_immediately() {
        let mut client = ClientConnection::connect("/", "h", None).expect("connect");
        client
            .close(WsCloseCode::Normal, "")
            .expect("close while connecting");
        assert!(client.is_closed());
        assert!(!client.is_handshaking());
        client.feed(&[]).expect("empty feed");
        assert!(matches!(client.feed(b"x"), Err(WsError::Closed)));
        let io = client.process().expect("closed process");
        assert_eq!(io.pending_rx, 0);
    }
}
