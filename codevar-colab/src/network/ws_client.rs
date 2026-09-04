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

//! WebSocket client connection (RFC 6455 §4.1 / §5).
//!
//! [`ClientConnection`] is I/O-agnostic: feed response bytes with [`Self::feed`],
//! drain request / frame bytes with [`Self::take_write`], and advance protocol
//! state with [`Self::process`].

use crate::network::ws_connection::{
    CommonState, ConnectionConfig, ConnectionState, IoState,
};
use crate::network::ws_error::{WsError, WsResult};
use crate::network::ws_frame::WsFrame;
use crate::network::ws_handshake::{WsClientHandshake, try_parse_response};
use crate::network::ws_ids::{Role, WsCloseCode};
use crate::network::ws_message::WsMessage;

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
                self.handshake.validate_response(&response).map_err(|e| {
                    self.common.state = ConnectionState::Closed;
                    e
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
