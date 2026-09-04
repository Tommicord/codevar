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

//! WebSocket server connection (RFC 6455 §4.2 / §5).
//!
//! [`ServerConnection`] waits for a client HTTP Upgrade request, validates
//! it, queues a `101 Switching Protocols` response, then exchanges framed
//! messages. Outbound frames are never masked (RFC 6455 §5.1).

use crate::network::ws_connection::{
    CommonState, ConnectionConfig, ConnectionState, IoState,
};
use crate::network::ws_error::{WsError, WsResult};
use crate::network::ws_frame::WsFrame;
use crate::network::ws_handshake::{WsServerHandshake, try_parse_request};
use crate::network::ws_ids::{Role, WsCloseCode};
use crate::network::ws_message::WsMessage;

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
