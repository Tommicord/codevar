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

//! Shared WebSocket connection state machine (RFC 6455 §5–§7).
//!
//! [`CommonState`] owns the RX/TX byte buffers, fragmentation reassembly,
//! automatic control-frame handling, and the OPEN / CLOSING / CLOSED
//! lifecycle. Client and server roles wrap this type and add the HTTP
//! opening handshake.

use crate::network::ws_error::{WsError, WsResult};
use crate::network::ws_frame::{
    WsFrame, decode_close_payload, random_mask_key, try_parse_frame,
};
use crate::network::ws_ids::{
    DEFAULT_MAX_FRAME_SIZE, DEFAULT_MAX_MESSAGE_SIZE, Role, WsCloseCode, WsOpcode,
};
use crate::network::ws_message::WsMessage;
use crate::network::ws_utf8::Utf8Validator;
use std::collections::VecDeque;

/// High-level WebSocket connection lifecycle (RFC 6455 §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// HTTP opening handshake in progress.
    Connecting,
    /// Handshake complete; data frames may be exchanged.
    Open,
    /// Close frame sent and/or received; waiting for peer Close or TCP teardown.
    Closing,
    /// Connection fully closed (cleanly or abnormally).
    Closed,
}

/// Snapshot of buffer / readiness state after processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoState {
    /// Bytes still buffered on the inbound side awaiting a complete frame.
    pub pending_rx: usize,
    /// Bytes queued for the transport.
    pub pending_tx: usize,
    /// Complete messages waiting for the application.
    pub pending_messages: usize,
}

/// Tunables for framing and reassembly limits.
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// Maximum payload size of a single frame.
    pub max_frame_size: usize,
    /// Maximum size of a reassembled data message.
    pub max_message_size: usize,
    /// When true, automatically reply to Ping with a matching Pong.
    pub auto_pong: bool,
    /// When true, automatically echo a Close frame when the peer closes first.
    pub auto_close_reply: bool,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            max_frame_size: DEFAULT_MAX_FRAME_SIZE,
            max_message_size: DEFAULT_MAX_MESSAGE_SIZE,
            auto_pong: true,
            auto_close_reply: true,
        }
    }
}

/// Fragment reassembly buffer for a multi-frame data message.
#[derive(Debug, Default)]
struct FragmentBuf {
    /// Opcode of the initial fragment (`Text` or `Binary`).
    opcode: Option<WsOpcode>,
    /// Accumulated payload bytes.
    data: Vec<u8>,
    /// Streaming UTF-8 validator (only for text messages).
    utf8: Utf8Validator,
}

impl FragmentBuf {
    fn clear(&mut self) {
        self.opcode = None;
        self.data.clear();
        self.utf8.reset();
    }

    fn is_active(&self) -> bool {
        self.opcode.is_some()
    }
}

/// Bidirectional framing state shared by client and server.
pub struct CommonState {
    /// Local role (determines masking rules).
    pub role: Role,
    /// Lifecycle state.
    pub state: ConnectionState,
    /// Framing configuration.
    pub config: ConnectionConfig,
    /// Inbound transport buffer (may hold a partial frame / HTTP message).
    rx_buf: Vec<u8>,
    /// Outbound transport buffer.
    tx_buf: Vec<u8>,
    /// Complete messages ready for the application.
    messages: VecDeque<WsMessage>,
    /// Active fragmented data message, if any.
    fragment: FragmentBuf,
    /// True after a Close frame has been sent locally.
    pub local_close_sent: bool,
    /// True after a Close frame has been received from the peer.
    pub peer_close_received: bool,
    /// Close code from the first Close frame received (RFC 6455 §7.1.5).
    pub peer_close_code: Option<WsCloseCode>,
    /// Close reason from the first Close frame received.
    pub peer_close_reason: Option<String>,
    /// Negotiated subprotocol after the handshake.
    pub protocol: Option<String>,
    /// Sticky fatal error (surfaced after a fail-close is queued).
    pub error: Option<WsError>,
}

impl CommonState {
    /// Creates shared state for `role` with `config`.
    #[must_use]
    pub fn new(role: Role, config: ConnectionConfig) -> Self {
        Self {
            role,
            state: ConnectionState::Connecting,
            config,
            rx_buf: Vec::new(),
            tx_buf: Vec::new(),
            messages: VecDeque::new(),
            fragment: FragmentBuf::default(),
            local_close_sent: false,
            peer_close_received: false,
            peer_close_code: None,
            peer_close_reason: None,
            protocol: None,
            error: None,
        }
    }

    /// Returns `true` while the opening handshake is incomplete.
    #[must_use]
    pub fn is_connecting(&self) -> bool {
        self.state == ConnectionState::Connecting
    }

    /// Returns `true` when data frames may be exchanged.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.state == ConnectionState::Open
    }

    /// Returns `true` when the connection is fully closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.state == ConnectionState::Closed
    }

    /// Returns `true` when outbound bytes are pending.
    #[must_use]
    pub fn wants_write(&self) -> bool {
        !self.tx_buf.is_empty()
    }

    /// Returns `true` when more network input may be useful.
    #[must_use]
    pub fn wants_read(&self) -> bool {
        !self.is_closed() && !self.peer_close_received
    }

    /// Appends transport bytes to the receive buffer.
    pub fn feed(&mut self, data: &[u8]) {
        self.rx_buf.extend_from_slice(data);
    }

    /// Drains all buffered outbound transport bytes.
    pub fn take_write(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.tx_buf)
    }

    /// Appends raw bytes to the outbound buffer (used for the HTTP handshake).
    pub fn queue_raw(&mut self, data: &[u8]) {
        self.tx_buf.extend_from_slice(data);
    }

    /// Number of outbound transport bytes pending.
    #[must_use]
    pub fn pending_tx_len(&self) -> usize {
        self.tx_buf.len()
    }

    /// Immutable view of the inbound buffer (for handshake parsers).
    #[must_use]
    pub fn rx_buf(&self) -> &[u8] {
        &self.rx_buf
    }

    /// Consumes `n` bytes from the front of the inbound buffer.
    pub fn consume_rx(&mut self, n: usize) {
        if n >= self.rx_buf.len() {
            self.rx_buf.clear();
        } else {
            self.rx_buf.drain(..n);
        }
    }

    /// Transitions from Connecting to Open after a successful handshake.
    pub fn mark_open(&mut self, protocol: Option<String>) {
        self.state = ConnectionState::Open;
        self.protocol = protocol;
    }

    /// Queues a frame for transmission, applying role-appropriate masking.
    pub fn send_frame(&mut self, frame: &WsFrame) -> WsResult<()> {
        if self.state == ConnectionState::Closed {
            return Err(WsError::Closed);
        }
        // After sending Close, only control frames already in-flight matter;
        // RFC 6455 §5.5.1: MUST NOT send further data frames.
        if self.local_close_sent && frame.header.opcode.is_data() {
            return Err(WsError::invalid_state(
                "cannot send data frames after Close",
            ));
        }
        if self.local_close_sent && frame.header.opcode == WsOpcode::Continuation {
            return Err(WsError::invalid_state(
                "cannot send continuation after Close",
            ));
        }
        let mask = if self.role.must_mask_outbound() {
            Some(random_mask_key()?)
        } else {
            None
        };
        frame.encode(&mut self.tx_buf, mask)?;
        Ok(())
    }

    /// Sends a complete text message (single frame).
    pub fn send_text(&mut self, text: &str) -> WsResult<()> {
        self.ensure_can_send_data()?;
        crate::network::ws_utf8::validate_utf8(text.as_bytes())?;
        self.send_frame(&WsFrame::text(text.as_bytes().to_vec()))
    }

    /// Sends a complete binary message (single frame).
    pub fn send_binary(&mut self, data: &[u8]) -> WsResult<()> {
        self.ensure_can_send_data()?;
        if data.len() > self.config.max_message_size {
            return Err(WsError::MessageTooBig {
                size: data.len(),
                limit: self.config.max_message_size,
            });
        }
        self.send_frame(&WsFrame::binary(data.to_vec()))
    }

    /// Sends a Ping control frame.
    pub fn send_ping(&mut self, payload: &[u8]) -> WsResult<()> {
        if self.state != ConnectionState::Open && self.state != ConnectionState::Closing {
            return Err(WsError::invalid_state("cannot ping in current state"));
        }
        if self.local_close_sent {
            return Err(WsError::invalid_state("cannot ping after Close"));
        }
        self.send_frame(&WsFrame::ping(payload)?)
    }

    /// Sends a Pong control frame.
    pub fn send_pong(&mut self, payload: &[u8]) -> WsResult<()> {
        if self.state == ConnectionState::Closed
            || self.state == ConnectionState::Connecting
        {
            return Err(WsError::invalid_state("cannot pong in current state"));
        }
        self.send_frame(&WsFrame::pong(payload)?)
    }

    /// Starts the closing handshake with `code` and `reason` (RFC 6455 §7.1.2).
    pub fn close(&mut self, code: WsCloseCode, reason: &str) -> WsResult<()> {
        if self.local_close_sent || self.state == ConnectionState::Closed {
            return Ok(());
        }
        if self.state == ConnectionState::Connecting {
            self.state = ConnectionState::Closed;
            return Ok(());
        }
        let frame = WsFrame::close(Some(code), reason)?;
        self.send_frame(&frame)?;
        self.local_close_sent = true;
        self.state = if self.peer_close_received {
            ConnectionState::Closed
        } else {
            ConnectionState::Closing
        };
        Ok(())
    }

    /// Fails the connection (RFC 6455 §7.1.7): optionally send Close, then mark closed.
    pub fn fail(&mut self, err: WsError) -> WsError {
        if let Some(code) = err.close_code() {
            if self.state == ConnectionState::Open
                || self.state == ConnectionState::Closing
            {
                if !self.local_close_sent {
                    let _ = self.close(code, "");
                }
            }
        }
        self.state = ConnectionState::Closed;
        self.error = Some(err.clone());
        err
    }

    /// Pops the next complete message, if any.
    pub fn read_message(&mut self) -> Option<WsMessage> {
        self.messages.pop_front()
    }

    /// Number of buffered complete messages.
    #[must_use]
    pub fn pending_messages(&self) -> usize {
        self.messages.len()
    }

    /// Processes buffered frames after the handshake has completed.
    pub fn process_frames(&mut self) -> WsResult<IoState> {
        if let Some(ref e) = self.error {
            return Err(e.clone());
        }
        if self.state == ConnectionState::Connecting {
            return Err(WsError::HandshakeNotComplete);
        }
        if self.state == ConnectionState::Closed && self.rx_buf.is_empty() {
            return Ok(self.io_state());
        }

        loop {
            match try_parse_frame(&self.rx_buf, self.role, self.config.max_frame_size) {
                Ok(Some((frame, consumed))) => {
                    self.consume_rx(consumed);
                    if let Err(e) = self.handle_frame(frame) {
                        return Err(self.fail(e));
                    }
                    if self.state == ConnectionState::Closed {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => return Err(self.fail(e)),
            }
        }
        Ok(self.io_state())
    }

    fn io_state(&self) -> IoState {
        IoState {
            pending_rx: self.rx_buf.len(),
            pending_tx: self.tx_buf.len(),
            pending_messages: self.messages.len(),
        }
    }

    fn ensure_can_send_data(&self) -> WsResult<()> {
        if self.state != ConnectionState::Open {
            return Err(WsError::invalid_state(
                "connection is not open for data frames",
            ));
        }
        if self.local_close_sent {
            return Err(WsError::invalid_state("cannot send data after Close"));
        }
        Ok(())
    }

    fn handle_frame(&mut self, frame: WsFrame) -> WsResult<()> {
        // After we have been instructed to fail, ignore further input.
        if self.state == ConnectionState::Closed {
            return Ok(());
        }

        match frame.header.opcode {
            WsOpcode::Text | WsOpcode::Binary => self.handle_data_start(frame)?,
            WsOpcode::Continuation => self.handle_continuation(frame)?,
            WsOpcode::Ping => self.handle_ping(frame)?,
            WsOpcode::Pong => {
                self.messages.push_back(WsMessage::Pong(frame.payload));
            }
            WsOpcode::Close => self.handle_close(frame)?,
        }
        Ok(())
    }

    fn handle_data_start(&mut self, frame: WsFrame) -> WsResult<()> {
        if self.fragment.is_active() {
            return Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                "new data frame while fragmentation in progress",
            ));
        }
        if self.peer_close_received || self.local_close_sent {
            // RFC: endpoint MAY continue processing until Close reply; we
            // drop data frames after Close to keep state simple.
            return Ok(());
        }
        if frame.header.fin {
            self.deliver_complete(frame.header.opcode, frame.payload)?;
        } else {
            self.begin_fragment(frame.header.opcode, frame.payload)?;
        }
        Ok(())
    }

    fn handle_continuation(&mut self, frame: WsFrame) -> WsResult<()> {
        if !self.fragment.is_active() {
            return Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                "unexpected continuation frame",
            ));
        }
        self.append_fragment(&frame.payload)?;
        if frame.header.fin {
            let opcode = self
                .fragment
                .opcode
                .ok_or_else(|| WsError::Internal("fragment opcode missing".into()))?;
            let data = std::mem::take(&mut self.fragment.data);
            if opcode == WsOpcode::Text {
                self.fragment.utf8.finish()?;
            }
            self.fragment.clear();
            self.deliver_complete(opcode, data)?;
        }
        Ok(())
    }

    fn begin_fragment(&mut self, opcode: WsOpcode, payload: Vec<u8>) -> WsResult<()> {
        if !opcode.is_data() {
            return Err(WsError::protocol(
                WsCloseCode::ProtocolError,
                "fragmented message must start with text or binary",
            ));
        }
        self.fragment.opcode = Some(opcode);
        self.fragment.data.clear();
        self.fragment.utf8.reset();
        self.append_fragment(&payload)?;
        Ok(())
    }

    fn append_fragment(&mut self, payload: &[u8]) -> WsResult<()> {
        let new_len = self
            .fragment
            .data
            .len()
            .checked_add(payload.len())
            .ok_or_else(|| WsError::MessageTooBig {
                size: usize::MAX,
                limit: self.config.max_message_size,
            })?;
        if new_len > self.config.max_message_size {
            return Err(WsError::MessageTooBig {
                size: new_len,
                limit: self.config.max_message_size,
            });
        }
        if self.fragment.opcode == Some(WsOpcode::Text) {
            self.fragment.utf8.feed(payload)?;
        }
        self.fragment.data.extend_from_slice(payload);
        Ok(())
    }

    fn deliver_complete(&mut self, opcode: WsOpcode, payload: Vec<u8>) -> WsResult<()> {
        if payload.len() > self.config.max_message_size {
            return Err(WsError::MessageTooBig {
                size: payload.len(),
                limit: self.config.max_message_size,
            });
        }
        match opcode {
            WsOpcode::Text => {
                crate::network::ws_utf8::validate_utf8(&payload)?;
                let text =
                    String::from_utf8(payload).map_err(|_| WsError::InvalidUtf8)?;
                self.messages.push_back(WsMessage::Text(text));
            }
            WsOpcode::Binary => {
                self.messages.push_back(WsMessage::Binary(payload));
            }
            _ => {
                return Err(WsError::Internal(
                    "deliver_complete called with non-data opcode".into(),
                ));
            }
        }
        Ok(())
    }

    fn handle_ping(&mut self, frame: WsFrame) -> WsResult<()> {
        if self.config.auto_pong
            && !self.local_close_sent
            && self.state != ConnectionState::Closed
        {
            self.send_frame(&WsFrame::pong(frame.payload.clone())?)?;
        }
        self.messages.push_back(WsMessage::Ping(frame.payload));
        Ok(())
    }

    fn handle_close(&mut self, frame: WsFrame) -> WsResult<()> {
        let (code, reason) = decode_close_payload(&frame.payload)?;
        if !self.peer_close_received {
            self.peer_close_code = Some(code);
            self.peer_close_reason = Some(reason.clone());
            self.peer_close_received = true;
            self.messages.push_back(WsMessage::Close {
                code,
                reason: reason.clone(),
            });
        }
        if !self.local_close_sent && self.config.auto_close_reply {
            // Echo the peer's code when sendable; otherwise use Normal.
            let reply = if code.is_sendable() {
                code
            } else {
                WsCloseCode::Normal
            };
            let close_frame = WsFrame::close(Some(reply), "")?;
            self.send_frame(&close_frame)?;
            self.local_close_sent = true;
        }
        self.state = if self.local_close_sent {
            ConnectionState::Closed
        } else {
            ConnectionState::Closing
        };
        Ok(())
    }
}
