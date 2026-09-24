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

//! Shared WebSocket connection state machine (RFC 6455 §5–§7).
//!
//! [`CommonState`] owns the RX/TX byte buffers, fragmentation reassembly,
//! automatic control-frame handling, and the OPEN / CLOSING / CLOSED
//! lifecycle. Client and server roles wrap this type and add the HTTP
//! opening handshake.

use crate::ws_error::{WsError, WsResult};
use crate::ws_frame::{WsFrame, decode_close_payload, random_mask_key, try_parse_frame};
use crate::ws_ids::{
    DEFAULT_MAX_FRAME_SIZE, DEFAULT_MAX_MESSAGE_SIZE, Role, WsCloseCode, WsOpcode,
};
use crate::ws_message::WsMessage;
use crate::ws_utf8::Utf8Validator;
use codevar_textlike_encode::encoding_utf8::{encode_text, utf8_valid_up_to};
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
    /// Streaming UTF-8 validator for text fragments (RFC 6455 §5.6).
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
    ///
    /// The payload is encoded to UTF-8 through `codevar-textlike-encode`
    /// and must fit within [`ConnectionConfig::max_message_size`].
    pub fn send_text(&mut self, text: &str) -> WsResult<()> {
        self.ensure_can_send_data()?;
        if text.len() > self.config.max_message_size {
            return Err(WsError::MessageTooBig {
                size: text.len(),
                limit: self.config.max_message_size,
            });
        }
        let payload = encode_text(text);
        self.send_frame(&WsFrame::text(payload))
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
        if let Some(code) = err.close_code()
            && (self.state == ConnectionState::Open
                || self.state == ConnectionState::Closing)
            && !self.local_close_sent
        {
            let _ = self.close(code, "");
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
        let new_len = self.fragment.data.len().checked_add(payload.len()).ok_or(
            WsError::MessageTooBig {
                size: usize::MAX,
                limit: self.config.max_message_size,
            },
        )?;
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
                if utf8_valid_up_to(&payload) != payload.len() {
                    return Err(WsError::InvalidUtf8);
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn open_client() -> CommonState {
        let mut s = CommonState::new(Role::Client, ConnectionConfig::default());
        s.mark_open(None);
        s
    }

    fn open_server() -> CommonState {
        let mut s = CommonState::new(Role::Server, ConnectionConfig::default());
        s.mark_open(None);
        s
    }

    fn encode(frame: &WsFrame, mask: Option<[u8; 4]>) -> Vec<u8> {
        let mut out = Vec::new();
        frame.encode(&mut out, mask).expect("encode");
        out
    }

    fn parse_outbound_from_client(bytes: &[u8]) -> WsFrame {
        let (frame, _) = try_parse_frame(bytes, Role::Server, DEFAULT_MAX_FRAME_SIZE)
            .expect("frame available")
            .expect("complete frame");
        frame
    }

    #[test]
    fn process_frames_requires_completed_handshake() {
        let mut s = CommonState::new(Role::Client, ConnectionConfig::default());
        assert!(s.is_connecting());
        assert!(!s.is_open());
        assert!(!s.is_closed());
        let err = s.process_frames().expect_err("not handshaked");
        assert_eq!(err, WsError::HandshakeNotComplete);
        s.mark_open(Some("chat".to_string()));
        assert!(s.is_open());
        assert_eq!(s.protocol.as_deref(), Some("chat"));
    }

    #[test]
    fn close_before_open_transitions_to_closed_without_frame() {
        let mut s = CommonState::new(Role::Client, ConnectionConfig::default());
        s.close(WsCloseCode::Normal, "").expect("close");
        assert!(s.is_closed());
        assert!(!s.local_close_sent);
        assert!(!s.wants_write());
        // Second close is a no-op.
        s.close(WsCloseCode::GoingAway, "").expect("close again");
        assert!(!s.wants_write());
        // A fully closed state with empty rx keeps processing trivially Ok.
        let io = s.process_frames().expect("closed process");
        assert_eq!(io.pending_rx, 0);
        assert_eq!(io.pending_tx, 0);
    }

    #[test]
    fn data_sends_require_open_state_and_control_limits() {
        let mut s = CommonState::new(Role::Client, ConnectionConfig::default());
        assert!(matches!(s.send_text("x"), Err(WsError::InvalidState(_))));
        assert!(matches!(s.send_binary(b"x"), Err(WsError::InvalidState(_))));
        assert!(matches!(s.send_ping(b""), Err(WsError::InvalidState(_))));
        assert!(matches!(s.send_pong(b""), Err(WsError::InvalidState(_))));

        s.mark_open(None);
        s.send_ping(b"hi").expect("ping while open");
        let big = [0u8; 126];
        assert!(matches!(s.send_ping(&big), Err(WsError::Protocol { .. })));

        s.close(WsCloseCode::Normal, "").expect("close");
        assert!(matches!(s.send_text("x"), Err(WsError::InvalidState(_))));
        assert!(matches!(s.send_ping(b""), Err(WsError::InvalidState(_))));
        // Pong remains permitted while closing.
        s.send_pong(b"ok").expect("pong while closing");
    }

    #[test]
    fn client_masks_outbound_and_server_does_not() {
        let mut client = CommonState::new(Role::Client, ConnectionConfig::default());
        client.mark_open(None);
        client.send_text("hello").expect("client send");
        let client_wire = client.take_write();
        assert_eq!(client_wire[0], 0x81); // FIN + text
        assert_ne!(client_wire[1] & 0x80, 0); // mask bit set
        let (frame, consumed) =
            try_parse_frame(&client_wire, Role::Server, DEFAULT_MAX_FRAME_SIZE)
                .expect("parse")
                .expect("complete");
        assert_eq!(consumed, client_wire.len());
        assert_eq!(frame.payload, b"hello");

        let mut server = CommonState::new(Role::Server, ConnectionConfig::default());
        server.mark_open(None);
        server.send_text("hello").expect("server send");
        let server_wire = server.take_write();
        assert_eq!(server_wire[0], 0x81);
        assert_eq!(server_wire[1] & 0x80, 0); // mask bit clear
        let (frame2, _) =
            try_parse_frame(&server_wire, Role::Client, DEFAULT_MAX_FRAME_SIZE)
                .expect("parse")
                .expect("complete");
        assert_eq!(frame2.payload, b"hello");
    }

    #[test]
    fn send_text_encodes_multibyte_utf8_and_roundtrips() {
        let text = "héllo wörld — café 🙂";
        let mut client = CommonState::new(Role::Client, ConnectionConfig::default());
        client.mark_open(None);
        client.send_text(text).expect("send multibyte");
        let wire = client.take_write();
        let frame = parse_outbound_from_client(&wire);
        assert_eq!(frame.header.opcode, WsOpcode::Text);
        assert_eq!(frame.payload, text.as_bytes());

        let mut server = CommonState::new(Role::Server, ConnectionConfig::default());
        server.mark_open(None);
        server.feed(&wire);
        server.process_frames().expect("receive");
        assert_eq!(
            server.read_message(),
            Some(WsMessage::Text(text.to_string()))
        );
    }

    #[test]
    fn send_text_rejects_payload_over_max_message_size() {
        let config = ConnectionConfig {
            max_message_size: 4,
            ..ConnectionConfig::default()
        };
        let mut s = CommonState::new(Role::Client, config);
        s.mark_open(None);
        let err = s.send_text("hello").expect_err("too big");
        assert_eq!(err, WsError::MessageTooBig { size: 5, limit: 4 });
        assert!(!s.wants_write());
        // Multi-byte characters count as their UTF-8 byte length.
        let err2 = s.send_text("ééé").expect_err("multibyte too big");
        assert_eq!(err2, WsError::MessageTooBig { size: 6, limit: 4 });
    }

    #[test]
    fn byte_by_byte_feed_matches_all_at_once() {
        let mut sender = CommonState::new(Role::Server, ConnectionConfig::default());
        sender.mark_open(None);
        sender.send_text("hello world").expect("send");
        let bytes = sender.take_write();

        let mut incremental = open_client();
        for b in &bytes {
            incremental.feed(&[*b]);
            incremental.process_frames().expect("incremental");
        }

        let mut at_once = open_client();
        at_once.feed(&bytes);
        at_once.process_frames().expect("all at once");

        let m1 = incremental.read_message();
        let m2 = at_once.read_message();
        assert_eq!(m1, m2);
        assert_eq!(m1, Some(WsMessage::Text("hello world".to_string())));
        assert!(incremental.rx_buf().is_empty());
        assert!(at_once.rx_buf().is_empty());
        assert_eq!(incremental.pending_messages(), 0);
        assert_eq!(incremental.pending_tx_len(), at_once.pending_tx_len());
    }

    #[test]
    fn truncated_frame_waits_for_remaining_bytes() {
        let mut s = open_client();
        let wire = encode(&WsFrame::text(b"partial"), None);
        assert!(wire.len() > 4);
        s.feed(&wire[..3]);
        let io = s.process_frames().expect("partial");
        assert_eq!(io.pending_rx, 3);
        assert_eq!(io.pending_messages, 0);
        assert!(s.read_message().is_none());

        s.feed(&wire[3..]);
        let io2 = s.process_frames().expect("rest");
        assert_eq!(io2.pending_rx, 0);
        match s.read_message() {
            Some(WsMessage::Text(t)) => assert_eq!(t, "partial"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn continuation_without_start_is_protocol_error() {
        let mut s = open_client();
        let orphan = WsFrame::new(true, WsOpcode::Continuation, b"orphan".to_vec());
        s.feed(&encode(&orphan, None));
        let err = s.process_frames().expect_err("continuation");
        assert!(matches!(
            err,
            WsError::Protocol {
                close_code: WsCloseCode::ProtocolError,
                ..
            }
        ));
        assert!(s.is_closed());
        // A fail-close frame is queued before teardown.
        assert!(s.wants_write());
    }

    #[test]
    fn new_data_frame_while_fragmenting_is_protocol_error() {
        let mut s = open_client();
        let start = WsFrame::new(false, WsOpcode::Text, b"ab".to_vec());
        s.feed(&encode(&start, None));
        s.process_frames().expect("start fragment");
        s.feed(&encode(&WsFrame::text(b"cd"), None));
        let err = s.process_frames().expect_err("interleaved data");
        assert!(matches!(err, WsError::Protocol { .. }));
        assert!(s.is_closed());
    }

    #[test]
    fn fragmented_binary_and_utf8_text_reassemble_correctly() {
        let mut s = open_client();
        let f0 = WsFrame::new(false, WsOpcode::Binary, vec![1, 2]);
        let f1 = WsFrame::new(false, WsOpcode::Continuation, vec![3, 4]);
        let f2 = WsFrame::new(true, WsOpcode::Continuation, vec![5]);
        s.feed(&encode(&f0, None));
        s.feed(&encode(&f1, None));
        s.feed(&encode(&f2, None));
        s.process_frames().expect("binary fragments");
        match s.read_message() {
            Some(WsMessage::Binary(b)) => {
                assert_eq!(b.as_slice(), &[1, 2, 3, 4, 5]);
            }
            other => panic!("unexpected {other:?}"),
        }

        // Multi-byte UTF-8 (é = C3 A9) split across the fragment boundary.
        let t0 = WsFrame::new(false, WsOpcode::Text, b"caf\xc3".to_vec());
        let t1 = WsFrame::new(true, WsOpcode::Continuation, b"\xa9!".to_vec());
        s.feed(&encode(&t0, None));
        s.feed(&encode(&t1, None));
        s.process_frames().expect("utf8 fragments");
        match s.read_message() {
            Some(WsMessage::Text(t)) => assert_eq!(t, "caf\u{e9}!"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn broken_utf8_in_fragments_and_single_frames_fails() {
        // Second fragment does not continue the pending sequence.
        let mut s = open_client();
        let t0 = WsFrame::new(false, WsOpcode::Text, b"caf\xc3".to_vec());
        s.feed(&encode(&t0, None));
        s.process_frames().expect("first fragment");
        let t1 = WsFrame::new(true, WsOpcode::Continuation, b"X".to_vec());
        s.feed(&encode(&t1, None));
        let err = s.process_frames().expect_err("broken continuation");
        assert_eq!(err, WsError::InvalidUtf8);
        assert!(s.is_closed());

        // Complete single frame with invalid UTF-8.
        let mut s2 = open_client();
        s2.feed(&encode(&WsFrame::text(vec![0xFF, 0xFE]), None));
        let err2 = s2.process_frames().expect_err("invalid utf8");
        assert_eq!(err2, WsError::InvalidUtf8);
    }

    #[test]
    fn oversize_declared_length_rejected_before_payload_arrives() {
        let config = ConnectionConfig {
            max_frame_size: 1024,
            max_message_size: 1024,
            ..ConnectionConfig::default()
        };
        let mut s = CommonState::new(Role::Client, config);
        s.mark_open(None);
        // FIN + binary, 64-bit length = 8 MiB; only the 10-byte header is fed.
        let header = [0x82u8, 0x7F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00];
        s.feed(&header);
        let err = s.process_frames().expect_err("too big");
        assert_eq!(
            err,
            WsError::MessageTooBig {
                size: 8 * 1024 * 1024,
                limit: 1024,
            }
        );
        // The connection fails closed and queues a fail-close frame.
        assert!(s.is_closed());
        assert!(s.wants_write());
        assert_eq!(s.error.as_ref(), Some(&err));
    }

    #[test]
    fn oversized_single_frame_and_fragments_respect_max_message_size() {
        let config = ConnectionConfig {
            max_frame_size: 1024,
            max_message_size: 100,
            ..ConnectionConfig::default()
        };
        // Complete frame within frame limit but over message limit.
        let mut s = CommonState::new(Role::Client, config.clone());
        s.mark_open(None);
        s.feed(&encode(&WsFrame::binary(vec![7u8; 200]), None));
        let err = s.process_frames().expect_err("single frame");
        assert_eq!(
            err,
            WsError::MessageTooBig {
                size: 200,
                limit: 100,
            }
        );

        // Fragment accumulation crosses the limit mid-message.
        let mut s2 = CommonState::new(Role::Client, config);
        s2.mark_open(None);
        let start = WsFrame::new(false, WsOpcode::Binary, vec![1u8; 60]);
        s2.feed(&encode(&start, None));
        s2.process_frames().expect("first fragment");
        let cont = WsFrame::new(true, WsOpcode::Continuation, vec![2u8; 60]);
        s2.feed(&encode(&cont, None));
        let err2 = s2.process_frames().expect_err("accumulated");
        assert_eq!(
            err2,
            WsError::MessageTooBig {
                size: 120,
                limit: 100,
            }
        );
    }

    #[test]
    fn send_binary_rejects_payload_over_max_message_size() {
        let config = ConnectionConfig {
            max_message_size: 64,
            ..ConnectionConfig::default()
        };
        let mut s = CommonState::new(Role::Client, config);
        s.mark_open(None);
        let data = [0u8; 65];
        let err = s.send_binary(&data).expect_err("too big");
        assert_eq!(
            err,
            WsError::MessageTooBig {
                size: 65,
                limit: 64,
            }
        );
        assert!(!s.wants_write());
    }

    #[test]
    fn ping_triggers_auto_pong_and_pong_is_surfaced() {
        let mut s = open_client();
        let ping = WsFrame::ping(b"tick").expect("ping frame");
        s.feed(&encode(&ping, None));
        s.process_frames().expect("ping");
        match s.read_message() {
            Some(WsMessage::Ping(p)) => assert_eq!(p, b"tick"),
            other => panic!("unexpected {other:?}"),
        }
        let pong = parse_outbound_from_client(&s.take_write());
        assert_eq!(pong.header.opcode, WsOpcode::Pong);
        assert_eq!(pong.payload, b"tick");

        // Unsolicited pong is surfaced without queuing a reply.
        let unsolicited = WsFrame::pong(b"tock").expect("pong frame");
        s.feed(&encode(&unsolicited, None));
        s.process_frames().expect("pong");
        match s.read_message() {
            Some(WsMessage::Pong(p)) => assert_eq!(p, b"tock"),
            other => panic!("unexpected {other:?}"),
        }
        assert!(!s.wants_write());
    }

    #[test]
    fn auto_pong_can_be_disabled() {
        let config = ConnectionConfig {
            auto_pong: false,
            ..ConnectionConfig::default()
        };
        let mut s = CommonState::new(Role::Client, config);
        s.mark_open(None);
        let ping = WsFrame::ping(b"tick").expect("ping frame");
        s.feed(&encode(&ping, None));
        s.process_frames().expect("ping");
        assert!(matches!(s.read_message(), Some(WsMessage::Ping(_))));
        assert!(!s.wants_write());
    }

    #[test]
    fn peer_close_with_reason_auto_replies_and_closes() {
        let mut s = open_client();
        let close =
            WsFrame::close(Some(WsCloseCode::GoingAway), "bye").expect("close frame");
        s.feed(&encode(&close, None));
        s.process_frames().expect("close");
        match s.read_message() {
            Some(WsMessage::Close { code, reason }) => {
                assert_eq!(code, WsCloseCode::GoingAway);
                assert_eq!(reason, "bye");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(s.peer_close_received);
        assert_eq!(s.peer_close_code, Some(WsCloseCode::GoingAway));
        assert_eq!(s.peer_close_reason.as_deref(), Some("bye"));
        assert!(s.is_closed());
        assert!(s.local_close_sent);
        assert!(!s.wants_read());

        let reply = parse_outbound_from_client(&s.take_write());
        assert_eq!(reply.header.opcode, WsOpcode::Close);
        let (code, reason) = decode_close_payload(&reply.payload).expect("payload");
        assert_eq!(code, WsCloseCode::GoingAway);
        assert_eq!(reason, "");

        // Re-processing an already-closed empty state is Ok.
        let io = s.process_frames().expect("closed");
        assert_eq!(io.pending_rx, 0);
    }

    #[test]
    fn empty_close_payload_maps_to_no_status_and_replies_normal() {
        let mut s = open_client();
        let empty = WsFrame::new(true, WsOpcode::Close, Vec::new());
        s.feed(&encode(&empty, None));
        s.process_frames().expect("close");
        match s.read_message() {
            Some(WsMessage::Close { code, reason }) => {
                assert_eq!(code, WsCloseCode::NoStatusReceived);
                assert_eq!(reason, "");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(s.is_closed());
        let reply = parse_outbound_from_client(&s.take_write());
        let (code, _) = decode_close_payload(&reply.payload).expect("payload");
        assert_eq!(code, WsCloseCode::Normal);
    }

    #[test]
    fn forbidden_close_codes_and_one_byte_payloads_are_rejected() {
        // 1005 must not appear on the wire.
        let mut s = open_client();
        let bad = WsFrame::new(true, WsOpcode::Close, vec![0x03, 0xED]);
        s.feed(&encode(&bad, None));
        let err = s.process_frames().expect_err("reserved code");
        assert!(matches!(err, WsError::Protocol { .. }));
        assert!(s.is_closed());

        // Single-byte close payload is a protocol error.
        let mut s2 = open_client();
        let one = WsFrame::new(true, WsOpcode::Close, vec![0x03]);
        s2.feed(&encode(&one, None));
        assert!(matches!(s2.process_frames(), Err(WsError::Protocol { .. })));
    }

    #[test]
    fn local_close_then_peer_close_completes_handshake() {
        let mut s = open_client();
        s.close(WsCloseCode::GoingAway, "bye").expect("close");
        assert_eq!(s.state, ConnectionState::Closing);
        assert!(s.local_close_sent);

        let sent = parse_outbound_from_client(&s.take_write());
        assert_eq!(sent.header.opcode, WsOpcode::Close);
        let (code, reason) = decode_close_payload(&sent.payload).expect("payload");
        assert_eq!(code, WsCloseCode::GoingAway);
        assert_eq!(reason, "bye");

        let peer_close =
            WsFrame::close(Some(WsCloseCode::Normal), "").expect("close frame");
        s.feed(&encode(&peer_close, None));
        s.process_frames().expect("peer close");
        assert!(s.is_closed());
        assert_eq!(s.peer_close_code, Some(WsCloseCode::Normal));
        assert!(matches!(s.send_text("x"), Err(WsError::InvalidState(_))));
        assert_eq!(s.send_frame(&WsFrame::text(b"x")), Err(WsError::Closed));
        // Closing again remains a no-op.
        s.close(WsCloseCode::Normal, "").expect("idempotent");
    }

    #[test]
    fn auto_close_reply_can_be_disabled() {
        let config = ConnectionConfig {
            auto_close_reply: false,
            ..ConnectionConfig::default()
        };
        let mut s = CommonState::new(Role::Client, config);
        s.mark_open(None);
        let peer_close =
            WsFrame::close(Some(WsCloseCode::Normal), "").expect("close frame");
        s.feed(&encode(&peer_close, None));
        s.process_frames().expect("close");
        assert_eq!(s.state, ConnectionState::Closing);
        assert!(!s.local_close_sent);
        assert!(s.peer_close_received);
        assert!(!s.wants_write());
        // Consume the surfaced Close message first.
        assert!(matches!(s.read_message(), Some(WsMessage::Close { .. })));
        // Data arriving after the peer's close is dropped silently.
        s.feed(&encode(&WsFrame::text(b"late"), None));
        s.process_frames().expect("late data");
        assert!(s.read_message().is_none());
        assert!(s.rx_buf().is_empty());
        // A local close can still be initiated afterwards.
        s.close(WsCloseCode::Normal, "").expect("local close");
        assert_eq!(s.state, ConnectionState::Closed);
        assert!(s.local_close_sent);
        assert!(s.wants_write());
    }

    #[test]
    fn close_during_fragmented_message_stops_reassembly() {
        let mut s = open_client();
        let frag = WsFrame::new(false, WsOpcode::Binary, vec![1, 2, 3]);
        s.feed(&encode(&frag, None));
        s.process_frames().expect("fragment");
        let close =
            WsFrame::close(Some(WsCloseCode::Normal), "stop").expect("close frame");
        s.feed(&encode(&close, None));
        s.process_frames().expect("close");
        assert!(s.is_closed());
        // Only the Close message is surfaced; the fragment is discarded.
        assert_eq!(s.pending_messages(), 1);
        match s.read_message() {
            Some(WsMessage::Close { code, reason }) => {
                assert_eq!(code, WsCloseCode::Normal);
                assert_eq!(reason, "stop");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(s.read_message().is_none());
    }

    #[test]
    fn data_frames_after_close_are_dropped() {
        let mut s = open_client();
        s.close(WsCloseCode::Normal, "").expect("close");
        assert_eq!(s.state, ConnectionState::Closing);
        s.feed(&encode(&WsFrame::text(b"late"), None));
        s.process_frames().expect("dropped");
        assert!(s.read_message().is_none());
        assert!(s.rx_buf().is_empty());

        // Complete the close handshake, then feed data to a closed connection.
        let peer_close =
            WsFrame::close(Some(WsCloseCode::Normal), "").expect("close frame");
        s.feed(&encode(&peer_close, None));
        s.process_frames().expect("peer close");
        assert!(s.is_closed());
        s.feed(&encode(&WsFrame::text(b"later"), None));
        s.process_frames().expect("ignored while closed");
        assert!(s.rx_buf().is_empty());
        assert!(matches!(s.read_message(), Some(WsMessage::Close { .. })));
    }

    #[test]
    fn masking_direction_violations_are_rejected() {
        // Server must receive masked frames.
        let mut server = open_server();
        server.feed(&encode(&WsFrame::text(b"x"), None));
        let err = server.process_frames().expect_err("unmasked to server");
        assert!(matches!(err, WsError::Protocol { .. }));

        // Client must receive unmasked frames.
        let mut client = open_client();
        client.feed(&encode(&WsFrame::text(b"x"), Some([1, 2, 3, 4])));
        let err2 = client.process_frames().expect_err("masked to client");
        assert!(matches!(err2, WsError::Protocol { .. }));
    }

    #[test]
    fn rsv_bits_are_rejected() {
        let mut s = open_client();
        let mut frame = WsFrame::text(b"x");
        frame.header.rsv1 = true;
        s.feed(&encode(&frame, None));
        let err = s.process_frames().expect_err("rsv1");
        assert!(matches!(
            err,
            WsError::Protocol {
                close_code: WsCloseCode::ProtocolError,
                ..
            }
        ));
        assert!(s.is_closed());
    }

    #[test]
    fn reserved_opcodes_are_rejected() {
        let mut s = open_client();
        // FIN + opcode 0x3 (reserved non-control).
        s.feed(&[0x83, 0x00]);
        let err = s.process_frames().expect_err("reserved data opcode");
        assert!(matches!(err, WsError::Protocol { .. }));

        let mut s2 = open_client();
        // FIN + opcode 0xB (reserved control).
        s2.feed(&[0x8B, 0x00]);
        assert!(matches!(s2.process_frames(), Err(WsError::Protocol { .. })));
    }

    #[test]
    fn malformed_control_and_length_frames_are_rejected() {
        // Ping declared with a 16-bit length of 128 (> 125).
        let mut s = open_client();
        s.feed(&[0x89, 0x7E, 0x00, 0x80]);
        let err = s.process_frames().expect_err("oversize ping");
        assert!(matches!(err, WsError::Protocol { .. }));

        // Fragmented ping (FIN clear).
        let mut s2 = open_client();
        s2.feed(&[0x09, 0x00]);
        assert!(matches!(s2.process_frames(), Err(WsError::Protocol { .. })));

        // Non-minimal 16-bit payload length (5 encoded as 16-bit).
        let mut s3 = open_client();
        s3.feed(&[0x82, 0x7E, 0x00, 0x05]);
        assert!(matches!(s3.process_frames(), Err(WsError::Protocol { .. })));
    }

    #[test]
    fn wrong_masking_key_corrupts_text_and_fails_utf8() {
        let mut s = open_server();
        let mut wire = encode(&WsFrame::text(b"hello world"), Some([1, 2, 3, 4]));
        // Tamper the mask key (bytes 2..6 for a small masked frame): the
        // payload is then unmasked with the wrong key.
        for b in &mut wire[2..6] {
            *b ^= 0xFF;
        }
        s.feed(&wire);
        let err = s.process_frames().expect_err("bad unmask");
        assert_eq!(err, WsError::InvalidUtf8);
        assert!(s.is_closed());
    }

    #[test]
    fn bit_flipped_binary_payload_is_delivered_corrupted() {
        let mut s = open_client();
        let original = vec![0x41u8; 64];
        let mut wire = encode(&WsFrame::binary(original.clone()), None);
        let last = wire.len() - 1;
        wire[last] ^= 0x01;
        s.feed(&wire);
        s.process_frames().expect("parse");
        match s.read_message() {
            Some(WsMessage::Binary(b)) => {
                assert_ne!(b, original);
                assert_eq!(b.len(), original.len());
                assert_eq!(b[63], original[63] ^ 0x01);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn fail_sets_sticky_error_and_closes() {
        let mut s = open_client();
        let err = WsError::protocol(WsCloseCode::ProtocolError, "boom");
        let returned = s.fail(err.clone());
        assert_eq!(returned, err);
        assert!(s.is_closed());
        assert_eq!(s.error.as_ref(), Some(&err));
        assert!(s.wants_write());
        let again = s.process_frames().expect_err("sticky");
        assert_eq!(again, err);
    }

    #[test]
    fn buffer_and_readiness_helpers() {
        let mut s = CommonState::new(Role::Client, ConnectionConfig::default());
        assert!(!s.wants_write());
        assert!(s.wants_read());
        s.queue_raw(b"abc");
        assert_eq!(s.pending_tx_len(), 3);
        assert!(s.wants_write());
        assert_eq!(s.take_write(), b"abc");
        assert!(!s.wants_write());

        s.feed(b"12345");
        assert_eq!(s.rx_buf(), b"12345");
        s.consume_rx(2);
        assert_eq!(s.rx_buf(), b"345");
        s.consume_rx(100);
        assert!(s.rx_buf().is_empty());

        // Peer close stops further reads.
        let mut s2 = open_client();
        let empty_close = WsFrame::new(true, WsOpcode::Close, Vec::new());
        s2.feed(&encode(&empty_close, None));
        s2.process_frames().expect("close");
        assert!(!s2.wants_read());
    }

    #[test]
    fn ten_thousand_small_messages_are_delivered_in_order() {
        let mut sender = CommonState::new(Role::Server, ConnectionConfig::default());
        sender.mark_open(None);
        let mut receiver = open_client();
        for i in 0..10_000u32 {
            sender.send_text(&format!("m{i}")).expect("send");
            let bytes = sender.take_write();
            receiver.feed(&bytes);
            receiver.process_frames().expect("process");
        }
        assert_eq!(receiver.pending_messages(), 10_000);
        for i in 0..10_000u32 {
            match receiver.read_message() {
                Some(WsMessage::Text(t)) => assert_eq!(t, format!("m{i}")),
                other => panic!("expected text at {i}, got {other:?}"),
            }
        }
        assert_eq!(receiver.pending_messages(), 0);
        assert!(receiver.rx_buf().is_empty());
    }

    #[test]
    fn large_binary_roundtrip_covers_all_length_encodings() {
        let mut client = CommonState::new(Role::Client, ConnectionConfig::default());
        client.mark_open(None);
        let mut server = CommonState::new(Role::Server, ConnectionConfig::default());
        server.mark_open(None);

        // 7-bit (<=125), 16-bit (<=65535) and 64-bit length encodings.
        for size in [200usize, 2_000, 1024 * 1024] {
            let payload: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
            client.send_binary(&payload).expect("send");
            let wire = client.take_write();
            server.feed(&wire);
            let io = server.process_frames().expect("process");
            assert_eq!(io.pending_rx, 0, "size {size}");
            match server.read_message() {
                Some(WsMessage::Binary(b)) => {
                    assert_eq!(b.len(), size);
                    assert_eq!(b, payload);
                }
                other => panic!("size {size}: unexpected {other:?}"),
            }
            // Drain the server's tx so the buffers do not grow unbounded.
            server.take_write();
        }
    }
}
