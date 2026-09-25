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

//! [`std::io`] adapter over a WebSocket connection.
//!
//! [`WebSocketStream`] wraps any [`Read`] + [`Write`] transport together with
//! a [`ClientConnection`] or [`ServerConnection`] and presents a byte-stream
//! view of *data* messages (text bytes or binary payloads). Control frames
//! are handled by the underlying connection (auto-pong / close echo).

use crate::ws_client::ClientConnection;
use crate::ws_error::{WsError, WsResult};
use crate::ws_ids::WsCloseCode;
use crate::ws_message::WsMessage;
use crate::ws_server::ServerConnection;
use std::io::{self, Read, Write};

/// Trait shared by client and server connections for the stream adapter.
pub trait WsSession {
    /// Feeds transport bytes.
    fn feed(&mut self, data: &[u8]) -> WsResult<()>;
    /// Drains transport bytes.
    fn take_write(&mut self) -> Vec<u8>;
    /// Advances protocol state.
    fn process(&mut self) -> WsResult<()>;
    /// Whether the handshake is still running.
    fn is_handshaking(&self) -> bool;
    /// Whether the connection is open for data.
    fn is_open(&self) -> bool;
    /// Whether outbound bytes are pending.
    fn wants_write(&self) -> bool;
    /// Sends a binary data message.
    fn send_binary(&mut self, data: &[u8]) -> WsResult<()>;
    /// Reads the next complete message.
    fn read_message(&mut self) -> WsResult<Option<WsMessage>>;
    /// Starts the closing handshake.
    fn close(&mut self, code: WsCloseCode, reason: &str) -> WsResult<()>;
}

impl WsSession for ClientConnection {
    fn feed(&mut self, data: &[u8]) -> WsResult<()> {
        ClientConnection::feed(self, data)
    }
    fn take_write(&mut self) -> Vec<u8> {
        ClientConnection::take_write(self)
    }
    fn process(&mut self) -> WsResult<()> {
        ClientConnection::process(self).map(|_| ())
    }
    fn is_handshaking(&self) -> bool {
        ClientConnection::is_handshaking(self)
    }
    fn is_open(&self) -> bool {
        ClientConnection::is_open(self)
    }
    fn wants_write(&self) -> bool {
        ClientConnection::wants_write(self)
    }
    fn send_binary(&mut self, data: &[u8]) -> WsResult<()> {
        ClientConnection::send_binary(self, data)
    }
    fn read_message(&mut self) -> WsResult<Option<WsMessage>> {
        ClientConnection::read_message(self)
    }
    fn close(&mut self, code: WsCloseCode, reason: &str) -> WsResult<()> {
        ClientConnection::close(self, code, reason)
    }
}

impl WsSession for ServerConnection {
    fn feed(&mut self, data: &[u8]) -> WsResult<()> {
        ServerConnection::feed(self, data)
    }
    fn take_write(&mut self) -> Vec<u8> {
        ServerConnection::take_write(self)
    }
    fn process(&mut self) -> WsResult<()> {
        ServerConnection::process(self).map(|_| ())
    }
    fn is_handshaking(&self) -> bool {
        ServerConnection::is_handshaking(self)
    }
    fn is_open(&self) -> bool {
        ServerConnection::is_open(self)
    }
    fn wants_write(&self) -> bool {
        ServerConnection::wants_write(self)
    }
    fn send_binary(&mut self, data: &[u8]) -> WsResult<()> {
        ServerConnection::send_binary(self, data)
    }
    fn read_message(&mut self) -> WsResult<Option<WsMessage>> {
        ServerConnection::read_message(self)
    }
    fn close(&mut self, code: WsCloseCode, reason: &str) -> WsResult<()> {
        ServerConnection::close(self, code, reason)
    }
}

/// Bidirectional WebSocket stream wrapping a byte-oriented transport.
///
/// Completes the handshake on first I/O, then exposes data-message payloads
/// through [`Read`] / [`Write`]. Text messages are delivered as their UTF-8
/// bytes; binary messages are delivered verbatim. Ping / Pong / Close are
/// consumed by the session and not returned from [`Read`].
pub struct WebSocketStream<S, C> {
    transport: S,
    conn: C,
    /// Leftover application bytes from a partially consumed message.
    app_rx: Vec<u8>,
    app_rx_pos: usize,
}

impl<S, C> WebSocketStream<S, C>
where
    S: Read + Write,
    C: WsSession,
{
    /// Wraps a transport and WebSocket session.
    #[must_use]
    pub fn new(transport: S, conn: C) -> Self {
        Self {
            transport,
            conn,
            app_rx: Vec::new(),
            app_rx_pos: 0,
        }
    }

    /// Immutable access to the session.
    #[must_use]
    pub fn conn(&self) -> &C {
        &self.conn
    }

    /// Mutable access to the session.
    pub fn conn_mut(&mut self) -> &mut C {
        &mut self.conn
    }

    /// Consumes the stream, returning the transport and session.
    #[must_use]
    pub fn into_inner(self) -> (S, C) {
        (self.transport, self.conn)
    }

    /// Drives the handshake until it completes or an error occurs.
    pub fn complete_handshake(&mut self) -> WsResult<()> {
        while self.conn.is_handshaking() {
            self.flush_ws()?;
            let mut buf = [0u8; 4096];
            let n = self.transport.read(&mut buf)?;
            if n == 0 {
                return Err(WsError::handshake("transport closed during WebSocket handshake"));
            }
            self.conn.feed(&buf[..n])?;
            self.conn.process()?;
            self.flush_ws()?;
        }
        Ok(())
    }

    /// Performs a clean Close handshake then shuts down writes on the transport.
    pub fn close(&mut self) -> WsResult<()> {
        self.conn.close(WsCloseCode::Normal, "")?;
        self.flush_ws()?;
        Ok(())
    }

    fn flush_ws(&mut self) -> WsResult<()> {
        let out = self.conn.take_write();
        if !out.is_empty() {
            self.transport.write_all(&out)?;
            self.transport.flush()?;
        }
        Ok(())
    }

    fn fill_app_rx(&mut self) -> WsResult<()> {
        if self.app_rx_pos < self.app_rx.len() {
            return Ok(());
        }
        self.app_rx.clear();
        self.app_rx_pos = 0;
        loop {
            if let Some(msg) = self.conn.read_message()? {
                match msg {
                    WsMessage::Text(t) => {
                        self.app_rx = t.into_bytes();
                        return Ok(());
                    }
                    WsMessage::Binary(b) => {
                        self.app_rx = b;
                        return Ok(());
                    }
                    WsMessage::Ping(_) | WsMessage::Pong(_) => continue,
                    WsMessage::Close { .. } => return Err(WsError::Closed),
                }
            }
            self.flush_ws()?;
            let mut buf = [0u8; 8192];
            let n = self.transport.read(&mut buf)?;
            if n == 0 {
                return Err(WsError::Closed);
            }
            self.conn.feed(&buf[..n])?;
            self.conn.process()?;
            self.flush_ws()?;
        }
    }
}

impl<S, C> Read for WebSocketStream<S, C>
where
    S: Read + Write,
    C: WsSession,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.conn.is_handshaking() {
            self.complete_handshake().map_err(io::Error::other)?;
        }
        self.fill_app_rx().map_err(io::Error::other)?;
        let available = &self.app_rx[self.app_rx_pos..];
        let n = available.len().min(buf.len());
        buf[..n].copy_from_slice(&available[..n]);
        self.app_rx_pos += n;
        Ok(n)
    }
}

impl<S, C> Write for WebSocketStream<S, C>
where
    S: Read + Write,
    C: WsSession,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.conn.is_handshaking() {
            self.complete_handshake().map_err(io::Error::other)?;
        }
        self.conn.send_binary(buf).map_err(io::Error::other)?;
        self.flush_ws().map_err(io::Error::other)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_ws().map_err(io::Error::other)?;
        self.transport.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ws_connection::ConnectionState;
    use crate::ws_frame::{WsFrame, try_parse_frame};
    use crate::ws_ids::{DEFAULT_MAX_FRAME_SIZE, Role, WsOpcode};

    /// In-memory transport: scripted reads, recorded writes.
    struct MemTransport {
        input: Vec<u8>,
        pos: usize,
        output: Vec<u8>,
        max_read: Option<usize>,
    }

    impl MemTransport {
        fn new(input: Vec<u8>, max_read: Option<usize>) -> Self {
            Self {
                input,
                pos: 0,
                output: Vec::new(),
                max_read,
            }
        }

        fn push_input(&mut self, data: &[u8]) {
            self.input.extend_from_slice(data);
        }
    }

    impl Read for MemTransport {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            let avail = self.input.len() - self.pos;
            if avail == 0 {
                return Ok(0); // EOF
            }
            let mut n = avail.min(buf.len());
            if let Some(max) = self.max_read {
                n = n.min(max);
            }
            if n == 0 {
                return Ok(0);
            }
            buf[..n].copy_from_slice(&self.input[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    impl Write for MemTransport {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.output.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// Builds a client stream whose transport already holds a valid 101
    /// response (the Upgrade request is drained to derive it).
    fn open_client_stream(max_read: Option<usize>) -> WebSocketStream<MemTransport, ClientConnection> {
        let mut client = ClientConnection::connect("/", "localhost", None).expect("connect");
        let request = client.take_write();
        let mut server = ServerConnection::accept(None).expect("accept");
        server.feed(&request).expect("feed");
        server.process().expect("process");
        assert!(server.is_open());
        let response = server.take_write();
        WebSocketStream::new(MemTransport::new(response, max_read), client)
    }

    fn unmasked(frame: &WsFrame) -> Vec<u8> {
        let mut out = Vec::new();
        frame.encode(&mut out, None).expect("encode");
        out
    }

    fn parse_from_server(bytes: &[u8]) -> WsFrame {
        let (frame, _) = try_parse_frame(bytes, Role::Server, DEFAULT_MAX_FRAME_SIZE)
            .expect("parse")
            .expect("complete");
        frame
    }

    #[test]
    fn complete_handshake_reads_preloaded_response() {
        let mut stream = open_client_stream(None);
        assert!(stream.conn().is_handshaking());
        stream.complete_handshake().expect("handshake");
        assert!(stream.conn().is_open());
        assert!(!stream.conn().is_handshaking());
        // Completing twice is a no-op.
        stream.complete_handshake().expect("idempotent");
    }

    #[test]
    fn eof_during_handshake_errors_and_request_was_flushed() {
        let client = ClientConnection::connect("/", "localhost", None).expect("connect");
        let mut stream = WebSocketStream::new(MemTransport::new(vec![], None), client);
        let err = stream.complete_handshake().expect_err("eof");
        assert!(matches!(err, WsError::Handshake(_)));
        assert!(
            err.to_string()
                .contains("transport closed during WebSocket handshake")
        );
        // The Upgrade request was flushed to the transport before the read.
        assert!(stream.transport.output.starts_with(b"GET / "));
        assert!(stream.conn().is_handshaking());
    }

    #[test]
    fn handshake_completes_with_single_byte_reads() {
        let mut stream = open_client_stream(Some(1));
        stream.complete_handshake().expect("handshake");
        assert!(stream.conn().is_open());
    }

    #[test]
    fn read_maps_handshake_eof_to_io_error() {
        let client = ClientConnection::connect("/", "localhost", None).expect("connect");
        let mut stream = WebSocketStream::new(MemTransport::new(vec![], None), client);
        let mut buf = [0u8; 8];
        let err = stream.read(&mut buf).expect_err("eof");
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert!(err.to_string().contains("transport closed"));
    }

    #[test]
    fn read_delivers_text_and_binary_payloads() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");

        stream
            .transport
            .push_input(&unmasked(&WsFrame::text(b"hello world")));
        let mut buf = [0u8; 64];
        let n = stream.read(&mut buf).expect("read");
        assert_eq!(&buf[..n], b"hello world");

        stream
            .transport
            .push_input(&unmasked(&WsFrame::binary(vec![9u8, 8, 7])));
        let n2 = stream.read(&mut buf).expect("read");
        assert_eq!(&buf[..n2], &[9, 8, 7]);
    }

    #[test]
    fn read_with_small_buffer_returns_message_in_chunks() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");
        stream
            .transport
            .push_input(&unmasked(&WsFrame::text(b"hello world")));

        let mut buf = [0u8; 5];
        let n = stream.read(&mut buf).expect("first chunk");
        assert_eq!(&buf[..n], b"hello");

        let n2 = stream.read(&mut buf).expect("second chunk");
        assert_eq!(&buf[..n2], b" worl");

        let n3 = stream.read(&mut buf).expect("third chunk");
        assert_eq!(&buf[..n3], b"d");
    }

    #[test]
    fn read_with_single_byte_transport_reads_still_assembles_frame() {
        let mut stream = open_client_stream(Some(1));
        stream.complete_handshake().expect("handshake");
        stream
            .transport
            .push_input(&unmasked(&WsFrame::text(b"chunked")));
        let mut buf = [0u8; 16];
        let n = stream.read(&mut buf).expect("read");
        assert_eq!(&buf[..n], b"chunked");
    }

    #[test]
    fn empty_message_read_returns_zero_but_stream_stays_usable() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");
        stream.transport.push_input(&unmasked(&WsFrame::text(b"")));

        let mut buf = [0u8; 8];
        // Quirk: an empty data message surfaces as Ok(0) from Read.
        let n = stream.read(&mut buf).expect("empty message");
        assert_eq!(n, 0);
        assert!(stream.conn().is_open());

        // The next message is still delivered.
        stream
            .transport
            .push_input(&unmasked(&WsFrame::text(b"hi")));
        let n2 = stream.read(&mut buf).expect("next message");
        assert_eq!(&buf[..n2], b"hi");
    }

    #[test]
    fn eof_mid_stream_maps_to_closed_io_error() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");
        // Transport input exhausted; no message buffered.
        let mut buf = [0u8; 8];
        let err = stream.read(&mut buf).expect_err("eof");
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert!(err.to_string().contains("closed"));
    }

    #[test]
    fn peer_close_message_surfaces_closed_error() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");
        let close = WsFrame::close(None, "").expect("close frame");
        stream.transport.push_input(&unmasked(&close));
        let mut buf = [0u8; 8];
        let err = stream.read(&mut buf).expect_err("peer close");
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert!(err.to_string().contains("closed"));
    }

    #[test]
    fn control_frames_are_not_delivered_as_data() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");
        let ping = WsFrame::ping(b"pp").expect("ping frame");
        let mut input = unmasked(&ping);
        let pong = WsFrame::pong(b"qq").expect("pong frame");
        input.extend(unmasked(&pong));
        input.extend(unmasked(&WsFrame::text(b"data")));
        stream.transport.push_input(&input);

        let mut buf = [0u8; 16];
        let n = stream.read(&mut buf).expect("read");
        assert_eq!(&buf[..n], b"data");
    }

    #[test]
    fn write_sends_binary_frame_and_returns_buffer_len() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");
        let written_before = stream.transport.output.len();

        let n = stream.write(b"abc").expect("write");
        assert_eq!(n, 3);
        assert!(stream.transport.output.len() > written_before);

        let frame = parse_from_server(&stream.transport.output[written_before..]);
        assert_eq!(frame.header.opcode, WsOpcode::Binary);
        assert_eq!(frame.payload, b"abc");
    }

    #[test]
    fn write_empty_buffer_succeeds_and_queues_empty_frame() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");
        let written_before = stream.transport.output.len();

        let n = stream.write(&[]).expect("empty write");
        assert_eq!(n, 0);
        assert!(stream.transport.output.len() > written_before);

        let frame = parse_from_server(&stream.transport.output[written_before..]);
        assert_eq!(frame.header.opcode, WsOpcode::Binary);
        assert!(frame.payload.is_empty());
    }

    #[test]
    fn write_before_handshake_completes_it_first() {
        let mut client = ClientConnection::connect("/", "localhost", None).expect("connect");
        let request = client.take_write();
        let mut server = ServerConnection::accept(None).expect("accept");
        server.feed(&request).expect("feed");
        server.process().expect("process");
        let response = server.take_write();

        let mut stream = WebSocketStream::new(MemTransport::new(response, None), client);
        assert!(stream.conn().is_handshaking());
        let n = stream.write(b"xy").expect("write");
        assert_eq!(n, 2);
        assert!(stream.conn().is_open());
        // The request was drained before wrapping, so the output holds only
        // the binary frame queued by this write.
        let frame = parse_from_server(&stream.transport.output);
        assert_eq!(frame.header.opcode, WsOpcode::Binary);
        assert_eq!(frame.payload, b"xy");
    }

    #[test]
    fn flush_pushes_queued_session_bytes_to_transport() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");
        let before = stream.transport.output.len();
        stream.conn_mut().send_binary(b"queued").expect("queue");
        assert_eq!(stream.transport.output.len(), before);
        stream.flush().expect("flush");
        assert!(stream.transport.output.len() > before);
        let frame = parse_from_server(&stream.transport.output[before..]);
        assert_eq!(frame.payload, b"queued");
    }

    #[test]
    fn close_queues_and_flushes_close_frame() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");
        let before = stream.transport.output.len();
        stream.close().expect("close");
        assert!(stream.transport.output.len() > before);
        let frame = parse_from_server(&stream.transport.output[before..]);
        assert_eq!(frame.header.opcode, WsOpcode::Close);
        assert_eq!(stream.conn().state(), ConnectionState::Closing);
    }

    #[test]
    fn into_inner_returns_transport_and_session() {
        let mut stream = open_client_stream(None);
        stream.complete_handshake().expect("handshake");
        stream.write_all(b"z").expect("write");
        let (transport, conn) = stream.into_inner();
        assert!(conn.is_open());
        assert!(!transport.output.is_empty());
    }
}
