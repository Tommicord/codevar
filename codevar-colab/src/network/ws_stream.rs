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

//! [`std::io`] adapter over a WebSocket connection.
//!
//! [`WebSocketStream`] wraps any [`Read`] + [`Write`] transport together with
//! a [`ClientConnection`] or [`ServerConnection`] and presents a byte-stream
//! view of *data* messages (text bytes or binary payloads). Control frames
//! are handled by the underlying connection (auto-pong / close echo).

use crate::network::ws_client::ClientConnection;
use crate::network::ws_error::{WsError, WsResult};
use crate::network::ws_ids::WsCloseCode;
use crate::network::ws_message::WsMessage;
use crate::network::ws_server::ServerConnection;
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
                return Err(WsError::handshake(
                    "transport closed during WebSocket handshake",
                ));
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
            self.complete_handshake()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        }
        self.fill_app_rx()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
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
            self.complete_handshake()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        }
        self.conn
            .send_binary(buf)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        self.flush_ws()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_ws()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        self.transport.flush()
    }
}
