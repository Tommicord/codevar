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

//! [`std::io`] adapters over [`TlsClientConnection`] / [`TlsServerConnection`].

use crate::network::tls_client::TlsClientConnection;
use crate::network::tls_error::{TlsError, TlsResult};
use crate::network::tls_server::TlsServerConnection;
use std::io::{self, Read, Write};

/// Trait shared by client and server connections for the stream adapter.
pub trait TlsSession {
    /// Feeds transport ciphertext.
    fn read_tls(&mut self, data: &[u8]) -> TlsResult<()>;
    /// Drains transport ciphertext.
    fn write_tls(&mut self) -> Vec<u8>;
    /// Processes records.
    fn process_new_packets(&mut self) -> TlsResult<()>;
    /// Reads application data.
    fn read_app(&mut self, buf: &mut [u8]) -> TlsResult<usize>;
    /// Writes application data.
    fn write_app(&mut self, data: &[u8]) -> TlsResult<()>;
    /// Whether handshake is in progress.
    fn is_handshaking(&self) -> bool;
    /// Whether outbound TLS bytes are pending.
    fn wants_write(&self) -> bool;
    /// Sends close_notify.
    fn send_close_notify(&mut self) -> TlsResult<()>;
}

impl TlsSession for TlsClientConnection {
    fn read_tls(&mut self, data: &[u8]) -> TlsResult<()> {
        TlsClientConnection::read_tls(self, data)
    }
    fn write_tls(&mut self) -> Vec<u8> {
        TlsClientConnection::write_tls(self)
    }
    fn process_new_packets(&mut self) -> TlsResult<()> {
        TlsClientConnection::process_new_packets(self).map(|_| ())
    }
    fn read_app(&mut self, buf: &mut [u8]) -> TlsResult<usize> {
        self.reader_read(buf)
    }
    fn write_app(&mut self, data: &[u8]) -> TlsResult<()> {
        self.writer_write(data)
    }
    fn is_handshaking(&self) -> bool {
        TlsClientConnection::is_handshaking(self)
    }
    fn wants_write(&self) -> bool {
        TlsClientConnection::wants_write(self)
    }
    fn send_close_notify(&mut self) -> TlsResult<()> {
        TlsClientConnection::send_close_notify(self)
    }
}

impl TlsSession for TlsServerConnection {
    fn read_tls(&mut self, data: &[u8]) -> TlsResult<()> {
        TlsServerConnection::read_tls(self, data)
    }
    fn write_tls(&mut self) -> Vec<u8> {
        TlsServerConnection::write_tls(self)
    }
    fn process_new_packets(&mut self) -> TlsResult<()> {
        TlsServerConnection::process_new_packets(self).map(|_| ())
    }
    fn read_app(&mut self, buf: &mut [u8]) -> TlsResult<usize> {
        self.reader_read(buf)
    }
    fn write_app(&mut self, data: &[u8]) -> TlsResult<()> {
        self.writer_write(data)
    }
    fn is_handshaking(&self) -> bool {
        TlsServerConnection::is_handshaking(self)
    }
    fn wants_write(&self) -> bool {
        TlsServerConnection::wants_write(self)
    }
    fn send_close_notify(&mut self) -> TlsResult<()> {
        TlsServerConnection::send_close_notify(self)
    }
}

/// Bidirectional TLS stream wrapping a byte-oriented transport.
///
/// Completes the handshake on first I/O, then encrypts application data.
pub struct TlsStream<S, C> {
    transport: S,
    conn: C,
}

impl<S, C> TlsStream<S, C>
where
    S: Read + Write,
    C: TlsSession,
{
    /// Wraps a transport and TLS session.
    #[must_use]
    pub fn new(transport: S, conn: C) -> Self {
        Self { transport, conn }
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

    /// Consumes the stream, returning transport and session.
    #[must_use]
    pub fn into_inner(self) -> (S, C) {
        (self.transport, self.conn)
    }

    /// Drives the handshake to completion.
    pub fn complete_handshake(&mut self) -> io::Result<()> {
        while self.conn.is_handshaking() {
            self.flush_tls()?;
            if self.conn.is_handshaking() {
                self.absorb_tls()?;
                self.conn.process_new_packets().map_err(tls_to_io)?;
            }
        }
        self.flush_tls()?;
        Ok(())
    }

    fn flush_tls(&mut self) -> io::Result<()> {
        while self.conn.wants_write() {
            let buf = self.conn.write_tls();
            if buf.is_empty() {
                break;
            }
            self.transport.write_all(&buf)?;
        }
        self.transport.flush()?;
        Ok(())
    }

    fn absorb_tls(&mut self) -> io::Result<()> {
        let mut buf = [0u8; 16_384];
        let n = self.transport.read(&mut buf)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "TLS transport closed during handshake",
            ));
        }
        self.conn.read_tls(&buf[..n]).map_err(tls_to_io)?;
        Ok(())
    }
}

impl<S, C> Read for TlsStream<S, C>
where
    S: Read + Write,
    C: TlsSession,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.complete_handshake()?;
        loop {
            match self.conn.read_app(buf) {
                Ok(n) => return Ok(n),
                Err(TlsError::WouldBlock) => {
                    self.absorb_tls()?;
                    self.conn.process_new_packets().map_err(tls_to_io)?;
                }
                Err(TlsError::Closed) => return Ok(0),
                Err(e) => return Err(tls_to_io(e)),
            }
        }
    }
}

impl<S, C> Write for TlsStream<S, C>
where
    S: Read + Write,
    C: TlsSession,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.complete_handshake()?;
        self.conn.write_app(buf).map_err(tls_to_io)?;
        self.flush_tls()?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_tls()
    }
}

fn tls_to_io(err: TlsError) -> io::Error {
    let kind = match &err {
        TlsError::WouldBlock => io::ErrorKind::WouldBlock,
        TlsError::Closed => io::ErrorKind::ConnectionAborted,
        TlsError::HandshakeNotComplete => io::ErrorKind::WouldBlock,
        _ => io::ErrorKind::Other,
    };
    io::Error::new(kind, err.to_string())
}
