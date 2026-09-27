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

//! [`std::io`] adapters over [`TlsClientConnection`] / [`TlsServerConnection`].

use crate::tls_client::TlsClientConnection;
use crate::tls_error::{TlsError, TlsResult};
use crate::tls_server::TlsServerConnection;
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
                self.conn
                    .process_new_packets()
                    .map_err(tls_to_io)?;
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
                    match self.conn.process_new_packets() {
                        Ok(()) => {}
                        Err(TlsError::Closed) => {
                            // Peer sent close_notify: app data queued before the
                            // alert must still be delivered, then EOF.
                            return match self.conn.read_app(buf) {
                                Ok(n) => Ok(n),
                                Err(TlsError::Closed) | Err(TlsError::WouldBlock) => Ok(0),
                                Err(e) => Err(tls_to_io(e)),
                            };
                        }
                        Err(e) => return Err(tls_to_io(e)),
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// Scripted in-memory transport: reads come from `rx`, writes land in `tx`.
    struct MemTransport {
        rx: Vec<u8>,
        pos: usize,
        tx: Vec<u8>,
        /// When set, the next read fails once with this kind.
        fail_read: Option<io::ErrorKind>,
    }

    impl MemTransport {
        fn new(rx: Vec<u8>) -> Self {
            Self { rx, pos: 0, tx: Vec::new(), fail_read: None }
        }
    }

    impl Read for MemTransport {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if let Some(kind) = self.fail_read.take() {
                return Err(io::Error::new(kind, "transport read failure"));
            }
            let avail = &self.rx[self.pos..];
            let n = avail.len().min(buf.len());
            buf[..n].copy_from_slice(&avail[..n]);
            self.pos += n;
            Ok(n)
        }
    }

    impl Write for MemTransport {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.tx.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// Scripted TLS session controlling every `TlsSession` call.
    #[derive(Default)]
    struct MockSession {
        handshaking: bool,
        /// Ciphertext emitted one chunk per `write_tls` call.
        out: VecDeque<Vec<u8>>,
        /// Bytes accepted by `read_tls`.
        tls_in: Vec<u8>,
        /// App plaintext available to `read_app`.
        app_rx: VecDeque<u8>,
        /// App plaintext captured by `write_app`.
        app_tx: Vec<Vec<u8>>,
        /// Error returned by the next `process_new_packets`.
        process_result: Option<TlsError>,
        /// App data delivered in the same flight as a `Closed` process result.
        appdata_with_close: Option<Vec<u8>>,
        /// Error returned by the next `read_app`.
        read_result: Option<TlsError>,
        /// Error returned by the next `write_app`.
        write_result: Option<TlsError>,
        process_calls: usize,
        close_notified: bool,
    }

    impl MockSession {
        fn idle() -> Self {
            Self { handshaking: false, ..Self::default() }
        }
    }

    impl TlsSession for MockSession {
        fn read_tls(&mut self, data: &[u8]) -> TlsResult<()> {
            self.tls_in.extend_from_slice(data);
            Ok(())
        }
        fn write_tls(&mut self) -> Vec<u8> {
            self.out.pop_front().unwrap_or_default()
        }
        fn process_new_packets(&mut self) -> TlsResult<()> {
            self.process_calls += 1;
            if let Some(data) = self.appdata_with_close.take() {
                self.app_rx.extend(data);
            }
            if let Some(err) = self.process_result.take() {
                if matches!(err, TlsError::Closed) {
                    self.close_notified = true;
                }
                return Err(err);
            }
            self.handshaking = false;
            Ok(())
        }
        fn read_app(&mut self, buf: &mut [u8]) -> TlsResult<usize> {
            if let Some(err) = self.read_result.take() {
                return Err(err);
            }
            if !self.app_rx.is_empty() {
                let n = buf.len().min(self.app_rx.len());
                for slot in buf.iter_mut().take(n) {
                    *slot = self.app_rx.pop_front().unwrap_or(0);
                }
                return Ok(n);
            }
            if self.close_notified {
                return Ok(0);
            }
            Err(TlsError::WouldBlock)
        }
        fn write_app(&mut self, data: &[u8]) -> TlsResult<()> {
            if let Some(err) = self.write_result.take() {
                return Err(err);
            }
            self.app_tx.push(data.to_vec());
            Ok(())
        }
        fn is_handshaking(&self) -> bool {
            self.handshaking
        }
        fn wants_write(&self) -> bool {
            !self.out.is_empty()
        }
        fn send_close_notify(&mut self) -> TlsResult<()> {
            self.out.push_back(b"close".to_vec());
            Ok(())
        }
    }

    #[test]
    fn handshake_flushes_flights_and_absorbs_input() {
        let transport = MemTransport::new(b"server flight".to_vec());
        let session = MockSession {
            handshaking: true,
            out: VecDeque::from(vec![b"client flight".to_vec()]),
            ..MockSession::default()
        };
        let mut stream = TlsStream::new(transport, session);
        stream.complete_handshake().unwrap();

        assert!(!stream.conn().is_handshaking());
        assert_eq!(stream.conn().process_calls, 1);
        assert_eq!(stream.conn().tls_in, b"server flight");
        let (t, _) = stream.into_inner();
        assert_eq!(t.tx, b"client flight");
    }

    #[test]
    fn handshake_with_no_pending_output_still_terminates() {
        let transport = MemTransport::new(b"resp".to_vec());
        let session = MockSession {
            handshaking: true,
            ..MockSession::default()
        };
        let mut stream = TlsStream::new(transport, session);
        stream.complete_handshake().unwrap();
        let (t, c) = stream.into_inner();
        assert!(t.tx.is_empty());
        assert!(!c.is_handshaking());
    }

    #[test]
    fn handshake_reports_transport_eof() {
        let transport = MemTransport::new(Vec::new());
        let session = MockSession {
            handshaking: true,
            ..MockSession::default()
        };
        let mut stream = TlsStream::new(transport, session);
        let err = stream.complete_handshake().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn handshake_propagates_process_errors() {
        let transport = MemTransport::new(b"data".to_vec());
        let session = MockSession {
            handshaking: true,
            process_result: Some(TlsError::certificate("bad cert")),
            ..MockSession::default()
        };
        let mut stream = TlsStream::new(transport, session);
        let err = stream.complete_handshake().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert!(err.to_string().contains("bad cert"));
    }

    #[test]
    fn read_delivers_buffered_app_data_without_transport_read() {
        let transport = MemTransport::new(Vec::new());
        let mut session = MockSession::idle();
        session.app_rx.extend(b"buffered data");
        let mut stream = TlsStream::new(transport, session);

        let mut buf = [0u8; 64];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"buffered data");
        // No transport read happened (rx stays empty).
        let (t, _) = stream.into_inner();
        assert_eq!(t.pos, 0);
    }

    #[test]
    fn read_pumps_transport_then_reports_unexpected_eof() {
        // The session never produces app data, so the read loop keeps pumping
        // the transport until it runs dry.
        let transport = MemTransport::new(b"cipher record".to_vec());
        let session = MockSession::idle();
        let mut stream = TlsStream::new(transport, session);

        let mut buf = [0u8; 64];
        let err = stream.read(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
        let (t, c) = stream.into_inner();
        assert_eq!(t.pos, c.tls_in.len());
        assert_eq!(c.tls_in, b"cipher record");
    }

    #[test]
    fn read_returns_eof_after_close_notify() {
        // close_notify arrives as a process error; no buffered app data.
        let transport = MemTransport::new(b"x".to_vec());
        let mut session = MockSession::idle();
        session.process_result = Some(TlsError::Closed);
        let mut stream = TlsStream::new(transport, session);

        let mut buf = [0u8; 16];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 0);
        assert!(stream.conn().close_notified);
    }

    #[test]
    fn read_delivers_data_queued_before_close_notify() {
        // Regression: app data processed in the same flight as close_notify
        // must be delivered before EOF (previously mapped to ConnectionAborted).
        let transport = MemTransport::new(b"x".to_vec());
        let mut session = MockSession::idle();
        session.process_result = Some(TlsError::Closed);
        session.appdata_with_close = Some(b"last words".to_vec());
        let mut stream = TlsStream::new(transport, session);

        let mut buf = [0u8; 64];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"last words");
        // Everything consumed: subsequent reads are EOF, not an error.
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn read_maps_session_errors_to_io_errors() {
        let transport = MemTransport::new(Vec::new());
        let mut session = MockSession::idle();
        session.read_result = Some(TlsError::certificate("decrypt failed"));
        let mut stream = TlsStream::new(transport, session);

        let mut buf = [0u8; 16];
        let err = stream.read(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert!(err.to_string().contains("decrypt failed"));
    }

    #[test]
    fn read_app_closed_directly_returns_eof() {
        let transport = MemTransport::new(Vec::new());
        let mut session = MockSession::idle();
        session.read_result = Some(TlsError::Closed);
        let mut stream = TlsStream::new(transport, session);

        let mut buf = [0u8; 16];
        assert_eq!(stream.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn read_app_handshake_not_complete_maps_to_would_block() {
        let transport = MemTransport::new(Vec::new());
        let mut session = MockSession::idle();
        session.read_result = Some(TlsError::HandshakeNotComplete);
        let mut stream = TlsStream::new(transport, session);

        let mut buf = [0u8; 16];
        let err = stream.read(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
    }

    #[test]
    fn read_propagates_would_block_from_session_without_error() {
        // Session returns WouldBlock, transport has no data yet -> UnexpectedEof
        // is the only outcome the synchronous adapter can surface.
        let transport = MemTransport::new(Vec::new());
        let session = MockSession::idle();
        let mut stream = TlsStream::new(transport, session);
        let mut buf = [0u8; 16];
        let err = stream.read(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn write_sends_app_data_through_session_and_flushes() {
        let transport = MemTransport::new(Vec::new());
        let mut session = MockSession::idle();
        session.out.push_back(b"pending record".to_vec());
        let mut stream = TlsStream::new(transport, session);

        let n = stream.write(b"hello world").unwrap();
        assert_eq!(n, 11);
        let (t, c) = stream.into_inner();
        assert_eq!(t.tx, b"pending record");
        assert_eq!(c.app_tx, vec![b"hello world".to_vec()]);
    }

    #[test]
    fn write_propagates_session_errors() {
        let transport = MemTransport::new(Vec::new());
        let mut session = MockSession::idle();
        session.write_result = Some(TlsError::Closed);
        let mut stream = TlsStream::new(transport, session);

        let err = stream.write(b"data").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionAborted);
        assert!(err.to_string().contains("closed"));
    }

    #[test]
    fn flush_drains_all_pending_tls_output() {
        let transport = MemTransport::new(Vec::new());
        let mut session = MockSession::idle();
        session.out = VecDeque::from(vec![b"one".to_vec(), b"two".to_vec()]);
        let mut stream = TlsStream::new(transport, session);
        stream.flush().unwrap();
        let (t, c) = stream.into_inner();
        assert_eq!(t.tx, b"onetwo");
        assert!(!c.wants_write());
    }

    #[test]
    fn write_completes_handshake_first() {
        let transport = MemTransport::new(b"peer flight".to_vec());
        let session = MockSession {
            handshaking: true,
            ..MockSession::default()
        };
        let mut stream = TlsStream::new(transport, session);
        let n = stream.write(b"after hs").unwrap();
        assert_eq!(n, 8);
        assert!(!stream.conn().is_handshaking());
        assert_eq!(stream.conn().tls_in, b"peer flight");
    }

    #[test]
    fn send_close_notify_queues_output() {
        let transport = MemTransport::new(Vec::new());
        let session = MockSession::idle();
        let mut stream = TlsStream::new(transport, session);
        stream.conn_mut().send_close_notify().unwrap();
        assert!(stream.conn().wants_write());
        stream.flush().unwrap();
        let (t, _) = stream.into_inner();
        assert_eq!(t.tx, b"close");
    }

    #[test]
    fn conn_accessors_and_into_inner() {
        let transport = MemTransport::new(Vec::new());
        let session = MockSession::idle();
        let mut stream = TlsStream::new(transport, session);
        assert!(!stream.conn().is_handshaking());
        stream.conn_mut().app_rx.push_back(1);
        assert_eq!(stream.conn().app_rx.len(), 1);
        let (t, c) = stream.into_inner();
        assert_eq!(c.app_rx.len(), 1);
        assert!(t.tx.is_empty());
    }

    #[test]
    fn tls_to_io_maps_error_kinds() {
        assert_eq!(
            tls_to_io(TlsError::WouldBlock).kind(),
            io::ErrorKind::WouldBlock
        );
        assert_eq!(
            tls_to_io(TlsError::Closed).kind(),
            io::ErrorKind::ConnectionAborted
        );
        assert_eq!(
            tls_to_io(TlsError::HandshakeNotComplete).kind(),
            io::ErrorKind::WouldBlock
        );
        assert_eq!(
            tls_to_io(TlsError::certificate("x")).kind(),
            io::ErrorKind::Other
        );
        assert_eq!(
            tls_to_io(TlsError::RandomFailed).kind(),
            io::ErrorKind::Other
        );
    }

    #[test]
    fn read_propagates_transport_read_errors() {
        let mut transport = MemTransport::new(Vec::new());
        transport.fail_read = Some(io::ErrorKind::ConnectionReset);
        let mut session = MockSession::idle();
        session.app_rx.push_back(b'x');
        let mut stream = TlsStream::new(transport, session);
        let mut buf = [0u8; 4];
        // First read is served from the buffer.
        assert_eq!(stream.read(&mut buf).unwrap(), 1);
        // Second read needs the transport, which now fails.
        let err = stream.read(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionReset);
    }
}
