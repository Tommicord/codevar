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

//! A synchronous D-Bus connection.
//!
//! `Connection<T>` owns a transport `T` implementing [`DbusTransport`]
//! and provides a small synchronous API: authentication, method
//! calls, signal emission, and message reception. The connection
//! buffers written bytes and queues incoming signals until the
//! caller consumes them.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use crate::dbus_auth::AuthSession;
use crate::dbus_error::{DbusError, DbusResult};
use crate::dbus_message::{BodyWriter, DbusMessage, DbusMessageStream, MessageKind};
use crate::dbus_transport::{DbusPollEvents, DbusTransport};

/// Default timeout for a method call, matching the D-Bus
/// specification default of 25 seconds.
pub const DEFAULT_CALL_TIMEOUT: core::time::Duration =
    core::time::Duration::from_millis(25_000);

/// A synchronous D-Bus connection.
pub struct Connection<T: DbusTransport> {
    transport: T,
    stream: DbusMessageStream,
    tx: Vec<u8>,
    tx_pos: usize,
    needs_write: bool,
    next_serial: u32,
    unique_name: Option<String>,
    incoming: VecDeque<DbusMessage>,
    pending_replies: Vec<(u32, DbusMessage)>,
}

impl<T: DbusTransport> Connection<T> {
    /// Creates a new connection backed by `transport`.
    #[must_use]
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            stream: DbusMessageStream::new(),
            tx: Vec::new(),
            tx_pos: 0,
            needs_write: false,
            next_serial: 1,
            unique_name: None,
            incoming: VecDeque::new(),
            pending_replies: Vec::new(),
        }
    }

    /// Returns the transport.
    #[must_use]
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Returns the transport mutably.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Consumes the connection and returns the transport.
    #[must_use]
    pub fn into_transport(self) -> T {
        self.transport
    }

    /// Performs the SASL authentication handshake using `uid`.
    ///
    /// Blocks until the server accepts the connection.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Auth`] when the server rejects all
    /// mechanisms or disconnects.
    pub fn authenticate(&mut self, uid: u32) -> DbusResult<()> {
        let mut session = AuthSession::with_external(uid);
        loop {
            match session.poll()? {
                crate::dbus_auth::AuthPoll::Write(bytes) => {
                    let mut cursor = &bytes[..];
                    while !cursor.is_empty() {
                        match self.transport.write(cursor) {
                            Ok(0) => return Err(DbusError::Disconnected),
                            Ok(n) => cursor = &cursor[n..],
                            Err(DbusError::WouldBlock) => {
                                self.wait_write(None)?;
                            }
                            Err(e) => return Err(e),
                        }
                    }
                }
                crate::dbus_auth::AuthPoll::Read => {
                    let mut buf = [0u8; 4096];
                    match self.transport.read(&mut buf) {
                        Ok(0) => return Err(DbusError::Disconnected),
                        Ok(n) => session.feed(&buf[..n])?,
                        Err(DbusError::WouldBlock) => {
                            self.wait_read(None)?;
                        }
                        Err(e) => return Err(e),
                    }
                }
                crate::dbus_auth::AuthPoll::Done => break,
            }
        }
        Ok(())
    }

    /// Sends `Hello` and stores the bus unique name.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Remote`] if the bus reports an error,
    /// or [`DbusError::Timeout`] if no reply arrives in time.
    pub fn hello(&mut self) -> DbusResult<()> {
        let mut message = DbusMessage::method_call(
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
            "Hello",
        )?;
        let serial = self.next_serial()?;
        message.set_serial(serial)?;
        self.send(&mut message)?;
        let reply = self.wait_reply(serial, DEFAULT_CALL_TIMEOUT)?;
        let mut reader = reply.body_reader();
        let unique = reader.read_str()?;
        self.unique_name = Some(String::from(unique));
        Ok(())
    }

    /// Returns the unique bus name advertised by the server.
    #[must_use]
    pub fn unique_name(&self) -> Option<&str> {
        self.unique_name.as_deref()
    }

    /// Sends a method call and waits for its reply.
    ///
    /// `body` marshals the arguments; the closure must produce a
    /// valid D-Bus signature.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Remote`] if the bus replies with an
    /// error, [`DbusError::Timeout`] if no reply arrives in time,
    /// or [`DbusError::InvalidName`] if any name is malformed.
    pub fn call<F>(
        &mut self,
        destination: &str,
        path: &str,
        interface: &str,
        member: &str,
        body: F,
        timeout: core::time::Duration,
    ) -> DbusResult<DbusMessage>
    where
        F: FnOnce(&mut BodyWriter) -> DbusResult<()>,
    {
        let mut message = DbusMessage::method_call(destination, path, interface, member)?;
        let serial = self.next_serial()?;
        message.set_serial(serial)?;
        message.build_body(body)?;
        self.send(&mut message)?;
        let reply = self.wait_reply(serial, timeout)?;
        if reply.kind() == MessageKind::Error {
            Err(self.remote_error(reply))
        } else {
            Ok(reply)
        }
    }

    /// Emits a signal without waiting for a reply.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidName`] if the path or interface
    /// is malformed, or [`DbusError::Io`] on transport errors.
    pub fn emit_signal<F>(
        &mut self,
        path: &str,
        interface: &str,
        member: &str,
        body: F,
    ) -> DbusResult<()>
    where
        F: FnOnce(&mut BodyWriter) -> DbusResult<()>,
    {
        let mut message = DbusMessage::signal(path, interface, member)?;
        let serial = self.next_serial()?;
        message.set_serial(serial)?;
        message.build_body(body)?;
        self.send(&mut message)
    }

    /// Calls `AddMatch` on the bus.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Remote`] if the bus reports an error,
    /// [`DbusError::Timeout`] on deadline expiry, or
    /// [`DbusError::InvalidName`] for a malformed rule.
    pub fn add_match(
        &mut self,
        rule: &str,
        timeout: core::time::Duration,
    ) -> DbusResult<()> {
        let mut message = DbusMessage::method_call(
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
            "AddMatch",
        )?;
        message.build_body(|body| body.write_str(rule))?;
        let serial = self.next_serial()?;
        message.set_serial(serial)?;
        self.send(&mut message)?;
        let reply = self.wait_reply(serial, timeout)?;
        if reply.kind() == MessageKind::Error {
            Err(self.remote_error(reply))
        } else {
            Ok(())
        }
    }

    /// Removes a previously installed match rule.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Remote`], [`DbusError::Timeout`], or
    /// [`DbusError::InvalidName`].
    pub fn remove_match(
        &mut self,
        rule: &str,
        timeout: core::time::Duration,
    ) -> DbusResult<()> {
        let mut message = DbusMessage::method_call(
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
            "RemoveMatch",
        )?;
        message.build_body(|body| body.write_str(rule))?;
        let serial = self.next_serial()?;
        message.set_serial(serial)?;
        self.send(&mut message)?;
        let reply = self.wait_reply(serial, timeout)?;
        if reply.kind() == MessageKind::Error {
            Err(self.remote_error(reply))
        } else {
            Ok(())
        }
    }

    /// Returns the next queued signal, if any, without blocking.
    ///
    /// This method drains the transport once and then returns any
    /// signal that arrived while draining.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Disconnected`] when the peer closed the
    /// connection.
    pub fn try_recv(&mut self) -> DbusResult<Option<DbusMessage>> {
        self.drain_available()?;
        Ok(self.incoming.pop_front())
    }

    /// Receives the next signal or method return, blocking until
    /// `timeout` elapses.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Timeout`] when no message arrives in
    /// time, or [`DbusError::Disconnected`] on peer close.
    pub fn recv_timeout(
        &mut self,
        timeout: core::time::Duration,
    ) -> DbusResult<DbusMessage> {
        let deadline = self
            .transport
            .now_ms()
            .saturating_add(timeout.as_millis() as u64);
        loop {
            if let Some(message) = self.incoming.pop_front() {
                return Ok(message);
            }
            let remaining = deadline.saturating_sub(self.transport.now_ms());
            if remaining == 0 {
                return Err(DbusError::Timeout);
            }
            self.wait_read(Some(core::time::Duration::from_millis(remaining)))?;
            self.drain_available()?;
        }
    }

    /// Returns a flush of pending outbound bytes, or `Ok(())`
    /// when the outbound buffer is empty.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Disconnected`] when the peer closed the
    /// connection while flushing.
    pub fn flush(&mut self) -> DbusResult<()> {
        self.flush_writable()
    }

    fn send(&mut self, message: &mut DbusMessage) -> DbusResult<()> {
        let serial = self.next_serial()?;
        message.set_serial(serial)?;
        let bytes = message.encode()?;
        self.tx.extend_from_slice(&bytes);
        self.needs_write = true;
        Ok(())
    }

    fn flush_writable(&mut self) -> DbusResult<()> {
        while self.tx_pos < self.tx.len() {
            match self.transport.write(&self.tx[self.tx_pos..]) {
                Ok(0) => return Err(DbusError::Disconnected),
                Ok(n) => self.tx_pos += n,
                Err(DbusError::WouldBlock) => return Ok(()),
                Err(e) => return Err(e),
            }
            if self.tx_pos == self.tx.len() {
                self.tx.clear();
                self.tx_pos = 0;
                self.needs_write = false;
            }
        }
        self.tx.clear();
        self.tx_pos = 0;
        self.needs_write = false;
        Ok(())
    }

    fn wait_read(&mut self, timeout: Option<core::time::Duration>) -> DbusResult<()> {
        self.wait_interest(timeout, DbusPollEvents::READABLE)
    }

    fn wait_write(&mut self, timeout: Option<core::time::Duration>) -> DbusResult<()> {
        self.wait_interest(timeout, DbusPollEvents::WRITABLE)
    }

    fn wait_interest(
        &mut self,
        timeout: Option<core::time::Duration>,
        interest: DbusPollEvents,
    ) -> DbusResult<()> {
        let _ = self.transport.wait(timeout, interest)?;
        Ok(())
    }

    fn drain_available(&mut self) -> DbusResult<()> {
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > 64 {
                break;
            }
            let mut buf = [0u8; 4096];
            match self.transport.read(&mut buf) {
                Ok(0) => return Err(DbusError::Disconnected),
                Ok(n) => self.stream.feed(&buf[..n]),
                Err(DbusError::WouldBlock) => break,
                Err(e) => return Err(e),
            }
            while let Some(message) = self.stream.next_message()? {
                self.dispatch(message);
            }
        }
        Ok(())
    }

    fn dispatch(&mut self, message: DbusMessage) {
        match message.kind() {
            MessageKind::MethodReturn | MessageKind::Error => {
                if let Some(serial) = message.reply_serial() {
                    self.pending_replies.push((serial, message));
                } else {
                    self.incoming.push_back(message);
                }
            }
            _ => {
                self.incoming.push_back(message);
            }
        }
    }

    fn wait_reply(
        &mut self,
        serial: u32,
        timeout: core::time::Duration,
    ) -> DbusResult<DbusMessage> {
        let deadline = self
            .transport
            .now_ms()
            .saturating_add(timeout.as_millis() as u64);
        loop {
            if let Some(pos) = self.pending_replies.iter().position(|(s, _)| *s == serial)
            {
                let (_, message) = self.pending_replies.remove(pos);
                return if message.kind() == MessageKind::Error {
                    Err(self.remote_error(message))
                } else {
                    Ok(message)
                };
            }
            let remaining = deadline.saturating_sub(self.transport.now_ms());
            if remaining == 0 {
                return Err(DbusError::Timeout);
            }
            let interest = if self.needs_write {
                DbusPollEvents::READABLE | DbusPollEvents::WRITABLE
            } else {
                DbusPollEvents::READABLE
            };
            let events = self
                .transport
                .wait(Some(core::time::Duration::from_millis(remaining)), interest)?;
            if events.contains(DbusPollEvents::WRITABLE) {
                self.flush_writable()?;
            }
            if events.contains(DbusPollEvents::READABLE)
                || events.contains(DbusPollEvents::HANGUP)
            {
                self.drain_available()?;
            }
        }
    }

    fn next_serial(&mut self) -> DbusResult<u32> {
        let serial = self.next_serial;
        self.next_serial = self.next_serial.wrapping_add(1);
        if self.next_serial == 0 {
            self.next_serial = 1;
        }
        Ok(serial)
    }

    fn remote_error(&self, message: DbusMessage) -> DbusError {
        let name = message.error_name().unwrap_or_default();
        let message_text = {
            let mut reader = message.body_reader();
            reader.read_str().unwrap_or_default()
        };
        DbusError::remote(name, message_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dbus_message::DbusMessage;
    use core::time::Duration;

    struct MockTransport {
        rx: VecDeque<Vec<u8>>,
        tx: Vec<Vec<u8>>,
        readable: bool,
        writable: bool,
    }

    impl MockTransport {
        fn new() -> Self {
            Self {
                rx: VecDeque::new(),
                tx: Vec::new(),
                readable: true,
                writable: true,
            }
        }
    }

    impl DbusTransport for MockTransport {
        fn read(&mut self, buf: &mut [u8]) -> DbusResult<usize> {
            if let Some(data) = self.rx.front() {
                let n = data.len().min(buf.len());
                buf[..n].copy_from_slice(&data[..n]);
                if n == data.len() {
                    self.rx.pop_front();
                } else {
                    let remaining = data[n..].to_vec();
                    self.rx.pop_front();
                    self.rx.push_back(remaining);
                }
                Ok(n)
            } else {
                Err(DbusError::WouldBlock)
            }
        }
        fn write(&mut self, buf: &[u8]) -> DbusResult<usize> {
            if self.writable {
                let n = buf.len();
                self.tx.push(buf.to_vec());
                Ok(n)
            } else {
                Err(DbusError::WouldBlock)
            }
        }
        fn wait(
            &mut self,
            _timeout: Option<Duration>,
            interest: DbusPollEvents,
        ) -> DbusResult<DbusPollEvents> {
            let mut out = DbusPollEvents::EMPTY;
            if interest.contains(DbusPollEvents::READABLE) && self.readable {
                out |= DbusPollEvents::READABLE;
            }
            if interest.contains(DbusPollEvents::WRITABLE) && self.writable {
                out |= DbusPollEvents::WRITABLE;
            }
            Ok(out)
        }
        fn now_ms(&self) -> u64 {
            0
        }
    }

    #[test]
    fn hello_round_trip() {
        let mut transport = MockTransport::new();
        transport.rx.push_back(b"OK deadbeef\r\n".to_vec());
        let mut conn = Connection::new(transport);
        conn.authenticate(0).unwrap();
        assert!(conn.unique_name().is_none());
        let mut hello_reply = DbusMessage::method_return(1);
        hello_reply.set_serial(1).unwrap();
        hello_reply.build_body(|body| body.write_str(":1")).unwrap();
        conn.transport.rx.push_back(hello_reply.encode().unwrap());
        conn.hello().unwrap();
        assert_eq!(conn.unique_name(), Some(":1"));
    }
}
