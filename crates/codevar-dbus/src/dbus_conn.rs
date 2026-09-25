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
use crate::dbus_transport::{DbusPollEvents, DbusTransport, close_fds};

/// Default timeout for a method call, matching the D-Bus
/// specification default of 25 seconds.
pub const DEFAULT_CALL_TIMEOUT: core::time::Duration = core::time::Duration::from_millis(25_000);

/// `RequestName` flag: let a later `REPLACE_EXISTING` requester take
/// the name away from this request.
pub const NAME_FLAG_ALLOW_REPLACEMENT: u32 = 0x1;
/// `RequestName` flag: take the name from its current owner when the
/// owner allowed replacement, instead of queueing.
pub const NAME_FLAG_REPLACE_EXISTING: u32 = 0x2;
/// `RequestName` flag: fail with [`NAME_REPLY_EXISTS`] instead of
/// queueing when the name already has an owner.
pub const NAME_FLAG_DO_NOT_QUEUE: u32 = 0x4;

/// `RequestName` reply: the caller became the primary owner.
pub const NAME_REPLY_PRIMARY_OWNER: u32 = 1;
/// `RequestName` reply: the caller is queued behind the current owner.
pub const NAME_REPLY_IN_QUEUE: u32 = 2;
/// `RequestName` reply: the name exists and the flags forbade
/// queueing.
pub const NAME_REPLY_EXISTS: u32 = 3;
/// `RequestName` reply: the caller already owned the name.
pub const NAME_REPLY_ALREADY_OWNER: u32 = 4;

/// One encoded outbound message plus the descriptors attached to the
/// first byte of `bytes`.
///
/// The descriptors are released when the segment is dropped unless
/// they were handed to the kernel (or moved out) earlier, so an
/// abandoned transmit queue cannot leak them.
struct TxSegment {
    bytes: Vec<u8>,
    pos: usize,
    fds: Vec<i32>,
}

impl TxSegment {
    /// Returns the unwritten tail of the segment.
    fn remaining(&self) -> &[u8] {
        &self.bytes[self
            .pos
            .min(
                self.bytes
                    .len(),
            )..]
    }
}

impl Drop for TxSegment {
    fn drop(&mut self) {
        close_fds(core::mem::take(&mut self.fds));
    }
}

/// A synchronous D-Bus connection.
///
/// Outbound messages are queued as per-message segments so that
/// descriptors passed with a message attach to exactly the byte
/// position where that message starts in the stream.
pub struct Connection<T: DbusTransport> {
    transport: T,
    stream: DbusMessageStream,
    tx: VecDeque<TxSegment>,
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
            tx: VecDeque::new(),
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
                        match self
                            .transport
                            .write(cursor)
                        {
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
                    match self
                        .transport
                        .read(&mut buf)
                    {
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
        self.unique_name
            .as_deref()
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
    pub fn emit_signal<F>(&mut self, path: &str, interface: &str, member: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut BodyWriter) -> DbusResult<()>,
    {
        let mut message = DbusMessage::signal(path, interface, member)?;
        let serial = self.next_serial()?;
        message.set_serial(serial)?;
        message.build_body(body)?;
        self.send(&mut message)
    }

    /// Sends a pre-built message and queues it for flushing.
    ///
    /// Descriptors attached to `message` are taken over by the
    /// connection; they are closed automatically when they reach the
    /// kernel or when the connection is dropped before that.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidState`] when the serial
    /// counter overflows or [`DbusError::Io`] on transport
    /// errors. When encoding fails the message is dropped and any
    /// descriptors it still holds are closed.
    pub fn send_message(&mut self, mut message: DbusMessage) -> DbusResult<()> {
        self.send(&mut message)
    }

    /// Calls `RequestName` on the bus to acquire ownership
    /// of `name`.
    ///
    /// `flags` is a bitwise combination of [`NAME_FLAG_ALLOW_REPLACEMENT],
    /// [`NAME_FLAG_REPLACE_EXISTING`] and [`NAME_FLAG_DO_NOT_QUEUE`];
    /// `0` queues behind the current owner.
    ///
    /// Returns the reply code from the `org.freedesktop.DBus.RequestName`
    /// reply: [`NAME_REPLY_PRIMARY_OWNER`], [`NAME_REPLY_IN_QUEUE`],
    /// [`NAME_REPLY_EXISTS`] or [`NAME_REPLY_ALREADY_OWNER`].
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Remote`] if the bus reports an
    /// `org.freedesktop.DBus.Error.*` failure, [`DbusError::Timeout`]
    /// on deadline expiry, or [`DbusError::InvalidName`] for a
    /// malformed name.
    pub fn request_name(&mut self, name: &str, flags: u32, timeout: core::time::Duration) -> DbusResult<u32> {
        let mut message = DbusMessage::method_call(
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
            "RequestName",
        )?;
        message.build_body(|body| {
            body.write_str(name)?;
            body.write_u32(flags)
        })?;
        let serial = self.next_serial()?;
        message.set_serial(serial)?;
        self.send(&mut message)?;
        let reply = self.wait_reply(serial, timeout)?;
        let mut reader = reply.body_reader();
        reader.read_u32()
    }

    /// Calls `AddMatch` on the bus.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Remote`] if the bus reports an error,
    /// [`DbusError::Timeout`] on deadline expiry, or
    /// [`DbusError::InvalidName`] for a malformed rule.
    pub fn add_match(&mut self, rule: &str, timeout: core::time::Duration) -> DbusResult<()> {
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
    pub fn remove_match(&mut self, rule: &str, timeout: core::time::Duration) -> DbusResult<()> {
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
        Ok(self
            .incoming
            .pop_front())
    }

    /// Receives the next signal or method return, blocking until
    /// `timeout` elapses.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Timeout`] when no message arrives in
    /// time, or [`DbusError::Disconnected`] on peer close.
    pub fn recv_timeout(&mut self, timeout: core::time::Duration) -> DbusResult<DbusMessage> {
        let deadline = self
            .transport
            .now_ms()
            .saturating_add(timeout.as_millis() as u64);
        loop {
            if let Some(message) = self
                .incoming
                .pop_front()
            {
                return Ok(message);
            }
            let remaining = deadline.saturating_sub(
                self.transport
                    .now_ms(),
            );
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
        // Callers that need the serial before waiting for a reply
        // assign it themselves (`hello`, `call`, ...); only assign one
        // here when the message does not carry a serial yet, so the
        // value on the wire always matches the one being awaited.
        if message.serial() == 0 {
            let serial = self.next_serial()?;
            message.set_serial(serial)?;
        }
        let bytes = message.encode()?;
        self.tx
            .push_back(TxSegment {
                bytes,
                pos: 0,
                fds: message.take_fds(),
            });
        self.needs_write = true;
        Ok(())
    }

    /// Writes queued segments in order.
    ///
    /// Each segment writes through [`DbusTransport::write_with_fds`]
    /// while it still carries descriptors and through
    /// [`DbusTransport::write`] afterward. Descriptors attach to the
    /// first byte of a segment, so on a partial write only the first
    /// chunk carries them and the remainder is written plain. After
    /// the kernel accepted the first chunk (`Ok(n)` with `n > 0`)
    /// our copies are released, because the kernel keeps its own
    /// references from then on.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::Disconnected`] when the peer closed the
    /// connection while flushing. On any hard error the remaining
    /// segments are dropped (closing descriptors that were never
    /// sent) so nothing leaks.
    fn flush_writable(&mut self) -> DbusResult<()> {
        while let Some(front) = self
            .tx
            .front()
        {
            if front.pos
                >= front
                    .bytes
                    .len()
            {
                self.tx
                    .pop_front();
                continue;
            }
            let had_fds = !front
                .fds
                .is_empty();
            let outcome = if had_fds {
                self.transport
                    .write_with_fds(front.remaining(), &front.fds)
            } else {
                self.transport
                    .write(front.remaining())
            };
            match outcome {
                Ok(0) => {
                    self.drop_tx_segments();
                    return Err(DbusError::Disconnected);
                }
                Ok(written) => {
                    if let Some(segment) = self
                        .tx
                        .front_mut()
                    {
                        segment.pos = segment
                            .pos
                            .saturating_add(written);
                        if had_fds {
                            // The kernel took its own references with
                            // the first bytes of the segment; the
                            // remainder travels without descriptors.
                            close_fds(core::mem::take(&mut segment.fds));
                        }
                    }
                }
                Err(DbusError::WouldBlock) => return Ok(()),
                Err(error) => {
                    self.drop_tx_segments();
                    return Err(error);
                }
            }
        }
        self.needs_write = false;
        Ok(())
    }

    /// Drops every queued segment, closing descriptors that were
    /// never handed to the kernel.
    fn drop_tx_segments(&mut self) {
        self.tx
            .clear();
        self.needs_write = false;
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
        let _ = self
            .transport
            .wait(timeout, interest)?;
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
            match self
                .transport
                .read(&mut buf)
            {
                Ok(0) => return Err(DbusError::Disconnected),
                Ok(n) => {
                    // Queue the descriptors the read reported before
                    // feeding its bytes, so descriptor order always
                    // matches message order in the stream.
                    let fds = self
                        .transport
                        .take_fds();
                    self.stream
                        .feed_with_fds(&buf[..n], fds);
                }
                Err(DbusError::WouldBlock) => break,
                Err(e) => return Err(e),
            }
            while let Some(message) = self
                .stream
                .next_message()?
            {
                self.dispatch(message);
            }
        }
        Ok(())
    }

    fn dispatch(&mut self, message: DbusMessage) {
        match message.kind() {
            MessageKind::MethodReturn | MessageKind::Error => {
                if let Some(serial) = message.reply_serial() {
                    self.pending_replies
                        .push((serial, message));
                } else {
                    self.incoming
                        .push_back(message);
                }
            }
            _ => {
                self.incoming
                    .push_back(message);
            }
        }
    }

    fn wait_reply(&mut self, serial: u32, timeout: core::time::Duration) -> DbusResult<DbusMessage> {
        let deadline = self
            .transport
            .now_ms()
            .saturating_add(timeout.as_millis() as u64);
        loop {
            if let Some(pos) = self
                .pending_replies
                .iter()
                .position(|(s, _)| *s == serial)
            {
                let (_, message) = self
                    .pending_replies
                    .remove(pos);
                return if message.kind() == MessageKind::Error {
                    Err(self.remote_error(message))
                } else {
                    Ok(message)
                };
            }
            let remaining = deadline.saturating_sub(
                self.transport
                    .now_ms(),
            );
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
            if events.contains(DbusPollEvents::READABLE) || events.contains(DbusPollEvents::HANGUP) {
                self.drain_available()?;
            }
        }
    }

    fn next_serial(&mut self) -> DbusResult<u32> {
        let serial = self.next_serial;
        self.next_serial = self
            .next_serial
            .wrapping_add(1);
        if self.next_serial == 0 {
            self.next_serial = 1;
        }
        Ok(serial)
    }

    fn remote_error(&self, message: DbusMessage) -> DbusError {
        let name = message
            .error_name()
            .unwrap_or_default();
        let message_text = {
            let mut reader = message.body_reader();
            reader
                .read_str()
                .unwrap_or_default()
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
        rx_fds: Vec<i32>,
        tx: Vec<(Vec<u8>, Vec<i32>)>,
        readable: bool,
        writable: bool,
    }

    impl MockTransport {
        fn new() -> Self {
            Self {
                rx: VecDeque::new(),
                rx_fds: Vec::new(),
                tx: Vec::new(),
                readable: true,
                writable: true,
            }
        }
    }

    impl DbusTransport for MockTransport {
        fn read(&mut self, buf: &mut [u8]) -> DbusResult<usize> {
            if let Some(data) = self
                .rx
                .front()
            {
                let n = data
                    .len()
                    .min(buf.len());
                buf[..n].copy_from_slice(&data[..n]);
                if n == data.len() {
                    self.rx
                        .pop_front();
                } else {
                    let remaining = data[n..].to_vec();
                    self.rx
                        .pop_front();
                    self.rx
                        .push_back(remaining);
                }
                Ok(n)
            } else {
                Err(DbusError::WouldBlock)
            }
        }
        fn write(&mut self, buf: &[u8]) -> DbusResult<usize> {
            if self.writable {
                let n = buf.len();
                self.tx
                    .push((buf.to_vec(), Vec::new()));
                Ok(n)
            } else {
                Err(DbusError::WouldBlock)
            }
        }
        fn write_with_fds(&mut self, buf: &[u8], fds: &[i32]) -> DbusResult<usize> {
            if self.writable {
                self.tx
                    .push((buf.to_vec(), fds.to_vec()));
                Ok(buf.len())
            } else {
                Err(DbusError::WouldBlock)
            }
        }
        fn take_fds(&mut self) -> Vec<i32> {
            core::mem::take(&mut self.rx_fds)
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
        transport
            .rx
            .push_back(b"OK deadbeef\r\n".to_vec());
        let mut conn = Connection::new(transport);
        conn.authenticate(0)
            .unwrap();
        assert!(
            conn.unique_name()
                .is_none()
        );
        let mut hello_reply = DbusMessage::method_return(1);
        hello_reply
            .set_serial(1)
            .unwrap();
        hello_reply
            .build_body(|body| body.write_str(":1"))
            .unwrap();
        conn.transport
            .rx
            .push_back(
                hello_reply
                    .encode()
                    .unwrap(),
            );
        conn.hello()
            .unwrap();
        assert_eq!(conn.unique_name(), Some(":1"));
    }

    #[test]
    fn request_name_sends_flags_and_returns_reply_code() {
        let mut transport = MockTransport::new();
        let mut reply = DbusMessage::method_return(1);
        reply
            .set_serial(100)
            .unwrap();
        reply
            .build_body(|body| body.write_u32(NAME_REPLY_IN_QUEUE))
            .unwrap();
        transport
            .rx
            .push_back(
                reply
                    .encode()
                    .unwrap(),
            );
        let mut conn = Connection::new(transport);

        let code = conn
            .request_name(
                "org.example.Service",
                NAME_FLAG_ALLOW_REPLACEMENT | NAME_FLAG_DO_NOT_QUEUE,
                Duration::from_secs(5),
            )
            // Justified: the canned bus reply is well formed.
            .unwrap();
        assert_eq!(code, NAME_REPLY_IN_QUEUE);

        // The outgoing message must carry the requested flags.
        let (bytes, fds) = conn
            .transport
            .tx
            .first()
            .unwrap();
        assert!(fds.is_empty());
        let sent = DbusMessage::decode(bytes).unwrap();
        assert_eq!(sent.member(), Some("RequestName"));
        let mut reader = sent.body_reader();
        assert_eq!(
            reader
                .read_str()
                .unwrap(),
            "org.example.Service"
        );
        assert_eq!(
            reader
                .read_u32()
                .unwrap(),
            NAME_FLAG_ALLOW_REPLACEMENT | NAME_FLAG_DO_NOT_QUEUE
        );
    }

    #[test]
    fn send_message_preserves_preassigned_serial() {
        let mut conn = Connection::new(MockTransport::new());
        let mut message = DbusMessage::method_return(7);
        message
            .set_serial(42)
            .unwrap();
        // Justified: fixed inputs cannot fail.
        conn.send_message(message)
            .unwrap();
        conn.flush()
            .unwrap();

        // The serial on the wire must match the one assigned by the
        // caller, otherwise `wait_reply` can never match the reply of
        // a real bus (which echoes the wire serial).
        let (bytes, _) = conn
            .transport
            .tx
            .first()
            .unwrap();
        let sent = DbusMessage::decode(bytes).unwrap();
        assert_eq!(sent.serial(), 42);
        assert_eq!(sent.reply_serial(), Some(7));
    }

    #[test]
    fn send_message_assigns_serial_when_unset() {
        let mut conn = Connection::new(MockTransport::new());
        let message = DbusMessage::method_return(7);
        // Justified: fixed inputs cannot fail.
        conn.send_message(message)
            .unwrap();
        conn.flush()
            .unwrap();

        let (bytes, _) = conn
            .transport
            .tx
            .first()
            .unwrap();
        let sent = DbusMessage::decode(bytes).unwrap();
        assert_ne!(sent.serial(), 0);
    }

    #[test]
    fn request_name_maps_bus_errors_to_remote_errors() {
        let mut transport = MockTransport::new();
        let mut error = DbusMessage::error(1, "org.freedesktop.DBus.Error.NameHasNoOwner").unwrap();
        error
            .set_serial(100)
            .unwrap();
        error
            .build_body(|body| body.write_str("no such name"))
            .unwrap();
        transport
            .rx
            .push_back(
                error
                    .encode()
                    .unwrap(),
            );
        let mut conn = Connection::new(transport);

        let Err(DbusError::Remote { name, .. }) =
            conn.request_name("a.b", NAME_FLAG_DO_NOT_QUEUE, Duration::from_secs(5))
        else {
            // The canned reply is an error, so the result must be too.
            panic!("request_name must surface bus errors as DbusError::Remote");
        };
        assert_eq!(name, "org.freedesktop.DBus.Error.NameHasNoOwner");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn flush_attaches_fds_to_the_right_segment() {
        // Justified: assertions describe the invariants of the test
        // setup itself, which cannot legitimately fail.
        let mut pipe = [-1i32; 2];
        // SAFETY: `pipe` is a valid two element array; on success the
        // kernel fills both entries with open descriptors.
        assert_eq!(unsafe { libc::pipe(pipe.as_mut_ptr()) }, 0);
        let fd = pipe[0];
        let mut conn = Connection::new(MockTransport::new());

        let first = DbusMessage::signal("/a", "b.C", "One").unwrap();
        conn.send_message(first)
            .unwrap();

        let mut second = DbusMessage::method_call("a.b", "/a", "b.C", "Two").unwrap();
        second
            .build_body(|body| body.write_fd(0))
            .unwrap();
        second.set_fds(vec![fd]);
        conn.send_message(second)
            .unwrap();

        conn.flush()
            .unwrap();

        let sent = &conn
            .transport
            .tx;
        assert_eq!(sent.len(), 2);
        // First message travels plain, the second one carries the fd.
        assert!(
            sent[0]
                .1
                .is_empty()
        );
        assert_eq!(
            DbusMessage::decode(&sent[0].0)
                .unwrap()
                .member(),
            Some("One")
        );
        assert_eq!(sent[1].1, vec![fd]);
        let decoded = DbusMessage::decode(&sent[1].0).unwrap();
        assert_eq!(decoded.member(), Some("Two"));
        assert_eq!(decoded.unix_fds(), 1);
        // The flush released our copy once the kernel took it.
        // SAFETY: `fcntl` only inspects the descriptor.
        assert_eq!(unsafe { libc::fcntl(fd, libc::F_GETFD) }, -1);
        // SAFETY: closing the write end of the pipe we still own.
        unsafe { libc::close(pipe[1]) };
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn try_recv_attaches_fds_received_with_a_message() {
        // Justified: the test setup cannot legitimately fail.
        let mut pipe = [-1i32; 2];
        // SAFETY: `pipe` is a valid two element array; on success the
        // kernel fills both entries with open descriptors.
        assert_eq!(unsafe { libc::pipe(pipe.as_mut_ptr()) }, 0);
        let fd = pipe[0];

        let mut call = DbusMessage::method_call("a.b", "/a", "b.C", "Open").unwrap();
        call.set_serial(1)
            .unwrap();
        call.build_body(|body| body.write_fd(0))
            .unwrap();
        call.set_fds(vec![fd]);
        let bytes = call
            .encode()
            .unwrap();
        // Keep the descriptor alive: it now travels out of band via
        // the transport instead of through the message.
        let attached = call.take_fds();

        let mut transport = MockTransport::new();
        transport
            .rx
            .push_back(bytes);
        transport.rx_fds = attached;
        let mut conn = Connection::new(transport);

        // Justified: the queued message is complete and well formed.
        let message = conn
            .try_recv()
            .unwrap()
            .unwrap();
        assert_eq!(message.member(), Some("Open"));
        assert_eq!(message.fds(), &[fd]);
        assert_eq!(
            message
                .body_reader()
                .read_fd()
                .unwrap(),
            0
        );
        assert_eq!(
            conn.try_recv()
                .unwrap(),
            None
        );

        // Dropping the message closes the descriptor it owns.
        drop(message);
        // SAFETY: `fcntl` only inspects the descriptor.
        assert_eq!(unsafe { libc::fcntl(fd, libc::F_GETFD) }, -1);
        // SAFETY: closing the write end of the pipe we still own.
        unsafe { libc::close(pipe[1]) };
    }
}
