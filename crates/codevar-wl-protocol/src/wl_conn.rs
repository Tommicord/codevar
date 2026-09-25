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

//! Byte transport and wire format codec shared by both sides.
//!
//! The module replaces `connection.c`: a [`WlTransport`] carries raw bytes
//! and file descriptors, a [`WlConnection`] buffers them and a [`WlClosure`]
//! encodes or decodes a single protocol message.

use core::time::Duration;

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::wl_error::{WlError, WlResult};
use crate::wl_handle::{
    MAX_MESSAGE_SIZE, WlArgType, WlArgument, WlArray, WlFd, WlFixed, WlMap, WlMessage, WlObject,
    WlPollEvents, arg_count, get_next_argument,
};

/// Opaque handle used to poll a transport from an event loop.
pub type WlHandle = u64;

/// Byte level transport of a Wayland connection.
///
/// Implementations wrap the platform socket or an in-memory channel. The
/// connection buffers written data, so [`WlTransport::send`] may write fewer
/// bytes than requested; implementations must either accept all file
/// descriptors passed alongside a successful write or none of them.
pub trait WlTransport {
    /// Reads pending bytes into `buf` and appends received file descriptors
    /// to `fds`.
    ///
    /// Returns `Ok(0)` when the peer closed the connection,
    /// [`WlError::WouldBlock`] when no data is available, and the number of
    /// bytes read otherwise.
    fn recv(&mut self, buf: &mut [u8], fds: &mut Vec<WlFd>) -> WlResult<usize>;

    /// Writes `data`, attaching `fds` to the write.
    ///
    /// Returns the number of bytes accepted. All `fds` are consumed when at
    /// least one byte is accepted.
    fn send(&mut self, data: &[u8], fds: &[WlFd]) -> WlResult<usize>;

    /// Waits until one of `mask` events is ready or `timeout` elapses.
    ///
    /// A `timeout` of `None` blocks indefinitely, `Some(Duration::ZERO)`
    /// polls without blocking.
    fn wait(&mut self, timeout: Option<Duration>, mask: WlPollEvents) -> WlResult<WlPollEvents>;

    /// Opaque handle used to poll this transport from an event loop.
    fn handle(&self) -> WlHandle;

    /// Releases a received file descriptor that is being discarded.
    ///
    /// The default implementation does nothing, which is correct for
    /// transports without a descriptor concept.
    fn release_fd(&mut self, _fd: WlFd) {}
}

/// A decoded protocol message.
///
/// The value carries the wire header, the static message signature and the
/// decoded arguments. Use [`WlConnection::demarshal`] to decode incoming
/// messages and [`WlConnection::queue_closure`] to encode outgoing ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WlClosure {
    /// Id of the object the message was sent on.
    pub sender_id: u32,
    /// Opcode of the message within its interface.
    pub opcode: u32,
    /// Static signature of the message.
    pub message: &'static WlMessage,
    /// Decoded arguments in signature order.
    pub args: Vec<WlArgument>,
}

impl WlClosure {
    /// Creates a closure after validating the arguments against `message`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when the argument count, an
    /// argument type or a nullability constraint does not match.
    pub fn new(
        sender_id: u32,
        opcode: u32,
        message: &'static WlMessage,
        args: Vec<WlArgument>,
    ) -> WlResult<Self> {
        validate_args(message, &args)?;
        Ok(Self {
            sender_id,
            opcode,
            message,
            args,
        })
    }

    /// Encodes the closure into wire format, including the message header.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when the arguments do not match
    /// the signature and [`WlError::MessageTooBig`] when the encoded message
    /// exceeds [`MAX_MESSAGE_SIZE`].
    pub fn encode(&self) -> WlResult<Vec<u8>> {
        validate_args(self.message, &self.args)?;
        let size = encoded_size(&self.args);
        if size > MAX_MESSAGE_SIZE {
            return Err(WlError::MessageTooBig(size));
        }
        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(&self.sender_id.to_le_bytes());
        let header = ((size as u32) << 16) | (self.opcode & 0x0000_ffff);
        bytes.extend_from_slice(&header.to_le_bytes());
        for arg in &self.args {
            match arg {
                WlArgument::Int(value) => bytes.extend_from_slice(&value.to_le_bytes()),
                WlArgument::Uint(value) | WlArgument::NewId(value) => {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                WlArgument::Fixed(value) => {
                    bytes.extend_from_slice(&value.to_raw().to_le_bytes());
                }
                WlArgument::Object(id) => bytes.extend_from_slice(&id.to_le_bytes()),
                WlArgument::Str(None) => bytes.extend_from_slice(&0u32.to_le_bytes()),
                WlArgument::Str(Some(value)) => {
                    let length = value.len() + 1;
                    bytes.extend_from_slice(&(length as u32).to_le_bytes());
                    bytes.extend_from_slice(value.as_bytes());
                    bytes.push(0);
                    pad_to_word(&mut bytes);
                }
                WlArgument::Array(None) => bytes.extend_from_slice(&0u32.to_le_bytes()),
                WlArgument::Array(Some(value)) => {
                    let data = value.as_bytes();
                    bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
                    bytes.extend_from_slice(data);
                    pad_to_word(&mut bytes);
                }
                // File descriptors travel out of band and occupy no bytes.
                WlArgument::Fd(_) => {}
            }
        }
        debug_assert_eq!(bytes.len(), size);
        Ok(bytes)
    }

    /// Returns the file descriptors carried by the arguments, in signature
    /// order.
    #[must_use]
    pub fn fd_args(&self) -> Vec<WlFd> {
        self.args
            .iter()
            .filter_map(|arg| match arg {
                WlArgument::Fd(fd) if *fd >= 0 => Some(*fd),
                _ => None,
            })
            .collect()
    }

    /// Marks all file descriptor arguments as taken.
    pub fn clear_fds(&mut self) {
        for arg in &mut self.args {
            if let WlArgument::Fd(fd) = arg {
                *fd = -1;
            }
        }
    }
}

fn validate_args(message: &'static WlMessage, args: &[WlArgument]) -> WlResult<()> {
    let expected = arg_count(message.signature);
    if args.len() != expected {
        return Err(WlError::invalid_argument(format!(
            "{} expects {expected} arguments, got {}",
            message.name,
            args.len()
        )));
    }
    for (arg, msg_arg) in args.iter().zip(message.args()) {
        arg.validate(msg_arg.details)?;
    }
    Ok(())
}

fn encoded_size(args: &[WlArgument]) -> usize {
    let mut size = 8usize;
    for arg in args {
        size += match arg {
            WlArgument::Fd(_) => 0,
            WlArgument::Str(None) | WlArgument::Array(None) => 4,
            WlArgument::Str(Some(value)) => 4 + align_up(value.len() + 1),
            WlArgument::Array(Some(value)) => 4 + align_up(value.len()),
            WlArgument::Int(_)
            | WlArgument::Uint(_)
            | WlArgument::Fixed(_)
            | WlArgument::Object(_)
            | WlArgument::NewId(_) => 4,
        };
    }
    size
}

const fn align_up(length: usize) -> usize {
    (length + 3) & !3
}

fn pad_to_word(bytes: &mut Vec<u8>) {
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
}

fn truncated() -> WlError {
    WlError::invalid_argument("message truncated while decoding")
}

fn read_u32(bytes: &[u8], pos: usize) -> WlResult<u32> {
    let end = pos.saturating_add(4);
    let slice = bytes.get(pos..end).ok_or_else(truncated)?;
    let mut value = [0u8; 4];
    value.copy_from_slice(slice);
    Ok(u32::from_le_bytes(value))
}

fn parse_args(
    bytes: &[u8],
    message: &'static WlMessage,
    fds: &mut VecDeque<WlFd>,
    taken_fds: &mut Vec<WlFd>,
) -> WlResult<Vec<WlArgument>> {
    let mut args = Vec::with_capacity(arg_count(message.signature));
    let mut pos = 8usize;
    let mut sig_pos = 0usize;
    while let Some((details, next)) = get_next_argument(message.signature, sig_pos) {
        sig_pos = next;
        match details.ty {
            WlArgType::Int => {
                args.push(WlArgument::Int(read_u32(bytes, pos)? as i32));
                pos += 4;
            }
            WlArgType::Uint => {
                args.push(WlArgument::Uint(read_u32(bytes, pos)?));
                pos += 4;
            }
            WlArgType::Fixed => {
                let raw = read_u32(bytes, pos)? as i32;
                pos += 4;
                args.push(WlArgument::Fixed(WlFixed::from_raw(raw)));
            }
            WlArgType::Object => {
                let id = read_u32(bytes, pos)?;
                pos += 4;
                if id == 0 && !details.nullable {
                    return Err(WlError::invalid_argument(
                        "null object received for a non-nullable argument",
                    ));
                }
                args.push(WlArgument::Object(id));
            }
            WlArgType::NewId => {
                let id = read_u32(bytes, pos)?;
                pos += 4;
                if id == 0 {
                    return Err(WlError::invalid_argument(
                        "null new id received for a new_id argument",
                    ));
                }
                args.push(WlArgument::NewId(id));
            }
            WlArgType::String => {
                let length = read_u32(bytes, pos)? as usize;
                pos += 4;
                if length == 0 {
                    if !details.nullable {
                        return Err(WlError::invalid_argument(
                            "null string received for a non-nullable argument",
                        ));
                    }
                    args.push(WlArgument::Str(None));
                    continue;
                }
                let words = length.div_ceil(4);
                let end = pos.saturating_add(words * 4);
                let data = bytes.get(pos..end).ok_or_else(truncated)?;
                let payload = data.get(..length).ok_or_else(truncated)?;
                if payload.last() != Some(&0) {
                    return Err(WlError::invalid_argument("string is not nul terminated"));
                }
                let text = &payload[..length - 1];
                if text.contains(&0) {
                    return Err(WlError::invalid_argument("string contains an embedded nul"));
                }
                let value = String::from_utf8(text.to_vec())
                    .map_err(|_| WlError::invalid_argument("string is not valid utf-8"))?;
                pos = end;
                args.push(WlArgument::Str(Some(value)));
            }
            WlArgType::Array => {
                let length = read_u32(bytes, pos)? as usize;
                pos += 4;
                let words = length.div_ceil(4);
                let end = pos.saturating_add(words * 4);
                let data = bytes.get(pos..end).ok_or_else(truncated)?;
                let payload = data.get(..length).ok_or_else(truncated)?;
                pos = end;
                args.push(WlArgument::Array(Some(WlArray::from_bytes(payload))));
            }
            WlArgType::Fd => {
                let fd = fds
                    .pop_front()
                    .ok_or_else(|| WlError::invalid_argument("file descriptor expected"))?;
                taken_fds.push(fd);
                args.push(WlArgument::Fd(fd));
            }
        }
    }
    Ok(args)
}

/// Reserves every `new_id` argument of `closure` in `map`.
///
/// # Errors
///
/// Returns [`WlError::InvalidArgument`] when the id belongs to the local id
/// space or is already in use, and [`WlError::TooManyObjects`] when the id
/// space is exhausted.
pub fn reserve_new_ids<T>(closure: &WlClosure, map: &mut WlMap<T>) -> WlResult<()> {
    for (arg, details) in closure.args.iter().zip(closure.message.args()) {
        if details.details.ty == WlArgType::NewId
            && let WlArgument::NewId(id) = arg
        {
            map.reserve_new(*id)?;
        }
    }
    Ok(())
}

/// Validates the `object` arguments of `closure` against `map`.
///
/// Arguments referring to an object the local side already destroyed are
/// rewritten to the null object, matching `wl_closure_lookup_objects`.
///
/// # Errors
///
/// Returns [`WlError::InvalidObject`] for unknown ids and
/// [`WlError::InvalidArgument`] when the object type does not match the
/// signature.
pub fn lookup_objects<T: WlObject>(closure: &mut WlClosure, map: &WlMap<T>) -> WlResult<()> {
    for (arg, details) in closure
        .args
        .iter_mut()
        .zip(closure.message.args())
    {
        if details.details.ty != WlArgType::Object {
            continue;
        }
        let id = match arg {
            WlArgument::Object(id) => *id,
            _ => continue,
        };
        if id == 0 {
            continue;
        }
        if map.is_zombie(id) {
            *arg = WlArgument::Object(0);
            continue;
        }
        let object = map.lookup(id).ok_or(WlError::InvalidObject(id))?;
        if let Some(expected) = details.interface
            && !expected.equal(object.interface())
        {
            return Err(WlError::invalid_argument(format!(
                "object {id} has interface {}, expected {}",
                object.interface().name,
                expected.name
            )));
        }
    }
    Ok(())
}

/// Buffered byte connection on top of a [`WlTransport`].
#[derive(Debug)]
pub struct WlConnection<T: WlTransport> {
    transport: T,
    input: Vec<u8>,
    input_fds: VecDeque<WlFd>,
    output: Vec<u8>,
    output_fds: Vec<WlFd>,
    disconnected: bool,
}

impl<T: WlTransport> WlConnection<T> {
    /// Wraps a transport in an empty connection.
    #[must_use]
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            input: Vec::new(),
            input_fds: VecDeque::new(),
            output: Vec::new(),
            output_fds: Vec::new(),
            disconnected: false,
        }
    }

    /// Returns the transport.
    #[inline]
    #[must_use]
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Mutably returns the transport.
    #[inline]
    #[must_use]
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Unwraps the connection, returning the transport.
    #[must_use]
    pub fn into_transport(self) -> T {
        self.transport
    }

    /// Returns the poll handle of the transport.
    #[inline]
    #[must_use]
    pub fn handle(&self) -> WlHandle {
        self.transport.handle()
    }

    /// Returns `true` once the peer closed the connection.
    #[inline]
    #[must_use]
    pub fn is_disconnected(&self) -> bool {
        self.disconnected
    }

    /// Reads from the transport into the input buffer.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::WouldBlock`] when no data is available and
    /// [`WlError::Disconnected`] when the peer closed the connection.
    pub fn read(&mut self) -> WlResult<usize> {
        if self.disconnected {
            return Err(WlError::Disconnected);
        }
        let mut buf = [0u8; MAX_MESSAGE_SIZE];
        let mut fds = Vec::new();
        let read = self.transport.recv(&mut buf, &mut fds)?;
        if read == 0 {
            self.disconnected = true;
            return Err(WlError::Disconnected);
        }
        self.input.extend_from_slice(&buf[..read]);
        self.input_fds.extend(fds);
        Ok(read)
    }

    /// Returns the number of buffered input bytes.
    #[inline]
    #[must_use]
    pub fn pending_input(&self) -> usize {
        self.input.len()
    }

    /// Returns the number of buffered output bytes.
    #[inline]
    #[must_use]
    pub fn pending_output(&self) -> usize {
        self.output.len()
    }

    /// Returns `true` when buffered output waits to be flushed.
    #[inline]
    #[must_use]
    pub fn wants_write(&self) -> bool {
        !self.output.is_empty()
    }

    /// Peeks at the header of the next message.
    ///
    /// Returns `(sender id, opcode, message size in bytes)` when at least one
    /// full header is buffered.
    #[must_use]
    pub fn peek(&self) -> Option<(u32, u32, u32)> {
        if self.input.len() < 8 {
            return None;
        }
        let sender = u32::from_le_bytes(self.input[0..4].try_into().ok()?);
        let header = u32::from_le_bytes(self.input[4..8].try_into().ok()?);
        Some((sender, header & 0x0000_ffff, header >> 16))
    }

    /// Drops `size` buffered input bytes without decoding them.
    pub fn consume(&mut self, size: usize) {
        let size = size.min(self.input.len());
        self.input.drain(..size);
    }

    /// Drops a message and releases the next `fd_count` file descriptors.
    ///
    /// Used to discard events of objects that no longer exist.
    pub fn skip_message(&mut self, size: usize, fd_count: usize) {
        self.consume(size);
        for _ in 0..fd_count {
            if let Some(fd) = self.input_fds.pop_front() {
                self.transport.release_fd(fd);
            }
        }
    }

    /// Releases all file descriptors held by the connection.
    pub fn release_fds(&mut self) {
        while let Some(fd) = self.input_fds.pop_front() {
            self.transport.release_fd(fd);
        }
        for fd in self.output_fds.drain(..) {
            self.transport.release_fd(fd);
        }
    }

    /// Decodes the next message using `message` as its signature.
    ///
    /// The buffered bytes and the consumed file descriptors are removed even
    /// when decoding fails. A message larger than [`MAX_MESSAGE_SIZE`] is
    /// reported without consuming data so the caller can drop the connection.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::WouldBlock`] when the message is not complete and
    /// [`WlError::InvalidArgument`] when the payload does not match the
    /// signature.
    pub fn demarshal(&mut self, message: &'static WlMessage) -> WlResult<WlClosure> {
        let (sender_id, opcode, size) = self
            .peek()
            .ok_or_else(|| WlError::invalid_argument("no message header buffered"))?;
        let size = size as usize;
        if size < 8 {
            self.consume(size);
            return Err(WlError::invalid_argument("message shorter than its header"));
        }
        if size > MAX_MESSAGE_SIZE {
            return Err(WlError::MessageTooBig(size));
        }
        if self.input.len() < size {
            return Err(WlError::WouldBlock);
        }
        let mut taken_fds = Vec::new();
        let parsed = {
            let bytes = &self.input[..size];
            parse_args(bytes, message, &mut self.input_fds, &mut taken_fds)
        };
        self.consume(size);
        match parsed {
            Ok(args) => Ok(WlClosure {
                sender_id,
                opcode,
                message,
                args,
            }),
            Err(error) => {
                for fd in taken_fds {
                    self.transport.release_fd(fd);
                }
                Err(error)
            }
        }
    }

    /// Encodes `closure` and appends it to the output buffer.
    ///
    /// File descriptor arguments are transferred to the connection.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when the arguments do not match
    /// the signature and [`WlError::MessageTooBig`] when the encoded message
    /// exceeds [`MAX_MESSAGE_SIZE`].
    pub fn queue_closure(&mut self, closure: &mut WlClosure) -> WlResult<()> {
        let bytes = closure.encode()?;
        let fds = closure.fd_args();
        closure.clear_fds();
        self.output.extend_from_slice(&bytes);
        self.output_fds.extend(fds);
        Ok(())
    }

    /// Writes buffered output to the transport.
    ///
    /// Returns the number of bytes written; buffered data may remain when
    /// the transport refuses to accept everything, in which case
    /// [`WlConnection::wants_write`] reports `true`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Disconnected`] when the peer is gone and
    /// [`WlError::Io`] for transport failures.
    pub fn flush(&mut self) -> WlResult<usize> {
        let mut written = 0usize;
        while !self.output.is_empty() {
            let fds = core::mem::take(&mut self.output_fds);
            match self.transport.send(&self.output, &fds) {
                Ok(0) => {
                    self.output_fds = fds;
                    break;
                }
                Ok(count) => {
                    let count = count.min(self.output.len());
                    written += count;
                    self.output.drain(..count);
                }
                Err(WlError::WouldBlock) => {
                    self.output_fds = fds;
                    break;
                }
                Err(error) => {
                    self.output_fds = fds;
                    if matches!(error, WlError::Io(_) | WlError::Disconnected) {
                        self.disconnected = true;
                    }
                    return Err(error);
                }
            }
        }
        Ok(written)
    }

    /// Waits for events on the transport.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Disconnected`] when the connection is gone.
    pub fn wait(&mut self, timeout: Option<Duration>, mask: WlPollEvents) -> WlResult<WlPollEvents> {
        if self.disconnected {
            return Err(WlError::Disconnected);
        }
        self.transport.wait(timeout, mask)
    }

    /// Releases a file descriptor the caller does not want to keep.
    pub fn release_fd(&mut self, fd: WlFd) {
        self.transport.release_fd(fd);
    }

    /// Releases leftover file descriptor arguments after a dispatch.
    pub fn release_argument_fds(&mut self, args: &mut [WlArgument]) {
        for arg in args {
            if let WlArgument::Fd(fd) = arg
                && *fd >= 0
            {
                let fd = core::mem::replace(fd, -1);
                self.transport.release_fd(fd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wl_handle::{
        CALLBACK_DONE, CALLBACK_INTERFACE, DISPLAY_ERROR, DISPLAY_GET_REGISTRY, DISPLAY_INTERFACE,
        DISPLAY_SYNC, REGISTRY_BIND, REGISTRY_INTERFACE, WlInterface, WlMapSide,
    };

    struct TestTransport {
        input: VecDeque<u8>,
        sent: Vec<u8>,
    }

    impl TestTransport {
        fn new() -> Self {
            Self {
                input: VecDeque::new(),
                sent: Vec::new(),
            }
        }
    }

    impl WlTransport for TestTransport {
        fn recv(&mut self, buf: &mut [u8], _fds: &mut Vec<WlFd>) -> WlResult<usize> {
            if self.input.is_empty() {
                return Err(WlError::WouldBlock);
            }
            let mut count = 0;
            while count < buf.len() {
                match self.input.pop_front() {
                    Some(byte) => {
                        buf[count] = byte;
                        count += 1;
                    }
                    None => break,
                }
            }
            Ok(count)
        }

        fn send(&mut self, data: &[u8], _fds: &[WlFd]) -> WlResult<usize> {
            self.sent.extend_from_slice(data);
            Ok(data.len())
        }

        fn wait(&mut self, _timeout: Option<Duration>, _mask: WlPollEvents) -> WlResult<WlPollEvents> {
            Ok(WlPollEvents::READABLE)
        }

        fn handle(&self) -> WlHandle {
            1
        }
    }

    #[test]
    fn encodes_get_registry_request() {
        let closure = WlClosure::new(
            1,
            DISPLAY_GET_REGISTRY,
            &DISPLAY_INTERFACE.requests[DISPLAY_GET_REGISTRY as usize],
            alloc::vec![WlArgument::NewId(2)],
        )
        .unwrap();
        let bytes = closure.encode().unwrap();
        assert_eq!(bytes, alloc::vec![1, 0, 0, 0, 1, 0, 12, 0, 2, 0, 0, 0]);
    }

    #[test]
    fn rejects_argument_mismatches() {
        let message = &DISPLAY_INTERFACE.requests[DISPLAY_SYNC as usize];
        assert!(WlClosure::new(1, DISPLAY_SYNC, message, alloc::vec![]).is_err());
        assert!(WlClosure::new(1, DISPLAY_SYNC, message, alloc::vec![WlArgument::Uint(1)],).is_err());
    }

    #[test]
    fn encodes_and_demarshals_registry_bind() {
        let message = &REGISTRY_INTERFACE.requests[REGISTRY_BIND as usize];
        let closure = WlClosure::new(
            2,
            REGISTRY_BIND,
            message,
            alloc::vec![
                WlArgument::Uint(4),
                WlArgument::Str(Some(String::from("wl_seat"))),
                WlArgument::Uint(5),
                WlArgument::NewId(10),
            ],
        )
        .unwrap();
        let bytes = closure.encode().unwrap();
        assert_eq!(bytes.len() % 4, 0);

        let mut connection = WlConnection::new(TestTransport::new());
        connection.input.extend_from_slice(&bytes);
        let decoded = connection.demarshal(message).unwrap();
        assert_eq!(decoded.sender_id, 2);
        assert_eq!(decoded.opcode, REGISTRY_BIND);
        assert_eq!(decoded.args[0], WlArgument::Uint(4));
        assert_eq!(decoded.args[1], WlArgument::Str(Some(String::from("wl_seat"))));
        assert_eq!(decoded.args[3], WlArgument::NewId(10));
        assert_eq!(connection.pending_input(), 0);
    }

    #[test]
    fn reports_invalid_and_incomplete_headers() {
        let message = &DISPLAY_INTERFACE.events[DISPLAY_ERROR as usize];
        let mut connection = WlConnection::new(TestTransport::new());
        // Header announces a four byte message, which cannot exist.
        connection
            .input
            .extend_from_slice(&[1, 0, 0, 0, 4, 0, 0, 0]);
        assert!(connection.demarshal(message).is_err());

        let mut connection = WlConnection::new(TestTransport::new());
        // Complete header for a twenty byte message, payload missing.
        connection
            .input
            .extend_from_slice(&[1, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0]);
        assert_eq!(connection.demarshal(message), Err(WlError::WouldBlock));

        // Announced size beyond the protocol limit is not consumed.
        let mut connection = WlConnection::new(TestTransport::new());
        connection
            .input
            .extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0x80]);
        assert_eq!(connection.demarshal(message), Err(WlError::MessageTooBig(0x8000)));
        assert_eq!(connection.pending_input(), 8);
    }

    #[test]
    fn flush_and_read_round_trip() {
        let mut connection = WlConnection::new(TestTransport::new());
        let message = &CALLBACK_INTERFACE.events[CALLBACK_DONE as usize];
        let mut closure =
            WlClosure::new(7, CALLBACK_DONE, message, alloc::vec![WlArgument::Uint(3)]).unwrap();
        connection.queue_closure(&mut closure).unwrap();
        assert!(connection.wants_write());
        let written = connection.flush().unwrap();
        assert_eq!(written, 12);
        assert!(!connection.wants_write());

        let sent = core::mem::take(&mut connection.transport.sent);
        connection.transport.input.extend(sent);
        connection.read().unwrap();
        let decoded = connection.demarshal(message).unwrap();
        assert_eq!(decoded.sender_id, 7);
        assert_eq!(decoded.args[0], WlArgument::Uint(3));
    }

    #[test]
    fn reserves_new_ids_and_validates_objects() {
        let message = &DISPLAY_INTERFACE.requests[DISPLAY_SYNC as usize];
        let closure = WlClosure::new(1, DISPLAY_SYNC, message, alloc::vec![WlArgument::NewId(3)]).unwrap();
        let mut map: WlMap<u32> = WlMap::new(WlMapSide::Server);
        // The client's id space grows densely, so ids 1 and 2 exist already.
        map.reserve_new(1).unwrap();
        map.reserve_new(2).unwrap();
        reserve_new_ids(&closure, &mut map).unwrap();
        assert!(map.is_vacant(3));

        let mut map: WlMap<WlProbe> = WlMap::new(WlMapSide::Client);
        map.insert_at(1, WlProbe).unwrap();
        let mut closure = WlClosure::new(
            1,
            DISPLAY_ERROR,
            &DISPLAY_INTERFACE.events[DISPLAY_ERROR as usize],
            alloc::vec![
                WlArgument::Object(1),
                WlArgument::Uint(0),
                WlArgument::Str(Some(String::from("boom"))),
            ],
        )
        .unwrap();
        lookup_objects(&mut closure, &map).unwrap();

        let mut closure = WlClosure::new(
            1,
            DISPLAY_ERROR,
            &DISPLAY_INTERFACE.events[DISPLAY_ERROR as usize],
            alloc::vec![
                WlArgument::Object(9),
                WlArgument::Uint(0),
                WlArgument::Str(Some(String::from("boom"))),
            ],
        )
        .unwrap();
        assert!(lookup_objects(&mut closure, &map).is_err());
    }

    struct WlProbe;

    impl WlObject for WlProbe {
        fn id(&self) -> u32 {
            1
        }

        fn interface(&self) -> &'static WlInterface {
            &DISPLAY_INTERFACE
        }

        fn version(&self) -> u32 {
            1
        }
    }
}
