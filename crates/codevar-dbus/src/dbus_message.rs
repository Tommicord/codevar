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

//! Message header, body builder and incremental stream decoder.
//!
//! A message starts with a fixed 12 byte header, followed by the
//! header field array `a(yv)` and the body. The [`DbusMessageStream`]
//! splits a byte stream back into messages, returning `Ok(None)` while
//! a message is still incomplete.

use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::dbus_error::{DbusError, DbusResult};
use crate::dbus_marshal::{ByteOrder, DbusReader, DbusWriter, MAX_ARRAY_LEN};
use crate::dbus_names::{
    is_valid_bus_name, is_valid_error_name, is_valid_interface_name, is_valid_member, validate_object_path,
};
use crate::dbus_signature::{
    SignatureIter, type_alignment, validate_array_element_type, validate_signature, validate_single_type,
};
use crate::dbus_transport::{DbusTransport, close_fds};

/// Major protocol version carried by every message.
pub const PROTOCOL_VERSION: u8 = 1;

/// Maximum size of an encoded message in bytes.
pub const MAX_MESSAGE_LEN: usize = 128 * 1024 * 1024;

/// Length of the fixed message header.
pub const FIXED_HEADER_LEN: usize = 12;

/// Flag: the method call does not expect a reply.
pub const FLAG_NO_REPLY_EXPECTED: u8 = 0x01;
/// Flag: the bus must not start a service owner for this call.
pub const FLAG_NO_AUTO_START: u8 = 0x02;
/// Flag: the caller is prepared to wait for interactive authorization.
pub const FLAG_ALLOW_INTERACTIVE_AUTHORIZATION: u8 = 0x04;

/// Header field code of `PATH`.
pub const FIELD_PATH: u8 = 1;
/// Header field code of `INTERFACE`.
pub const FIELD_INTERFACE: u8 = 2;
/// Header field code of `MEMBER`.
pub const FIELD_MEMBER: u8 = 3;
/// Header field code of `ERROR_NAME`.
pub const FIELD_ERROR_NAME: u8 = 4;
/// Header field code of `REPLY_SERIAL`.
pub const FIELD_REPLY_SERIAL: u8 = 5;
/// Header field code of `DESTINATION`.
pub const FIELD_DESTINATION: u8 = 6;
/// Header field code of `SENDER`.
pub const FIELD_SENDER: u8 = 7;
/// Header field code of `SIGNATURE`.
pub const FIELD_SIGNATURE: u8 = 8;
/// Header field code of `UNIX_FDS`.
pub const FIELD_UNIX_FDS: u8 = 9;

fn read_u32_at(bytes: &[u8], position: usize, order: ByteOrder) -> DbusResult<u32> {
    let slice = bytes
        .get(position..position.saturating_add(4))
        .ok_or_else(|| DbusError::invalid_message("message truncated while decoding"))?;
    let mut array = [0u8; 4];
    array.copy_from_slice(slice);
    Ok(match order {
        ByteOrder::Little => u32::from_le_bytes(array),
        ByteOrder::Big => u32::from_be_bytes(array),
    })
}

/// Type of D-Bus message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageKind {
    /// A request that may prompt a reply.
    MethodCall = 1,
    /// The reply of a successful method call.
    MethodReturn = 2,
    /// The reply of a failed method call.
    Error = 3,
    /// A broadcast emission from an object.
    Signal = 4,
}

impl MessageKind {
    /// Returns the kind for a wire type byte, `None` when unknown.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::MethodCall),
            2 => Some(Self::MethodReturn),
            3 => Some(Self::Error),
            4 => Some(Self::Signal),
            _ => None,
        }
    }

    /// Returns the wire type byte of this kind.
    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Builder that records the body bytes together with its signature.
///
/// Scalar writes append one type code to the signature, container
/// writes record the complete type up front and suppress recording
/// while their body runs, so the closure must write values that match
/// the declared element, struct or variant signature.
#[derive(Debug, Clone)]
pub struct BodyWriter {
    writer: DbusWriter,
    signature: String,
    recording: bool,
}

impl BodyWriter {
    /// Creates an empty body writer using `order`.
    #[must_use]
    pub fn new(order: ByteOrder) -> Self {
        Self {
            writer: DbusWriter::new(order),
            signature: String::new(),
            recording: true,
        }
    }

    /// Returns the signature recorded so far.
    #[inline]
    #[must_use]
    pub fn signature(&self) -> &str {
        &self.signature
    }

    /// Returns the current position inside the body.
    #[inline]
    #[must_use]
    pub fn position(&self) -> usize {
        self.writer
            .position()
    }

    /// Consumes the builder, returning the bytes and the signature.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u8>, String) {
        (
            self.writer
                .into_bytes(),
            self.signature,
        )
    }

    #[inline]
    fn record(&mut self, code: u8) {
        if self.recording {
            self.signature
                .push(char::from(code));
        }
    }

    /// Appends a `BYTE` value.
    pub fn write_u8(&mut self, value: u8) -> DbusResult<()> {
        self.record(b'y');
        self.writer
            .write_u8(value);
        Ok(())
    }

    /// Appends a `UINT16` value.
    pub fn write_u16(&mut self, value: u16) -> DbusResult<()> {
        self.record(b'q');
        self.writer
            .write_u16(value);
        Ok(())
    }

    /// Appends a `UINT32` value.
    pub fn write_u32(&mut self, value: u32) -> DbusResult<()> {
        self.record(b'u');
        self.writer
            .write_u32(value);
        Ok(())
    }

    /// Appends an `INT16` value.
    pub fn write_i16(&mut self, value: i16) -> DbusResult<()> {
        self.record(b'n');
        self.writer
            .write_i16(value);
        Ok(())
    }

    /// Appends an `INT32` value.
    pub fn write_i32(&mut self, value: i32) -> DbusResult<()> {
        self.record(b'i');
        self.writer
            .write_i32(value);
        Ok(())
    }

    /// Appends a `UINT64` value.
    pub fn write_u64(&mut self, value: u64) -> DbusResult<()> {
        self.record(b't');
        self.writer
            .write_u64(value);
        Ok(())
    }

    /// Appends an `INT64` value.
    pub fn write_i64(&mut self, value: i64) -> DbusResult<()> {
        self.record(b'x');
        self.writer
            .write_i64(value);
        Ok(())
    }

    /// Appends a `BOOLEAN` value.
    pub fn write_bool(&mut self, value: bool) -> DbusResult<()> {
        self.record(b'b');
        self.writer
            .write_bool(value);
        Ok(())
    }

    /// Appends a `DOUBLE` value.
    pub fn write_f64(&mut self, value: f64) -> DbusResult<()> {
        self.record(b'd');
        self.writer
            .write_f64(value);
        Ok(())
    }

    /// Appends an `UNIX_FD` (`h`) value carrying `index`.
    ///
    /// `index` selects the descriptor inside the list attached to the
    /// message; it must be less than the number of descriptors set
    /// with [`DbusMessage::set_fds`].
    pub fn write_fd(&mut self, index: u32) -> DbusResult<()> {
        self.record(b'h');
        self.writer
            .write_fd(index);
        Ok(())
    }

    /// Appends a `STRING` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when `value` contains an
    /// embedded nul byte.
    pub fn write_str(&mut self, value: &str) -> DbusResult<()> {
        self.writer
            .write_str(value)?;
        self.record(b's');
        Ok(())
    }

    /// Appends an `OBJECT_PATH` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidName`] when `path` is not a valid
    /// object path.
    pub fn write_object_path(&mut self, path: &str) -> DbusResult<()> {
        self.writer
            .write_object_path(path)?;
        self.record(b'o');
        Ok(())
    }

    /// Appends a `SIGNATURE` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidSignature`] when `sig` is invalid.
    pub fn write_signature(&mut self, sig: &str) -> DbusResult<()> {
        self.writer
            .write_signature(sig)?;
        self.record(b'g');
        Ok(())
    }

    /// Appends an array whose element type is `element_sig`.
    ///
    /// The closure must write exactly the elements described by
    /// `element_sig`.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidSignature`] when `element_sig` is
    /// not a single valid type, [`DbusError::InvalidMessage`] when the
    /// encoded array exceeds [`MAX_ARRAY_LEN`], plus whatever `body`
    /// returns.
    pub fn write_array<F>(&mut self, element_sig: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        validate_array_element_type(element_sig)?;
        let code = element_sig.as_bytes()[0];
        let alignment = type_alignment(code)
            .ok_or_else(|| DbusError::invalid_signature(alloc::format!("invalid type code: {code}")))?;
        if self.recording {
            self.signature
                .push('a');
            self.signature
                .push_str(element_sig);
        }
        self.writer
            .align(8);
        let length_pos = self
            .writer
            .position();
        self.writer
            .write_u32(0);
        self.writer
            .align(alignment);
        let start = self
            .writer
            .position();
        let was_recording = core::mem::replace(&mut self.recording, false);
        let result = body(self);
        self.recording = was_recording;
        result?;
        let length = self
            .writer
            .position()
            .saturating_sub(start);
        if length > MAX_ARRAY_LEN {
            return Err(DbusError::invalid_message(alloc::format!(
                "array of {length} bytes exceeds the limit of {MAX_ARRAY_LEN}"
            )));
        }
        self.writer
            .patch_u32(length_pos, length as u32)
    }

    /// Appends a struct whose field types are `fields_sig`.
    ///
    /// The closure must write exactly the fields described by
    /// `fields_sig`.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidSignature`] when `fields_sig` is not
    /// a non-empty valid signature, plus whatever `body` returns.
    pub fn write_struct<F>(&mut self, fields_sig: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        if fields_sig.is_empty() {
            return Err(DbusError::invalid_signature("struct needs at least one field"));
        }
        validate_signature(fields_sig)?;
        if self.recording {
            self.signature
                .push('(');
            self.signature
                .push_str(fields_sig);
            self.signature
                .push(')');
        }
        self.writer
            .align(8);
        let was_recording = core::mem::replace(&mut self.recording, false);
        let result = body(self);
        self.recording = was_recording;
        result
    }

    /// Appends a variant holding the value written by `body`.
    ///
    /// `sig` must describe exactly one type and match what `body`
    /// writes.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidSignature`] when `sig` is not a
    /// single valid type, plus whatever `body` returns.
    pub fn write_variant<F>(&mut self, sig: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        validate_single_type(sig)?;
        if self.recording {
            self.signature
                .push('v');
        }
        self.writer
            .write_signature(sig)?;
        let was_recording = core::mem::replace(&mut self.recording, false);
        let result = body(self);
        self.recording = was_recording;
        result
    }
}

/// A decoded or to-be-sent D-Bus message.
///
/// Header values are owned, so a message can be queued independently
/// of the buffer it was decoded from.
///
/// # File descriptors
///
/// The `fds` list holds raw Unix descriptors announced by the
/// `UNIX_FDS` header field. The message **owns** them: they are
/// closed when the message is dropped unless they were moved out
/// first with [`take_fds`](Self::take_fds), which transfers
/// ownership to the caller.
///
/// - when *sending*, [`DbusMessage::send`] and
///   [`Connection::send_message`](crate::Connection::send_message)
///   take the descriptors out and either hand them to the kernel or
///   keep them queued for a later flush;
/// - when *receiving*, descriptors moved in by
///   [`DbusMessageStream::feed_with_fds`] are owned by the message
///   and by the application once it takes them.
///
/// The type deliberately does not implement `Clone`: a shallow copy
/// would duplicate descriptor numbers and end in a double close.
#[derive(Debug, PartialEq, Eq)]
pub struct DbusMessage {
    kind: MessageKind,
    flags: u8,
    order: ByteOrder,
    serial: u32,
    path: Option<String>,
    interface: Option<String>,
    member: Option<String>,
    error_name: Option<String>,
    reply_serial: Option<u32>,
    destination: Option<String>,
    sender: Option<String>,
    signature: String,
    unix_fds: u32,
    fds: Vec<i32>,
    body: Vec<u8>,
}

impl DbusMessage {
    fn empty(kind: MessageKind, order: ByteOrder) -> Self {
        Self {
            kind,
            flags: 0,
            order,
            serial: 0,
            path: None,
            interface: None,
            member: None,
            error_name: None,
            reply_serial: None,
            destination: None,
            sender: None,
            signature: String::new(),
            unix_fds: 0,
            fds: Vec::new(),
            body: Vec::new(),
        }
    }

    /// Creates a method call message.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidName`] when any of the names is
    /// invalid.
    pub fn method_call(destination: &str, path: &str, interface: &str, member: &str) -> DbusResult<Self> {
        if !is_valid_bus_name(destination) {
            return Err(DbusError::invalid_name(alloc::format!(
                "invalid destination: {destination}"
            )));
        }
        validate_object_path(path)?;
        if !is_valid_interface_name(interface) {
            return Err(DbusError::invalid_name(alloc::format!(
                "invalid interface: {interface}"
            )));
        }
        if !is_valid_member(member) {
            return Err(DbusError::invalid_name(alloc::format!(
                "invalid member: {member}"
            )));
        }
        let mut message = Self::empty(MessageKind::MethodCall, ByteOrder::Little);
        message.destination = Some(destination.to_string());
        message.path = Some(path.to_string());
        message.interface = Some(interface.to_string());
        message.member = Some(member.to_string());
        Ok(message)
    }

    /// Creates a signal message.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidName`] when any of the names is
    /// invalid.
    pub fn signal(path: &str, interface: &str, member: &str) -> DbusResult<Self> {
        validate_object_path(path)?;
        if !is_valid_interface_name(interface) {
            return Err(DbusError::invalid_name(alloc::format!(
                "invalid interface: {interface}"
            )));
        }
        if !is_valid_member(member) {
            return Err(DbusError::invalid_name(alloc::format!(
                "invalid member: {member}"
            )));
        }
        let mut message = Self::empty(MessageKind::Signal, ByteOrder::Little);
        message.path = Some(path.to_string());
        message.interface = Some(interface.to_string());
        message.member = Some(member.to_string());
        Ok(message)
    }

    /// Creates the successful reply to the call identified by
    /// `reply_serial`.
    #[must_use]
    pub fn method_return(reply_serial: u32) -> Self {
        let mut message = Self::empty(MessageKind::MethodReturn, ByteOrder::Little);
        message.reply_serial = Some(reply_serial);
        message
    }

    /// Creates the error reply to the call identified by `reply_serial`.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidName`] when `error_name` is not a
    /// valid error name.
    pub fn error(reply_serial: u32, error_name: &str) -> DbusResult<Self> {
        if !is_valid_error_name(error_name) {
            return Err(DbusError::invalid_name(alloc::format!(
                "invalid error name: {error_name}"
            )));
        }
        let mut message = Self::empty(MessageKind::Error, ByteOrder::Little);
        message.reply_serial = Some(reply_serial);
        message.error_name = Some(error_name.to_string());
        Ok(message)
    }

    /// Returns the message kind.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> MessageKind {
        self.kind
    }

    /// Returns the flag byte.
    #[inline]
    #[must_use]
    pub const fn flags(&self) -> u8 {
        self.flags
    }

    /// Replaces the flag byte.
    pub const fn set_flags(&mut self, flags: u8) {
        self.flags = flags;
    }

    /// Marks the message as not expecting a reply.
    pub fn set_no_reply_expected(&mut self) {
        self.flags |= FLAG_NO_REPLY_EXPECTED;
    }

    /// Returns `true` when a reply to this method call is expected.
    #[inline]
    #[must_use]
    pub const fn is_reply_expected(&self) -> bool {
        matches!(self.kind, MessageKind::MethodCall) && self.flags & FLAG_NO_REPLY_EXPECTED == 0
    }

    /// Returns the message serial.
    #[inline]
    #[must_use]
    pub const fn serial(&self) -> u32 {
        self.serial
    }

    /// Replaces the message serial, which must not be zero.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidState`] when `serial` is zero.
    pub fn set_serial(&mut self, serial: u32) -> DbusResult<()> {
        if serial == 0 {
            return Err(DbusError::invalid_state("serial must not be zero"));
        }
        self.serial = serial;
        Ok(())
    }

    /// Returns the object path, when present.
    #[inline]
    #[must_use]
    pub fn path(&self) -> Option<&str> {
        self.path
            .as_deref()
    }

    /// Returns the interface, when present.
    #[inline]
    #[must_use]
    pub fn interface(&self) -> Option<&str> {
        self.interface
            .as_deref()
    }

    /// Returns the member (method or signal name), when present.
    #[inline]
    #[must_use]
    pub fn member(&self) -> Option<&str> {
        self.member
            .as_deref()
    }

    /// Returns the error name, when present.
    #[inline]
    #[must_use]
    pub fn error_name(&self) -> Option<&str> {
        self.error_name
            .as_deref()
    }

    /// Returns the serial of the replied-to call, when present.
    #[inline]
    #[must_use]
    pub const fn reply_serial(&self) -> Option<u32> {
        self.reply_serial
    }

    /// Returns the destination, when present.
    #[inline]
    #[must_use]
    pub fn destination(&self) -> Option<&str> {
        self.destination
            .as_deref()
    }

    /// Replaces the destination.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidName`] when `destination` is not a
    /// valid bus name.
    pub fn set_destination(&mut self, destination: &str) -> DbusResult<()> {
        if !is_valid_bus_name(destination) {
            return Err(DbusError::invalid_name(alloc::format!(
                "invalid destination: {destination}"
            )));
        }
        self.destination = Some(destination.to_string());
        Ok(())
    }

    /// Returns the sender, when present.
    #[inline]
    #[must_use]
    pub fn sender(&self) -> Option<&str> {
        self.sender
            .as_deref()
    }

    /// Replaces the sender, which the connection sets to its unique
    /// name.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidName`] when `sender` is not a valid
    /// bus name.
    pub fn set_sender(&mut self, sender: &str) -> DbusResult<()> {
        if !is_valid_bus_name(sender) {
            return Err(DbusError::invalid_name(alloc::format!(
                "invalid sender: {sender}"
            )));
        }
        self.sender = Some(sender.to_string());
        Ok(())
    }

    /// Returns the body signature.
    #[inline]
    #[must_use]
    pub fn signature(&self) -> &str {
        &self.signature
    }

    /// Returns the raw body bytes.
    #[inline]
    #[must_use]
    pub fn body(&self) -> &[u8] {
        &self.body
    }

    /// Returns a reader over the raw body bytes.
    #[inline]
    #[must_use]
    pub fn body_reader(&self) -> DbusReader<'_> {
        DbusReader::new(&self.body, self.order)
    }

    /// Returns the announced number of file descriptors.
    #[inline]
    #[must_use]
    pub const fn unix_fds(&self) -> u32 {
        self.unix_fds
    }

    /// Returns the raw file descriptors attached to this message.
    ///
    /// The list is empty when no descriptors are attached (including
    /// plain messages and messages whose descriptors were moved out
    /// by [`take_fds`](Self::take_fds)). See the type-level
    /// documentation for the ownership rules.
    #[inline]
    #[must_use]
    pub fn fds(&self) -> &[i32] {
        &self.fds
    }

    /// Moves the attached descriptors out of the message, transferring
    /// ownership to the caller.
    ///
    /// The caller must close the returned descriptors exactly once.
    /// The announced count from [`unix_fds`](Self::unix_fds) is kept,
    /// so re-encoding a received message still announces the same
    /// number of descriptors even though they are no longer attached;
    /// attach replacements with [`set_fds`](Self::set_fds) before
    /// forwarding.
    pub fn take_fds(&mut self) -> Vec<i32> {
        core::mem::take(&mut self.fds)
    }

    /// Attaches `fds` to the message and announces `fds.len()`
    /// descriptors in the `UNIX_FDS` header field.
    ///
    /// The message takes ownership of the raw descriptors and closes
    /// them on drop unless they are moved out again with
    /// [`take_fds`](Self::take_fds).
    pub fn set_fds(&mut self, fds: Vec<i32>) {
        self.unix_fds = fds.len() as u32;
        self.fds = fds;
    }

    /// Returns the byte order used when encoding this message.
    #[inline]
    #[must_use]
    pub const fn order(&self) -> ByteOrder {
        self.order
    }

    /// Builds the message body, recording its signature automatically.
    ///
    /// Any body built earlier is replaced.
    ///
    /// # Errors
    ///
    /// Returns whatever `body` returns.
    pub fn build_body<F>(&mut self, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut BodyWriter) -> DbusResult<()>,
    {
        let mut writer = BodyWriter::new(self.order);
        body(&mut writer)?;
        let (data, signature) = writer.into_parts();
        self.body = data;
        self.signature = signature;
        Ok(())
    }

    /// Returns the descriptor count announced by the `UNIX_FDS`
    /// header field when encoding.
    ///
    /// Attached descriptors win over the decoded count so that a
    /// message built with [`set_fds`](Self::set_fds) always announces
    /// the number of descriptors it actually carries.
    fn announced_fd_count(&self) -> u32 {
        if !self
            .fds
            .is_empty()
        {
            self.fds
                .len() as u32
        } else {
            self.unix_fds
        }
    }

    fn validate_for_encode(&self) -> DbusResult<()> {
        if self.serial == 0 {
            return Err(DbusError::invalid_state("message serial must not be zero"));
        }
        if self
            .signature
            .as_bytes()
            .contains(&b'h')
            && self.announced_fd_count() == 0
        {
            return Err(DbusError::invalid_message(
                "file descriptor argument without an announced UNIX_FDS header",
            ));
        }
        validate_signature(&self.signature)?;
        match self.kind {
            MessageKind::MethodCall => {
                if self
                    .member
                    .is_none()
                {
                    return Err(DbusError::invalid_state("method call needs a member"));
                }
            }
            MessageKind::Signal => {
                if self
                    .path
                    .is_none()
                    || self
                        .member
                        .is_none()
                {
                    return Err(DbusError::invalid_state("signal needs a path and a member"));
                }
            }
            MessageKind::MethodReturn => {
                if self
                    .reply_serial
                    .is_none()
                {
                    return Err(DbusError::invalid_state("reply needs a reply serial"));
                }
            }
            MessageKind::Error => {
                if self
                    .reply_serial
                    .is_none()
                    || self
                        .error_name
                        .is_none()
                {
                    return Err(DbusError::invalid_state(
                        "error reply needs a reply serial and an error name",
                    ));
                }
            }
        }
        if self
            .body
            .is_empty()
            != self
                .signature
                .is_empty()
        {
            return Err(DbusError::invalid_message(
                "body length does not match the signature",
            ));
        }
        if self
            .body
            .len()
            > MAX_MESSAGE_LEN
        {
            return Err(DbusError::MessageTooBig(
                self.body
                    .len(),
            ));
        }
        Ok(())
    }

    /// Encodes the message into wire format.
    ///
    /// When descriptors are attached (or a `UNIX_FDS` count was
    /// decoded earlier) the `UNIX_FDS` header field is included so
    /// the peer knows how many descriptors accompany the message.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidState`] when a required header
    /// field of the message kind is missing, [`DbusError::InvalidName`]
    /// or [`DbusError::InvalidSignature`] for invalid fields,
    /// [`DbusError::InvalidMessage`] when `h` arguments are used
    /// without an announced descriptor count, and
    /// [`DbusError::MessageTooBig`] when the message exceeds
    /// [`MAX_MESSAGE_LEN`].
    pub fn encode(&self) -> DbusResult<Vec<u8>> {
        self.validate_for_encode()?;
        let body_len = self
            .body
            .len();
        let mut writer = DbusWriter::with_capacity(self.order, body_len + 64);
        writer.write_u8(
            self.order
                .marker(),
        );
        writer.write_u8(
            self.kind
                .as_u8(),
        );
        writer.write_u8(self.flags);
        writer.write_u8(PROTOCOL_VERSION);
        writer.write_u32(body_len as u32);
        writer.write_u32(self.serial);
        writer.align(8);
        let length_pos = writer.position();
        writer.write_u32(0);
        writer.align(8);
        let fields_start = writer.position();

        if let Some(path) = &self.path {
            let path = path.clone();
            write_field(&mut writer, FIELD_PATH, "o", |writer| {
                writer.write_object_path(&path)
            })?;
        }
        if let Some(interface) = &self.interface {
            let interface = interface.clone();
            write_field(&mut writer, FIELD_INTERFACE, "s", |writer| {
                writer.write_str(&interface)
            })?;
        }
        if let Some(member) = &self.member {
            let member = member.clone();
            write_field(&mut writer, FIELD_MEMBER, "s", |writer| writer.write_str(&member))?;
        }
        if let Some(error_name) = &self.error_name {
            let error_name = error_name.clone();
            write_field(&mut writer, FIELD_ERROR_NAME, "s", |writer| {
                writer.write_str(&error_name)
            })?;
        }
        if let Some(reply_serial) = self.reply_serial {
            write_field(&mut writer, FIELD_REPLY_SERIAL, "u", |writer| {
                writer.write_u32(reply_serial);
                Ok(())
            })?;
        }
        if let Some(destination) = &self.destination {
            let destination = destination.clone();
            write_field(&mut writer, FIELD_DESTINATION, "s", |writer| {
                writer.write_str(&destination)
            })?;
        }
        if let Some(sender) = &self.sender {
            let sender = sender.clone();
            write_field(&mut writer, FIELD_SENDER, "s", |writer| writer.write_str(&sender))?;
        }
        if !self
            .signature
            .is_empty()
        {
            let signature = self
                .signature
                .clone();
            write_field(&mut writer, FIELD_SIGNATURE, "g", |writer| {
                writer.write_signature(&signature)
            })?;
        }
        let fd_count = self.announced_fd_count();
        if fd_count > 0 {
            write_field(&mut writer, FIELD_UNIX_FDS, "u", |writer| {
                writer.write_u32(fd_count);
                Ok(())
            })?;
        }

        let fields_len = writer
            .position()
            .saturating_sub(fields_start);
        if fields_len > MAX_ARRAY_LEN {
            return Err(DbusError::MessageTooBig(fields_len));
        }
        writer.patch_u32(length_pos, fields_len as u32)?;
        writer.align(8);
        writer.write_bytes(&self.body);
        let total = writer.position();
        if total > MAX_MESSAGE_LEN {
            return Err(DbusError::MessageTooBig(total));
        }
        writer.patch_u32(4, body_len as u32)?;
        Ok(writer.into_bytes())
    }

    /// Encodes the message and writes it to `transport`, attaching
    /// any descriptors from [`fds`](Self::fds) to the stream position
    /// where the message starts.
    ///
    /// Returns the number of bytes written, or an error if the
    /// message exceeds [`MAX_MESSAGE_LEN`] or the transport reports
    /// a failure. When descriptors were attached and the result is
    /// `Ok(n)` with `n > 0`, the kernel has taken its own reference
    /// to them and the caller should close its copies; otherwise the
    /// caller keeps ownership.
    pub fn send(&self, transport: &mut dyn DbusTransport) -> DbusResult<usize> {
        let bytes = self.encode()?;
        transport.write_with_fds(&bytes, &self.fds)
    }

    /// Decodes a complete message from `bytes`.
    ///
    /// The `UNIX_FDS` header field is parsed into
    /// [`unix_fds`](Self::unix_fds); the descriptors themselves are
    /// not part of the byte stream and are supplied separately, e.g.
    /// through [`DbusMessageStream::feed_with_fds`].
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] for malformed headers,
    /// missing required fields, `h` arguments without an announced
    /// descriptor count or inconsistent bodies.
    pub fn decode(bytes: &[u8]) -> DbusResult<Self> {
        let marker = bytes
            .first()
            .copied()
            .ok_or_else(truncated)?;
        let order = ByteOrder::from_marker(marker)
            .ok_or_else(|| DbusError::invalid_message("invalid endianness marker"))?;
        let mut reader = DbusReader::new(bytes, order);
        reader.read_u8()?;
        let kind = MessageKind::from_u8(reader.read_u8()?)
            .ok_or_else(|| DbusError::invalid_message("unknown message type"))?;
        let flags = reader.read_u8()?;
        let version = reader.read_u8()?;
        if version != PROTOCOL_VERSION {
            return Err(DbusError::invalid_message(alloc::format!(
                "unsupported protocol version {version}"
            )));
        }
        let body_len = reader.read_u32()? as usize;
        let serial = reader.read_u32()?;
        if serial == 0 {
            return Err(DbusError::invalid_message("message serial must not be zero"));
        }
        if body_len > MAX_MESSAGE_LEN {
            return Err(DbusError::MessageTooBig(body_len));
        }

        let mut message = Self::empty(kind, order);
        message.flags = flags;
        message.serial = serial;

        let mut seen = 0u32;
        let mut fields = reader.read_array(8)?;
        while !fields.is_empty() {
            fields.read_struct()?;
            let code = fields.read_u8()?;
            let sig = fields.read_variant_signature()?;
            let bit = 1u32 << code.min(31);
            if code <= FIELD_UNIX_FDS && seen & bit != 0 {
                return Err(DbusError::invalid_message(alloc::format!(
                    "duplicate header field {code}"
                )));
            }
            if code <= FIELD_UNIX_FDS {
                seen |= bit;
            }
            match code {
                FIELD_PATH => {
                    expect_signature(sig, b'o', code)?;
                    message.path = Some(
                        fields
                            .read_object_path()?
                            .to_string(),
                    );
                }
                FIELD_INTERFACE => {
                    expect_signature(sig, b's', code)?;
                    let value = fields
                        .read_str()?
                        .to_string();
                    if !is_valid_interface_name(&value) {
                        return Err(DbusError::invalid_name(alloc::format!(
                            "invalid interface in header: {value}"
                        )));
                    }
                    message.interface = Some(value);
                }
                FIELD_MEMBER => {
                    expect_signature(sig, b's', code)?;
                    let value = fields
                        .read_str()?
                        .to_string();
                    if !is_valid_member(&value) {
                        return Err(DbusError::invalid_name(alloc::format!(
                            "invalid member in header: {value}"
                        )));
                    }
                    message.member = Some(value);
                }
                FIELD_ERROR_NAME => {
                    expect_signature(sig, b's', code)?;
                    let value = fields
                        .read_str()?
                        .to_string();
                    if !is_valid_error_name(&value) {
                        return Err(DbusError::invalid_name(alloc::format!(
                            "invalid error name in header: {value}"
                        )));
                    }
                    message.error_name = Some(value);
                }
                FIELD_REPLY_SERIAL => {
                    expect_signature(sig, b'u', code)?;
                    let value = fields.read_u32()?;
                    if value == 0 {
                        return Err(DbusError::invalid_message("reply serial must not be zero"));
                    }
                    message.reply_serial = Some(value);
                }
                FIELD_DESTINATION => {
                    expect_signature(sig, b's', code)?;
                    let value = fields
                        .read_str()?
                        .to_string();
                    if !is_valid_bus_name(&value) {
                        return Err(DbusError::invalid_name(alloc::format!(
                            "invalid destination in header: {value}"
                        )));
                    }
                    message.destination = Some(value);
                }
                FIELD_SENDER => {
                    expect_signature(sig, b's', code)?;
                    let value = fields
                        .read_str()?
                        .to_string();
                    if !is_valid_bus_name(&value) {
                        return Err(DbusError::invalid_name(alloc::format!(
                            "invalid sender in header: {value}"
                        )));
                    }
                    message.sender = Some(value);
                }
                FIELD_SIGNATURE => {
                    expect_signature(sig, b'g', code)?;
                    message.signature = fields
                        .read_signature()?
                        .to_string();
                }
                FIELD_UNIX_FDS => {
                    expect_signature(sig, b'u', code)?;
                    message.unix_fds = fields.read_u32()?;
                }
                _ => skip_value(&mut fields, sig)?,
            }
        }

        reader.align(8)?;
        message.body = reader
            .read_bytes(body_len)?
            .to_vec();
        if !reader.is_empty() {
            return Err(DbusError::invalid_message("trailing bytes after the message"));
        }

        validate_decode(&message)?;
        Ok(message)
    }
}

impl Drop for DbusMessage {
    fn drop(&mut self) {
        // A message that still holds descriptors nobody took owns
        // them, so release them here to keep the RAII contract.
        close_fds(core::mem::take(&mut self.fds));
    }
}

fn truncated() -> DbusError {
    DbusError::invalid_message("message truncated while decoding")
}

fn expect_signature(actual: &str, expected: u8, code: u8) -> DbusResult<()> {
    if actual.as_bytes() == [expected] {
        Ok(())
    } else {
        Err(DbusError::invalid_message(alloc::format!(
            "header field {code} has signature {actual}, expected {}",
            char::from(expected)
        )))
    }
}

fn write_field<F>(writer: &mut DbusWriter, code: u8, sig: &str, value: F) -> DbusResult<()>
where
    F: FnOnce(&mut DbusWriter) -> DbusResult<()>,
{
    writer.align(8);
    writer.write_u8(code);
    writer.write_variant(sig, value)
}

fn validate_decode(message: &DbusMessage) -> DbusResult<()> {
    match message.kind {
        MessageKind::MethodCall => {
            if message
                .member
                .is_none()
            {
                return Err(DbusError::invalid_message("method call without a member"));
            }
        }
        MessageKind::Signal => {
            if message
                .path
                .is_none()
                || message
                    .member
                    .is_none()
            {
                return Err(DbusError::invalid_message("signal without a path or member"));
            }
        }
        MessageKind::MethodReturn => {
            if message
                .reply_serial
                .is_none()
            {
                return Err(DbusError::invalid_message("reply without a reply serial"));
            }
        }
        MessageKind::Error => {
            if message
                .reply_serial
                .is_none()
                || message
                    .error_name
                    .is_none()
            {
                return Err(DbusError::invalid_message(
                    "error reply without a reply serial or error name",
                ));
            }
        }
    }
    if message
        .body
        .is_empty()
        != message
            .signature
            .is_empty()
    {
        return Err(DbusError::invalid_message(
            "body length does not match the signature",
        ));
    }
    if message
        .signature
        .as_bytes()
        .contains(&b'h')
        && message.unix_fds == 0
    {
        return Err(DbusError::invalid_message(
            "file descriptor argument without an announced UNIX_FDS header",
        ));
    }
    Ok(())
}

/// Skips one value of type `sig` without decoding it.
fn skip_value(reader: &mut DbusReader<'_>, sig: &str) -> DbusResult<()> {
    let code = sig
        .as_bytes()
        .first()
        .copied()
        .ok_or_else(|| DbusError::invalid_signature("empty signature while skipping a value"))?;
    match code {
        b'y' => {
            reader.read_u8()?;
        }
        b'b' | b'i' | b'u' | b'h' => {
            reader.read_u32()?;
        }
        b'n' | b'q' => {
            reader.read_u16()?;
        }
        b'x' | b't' | b'd' => {
            reader.read_u64()?;
        }
        b's' | b'o' => {
            reader.read_str()?;
        }
        b'g' => {
            reader.read_signature()?;
        }
        b'v' => {
            let inner = reader.read_variant_signature()?;
            skip_value(reader, inner)?;
        }
        b'a' => {
            let element = &sig[1..];
            let element_code = element
                .as_bytes()
                .first()
                .copied()
                .ok_or_else(|| DbusError::invalid_signature("array without an element type"))?;
            let alignment = type_alignment(element_code).ok_or_else(|| {
                DbusError::invalid_signature(alloc::format!("invalid type code: {element_code}"))
            })?;
            let mut elements = reader.read_array(alignment)?;
            while !elements.is_empty() {
                skip_value(&mut elements, element)?;
            }
        }
        b'(' => {
            reader.read_struct()?;
            let inner = sig
                .strip_prefix('(')
                .and_then(|rest| rest.strip_suffix(')'))
                .ok_or_else(|| DbusError::invalid_signature("malformed struct signature"))?;
            for field in SignatureIter::new(inner) {
                skip_value(reader, field)?;
            }
        }
        b'{' => {
            reader.read_struct()?;
            let inner = sig
                .strip_prefix('{')
                .and_then(|rest| rest.strip_suffix('}'))
                .ok_or_else(|| DbusError::invalid_signature("malformed dict signature"))?;
            let mut parts = SignatureIter::new(inner);
            let key = parts
                .next()
                .ok_or_else(|| DbusError::invalid_signature("dict without a key"))?;
            let value = parts
                .next()
                .ok_or_else(|| DbusError::invalid_signature("dict without a value"))?;
            skip_value(reader, key)?;
            skip_value(reader, value)?;
        }
        _ => {
            return Err(DbusError::invalid_signature(alloc::format!(
                "invalid type code: {code}"
            )));
        }
    }
    Ok(())
}

/// Incremental decoder that splits a byte stream into messages.
///
/// Descriptors sent with messages are queued first by
/// [`feed_with_fds`](Self::feed_with_fds) and moved into each message
/// as soon as it is complete, so the queue order always matches the
/// message order. Queued descriptors that never reach a message are
/// closed when the stream is [`clear`](Self::clear)ed or dropped.
#[derive(Debug, Default)]
pub struct DbusMessageStream {
    buffer: Vec<u8>,
    pending_fds: VecDeque<i32>,
}

impl DbusMessageStream {
    /// Creates an empty stream decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            pending_fds: VecDeque::new(),
        }
    }

    /// Appends bytes read from the transport.
    ///
    /// Equivalent to [`feed_with_fds`](Self::feed_with_fds) with an
    /// empty descriptor list; use `feed_with_fds` whenever the read
    /// that produced `data` also carried descriptors.
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer
            .extend_from_slice(data);
    }

    /// Queues `fds` and then appends `data` read from the transport.
    ///
    /// The descriptors must come from the *same* transport read as
    /// `data`: on Linux `SCM_RIGHTS` descriptors attach to a byte
    /// position and are returned by the read that reaches it, so
    /// queueing them before the bytes keeps descriptor order aligned
    /// with message order. Descriptors are owned by the stream until
    /// they are moved into a completed message, after which they
    /// belong to the application (see [`DbusMessage`] for the
    /// ownership rules).
    pub fn feed_with_fds(&mut self, data: &[u8], fds: Vec<i32>) {
        self.pending_fds
            .extend(fds);
        self.buffer
            .extend_from_slice(data);
    }

    /// Returns the number of buffered bytes.
    #[inline]
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.buffer
            .len()
    }

    /// Returns the number of descriptors queued but not yet attached
    /// to a message.
    #[inline]
    #[must_use]
    pub fn pending_fds(&self) -> usize {
        self.pending_fds
            .len()
    }

    /// Drops all buffered bytes and closes queued descriptors that
    /// never reached a message.
    pub fn clear(&mut self) {
        self.buffer
            .clear();
        close_fds(core::mem::take(&mut self.pending_fds));
    }

    /// Returns the next complete message, `Ok(None)` while incomplete.
    ///
    /// Unknown message types are skipped, as required by the
    /// specification.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] for malformed framing,
    /// a message whose `UNIX_FDS` count exceeds the number of
    /// queued descriptors (the message bytes are kept so the caller
    /// may queue the missing descriptors and retry), and
    /// [`DbusError::MessageTooBig`] for oversized messages without
    /// consuming the offending bytes, so the caller can drop the
    /// connection.
    pub fn next_message(&mut self) -> DbusResult<Option<DbusMessage>> {
        loop {
            if self
                .buffer
                .len()
                < FIXED_HEADER_LEN
            {
                return Ok(None);
            }
            let marker = self.buffer[0];
            let order = ByteOrder::from_marker(marker)
                .ok_or_else(|| DbusError::invalid_message("invalid endianness marker"))?;
            if self.buffer[3] != PROTOCOL_VERSION {
                return Err(DbusError::invalid_message(alloc::format!(
                    "unsupported protocol version {}",
                    self.buffer[3]
                )));
            }
            let body_len = read_u32_at(&self.buffer, 4, order)? as usize;
            let serial = read_u32_at(&self.buffer, 8, order)?;
            if serial == 0 {
                return Err(DbusError::invalid_message("message serial must not be zero"));
            }
            if body_len > MAX_MESSAGE_LEN {
                return Err(DbusError::MessageTooBig(body_len));
            }
            if self
                .buffer
                .len()
                < 20
            {
                return Ok(None);
            }
            let fields_len = read_u32_at(&self.buffer, 16, order)? as usize;
            if fields_len > MAX_ARRAY_LEN {
                return Err(DbusError::MessageTooBig(fields_len));
            }
            let fields_end = 24usize
                .checked_add(fields_len)
                .ok_or(DbusError::MessageTooBig(fields_len))?;
            let header_end = fields_end
                .checked_add(7)
                .map(|value| value & !7)
                .ok_or(DbusError::MessageTooBig(fields_len))?;
            let total = header_end
                .checked_add(body_len)
                .ok_or(DbusError::MessageTooBig(body_len))?;
            if total > MAX_MESSAGE_LEN {
                return Err(DbusError::MessageTooBig(total));
            }
            if self
                .buffer
                .len()
                < total
            {
                return Ok(None);
            }
            let decoded = if MessageKind::from_u8(self.buffer[1]).is_some() {
                Some(DbusMessage::decode(&self.buffer[..total]))
            } else {
                None
            };
            if let Some(Ok(ref message)) = decoded {
                let needed = message.unix_fds() as usize;
                if self
                    .pending_fds
                    .len()
                    < needed
                {
                    return Err(DbusError::invalid_message(alloc::format!(
                        "message announces {needed} file descriptors but only {} are queued",
                        self.pending_fds
                            .len()
                    )));
                }
            }
            self.buffer
                .drain(..total);
            if let Some(result) = decoded {
                let mut message = result?;
                let needed = message.unix_fds() as usize;
                if needed > 0 {
                    let fds = self
                        .pending_fds
                        .drain(..needed)
                        .collect();
                    message.set_fds(fds);
                }
                return Ok(Some(message));
            }
        }
    }
}

impl Drop for DbusMessageStream {
    fn drop(&mut self) {
        // Descriptors never attached to a message are still owned by
        // the stream; release them so they cannot leak.
        close_fds(core::mem::take(&mut self.pending_fds));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn hello_call() -> DbusMessage {
        DbusMessage::method_call(
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
            "Hello",
        )
        .unwrap()
    }

    #[test]
    fn encodes_hello_call_with_golden_bytes() {
        let mut message = hello_call();
        message
            .set_serial(1)
            .unwrap();
        let bytes = message
            .encode()
            .unwrap();
        let expected = vec![
            // Fixed header: 'l', METHOD_CALL, flags 0, version 1.
            0x6c, 0x01, 0x00, 0x01, // body length 0.
            0x00, 0x00, 0x00, 0x00, // serial 1.
            0x01, 0x00, 0x00, 0x00, // padding to the header field array.
            0x00, 0x00, 0x00, 0x00, // header field array length: 109.
            0x6d, 0x00, 0x00, 0x00, // padding to the first struct.
            0x00, 0x00, 0x00, 0x00, // PATH field.
            0x01, 0x01, 0x6f, 0x00, 0x15, 0x00, 0x00, 0x00, // "/org/freedesktop/DBus"
            0x2f, 0x6f, 0x72, 0x67, 0x2f, 0x66, 0x72, 0x65, 0x65, 0x64, 0x65, 0x73, 0x6b, 0x74, 0x6f, 0x70,
            0x2f, 0x44, 0x42, 0x75, 0x73, 0x00, // padding to the next struct.
            0x00, 0x00, // INTERFACE field.
            0x02, 0x01, 0x73, 0x00, 0x14, 0x00, 0x00, 0x00, // "org.freedesktop.DBus"
            0x6f, 0x72, 0x67, 0x2e, 0x66, 0x72, 0x65, 0x65, 0x64, 0x65, 0x73, 0x6b, 0x74, 0x6f, 0x70, 0x2e,
            0x44, 0x42, 0x75, 0x73, 0x00, // padding to the next struct.
            0x00, 0x00, 0x00, // MEMBER field.
            0x03, 0x01, 0x73, 0x00, 0x05, 0x00, 0x00, 0x00, // "Hello"
            0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x00, // padding to the next struct.
            0x00, 0x00, // DESTINATION field.
            0x06, 0x01, 0x73, 0x00, 0x14, 0x00, 0x00, 0x00, // "org.freedesktop.DBus"
            0x6f, 0x72, 0x67, 0x2e, 0x66, 0x72, 0x65, 0x65, 0x64, 0x65, 0x73, 0x6b, 0x74, 0x6f, 0x70, 0x2e,
            0x44, 0x42, 0x75, 0x73, 0x00, // padding to the 8 byte header boundary.
            0x00, 0x00, 0x00,
        ];
        assert_eq!(bytes, expected);
        assert_eq!(bytes.len() % 8, 0);
    }

    #[test]
    fn round_trips_every_message_kind() {
        let mut call = hello_call();
        call.set_serial(7)
            .unwrap();
        call.set_no_reply_expected();
        call.build_body(|body| {
            body.write_str("codevar")?;
            body.write_u32(42)?;
            Ok(())
        })
        .unwrap();
        let bytes = call
            .encode()
            .unwrap();
        let decoded = DbusMessage::decode(&bytes).unwrap();
        assert_eq!(decoded.kind(), MessageKind::MethodCall);
        assert_eq!(decoded.serial(), 7);
        assert_eq!(decoded.destination(), Some("org.freedesktop.DBus"));
        assert_eq!(decoded.path(), Some("/org/freedesktop/DBus"));
        assert_eq!(decoded.interface(), Some("org.freedesktop.DBus"));
        assert_eq!(decoded.member(), Some("Hello"));
        assert_eq!(decoded.signature(), "su");
        assert!(!decoded.is_reply_expected());
        let mut reader = decoded.body_reader();
        assert_eq!(
            reader
                .read_str()
                .unwrap(),
            "codevar"
        );
        assert_eq!(
            reader
                .read_u32()
                .unwrap(),
            42
        );

        let mut reply = DbusMessage::method_return(7);
        reply
            .set_serial(9)
            .unwrap();
        reply
            .build_body(|body| body.write_bool(true))
            .unwrap();
        let decoded = DbusMessage::decode(
            &reply
                .encode()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(decoded.kind(), MessageKind::MethodReturn);
        assert_eq!(decoded.reply_serial(), Some(7));
        assert_eq!(decoded.signature(), "b");

        let mut error = DbusMessage::error(7, "org.freedesktop.DBus.Error.Failed").unwrap();
        error
            .set_serial(10)
            .unwrap();
        error
            .build_body(|body| body.write_str("boom"))
            .unwrap();
        let decoded = DbusMessage::decode(
            &error
                .encode()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(decoded.kind(), MessageKind::Error);
        assert_eq!(decoded.error_name(), Some("org.freedesktop.DBus.Error.Failed"));
        assert_eq!(decoded.reply_serial(), Some(7));

        let mut signal = DbusMessage::signal("/org/example", "org.example.Interface", "Changed").unwrap();
        signal
            .set_serial(11)
            .unwrap();
        let decoded = DbusMessage::decode(
            &signal
                .encode()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(decoded.kind(), MessageKind::Signal);
        assert_eq!(decoded.path(), Some("/org/example"));
        assert_eq!(decoded.member(), Some("Changed"));
        assert_eq!(decoded.signature(), "");
        assert!(
            decoded
                .body()
                .is_empty()
        );
    }

    #[test]
    fn round_trips_nested_body_values() {
        let mut message =
            DbusMessage::method_call("org.example.Service", "/org/example", "org.example.Iface", "Send")
                .unwrap();
        message
            .set_serial(3)
            .unwrap();
        message
            .build_body(|body| {
                body.write_i64(-5)?;
                body.write_array("s", |body| {
                    body.write_str("alpha")?;
                    body.write_str("beta")?;
                    Ok(())
                })?;
                body.write_struct("iu", |body| {
                    body.write_u8(9)?;
                    body.write_u32(1000)?;
                    Ok(())
                })?;
                body.write_variant("d", |body| {
                    body.write_f64(2.5)?;
                    Ok(())
                })?;
                Ok(())
            })
            .unwrap();
        assert_eq!(message.signature(), "xas(iu)v");

        let bytes = message
            .encode()
            .unwrap();
        let decoded = DbusMessage::decode(&bytes).unwrap();
        assert_eq!(decoded.signature(), message.signature());
        let mut reader = decoded.body_reader();
        assert_eq!(
            reader
                .read_i64()
                .unwrap(),
            -5
        );
        let mut array = reader
            .read_array(4)
            .unwrap();
        assert_eq!(
            array
                .read_str()
                .unwrap(),
            "alpha"
        );
        assert_eq!(
            array
                .read_str()
                .unwrap(),
            "beta"
        );
        reader
            .read_struct()
            .unwrap();
        assert_eq!(
            reader
                .read_u8()
                .unwrap(),
            9
        );
        assert_eq!(
            reader
                .read_u32()
                .unwrap(),
            1000
        );
        assert_eq!(
            reader
                .read_variant_signature()
                .unwrap(),
            "d"
        );
        assert_eq!(
            reader
                .read_f64()
                .unwrap(),
            2.5
        );
    }

    #[test]
    fn rejects_missing_required_fields() {
        let mut call = DbusMessage::method_call("a.b", "/x", "a.b.C", "Go").unwrap();
        call.set_serial(1)
            .unwrap();
        call.member = None;
        assert!(
            call.encode()
                .is_err()
        );

        let mut reply = DbusMessage::method_return(5);
        reply
            .set_serial(2)
            .unwrap();
        reply.reply_serial = None;
        assert!(
            reply
                .encode()
                .is_err()
        );

        let mut signal = DbusMessage::signal("/x", "a.b.C", "Ping").unwrap();
        signal
            .set_serial(3)
            .unwrap();
        signal.path = None;
        assert!(
            signal
                .encode()
                .is_err()
        );

        let call = hello_call();
        assert!(
            call.encode()
                .is_err()
        );
        assert_eq!(
            call.encode(),
            Err(DbusError::InvalidState(String::from(
                "message serial must not be zero"
            )))
        );
    }

    #[test]
    fn rejects_fd_arguments_without_an_announced_count() {
        let mut call = hello_call();
        call.set_serial(1)
            .unwrap();
        call.signature = String::from("h");
        call.body = vec![0, 0, 0, 0];
        // Without a UNIX_FDS header the `h` index resolves nowhere.
        assert!(matches!(call.encode(), Err(DbusError::InvalidMessage(_))));

        // A message decoded from the wire with `h` arguments but no
        // UNIX_FDS header is rejected the same way. The header field
        // is private, so craft the frame directly.
        let mut writer = DbusWriter::new(ByteOrder::Little);
        writer.write_u8(b'l');
        writer.write_u8(MessageKind::MethodCall.as_u8());
        writer.write_u8(0);
        writer.write_u8(PROTOCOL_VERSION);
        writer.write_u32(4);
        writer.write_u32(1);
        writer.align(8);
        let length_pos = writer.position();
        writer.write_u32(0);
        writer.align(8);
        let fields_start = writer.position();
        write_field(&mut writer, FIELD_MEMBER, "s", |writer| writer.write_str("Open"))
            // Justified: the field is well formed by construction.
            .unwrap();
        write_field(&mut writer, FIELD_SIGNATURE, "g", |writer| {
            writer.write_signature("h")
        })
        // Justified: the field is well formed by construction.
        .unwrap();
        let fields_len = writer.position() - fields_start;
        writer
            .patch_u32(length_pos, fields_len as u32)
            // Justified: `length_pos` was just reserved.
            .unwrap();
        writer.align(8);
        writer.write_fd(0);
        let bytes = writer.into_bytes();
        assert!(matches!(
            DbusMessage::decode(&bytes),
            Err(DbusError::InvalidMessage(_))
        ));
    }

    #[test]
    fn round_trips_the_unix_fds_header_and_body_index() {
        let mut call = hello_call();
        call.set_serial(1)
            .unwrap();
        call.build_body(|body| {
            body.write_fd(0)?;
            body.write_str("payload")
        })
        .unwrap();
        assert_eq!(call.signature(), "hs");
        // A real descriptor: dropping the message below closes it.
        let mut pipe = [-1i32; 2];
        // SAFETY: `pipe` is a valid two element array; on success the
        // kernel fills both entries with open descriptors.
        assert_eq!(unsafe { libc::pipe(pipe.as_mut_ptr()) }, 0);
        call.set_fds(vec![pipe[0]]);

        let bytes = call
            .encode()
            .unwrap();
        // The header announces the count even though the descriptors
        // themselves travel out of band.
        let decoded = DbusMessage::decode(&bytes).unwrap();
        assert_eq!(decoded.unix_fds(), 1);
        assert_eq!(decoded.fds(), &[] as &[i32]);
        let mut reader = decoded.body_reader();
        assert_eq!(
            reader
                .read_fd()
                .unwrap(),
            0
        );
        assert_eq!(
            reader
                .read_str()
                .unwrap(),
            "payload"
        );

        // The original message still owns its descriptor and closes
        // it when it goes out of scope.
        assert_eq!(call.fds(), &[pipe[0]]);
        drop(call);
        drop(decoded);
        // SAFETY: `fcntl` only inspects the descriptor.
        assert_eq!(unsafe { libc::fcntl(pipe[0], libc::F_GETFD) }, -1);
        // SAFETY: closing the write end this test still owns.
        unsafe { libc::close(pipe[1]) };
    }

    #[test]
    fn stream_moves_queued_fds_into_completed_messages() {
        // Placeholder descriptor numbers: negative values are skipped
        // when messages are dropped, so no foreign descriptor can be
        // closed by accident.
        let mut first = hello_call();
        first
            .set_serial(1)
            .unwrap();
        first
            .build_body(|body| body.write_fd(0))
            .unwrap();
        first.set_fds(vec![-1]);
        let first_bytes = first
            .encode()
            .unwrap();
        let first_fds = first.take_fds();

        let mut second = DbusMessage::signal("/a", "b.C", "Two").unwrap();
        second
            .set_serial(2)
            .unwrap();
        second
            .build_body(|body| body.write_fd(0))
            .unwrap();
        second.set_fds(vec![-2, -3]);
        let second_bytes = second
            .encode()
            .unwrap();
        let second_fds = second.take_fds();

        let mut stream = DbusMessageStream::new();
        stream.feed_with_fds(&first_bytes, first_fds);
        stream.feed_with_fds(&second_bytes, second_fds);
        assert_eq!(stream.pending_fds(), 3);

        // Justified: both messages were encoded completely.
        let decoded_first = stream
            .next_message()
            .unwrap()
            .unwrap();
        assert_eq!(decoded_first.member(), Some("Hello"));
        assert_eq!(decoded_first.fds(), &[-1]);
        let decoded_second = stream
            .next_message()
            .unwrap()
            .unwrap();
        assert_eq!(decoded_second.member(), Some("Two"));
        assert_eq!(decoded_second.fds(), &[-2, -3]);
        assert_eq!(stream.pending_fds(), 0);
        assert_eq!(
            stream
                .next_message()
                .unwrap(),
            None
        );
    }

    #[test]
    fn stream_reports_missing_fds_without_consuming_the_message() {
        // Placeholder descriptor numbers, see the test above.
        let mut call = hello_call();
        call.set_serial(1)
            .unwrap();
        call.build_body(|body| {
            body.write_fd(0)?;
            body.write_fd(1)
        })
        .unwrap();
        call.set_fds(vec![-1, -2]);
        let bytes = call
            .encode()
            .unwrap();
        let fds = call.take_fds();

        let mut stream = DbusMessageStream::new();
        stream.feed_with_fds(&bytes, fds[..1].to_vec());
        let error = stream
            .next_message()
            .unwrap_err();
        assert!(matches!(error, DbusError::InvalidMessage(_)));
        // The bytes are kept, so supplying the missing descriptor
        // lets the caller recover.
        assert_eq!(stream.buffered(), bytes.len());
        assert_eq!(stream.pending_fds(), 1);

        stream.feed_with_fds(&[], fds[1..].to_vec());
        // Justified: both descriptors are queued now.
        let message = stream
            .next_message()
            .unwrap()
            .unwrap();
        assert_eq!(message.fds(), &[-1, -2]);
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn stream_closes_fds_that_never_reach_a_message() {
        // Justified: the pipe cannot legitimately fail here.
        let mut pipe = [-1i32; 2];
        // SAFETY: `pipe` is a valid two element array; on success the
        // kernel fills both entries with open descriptors.
        assert_eq!(unsafe { libc::pipe(pipe.as_mut_ptr()) }, 0);
        {
            let mut stream = DbusMessageStream::new();
            // Bytes of a message that never completes plus its fd.
            stream.feed_with_fds(&[0x6c], vec![pipe[0]]);
            assert_eq!(stream.pending_fds(), 1);
        }
        // SAFETY: `fcntl` only inspects the descriptor.
        assert_eq!(unsafe { libc::fcntl(pipe[0], libc::F_GETFD) }, -1);
        // SAFETY: closing the write end this test still owns.
        unsafe { libc::close(pipe[1]) };
    }

    #[test]
    fn rejects_body_signature_mismatch() {
        let mut call = hello_call();
        call.set_serial(1)
            .unwrap();
        call.signature = String::from("s");
        assert!(
            call.encode()
                .is_err()
        );
    }

    #[test]
    fn decodes_big_endian_messages() {
        let mut writer = DbusWriter::new(ByteOrder::Big);
        writer.write_u8(ByteOrder::Big.marker());
        writer.write_u8(MessageKind::MethodReturn.as_u8());
        writer.write_u8(0);
        writer.write_u8(PROTOCOL_VERSION);
        writer.write_u32(4);
        writer.write_u32(5);
        writer.align(8);
        let length_pos = writer.position();
        writer.write_u32(0);
        writer.align(8);
        let fields_start = writer.position();
        write_field(&mut writer, FIELD_REPLY_SERIAL, "u", |writer| {
            writer.write_u32(4);
            Ok(())
        })
        .unwrap();
        write_field(&mut writer, FIELD_SIGNATURE, "g", |writer| {
            writer.write_signature("u")
        })
        .unwrap();
        let fields_len = writer.position() - fields_start;
        writer
            .patch_u32(length_pos, fields_len as u32)
            .unwrap();
        writer.align(8);
        writer.write_u32(0x1122_3344);
        let bytes = writer.into_bytes();
        assert_eq!(bytes[0], b'B');

        let decoded = DbusMessage::decode(&bytes).unwrap();
        assert_eq!(decoded.order(), ByteOrder::Big);
        assert_eq!(decoded.kind(), MessageKind::MethodReturn);
        assert_eq!(decoded.reply_serial(), Some(4));
        assert_eq!(decoded.signature(), "u");
        assert_eq!(
            decoded
                .body_reader()
                .read_u32()
                .unwrap(),
            0x1122_3344
        );
    }

    #[test]
    fn stream_returns_none_until_complete() {
        let mut message = hello_call();
        message
            .set_serial(1)
            .unwrap();
        let bytes = message
            .encode()
            .unwrap();

        let mut stream = DbusMessageStream::new();
        stream.feed(&bytes[..5]);
        assert_eq!(
            stream
                .next_message()
                .unwrap(),
            None
        );
        stream.feed(&bytes[5..19]);
        assert_eq!(
            stream
                .next_message()
                .unwrap(),
            None
        );
        stream.feed(&bytes[19..]);
        let decoded = stream
            .next_message()
            .unwrap();
        assert!(decoded.is_some());
        assert_eq!(
            stream
                .next_message()
                .unwrap(),
            None
        );
        assert_eq!(stream.buffered(), 0);
    }

    #[test]
    fn stream_decodes_back_to_back_messages() {
        let mut first = hello_call();
        first
            .set_serial(1)
            .unwrap();
        let mut second = DbusMessage::signal("/a", "b.C", "Ping").unwrap();
        second
            .set_serial(2)
            .unwrap();
        let mut data = first
            .encode()
            .unwrap();
        data.extend_from_slice(
            &second
                .encode()
                .unwrap(),
        );

        let mut stream = DbusMessageStream::new();
        stream.feed(&data);
        let first_decoded = stream
            .next_message()
            .unwrap()
            .unwrap();
        assert_eq!(first_decoded.member(), Some("Hello"));
        let second_decoded = stream
            .next_message()
            .unwrap()
            .unwrap();
        assert_eq!(second_decoded.kind(), MessageKind::Signal);
        assert_eq!(
            stream
                .next_message()
                .unwrap(),
            None
        );
    }

    #[test]
    fn stream_skips_unknown_message_types() {
        let mut message = hello_call();
        message
            .set_serial(1)
            .unwrap();
        let mut bytes = message
            .encode()
            .unwrap();
        bytes[1] = 9;
        let mut stream = DbusMessageStream::new();
        stream.feed(&bytes);
        assert_eq!(
            stream
                .next_message()
                .unwrap(),
            None
        );
        assert_eq!(stream.buffered(), 0);
    }

    #[test]
    fn stream_rejects_broken_framing() {
        let mut stream = DbusMessageStream::new();
        stream.feed(&[0x78, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]);
        assert!(
            stream
                .next_message()
                .is_err()
        );

        let mut stream = DbusMessageStream::new();
        stream.feed(&[0x6c, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0]);
        assert!(
            stream
                .next_message()
                .is_err()
        );

        let mut stream = DbusMessageStream::new();
        stream.feed(&[0x6c, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(
            stream
                .next_message()
                .is_err()
        );

        // A body beyond the size limit is reported without consuming.
        let mut stream = DbusMessageStream::new();
        stream.feed(&[0x6c, 1, 0, 1, 0xff, 0xff, 0xff, 0x7f, 1, 0, 0, 0]);
        assert_eq!(stream.next_message(), Err(DbusError::MessageTooBig(0x7fff_ffff)));
        assert_eq!(stream.buffered(), 12);
    }

    #[test]
    fn decode_rejects_duplicate_and_unknown_typed_fields() {
        let mut writer = DbusWriter::new(ByteOrder::Little);
        writer.write_u8(b'l');
        writer.write_u8(MessageKind::MethodCall.as_u8());
        writer.write_u8(0);
        writer.write_u8(PROTOCOL_VERSION);
        writer.write_u32(0);
        writer.write_u32(1);
        writer.align(8);
        let length_pos = writer.position();
        writer.write_u32(0);
        writer.align(8);
        let fields_start = writer.position();
        write_field(&mut writer, FIELD_MEMBER, "s", |writer| writer.write_str("Hello")).unwrap();
        write_field(&mut writer, FIELD_MEMBER, "s", |writer| writer.write_str("Hello")).unwrap();
        let fields_len = writer.position() - fields_start;
        writer
            .patch_u32(length_pos, fields_len as u32)
            .unwrap();
        writer.align(8);
        let bytes = writer.into_bytes();
        assert!(DbusMessage::decode(&bytes).is_err());
    }

    #[test]
    fn message_kind_conversions() {
        assert_eq!(MessageKind::from_u8(1), Some(MessageKind::MethodCall));
        assert_eq!(MessageKind::from_u8(4), Some(MessageKind::Signal));
        assert_eq!(MessageKind::from_u8(0), None);
        assert_eq!(MessageKind::from_u8(5), None);
        assert_eq!(MessageKind::Signal.as_u8(), 4);
    }

    #[test]
    fn body_writer_encodes_dict_entry_arrays() {
        // Justified: these fixed arguments cannot fail to build.
        let mut message = DbusMessage::method_call(
            "org.example.Test",
            "/org/example/Test",
            "org.example.Test",
            "Options",
        )
        .unwrap();
        message
            .build_body(|body| {
                body.write_array("{sv}", |body| {
                    body.write_struct("sv", |body| {
                        body.write_str("token")?;
                        body.write_variant("s", |body| body.write_str("abc123"))
                    })?;
                    body.write_struct("sv", |body| {
                        body.write_str("count")?;
                        body.write_variant("u", |body| body.write_u32(7))
                    })
                })
            })
            .unwrap();
        assert_eq!(message.signature(), "a{sv}");

        let mut reader = message.body_reader();
        let mut dict = reader
            .read_array(8)
            .unwrap();
        let mut seen = 0;
        while dict.remaining() > 0 {
            dict.read_struct()
                .unwrap();
            let key = dict
                .read_str()
                .unwrap();
            let sig = dict
                .read_variant_signature()
                .unwrap();
            match sig {
                "s" => {
                    assert_eq!(key, "token");
                    assert_eq!(
                        dict.read_str()
                            .unwrap(),
                        "abc123"
                    );
                }
                "u" => {
                    assert_eq!(key, "count");
                    assert_eq!(
                        dict.read_u32()
                            .unwrap(),
                        7
                    );
                }
                other => panic!("unexpected variant signature {other}"),
            }
            seen += 1;
        }
        assert_eq!(seen, 2);
    }
}
