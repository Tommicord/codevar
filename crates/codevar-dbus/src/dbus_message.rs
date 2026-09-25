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

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::dbus_error::{DbusError, DbusResult};
use crate::dbus_marshal::{ByteOrder, DbusReader, DbusWriter, MAX_ARRAY_LEN};
use crate::dbus_names::{
    is_valid_bus_name, is_valid_error_name, is_valid_interface_name, is_valid_member,
    validate_object_path,
};
use crate::dbus_signature::{
    SignatureIter, type_alignment, validate_signature, validate_single_type,
};

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

/// Type of a D-Bus message.
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
        self.writer.position()
    }

    /// Consumes the builder, returning the bytes and the signature.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u8>, String) {
        (self.writer.into_bytes(), self.signature)
    }

    #[inline]
    fn record(&mut self, code: u8) {
        if self.recording {
            self.signature.push(char::from(code));
        }
    }

    /// Appends a `BYTE` value.
    pub fn write_u8(&mut self, value: u8) -> DbusResult<()> {
        self.record(b'y');
        self.writer.write_u8(value);
        Ok(())
    }

    /// Appends a `UINT16` value.
    pub fn write_u16(&mut self, value: u16) -> DbusResult<()> {
        self.record(b'q');
        self.writer.write_u16(value);
        Ok(())
    }

    /// Appends a `UINT32` value.
    pub fn write_u32(&mut self, value: u32) -> DbusResult<()> {
        self.record(b'u');
        self.writer.write_u32(value);
        Ok(())
    }

    /// Appends an `INT16` value.
    pub fn write_i16(&mut self, value: i16) -> DbusResult<()> {
        self.record(b'n');
        self.writer.write_i16(value);
        Ok(())
    }

    /// Appends an `INT32` value.
    pub fn write_i32(&mut self, value: i32) -> DbusResult<()> {
        self.record(b'i');
        self.writer.write_i32(value);
        Ok(())
    }

    /// Appends a `UINT64` value.
    pub fn write_u64(&mut self, value: u64) -> DbusResult<()> {
        self.record(b't');
        self.writer.write_u64(value);
        Ok(())
    }

    /// Appends an `INT64` value.
    pub fn write_i64(&mut self, value: i64) -> DbusResult<()> {
        self.record(b'x');
        self.writer.write_i64(value);
        Ok(())
    }

    /// Appends a `BOOLEAN` value.
    pub fn write_bool(&mut self, value: bool) -> DbusResult<()> {
        self.record(b'b');
        self.writer.write_bool(value);
        Ok(())
    }

    /// Appends a `DOUBLE` value.
    pub fn write_f64(&mut self, value: f64) -> DbusResult<()> {
        self.record(b'd');
        self.writer.write_f64(value);
        Ok(())
    }

    /// Appends a `STRING` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] when `value` contains an
    /// embedded nul byte.
    pub fn write_str(&mut self, value: &str) -> DbusResult<()> {
        self.writer.write_str(value)?;
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
        self.writer.write_object_path(path)?;
        self.record(b'o');
        Ok(())
    }

    /// Appends a `SIGNATURE` value.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidSignature`] when `sig` is invalid.
    pub fn write_signature(&mut self, sig: &str) -> DbusResult<()> {
        self.writer.write_signature(sig)?;
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
        validate_single_type(element_sig)?;
        let code = element_sig.as_bytes()[0];
        let alignment = type_alignment(code).ok_or_else(|| {
            DbusError::invalid_signature(alloc::format!("invalid type code: {code}"))
        })?;
        if self.recording {
            self.signature.push('a');
            self.signature.push_str(element_sig);
        }
        self.writer.align(8);
        let length_pos = self.writer.position();
        self.writer.write_u32(0);
        self.writer.align(alignment);
        let start = self.writer.position();
        let was_recording = core::mem::replace(&mut self.recording, false);
        let result = body(self);
        self.recording = was_recording;
        result?;
        let length = self.writer.position().saturating_sub(start);
        if length > MAX_ARRAY_LEN {
            return Err(DbusError::invalid_message(alloc::format!(
                "array of {length} bytes exceeds the limit of {MAX_ARRAY_LEN}"
            )));
        }
        self.writer.patch_u32(length_pos, length as u32)
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
            return Err(DbusError::invalid_signature(
                "struct needs at least one field",
            ));
        }
        validate_signature(fields_sig)?;
        if self.recording {
            self.signature.push('(');
            self.signature.push_str(fields_sig);
            self.signature.push(')');
        }
        self.writer.align(8);
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
            self.signature.push('v');
        }
        self.writer.write_signature(sig)?;
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
            body: Vec::new(),
        }
    }

    /// Creates a method call message.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidName`] when any of the names is
    /// invalid.
    pub fn method_call(
        destination: &str,
        path: &str,
        interface: &str,
        member: &str,
    ) -> DbusResult<Self> {
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
        matches!(self.kind, MessageKind::MethodCall)
            && self.flags & FLAG_NO_REPLY_EXPECTED == 0
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
        self.path.as_deref()
    }

    /// Returns the interface, when present.
    #[inline]
    #[must_use]
    pub fn interface(&self) -> Option<&str> {
        self.interface.as_deref()
    }

    /// Returns the member (method or signal name), when present.
    #[inline]
    #[must_use]
    pub fn member(&self) -> Option<&str> {
        self.member.as_deref()
    }

    /// Returns the error name, when present.
    #[inline]
    #[must_use]
    pub fn error_name(&self) -> Option<&str> {
        self.error_name.as_deref()
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
        self.destination.as_deref()
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
        self.sender.as_deref()
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

    fn validate_for_encode(&self) -> DbusResult<()> {
        if self.serial == 0 {
            return Err(DbusError::invalid_state("message serial must not be zero"));
        }
        if self.unix_fds > 0 {
            return Err(DbusError::unsupported(
                "file descriptor passing is not supported",
            ));
        }
        if self.signature.as_bytes().contains(&b'h') {
            return Err(DbusError::unsupported(
                "file descriptor arguments are not supported",
            ));
        }
        validate_signature(&self.signature)?;
        match self.kind {
            MessageKind::MethodCall => {
                if self.member.is_none() {
                    return Err(DbusError::invalid_state("method call needs a member"));
                }
            }
            MessageKind::Signal => {
                if self.path.is_none() || self.member.is_none() {
                    return Err(DbusError::invalid_state(
                        "signal needs a path and a member",
                    ));
                }
            }
            MessageKind::MethodReturn => {
                if self.reply_serial.is_none() {
                    return Err(DbusError::invalid_state("reply needs a reply serial"));
                }
            }
            MessageKind::Error => {
                if self.reply_serial.is_none() || self.error_name.is_none() {
                    return Err(DbusError::invalid_state(
                        "error reply needs a reply serial and an error name",
                    ));
                }
            }
        }
        if self.body.is_empty() != self.signature.is_empty() {
            return Err(DbusError::invalid_message(
                "body length does not match the signature",
            ));
        }
        if self.body.len() > MAX_MESSAGE_LEN {
            return Err(DbusError::MessageTooBig(self.body.len()));
        }
        Ok(())
    }

    /// Encodes the message into wire format.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidState`] when a required header
    /// field of the message kind is missing, [`DbusError::InvalidName`]
    /// or [`DbusError::InvalidSignature`] for invalid fields,
    /// [`DbusError::Unsupported`] for file descriptors and
    /// [`DbusError::MessageTooBig`] when the message exceeds
    /// [`MAX_MESSAGE_LEN`].
    pub fn encode(&self) -> DbusResult<Vec<u8>> {
        self.validate_for_encode()?;
        let body_len = self.body.len();
        let mut writer = DbusWriter::with_capacity(self.order, body_len + 64);
        writer.write_u8(self.order.marker());
        writer.write_u8(self.kind.as_u8());
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
            write_field(&mut writer, FIELD_MEMBER, "s", |writer| {
                writer.write_str(&member)
            })?;
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
            write_field(&mut writer, FIELD_SENDER, "s", |writer| {
                writer.write_str(&sender)
            })?;
        }
        if !self.signature.is_empty() {
            let signature = self.signature.clone();
            write_field(&mut writer, FIELD_SIGNATURE, "g", |writer| {
                writer.write_signature(&signature)
            })?;
        }

        let fields_len = writer.position().saturating_sub(fields_start);
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

    /// Decodes a complete message from `bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] for malformed headers,
    /// missing required fields or inconsistent bodies, and
    /// [`DbusError::Unsupported`] for file descriptor passing.
    pub fn decode(bytes: &[u8]) -> DbusResult<Self> {
        let marker = bytes.first().copied().ok_or_else(truncated)?;
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
            return Err(DbusError::invalid_message(
                "message serial must not be zero",
            ));
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
                    message.path = Some(fields.read_object_path()?.to_string());
                }
                FIELD_INTERFACE => {
                    expect_signature(sig, b's', code)?;
                    let value = fields.read_str()?.to_string();
                    if !is_valid_interface_name(&value) {
                        return Err(DbusError::invalid_name(alloc::format!(
                            "invalid interface in header: {value}"
                        )));
                    }
                    message.interface = Some(value);
                }
                FIELD_MEMBER => {
                    expect_signature(sig, b's', code)?;
                    let value = fields.read_str()?.to_string();
                    if !is_valid_member(&value) {
                        return Err(DbusError::invalid_name(alloc::format!(
                            "invalid member in header: {value}"
                        )));
                    }
                    message.member = Some(value);
                }
                FIELD_ERROR_NAME => {
                    expect_signature(sig, b's', code)?;
                    let value = fields.read_str()?.to_string();
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
                        return Err(DbusError::invalid_message(
                            "reply serial must not be zero",
                        ));
                    }
                    message.reply_serial = Some(value);
                }
                FIELD_DESTINATION => {
                    expect_signature(sig, b's', code)?;
                    let value = fields.read_str()?.to_string();
                    if !is_valid_bus_name(&value) {
                        return Err(DbusError::invalid_name(alloc::format!(
                            "invalid destination in header: {value}"
                        )));
                    }
                    message.destination = Some(value);
                }
                FIELD_SENDER => {
                    expect_signature(sig, b's', code)?;
                    let value = fields.read_str()?.to_string();
                    if !is_valid_bus_name(&value) {
                        return Err(DbusError::invalid_name(alloc::format!(
                            "invalid sender in header: {value}"
                        )));
                    }
                    message.sender = Some(value);
                }
                FIELD_SIGNATURE => {
                    expect_signature(sig, b'g', code)?;
                    message.signature = fields.read_signature()?.to_string();
                }
                FIELD_UNIX_FDS => {
                    expect_signature(sig, b'u', code)?;
                    message.unix_fds = fields.read_u32()?;
                }
                _ => skip_value(&mut fields, sig)?,
            }
        }

        reader.align(8)?;
        message.body = reader.read_bytes(body_len)?.to_vec();
        if !reader.is_empty() {
            return Err(DbusError::invalid_message(
                "trailing bytes after the message",
            ));
        }

        validate_decode(&message)?;
        Ok(message)
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

fn write_field<F>(
    writer: &mut DbusWriter,
    code: u8,
    sig: &str,
    value: F,
) -> DbusResult<()>
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
            if message.member.is_none() {
                return Err(DbusError::invalid_message("method call without a member"));
            }
        }
        MessageKind::Signal => {
            if message.path.is_none() || message.member.is_none() {
                return Err(DbusError::invalid_message(
                    "signal without a path or member",
                ));
            }
        }
        MessageKind::MethodReturn => {
            if message.reply_serial.is_none() {
                return Err(DbusError::invalid_message("reply without a reply serial"));
            }
        }
        MessageKind::Error => {
            if message.reply_serial.is_none() || message.error_name.is_none() {
                return Err(DbusError::invalid_message(
                    "error reply without a reply serial or error name",
                ));
            }
        }
    }
    if message.body.is_empty() != message.signature.is_empty() {
        return Err(DbusError::invalid_message(
            "body length does not match the signature",
        ));
    }
    if message.unix_fds > 0 {
        return Err(DbusError::unsupported(
            "file descriptor passing is not supported",
        ));
    }
    if message.signature.as_bytes().contains(&b'h') {
        return Err(DbusError::unsupported(
            "file descriptor arguments are not supported",
        ));
    }
    Ok(())
}

/// Skips one value of type `sig` without decoding it.
fn skip_value(reader: &mut DbusReader<'_>, sig: &str) -> DbusResult<()> {
    let code = sig.as_bytes().first().copied().ok_or_else(|| {
        DbusError::invalid_signature("empty signature while skipping a value")
    })?;
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
            let element_code = element.as_bytes().first().copied().ok_or_else(|| {
                DbusError::invalid_signature("array without an element type")
            })?;
            let alignment = type_alignment(element_code).ok_or_else(|| {
                DbusError::invalid_signature(alloc::format!(
                    "invalid type code: {element_code}"
                ))
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
                .ok_or_else(|| {
                    DbusError::invalid_signature("malformed struct signature")
                })?;
            for field in SignatureIter::new(inner) {
                skip_value(reader, field)?;
            }
        }
        b'{' => {
            reader.read_struct()?;
            let inner = sig
                .strip_prefix('{')
                .and_then(|rest| rest.strip_suffix('}'))
                .ok_or_else(|| {
                    DbusError::invalid_signature("malformed dict signature")
                })?;
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
#[derive(Debug, Clone, Default)]
pub struct DbusMessageStream {
    buffer: Vec<u8>,
}

impl DbusMessageStream {
    /// Creates an empty stream decoder.
    #[must_use]
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Appends bytes read from the transport.
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Returns the number of buffered bytes.
    #[inline]
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Drops all buffered bytes.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Returns the next complete message, `Ok(None)` while incomplete.
    ///
    /// Unknown message types are skipped, as required by the
    /// specification.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidMessage`] for malformed framing and
    /// [`DbusError::MessageTooBig`] for oversized messages without
    /// consuming the offending bytes, so the caller can drop the
    /// connection.
    pub fn next_message(&mut self) -> DbusResult<Option<DbusMessage>> {
        loop {
            if self.buffer.len() < FIXED_HEADER_LEN {
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
                return Err(DbusError::invalid_message(
                    "message serial must not be zero",
                ));
            }
            if body_len > MAX_MESSAGE_LEN {
                return Err(DbusError::MessageTooBig(body_len));
            }
            if self.buffer.len() < 20 {
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
            if self.buffer.len() < total {
                return Ok(None);
            }
            let decoded = if MessageKind::from_u8(self.buffer[1]).is_some() {
                Some(DbusMessage::decode(&self.buffer[..total]))
            } else {
                None
            };
            self.buffer.drain(..total);
            if let Some(result) = decoded {
                return result.map(Some);
            }
        }
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
        message.set_serial(1).unwrap();
        let bytes = message.encode().unwrap();
        let expected = vec![
            // Fixed header: 'l', METHOD_CALL, flags 0, version 1.
            0x6c, 0x01, 0x00, 0x01, // body length 0.
            0x00, 0x00, 0x00, 0x00, // serial 1.
            0x01, 0x00, 0x00, 0x00, // padding to the header field array.
            0x00, 0x00, 0x00, 0x00, // header field array length: 109.
            0x6d, 0x00, 0x00, 0x00, // padding to the first struct.
            0x00, 0x00, 0x00, 0x00, // PATH field.
            0x01, 0x01, 0x6f, 0x00, 0x15, 0x00, 0x00,
            0x00, // "/org/freedesktop/DBus"
            0x2f, 0x6f, 0x72, 0x67, 0x2f, 0x66, 0x72, 0x65, 0x65, 0x64, 0x65, 0x73, 0x6b,
            0x74, 0x6f, 0x70, 0x2f, 0x44, 0x42, 0x75, 0x73,
            0x00, // padding to the next struct.
            0x00, 0x00, // INTERFACE field.
            0x02, 0x01, 0x73, 0x00, 0x14, 0x00, 0x00,
            0x00, // "org.freedesktop.DBus"
            0x6f, 0x72, 0x67, 0x2e, 0x66, 0x72, 0x65, 0x65, 0x64, 0x65, 0x73, 0x6b, 0x74,
            0x6f, 0x70, 0x2e, 0x44, 0x42, 0x75, 0x73,
            0x00, // padding to the next struct.
            0x00, 0x00, 0x00, // MEMBER field.
            0x03, 0x01, 0x73, 0x00, 0x05, 0x00, 0x00, 0x00, // "Hello"
            0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x00, // padding to the next struct.
            0x00, 0x00, // DESTINATION field.
            0x06, 0x01, 0x73, 0x00, 0x14, 0x00, 0x00,
            0x00, // "org.freedesktop.DBus"
            0x6f, 0x72, 0x67, 0x2e, 0x66, 0x72, 0x65, 0x65, 0x64, 0x65, 0x73, 0x6b, 0x74,
            0x6f, 0x70, 0x2e, 0x44, 0x42, 0x75, 0x73,
            0x00, // padding to the 8 byte header boundary.
            0x00, 0x00, 0x00,
        ];
        assert_eq!(bytes, expected);
        assert_eq!(bytes.len() % 8, 0);
    }

    #[test]
    fn round_trips_every_message_kind() {
        let mut call = hello_call();
        call.set_serial(7).unwrap();
        call.set_no_reply_expected();
        call.build_body(|body| {
            body.write_str("codevar")?;
            body.write_u32(42)?;
            Ok(())
        })
        .unwrap();
        let bytes = call.encode().unwrap();
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
        assert_eq!(reader.read_str().unwrap(), "codevar");
        assert_eq!(reader.read_u32().unwrap(), 42);

        let mut reply = DbusMessage::method_return(7);
        reply.set_serial(9).unwrap();
        reply.build_body(|body| body.write_bool(true)).unwrap();
        let decoded = DbusMessage::decode(&reply.encode().unwrap()).unwrap();
        assert_eq!(decoded.kind(), MessageKind::MethodReturn);
        assert_eq!(decoded.reply_serial(), Some(7));
        assert_eq!(decoded.signature(), "b");

        let mut error =
            DbusMessage::error(7, "org.freedesktop.DBus.Error.Failed").unwrap();
        error.set_serial(10).unwrap();
        error.build_body(|body| body.write_str("boom")).unwrap();
        let decoded = DbusMessage::decode(&error.encode().unwrap()).unwrap();
        assert_eq!(decoded.kind(), MessageKind::Error);
        assert_eq!(
            decoded.error_name(),
            Some("org.freedesktop.DBus.Error.Failed")
        );
        assert_eq!(decoded.reply_serial(), Some(7));

        let mut signal =
            DbusMessage::signal("/org/example", "org.example.Interface", "Changed")
                .unwrap();
        signal.set_serial(11).unwrap();
        let decoded = DbusMessage::decode(&signal.encode().unwrap()).unwrap();
        assert_eq!(decoded.kind(), MessageKind::Signal);
        assert_eq!(decoded.path(), Some("/org/example"));
        assert_eq!(decoded.member(), Some("Changed"));
        assert_eq!(decoded.signature(), "");
        assert!(decoded.body().is_empty());
    }

    #[test]
    fn round_trips_nested_body_values() {
        let mut message = DbusMessage::method_call(
            "org.example.Service",
            "/org/example",
            "org.example.Iface",
            "Send",
        )
        .unwrap();
        message.set_serial(3).unwrap();
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

        let bytes = message.encode().unwrap();
        let decoded = DbusMessage::decode(&bytes).unwrap();
        assert_eq!(decoded.signature(), message.signature());
        let mut reader = decoded.body_reader();
        assert_eq!(reader.read_i64().unwrap(), -5);
        let mut array = reader.read_array(4).unwrap();
        assert_eq!(array.read_str().unwrap(), "alpha");
        assert_eq!(array.read_str().unwrap(), "beta");
        reader.read_struct().unwrap();
        assert_eq!(reader.read_u8().unwrap(), 9);
        assert_eq!(reader.read_u32().unwrap(), 1000);
        assert_eq!(reader.read_variant_signature().unwrap(), "d");
        assert_eq!(reader.read_f64().unwrap(), 2.5);
    }

    #[test]
    fn rejects_missing_required_fields() {
        let mut call = DbusMessage::method_call("a.b", "/x", "a.b.C", "Go").unwrap();
        call.set_serial(1).unwrap();
        call.member = None;
        assert!(call.encode().is_err());

        let mut reply = DbusMessage::method_return(5);
        reply.set_serial(2).unwrap();
        reply.reply_serial = None;
        assert!(reply.encode().is_err());

        let mut signal = DbusMessage::signal("/x", "a.b.C", "Ping").unwrap();
        signal.set_serial(3).unwrap();
        signal.path = None;
        assert!(signal.encode().is_err());

        let call = hello_call();
        assert!(call.encode().is_err());
        assert_eq!(
            call.encode(),
            Err(DbusError::InvalidState(String::from(
                "message serial must not be zero"
            )))
        );
    }

    #[test]
    fn rejects_file_descriptors() {
        let mut call = hello_call();
        call.set_serial(1).unwrap();
        call.signature = String::from("h");
        call.body = vec![1, 0, 0, 0];
        assert!(matches!(call.encode(), Err(DbusError::Unsupported(_))));

        let mut message = hello_call();
        message.set_serial(1).unwrap();
        message.unix_fds = 1;
        assert!(matches!(message.encode(), Err(DbusError::Unsupported(_))));
    }

    #[test]
    fn rejects_body_signature_mismatch() {
        let mut call = hello_call();
        call.set_serial(1).unwrap();
        call.signature = String::from("s");
        assert!(call.encode().is_err());
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
        writer.patch_u32(length_pos, fields_len as u32).unwrap();
        writer.align(8);
        writer.write_u32(0x1122_3344);
        let bytes = writer.into_bytes();
        assert_eq!(bytes[0], b'B');

        let decoded = DbusMessage::decode(&bytes).unwrap();
        assert_eq!(decoded.order(), ByteOrder::Big);
        assert_eq!(decoded.kind(), MessageKind::MethodReturn);
        assert_eq!(decoded.reply_serial(), Some(4));
        assert_eq!(decoded.signature(), "u");
        assert_eq!(decoded.body_reader().read_u32().unwrap(), 0x1122_3344);
    }

    #[test]
    fn stream_returns_none_until_complete() {
        let mut message = hello_call();
        message.set_serial(1).unwrap();
        let bytes = message.encode().unwrap();

        let mut stream = DbusMessageStream::new();
        stream.feed(&bytes[..5]);
        assert_eq!(stream.next_message().unwrap(), None);
        stream.feed(&bytes[5..19]);
        assert_eq!(stream.next_message().unwrap(), None);
        stream.feed(&bytes[19..]);
        let decoded = stream.next_message().unwrap();
        assert!(decoded.is_some());
        assert_eq!(stream.next_message().unwrap(), None);
        assert_eq!(stream.buffered(), 0);
    }

    #[test]
    fn stream_decodes_back_to_back_messages() {
        let mut first = hello_call();
        first.set_serial(1).unwrap();
        let mut second = DbusMessage::signal("/a", "b.C", "Ping").unwrap();
        second.set_serial(2).unwrap();
        let mut data = first.encode().unwrap();
        data.extend_from_slice(&second.encode().unwrap());

        let mut stream = DbusMessageStream::new();
        stream.feed(&data);
        let first_decoded = stream.next_message().unwrap().unwrap();
        assert_eq!(first_decoded.member(), Some("Hello"));
        let second_decoded = stream.next_message().unwrap().unwrap();
        assert_eq!(second_decoded.kind(), MessageKind::Signal);
        assert_eq!(stream.next_message().unwrap(), None);
    }

    #[test]
    fn stream_skips_unknown_message_types() {
        let mut message = hello_call();
        message.set_serial(1).unwrap();
        let mut bytes = message.encode().unwrap();
        bytes[1] = 9;
        let mut stream = DbusMessageStream::new();
        stream.feed(&bytes);
        assert_eq!(stream.next_message().unwrap(), None);
        assert_eq!(stream.buffered(), 0);
    }

    #[test]
    fn stream_rejects_broken_framing() {
        let mut stream = DbusMessageStream::new();
        stream.feed(&[0x78, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]);
        assert!(stream.next_message().is_err());

        let mut stream = DbusMessageStream::new();
        stream.feed(&[0x6c, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0]);
        assert!(stream.next_message().is_err());

        let mut stream = DbusMessageStream::new();
        stream.feed(&[0x6c, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(stream.next_message().is_err());

        // A body beyond the size limit is reported without consuming.
        let mut stream = DbusMessageStream::new();
        stream.feed(&[0x6c, 1, 0, 1, 0xff, 0xff, 0xff, 0x7f, 1, 0, 0, 0]);
        assert_eq!(
            stream.next_message(),
            Err(DbusError::MessageTooBig(0x7fff_ffff))
        );
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
        write_field(&mut writer, FIELD_MEMBER, "s", |writer| {
            writer.write_str("Hello")
        })
        .unwrap();
        write_field(&mut writer, FIELD_MEMBER, "s", |writer| {
            writer.write_str("Hello")
        })
        .unwrap();
        let fields_len = writer.position() - fields_start;
        writer.patch_u32(length_pos, fields_len as u32).unwrap();
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
}
