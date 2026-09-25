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

//! Protocol primitives shared by the Wayland client and server ports.
//!
//! The module mirrors `wayland-util.h` and the parts of `wayland-private.h`
//! that describe the wire format: message signatures, interface metadata,
//! argument values, the object map and the intrusive list utilities.

use core::fmt;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::wl_error::{WlError, WlResult};

/// Maximum size of a protocol message, including the 8 byte header.
pub const MAX_MESSAGE_SIZE: usize = 4096;

/// Maximum number of 32 bit words in a protocol message.
pub const MAX_MESSAGE_WORDS: usize = MAX_MESSAGE_SIZE / 4;

/// First id of the server side id space.
pub const SERVER_ID_START: u32 = 0xff00_0000;

/// Maximum number of objects tracked by a single [`WlMap`].
pub const MAP_MAX_OBJECTS: u32 = 0x00f0_0000;

/// Maximum number of arguments in a single message.
pub const MAX_CLOSURE_ARGS: usize = 20;

/// File descriptor type used by transports and the wire format.
pub type WlFd = i32;

/// The file descriptor is readable.
pub const EVENT_READABLE: u32 = 0x01;
/// The file descriptor is writable.
pub const EVENT_WRITABLE: u32 = 0x02;
/// The peer hung up.
pub const EVENT_HANGUP: u32 = 0x04;
/// An error occurred on the file descriptor.
pub const EVENT_ERROR: u32 = 0x08;

/// Set of event bits as returned by pollers and transports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct WlPollEvents(u32);

impl WlPollEvents {
    /// No events.
    pub const EMPTY: Self = Self(0);
    /// The file descriptor is readable.
    pub const READABLE: Self = Self(EVENT_READABLE);
    /// The file descriptor is writable.
    pub const WRITABLE: Self = Self(EVENT_WRITABLE);
    /// The peer hung up.
    pub const HANGUP: Self = Self(EVENT_HANGUP);
    /// An error occurred on the file descriptor.
    pub const ERROR: Self = Self(EVENT_ERROR);
    /// All event bits.
    pub const ALL: Self = Self(EVENT_READABLE | EVENT_WRITABLE | EVENT_HANGUP | EVENT_ERROR);

    /// Creates an event set from raw bits.
    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw bits of the event set.
    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns `true` when every bit of `other` is set.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns `true` when no bit is set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Adds the bits of `other`.
    #[inline]
    pub const fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Removes the bits of `other`.
    #[inline]
    pub const fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    /// Returns the union of both event sets.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the intersection of both event sets.
    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
}

impl core::ops::BitOr for WlPollEvents {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl core::ops::BitOrAssign for WlPollEvents {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.insert(rhs);
    }
}

impl core::ops::BitAnd for WlPollEvents {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

impl core::ops::BitAndAssign for WlPollEvents {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.intersection(rhs);
    }
}

/// `wl_display.sync` request opcode.
pub const DISPLAY_SYNC: u32 = 0;
/// `wl_display.get_registry` request opcode.
pub const DISPLAY_GET_REGISTRY: u32 = 1;
/// `wl_display.error` event opcode.
pub const DISPLAY_ERROR: u32 = 0;
/// `wl_display.delete_id` event opcode.
pub const DISPLAY_DELETE_ID: u32 = 1;
/// `wl_registry.bind` request opcode.
pub const REGISTRY_BIND: u32 = 0;
/// `wl_registry.global` event opcode.
pub const REGISTRY_GLOBAL: u32 = 0;
/// `wl_registry.global_remove` event opcode.
pub const REGISTRY_GLOBAL_REMOVE: u32 = 1;
/// `wl_callback.done` event opcode.
pub const CALLBACK_DONE: u32 = 0;

/// Fixed point 24.8 number as used by the wire format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WlFixed(i32);

impl WlFixed {
    /// Creates a fixed point number from its raw 24.8 representation.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }

    /// Returns the raw 24.8 representation.
    #[inline]
    pub const fn to_raw(self) -> i32 {
        self.0
    }

    /// Converts a floating point number to 24.8 fixed point.
    ///
    /// Fractions are rounded half away from zero, matching the C
    /// `wl_fixed_from_double` conversion.
    #[inline]
    #[must_use]
    pub fn from_double(value: f64) -> Self {
        let scaled = value * 256.0;
        let truncated = scaled as i32;
        let fraction = scaled - f64::from(truncated);
        let rounded = if fraction >= 0.5 {
            truncated.saturating_add(1)
        } else if fraction <= -0.5 {
            truncated.saturating_sub(1)
        } else {
            truncated
        };
        Self(rounded)
    }

    /// Converts a 24.8 fixed point number to floating point.
    #[inline]
    #[must_use]
    pub fn to_double(self) -> f64 {
        f64::from(self.0) / 256.0
    }

    /// Converts an integer to 24.8 fixed point.
    #[inline]
    #[must_use]
    pub const fn from_int(value: i32) -> Self {
        Self(value.saturating_mul(256))
    }

    /// Returns the integer component of the fixed point number.
    #[inline]
    #[must_use]
    pub const fn to_int(self) -> i32 {
        self.0 / 256
    }
}

impl From<f64> for WlFixed {
    #[inline]
    fn from(value: f64) -> Self {
        Self::from_double(value)
    }
}

impl From<WlFixed> for f64 {
    #[inline]
    fn from(value: WlFixed) -> Self {
        value.to_double()
    }
}

impl From<i32> for WlFixed {
    #[inline]
    fn from(value: i32) -> Self {
        Self::from_int(value)
    }
}

impl From<WlFixed> for i32 {
    #[inline]
    fn from(value: WlFixed) -> Self {
        value.to_int()
    }
}

impl fmt::Display for WlFixed {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_double())
    }
}

/// Protocol argument type symbols as used in message signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WlArgType {
    /// `i` — 32 bit signed integer.
    Int = b'i',
    /// `u` — 32 bit unsigned integer.
    Uint = b'u',
    /// `f` — 24.8 fixed point number.
    Fixed = b'f',
    /// `s` — nullable string.
    String = b's',
    /// `o` — object id.
    Object = b'o',
    /// `n` — new object id.
    NewId = b'n',
    /// `a` — opaque byte array.
    Array = b'a',
    /// `h` — file descriptor.
    Fd = b'h',
}

impl WlArgType {
    /// Returns the type matching a signature symbol.
    #[inline]
    #[must_use]
    pub const fn from_u8(symbol: u8) -> Option<Self> {
        match symbol {
            b'i' => Some(Self::Int),
            b'u' => Some(Self::Uint),
            b'f' => Some(Self::Fixed),
            b's' => Some(Self::String),
            b'o' => Some(Self::Object),
            b'n' => Some(Self::NewId),
            b'a' => Some(Self::Array),
            b'h' => Some(Self::Fd),
            _ => None,
        }
    }

    /// Returns the signature symbol of the type.
    #[inline]
    pub const fn as_char(self) -> char {
        self as u8 as char
    }
}

impl fmt::Display for WlArgType {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

/// Decoded details of a single message argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WlArgDetails {
    /// Argument type.
    pub ty: WlArgType,
    /// Whether the signature marked the argument as nullable with `?`.
    pub nullable: bool,
}

impl WlArgDetails {
    /// Creates argument details.
    #[inline]
    #[must_use]
    pub const fn new(ty: WlArgType, nullable: bool) -> Self {
        Self { ty, nullable }
    }
}

/// Parses the argument starting at `pos` in `signature`.
///
/// Leading digits (the `since` version prefix) and `?` markers are consumed
/// while scanning. Returns the argument details and the position just after
/// the parsed argument, or `None` when the signature is exhausted.
#[must_use]
pub fn get_next_argument(signature: &str, pos: usize) -> Option<(WlArgDetails, usize)> {
    let bytes = signature.as_bytes();
    let mut nullable = false;
    let mut index = pos;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'?' {
            nullable = true;
            index += 1;
            continue;
        }
        if byte.is_ascii_digit() {
            index += 1;
            continue;
        }
        if let Some(ty) = WlArgType::from_u8(byte) {
            return Some((WlArgDetails::new(ty, nullable), index + 1));
        }
        index += 1;
    }
    None
}

/// Counts the arguments described by a message signature.
#[must_use]
pub fn arg_count(signature: &str) -> usize {
    let mut count = 0;
    let mut pos = 0;
    while let Some((_, next)) = get_next_argument(signature, pos) {
        count += 1;
        pos = next;
    }
    count
}

/// Returns the `since` version prefix of a signature.
///
/// A signature without a version prefix reports `1`, matching libwayland.
#[must_use]
pub fn message_since(signature: &str) -> u32 {
    let mut value = 0u32;
    for byte in signature.bytes() {
        if !byte.is_ascii_digit() {
            break;
        }
        value = value
            .saturating_mul(10)
            .saturating_add(u32::from(byte - b'0'));
    }
    if value == 0 { 1 } else { value }
}

/// Counts the file descriptor arguments of a signature.
#[must_use]
pub fn count_fds(signature: &str) -> usize {
    let mut count = 0;
    let mut pos = 0;
    while let Some((details, next)) = get_next_argument(signature, pos) {
        if details.ty == WlArgType::Fd {
            count += 1;
        }
        pos = next;
    }
    count
}

/// Counts the array arguments of a signature.
#[must_use]
pub fn count_arrays(signature: &str) -> usize {
    let mut count = 0;
    let mut pos = 0;
    while let Some((details, next)) = get_next_argument(signature, pos) {
        if details.ty == WlArgType::Array {
            count += 1;
        }
        pos = next;
    }
    count
}

/// A single expanded argument of a [`WlMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WlMessageArg {
    /// Position of the argument in the expanded signature.
    pub index: usize,
    /// Type and nullability of the argument.
    pub details: WlArgDetails,
    /// Interface of object and new_id arguments, `None` otherwise.
    pub interface: Option<&'static WlInterface>,
}

/// Iterator over the expanded arguments of a [`WlMessage`].
#[derive(Debug, Clone, Copy)]
pub struct WlMessageArgs {
    signature: &'static str,
    pos: usize,
    index: usize,
    types: &'static [Option<&'static WlInterface>],
}

impl Iterator for WlMessageArgs {
    type Item = WlMessageArg;

    fn next(&mut self) -> Option<Self::Item> {
        let (details, next) = get_next_argument(self.signature, self.pos)?;
        let item = WlMessageArg {
            index: self.index,
            details,
            interface: self
                .types
                .get(self.index)
                .copied()
                .flatten(),
        };
        self.pos = next;
        self.index += 1;
        Some(item)
    }
}

/// Signature of a single request or event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WlMessage {
    /// Message name as used on the wire.
    pub name: &'static str,
    /// Message signature, see the module documentation of `wayland-util.h`.
    pub signature: &'static str,
    /// Interfaces of object and new_id arguments, aligned to expanded slots.
    pub types: &'static [Option<&'static WlInterface>],
}

impl WlMessage {
    /// Creates a message description.
    #[inline]
    pub const fn new(
        name: &'static str,
        signature: &'static str,
        types: &'static [Option<&'static WlInterface>],
    ) -> Self {
        Self {
            name,
            signature,
            types,
        }
    }

    /// Returns the number of arguments of the message.
    #[inline]
    #[must_use]
    pub fn arg_count(&self) -> usize {
        arg_count(self.signature)
    }

    /// Returns the protocol version the message was introduced in.
    #[inline]
    #[must_use]
    pub fn since(&self) -> u32 {
        message_since(self.signature)
    }

    /// Returns the number of file descriptor arguments.
    #[inline]
    #[must_use]
    pub fn fd_count(&self) -> usize {
        count_fds(self.signature)
    }

    /// Returns the number of array arguments.
    #[inline]
    #[must_use]
    pub fn array_count(&self) -> usize {
        count_arrays(self.signature)
    }

    /// Iterates over the expanded arguments of the message.
    #[inline]
    #[must_use]
    pub fn args(&self) -> WlMessageArgs {
        WlMessageArgs {
            signature: self.signature,
            pos: 0,
            index: 0,
            types: self.types,
        }
    }

    /// Returns the interface of the argument at `index`.
    #[inline]
    #[must_use]
    pub fn type_at(&self, index: usize) -> Option<&'static WlInterface> {
        self.types
            .get(index)
            .copied()
            .flatten()
    }
}

/// Description of a protocol object type.
#[derive(Debug, PartialEq, Eq)]
pub struct WlInterface {
    /// Interface name as used on the wire.
    pub name: &'static str,
    /// Maximum version of the interface.
    pub version: u32,
    /// Request signatures, indexed by opcode.
    pub requests: &'static [WlMessage],
    /// Event signatures, indexed by opcode.
    pub events: &'static [WlMessage],
}

impl WlInterface {
    /// Creates an interface description.
    #[inline]
    pub const fn new(
        name: &'static str,
        version: u32,
        requests: &'static [WlMessage],
        events: &'static [WlMessage],
    ) -> Self {
        Self {
            name,
            version,
            requests,
            events,
        }
    }

    /// Looks up a request by name, returning its opcode and signature.
    #[must_use]
    pub fn request(&self, name: &str) -> Option<(u32, &WlMessage)> {
        self.requests
            .iter()
            .enumerate()
            .find(|(_, message)| message.name == name)
            .map(|(opcode, message)| (opcode as u32, message))
    }

    /// Looks up an event by name, returning its opcode and signature.
    #[must_use]
    pub fn event(&self, name: &str) -> Option<(u32, &WlMessage)> {
        self.events
            .iter()
            .enumerate()
            .find(|(_, message)| message.name == name)
            .map(|(opcode, message)| (opcode as u32, message))
    }

    /// Returns the request registered at `opcode`.
    #[inline]
    #[must_use]
    pub fn request_at(&self, opcode: u32) -> Option<&WlMessage> {
        self.requests
            .get(opcode as usize)
    }

    /// Returns the event registered at `opcode`.
    #[inline]
    #[must_use]
    pub fn event_at(&self, opcode: u32) -> Option<&WlMessage> {
        self.events
            .get(opcode as usize)
    }

    /// Returns `true` when both descriptions refer to the same interface.
    #[must_use]
    pub fn equal(&self, other: &Self) -> bool {
        core::ptr::eq(self, other) || self.name == other.name
    }
}

impl fmt::Display for WlInterface {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

/// `wl_display` core interface.
pub static DISPLAY_INTERFACE: WlInterface = WlInterface {
    name: "wl_display",
    version: 1,
    requests: &[
        WlMessage::new("sync", "n", &[Some(&CALLBACK_INTERFACE)]),
        WlMessage::new("get_registry", "n", &[Some(&REGISTRY_INTERFACE)]),
    ],
    events: &[
        WlMessage::new("error", "ous", &[None, None, None]),
        WlMessage::new("delete_id", "u", &[None]),
    ],
};

/// `wl_registry` core interface.
pub static REGISTRY_INTERFACE: WlInterface = WlInterface {
    name: "wl_registry",
    version: 1,
    requests: &[WlMessage::new("bind", "usun", &[None, None, None, None])],
    events: &[
        WlMessage::new("global", "usu", &[None, None, None]),
        WlMessage::new("global_remove", "u", &[None]),
    ],
};

/// `wl_callback` core interface.
pub static CALLBACK_INTERFACE: WlInterface = WlInterface {
    name: "wl_callback",
    version: 1,
    requests: &[],
    events: &[WlMessage::new("done", "u", &[None])],
};

/// Protocol error codes of `enum wl_display_error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum WlDisplayError {
    /// Object id sent by the client is unknown.
    InvalidObject = 0,
    /// Request is not implemented by the object.
    InvalidMethod = 1,
    /// Out of memory while creating a protocol object.
    NoMemory = 2,
    /// Implementation of an object failed.
    Implementation = 3,
}

impl WlDisplayError {
    /// Returns the wire code of the error.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a wire code, returning `None` for unknown codes.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::InvalidObject),
            1 => Some(Self::InvalidMethod),
            2 => Some(Self::NoMemory),
            3 => Some(Self::Implementation),
            _ => None,
        }
    }

    /// Returns the symbolic name of the error.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvalidObject => "invalid_object",
            Self::InvalidMethod => "invalid_method",
            Self::NoMemory => "no_memory",
            Self::Implementation => "implementation",
        }
    }
}

impl fmt::Display for WlDisplayError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Growable byte array used by `array` arguments.
///
/// Like `wl_array`, the value only grows until it is released; it exists so
/// that array arguments keep the shape of the C API.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WlArray(Vec<u8>);

impl WlArray {
    /// Creates an empty array.
    #[must_use]
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Creates an array with the contents of `bytes`.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(bytes.to_vec())
    }

    /// Returns the contents of the array.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Returns the number of bytes in the array.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.0
            .len()
    }

    /// Returns `true` when the array holds no bytes.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0
            .is_empty()
    }

    /// Appends a single byte.
    #[inline]
    pub fn push(&mut self, byte: u8) {
        self.0
            .push(byte);
    }

    /// Appends the contents of `bytes`.
    #[inline]
    pub fn extend_from_slice(&mut self, bytes: &[u8]) {
        self.0
            .extend_from_slice(bytes);
    }

    /// Removes all bytes from the array.
    #[inline]
    pub fn clear(&mut self) {
        self.0
            .clear();
    }

    /// Consumes the array, returning its bytes.
    #[inline]
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

impl From<Vec<u8>> for WlArray {
    #[inline]
    fn from(value: Vec<u8>) -> Self {
        Self(value)
    }
}

impl From<&[u8]> for WlArray {
    #[inline]
    fn from(value: &[u8]) -> Self {
        Self::from_bytes(value)
    }
}

impl From<WlArray> for Vec<u8> {
    #[inline]
    fn from(value: WlArray) -> Self {
        value.0
    }
}

/// A single decoded or to-be-encoded message argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WlArgument {
    /// `i` — 32 bit signed integer.
    Int(i32),
    /// `u` — 32 bit unsigned integer.
    Uint(u32),
    /// `f` — 24.8 fixed point number.
    Fixed(WlFixed),
    /// `s` — string, `None` encodes a null string.
    Str(Option<String>),
    /// `o` — object id, `0` encodes a null object.
    Object(u32),
    /// `n` — new object id.
    NewId(u32),
    /// `a` — byte array, `None` encodes a null array.
    Array(Option<WlArray>),
    /// `h` — file descriptor.
    Fd(WlFd),
}

impl WlArgument {
    /// Returns the argument type of the value.
    #[must_use]
    pub const fn arg_type(&self) -> WlArgType {
        match self {
            Self::Int(_) => WlArgType::Int,
            Self::Uint(_) => WlArgType::Uint,
            Self::Fixed(_) => WlArgType::Fixed,
            Self::Str(_) => WlArgType::String,
            Self::Object(_) => WlArgType::Object,
            Self::NewId(_) => WlArgType::NewId,
            Self::Array(_) => WlArgType::Array,
            Self::Fd(_) => WlArgType::Fd,
        }
    }

    /// Returns `true` when the value encodes a null object or null string.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        match self {
            Self::Str(value) => value.is_none(),
            Self::Array(value) => value.is_none(),
            Self::Object(id) => *id == 0,
            _ => false,
        }
    }

    /// Checks that the value can be marshalled for `details`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when the value does not match the
    /// argument type or a non-nullable argument carries a null value.
    pub fn validate(&self, details: WlArgDetails) -> WlResult<()> {
        if self.arg_type() != details.ty {
            return Err(WlError::invalid_argument(format_arg_type_mismatch(
                details.ty,
                self.arg_type(),
            )));
        }
        if !details.nullable && self.is_null() {
            return Err(WlError::invalid_argument(format_null_argument(details.ty)));
        }
        Ok(())
    }

    /// Claims a file descriptor argument.
    ///
    /// The argument is replaced with `-1` so the dispatcher knows the
    /// descriptor was taken and must not be released after the handler
    /// returned. Returns `None` when the argument is not a live file
    /// descriptor.
    pub fn take_fd(&mut self) -> Option<WlFd> {
        match self {
            Self::Fd(fd) if *fd >= 0 => Some(core::mem::replace(fd, -1)),
            _ => None,
        }
    }
}

impl From<u32> for WlArgument {
    #[inline]
    fn from(value: u32) -> Self {
        Self::Uint(value)
    }
}

impl From<i32> for WlArgument {
    #[inline]
    fn from(value: i32) -> Self {
        Self::Int(value)
    }
}

impl From<&str> for WlArgument {
    #[inline]
    fn from(value: &str) -> Self {
        Self::Str(Some(String::from(value)))
    }
}

impl From<String> for WlArgument {
    #[inline]
    fn from(value: String) -> Self {
        Self::Str(Some(value))
    }
}

impl From<WlFixed> for WlArgument {
    #[inline]
    fn from(value: WlFixed) -> Self {
        Self::Fixed(value)
    }
}

impl From<WlArray> for WlArgument {
    #[inline]
    fn from(value: WlArray) -> Self {
        Self::Array(Some(value))
    }
}

impl From<&WlArray> for WlArgument {
    #[inline]
    fn from(value: &WlArray) -> Self {
        Self::Array(Some(value.clone()))
    }
}

fn format_arg_type_mismatch(expected: WlArgType, actual: WlArgType) -> String {
    format!("expected {expected} argument, got {actual}")
}

fn format_null_argument(ty: WlArgType) -> String {
    format!("null value passed for non-nullable {ty} argument")
}

/// Common accessors shared by client proxies and server resources.
pub trait WlObject {
    /// Returns the object id on the wire.
    fn id(&self) -> u32;

    /// Returns the interface of the object.
    fn interface(&self) -> &'static WlInterface;

    /// Returns the protocol version of the object.
    fn version(&self) -> u32;
}

/// Side of an object id space, as tracked by a [`WlMap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WlMapSide {
    /// Ids allocated by the client, starting at `1`.
    Client,
    /// Ids allocated by the server, starting at [`SERVER_ID_START`].
    Server,
}

impl WlMapSide {
    /// Returns the first id of the id space.
    #[inline]
    pub const fn base(self) -> u32 {
        match self {
            Self::Client => 0,
            Self::Server => SERVER_ID_START,
        }
    }

    /// Returns the side an id belongs to.
    #[inline]
    #[must_use]
    pub const fn of_id(id: u32) -> Self {
        if id < SERVER_ID_START {
            Self::Client
        } else {
            Self::Server
        }
    }

    /// Returns `true` when the id belongs to this side.
    #[inline]
    pub const fn owns(self, id: u32) -> bool {
        matches!(
            (self, id < SERVER_ID_START),
            (Self::Client, true) | (Self::Server, false)
        )
    }
}

/// State of a single entry of a [`WlMap`].
#[derive(Debug)]
enum WlMapEntry<T> {
    /// The entry is on the free list and may be reused by `insert_new`.
    Free,
    /// The id exists but no object is attached to it.
    Vacant,
    /// The id refers to a live object.
    Live(T),
    /// The client destroyed the object and waits for `delete_id`.
    Zombie { interface: &'static WlInterface },
}

/// Object id map as implemented by `wl_map`.
///
/// The map tracks the client and server id space separately. Ids below
/// [`SERVER_ID_START`] index the client array directly, larger ids index
/// the server array after subtracting the base.
///
/// # Deviation from libwayland
///
/// libwayland keeps a single free list shared by both id spaces. This map
/// keeps one free list per id space, which prevents an id from one space from
/// being handed out for the other space.
#[derive(Debug)]
pub struct WlMap<T> {
    side: WlMapSide,
    client: Vec<WlMapEntry<T>>,
    server: Vec<WlMapEntry<T>>,
    free_client: Vec<u32>,
    free_server: Vec<u32>,
}

impl<T> WlMap<T> {
    /// Creates a map for the given allocating side.
    ///
    /// Index `0` of the client id space is reserved, so the first id handed
    /// out by [`WlMap::insert_new`] on the client side is `1`.
    #[must_use]
    pub fn new(side: WlMapSide) -> Self {
        Self {
            side,
            client: Vec::from([WlMapEntry::Vacant]),
            server: Vec::new(),
            free_client: Vec::new(),
            free_server: Vec::new(),
        }
    }

    /// Returns the side this map allocates ids for.
    #[inline]
    #[must_use]
    pub const fn side(&self) -> WlMapSide {
        self.side
    }

    /// Returns the total number of tracked ids.
    #[must_use]
    pub fn len(&self) -> usize {
        self.client
            .len()
            + self
                .server
                .len()
    }

    /// Returns `true` when no id has ever been allocated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn parts(&self, id: u32) -> Option<(&Vec<WlMapEntry<T>>, usize)> {
        if id < SERVER_ID_START {
            Some((&self.client, id as usize))
        } else {
            let index = id - SERVER_ID_START;
            if index > MAP_MAX_OBJECTS {
                return None;
            }
            Some((&self.server, index as usize))
        }
    }

    fn parts_mut(&mut self, id: u32) -> Option<(&mut Vec<WlMapEntry<T>>, usize)> {
        if id < SERVER_ID_START {
            Some((&mut self.client, id as usize))
        } else {
            let index = id - SERVER_ID_START;
            if index > MAP_MAX_OBJECTS {
                return None;
            }
            Some((&mut self.server, index as usize))
        }
    }

    fn entry(&self, id: u32) -> Option<&WlMapEntry<T>> {
        let (entries, index) = self.parts(id)?;
        entries.get(index)
    }

    /// Allocates a new id on the side of the map.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::TooManyObjects`] when [`MAP_MAX_OBJECTS`] is
    /// exhausted.
    pub fn insert_new(&mut self, data: T) -> WlResult<u32> {
        let (entries, free) = match self.side {
            WlMapSide::Client => (&mut self.client, &mut self.free_client),
            WlMapSide::Server => (&mut self.server, &mut self.free_server),
        };
        if entries.len() as u32 > MAP_MAX_OBJECTS {
            return Err(WlError::TooManyObjects);
        }
        let mut reused = None;
        while let Some(index) = free.pop() {
            if matches!(entries.get(index as usize), Some(WlMapEntry::Free)) {
                reused = Some(index as usize);
                break;
            }
        }
        let index = match reused {
            Some(index) => {
                entries[index] = WlMapEntry::Live(data);
                index
            }
            None => {
                if entries.len() as u32 > MAP_MAX_OBJECTS {
                    return Err(WlError::TooManyObjects);
                }
                entries.push(WlMapEntry::Live(data));
                entries.len() - 1
            }
        };
        Ok(index as u32
            + self
                .side
                .base())
    }

    /// Inserts an object at a caller chosen id.
    ///
    /// The id may belong to either side; ids beyond the current end of the
    /// id space are appended.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::TooManyObjects`] when `id` is out of range and
    /// [`WlError::InvalidArgument`] when `id` skips ahead in the id space.
    pub fn insert_at(&mut self, id: u32, data: T) -> WlResult<()> {
        let (entries, index) = self
            .parts_mut(id)
            .ok_or(WlError::InvalidObject(id))?;
        if index as u32 > MAP_MAX_OBJECTS {
            return Err(WlError::TooManyObjects);
        }
        if index > entries.len() {
            return Err(WlError::invalid_argument(format!(
                "id {id} skips ahead in the object map"
            )));
        }
        if index == entries.len() {
            entries.push(WlMapEntry::Live(data));
        } else {
            entries[index] = WlMapEntry::Live(data);
        }
        Ok(())
    }

    /// Reserves an id owned by the peer side of the map.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when the id belongs to the
    /// allocating side of the map or is already in use, and
    /// [`WlError::TooManyObjects`] when the id is out of range.
    pub fn reserve_new(&mut self, id: u32) -> WlResult<()> {
        if self
            .side
            .owns(id)
        {
            return Err(WlError::invalid_argument(format!(
                "id {id} belongs to the local id space"
            )));
        }
        let (entries, index) = self
            .parts_mut(id)
            .ok_or(WlError::InvalidObject(id))?;
        if index as u32 > MAP_MAX_OBJECTS {
            return Err(WlError::TooManyObjects);
        }
        match index.cmp(&entries.len()) {
            core::cmp::Ordering::Greater => Err(WlError::invalid_argument(format!(
                "id {id} skips ahead in the object map"
            ))),
            core::cmp::Ordering::Equal => {
                entries.push(WlMapEntry::Vacant);
                Ok(())
            }
            core::cmp::Ordering::Less => match &entries[index] {
                WlMapEntry::Vacant => Ok(()),
                _ => Err(WlError::invalid_argument(format!("id {id} is already in use"))),
            },
        }
    }

    /// Marks an id as existing but without an attached object.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`WlMap::insert_at`].
    pub fn vacate_at(&mut self, id: u32) -> WlResult<()> {
        let (entries, index) = self
            .parts_mut(id)
            .ok_or(WlError::InvalidObject(id))?;
        if index as u32 > MAP_MAX_OBJECTS {
            return Err(WlError::TooManyObjects);
        }
        if index > entries.len() {
            return Err(WlError::invalid_argument(format!(
                "id {id} skips ahead in the object map"
            )));
        }
        if index == entries.len() {
            entries.push(WlMapEntry::Vacant);
        } else {
            entries[index] = WlMapEntry::Vacant;
        }
        Ok(())
    }

    /// Turns a live entry into a zombie waiting for `delete_id`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidObject`] when the id has no live object.
    pub fn make_zombie(&mut self, id: u32, interface: &'static WlInterface) -> WlResult<()> {
        let (entries, index) = self
            .parts_mut(id)
            .ok_or(WlError::InvalidObject(id))?;
        match entries.get_mut(index) {
            Some(entry @ WlMapEntry::Live(_)) => {
                *entry = WlMapEntry::Zombie { interface };
                Ok(())
            }
            _ => Err(WlError::InvalidObject(id)),
        }
    }

    /// Frees an id on the side of the map and returns it for reuse.
    ///
    /// Returns `true` when the id was present.
    pub fn remove(&mut self, id: u32) -> bool {
        if !self
            .side
            .owns(id)
        {
            return false;
        }
        let index = if id < SERVER_ID_START {
            id as usize
        } else {
            (id - SERVER_ID_START) as usize
        };
        let (entries, free) = match self.side {
            WlMapSide::Client => (&mut self.client, &mut self.free_client),
            WlMapSide::Server => (&mut self.server, &mut self.free_server),
        };
        match entries.get_mut(index) {
            Some(entry) if !matches!(entry, WlMapEntry::Free) => {
                *entry = WlMapEntry::Free;
                free.push(index as u32);
                true
            }
            _ => false,
        }
    }

    /// Looks up the object attached to an id.
    #[must_use]
    pub fn lookup(&self, id: u32) -> Option<&T> {
        match self.entry(id) {
            Some(WlMapEntry::Live(data)) => Some(data),
            _ => None,
        }
    }

    /// Mutably looks up the object attached to an id.
    #[must_use]
    pub fn lookup_mut(&mut self, id: u32) -> Option<&mut T> {
        let (entries, index) = self.parts_mut(id)?;
        match entries.get_mut(index) {
            Some(WlMapEntry::Live(data)) => Some(data),
            _ => None,
        }
    }

    /// Takes the object attached to an id out of the map.
    ///
    /// The entry stays in place but no longer holds an object, matching
    /// libwayland's behaviour of clearing a pointer before freeing it.
    #[must_use]
    pub fn take(&mut self, id: u32) -> Option<T> {
        let (entries, index) = self.parts_mut(id)?;
        let entry = entries.get_mut(index)?;
        if !matches!(entry, WlMapEntry::Live(_)) {
            return None;
        }
        match core::mem::replace(entry, WlMapEntry::Vacant) {
            WlMapEntry::Live(data) => Some(data),
            _ => None,
        }
    }

    /// Returns `true` when the id refers to a destroyed client object.
    #[must_use]
    pub fn is_zombie(&self, id: u32) -> bool {
        matches!(self.side, WlMapSide::Client) && matches!(self.entry(id), Some(WlMapEntry::Zombie { .. }))
    }

    /// Returns the interface of a zombie entry.
    #[must_use]
    pub fn zombie_interface(&self, id: u32) -> Option<&'static WlInterface> {
        match self.entry(id) {
            Some(WlMapEntry::Zombie { interface }) => Some(interface),
            _ => None,
        }
    }

    /// Returns `true` when the id is allocated in the map.
    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        matches!(
            self.entry(id),
            Some(WlMapEntry::Live(_)) | Some(WlMapEntry::Zombie { .. })
        )
    }

    /// Returns `true` when the id exists without an attached object.
    #[must_use]
    pub fn is_vacant(&self, id: u32) -> bool {
        matches!(self.entry(id), Some(WlMapEntry::Vacant) | None)
    }

    /// Iterates over all live objects with their ids.
    pub fn iter(&self) -> WlMapIter<'_, T> {
        WlMapIter {
            client: &self.client,
            server: &self.server,
            index: 0,
            side: WlMapSide::Client,
            base: 0,
        }
    }
}

impl<T> Default for WlMap<T> {
    #[inline]
    fn default() -> Self {
        Self::new(WlMapSide::Client)
    }
}

/// Iterator over the live entries of a [`WlMap`].
#[derive(Debug, Clone)]
pub struct WlMapIter<'a, T> {
    client: &'a [WlMapEntry<T>],
    server: &'a [WlMapEntry<T>],
    index: usize,
    side: WlMapSide,
    base: u32,
}

impl<'a, T> Iterator for WlMapIter<'a, T> {
    type Item = (u32, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let entries = match self.side {
                WlMapSide::Client => self.client,
                WlMapSide::Server => self.server,
            };
            if self.index < entries.len() {
                let index = self.index;
                self.index += 1;
                if let WlMapEntry::Live(data) = &entries[index] {
                    return Some((index as u32 + self.base, data));
                }
            } else if self.side == WlMapSide::Client {
                self.side = WlMapSide::Server;
                self.base = SERVER_ID_START;
                self.index = 0;
            } else {
                return None;
            }
        }
    }
}

/// Node of an intrusive doubly linked list.
///
/// An initialized node points to itself. Nodes must not be moved once they
/// are part of a list; keep them in a pinned allocation such as `Box::pin`.
#[derive(Debug)]
pub struct WlList {
    prev: *mut WlList,
    next: *mut WlList,
}

impl WlList {
    /// Creates a detached node.
    ///
    /// Call [`WlList::init`] to make the node point to itself, and do so
    /// again after the containing value reached its final address.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            prev: core::ptr::null_mut(),
            next: core::ptr::null_mut(),
        }
    }

    /// Initializes the node as an empty list head or detached element.
    pub fn init(&mut self) {
        let this = core::ptr::from_mut(self);
        self.prev = this;
        self.next = this;
    }

    /// Inserts `elm` after this node.
    ///
    /// When `self` is the list head, `elm` becomes the first element.
    ///
    /// The caller must ensure `elm` is not already linked into a list.
    pub fn insert(&mut self, elm: &mut WlList) {
        let next = self.next;
        elm.prev = core::ptr::from_mut(self);
        elm.next = next;
        // Safety: `elm` is detached and exclusively borrowed, so writing the
        // neighbor links only touches nodes reachable from the list that owns
        // `self`, never `self` or `elm` themselves.
        unsafe {
            (*next).prev = core::ptr::from_mut(elm);
        }
        self.next = core::ptr::from_mut(elm);
    }

    /// Unlinks the node from its list and re-initializes it.
    pub fn remove(&mut self) {
        if self
            .prev
            .is_null()
            || self
                .next
                .is_null()
        {
            self.init();
            return;
        }
        // Safety: for a linked node both neighbors are valid list nodes owned
        // by the same structure graph; the node itself is exclusively borrowed
        // through `&mut self`.
        unsafe {
            (*self.prev).next = self.next;
            (*self.next).prev = self.prev;
        }
        self.init();
    }

    /// Appends all elements of `other` after this node.
    ///
    /// `other` is left empty.
    pub fn append_list(&mut self, other: &mut WlList) {
        if other.is_empty() {
            return;
        }
        // Safety: `other` is non-empty, so its first and last nodes are valid
        // nodes of the same list; `self` and `other` are distinct lists.
        unsafe {
            let other_first = other.next;
            let other_last = other.prev;
            (*other_last).next = core::ptr::from_mut(self);
            (*self.prev).next = other_first;
            (*other_first).prev = self.prev;
            self.prev = other_last;
        }
        other.init();
    }

    /// Returns `true` when the list has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        core::ptr::eq(self.next, self)
    }

    /// Returns the number of elements, in `O(n)`.
    #[must_use]
    pub fn length(&self) -> usize {
        let mut count = 0;
        for _ in self.iter() {
            count += 1;
        }
        count
    }

    /// Iterates over the nodes of the list.
    ///
    /// The iterator yields raw pointers so callers can use
    /// [`wl_container_of!`] to reach the containing structure.
    #[must_use]
    pub fn iter(&self) -> WlListIter {
        WlListIter {
            next: self.next,
            head: core::ptr::from_ref(self),
        }
    }
}

impl Default for WlList {
    #[inline]
    fn default() -> Self {
        let mut list = Self::new();
        list.init();
        list
    }
}

impl Drop for WlList {
    fn drop(&mut self) {
        if !self.is_empty() {
            self.remove();
        }
    }
}

/// Iterator over the nodes of a [`WlList`].
#[derive(Debug, Clone, Copy)]
pub struct WlListIter {
    next: *mut WlList,
    head: *const WlList,
}

impl Iterator for WlListIter {
    type Item = *mut WlList;

    fn next(&mut self) -> Option<Self::Item> {
        if core::ptr::eq(
            self.next
                .cast_const(),
            self.head,
        ) {
            return None;
        }
        let current = self.next;
        if current.is_null() {
            return None;
        }
        // Safety: `current` was produced by walking the list from a valid
        // head, so it points at a linked node whose `next` field is valid.
        self.next = unsafe { (*current).next };
        Some(current)
    }
}

/// Computes the address of the structure containing `ptr`.
///
/// # Examples
///
/// ```
/// use codevar_wl_protocol::wl_container_of;
///
/// struct Node { link: codevar_wl_protocol::WlList, value: u32 }
/// let mut node = Node { link: codevar_wl_protocol::WlList::new(), value: 7 };
/// node.link.init();
/// let link: *mut codevar_wl_protocol::WlList = &mut node.link;
/// let node_ptr = wl_container_of!(link, Node, link);
/// assert_eq!(unsafe { (*node_ptr).value }, 7);
/// ```
#[macro_export]
macro_rules! wl_container_of {
    ($ptr:expr, $ty:ty, $member:ident) => {{
        let member = $ptr as *const core::primitive::u8;
        let offset = ::core::mem::offset_of!($ty, $member);
        (member as usize - offset) as *mut $ty
    }};
}

/// Identifier returned when registering a signal listener.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WlListenerId(u64);

/// A single listener registered on a [`WlSignal`].
type WlListener<C, T> = Box<dyn FnMut(&mut C, &T)>;

/// Listener registry mirroring `wl_signal`.
///
/// A signal is stored inside the object it belongs to. Because listeners
/// receive `&mut` access to that object, the signal must be moved out of the
/// object before it is emitted; use the [`wl_signal_emit!`] macro for that.
pub struct WlSignal<C, T> {
    next_id: u64,
    listeners: Vec<(WlListenerId, WlListener<C, T>)>,
}

impl<C, T> fmt::Debug for WlSignal<C, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WlSignal")
            .field(
                "listeners",
                &self
                    .listeners
                    .len(),
            )
            .finish_non_exhaustive()
    }
}

impl<C, T> WlSignal<C, T> {
    /// Creates an empty signal.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next_id: 0,
            listeners: Vec::new(),
        }
    }

    /// Registers a listener and returns its id.
    pub fn add(&mut self, listener: impl FnMut(&mut C, &T) + 'static) -> WlListenerId {
        self.next_id += 1;
        let id = WlListenerId(self.next_id);
        self.listeners
            .push((id, Box::new(listener)));
        id
    }

    /// Removes a listener, returning `true` when it was registered.
    pub fn remove(&mut self, id: WlListenerId) -> bool {
        let before = self
            .listeners
            .len();
        self.listeners
            .retain(|(listener_id, _)| *listener_id != id);
        self.listeners
            .len()
            != before
    }

    /// Returns the number of registered listeners.
    #[must_use]
    pub fn len(&self) -> usize {
        self.listeners
            .len()
    }

    /// Returns `true` when no listener is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.listeners
            .is_empty()
    }

    /// Calls every listener with mutable access to `container`.
    ///
    /// The signal itself must already be removed from `container`, which the
    /// [`wl_signal_emit!`] macro takes care of. Listeners added to the signal
    /// while it is emitted are merged back by
    /// [`WlSignal::absorb`].
    pub fn emit(&mut self, container: &mut C, payload: &T) {
        for index in 0..self
            .listeners
            .len()
        {
            if let Some((_, listener)) = self
                .listeners
                .get_mut(index)
            {
                listener(container, payload);
            }
        }
    }

    /// Merges listeners registered on `pending` while the signal was emitted.
    pub fn absorb(&mut self, pending: &mut Self) {
        self.listeners
            .append(&mut pending.listeners);
        self.next_id = self
            .next_id
            .max(pending.next_id);
        pending.next_id = self.next_id;
    }
}

impl<C, T> Default for WlSignal<C, T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Emits a signal that is stored as a field of `container`.
///
/// The signal is moved out of the container while listeners run, so they may
/// take `&mut` access to it.
///
/// # Examples
///
/// ```
/// use codevar_wl_protocol::{WlSignal, wl_signal_emit};
///
/// struct Display { destroy: WlSignal<Display, ()> }
/// let mut display = Display { destroy: WlSignal::new() };
/// display.destroy.add(|_, _| {});
/// wl_signal_emit!(display, destroy, ());
/// assert_eq!(display.destroy.len(), 1);
/// ```
#[macro_export]
macro_rules! wl_signal_emit {
    ($container:expr, $field:ident, $payload:expr) => {{
        let mut taken = ::core::mem::take(&mut $container.$field);
        taken.emit(&mut $container, &$payload);
        taken.absorb(&mut $container.$field);
        $container.$field = taken;
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_conversions_round_trip() {
        let fixed = WlFixed::from_double(1.5);
        assert_eq!(fixed.to_raw(), 384);
        assert_eq!(fixed.to_double(), 1.5);
        assert_eq!(WlFixed::from_int(3).to_raw(), 768);
        assert_eq!(WlFixed::from_raw(384).to_int(), 1);
    }

    #[test]
    fn signature_parsing_matches_libwayland() {
        assert_eq!(arg_count("2u?o"), 2);
        assert_eq!(message_since("2u?o"), 2);
        assert_eq!(message_since("u"), 1);
        assert_eq!(count_fds("shu"), 1);
        assert_eq!(count_arrays("as"), 1);

        let (details, pos) = get_next_argument("2u?o", 0).unwrap();
        assert_eq!(details.ty, WlArgType::Uint);
        assert!(!details.nullable);
        let (details, pos) = get_next_argument("2u?o", pos).unwrap();
        assert_eq!(details.ty, WlArgType::Object);
        assert!(details.nullable);
        assert!(get_next_argument("2u?o", pos).is_none());
    }

    #[test]
    fn builtin_interfaces_have_consistent_types() {
        for interface in [&DISPLAY_INTERFACE, &REGISTRY_INTERFACE, &CALLBACK_INTERFACE] {
            for message in interface
                .requests
                .iter()
                .chain(
                    interface
                        .events
                        .iter(),
                )
            {
                assert_eq!(
                    message.arg_count(),
                    message
                        .types
                        .len()
                );
            }
        }
        assert!(
            DISPLAY_INTERFACE
                .request("sync")
                .is_some()
        );
        assert!(
            DISPLAY_INTERFACE
                .event("delete_id")
                .is_some()
        );
        assert_eq!(REGISTRY_INTERFACE.requests[0].name, "bind");
        assert!(DISPLAY_INTERFACE.equal(&DISPLAY_INTERFACE));
        assert!(!DISPLAY_INTERFACE.equal(&REGISTRY_INTERFACE));
    }

    #[test]
    fn argument_validation_checks_type_and_nullability() {
        let nullable = WlArgDetails::new(WlArgType::String, true);
        let strict = WlArgDetails::new(WlArgType::String, false);
        assert!(
            WlArgument::Str(None)
                .validate(nullable)
                .is_ok()
        );
        assert!(
            WlArgument::Str(None)
                .validate(strict)
                .is_err()
        );
        assert!(
            WlArgument::Uint(1)
                .validate(strict)
                .is_err()
        );
    }

    #[test]
    fn map_allocates_frees_and_reuses_client_ids() {
        let mut map: WlMap<u32> = WlMap::new(WlMapSide::Client);
        let first = map
            .insert_new(10)
            .unwrap();
        assert_eq!(first, 1);
        let second = map
            .insert_new(20)
            .unwrap();
        assert_eq!(second, 2);
        assert_eq!(map.lookup(1), Some(&10));

        assert!(map.remove(1));
        assert_eq!(map.lookup(1), None);
        let reused = map
            .insert_new(30)
            .unwrap();
        assert_eq!(reused, 1);
        assert_eq!(map.lookup(1), Some(&30));
        assert_eq!(map.lookup(2), Some(&20));
    }

    #[test]
    fn map_reserves_peer_ids_only() {
        let mut map: WlMap<u32> = WlMap::new(WlMapSide::Server);
        // Reserve the whole dense prefix the peer could have allocated.
        for id in 1..=7 {
            map.reserve_new(id)
                .unwrap();
        }
        assert!(map.is_vacant(7));
        assert!(
            map.reserve_new(9)
                .is_err()
        );
        assert!(
            map.reserve_new(SERVER_ID_START)
                .is_err()
        );

        map.insert_at(7, 42)
            .unwrap();
        assert_eq!(map.lookup(7), Some(&42));
        assert!(
            map.reserve_new(7)
                .is_err()
        );

        let server_id = map
            .insert_new(99)
            .unwrap();
        assert_eq!(server_id, SERVER_ID_START);
        assert_eq!(map.lookup(server_id), Some(&99));
    }

    #[test]
    fn map_tracks_zombies_until_removed() {
        let mut map: WlMap<u32> = WlMap::new(WlMapSide::Client);
        let id = map
            .insert_new(1)
            .unwrap();
        map.make_zombie(id, &REGISTRY_INTERFACE)
            .unwrap();
        assert!(map.is_zombie(id));
        assert_eq!(map.lookup(id), None);
        assert_eq!(map.zombie_interface(id), Some(&REGISTRY_INTERFACE));
        assert!(map.remove(id));
        assert!(!map.is_zombie(id));
    }

    #[test]
    fn map_take_leaves_vacant_entry() {
        let mut map: WlMap<u32> = WlMap::new(WlMapSide::Server);
        assert_eq!(map.take(1), None);
        map.insert_at(1, 5)
            .unwrap();
        assert_eq!(map.take(1), Some(5));
        assert!(map.is_vacant(1));
        map.insert_at(1, 6)
            .unwrap();
        assert_eq!(map.lookup(1), Some(&6));
    }

    #[test]
    fn map_iterates_live_entries_in_id_order() {
        let mut map: WlMap<u32> = WlMap::new(WlMapSide::Server);
        map.insert_at(1, 10)
            .unwrap();
        let server_id = map
            .insert_new(20)
            .unwrap();
        let collected: Vec<(u32, &u32)> = map
            .iter()
            .collect();
        assert_eq!(collected, alloc::vec![(1, &10), (server_id, &20)]);
    }

    #[test]
    fn list_links_unlinks_and_relinks() {
        let mut head = WlList::new();
        head.init();
        let mut first = WlList::new();
        let mut second = WlList::new();
        first.init();
        second.init();

        head.insert(&mut first);
        head.insert(&mut second);
        assert_eq!(head.length(), 2);
        assert!(!head.is_empty());

        second.remove();
        assert_eq!(head.length(), 1);
        first.remove();
        assert!(head.is_empty());
    }

    #[test]
    fn container_of_macro_finds_owner() {
        struct Holder {
            value: u32,
            link: WlList,
        }
        let mut holder = Box::pin(Holder {
            value: 77,
            link: WlList::new(),
        });
        holder
            .link
            .init();
        let owner: *mut Holder = &mut *holder;
        unsafe {
            let link = core::ptr::addr_of_mut!((*owner).link);
            let recovered = wl_container_of!(link, Holder, link);
            assert_eq!(recovered, owner);
            assert_eq!((*recovered).value, 77);
        }
    }

    #[test]
    fn signal_emits_and_removes_listeners() {
        struct Source {
            signal: WlSignal<Source, u32>,
            seen: u32,
        }
        let mut source = Source {
            signal: WlSignal::new(),
            seen: 0,
        };
        let id = source
            .signal
            .add(|container, payload| {
                container.seen += payload;
            });
        wl_signal_emit!(source, signal, &3);
        assert_eq!(source.seen, 3);
        assert!(
            source
                .signal
                .remove(id)
        );
        wl_signal_emit!(source, signal, &5);
        assert_eq!(source.seen, 3);
    }
}
