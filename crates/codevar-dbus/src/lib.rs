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

//! D-Bus client implementation for Linux.
//!
//! The crate implements the D-Bus wire protocol in `no_std` code with
//! borrowed decoding, so linking it does not pull `std` into the binary.
//! Pluggable transports keep the connection independent of the socket
//! layer.
//!
//! # Example
//!
//! ```no_run
//! use codevar_dbus::{Connection, DbusTransport, DbusPollEvents, DbusError};
//! use core::time::Duration;
//!
//! fn run<T: DbusTransport>(transport: T) -> Result<(), DbusError> {
//!     let mut conn = Connection::new(transport);
//!     conn.authenticate(1000)?;
//!     conn.hello()?;
//!     conn.call(
//!         "org.freedesktop.DBus",
//!         "/org/freedesktop/DBus",
//!         "org.freedesktop.DBus",
//!         "ListNames",
//!         |body| body.write_array("s", |body| Ok(())),
//!         Duration::from_secs(10),
//!     )?;
//!     Ok(())
//! }
//! ```

#![cfg_attr(not(test), no_std)]

extern crate alloc;

mod dbus_addr;
mod dbus_auth;
mod dbus_conn;
mod dbus_error;
mod dbus_marshal;
mod dbus_message;
mod dbus_names;
mod dbus_signature;
mod dbus_transport;

pub use dbus_addr::{
    DbusAddress, SESSION_BUS_FILE, SYSTEM_BUS_SOCKET, SYSTEM_BUS_SOCKET_LEGACY,
    percent_decode,
};
pub use dbus_auth::{AuthPoll, AuthSession};
pub use dbus_conn::{Connection, DEFAULT_CALL_TIMEOUT};
pub use dbus_error::{DbusError, DbusResult};
pub use dbus_marshal::{ByteOrder, DbusReader, DbusWriter, MAX_ARRAY_LEN};
pub use dbus_message::{
    BodyWriter, DbusMessage, DbusMessageStream, FIELD_DESTINATION, FIELD_ERROR_NAME,
    FIELD_INTERFACE, FIELD_MEMBER, FIELD_PATH, FIELD_REPLY_SERIAL, FIELD_SENDER,
    FIELD_SIGNATURE, FIELD_UNIX_FDS, FIXED_HEADER_LEN,
    FLAG_ALLOW_INTERACTIVE_AUTHORIZATION, FLAG_NO_AUTO_START, FLAG_NO_REPLY_EXPECTED,
    MAX_MESSAGE_LEN, MessageKind, PROTOCOL_VERSION,
};
pub use dbus_names::{
    MAX_NAME_LEN, is_valid_bus_name, is_valid_error_name, is_valid_interface_name,
    is_valid_member, is_valid_object_path, is_valid_unique_name, validate_bus_name,
    validate_interface_name, validate_member, validate_object_path,
};
pub use dbus_signature::{
    MAX_ARRAY_DEPTH, MAX_SIGNATURE_LEN, MAX_STRUCT_DEPTH, SignatureIter, is_basic_type,
    is_container_type, single_complete_type_len, type_alignment, type_fixed_size,
    validate_signature, validate_single_type,
};
pub use dbus_transport::{DbusPollEvents, DbusTransport};

#[cfg(all(unix, not(target_arch = "wasm32")))]
pub mod dbus_unix;
