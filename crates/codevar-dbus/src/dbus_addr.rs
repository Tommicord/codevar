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

//! D-Bus server address parsing.
//!
//! An address list separates several addresses with `;`, each address is
//! a transport name followed by `,` separated `key=value` pairs:
//!
//! ```text
//! unix:path=/run/user/1000/bus,guid=1234abcd
//! ```
//!
//! Values use percent escaping; the unescaped byte set is
//! `[-0-9A-Za-z_/.*]`. Malformed escapes are rejected, while unescaped
//! bytes outside the optional set are accepted leniently, matching the
//! behaviour of deployed buses.

use alloc::string::String;
use alloc::vec::Vec;

use crate::dbus_error::{DbusError, DbusResult};

/// Socket path of the well-known system bus.
pub const SYSTEM_BUS_SOCKET: &str = "/run/dbus/system_bus_socket";
/// Legacy socket path of the well-known system bus.
pub const SYSTEM_BUS_SOCKET_LEGACY: &str = "/var/run/dbus/system_bus_socket";
/// File name of the session bus socket inside `$XDG_RUNTIME_DIR`.
pub const SESSION_BUS_FILE: &str = "bus";

/// A single parsed server address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbusAddress {
    /// Transport name before the `:`, e.g. `unix`.
    transport: String,
    /// Escaped-decoded `key=value` pairs in wire order.
    keys: Vec<(String, String)>,
}

impl DbusAddress {
    /// Parses a `;` separated address list.
    ///
    /// Empty segments are skipped; at least one address is required.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidAddress`] when the list contains no
    /// address or a segment is malformed.
    pub fn parse_all(list: &str) -> DbusResult<Vec<Self>> {
        let mut addresses = Vec::new();
        for segment in list.split(';') {
            if segment.is_empty() {
                continue;
            }
            addresses.push(Self::parse(segment)?);
        }
        if addresses.is_empty() {
            return Err(DbusError::invalid_address("address list is empty"));
        }
        Ok(addresses)
    }

    /// Parses one `transport:key=value,...` address.
    ///
    /// # Errors
    ///
    /// Returns [`DbusError::InvalidAddress`] when the transport or a key
    /// is missing or an escape sequence is malformed.
    pub fn parse(entry: &str) -> DbusResult<Self> {
        let (transport, rest) = entry
            .split_once(':')
            .ok_or_else(|| {
                DbusError::invalid_address(alloc::format!(
                    "address `{entry}` is missing the transport separator"
                ))
            })?;
        if transport.is_empty() {
            return Err(DbusError::invalid_address("transport name is empty"));
        }
        if transport.contains('=') || transport.contains(',') {
            return Err(DbusError::invalid_address(alloc::format!(
                "invalid transport name `{transport}`"
            )));
        }
        let mut keys = Vec::new();
        for pair in rest.split(',') {
            if pair.is_empty() {
                continue;
            }
            let (key, value) = pair
                .split_once('=')
                .ok_or_else(|| {
                    DbusError::invalid_address(alloc::format!("address entry `{pair}` is missing `=`"))
                })?;
            if key.is_empty() {
                return Err(DbusError::invalid_address("address key is empty"));
            }
            keys.push((percent_decode(key)?, percent_decode(value)?));
        }
        Ok(Self {
            transport: String::from(transport),
            keys,
        })
    }

    /// Returns the transport name, e.g. `unix`.
    #[must_use]
    pub fn transport(&self) -> &str {
        &self.transport
    }

    /// Returns the decoded value stored under `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.keys
            .iter()
            .find(|(entry_key, _)| entry_key == key)
            .map(|(_, value)| value.as_str())
    }

    /// Returns the `guid` key of the address, when present.
    #[must_use]
    pub fn guid(&self) -> Option<&str> {
        self.get("guid")
    }

    /// Returns `true` for the `unix` transport.
    #[must_use]
    pub fn is_unix(&self) -> bool {
        self.transport == "unix"
    }

    /// Returns `true` when the address carries enough information to
    /// open a connection.
    #[must_use]
    pub fn is_connectable(&self) -> bool {
        if !self.is_unix() {
            return false;
        }
        self.get("path")
            .is_some_and(|path| !path.is_empty())
            || self
                .get("abstract")
                .is_some_and(|name| !name.is_empty())
    }
}

/// Decodes the percent escapes of one address key or value.
///
/// # Errors
///
/// Returns [`DbusError::InvalidAddress`] when a `%` is not followed by
/// two hexadecimal digits.
pub fn percent_decode(value: &str) -> DbusResult<String> {
    let bytes = value.as_bytes();
    if !bytes.contains(&b'%') {
        return Ok(String::from(value));
    }
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte != b'%' {
            decoded.push(byte);
            index += 1;
            continue;
        }
        let high = bytes
            .get(index + 1)
            .and_then(|&digit| hex_digit(digit))
            .ok_or_else(|| {
                DbusError::invalid_address(alloc::format!("invalid escape sequence in `{value}`"))
            })?;
        let low = bytes
            .get(index + 2)
            .and_then(|&digit| hex_digit(digit))
            .ok_or_else(|| {
                DbusError::invalid_address(alloc::format!("invalid escape sequence in `{value}`"))
            })?;
        decoded.push(high << 4 | low);
        index += 3;
    }
    String::from_utf8(decoded)
        .map_err(|_| DbusError::invalid_address(alloc::format!("`{value}` does not decode to valid UTF-8")))
}

const fn hex_digit(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        b'A'..=b'F' => Some(digit - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_unix_path_with_guid() {
        let address = DbusAddress::parse("unix:path=/run/user/1000/bus,guid=1234abcd").unwrap();
        assert_eq!(address.transport(), "unix");
        assert!(address.is_unix());
        assert!(address.is_connectable());
        assert_eq!(address.get("path"), Some("/run/user/1000/bus"));
        assert_eq!(address.guid(), Some("1234abcd"));
    }

    #[test]
    fn parses_abstract_socket_names() {
        let address = DbusAddress::parse("unix:abstract=%2Ftmp%2Fdbus-abc").unwrap();
        assert_eq!(address.get("abstract"), Some("/tmp/dbus-abc"));
        assert!(address.is_connectable());
    }

    #[test]
    fn rejects_malformed_escapes() {
        assert!(DbusAddress::parse("unix:path=%zz").is_err());
        assert!(DbusAddress::parse("unix:path=/x%2").is_err());
        assert!(DbusAddress::parse("unix:path=trailing%").is_err());
    }

    #[test]
    fn splits_address_lists_and_skips_empty_segments() {
        let addresses = DbusAddress::parse_all("unix:path=/a;unix:path=/b;").unwrap();
        assert_eq!(addresses.len(), 2);
        assert_eq!(addresses[0].get("path"), Some("/a"));
        assert_eq!(addresses[1].get("path"), Some("/b"));
        assert!(DbusAddress::parse_all("").is_err());
        assert!(DbusAddress::parse_all(";;").is_err());
    }

    #[test]
    fn rejects_structurally_invalid_addresses() {
        assert!(DbusAddress::parse("unixpath=/x").is_err());
        assert!(DbusAddress::parse(":path=/x").is_err());
        assert!(DbusAddress::parse("unix:path").is_err());
        assert!(DbusAddress::parse("unix:=/x").is_err());
        assert!(DbusAddress::parse("un,ix:path=/x").is_err());
    }

    #[test]
    fn non_unix_addresses_are_not_connectable() {
        let address = DbusAddress::parse("tcp:host=127.0.0.1,port=4242").unwrap();
        assert!(!address.is_unix());
        assert!(!address.is_connectable());
        assert_eq!(address.get("port"), Some("4242"));

        let empty = DbusAddress::parse("unix:guid=abcd").unwrap();
        assert!(empty.is_unix());
        assert!(!empty.is_connectable());
    }

    #[test]
    fn decodes_hexadecimal_escapes() {
        assert_eq!(percent_decode("a%41b").unwrap(), "aAb");
        assert_eq!(percent_decode("%00").unwrap(), "\0");
        assert_eq!(percent_decode("plain").unwrap(), "plain");
        assert!(percent_decode("%1").is_err());
        assert!(percent_decode("%zz").is_err());
    }
}
