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

//! Validation of the names that appear in D-Bus messages.
//!
//! The rules follow the "Valid Names" section of the D-Bus
//! specification: bus names, interfaces, error names and members are
//! limited to 255 bytes of ASCII with element rules that depend on the
//! kind of name.

use crate::dbus_error::{DbusError, DbusResult};

/// Maximum length of bus names, interfaces and members.
pub const MAX_NAME_LEN: usize = 255;

fn is_name_char(byte: u8, allow_hyphen: bool) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || (allow_hyphen && byte == b'-')
}

fn is_element_start(byte: u8, allow_digit: bool) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_' || (allow_digit && byte.is_ascii_digit())
}

/// Splits a `.`-separated name and checks every element.
///
/// `allow_hyphen` permits `-` in elements, `allow_digit` permits
/// elements that begin with a digit (unique connection names only) and
/// `min_elements` is the minimum number of elements the name must have.
fn check_elements(name: &str, allow_hyphen: bool, allow_digit: bool, min: usize) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_NAME_LEN {
        return false;
    }
    if bytes[0] == b'.' {
        return false;
    }
    let mut elements = 0usize;
    let mut start = 0usize;
    let mut index = 0usize;
    loop {
        let is_last = index == bytes.len();
        if is_last || bytes[index] == b'.' {
            if index == start {
                return false;
            }
            if !is_element_start(bytes[start], allow_digit) {
                return false;
            }
            for byte in &bytes[start..index] {
                if !is_name_char(*byte, allow_hyphen) {
                    return false;
                }
            }
            elements += 1;
            start = index + 1;
        }
        if is_last {
            break;
        }
        index += 1;
    }
    elements >= min
}

/// Returns `true` when `path` is a valid object path.
///
/// A path starts with `/` and is followed by non-empty elements of
/// `[A-Z][a-z][0-9]_` separated by `/`. Only the root path `/` may end
/// with a slash.
#[must_use]
pub fn is_valid_object_path(path: &str) -> bool {
    let bytes = path.as_bytes();
    if bytes.first() != Some(&b'/') {
        return false;
    }
    if bytes.len() == 1 {
        return true;
    }
    if bytes.last() == Some(&b'/') {
        return false;
    }
    let mut start = 1usize;
    let mut index = 1usize;
    loop {
        let is_last = index == bytes.len();
        if is_last || bytes[index] == b'/' {
            if index == start {
                return false;
            }
            for byte in &bytes[start..index] {
                if !is_name_char(*byte, false) {
                    return false;
                }
            }
            start = index + 1;
        }
        if is_last {
            break;
        }
        index += 1;
    }
    true
}

/// Returns `true` when `name` is a valid interface name.
///
/// Interface names have at least two `.`-separated elements whose
/// characters are `[A-Z][a-z][0-9]_` and that never begin with a digit.
#[must_use]
pub fn is_valid_interface_name(name: &str) -> bool {
    check_elements(name, false, false, 2)
}

/// Returns `true` when `name` is a valid error name.
///
/// Error names follow the same rules as interface names.
#[must_use]
pub fn is_valid_error_name(name: &str) -> bool {
    is_valid_interface_name(name)
}

/// Returns `true` when `name` is a valid member (method or signal)
/// name.
///
/// Members are 1 to 255 bytes of `[A-Z][a-z][0-9]_`, never begin with
/// a digit and never contain a `.`.
#[must_use]
pub fn is_valid_member(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_NAME_LEN {
        return false;
    }
    if !is_element_start(bytes[0], false) {
        return false;
    }
    bytes
        .iter()
        .all(|byte| is_name_char(*byte, false))
}

/// Returns `true` when `name` is a valid bus name of any kind.
///
/// Bus names have at least two `.`-separated elements of
/// `[A-Z][a-z][0-9]_-`. Only unique connection names (leading `:`) may
/// have elements that begin with a digit.
#[must_use]
pub fn is_valid_bus_name(name: &str) -> bool {
    let unique = name.as_bytes().first() == Some(&b':');
    let rest = if unique { &name[1..] } else { name };
    check_elements(rest, true, unique, 2)
}

/// Returns `true` when `name` is a valid unique connection name.
#[must_use]
pub fn is_valid_unique_name(name: &str) -> bool {
    name.as_bytes().first() == Some(&b':') && is_valid_bus_name(name)
}

/// Validates an object path, reporting the failure as [`DbusError`].
///
/// # Errors
///
/// Returns [`DbusError::InvalidName`] when `path` is not a valid
/// object path.
pub fn validate_object_path(path: &str) -> DbusResult<()> {
    if is_valid_object_path(path) {
        Ok(())
    } else {
        Err(DbusError::invalid_name(alloc::format!(
            "invalid object path: {path}"
        )))
    }
}

/// Validates an interface name, reporting the failure as [`DbusError`].
///
/// # Errors
///
/// Returns [`DbusError::InvalidName`] when `name` is not a valid
/// interface name.
pub fn validate_interface_name(name: &str) -> DbusResult<()> {
    if is_valid_interface_name(name) {
        Ok(())
    } else {
        Err(DbusError::invalid_name(alloc::format!(
            "invalid interface name: {name}"
        )))
    }
}

/// Validates a bus name, reporting the failure as [`DbusError`].
///
/// # Errors
///
/// Returns [`DbusError::InvalidName`] when `name` is not a valid bus
/// name.
pub fn validate_bus_name(name: &str) -> DbusResult<()> {
    if is_valid_bus_name(name) {
        Ok(())
    } else {
        Err(DbusError::invalid_name(alloc::format!(
            "invalid bus name: {name}"
        )))
    }
}

/// Validates a member name, reporting the failure as [`DbusError`].
///
/// # Errors
///
/// Returns [`DbusError::InvalidName`] when `name` is not a valid
/// member name.
pub fn validate_member(name: &str) -> DbusResult<()> {
    if is_valid_member(name) {
        Ok(())
    } else {
        Err(DbusError::invalid_name(alloc::format!(
            "invalid member name: {name}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_and_rejects_object_paths() {
        assert!(is_valid_object_path("/"));
        assert!(is_valid_object_path("/org/freedesktop/DBus"));
        assert!(is_valid_object_path("/a1/_b"));
        assert!(!is_valid_object_path(""));
        assert!(!is_valid_object_path("org/freedesktop"));
        assert!(!is_valid_object_path("//a"));
        assert!(!is_valid_object_path("/a//b"));
        assert!(!is_valid_object_path("/a/"));
        assert!(!is_valid_object_path("/a-b"));
        // Object path elements may start with a digit, only their
        // character set is restricted.
        assert!(is_valid_object_path("/1a"));
    }

    #[test]
    fn accepts_and_rejects_interface_names() {
        assert!(is_valid_interface_name("org.freedesktop.DBus"));
        assert!(is_valid_interface_name("a.b"));
        assert!(is_valid_interface_name("com.example._7zip"));
        assert!(!is_valid_interface_name("org"));
        assert!(!is_valid_interface_name(""));
        assert!(!is_valid_interface_name("org.7zip.App"));
        assert!(!is_valid_interface_name("org.ex-ample.App"));
        assert!(!is_valid_interface_name(".org.App"));
        assert!(!is_valid_interface_name("org..App"));
        assert!(!is_valid_interface_name("org.App."));
    }

    #[test]
    fn accepts_and_rejects_bus_names() {
        assert!(is_valid_bus_name(":1.42"));
        assert!(is_valid_bus_name("org.example.Editor"));
        assert!(is_valid_bus_name(":1.0"));
        assert!(is_valid_bus_name("org.example.some-name"));
        assert!(!is_valid_bus_name(":1"));
        assert!(!is_valid_bus_name("org"));
        assert!(!is_valid_bus_name(".org"));
        assert!(!is_valid_bus_name("org.7zip"));
        assert!(!is_valid_bus_name("org.example."));
        assert!(is_valid_unique_name(":1.42"));
        assert!(!is_valid_unique_name("org.example.Editor"));
    }

    #[test]
    fn accepts_and_rejects_members() {
        assert!(is_valid_member("Hello"));
        assert!(is_valid_member("ItemsChanged"));
        assert!(is_valid_member("_private"));
        assert!(!is_valid_member(""));
        assert!(!is_valid_member("1Hello"));
        assert!(!is_valid_member("Get.Name"));
        assert!(!is_valid_member("Get-Name"));
        assert!(!is_valid_member(&"x".repeat(MAX_NAME_LEN + 1)));
    }

    #[test]
    fn enforces_the_name_length_limit() {
        let mut name = String::from("org.example.");
        name.push_str(&"a".repeat(MAX_NAME_LEN));
        assert!(!is_valid_interface_name(&name));
    }

    #[test]
    fn validate_helpers_report_errors() {
        assert!(validate_object_path("/ok").is_ok());
        assert!(validate_object_path("bad").is_err());
        assert!(validate_interface_name("a.b").is_ok());
        assert!(validate_interface_name("a").is_err());
        assert!(validate_bus_name(":1.1").is_ok());
        assert!(validate_bus_name("nope").is_err());
        assert!(validate_member("Hello").is_ok());
        assert!(validate_member("no.dot").is_err());
    }
}
