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

//! Shared helpers for every portal frontend.
//!
//! Ported from `shared/xdp-utils.c` in the C reference: app-id, token
//! and filename validators, a `GKeyFile`-style parser, the D-Bus
//! `a{sv}` option map with [`filter_options`], base64 token
//! generation, shell quoting helpers and document portal path
//! remapping.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use codevar_dbus::{
    BodyWriter, DbusReader, DbusResult, DbusWriter, SignatureIter, single_complete_type_len, type_alignment,
    validate_array_element_type, validate_signature, validate_single_type,
};
use spin::Mutex;

use crate::xdp_error::PortalError;

/// The `options` argument of portal methods: a D-Bus `a{sv}` map with
/// deterministic iteration order.
pub type OptionMap = BTreeMap<String, PortalValue>;

/// Returns whether `c` is allowed in an app id element or token.
const fn is_name_character(c: u8, allow_dash: bool) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || (allow_dash && c == b'-')
}

/// Returns whether `app_id` is a valid application id.
///
/// Same rules as `xdp_is_valid_app_id` in the C reference: at most 255
/// bytes, at least one dot, no leading or trailing dot, elements built
/// from alphanumerics and underscores with dashes allowed only in the
/// final element.
#[must_use]
pub fn is_valid_app_id(app_id: &str) -> bool {
    let len = app_id.len();
    if len == 0 || len > 255 {
        return false;
    }
    let bytes = app_id.as_bytes();
    if bytes[0] == b'.' {
        return false;
    }
    let last_dot = bytes.iter().rposition(|&b| b == b'.');
    let mut dot_count = 0u32;
    let mut last_element = false;
    let mut index = 0;
    while index < len {
        if bytes[index] == b'.' {
            last_element = Some(index) == last_dot;
            index += 1;
            if index == len {
                return false;
            }
            dot_count += 1;
        }
        if !is_name_character(bytes[index], last_element) {
            return false;
        }
        index += 1;
    }
    dot_count >= 1
}

/// Returns whether `token` can be used as the handle token of a
/// request: every character is alphanumeric or an underscore, and
/// `/foo/{token}` is a valid object path.
#[must_use]
pub fn is_valid_token(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    if !token.bytes().all(|c| is_name_character(c, false)) {
        return false;
    }
    let path = format!("/foo/{token}");
    codevar_dbus::is_valid_object_path(&path)
}

/// Returns whether `filename` is usable as a bare file name: non
/// empty, not `.` or `..` and free of path separators.
#[must_use]
pub fn is_valid_filename(filename: &str) -> bool {
    if filename.is_empty() || filename == "." || filename == ".." {
        return false;
    }
    !filename.contains('/')
}

/// A `GKeyFile`-style INI file with ordered groups and duplicate keys.
///
/// Parsing follows `g_key_file_load_from_data`: comments start with
/// `#` or `;` in column one, group headers are `[name]`, key lines are
/// split at the first `=` and blank lines are skipped. Anything else
/// aborts the parse, mirroring how GLib rejects the whole file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyFile {
    groups: Vec<(String, Vec<(String, String)>)>,
}

impl KeyFile {
    /// Creates an empty key file.
    #[must_use]
    pub const fn new() -> Self {
        Self { groups: Vec::new() }
    }

    /// Parses the contents of a key file.
    ///
    /// Follows `GKeyFile` rules: leading whitespace is ignored,
    /// comments start with `#`, keys and values are trimmed of
    /// surrounding whitespace at the split point, a duplicate group
    /// header re-opens the earlier group and duplicate keys keep both
    /// entries (the last one wins on read).
    ///
    /// # Errors
    ///
    /// Returns [`PortalError::InvalidArgument`] when a line is neither
    /// a comment, a group header nor a key/value pair, when a key line
    /// appears before any group, or when a group header is malformed.
    pub fn parse(data: &str) -> Result<Self, PortalError> {
        let mut file = Self::new();
        let mut current_group: Option<usize> = None;
        for (index, raw_line) in data.split('\n').enumerate() {
            let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
            let line = line.trim_start();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line_number = index + 1;
            if let Some(rest) = line.strip_prefix('[') {
                let name = rest.strip_suffix(']').ok_or_else(|| {
                    PortalError::InvalidArgument(format!("line {line_number}: unterminated group header"))
                })?;
                if name.is_empty() {
                    return Err(PortalError::InvalidArgument(format!(
                        "line {line_number}: empty group name"
                    )));
                }
                current_group = Some(
                    match file
                        .groups
                        .iter()
                        .position(|(entry, _)| entry == name)
                    {
                        Some(position) => position,
                        None => {
                            file.groups.push((String::from(name), Vec::new()));
                            file.groups.len() - 1
                        }
                    },
                );
                continue;
            }
            let Some(eq) = line.find('=') else {
                return Err(PortalError::InvalidArgument(format!(
                    "line {line_number}: not a key/value pair"
                )));
            };
            let key = line[..eq].trim();
            if key.is_empty() {
                return Err(PortalError::InvalidArgument(format!(
                    "line {line_number}: empty key"
                )));
            }
            let Some(group_index) = current_group else {
                return Err(PortalError::InvalidArgument(format!(
                    "line {line_number}: key/value pair outside of any group"
                )));
            };
            let value = line[eq + 1..].trim_start();
            file.groups[group_index]
                .1
                .push((String::from(key), String::from(value)));
        }
        Ok(file)
    }

    /// Returns whether `group` exists.
    #[must_use]
    pub fn has_group(&self, group: &str) -> bool {
        self.groups.iter().any(|(name, _)| name == group)
    }

    /// Iterates over `(group, entries)` in file order.
    pub fn iter_groups(&self) -> impl Iterator<Item = (&str, &[(String, String)])> {
        self.groups
            .iter()
            .map(|(name, entries)| (name.as_str(), entries.as_slice()))
    }

    /// Returns the raw value stored under `key` without unescaping;
    /// for duplicate keys the last entry wins, like
    /// `g_key_file_get_value`.
    #[must_use]
    pub fn value(&self, group: &str, key: &str) -> Option<&str> {
        let entries = self.find_group(group)?;
        entries
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Returns the value stored under `key`, resolving backslash
    /// escapes like `g_key_file_get_string`; `None` when the key is
    /// missing or holds an invalid escape sequence.
    #[must_use]
    pub fn get(&self, group: &str, key: &str) -> Option<String> {
        self.value(group, key).and_then(unescape)
    }

    /// Returns the `;`-separated list stored under `key`, or `None`
    /// when the key is missing or an element holds an invalid escape.
    ///
    /// Follows `g_key_file_get_string_list`: a backslash escapes the
    /// character that follows it for splitting purposes, one trailing
    /// empty element is dropped (so `a;b;` yields two elements) and
    /// every element is unescaped.
    #[must_use]
    pub fn list(&self, group: &str, key: &str) -> Option<Vec<String>> {
        let raw = self.value(group, key)?;
        let mut parts = split_escaped_list(raw);
        if parts.last().is_some_and(String::is_empty) {
            parts.pop();
        }
        parts.iter().map(|part| unescape(part)).collect()
    }

    /// Returns the boolean stored under `key`, or `default` when the
    /// key is missing or does not hold exactly `true` or `false`,
    /// like `g_key_file_get_boolean`.
    #[must_use]
    pub fn boolean(&self, group: &str, key: &str, default: bool) -> bool {
        match self.value(group, key) {
            Some("true") => true,
            Some("false") => false,
            _ => default,
        }
    }

    /// Returns the raw entries of `group` in file order.
    #[must_use]
    pub fn entries(&self, group: &str) -> Option<&[(String, String)]> {
        self.find_group(group).map(Vec::as_slice)
    }

    /// Sets `key` to `value` inside `group`, creating the group when
    /// needed, replacing the first existing occurrence in place or
    /// appending a new entry.
    pub fn set(&mut self, group: &str, key: &str, value: &str) {
        let position = self
            .groups
            .iter()
            .position(|(name, _)| name == group);
        let group_index = match position {
            Some(index) => index,
            None => {
                self.groups
                    .push((String::from(group), Vec::new()));
                self.groups.len() - 1
            }
        };
        let entries = &mut self.groups[group_index].1;
        match entries.iter_mut().find(|(k, _)| k == key) {
            Some(entry) => entry.1 = String::from(value),
            None => entries.push((String::from(key), String::from(value))),
        }
    }

    /// Removes the first occurrence of `key` from `group` and returns
    /// whether it existed.
    pub fn remove(&mut self, group: &str, key: &str) -> bool {
        let Some(group_index) = self
            .groups
            .iter()
            .position(|(name, _)| name == group)
        else {
            return false;
        };
        let entries = &mut self.groups[group_index].1;
        let Some(entry_index) = entries.iter().position(|(k, _)| k == key) else {
            return false;
        };
        entries.remove(entry_index);
        true
    }

    /// Serializes the file in `g_key_file_to_data` layout: group
    /// headers, `key=value` lines and a blank line between groups.
    #[must_use]
    pub fn to_data(&self) -> String {
        let mut out = String::new();
        for (index, (name, entries)) in self.groups.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            out.push('[');
            out.push_str(name);
            out.push_str("]\n");
            for (key, value) in entries {
                out.push_str(key);
                out.push('=');
                out.push_str(value);
                out.push('\n');
            }
        }
        out
    }

    fn find_group(&self, group: &str) -> Option<&Vec<(String, String)>> {
        self.groups
            .iter()
            .find(|(name, _)| name == group)
            .map(|(_, entries)| entries)
    }
}

/// Resolves the backslash escapes understood by `GKeyFile` string
/// reads (`\s`, `\t`, `\r`, `\n`, `\\` and `\;`), returning `None` on
/// an unknown or dangling escape, like `g_key_file_get_string`.
fn unescape(value: &str) -> Option<String> {
    if !value.contains('\\') {
        return Some(String::from(value));
    }
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next()? {
            's' => out.push(' '),
            'n' => out.push('\n'),
            't' => out.push('\t'),
            'r' => out.push('\r'),
            '\\' => out.push('\\'),
            ';' => out.push(';'),
            _ => return None,
        }
    }
    Some(out)
}

/// Splits a list value on `;`, treating `\x` as a single escaped
/// character that cannot separate elements.
fn split_escaped_list(value: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            current.push(c);
            if let Some(next) = chars.next() {
                current.push(next);
            }
            continue;
        }
        if c == ';' {
            parts.push(core::mem::take(&mut current));
            continue;
        }
        current.push(c);
    }
    parts.push(current);
    parts
}

/// A decoded D-Bus variant payload, mirroring the subset of the wire
/// format used by portal options.
#[derive(Clone, Debug, PartialEq)]
pub enum PortalValue {
    /// A `BOOLEAN` (`b`) value.
    Bool(bool),
    /// A `BYTE` (`y`) value.
    Byte(u8),
    /// An `INT16` (`n`) value.
    I16(i16),
    /// A `UINT16` (`q`) value.
    U16(u16),
    /// An `INT32` (`i`) value.
    I32(i32),
    /// A `UINT32` (`u`) value.
    U32(u32),
    /// An `INT64` (`x`) value.
    I64(i64),
    /// A `UINT64` (`t`) value.
    U64(u64),
    /// A `DOUBLE` (`d`) value.
    F64(f64),
    /// A `STRING` (`s`) value.
    Str(String),
    /// An `OBJECT_PATH` (`o`) value.
    ObjectPath(String),
    /// A `SIGNATURE` (`g`) value.
    Signature(String),
    /// A `UNIX_FD` (`h`) value; the number indexes the message file
    /// descriptor list.
    Handle(u32),
    /// An `ARRAY` (`a`) value with its element signature; empty arrays
    /// keep the element type of the encoded signature.
    Array(String, Vec<PortalValue>),
    /// A `STRUCT` (`(...)`) value.
    Struct(Vec<PortalValue>),
    /// A `DICT_ENTRY` (`{...}`) value as found inside arrays.
    DictEntry(Box<PortalValue>, Box<PortalValue>),
    /// A `VARIANT` (`v`) value wrapping another value.
    Variant(Box<PortalValue>),
}

impl PortalValue {
    /// Builds a string value.
    #[must_use]
    pub fn string(value: impl Into<String>) -> Self {
        Self::Str(value.into())
    }

    /// Builds an `as` array of strings.
    #[must_use]
    pub fn string_array<I, S>(values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Array(
            String::from("s"),
            values
                .into_iter()
                .map(Into::into)
                .map(Self::Str)
                .collect(),
        )
    }

    /// Builds an array of an arbitrary element type after validating
    /// the element signature and every element against it.
    ///
    /// # Errors
    ///
    /// Returns [`PortalError::InvalidArgument`] when
    /// `element_signature` is not a single valid D-Bus type or an
    /// element does not match it.
    pub fn new_array(element_signature: &str, items: Vec<PortalValue>) -> Result<Self, PortalError> {
        validate_array_element_type(element_signature)
            .map_err(|err| PortalError::InvalidArgument(err.to_string()))?;
        if items
            .iter()
            .any(|item| !item.matches_one(element_signature))
        {
            return Err(PortalError::InvalidArgument(format!(
                "array element does not match signature {element_signature}"
            )));
        }
        Ok(Self::Array(String::from(element_signature), items))
    }

    /// Returns the complete D-Bus type signature of this value.
    #[must_use]
    pub fn signature(&self) -> String {
        match self {
            Self::Bool(_) => String::from("b"),
            Self::Byte(_) => String::from("y"),
            Self::I16(_) => String::from("n"),
            Self::U16(_) => String::from("q"),
            Self::I32(_) => String::from("i"),
            Self::U32(_) => String::from("u"),
            Self::I64(_) => String::from("x"),
            Self::U64(_) => String::from("t"),
            Self::F64(_) => String::from("d"),
            Self::Str(_) => String::from("s"),
            Self::ObjectPath(_) => String::from("o"),
            Self::Signature(_) => String::from("g"),
            Self::Handle(_) => String::from("h"),
            Self::Array(element, _) => format!("a{element}"),
            Self::Struct(fields) => {
                let mut out = String::from("(");
                for field in fields {
                    out.push_str(&field.signature());
                }
                out.push(')');
                out
            }
            Self::DictEntry(key, value) => {
                format!("{{{}{}}}", key.signature(), value.signature())
            }
            Self::Variant(_) => String::from("v"),
        }
    }

    /// Returns whether this value encodes exactly the type described
    /// by `signature`.
    #[must_use]
    pub fn matches_signature(&self, signature: &str) -> bool {
        match single_complete_type_len(signature) {
            Some(len) if len == signature.len() => self.matches_one(signature),
            _ => false,
        }
    }

    /// Returns whether this value encodes `signature`, which must be
    /// one complete D-Bus type.
    fn matches_one(&self, signature: &str) -> bool {
        match self {
            Self::Bool(_) => signature == "b",
            Self::Byte(_) => signature == "y",
            Self::I16(_) => signature == "n",
            Self::U16(_) => signature == "q",
            Self::I32(_) => signature == "i",
            Self::U32(_) => signature == "u",
            Self::I64(_) => signature == "x",
            Self::U64(_) => signature == "t",
            Self::F64(_) => signature == "d",
            Self::Str(_) => signature == "s",
            Self::ObjectPath(_) => signature == "o",
            Self::Signature(_) => signature == "g",
            Self::Handle(_) => signature == "h",
            Self::Variant(_) => signature == "v",
            Self::Array(element, items) => match signature.strip_prefix('a') {
                Some(rest) if rest == element => items.iter().all(|item| item.matches_one(element)),
                _ => false,
            },
            Self::Struct(fields) => {
                if !signature.starts_with('(') || !signature.ends_with(')') {
                    return false;
                }
                let mut iter = SignatureIter::new(&signature[1..signature.len() - 1]);
                for field in fields {
                    match iter.next() {
                        Some(child) if field.matches_one(child) => {}
                        _ => return false,
                    }
                }
                iter.next().is_none()
            }
            Self::DictEntry(key, value) => {
                if !signature.starts_with('{') || !signature.ends_with('}') {
                    return false;
                }
                let mut iter = SignatureIter::new(&signature[1..signature.len() - 1]);
                match (iter.next(), iter.next(), iter.next()) {
                    (Some(key_sig), Some(value_sig), None) => {
                        key.matches_one(key_sig) && value.matches_one(value_sig)
                    }
                    _ => false,
                }
            }
        }
    }

    /// Decodes one value of `signature` from `reader`.
    ///
    /// # Errors
    ///
    /// Returns a `DbusError` when `signature` is not a single valid
    /// type or the payload ends early.
    pub fn decode(reader: &mut DbusReader<'_>, signature: &str) -> DbusResult<Self> {
        validate_single_type(signature)?;
        Self::decode_one(reader, signature)
    }

    /// Decodes the payload of a `VARIANT` value from `reader`,
    /// including its signature header.
    ///
    /// # Errors
    ///
    /// Returns a `DbusError` when the header is malformed or the
    /// payload ends early.
    pub fn decode_variant(reader: &mut DbusReader<'_>) -> DbusResult<Self> {
        let signature = reader.read_variant_signature()?;
        Self::decode_one(reader, signature)
    }

    fn decode_one(reader: &mut DbusReader<'_>, signature: &str) -> DbusResult<Self> {
        let value = match signature.as_bytes()[0] {
            b'y' => Self::Byte(reader.read_u8()?),
            b'b' => Self::Bool(reader.read_bool()?),
            b'n' => Self::I16(reader.read_i16()?),
            b'q' => Self::U16(reader.read_u16()?),
            b'i' => Self::I32(reader.read_i32()?),
            b'u' => Self::U32(reader.read_u32()?),
            b'x' => Self::I64(reader.read_i64()?),
            b't' => Self::U64(reader.read_u64()?),
            b'd' => Self::F64(reader.read_f64()?),
            b's' => Self::Str(reader.read_str()?.to_string()),
            b'o' => Self::ObjectPath(reader.read_object_path()?.to_string()),
            b'g' => Self::Signature(reader.read_signature()?.to_string()),
            b'h' => Self::Handle(reader.read_fd()?),
            b'v' => Self::Variant(Box::new(Self::decode_variant(reader)?)),
            b'a' => {
                let element = &signature[1..];
                let code = element
                    .as_bytes()
                    .first()
                    .copied()
                    .unwrap_or_default();
                let alignment = type_alignment(code).ok_or_else(|| {
                    codevar_dbus::DbusError::invalid_signature(format!("invalid element type code: {code}"))
                })?;
                let mut array = reader.read_array(alignment)?;
                let mut items = Vec::new();
                while !array.is_empty() {
                    items.push(Self::decode_one(&mut array, element)?);
                }
                Self::Array(element.to_string(), items)
            }
            b'(' => {
                reader.read_struct()?;
                let mut fields = Vec::new();
                for field in SignatureIter::new(&signature[1..signature.len() - 1]) {
                    fields.push(Self::decode_one(reader, field)?);
                }
                Self::Struct(fields)
            }
            b'{' => {
                reader.read_struct()?;
                let mut iter = SignatureIter::new(&signature[1..signature.len() - 1]);
                let key_signature = iter.next().ok_or_else(|| {
                    codevar_dbus::DbusError::invalid_signature("dict entry without key type")
                })?;
                let value_signature = iter.next().ok_or_else(|| {
                    codevar_dbus::DbusError::invalid_signature("dict entry without value type")
                })?;
                let key = Self::decode_one(reader, key_signature)?;
                let value = Self::decode_one(reader, value_signature)?;
                Self::DictEntry(Box::new(key), Box::new(value))
            }
            code => {
                return Err(codevar_dbus::DbusError::invalid_signature(format!(
                    "invalid type code: {code}"
                )));
            }
        };
        Ok(value)
    }
}

/// Writes `value` through `writer`.
pub(crate) fn write_value<W: ValueWriter>(writer: &mut W, value: &PortalValue) -> DbusResult<()> {
    match value {
        PortalValue::Bool(v) => writer.write_bool(*v),
        PortalValue::Byte(v) => writer.write_u8(*v),
        PortalValue::I16(v) => writer.write_i16(*v),
        PortalValue::U16(v) => writer.write_u16(*v),
        PortalValue::I32(v) => writer.write_i32(*v),
        PortalValue::U32(v) => writer.write_u32(*v),
        PortalValue::I64(v) => writer.write_i64(*v),
        PortalValue::U64(v) => writer.write_u64(*v),
        PortalValue::F64(v) => writer.write_f64(*v),
        PortalValue::Str(v) => writer.write_str(v),
        PortalValue::ObjectPath(v) => writer.write_object_path(v),
        PortalValue::Signature(v) => writer.write_signature(v),
        PortalValue::Handle(v) => writer.write_fd(*v),
        PortalValue::Array(element, items) => writer.write_array(element, |inner| {
            for item in items {
                write_value(inner, item)?;
            }
            Ok(())
        }),
        PortalValue::Struct(fields) => {
            let mut fields_signature = String::new();
            for field in fields {
                fields_signature.push_str(&field.signature());
            }
            writer.write_struct(&fields_signature, |inner| {
                for field in fields {
                    write_value(inner, field)?;
                }
                Ok(())
            })
        }
        PortalValue::DictEntry(key, entry_value) => {
            let mut fields_signature = key.signature();
            fields_signature.push_str(&entry_value.signature());
            writer.write_struct(&fields_signature, |inner| {
                write_value(inner, key)?;
                write_value(inner, entry_value)
            })
        }
        PortalValue::Variant(inner) => {
            let signature = inner.signature();
            writer.write_variant(&signature, |child| write_value(child, inner))
        }
    }
}

/// The write operations shared by [`BodyWriter`] and [`DbusWriter`]
/// so option maps can be encoded into full messages and into raw
/// bodies alike.
trait ValueWriter {
    /// Appends a `BOOLEAN`.
    fn write_bool(&mut self, value: bool) -> DbusResult<()>;
    /// Appends a `BYTE`.
    fn write_u8(&mut self, value: u8) -> DbusResult<()>;
    /// Appends an `INT16`.
    fn write_i16(&mut self, value: i16) -> DbusResult<()>;
    /// Appends a `UINT16`.
    fn write_u16(&mut self, value: u16) -> DbusResult<()>;
    /// Appends an `INT32`.
    fn write_i32(&mut self, value: i32) -> DbusResult<()>;
    /// Appends a `UINT32`.
    fn write_u32(&mut self, value: u32) -> DbusResult<()>;
    /// Appends an `INT64`.
    fn write_i64(&mut self, value: i64) -> DbusResult<()>;
    /// Appends a `UINT64`.
    fn write_u64(&mut self, value: u64) -> DbusResult<()>;
    /// Appends a `DOUBLE`.
    fn write_f64(&mut self, value: f64) -> DbusResult<()>;
    /// Appends a `UNIX_FD`.
    fn write_fd(&mut self, index: u32) -> DbusResult<()>;
    /// Appends a `STRING`.
    fn write_str(&mut self, value: &str) -> DbusResult<()>;
    /// Appends an `OBJECT_PATH`.
    fn write_object_path(&mut self, value: &str) -> DbusResult<()>;
    /// Appends a `SIGNATURE`.
    fn write_signature(&mut self, value: &str) -> DbusResult<()>;
    /// Appends an array of `element_signature` around `body`.
    fn write_array<F>(&mut self, element_signature: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>;
    /// Appends a struct of `fields_signature` around `body`.
    fn write_struct<F>(&mut self, fields_signature: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>;
    /// Appends a variant of `signature` around `body`.
    fn write_variant<F>(&mut self, signature: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>;
}

impl ValueWriter for BodyWriter {
    fn write_bool(&mut self, value: bool) -> DbusResult<()> {
        BodyWriter::write_bool(self, value)
    }

    fn write_u8(&mut self, value: u8) -> DbusResult<()> {
        BodyWriter::write_u8(self, value)
    }

    fn write_i16(&mut self, value: i16) -> DbusResult<()> {
        BodyWriter::write_i16(self, value)
    }

    fn write_u16(&mut self, value: u16) -> DbusResult<()> {
        BodyWriter::write_u16(self, value)
    }

    fn write_i32(&mut self, value: i32) -> DbusResult<()> {
        BodyWriter::write_i32(self, value)
    }

    fn write_u32(&mut self, value: u32) -> DbusResult<()> {
        BodyWriter::write_u32(self, value)
    }

    fn write_i64(&mut self, value: i64) -> DbusResult<()> {
        BodyWriter::write_i64(self, value)
    }

    fn write_u64(&mut self, value: u64) -> DbusResult<()> {
        BodyWriter::write_u64(self, value)
    }

    fn write_f64(&mut self, value: f64) -> DbusResult<()> {
        BodyWriter::write_f64(self, value)
    }

    fn write_fd(&mut self, index: u32) -> DbusResult<()> {
        BodyWriter::write_fd(self, index)
    }

    fn write_str(&mut self, value: &str) -> DbusResult<()> {
        BodyWriter::write_str(self, value)
    }

    fn write_object_path(&mut self, value: &str) -> DbusResult<()> {
        BodyWriter::write_object_path(self, value)
    }

    fn write_signature(&mut self, value: &str) -> DbusResult<()> {
        BodyWriter::write_signature(self, value)
    }

    fn write_array<F>(&mut self, element_signature: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        BodyWriter::write_array(self, element_signature, body)
    }

    fn write_struct<F>(&mut self, fields_signature: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        BodyWriter::write_struct(self, fields_signature, body)
    }

    fn write_variant<F>(&mut self, signature: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        BodyWriter::write_variant(self, signature, body)
    }
}

impl ValueWriter for DbusWriter {
    fn write_bool(&mut self, value: bool) -> DbusResult<()> {
        DbusWriter::write_bool(self, value);
        Ok(())
    }

    fn write_u8(&mut self, value: u8) -> DbusResult<()> {
        DbusWriter::write_u8(self, value);
        Ok(())
    }

    fn write_i16(&mut self, value: i16) -> DbusResult<()> {
        DbusWriter::write_i16(self, value);
        Ok(())
    }

    fn write_u16(&mut self, value: u16) -> DbusResult<()> {
        DbusWriter::write_u16(self, value);
        Ok(())
    }

    fn write_i32(&mut self, value: i32) -> DbusResult<()> {
        DbusWriter::write_i32(self, value);
        Ok(())
    }

    fn write_u32(&mut self, value: u32) -> DbusResult<()> {
        DbusWriter::write_u32(self, value);
        Ok(())
    }

    fn write_i64(&mut self, value: i64) -> DbusResult<()> {
        DbusWriter::write_i64(self, value);
        Ok(())
    }

    fn write_u64(&mut self, value: u64) -> DbusResult<()> {
        DbusWriter::write_u64(self, value);
        Ok(())
    }

    fn write_f64(&mut self, value: f64) -> DbusResult<()> {
        DbusWriter::write_f64(self, value);
        Ok(())
    }

    fn write_fd(&mut self, index: u32) -> DbusResult<()> {
        DbusWriter::write_fd(self, index);
        Ok(())
    }

    fn write_str(&mut self, value: &str) -> DbusResult<()> {
        DbusWriter::write_str(self, value)
    }

    fn write_object_path(&mut self, value: &str) -> DbusResult<()> {
        DbusWriter::write_object_path(self, value)
    }

    fn write_signature(&mut self, value: &str) -> DbusResult<()> {
        DbusWriter::write_signature(self, value)
    }

    fn write_array<F>(&mut self, element_signature: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        validate_array_element_type(element_signature)?;
        let alignment = DbusWriter::first_type_alignment(element_signature)?;
        DbusWriter::write_array(self, alignment, body)
    }

    fn write_struct<F>(&mut self, fields_signature: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        validate_signature(fields_signature)?;
        DbusWriter::write_struct(self, body)
    }

    fn write_variant<F>(&mut self, signature: &str, body: F) -> DbusResult<()>
    where
        F: FnOnce(&mut Self) -> DbusResult<()>,
    {
        DbusWriter::write_variant(self, signature, body)
    }
}

/// Encodes `options` as an `a{sv}` array inside a message body,
/// recording the signature `a{sv}` on `writer`.
///
/// # Errors
///
/// Returns a `DbusError` when an option cannot be encoded, e.g. an
/// object path or string that fails validation.
pub fn encode_options(writer: &mut BodyWriter, options: &OptionMap) -> DbusResult<()> {
    writer.write_array("{sv}", |inner| {
        for (key, value) in options {
            inner.write_struct("sv", |entry| {
                entry.write_str(key)?;
                let signature = value.signature();
                entry.write_variant(&signature, |slot| write_value(slot, value))
            })?;
        }
        Ok(())
    })
}

/// Encodes `options` as an `a{sv}` array into a raw body writer, e.g.
/// when building a standalone body with an explicit signature.
///
/// # Errors
///
/// Returns a `DbusError` when an option cannot be encoded, e.g. an
/// object path or string that fails validation.
pub fn encode_options_raw(writer: &mut DbusWriter, options: &OptionMap) -> DbusResult<()> {
    <DbusWriter as ValueWriter>::write_array(writer, "{sv}", |inner| {
        for (key, value) in options {
            <DbusWriter as ValueWriter>::write_struct(inner, "sv", |entry| {
                entry.write_str(key)?;
                let signature = value.signature();
                entry.write_variant(&signature, |slot| write_value(slot, value))
            })?;
        }
        Ok(())
    })
}

/// Decodes an `a{sv}` array from `reader`, which must be positioned
/// at the start of the array.
///
/// # Errors
///
/// Returns a `DbusError` when the payload is not a well-formed
/// `a{sv}` array.
pub fn decode_options(reader: &mut DbusReader<'_>) -> DbusResult<OptionMap> {
    let mut array = reader.read_array(8)?;
    let mut options = OptionMap::new();
    while !array.is_empty() {
        array.read_struct()?;
        let key = array.read_str()?.to_string();
        let value = PortalValue::decode_variant(&mut array)?;
        options.insert(key, value);
    }
    Ok(options)
}

/// Semantic validation callback for one supported option key, called
/// with the key, its value and the full incoming option map.
pub type OptionKeyValidate = fn(&str, &PortalValue, &OptionMap) -> Result<(), PortalError>;

/// One supported option key, mirroring `XdpOptionKey` in the C
/// reference.
pub struct OptionKey {
    /// Name of the option as it appears inside `a{sv}`.
    pub key: &'static str,
    /// D-Bus type the value must have, e.g. `b` or `a{sv}`.
    pub type_signature: &'static str,
    /// Optional semantic validation run after the type check.
    pub validate: Option<OptionKeyValidate>,
}

impl OptionKey {
    /// Creates a supported key that only checks the value type.
    #[must_use]
    pub const fn new(key: &'static str, type_signature: &'static str) -> Self {
        Self {
            key,
            type_signature,
            validate: None,
        }
    }

    /// Creates a supported key with an extra validation callback.
    #[must_use]
    pub const fn with_validate(
        key: &'static str,
        type_signature: &'static str,
        validate: OptionKeyValidate,
    ) -> Self {
        Self {
            key,
            type_signature,
            validate: Some(validate),
        }
    }
}

/// Extracts the supported options from `options`, validating types
/// and running per-key callbacks like `xdp_filter_options`.
///
/// Unknown keys are ignored; the first validation failure is kept and
/// returned after every supported key has been visited.
///
/// # Errors
///
/// Returns [`PortalError::InvalidArgument`] when a supported key
/// holds the wrong type, or the error produced by a validation
/// callback.
pub fn filter_options(options: &OptionMap, supported: &[OptionKey]) -> Result<OptionMap, PortalError> {
    let mut filtered = OptionMap::new();
    let mut first_error: Option<PortalError> = None;
    for supported_key in supported {
        let Some(value) = options.get(supported_key.key) else {
            continue;
        };
        if !value.matches_signature(supported_key.type_signature) {
            if first_error.is_none() {
                first_error = Some(PortalError::InvalidArgument(format!(
                    "Expected type '{}' for option '{}', got '{}'",
                    supported_key.type_signature,
                    supported_key.key,
                    value.signature()
                )));
            }
            continue;
        }
        if let Some(validate) = supported_key.validate
            && let Err(error) = validate(supported_key.key, value, options)
        {
            if first_error.is_none() {
                first_error = Some(error);
            }
            continue;
        }
        filtered.insert(String::from(supported_key.key), value.clone());
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(filtered),
    }
}

/// Extracts and filters options directly from a `DbusReader`.
///
/// This decodes the `a{sv}` array from the reader and then applies
/// [`filter_options`] to validate and filter the options.
///
/// # Errors
///
/// Returns [`PortalError::InvalidArgument`] when the options cannot
/// be decoded, or when a supported key holds the wrong type.
pub fn filter_options_from_reader(
    reader: &mut DbusReader<'_>,
    supported: &[OptionKey],
) -> Result<OptionMap, PortalError> {
    let options = decode_options(reader)?;
    filter_options(&options, supported)
}

/// Reads the environment variable `name`.
///
/// Returns `None` when the variable is unset or the target does not
/// provide an environment (wasm and non-unix builds).
#[must_use]
pub fn env_var(name: &str) -> Option<String> {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        if name.contains('\0') {
            return None;
        }
        // SAFETY: `name` is a valid NUL-terminated C string; `getenv`
        // returns either null or a pointer to a string that outlives
        // this call, which is copied before returning.
        let ptr = unsafe { libc::getenv(name.as_ptr().cast::<libc::c_char>()) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: the pointer is non-null and stays valid while no
        // environment mutation happens; it is copied immediately.
        let cstr = unsafe { core::ffi::CStr::from_ptr(ptr) };
        cstr.to_str().ok().map(String::from)
    }
    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        let _ = name;
        None
    }
}

/// Fills `buffer` with random bytes from the kernel.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when `getrandom` reports an error
/// other than `EINTR`.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn fill_random(buffer: &mut [u8]) -> Result<(), PortalError> {
    let mut offset = 0;
    while offset < buffer.len() {
        // SAFETY: `buffer[offset..]` is a valid writable slice and the
        // kernel writes at most its length into it.
        let written = unsafe {
            libc::getrandom(
                buffer[offset..]
                    .as_mut_ptr()
                    .cast::<libc::c_void>(),
                buffer.len() - offset,
                0,
            )
        };
        if written < 0 {
            // SAFETY: `getrandom` reported an error, so `errno` is set.
            let errno = unsafe { *libc::__errno_location() };
            if errno == libc::EINTR {
                continue;
            }
            return Err(PortalError::Failed(format!(
                "failed to get random data: errno {errno}"
            )));
        }
        match usize::try_from(written) {
            Ok(0) | Err(_) => {
                return Err(PortalError::Failed(String::from(
                    "failed to get random data: short read",
                )));
            }
            Ok(count) => offset += count,
        }
    }
    Ok(())
}

/// Random bytes are unavailable outside unix targets.
#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
fn fill_random(buffer: &mut [u8]) -> Result<(), PortalError> {
    let _ = buffer;
    Err(PortalError::Failed(String::from(
        "random data unavailable on this target",
    )))
}

/// Encodes bytes as D-Bus-path-safe base64 (`[A-Za-z0-9_]`, no
/// padding): `+` and `/` alias to `_` and encoding stops at the first
/// `=`.
fn encode_base64_for_dbus(data: &[u8]) -> String {
    let mut encoded = codevar_base::basic_base64::encode(data);
    if let Some(padding) = encoded.find('=') {
        encoded.truncate(padding);
    }
    encoded.replace(['+', '/'], "_")
}

/// Generates a 16-byte request token as D-Bus-path-safe base64.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the kernel cannot provide
/// random bytes.
pub fn generate_token() -> Result<String, PortalError> {
    let mut bytes = [0u8; 16];
    fill_random(&mut bytes)?;
    Ok(encode_base64_for_dbus(&bytes))
}

/// Generates a 16-byte document or session key as D-Bus-path-safe
/// base64.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the kernel cannot provide
/// random bytes.
pub fn generate_key() -> Result<String, PortalError> {
    let mut bytes = [0u8; 16];
    fill_random(&mut bytes)?;
    Ok(encode_base64_for_dbus(&bytes))
}

/// Returns whether `arg` needs shell quoting.
const fn shell_needs_quoting(arg: &str) -> bool {
    let bytes = arg.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let c = bytes[index];
        let safe =
            c.is_ascii_alphanumeric() || matches!(c, b'-' | b'/' | b'~' | b':' | b'.' | b'_' | b'=' | b'@');
        if !safe {
            return true;
        }
        index += 1;
    }
    false
}

/// Quotes `arg` with single quotes, escaping embedded quotes the way
/// `g_shell_quote` does.
#[must_use]
pub fn shell_quote(arg: &str) -> String {
    let mut out = String::with_capacity(arg.len() + 2);
    out.push('\'');
    for c in arg.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// Quotes `arg` only when `quote_escape` is set and the value needs
/// it, like `xdp_maybe_quote`.
#[must_use]
pub fn maybe_quote(arg: &str, quote_escape: bool) -> String {
    if quote_escape && shell_needs_quoting(arg) {
        shell_quote(arg)
    } else {
        String::from(arg)
    }
}

/// Joins `args` with spaces, quoting each argument like
/// `xdp_maybe_quote_argv`.
#[must_use]
pub fn maybe_quote_argv(args: &[&str], quote_escape: bool) -> String {
    let mut out = String::new();
    for (index, arg) in args.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(&maybe_quote(arg, quote_escape));
    }
    out
}

/// Splits a command line into words following shell quoting rules,
/// the way `g_shell_parse_argv` does: whitespace separates words,
/// single quotes are literal, double quotes honour backslash escapes
/// for `` ` ``, `"`, `$`, `\` and newlines, and other backslashes
/// escape the next character outside quotes.
///
/// # Errors
///
/// Returns [`PortalError::InvalidArgument`] when a quote is left
/// unterminated or the command line is empty.
pub fn shell_parse_argv(command: &str) -> Result<Vec<String>, PortalError> {
    #[derive(PartialEq)]
    enum State {
        Unquoted,
        SingleQuoted,
        DoubleQuoted,
    }

    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut escaped = false;
    let mut state = State::Unquoted;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        match state {
            State::Unquoted => {
                if escaped {
                    current.push(c);
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                    in_word = true;
                } else if c == '\'' {
                    state = State::SingleQuoted;
                    in_word = true;
                } else if c == '"' {
                    state = State::DoubleQuoted;
                    in_word = true;
                } else if c.is_whitespace() {
                    if in_word {
                        words.push(core::mem::take(&mut current));
                        in_word = false;
                    }
                } else {
                    current.push(c);
                    in_word = true;
                }
            }
            State::SingleQuoted => {
                if c == '\'' {
                    state = State::Unquoted;
                } else {
                    current.push(c);
                }
            }
            State::DoubleQuoted => {
                if c == '"' {
                    state = State::Unquoted;
                } else if c == '\\' {
                    match chars.peek() {
                        Some(&next) if matches!(next, '`' | '"' | '$' | '\\') => {
                            chars.next();
                            current.push(next);
                        }
                        Some(&'\n') => {
                            chars.next();
                        }
                        _ => current.push('\\'),
                    }
                } else {
                    current.push(c);
                }
            }
        }
    }
    if state != State::Unquoted {
        return Err(PortalError::InvalidArgument(String::from("unmatched quote")));
    }
    if escaped {
        current.push('\\');
        in_word = true;
    }
    if in_word {
        words.push(current);
    }
    if words.is_empty() {
        return Err(PortalError::InvalidArgument(String::from("empty command line")));
    }
    Ok(words)
}

/// Cached path of the document portal FUSE mount, set once after the
/// portal announces it.
static DOCUMENTS_MOUNTPOINT: Mutex<Option<String>> = Mutex::new(None);

/// Stores the mount point reported by the document portal's
/// `GetMountPoint` method; `None` clears it.
pub fn set_documents_mountpoint(path: Option<&str>) {
    let mut slot = DOCUMENTS_MOUNTPOINT.lock();
    *slot = path.map(String::from);
}

/// Returns the cached document portal mount point.
#[must_use]
pub fn documents_mountpoint() -> Option<String> {
    DOCUMENTS_MOUNTPOINT.lock().clone()
}

/// Converts `path` into its `/by-app/{app_id}` alias when it lives
/// inside the document portal mount, like
/// `xdp_get_alternate_document_path`.
#[must_use]
pub fn get_alternate_document_path(path: &str, app_id: &str) -> Option<String> {
    if app_id.is_empty() {
        return None;
    }
    let mountpoint = documents_mountpoint()?;
    if !path.starts_with(&mountpoint) {
        return None;
    }
    let rest = &path[mountpoint.len()..];
    let rest = rest.strip_prefix('/')?;
    Some(format!("{mountpoint}/by-app/{app_id}/{rest}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_dbus::{ByteOrder, DbusReader, DbusWriter};

    // unwrap() calls in tests are safe: every one operates on values
    // constructed by the code under test with known-good inputs.

    #[test]
    fn validates_app_ids_like_the_c_implementation() {
        assert!(is_valid_app_id("org.gnome.Calculator"));
        assert!(is_valid_app_id("1234.5"));
        assert!(is_valid_app_id("com.example.app-2"));
        assert!(is_valid_app_id("snap.foo"));
        assert!(!is_valid_app_id(""));
        assert!(!is_valid_app_id(".hidden"));
        assert!(!is_valid_app_id("trailing."));
        assert!(!is_valid_app_id("double..dot"));
        assert!(!is_valid_app_id("no-dot"));
        assert!(!is_valid_app_id("a.b-early.dash"));
        assert!(!is_valid_app_id(&format!("x.{}", "a".repeat(256))));
    }

    #[test]
    fn validates_tokens_and_filenames() {
        assert!(is_valid_token("handle_1"));
        assert!(is_valid_token("1"));
        assert!(!is_valid_token(""));
        assert!(!is_valid_token("has-dash"));
        assert!(!is_valid_token("has/slash"));
        assert!(is_valid_filename("report.pdf"));
        assert!(!is_valid_filename(""));
        assert!(!is_valid_filename("."));
        assert!(!is_valid_filename(".."));
        assert!(!is_valid_filename("a/b"));
    }

    #[test]
    fn parses_key_files_and_round_trips() {
        let file = KeyFile::parse(
            "# comment\n[portal]\nDBusName=org.example.Portal\nInterfaces=a.b;c.d;\n\n# other\n[extra]\nflag=true\n",
        )
        .unwrap();
        assert!(file.has_group("portal"));
        assert_eq!(
            file.get("portal", "DBusName").as_deref(),
            Some("org.example.Portal")
        );
        assert_eq!(
            file.list("portal", "Interfaces").unwrap(),
            Vec::from([String::from("a.b"), String::from("c.d")])
        );
        assert!(file.boolean("extra", "flag", false));
        assert!(!file.boolean("extra", "missing", false));
        assert_eq!(file.get("portal", "missing"), None);

        let mut edited = file.clone();
        edited.set("portal", "DBusName", "org.example.Other");
        edited.set("portal", "UseIn", "gnome");
        edited.set("new", "k", "v");
        assert_eq!(
            edited.get("portal", "DBusName").as_deref(),
            Some("org.example.Other")
        );
        assert!(edited.remove("portal", "UseIn"));
        assert!(!edited.remove("portal", "UseIn"));
        assert!(edited.has_group("new"));

        let text = edited.to_data();
        let reparsed = KeyFile::parse(&text).unwrap();
        assert_eq!(reparsed, edited);
    }

    #[test]
    fn rejects_malformed_key_files() {
        assert!(KeyFile::parse("DBusName=x\n").is_err());
        assert!(KeyFile::parse("[portal]\nnovalue\n").is_err());
        assert!(KeyFile::parse("[portal\n").is_err());
        assert!(KeyFile::parse("[]\n").is_err());
        assert!(KeyFile::parse("[g]\n;semicolon comment\n").is_err());
        assert!(KeyFile::parse("[g]\n=v\n").is_err());
    }

    #[test]
    fn unescapes_glib_string_sequences() {
        let file = KeyFile::parse("[g]\na=one\\stwo\\nline\\;semi\\\\slash\n").unwrap();
        assert_eq!(file.get("g", "a").unwrap(), "one two\nline;semi\\slash");

        let invalid = KeyFile::parse("[g]\na=x\\qy\n").unwrap();
        assert_eq!(invalid.get("g", "a"), None);
        assert_eq!(invalid.list("g", "a"), None);

        let escaped = KeyFile::parse("[g]\na=x\\;y;z;\n").unwrap();
        assert_eq!(
            escaped.list("g", "a").unwrap(),
            Vec::from([String::from("x;y"), String::from("z")])
        );

        let empty = KeyFile::parse("[g]\na=\n").unwrap();
        assert!(empty.list("g", "a").unwrap().is_empty());

        let merged = KeyFile::parse("[g]\na=1\n[g]\nb=2\n").unwrap();
        assert_eq!(merged.get("g", "a").as_deref(), Some("1"));
        assert_eq!(merged.get("g", "b").as_deref(), Some("2"));

        let duplicate = KeyFile::parse("[g]\nk=first\nk=last\n").unwrap();
        assert_eq!(duplicate.get("g", "k").as_deref(), Some("last"));
        assert_eq!(duplicate.entries("g").unwrap().len(), 2);
    }

    #[test]
    fn checks_option_value_signatures() {
        let value = PortalValue::Bool(true);
        assert!(value.matches_signature("b"));
        assert!(!value.matches_signature("s"));
        assert!(!value.matches_signature("bs"));

        let empty = PortalValue::Array(String::from("s"), Vec::new());
        assert!(empty.matches_signature("as"));
        assert!(!empty.matches_signature("ai"));

        let nested = PortalValue::new_array(
            "{sv}",
            Vec::from([PortalValue::DictEntry(
                Box::new(PortalValue::string("k")),
                Box::new(PortalValue::Variant(Box::new(PortalValue::U32(1)))),
            )]),
        )
        .unwrap();
        assert!(nested.matches_signature("a{sv}"));
        assert_eq!(nested.signature(), "a{sv}");

        assert!(PortalValue::string_array(Vec::from(["a", "b"])).matches_signature("as"));
        let error = PortalValue::new_array("s", Vec::from([PortalValue::Byte(1)]));
        assert!(matches!(error, Err(PortalError::InvalidArgument(_))));
    }

    #[test]
    fn encodes_and_decodes_option_maps() {
        let mut options = OptionMap::new();
        options.insert(String::from("interactive"), PortalValue::Bool(true));
        options.insert(String::from("token"), PortalValue::string("abc"));
        options.insert(
            String::from("handle"),
            PortalValue::ObjectPath(String::from("/h/1")),
        );
        options.insert(String::from("count"), PortalValue::U32(7));
        options.insert(String::from("names"), PortalValue::string_array(Vec::from(["x"])));
        options.insert(
            String::from("wrapped"),
            PortalValue::Variant(Box::new(PortalValue::F64(0.5))),
        );

        let mut body = BodyWriter::new(ByteOrder::Little);
        encode_options(&mut body, &options).unwrap();
        let (bytes, signature) = body.into_parts();
        assert_eq!(signature, "a{sv}");

        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        let decoded = decode_options(&mut reader).unwrap();
        assert!(reader.is_empty());
        assert_eq!(decoded, options);
    }

    #[test]
    fn encodes_options_into_raw_bodies() {
        let mut options = OptionMap::new();
        options.insert(String::from("s"), PortalValue::string("v"));
        let mut writer = DbusWriter::new(ByteOrder::Little);
        encode_options_raw(&mut writer, &options).unwrap();
        let bytes = writer.into_bytes();
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        let decoded = decode_options(&mut reader).unwrap();
        assert_eq!(decoded, options);
    }

    fn validate_flag(_key: &str, value: &PortalValue, _options: &OptionMap) -> Result<(), PortalError> {
        if *value == PortalValue::Bool(true) {
            Ok(())
        } else {
            Err(PortalError::InvalidArgument(String::from("flag must be true")))
        }
    }

    #[test]
    fn filters_options_like_xdp_filter_options() {
        let supported = [
            OptionKey::new("interactive", "b"),
            OptionKey::with_validate("flag", "b", validate_flag),
        ];

        let mut options = OptionMap::new();
        options.insert(String::from("interactive"), PortalValue::Bool(true));
        options.insert(String::from("unknown"), PortalValue::string("x"));
        let filtered = filter_options(&options, &supported).unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("interactive"));

        let mut wrong_type = OptionMap::new();
        wrong_type.insert(String::from("interactive"), PortalValue::string("nope"));
        let error = filter_options(&wrong_type, &supported).unwrap_err();
        assert_eq!(
            error.message(),
            "Expected type 'b' for option 'interactive', got 's'"
        );

        let mut invalid = OptionMap::new();
        invalid.insert(String::from("flag"), PortalValue::Bool(false));
        let error = filter_options(&invalid, &supported).unwrap_err();
        assert!(matches!(error, PortalError::InvalidArgument(_)));
        assert_eq!(error.message(), "flag must be true");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn generates_path_safe_tokens() {
        // unwrap() is safe: random generation only fails when the
        // kernel refuses to provide bytes, which does not happen in
        // the test environment.
        let token = generate_token().unwrap();
        let key = generate_key().unwrap();
        assert_eq!(token.len(), 22);
        assert_eq!(key.len(), 22);
        assert!(
            token
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        );
        assert!(
            key.bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        );
        assert_ne!(token, generate_key().unwrap());
    }

    #[test]
    fn parses_shell_command_lines() {
        assert_eq!(
            shell_parse_argv("gedit --new-window 'my file.txt'").unwrap(),
            Vec::from([
                String::from("gedit"),
                String::from("--new-window"),
                String::from("my file.txt")
            ])
        );
        assert_eq!(
            shell_parse_argv("a \"b c\" d\\ e").unwrap(),
            Vec::from([String::from("a"), String::from("b c"), String::from("d e")])
        );
        assert_eq!(
            shell_parse_argv("a 'it'\\''s'").unwrap(),
            Vec::from([String::from("a"), String::from("it's")])
        );
        assert!(shell_parse_argv("").is_err());
        assert!(shell_parse_argv("   ").is_err());
        assert!(shell_parse_argv("open \"unterminated").is_err());
    }

    #[test]
    fn quotes_only_when_needed() {
        assert_eq!(maybe_quote("plain", true), "plain");
        assert_eq!(maybe_quote("plain", false), "plain");
        assert_eq!(maybe_quote("needs quoting", true), "'needs quoting'");
        assert_eq!(maybe_quote("needs quoting", false), "needs quoting");
        assert_eq!(maybe_quote_argv(&["a b", "c"], true), "'a b' c");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn remaps_paths_inside_the_document_mount() {
        set_documents_mountpoint(Some("/run/user/1000/doc"));
        assert_eq!(documents_mountpoint().as_deref(), Some("/run/user/1000/doc"));
        assert_eq!(
            get_alternate_document_path("/run/user/1000/doc/abcd/file.txt", "org.example.App").as_deref(),
            Some("/run/user/1000/doc/by-app/org.example.App/abcd/file.txt")
        );
        assert_eq!(
            get_alternate_document_path("/etc/passwd", "org.example.App"),
            None
        );
        assert_eq!(get_alternate_document_path("/run/user/1000/doc/abcd/x", ""), None);
        set_documents_mountpoint(None);
        assert_eq!(documents_mountpoint(), None);
        assert_eq!(get_alternate_document_path("/run/user/1000/doc/x", "a.b"), None);
    }
}
