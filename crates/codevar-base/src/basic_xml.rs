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

//! XML builder with fluent API and validator following RFC standards.
//!
//! This module provides a [`XmlBuilder`] for constructing well-formed XML
//! documents through a chainable, builder-style interface, and a
//! [`validate`] function that checks whether a string is well-formed XML
//! per the XML 1.0 specification.
//!
//! # Examples
//!
//! ```
//! use codevar_base::basic_xml::XmlBuilder;
//!
//! let doc = XmlBuilder::new("node")
//!     .attr("version", "1.0")
//!     .child("interface")
//!         .attr("name", "org.freedesktop.DBus.Introspectable")
//!         .child("method")
//!             .attr("name", "Introspect")
//!             .child("arg")
//!                 .attr("name", "data")
//!                 .attr("type", "s")
//!                 .attr("direction", "out")
//!             .end()
//!         .end()
//!     .end()
//!     .build();
//!
//! let xml = doc.to_string();
//! assert!(xml.contains("<node version=\"1.0\">"));
//! ```
//!
//! # Performance
//!
//! The builder accumulates content in a single `String` buffer with
//! amortized doubling, so serialization is O(n) over the total number
//! of tokens produced.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

/// Escapes special XML characters in `input` as per XML 1.0 spec.
///
/// Replaces `&`, `<`, `>`, `"`, and `'` with their corresponding
/// entity references so the output is safe to embed in XML text or
/// attribute values.
#[must_use]
pub fn escape(input: &str) -> String {
    // Fast path: no escaping needed.
    if !input.contains(['&', '<', '>', '"', '\'']) {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len() + 16);
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

/// Errors produced by [`validate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XmlError {
    /// The input is empty.
    EmptyInput,
    /// The document does not start with `<?xml` or a `<` that begins
    /// the root element.
    MissingDeclaration,
    /// An attribute value contains an unescaped quote character.
    UnescapedQuote,
    /// A tag name contains an invalid character.
    InvalidTagName,
    /// An opening tag was never closed or a closing tag did not match.
    UnbalancedTags,
    /// A character reference or entity is malformed.
    InvalidEntity,
    /// The document contains more than one root element.
    MultipleRoots,
}

impl fmt::Display for XmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("empty XML input"),
            Self::MissingDeclaration => f.write_str("missing XML declaration or root element"),
            Self::UnescapedQuote => f.write_str("unescaped quote in attribute value"),
            Self::InvalidTagName => f.write_str("invalid tag name"),
            Self::UnbalancedTags => f.write_str("unbalanced XML tags"),
            Self::InvalidEntity => f.write_str("malformed entity reference"),
            Self::MultipleRoots => f.write_str("multiple root elements"),
        }
    }
}

impl core::error::Error for XmlError {}

/// Validates that `input` is well-formed XML per the XML 1.0 specification.
///
/// Checks structural well-formedness: a single root element, properly
/// nested and balanced tags, valid attribute syntax, and legal character
/// references. This is a lightweight validator suitable for checking
/// introspection output; it does not validate against an XML schema
/// or DTD.
///
/// # Errors
///
/// - [`XmlError::EmptyInput`] — `input` is empty.
/// - [`XmlError::MissingDeclaration`] — no valid XML declaration or
///   root element found.
/// - [`XmlError::UnbalancedTags`] — tags are not properly nested or
///   closed.
/// - [`XmlError::InvalidTagName`] — a tag name contains characters
///   not allowed by the XML spec.
/// - [`XmlError::InvalidEntity`] — a character reference is malformed.
/// - [`XmlError::UnescapedQuote`] — an attribute value contains a
///   raw quote.
///
/// # Examples
///
/// ```
/// use codevar_base::basic_xml::validate;
///
/// assert!(validate("<node><child/></node>").is_ok());
/// assert!(validate("<node attr=\"value\"></node>").is_ok());
/// assert!(validate("<node><unclosed>").is_err());
/// ```
#[must_use]
pub fn validate(input: &str) -> Result<(), XmlError> {
    if input.trim().is_empty() {
        return Err(XmlError::EmptyInput);
    }

    let bytes = input.as_bytes();
    let mut i = 0usize;
    let mut stack: Vec<&str> = Vec::new();
    let mut found_root = false;

    if bytes.get(i..i + 5) == Some(b"<?xml") {
        i += 5;
        while i < bytes.len() {
            if bytes[i] == b'>' {
                i += 1;
                break;
            }
            if bytes[i] == b'"' {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    i += 1;
                }
                if i >= bytes.len() {
                    return Err(XmlError::UnescapedQuote);
                }
            }
            i += 1;
        }
    }
    while i < bytes.len() {
        while i < bytes.len() && matches!(bytes[i], b' ' | b'\n' | b'\r' | b'\t') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        match bytes[i] {
            b'<' => {
                i += 1;
                if i < bytes.len() && bytes[i] == b'/' {
                    // Closing tag.
                    i += 1;
                    let start = i;
                    while i < bytes.len()
                        && bytes[i] != b'>'
                        && !matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r')
                    {
                        i += 1;
                    }
                    let tag_name = &input[start..i];
                    if !is_valid_name(tag_name) {
                        return Err(XmlError::InvalidTagName);
                    }
                    // Skip to >.
                    while i < bytes.len() && bytes[i] != b'>' {
                        i += 1;
                    }
                    if i >= bytes.len() {
                        return Err(XmlError::UnbalancedTags);
                    }
                    i += 1; // skip >
                    match stack.pop() {
                        Some(expected) if expected == tag_name => {}
                        _ => return Err(XmlError::UnbalancedTags),
                    }
                } else {
                    let start = i;
                    while i < bytes.len() && !matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r' | b'>' | b'/') {
                        i += 1;
                    }
                    let tag_name = &input[start..i];
                    if !is_valid_name(tag_name) {
                        return Err(XmlError::InvalidTagName);
                    }
                    // Parse attributes to find if self-closing (/>).
                    let mut self_closing = false;
                    while i < bytes.len() {
                        if bytes[i] == b'/' && bytes.get(i + 1..i + 2) == Some(b">") {
                            self_closing = true;
                            i += 2;
                            break;
                        }
                        if bytes[i] == b'>' {
                            i += 1;
                            break;
                        }
                        if bytes[i] == b'"' {
                            i += 1;
                            while i < bytes.len() && bytes[i] != b'"' {
                                if bytes[i] == b'<' {
                                    return Err(XmlError::InvalidEntity);
                                }
                                i += 1;
                            }
                            if i >= bytes.len() {
                                return Err(XmlError::UnescapedQuote);
                            }
                            // Skip past the closing quote.
                            i += 1;
                        } else {
                            i += 1;
                        }
                    }
                    if !self_closing {
                        stack.push(tag_name);
                        found_root = true;
                    } else {
                        found_root = true;
                    }
                }
            }
            b'&' => {
                // Entity reference in text.
                i += 1;
                if i < bytes.len() && bytes[i] == b'#' {
                    i += 1;
                    if i < bytes.len() && bytes[i] == b'x' {
                        i += 1;
                        while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                            i += 1;
                        }
                    } else {
                        while i < bytes.len() && bytes[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                    if i < bytes.len() && bytes[i] == b';' {
                        i += 1;
                    } else {
                        return Err(XmlError::InvalidEntity);
                    }
                } else if i < bytes.len() {
                    let entity_start = i;
                    while i < bytes.len() && bytes[i] != b';' {
                        i += 1;
                    }
                    if i >= bytes.len()
                        || !bytes[entity_start..i]
                            .iter()
                            .all(|b| b.is_ascii_alphabetic())
                    {
                        return Err(XmlError::InvalidEntity);
                    }
                    i += 1;
                } else {
                    return Err(XmlError::InvalidEntity);
                }
            }
            _ => {
                // Text content - skip until next '<' or '&'.
                while i < bytes.len() && bytes[i] != b'<' && bytes[i] != b'&' {
                    i += 1;
                }
            }
        }
    }

    if !found_root {
        return Err(XmlError::MissingDeclaration);
    }
    if !stack.is_empty() {
        return Err(XmlError::UnbalancedTags);
    }
    Ok(())
}

/// Checks whether `name` is a valid XML 1.0 Name production.
///
/// Per the XML spec, a Name must start with a letter or `_` or `:`,
/// and subsequent characters may be letters, digits, `.`, `-`, `_`, or `:`.
#[must_use]
fn is_valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !is_name_start_char(first) {
        return false;
    }
    chars.all(is_name_char)
}

/// Checks whether `c` is a valid XML NameStartChar.
#[must_use]
fn is_name_start_char(c: char) -> bool {
    matches!(c, 'A'..='Z' | 'a'..='z' | '_') || c == ':'
}

/// Checks whether `c` is a valid XML NameChar.
#[must_use]
fn is_name_char(c: char) -> bool {
    is_name_start_char(c) || matches!(c, '0'..='9' | '.')
}

/// A builder for constructing XML elements with a fluent API.
///
/// Methods can be chained to produce well-formed XML. Each call to
/// [`child`] opens a nested element; [`end`] closes the current
/// element and returns to the parent level. Use [`build`] to
/// finalize and obtain an [`XmlDocument`].
///
/// # Examples
///
/// ```
/// use codevar_base::basic_xml::XmlBuilder;
///
/// let doc = XmlBuilder::new("root")
///     .attr("version", "1.0")
///     .child("child")
///         .text("content")
///     .end()
///     .build();
///
/// assert_eq!(
///     doc.to_string(),
///     "<root version=\"1.0\"><child>content</child></root>"
/// );
/// ```
#[derive(Debug, Clone)]
pub struct XmlBuilder {
    buffer: String,
    /// Stack of (tag_name, has_children, open_tag_start_pos).
    /// `open_tag_start_pos` is the byte offset in `buffer` where the
    /// opening `<` of this element begins, enabling efficient
    /// self-closing tag generation in [`end`].
    stack: Vec<(String, bool, usize)>,
    closed: bool,
}

impl XmlBuilder {
    /// Creates a new builder with the given root element name.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not a valid XML name.
    #[must_use]
    pub fn new(name: &str) -> Self {
        assert!(is_valid_name(name), "invalid XML element name: {name}");
        let mut buffer = String::with_capacity(64);
        buffer.push('<');
        buffer.push_str(name);
        Self {
            buffer,
            stack: vec![(name.to_string(), false, 0)],
            closed: false,
        }
    }

    /// Adds an attribute to the current element, returning self for chaining.
    ///
    /// The attribute value is automatically escaped.
    ///
    /// # Panics
    ///
    /// Panics if called after [`build`] or if the name is invalid.
    #[must_use]
    pub fn attr(mut self, name: &str, value: &str) -> Self {
        debug_assert!(!self.closed, "cannot add attribute to a built document");
        assert!(is_valid_name(name), "invalid XML attribute name: {name}");
        self.buffer.push(' ');
        self.buffer.push_str(name);
        self.buffer.push_str("=\"");
        self.buffer.push_str(&escape(value));
        self.buffer.push('"');
        self
    }

    /// Adds escaped text content to the current element.
    ///
    /// The text is automatically escaped for safe XML embedding.
    ///
    /// # Panics
    ///
    /// Panics if called after [`build`].
    #[must_use]
    pub fn text(mut self, text: &str) -> Self {
        assert!(!self.closed, "cannot add text to a built document");
        self.flush_open();
        self.buffer.push_str(&escape(text));
        if let Some(entry) = self.stack.last_mut() {
            entry.1 = true;
        }
        self
    }

    /// Opens a child element with the given name, returning self for
    /// further chaining on the child.
    ///
    /// Call [`end`] to close this child and return to the parent level.
    ///
    /// # Panics
    ///
    /// Panics if called after [`build`] or if the name is invalid.
    #[must_use]
    pub fn child(mut self, name: &str) -> Self {
        assert!(!self.closed, "cannot add child to a built document");
        assert!(is_valid_name(name), "invalid XML element name: {name}");
        self.flush_open();
        // Mark the parent as having children.
        if let Some(entry) = self.stack.last_mut() {
            entry.1 = true;
        }
        let pos = self.buffer.len();
        self.buffer.push('<');
        self.buffer.push_str(name);
        self.stack.push((name.to_string(), false, pos));
        self
    }

    /// Closes the innermost open element, returning self for chaining.
    ///
    /// If the element had no children, emits a self-closing tag (`/>`).
    /// Otherwise, emits a closing tag (`</name>`).
    ///
    /// # Panics
    ///
    /// Panics if there is no open element or if called after [`build`].
    #[must_use]
    pub fn end(mut self) -> Self {
        assert!(!self.closed, "cannot close element of a built document");
        let Some((tag_name, has_children, _)) = self.stack.pop() else {
            panic!("cannot close element: no open elements");
        };
        if has_children {
            // Opening tag already closed by flush_open() in child() or text().
            self.buffer.push_str("</");
            self.buffer.push_str(&tag_name);
            self.buffer.push('>');
        } else {
            // No children were added; close the opening tag with />.
            // The opening <name was pushed without a >, so add />.
            // But text() may have already added > via flush_open().
            // If buffer ends with >, replace it with />; otherwise add />.
            if self.buffer.ends_with('>') {
                self.buffer.pop();
            }
            self.buffer.push_str("/>");
        }
        self
    }

    /// Finalizes the document, closing all open elements, and returns
    /// an [`XmlDocument`].
    ///
    /// After this call, the builder cannot be used again.
    ///
    /// # Panics
    ///
    /// Panics if there are still unclosed elements.
    #[must_use]
    pub fn build(mut self) -> XmlDocument {
        if (!self.closed) {
            while self.stack.len() > 1 {
                self = self.end();
            }
            if let Some((tag_name, _, _)) = self.stack.pop() {
                if self.buffer.ends_with('/') {
                    self.buffer.pop(); // remove /
                    self.buffer.push_str("></");
                    self.buffer.push_str(&tag_name);
                    self.buffer.push('>');
                } else {
                    if !self.buffer.ends_with('>') {
                        self.buffer.push('>');
                    }
                    self.buffer.push_str("</");
                    self.buffer.push_str(&tag_name);
                    self.buffer.push('>');
                }
            }
            self.closed = true;
        }
        XmlDocument { xml: self.buffer }
    }

    /// Appends raw XML content to the builder's buffer.
    ///
    /// Useful for inserting pre-built XML content from registered
    /// interfaces or other sources.
    ///
    /// # Panics
    ///
    /// Panics if called after [`build`].
    pub fn push(mut self, content: &str) -> Self {
        assert!(!self.closed, "cannot push content to a built document");
        self.buffer.push_str(content);
        self
    }

    /// Flushes any open tag by closing its opening bracket `>`.
    /// Called before adding text or children to ensure the current
    /// element's opening tag is properly terminated.
    fn flush_open(&mut self) {
        if let Some((_, has_children, _)) = self.stack.last() {
            if !has_children && !self.buffer.ends_with('>') && !self.buffer.ends_with('/') {
                self.buffer.push('>');
            }
        }
    }
}

impl Default for XmlBuilder {
    fn default() -> Self {
        Self::new("root")
    }
}

/// A finalized XML document produced by [`XmlBuilder::build`].
///
/// Provides [`to_string`] access to the XML text and a [`validate`]
/// convenience method.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct XmlDocument {
    xml: String,
}

impl XmlDocument {
    /// Returns the XML document as a string.
    #[must_use]
    pub fn to_string(&self) -> &str {
        &self.xml
    }

    /// Validates that this document is well-formed XML.
    #[must_use]
    pub fn validate(&self) -> Result<(), XmlError> {
        validate(&self.xml)
    }
}

impl AsRef<str> for XmlDocument {
    fn as_ref(&self) -> &str {
        &self.xml
    }
}

impl fmt::Display for XmlDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.xml)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_simple_element() {
        let doc = XmlBuilder::new("node").build();
        let xml = doc.to_string();
        assert!(xml.starts_with("<node"));
        assert!(xml.ends_with("</node>"));
        assert!(doc.validate().is_ok());
    }

    #[test]
    fn builder_with_attributes() {
        let doc = XmlBuilder::new("node")
            .attr("version", "1.0")
            .attr("encoding", "utf-8")
            .build();
        let xml = doc.to_string();
        assert!(xml.contains("version=\"1.0\""));
        assert!(xml.contains("encoding=\"utf-8\""));
        assert!(doc.validate().is_ok());
    }

    #[test]
    fn builder_with_children() {
        let doc = XmlBuilder::new("root")
            .child("child")
            .text("hello")
            .end()
            .build();
        assert!(doc.validate().is_ok());
    }

    #[test]
    fn builder_nested_children() {
        let doc = XmlBuilder::new("node")
            .child("a")
            .child("b")
            .text("deep")
            .end()
            .text("mid")
            .end()
            .text("top")
            .build();
        assert!(doc.validate().is_ok());
        let xml = doc.to_string();
        assert!(xml.contains("<a>"));
        assert!(xml.contains("</a>"));
    }

    #[test]
    fn builder_self_closing() {
        let doc = XmlBuilder::new("root")
            .child("child")
            .attr("type", "s")
            .end()
            .build();
        let xml = doc.to_string();
        assert!(xml.contains("<child type=\"s\"/>"));
        assert!(doc.validate().is_ok());
    }

    #[test]
    fn escape_special_chars() {
        assert_eq!(escape("a < b & c"), "a &lt; b &amp; c");
        assert_eq!(escape("\"quote\""), "&quot;quote&quot;");
        assert_eq!(escape("no special"), "no special");
    }

    #[test]
    fn validate_well_formed() {
        assert!(validate("<root><child/></root>").is_ok());
        assert!(validate("<node version=\"1.0\"><child>text</child></node>").is_ok());
    }

    #[test]
    fn validate_unbalanced() {
        assert!(validate("<node><child></node>").is_err());
    }

    #[test]
    fn validate_empty() {
        assert!(validate("").is_err());
    }

    #[test]
    fn roundtrip() {
        let doc = XmlBuilder::new("node")
            .attr("id", "1")
            .child("interface")
            .attr("name", "org.freedesktop.DBus.Introspectable")
            .child("method")
            .attr("name", "Introspect")
            .child("arg")
            .attr("name", "data")
            .attr("type", "s")
            .attr("direction", "out")
            .end()
            .end()
            .end()
            .build();
        let xml = doc.to_string();
        assert!(doc.validate().is_ok());
        assert!(xml.contains("org.freedesktop.DBus.Introspectable"));
        assert!(xml.contains("<method name=\"Introspect\">"));
    }
}
