//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Strict RFC 3986 URL parsing, validation, and percent-encoding.
//!
//! [`parse`] splits an absolute URL into its [`Url`] components without
//! copying anything: every field borrows directly from the input string,
//! so parsing is a handful of ASCII scans over the input. [`is_valid`]
//! reports whether [`parse`] would succeed, without keeping the result.
//!
//! [`encode`] percent-encodes every byte outside the RFC 3986 unreserved
//! set (`A-Z a-z 0-9 - . _ ~`) as `%XX` with uppercase hex digits, and
//! [`decode`] reverses that transformation, rejecting malformed `%`
//! sequences and percent-decoded bytes that are not valid UTF-8.
//! [`decode_bytes`] provides the raw byte form for callers that do not
//! need text.
//!
//! Validation is deliberately strict, so a URL accepted here is safe to
//! hand to a network stack:
//!
//! - only absolute URLs parse; a `scheme` must lead the input;
//! - hosts must be a bracketed `IPv6` literal, a dotted-quad `IPv4`
//!   address, or a `reg-name` of unreserved/sub-delimiter characters and
//!   percent-escapes (the `file` scheme additionally allows an empty
//!   host, as in `file:///etc/hosts`);
//! - ports must be decimal digits in `0..=65535`;
//! - each component accepts only its production's characters, and raw
//!   non-ASCII bytes are rejected — run text through [`encode`] first.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Uppercase hex digits used to render one byte as `%XX`.
const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

/// Largest port number accepted by [`parse`] (RFC 6335).
const MAX_PORT: u32 = 65_535;

/// Parses and validates a complete absolute URL.
///
/// The returned [`Url`] borrows every component from `input`, so no
/// allocation happens on this path. See the [module documentation](self)
/// for the exact subset of RFC 3986 that is accepted.
///
/// # Errors
///
/// - [`UrlError::EmptyInput`] — `input` is empty.
/// - [`UrlError::MissingScheme`] — `input` contains no `:`, so it is not
///   an absolute URL.
/// - [`UrlError::InvalidScheme`] — the text before the first `:` is not
///   a valid scheme.
/// - [`UrlError::InvalidCharacter`] — a byte outside the allowed set of
///   the component it appears in; holds the byte offset in `input`.
/// - [`UrlError::InvalidPercentEncoding`] — a `%` not followed by two
///   hex digits; holds the byte offset of that `%`.
/// - [`UrlError::EmptyHost`] — `//` was present but no host follows it.
/// - [`UrlError::InvalidHost`] — the host is neither a valid `IPv4`
///   address nor a `reg-name`.
/// - [`UrlError::InvalidIpv6`] — a bracketed host is not a valid `IPv6`
///   address, or brackets are used in an invalid position.
/// - [`UrlError::InvalidPort`] — the port is empty or exceeds
///   `0..=65535`.
///
/// # Performance
///
/// O(n) over `input` with a bounded number of scans and no allocation.
///
/// # Examples
///
/// ```
/// use codevar_base::basic_url::parse;
///
/// let url = parse("https://example.com:8080/a?b=c#d").unwrap();
/// assert_eq!(url.scheme(), "https");
/// assert_eq!(url.authority().unwrap().port_number(), Some(8080));
/// assert_eq!(url.path(), "/a");
/// assert_eq!(url.query(), Some("b=c"));
/// assert_eq!(url.fragment(), Some("d"));
/// ```
pub fn parse(input: &str) -> Result<Url<'_>, UrlError> {
    if input.is_empty() {
        return Err(UrlError::EmptyInput);
    }

    let scheme_end = input.find(':').ok_or(UrlError::MissingScheme)?;
    validate_scheme(&input[..scheme_end])?;

    let hier_start = scheme_end + 1;
    let hash = input[hier_start..]
        .find('#')
        .map_or(input.len(), |rel| hier_start + rel);
    let question = input[hier_start..hash]
        .find('?')
        .map_or(hash, |rel| hier_start + rel);

    let (authority, path_start, path_end) =
        if input[hier_start..question].starts_with("//") {
            let scan = hier_start + 2;
            let end = input[scan..question]
                .find('/')
                .map_or(question, |rel| scan + rel);
            (Some((scan, end)), end, question)
        } else {
            (None, hier_start, question)
        };

    let allow_empty_host = input[..scheme_end].eq_ignore_ascii_case("file");
    let authority = authority
        .map(|(start, end)| parse_authority(&input[start..end], start, allow_empty_host))
        .transpose()?;

    validate_component(&input[path_start..path_end], path_start, is_path_char)?;

    let query = if question < hash {
        validate_component(&input[question + 1..hash], question + 1, is_query_char)?;
        Some(&input[question + 1..hash])
    } else {
        None
    };

    let fragment = if hash < input.len() {
        validate_component(&input[hash + 1..], hash + 1, is_query_char)?;
        Some(&input[hash + 1..])
    } else {
        None
    };

    Ok(Url {
        scheme: &input[..scheme_end],
        authority,
        path: &input[path_start..path_end],
        query,
        fragment,
    })
}

/// Returns whether [`parse`] accepts `input`.
///
/// # Performance
///
/// O(n) over `input`; identical cost to [`parse`] apart from building no
/// [`Url`] for the caller.
///
/// # Examples
///
/// ```
/// use codevar_base::basic_url::is_valid;
///
/// assert!(is_valid("https://example.com/x"));
/// assert!(!is_valid("example.com/x"));
/// ```
#[must_use]
pub fn is_valid(input: &str) -> bool {
    parse(input).is_ok()
}

/// Percent-encodes `input` so it may be embedded in a URL component.
///
/// Every byte outside the RFC 3986 unreserved set (`A-Z a-z 0-9 - . _ ~`)
/// becomes `%XX` with uppercase hex digits; no other transformation is
/// applied. Structural delimiters (`/`, `?`, `#`, ...) are encoded too,
/// so use this for a single component rather than for a whole URL.
///
/// # Performance
///
/// Two passes over `input` (one to size the buffer exactly, one to
/// write); the result is always `input.len() + 2 * <escaped bytes>`
/// bytes long.
///
/// # Examples
///
/// ```
/// use codevar_base::basic_url::encode;
///
/// assert_eq!(encode("hello world"), "hello%20world");
/// assert_eq!(encode("a/b?c=d"), "a%2Fb%3Fc%3Dd");
/// assert_eq!(encode("safe-1.2_3~"), "safe-1.2_3~");
/// ```
#[must_use]
pub fn encode(input: &str) -> String {
    encode_bytes(input.as_bytes())
}

/// Percent-encodes arbitrary bytes with the same rules as [`encode`].
///
/// # Performance
///
/// Two passes over `input`; output length is exactly
/// `input.len() + 2 * <escaped bytes>`.
///
/// # Examples
///
/// ```
/// use codevar_base::basic_url::encode_bytes;
///
/// assert_eq!(encode_bytes(&[0x00, 0xFF]), "%00%FF");
/// ```
#[must_use]
pub fn encode_bytes(input: &[u8]) -> String {
    let kept = input.iter().filter(|&&byte| is_unreserved(byte)).count();
    let mut out = String::with_capacity(kept + (input.len() - kept) * 3);
    for &byte in input {
        if is_unreserved(byte) {
            out.push(char::from(byte));
        } else {
            out.push('%');
            out.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
            out.push(char::from(HEX_DIGITS[usize::from(byte & 0x0F)]));
        }
    }
    out
}

/// Decodes percent-encoded text into a UTF-8 string.
///
/// `+` is not treated as a space (this is not form encoding). Bytes that
/// were never percent-encoded are passed through unchanged.
///
/// # Errors
///
/// - [`UrlError::InvalidPercentEncoding`] — a `%` is not followed by two
///   hex digits; holds the byte offset of that `%`.
/// - [`UrlError::InvalidUtf8`] — the decoded bytes are not valid UTF-8.
///
/// # Performance
///
/// Single pass over `input`, one output allocation.
///
/// # Examples
///
/// ```
/// use codevar_base::basic_url::decode;
///
/// assert_eq!(decode("hello%20world").unwrap(), "hello world");
/// assert_eq!(decode("a+b").unwrap(), "a+b");
/// assert!(decode("%FF").is_err());
/// ```
pub fn decode(input: &str) -> Result<String, UrlError> {
    String::from_utf8(decode_bytes(input)?).map_err(|_| UrlError::InvalidUtf8)
}

/// Decodes percent-encoded text into raw bytes.
///
/// Behaves like [`decode`] but performs no UTF-8 validation, so it
/// succeeds for any byte sequence that [`encode_bytes`] could produce.
///
/// # Errors
///
/// - [`UrlError::InvalidPercentEncoding`] — a `%` is not followed by two
///   hex digits; holds the byte offset of that `%`.
///
/// # Performance
///
/// Single pass over `input`, one output allocation; output length never
/// exceeds `input.len()`.
///
/// # Examples
///
/// ```
/// use codevar_base::basic_url::decode_bytes;
///
/// assert_eq!(decode_bytes("%00%FF").unwrap(), vec![0x00, 0xFF]);
/// assert_eq!(decode_bytes("plain").unwrap(), b"plain".to_vec());
/// ```
pub fn decode_bytes(input: &str) -> Result<Vec<u8>, UrlError> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        let Some(triple) = bytes.get(i + 1..i + 3) else {
            return Err(UrlError::InvalidPercentEncoding(i));
        };
        let high = hex_value(triple[0]).ok_or(UrlError::InvalidPercentEncoding(i))?;
        let low = hex_value(triple[1]).ok_or(UrlError::InvalidPercentEncoding(i))?;
        out.push((high << 4) | low);
        i += 3;
    }
    Ok(out)
}

/// A parsed URL, borrowing every component from the input string.
///
/// Produced by [`parse`]; the [`Display`](fmt::Display) implementation
/// reproduces the input byte for byte, because components are kept as
/// raw slices rather than normalized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Url<'a> {
    /// Text before the first `:`, stored verbatim.
    scheme: &'a str,
    /// `//`-introduced authority, present only when the URL has one.
    authority: Option<Authority<'a>>,
    /// Path after the authority (or after the scheme), possibly empty.
    path: &'a str,
    /// Text after the first `?`, excluded from the path.
    query: Option<&'a str>,
    /// Text after the first `#`.
    fragment: Option<&'a str>,
}

impl<'a> Url<'a> {
    /// The scheme, preserving the case it had in the input.
    #[must_use]
    pub fn scheme(&self) -> &'a str {
        self.scheme
    }

    /// The authority, or `None` for scheme-only forms such as
    /// `mailto:user@example.com`.
    #[must_use]
    pub fn authority(&self) -> Option<&Authority<'a>> {
        self.authority.as_ref()
    }

    /// The host of the authority, or `None` when there is no authority.
    #[must_use]
    pub fn host(&self) -> Option<Host<'a>> {
        self.authority.map(|authority| authority.host)
    }

    /// The path, possibly empty (for example `http://example.com`).
    #[must_use]
    pub fn path(&self) -> &'a str {
        self.path
    }

    /// The query without its leading `?`; `Some("")` for `...?`.
    #[must_use]
    pub fn query(&self) -> Option<&'a str> {
        self.query
    }

    /// The fragment without its leading `#`; `Some("")` for `...#`.
    #[must_use]
    pub fn fragment(&self) -> Option<&'a str> {
        self.fragment
    }
}

impl fmt::Display for Url<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.scheme)?;
        f.write_str(":")?;
        if let Some(authority) = &self.authority {
            f.write_str("//")?;
            write!(f, "{authority}")?;
        }
        f.write_str(self.path)?;
        if let Some(query) = self.query {
            f.write_str("?")?;
            f.write_str(query)?;
        }
        if let Some(fragment) = self.fragment {
            f.write_str("#")?;
            f.write_str(fragment)?;
        }
        Ok(())
    }
}

/// The `userinfo@host:port` part of a URL, split into its pieces.
///
/// Produced by [`parse`]; every field borrows from the parsed input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Authority<'a> {
    /// Text before the first `@`, including any `:password`.
    userinfo: Option<&'a str>,
    /// The host, already classified and validated.
    host: Host<'a>,
    /// The port as digits, without the leading `:`.
    port: Option<&'a str>,
}

impl<'a> Authority<'a> {
    /// The userinfo (user and optional `:password`) without the
    /// trailing `@`, or `None` when the authority has none.
    #[must_use]
    pub fn userinfo(&self) -> Option<&'a str> {
        self.userinfo
    }

    /// The host, classified as domain, `IPv4`, or `IPv6`.
    #[must_use]
    pub fn host(&self) -> Host<'a> {
        self.host
    }

    /// The port as written in the input, without the leading `:`.
    #[must_use]
    pub fn port(&self) -> Option<&'a str> {
        self.port
    }

    /// The port as a number, or `None` when the URL has no port.
    #[must_use]
    pub fn port_number(&self) -> Option<u16> {
        self.port.and_then(|port| port.parse().ok())
    }
}

impl fmt::Display for Authority<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(userinfo) = self.userinfo {
            f.write_str(userinfo)?;
            f.write_str("@")?;
        }
        match self.host {
            Host::Ipv6(host) => write!(f, "[{host}]")?,
            Host::Domain(host) | Host::Ipv4(host) => f.write_str(host)?,
        }
        if let Some(port) = self.port {
            f.write_str(":")?;
            f.write_str(port)?;
        }
        Ok(())
    }
}

/// The host of a URL authority, in the form it was validated as.
///
/// [`parse`] never returns an unvalidated host: every variant's payload
/// is exactly the substring of the input that was checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Host<'a> {
    /// A `reg-name` of unreserved/sub-delimiter characters and
    /// percent-escapes (for example `example.com`). The `file` scheme
    /// permits an empty host, which is stored as `Domain("")`.
    Domain(&'a str),
    /// A dotted-quad `IPv4` address, stored without any normalization
    /// (leading zeroes are rejected at parse time).
    Ipv4(&'a str),
    /// An `IPv6` address, stored without the surrounding `[]`.
    Ipv6(&'a str),
}

impl<'a> Host<'a> {
    /// The host text exactly as it appears in the input (without `[]`
    /// for `IPv6` literals).
    #[must_use]
    pub const fn as_str(self) -> &'a str {
        match self {
            Self::Domain(host) | Self::Ipv4(host) | Self::Ipv6(host) => host,
        }
    }
}

/// Errors produced by [`parse`], [`decode`], and [`decode_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UrlError {
    /// The input string is empty.
    EmptyInput,
    /// The input contains no `:`, so it is not an absolute URL.
    MissingScheme,
    /// The text before the first `:` is not a valid scheme.
    InvalidScheme,
    /// A byte outside the allowed set of the component it appears in;
    /// holds the byte offset of the first offending byte.
    InvalidCharacter(usize),
    /// A `%` that is not followed by two hex digits; holds the byte
    /// offset of that `%`.
    InvalidPercentEncoding(usize),
    /// The authority introduces a host but the host is empty.
    EmptyHost,
    /// The host is neither a valid `IPv4` address nor a `reg-name`.
    InvalidHost,
    /// A bracketed host is not a valid `IPv6` address, or brackets
    /// appear where only a `reg-name` may.
    InvalidIpv6,
    /// The port is empty, not made of digits, or exceeds `0..=65535`.
    InvalidPort,
    /// The percent-decoded bytes are not valid UTF-8.
    InvalidUtf8,
}

impl fmt::Display for UrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("empty URL"),
            Self::MissingScheme => f.write_str("URL is missing a scheme"),
            Self::InvalidScheme => f.write_str("invalid URL scheme"),
            Self::InvalidCharacter(offset) => {
                write!(f, "invalid character at byte offset {offset}")
            }
            Self::InvalidPercentEncoding(offset) => {
                write!(f, "invalid percent-encoding at byte offset {offset}")
            }
            Self::EmptyHost => f.write_str("URL authority has an empty host"),
            Self::InvalidHost => f.write_str("invalid URL host"),
            Self::InvalidIpv6 => f.write_str("invalid IPv6 host address"),
            Self::InvalidPort => f.write_str("invalid URL port"),
            Self::InvalidUtf8 => f.write_str("percent-decoded URL is not valid UTF-8"),
        }
    }
}

impl core::error::Error for UrlError {}

/// Validates the scheme production: `ALPHA *( ALPHA / DIGIT / "+" /
/// "-" / "." )`.
///
/// `scheme` is the text before the first `:` of the input.
fn validate_scheme(scheme: &str) -> Result<(), UrlError> {
    let bytes = scheme.as_bytes();
    let (first, rest) = bytes.split_first().ok_or(UrlError::InvalidScheme)?;
    if !first.is_ascii_alphabetic() {
        return Err(UrlError::InvalidScheme);
    }
    let valid = rest
        .iter()
        .all(|&byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'));
    if valid {
        Ok(())
    } else {
        Err(UrlError::InvalidScheme)
    }
}

/// Validates one component of the URL against `allowed`, treating
/// `%XX` triples as a single unit.
///
/// `base` is the byte offset of `component` within the whole URL, so
/// the offsets carried by [`UrlError`] are absolute.
fn validate_component<F>(component: &str, base: usize, allowed: F) -> Result<(), UrlError>
where
    F: Fn(u8) -> bool,
{
    let bytes = component.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            if !allowed(bytes[i]) {
                return Err(UrlError::InvalidCharacter(base + i));
            }
            i += 1;
            continue;
        }
        if i + 3 > bytes.len() {
            return Err(UrlError::InvalidPercentEncoding(base + i));
        }
        if !bytes[i + 1].is_ascii_hexdigit() || !bytes[i + 2].is_ascii_hexdigit() {
            return Err(UrlError::InvalidPercentEncoding(base + i));
        }
        i += 3;
    }
    Ok(())
}

/// Validates the port production: decimal digits whose value fits in
/// `0..=65535`.
///
/// `base` is the byte offset of `port` within the whole URL.
fn validate_port(port: &str, base: usize) -> Result<(), UrlError> {
    let bytes = port.as_bytes();
    if bytes.is_empty() {
        return Err(UrlError::InvalidPort);
    }
    let mut value: u32 = 0;
    for (offset, &byte) in bytes.iter().enumerate() {
        if !byte.is_ascii_digit() {
            return Err(UrlError::InvalidCharacter(base + offset));
        }
        value = value * 10 + u32::from(byte - b'0');
        if value > MAX_PORT {
            return Err(UrlError::InvalidPort);
        }
    }
    Ok(())
}

/// Splits `authority` (the text between `//` and the next `/`, `?`, or
/// end of input) into userinfo, host, and port, validating each piece.
///
/// `base` is the byte offset of `authority` within the whole URL.
/// `allow_empty_host` is set for the `file` scheme, whose URLs such as
/// `file:///etc/hosts` legitimately carry no host.
fn parse_authority<'a>(
    authority: &'a str,
    base: usize,
    allow_empty_host: bool,
) -> Result<Authority<'a>, UrlError> {
    let at = authority.find('@');
    if let Some(at) = at {
        validate_component(&authority[..at], base, is_userinfo_char)?;
    }
    let hostport_start = at.map_or(0, |at| at + 1);
    let (host, port) = split_host_port(
        &authority[hostport_start..],
        base + hostport_start,
        allow_empty_host,
    )?;
    Ok(Authority {
        userinfo: at.map(|at| &authority[..at]),
        host,
        port,
    })
}

/// Splits `hostport` at the first `:` outside brackets (or at the `]`
/// closing an `IPv6` literal) and validates both halves.
///
/// `base` is the byte offset of `hostport` within the whole URL.
fn split_host_port<'a>(
    hostport: &'a str,
    base: usize,
    allow_empty_host: bool,
) -> Result<(Host<'a>, Option<&'a str>), UrlError> {
    if let Some(after) = hostport.strip_prefix('[') {
        let close = after.find(']').ok_or(UrlError::InvalidIpv6)?;
        let raw = &after[..close];
        if !is_valid_ipv6(raw) {
            return Err(UrlError::InvalidIpv6);
        }
        let tail = &after[close + 1..];
        if tail.is_empty() {
            return Ok((Host::Ipv6(raw), None));
        }
        let port = tail.strip_prefix(':').ok_or(UrlError::InvalidHost)?;
        validate_port(port, base + close + 3)?;
        return Ok((Host::Ipv6(raw), Some(port)));
    }
    if hostport.contains('[') || hostport.contains(']') {
        return Err(UrlError::InvalidHost);
    }
    match hostport.find(':') {
        Some(colon) => {
            let host = classify_host(&hostport[..colon], base, allow_empty_host)?;
            validate_port(&hostport[colon + 1..], base + colon + 1)?;
            Ok((host, Some(&hostport[colon + 1..])))
        }
        None => Ok((classify_host(hostport, base, allow_empty_host)?, None)),
    }
}

/// Classifies an unbracketed host as a `reg-name` or an `IPv4`
/// address, rejecting anything that is neither.
///
/// A host made only of digits and dots must be a valid dotted-quad
/// `IPv4` address, so typos such as `1.2.3` or `999.1.1.1` fail
/// instead of silently becoming a domain name.
///
/// `base` is the byte offset of `host` within the whole URL.
fn classify_host<'a>(
    host: &'a str,
    base: usize,
    allow_empty_host: bool,
) -> Result<Host<'a>, UrlError> {
    if host.is_empty() {
        return if allow_empty_host {
            Ok(Host::Domain(""))
        } else {
            Err(UrlError::EmptyHost)
        };
    }
    if host
        .bytes()
        .all(|byte| byte.is_ascii_digit() || byte == b'.')
    {
        return if is_valid_ipv4(host) {
            Ok(Host::Ipv4(host))
        } else {
            Err(UrlError::InvalidHost)
        };
    }
    validate_component(host, base, is_reg_name_char)?;
    Ok(Host::Domain(host))
}

/// Validates the `IPv4address` production: exactly four decimal
/// octets in `0..=255` with no leading zeroes.
fn is_valid_ipv4(host: &str) -> bool {
    let bytes = host.as_bytes();
    let mut i = 0;
    let mut octets = 0;
    while i < bytes.len() {
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let digits = i - start;
        if digits == 0 || digits > 3 {
            return false;
        }
        if digits > 1 && bytes[start] == b'0' {
            return false;
        }
        let mut value: u32 = 0;
        for &byte in &bytes[start..i] {
            value = value * 10 + u32::from(byte - b'0');
        }
        if value > 255 {
            return false;
        }
        octets += 1;
        if i == bytes.len() {
            break;
        }
        if bytes[i] != b'.' {
            return false;
        }
        i += 1;
        if i == bytes.len() {
            return false;
        }
    }
    octets == 4
}

/// Validates the `IPv6address` production, including `::` compression
/// (at most once) and an optional trailing `IPv4address` piece.
fn is_valid_ipv6(host: &str) -> bool {
    let bytes = host.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut i = 0;
    let mut groups = 0;
    let mut compressed = false;

    if bytes[0] == b':' {
        if bytes.len() < 2 || bytes[1] != b':' {
            return false;
        }
        compressed = true;
        i = 2;
        if i == bytes.len() {
            return true;
        }
    }

    while i < bytes.len() {
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
            i += 1;
        }
        let digits = i - start;
        if digits == 0 {
            return false;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            if !is_valid_ipv4(&host[start..]) {
                return false;
            }
            groups += 2;
            break;
        }
        if digits > 4 {
            return false;
        }
        groups += 1;
        if i == bytes.len() {
            break;
        }
        if bytes[i] != b':' {
            return false;
        }
        i += 1;
        if i == bytes.len() {
            return false;
        }
        if bytes[i] == b':' {
            if compressed {
                return false;
            }
            compressed = true;
            i += 1;
            if i == bytes.len() {
                break;
            }
        }
    }

    if compressed { groups < 8 } else { groups == 8 }
}

/// RFC 3986 `unreserved`: ASCII letters, digits, `-`, `.`, `_`, `~`.
#[inline]
fn is_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
}

/// RFC 3986 `sub-delims`.
#[inline]
fn is_sub_delim(byte: u8) -> bool {
    matches!(
        byte,
        b'!' | b'$' | b'&' | b'\'' | b'(' | b')' | b'*' | b'+' | b',' | b';' | b'='
    )
}

/// RFC 3986 `reg-name`: `unreserved / sub-delims / pct-encoded`.
#[inline]
fn is_reg_name_char(byte: u8) -> bool {
    is_unreserved(byte) || is_sub_delim(byte)
}

/// RFC 3986 `userinfo`: `unreserved / sub-delims / ":" / pct-encoded`.
#[inline]
fn is_userinfo_char(byte: u8) -> bool {
    is_reg_name_char(byte) || byte == b':'
}

/// RFC 3986 `pchar` plus `/`, which is what a path segment separator
/// needs.
#[inline]
fn is_path_char(byte: u8) -> bool {
    is_reg_name_char(byte) || matches!(byte, b':' | b'@' | b'/')
}

/// RFC 3986 `query` and `fragment`: `pchar / "/" / "?"`.
#[inline]
fn is_query_char(byte: u8) -> bool {
    is_path_char(byte) || byte == b'?'
}

/// Converts one hex digit to its value, or `None` for other bytes.
#[inline]
fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// Inputs that must parse, validate, and round-trip through
    /// [`Url`]'s `Display` implementation unchanged.
    const VALID: &[&str] = &[
        "http://example.com",
        "http://example.com/",
        "https://example.com:8080/path/to/resource",
        "https://user:pass@example.com/a?b=c#d",
        "https://[::1]:8443/x",
        "https://[2001:db8::1]/",
        "https://[::ffff:192.168.0.1]/",
        "http://192.168.0.1:80/",
        "http://localhost/a%20b",
        "http://ex%41mple.com/",
        "http://user@host/path",
        "http://example.com/a@b",
        "http://example.com/path%2Fwith%2Fencoded",
        "http://example.com?",
        "http://example.com#",
        "http://example.com/?x#y",
        "http://host/path?query=with/slash#frag?with?marks",
        "HTTPS://EXAMPLE.COM/A?B=C#D",
        "file:///etc/hosts",
        "FILE:///tmp",
        "mailto:someone@example.com",
        "urn:isbn:0451450523",
        "data:text/plain,hello%20world",
        "custom-scheme://a.b.c/d+e;f~g",
    ];

    /// Inputs that must be rejected, one and all.
    const INVALID: &[&str] = &[
        "",
        "example.com",
        "//example.com",
        ":foo",
        "1http://x",
        "+http://x",
        "ht tp://x",
        "http://",
        "http:///",
        "http:///path",
        "http://:8080/",
        "http://exa mple.com/",
        "http://exämple.com/",
        "http://example.com/a b",
        "http://example.com/%zz",
        "http://example.com/%2",
        "http://example.com/#frag#x",
        "http://example.com/[bracket]",
        "http://example.com:65536/",
        "http://example.com:abc/",
        "http://example.com:",
        "http://999.1.1.1/",
        "http://1.2.3/",
        "http://192.168.0.01/",
        "http://256.1.1.1/",
        "http://[::1/",
        "http://[zz::1]/",
        "http://[::1]x/",
        "http://[1:2:3:4:5:6:7:8:9]/",
        "http://[1:2::3::4]/",
        "http://::1/",
        "http://?query",
    ];

    #[test]
    fn parses_complete_url() {
        let input = "https://user:pass@example.com:8443/a/b%20c?x=1&y=2#frag";
        let url = parse(input).unwrap();
        assert_eq!(url.scheme(), "https");
        let authority = url.authority().unwrap();
        assert_eq!(authority.userinfo(), Some("user:pass"));
        assert_eq!(authority.host(), Host::Domain("example.com"));
        assert_eq!(authority.port(), Some("8443"));
        assert_eq!(authority.port_number(), Some(8443));
        assert_eq!(url.host(), Some(Host::Domain("example.com")));
        assert_eq!(url.path(), "/a/b%20c");
        assert_eq!(url.query(), Some("x=1&y=2"));
        assert_eq!(url.fragment(), Some("frag"));
        assert_eq!(url.to_string(), input);
    }

    #[test]
    fn parses_urls_without_authority() {
        let mailto = parse("mailto:someone@example.com").unwrap();
        assert_eq!(mailto.scheme(), "mailto");
        assert!(mailto.authority().is_none());
        assert!(mailto.host().is_none());
        assert_eq!(mailto.path(), "someone@example.com");
        assert!(mailto.query().is_none());
        assert!(mailto.fragment().is_none());

        assert_eq!(
            parse("urn:isbn:0451450523").unwrap().path(),
            "isbn:0451450523"
        );
        assert_eq!(
            parse("data:text/plain,hello%20world").unwrap().path(),
            "text/plain,hello%20world"
        );
        assert_eq!(parse("mailto:a@b").unwrap().to_string(), "mailto:a@b");
    }

    #[test]
    fn parses_empty_query_and_fragment() {
        let url = parse("http://example.com?").unwrap();
        assert_eq!(url.query(), Some(""));
        assert!(url.fragment().is_none());

        let url = parse("http://example.com#").unwrap();
        assert_eq!(url.fragment(), Some(""));
        assert!(url.query().is_none());

        let url = parse("http://example.com/?a#").unwrap();
        assert_eq!(url.query(), Some("a"));
        assert_eq!(url.fragment(), Some(""));

        let url = parse("http://example.com/p?q=1/2#f?g").unwrap();
        assert_eq!(url.path(), "/p");
        assert_eq!(url.query(), Some("q=1/2"));
        assert_eq!(url.fragment(), Some("f?g"));
    }

    #[test]
    fn parses_file_urls_with_empty_host() {
        let url = parse("file:///etc/hosts").unwrap();
        assert_eq!(url.authority().unwrap().host(), Host::Domain(""));
        assert_eq!(url.path(), "/etc/hosts");
        assert_eq!(url.to_string(), "file:///etc/hosts");
        assert!(is_valid("FILE:///tmp"));
        // Only the `file` scheme may drop the host.
        assert_eq!(parse("http:///path").unwrap_err(), UrlError::EmptyHost);
    }

    #[test]
    fn classifies_hosts_by_their_validated_form() {
        assert_eq!(
            parse("http://example.com/").unwrap().host(),
            Some(Host::Domain("example.com"))
        );
        assert_eq!(
            parse("http://192.168.0.1/").unwrap().host(),
            Some(Host::Ipv4("192.168.0.1"))
        );
        assert_eq!(
            parse("http://[::1]/").unwrap().host(),
            Some(Host::Ipv6("::1"))
        );
        assert_eq!(Host::Ipv6("::1").as_str(), "::1");
        assert_eq!(Host::Domain("a.b").as_str(), "a.b");
    }

    #[test]
    fn validates_ipv6_literals() {
        for host in [
            "::1",
            "::",
            "2001:db8::1",
            "::ffff:192.168.0.1",
            "1:2:3:4:5:6:7:8",
            "1:2:3:4:5:6:1.2.3.4",
            "FE80::DB8",
        ] {
            let input = alloc::format!("https://[{host}]/");
            assert_eq!(
                parse(&input).unwrap().host(),
                Some(Host::Ipv6(host)),
                "host {host}"
            );
        }
        for host in [
            "",
            ":",
            ":::",
            "1:2",
            "1:2:3:4:5:6:7",
            "1:2:3:4:5:6:7:8:9",
            "1::2::3",
            "12345::",
            "g::1",
            "1:2:3:4:5:6:7:8:",
            "1:2:3:4:5:6:1.2.3.4.5",
        ] {
            assert_eq!(
                parse(&alloc::format!("https://[{host}]/")).unwrap_err(),
                UrlError::InvalidIpv6,
                "host {host}"
            );
        }
    }

    #[test]
    fn validates_ports() {
        let port_of =
            |input: &str| parse(input).unwrap().authority().unwrap().port_number();
        assert_eq!(port_of("http://h:0/"), Some(0));
        assert_eq!(port_of("http://h:65535/"), Some(65535));
        assert_eq!(port_of("http://h/"), None);
        assert_eq!(port_of("https://[::1]:8443/"), Some(8443));
        assert_eq!(parse("http://h:65536/").unwrap_err(), UrlError::InvalidPort);
        assert_eq!(
            parse("http://h:99999999999999999999/").unwrap_err(),
            UrlError::InvalidPort
        );
        assert_eq!(parse("http://h:").unwrap_err(), UrlError::InvalidPort);
        assert_eq!(
            parse("http://h:80a/").unwrap_err(),
            UrlError::InvalidCharacter(11)
        );
    }

    #[test]
    fn reports_absolute_byte_offsets_for_invalid_characters() {
        assert_eq!(
            parse("http://x/a b").unwrap_err(),
            UrlError::InvalidCharacter(10)
        );
        assert_eq!(
            parse("http://x/p?q u").unwrap_err(),
            UrlError::InvalidCharacter(12)
        );
        assert_eq!(
            parse("http://x/#f#g").unwrap_err(),
            UrlError::InvalidCharacter(11)
        );
        assert_eq!(
            parse("http://user:pa ss@x/").unwrap_err(),
            UrlError::InvalidCharacter(14)
        );
        // The offset counts bytes, not characters.
        assert_eq!(
            parse("http://exämple.com/").unwrap_err(),
            UrlError::InvalidCharacter(9)
        );
        assert_eq!(
            parse("http://x/ä").unwrap_err(),
            UrlError::InvalidCharacter(9)
        );
    }

    #[test]
    fn rejects_malformed_percent_encoding_anywhere() {
        assert_eq!(
            parse("http://x/%zz").unwrap_err(),
            UrlError::InvalidPercentEncoding(9)
        );
        assert_eq!(
            parse("http://x/%2").unwrap_err(),
            UrlError::InvalidPercentEncoding(9)
        );
        assert_eq!(
            parse("http://x/a%2?q=b%").unwrap_err(),
            UrlError::InvalidPercentEncoding(10)
        );
        assert_eq!(
            parse("http://x/?%").unwrap_err(),
            UrlError::InvalidPercentEncoding(10)
        );
        assert_eq!(
            parse("http://x/#a%").unwrap_err(),
            UrlError::InvalidPercentEncoding(11)
        );
        assert_eq!(
            parse("http://a%zz@x/").unwrap_err(),
            UrlError::InvalidPercentEncoding(8)
        );
        assert_eq!(
            parse("http://ex%zz.com/").unwrap_err(),
            UrlError::InvalidPercentEncoding(9)
        );
    }

    #[test]
    fn rejects_invalid_schemes_and_missing_schemes() {
        assert_eq!(parse("").unwrap_err(), UrlError::EmptyInput);
        assert_eq!(parse("example.com/x").unwrap_err(), UrlError::MissingScheme);
        assert_eq!(parse("//example.com").unwrap_err(), UrlError::MissingScheme);
        assert_eq!(parse(":foo").unwrap_err(), UrlError::InvalidScheme);
        assert_eq!(parse("1http://x").unwrap_err(), UrlError::InvalidScheme);
        assert_eq!(parse("+http://x").unwrap_err(), UrlError::InvalidScheme);
        assert_eq!(parse("ht tp://x").unwrap_err(), UrlError::InvalidScheme);
        // `..` is valid per RFC 3986 (`.` is allowed in schemes).
    }

    #[test]
    fn rejects_empty_and_malformed_hosts() {
        assert_eq!(parse("http://").unwrap_err(), UrlError::EmptyHost);
        assert_eq!(parse("http://:8080/").unwrap_err(), UrlError::EmptyHost);
        assert_eq!(parse("http://?query").unwrap_err(), UrlError::EmptyHost);
        assert_eq!(
            parse("http://999.1.1.1/").unwrap_err(),
            UrlError::InvalidHost
        );
        assert_eq!(parse("http://1.2.3/").unwrap_err(), UrlError::InvalidHost);
        assert_eq!(
            parse("http://192.168.0.01/").unwrap_err(),
            UrlError::InvalidHost
        );
        assert_eq!(parse("http://[::1/").unwrap_err(), UrlError::InvalidIpv6);
        assert_eq!(parse("http://[zz::1]/").unwrap_err(), UrlError::InvalidIpv6);
        assert_eq!(parse("http://[::1]x/").unwrap_err(), UrlError::InvalidHost);
        assert_eq!(
            parse("http://example.com/[bracket]").unwrap_err(),
            UrlError::InvalidCharacter(19)
        );
    }

    #[test]
    fn accepts_scheme_case_but_never_normalizes() {
        let url = parse("HTTPS://EXAMPLE.COM/A?B=C#D").unwrap();
        assert_eq!(url.scheme(), "HTTPS");
        assert_eq!(url.path(), "/A");
        assert_eq!(url.to_string(), "HTTPS://EXAMPLE.COM/A?B=C#D");
    }

    #[test]
    fn display_round_trips_every_valid_url() {
        for input in VALID {
            let url = parse(input).unwrap_or_else(|error| {
                panic!("unexpected error for {input:?}: {error}");
            });
            assert_eq!(url.to_string(), *input, "{input}");
        }
    }

    #[test]
    fn is_valid_agrees_with_parse() {
        for input in VALID {
            assert!(is_valid(input), "expected valid: {input}");
            assert!(parse(input).is_ok(), "expected valid: {input}");
        }
        for input in INVALID {
            assert!(!is_valid(input), "expected invalid: {input}");
            assert!(parse(input).is_err(), "expected invalid: {input}");
        }
    }

    #[test]
    fn components_borrow_from_the_input() {
        let input = "https://example.com/path?query#frag";
        let url = parse(input).unwrap();
        let start = input.as_ptr() as usize;
        let end = start + input.len();
        for component in [
            url.scheme(),
            url.path(),
            url.query().unwrap_or(""),
            url.fragment().unwrap_or(""),
        ] {
            let offset = component.as_ptr() as usize;
            assert!(offset >= start && offset + component.len() <= end);
        }
    }

    #[test]
    fn encode_keeps_only_unreserved_bytes() {
        assert_eq!(encode(""), "");
        assert_eq!(encode("AZaz09-._~"), "AZaz09-._~");
        assert_eq!(encode(" "), "%20");
        assert_eq!(encode("/"), "%2F");
        assert_eq!(encode("+"), "%2B");
        assert_eq!(encode("ü"), "%C3%BC");
        assert_eq!(encode_bytes(&[0x00, 0xFF]), "%00%FF");
        // Hex digits are uppercase, matching the RFC 3986 examples.
        assert_eq!(encode("\u{7f}"), "%7F");
    }

    #[test]
    fn encode_bytes_round_trips_every_byte() {
        for value in 0u16..=255 {
            let byte = u8::try_from(value).unwrap();
            let encoded = encode_bytes(&[byte]);
            let expected_len = if is_unreserved(byte) { 1 } else { 3 };
            assert_eq!(encoded.len(), expected_len, "byte {byte:#04x}");
            assert_eq!(
                decode_bytes(&encoded).unwrap(),
                alloc::vec![byte],
                "byte {byte:#04x}"
            );
        }
    }

    #[test]
    fn encode_decode_round_trips_text() {
        for text in ["", "plain", "hello world", "a/b?c=d#e", "100% sure", "« »"] {
            let encoded = encode(text);
            assert_eq!(decode(&encoded).unwrap(), text, "{text}");
        }
    }

    #[test]
    fn decode_rejects_malformed_percent_sequences() {
        assert_eq!(decode("").unwrap(), "");
        assert_eq!(decode("+").unwrap(), "+");
        assert_eq!(
            decode("%").unwrap_err(),
            UrlError::InvalidPercentEncoding(0)
        );
        assert_eq!(
            decode("%2").unwrap_err(),
            UrlError::InvalidPercentEncoding(0)
        );
        assert_eq!(
            decode("%zz").unwrap_err(),
            UrlError::InvalidPercentEncoding(0)
        );
        assert_eq!(
            decode("a%2G").unwrap_err(),
            UrlError::InvalidPercentEncoding(1)
        );
        assert_eq!(decode("%2f").unwrap(), "/");
        assert_eq!(decode("%2F").unwrap(), "/");
    }

    #[test]
    fn decode_rejects_invalid_utf8() {
        assert_eq!(decode("%FF").unwrap_err(), UrlError::InvalidUtf8);
        assert_eq!(decode("%C3").unwrap_err(), UrlError::InvalidUtf8);
        assert_eq!(decode_bytes("%FF").unwrap(), alloc::vec![0xFF]);
    }

    /// The exact messages matter: callers surface `UrlError`'s display
    /// text as their own validation error payload.
    #[test]
    fn error_messages_are_stable() {
        assert_eq!(UrlError::EmptyInput.to_string(), "empty URL");
        assert_eq!(
            UrlError::MissingScheme.to_string(),
            "URL is missing a scheme"
        );
        assert_eq!(UrlError::InvalidScheme.to_string(), "invalid URL scheme");
        assert_eq!(
            UrlError::InvalidCharacter(7).to_string(),
            "invalid character at byte offset 7"
        );
        assert_eq!(
            UrlError::InvalidPercentEncoding(7).to_string(),
            "invalid percent-encoding at byte offset 7"
        );
        assert_eq!(
            UrlError::EmptyHost.to_string(),
            "URL authority has an empty host"
        );
        assert_eq!(UrlError::InvalidHost.to_string(), "invalid URL host");
        assert_eq!(
            UrlError::InvalidIpv6.to_string(),
            "invalid IPv6 host address"
        );
        assert_eq!(UrlError::InvalidPort.to_string(), "invalid URL port");
        assert_eq!(
            UrlError::InvalidUtf8.to_string(),
            "percent-decoded URL is not valid UTF-8"
        );
    }
}
