//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Shared literal parsing used by both the lexical checks and the
//! type-checking pass.
//!
//! Keeping digit parsing, suffix tables, and escape decoding in one place
//! means a literal gets exactly one interpretation: the value range check
//! in `lexcheck` and the type assigned in `analyzer` can never disagree.

use alloc::string::String;

use codevar_ocl_lex::Base;

use crate::types::Scalar;

/// An escape sequence the decoder does not define.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EscapeIssue {
    /// Byte offset of the backslash inside the decoded text.
    pub(crate) offset: u32,
    /// The character after the backslash.
    pub(crate) ch: char,
}

/// Strips the `r#` escape from a raw identifier, leaving other names alone.
pub(crate) fn strip_raw_ident(name: &str) -> &str {
    name.strip_prefix("r#").unwrap_or(name)
}

/// The integer scalar a literal suffix selects, `None` when unrecognized.
///
/// Both spellings are accepted: the Rust-style names the parser tests use
/// and the C suffixes (`u`, `l`, `ul`, `ll`, …) OpenCL source is full of.
pub(crate) fn int_scalar_for_suffix(suffix: &str) -> Option<Scalar> {
    match suffix {
        "i8" => Some(Scalar::I8),
        "i16" => Some(Scalar::I16),
        "i32" => Some(Scalar::I32),
        "i64" => Some(Scalar::I64),
        "u8" => Some(Scalar::U8),
        "u16" => Some(Scalar::U16),
        "u32" => Some(Scalar::U32),
        "u64" => Some(Scalar::U64),
        "isize" => Some(Scalar::I64),
        "usize" => Some(Scalar::U64),
        _ if suffix.eq_ignore_ascii_case("u") => Some(Scalar::U32),
        _ if suffix.eq_ignore_ascii_case("l") => Some(Scalar::I64),
        _ if suffix.eq_ignore_ascii_case("ll") => Some(Scalar::I64),
        _ if suffix.eq_ignore_ascii_case("ul") || suffix.eq_ignore_ascii_case("lu") => Some(Scalar::U64),
        _ if suffix.eq_ignore_ascii_case("ull") || suffix.eq_ignore_ascii_case("llu") => Some(Scalar::U64),
        _ => None,
    }
}

/// The float scalar a literal suffix selects, `None` when unrecognized.
///
/// `f` and `F` mean `float`, the C spelling every OpenCL kernel uses.
pub(crate) fn float_scalar_for_suffix(suffix: &str) -> Option<Scalar> {
    match suffix {
        "f32" => Some(Scalar::F32),
        "f64" => Some(Scalar::F64),
        _ if suffix.eq_ignore_ascii_case("f") => Some(Scalar::F32),
        _ => None,
    }
}

/// Parses the digits of an integer literal, underscores removed.
///
/// Returns `None` when the text is empty or does not fit in an `i128`,
/// which the caller reports as an out-of-range literal.
pub(crate) fn int_digits_value(text: &str, base: Base) -> Option<i128> {
    let digits = strip_radix_prefix(text, base);
    let mut cleaned = String::with_capacity(digits.len());
    cleaned.extend(
        digits
            .chars()
            .filter(|character| *character != '_'),
    );
    if cleaned.is_empty() {
        return None;
    }
    i128::from_str_radix(&cleaned, base as u32).ok()
}

/// Removes the `0x`/`0b`/`0o` prefix; decimal and C-style octal keep the
/// leading digit so `0755` parses as the octal constant it is.
fn strip_radix_prefix(text: &str, base: Base) -> &str {
    let has_o_prefix = text
        .get(1..2)
        .is_some_and(|letter| letter.eq_ignore_ascii_case("o"));
    match base {
        Base::Binary | Base::Hexadecimal => text.get(2..).unwrap_or(text),
        Base::Octal if has_o_prefix => text.get(2..).unwrap_or(text),
        _ => text,
    }
}

/// Parses a decimal float literal; hexadecimal floats return `None`
/// because `f64::from_str` does not implement the `0x1.8p3` grammar.
pub(crate) fn float_value(text: &str) -> Option<f64> {
    let mut cleaned = String::with_capacity(text.len());
    cleaned.extend(text.chars().filter(|character| *character != '_'));
    cleaned.parse::<f64>().ok()
}

/// Resolves `\n`, `\xNN`, `\u{...}`, and friends inside a literal body.
///
/// # Errors
///
/// Returns the first escape the dialect does not define, with its offset
/// inside `text`, so the caller can point at it.
pub(crate) fn decode_escapes(text: &str) -> Result<String, EscapeIssue> {
    let bytes = text.as_bytes();
    let mut decoded = String::with_capacity(text.len());
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'\\' {
            let rest = &text[index..];
            let Some(character) = rest.chars().next() else {
                break;
            };
            decoded.push(character);
            index += character.len_utf8();
            continue;
        }
        let start = index;
        index += 1;
        let Some(&kind) = bytes.get(index) else {
            return Err(EscapeIssue {
                offset: start as u32,
                ch: '\\',
            });
        };
        index += 1;
        match kind {
            b'n' => decoded.push('\n'),
            b'r' => decoded.push('\r'),
            b't' => decoded.push('\t'),
            b'0' => decoded.push('\0'),
            b'\\' => decoded.push('\\'),
            b'\'' => decoded.push('\''),
            b'"' => decoded.push('"'),
            b'x' => {
                let value = parse_hex_digits(&text[index..], 2).ok_or(EscapeIssue {
                    offset: start as u32,
                    ch: 'x',
                })?;
                index += 2;
                decoded.push(char::from_u32(value).ok_or(EscapeIssue {
                    offset: start as u32,
                    ch: 'x',
                })?);
            }
            b'u' => {
                let (character, consumed) = parse_unicode_escape(&text[index..]).ok_or(EscapeIssue {
                    offset: start as u32,
                    ch: 'u',
                })?;
                index += consumed;
                decoded.push(character);
            }
            other => {
                let ch = text[start + 1..]
                    .chars()
                    .next()
                    .unwrap_or(other as char);
                return Err(EscapeIssue {
                    offset: start as u32,
                    ch,
                });
            }
        }
    }
    Ok(decoded)
}

/// Reads exactly `count` hexadecimal digits from the front of `text`.
fn parse_hex_digits(text: &str, count: usize) -> Option<u32> {
    let digits = text.get(..count)?;
    let mut value = 0u32;
    for character in digits.chars() {
        let digit = character.to_digit(16)?;
        value = value * 16 + digit;
    }
    Some(value)
}

/// Reads `{…}` after a `\u` escape, returning the character and the number
/// of bytes consumed after the backslash-u marker.
fn parse_unicode_escape(text: &str) -> Option<(char, usize)> {
    let after_open = text.strip_prefix('{')?;
    let digits = after_open
        .chars()
        .take_while(char::is_ascii_hexdigit)
        .count();
    if digits == 0 || digits > 6 {
        return None;
    }
    let value = parse_hex_digits(after_open, digits)?;
    let after_digits = after_open.get(digits..)?;
    let after_close = after_digits.strip_prefix('}')?;
    let consumed = text.len() - after_close.len();
    char::from_u32(value).map(|character| (character, consumed))
}
