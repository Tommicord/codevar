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

//! Lexical validation: turning the lexer's error flags into diagnostics.
//!
//! `codevar-ocl-lex` never reports problems itself — unterminated
//! literals, empty numbers, and stray characters ride on token variants.
//! This pass walks the token stream once and converts those flags into
//! spanned [`Diagnostic`]s with stable codes, including the checks that
//! need the literal's *text*: suffix tables, digit range, escape
//! decoding, and character-literal length.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use codevar_ocl_lex::{Base, LiteralKind, Token, TokenKind, tokenize, validate_raw_str};
use codevar_ocl_parse::Span;

use crate::diagnostic::{Diagnostic, codes};
use crate::literal::{
    decode_escapes, float_scalar_for_suffix, float_value, int_digits_value, int_scalar_for_suffix,
};
use crate::types::Scalar;

/// Reports every lexical problem in `source`, in source order.
pub(crate) fn check(source: &str, diags: &mut Vec<Diagnostic>) {
    let mut offset: u32 = 0;
    for token in tokenize(source) {
        let start = offset;
        let end = start.saturating_add(token.len);
        let Some(text) = source.get(start as usize..end as usize) else {
            break;
        };
        check_token(start, text, token, diags);
        offset = end;
    }
}

/// Classifies a single token and pushes its diagnostic, if any.
fn check_token(start: u32, text: &str, token: Token, diags: &mut Vec<Diagnostic>) {
    match token.kind {
        TokenKind::Literal { kind, suffix_start } => {
            check_literal(start, text, kind, suffix_start, diags);
        }
        TokenKind::BlockComment {
            terminated: false, ..
        } => push(
            diags,
            start,
            token.len,
            "unterminated block comment",
            codes::UNTERMINATED_BLOCK_COMMENT,
        ),
        TokenKind::Lifetime {
            starts_with_number: true,
        } => push(
            diags,
            start,
            token.len,
            "lifetimes cannot start with a number",
            codes::LIFETIME_STARTS_WITH_NUMBER,
        ),
        TokenKind::Unknown => {
            let character = text.chars().next().unwrap_or('\u{fffd}');
            push(
                diags,
                start,
                token.len,
                format!("character not allowed in source code: {character:?}"),
                codes::UNKNOWN_CHARACTER,
            );
        }
        TokenKind::UnknownPrefix => push(
            diags,
            start,
            token.len,
            format!("unknown literal prefix `{text}`"),
            codes::UNKNOWN_LITERAL_PREFIX,
        ),
        _ => {}
    }
}

/// Pushes a single-label diagnostic covering `[start, start + len)`.
fn push(diags: &mut Vec<Diagnostic>, start: u32, len: u32, message: impl Into<String>, code: &'static str) {
    diags.push(Diagnostic::error(Span::new(start, len), message).with_code(code));
}

/// Validates one literal token: termination flags, suffix tables, digit
/// ranges, escape sequences, and character-literal length.
fn check_literal(start: u32, text: &str, kind: LiteralKind, suffix_start: u32, diags: &mut Vec<Diagnostic>) {
    let split = (suffix_start as usize).min(text.len());
    let (body, suffix) = text.split_at(split);
    match kind {
        LiteralKind::Int { empty_int: true, .. } => {
            push(
                diags,
                start,
                text.len() as u32,
                "integer literal has no digits",
                codes::EMPTY_INTEGER_LITERAL,
            );
        }
        LiteralKind::Int { base, .. } => check_int(start, text, body, suffix, base, diags),
        LiteralKind::Float {
            base,
            empty_exponent: true,
        } => {
            let message = if base == Base::Hexadecimal {
                "hexadecimal float literal requires an exponent"
            } else {
                "exponent has no digits"
            };
            push(diags, start, text.len() as u32, message, codes::EMPTY_EXPONENT);
        }
        LiteralKind::Float { base, .. } => check_float(start, text, body, suffix, base, diags),
        LiteralKind::Char { terminated: false } => push(
            diags,
            start,
            text.len() as u32,
            "unterminated character literal",
            codes::UNTERMINATED_CHAR,
        ),
        LiteralKind::Char { .. } => check_char(start, text, body, suffix, diags),
        LiteralKind::Str { terminated: false } => push(
            diags,
            start,
            text.len() as u32,
            "unterminated string literal",
            codes::UNTERMINATED_STRING,
        ),
        LiteralKind::Str { .. } => check_string(start, body, suffix, diags),
        LiteralKind::RawStr { n_hashes } => check_raw_str(start, text, body, suffix, n_hashes, diags),
    }
}

/// Integer literals: suffix table and value range.
fn check_int(start: u32, text: &str, body: &str, suffix: &str, base: Base, diags: &mut Vec<Diagnostic>) {
    if suffix.is_empty() {
        let range = match int_digits_value(body, base) {
            Some(value) if fits_any_int(value) => None,
            _ => Some("integer literal is too large to fit in any type"),
        };
        if let Some(message) = range {
            push(
                diags,
                start,
                text.len() as u32,
                message,
                codes::LITERAL_OUT_OF_RANGE,
            );
        }
        return;
    }
    let Some(scalar) = int_scalar_for_suffix(suffix) else {
        let diagnostic = Diagnostic::error(
            Span::new(start + body.len() as u32, suffix.len() as u32),
            format!("invalid literal suffix `{suffix}`"),
        )
        .with_code(codes::INVALID_LITERAL_SUFFIX)
        .with_help(
            "expected one of `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `isize`, or `usize`",
        );
        diags.push(diagnostic);
        return;
    };
    let display = scalar.as_str();
    let in_range = match int_digits_value(body, base) {
        Some(value) => scalar.fits_int(value),
        None => false,
    };
    if !in_range {
        push(
            diags,
            start,
            text.len() as u32,
            format!("literal out of range for `{display}`"),
            codes::LITERAL_OUT_OF_RANGE,
        );
    }
}

/// True when at least one built-in integer type can hold `value`.
fn fits_any_int(value: i128) -> bool {
    [
        Scalar::I8,
        Scalar::I16,
        Scalar::I32,
        Scalar::I64,
        Scalar::U8,
        Scalar::U16,
        Scalar::U32,
        Scalar::U64,
    ]
    .iter()
    .any(|scalar| scalar.fits_int(value))
}

/// Float literals: suffix table and value range.
fn check_float(start: u32, text: &str, body: &str, suffix: &str, base: Base, diags: &mut Vec<Diagnostic>) {
    if !suffix.is_empty() && float_scalar_for_suffix(suffix).is_none() {
        diags.push(
            Diagnostic::error(
                Span::new(start + body.len() as u32, suffix.len() as u32),
                format!("invalid literal suffix `{suffix}`"),
            )
            .with_code(codes::INVALID_LITERAL_SUFFIX)
            .with_help("expected `f32`, `f64`, or the C-style `f` suffix"),
        );
        return;
    }
    if base == Base::Hexadecimal {
        return;
    }
    let Some(value) = float_value(body) else {
        return;
    };
    let limit = match float_scalar_for_suffix(suffix) {
        Some(Scalar::F32) => Some(f32::MAX as f64),
        Some(Scalar::F64) => None,
        _ => {
            if value.is_finite() {
                return;
            }
            Some(f64::MAX)
        }
    };
    let out_of_range = value.is_nan() || !value.is_finite() || limit.is_some_and(|limit| value.abs() > limit);
    if out_of_range {
        let message = match float_scalar_for_suffix(suffix) {
            Some(scalar) => format!("literal out of range for `{}`", scalar.as_str()),
            None => String::from("literal out of range"),
        };
        push(
            diags,
            start,
            text.len() as u32,
            message,
            codes::LITERAL_OUT_OF_RANGE,
        );
    }
}

/// Character literals: suffixes, escapes, and exactly one character.
fn check_char(start: u32, text: &str, body: &str, suffix: &str, diags: &mut Vec<Diagnostic>) {
    if !suffix.is_empty() {
        push_suffix_error(diags, start, body, suffix, "character");
        return;
    }
    let inner = strip_quotes(body);
    let decoded = match decode_escapes(inner) {
        Ok(decoded) => decoded,
        Err(issue) => {
            push_escape(diags, start, issue.offset, issue.ch);
            return;
        }
    };
    let count = decoded.chars().count();
    if count != 1 {
        push(
            diags,
            start,
            text.len() as u32,
            format!("character literal must contain exactly one character, found {count}"),
            codes::CHAR_LITERAL_TOO_LONG,
        );
    }
}

/// String literals: suffixes and escapes (raw strings skip this).
fn check_string(start: u32, body: &str, suffix: &str, diags: &mut Vec<Diagnostic>) {
    if !suffix.is_empty() {
        push_suffix_error(diags, start, body, suffix, "string");
        return;
    }
    let inner = strip_quotes(body);
    if let Err(issue) = decode_escapes(inner) {
        push_escape(diags, start, issue.offset, issue.ch);
    }
}

/// Raw strings: lexer flag, precise message, and suffixes.
fn check_raw_str(
    start: u32,
    text: &str,
    body: &str,
    suffix: &str,
    n_hashes: Option<u8>,
    diags: &mut Vec<Diagnostic>,
) {
    if n_hashes.is_none() {
        let message = validate_raw_str(body, 1).map_or_else(
            |error| error.to_string(),
            |_| String::from("invalid raw string literal"),
        );
        push(
            diags,
            start,
            text.len() as u32,
            message,
            codes::INVALID_RAW_STRING,
        );
    }
    if !suffix.is_empty() {
        push_suffix_error(diags, start, body, suffix, "string");
    }
}

/// Reports a suffix on a literal kind that cannot take one.
fn push_suffix_error(diags: &mut Vec<Diagnostic>, start: u32, body: &str, suffix: &str, kind: &str) {
    diags.push(
        Diagnostic::error(
            Span::new(start + body.len() as u32, suffix.len() as u32),
            format!("invalid literal suffix `{suffix}`"),
        )
        .with_code(codes::INVALID_LITERAL_SUFFIX)
        .with_help(format!("{kind} literals cannot have a suffix")),
    );
}

/// Reports an undefined escape at its offset inside the literal.
fn push_escape(diags: &mut Vec<Diagnostic>, start: u32, offset: u32, character: char) {
    diags.push(
        Diagnostic::error(
            Span::new(start + offset + 1, 1 + character.len_utf8() as u32),
            format!("unknown escape sequence: `\\{character}`"),
        )
        .with_code(codes::UNKNOWN_ESCAPE_SEQUENCE)
        .with_help("escape the backslash itself as `\\\\` if that was intended"),
    );
}

/// Removes the surrounding quotes from a literal body.
fn strip_quotes(body: &str) -> &str {
    if matches!(body.as_bytes().first(), Some(b'\'' | b'"')) && body.len() >= 2 {
        body.get(1..body.len() - 1).unwrap_or(body)
    } else {
        body
    }
}
