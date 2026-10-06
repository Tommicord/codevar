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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Low-level lexer for the Codevar OpenCL dialect (Rust-like syntax).
//!
//! The lexer follows the `rustc_lexer` design: it is a pure function of the
//! source text that produces [`Token`]s pairing a [`TokenKind`] with a byte
//! length, and never reports diagnostics — lexical problems are flags on the
//! tokens (e.g. [`LiteralKind::Str::terminated`]). Consequences of the
//! design, which higher layers rely on:
//!
//! - **No spans.** Tokens carry relative lengths; consumers accumulate them
//!   in source order to derive byte offsets.
//! - **Trivia is first-class.** Whitespace and comments are tokens, so a
//!   lossless syntax tree can be built over them.
//! - **Keywords are not resolved.** `kernel`, `return`, … lex as
//!   [`TokenKind::Ident`]; the parse layer owns keyword recognition.
//! - **Multi-character operators are not glued.** `->` lexes as two tokens
//!   (`-`, `>`), the parse layer joins them like `rustc`'s token trees.
//! - **Lifetimes are core tokens.** The dialect has Rust-like ownership
//!   semantics, so `'a` is a [`TokenKind::Lifetime`], not punctuation.
//!
//! # Examples
//!
//! ```
//! use codevar_ocl_lex::{tokenize, TokenKind};
//!
//! let kinds: Vec<TokenKind> = tokenize("a + b").map(|token| token.kind).collect();
//! assert_eq!(kinds.len(), 5);
//! assert_eq!(kinds[0], TokenKind::Ident);
//! assert_eq!(kinds[2], TokenKind::Plus);
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]
extern crate alloc;

mod cursor;
mod token;

#[cfg(test)]
use crate::TokenKind::*;
#[cfg(test)]
use alloc::vec::Vec;
pub use cursor::Cursor;
pub use token::{Base, DocStyle, LiteralKind, RawStrError, Token, TokenKind};

/// Creates an iterator that produces tokens from the input string.
///
/// Trivia tokens are included, and the iterator stops *before* yielding
/// [`TokenKind::Eof`]. The sum of the tokens' `len` fields always equals
/// `input.len()`.
///
/// # Examples
///
/// ```
/// use codevar_ocl_lex::{tokenize, TokenKind};
///
/// let tokens: Vec<TokenKind> = tokenize("x;").map(|token| token.kind).collect();
/// assert_eq!(tokens, [TokenKind::Ident, TokenKind::Semi]);
/// ```
pub fn tokenize(input: &str) -> impl Iterator<Item = Token> {
    let mut cursor = Cursor::new(input);
    core::iter::from_fn(move || {
        let token = cursor.advance_token();
        if token.kind != TokenKind::Eof {
            Some(token)
        } else {
            None
        }
    })
}

/// True if `c` is whitespace for this dialect.
///
/// The set is `Pattern_White_Space`, a stable Unicode subset: tab through
/// carriage return (`\t`, `\n`, `\v`, `\f`, `\r`), space, NEL (U+0085),
/// LRM/RLM (U+200E/U+200F), and the line/paragraph separators (U+2028,
/// U+2029).
#[must_use]
pub fn is_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'..='\u{000D}' | '\u{0020}' | '\u{0085}' | '\u{200E}' | '\u{200F}' | '\u{2028}' | '\u{2029}'
    )
}

/// True if `c` is horizontal whitespace (tab and space).
#[must_use]
pub fn is_horizontal_whitespace(c: char) -> bool {
    matches!(c, '\u{0009}' | '\u{0020}')
}

/// True if `c` is valid as the first character of an identifier.
///
/// XID_Start plus `_` (which is formally not part of XID_Start).
#[must_use]
pub fn is_id_start(c: char) -> bool {
    c == '_' || unicode_ident::is_xid_start(c)
}

/// True if `c` is valid as a non-first character of an identifier.
#[must_use]
pub fn is_id_continue(c: char) -> bool {
    unicode_ident::is_xid_continue(c)
}

/// True if `string` is lexically a single identifier.
#[must_use]
pub fn is_ident(string: &str) -> bool {
    let mut chars = string.chars();
    if let Some(start) = chars.next() {
        is_id_start(start) && chars.all(is_id_continue)
    } else {
        false
    }
}

/// Validates a raw string literal such as `r##"ab"c##`.
///
/// `prefix_len` is the byte length of the prefix before the `#`s and the
/// opening quote (the `r` of `r"…"`, i.e. `1`). `input` should be the raw
/// string as it appears in source; trailing suffix text is ignored.
///
/// The lexer already flags invalid raw strings as
/// [`LiteralKind::RawStr { n_hashes: None }`](LiteralKind::RawStr); this
/// function exists to turn that flag into a precise message.
///
/// # Errors
///
/// Returns the reason the literal is malformed: a bad character between `r`
/// and the quote ([`RawStrError::InvalidStarter`]), a missing terminator
/// ([`RawStrError::NoTerminator`]), or more than 255 `#`s
/// ([`RawStrError::TooManyDelimiters`]).
///
/// # Panics
///
/// Never panics; an empty or truncated `input` reports
/// [`RawStrError::InvalidStarter`].
pub fn validate_raw_str(input: &str, prefix_len: u32) -> Result<(), RawStrError> {
    let mut cursor = Cursor::new(input);
    for _ in 0..prefix_len {
        if cursor.bump().is_none() {
            break;
        }
    }
    cursor
        .raw_double_quoted_string(prefix_len)
        .map(|_| ())
}

#[cfg(test)]
fn kinds(source: &str) -> Vec<TokenKind> {
    tokenize(source).map(|token| token.kind).collect()
}

#[cfg(test)]
fn first(source: &str) -> TokenKind {
    tokenize(source)
        .next()
        .map_or(Unknown, |token| token.kind)
}

#[cfg(test)]
mod tests {
    use super::LiteralKind::*;
    use super::*;
    use alloc::format;
    use alloc::vec::Vec;

    #[test]
    fn empty_input_yields_no_tokens() {
        assert!(tokenize("").next().is_none());
    }

    #[test]
    fn eof_is_returned_only_by_advance_token() {
        let mut cursor = Cursor::new("");
        assert_eq!(cursor.advance_token().kind, TokenKind::Eof);
        assert_eq!(cursor.advance_token().kind, TokenKind::Eof);
    }

    #[test]
    fn token_lengths_sum_to_source_length() {
        let sources = [
            "",
            " \t\r\n ",
            "kernel void main() { return; }",
            "'a' 'static '0 ''' 'abc' '",
            "\"abc\" \"unterminated\nnext",
            "\"line \\\ncontinued\"",
            "r\"raw\" r#\"raw\"# r#\"unterminated",
            "r~\"broken\"",
            "0 0755 0x1F 0b1010 0o17 1_000 1.0 1e10 1e 0x1.8p3 0x1.8 1.0f 1u8",
            "// line comment\n/* block */ /* unterminated",
            "r#type _x1 aé foo\"bar\" №",
            "0..2 12.foo -> :: << == =>",
            "/* /* nested */ still */ done",
        ];
        for source in sources {
            let total: usize = tokenize(source)
                .map(|token| token.len as usize)
                .sum();
            assert_eq!(total, source.len(), "length mismatch for source {source:?}");
        }
    }

    #[test]
    fn keywords_are_plain_identifiers() {
        assert_eq!(kinds("kernel void"), [Ident, Whitespace, Ident]);
        assert_eq!(first("return"), Ident);
    }

    #[test]
    fn whitespace_is_a_single_run() {
        assert_eq!(kinds(" \t\n "), [Whitespace]);
        assert_eq!(kinds("a \t\n b"), [Ident, Whitespace, Ident]);
    }

    #[test]
    fn identifiers_and_raw_identifiers() {
        assert_eq!(first("foo"), Ident);
        assert_eq!(first("_"), Ident);
        assert_eq!(first("get_global_id"), Ident);
        assert_eq!(first("aé"), Ident);
        assert_eq!(first("r#type"), RawIdent);
        let raw = tokenize("r#type")
            .next()
            .map_or(0, |token| token.len);
        assert_eq!(raw, 6);
    }

    #[test]
    fn line_comment_doc_styles() {
        assert_eq!(first("// plain"), LineComment { doc_style: None });
        assert_eq!(
            first("/// outer"),
            LineComment {
                doc_style: Some(DocStyle::Outer)
            }
        );
        assert_eq!(
            first("//! inner"),
            LineComment {
                doc_style: Some(DocStyle::Inner)
            }
        );
        assert_eq!(first("//// four"), LineComment { doc_style: None });
        assert_eq!(first("//"), LineComment { doc_style: None });
    }

    #[test]
    fn block_comment_doc_styles_and_nesting() {
        assert_eq!(
            first("/* plain */"),
            BlockComment {
                doc_style: None,
                terminated: true
            }
        );
        assert_eq!(
            first("/** outer */"),
            BlockComment {
                doc_style: Some(DocStyle::Outer),
                terminated: true
            }
        );
        assert_eq!(
            first("/*! inner */"),
            BlockComment {
                doc_style: Some(DocStyle::Inner),
                terminated: true
            }
        );
        assert_eq!(
            first("/**/"),
            BlockComment {
                doc_style: None,
                terminated: true
            }
        );
        assert_eq!(
            first("/* /* */"),
            BlockComment {
                doc_style: None,
                terminated: false
            }
        );
        assert_eq!(
            first("/*"),
            BlockComment {
                doc_style: None,
                terminated: false
            }
        );
        assert_eq!(
            first("/* /* nested */ still */"),
            BlockComment {
                doc_style: None,
                terminated: true
            }
        );
    }

    #[test]
    fn lifetimes() {
        assert_eq!(
            first("'a"),
            Lifetime {
                starts_with_number: false
            }
        );
        assert_eq!(
            first("'static"),
            Lifetime {
                starts_with_number: false
            }
        );
        assert_eq!(
            first("'0"),
            Lifetime {
                starts_with_number: true
            }
        );
        assert_eq!(
            first("'a'"),
            Literal {
                kind: Char { terminated: true },
                suffix_start: 3
            }
        );
    }

    #[test]
    fn character_literals() {
        assert_eq!(
            first("'x'"),
            Literal {
                kind: Char { terminated: true },
                suffix_start: 3
            }
        );
        assert_eq!(
            first("'''"),
            Literal {
                kind: Char { terminated: true },
                suffix_start: 3
            }
        );
        assert_eq!(
            first("';'"),
            Literal {
                kind: Char { terminated: true },
                suffix_start: 3
            }
        );
        assert_eq!(
            first("'\\n'"),
            Literal {
                kind: Char { terminated: true },
                suffix_start: 4
            }
        );
        assert_eq!(
            first("'abc'"),
            Literal {
                kind: Char { terminated: true },
                suffix_start: 5
            }
        );
        assert_eq!(
            first("'a"),
            Lifetime {
                starts_with_number: false
            }
        );
        assert_eq!(
            first("';"),
            Literal {
                kind: Char { terminated: false },
                suffix_start: 2
            }
        );
        assert_eq!(
            first("'"),
            Literal {
                kind: Char { terminated: false },
                suffix_start: 1
            }
        );
    }

    #[test]
    fn string_literals() {
        assert_eq!(
            first("\"abc\""),
            Literal {
                kind: Str { terminated: true },
                suffix_start: 5
            }
        );
        assert_eq!(
            first("\"abc"),
            Literal {
                kind: Str { terminated: false },
                suffix_start: 4
            }
        );
        assert_eq!(
            first("\"a\\\"b\""),
            Literal {
                kind: Str { terminated: true },
                suffix_start: 6
            }
        );
        assert_eq!(
            kinds("\"ab\ncd\""),
            [
                Literal {
                    kind: Str { terminated: false },
                    suffix_start: 3
                },
                Whitespace,
                UnknownPrefix,
                Literal {
                    kind: Str { terminated: false },
                    suffix_start: 1
                },
            ]
        );
    }

    #[test]
    fn string_line_continuation_keeps_literal_open() {
        let source = "\"ab\\\ncd\"";
        let tokens: Vec<_> = tokenize(source).collect();
        assert_eq!(tokens.len(), 1);
        assert_eq!(
            tokens[0].kind,
            Literal {
                kind: Str { terminated: true },
                suffix_start: source.len() as u32,
            }
        );
    }

    #[test]
    fn string_suffix_is_part_of_the_token() {
        assert_eq!(
            first("\"abc\"x"),
            Literal {
                kind: Str { terminated: true },
                suffix_start: 5
            }
        );
        let token = tokenize("\"abc\"x")
            .next()
            .map_or(0, |token| token.len);
        assert_eq!(token, 6);
    }

    #[test]
    fn raw_string_literals() {
        assert_eq!(
            first("r\"abc\""),
            Literal {
                kind: RawStr { n_hashes: Some(0) },
                suffix_start: 6
            }
        );
        assert_eq!(
            first("r#\"ab\"c\"#"),
            Literal {
                kind: RawStr { n_hashes: Some(1) },
                suffix_start: 9
            }
        );
        assert_eq!(
            first("r#\"unterminated"),
            Literal {
                kind: RawStr { n_hashes: None },
                suffix_start: 15
            }
        );
        assert_eq!(first("r~\"broken\""), Ident);
        assert_eq!(
            first("r#~\"y\""),
            Literal {
                kind: RawStr { n_hashes: None },
                suffix_start: 3
            }
        );
    }

    #[test]
    fn validate_raw_str_reports_specific_errors() {
        assert!(validate_raw_str("r\"abc\"", 1).is_ok());
        assert!(validate_raw_str("r#\"ab\"c\"#", 1).is_ok());
        assert!(matches!(
            validate_raw_str("r#\"abc", 1),
            Err(RawStrError::NoTerminator { expected: 1, .. })
        ));
        assert!(matches!(
            validate_raw_str("r~\"x\"", 1),
            Err(RawStrError::InvalidStarter { bad_char: '~' })
        ));
        let many = format!("r{}\"x\"{}", "#".repeat(256), "#".repeat(256));
        assert!(matches!(
            validate_raw_str(&many, 1),
            Err(RawStrError::TooManyDelimiters { found: 256 })
        ));
    }

    #[test]
    fn integer_literals() {
        assert_eq!(
            first("123"),
            Literal {
                kind: Int {
                    base: Base::Decimal,
                    empty_int: false
                },
                suffix_start: 3
            }
        );
        assert_eq!(
            first("0x1F"),
            Literal {
                kind: Int {
                    base: Base::Hexadecimal,
                    empty_int: false
                },
                suffix_start: 4
            }
        );
        assert_eq!(
            first("0b1010"),
            Literal {
                kind: Int {
                    base: Base::Binary,
                    empty_int: false
                },
                suffix_start: 6
            }
        );
        assert_eq!(
            first("0o17"),
            Literal {
                kind: Int {
                    base: Base::Octal,
                    empty_int: false
                },
                suffix_start: 4
            }
        );
        assert_eq!(
            first("0755"),
            Literal {
                kind: Int {
                    base: Base::Octal,
                    empty_int: false
                },
                suffix_start: 4
            }
        );
        assert_eq!(
            first("0"),
            Literal {
                kind: Int {
                    base: Base::Decimal,
                    empty_int: false
                },
                suffix_start: 1
            }
        );
        assert_eq!(
            first("0x"),
            Literal {
                kind: Int {
                    base: Base::Hexadecimal,
                    empty_int: true
                },
                suffix_start: 2
            }
        );
        assert_eq!(
            first("0b"),
            Literal {
                kind: Int {
                    base: Base::Binary,
                    empty_int: true
                },
                suffix_start: 2
            }
        );
        assert_eq!(
            first("1_000"),
            Literal {
                kind: Int {
                    base: Base::Decimal,
                    empty_int: false
                },
                suffix_start: 5
            }
        );
    }

    #[test]
    fn float_literals() {
        assert_eq!(
            first("1.0"),
            Literal {
                kind: Float {
                    base: Base::Decimal,
                    empty_exponent: false
                },
                suffix_start: 3
            }
        );
        assert_eq!(
            first("1e10"),
            Literal {
                kind: Float {
                    base: Base::Decimal,
                    empty_exponent: false
                },
                suffix_start: 4
            }
        );
        assert_eq!(
            first("1e"),
            Literal {
                kind: Float {
                    base: Base::Decimal,
                    empty_exponent: true
                },
                suffix_start: 2
            }
        );
        assert_eq!(
            first("0x1.8p3"),
            Literal {
                kind: Float {
                    base: Base::Hexadecimal,
                    empty_exponent: false
                },
                suffix_start: 7
            }
        );
        assert_eq!(
            first("0x1.8"),
            Literal {
                kind: Float {
                    base: Base::Hexadecimal,
                    empty_exponent: true
                },
                suffix_start: 5
            }
        );
        assert_eq!(
            first("1.0f"),
            Literal {
                kind: Float {
                    base: Base::Decimal,
                    empty_exponent: false
                },
                suffix_start: 3
            }
        );
        let token = tokenize("1.0f")
            .next()
            .map_or(0, |token| token.len);
        assert_eq!(token, 4);
    }

    #[test]
    fn integer_suffix_is_part_of_the_token() {
        assert_eq!(
            first("1u8"),
            Literal {
                kind: Int {
                    base: Base::Decimal,
                    empty_int: false
                },
                suffix_start: 1
            }
        );
        let token = tokenize("1u8")
            .next()
            .map_or(0, |token| token.len);
        assert_eq!(token, 3);
    }

    #[test]
    fn numbers_do_not_swallow_ranges_or_field_access() {
        assert_eq!(
            kinds("0..2"),
            [
                Literal {
                    kind: Int {
                        base: Base::Decimal,
                        empty_int: false
                    },
                    suffix_start: 1
                },
                Dot,
                Dot,
                Literal {
                    kind: Int {
                        base: Base::Decimal,
                        empty_int: false
                    },
                    suffix_start: 1
                },
            ]
        );
        assert_eq!(
            kinds("12.foo"),
            [
                Literal {
                    kind: Int {
                        base: Base::Decimal,
                        empty_int: false
                    },
                    suffix_start: 2
                },
                Dot,
                Ident,
            ]
        );
    }

    #[test]
    fn punctuation_is_single_character() {
        assert_eq!(
            kinds(";,.(){}[]@#~?:$=!<>-&|+*/^%"),
            [
                Semi,
                Comma,
                Dot,
                OpenParen,
                CloseParen,
                OpenBrace,
                CloseBrace,
                OpenBracket,
                CloseBracket,
                At,
                Pound,
                Tilde,
                Question,
                Colon,
                Dollar,
                Eq,
                Bang,
                Lt,
                Gt,
                Minus,
                And,
                Or,
                Plus,
                Star,
                Slash,
                Caret,
                Percent,
            ]
        );
    }

    #[test]
    fn multi_character_operators_are_not_glued() {
        assert_eq!(kinds("->::<<=="), [Minus, Gt, Colon, Colon, Lt, Lt, Eq, Eq]);
    }

    #[test]
    fn unknown_literal_prefixes() {
        assert_eq!(
            kinds("foo\"bar\""),
            [
                UnknownPrefix,
                Literal {
                    kind: Str { terminated: true },
                    suffix_start: 5
                }
            ]
        );
        assert_eq!(
            kinds("foo'x'"),
            [
                UnknownPrefix,
                Literal {
                    kind: Char { terminated: true },
                    suffix_start: 3
                }
            ]
        );
        assert_eq!(
            kinds("b\"abc\""),
            [
                UnknownPrefix,
                Literal {
                    kind: Str { terminated: true },
                    suffix_start: 5
                }
            ]
        );
    }

    #[test]
    fn unknown_characters_keep_their_utf8_length() {
        assert_eq!(first("№"), Unknown);
        let token = tokenize("№").next().map_or(0, |token| token.len);
        assert_eq!(token, 3);
        assert_eq!(kinds("a🚀b"), [Ident, Unknown, Ident]);
    }

    #[test]
    fn predicates() {
        assert!(is_whitespace(' '));
        assert!(is_whitespace('\n'));
        assert!(is_whitespace('\u{2028}'));
        assert!(!is_whitespace('a'));
        assert!(is_horizontal_whitespace('\t'));
        assert!(!is_horizontal_whitespace('\n'));
        assert!(is_id_start('a'));
        assert!(is_id_start('_'));
        assert!(!is_id_start('1'));
        assert!(is_id_continue('1'));
        assert!(is_id_continue('_'));
        assert!(!is_id_continue('-'));
        assert!(is_ident("foo_bar"));
        assert!(is_ident("_"));
        assert!(!is_ident(""));
        assert!(!is_ident("1a"));
        assert!(!is_ident("a-b"));
    }
}
