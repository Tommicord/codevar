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

//! Lossless lexer for Negex regular expression strings.
//!
//! The lexer follows the `rustc_lexer`/`codevar-ocl-lex` design: it is a
//! pure function of the pattern text that produces [`Token`]s pairing a
//! [`TokenKind`] with a byte length, and never reports diagnostics —
//! lexical problems are flags on the tokens.  Consequences of the design:
//!
//! - **No spans.** Tokens carry relative lengths; consumers accumulate them
//!   in source order to derive byte offsets.
//! - **Trivia is first-class.** Whitespace (outside character classes) is a
//!   token, so a lossless token tree can be built.
//! - **Escapes are single tokens.** `\d`, `\w`, `\s`, `\p{L}`, `\n` … each
//!   lex as a single escape token carrying the escaped text.
//! - **Multi-character operators are not glued.** `\{`, `\}`, `(` … lex as
//!   individual punctuation; the parse layer owns grouping.
//!
//! # Examples
//!
//! ```
//! use codevar_negex_lex::{tokenize, TokenKind};
//!
//! let kinds: Vec<TokenKind> = tokenize("a*b").map(|token| token.kind).collect();
//! assert_eq!(kinds, [TokenKind::Literal('a'), TokenKind::Star, TokenKind::Literal('b')]);
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod cursor;
mod token;

pub use cursor::Cursor;
pub use token::{Token, TokenKind};

/// Creates an iterator that produces tokens from the regex pattern.
///
/// Trivia tokens are included, and the iterator stops *before* yielding
/// [`TokenKind::Eof`]. The sum of the tokens' `len` fields always equals
/// `input.len()`.
///
/// # Examples
///
/// ```
/// use codevar_negex_lex::tokenize;
///
/// let tokens: Vec<_> = tokenize("ab").collect();
/// assert_eq!(tokens.len(), 2);
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

/// True if `c` is whitespace outside a character class.
#[must_use]
pub fn is_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'..='\u{000D}' | '\u{0020}' | '\u{0085}' | '\u{200E}' | '\u{200F}' | '\u{2028}' | '\u{2029}'
    )
}

#[cfg(test)]
mod tests {
    use super::TokenKind::*;
    use super::*;

    #[cfg(test)]
    fn kinds(input: &str) -> Vec<TokenKind> {
        tokenize(input).map(|token| token.kind).collect()
    }

    #[test]
    fn empty_pattern_yields_no_tokens() {
        assert!(tokenize("").next().is_none());
    }

    #[test]
    fn token_lengths_sum_to_pattern_length() {
        let patterns = [
            "",
            " ",
            "a*b",
            "a|b",
            "(abc)",
            "[0-9]",
            "a{2,4}",
            "a\\nb",
            "\\p{L}",
            "a{2,}",
            "[^0-9]",
            "\\d\\w\\s",
            r"\.\*\+\?\|\(\)\[\]\{\}",
        ];
        for pattern in patterns {
            let total: usize = tokenize(pattern)
                .map(|token| token.len as usize)
                .sum();
            assert_eq!(total, pattern.len(), "length mismatch for pattern {pattern:?}");
        }
    }

    #[test]
    fn every_token_kind() {
        // Literal, Pipe, Star, Plus, Question, OpenParen, CloseParen,
        // OpenBracket, CloseBracket, OpenBrace, CloseBrace, Comma, Dot,
        // Caret, Dollar, Hyphen, Whitespace, Unknown, Eof are all covered
        // by the individual tests below; this test asserts the enum is
        // non-exhaustive-free by matching each variant at least once.
        let _ = kinds("a|b*+?()[]{}.,^$- ");
    }

    #[test]
    fn alternation() {
        assert_eq!(
            kinds("a|b|c"),
            [Literal('a'), Pipe, Literal('b'), Pipe, Literal('c')]
        );
    }

    #[test]
    fn quantifiers() {
        assert_eq!(kinds("a*"), [Literal('a'), Star]);
        assert_eq!(kinds("a+"), [Literal('a'), Plus]);
        assert_eq!(kinds("a?"), [Literal('a'), Question]);
        assert_eq!(kinds("a{2}"), [Literal('a'), OpenBrace, Literal('2'), CloseBrace]);
        assert_eq!(
            kinds("a{2,4}"),
            [
                Literal('a'),
                OpenBrace,
                Literal('2'),
                Comma,
                Literal('4'),
                CloseBrace,
            ]
        );
        assert_eq!(
            kinds("a{2,}"),
            [Literal('a'), OpenBrace, Literal('2'), Comma, CloseBrace]
        );
    }

    #[test]
    fn groups() {
        assert_eq!(
            kinds("(abc)"),
            [OpenParen, Literal('a'), Literal('b'), Literal('c'), CloseParen,]
        );
        assert_eq!(kinds("()"), [OpenParen, CloseParen]);
    }

    #[test]
    fn char_classes() {
        assert_eq!(
            kinds("[0-9]"),
            [
                CharClassStart('\0'),
                Literal('0'),
                Hyphen,
                Literal('9'),
                CloseBracket,
            ]
        );
        assert_eq!(
            kinds("[^0-9]"),
            [
                CharClassStart('^'),
                Literal('0'),
                Hyphen,
                Literal('9'),
                CloseBracket,
            ]
        );
        // `[^]` is a negated introducer followed by `]`, not a literal `^`.
        assert_eq!(kinds("[^]"), [CharClassStart('^'), CloseBracket]);
    }

    #[test]
    fn escape_sequences() {
        assert_eq!(kinds(r"\d"), [Escape('d')]);
        assert_eq!(kinds(r"\w"), [Escape('w')]);
        assert_eq!(kinds(r"\s"), [Escape('s')]);
        assert_eq!(kinds(r"\n"), [Escape('n')]);
        assert_eq!(kinds(r"\r"), [Escape('r')]);
        assert_eq!(kinds(r"\t"), [Escape('t')]);
        assert_eq!(kinds(r"\\"), [Escape('\\')]);
    }

    #[test]
    fn escaped_metacharacters() {
        let escaped = r"\.\*\+\?\|\(\)\[\]\{\}\\\-\^\$\.";
        let kinds = kinds(escaped);
        // Every escaped metachar is a single Escape token.
        assert!(
            kinds
                .iter()
                .all(|k| matches!(k, TokenKind::Escape(_)))
        );
        assert_eq!(kinds.len(), 16);
    }

    #[test]
    fn unicode_property_escapes() {
        let token = tokenize(r"\p{L}").next().unwrap();
        assert_eq!(token.kind, TokenKind::Escape('p'));
        assert_eq!(token.len, 5);

        let token = tokenize(r"\P{S}").next().unwrap();
        assert_eq!(token.kind, TokenKind::Escape('P'));
        assert_eq!(token.len, 5);

        let token = tokenize(r"\p{Nd}").next().unwrap();
        assert_eq!(token.kind, TokenKind::Escape('p'));
        assert_eq!(token.len, 6);
    }

    #[test]
    fn unclosed_delimiters_recover() {
        // Unclosed character class: no CloseBracket, iterator ends at Eof.
        let tokens: Vec<_> = tokenize("[abc").collect();
        assert_eq!(
            tokens.iter().map(|t| t.kind).collect::<Vec<_>>(),
            [CharClassStart('\0'), Literal('a'), Literal('b'), Literal('c')]
        );
        assert_eq!(
            tokens
                .iter()
                .map(|t| t.len as usize)
                .sum::<usize>(),
            4
        );

        // Unclosed group.
        let tokens: Vec<_> = tokenize("(abc").collect();
        assert_eq!(
            tokens.iter().map(|t| t.kind).collect::<Vec<_>>(),
            [OpenParen, Literal('a'), Literal('b'), Literal('c')]
        );
        assert_eq!(
            tokens
                .iter()
                .map(|t| t.len as usize)
                .sum::<usize>(),
            4
        );

        // Unclosed brace.
        let tokens: Vec<_> = tokenize("a{2").collect();
        assert_eq!(
            tokens.iter().map(|t| t.kind).collect::<Vec<_>>(),
            [Literal('a'), OpenBrace, Literal('2')]
        );
        assert_eq!(
            tokens
                .iter()
                .map(|t| t.len as usize)
                .sum::<usize>(),
            3
        );
    }

    #[test]
    fn whitespace_is_token() {
        assert_eq!(kinds(" "), [Whitespace]);
        assert_eq!(kinds("a b"), [Literal('a'), Whitespace, Literal('b')]);
    }

    #[test]
    fn predicates() {
        assert!(is_whitespace(' '));
        assert!(is_whitespace('\n'));
        assert!(!is_whitespace('a'));
    }
}
