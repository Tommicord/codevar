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

//! A single token produced by the Negex regex lexer.
//!
//! Tokens carry a [`TokenKind`] and a byte length, mirroring the design of
//! `codevar-ocl-lex` and `rustc_lexer`.

/// A token pairing a classification with its byte length in the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    /// The token's classification.
    pub kind: TokenKind,
    /// Length of the token in bytes.
    pub len: u32,
}

/// The classification of a regex lexeme.
///
/// This is a lossless classification: every byte of the input is covered by
/// exactly one token, and unrecognized or malformed constructs are reported
/// as flags on tokens rather than as diagnostics, mirroring the
/// `rustc_lexer` model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenKind {
    /// A literal character that matches itself (`a`, `z`, `0`, …).
    Literal(char),
    /// An alternation operator: `|`.
    Pipe,
    /// A Kleene star: `*`.
    Star,
    /// One-or-more: `+`.
    Plus,
    /// Zero-or-one: `?`.
    Question,
    /// An opening group: `(`.
    OpenParen,
    /// A closing group: `)`.
    CloseParen,
    /// An opening character class: `[`.
    OpenBracket,
    /// A closing character class: `]`.
    CloseBracket,
    /// An opening brace for a range quantifier: `{`.
    OpenBrace,
    /// A closing brace for a range quantifier: `}`.
    CloseBrace,
    /// A range quantifier separator: `,`.
    Comma,
    /// Any-character: `.`.
    Dot,
    /// A backslash escape, carrying the escaped character.
    ///
    /// This covers simple escapes (`\n`, `\.`, `\\` …), multi-character
    /// escapes (`\d`, `\w`, `\s`), and Unicode property escapes
    /// (`\p{L}`, `\P{S}` …). The carried `char` is the first character
    /// after the backslash.
    Escape(char),
    /// Start-of-input anchor: `^`.
    Caret,
    /// End-of-input anchor: `$`.
    Dollar,
    /// A hyphen inside a character class: `-`.
    Hyphen,
    /// End of input.
    #[default]
    Eof,
    /// Whitespace outside a character class.
    Whitespace,
    /// An unrecognized character.
    Unknown,
    /// The beginning of a character class: `[` or `[^`. The carried `char`
    /// is `^` when the class is negated, otherwise `NUL`. Covers the `[`
    /// and optional `^`.
    CharClassStart(char),
}

impl Token {
    /// Creates a token of the given kind and length.
    #[must_use]
    pub const fn new(kind: TokenKind, len: u32) -> Self {
        Self { kind, len }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::default::Default;

    #[test]
    fn token_new_sets_fields() {
        let token = Token::new(TokenKind::Literal('x'), 1);
        assert_eq!(token.kind, TokenKind::Literal('x'));
        assert_eq!(token.len, 1);
    }

    #[test]
    fn default_kind_is_eof() {
        assert_eq!(TokenKind::default(), TokenKind::Eof);
    }
}
