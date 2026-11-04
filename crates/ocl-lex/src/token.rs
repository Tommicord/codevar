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

//! Token types for the Codevar OpenCL lexer.
//!
//! Mirrors the design of [`rustc_lexer`]'s token model: a token is a compact
//! `{ kind, len }` pair covering a consecutive byte range of the source.
//! Lexical problems are encoded as flags on the token variants (for example
//! [`LiteralKind::Str::terminated`]) instead of diagnostics, keeping this
//! layer pure and `no_std`.
//!
//! [`rustc_lexer`]: https://github.com/rust-lang/rust/blob/master/compiler/rustc_lexer/src/lib.rs

/// A single lexed token: a kind tag plus the byte length of the source text
/// it covers.
///
/// Tokens carry no absolute position; the consumer accumulates lengths in
/// source order to derive byte offsets (the `rustc_lexer` convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    /// What kind of lexeme was matched.
    pub kind: TokenKind,
    /// Length of the token in bytes (always ≥ 1 except for [`TokenKind::Eof`]).
    pub len: u32,
}

impl Token {
    pub(crate) const fn new(kind: TokenKind, len: u32) -> Self {
        Self { kind, len }
    }
}

/// Kind of a lexed token.
///
/// Trivia (whitespace and comments) are first-class tokens so that a higher
/// layer can attach them to a lossless syntax tree. Multi-character operators
/// (`->`, `::`, `<<`, …) are *not* produced here; the parse layer glues
/// adjacent single-character tokens, exactly like `rustc`'s token-tree stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// A line comment, e.g. `// comment` or `/// doc comment`.
    LineComment {
        /// Doc-comment style when this is `///` or `//!`.
        doc_style: Option<DocStyle>,
    },
    /// A block comment, e.g. `/* comment */`.
    ///
    /// Block comments nest, so `/* /* */` is *not* terminated.
    BlockComment {
        /// Doc-comment style when this is `/** … */` or `/*! … */`.
        doc_style: Option<DocStyle>,
        /// Whether a matching `*/` was found before end of input.
        terminated: bool,
    },
    /// A run of whitespace characters.
    Whitespace,
    /// An identifier, e.g. `kernel` or `get_global_id`.
    Ident,
    /// A raw identifier, e.g. `r#type`.
    RawIdent,
    /// A lifetime, e.g. `'a` or `'static`.
    ///
    /// Codevar's OpenCL dialect has Rust-like ownership semantics, so
    /// lifetimes are a core token; `'0`-style lifetimes set
    /// `starts_with_number` for better diagnostics.
    Lifetime {
        /// Whether the lifetime starts with a digit (always an error).
        starts_with_number: bool,
    },
    /// An unknown literal prefix such as `foo"bar"` or `foo#`.
    ///
    /// Only the prefix (`foo`) is part of this token; the separator that
    /// follows is lexed as its own token. Kept for error recovery.
    UnknownPrefix,
    /// A numeric, character, string, or raw-string literal.
    ///
    /// The optional suffix (`u8`, `f`, `ul`, …) is included in the token
    /// text but delimited by `suffix_start` (a byte offset relative to the
    /// token start; equal to the token length when there is no suffix).
    Literal {
        /// Discriminant of the literal shape.
        kind: LiteralKind,
        /// Byte offset within the token where a literal suffix begins.
        suffix_start: u32,
    },

    /// `;`
    Semi,
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `(`
    OpenParen,
    /// `)`
    CloseParen,
    /// `{`
    OpenBrace,
    /// `}`
    CloseBrace,
    /// `[`
    OpenBracket,
    /// `]`
    CloseBracket,
    /// `@`
    At,
    /// `#`
    Pound,
    /// `~`
    Tilde,
    /// `?`
    Question,
    /// `:`
    Colon,
    /// `$`
    Dollar,
    /// `=`
    Eq,
    /// `!`
    Bang,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `-`
    Minus,
    /// `&`
    And,
    /// `|`
    Or,
    /// `+`
    Plus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `^`
    Caret,
    /// `%`
    Percent,

    /// A character the lexer does not recognize (e.g. `№` or an emoji).
    ///
    /// Produced for error recovery; the parser reports it.
    Unknown,
    /// End of input. Never yielded by [`tokenize`](crate::tokenize); returned
    /// only by [`Cursor::advance_token`](crate::Cursor::advance_token).
    Eof,
}

/// Doc-comment flavor, mirroring Rust's `///` (outer) and `//!` (inner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocStyle {
    /// Outer doc comment (`///` / `/** */`), documenting the next item.
    Outer,
    /// Inner doc comment (`//!` / `/*! */`), documenting the enclosing item.
    Inner,
}

/// Shape of a [`TokenKind::Literal`] token.
///
/// The literal suffix is *not* part of this decision: `1f32` and `1u8` both
/// classify as [`LiteralKind::Int`]; the parser inspects `suffix_start` to
/// interpret the suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LiteralKind {
    /// Integer literal: `123`, `0xFF`, `0b101`, `0o17`, `2ul`.
    Int {
        /// Radix implied by the literal's prefix.
        base: Base,
        /// Digits are missing after the prefix (e.g. `0x`, `0b`).
        empty_int: bool,
    },
    /// Floating-point literal: `1.0`, `1e9`, `1.0f`, `0x1.8p3`.
    Float {
        /// Radix implied by the literal's prefix (hex floats use `Base::Hexadecimal`).
        base: Base,
        /// The exponent marker is missing/empty: `1e`, or a hex float without
        /// a required `p` exponent such as `0x1.8`.
        empty_exponent: bool,
    },
    /// Character literal: `'a'`, `'\n'`, `'''`.
    Char {
        /// Whether a closing `'` was found before end of input or newline.
        terminated: bool,
    },
    /// String literal: `"abc"`, `"abc`.
    Str {
        /// Whether a closing `"` was found before end of input or newline.
        terminated: bool,
    },
    /// Raw string literal: `r"abc"`, `r#"ab"c"#`.
    ///
    /// `None` means the literal is invalid (unterminated, too many `#`s, or
    /// a bad character between `r` and `"`); use
    /// [`validate_raw_str`](crate::validate_raw_str) for the precise error.
    RawStr {
        /// Number of `#` delimiters, or `None` when invalid.
        n_hashes: Option<u8>,
    },
}

/// Radix of a numeric literal, implied by its prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Base {
    /// Prefix `0b` (e.g. `0b1010`).
    Binary = 2,
    /// Prefix `0o`, or a leading `0` followed by octal digits (C-style,
    /// e.g. `0755`) — OpenCL C follows C integer-constant rules.
    Octal = 8,
    /// No prefix (e.g. `42`).
    Decimal = 10,
    /// Prefix `0x` (e.g. `0xFF`).
    Hexadecimal = 16,
}

/// Why a raw string literal is invalid, reported by
/// [`validate_raw_str`](crate::validate_raw_str).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawStrError {
    /// A non-`#` character appears between `r`/`br` and the opening quote,
    /// e.g. `r~"abc"~`.
    InvalidStarter {
        /// The offending character.
        bad_char: char,
    },
    /// The string was never terminated by `"` followed by `expected` `#`s.
    NoTerminator {
        /// Number of `#`s that would be required to terminate the string.
        expected: u32,
        /// Largest number of `#`s actually found after a candidate `"`.
        found: u32,
        /// Byte offset (within the validated string, including the `r`
        /// prefix) of the best termination candidate found, if any.
        possible_terminator_offset: Option<u32>,
    },
    /// More than 255 `#` delimiters were used.
    TooManyDelimiters {
        /// Number of `#`s found.
        found: u32,
    },
}

impl core::fmt::Display for RawStrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidStarter { bad_char } => {
                write!(
                    f,
                    "invalid character `{bad_char}` between `r` and `\"` in raw string"
                )
            }
            Self::NoTerminator {
                expected,
                found,
                possible_terminator_offset,
            } => {
                write!(f, "missing terminating `\"` followed by {expected} `#`")?;
                if *found > 0 {
                    write!(f, " (found at most {found}")?;
                    match possible_terminator_offset {
                        Some(offset) => write!(f, ", candidate at offset {offset})")?,
                        None => write!(f, ")")?,
                    }
                }
                Ok(())
            }
            Self::TooManyDelimiters { found } => {
                write!(f, "too many `#` delimiters in raw string: {found} > 255")
            }
        }
    }
}

impl core::error::Error for RawStrError {}
