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

//! The peekable cursor that drives the lexer state machine.
//!
//! The cursor borrows the source string and consumes it character by
//! character. [`Cursor::advance_token`] assembles one [`Token`] per call,
//! reporting lexical problems as flags on the token instead of diagnostics
//! (the `rustc_lexer` model), so this layer stays pure and `no_std`.

use crate::LiteralKind::*;
use crate::TokenKind::*;
use crate::{
    Base, DocStyle, LiteralKind, RawStrError, Token, TokenKind, is_id_continue, is_id_start, is_whitespace,
};

/// Placeholder returned when the input is exhausted.
///
/// A real NUL byte in the source is indistinguishable from this value in
/// peeks; the callers that care check [`Cursor::is_eof`] as well.
const EOF_CHAR: char = '\0';

/// A cursor over the remaining source text.
///
/// Tokens carry no absolute position: consumers accumulate the `len` fields
/// of produced [`Token`]s to derive byte offsets. Position bookkeeping
/// ([`Cursor::pos_within_token`]) is relative to the start of the token
/// currently being assembled.
pub struct Cursor<'a> {
    /// Byte length of the not-yet-consumed suffix at the last reset point.
    len_remaining: usize,
    /// Iterator over the remaining characters. Slightly faster than a `&str`.
    chars: core::str::Chars<'a>,
    /// Last consumed symbol, tracked only in debug builds for the
    /// `debug_assert!`s guarding the lexing routines.
    #[cfg(debug_assertions)]
    prev: char,
}

impl<'a> Cursor<'a> {
    /// Creates a cursor positioned at the start of `input`.
    #[must_use]
    pub fn new(input: &'a str) -> Cursor<'a> {
        Cursor {
            len_remaining: input.len(),
            chars: input.chars(),
            #[cfg(debug_assertions)]
            prev: EOF_CHAR,
        }
    }

    /// Returns the not-yet-consumed suffix of the original input.
    #[must_use]
    pub fn as_str(&self) -> &'a str {
        self.chars.as_str()
    }

    /// Peeks the next symbol without consuming it, or [`EOF_CHAR`] at end of
    /// input.
    ///
    /// Getting [`EOF_CHAR`] does not always mean the end of the input; check
    /// [`Cursor::is_eof`] to be sure.
    #[must_use]
    pub fn first(&self) -> char {
        self.chars.clone().next().unwrap_or(EOF_CHAR)
    }

    /// Returns the last consumed symbol, or [`EOF_CHAR`] in release builds
    /// (debug builds only).
    pub(crate) fn prev(&self) -> char {
        #[cfg(debug_assertions)]
        {
            self.prev
        }

        #[cfg(not(debug_assertions))]
        {
            EOF_CHAR
        }
    }

    /// Peeks the second symbol without consuming it, or [`EOF_CHAR`].
    pub(crate) fn second(&self) -> char {
        let mut iter = self.chars.clone();
        iter.next();
        iter.next().unwrap_or(EOF_CHAR)
    }

    /// Checks whether there is nothing more to consume.
    pub(crate) fn is_eof(&self) -> bool {
        self.chars.as_str().is_empty()
    }

    /// Returns the bytes consumed since the last
    /// [`Cursor::reset_pos_within_token`].
    pub(crate) fn pos_within_token(&self) -> u32 {
        (self.len_remaining - self.chars.as_str().len()) as u32
    }

    /// Resets the consumed-bytes counter to zero.
    pub(crate) fn reset_pos_within_token(&mut self) {
        self.len_remaining = self.chars.as_str().len();
    }

    /// Moves to the next character, returning it, or `None` at end of input.
    pub(crate) fn bump(&mut self) -> Option<char> {
        let c = self.chars.next()?;

        #[cfg(debug_assertions)]
        {
            self.prev = c;
        }

        Some(c)
    }

    /// Consumes symbols while `predicate` holds, stopping at end of input.
    pub(crate) fn eat_while(&mut self, mut predicate: impl FnMut(char) -> bool) {
        while !self.is_eof() && predicate(self.first()) {
            self.bump();
        }
    }

    /// Positions the cursor at the next occurrence of `byte`, or at the end
    /// of the input if it never occurs. The byte itself is not consumed.
    pub(crate) fn eat_until(&mut self, byte: u8) {
        let rest = self.as_str();
        let index = rest
            .as_bytes()
            .iter()
            .position(|&candidate| candidate == byte);
        let remainder = index
            .and_then(|position| rest.get(position..))
            .unwrap_or("");
        self.chars = remainder.chars();
    }

    /// Parses the next token from the input.
    ///
    /// Returns [`TokenKind::Eof`] exactly once, when the input is exhausted;
    /// the token's `len` covers every byte consumed by this call, including
    /// leading trivia. Multi-character operators (`->`, `::`, `<<`, …) are
    /// intentionally left as adjacent single-character tokens for the parse
    /// layer to glue together.
    pub fn advance_token(&mut self) -> Token {
        let Some(first_char) = self.bump() else {
            return Token::new(TokenKind::Eof, 0);
        };

        let token_kind = match first_char {
            '/' => match self.first() {
                '/' => self.line_comment(),
                '*' => self.block_comment(),
                _ => Slash,
            },

            c if is_whitespace(c) => self.whitespace(),

            'r' => match (self.first(), self.second()) {
                ('#', c1) if is_id_start(c1) => {
                    self.bump();
                    self.eat_identifier();
                    RawIdent
                }
                ('#', _) | ('"', _) => {
                    let result = self.raw_double_quoted_string(1);
                    let suffix_start = self.pos_within_token();
                    if result.is_ok() {
                        self.eat_literal_suffix();
                    }
                    Literal {
                        kind: RawStr {
                            n_hashes: result.ok(),
                        },
                        suffix_start,
                    }
                }
                _ => self.ident_or_unknown_prefix(),
            },

            c if is_id_start(c) => self.ident_or_unknown_prefix(),

            c @ '0'..='9' => {
                let literal_kind = self.number(c);
                let suffix_start = self.pos_within_token();
                self.eat_literal_suffix();
                Literal {
                    kind: literal_kind,
                    suffix_start,
                }
            }

            ';' => Semi,
            ',' => Comma,
            '.' => Dot,
            '(' => OpenParen,
            ')' => CloseParen,
            '{' => OpenBrace,
            '}' => CloseBrace,
            '[' => OpenBracket,
            ']' => CloseBracket,
            '@' => At,
            '#' => Pound,
            '~' => Tilde,
            '?' => Question,
            ':' => Colon,
            '$' => Dollar,
            '=' => Eq,
            '!' => Bang,
            '<' => Lt,
            '>' => Gt,
            '-' => Minus,
            '&' => And,
            '|' => Or,
            '+' => Plus,
            '*' => Star,
            '^' => Caret,
            '%' => Percent,

            '\'' => self.lifetime_or_char(),

            '"' => {
                let terminated = self.double_quoted_string();
                let suffix_start = self.pos_within_token();
                if terminated {
                    self.eat_literal_suffix();
                }
                Literal {
                    kind: Str { terminated },
                    suffix_start,
                }
            }

            _ => Unknown,
        };

        let token = Token::new(token_kind, self.pos_within_token());
        self.reset_pos_within_token();
        token
    }

    /// Consumes a `//` line comment, up to but excluding the newline.
    ///
    /// `//!` marks an inner doc comment and `///` (not followed by a third
    /// slash) an outer one; `////` and longer runs are ordinary comments.
    fn line_comment(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '/' && self.first() == '/');
        self.bump();

        let doc_style = match self.first() {
            '!' => Some(DocStyle::Inner),
            '/' if self.second() != '/' => Some(DocStyle::Outer),
            _ => None,
        };

        self.eat_until(b'\n');
        LineComment { doc_style }
    }

    /// Consumes a `/* … */` block comment; block comments nest, so
    /// `/* /* */` is unterminated.
    ///
    /// `/*!` marks an inner doc comment and `/**` (with a non-`*`/`/` follow)
    /// an outer one.
    fn block_comment(&mut self) -> TokenKind {
        debug_assert!(self.prev() == '/' && self.first() == '*');
        self.bump();

        let doc_style = match self.first() {
            '!' => Some(DocStyle::Inner),
            '*' if !matches!(self.second(), '*' | '/') => Some(DocStyle::Outer),
            _ => None,
        };

        let mut depth = 1usize;
        while let Some(c) = self.bump() {
            match c {
                '/' if self.first() == '*' => {
                    self.bump();
                    depth += 1;
                }
                '*' if self.first() == '/' => {
                    self.bump();
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        break;
                    }
                }
                _ => (),
            }
        }

        BlockComment {
            doc_style,
            terminated: depth == 0,
        }
    }

    /// Consumes the run of whitespace starting at the previous character.
    fn whitespace(&mut self) -> TokenKind {
        debug_assert!(is_whitespace(self.prev()));
        self.eat_while(is_whitespace);
        Whitespace
    }

    /// Consumes an identifier, or the prefix of a malformed literal.
    ///
    /// An identifier directly followed by `#`, `"`, or `'` is the unknown
    /// prefix of a literal the dialect does not define (e.g. `foo"bar"` or
    /// `b"x"`); only the prefix becomes this token, for error recovery.
    fn ident_or_unknown_prefix(&mut self) -> TokenKind {
        debug_assert!(is_id_start(self.prev()));
        self.eat_while(is_id_continue);
        match self.first() {
            '#' | '"' | '\'' => UnknownPrefix,
            _ => Ident,
        }
    }

    /// Consumes a numeric literal; the first digit is already consumed.
    ///
    /// Handles `0b`/`0o`/`0x` prefixes, OpenCL C-style octal literals written
    /// with a leading `0` (`0755`), decimal and hexadecimal floats (C
    /// hexadecimal floats such as `0x1.8p3` require a `p` exponent), digit
    /// separators (`1_000`), and literal suffixes (`1u8`, `1.0f`).
    fn number(&mut self, first_digit: char) -> LiteralKind {
        debug_assert!(first_digit.is_ascii_digit());
        let mut base = Base::Decimal;
        if first_digit == '0' {
            match self.first() {
                'b' => {
                    base = Base::Binary;
                    self.bump();
                    if !self.eat_decimal_digits() {
                        return Int {
                            base,
                            empty_int: true,
                        };
                    }
                }
                'o' => {
                    base = Base::Octal;
                    self.bump();
                    if !self.eat_decimal_digits() {
                        return Int {
                            base,
                            empty_int: true,
                        };
                    }
                }
                'x' => {
                    base = Base::Hexadecimal;
                    self.bump();
                    if !self.eat_hexadecimal_digits() {
                        return Int {
                            base,
                            empty_int: true,
                        };
                    }
                }
                '0'..='9' => {
                    base = Base::Octal;
                    self.eat_decimal_digits();
                }
                '_' => {
                    self.eat_decimal_digits();
                }
                '.' | 'e' | 'E' => {}
                _ => {
                    return Int {
                        base,
                        empty_int: false,
                    };
                }
            }
        } else {
            self.eat_decimal_digits();
        }

        let is_hex = base == Base::Hexadecimal;
        match self.first() {
            '.' if self.second() != '.' && !is_id_start(self.second()) => {
                self.bump();
                let mut empty_exponent;
                if is_hex {
                    self.eat_hexadecimal_digits();
                    empty_exponent = true;
                    if matches!(self.first(), 'p' | 'P') {
                        self.bump();
                        empty_exponent = !self.eat_float_exponent();
                    }
                } else {
                    empty_exponent = false;
                    if self.first().is_ascii_digit() {
                        self.eat_decimal_digits();
                        if matches!(self.first(), 'e' | 'E') {
                            self.bump();
                            empty_exponent = !self.eat_float_exponent();
                        }
                    }
                }
                Float { base, empty_exponent }
            }
            'e' | 'E' => {
                self.bump();
                Float {
                    base,
                    empty_exponent: !self.eat_float_exponent(),
                }
            }
            'p' | 'P' if is_hex => {
                self.bump();
                Float {
                    base,
                    empty_exponent: !self.eat_float_exponent(),
                }
            }
            _ => Int {
                base,
                empty_int: false,
            },
        }
    }

    /// Consumes a lifetime (`'a`, `'static`) or a character literal (`'a'`).
    ///
    /// A lifetime start followed by a digit (`'0`) is recorded via
    /// [`LiteralKind`]-independent `starts_with_number` so the parser can
    /// report an invalid lifetime rather than an unterminated char literal.
    /// Contents followed by a closing quote (`'abc'`) lex as a multi-character
    /// character literal for the parser to reject.
    fn lifetime_or_char(&mut self) -> TokenKind {
        debug_assert_eq!(self.prev(), '\'');

        let can_be_a_lifetime = if self.second() == '\'' {
            false
        } else {
            is_id_start(self.first()) || self.first().is_ascii_digit()
        };

        if !can_be_a_lifetime {
            let terminated = self.single_quoted_string();
            let suffix_start = self.pos_within_token();
            if terminated {
                self.eat_literal_suffix();
            }
            return Literal {
                kind: Char { terminated },
                suffix_start,
            };
        }

        let starts_with_number = self.first().is_ascii_digit();
        self.bump();
        self.eat_while(is_id_continue);

        match self.first() {
            '\'' => {
                self.bump();
                Literal {
                    kind: Char { terminated: true },
                    suffix_start: self.pos_within_token(),
                }
            }
            _ => Lifetime { starts_with_number },
        }
    }

    /// Consumes the body of a character literal after the opening quote.
    ///
    /// Returns `true` when the closing quote was found. A newline that is not
    /// immediately followed by another quote ends the literal early
    /// (recovery), as does a backslash directly before a line break, so the
    /// rest of the file stays lexable.
    fn single_quoted_string(&mut self) -> bool {
        debug_assert_eq!(self.prev(), '\'');
        if self.second() == '\'' && self.first() != '\\' {
            self.bump();
            self.bump();
            return true;
        }

        loop {
            match self.first() {
                '\'' => {
                    self.bump();
                    return true;
                }
                '/' => break,
                '\n' if self.second() != '\'' => break,
                '\\' if matches!(self.second(), '\n' | '\r') => break,
                EOF_CHAR if self.is_eof() => break,
                '\\' => {
                    self.bump();
                    self.bump();
                }
                _ => {
                    self.bump();
                }
            }
        }
        false
    }

    /// Consumes the body of a double-quoted string after the opening quote.
    ///
    /// Returns `true` when the closing quote was found. An unescaped newline
    /// ends the literal early (`terminated: false`) — like C, a plain string
    /// may not span lines — while a backslash directly followed by a newline
    /// is a line continuation that keeps the literal open across lines.
    fn double_quoted_string(&mut self) -> bool {
        debug_assert_eq!(self.prev(), '"');
        loop {
            match self.first() {
                '"' => {
                    self.bump();
                    return true;
                }
                '\n' | '\r' => return false,
                '\\' => {
                    self.bump();
                    match self.first() {
                        '\n' => {
                            self.bump();
                        }
                        '\r' => {
                            self.bump();
                            if self.first() == '\n' {
                                self.bump();
                            }
                        }
                        _ if !self.is_eof() => {
                            self.bump();
                        }
                        _ => (),
                    }
                }
                EOF_CHAR if self.is_eof() => return false,
                _ => {
                    self.bump();
                }
            }
        }
    }

    /// Consumes a raw string literal, returning its `#` count or the reason
    /// it is malformed. `prefix_len` is the byte length of the prefix before
    /// the `#`s and the opening quote (always `1`: the `r`).
    pub(crate) fn raw_double_quoted_string(&mut self, prefix_len: u32) -> Result<u8, RawStrError> {
        let n_hashes = self.raw_string_unvalidated(prefix_len)?;
        match u8::try_from(n_hashes) {
            Ok(count) => Ok(count),
            Err(_) => Err(RawStrError::TooManyDelimiters { found: n_hashes }),
        }
    }

    /// The unvalidated core of [`Cursor::raw_double_quoted_string`].
    ///
    /// Tracks the candidate terminator with the most `#`s seen so far so
    /// [`RawStrError::NoTerminator`] can point at where the literal should
    /// have ended.
    fn raw_string_unvalidated(&mut self, prefix_len: u32) -> Result<u32, RawStrError> {
        debug_assert_eq!(self.prev(), 'r');
        let start_pos = self.pos_within_token();
        let mut possible_terminator_offset = None;
        let mut max_hashes = 0;

        let mut eaten = 0;
        while self.first() == '#' {
            eaten += 1;
            self.bump();
        }
        let n_start_hashes = eaten;
        match self.bump() {
            Some('"') => (),
            other => {
                return Err(RawStrError::InvalidStarter {
                    bad_char: other.unwrap_or(EOF_CHAR),
                });
            }
        }
        loop {
            self.eat_until(b'"');
            if self.is_eof() {
                return Err(RawStrError::NoTerminator {
                    expected: n_start_hashes,
                    found: max_hashes,
                    possible_terminator_offset,
                });
            }
            self.bump();
            let mut n_end_hashes = 0;
            while self.first() == '#' && n_end_hashes < n_start_hashes {
                n_end_hashes += 1;
                self.bump();
            }

            if n_end_hashes == n_start_hashes {
                return Ok(n_start_hashes);
            } else if n_end_hashes > max_hashes {
                possible_terminator_offset =
                    Some(self.pos_within_token() - start_pos - n_end_hashes + prefix_len);
                max_hashes = n_end_hashes;
            }
        }
    }

    /// Consumes decimal digits and underscores; `true` if any digit was met.
    fn eat_decimal_digits(&mut self) -> bool {
        let mut has_digits = false;
        while !self.is_eof() {
            match self.first() {
                '_' => {
                    self.bump();
                }
                '0'..='9' => {
                    has_digits = true;
                    self.bump();
                }
                _ => break,
            }
        }
        has_digits
    }

    /// Consumes hexadecimal digits and underscores; `true` if any digit was met.
    fn eat_hexadecimal_digits(&mut self) -> bool {
        let mut has_digits = false;
        while !self.is_eof() {
            match self.first() {
                '_' => {
                    self.bump();
                }
                '0'..='9' | 'a'..='f' | 'A'..='F' => {
                    has_digits = true;
                    self.bump();
                }
                _ => break,
            }
        }
        has_digits
    }

    /// Consumes an exponent (`e10`, `e-3`, `p+2`); `true` if any digit was met.
    fn eat_float_exponent(&mut self) -> bool {
        debug_assert!(matches!(self.prev(), 'e' | 'E' | 'p' | 'P'));
        if matches!(self.first(), '-' | '+') {
            self.bump();
        }
        self.eat_decimal_digits()
    }

    /// Consumes the suffix of a literal (e.g. the `u8` of `1u8`).
    fn eat_literal_suffix(&mut self) {
        self.eat_identifier();
    }

    /// Consumes an identifier, succeeding on `_` (not a valid identifier
    /// start, but a valid suffix).
    fn eat_identifier(&mut self) {
        if !is_id_start(self.first()) {
            return;
        }
        self.bump();
        self.eat_while(is_id_continue);
    }
}
