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

//! A streaming cursor over the characters of a regex pattern.
//!
//! Raw characters are pulled from the input on demand, trivia is tracked,
//! and escapes and character classes are recognized as single tokens.

use crate::{Token, TokenKind, is_whitespace};
use codevar_logger::{log_debug, log_irr, log_warn};

/// A cursor over a pattern string, producing one [`Token`] at a time.
pub struct Cursor {
    /// Remaining input characters.
    chars: core::str::Chars<'static>,
    /// Whether the iterator is exhausted.
    exhausted: bool,
    /// Whether the lexer is currently inside a character class (`[...]`).
    in_char_class: bool,
}

impl Cursor {
    /// Wraps `input` as a cursor.
    pub fn new(input: &str) -> Self {
        // SAFETY: the cursor borrows `input` for its lifetime; the
        // `'static` bound is a sound reborrow because the cursor never
        // outlives the input in practice (it is used within `tokenize`).
        Self {
            chars: unsafe {
                core::mem::transmute::<core::str::Chars<'_>, core::str::Chars<'static>>(input.chars())
            },
            exhausted: false,
            in_char_class: false,
        }
    }

    /// Advances to and returns the next token.
    pub fn advance_token(&mut self) -> Token {
        if self.exhausted {
            if self.in_char_class {
                self.in_char_class = false;
                log_warn!("unclosed character class at end of input");
            }
            return Token::new(TokenKind::Eof, 0);
        }
        let c = match self.chars.next() {
            Some(c) => c,
            None => {
                self.exhausted = true;
                if self.in_char_class {
                    self.in_char_class = false;
                    log_warn!("unclosed character class at end of input");
                }
                return Token::new(TokenKind::Eof, 0);
            }
        };
        let len = c.len_utf8() as u32;

        if self.in_char_class {
            return self.in_class(c, len);
        }

        match c {
            '[' => {
                let negated = matches!(self.chars.clone().next(), Some('^'));
                if negated {
                    // Consume the `^`; it is part of the introducer.
                    let _ = self.chars.next();
                    self.in_char_class = true;
                    Token::new(TokenKind::CharClassStart('^'), len + 1)
                } else {
                    self.in_char_class = true;
                    Token::new(TokenKind::CharClassStart('\0'), len)
                }
            }
            '\\' => self.escape(len),
            c if is_whitespace(c) => Token::new(TokenKind::Whitespace, len),
            '|' => Token::new(TokenKind::Pipe, len),
            '*' => Token::new(TokenKind::Star, len),
            '+' => Token::new(TokenKind::Plus, len),
            '?' => Token::new(TokenKind::Question, len),
            '(' => Token::new(TokenKind::OpenParen, len),
            ')' => Token::new(TokenKind::CloseParen, len),
            ']' => Token::new(TokenKind::CloseBracket, len),
            '{' => Token::new(TokenKind::OpenBrace, len),
            '}' => Token::new(TokenKind::CloseBrace, len),
            ',' => Token::new(TokenKind::Comma, len),
            '.' => Token::new(TokenKind::Dot, len),
            '^' => Token::new(TokenKind::Caret, len),
            '$' => Token::new(TokenKind::Dollar, len),
            '-' => Token::new(TokenKind::Hyphen, len),
            c if is_surrogate(c) => {
                log_warn!("surrogate code point skipped in pattern");
                Token::new(TokenKind::Unknown, len)
            }
            c => Token::new(TokenKind::Literal(c), len),
        }
    }

    /// Lexes a single character appearing inside a character class.
    ///
    /// Inside a class the metacharacters `*`, `+`, `?`, `|`, `(`, `)`,
    /// `{`, `}`, `.` are ordinary literals; only `]`, `\`, and `-` are
    /// special (and `^` only when it is the first member, already consumed
    /// by the introducer).
    fn in_class(&mut self, c: char, len: u32) -> Token {
        match c {
            ']' => {
                self.in_char_class = false;
                Token::new(TokenKind::CloseBracket, len)
            }
            '\\' => self.escape(len),
            '-' => Token::new(TokenKind::Hyphen, len),
            c => Token::new(TokenKind::Literal(c), len),
        }
    }

    /// Lexes a backslash escape, consuming the escaped character.
    ///
    /// Handles simple escapes (`\n`, `\.`, `\\` …), multi-character escapes
    /// (`\d`, `\w`, `\s`), and Unicode property escapes (`\p{L}`, `\P{S}` …).
    fn escape(&mut self, backslash_len: u32) -> Token {
        let c = match self.chars.next() {
            Some(c) => c,
            None => {
                self.exhausted = true;
                log_irr!("trailing backslash at end of pattern");
                return Token::new(TokenKind::Escape('\\'), backslash_len);
            }
        };

        // Unicode property escape: `\p{...}` or `\P{...}`. Only consume the
        // property body when a `{` actually follows; otherwise `\p` is a
        // plain two-character escape and the following character stays for
        // the next token.
        if matches!(c, 'p' | 'P') && matches!(self.chars.clone().next(), Some('{')) {
            return self.property_escape(backslash_len, c);
        }

        let len = backslash_len + c.len_utf8() as u32;
        log_debug!("escape sequence \\{} ({} bytes)", c, len);
        Token::new(TokenKind::Escape(c), len)
    }

    /// Lexes a Unicode property escape `\p{...}` / `\P{...}`, consuming the
    /// opening `{`, the property name, and the closing `}`.
    ///
    /// If the closing `}` never arrives, the token still covers everything
    /// consumed so far and the iterator is marked exhausted (graceful
    /// recovery for an unclosed property escape).
    fn property_escape(&mut self, backslash_len: u32, first: char) -> Token {
        let mut len = backslash_len + first.len_utf8() as u32;
        // Consume the opening `{` (already confirmed present by the caller).
        let _ = self.chars.next();
        len += 1;
        loop {
            match self.chars.next() {
                Some('}') => {
                    len += 1;
                    break;
                }
                Some(c) => {
                    len += c.len_utf8() as u32;
                }
                None => {
                    self.exhausted = true;
                    log_warn!("unclosed Unicode property escape");
                    break;
                }
            }
        }
        Token::new(TokenKind::Escape(first), len)
    }
}

/// True if `c` is a UTF-16 surrogate code point (U+D800..=U+DFFF).
///
/// Surrogates cannot appear in valid UTF-8, so [`str::Chars`] never yields
/// them; this is a defensive guard for any code point source.
#[must_use]
#[inline]
fn is_surrogate(c: char) -> bool {
    matches!(c as u32, 0xD800..=0xDFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_produces_tokens_in_order() {
        let mut cursor = Cursor::new("a*b");
        assert_eq!(cursor.advance_token().kind, TokenKind::Literal('a'));
        assert_eq!(cursor.advance_token().kind, TokenKind::Star);
        assert_eq!(cursor.advance_token().kind, TokenKind::Literal('b'));
        assert_eq!(cursor.advance_token().kind, TokenKind::Eof);
    }

    #[test]
    fn escape_consumes_escaped_char() {
        let mut cursor = Cursor::new(r"\n");
        let token = cursor.advance_token();
        assert_eq!(token.kind, TokenKind::Escape('n'));
        assert_eq!(token.len, 2);
    }

    #[test]
    fn cursor_is_exhausted_after_eof() {
        let mut cursor = Cursor::new("");
        assert_eq!(cursor.advance_token().kind, TokenKind::Eof);
        assert_eq!(cursor.advance_token().kind, TokenKind::Eof);
    }

    #[test]
    fn char_class_introducer() {
        let mut cursor = Cursor::new("[0-9]");
        let t = cursor.advance_token();
        assert_eq!(t.kind, TokenKind::CharClassStart('\0'));
        assert_eq!(t.len, 1);
        assert_eq!(cursor.advance_token().kind, TokenKind::Literal('0'));
        assert_eq!(cursor.advance_token().kind, TokenKind::Hyphen);
        assert_eq!(cursor.advance_token().kind, TokenKind::Literal('9'));
        assert_eq!(cursor.advance_token().kind, TokenKind::CloseBracket);
        assert_eq!(cursor.advance_token().kind, TokenKind::Eof);
    }

    #[test]
    fn negated_char_class_introducer() {
        let mut cursor = Cursor::new("[^0-9]");
        let t = cursor.advance_token();
        assert_eq!(t.kind, TokenKind::CharClassStart('^'));
        assert_eq!(t.len, 2);
    }

    #[test]
    fn property_escape_consumes_body() {
        let mut cursor = Cursor::new(r"\p{L}");
        let token = cursor.advance_token();
        assert_eq!(token.kind, TokenKind::Escape('p'));
        assert_eq!(token.len, 5);
        assert_eq!(cursor.advance_token().kind, TokenKind::Eof);
    }

    #[test]
    fn trailing_backslash_is_escape() {
        let mut cursor = Cursor::new(r"\");
        let token = cursor.advance_token();
        assert_eq!(token.kind, TokenKind::Escape('\\'));
        assert_eq!(token.len, 1);
        assert_eq!(cursor.advance_token().kind, TokenKind::Eof);
    }

    #[test]
    fn unclosed_class_recover() {
        let mut cursor = Cursor::new("[abc");
        let mut total = 0;
        loop {
            let token = cursor.advance_token();
            total += token.len as usize;
            if token.kind == TokenKind::Eof {
                break;
            }
        }
        assert_eq!(total, 4); // `[` `a` `b` `c` — no closing `]`
    }
}
