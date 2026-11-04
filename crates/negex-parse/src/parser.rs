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

//! The parser: recursive descent over the I-Regexp ABNF grammar.
//!
//! Every loop either consumes a token or reports an error, which keeps
//! recovery live on malformed input.  A maximum nesting depth prevents
//! stack overflow on deeply nested patterns.

use crate::ast::*;
use crate::{ParseError, Span};
use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use codevar_logger::{log_debug, log_warn};
use codevar_negex_lex::{Token, TokenKind};

/// Maximum nesting depth for groups before the parser reports a limit
/// instead of overflowing the native stack.
const MAX_DEPTH: u32 = 64;

/// Extracts the category name from a `\p{...}` / `\P{...}` escape's source
/// text.
///
/// Returns an empty name when the escape is malformed (for example `\p` with
/// no braces, or an unclosed `\p{L`).  Encoding that as an empty name lets
/// code generation reject it as an unknown property instead of silently
/// treating it as a wildcard.
fn property_name(text: &str) -> String {
    let body = text
        .strip_prefix("\\p{")
        .or_else(|| text.strip_prefix("\\P{"))
        .and_then(|rest| rest.strip_suffix('}'));
    match body {
        Some(name) => String::from(name),
        None => String::new(),
    }
}

/// A cursor over tokens with lookahead.
pub(crate) struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    pattern: &'a str,
    depth: u32,
    errors: Vec<ParseError>,
}

impl<'a> Parser<'a> {
    /// Creates a parser over `tokens` of `pattern`.
    pub(crate) fn new(pattern: &'a str, tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            pos: 0,
            pattern,
            depth: 0,
            errors: Vec::new(),
        }
    }

    /// The source text of the current token.
    ///
    /// Tokens carry lengths, not text, so multi-character escapes such as
    /// `\p{L}` recover their property name by slicing the original pattern.
    #[inline]
    fn current_text(&self) -> &'a str {
        let offset = self.offset() as usize;
        let len = self.peek().len as usize;
        self.pattern
            .get(offset..offset.saturating_add(len))
            .unwrap_or("")
    }

    /// Byte offset of the current token.
    #[inline]
    fn offset(&self) -> u32 {
        self.tokens
            .get(..self.pos)
            .map(|slice| slice.iter().map(|t| t.len).sum())
            .unwrap_or(0)
    }

    /// Returns the current token without consuming it.
    #[inline]
    fn peek(&self) -> Token {
        self.tokens
            .get(self.pos)
            .copied()
            .unwrap_or(Token::new(TokenKind::Eof, 0))
    }

    /// Returns the current token's kind without consuming it.
    #[inline]
    fn peek_kind(&self) -> TokenKind {
        self.peek().kind
    }

    /// Consumes and returns the current token.
    #[inline]
    fn bump(&mut self) -> Token {
        let token = self.peek();
        if token.kind != TokenKind::Eof {
            self.pos += 1;
        }
        token
    }

    /// Consumes the current token if it matches `kind`, otherwise reports
    /// an error and returns `false`.
    fn expect(&mut self, kind: TokenKind) -> bool {
        let token = self.peek();
        if token.kind == kind {
            self.bump();
            true
        } else {
            let span = Span::new(self.offset(), token.len);
            self.errors.push(ParseError {
                span,
                message: format!("expected `{kind:?}`, found `{:?}`", token.kind),
            });
            log_debug!(
                "expected {:?} at offset {}, found {:?}",
                kind,
                self.offset(),
                token.kind
            );
            false
        }
    }

    /// Reports an error at the current position.
    fn error(&mut self, message: String) {
        let token = self.peek();
        let offset = self.offset();
        log_debug!("parse error at offset {offset}: {message}");
        self.errors.push(ParseError {
            span: Span::new(offset, token.len),
            message,
        });
    }

    /// Takes the accumulated errors, sorting them by source position.
    pub(crate) fn take_errors(&mut self) -> Vec<ParseError> {
        self.errors
            .sort_by_key(|error| (error.span.offset, error.span.len));
        core::mem::take(&mut self.errors)
    }

    /// Parses the top-level regex: `i-regexp = branch *( "|" branch )`.
    pub(crate) fn parse_regex(&mut self) -> Regex {
        let start = self.offset();
        let branch = self.parse_branch();
        let mut branches = vec![branch];
        while self.peek_kind() == TokenKind::Pipe {
            self.bump(); // consume `|`
            let branch = self.parse_branch();
            branches.push(branch);
        }
        let end = self.offset();
        let span = Span::new(start, end.saturating_sub(start));
        if branches.len() == 1 {
            Regex::new(branches.remove(0), span)
        } else {
            Regex::alternation(branches, span)
        }
    }

    /// Parses a branch: `branch = *piece`.
    fn parse_branch(&mut self) -> Branch {
        let start = self.offset();
        let mut pieces = Vec::new();
        while !matches!(
            self.peek_kind(),
            TokenKind::Pipe | TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::Eof
        ) {
            pieces.push(self.parse_piece());
        }
        let end = self.offset();
        let span = Span::new(start, end.saturating_sub(start));
        Branch { pieces, span }
    }

    /// Parses a piece: `piece = atom [ quantifier ]`.
    fn parse_piece(&mut self) -> Piece {
        let start = self.offset();
        let atom = self.parse_atom();
        let quantifier = if matches!(
            self.peek_kind(),
            TokenKind::Star | TokenKind::Plus | TokenKind::Question | TokenKind::OpenBrace
        ) {
            Some(self.parse_quantifier())
        } else {
            None
        };
        let end = self.offset();
        let span = Span::new(start, end.saturating_sub(start));
        Piece {
            atom,
            quantifier,
            span,
        }
    }

    /// Parses a quantifier: `* / + / ? / range-quantifier`.
    fn parse_quantifier(&mut self) -> Quantifier {
        let kind = self.peek_kind();
        match kind {
            TokenKind::Star => {
                self.bump();
                Quantifier::Star
            }
            TokenKind::Plus => {
                self.bump();
                Quantifier::Plus
            }
            TokenKind::Question => {
                self.bump();
                Quantifier::Question
            }
            TokenKind::OpenBrace => self.parse_range_quantifier(),
            _ => {
                let message = String::from("expected quantifier");
                self.error(message);
                Quantifier::Question
            }
        }
    }

    /// Parses a range quantifier: `{" { QuantExact [ "," [ QuantExact ] ] "}"}`.
    fn parse_range_quantifier(&mut self) -> Quantifier {
        let start = self.offset();
        self.expect(TokenKind::OpenBrace);
        let min = self.parse_quant_exact();
        let max = if self.peek_kind() == TokenKind::Comma {
            self.bump(); // consume `,`
            if self.peek_kind() != TokenKind::CloseBrace {
                Some(self.parse_quant_exact())
            } else {
                None // `{min,}` means min to unbounded
            }
        } else {
            None // `{min}` means exactly min
        };
        self.expect(TokenKind::CloseBrace);
        let end = self.offset();
        let _span = Span::new(start, end.saturating_sub(start));
        if min > u32::MAX / 2 {
            self.errors.push(ParseError {
                span: Span::new(start, end.saturating_sub(start)),
                message: String::from("range quantifier bound too large"),
            });
            log_warn!("range quantifier bound too large: {min}");
        }
        Quantifier::Range { min, max }
    }

    /// Parses a QuantExact: `1*%x30-39`.
    fn parse_quant_exact(&mut self) -> u32 {
        let _start = self.offset();
        let mut value: u32 = 0;
        let mut digits = 0;
        while matches!(self.peek_kind(), TokenKind::Literal(c) if c.is_ascii_digit()) {
            let token = self.bump();
            if let TokenKind::Literal(c) = token.kind {
                value = value
                    .saturating_mul(10)
                    .saturating_add(c.to_digit(10).unwrap_or(0));
                digits += 1;
            }
        }
        if digits == 0 {
            let token = self.peek();
            self.errors.push(ParseError {
                span: Span::new(self.offset(), token.len),
                message: String::from("expected digit in range quantifier"),
            });
        }
        value
    }

    /// Parses an atom: `NormalChar / charClass / ( "(" i-regexp ")" )`.
    fn parse_atom(&mut self) -> Atom {
        let start = self.offset();
        let kind = self.peek_kind();
        let mut atom = match kind {
            TokenKind::Literal(c) => {
                self.bump();
                Atom::Literal(c)
            }
            TokenKind::Dot => {
                self.bump();
                Atom::Any
            }
            TokenKind::OpenParen => {
                self.bump(); // consume `(`
                self.depth += 1;
                if self.depth > MAX_DEPTH {
                    self.error(format!("nesting depth exceeds {MAX_DEPTH}"));
                    self.depth -= 1;
                    // Skip to matching `)`
                    let mut nest = 1u32;
                    while nest > 0 {
                        match self.bump().kind {
                            TokenKind::OpenParen => nest += 1,
                            TokenKind::CloseParen => nest -= 1,
                            TokenKind::Eof => break,
                            _ => {}
                        }
                    }
                    Atom::Error {
                        span: Span::new(start, self.offset().saturating_sub(start)),
                    }
                } else {
                    let inner = self.parse_regex();
                    self.depth -= 1;
                    self.expect(TokenKind::CloseParen);
                    Atom::Group(Box::new(inner))
                }
            }
            TokenKind::CharClassStart(negated) => self.parse_char_class(negated != '\0'),
            TokenKind::Escape(c) => {
                let text = self.current_text();
                self.bump();
                self.parse_escape(c, text)
            }
            TokenKind::Caret => {
                // `^` is a normal character in I-Regexp (RFC 9485 has no
                // anchors): it matches itself.
                self.bump();
                Atom::Literal('^')
            }
            TokenKind::Dollar => {
                // `$` is a normal character in I-Regexp.
                self.bump();
                Atom::Literal('$')
            }
            _ => {
                let token = self.peek();
                self.errors.push(ParseError {
                    span: Span::new(self.offset(), token.len),
                    message: format!("unexpected token `{:?}` in atom", token.kind),
                });
                self.bump();
                Atom::Error {
                    span: Span::new(start, self.offset().saturating_sub(start)),
                }
            }
        };
        let end = self.offset();
        if let Atom::Error { ref mut span } = atom {
            *span = Span::new(start, end.saturating_sub(start));
        }
        atom
    }

    /// Parses an escape sequence after the backslash.
    ///
    /// `text` is the whole escape's source text, needed for the property
    /// escapes whose category name is not part of the token kind.
    fn parse_escape(&mut self, c: char, text: &str) -> Atom {
        match c {
            'n' => Atom::Literal('\n'),
            'r' => Atom::Literal('\r'),
            't' => Atom::Literal('\t'),
            '\\' => Atom::Literal('\\'),
            '|' => Atom::Literal('|'),
            '*' => Atom::Literal('*'),
            '+' => Atom::Literal('+'),
            '?' => Atom::Literal('?'),
            '.' => Atom::Literal('.'),
            '(' => Atom::Literal('('),
            ')' => Atom::Literal(')'),
            '[' => Atom::Literal('['),
            ']' => Atom::Literal(']'),
            '{' => Atom::Literal('{'),
            '}' => Atom::Literal('}'),
            '-' => Atom::Literal('-'),
            '^' => Atom::Literal('^'),
            '$' => Atom::Literal('$'),
            'd' => {
                // \d: digit class [0-9]
                Atom::Class(Class::Positive {
                    items: vec![ClassItem::Range { start: '0', end: '9' }],
                    negated: false,
                })
            }
            's' => {
                // \s: whitespace
                Atom::Class(Class::Positive {
                    items: vec![
                        ClassItem::Char(' '),
                        ClassItem::Char('\t'),
                        ClassItem::Char('\n'),
                        ClassItem::Char('\r'),
                    ],
                    negated: false,
                })
            }
            'w' => {
                // \w: word character [A-Za-z0-9_]
                Atom::Class(Class::Positive {
                    items: vec![
                        ClassItem::Range { start: 'A', end: 'Z' },
                        ClassItem::Range { start: 'a', end: 'z' },
                        ClassItem::Range { start: '0', end: '9' },
                        ClassItem::Char('_'),
                    ],
                    negated: false,
                })
            }
            'p' => {
                // \p{IsCategory}: Unicode property (positive)
                Atom::Class(Class::Property {
                    category: property_name(text),
                    negated: false,
                })
            }
            'P' => {
                // \P{IsCategory}: Unicode property (negated)
                Atom::Class(Class::Property {
                    category: property_name(text),
                    negated: true,
                })
            }
            c => Atom::Literal(c),
        }
    }

    /// Parses a character class: `charClassExpr`.
    ///
    /// `negated` comes from the `CharClassStart` token (`[` vs `[^`), which
    /// already consumed the optional `^`.
    fn parse_char_class(&mut self, negated: bool) -> Atom {
        let start = self.offset();
        self.bump(); // consume the CharClassStart (`[` or `[^`) token
        let mut items = Vec::new();
        while self.peek_kind() != TokenKind::CloseBracket && self.peek_kind() != TokenKind::Eof {
            let kind = self.peek_kind();
            match kind {
                TokenKind::Hyphen => {
                    // A leading/trailing `-` is a literal hyphen.
                    self.bump();
                    items.push(ClassItem::Char('-'));
                }
                TokenKind::Escape(c) => {
                    let text = self.current_text();
                    self.bump();
                    match c {
                        'p' => items.push(ClassItem::Property {
                            category: property_name(text),
                            negated: false,
                        }),
                        'P' => items.push(ClassItem::Property {
                            category: property_name(text),
                            negated: true,
                        }),
                        c => items.push(ClassItem::Char(c)),
                    }
                }
                TokenKind::Literal(c) => {
                    self.bump();
                    // Check for a range: `a-z`
                    if self.peek_kind() == TokenKind::Hyphen {
                        // Look ahead: if after hyphen is a literal or escape, it's a range
                        let hyphen_pos = self.pos;
                        self.bump(); // consume `-`
                        match self.peek_kind() {
                            TokenKind::Literal(end) | TokenKind::Escape(end) => {
                                self.bump();
                                let end_char = match end {
                                    'n' => '\n',
                                    'r' => '\r',
                                    't' => '\t',
                                    ch => ch,
                                };
                                items.push(ClassItem::Range {
                                    start: c,
                                    end: end_char,
                                });
                            }
                            _ => {
                                // Not a range, put the hyphen back as literal
                                self.pos = hyphen_pos;
                                items.push(ClassItem::Char(c));
                                items.push(ClassItem::Char('-'));
                            }
                        }
                    } else {
                        items.push(ClassItem::Char(c));
                    }
                }
                _ => {
                    let token = self.peek();
                    self.errors.push(ParseError {
                        span: Span::new(self.offset(), token.len),
                        message: format!("unexpected token `{:?}` in character class", token.kind),
                    });
                    self.bump();
                }
            }
        }
        self.expect(TokenKind::CloseBracket);
        if items.is_empty() {
            let span = Span::new(start, self.offset().saturating_sub(start));
            self.errors.push(ParseError {
                span,
                message: String::from("character class must not be empty"),
            });
            log_warn!("empty character class at offset {}", start);
        }
        Atom::Class(Class::Positive { items, negated })
    }
}
