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

//! The parser: a Pratt expression parser over significant tokens, wrapped
//! by recursive descent for items, statements, and types.
//!
//! Multi-character operators are composed *on demand* from adjacent
//! single-character tokens (rustc's `Spacing::Joint` discipline), so
//! constructs such as nested generics `A<B<C>>` never see a glued `>>`.
//! Every loop either consumes a token or reports an error, which keeps
//! recovery live on malformed input.
//!
//! Tokens are pulled through a [`Cursor`], a small streaming lookahead
//! ring that filters trivia and accumulates spans from token lengths, so
//! neither [`parse_program`] nor [`ItemStream`](crate::ItemStream)
//! materializes the file's token stream.

use crate::ast::*;
use crate::stream::ItemOutcome;
use crate::{ParseError, Span};
use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use codevar_ocl_lex::{Token, TokenKind};

/// Maximum nesting depth for expressions, types, blocks, and `if` chains
/// before the parser reports a recursion limit instead of overflowing the
/// native stack.
///
/// A parenthesized expression costs about six parser frames per depth
/// unit, and debug builds make those frames several kilobytes each; the
/// limit keeps the worst case comfortably inside the 2 MiB stack that
/// spawned test and worker threads get on Linux.
const MAX_DEPTH: u32 = 64;

/// Binding power of the `as` cast operator.
const BP_CAST: u8 = 12;
/// Binding power of prefix operators (`-x`, `*x`, `&x`, `!x`).
const BP_UNARY: u8 = 15;
/// Binding power of the lowest-binding range operand position.
const BP_RANGE_OPERAND: u8 = 3;
/// Binding power of assignment (right-associative).
const BP_ASSIGN: u8 = 1;
/// Binding power of ranges.
const BP_RANGE: u8 = 2;

/// A non-trivia token paired with its source span.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SigToken {
    /// The lexed token classification.
    kind: TokenKind,
    /// Span of the token in the source.
    span: Span,
}

/// Whether the token is trivia (dropped by the parser, kept by the trees).
fn is_trivia(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Whitespace | TokenKind::LineComment { .. } | TokenKind::BlockComment { .. }
    )
}

/// Number of significant tokens the lookahead ring holds.
///
/// The parser peeks at most three tokens ahead (`kind_at(2)` and the
/// `at_joint3` chain), so four slots leave one spare.
const RING_CAPACITY: usize = 4;

/// A streaming cursor over the significant tokens of a source.
///
/// Raw [`Token`]s are pulled from the iterator on demand: trivia is
/// filtered as it arrives, spans are accumulated from token lengths
/// starting at byte offset 0, and at most [`RING_CAPACITY`] significant
/// tokens are buffered at once, so parsing never materializes the whole
/// file's token stream.
///
/// Invariant: after [`Cursor::new`] and after every [`Cursor::bump`],
/// the ring holds `RING_CAPACITY` tokens or the iterator is exhausted —
/// so an empty ring means end of input.
pub(crate) struct Cursor<I> {
    /// Raw token stream, trivia included.
    tokens: I,
    /// Lookahead ring; `ring[0]` is the next token to consume.
    ring: [Option<SigToken>; RING_CAPACITY],
    /// Number of valid entries in `ring`.
    len: usize,
    /// Byte offset of the next raw token (lengths accumulate from 0).
    offset: u32,
    /// Whether the iterator has returned `None`.
    exhausted: bool,
    /// Most recently consumed token, for previous-token queries.
    prev: Option<SigToken>,
    /// Total significant tokens consumed so far (the parse position).
    consumed: usize,
}

impl<I: Iterator<Item = Token>> Cursor<I> {
    /// Wraps `tokens`, buffering the first lookahead tokens.
    pub(crate) fn new(tokens: I) -> Self {
        let mut cursor = Self {
            tokens,
            ring: [None; RING_CAPACITY],
            len: 0,
            offset: 0,
            exhausted: false,
            prev: None,
            consumed: 0,
        };
        cursor.fill();
        cursor
    }

    /// Refills the ring until it is full or the input is exhausted.
    fn fill(&mut self) {
        while self.len < RING_CAPACITY && !self.exhausted {
            let Some(token) = self.tokens.next() else {
                self.exhausted = true;
                break;
            };
            let span = Span::new(self.offset, token.len);
            self.offset = self.offset.saturating_add(token.len);
            if !is_trivia(token.kind) {
                self.ring[self.len] = Some(SigToken {
                    kind: token.kind,
                    span,
                });
                self.len += 1;
            }
        }
    }

    /// The next significant token, without consuming it.
    pub(crate) fn peek(&self) -> Option<SigToken> {
        self.ring[0]
    }

    /// The significant token `n` positions ahead.
    pub(crate) fn peek_n(&self, n: usize) -> Option<SigToken> {
        self.ring.get(n).copied().flatten()
    }

    /// Consumes and returns the next significant token.
    fn bump(&mut self) -> Option<SigToken> {
        let token = self.ring[0];
        token?;
        let mut index = 0;
        while index + 1 < self.len {
            self.ring[index] = self.ring[index + 1];
            index += 1;
        }
        self.len -= 1;
        self.ring[self.len] = None;
        self.prev = token;
        self.consumed = self.consumed.saturating_add(1);
        self.fill();
        token
    }

    /// Whether the input is exhausted (an empty ring implies this).
    pub(crate) fn at_end(&self) -> bool {
        self.len == 0
    }
}

/// Parses a program from a streaming token source.
pub(crate) fn parse_program<'a>(
    source: &'a str,
    tokens: impl Iterator<Item = Token> + 'a,
) -> (Program, Vec<ParseError>) {
    let mut parser = Parser::new(source, tokens);
    let items = parser.parse_items();
    let program = Program {
        items,
        span: Span::new(0, source.len() as u32),
    };
    (program, parser.errors)
}

/// Which infix form the upcoming tokens represent.
#[derive(Debug, Clone, Copy)]
enum Infix {
    /// A binary operator: `a + b`.
    Binary {
        /// The operator.
        op: BinaryOp,
        /// Left binding power.
        lbp: u8,
        /// Number of tokens the operator spans.
        tokens: u8,
    },
    /// An assignment: `a = b` or a compound form such as `a += b`.
    Assign {
        /// The compound operator, or `None` for plain `=`.
        op: Option<BinaryOp>,
        /// Number of tokens the operator spans.
        tokens: u8,
    },
    /// A range: `a..b` / `a..=b`.
    Range {
        /// Whether the range is inclusive.
        inclusive: bool,
        /// Number of tokens the operator spans.
        tokens: u8,
    },
}

impl Infix {
    /// Left binding power of this operator.
    fn lbp(self) -> u8 {
        match self {
            Self::Binary { lbp, .. } => lbp,
            Self::Assign { .. } => BP_ASSIGN,
            Self::Range { .. } => BP_RANGE,
        }
    }
}

/// Left binding power of a simple binary operator.
fn binary_lbp(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Add | BinaryOp::Sub => 10,
        BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => 11,
        BinaryOp::Shl | BinaryOp::Shr => 9,
        BinaryOp::BitAnd => 8,
        BinaryOp::BitXor => 7,
        BinaryOp::BitOr => 6,
        BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => 5,
        BinaryOp::And => 4,
        BinaryOp::Or => 3,
    }
}

/// Whether the operator is a non-associative comparison.
fn is_comparison(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
    )
}

/// Maps a single-character token to its binary operator.
fn simple_binary(kind: TokenKind) -> Option<BinaryOp> {
    Some(match kind {
        TokenKind::Plus => BinaryOp::Add,
        TokenKind::Minus => BinaryOp::Sub,
        TokenKind::Star => BinaryOp::Mul,
        TokenKind::Slash => BinaryOp::Div,
        TokenKind::Percent => BinaryOp::Rem,
        TokenKind::And => BinaryOp::BitAnd,
        TokenKind::Or => BinaryOp::BitOr,
        TokenKind::Caret => BinaryOp::BitXor,
        _ => return None,
    })
}

/// Maps a single-character token to its compound-assignment operator.
fn compound_binary(kind: TokenKind) -> Option<BinaryOp> {
    Some(match kind {
        TokenKind::Plus => BinaryOp::Add,
        TokenKind::Minus => BinaryOp::Sub,
        TokenKind::Star => BinaryOp::Mul,
        TokenKind::Slash => BinaryOp::Div,
        TokenKind::Percent => BinaryOp::Rem,
        TokenKind::And => BinaryOp::BitAnd,
        TokenKind::Or => BinaryOp::BitOr,
        TokenKind::Caret => BinaryOp::BitXor,
        _ => return None,
    })
}

/// Whether the token can start an expression.
fn is_terminator(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Semi
            | TokenKind::Comma
            | TokenKind::CloseParen
            | TokenKind::CloseBracket
            | TokenKind::CloseBrace
    )
}

/// The recursive-descent parser state.
pub(crate) struct Parser<'a, I> {
    /// Full source text, for slicing spans.
    source: &'a str,
    /// Streaming cursor over the significant tokens.
    cursor: Cursor<I>,
    /// Diagnostics collected so far.
    errors: Vec<ParseError>,
    /// Current nesting depth for the recursion limit.
    depth: u32,
    /// Whether the recursion limit was already reported.
    depth_reported: bool,
    /// Next [`NodeId`] to mint; the only source of node identities.
    next_node_id: u32,
}

impl<'a, I: Iterator<Item = Token>> Parser<'a, I> {
    /// Creates a parser whose token stream starts at byte offset 0.
    pub(crate) fn new(source: &'a str, tokens: I) -> Self {
        Self {
            source,
            cursor: Cursor::new(tokens),
            errors: Vec::new(),
            depth: 0,
            depth_reported: false,
            next_node_id: 0,
        }
    }

    /// Parses the next top-level item, resetting the per-item state.
    ///
    /// Returns `None` at end of input. The cursor — and with it the
    /// source position, previous token, and consumed-token counter —
    /// persists across calls, while `depth`, `depth_reported`, and
    /// `next_node_id` restart at zero so every item owns a fresh
    /// [`NodeId`] space. The stream always advances or reaches EOF: an
    /// item that consumed nothing is followed by a forced single-token
    /// step, so malformed input cannot hang the caller.
    pub(crate) fn parse_next_item(&mut self) -> Option<ItemOutcome> {
        if self.at_eof() {
            return None;
        }
        self.depth = 0;
        self.depth_reported = false;
        self.next_node_id = 0;
        let start = self.cursor.consumed;
        let item = self.parse_item();
        let _ = self.force_progress(start);
        Some(ItemOutcome {
            item,
            errors: core::mem::take(&mut self.errors),
        })
    }

    /// Mints the identity for the next [`Expr`] or [`Pat`] node.
    fn mint_id(&mut self) -> NodeId {
        let id = NodeId::from_raw(self.next_node_id);
        self.next_node_id = self.next_node_id.saturating_add(1);
        debug_assert!(
            id != NodeId::DUMMY,
            "node id space exhausted: cannot mint another identity"
        );
        id
    }

    /// Span at end of input.
    fn eof_span(&self) -> Span {
        let len = u32::try_from(self.source.len()).unwrap_or(u32::MAX);
        Span::new(len, 0)
    }

    /// The upcoming token, if any.
    fn peek(&self) -> Option<SigToken> {
        self.cursor.peek()
    }

    /// The token `n` positions ahead.
    fn peek_n(&self, n: usize) -> Option<SigToken> {
        self.cursor.peek_n(n)
    }

    /// The most recently consumed token, if any.
    fn prev(&self) -> Option<SigToken> {
        self.cursor.prev
    }

    /// Span of the most recently consumed token, or end of input.
    fn prev_span(&self) -> Span {
        self.cursor
            .prev
            .map_or_else(|| self.eof_span(), |token| token.span)
    }

    /// Kind of the most recently consumed token, if any.
    fn prev_kind(&self) -> Option<TokenKind> {
        self.cursor.prev.map(|token| token.kind)
    }

    /// The number of significant tokens consumed so far.
    fn consumed(&self) -> usize {
        self.cursor.consumed
    }

    /// The upcoming token kind.
    fn kind(&self) -> Option<TokenKind> {
        self.peek().map(|token| token.kind)
    }

    /// Span of the upcoming token, or end of input.
    fn span(&self) -> Span {
        self.peek()
            .map_or_else(|| self.eof_span(), |token| token.span)
    }

    /// Source text of a span, empty when the range is invalid.
    fn text(&self, span: Span) -> &'a str {
        let start = span.offset as usize;
        let end = start.saturating_add(span.len as usize);
        self.source.get(start..end).unwrap_or("")
    }

    /// Text of the upcoming token.
    fn peek_text(&self) -> &'a str {
        self.peek()
            .map_or("", |token| self.text(token.span))
    }

    /// Whether input is exhausted.
    fn at_eof(&self) -> bool {
        self.cursor.at_end()
    }

    /// Consumes and returns the upcoming token.
    fn bump(&mut self) -> Option<SigToken> {
        self.cursor.bump()
    }

    /// Consumes `n` tokens that a prior check verified are present.
    fn bump_n(&mut self, n: u8) {
        for _ in 0..n {
            if self.cursor.bump().is_none() {
                break;
            }
        }
    }

    /// Records a diagnostic at `span`.
    fn error(&mut self, span: Span, message: String) {
        self.errors.push(ParseError { span, message });
    }

    /// Whether the next token is `kind`.
    fn at(&self, kind: TokenKind) -> bool {
        self.kind() == Some(kind)
    }

    /// Consumes the token if it is `kind`.
    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.at(kind) {
            let _ = self.bump();
            true
        } else {
            false
        }
    }

    /// Consumes `kind`, or records `what` as missing without consuming.
    fn expect(&mut self, kind: TokenKind, what: &str) -> Option<Span> {
        if self.eat(kind) {
            return self.prev().map(|token| token.span);
        }
        let span = self.span();
        self.error(span, format!("expected {what}"));
        None
    }

    /// Whether the upcoming token is the keyword `kw`.
    fn at_kw(&self, kw: &str) -> bool {
        self.at(TokenKind::Ident) && self.peek_text() == kw
    }

    /// Consumes the keyword `kw` if present.
    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.at_kw(kw) {
            let _ = self.bump();
            true
        } else {
            false
        }
    }

    /// Whether the upcoming token is a (raw) identifier.
    fn at_ident(&self) -> bool {
        matches!(self.kind(), Some(TokenKind::Ident | TokenKind::RawIdent))
    }

    /// Whether the upcoming token is a lifetime.
    fn at_lifetime(&self) -> bool {
        matches!(self.kind(), Some(TokenKind::Lifetime { .. }))
    }

    /// Consumes an identifier and returns its span and text.
    fn bump_ident(&mut self) -> Option<(Span, String)> {
        if !self.at_ident() {
            return None;
        }
        let span = self.bump()?.span;
        Some((span, String::from(self.text(span))))
    }

    /// Consumes an identifier or records `what` as missing.
    fn expect_ident(&mut self, what: &str) -> Option<(Span, String)> {
        if let Some(ident) = self.bump_ident() {
            return Some(ident);
        }
        let span = self.span();
        self.error(span, format!("expected {what}"));
        None
    }

    /// Whether the tokens `n` and `n + 1` ahead are adjacent in source.
    ///
    /// Trivia lengths are included in the accumulated spans, so only
    /// genuinely glued tokens such as `->` or `..=` report adjacency.
    fn joint_at(&self, n: usize) -> bool {
        match (self.peek_n(n), self.peek_n(n + 1)) {
            (Some(a), Some(b)) => a.span.offset.saturating_add(a.span.len) == b.span.offset,
            _ => false,
        }
    }

    /// Whether the next two tokens are the joint pair `a`, `b`.
    fn at_joint(&self, a: TokenKind, b: TokenKind) -> bool {
        self.at(a) && self.joint_at(0) && self.kind_at(1) == Some(b)
    }

    /// Whether the next three tokens are the joint chain `a`, `b`, `c`.
    fn at_joint3(&self, a: TokenKind, b: TokenKind, c: TokenKind) -> bool {
        self.at(a)
            && self.joint_at(0)
            && self.kind_at(1) == Some(b)
            && self.joint_at(1)
            && self.kind_at(2) == Some(c)
    }

    /// Kind of the token `n` ahead.
    fn kind_at(&self, n: usize) -> Option<TokenKind> {
        self.peek_n(n).map(|token| token.kind)
    }

    /// Enters a recursive production; reports the recursion limit once.
    fn enter(&mut self) -> bool {
        if self.depth >= MAX_DEPTH {
            if !self.depth_reported {
                self.depth_reported = true;
                let span = self.span();
                self.error(span, String::from("recursion limit exceeded"));
            }
            return false;
        }
        self.depth += 1;
        true
    }

    /// Leaves a recursive production entered with [`Parser::enter`].
    fn leave(&mut self) {
        self.depth -= 1;
    }

    /// Consumes one token when a recovery loop made no progress.
    ///
    /// `start` is the consumed-token counter observed before the loop
    /// began. Returns `false` only at end of input, where loops must
    /// stop.
    fn force_progress(&mut self, start: usize) -> bool {
        if self.cursor.consumed > start {
            return true;
        }
        if self.at_eof() {
            return false;
        }
        let _ = self.bump();
        true
    }
}

/// Joins two spans into a covering span.
fn join(a: Span, b: Span) -> Span {
    Span::new(a.offset, b.end().saturating_sub(a.offset))
}

impl<I: Iterator<Item = Token>> Parser<'_, I> {
    /// Parses all top-level items until end of input.
    fn parse_items(&mut self) -> Vec<Item> {
        let mut items = Vec::new();
        while !self.at_eof() {
            let start = self.consumed();
            items.push(self.parse_item());
            if !self.force_progress(start) {
                break;
            }
        }
        items
    }

    /// Parses one item, including its leading attributes.
    fn parse_item(&mut self) -> Item {
        let attrs = self.parse_attrs();
        let start = attrs
            .first()
            .map_or(self.span(), |attr| attr.span);
        if self.at_kw("fn") {
            return self.parse_fn(attrs, start);
        }
        if self.at_kw("struct") {
            return self.parse_struct(attrs, start);
        }
        if self.at_kw("type") {
            return self.parse_type_alias(attrs, start);
        }
        let span = self.span();
        self.error(span, String::from("expected item"));
        Item {
            attrs,
            kind: ItemKind::Error,
            span: join(start, span),
        }
    }

    /// Consumes a run of `#[…]` attributes as opaque spans.
    fn parse_attrs(&mut self) -> Vec<Attr> {
        let mut attrs = Vec::new();
        while self.at(TokenKind::Pound) && self.at_joint(TokenKind::Pound, TokenKind::OpenBracket) {
            let start = self
                .bump()
                .map_or(self.eof_span(), |token| token.span);
            self.bump();
            let mut depth = 1u32;
            let mut end = start;
            while depth > 0 && !self.at_eof() {
                let token = self
                    .bump()
                    .map_or(self.eof_span(), |token| token.span);
                end = token;
                match self.prev_kind() {
                    Some(TokenKind::OpenBracket) => depth += 1,
                    Some(TokenKind::CloseBracket) => depth -= 1,
                    _ => {}
                }
            }
            if depth > 0 {
                let span = self.span();
                self.error(span, String::from("unclosed attribute"));
            }
            attrs.push(Attr {
                span: join(start, end),
            });
        }
        attrs
    }

    /// Parses `fn` with already-collected attributes.
    fn parse_fn(&mut self, attrs: Vec<Attr>, start: Span) -> Item {
        self.bump();
        let name = self
            .expect_ident("function name")
            .map_or_else(String::new, |(_, name)| name);
        let generics = self.parse_generic_params();
        self.expect(TokenKind::OpenParen, "`(`");
        let params = self.parse_params();
        let ret = self.parse_fn_return();
        let body = if self.at(TokenKind::OpenBrace) {
            self.parse_block()
        } else {
            let span = self.span();
            self.error(span, String::from("expected function body"));
            Block {
                stmts: Vec::new(),
                tail: None,
                span,
            }
        };
        let body_span = body.span;
        Item {
            attrs,
            span: join(start, body_span),
            kind: ItemKind::Fn(FnItem {
                name,
                generics,
                params,
                ret,
                body,
                span: join(start, body_span),
            }),
        }
    }

    /// Parses the optional `-> Type` return annotation.
    fn parse_fn_return(&mut self) -> Option<Type> {
        if !self.at(TokenKind::Minus) {
            return None;
        }
        if self.at_joint(TokenKind::Minus, TokenKind::Gt) {
            self.bump_n(2);
            return Some(self.parse_type());
        }
        let span = self.span();
        self.error(span, String::from("expected `->`"));
        self.bump();
        None
    }

    /// Parses `(pattern: Type, …)`.
    fn parse_params(&mut self) -> Vec<Param> {
        let mut params = Vec::new();
        loop {
            if self.at(TokenKind::CloseParen) || self.at_eof() {
                break;
            }
            let start = self.span();
            let pat = self.parse_pat();
            self.expect(TokenKind::Colon, "`:`");
            let ty = self.parse_type();
            params.push(Param {
                span: join(start, ty.span),
                pat,
                ty,
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::CloseParen, "`)`");
        params
    }

    /// Parses `<lifetime, Type, …>` after a name; empty when absent.
    fn parse_generic_params(&mut self) -> Vec<GenericParam> {
        if !self.at(TokenKind::Lt) {
            return Vec::new();
        }
        self.bump();
        let mut params = Vec::new();
        loop {
            if self.at(TokenKind::Gt) || self.at_eof() {
                break;
            }
            if self.at_lifetime() {
                let span = self
                    .bump()
                    .map_or(self.eof_span(), |token| token.span);
                let text = self.text(span);
                params.push(GenericParam::Lifetime {
                    name: String::from(text.strip_prefix('\'').unwrap_or(text)),
                    span,
                });
            } else if let Some((span, name)) = self.bump_ident() {
                params.push(GenericParam::Type { name, span });
            } else {
                let span = self.span();
                self.error(span, String::from("expected generic parameter"));
                break;
            }
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::Gt, "`>`");
        params
    }

    /// Parses `struct Name … { … }` or `struct Name;`.
    fn parse_struct(&mut self, attrs: Vec<Attr>, start: Span) -> Item {
        self.bump();
        let name = self
            .expect_ident("struct name")
            .map_or_else(String::new, |(_, name)| name);
        let generics = self.parse_generic_params();
        let mut fields = Vec::new();
        let mut unit = false;
        let end = if self.at(TokenKind::OpenBrace) {
            self.bump();
            loop {
                if self.at(TokenKind::CloseBrace) || self.at_eof() {
                    break;
                }
                let field_index = self.consumed();
                let field_start = self.span();
                let Some((_, field_name)) = self.expect_ident("field name") else {
                    if !self.force_progress(field_index) {
                        break;
                    }
                    continue;
                };
                self.expect(TokenKind::Colon, "`:`");
                let ty = self.parse_type();
                fields.push(Field {
                    span: join(field_start, ty.span),
                    name: field_name,
                    ty,
                });
                if !self.eat(TokenKind::Comma) {
                    break;
                }
            }
            self.expect(TokenKind::CloseBrace, "`}`")
                .unwrap_or_else(|| self.span())
        } else if self.at(TokenKind::Semi) {
            unit = true;
            let _ = self.bump();
            self.prev_span()
        } else {
            let span = self.span();
            self.error(span, String::from("expected `{` or `;`"));
            span
        };
        Item {
            attrs,
            span: join(start, end),
            kind: ItemKind::Struct(StructItem {
                name,
                generics,
                fields,
                unit,
                span: join(start, end),
            }),
        }
    }

    /// Parses `type Name = T;`.
    fn parse_type_alias(&mut self, attrs: Vec<Attr>, start: Span) -> Item {
        self.bump();
        let name = self
            .expect_ident("type alias name")
            .map_or_else(String::new, |(_, name)| name);
        let generics = self.parse_generic_params();
        self.expect(TokenKind::Eq, "`=`");
        let aliased = self.parse_type();
        let end = self
            .expect(TokenKind::Semi, "`;`")
            .unwrap_or_else(|| self.span());
        Item {
            attrs,
            span: join(start, end),
            kind: ItemKind::TypeAlias(TypeAliasItem {
                name,
                generics,
                aliased,
                span: join(start, end),
            }),
        }
    }
}

impl<I: Iterator<Item = Token>> Parser<'_, I> {
    /// Parses a binding pattern: `x`, `mut x`, or `_`.
    fn parse_pat(&mut self) -> Pat {
        let start = self.span();
        if self.at_kw("mut") {
            self.bump();
            match self.expect_ident("pattern name") {
                Some((span, name)) => Pat {
                    id: self.mint_id(),
                    kind: PatKind::Ident { name, mutable: true },
                    span: join(start, span),
                },
                None => Pat {
                    id: self.mint_id(),
                    kind: PatKind::Error,
                    span: start,
                },
            }
        } else if let Some((span, name)) = self.bump_ident() {
            if name == "_" {
                Pat {
                    id: self.mint_id(),
                    kind: PatKind::Wild,
                    span,
                }
            } else {
                Pat {
                    id: self.mint_id(),
                    kind: PatKind::Ident { name, mutable: false },
                    span,
                }
            }
        } else {
            let span = self.span();
            self.error(span, String::from("expected pattern"));
            Pat {
                id: self.mint_id(),
                kind: PatKind::Error,
                span,
            }
        }
    }

    /// Parses a type expression.
    fn parse_type(&mut self) -> Type {
        if !self.enter() {
            let span = self.span();
            return Type {
                kind: TypeKind::Error,
                span,
            };
        }
        let result = self.parse_type_inner();
        self.leave();
        result
    }

    /// Body of [`Parser::parse_type`] behind the recursion limit.
    fn parse_type_inner(&mut self) -> Type {
        let start = self.span();
        match self.kind() {
            Some(TokenKind::And) => {
                self.bump();
                let lifetime = if self.at_lifetime() {
                    let span = self
                        .bump()
                        .map_or(self.eof_span(), |token| token.span);
                    let text = self.text(span);
                    Some(String::from(text.strip_prefix('\'').unwrap_or(text)))
                } else {
                    None
                };
                let mutable = self.eat_kw("mut");
                let inner = self.parse_type();
                Type {
                    span: join(start, inner.span),
                    kind: TypeKind::Ref {
                        lifetime,
                        mutable,
                        inner: Box::new(inner),
                    },
                }
            }
            Some(TokenKind::Star) => {
                self.bump();
                let mutable = if self.eat_kw("mut") {
                    true
                } else if self.eat_kw("const") {
                    false
                } else {
                    let span = self.span();
                    self.error(span, String::from("expected `const` or `mut`"));
                    false
                };
                let inner = self.parse_type();
                Type {
                    span: join(start, inner.span),
                    kind: TypeKind::Ptr {
                        mutable,
                        inner: Box::new(inner),
                    },
                }
            }
            Some(TokenKind::OpenParen) => {
                self.bump();
                if self.eat(TokenKind::CloseParen) {
                    return Type {
                        kind: TypeKind::Tuple(Vec::new()),
                        span: join(start, self.prev_span()),
                    };
                }
                let mut types = Vec::new();
                loop {
                    types.push(self.parse_type());
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                    if self.at(TokenKind::CloseParen) || self.at_eof() {
                        break;
                    }
                }
                let trailing_comma = matches!(self.prev_kind(), Some(TokenKind::Comma));
                let end = self
                    .expect(TokenKind::CloseParen, "`)`")
                    .unwrap_or_else(|| self.span());
                if types.len() == 1 && !trailing_comma {
                    let inner = types.pop().unwrap_or(Type {
                        kind: TypeKind::Error,
                        span: start,
                    });
                    return Type {
                        kind: inner.kind,
                        span: join(start, end),
                    };
                }
                Type {
                    kind: TypeKind::Tuple(types),
                    span: join(start, end),
                }
            }
            Some(TokenKind::OpenBracket) => {
                self.bump();
                let elem = self.parse_type();
                if self.eat(TokenKind::Semi) {
                    let len = self.parse_expr(0);
                    let end = self
                        .expect(TokenKind::CloseBracket, "`]`")
                        .unwrap_or_else(|| self.span());
                    Type {
                        span: join(start, end),
                        kind: TypeKind::Array {
                            elem: Box::new(elem),
                            len: Box::new(len),
                        },
                    }
                } else {
                    let end = self
                        .expect(TokenKind::CloseBracket, "`]`")
                        .unwrap_or_else(|| self.span());
                    Type {
                        span: join(start, end),
                        kind: TypeKind::Slice(Box::new(elem)),
                    }
                }
            }
            Some(TokenKind::Ident | TokenKind::RawIdent) => {
                let path = self.parse_path();
                let args = if self.at(TokenKind::Lt) {
                    self.parse_generic_args()
                } else {
                    Vec::new()
                };
                let last = args.last().map_or(path.span, |arg| match arg {
                    GenericArg::Type(ty) => ty.span,
                    GenericArg::Lifetime { span, .. } => *span,
                });
                let full = join(path.span, last);
                Type {
                    span: full,
                    kind: TypeKind::Path(TypePath {
                        segments: path.segments,
                        args,
                        span: full,
                    }),
                }
            }
            _ => {
                let span = self.span();
                self.error(span, String::from("expected type"));
                Type {
                    kind: TypeKind::Error,
                    span,
                }
            }
        }
    }

    /// Parses `<Type, 'a, …>` generic arguments.
    fn parse_generic_args(&mut self) -> Vec<GenericArg> {
        self.bump();
        let mut args = Vec::new();
        loop {
            if self.at(TokenKind::Gt) || self.at_eof() {
                break;
            }
            if self.at_lifetime() {
                let span = self
                    .bump()
                    .map_or(self.eof_span(), |token| token.span);
                let text = self.text(span);
                args.push(GenericArg::Lifetime {
                    name: String::from(text.strip_prefix('\'').unwrap_or(text)),
                    span,
                });
            } else if self.starts_type() {
                args.push(GenericArg::Type(self.parse_type()));
            } else {
                let span = self.span();
                self.error(span, String::from("expected generic argument"));
                break;
            }
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::Gt, "`>`");
        args
    }

    /// Whether the upcoming token can begin a type.
    fn starts_type(&self) -> bool {
        matches!(
            self.kind(),
            Some(
                TokenKind::And
                    | TokenKind::Star
                    | TokenKind::OpenParen
                    | TokenKind::OpenBracket
                    | TokenKind::Ident
                    | TokenKind::RawIdent
            )
        )
    }

    /// Parses a `{ … }` block with statements and an optional tail.
    fn parse_block(&mut self) -> Block {
        let Some(open) = self.expect(TokenKind::OpenBrace, "`{`") else {
            let span = self.span();
            return Block {
                stmts: Vec::new(),
                tail: None,
                span,
            };
        };
        if !self.enter() {
            return Block {
                stmts: Vec::new(),
                tail: None,
                span: open,
            };
        }
        let mut stmts = Vec::new();
        let mut tail = None;
        loop {
            if self.at(TokenKind::CloseBrace) || self.at_eof() {
                break;
            }
            let start = self.consumed();
            match self.parse_stmt() {
                StmtOutcome::Stmt(stmt) => stmts.push(stmt),
                StmtOutcome::Tail(expr) => {
                    tail = Some(Box::new(expr));
                    break;
                }
                StmtOutcome::Stopped => break,
            }
            if !self.force_progress(start) {
                break;
            }
        }
        let end = self
            .expect(TokenKind::CloseBrace, "`}`")
            .unwrap_or_else(|| self.span());
        self.leave();
        Block {
            stmts,
            tail,
            span: join(open, end),
        }
    }

    /// Parses one statement.
    fn parse_stmt(&mut self) -> StmtOutcome {
        if self.at_kw("let") {
            let start = self.span();
            self.bump();
            let pat = self.parse_pat();
            let mut ty = None;
            if self.eat(TokenKind::Colon) {
                ty = Some(self.parse_type());
            }
            let mut init = None;
            if self.eat(TokenKind::Eq) {
                init = Some(self.parse_expr(0));
            }
            let end = self
                .expect(TokenKind::Semi, "`;`")
                .unwrap_or_else(|| self.span());
            return StmtOutcome::Stmt(Stmt {
                kind: StmtKind::Let(LetStmt {
                    pat,
                    ty,
                    init,
                    span: join(start, end),
                }),
                span: join(start, end),
            });
        }
        if self.at(TokenKind::Semi) {
            let span = self
                .bump()
                .map_or(self.eof_span(), |token| token.span);
            return StmtOutcome::Stmt(Stmt {
                kind: StmtKind::Empty,
                span,
            });
        }
        if self.at(TokenKind::CloseBrace) {
            return StmtOutcome::Stopped;
        }
        let expr = self.parse_expr(0);
        let expr_span = expr.span;
        if self.eat(TokenKind::Semi) {
            let end = self.prev_span();
            return StmtOutcome::Stmt(Stmt {
                kind: StmtKind::Expr(expr),
                span: join(expr_span, end),
            });
        }
        if self.at(TokenKind::CloseBrace) {
            return StmtOutcome::Tail(expr);
        }
        let block_like = matches!(
            expr.kind,
            ExprKind::Block(_)
                | ExprKind::If { .. }
                | ExprKind::While { .. }
                | ExprKind::Loop { .. }
                | ExprKind::For { .. }
        );
        if !block_like {
            let span = self.span();
            self.error(span, String::from("expected `;`"));
        }
        StmtOutcome::Stmt(Stmt {
            kind: StmtKind::Expr(expr),
            span: expr_span,
        })
    }
}

/// Result of parsing a single statement.
#[derive(Debug)]
enum StmtOutcome {
    /// A complete statement; parsing continues.
    Stmt(Stmt),
    /// A tail expression immediately before `}`.
    Tail(Expr),
    /// The block should end (only at `}`).
    Stopped,
}

impl<I: Iterator<Item = Token>> Parser<'_, I> {
    /// Parses an expression whose operators bind at least as tight as
    /// `min_bp`.
    fn parse_expr(&mut self, min_bp: u8) -> Expr {
        if !self.enter() {
            let span = self.span();
            return Expr {
                id: self.mint_id(),
                kind: ExprKind::Error,
                span,
            };
        }
        let result = self.parse_expr_inner(min_bp);
        self.leave();
        result
    }

    /// Precedence-climbing loop; multi-character operators are composed
    /// from adjacent tokens on demand.
    fn parse_expr_inner(&mut self, min_bp: u8) -> Expr {
        let mut lhs = self.parse_unary();
        let mut cmp_pending = false;
        loop {
            if self.at_kw("as") {
                if BP_CAST < min_bp {
                    break;
                }
                self.bump();
                let ty = self.parse_type();
                let span = join(lhs.span, ty.span);
                lhs = Expr {
                    id: self.mint_id(),
                    kind: ExprKind::Cast {
                        expr: Box::new(lhs),
                        ty: Box::new(ty),
                    },
                    span,
                };
                cmp_pending = false;
                continue;
            }
            let Some(infix) = self.peek_infix() else {
                break;
            };
            let lbp = infix.lbp();
            if lbp < min_bp {
                break;
            }
            match infix {
                Infix::Binary { op, tokens, .. } => {
                    if is_comparison(op) && cmp_pending {
                        let span = self.span();
                        self.error(span, String::from("comparison operators are non-associative"));
                    }
                    self.bump_n(tokens);
                    let rhs = self.parse_expr(lbp + 1);
                    let span = join(lhs.span, rhs.span);
                    lhs = Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Binary {
                            op,
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        },
                        span,
                    };
                    cmp_pending = is_comparison(op);
                }
                Infix::Assign { op, tokens } => {
                    self.bump_n(tokens);
                    let rhs = self.parse_expr(BP_ASSIGN);
                    let span = join(lhs.span, rhs.span);
                    lhs = Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Assign {
                            op,
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        },
                        span,
                    };
                    cmp_pending = false;
                }
                Infix::Range { inclusive, tokens } => {
                    self.bump_n(tokens);
                    let end = self.range_operand();
                    let span = end
                        .as_deref()
                        .map_or(lhs.span, |expr| join(lhs.span, expr.span));
                    lhs = Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Range {
                            start: Some(Box::new(lhs)),
                            end,
                            inclusive,
                        },
                        span,
                    };
                    cmp_pending = false;
                }
            }
        }
        lhs
    }

    /// Classifies the upcoming tokens as an infix operator, if any.
    fn peek_infix(&self) -> Option<Infix> {
        let kind = self.kind()?;
        match kind {
            TokenKind::Eq => {
                if self.at_joint(TokenKind::Eq, TokenKind::Eq) {
                    Some(Infix::Binary {
                        op: BinaryOp::Eq,
                        lbp: 5,
                        tokens: 2,
                    })
                } else {
                    Some(Infix::Assign { op: None, tokens: 1 })
                }
            }
            TokenKind::Bang if self.at_joint(TokenKind::Bang, TokenKind::Eq) => Some(Infix::Binary {
                op: BinaryOp::Ne,
                lbp: 5,
                tokens: 2,
            }),
            TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Star
            | TokenKind::Slash
            | TokenKind::Percent
            | TokenKind::And
            | TokenKind::Or
            | TokenKind::Caret => {
                if kind == TokenKind::And && self.at_joint(TokenKind::And, TokenKind::And) {
                    return Some(Infix::Binary {
                        op: BinaryOp::And,
                        lbp: 4,
                        tokens: 2,
                    });
                }
                if kind == TokenKind::Or && self.at_joint(TokenKind::Or, TokenKind::Or) {
                    return Some(Infix::Binary {
                        op: BinaryOp::Or,
                        lbp: 3,
                        tokens: 2,
                    });
                }
                if self.at_joint(kind, TokenKind::Eq) {
                    return Some(Infix::Assign {
                        op: compound_binary(kind),
                        tokens: 2,
                    });
                }
                let op = simple_binary(kind)?;
                Some(Infix::Binary {
                    op,
                    lbp: binary_lbp(op),
                    tokens: 1,
                })
            }
            TokenKind::Lt => {
                if self.at_joint(TokenKind::Lt, TokenKind::Eq) {
                    Some(Infix::Binary {
                        op: BinaryOp::Le,
                        lbp: 5,
                        tokens: 2,
                    })
                } else if self.at_joint(TokenKind::Lt, TokenKind::Lt) {
                    Some(Infix::Binary {
                        op: BinaryOp::Shl,
                        lbp: 9,
                        tokens: 2,
                    })
                } else {
                    Some(Infix::Binary {
                        op: BinaryOp::Lt,
                        lbp: 5,
                        tokens: 1,
                    })
                }
            }
            TokenKind::Gt => {
                if self.at_joint(TokenKind::Gt, TokenKind::Eq) {
                    Some(Infix::Binary {
                        op: BinaryOp::Ge,
                        lbp: 5,
                        tokens: 2,
                    })
                } else if self.at_joint(TokenKind::Gt, TokenKind::Gt) {
                    Some(Infix::Binary {
                        op: BinaryOp::Shr,
                        lbp: 9,
                        tokens: 2,
                    })
                } else {
                    Some(Infix::Binary {
                        op: BinaryOp::Gt,
                        lbp: 5,
                        tokens: 1,
                    })
                }
            }
            TokenKind::Dot => {
                if self.at_joint3(TokenKind::Dot, TokenKind::Dot, TokenKind::Eq) {
                    Some(Infix::Range {
                        inclusive: true,
                        tokens: 3,
                    })
                } else if self.at_joint(TokenKind::Dot, TokenKind::Dot) {
                    Some(Infix::Range {
                        inclusive: false,
                        tokens: 2,
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Parses the optional right-hand side of a range.
    fn range_operand(&mut self) -> Option<Box<Expr>> {
        if self.starts_range_operand() {
            Some(Box::new(self.parse_expr(BP_RANGE_OPERAND)))
        } else {
            None
        }
    }

    /// Whether an expression may follow `..`.
    ///
    /// Block-like openers are excluded so that `for i in 0.. { … }` ends
    /// the range before the loop body (rust-analyzer's restriction).
    fn starts_range_operand(&self) -> bool {
        match self.kind() {
            None => false,
            Some(kind) if is_terminator(kind) => false,
            Some(TokenKind::OpenBrace) => false,
            Some(TokenKind::Ident | TokenKind::RawIdent) => {
                !matches!(self.peek_text(), "if" | "while" | "loop" | "for")
            }
            Some(_) => true,
        }
    }

    /// Whether no operand follows `return` / `break`.
    fn at_value_end(&self) -> bool {
        self.kind().is_none_or(is_terminator)
    }

    /// Parses prefix operators and, below them, a postfix expression.
    fn parse_unary(&mut self) -> Expr {
        let start = self.span();
        match self.kind() {
            Some(TokenKind::Minus) | Some(TokenKind::Bang) | Some(TokenKind::Star) => {
                let op = match self.kind() {
                    Some(TokenKind::Minus) => UnaryOp::Neg,
                    Some(TokenKind::Bang) => UnaryOp::Not,
                    _ => UnaryOp::Deref,
                };
                self.bump();
                let expr = self.parse_expr(BP_UNARY);
                let span = join(start, expr.span);
                Expr {
                    id: self.mint_id(),
                    kind: ExprKind::Unary {
                        op,
                        expr: Box::new(expr),
                    },
                    span,
                }
            }
            Some(TokenKind::And) => {
                self.bump();
                let mutable = self.eat_kw("mut");
                let expr = self.parse_expr(BP_UNARY);
                let span = join(start, expr.span);
                Expr {
                    id: self.mint_id(),
                    kind: ExprKind::Unary {
                        op: UnaryOp::AddrOf { mutable },
                        expr: Box::new(expr),
                    },
                    span,
                }
            }
            Some(TokenKind::Dot) if self.at_joint(TokenKind::Dot, TokenKind::Dot) => {
                let inclusive = self.at_joint3(TokenKind::Dot, TokenKind::Dot, TokenKind::Eq);
                self.bump_n(if inclusive { 3 } else { 2 });
                let end = self.range_operand();
                let span = end
                    .as_deref()
                    .map_or(start, |expr| join(start, expr.span));
                Expr {
                    id: self.mint_id(),
                    kind: ExprKind::Range {
                        start: None,
                        end,
                        inclusive,
                    },
                    span,
                }
            }
            _ => self.parse_postfix(),
        }
    }

    /// Parses a primary expression followed by postfix operators.
    fn parse_postfix(&mut self) -> Expr {
        let mut expr = self.parse_primary();
        loop {
            match self.kind() {
                Some(TokenKind::OpenParen) => {
                    self.bump();
                    let (args, end) = self.parse_call_args();
                    let span = join(expr.span, end);
                    expr = Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Call {
                            callee: Box::new(expr),
                            args,
                        },
                        span,
                    };
                }
                Some(TokenKind::OpenBracket) => {
                    self.bump();
                    let index = self.parse_expr(0);
                    let end = self
                        .expect(TokenKind::CloseBracket, "`]`")
                        .unwrap_or_else(|| self.span());
                    let span = join(expr.span, end);
                    expr = Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Index {
                            expr: Box::new(expr),
                            index: Box::new(index),
                        },
                        span,
                    };
                }
                Some(TokenKind::Dot) if !self.at_joint(TokenKind::Dot, TokenKind::Dot) => {
                    self.bump();
                    let member = if let Some((_, name)) = self.bump_ident() {
                        name
                    } else if matches!(
                        self.kind(),
                        Some(TokenKind::Literal {
                            kind: codevar_ocl_lex::LiteralKind::Int { .. },
                            ..
                        })
                    ) {
                        let span = self
                            .bump()
                            .map_or(self.eof_span(), |token| token.span);
                        String::from(self.text(span))
                    } else {
                        let span = self.span();
                        self.error(span, String::from("expected field name"));
                        break;
                    };
                    let span = join(expr.span, self.prev_span());
                    expr = Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Field {
                            expr: Box::new(expr),
                            name: member,
                        },
                        span,
                    };
                }
                Some(TokenKind::Question) => {
                    self.bump();
                    let span = join(expr.span, self.prev_span());
                    expr = Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Try { expr: Box::new(expr) },
                        span,
                    };
                }
                _ => break,
            }
        }
        expr
    }

    /// Parses `a, b, …` up to and including the closing `)`.
    ///
    /// Returns the arguments and the span of the closing token (or the
    /// recovery span when it is missing).
    fn parse_call_args(&mut self) -> (Vec<Expr>, Span) {
        let mut args = Vec::new();
        loop {
            if self.at(TokenKind::CloseParen) || self.at_eof() {
                break;
            }
            args.push(self.parse_expr(0));
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        let end = self
            .expect(TokenKind::CloseParen, "`)`")
            .unwrap_or_else(|| self.span());
        (args, end)
    }

    /// Parses a primary expression.
    fn parse_primary(&mut self) -> Expr {
        let start = self.span();
        match self.kind() {
            None => {
                let span = self.eof_span();
                self.error(span, String::from("expected expression"));
                Expr {
                    id: self.mint_id(),
                    kind: ExprKind::Error,
                    span,
                }
            }
            Some(TokenKind::Literal { .. }) => {
                let token = self.peek().unwrap_or(SigToken {
                    kind: TokenKind::Eof,
                    span: start,
                });
                self.bump();
                let span = token.span;
                let text = String::from(self.text(span));
                Expr {
                    id: self.mint_id(),
                    kind: ExprKind::Literal {
                        text,
                        kind: token.kind,
                    },
                    span,
                }
            }
            Some(TokenKind::Ident | TokenKind::RawIdent) => match self.peek_text() {
                "true" => {
                    self.bump();
                    Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Bool(true),
                        span: start,
                    }
                }
                "false" => {
                    self.bump();
                    Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Bool(false),
                        span: start,
                    }
                }
                "if" => self.parse_if(),
                "while" => self.parse_while(),
                "loop" => self.parse_loop(),
                "for" => self.parse_for(),
                "return" | "break" => {
                    let is_return = self.peek_text() == "return";
                    self.bump();
                    let value = if self.at_value_end() {
                        None
                    } else {
                        Some(Box::new(self.parse_expr(BP_ASSIGN)))
                    };
                    let span = value
                        .as_deref()
                        .map_or(start, |expr| join(start, expr.span));
                    Expr {
                        id: self.mint_id(),
                        kind: if is_return {
                            ExprKind::Return { expr: value }
                        } else {
                            ExprKind::Break { expr: value }
                        },
                        span,
                    }
                }
                "continue" => {
                    self.bump();
                    Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Continue,
                        span: start,
                    }
                }
                "else" | "fn" | "let" | "struct" | "type" | "in" | "as" | "mut" => {
                    self.error(start, String::from("expected expression"));
                    Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Error,
                        span: start,
                    }
                }
                _ => {
                    let path = self.parse_path();
                    let span = path.span;
                    Expr {
                        id: self.mint_id(),
                        kind: ExprKind::Path(path),
                        span,
                    }
                }
            },
            Some(TokenKind::OpenBrace) => {
                let block = self.parse_block();
                let span = block.span;
                Expr {
                    id: self.mint_id(),
                    kind: ExprKind::Block(block),
                    span,
                }
            }
            Some(TokenKind::OpenParen) => self.parse_paren_or_tuple(),
            Some(TokenKind::OpenBracket) => self.parse_array(),
            Some(TokenKind::Unknown | TokenKind::UnknownPrefix) => {
                let token = self
                    .bump()
                    .map_or(self.eof_span(), |token| token.span);
                self.error(token, String::from("unexpected character"));
                Expr {
                    id: self.mint_id(),
                    kind: ExprKind::Error,
                    span: token,
                }
            }
            Some(_) => {
                self.error(start, String::from("expected expression"));
                Expr {
                    id: self.mint_id(),
                    kind: ExprKind::Error,
                    span: start,
                }
            }
        }
    }

    /// Parses `(…)`: unit, parenthesized expression, or tuple.
    fn parse_paren_or_tuple(&mut self) -> Expr {
        let start = self
            .bump()
            .map_or(self.eof_span(), |token| token.span);
        if self.at(TokenKind::CloseParen) {
            let end = self
                .bump()
                .map_or(self.eof_span(), |token| token.span);
            return Expr {
                id: self.mint_id(),
                kind: ExprKind::Tuple(Vec::new()),
                span: join(start, end),
            };
        }
        let first = self.parse_expr(0);
        if self.at(TokenKind::CloseParen) {
            let end = self
                .bump()
                .map_or(self.eof_span(), |token| token.span);
            return Expr {
                id: self.mint_id(),
                kind: first.kind,
                span: join(start, end),
            };
        }
        let mut elems = vec![first];
        if self.eat(TokenKind::Comma) {
            loop {
                if self.at(TokenKind::CloseParen) || self.at_eof() {
                    break;
                }
                elems.push(self.parse_expr(0));
                if !self.eat(TokenKind::Comma) {
                    break;
                }
            }
        }
        let end = self
            .expect(TokenKind::CloseParen, "`)`")
            .unwrap_or_else(|| self.span());
        Expr {
            id: self.mint_id(),
            kind: ExprKind::Tuple(elems),
            span: join(start, end),
        }
    }

    /// Parses `[a, b]`.
    fn parse_array(&mut self) -> Expr {
        let start = self
            .bump()
            .map_or(self.eof_span(), |token| token.span);
        let mut elems = Vec::new();
        loop {
            if self.at(TokenKind::CloseBracket) || self.at_eof() {
                break;
            }
            elems.push(self.parse_expr(0));
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        let end = self
            .expect(TokenKind::CloseBracket, "`]`")
            .unwrap_or_else(|| self.span());
        Expr {
            id: self.mint_id(),
            kind: ExprKind::Array(elems),
            span: join(start, end),
        }
    }

    /// Parses a `a::b` path.
    fn parse_path(&mut self) -> Path {
        let start = self.span();
        let mut segments = Vec::new();
        match self.bump_ident() {
            Some((span, name)) => segments.push(Segment { name, span }),
            None => {
                self.error(start, String::from("expected identifier"));
                return Path {
                    segments,
                    span: start,
                };
            }
        }
        while self.at_joint(TokenKind::Colon, TokenKind::Colon) {
            self.bump_n(2);
            match self.bump_ident() {
                Some((span, name)) => segments.push(Segment { name, span }),
                None => {
                    let span = self.span();
                    self.error(span, String::from("expected path segment"));
                    break;
                }
            }
        }
        let end = segments
            .last()
            .map_or(start, |segment| segment.span);
        Path {
            segments,
            span: join(start, end),
        }
    }

    /// Parses an `if` / `else if` chain with depth protection.
    fn parse_if(&mut self) -> Expr {
        if !self.enter() {
            let span = self.span();
            return Expr {
                id: self.mint_id(),
                kind: ExprKind::Error,
                span,
            };
        }
        let result = self.parse_if_inner();
        self.leave();
        result
    }

    /// Body of [`Parser::parse_if`] behind the recursion limit.
    fn parse_if_inner(&mut self) -> Expr {
        let start = self.span();
        self.bump();
        let cond = self.parse_expr(0);
        let then = self.parse_block();
        let mut span = join(start, then.span);
        let else_branch = if self.eat_kw("else") {
            if self.at_kw("if") {
                let branch = self.parse_if();
                span = join(span, branch.span);
                Some(Box::new(branch))
            } else if self.at(TokenKind::OpenBrace) {
                let block = self.parse_block();
                let block_span = block.span;
                span = join(span, block_span);
                Some(Box::new(Expr {
                    id: self.mint_id(),
                    kind: ExprKind::Block(block),
                    span: block_span,
                }))
            } else {
                let branch_span = self.span();
                self.error(branch_span, String::from("expected `{` or `if` after `else`"));
                None
            }
        } else {
            None
        };
        Expr {
            id: self.mint_id(),
            kind: ExprKind::If {
                cond: Box::new(cond),
                then,
                else_branch,
            },
            span,
        }
    }

    /// Parses `while cond { … }`.
    fn parse_while(&mut self) -> Expr {
        let start = self.span();
        self.bump();
        let cond = self.parse_expr(0);
        let body = self.parse_block();
        let span = join(start, body.span);
        Expr {
            id: self.mint_id(),
            kind: ExprKind::While {
                cond: Box::new(cond),
                body,
            },
            span,
        }
    }

    /// Parses `loop { … }`.
    fn parse_loop(&mut self) -> Expr {
        let start = self.span();
        self.bump();
        let body = self.parse_block();
        let span = join(start, body.span);
        Expr {
            id: self.mint_id(),
            kind: ExprKind::Loop { body },
            span,
        }
    }

    /// Parses `for pat in expr { … }`.
    fn parse_for(&mut self) -> Expr {
        let start = self.span();
        self.bump();
        let pat = self.parse_pat();
        if !self.eat_kw("in") {
            let span = self.span();
            self.error(span, String::from("expected `in`"));
        }
        let iter = self.parse_expr(0);
        let body = self.parse_block();
        let span = join(start, body.span);
        Expr {
            id: self.mint_id(),
            kind: ExprKind::For {
                pat,
                iter: Box::new(iter),
                body,
            },
            span,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_ocl_lex::tokenize;

    /// Builds a parser positioned at the start of `tokens`.
    fn parser_at<'a>(
        source: &'a str,
        tokens: impl Iterator<Item = Token> + 'a,
    ) -> Parser<'a, impl Iterator<Item = Token> + 'a> {
        Parser::new(source, tokens)
    }

    #[test]
    fn binding_power_ladder_orders_every_operator() {
        assert_eq!(binary_lbp(BinaryOp::Mul), 11);
        assert_eq!(binary_lbp(BinaryOp::Div), 11);
        assert_eq!(binary_lbp(BinaryOp::Add), 10);
        assert_eq!(binary_lbp(BinaryOp::Sub), 10);
        assert_eq!(binary_lbp(BinaryOp::Shl), 9);
        assert_eq!(binary_lbp(BinaryOp::Shr), 9);
        assert_eq!(binary_lbp(BinaryOp::BitAnd), 8);
        assert_eq!(binary_lbp(BinaryOp::BitXor), 7);
        assert_eq!(binary_lbp(BinaryOp::BitOr), 6);
        assert_eq!(binary_lbp(BinaryOp::Gt), 5);
        assert_eq!(binary_lbp(BinaryOp::And), 4);
        assert_eq!(binary_lbp(BinaryOp::Or), 3);
        assert!(binary_lbp(BinaryOp::Mul) > binary_lbp(BinaryOp::Add));
        assert!(binary_lbp(BinaryOp::Add) > binary_lbp(BinaryOp::Shl));
        assert!(binary_lbp(BinaryOp::BitAnd) > binary_lbp(BinaryOp::BitOr));
        assert!(binary_lbp(BinaryOp::And) > binary_lbp(BinaryOp::Or));
        const {
            assert!(BP_ASSIGN < BP_RANGE);
            assert!(BP_RANGE < BP_RANGE_OPERAND);
            assert!(BP_RANGE_OPERAND < BP_CAST);
            assert!(BP_CAST < BP_UNARY);
        }
    }

    #[test]
    fn comparisons_are_classified_for_non_associativity() {
        assert!(is_comparison(BinaryOp::Eq));
        assert!(is_comparison(BinaryOp::Ne));
        assert!(is_comparison(BinaryOp::Lt));
        assert!(is_comparison(BinaryOp::Le));
        assert!(is_comparison(BinaryOp::Gt));
        assert!(is_comparison(BinaryOp::Ge));
        assert!(!is_comparison(BinaryOp::Add));
        assert!(!is_comparison(BinaryOp::And));
    }

    #[test]
    fn single_character_tokens_map_to_binary_operators() {
        assert!(matches!(simple_binary(TokenKind::Plus), Some(BinaryOp::Add)));
        assert!(matches!(simple_binary(TokenKind::Minus), Some(BinaryOp::Sub)));
        assert!(matches!(simple_binary(TokenKind::Star), Some(BinaryOp::Mul)));
        assert!(matches!(compound_binary(TokenKind::Slash), Some(BinaryOp::Div)));
        assert!(matches!(compound_binary(TokenKind::Percent), Some(BinaryOp::Rem)));
        assert!(simple_binary(TokenKind::Whitespace).is_none());
        assert!(simple_binary(TokenKind::Eq).is_none());
        assert!(compound_binary(TokenKind::Bang).is_none());
    }

    #[test]
    fn terminators_end_value_expressions() {
        for kind in [
            TokenKind::Semi,
            TokenKind::Comma,
            TokenKind::CloseParen,
            TokenKind::CloseBracket,
            TokenKind::CloseBrace,
        ] {
            assert!(is_terminator(kind));
        }
        assert!(!is_terminator(TokenKind::Ident));
        assert!(!is_terminator(TokenKind::OpenBrace));
        assert!(!is_terminator(TokenKind::Plus));
    }

    #[test]
    fn trivia_is_filtered_from_the_parser_stream() {
        assert!(is_trivia(TokenKind::Whitespace));
        assert!(is_trivia(TokenKind::LineComment { doc_style: None }));
        assert!(is_trivia(TokenKind::BlockComment {
            doc_style: None,
            terminated: true
        }));
        assert!(!is_trivia(TokenKind::Ident));
        assert!(!is_trivia(TokenKind::Semi));

        let source = "fn f() { }";
        let mut cursor = Cursor::new(tokenize(source));
        let mut significant = Vec::new();
        while let Some(token) = cursor.bump() {
            assert!(!is_trivia(token.kind));
            significant.push(token);
        }
        assert_eq!(significant.len(), 6);
        assert!(cursor.at_end());
        let last_end = significant
            .last()
            .map_or(0, |token| token.span.end());
        assert_eq!(last_end as usize, source.len(), "trivia lengths must count");
    }

    #[test]
    fn join_spans_covers_both_ranges() {
        assert_eq!(join(Span::new(2, 3), Span::new(10, 4)), Span::new(2, 12));
        assert_eq!(join(Span::new(2, 3), Span::new(2, 3)), Span::new(2, 3));
        assert_eq!(join(Span::new(10, 4), Span::new(2, 3)), Span::new(10, 0));
    }

    #[test]
    fn joint_glue_requires_source_adjacency() {
        let mut parser = parser_at("<= ..=", tokenize("<= ..="));
        assert!(parser.at_joint(TokenKind::Lt, TokenKind::Eq));
        assert!(!parser.at_joint(TokenKind::Lt, TokenKind::Dot));
        // Step over `<=` to sit on the `..=` chain across the gap.
        assert!(parser.bump().is_some());
        assert!(parser.bump().is_some());
        assert!(parser.at_joint(TokenKind::Dot, TokenKind::Dot));
        assert!(parser.at_joint3(TokenKind::Dot, TokenKind::Dot, TokenKind::Eq));
        assert!(!parser.at_joint3(TokenKind::Lt, TokenKind::Eq, TokenKind::Dot));

        let spaced = parser_at("< =", tokenize("< ="));
        assert!(!spaced.at_joint(TokenKind::Lt, TokenKind::Eq));
    }

    #[test]
    fn recursion_limit_reports_once_and_stops() {
        let mut parser = parser_at("", core::iter::empty::<Token>());
        let mut entered = 0u32;
        while parser.enter() {
            entered += 1;
        }
        assert_eq!(entered, MAX_DEPTH);
        assert_eq!(parser.errors.len(), 1);
        assert_eq!(parser.errors[0].message, "recursion limit exceeded");
        assert!(!parser.enter());
        assert_eq!(parser.errors.len(), 1);
        parser.leave();
        assert!(parser.enter());
    }

    #[test]
    fn force_progress_advances_or_stops_at_eof() {
        let mut parser = parser_at("x;", tokenize("x;"));
        assert!(parser.force_progress(0));
        assert_eq!(parser.consumed(), 1);
        assert!(parser.force_progress(0));
        assert_eq!(parser.consumed(), 1);
        assert!(parser.force_progress(1));
        assert_eq!(parser.consumed(), 2);
        assert!(!parser.force_progress(2));
    }

    #[test]
    fn value_end_and_lifetime_probes_match_the_stream() {
        assert!(parser_at(";", tokenize(";")).at_value_end());

        let ident = parser_at("x", tokenize("x"));
        assert!(!ident.at_value_end());
        assert!(!ident.at_lifetime());
        assert!(parser_at("", core::iter::empty::<Token>()).at_value_end());

        assert!(parser_at("'a", tokenize("'a")).at_lifetime());
    }
}
