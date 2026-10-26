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

//! Lossless token trees: balanced delimiter groups over the raw token
//! stream, with trivia preserved as leaves.
//!
//! The builder mirrors rustc's `tokenstream` stage: every source byte
//! belongs to exactly one tree node, groups are nested by bracket type, and
//! unbalanced input is repaired with [`ParseError`] diagnostics instead of
//! aborting.

use crate::{ParseError, Span};
use alloc::format;
use alloc::vec::Vec;
use codevar_ocl_lex::{Token, TokenKind, tokenize};

/// Delimiter pair of a [`TokenTreeKind::Group`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delimiter {
    /// `(` … `)`
    Parenthesis,
    /// `{` … `}`
    Brace,
    /// `[` … `]`
    Bracket,
}

/// One node of the lossless token tree.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenTree {
    /// Span of this node, including any delimiters it owns.
    pub span: Span,
    /// Leaf token or nested group.
    pub kind: TokenTreeKind,
}

/// Payload of a [`TokenTree`].
#[derive(Debug, Clone, PartialEq)]
pub enum TokenTreeKind {
    /// A single source token, trivia included.
    Leaf {
        /// The lexed token classification.
        kind: TokenKind,
    },
    /// A balanced `open children close` region.
    Group {
        /// Which delimiter pair this group uses.
        delimiter: Delimiter,
        /// Span of the opening delimiter.
        open: Span,
        /// Trees between the delimiters.
        children: Vec<TokenTree>,
        /// Span of the closing delimiter; `None` when input ended first.
        close: Option<Span>,
    },
}

/// Collects `(kind, span)` pairs covering every source byte in order.
pub(crate) fn collect_tokens(source: &str) -> Vec<(TokenKind, Span)> {
    let mut offset = 0u32;
    tokenize(source)
        .map(|token: Token| {
            let span = Span::new(offset, token.len);
            offset = offset.saturating_add(token.len);
            (token.kind, span)
        })
        .collect()
}

/// Builds token trees over pre-collected tokens.
pub(crate) fn build_from_tokens(
    tokens: &[(TokenKind, Span)],
    source_len: u32,
) -> (Vec<TokenTree>, Vec<ParseError>) {
    let mut builder = Builder {
        tokens,
        index: 0,
        source_len,
        errors: Vec::new(),
    };
    let (trees, _) = builder.parse_children(None);
    (trees, builder.errors)
}

/// Builds the lossless token tree for a source file.
///
/// Returns the top-level trees (covering the whole file) and recovery
/// diagnostics for unbalanced or mismatched delimiters.
pub fn build_token_trees(source: &str) -> (Vec<TokenTree>, Vec<ParseError>) {
    let tokens = collect_tokens(source);
    build_from_tokens(&tokens, source.len() as u32)
}

/// Maps an opening delimiter to its expected closer.
fn expected_closer(kind: TokenKind) -> Option<TokenKind> {
    match kind {
        TokenKind::OpenParen => Some(TokenKind::CloseParen),
        TokenKind::OpenBrace => Some(TokenKind::CloseBrace),
        TokenKind::OpenBracket => Some(TokenKind::CloseBracket),
        _ => None,
    }
}

/// Whether the token closes some delimiter.
fn is_closer(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::CloseParen | TokenKind::CloseBrace | TokenKind::CloseBracket
    )
}

/// Stable display text for a delimiter token, e.g. `)` .
fn delimiter_text(kind: TokenKind) -> &'static str {
    match kind {
        TokenKind::OpenParen | TokenKind::CloseParen => ")",
        TokenKind::OpenBrace | TokenKind::CloseBrace => "}",
        TokenKind::OpenBracket | TokenKind::CloseBracket => "]",
        _ => "?",
    }
}

/// Which [`Delimiter`] a token opens.
fn delimiter_of(kind: TokenKind) -> Delimiter {
    match kind {
        TokenKind::OpenBrace => Delimiter::Brace,
        TokenKind::OpenBracket => Delimiter::Bracket,
        _ => Delimiter::Parenthesis,
    }
}

/// Recursive builder state.
struct Builder<'a> {
    /// All tokens with computed spans.
    tokens: &'a [(TokenKind, Span)],
    /// Index of the next unconsumed token.
    index: usize,
    /// Length of the source, for spans at end of input.
    source_len: u32,
    /// Recovery diagnostics collected so far.
    errors: Vec<ParseError>,
}

impl Builder<'_> {
    /// Parses trees until EOF, or until `closer` (if given) is consumed.
    ///
    /// Returns the children plus the span of the consumed closer (`None`
    /// at EOF). When a closer is expected, it is consumed here; otherwise
    /// the trees reach end of input.
    fn parse_children(&mut self, closer: Option<TokenKind>) -> (Vec<TokenTree>, Option<Span>) {
        let mut children = Vec::new();
        while let Some((kind, span)) = self.peek() {
            if is_closer(kind) {
                if Some(kind) == closer {
                    self.index += 1;
                    return (children, Some(span));
                }
                if closer.is_some() {
                    self.errors.push(ParseError {
                        span,
                        message: format!(
                            "expected `{}` before `{}`",
                            closer.map_or("?", delimiter_text),
                            delimiter_text(kind)
                        ),
                    });
                } else {
                    self.errors.push(ParseError {
                        span,
                        message: format!("mismatched closing delimiter `{}`", delimiter_text(kind)),
                    });
                }
                self.index += 1;
                children.push(TokenTree {
                    span,
                    kind: TokenTreeKind::Leaf { kind },
                });
                continue;
            }
            if let Some(expected) = expected_closer(kind) {
                self.index += 1;
                let (inner, close) = self.parse_children(Some(expected));
                let group_span = Span::new(
                    span.offset,
                    close.map_or_else(
                        || self.source_len.saturating_sub(span.offset),
                        |c| c.end().saturating_sub(span.offset),
                    ),
                );
                children.push(TokenTree {
                    span: group_span,
                    kind: TokenTreeKind::Group {
                        delimiter: delimiter_of(kind),
                        open: span,
                        children: inner,
                        close,
                    },
                });
                continue;
            }
            self.index += 1;
            children.push(TokenTree {
                span,
                kind: TokenTreeKind::Leaf { kind },
            });
        }
        if let Some(expected) = closer {
            self.errors.push(ParseError {
                span: self.end_span(),
                message: format!("unclosed delimiter, expected `{}`", delimiter_text(expected)),
            });
        }
        (children, None)
    }

    /// Peeks the current token.
    fn peek(&self) -> Option<(TokenKind, Span)> {
        self.tokens.get(self.index).copied()
    }

    /// Span at end of input, for diagnostics past the last token.
    fn end_span(&self) -> Span {
        Span::new(self.source_len, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_delimiters_map_to_their_closers() {
        assert_eq!(expected_closer(TokenKind::OpenParen), Some(TokenKind::CloseParen));
        assert_eq!(expected_closer(TokenKind::OpenBrace), Some(TokenKind::CloseBrace));
        assert_eq!(
            expected_closer(TokenKind::OpenBracket),
            Some(TokenKind::CloseBracket)
        );
        assert_eq!(expected_closer(TokenKind::Ident), None);
        assert_eq!(expected_closer(TokenKind::CloseParen), None);
    }

    #[test]
    fn closer_classification_covers_every_delimiter_token() {
        assert!(is_closer(TokenKind::CloseParen));
        assert!(is_closer(TokenKind::CloseBrace));
        assert!(is_closer(TokenKind::CloseBracket));
        assert!(!is_closer(TokenKind::OpenParen));
        assert!(!is_closer(TokenKind::OpenBrace));
        assert!(!is_closer(TokenKind::Ident));
    }

    #[test]
    fn delimiter_text_reports_the_closing_glyph() {
        assert_eq!(delimiter_text(TokenKind::OpenParen), ")");
        assert_eq!(delimiter_text(TokenKind::CloseParen), ")");
        assert_eq!(delimiter_text(TokenKind::OpenBrace), "}");
        assert_eq!(delimiter_text(TokenKind::CloseBracket), "]");
        assert_eq!(delimiter_text(TokenKind::Ident), "?");
    }

    #[test]
    fn delimiter_of_maps_every_opener() {
        assert_eq!(delimiter_of(TokenKind::OpenParen), Delimiter::Parenthesis);
        assert_eq!(delimiter_of(TokenKind::OpenBrace), Delimiter::Brace);
        assert_eq!(delimiter_of(TokenKind::OpenBracket), Delimiter::Bracket);
        assert_eq!(delimiter_of(TokenKind::Ident), Delimiter::Parenthesis);
    }

    #[test]
    fn nested_groups_balance_by_delimiter_kind() {
        let (trees, errors) = build_token_trees("(a[b]c)");
        assert!(errors.is_empty());
        let [tree] = trees.as_slice() else {
            panic!("expected a single top-level group");
        };
        assert_eq!(tree.span, Span::new(0, 7));
        let TokenTreeKind::Group {
            delimiter,
            open,
            children,
            close,
        } = &tree.kind
        else {
            panic!("expected a group");
        };
        assert_eq!(*delimiter, Delimiter::Parenthesis);
        assert_eq!(*open, Span::new(0, 1));
        assert_eq!(*close, Some(Span::new(6, 1)));
        assert_eq!(children.len(), 3);
        let TokenTreeKind::Group {
            delimiter,
            children,
            close,
            ..
        } = &children[1].kind
        else {
            panic!("expected a nested bracket group");
        };
        assert_eq!(*delimiter, Delimiter::Bracket);
        assert_eq!(*close, Some(Span::new(4, 1)));
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn mismatched_and_unclosed_delimiters_recover_with_diagnostics() {
        let (trees, errors) = build_token_trees("([)");
        assert_eq!(errors.len(), 3);
        assert_eq!(errors[0].message, "expected `]` before `)`");
        assert_eq!(errors[1].message, "unclosed delimiter, expected `]`");
        assert_eq!(errors[2].message, "unclosed delimiter, expected `)`");

        let [tree] = trees.as_slice() else {
            panic!("expected a single top-level group");
        };
        let TokenTreeKind::Group { children, close, .. } = &tree.kind else {
            panic!("expected a group");
        };
        assert_eq!(*close, None);
        assert_eq!(children.len(), 1);
        let TokenTreeKind::Group {
            children: inner,
            close: inner_close,
            ..
        } = &children[0].kind
        else {
            panic!("expected the recovered bracket group");
        };
        assert_eq!(*inner_close, None);
        assert_eq!(inner.len(), 1);
    }
}
