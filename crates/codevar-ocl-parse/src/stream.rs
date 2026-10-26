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

//! Item-at-a-time parsing for the streaming SPIR-V pipeline.
//!
//! [`ItemStream`] wraps a token iterator and yields one top-level
//! [`ItemOutcome`] per [`next_item`](ItemStream::next_item) call, so a
//! large kernel never has its token stream, token trees, or whole-file
//! AST materialized at once: the parser buffers four significant tokens
//! ahead and holds exactly one item at a time.
//!
//! ```
//! let source = "fn a() {} struct S { x: int }";
//! let mut stream = codevar_ocl_parse::ItemStream::new(source);
//! let mut kinds = 0;
//! while let Some(outcome) = stream.next_item() {
//!     assert!(outcome.errors.is_empty());
//!     kinds += 1;
//! }
//! assert_eq!(kinds, 2);
//! ```

use crate::ast::{Item, ItemKind};
use crate::parser::Parser;
use crate::{ParseError, Span};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use codevar_ocl_lex::{Token, tokenize};

/// The result of parsing one top-level item.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemOutcome {
    /// The parsed item; `ItemKind::Error` when recovery failed.
    pub item: Item,
    /// Diagnostics produced while parsing this item.
    pub errors: Vec<ParseError>,
}

/// A streaming, item-at-a-time parser over an in-memory source.
///
/// The stream keeps its source position between items while resetting
/// the per-item parser state, so [`next_item`](Self::next_item) never
/// re-lexes and never buffers more than a handful of tokens.
pub struct ItemStream<'a, I> {
    /// Parser owning the persistent cursor; `None` for streams built
    /// from a bare token iterator (see [`ItemStream::tokens`]).
    parser: Option<Parser<'a, I>>,
    /// Whether the missing-source diagnostic was already reported.
    reported_missing_source: bool,
}

impl<'a, I> ItemStream<'a, I>
where
    I: Iterator<Item = Token> + 'a,
{
    /// Wraps an existing token iterator whose token lengths cover the
    /// source in order, starting at byte offset 0.
    ///
    /// A [`Token`] carries only a kind and a byte length — the lexeme
    /// text lives in the source — so a stream built from tokens alone
    /// cannot recover identifier or keyword text. Such a stream reports
    /// that limitation once, as a single [`ItemKind::Error`] outcome
    /// with a diagnostic, and then ends; use [`new`](Self::new), which
    /// tokenizes the source itself, to parse items.
    pub fn tokens(tokens: I) -> Self {
        let _ = tokens;
        Self {
            parser: None,
            reported_missing_source: false,
        }
    }

    /// Parses the next top-level item, or `None` at end of input.
    ///
    /// The stream always advances or reaches EOF (no hangs): a failed
    /// item yields [`ItemKind::Error`] plus diagnostics. NodeIds restart
    /// from 0 for each item (per-item id spaces; consumers pair each
    /// item with per-item analysis tables).
    pub fn next_item(&mut self) -> Option<ItemOutcome> {
        if let Some(parser) = self.parser.as_mut() {
            return parser.parse_next_item();
        }
        if self.reported_missing_source {
            return None;
        }
        self.reported_missing_source = true;
        let span = Span::new(0, 0);
        Some(ItemOutcome {
            item: Item {
                attrs: Vec::new(),
                kind: ItemKind::Error,
                span,
            },
            errors: vec![ParseError {
                span,
                message: String::from(
                    "token stream carries no source text; use `ItemStream::new` to parse items",
                ),
            }],
        })
    }
}

impl<'a> ItemStream<'a, core::iter::Empty<Token>> {
    /// Convenience: an item stream over `tokenize(source)`.
    pub fn new(source: &'a str) -> ItemStream<'a, impl Iterator<Item = Token> + 'a> {
        ItemStream {
            parser: Some(Parser::new(source, tokenize(source))),
            reported_missing_source: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{ExprKind, TypeKind};
    use crate::{NodeId, Visitor, parse};
    use alloc::vec::Vec;

    /// Collects every `NodeId` of one item in visitor order.
    fn ids_of(item: &Item) -> Vec<NodeId> {
        #[derive(Default)]
        struct Collector {
            ids: Vec<NodeId>,
        }
        impl Visitor for Collector {
            fn visit_expr(&mut self, expr: &crate::ast::Expr) {
                self.ids.push(expr.id);
                crate::visit::walk_expr(self, expr);
            }
            fn visit_pat(&mut self, pat: &crate::ast::Pat) {
                self.ids.push(pat.id);
                crate::visit::walk_pat(self, pat);
            }
        }
        let mut collector = Collector::default();
        crate::visit::walk_item(&mut collector, item);
        collector.ids
    }

    /// Drains the stream into items and pooled diagnostics.
    fn drain<'s, S: Iterator<Item = Token> + 's>(
        stream: &mut ItemStream<'s, S>,
    ) -> (Vec<Item>, Vec<ParseError>) {
        let mut items = Vec::new();
        let mut errors = Vec::new();
        while let Some(outcome) = stream.next_item() {
            items.push(outcome.item);
            errors.extend(outcome.errors);
        }
        (items, errors)
    }

    #[test]
    fn yields_items_in_source_order_with_their_kinds() {
        let source = "#[kernel] fn a() {} fn b(x: int) -> int { x } struct S { a: int } type T = int";
        let mut stream = ItemStream::new(source);
        let first = stream.next_item().expect("first item");
        assert!(first.errors.is_empty(), "{:?}", first.errors);
        assert_eq!(first.item.attrs.len(), 1);
        let ItemKind::Fn(func) = &first.item.kind else {
            panic!("expected the attributed function, found {:?}", first.item.kind);
        };
        assert_eq!(func.name, "a");

        let second = stream.next_item().expect("second item");
        assert!(second.errors.is_empty(), "{:?}", second.errors);
        let ItemKind::Fn(func) = &second.item.kind else {
            panic!("expected the second function, found {:?}", second.item.kind);
        };
        assert_eq!(func.name, "b");
        assert_eq!(func.params.len(), 1);
        assert!(func.ret.is_some());

        let third = stream.next_item().expect("third item");
        assert!(third.errors.is_empty(), "{:?}", third.errors);
        let ItemKind::Struct(strukt) = &third.item.kind else {
            panic!("expected the struct, found {:?}", third.item.kind);
        };
        assert_eq!(strukt.name, "S");
        assert_eq!(strukt.fields.len(), 1);

        let fourth = stream.next_item().expect("fourth item");
        let ItemKind::TypeAlias(alias) = &fourth.item.kind else {
            panic!("expected the type alias, found {:?}", fourth.item.kind);
        };
        assert_eq!(alias.name, "T");
        assert!(matches!(alias.aliased.kind, TypeKind::Path(_)));

        assert!(stream.next_item().is_none(), "stream must reach EOF");
    }

    #[test]
    fn failed_item_reports_errors_and_the_stream_continues() {
        let source = "123 fn ok() {}";
        let mut stream = ItemStream::new(source);

        let failed = stream.next_item().expect("error item");
        assert!(matches!(failed.item.kind, ItemKind::Error));
        assert!(
            failed
                .errors
                .iter()
                .any(|error| error.message.contains("expected item")),
            "{:?}",
            failed.errors
        );

        let recovered = stream.next_item().expect("recovered item");
        assert!(recovered.errors.is_empty(), "{:?}", recovered.errors);
        let ItemKind::Fn(func) = &recovered.item.kind else {
            panic!(
                "expected the function after recovery, found {:?}",
                recovered.item.kind
            );
        };
        assert_eq!(func.name, "ok");
        assert!(stream.next_item().is_none(), "stream must reach EOF");
    }

    #[test]
    fn joint_operators_glue_through_the_stream() {
        let source = "fn f() -> int { a ..= b }";
        let mut stream = ItemStream::new(source);
        let outcome = stream.next_item().expect("item");
        assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
        let ItemKind::Fn(func) = &outcome.item.kind else {
            panic!("expected a function, found {:?}", outcome.item.kind);
        };
        let ret = func.ret.as_ref().expect("return type after `->`");
        assert!(matches!(ret.kind, TypeKind::Path(ref path) if path.segments[0].name == "int"));
        let tail = func
            .body
            .tail
            .as_deref()
            .expect("tail expression");
        assert!(
            matches!(tail.kind, ExprKind::Range { inclusive: true, .. }),
            "{:?}",
            tail.kind
        );

        // Split `- >` is *not* the `->` operator: joint detection is by
        // adjacent spans, and the whitespace between them breaks glue.
        let mut spaced = ItemStream::new("fn f() - > int {}");
        let outcome = spaced.next_item().expect("item");
        assert!(
            outcome
                .errors
                .iter()
                .any(|error| error.message.contains("`->`")),
            "{:?}",
            outcome.errors
        );
        let ItemKind::Fn(func) = &outcome.item.kind else {
            panic!("expected a function, found {:?}", outcome.item.kind);
        };
        assert!(func.ret.is_none());
    }

    #[test]
    fn items_and_errors_match_the_whole_file_parse() {
        let source = "#[kernel] fn a() {} fn b(x: int) -> int { x } \
                      struct S { a: int } type T = int; fn c() { let y = 1 }";
        let parsed = parse(source);
        let mut stream = ItemStream::new(source);
        let (items, errors) = drain(&mut stream);
        assert_eq!(items, parsed.program.items);
        assert_eq!(errors, parsed.errors);
        assert_eq!(errors.len(), 1, "{errors:?}");
    }

    #[test]
    fn node_ids_restart_for_every_item() {
        let source = "fn a(x: int) -> int { x } fn b(y: int) -> int { y }";
        let mut stream = ItemStream::new(source);
        let first = stream.next_item().expect("first item");
        let second = stream.next_item().expect("second item");
        let first_ids = ids_of(&first.item);
        let second_ids = ids_of(&second.item);
        assert!(!first_ids.is_empty(), "expected minted ids");
        assert_eq!(first_ids, second_ids, "per-item id spaces must restart");
        assert_eq!(
            first_ids.first().map(|id| id.as_raw()),
            Some(0),
            "the first item must start its id space at 0"
        );
    }

    #[test]
    fn trailing_trivia_ends_the_stream() {
        let mut stream = ItemStream::new("fn a() {} // trailing\n\n");
        assert!(stream.next_item().is_some());
        assert!(stream.next_item().is_none());
        assert!(stream.next_item().is_none(), "EOF must be stable");
    }

    #[test]
    fn token_only_stream_reports_missing_source_once() {
        let mut stream = ItemStream::tokens(tokenize("fn a() {}"));
        let outcome = stream.next_item().expect("diagnostic item");
        assert!(matches!(outcome.item.kind, ItemKind::Error));
        assert!(!outcome.errors.is_empty());
        assert!(stream.next_item().is_none());
    }
}
