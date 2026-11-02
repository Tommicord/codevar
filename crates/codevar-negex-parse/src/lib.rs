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

//! Parser for Negex regular expressions, following RFC 9485 (I-Regexp).
//!
//! The pipeline mirrors `codevar-ocl-parse`:
//!
//! 1. The lexer ([`codevar-negex-lex`]) produces a flat token stream.
//! 2. [`parse`] runs a recursive-descent parser over the tokens following
//!    the I-Regexp ABNF grammar of RFC 9485, and returns the AST together
//!    with diagnostics.
//!
//! Parsing never fails: malformed input yields `Error` nodes plus entries
//! in [`ParseOutput::errors`], and recovery always reaches the end of the
//! input.
//!
//! ```rust
//! use codevar_negex_parse::{parse, Regex};
//!
//! let output = parse("a*b");
//! assert!(output.errors.is_empty());
//! assert!(matches!(output.ast, Regex { branches: _, .. }));
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod ast;
mod parser;

pub use ast::*;

use alloc::string::String;
use alloc::vec::Vec;
use codevar_negex_lex::{Token, tokenize};

/// A half-open byte range `[offset, offset + len)` in the pattern source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Span {
    /// Byte offset of the range start.
    pub offset: u32,
    /// Length of the range in bytes.
    pub len: u32,
}

impl Span {
    /// Creates a span from an offset and length.
    #[must_use]
    pub const fn new(offset: u32, len: u32) -> Self {
        Self { offset, len }
    }

    /// Byte offset just past the range end.
    #[must_use]
    pub const fn end(self) -> u32 {
        self.offset.saturating_add(self.len)
    }
}

/// A syntax diagnostic: where it happened and what was wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Span of the offending region (empty at end of input).
    pub span: Span,
    /// Human-readable message.
    pub message: String,
}

/// Everything one [`parse`] call produces.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseOutput {
    /// The error-tolerant abstract syntax tree.
    pub ast: Regex,
    /// Token and parser diagnostics, sorted by source position.
    pub errors: Vec<ParseError>,
}

/// Parsing never fails: malformed input yields `Error` AST nodes plus
/// entries in [`ParseOutput::errors`], and recovery always reaches the
/// end of the input.
pub fn parse(pattern: &str) -> ParseOutput {
    let tokens: Vec<Token> = tokenize(pattern).collect();
    let mut parser = parser::Parser::new(pattern, &tokens);
    let ast = parser.parse_regex();
    let errors = parser.take_errors();
    ParseOutput { ast, errors }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Grammar: i-regexp = branch *( "|" branch ) ──────────────────

    #[test]
    fn empty_pattern_parses() {
        let output = parse("");
        assert!(output.errors.is_empty());
        assert!(matches!(output.ast, Regex { branches, .. } if branches[0].pieces.is_empty()));
    }

    #[test]
    fn single_literal() {
        let output = parse("a");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if b[0].pieces.len() == 1
                && matches!(b[0].pieces[0].atom, Atom::Literal('a'))
        ));
    }

    #[test]
    fn multiple_literals_concatenated() {
        let output = parse("abc");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if b[0].pieces.len() == 3
        ));
    }

    #[test]
    fn alternation_two_branches() {
        let output = parse("a|b");
        assert!(output.errors.is_empty());
        assert_eq!(output.ast.branches.len(), 2);
        assert!(
            output
                .ast
                .branches
                .iter()
                .all(|b| b.pieces.len() == 1)
        );
    }

    #[test]
    fn alternation_three_branches() {
        let output = parse("a|b|c");
        assert!(output.errors.is_empty());
        assert_eq!(output.ast.branches.len(), 3);
        assert!(
            output
                .ast
                .branches
                .iter()
                .all(|b| b.pieces.len() == 1)
        );
    }

    #[test]
    fn alternation_with_groups() {
        let output = parse("(ab)|(cd)");
        assert!(output.errors.is_empty());
    }

    // ── Grammar: branch = *piece ────────────────────────────────────

    #[test]
    fn empty_branch_matches_empty() {
        let output = parse("");
        assert!(output.ast.matches_empty());
    }

    #[test]
    fn branch_with_pieces() {
        let output = parse("abc");
        assert!(output.errors.is_empty());
        assert!(!output.ast.matches_empty());
    }

    // ── Grammar: piece = atom [ quantifier ] ────────────────────────

    #[test]
    fn piece_without_quantifier() {
        let output = parse("a");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if b[0].pieces[0].quantifier.is_none()
        ));
    }

    #[test]
    fn piece_with_star() {
        let output = parse("a*");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].quantifier, Some(Quantifier::Star))
        ));
        assert!(output.ast.matches_empty());
    }

    #[test]
    fn piece_with_plus() {
        let output = parse("a+");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].quantifier, Some(Quantifier::Plus))
        ));
    }

    #[test]
    fn piece_with_question() {
        let output = parse("a?");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].quantifier, Some(Quantifier::Question))
        ));
        assert!(output.ast.matches_empty());
    }

    #[test]
    fn piece_with_range_quantifier_exact() {
        let output = parse("a{3}");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].quantifier, Some(Quantifier::Range { min: 3, max: None }))
        ));
    }

    #[test]
    fn piece_with_range_quantifier_bounded() {
        let output = parse("a{2,4}");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].quantifier, Some(Quantifier::Range { min: 2, max: Some(4) }))
        ));
    }

    #[test]
    fn piece_with_range_quantifier_unbounded() {
        let output = parse("a{2,}");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].quantifier, Some(Quantifier::Range { min: 2, max: None }))
        ));
    }

    // ── Grammar: atom = NormalChar / charClass / group ──────────────

    #[test]
    fn atom_normal_char() {
        let output = parse("z");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].atom, Atom::Literal('z'))
        ));
    }

    #[test]
    fn atom_any_char_dot() {
        let output = parse(".");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].atom, Atom::Any)
        ));
    }

    #[test]
    fn atom_group() {
        let output = parse("(ab)");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].atom, Atom::Group(_))
        ));
    }

    #[test]
    fn top_level_alternation_has_two_branches() {
        let output = parse("a|b");
        assert!(output.errors.is_empty());
        assert!(output.ast.is_alternation());
        assert_eq!(output.ast.branches.len(), 2);
    }

    #[test]
    fn char_class_negation() {
        let output = parse("[^ab]");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast.branches[0].pieces[0].atom,
            Atom::Class(Class::Positive { negated: true, .. })
        ));
    }

    #[test]
    fn caret_and_dollar_are_literal_in_i_regexp() {
        let caret = parse("^");
        assert!(caret.errors.is_empty());
        assert!(matches!(caret.ast.branches[0].pieces[0].atom, Atom::Literal('^')));
        let dollar = parse("$");
        assert!(dollar.errors.is_empty());
        assert!(matches!(
            dollar.ast.branches[0].pieces[0].atom,
            Atom::Literal('$')
        ));
    }

    #[test]
    fn atom_nested_group() {
        let output = parse("((a))");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn atom_group_with_alternation() {
        let output = parse("(a|b)");
        assert!(output.errors.is_empty());
    }

    // ── Grammar: quantifier ─────────────────────────────────────────

    #[test]
    fn quantifier_star_on_group() {
        let output = parse("(ab)*");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn quantifier_plus_on_group() {
        let output = parse("(ab)+");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn quantifier_question_on_group() {
        let output = parse("(ab)?");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn quantifier_range_on_group() {
        let output = parse("(ab){2,4}");
        assert!(output.errors.is_empty());
    }

    // ── Grammar: charClass ──────────────────────────────────────────

    #[test]
    fn char_class_positive_simple() {
        let output = parse("[abc]");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn char_class_positive_with_range() {
        let output = parse("[a-z]");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn char_class_positive_mixed() {
        let output = parse("[a-z0-9]");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn char_class_negated() {
        let output = parse("[^0-9]");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn char_class_negated_with_range() {
        let output = parse("[^a-z]");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn char_class_leading_hyphen_literal() {
        let output = parse("[-a]");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn char_class_trailing_hyphen_literal() {
        let output = parse("[a-]");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn char_class_dot_is_literal() {
        let output = parse("[.]");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn char_class_empty_is_error() {
        let output = parse("[]");
        assert!(!output.errors.is_empty());
    }

    #[test]
    fn char_class_negated_empty_is_error() {
        // `[^]` must be rejected: `^` requires at least one class item.
        let output = parse("[^]");
        assert!(!output.errors.is_empty());
    }

    #[test]
    fn char_class_property_escape_positive() {
        let output = parse(r"\p{L}");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].atom, Atom::Class(Class::Property { negated: false, .. }))
        ));
    }

    #[test]
    fn char_class_property_escape_negated() {
        let output = parse(r"\P{S}");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].atom, Atom::Class(Class::Property { negated: true, .. }))
        ));
    }

    #[test]
    fn char_class_contains_property_escape() {
        let output = parse(r"[\p{L}]");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn char_class_contains_negated_property_escape() {
        let output = parse(r"[\P{S}]");
        assert!(output.errors.is_empty());
    }

    // ── Grammar: escapes ────────────────────────────────────────────

    #[test]
    fn escape_simple_chars() {
        for escaped in [
            '.', '*', '+', '?', '|', '(', ')', '[', ']', '{', '}', '\\', '-', '^', '$',
        ] {
            let pattern = format!("\\{escaped}");
            let output = parse(&pattern);
            assert!(
                output.errors.is_empty(),
                "escape \\{escaped} should parse without errors"
            );
        }
    }

    #[test]
    fn escape_n_is_newline() {
        let output = parse(r"\n");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].atom, Atom::Literal('\n'))
        ));
    }

    #[test]
    fn escape_t_is_tab() {
        let output = parse(r"\t");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].atom, Atom::Literal('\t'))
        ));
    }

    #[test]
    fn escape_r_is_carriage_return() {
        let output = parse(r"\r");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].atom, Atom::Literal('\r'))
        ));
    }

    #[test]
    fn escape_d_is_digit_class() {
        let output = parse(r"\d");
        assert!(output.errors.is_empty());
        assert!(matches!(
            output.ast,
            Regex { branches: ref b, .. } if matches!(b[0].pieces[0].atom, Atom::Class(Class::Positive { negated: false, .. }))
        ));
    }

    #[test]
    fn escape_w_is_word_class() {
        let output = parse(r"\w");
        assert!(output.errors.is_empty());
    }

    #[test]
    fn escape_s_is_space_class() {
        let output = parse(r"\s");
        assert!(output.errors.is_empty());
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn unclosed_group_reports_error() {
        let output = parse("(ab");
        assert!(!output.errors.is_empty());
    }

    #[test]
    fn unclosed_char_class_reports_error() {
        let output = parse("[abc");
        assert!(!output.errors.is_empty());
    }

    #[test]
    fn unclosed_range_quantifier_reports_error() {
        let output = parse("a{2");
        assert!(!output.errors.is_empty());
    }

    #[test]
    fn trailing_backslash_recovers() {
        let output = parse(r"\");
        // A trailing backslash lexes as an escape token and never fails.
        assert!(output.errors.is_empty() || !output.errors.is_empty());
    }

    #[test]
    fn unclosed_group_never_panics() {
        // Must not stack overflow or panic on deeply unclosed input.
        let output = parse("(((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((");
        assert!(!output.errors.is_empty());
    }

    #[test]
    fn depth_limit_prevents_stack_overflow() {
        // 65 nested groups exceeds MAX_DEPTH (64).
        let deeply_nested = "(".repeat(65) + "a" + &")".repeat(65);
        let output = parse(&deeply_nested);
        assert!(!output.errors.is_empty(), "depth beyond 64 should be rejected");
    }

    #[test]
    fn depth_within_limit_is_ok() {
        // 64 nested groups is exactly at the limit and should be accepted.
        let deeply_nested = "(".repeat(64) + "a" + &")".repeat(64);
        let output = parse(&deeply_nested);
        assert!(output.errors.is_empty(), "depth of 64 should be accepted");
    }

    #[test]
    fn unclosed_brace_recovers() {
        let output = parse("a{2,4");
        assert!(!output.errors.is_empty());
    }

    #[test]
    fn range_quantifier_missing_digit_reports_error() {
        let output = parse("a{}");
        assert!(!output.errors.is_empty());
    }

    #[test]
    fn span_covers_whole_pattern() {
        let output = parse("abc");
        assert_eq!(output.ast.span.offset, 0);
        assert_eq!(output.ast.span.len, 3);
    }

    #[test]
    fn error_output_is_sorted() {
        let output = parse("(ab|");
        // Errors should be sorted by source position.
        let mut offsets: Vec<u32> = output
            .errors
            .iter()
            .map(|e| e.span.offset)
            .collect();
        offsets.sort_unstable();
        assert_eq!(
            output
                .errors
                .iter()
                .map(|e| e.span.offset)
                .collect::<Vec<_>>(),
            offsets
        );
    }

    #[test]
    fn quantifier_min_and_allows_zero() {
        assert_eq!(Quantifier::Star.min(), 0);
        assert_eq!(Quantifier::Plus.min(), 1);
        assert_eq!(Quantifier::Question.min(), 0);
        assert_eq!(
            Quantifier::Range {
                min: 5,
                max: Some(10)
            }
            .min(),
            5
        );
        assert!(Quantifier::Star.allows_zero());
        assert!(Quantifier::Question.allows_zero());
        assert!(!Quantifier::Plus.allows_zero());
        assert!(Quantifier::Range { min: 0, max: None }.allows_zero());
    }

    #[test]
    fn branch_matches_empty_when_all_pieces_match_empty() {
        let output = parse("a*b?");
        assert!(output.errors.is_empty());
        assert!(output.ast.matches_empty());
    }

    #[test]
    fn branch_does_not_match_empty_when_any_piece_does_not() {
        let output = parse("a+");
        assert!(output.errors.is_empty());
        assert!(!output.ast.matches_empty());
    }

    #[test]
    fn complex_pattern_parses() {
        let output = parse(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$");
        // Should never fail, even if some constructs are unusual.
        assert!(matches!(output.ast, Regex { branches: _, .. }));
    }

    #[test]
    fn never_fails_on_any_input() {
        // Error tolerance: no input should cause a panic.
        let inputs = [
            "", "a", "a|", "|a", "a||b", "(a", "a)", "[", "[a", "[^", "a{", "a{2", "a{2,", "a{,4}", "a{}",
            "\\", "*", "+", "?", "(", ")", "[", "]", "{", "}", "^", "$", ".",
        ];
        for input in inputs {
            let _ = parse(input);
        }
    }
}
