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

//! Abstract syntax tree for Negex regular expressions.
//!
//! The tree follows the I-Regexp ABNF grammar of RFC 9485:
//!
//! ```abnf
//! i-regexp = branch *( "|" branch )
//! branch = *piece
//! piece = atom [ quantifier ]
//! quantifier = "*" / "+" / "?" / range-quantifier
//! range-quantifier = "{" QuantExact [ "," [ QuantExact ] ] "}"
//! QuantExact = 1*%x30-39 ; '0'-'9'
//! atom = NormalChar / charClass / ( "(" i-regexp ")" )
//! charClass = "." / SingleCharEsc / charClassEsc / charClassExpr
//! charClassEsc = catEsc / complEsc
//! charClassExpr = "[" [ "^" ] ( "-" / CCE1 ) *CCE1 [ "-" ] "]"
//! ```

use crate::Span;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// A complete regular expression: `i-regexp = branch *( "|" branch )`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Regex {
    /// The alternatives of this expression.  A plain concatenation has a
    /// single branch; `a|b` has two.  A match succeeds when any alternative
    /// matches.
    pub branches: Vec<Branch>,
    /// Span covering the whole pattern.
    pub span: Span,
}

impl Regex {
    /// Creates a regex with a single branch.
    #[must_use]
    pub fn new(branch: Branch, span: Span) -> Self {
        Self {
            branches: vec![branch],
            span,
        }
    }

    /// Creates a regex from two or more alternatives.
    #[must_use]
    pub fn alternation(branches: Vec<Branch>, span: Span) -> Self {
        Self { branches, span }
    }

    /// Whether this expression has more than one alternative.
    #[must_use]
    #[inline]
    pub fn is_alternation(&self) -> bool {
        self.branches.len() > 1
    }

    /// Whether this regex can match the empty string.
    #[must_use]
    pub fn matches_empty(&self) -> bool {
        self.branches.iter().any(Branch::matches_empty)
    }
}

/// A branch: one or more pieces concatenated, or empty (`branch = *piece`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Branch {
    /// The pieces in order.
    pub pieces: Vec<Piece>,
    /// Span covering the branch.
    pub span: Span,
}

impl Branch {
    /// Creates an empty branch (matches the empty string).
    #[must_use]
    pub fn empty(span: Span) -> Self {
        Self {
            pieces: Vec::new(),
            span,
        }
    }

    /// Whether this branch can match the empty string.
    #[must_use]
    pub fn matches_empty(&self) -> bool {
        self.pieces.is_empty() || self.pieces.iter().all(|p| p.matches_empty())
    }
}

/// A piece: an atom with an optional quantifier (`piece = atom [ quantifier ]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    /// The atom.
    pub atom: Atom,
    /// The optional quantifier.
    pub quantifier: Option<Quantifier>,
    /// Span covering the piece.
    pub span: Span,
}

impl Piece {
    /// Whether this piece can match the empty string.
    #[must_use]
    pub fn matches_empty(&self) -> bool {
        match self.quantifier {
            None => self.atom.matches_empty(),
            Some(Quantifier::Star) | Some(Quantifier::Question) => true,
            Some(Quantifier::Range { min, .. }) => min == 0 || self.atom.matches_empty(),
            Some(Quantifier::Plus) => self.atom.matches_empty(),
        }
    }
}

/// An atom: `NormalChar / charClass / ( "(" i-regexp ")" )`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Atom {
    /// A literal character that matches itself.
    Literal(char),
    /// Any character except line terminators: `.`.
    Any,
    /// A character class.
    Class(Class),
    /// A grouped sub-expression: `(` i-regexp `)`.
    Group(Box<Regex>),
    /// A parse error at this position.
    Error {
        /// Span of the error.
        span: Span,
    },
}

impl Atom {
    /// Whether this atom can match the empty string.
    #[must_use]
    pub fn matches_empty(&self) -> bool {
        match self {
            Self::Group(regex) => regex.matches_empty(),
            Self::Error { .. } => false,
            _ => false,
        }
    }
}

/// A character class: `.` / SingleCharEsc / charClassEsc / charClassExpr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Class {
    /// A positive or negated character class expression: `[...]` / `[^...]`.
    Positive {
        /// The items in the class.
        items: Vec<ClassItem>,
        /// Whether the class is negated (`[^...]`).
        negated: bool,
    },
    /// A Unicode property escape: `\p{IsCategory}` / `\P{IsCategory}`.
    Property {
        /// The property category name (e.g. `L`, `Nd`).
        category: String,
        /// Whether the property is negated (`\P{...}`).
        negated: bool,
    },
    /// A parse error at this position.
    Error {
        /// Span of the error.
        span: Span,
    },
}

/// One item inside a character class expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassItem {
    /// A single literal character.
    Char(char),
    /// A range `start..=end` (inclusive).
    Range {
        /// First character of the range.
        start: char,
        /// Last character of the range.
        end: char,
    },
    /// A Unicode property escape inside a class.
    Property {
        /// The property category name (e.g. `L`, `Nd`).
        category: String,
        /// Whether the property is negated (`\P{...}`).
        negated: bool,
    },
    /// A parse error at this position.
    Error {
        /// Span of the error.
        span: Span,
    },
}

/// A quantifier applied to a piece: `*` / `+` / `?` / range-quantifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Quantifier {
    /// Zero or more: `*`.
    Star,
    /// One or more: `+`.
    Plus,
    /// Zero or one: `?`.
    Question,
    /// A bounded range: `{min,max}` / `{min}` / `{min,}`.
    Range {
        /// Minimum number of repetitions.
        min: u32,
        /// Maximum number of repetitions, or `None` for unbounded.
        max: Option<u32>,
    },
}

impl Quantifier {
    /// The minimum number of repetitions.
    #[must_use]
    pub const fn min(&self) -> u32 {
        match self {
            Self::Star | Self::Question => 0,
            Self::Plus => 1,
            Self::Range { min, .. } => *min,
        }
    }

    /// Whether the quantifier allows zero repetitions.
    #[must_use]
    pub const fn allows_zero(&self) -> bool {
        self.min() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantifier_min_values() {
        assert_eq!(Quantifier::Star.min(), 0);
        assert_eq!(Quantifier::Plus.min(), 1);
        assert_eq!(Quantifier::Question.min(), 0);
        assert_eq!(Quantifier::Range { min: 2, max: Some(4) }.min(), 2);
        assert_eq!(Quantifier::Range { min: 3, max: None }.min(), 3);
    }

    #[test]
    fn quantifier_allows_zero() {
        assert!(Quantifier::Star.allows_zero());
        assert!(Quantifier::Question.allows_zero());
        assert!(!Quantifier::Plus.allows_zero());
        assert!(Quantifier::Range { min: 0, max: Some(5) }.allows_zero());
        assert!(!Quantifier::Range { min: 1, max: Some(5) }.allows_zero());
    }

    #[test]
    fn empty_branch_matches_empty() {
        assert!(Branch::empty(Span::new(0, 0)).matches_empty());
    }

    #[test]
    fn star_piece_matches_empty() {
        let piece = Piece {
            atom: Atom::Literal('a'),
            quantifier: Some(Quantifier::Star),
            span: Span::new(0, 1),
        };
        assert!(piece.matches_empty());
    }

    #[test]
    fn plus_piece_does_not_match_empty() {
        let piece = Piece {
            atom: Atom::Literal('a'),
            quantifier: Some(Quantifier::Plus),
            span: Span::new(0, 1),
        };
        assert!(!piece.matches_empty());
    }
}
