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

//! Errors produced while building and lowering a regex automaton.
//!
//! The crate never reports failure through a bare `String`: every fallible
//! operation returns a [`NegexError`], which carries the structured context
//! a caller needs to react to the failure (which quantifier was bad, how
//! far construction got before the state cap, which builder operation
//! rejected the module) and chains the underlying [`BuildError`] as its
//! source.

use alloc::string::String;
use codevar_ocl_ir::ir::BuildError;
use core::fmt;

/// A failure while constructing an NFA/DFA or lowering it to IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegexError {
    /// A range quantifier is malformed: its minimum exceeds its maximum, or
    /// it asks for more than [`crate::MAX_RANGE_EXPANSION`] copies.
    InvalidRangeQuantifier {
        /// The requested minimum repetition count.
        min: u32,
        /// The requested maximum repetition count, or `None` for `{min,}`.
        max: Option<u32>,
    },
    /// Determinization produced more than [`crate::MAX_DFA_STATES`] states.
    ///
    /// The cap keeps memory bounded and prevents a pathological pattern from
    /// exhausting the allocator; I-Regexp patterns are expected to stay far
    /// below it.
    TooManyStates {
        /// Number of states produced when construction was aborted.
        states: usize,
        /// The configured cap.
        limit: usize,
    },
    /// [`lower_dfa`](crate::lower_dfa) was called with a DFA that has no
    /// states, so there is no start state to lower.
    EmptyDfa,
    /// The IR builder rejected an operation.
    ///
    /// This is an internal invariant failure: the lowering code builds a
    /// module of a fixed shape, so a rejection means the builder contract
    /// was violated (for example, emitting into a function without a body).
    Module {
        /// The builder entry point that failed.
        operation: &'static str,
        /// The underlying builder error.
        source: BuildError,
    },
    /// The match function was created without its single `*mut MatchItem`
    /// argument.
    ///
    /// This is an internal invariant failure: the signature is built by the
    /// same function that later reads the argument.
    MissingMatchArgument,
    /// A DFA transition has no lowering block.
    ///
    /// This is an internal invariant failure: the block list and the
    /// transition list are produced from the same iteration.
    MissingMatchBlock {
        /// The DFA state that owns the transition.
        state: usize,
        /// The target state of the transition.
        target: usize,
    },
    /// A character class contains a scalar value above `0xFF`.
    ///
    /// Negex matches bytes: a class range is a set of byte values, so a
    /// range end above U+00FF cannot be represented.  Non-ASCII *literals*
    /// are still supported because they lower to their UTF-8 byte sequence;
    /// only class ranges are byte-bounded.
    ClassOutOfByteRange {
        /// The offending scalar value.
        scalar: char,
    },
    /// A character class range is backwards (`start > end`).
    InvalidClassRange {
        /// First character of the range.
        start: char,
        /// Last character of the range.
        end: char,
    },
    /// A `\p{...}` category has no byte-level definition in this engine.
    UnknownProperty {
        /// The requested category name.
        category: String,
    },
    /// A negated class contains non-ASCII scalars, which the byte-level
    /// complement cannot express exactly.
    UnsupportedNegatedClass,
    /// A parse-error placeholder reached code generation.
    ///
    /// The parser attaches [`Atom::Error`](codevar_negex_parse::Atom::Error)
    /// nodes so recovery can continue; compilation refuses to lower them.
    InvalidClassAst,
}

impl fmt::Display for NegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRangeQuantifier { min, max } => match max {
                Some(max) => write!(
                    f,
                    "invalid range quantifier {{{min},{max}}}: minimum exceeds maximum or \
                     exceeds the expansion cap of {}",
                    crate::MAX_RANGE_EXPANSION
                ),
                None => write!(
                    f,
                    "invalid range quantifier {{{min},}}: minimum exceeds the expansion cap of {}",
                    crate::MAX_RANGE_EXPANSION
                ),
            },
            Self::TooManyStates { states, limit } => {
                write!(f, "DFA exceeded its {limit}-state cap (produced {states} states)")
            }
            Self::EmptyDfa => f.write_str("cannot lower an empty DFA"),
            Self::Module { operation, source } => {
                write!(f, "IR builder rejected `{operation}`: {source}")
            }
            Self::MissingMatchArgument => f.write_str("match function has no argument 0"),
            Self::MissingMatchBlock { state, target } => {
                write!(f, "no lowering block for state {state} -> {target}")
            }
            Self::ClassOutOfByteRange { scalar } => write!(
                f,
                "character U+{:04X} is outside the engine's byte range U+0000-U+00FF",
                *scalar as u32
            ),
            Self::InvalidClassRange { start, end } => write!(
                f,
                "character class range U+{:04X}-U+{:04X} is backwards",
                *start as u32, *end as u32
            ),
            Self::UnknownProperty { category } => {
                write!(f, "unknown Unicode property category `{category}`")
            }
            Self::UnsupportedNegatedClass => {
                f.write_str("negated character class contains a non-ASCII scalar")
            }
            Self::InvalidClassAst => f.write_str("parse-error placeholder reached code generation"),
        }
    }
}

impl core::error::Error for NegexError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Module { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_quantifier_message_includes_bounds() {
        let error = NegexError::InvalidRangeQuantifier { min: 5, max: Some(2) };
        let message = alloc::format!("{error}");
        assert!(message.contains("{5,2}"), "{message}");
    }

    #[test]
    fn open_ended_range_message_names_the_cap() {
        let error = NegexError::InvalidRangeQuantifier { min: 99, max: None };
        let message = alloc::format!("{error}");
        assert!(message.contains("{99,}"), "{message}");
    }

    #[test]
    fn state_cap_message_reports_both_numbers() {
        let error = NegexError::TooManyStates {
            states: 5000,
            limit: crate::MAX_DFA_STATES,
        };
        let message = alloc::format!("{error}");
        assert!(message.contains("5000"), "{message}");
        assert!(message.contains("4096"), "{message}");
    }

    #[test]
    fn builder_error_is_exposed_as_source() {
        use core::error::Error;

        let error = NegexError::Module {
            operation: "push_block",
            source: BuildError::MissingBody,
        };
        assert!(
            Error::source(&error).is_some(),
            "builder error must chain as a source"
        );
        let message = alloc::format!("{error}");
        assert!(message.contains("push_block"), "{message}");
        assert!(message.contains("body"), "{message}");
    }
}
