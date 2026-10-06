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

//! Parser for the Codevar OpenCL dialect.
//!
//! The pipeline mirrors rustc's front end:
//!
//! 1. The lexer ([`codevar-ocl-lex`]) produces a flat, lossless token
//!    stream (trivia included).
//! 2. [`build_token_trees`] groups it into balanced delimiter trees while
//!    recording recovery diagnostics for unbalanced input.
//! 3. [`parse`] runs a Pratt expression parser plus recursive descent for
//!    items, statements, and types over the *significant* tokens, and
//!    returns the AST together with both error sets.
//! 4. [`optimize`] rewrites the AST in place — constant folding,
//!    algebraic simplification, dead-statement removal — following the
//!    design of GCC's and Clang's expression folders.
//!
//! ```
//! let output = codevar_ocl_parse::parse("fn add(a: i32, b: i32) -> i32 { a + b }");
//! assert!(output.errors.is_empty());
//! assert_eq!(output.program.items.len(), 1);
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod ast;
mod optimize;
mod parser;
mod token_tree;

#[cfg(test)]
mod tests;

pub use ast::*;
pub use optimize::{OptReport, optimize};
pub use token_tree::{Delimiter, TokenTree, TokenTreeKind, build_token_trees};

use alloc::string::String;
use alloc::vec::Vec;

/// A half-open byte range `[offset, offset + len)` in the source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Span {
    /// Byte offset of the range start.
    pub offset: u32,
    /// Length of the range in bytes.
    pub len: u32,
}

impl Span {
    /// Creates a span from an offset and length.
    pub const fn new(offset: u32, len: u32) -> Self {
        Self { offset, len }
    }

    /// Byte offset just past the range end.
    pub const fn end(self) -> u32 {
        self.offset.saturating_add(self.len)
    }
}

/// A syntax diagnostic: where it happened and what was wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Span of the offending region (empty at end of input).
    pub span: Span,
    /// Human-readable message, e.g. `` expected `;` ``.
    pub message: String,
}

/// Everything one [`parse`] call produces.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseOutput {
    /// The error-tolerant abstract syntax tree.
    pub program: Program,
    /// Lossless token trees covering every source byte.
    pub trees: Vec<TokenTree>,
    /// Token-tree and parser diagnostics, sorted by source position.
    pub errors: Vec<ParseError>,
}

/// Parses a source file into AST, token trees, and diagnostics.
///
/// Parsing never fails: malformed input yields `Error` nodes plus entries
/// in [`ParseOutput::errors`], and recovery always reaches the end of the
/// input.
pub fn parse(source: &str) -> ParseOutput {
    let tokens = token_tree::collect_tokens(source);
    let source_len = u32::try_from(source.len()).unwrap_or(u32::MAX);
    let (trees, mut errors) = token_tree::build_from_tokens(&tokens, source_len);
    let significant = parser::significant_tokens(&tokens);
    let (program, parse_errors) = parser::parse_program(source, &significant);
    errors.extend(parse_errors);
    errors.sort_by_key(|error| (error.span.offset, error.span.len));
    ParseOutput {
        program,
        trees,
        errors,
    }
}
