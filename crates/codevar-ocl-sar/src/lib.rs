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

//! Semantic analyzer for the Codevar OpenCL dialect.
//!
//! The analyzer is the stage between parsing and code generation, modeled
//! on rustc's `rustc_analysis` and Clang's `Sema`: it validates the source
//! text that the parser cannot, then reports everything it finds as
//! [`Diagnostic`]s instead of stopping at the first problem.
//!
//! Analysis runs in gated stages, exactly like `rustc`'s
//! `abort_if_errors`:
//!
//! 1. Unicode validation rejects control, bidirectional, invisible, and
//!    noncharacter code points.
//! 2. Lexical validation turns the lexer's error flags into diagnostics.
//! 3. Parsing produces an error-tolerant AST plus syntax errors.
//! 4. Only when the first three stages are error-free does the type
//!    checker run over the tree — one mistake never cascades.
//!
//! # Examples
//!
//! ```
//! use codevar_ocl_sar::{analyze, codes};
//!
//! let output = analyze("fn add(a: int, b: int) -> int { a + b }");
//! assert!(!output.has_errors());
//!
//! let broken = analyze("fn f() -> int { true }");
//! assert!(broken.has_errors());
//! assert!(broken.find_code(codes::MISMATCHED_TYPES).is_some());
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod analyzer;
mod builtins;
mod confusable;
mod diagnostic;
mod emit;
mod lexcheck;
mod literal;
mod tables;
mod types;
mod unicode;

#[cfg(test)]
mod tests;

use alloc::string::String;
use alloc::vec::Vec;

use codevar_ocl_parse::parse;

pub use analyzer::{DeclKind, Declaration};
pub use builtins::{Builtin, BuiltinKind, builtins, lookup_builtin_fn};
pub use codevar_ocl_parse::{NodeId, Span};
pub use confusable::confusable_skeleton;
pub use diagnostic::{Diagnostic, Label, MessageBuilder, Severity, codes};
pub use emit::{ColorChoice, emit_stderr, render};
pub use tables::{Res, ResolutionTable, TypeTable};
pub use types::{BuiltinType, Scalar, Ty, coerce, lookup_builtin, substitute, unify};

/// Everything one [`analyze`] call produces.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisOutput {
    /// Every diagnostic, sorted by source position; errors first on ties.
    pub diagnostics: Vec<Diagnostic>,
    /// Declarations the analyzer resolved, in source order.
    pub declarations: Vec<Declaration>,
    /// Type of every type-checked expression, keyed by [`NodeId`].
    ///
    /// The table describes the tree that was analyzed: [`analyze`] runs
    /// the analyzer directly on the parse output, so folding with
    /// [`optimize`](codevar_ocl_parse::optimize) afterward must not be
    /// applied to a tree whose tables are still consulted.
    pub types: TypeTable,
    /// Resolution of every resolved name use, keyed by [`NodeId`].
    pub resolutions: ResolutionTable,
}

impl AnalysisOutput {
    /// True when at least one diagnostic is an error.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }

    /// Number of error diagnostics.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_error())
            .count()
    }

    /// Number of warning diagnostics.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_warning())
            .count()
    }

    /// The first diagnostic carrying `code`, if any.
    #[must_use]
    pub fn find_code(&self, code: &str) -> Option<&Diagnostic> {
        self.diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == Some(code))
    }
}

/// Runs the full front end over `source` and returns every diagnostic.
///
/// The stages are gated: lexical or syntax errors suppress semantic
/// analysis so one broken token cannot produce a wall of type errors.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::{analyze, codes};
///
/// let output = analyze("fn f() { let x: int = true; }");
/// assert!(output.find_code(codes::MISMATCHED_TYPES).is_some());
/// ```
#[must_use]
pub fn analyze(source: &str) -> AnalysisOutput {
    let mut diagnostics = Vec::new();
    unicode::check(source, &mut diagnostics);
    lexcheck::check(source, &mut diagnostics);
    if diagnostics.iter().any(Diagnostic::is_error) {
        sort_diagnostics(&mut diagnostics);
        return AnalysisOutput {
            diagnostics,
            declarations: Vec::new(),
            types: TypeTable::new(),
            resolutions: ResolutionTable::new(),
        };
    }

    let parsed = parse(source);
    let syntax_failed = !parsed.errors.is_empty();
    for error in &parsed.errors {
        diagnostics.push(Diagnostic::error(error.span, error.message.clone()));
    }
    if syntax_failed {
        sort_diagnostics(&mut diagnostics);
        return AnalysisOutput {
            diagnostics,
            declarations: Vec::new(),
            types: TypeTable::new(),
            resolutions: ResolutionTable::new(),
        };
    }

    let analyzed = analyzer::run(source, &parsed.program);
    diagnostics.extend(analyzed.diagnostics);
    sort_diagnostics(&mut diagnostics);
    AnalysisOutput {
        diagnostics,
        declarations: analyzed.declarations,
        types: analyzed.types,
        resolutions: analyzed.resolutions,
    }
}

/// Runs [`analyze`] over raw bytes.
///
/// Input that is not valid UTF-8 is reported as a single fatal diagnostic
/// at the first invalid byte; the surrounding text is decoded lossily for
/// the excerpt so the caret still lands in the right column.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::{analyze_bytes, codes};
///
/// let mut bytes = Vec::from(&b"fn f() {}"[..]);
/// bytes.push(0xFF);
/// let output = analyze_bytes(&bytes);
/// assert_eq!(output.find_code(codes::INVALID_UTF8).is_some(), true);
/// ```
#[must_use]
pub fn analyze_bytes(bytes: &[u8]) -> AnalysisOutput {
    let source = match core::str::from_utf8(bytes) {
        Ok(source) => String::from(source),
        Err(error) => {
            let offset = error.valid_up_to() as u32;
            let diagnostic = Diagnostic::error(Span::new(offset, 1), "input is not valid UTF-8")
                .with_code(codes::INVALID_UTF8)
                .with_label(Span::new(offset, 1), "invalid UTF-8 starts here")
                .with_note("source files must be UTF-8 encoded");
            return AnalysisOutput {
                diagnostics: alloc::vec![diagnostic],
                declarations: Vec::new(),
                types: TypeTable::new(),
                resolutions: ResolutionTable::new(),
            };
        }
    };
    analyze(&source)
}

/// Orders diagnostics by position, errors before warnings on ties.
fn sort_diagnostics(diagnostics: &mut [Diagnostic]) {
    diagnostics.sort_by_key(|diagnostic| (diagnostic.span.offset, diagnostic.severity));
}
