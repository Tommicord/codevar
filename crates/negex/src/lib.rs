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

//! Negex: a high-performance regex engine compiling patterns to native
//! machine code.
//!
//! The public API compiles a regex string at run time into a native
//! machine-code blob and executes it through a custom trampoline (see
//! [`Trampoline`]).  An interpreter fallback ([`Interpreter`] via
//! [`CompiledRegex::exec`]) is available for targets without a native
//! back end.
//!
//! # Example
//!
//! ```no_run
//! use codevar_negex::{ExecError, Negex};
//!
//! let engine = Negex::new();
//! let pattern = r"[a-z]+";
//! let compiled = engine.compile(pattern).expect("compiles");
//! let result = compiled.exec(b"hello world").expect("exec runs");
//! if result.is_match() {
//!     println!("matched at {}..{}", result.start, result.end);
//! }
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod regex;
mod trampoline;

pub use regex::Interpreter;
pub use trampoline::{Trampoline, TrampolineError};

// Re-export the ABI match item so examples and dependents can name it.
pub use codevar_negex_abi::MatchItem;

use alloc::vec::Vec;
use codevar_logger::{log_debug, log_warn};
use codevar_negex_abi::{self as abi};
use codevar_negex_codegen::{Determinized, NegexError, build, lower_dfa};
use codevar_negex_native_codegen::{CodegenError, CompiledNative, compile_negex_match};
use codevar_negex_parse::parse;
use codevar_ocl_native_codegen::pipeline::CodegenOptions;
use codevar_ocl_tc::host_target;

/// The Negex engine: compiles and executes regular expressions.
pub struct Negex {
    /// Whether the host has a native codegen back end.
    has_native: bool,
}

impl Negex {
    /// Creates a new engine for the host target.
    #[must_use]
    pub fn new() -> Self {
        Self {
            has_native: host_target().is_ok(),
        }
    }

    /// Compiles `pattern` to a [`CompiledRegex`].
    ///
    /// # Errors
    ///
    /// Returns an error when the pattern does not parse, does not compile
    /// to a DFA, or cannot be lowered to native code.
    pub fn compile(&self, pattern: &str) -> Result<CompiledRegex, CompileError> {
        let output = parse(pattern);
        if !output.errors.is_empty() {
            log_warn!("pattern '{pattern}' has {} parse errors", output.errors.len());
            return Err(CompileError::Parse {
                errors: output.errors,
            });
        }

        let (nfa, start) = build(&output.ast).map_err(CompileError::Codegen)?;
        let dfa = Determinized::from_nfa(&nfa, start).map_err(CompileError::Codegen)?;
        let reverse = dfa.reverse().map_err(CompileError::Codegen)?;

        let module = lower_dfa(&dfa).map_err(CompileError::Codegen)?;

        let native = if self.has_native {
            if let Ok(target) = host_target() {
                let options = CodegenOptions::default();
                match compile_negex_match(&module, target.as_target(), &options) {
                    Ok(compiled) => Some(compiled),
                    Err(err) => {
                        log_warn!("native codegen failed for pattern '{pattern}': {err}");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // The generated code steps through the transition table held here;
        // the interpreter reads the DFA directly, so an interpreted compile
        // keeps no table.
        let trans_table = if native.is_some() {
            codevar_negex_codegen::transition_table(&dfa)
        } else {
            Vec::new()
        };

        Ok(CompiledRegex {
            dfa,
            reverse,
            native,
            trans_table,
        })
    }
}

impl Default for Negex {
    fn default() -> Self {
        Self::new()
    }
}

/// Why a regex could not be compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// The pattern did not parse.
    Parse {
        /// The parse errors.
        errors: Vec<codevar_negex_parse::ParseError>,
    },
    /// Lowering to IR failed.
    Codegen(NegexError),
    /// Native codegen failed.
    Native(CodegenError),
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Parse { errors } => {
                write!(f, "pattern did not parse ({} errors)", errors.len())
            }
            Self::Codegen(error) => write!(f, "codegen failed: {error}"),
            Self::Native(error) => write!(f, "native codegen failed: {error}"),
        }
    }
}

impl core::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Codegen(error) => Some(error),
            Self::Native(_) => None,
            Self::Parse { .. } => None,
        }
    }
}

/// Why a match could not be executed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecError {
    /// No native trampoline back end is available.
    NoTrampoline,
    /// No interpreter is available.
    NoInterpreter,
}

impl core::fmt::Display for ExecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoTrampoline => f.write_str("no native trampoline back end"),
            Self::NoInterpreter => f.write_str("no interpreter available"),
        }
    }
}

impl core::error::Error for ExecError {}

/// Result of a regex match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchResult {
    /// Whether the input matched.
    pub is_match: bool,
    /// Start offset of the match (valid when `is_match`).
    pub start: usize,
    /// End offset of the match (valid when `is_match`).
    pub end: usize,
}

impl MatchResult {
    /// Returns `true` if the match succeeded.
    #[must_use]
    #[inline]
    pub fn is_match(&self) -> bool {
        self.is_match
    }
}

/// A compiled regex, ready to execute against input.
#[derive(Debug, Clone)]
pub struct CompiledRegex {
    /// The DFA (used by the interpreter fallback).
    dfa: Determinized,
    /// The reverse DFA, used to find the leftmost start of a match.
    reverse: Determinized,
    /// The native machine code (when a host back end is available).
    native: Option<CompiledNative>,
    /// The DFA transition table the native code steps through.
    ///
    /// The generated matcher dereferences this through the
    /// [`MatchItem`](codevar_negex_abi::MatchItem) it is handed, so it must
    /// live as long as any match built from this regex.  Empty when no
    /// native code was produced: the interpreter needs no table.
    trans_table: Vec<u32>,
}

impl CompiledRegex {
    /// Returns the interpreter for direct DFA execution.
    ///
    /// This bypasses the native trampoline and runs the DFA directly in Rust.
    /// Useful for benchmarks, testing, or when native codegen is unavailable.
    #[must_use]
    pub fn interpreter(&self) -> Option<Interpreter> {
        Some(Interpreter::new(self.dfa.clone()))
    }

    /// Executes the compiled regex against `input`, returning a
    /// [`MatchResult`].
    ///
    /// Uses the native trampoline when available, otherwise the
    /// interpreter fallback.  If the native trampoline cannot be created
    /// or called, falls back to the interpreter.
    #[inline]
    pub fn exec(&self, input: &[u8]) -> Result<MatchResult, ExecError> {
        let Some(start) = self.find_start(input) else {
            return Ok(MatchResult {
                is_match: false,
                start: 0,
                end: 0,
            });
        };

        let tail = input.get(start..).unwrap_or(&[]);
        if let Some(native) = &self.native {
            match self.exec_native(tail, start, native) {
                Ok(result) => return Ok(result),
                Err(ExecError::NoTrampoline) => {
                    log_debug!("native trampoline unavailable, falling back to interpreter");
                }
                Err(e) => return Err(e),
            }
        }
        self.exec_interp(tail, start)
    }

    /// Executes `input` through the interpreter `interp`.
    ///
    /// Performs the same leftmost search as [`Self::exec`] and then runs
    /// the anchored match in `interp`, so a caller that keeps one
    /// [`Interpreter`] alive times interpreted matching on exactly the
    /// work the native trampoline is given.  [`Self::exec`] clones the DFA
    /// into a fresh interpreter every time it falls back; this does not.
    #[inline]
    pub fn exec_interpreted_with(
        &self,
        interp: &Interpreter,
        input: &[u8],
    ) -> Result<MatchResult, ExecError> {
        let Some(start) = self.find_start(input) else {
            return Ok(MatchResult {
                is_match: false,
                start: 0,
                end: 0,
            });
        };
        let tail = input.get(start..).unwrap_or(&[]);
        Ok(Self::interp_result(interp, tail, start))
    }

    /// Finds the leftmost offset at which a match can begin.
    ///
    /// Runs the reverse DFA over the input from the last byte backwards.
    /// A reverse state is accepting when it contains the forward start
    /// state, so the last accepting position seen while scanning right to
    /// left is the smallest start offset that has a match.  Scanning stops
    /// when the reverse automaton dies, because no earlier position can
    /// match either.  This is linear in the input length.
    #[inline]
    fn find_start(&self, input: &[u8]) -> Option<usize> {
        // A pattern with no accepting state can never match; skip the scan.
        if !self.dfa.states.iter().any(|state| state.accept) {
            return None;
        }

        let mut state = self.reverse.start;
        let mut best = self
            .reverse
            .states
            .get(state)
            .filter(|st| st.accept)
            .map(|_| input.len());

        for index in (0..input.len()).rev() {
            let byte = input[index] as usize;
            let Some(next) = self
                .reverse
                .states
                .get(state)
                .and_then(|st| st.transitions.get(byte).copied().flatten())
            else {
                break;
            };
            state = next;
            if self
                .reverse
                .states
                .get(state)
                .is_some_and(|st| st.accept)
            {
                best = Some(index);
            }
        }

        best
    }

    /// Executes via the native trampoline.
    ///
    /// `offset` is the absolute input offset of `input`, which starts at the
    /// match start found by [`Self::find_start`]; reported spans are shifted
    /// by it.
    fn exec_native(
        &self,
        input: &[u8],
        offset: usize,
        native: &CompiledNative,
    ) -> Result<MatchResult, ExecError> {
        let trampoline =
            Trampoline::new(&native.bytes, native.entry_offset).map_err(|_| ExecError::NoTrampoline)?;
        let mut item = abi::MatchItem::with_table(input, &self.trans_table);
        // SAFETY: The trampoline owns executable memory containing
        // position-independent machine code that follows the Negex ABI: a
        // single `*mut MatchItem` argument.  `item` is a valid, fully
        // initialized `MatchItem` whose `input` points at `input` and whose
        // `trans_table` points at `state_count * TRANS_TABLE_ROW` entries
        // owned by `self`, both live for the duration of the call.
        trampoline.call(&mut item);
        Ok(MatchResult {
            is_match: item.result != 0,
            start: offset + item.start as usize,
            end: offset + item.end as usize,
        })
    }

    /// Executes via the interpreter fallback.
    fn exec_interp(&self, input: &[u8], offset: usize) -> Result<MatchResult, ExecError> {
        let interp = Interpreter::new(self.dfa.clone());
        Ok(Self::interp_result(&interp, input, offset))
    }

    /// Runs the anchored interpreter and shifts the reported span by
    /// `offset`, the absolute input offset of `input`.
    fn interp_result(interp: &Interpreter, input: &[u8], offset: usize) -> MatchResult {
        let mut item = abi::MatchItem::new(input);
        let matched = interp.exec(input, &mut item);
        MatchResult {
            is_match: matched,
            start: offset + item.start as usize,
            end: offset + item.end as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_creates() {
        let _engine = Negex::new();
    }

    #[test]
    fn compile_rejects_bad_pattern() {
        let engine = Negex::new();
        assert!(engine.compile("[").is_err());
    }

    #[test]
    fn compile_literal_searches_anywhere() {
        let engine = Negex::new();
        let compiled = engine.compile("a").expect("'a' compiles");

        let result = compiled.exec(b"a").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (0, 1));

        let result = compiled.exec(b"xax").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (1, 2));

        let result = compiled.exec(b"b").expect("exec runs");
        assert!(!result.is_match());
    }

    #[test]
    fn compile_star() {
        let engine = Negex::new();
        let compiled = engine.compile("a*").expect("'a*' compiles");

        let result = compiled.exec(b"").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (0, 0));

        let result = compiled.exec(b"aaa").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (0, 3));

        // The leftmost match is the longest run of `a`.
        let result = compiled.exec(b"aab").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (0, 2));
    }

    #[test]
    fn compile_plus() {
        let engine = Negex::new();
        let compiled = engine.compile("a+").expect("'a+' compiles");

        let result = compiled.exec(b"").expect("exec runs");
        assert!(!result.is_match());

        let result = compiled.exec(b"a").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (0, 1));

        let result = compiled.exec(b"baaa").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (1, 4));
    }

    #[test]
    fn compile_alternation() {
        let engine = Negex::new();
        let compiled = engine
            .compile("cat|dog")
            .expect("'cat|dog' compiles");

        let result = compiled.exec(b"the cat sat").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (4, 7));

        let result = compiled.exec(b"a dog ran").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (2, 5));

        let result = compiled.exec(b"neither").expect("exec runs");
        assert!(!result.is_match());
    }

    #[test]
    fn leftmost_match_wins_over_earlier_alternative() {
        // `a|ab` on "ab": start 0 is leftmost, and the longest match at that
        // start is "ab".
        let engine = Negex::new();
        let compiled = engine.compile("a|ab").expect("'a|ab' compiles");
        let result = compiled.exec(b"ab").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (0, 2));
    }

    #[test]
    fn multiline_literal_search() {
        let engine = Negex::new();
        let compiled = engine.compile("world").expect("'world' compiles");
        let result = compiled.exec(b"hello world").expect("exec runs");
        assert_eq!((result.start, result.end), (6, 11));
    }

    #[test]
    fn compile_empty_pattern() {
        let engine = Negex::new();
        let compiled = engine.compile("").expect("'' compiles");
        let result = compiled.exec(b"abc").expect("exec runs");
        assert!(result.is_match());
        assert_eq!((result.start, result.end), (0, 0));
    }

    #[test]
    fn exec_returns_no_match() {
        let engine = Negex::new();
        let compiled = engine.compile("abc").expect("'abc' compiles");
        let result = compiled.exec(b"xyz").expect("exec runs");
        assert!(!result.is_match());
    }

    #[test]
    fn exec_empty_input_non_empty_pattern() {
        let engine = Negex::new();
        let compiled = engine.compile("a").expect("'a' compiles");
        let result = compiled.exec(b"").expect("exec runs");
        assert!(!result.is_match());
    }

    #[test]
    fn match_result_is_match_method() {
        let engine = Negex::new();
        let compiled = engine.compile("a").expect("'a' compiles");
        assert!(compiled.exec(b"a").expect("exec runs").is_match());
        assert!(!compiled.exec(b"b").expect("exec runs").is_match());
    }

    #[test]
    fn compile_error_is_debuggable() {
        let engine = Negex::new();
        let err = engine.compile("[").unwrap_err();
        let dbg = format!("{err:?}");
        assert!(!dbg.is_empty());
    }

    #[test]
    fn exec_error_is_debuggable() {
        assert_eq!(format!("{:?}", ExecError::NoTrampoline), "NoTrampoline");
        assert_eq!(format!("{:?}", ExecError::NoInterpreter), "NoInterpreter");
    }

    #[test]
    fn long_non_matching_input_is_linear() {
        let engine = Negex::new();
        let compiled = engine.compile("abc").expect("'abc' compiles");
        let input = vec![b'a'; 10_000];
        let result = compiled.exec(&input).expect("exec runs");
        assert!(!result.is_match());
    }

    #[test]
    fn native_and_interpreter_agree_on_a_long_run() {
        // A run longer than the unrolled prologue exercises the loop the
        // optimiser promotes into registers, where the compiled matcher and
        // the interpreter diverged.  `native.is_some()` keeps the comparison
        // honest: `exec` falls back to the interpreter when codegen fails,
        // which would make both sides agree trivially.
        let engine = Negex::new();
        // (pattern, the byte the scan runs over, the sample at the end).
        let cases = [
            ("a+", b'a', &b"ab"[..]),
            ("a*", b'a', &b"ab"[..]),
            (r"\w+", b' ', &b"variableName123"[..]),
        ];
        for (pattern, filler, sample) in cases {
            let mut input: Vec<u8> = vec![filler; 4_096];
            input.truncate(input.len().saturating_sub(sample.len()));
            input.extend_from_slice(sample);

            let compiled = engine.compile(pattern).expect("pattern compiles");
            assert!(compiled.native.is_some(), "'{pattern}' compiles natively");
            let interp = compiled
                .interpreter()
                .expect("interpreter available");
            let native = compiled.exec(&input).expect("exec runs");
            let interpreted = compiled
                .exec_interpreted_with(&interp, &input)
                .expect("exec runs");
            assert!(native.is_match(), "'{pattern}' matches its sample");
            assert_eq!(native, interpreted, "the executors disagree on '{pattern}'");
        }
    }

    #[test]
    fn a_pattern_without_native_code_still_matches() {
        // A wide-alternation DFA keeps more values live across its back edge
        // than the register file holds, so the allocator declines and `exec`
        // answers through the interpreter.  Which executor ran is not the
        // contract under test; the span is.
        let engine = Negex::new();
        // (pattern, filler byte, sample at the end, expected span).
        let cases = [
            ("cat|dog|bird|fish", b'x', &b"bird"[..], 4_092usize, 4_096),
            ("(a+)+b", b'a', &b"b"[..], 0, 4_096),
        ];
        for (pattern, filler, sample, start, end) in cases {
            let size = 4_096;
            let mut input: Vec<u8> = vec![filler; size];
            input.truncate(size - sample.len());
            input.extend_from_slice(sample);

            let compiled = engine.compile(pattern).expect("pattern compiles");
            let result = compiled.exec(&input).expect("exec runs");
            assert!(result.is_match(), "'{pattern}' matches its sample");
            assert_eq!(
                (result.start, result.end),
                (start, end),
                "'{pattern}' matches the wrong span"
            );
        }
    }

    #[test]
    fn trampoline_new_rejects_bad_entry_offset() {
        let error = Trampoline::new(&[], 0).expect_err("empty code");
        assert!(matches!(error, TrampolineError::BadEntryOffset { .. }));
    }
}
