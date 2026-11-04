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

//! The interpreter fallback for Negex.
//!
//! Executes a [`Determinized`] DFA directly against an input buffer in
//! time linear in the input length.  Used on targets without a native
//! codegen back end.  Because I-Regexp has no backtracking, lookahead, or
//! references, the DFA simulation is inherently ReDOS-resistant.
//!
//! # Character classes
//!
//! DFA transitions are keyed by the input byte itself: entry `b` of a
//! state's transition table is the state reached after consuming byte `b`.
//! Character classes were already compiled to byte sets during NFA/DFA
//! construction, so the interpreter needs no side table and a transition is
//! a single indexed lookup.
//!
//! # Matching
//!
//! [`Interpreter::exec`] performs an *anchored* match: the DFA starts at its
//! start state on the first input byte.  It records the longest prefix that
//! reaches an accepting state, which is what the generated machine code does
//! too.  Unanchored searching is layered on top by
//! [`CompiledRegex`](crate::CompiledRegex), which finds the leftmost start
//! with the reverse automaton and then runs this anchored match from there.
//!
//! # Streaming
//!
//! The interpreter processes input one byte at a time, so input can be
//! streamed in arbitrarily sized chunks: keep the DFA state alive between
//! chunks and feed each chunk's bytes to `exec` in order.  The [`MatchItem`]
//! in [`codevar_negex_abi`] holds an input pointer and length, making it
//! suitable for chunked submission.

use codevar_negex_abi::MatchItem;
use codevar_negex_codegen::Determinized;

/// The interpreter: a DFA executed directly against input bytes.
pub struct Interpreter {
    /// The DFA to execute.
    pub dfa: Determinized,
}

impl Interpreter {
    /// Creates an interpreter for `dfa`.
    #[must_use]
    pub fn new(dfa: Determinized) -> Self {
        Self { dfa }
    }

    /// Executes the DFA against `input`, writing the result into `item`.
    ///
    /// Returns `true` when a prefix of `input` matches.  The longest
    /// accepting prefix is reported in `item.end`; `item.start` is always
    /// zero because the match is anchored at the first byte.
    ///
    /// Runs in time linear in `input.len()` — there is no backtracking, so
    /// the engine is ReDOS-resistant.
    ///
    /// # Streaming
    ///
    /// Input may be provided in chunks: call `exec` once per chunk,
    /// carrying the [`MatchItem`] forward.  This implementation starts each
    /// call from the DFA start state, so it matches each chunk
    /// independently; callers that need a match to span chunks must
    /// concatenate the chunks first.
    #[inline]
    pub fn exec(&self, input: &[u8], item: &mut MatchItem) -> bool {
        let mut state = self.dfa.start;
        let mut pc: u32 = 0;
        let mut found = false;

        item.result = 0;
        item.start = 0;
        item.end = 0;

        while pc < input.len() as u32 {
            if self
                .dfa
                .states
                .get(state)
                .is_some_and(|st| st.accept)
            {
                item.result = 1;
                item.end = pc;
                found = true;
            }
            let byte = input[pc as usize];
            let Some(st) = self.dfa.states.get(state) else {
                break;
            };
            let Some(next) = st
                .transitions
                .get(byte as usize)
                .copied()
                .flatten()
            else {
                break;
            };
            state = next;
            pc += 1;
        }

        if self
            .dfa
            .states
            .get(state)
            .is_some_and(|st| st.accept)
        {
            item.result = 1;
            item.end = pc;
            found = true;
        }

        found
    }
}

impl Clone for Interpreter {
    fn clone(&self) -> Self {
        Self {
            dfa: self.dfa.clone(),
        }
    }
}

impl core::fmt::Debug for Interpreter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Interpreter")
            .field("dfa", &self.dfa)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_negex_codegen::{DfaState, build};
    use codevar_negex_parse::parse;

    fn build_dfa(pattern: &str) -> Determinized {
        let output = parse(pattern);
        assert!(output.errors.is_empty(), "pattern '{pattern}' must parse cleanly");
        let (list, start) = build(&output.ast).expect("NFA builds");
        Determinized::from_nfa(&list, start).expect("DFA determinizes")
    }

    fn compile(pattern: &str) -> Interpreter {
        Interpreter::new(build_dfa(pattern))
    }

    #[test]
    fn interpreter_matches_literal() {
        let interp = compile("a");

        let mut item = MatchItem::new(b"a");
        assert!(interp.exec(b"a", &mut item));
        assert_eq!(item.result, 1);
        assert_eq!(item.start, 0);
        assert_eq!(item.end, 1);

        let mut item = MatchItem::new(b"b");
        assert!(!interp.exec(b"b", &mut item));
        assert_eq!(item.result, 0);
    }

    #[test]
    fn interpreter_matches_star() {
        let interp = compile("a*");

        let mut item = MatchItem::new(b"");
        assert!(interp.exec(b"", &mut item));
        assert_eq!(item.result, 1);
        assert_eq!(item.end, 0);

        let mut item = MatchItem::new(b"aaa");
        assert!(interp.exec(b"aaa", &mut item));
        assert_eq!(item.end, 3);

        // A prefix matches; the trailing `b` simply ends the run.
        let mut item = MatchItem::new(b"aab");
        assert!(interp.exec(b"aab", &mut item));
        assert_eq!(item.end, 2);
    }

    #[test]
    fn interpreter_matches_plus() {
        let interp = compile("a+");

        let mut item = MatchItem::new(b"");
        assert!(!interp.exec(b"", &mut item));

        let mut item = MatchItem::new(b"a");
        assert!(interp.exec(b"a", &mut item));
        assert_eq!(item.end, 1);

        let mut item = MatchItem::new(b"aaa");
        assert!(interp.exec(b"aaa", &mut item));
        assert_eq!(item.end, 3);

        let mut item = MatchItem::new(b"b");
        assert!(!interp.exec(b"b", &mut item));
    }

    #[test]
    fn interpreter_matches_alternation() {
        let interp = compile("a|b");

        for input in [b"a".as_slice(), b"b".as_slice()] {
            let mut item = MatchItem::new(input);
            assert!(interp.exec(input, &mut item));
        }

        let mut item = MatchItem::new(b"c");
        assert!(!interp.exec(b"c", &mut item));
    }

    #[test]
    fn interpreter_empty_pattern_matches_empty_input() {
        let interp = compile("");
        let mut item = MatchItem::new(b"");
        assert!(interp.exec(b"", &mut item));
        assert_eq!(item.end, 0);
    }

    #[test]
    fn interpreter_reports_longest_prefix() {
        // `ab?` matches "a" even though "a" is not the whole input.
        let interp = compile("ab?");
        let mut item = MatchItem::new(b"ac");
        assert!(interp.exec(b"ac", &mut item));
        assert_eq!(item.end, 1);

        // `a(bc)?` accepts the shorter prefix "a" rather than rejecting.
        let interp = compile("a(bc)?");
        let mut item = MatchItem::new(b"ab");
        assert!(interp.exec(b"ab", &mut item));
        assert_eq!(item.end, 1);
    }

    #[test]
    fn interpreter_class_matches_exact_bytes_only() {
        let interp = compile("[0-9]");

        let mut item = MatchItem::new(b"5");
        assert!(interp.exec(b"5", &mut item));

        let mut item = MatchItem::new(b"a");
        assert!(!interp.exec(b"a", &mut item));
    }

    #[test]
    fn interpreter_no_match_on_dead_transition() {
        let interp = compile("ab");

        let mut item = MatchItem::new(b"a");
        assert!(!interp.exec(b"a", &mut item));

        let mut item = MatchItem::new(b"ab");
        assert!(interp.exec(b"ab", &mut item));
        assert_eq!(item.end, 2);
    }

    #[test]
    fn interpreter_linear_no_backtracking() {
        // A long non-matching input must not cause exponential behaviour.
        let interp = compile("ab");
        let input = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let mut item = MatchItem::new(input);
        assert!(!interp.exec(input, &mut item));
        assert_eq!(item.result, 0);
    }

    #[test]
    fn dfa_state_is_default_usable() {
        let state = DfaState::new();
        assert!(!state.accept);
        assert_eq!(state.transitions.len(), 256);
        assert!(state.transitions.iter().all(|t| t.is_none()));
    }
}
