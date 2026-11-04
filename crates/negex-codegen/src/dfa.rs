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

//! Subset construction: determinizes the Thompson NFA into a DFA.
//!
//! Each DFA state is a set of NFA states (epsilon-closure of a read
//! state's `out` targets plus the state itself).  Transitions are keyed by
//! input byte.  Because I-Regexp has no backtracking, the DFA matches in
//! time linear in the input length — the engine is inherently ReDOS-resistant.
//!
//! # ReDOS prevention
//!
//! I-Regexp (RFC 9485) has no backreferences, lookahead, or general
//! backtracking.  Subset construction therefore produces a DFA whose
//! matching cost is `O(input length)` regardless of the pattern's
//! quantifier nesting.  The number of DFA states is capped at
//! [`MAX_DFA_STATES`] and range quantifier expansion is capped at
//! [`super::MAX_RANGE_EXPANSION`], so construction is bounded too.

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use codevar_logger::log_debug;

use super::{DfaState, MAX_DFA_STATES, NfaState, StateList};
use crate::NegexError;

/// The result of determinizing an NFA: a DFA plus its start state.
#[derive(Debug, Clone)]
pub struct Determinized {
    /// The DFA states.
    pub states: Vec<DfaState>,
    /// The start state index.
    pub start: usize,
    /// Whether the empty string is accepted (start state's closure
    /// contains a match state).
    pub matches_empty: bool,
}

impl Determinized {
    /// Builds a DFA from the NFA state list, keying transitions by input
    /// byte.
    ///
    /// # Errors
    ///
    /// Returns an error when the DFA would exceed [`MAX_DFA_STATES`]
    /// states, keeping memory bounded for embedded use.
    pub fn from_nfa(list: &StateList, start: usize) -> Result<Self, NegexError> {
        let nfa_states = list.as_slice();
        log_debug!("dfa: starting subset construction, start={start}");

        // Epsilon-closure of a single state.
        let closure = |state: usize, visited: &mut BTreeSet<usize>| {
            let mut stack = vec![state];
            while let Some(s) = stack.pop() {
                if !visited.insert(s) {
                    continue;
                }
                match nfa_states[s] {
                    NfaState::Split { out, out1 } => {
                        stack.push(out);
                        stack.push(out1);
                    }
                    // A read state's `out` is reached only *after* consuming
                    // its character class, so it is not an epsilon edge and
                    // must not be followed here.
                    NfaState::Read { .. } => {}
                    NfaState::Match => {}
                }
            }
        };

        // Maps a frozenset of NFA states to a DFA state index.
        let mut state_ids: BTreeMap<BTreeSet<usize>, usize> = BTreeMap::new();
        let mut dfa_states: Vec<DfaState> = Vec::new();
        let mut work: VecDeque<BTreeSet<usize>> = VecDeque::new();

        let mut start_set = BTreeSet::new();
        closure(start, &mut start_set);
        let matches_empty = start_set.iter().any(|&s| {
            nfa_states
                .get(s)
                .map(NfaState::is_match)
                .unwrap_or(false)
        });
        log_debug!(
            "dfa: start closure has {} states, matches_empty={}",
            start_set.len(),
            matches_empty
        );
        let start_id = new_state(&start_set, nfa_states, &mut state_ids, &mut dfa_states);
        work.push_back(start_set);

        while let Some(set) = work.pop_front() {
            let id = state_ids[&set];
            // For each byte in the alphabet, compute the move.
            for byte in 0..=255u16 {
                let byte = byte as u8;
                let mut next = BTreeSet::new();
                for &s in &set {
                    if let Some(NfaState::Read { class, out }) = nfa_states.get(s)
                        && class.contains(byte)
                    {
                        closure(*out, &mut next);
                    }
                }
                if next.is_empty() {
                    // Transition to dead state (represented as None).
                    if let Some(st) = dfa_states.get_mut(id) {
                        st.transitions[byte as usize] = None;
                    }
                    continue;
                }
                let next_id = if let Some(&existing) = state_ids.get(&next) {
                    existing
                } else {
                    if dfa_states.len() >= MAX_DFA_STATES {
                        return Err(NegexError::TooManyStates {
                            states: dfa_states.len() + 1,
                            limit: MAX_DFA_STATES,
                        });
                    }
                    let nid = new_state(&next, nfa_states, &mut state_ids, &mut dfa_states);
                    work.push_back(next.clone());
                    nid
                };
                if let Some(st) = dfa_states.get_mut(id) {
                    st.transitions[byte as usize] = Some(next_id);
                }
            }
        }

        log_debug!("dfa: produced {} states", dfa_states.len());
        Ok(Self {
            states: dfa_states,
            start: start_id,
            matches_empty,
        })
    }

    /// Returns the number of DFA states.
    #[must_use]
    #[inline]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Returns `true` when the DFA has no states.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Builds the reverse search automaton used to find the leftmost start
    /// of a match.
    ///
    /// This is a subset construction over the reversed transition relation.
    /// The start set is every accepting state of this DFA, and every reverse
    /// state also contains that set, because a match may consume only a
    /// prefix of the remaining input — including the empty prefix.  From a
    /// set `S` on byte `b` the next set is `F ∪ { q | δ(q, b) ∈ S }`, where
    /// `F` is the accept set.
    ///
    /// A reverse state is accepting when it contains this DFA's start state,
    /// so scanning the input from the last byte backwards and remembering
    /// the last accepting position yields the leftmost offset at which a
    /// match can begin.
    ///
    /// # Errors
    ///
    /// Returns [`NegexError::TooManyStates`] when the reverse automaton
    /// exceeds [`MAX_DFA_STATES`], keeping memory bounded.
    pub fn reverse(&self) -> Result<Self, NegexError> {
        if self.states.is_empty() {
            return Err(NegexError::EmptyDfa);
        }

        let mut state_ids: BTreeMap<BTreeSet<usize>, usize> = BTreeMap::new();
        let mut states: Vec<DfaState> = Vec::new();
        let mut work: VecDeque<BTreeSet<usize>> = VecDeque::new();

        let mut start_set = BTreeSet::new();
        for (index, state) in self.states.iter().enumerate() {
            if state.accept {
                start_set.insert(index);
            }
        }
        // Every reverse state also contains the accept set: from any
        // position a match may consume only a prefix of the remaining input,
        // including the empty prefix.  Carrying `start_set` in each state is
        // what turns the reverse automaton into a "does a match start here?"
        // oracle rather than a full-suffix matcher.
        let final_set = start_set.clone();

        let start = register_reverse(&start_set, self.start, &mut state_ids, &mut states);
        work.push_back(start_set);

        while let Some(set) = work.pop_front() {
            let id = state_ids[&set];
            for byte in 0..=255u16 {
                let byte = byte as usize;
                let mut next = final_set.clone();
                for (source, state) in self.states.iter().enumerate() {
                    if let Some(target) = state.transitions.get(byte).copied().flatten()
                        && set.contains(&target)
                    {
                        next.insert(source);
                    }
                }
                let next_id = if let Some(&existing) = state_ids.get(&next) {
                    existing
                } else {
                    if states.len() >= MAX_DFA_STATES {
                        return Err(NegexError::TooManyStates {
                            states: states.len() + 1,
                            limit: MAX_DFA_STATES,
                        });
                    }
                    let registered = register_reverse(&next, self.start, &mut state_ids, &mut states);
                    work.push_back(next);
                    registered
                };
                if let Some(state) = states.get_mut(id) {
                    state.transitions[byte] = Some(next_id);
                }
            }
        }

        log_debug!("dfa: reverse automaton has {} states", states.len());
        Ok(Self {
            states,
            start,
            // The reverse start set is accepting exactly when the original
            // start state is accepting, i.e. the pattern matches empty.
            matches_empty: self.matches_empty,
        })
    }
}

/// Registers a reverse-DFA state (a set of forward states).
fn register_reverse(
    set: &BTreeSet<usize>,
    forward_start: usize,
    state_ids: &mut BTreeMap<BTreeSet<usize>, usize>,
    states: &mut Vec<DfaState>,
) -> usize {
    if let Some(&existing) = state_ids.get(set) {
        return existing;
    }
    let id = states.len();
    state_ids.insert(set.clone(), id);
    let mut state = DfaState::new();
    state.accept = set.contains(&forward_start);
    states.push(state);
    id
}

/// Registers a set of NFA states as a new DFA state (if not already
/// present) and returns its id.
#[inline]
fn new_state(
    set: &BTreeSet<usize>,
    nfa_states: &[NfaState],
    state_ids: &mut BTreeMap<BTreeSet<usize>, usize>,
    dfa_states: &mut Vec<DfaState>,
) -> usize {
    if let Some(&existing) = state_ids.get(set) {
        return existing;
    }
    let id = dfa_states.len();
    state_ids.insert(set.clone(), id);
    let accept = set.iter().any(|&s| {
        nfa_states
            .get(s)
            .map(NfaState::is_match)
            .unwrap_or(false)
    });
    dfa_states.push(DfaState::new());
    dfa_states[id].accept = accept;
    log_debug!("dfa: new state {id}, accept={}, nfa_count={}", accept, set.len());
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nfa::build;
    use codevar_negex_parse::parse;

    fn determinize(pattern: &str) -> Determinized {
        let output = parse(pattern);
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        Determinized::from_nfa(&list, start).expect("determinizes")
    }

    #[test]
    fn reverse_of_literal_accepts_after_consuming_literal() {
        let dfa = determinize("a");
        let reverse = dfa.reverse().expect("reverses");
        assert!(!reverse.states[reverse.start].accept);
        let next =
            reverse.states[reverse.start].transitions[b'a' as usize].expect("reverse transition on `a`");
        assert!(reverse.states[next].accept, "start is reachable after `a`");
    }

    #[test]
    fn reverse_of_empty_pattern_accepts_immediately() {
        let dfa = determinize("");
        let reverse = dfa.reverse().expect("reverses");
        assert!(reverse.states[reverse.start].accept);
    }

    #[test]
    fn reverse_transitions_are_total_for_matching_patterns() {
        // Every byte keeps the prefix-search automaton alive because each
        // reverse state contains the accept set.
        let dfa = determinize("ab");
        let reverse = dfa.reverse().expect("reverses");
        assert!(
            reverse.states[reverse.start]
                .transitions
                .iter()
                .all(Option::is_some)
        );
    }

    #[test]
    fn reverse_of_a_plus_finds_start_after_a() {
        let dfa = determinize("a+");
        let reverse = dfa.reverse().expect("reverses");
        let after_a =
            reverse.states[reverse.start].transitions[b'a' as usize].expect("reverse transition on `a`");
        assert!(reverse.states[after_a].accept);
    }

    #[test]
    fn print_dfa_transitions() {
        for pattern in ["a", "a*", "cat|dog", "world"] {
            let dfa = determinize(pattern);
            println!("\n=== Pattern: {} ===", pattern);
            println!("DFA start: {}, states: {}", dfa.start, dfa.states.len());
            for (i, state) in dfa.states.iter().enumerate() {
                println!("State {}: accept={}", i, state.accept);
                for byte in 0..=255u8 {
                    if let Some(target) = state.transitions[byte as usize] {
                        println!("  byte {} ('{}') -> state {}", byte, byte as char, target);
                    }
                }
            }
        }
    }

    #[test]
    fn determinize_literal() {
        let output = parse("a");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(!dfa.states.is_empty());
    }

    #[test]
    fn determinize_star() {
        let output = parse("a*");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(!dfa.states.is_empty());
    }

    #[test]
    fn determinize_alternation() {
        let output = parse("a|b");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(!dfa.states.is_empty());
    }

    #[test]
    fn determinize_empty() {
        let output = parse("");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(!dfa.states.is_empty());
    }

    #[test]
    fn accept_state_marked() {
        // "a" should have an accepting state.
        let output = parse("a");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(dfa.states.iter().any(|s| s.accept));
    }

    #[test]
    fn matches_empty_for_star() {
        let output = parse("a*");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(dfa.matches_empty, "a* matches empty");
    }

    #[test]
    fn matches_empty_false_for_literal() {
        let output = parse("a");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(!dfa.matches_empty, "a does not match empty");
    }

    #[test]
    fn state_count_bounded() {
        let output = parse("a");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(dfa.state_count() <= MAX_DFA_STATES);
    }

    #[test]
    fn state_count_and_is_empty() {
        let output = parse("a");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(!dfa.is_empty());
        assert!(dfa.state_count() > 0);
    }

    #[test]
    fn plus_does_not_match_empty() {
        let output = parse("a+");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(!dfa.matches_empty, "a+ does not match empty");
    }

    #[test]
    fn question_matches_empty() {
        let output = parse("a?");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        let dfa = Determinized::from_nfa(&list, start).expect("determinizes");
        assert!(dfa.matches_empty, "a? matches empty");
    }
}
