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

//! Target-independent regex codegen: NFA construction, DFA
//! construction/minimization, and lowering to the Codevar IR.
//!
//! This crate takes a parsed regex AST (RFC 9485 / I-Regexp) and turns it
//! into a deterministic finite automaton, then lowers that DFA into a
//! [`codevar_ocl_ir::ir::Module`] that the native code generator can
//! compile to machine code.
//!
//! Because I-Regexp has no backreferences, lookahead, or backtracking,
//! the resulting DFA matches in time linear in the input length — the
//! engine is inherently ReDOS-resistant.
//!
//! # Pipeline
//!
//! 1. **[`nfa`] — Thompson NFA.**  The regex AST becomes an
//!    epsilon-NFA with one state per atom/quantifier.
//! 2. **[`dfa`] — Subset construction.**  The NFA is determinized into a
//!    DFA.  DFA states are sets of NFA states, so the number of states is
//!    bounded by `2^n` in theory but small in practice for I-Regexp.
//! 3. **[`lower`] — IR lowering.**  The DFA is lowered to a
//!    [`codevar_ocl_ir::ir::Module`] with one basic block per DFA state
//!    and a byte-scanning loop.  Range quantifiers are expanded into
//!    counting states, capped at [`MAX_RANGE_EXPANSION`] to prevent
//!    state explosion.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod class;
mod dfa;
mod error;
mod lower;
mod nfa;

pub use class::*;
pub use dfa::*;
pub use error::*;
pub use lower::*;
pub use nfa::*;

use alloc::vec;
use alloc::vec::Vec;

/// Maximum number of states a range quantifier may expand into, to keep
/// DFA construction bounded and prevent ReDOS via state explosion.
pub const MAX_RANGE_EXPANSION: u32 = 64;

/// Maximum number of DFA states before construction is aborted with an
/// error, keeping memory bounded for embedded use.
pub const MAX_DFA_STATES: usize = 4096;

/// A compiled regex program: a DFA plus the metadata needed to run it.
#[derive(Debug, Clone)]
pub struct CompiledProgram {
    /// The DFA states.
    pub states: Vec<DfaState>,
    /// The start state index.
    pub start: usize,
    /// Whether the empty string matches.
    pub matches_empty: bool,
    /// Whether the regex is anchored (must match the full input).
    pub anchored: bool,
}

impl CompiledProgram {
    /// Creates a new compiled program.
    #[must_use]
    pub fn new(states: Vec<DfaState>, start: usize, matches_empty: bool) -> Self {
        Self {
            states,
            start,
            matches_empty,
            anchored: false,
        }
    }

    /// Returns the number of DFA states.
    #[must_use]
    #[inline]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }
}

/// One DFA state: transitions keyed by input byte, plus an accept flag.
#[derive(Debug, Clone)]
pub struct DfaState {
    /// Whether this state is accepting.
    pub accept: bool,
    /// Transitions: for each byte value `0..=255`, the target state index.
    /// `None` means the transition leads to the dead state.
    pub transitions: Vec<Option<usize>>,
    /// Optional repeat counter: `(count, max)` for `{min,max}` states.
    pub repeat: Option<(u32, u32)>,
}

impl DfaState {
    /// Creates a non-accepting state with 256 byte transitions, all dead.
    #[must_use]
    pub fn new() -> Self {
        Self {
            accept: false,
            transitions: vec![None; 256],
            repeat: None,
        }
    }
}

impl Default for DfaState {
    fn default() -> Self {
        Self::new()
    }
}

/// The set of bytes a character-class atom matches, as a 256-bit bitmap.
///
/// The engine matches bytes, so every atom — a literal, `.`, or a `[...]`
/// class — compiles down to one `ByteClass`.  A bitmap makes membership a
/// single shift-and-test (and, in generated code, a handful of integer
/// operations), which keeps the class test off the match loop's hot path.
///
/// Bit `b` is set when byte `b` is a member.  The four words are stored
/// little-endian by byte index: word `b / 64`, bit `b % 64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteClass {
    /// Bitmap words, indexed by `byte / 64`.
    bits: [u64; 4],
}

impl Default for ByteClass {
    fn default() -> Self {
        Self::empty()
    }
}

impl ByteClass {
    /// The empty class: matches no byte.
    #[must_use]
    #[inline]
    pub const fn empty() -> Self {
        Self { bits: [0; 4] }
    }

    /// The universal class: matches every byte.
    #[must_use]
    #[inline]
    pub const fn any() -> Self {
        Self { bits: [u64::MAX; 4] }
    }

    /// A class matching exactly `byte`.
    #[must_use]
    #[inline]
    pub const fn singleton(byte: u8) -> Self {
        let mut class = Self::empty();
        class.insert(byte);
        class
    }
    /// A class matching the inclusive byte range `start..=end`.
    #[must_use]
    #[inline]
    pub fn range(start: u8, end: u8) -> Self {
        let mut class = Self::empty();
        class.insert_range(start, end);
        class
    }

    /// Adds `byte` to the class.
    #[inline]
    pub const fn insert(&mut self, byte: u8) {
        self.bits[byte as usize / 64] |= 1u64 << (byte % 64);
    }

    /// Adds every byte in the inclusive range `start..=end`.
    #[inline]
    pub fn insert_range(&mut self, start: u8, end: u8) {
        if start > end {
            return;
        }
        let mut byte = u16::from(start);
        while byte <= u16::from(end) {
            self.insert(byte as u8);
            byte += 1;
        }
    }

    /// Removes `byte` from the class.
    #[inline]
    pub fn remove(&mut self, byte: u8) {
        self.bits[byte as usize / 64] &= !(1u64 << (byte % 64));
    }

    /// Adds every byte in `other` to this class.
    #[inline]
    pub fn union(&mut self, other: Self) {
        for (word, add) in self.bits.iter_mut().zip(other.bits) {
            *word |= add;
        }
    }

    /// Inverts the class over the full byte range.
    #[inline]
    pub fn invert(&mut self) {
        for word in &mut self.bits {
            *word = !*word;
        }
    }

    /// Whether `byte` is a member.
    #[must_use]
    #[inline]
    pub const fn contains(&self, byte: u8) -> bool {
        self.bits[byte as usize / 64] & (1u64 << (byte % 64)) != 0
    }

    /// The bitmap words, indexed by `byte / 64`.
    #[must_use]
    #[inline]
    pub const fn words(&self) -> &[u64; 4] {
        &self.bits
    }

    /// Whether the class matches no byte.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.bits[0] == 0 && self.bits[1] == 0 && self.bits[2] == 0 && self.bits[3] == 0
    }

    /// Whether the class matches every byte.
    #[must_use]
    #[inline]
    pub const fn is_any(&self) -> bool {
        self.bits[0] == u64::MAX
            && self.bits[1] == u64::MAX
            && self.bits[2] == u64::MAX
            && self.bits[3] == u64::MAX
    }
}

/// The dead state id: a transition to this state means no match possible.
pub const DEAD_STATE: usize = usize::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_program_state_count() {
        let program = CompiledProgram::new(vec![], 0, true);
        assert_eq!(program.state_count(), 0);
        assert!(program.matches_empty);
    }

    #[test]
    fn dfa_state_new_has_no_transitions() {
        let state = DfaState::new();
        assert!(!state.accept);
        assert_eq!(state.transitions.len(), 256);
        assert!(state.transitions.iter().all(|t| t.is_none()));
    }
}
