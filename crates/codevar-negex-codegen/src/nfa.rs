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

//! Thompson NFA construction from a regex AST.
//!
//! Each NFA state has at most two outgoing transitions: a "label" edge
//! (consuming a character class) and an epsilon edge (consuming nothing).
//! A state is either a split (two epsilon edges), a match (accepting), or
//! a read (one labeled edge plus one epsilon edge).
//!
//! Construction is fragment-based with dangling-epsilon-arrow patching,
//! following Thompson's classic construction.  Each quantifier, alternation,
//! and group produces a fragment with a start state and a list of dangling
//! arrow targets that get patched when the fragment is connected.

use alloc::vec;
use alloc::vec::Vec;
use codevar_logger::log_debug;
use codevar_negex_parse::{Atom, Branch as AstBranch, Piece as AstPiece, Quantifier, Regex};

use crate::{ByteClass, NegexError};

/// One NFA state.
#[derive(Debug, Clone)]
pub enum NfaState {
    /// A split state: two epsilon transitions to `out` and `out1`.
    ///
    /// `out` is the first epsilon target; `out1` is the second.
    Split {
        /// First epsilon transition target.
        out: usize,
        /// Second epsilon transition target.
        out1: usize,
    },
    /// A read state: consumes a character class, then epsilon to `out`.
    Read {
        /// The set of bytes this state consumes.
        class: ByteClass,
        /// Epsilon transition target after consuming the class.
        out: usize,
    },
    /// A match state: accepting.
    Match,
}

impl NfaState {
    /// Whether this state is a match (accepting) state.
    #[must_use]
    #[inline]
    pub const fn is_match(&self) -> bool {
        matches!(self, Self::Match)
    }
}

/// An NFA fragment: a start state and a list of dangling epsilon arrow
/// addresses (indices into `Out`/`Out1` fields) that need patching.
#[derive(Debug, Clone)]
pub struct Frag {
    /// The start state of the fragment.
    pub start: usize,
    /// Dangling arrow targets to patch. Each entry is an index into a
    /// state's `Out`/`Out1` field, encoded as `state_index * 2 + flag`
    /// where `flag` is 0 for `out` and 1 for `out1`.
    pub outs: Vec<usize>,
}

/// A list of states being built.
#[derive(Debug, Clone)]
pub struct StateList {
    states: Vec<NfaState>,
}

impl StateList {
    /// Creates an empty state list.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Appends a state and returns its index.
    #[inline]
    pub fn add(&mut self, state: NfaState) -> usize {
        let index = self.states.len();
        log_debug!("nfa: added state {index} ({:?})", state);
        self.states.push(state);
        index
    }

    /// Returns the number of states.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns `true` when the list contains no states.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Returns a reference to state `i`.
    #[must_use]
    #[inline]
    pub fn get(&self, i: usize) -> Option<&NfaState> {
        self.states.get(i)
    }

    /// Returns all states as a slice.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[NfaState] {
        &self.states
    }

    /// Returns a mutable reference to state `i`.
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> Option<&mut NfaState> {
        self.states.get_mut(i)
    }

    /// Consumes the list into a vector.
    #[must_use]
    #[inline]
    pub fn into_vec(self) -> Vec<NfaState> {
        self.states
    }
}

impl Default for StateList {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Encodes an arrow address as `state_index * 2 + flag` (flag 0 = `out`,
/// flag 1 = `out1`).
#[inline]
fn addr(state: usize, flag: usize) -> usize {
    state * 2 + flag
}

/// The field index (0 for `out`, 1 for `out1`) encoded in an address.
#[inline]
fn flag_of(addr: usize) -> usize {
    addr & 1
}

/// The state index encoded in an address.
#[inline]
fn state_of(addr: usize) -> usize {
    addr >> 1
}

/// Patches all dangling arrows in `outs` to point to `target`.
#[inline]
fn patch(list: &mut StateList, outs: &[usize], target: usize) {
    for addr in outs {
        let state = state_of(*addr);
        let flag = flag_of(*addr);
        if let Some(st) = list.get_mut(state) {
            match st {
                NfaState::Split { out, out1 } => {
                    if flag == 0 {
                        *out = target;
                    } else {
                        *out1 = target;
                    }
                }
                NfaState::Read { out, .. } => {
                    *out = target;
                }
                NfaState::Match => {}
            }
        }
    }
    log_debug!("nfa: patched {} arrows -> state {target}", outs.len());
}

/// Builds an NFA from a regex AST, returning the state list and the start
/// state index.
///
/// # Errors
///
/// Returns an error when the regex expands beyond
/// [`super::MAX_RANGE_EXPANSION`] copies per range quantifier, or when a
/// character class cannot be represented as a byte set.  Keeping both
/// bounds enforced means NFA construction is bounded and ReDOS-resistant.
pub fn build(ast: &Regex) -> Result<(StateList, usize), NegexError> {
    log_debug!("nfa: building NFA from AST");
    let mut list = StateList::new();
    let frag = compile_regex(ast, &mut list)?;
    let accept = list.add(NfaState::Match);
    patch(&mut list, &frag.outs, accept);
    log_debug!(
        "nfa: built {} states, start={}, accept={}",
        list.len(),
        frag.start,
        accept
    );
    Ok((list, frag.start))
}

/// Compiles a full regex into an NFA fragment.
///
/// A single-branch expression compiles as a concatenation.  Two or more
/// branches compile as a Thompson alternation: a chain of split states, one
/// per alternative, whose outgoing epsilon edges enter each branch.
fn compile_regex(ast: &Regex, list: &mut StateList) -> Result<Frag, NegexError> {
    if ast.branches.len() == 1 {
        return compile_branch(&ast.branches[0], list);
    }

    let mut frags = Vec::with_capacity(ast.branches.len());
    for branch in &ast.branches {
        frags.push(compile_branch(branch, list)?);
    }

    // `frags.len() - 1` split states chain the alternatives: split `i`
    // takes branch `i` on one edge and the next split on the other, and the
    // last split takes the final two branches.
    let splits = frags.len() - 1;
    let mut split_ids = Vec::with_capacity(splits);
    for _ in 0..splits {
        split_ids.push(list.add(NfaState::Split { out: 0, out1: 0 }));
    }

    for index in 0..splits {
        let split = split_ids[index];
        let taken = frags[index].start;
        let otherwise = if index + 1 < splits {
            split_ids[index + 1]
        } else {
            frags[index + 1].start
        };
        if let Some(state) = list.get_mut(split)
            && let NfaState::Split { out, out1 } = state
        {
            *out = taken;
            *out1 = otherwise;
        }
    }

    let mut outs = Vec::new();
    for frag in &frags {
        outs.extend_from_slice(&frag.outs);
    }

    Ok(Frag {
        start: split_ids[0],
        outs,
    })
}

/// Compiles a branch: concatenation of pieces.
fn compile_branch(branch: &AstBranch, list: &mut StateList) -> Result<Frag, NegexError> {
    if branch.pieces.is_empty() {
        // Empty branch: pure epsilon; two dangling outs.
        let start = list.add(NfaState::Split { out: 0, out1: 0 });
        return Ok(Frag {
            start,
            outs: vec![addr(start, 0), addr(start, 1)],
        });
    }
    let mut frag = compile_piece(&branch.pieces[0], list)?;
    for piece in &branch.pieces[1..] {
        let next = compile_piece(piece, list)?;
        patch(list, &frag.outs, next.start);
        frag.outs = next.outs;
    }
    Ok(frag)
}

/// Compiles a piece: atom plus optional quantifier.
fn compile_piece(piece: &AstPiece, list: &mut StateList) -> Result<Frag, NegexError> {
    match &piece.quantifier {
        None => compile_atom(&piece.atom, list),
        Some(Quantifier::Star) => compile_star(&piece.atom, list),
        Some(Quantifier::Plus) => {
            // e+ = e e*
            let first = compile_atom(&piece.atom, list)?;
            let rest = compile_star(&piece.atom, list)?;
            patch(list, &first.outs, rest.start);
            Ok(Frag {
                start: first.start,
                outs: rest.outs,
            })
        }
        Some(Quantifier::Question) => {
            // e? = split(e, epsilon)
            let start = list.add(NfaState::Split { out: 0, out1: 0 });
            let body = compile_atom(&piece.atom, list)?;
            if let Some(st) = list.get_mut(start)
                && let NfaState::Split { out, .. } = st
            {
                *out = body.start;
            }
            let mut outs = body.outs;
            outs.push(addr(start, 1));
            Ok(Frag { start, outs })
        }
        Some(Quantifier::Range { min, max }) => compile_range(&piece.atom, *min, *max, list),
    }
}

/// Compiles `e*`: split(e, skip) with e's outs looping back to start.
fn compile_star(atom: &Atom, list: &mut StateList) -> Result<Frag, NegexError> {
    let start = list.add(NfaState::Split { out: 0, out1: 0 });
    let body = compile_atom(atom, list)?;
    let skip = list.add(NfaState::Split { out: 0, out1: 0 });
    if let Some(st) = list.get_mut(start)
        && let NfaState::Split { out, out1 } = st
    {
        *out = body.start;
        *out1 = skip;
    }
    patch(list, &body.outs, start);
    Ok(Frag {
        start,
        outs: vec![addr(skip, 0), addr(skip, 1)],
    })
}

/// Compiles a bounded range quantifier `{min,max}`.
///
/// The `{min,}` form is treated as `{min, MAX_RANGE_EXPANSION}` so the NFA
/// stays bounded.  `{min}` (exact count) is `max = min`.
fn compile_range(atom: &Atom, min: u32, max: Option<u32>, list: &mut StateList) -> Result<Frag, NegexError> {
    let requested_max = max;
    let max = max.unwrap_or(super::MAX_RANGE_EXPANSION);
    if min > max || min > super::MAX_RANGE_EXPANSION {
        return Err(NegexError::InvalidRangeQuantifier {
            min,
            max: requested_max,
        });
    }
    log_debug!("nfa: compiling range quantifier {{min={min}, max={max:?}}}",);

    // Build `min` mandatory copies concatenated.
    let mut frag: Option<Frag> = None;
    for _ in 0..min {
        let copy = compile_atom(atom, list)?;
        match &mut frag {
            None => frag = Some(copy),
            Some(prev) => {
                patch(list, &prev.outs, copy.start);
                prev.outs = copy.outs;
            }
        }
    }
    let mut frag = frag.unwrap_or_else(|| Frag {
        start: list.add(NfaState::Split { out: 0, out1: 0 }),
        outs: Vec::new(),
    });
    // Build `max - min` optional copies: split(copy, skip) chained.
    for _ in min..max {
        let copy_start = list.add(NfaState::Split { out: 0, out1: 0 });
        let body = compile_atom(atom, list)?;
        let skip = list.add(NfaState::Split { out: 0, out1: 0 });
        if let Some(st) = list.get_mut(copy_start)
            && let NfaState::Split { out, out1 } = st
        {
            *out = body.start;
            *out1 = skip;
        }
        patch(list, &body.outs, copy_start);
        let old_outs = core::mem::take(&mut frag.outs);
        patch(list, &old_outs, copy_start);
        frag.outs = vec![addr(skip, 0), addr(skip, 1)];
        if let Some(st) = list.get_mut(frag.start)
            && let NfaState::Split { out, .. } = st
        {
            *out = copy_start;
        }
    }
    Ok(frag)
}

/// Compiles an atom into an NFA fragment.
fn compile_atom(atom: &Atom, list: &mut StateList) -> Result<Frag, NegexError> {
    match atom {
        Atom::Literal(scalar) => compile_scalar(*scalar, list),
        Atom::Any => read_state(any_except_line_terminators(), list),
        Atom::Class(class) => {
            let bytes = crate::class::byte_class(class)?;
            read_state(bytes, list)
        }
        Atom::Group(regex) => compile_regex(regex, list),
        Atom::Error { .. } => Err(NegexError::InvalidClassAst),
    }
}

/// Compiles one Unicode scalar as the byte sequence of its UTF-8 encoding.
///
/// The engine matches bytes, so a non-ASCII literal becomes a short chain of
/// single-byte reads.  This keeps literal matching exact for any scalar while
/// leaving the automaton byte-oriented.
fn compile_scalar(scalar: char, list: &mut StateList) -> Result<Frag, NegexError> {
    let mut buffer = [0u8; 4];
    let encoded = scalar.encode_utf8(&mut buffer).as_bytes();
    compile_bytes(encoded, list)
}

/// Compiles a byte sequence into a chain of single-byte reads.
fn compile_bytes(bytes: &[u8], list: &mut StateList) -> Result<Frag, NegexError> {
    let Some((&first, rest)) = bytes.split_first() else {
        // An empty sequence is pure epsilon.
        let start = list.add(NfaState::Split { out: 0, out1: 0 });
        return Ok(Frag {
            start,
            outs: vec![addr(start, 0), addr(start, 1)],
        });
    };
    let mut frag = read_state(ByteClass::singleton(first), list)?;
    for &byte in rest {
        let next = read_state(ByteClass::singleton(byte), list)?;
        patch(list, &frag.outs, next.start);
        frag.outs = next.outs;
    }
    Ok(frag)
}

/// The class matched by `.`: every byte except the line terminators `\n` and
/// `\r`.
fn any_except_line_terminators() -> ByteClass {
    let mut class = ByteClass::any();
    class.remove(b'\n');
    class.remove(b'\r');
    class
}

fn read_state(class: ByteClass, list: &mut StateList) -> Result<Frag, NegexError> {
    let out = list.add(NfaState::Split { out: 0, out1: 0 });
    let start = list.add(NfaState::Read { class, out });
    Ok(Frag {
        start,
        outs: vec![addr(out, 0), addr(out, 1)],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_negex_parse::parse;

    #[test]
    fn build_literal() {
        let output = parse("a");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn build_star() {
        let output = parse("a*");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn build_plus() {
        let output = parse("a+");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn build_question() {
        let output = parse("a?");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn build_alternation() {
        let output = parse("a|b");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn build_range_quantifier() {
        let output = parse("a{2,4}");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn build_unbounded_range() {
        let output = parse("a{2,}");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn build_exact_count() {
        let output = parse("a{3}");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn empty_regex_builds() {
        let output = parse("");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn group_builds() {
        let output = parse("(ab)");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn range_expansion_capped() {
        // {100,200} exceeds MAX_RANGE_EXPANSION (64) for the max side.
        let output = parse("a{100,200}");
        assert!(output.errors.is_empty());
        let result = build(&output.ast);
        assert!(result.is_err(), "range expansion must be capped");
    }

    #[test]
    fn invalid_range_rejected() {
        let output = parse("a{5,2}");
        assert!(output.errors.is_empty());
        let result = build(&output.ast);
        assert!(result.is_err(), "min > max must be rejected");
    }

    #[test]
    fn state_list_len_and_is_empty() {
        let mut list = StateList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        let _i = list.add(NfaState::Match);
        assert!(!list.is_empty());
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn nfa_state_is_match() {
        assert!(NfaState::Match.is_match());
        assert!(
            !NfaState::Read {
                class: ByteClass::singleton(b'a'),
                out: 0
            }
            .is_match()
        );
        assert!(!NfaState::Split { out: 0, out1: 0 }.is_match());
    }

    #[test]
    fn alternation_uses_group_atoms() {
        // "a|b" parses as a branch with two group-wrapped alternatives.
        let output = parse("a|b");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
        // At least: split + read(a) + split + read(b) + match
        assert!(list.len() >= 5);
    }

    #[test]
    fn quantified_group_builds() {
        let output = parse("(ab)*");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }

    #[test]
    fn char_class_builds() {
        let output = parse("[0-9]");
        assert!(output.errors.is_empty());
        let (list, start) = build(&output.ast).expect("builds");
        assert!(start < list.len());
    }
}
