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

//! Vectorization of the intermediate representation.
//!
//! Two shapes of code profit from vectors, and the module handles both.
//!
//! * **Straight-line code.**  A group of independent operations on
//!   neighbouring values — four additions of four different numbers —
//!   becomes one vector addition.  The values must be independent of
//!   each other, and the group must be worth a vector.
//! * **Counted loops.**  A loop that walks a range with a unit stride
//!   can be executed a vector at a time: the body is executed on a
//!   vector of values, and the elements that do not fill a whole
//!   vector are handled by a scalar remainder.
//!
//! The vectorizer works on the intermediate representation, where a
//! vector is a type of its own: the operations it writes are ordinary
//! operations over a vector type, which the back ends then lower to
//! the vector instructions their architecture has — or to a sequence
//! of scalar instructions when it has none.
//!
//! # What the vectorizer handles today
//!
//! ## Straight-line groups
//!
//! A group is a run of instructions inside one block that share
//!
//! * one opcode — a binary operation, a unary operation, a comparison,
//!   or a conversion —
//! * one operand type, and
//! * one result type,
//!
//! and whose members are independent of one another: no member reads
//! the result of another member.  A group needs at least
//! [`VectorConfig::min_group_size`] members and holds at most
//! [`VectorConfig::max_lanes`] of them; a longer run is cut into
//! several groups.
//!
//! The members of a group do not have to sit next to one another, so
//! four additions of four loads that follow one another are one
//! group.  The vector operation is placed where the operands of every
//! member are available; a member that lies before that place moves
//! behind the vector operation, which is only done when nothing reads
//! the member before it.
//!
//! The rewrite keeps the original results: each member's instruction
//! becomes a read of one lane of the new vector operation, so every
//! user of the group keeps working without being rewritten.  The
//! vector itself is assembled with a composite construction over the
//! members' operands.
//!
//! ## Counted loops
//!
//! A loop is counted when its header holds an induction variable, the
//! comparison of that variable against a bound, and an increment that
//! steps the variable by one.  The comparison may sit in the header
//! itself or in the block the header branches to.  The loop body must
//!
//! * reach the memory it touches through a single index that walks
//!   the induction variable with a unit stride,
//! * keep no value alive from one iteration to the next — the body
//!   holds no phi — and
//! * compute with binary operations, unary operations, comparisons,
//!   conversions, copies, loads, and stores only.
//!
//! Such a loop is rewritten into two loops: a vector loop that walks
//! the range [`VectorConfig::max_lanes`] elements at a time, and the
//! original loop, which keeps the elements that do not fill a whole
//! vector.  The vector loop's body holds one load per lane, one
//! vector operation, and one store per lane.
//!
//! # What the vectorizer refuses
//!
//! A rewrite that is not certainly right is worse than no rewrite at
//! all, so the vectorizer leaves the code alone whenever the pattern
//! does not match exactly.  In particular it refuses
//!
//! * groups whose members depend on one another, or whose operands
//!   are not yet available where the vector operation goes,
//! * groups of a single member, or of more members than the lane
//!   count allows,
//! * groups of loads: a straight-line group is a group of
//!   arithmetic, and the loads that feed it keep their own form,
//! * loops whose body holds a call, a barrier, a phi, a nested
//!   branch, or any other operation outside the list above,
//! * loops whose index does not walk the induction variable with a
//!   unit stride,
//! * loops that keep a value alive across iterations,
//! * loops whose induction variable does not start from a value the
//!   preheader computes, and
//! * loops whose memory already holds vectors: those are already a
//!   vector wide, and there is nothing left to widen.
//!
//! # Memory accesses
//!
//! The intermediate representation cannot build a pointer to a vector
//! type out of a pointer to a scalar element — it has no pointer
//! reinterpretation — so the vector loop gathers the elements of its
//! lane with one load per lane and scatters them back with one store
//! per lane.  The arithmetic between the gather and the scatter is a
//! single vector operation.

use codevar_ocl_ir::ir::Module;

use crate::error::Result;

mod loops;
mod slp;

#[cfg(test)]
mod tests;

/// How hard the vectorizer works.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorConfig {
    /// The largest number of lanes a vector may have.
    pub max_lanes: u32,
    /// Whether counted loops are vectorized.
    pub loop_vectorization: bool,
    /// Whether straight-line groups are vectorized.
    pub slp_vectorization: bool,
    /// The smallest number of operations a straight-line group must
    /// hold to be worth a vector.
    pub min_group_size: u32,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            max_lanes: 4,
            loop_vectorization: true,
            slp_vectorization: true,
            min_group_size: 2,
        }
    }
}

/// Vectorizes `module` in place.
///
/// The vectorizer runs the counted-loop rewrite first and the
/// straight-line rewrite second, so that the straight-line pass also
/// sees the scalar remainder loop a counted loop leaves behind.
///
/// Running the vectorizer twice changes nothing the second time: a
/// vectorized group no longer holds the scalar operations it was
/// built from, and a vectorized loop no longer matches the shape the
/// loop rewrite recognizes.
///
/// # Errors
///
/// Returns [`CodegenError::Optimization`](crate::error::CodegenError::Optimization)
/// when the module cannot be rewritten; the module is then left in
/// the state the vectorizer found it in.
pub fn vectorize(module: &mut Module, config: &VectorConfig) -> Result<()> {
    if config.max_lanes < 2 {
        return Ok(());
    }
    if config.loop_vectorization {
        loops::vectorize_loops(module, config);
    }
    if config.slp_vectorization {
        slp::vectorize_straight_line(module, config);
    }
    Ok(())
}
