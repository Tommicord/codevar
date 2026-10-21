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

//! The individual optimization passes.
//!
//! Every pass rewrites the module in place and leaves it verifiable.
//! A pass that cannot prove what it needs leaves the module alone
//! rather than guessing.

use codevar_ocl_ir::ir::Module;

use crate::error::Result;

mod cfg;
mod dce;
mod fold;
mod gvn;
mod inline;
mod instcombine;
mod licm;
mod mem2reg;
mod sccp;
mod unroll;

/// Promotes the function's stack slots into registers.
///
/// The slots whose address never escapes become values: the loads and
/// the stores disappear and the places where the value differs
/// between the incoming edges receive a merge.
///
/// # Errors
///
/// Returns an error when the module cannot be rewritten.
pub fn mem2reg(module: &mut Module) -> Result<()> {
    mem2reg::run(module)
}

/// Evaluates the operations whose operands are all known.
///
/// The integer arithmetic honours the width and the signedness of the
/// type, the floating-point arithmetic honours the precision of the
/// type, and the comparisons and the conversions follow the same
/// rules.  An operation whose result the intermediate representation
/// cannot hold — a division by zero, a shift past the width of the
/// type — is left alone.
///
/// # Errors
///
/// Returns an error when the module cannot be rewritten.
pub fn constant_folding(module: &mut Module) -> Result<()> {
    fold::run(module)
}

/// Propagates the constant arguments through the control flow.
///
/// A branch on a known condition makes one of its edges impossible; the
/// block behind that edge never runs, and the values it defines are
/// undefined.  The pass does not touch the control flow itself: the
/// control-flow simplification removes the blocks the impossible edges
/// lead to.
///
/// # Errors
///
/// Returns an error when the module cannot be rewritten.
pub fn sccp(module: &mut Module) -> Result<()> {
    sccp::run(module)
}

/// Shares the computations that are provably the same.
///
/// Two computations are the same when they are the same operation over
/// the same values.  The sharing is safe because the pass walks the
/// dominator tree, so a computation is reused only where its first
/// definition dominates the use.
///
/// # Errors
///
/// Returns an error when the module cannot be rewritten.
pub fn gvn(module: &mut Module) -> Result<()> {
    gvn::run(module)
}

/// Rewrites the operations into their simplest equivalent form.
///
/// The pass collapses the identities of the integer arithmetic, the
/// selects on a known condition, the comparisons whose operands are
/// known, the copies of copies, and the negations of negations.
///
/// # Errors
///
/// Returns an error when the module cannot be rewritten.
pub fn instcombine(module: &mut Module) -> Result<()> {
    instcombine::run(module)
}

/// Folds the branches whose condition is known and removes the
/// unreachable blocks.
///
/// A block that falls through into its single successor is merged into
/// it.  The branch of a loop header keeps the loop's shape, and the
/// merge a branch names goes away with it.
///
/// # Errors
///
/// Returns an error when the module cannot be rewritten.
pub fn simplify_cfg(module: &mut Module) -> Result<()> {
    cfg::run(module)
}

/// Removes the computations nobody uses.
///
/// The stores, the calls, the extended instructions, and the barriers
/// stay whatever their result is worth.
///
/// # Errors
///
/// Returns an error when the module cannot be rewritten.
pub fn dce(module: &mut Module) -> Result<()> {
    dce::run(module)
}

/// Brings the body of a called function into its caller.
///
/// The callee must be small enough — its body must hold at most
/// `threshold` instructions — and must not call its caller.
///
/// # Errors
///
/// Returns an error when the module cannot be rewritten.
pub fn inline(module: &mut Module, threshold: u32) -> Result<()> {
    inline::run(module, threshold)
}

/// Moves the loop-invariant computations out of their loop.
///
/// The computations move into the block that enters the loop, and only
/// when their operands reach that block.
///
/// # Errors
///
/// Returns an error when the module cannot be rewritten.
pub fn licm(module: &mut Module) -> Result<()> {
    licm::run(module)
}

/// Duplicates the body of a small counted loop.
///
/// The loop must count with an induction variable whose start, step,
/// and bound are known.  The pass writes out the first iterations and
/// leaves the loop to run the rest when the count is larger than
/// `limit`.
///
/// # Errors
///
/// Returns an error when the module cannot be rewritten.
pub fn unroll(module: &mut Module, limit: u32) -> Result<()> {
    unroll::run(module, limit)
}
