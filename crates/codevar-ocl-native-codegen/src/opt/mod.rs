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

//! Optimization of the intermediate representation.
//!
//! The optimizer rewrites a module into an equivalent but simpler
//! form.  The passes work on the intermediate representation itself, so
//! every one of them can be run, tested, and measured on its own:
//!
//! * **[`passes::mem2reg`]** — promotes the stack slots the front end
//!   created into registers, which is what turns the alloca-based
//!   shape the lowering produces into the SSA shape the later passes
//!   expect.
//! * **[`passes::constant_folding`]** — evaluates the operations whose
//!   operands are all known.
//! * **[`passes::sccp`]** — propagates the constant arguments through
//!   the control flow, so a branch on a known condition folds away.
//! * **[`passes::gvn`]** — shares the computations that are provably
//!   the same.
//! * **[`passes::instcombine`]** — rewrites the operations into their
//!   simplest equivalent form.
//! * **[`passes::simplify_cfg`]** — folds the branches whose condition
//!   is known and removes the blocks nobody reaches.
//! * **[`passes::dce`]** — removes the computations nobody uses.
//! * **[`passes::inline`]** — brings the body of a called function
//!   into its caller.
//! * **[`passes::licm`]** — moves the loop-invariant computations out
//!   of the loop.
//! * **[`passes::unroll`]** — duplicates the body of a small counted
//!   loop so the branches disappear.
//!
//! [`optimize`] runs them in the order that gives each pass the most
//! to work with, and every pass leaves the module verifiable.
//!
//! # Optimization levels
//!
//! Which passes run is selected by an [`Optimizations`] set, a bit mask of
//! [`Optimization`] variants combined with `|`.  The compiler driver's `-O`
//! flag maps onto four presets: [`Optimizations::O0`] runs nothing,
//! [`Optimizations::O1`] runs the cheap local passes, [`Optimizations::O2`]
//! runs every pass including the vectorizer, and [`Optimizations::O3`]
//! repeats the structural passes at twice the configured limits.
//!
//! # What a pass may and may not do
//!
//! Every pass is conservative: it rewrites only what it can prove, and
//! leaves the rest of the module alone.  A pass that cannot decide
//! does not guess.
//!
//! One shape of the intermediate representation constrains all of
//! them.  The value arena is append-only: a value that an instruction
//! defined cannot leave the module, and the verifier rejects a value
//! that no instruction defines.  A pass that removes a computation
//! therefore keeps its result defined — the operation becomes a copy
//! of the undefined value — and the computation itself disappears.
//! The blocks of a function are identified by their position, so a
//! pass that removes a block renumbers the handles that name it.
//!
//! The passes and what they refuse to do:
//!
//! | Pass | Does | Refuses |
//! | --- | --- | --- |
//! | `mem2reg` | promotes the slots whose address never escapes | a slot whose address reaches a call or an access chain, a slot an unreachable block mentions, a slot that does not start in the entry block, a slot whose merge would not have one incoming per predecessor |
//! | `constant_folding` | evaluates the arithmetic, the comparisons, the selects, and the conversions | an operation whose result is not representable, a division by zero, a shift past the width of the type |
//! | `sccp` | propagates the constants, marks the impossible edges, and makes the values of the blocks that never run undefined | the control flow itself, which the control-flow simplification owns |
//! | `gvn` | shares the computations over the same values | the loads, whose memory may hold a different value at a different point |
//! | `instcombine` | collapses the identities of the integer arithmetic and the copies of copies | the floating-point identities, where `x + 0.0` is not `x` |
//! | `simplify_cfg` | folds the branches on a known condition, removes the unreachable blocks, merges the blocks that fall through | the branch of a loop header, the merge of a block that holds one, a merge whose successor has another predecessor |
//! | `dce` | removes the computations nobody uses | the stores, the calls, the extended instructions, and the barriers |
//! | `inline` | brings a small callee into its caller | a callee that calls its caller, a callee without a body, a callee larger than the threshold |
//! | `licm` | moves the invariant computations into the block that enters the loop | a loop without a single entering block, a computation whose operands do not reach that block, a load |
//! | `unroll` | writes out the iterations of a counted loop | a loop whose count is unknown, a body that holds a merge, a loop whose values are used outside its body |

use core::ops::{BitOr, BitOrAssign};

use codevar_ocl_ir::ir::Module;

use crate::error::Result;

pub(crate) mod util;

pub mod passes;

#[cfg(test)]
mod tests;

/// One optimization the pipeline may run on a module.
///
/// The variants are single bits, so a set of them is an [`Optimizations`]
/// value.  Keeping the optimizations in a bit set means a configuration
/// never has to carry one `bool` per optimization, and the levels the
/// compiler driver exposes are simply presets of the set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Optimization {
    /// Promote the function's stack slots into registers (the `mem2reg`
    /// pass).
    Mem2Reg = 1 << 0,
    /// Evaluate the operations whose operands are all known.
    ConstantFolding = 1 << 1,
    /// Propagate the constant arguments through the control flow.
    ConstantPropagation = 1 << 2,
    /// Rewrite the operations into their simplest equivalent form.
    InstructionCombine = 1 << 3,
    /// Share the computations that are provably the same.
    CommonSubexpression = 1 << 4,
    /// Fold the branches whose condition is known and remove the
    /// unreachable blocks.
    ControlFlow = 1 << 5,
    /// Remove the computations nobody uses.
    DeadCode = 1 << 6,
    /// Bring the body of a small callee into its caller.
    Inlineable = 1 << 7,
    /// Move the loop-invariant computations out of their loop.
    LoopInvariant = 1 << 8,
    /// Duplicate the body of a small counted loop.
    Unrollable = 1 << 9,
    /// Rewrite the straight-line and counted-loop code into vector
    /// operations.
    Vectorizable = 1 << 10,
    /// Repeat the structural passes with larger inlining and unrolling
    /// limits.
    Aggressive = 1 << 11,
}

impl Optimization {
    /// The bit of this optimization.
    #[inline]
    #[must_use]
    pub const fn bit(self) -> u32 {
        self as u32
    }
}

/// A set of [`Optimization`]s.
///
/// The set is a bit mask with `BitOr`/`BitOrAssign` implementations, so a
/// caller combines the optimizations it wants without storing one flag per
/// optimization.  The presets [`Optimizations::O0`] through
/// [`Optimizations::O3`] match the `-O` levels of the compiler driver.
///
/// # Examples
///
/// ```
/// use codevar_ocl_native_codegen::opt::{Optimization, Optimizations};
///
/// let level = Optimizations::O1 | Optimization::Inlineable;
/// assert!(level.has(Optimization::Mem2Reg));
/// assert!(level.has(Optimization::Inlineable));
/// assert!(!Optimizations::O0.has(Optimization::Mem2Reg));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Optimizations(u32);

impl Optimizations {
    /// No optimization at all.
    pub const NONE: Self = Self(0);

    /// `-O0`: keep the module as it is.
    pub const O0: Self = Self::NONE;

    /// `-O1`: the cheap local passes.
    pub const O1: Self = Self(
        Optimization::Mem2Reg.bit()
            | Optimization::ConstantFolding.bit()
            | Optimization::ConstantPropagation.bit()
            | Optimization::InstructionCombine.bit()
            | Optimization::CommonSubexpression.bit()
            | Optimization::ControlFlow.bit()
            | Optimization::DeadCode.bit(),
    );

    /// `-O2`: every optimizer pass, including vectorization.
    pub const O2: Self = Self(
        Self::O1.0
            | Optimization::Inlineable.bit()
            | Optimization::LoopInvariant.bit()
            | Optimization::Unrollable.bit()
            | Optimization::Vectorizable.bit(),
    );

    /// `-O3`: [`Optimizations::O2`] plus the aggressive escalation.
    pub const O3: Self = Self(Self::O2.0 | Optimization::Aggressive.bit());

    /// Every optimization known to this crate.
    pub const ALL: Self = Self((1 << 12) - 1);

    /// Creates a set from raw bits (unknown bits are preserved).
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// The raw bits of the set.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// The set that selects exactly `optimization`.
    #[inline]
    #[must_use]
    pub const fn only(optimization: Optimization) -> Self {
        Self(optimization.bit())
    }

    /// `true` when no optimization is selected.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// `true` when every optimization of `other` is present.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// `true` when any optimization of `other` is present.
    #[inline]
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// `true` when `optimization` is present.
    #[inline]
    #[must_use]
    pub const fn has(self, optimization: Optimization) -> bool {
        (self.0 & optimization.bit()) != 0
    }

    /// `self` with `optimization` added.
    #[inline]
    #[must_use]
    pub const fn with(self, optimization: Optimization) -> Self {
        Self(self.0 | optimization.bit())
    }

    /// The union of two sets.
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The set the driver's `-O` level selects, or `None` for a level the
    /// compiler does not know.
    #[inline]
    #[must_use]
    pub const fn from_level(level: u8) -> Option<Self> {
        match level {
            0 => Some(Self::O0),
            1 => Some(Self::O1),
            2 => Some(Self::O2),
            3 => Some(Self::O3),
            _ => None,
        }
    }
}

impl BitOr for Optimizations {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Optimizations {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitOr<Optimization> for Optimizations {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Optimization) -> Self {
        self.with(rhs)
    }
}

impl BitOr<Optimizations> for Optimization {
    type Output = Optimizations;
    #[inline]
    fn bitor(self, rhs: Optimizations) -> Optimizations {
        rhs.with(self)
    }
}

impl BitOr for Optimization {
    type Output = Optimizations;
    #[inline]
    fn bitor(self, rhs: Self) -> Optimizations {
        Optimizations::only(self).with(rhs)
    }
}

impl From<Optimization> for Optimizations {
    #[inline]
    fn from(optimization: Optimization) -> Self {
        Self::only(optimization)
    }
}

/// How hard the optimizer works.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptConfig {
    /// The optimizations the pipeline runs; [`Optimizations::O0`] keeps
    /// the module as it is.
    pub optimizations: Optimizations,
    /// The largest function the inliner brings into its caller, in
    /// instructions.
    pub inline_threshold: u32,
    /// The largest number of iterations the unroller duplicates.
    pub unroll_limit: u32,
}

impl Default for OptConfig {
    fn default() -> Self {
        Self {
            optimizations: Optimizations::O2,
            inline_threshold: 48,
            unroll_limit: 8,
        }
    }
}

/// Optimizes `module` in place according to `config`.
///
/// The passes whose [`Optimization`] bit is set run in the order that gives
/// each of them the most to work with, and every pass leaves the module
/// verifiable.  An empty selection ([`Optimizations::O0`]) runs nothing.
///
/// # Errors
///
/// Returns [`CodegenError::Optimization`](crate::error::CodegenError::Optimization)
/// when a pass cannot rewrite the module; the module is then left in
/// the state the pass found it in.
pub fn optimize(module: &mut Module, config: &OptConfig) -> Result<()> {
    let selected = config.optimizations;
    if selected.is_empty() {
        return Ok(());
    }
    if selected.has(Optimization::Mem2Reg) {
        passes::mem2reg(module)?;
    }
    if selected.has(Optimization::ConstantFolding) {
        passes::constant_folding(module)?;
    }
    if selected.has(Optimization::ConstantPropagation) {
        passes::sccp(module)?;
    }
    if selected.has(Optimization::InstructionCombine) {
        passes::instcombine(module)?;
    }
    if selected.has(Optimization::CommonSubexpression) {
        passes::gvn(module)?;
    }
    if selected.has(Optimization::ControlFlow) {
        passes::simplify_cfg(module)?;
    }
    if selected.has(Optimization::DeadCode) {
        passes::dce(module)?;
    }
    let structural = selected.intersects(
        Optimizations::only(Optimization::Inlineable)
            | Optimization::LoopInvariant
            | Optimization::Unrollable,
    );
    if structural {
        structural_round(module, config, 1)?;
    }
    if selected.has(Optimization::Aggressive) {
        structural_round(module, config, 2)?;
    }
    Ok(())
}

/// Runs the structural passes — inlining, loop-invariant motion, and
/// unrolling — and then cleans up after them.
///
/// `scale` multiplies the inliner's and the unroller's limits, so the
/// aggressive round of [`Optimization::Aggressive`] runs them at twice
/// the configured size.
fn structural_round(module: &mut Module, config: &OptConfig, scale: u32) -> Result<()> {
    let selected = config.optimizations;
    if selected.has(Optimization::Inlineable) {
        passes::inline(module, config.inline_threshold.saturating_mul(scale))?;
    }
    if selected.has(Optimization::LoopInvariant) {
        passes::licm(module)?;
    }
    if selected.has(Optimization::Unrollable) {
        passes::unroll(module, config.unroll_limit.saturating_mul(scale))?;
    }
    if selected.has(Optimization::ConstantFolding) {
        passes::constant_folding(module)?;
    }
    if selected.has(Optimization::CommonSubexpression) {
        passes::gvn(module)?;
    }
    if selected.has(Optimization::ControlFlow) {
        passes::simplify_cfg(module)?;
    }
    if selected.has(Optimization::DeadCode) {
        passes::dce(module)?;
    }
    Ok(())
}
