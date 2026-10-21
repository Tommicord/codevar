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

use codevar_ocl_ir::ir::Module;

use crate::error::Result;

pub(crate) mod util;

pub mod passes;

#[cfg(test)]
mod tests;

/// How hard the optimizer works.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptConfig {
    /// The optimization level: zero keeps the module as it is, one
    /// runs the cheap local passes, two runs every pass.
    pub level: u8,
    /// The largest function the inliner brings into its caller, in
    /// instructions.
    pub inline_threshold: u32,
    /// Whether counted loops are unrolled.
    pub unroll: bool,
    /// The largest number of iterations the unroller duplicates.
    pub unroll_limit: u32,
}

impl Default for OptConfig {
    fn default() -> Self {
        Self {
            level: 2,
            inline_threshold: 48,
            unroll: true,
            unroll_limit: 8,
        }
    }
}

/// Optimizes `module` in place.
///
/// # Errors
///
/// Returns [`CodegenError::Optimization`](crate::error::CodegenError::Optimization)
/// when a pass cannot rewrite the module; the module is then left in
/// the state the pass found it in.
pub fn optimize(module: &mut Module, config: &OptConfig) -> Result<()> {
    if config.level == 0 {
        return Ok(());
    }
    passes::mem2reg(module)?;
    passes::constant_folding(module)?;
    passes::sccp(module)?;
    passes::instcombine(module)?;
    passes::gvn(module)?;
    passes::simplify_cfg(module)?;
    passes::dce(module)?;
    if config.level >= 2 {
        passes::inline(module, config.inline_threshold)?;
        passes::licm(module)?;
        if config.unroll {
            passes::unroll(module, config.unroll_limit)?;
        }
        passes::constant_folding(module)?;
        passes::gvn(module)?;
        passes::simplify_cfg(module)?;
        passes::dce(module)?;
    }
    Ok(())
}
