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

//! Global value numbering.
//!
//! The pass shares the computations that are provably the same.  Two
//! computations are the same when they are the same operation over the
//! same operands, and when the operands are the same values - the
//! comparison is structural, not by name.
//!
//! The sharing is safe because the pass walks the dominator tree: the
//! table of known computations holds the definitions of the current
//! block and of its ancestors only, so a computation is reused only
//! where its first definition dominates the use.  The table is scoped
//! to the dominator tree: leaving a block restores the entries its
//! ancestors had.
//!
//! Only the operations that compute a value from other values take
//! part.  A load is left alone: the memory it reads may hold a
//! different value at a different point of the program.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{BinOp, CmpOp, Module, Op, TypeId, ValueId};

use crate::error::Result;
use crate::opt::util::{Cfg, dominator_children, dominators};

/// The identity of a computation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ExprKey {
    /// An operation with a fixed number of operands.
    Fixed {
        /// Which kind of operation this is.
        variant: u8,
        /// Which operation this is.
        code: &'static str,
        /// The type of the result.
        ty: TypeId,
        /// The operands.
        operands: [ValueId; 3],
        /// How many operands are meaningful.
        len: u8,
    },
    /// An operation with a list of operands.
    List {
        /// Which kind of operation this is.
        variant: u8,
        /// Which operation this is.
        code: &'static str,
        /// The type of the result.
        ty: TypeId,
        /// The operands.
        operands: Vec<ValueId>,
    },
    /// An extraction with literal component indices.
    Extract {
        /// Which kind of operation this is.
        variant: u8,
        /// Which operation this is.
        code: &'static str,
        /// The type of the result.
        ty: TypeId,
        /// The composite the component comes from.
        composite: ValueId,
        /// The literal component indices.
        indices: Vec<u32>,
    },
}

/// True when the operands of the operation may be swapped.
fn commutative(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::IAdd
            | BinOp::IMul
            | BinOp::FAdd
            | BinOp::FMul
            | BinOp::BitwiseAnd
            | BinOp::BitwiseOr
            | BinOp::BitwiseXor
            | BinOp::LogicalAnd
            | BinOp::LogicalOr
    )
}

/// True when the operands of the comparison may be swapped.
fn symmetric(op: CmpOp) -> bool {
    matches!(
        op,
        CmpOp::IEqual | CmpOp::INotEqual | CmpOp::FOrdEqual | CmpOp::FOrdNotEqual
    )
}

/// The identity of the computation an operation performs.
fn key_of(op: &Op, ty: TypeId) -> Option<ExprKey> {
    let operands = [ValueId(0), ValueId(0), ValueId(0)];
    Some(match op {
        Op::Unary { op, operand } => ExprKey::Fixed {
            variant: 1,
            code: op.as_str(),
            ty,
            operands: [*operand, operands[1], operands[2]],
            len: 1,
        },
        Op::Binary { op, lhs, rhs } => {
            let (first, second) = if commutative(*op) && lhs > rhs {
                (*rhs, *lhs)
            } else {
                (*lhs, *rhs)
            };
            ExprKey::Fixed {
                variant: 2,
                code: op.as_str(),
                ty,
                operands: [first, second, operands[2]],
                len: 2,
            }
        }
        Op::Compare { op, lhs, rhs } => {
            let (first, second) = if symmetric(*op) && lhs > rhs {
                (*rhs, *lhs)
            } else {
                (*lhs, *rhs)
            };
            ExprKey::Fixed {
                variant: 3,
                code: op.as_str(),
                ty,
                operands: [first, second, operands[2]],
                len: 2,
            }
        }
        Op::Select { cond, a, b } => ExprKey::Fixed {
            variant: 4,
            code: "Select",
            ty,
            operands: [*cond, *a, *b],
            len: 3,
        },
        Op::Convert { op, operand } => ExprKey::Fixed {
            variant: 5,
            code: op.as_str(),
            ty,
            operands: [*operand, operands[1], operands[2]],
            len: 1,
        },
        Op::CopyObject { operand } => ExprKey::Fixed {
            variant: 6,
            code: "CopyObject",
            ty,
            operands: [*operand, operands[1], operands[2]],
            len: 1,
        },
        Op::AccessChain { base, indices } => {
            let mut all = Vec::with_capacity(indices.len() + 1);
            all.push(*base);
            all.extend(indices.iter().copied());
            ExprKey::List {
                variant: 7,
                code: "AccessChain",
                ty,
                operands: all,
            }
        }
        Op::PtrAccessChain { base, indices } => {
            let mut all = Vec::with_capacity(indices.len() + 1);
            all.push(*base);
            all.extend(indices.iter().copied());
            ExprKey::List {
                variant: 8,
                code: "PtrAccessChain",
                ty,
                operands: all,
            }
        }
        Op::CompositeConstruct { constituents } => ExprKey::List {
            variant: 9,
            code: "CompositeConstruct",
            ty,
            operands: constituents.clone(),
        },
        Op::CompositeExtract { composite, indices } => ExprKey::Extract {
            variant: 10,
            code: "CompositeExtract",
            ty,
            composite: *composite,
            indices: indices.clone(),
        },
        _ => return None,
    })
}

/// Shares the computations of every function.
pub(crate) fn run(module: &mut Module) -> Result<()> {
    let functions: Vec<ValueId> = module.functions().to_vec();
    for func in functions {
        number_function(module, func)?;
    }
    Ok(())
}

/// Shares the computations of one function.
fn number_function(module: &mut Module, func: ValueId) -> Result<()> {
    let Some(body) = module
        .function(func)
        .and_then(|f| f.body.as_ref())
    else {
        return Ok(());
    };
    let cfg = Cfg::build(body);
    let dom = dominators(&cfg);
    let children = dominator_children(&cfg, &dom);

    // The table of known computations, scoped to the dominator tree.
    let mut table: BTreeMap<ExprKey, ValueId> = BTreeMap::new();
    let mut scopes: Vec<Vec<(ExprKey, Option<ValueId>)>> = Vec::new();
    let mut stack: Vec<(usize, bool)> = Vec::new();
    let mut shared: Vec<(usize, usize, ValueId)> = Vec::new();
    stack.push((0usize, false));
    while let Some((block, leaving)) = stack.pop() {
        if leaving {
            if let Some(saved) = scopes.pop() {
                for (key, previous) in saved {
                    match previous {
                        Some(value) => {
                            table.insert(key, value);
                        }
                        None => {
                            table.remove(&key);
                        }
                    }
                }
            }
            continue;
        }
        let mut saved: Vec<(ExprKey, Option<ValueId>)> = Vec::new();
        if let Some(block_def) = body.get(block) {
            for (index, inst) in block_def.insts.iter().enumerate() {
                let Some(result) = inst.result else {
                    continue;
                };
                let Some(ty) = inst.ty else {
                    continue;
                };
                let Some(key) = key_of(&inst.op, ty) else {
                    continue;
                };
                if let Some(known) = table.get(&key).copied() {
                    // The known computation dominates this position:
                    // it comes from this block or from an ancestor.
                    shared.push((block, index, known));
                    continue;
                }
                let previous = table.insert(key.clone(), result);
                saved.push((key, previous));
            }
        }
        scopes.push(saved);
        stack.push((block, true));
        if let Some(list) = children.get(block) {
            for &child in list.iter().rev() {
                stack.push((child, false));
            }
        }
    }
    if shared.is_empty() {
        return Ok(());
    }

    // The rewrite: the repeated computation becomes a copy of the
    // computation that already exists.
    let Some(function) = module.function_mut(func) else {
        return Ok(());
    };
    let Some(body) = function.body.as_mut() else {
        return Ok(());
    };
    for (block, index, known) in shared {
        let Some(block) = body.get_mut(block) else {
            continue;
        };
        let Some(inst) = block.insts.get_mut(index) else {
            continue;
        };
        inst.op = Op::CopyObject { operand: known };
    }
    Ok(())
}
