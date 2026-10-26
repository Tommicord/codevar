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

//! Removal of the computations nobody uses.
//!
//! An instruction goes when nothing uses its result and the operation
//! has no effect of its own.  The stores, the calls, the extended
//! instructions, and the barriers stay whatever their result is worth,
//! because they change what the program does.  A value a merge reads
//! is a use like any other, so the computation that produced it stays
//! too.
//!
//! The removal is repeated until nothing changes: a computation that
//! only fed another dead computation goes as well.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{BasicBlock, ConstValue, Inst, Module, Op, TypeId, ValueId};

use crate::error::Result;
use crate::opt::util::{collect_operands, is_removable, replace_with_copy};

/// Removes the dead computations of every function.
pub(crate) fn run(module: &mut Module) -> Result<()> {
    let functions: Vec<ValueId> = module.functions().to_vec();
    for func in functions {
        eliminate(module, func)?;
    }
    Ok(())
}

/// Removes the dead computations of one function.
fn eliminate(module: &mut Module, func: ValueId) -> Result<()> {
    let mut removed: BTreeSet<ValueId> = BTreeSet::new();
    loop {
        // The values the function still uses.
        let Some(body) = module
            .function(func)
            .and_then(|f| f.body.as_ref())
        else {
            return Ok(());
        };
        let mut used: BTreeSet<ValueId> = BTreeSet::new();
        for block in body {
            for inst in &block.insts {
                let mut operands = Vec::new();
                collect_operands(&inst.op, &mut operands);
                used.extend(operands);
            }
        }

        // The instructions nobody needs.
        let mut dead: Vec<(usize, usize, ValueId, TypeId)> = Vec::new();
        for (block_index, block) in body.iter().enumerate() {
            for (inst_index, inst) in block.insts.iter().enumerate() {
                let Some(result) = inst.result else {
                    continue;
                };
                let Some(ty) = inst.ty else {
                    continue;
                };
                if removed.contains(&result) || used.contains(&result) {
                    continue;
                }
                if !is_removable(&inst.op) {
                    continue;
                }
                dead.push((block_index, inst_index, result, ty));
            }
        }
        if dead.is_empty() {
            return Ok(());
        }

        // The undefined values the rewrite needs.
        let mut undefs: BTreeMap<TypeId, ValueId> = BTreeMap::new();
        for (_, _, _, ty) in &dead {
            if undefs.contains_key(ty) {
                continue;
            }
            let Ok(value) = module.intern_const(*ty, ConstValue::Undef) else {
                continue;
            };
            undefs.insert(*ty, value);
        }

        // The rewrite: the operation becomes a copy of the undefined
        // value, so the value stays defined without the computation.
        let Some(function) = module.function_mut(func) else {
            return Ok(());
        };
        let Some(body) = function.body.as_mut() else {
            return Ok(());
        };
        // The dead instructions of every block, by their position.
        let mut per_block: BTreeMap<usize, Vec<(usize, ValueId, TypeId)>> = BTreeMap::new();
        for (block_index, inst_index, result, ty) in dead {
            per_block
                .entry(block_index)
                .or_default()
                .push((inst_index, result, ty));
        }
        for (block_index, list) in per_block {
            let Some(block) = body.get_mut(block_index) else {
                continue;
            };
            rewrite_block(block, &list, &undefs, &mut removed);
        }
    }
}

/// Rewrites one block: every dead computation becomes a copy of the
/// undefined value.
///
/// A merge block starts with its merges, and a merge that nobody reads
/// is replaced by a copy — which is not a merge, so it moves behind
/// the merges that stay.  The block therefore keeps the shape the
/// verifier expects.
fn rewrite_block(
    block: &mut BasicBlock,
    dead: &[(usize, ValueId, TypeId)],
    undefs: &BTreeMap<TypeId, ValueId>,
    removed: &mut BTreeSet<ValueId>,
) {
    let mut dead_by_index: BTreeMap<usize, (ValueId, TypeId)> = BTreeMap::new();
    for (inst_index, result, ty) in dead {
        dead_by_index.insert(*inst_index, (*result, *ty));
    }
    let old = core::mem::take(&mut block.insts);
    let mut merges: Vec<Inst> = Vec::new();
    let mut rest: Vec<Inst> = Vec::new();
    let mut copies: Vec<Inst> = Vec::new();
    for (inst_index, mut inst) in old.into_iter().enumerate() {
        let is_merge = matches!(inst.op, Op::Phi { .. });
        let dead_entry = dead_by_index.get(&inst_index).copied();
        match (is_merge, dead_entry) {
            (true, Some((result, ty))) => {
                if removed.contains(&result) {
                    continue;
                }
                let Some(undef) = undefs.get(&ty).copied() else {
                    merges.push(inst);
                    continue;
                };
                // The merge goes away; its value stays defined by a
                // copy that sits behind the merges that remain.
                copies.push(Inst::def(result, ty, Op::CopyObject { operand: undef }));
                removed.insert(result);
            }
            (true, None) => merges.push(inst),
            (false, Some((result, ty))) => {
                if removed.contains(&result) {
                    continue;
                }
                let Some(undef) = undefs.get(&ty).copied() else {
                    rest.push(inst);
                    continue;
                };
                replace_with_copy(&mut inst, undef);
                removed.insert(result);
                rest.push(inst);
            }
            (false, None) => rest.push(inst),
        }
    }
    let mut fresh = merges;
    fresh.extend(copies);
    fresh.extend(rest);
    block.insts = fresh;
}
