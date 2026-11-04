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

//! Sparse conditional constant propagation.
//!
//! The pass follows the constants through the control flow without
//! walking every path of the function:
//!
//! * Every value starts unknown.  A value becomes a constant when the
//!   operation that defines it has only constant operands, and it
//!   becomes undefined when the block that defines it can never run.
//! * A branch on a known condition makes one of its edges impossible.
//!   The block behind that edge is never executed, so the values it
//!   defines are undefined, and the merge it feeds takes the value of
//!   the edge that can still happen.
//! * A merge whose feasible incoming edges all carry the same constant
//!   is that constant.
//!
//! The pass does not touch the control flow: the branch that folds
//! away is left in place for the control-flow simplification, which
//! removes the block the impossible edge leads to.  A value the pass
//! knows becomes a copy of the constant it holds, so the value keeps
//! its identity and every use of it stays valid.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use codevar_ocl_ir::ir::{ConstValue, Module, Op, TypeId, ValueId};

use crate::error::Result;
use crate::opt::passes::fold::evaluate;
use crate::opt::util::{Cfg, ConstSource, known_payload, replace_with_copy};

/// What the pass knows about a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lattice {
    /// The value is a known constant.
    Const(ConstValue),
    /// The value is never produced.
    Undef,
}

/// The knowledge the pass has, seen by the constant evaluator.
struct Source<'a> {
    /// The module the values belong to.
    module: &'a Module,
    /// What is known about every value.
    states: &'a BTreeMap<ValueId, Lattice>,
}

impl ConstSource for Source<'_> {
    fn payload(&self, value: ValueId) -> Option<ConstValue> {
        match self.states.get(&value) {
            Some(Lattice::Const(constant)) => Some(*constant),
            _ => self.module.constant_value(value),
        }
    }
}

/// Propagates the constants through the control flow.
pub(crate) fn run(module: &mut Module) -> Result<()> {
    let functions: Vec<ValueId> = module.functions().to_vec();
    for func in functions {
        propagate_function(module, func)?;
    }
    Ok(())
}

/// Propagates the constants of one function.
fn propagate_function(module: &mut Module, func: ValueId) -> Result<()> {
    let Some(body) = module
        .function(func)
        .and_then(|f| f.body.as_ref())
    else {
        return Ok(());
    };
    let cfg = Cfg::build(body);
    let mut states: BTreeMap<ValueId, Lattice> = BTreeMap::new();
    // The edges that can still happen.  An edge that a known
    // condition rules out never comes back.
    let mut allowed: Vec<Vec<bool>> = cfg
        .succs
        .iter()
        .map(|successors| alloc::vec![true; successors.len()])
        .collect();

    // The values the blocks that cannot run define are undefined.
    for (block_index, block) in body.iter().enumerate() {
        if cfg
            .reachable
            .get(block_index)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        for inst in &block.insts {
            if let Some(result) = inst.result {
                states.entry(result).or_insert(Lattice::Undef);
            }
        }
    }

    // The fixed point: the constants flow forward, and the branches
    // on a known constant make an edge impossible.
    loop {
        let mut changed = false;
        let source = Source {
            module,
            states: &states,
        };

        // The blocks the surviving edges lead to.
        let mut executable = vec![false; cfg.len()];
        if let Some(flag) = executable.first_mut() {
            *flag = true;
        }
        let mut queue: Vec<usize> = vec![0];
        while let Some(block) = queue.pop() {
            let Some(successors) = cfg.succs.get(block) else {
                continue;
            };
            for (edge, &successor) in successors.iter().enumerate() {
                if !allowed
                    .get(block)
                    .and_then(|list| list.get(edge))
                    .copied()
                    .unwrap_or(false)
                {
                    continue;
                }
                if let Some(flag) = executable.get_mut(successor)
                    && !*flag
                {
                    *flag = true;
                    queue.push(successor);
                }
            }
        }

        // The edges a known condition rules out.
        for (block_index, block) in body.iter().enumerate() {
            if !executable
                .get(block_index)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let Some(last) = block.insts.last() else {
                continue;
            };
            let Op::BranchConditional { cond, .. } = &last.op else {
                continue;
            };
            let Some(ConstValue::Bool(bit)) = known_payload(&source, *cond) else {
                continue;
            };
            let Some(list) = allowed.get_mut(block_index) else {
                continue;
            };
            // The successors of a conditional branch are the true edge
            // and the false edge, in that order.
            let edge = if bit { 1 } else { 0 };
            if let Some(slot) = list.get_mut(edge)
                && *slot
            {
                *slot = false;
                changed = true;
            }
        }

        // The values the executable blocks compute.
        let mut updates: Vec<(ValueId, Lattice)> = Vec::new();
        for (block_index, block) in body.iter().enumerate() {
            if !executable
                .get(block_index)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            for inst in &block.insts {
                let Some(result) = inst.result else {
                    continue;
                };
                let Some(ty) = inst.ty else {
                    continue;
                };
                let lattice = match &inst.op {
                    Op::Phi { incomings } => match phi_lattice(&states, &executable, incomings) {
                        Some(lattice) => lattice,
                        None => continue,
                    },
                    _ => match evaluate(module, &source, &inst.op, ty) {
                        Some(value) => Lattice::Const(value),
                        None => continue,
                    },
                };
                if states.get(&result) != Some(&lattice) {
                    updates.push((result, lattice));
                }
            }
        }
        for (value, lattice) in updates {
            if states.insert(value, lattice) != Some(lattice) {
                changed = true;
            }
        }

        if !changed {
            // The blocks that cannot run define values that are never
            // produced.
            for (block_index, block) in body.iter().enumerate() {
                if executable
                    .get(block_index)
                    .copied()
                    .unwrap_or(false)
                {
                    continue;
                }
                for inst in &block.insts {
                    let Some(result) = inst.result else {
                        continue;
                    };
                    if states.get(&result) != Some(&Lattice::Undef) {
                        states.insert(result, Lattice::Undef);
                    }
                }
            }
            break;
        }
    }

    // The rewrite: a value the pass knows becomes a copy of what it
    // holds.
    let mut rewrites: Vec<(usize, usize, Option<ConstValue>)> = Vec::new();
    for (block_index, block) in body.iter().enumerate() {
        for (inst_index, inst) in block.insts.iter().enumerate() {
            let Some(result) = inst.result else {
                continue;
            };
            match states.get(&result) {
                Some(Lattice::Const(constant)) => {
                    if matches!(inst.op, Op::CopyObject { .. }) {
                        continue;
                    }
                    rewrites.push((block_index, inst_index, Some(*constant)));
                }
                Some(Lattice::Undef) => {
                    if matches!(inst.op, Op::CopyObject { .. }) {
                        continue;
                    }
                    rewrites.push((block_index, inst_index, None));
                }
                _ => {}
            }
        }
    }
    if rewrites.is_empty() {
        return Ok(());
    }

    // The constants the rewrites need.
    let mut wanted: Vec<(usize, usize, TypeId, ConstValue)> = Vec::with_capacity(rewrites.len());
    for (block_index, inst_index, value) in &rewrites {
        let Some(inst) = body
            .get(*block_index)
            .and_then(|b| b.insts.get(*inst_index))
        else {
            continue;
        };
        let Some(ty) = inst.ty else {
            continue;
        };
        let payload = value.unwrap_or(ConstValue::Undef);
        wanted.push((*block_index, *inst_index, ty, payload));
    }
    let mut constants: Vec<(usize, usize, ValueId)> = Vec::with_capacity(wanted.len());
    for (block_index, inst_index, ty, payload) in wanted {
        let Ok(constant) = module.intern_const(ty, payload) else {
            continue;
        };
        constants.push((block_index, inst_index, constant));
    }

    let Some(function) = module.function_mut(func) else {
        return Ok(());
    };
    let Some(body) = function.body.as_mut() else {
        return Ok(());
    };
    for (block_index, inst_index, constant) in constants {
        let Some(block) = body.get_mut(block_index) else {
            continue;
        };
        let Some(inst) = block.insts.get_mut(inst_index) else {
            continue;
        };
        replace_with_copy(inst, constant);
    }
    Ok(())
}

/// The value a merge holds, when every edge that can happen agrees.
fn phi_lattice(
    states: &BTreeMap<ValueId, Lattice>,
    executable: &[bool],
    incomings: &[(ValueId, codevar_ocl_ir::ir::BlockId)],
) -> Option<Lattice> {
    let mut value: Option<ConstValue> = None;
    let mut agreeing = true;
    for (incoming, predecessor) in incomings {
        if !executable
            .get(predecessor.index())
            .copied()
            .unwrap_or(false)
        {
            // The edge cannot happen: the incoming does not count.
            continue;
        }
        match states.get(incoming) {
            Some(Lattice::Const(constant)) => match value {
                Some(previous) if previous != *constant => agreeing = false,
                _ => value = Some(*constant),
            },
            Some(Lattice::Undef) => return Some(Lattice::Undef),
            _ => agreeing = false,
        }
    }
    if !agreeing {
        return None;
    }
    Some(match value {
        Some(constant) => Lattice::Const(constant),
        None => Lattice::Undef,
    })
}
