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

//! Lowers a [`Determinized`] DFA to a [`codevar_ocl_ir::ir::Module`] that
//! implements the DFA match loop.  The generated function follows the
//! Negex ABI: it takes a single `i64` (`*mut MatchItem`) argument, reads
//! the input buffer, its length and the DFA transition table from it, runs
//! the DFA, and writes the result back.
//!
//! The generated function `negex_match(i64)` has this structure:
//!
//! - `entry`: load input pointer/length and the transition table from the
//!   MatchItem, clear the result span, then branch to `loop` when the input
//!   is non-empty and to `tail` when it is not.
//! - `loop`: the only hot block.  `off` (the input offset) and `row` (the
//!   current table row) arrive as phis; load the byte at `off`, read the
//!   entry it indexes in `row`, and branch to `rec` when the entry's sign bit
//!   says the *current* state accepts.
//! - `rec`: store `off` as the accepted end, then fall into `next`.
//! - `next`: turn the entry's row field into the next `row` pointer with one
//!   scaled index, advance `off`, and branch back to `loop` while `off` is
//!   still inside the input.
//! - `tail`: end of input; read the accept flag of the row reached and branch
//!   to `tail_hit` or `tail_miss`.
//! - `tail_hit`: the whole tail matched, so report `end = len`.
//! - `tail_miss`: when `rec` published an accepted end, branch to
//!   `miss_found`; otherwise return with the span cleared.
//! - `miss_found`: report the end `rec` published.
//! - `ret`: return.
//!
//! The acceptance flag and the target row ride in the same table entry, so
//! the loop never dispatches on a state number and never tests for a dead
//! transition — a dead transition names the all-zero row, which points back
//! at itself until the input runs out.  The module therefore has a fixed
//! shape — nine blocks — whatever the DFA's state count.  See
//! [`transition_table`].
//!
//! Because the loop records every accepting prefix, the reported match is
//! the longest prefix accepted from the first byte.  Unanchored searching
//! is layered on top by the engine: it finds the leftmost start with the
//! reverse automaton and then calls this function on the sub-slice.
//!
//! Because each loop iteration consumes exactly one byte and there are no
//! epsilon transitions or backtracking, matching is linear in the input
//! length — ReDOS-resistant.
//!
//! # ReDOS prevention
//!
//! I-Regexp (RFC 9485) has no backreferences, lookahead, or general
//! backtracking, so subset construction produces a DFA whose matching cost
//! is `O(input length)` regardless of quantifier nesting.  The number of
//! DFA states is capped at [`super::MAX_DFA_STATES`] and range quantifier
//! expansion is capped at [`super::MAX_RANGE_EXPANSION`], so construction
//! is bounded too.

use super::Determinized;
use crate::NegexError;
use alloc::vec;
use alloc::vec::Vec;
use codevar_logger::log_debug;
use codevar_negex_abi::{MatchItemField, TRANS_TABLE_ROW};
use codevar_ocl_ir::ir::{
    BinOp, BlockId, CmpOp, ConstValue, ConvOp, Inst, Linkage, Module, Op, Storage, Target, TypeId, ValueId,
};

/// The Negex match function name.
const MATCH_FN: &str = "negex_match";

/// MatchItem field offsets, read from the ABI crate so the layout and the
/// generated accesses cannot drift apart.
const OFF_INPUT_PTR: u64 = MatchItemField::Input.offset() as u64;
const OFF_INPUT_LEN: u64 = MatchItemField::InputLen.offset() as u64;
const OFF_RESULT: u64 = MatchItemField::Result.offset() as u64;
const OFF_START: u64 = MatchItemField::Start.offset() as u64;
const OFF_END: u64 = MatchItemField::End.offset() as u64;
/// MatchItem `trans_table` offset.
const OFF_TRANS_TABLE: u64 = MatchItemField::TransTable.offset() as u64;

/// The row field of a transition-table entry.
///
/// A row is named by the index of its first entry, so the field holds the
/// flat index of the row a transition leads to; see [`transition_table`].
pub const TRANS_ROW_MASK: u32 = (1 << 31) - 1;

/// Set in an entry whose source state accepts.
///
/// The flag belongs to the state the entry was read *from*, so the matcher
/// learns whether to record an acceptance at the current position without
/// dispatching on the state at all.  It is the entry's sign bit, which makes
/// a signed compare against zero the whole test.
pub const TRANS_ACCEPT: u32 = 1 << 31;

/// Builds the row-major transition table a compiled matcher steps through.
///
/// The table holds one row per state plus a **dead row** at index 0, so state
/// `s` owns row `s + 1` and its entries live at `(s + 1) * TRANS_TABLE_ROW`.
/// An entry is the *flat index of the row to step to*, or-ed with
/// [`TRANS_ACCEPT`] when the entry's source state accepts.  A missing
/// transition names row 0, whose entries are all zero: the dead row therefore
/// self-loops, consumes the rest of the input without ever accepting, and
/// needs no test in the loop at all.
///
/// Naming a row by its first entry index, rather than by a state number, is
/// what keeps the loop short.  Indexed addressing scales an index only by the
/// entry width, so a state id would still need a multiply by the row stride;
/// a flat index turns one scaled address into the row pointer directly.
///
/// A flat row index must fit [`TRANS_ROW_MASK`]; determinization caps the
/// state count at [`crate::MAX_DFA_STATES`], and even the last row's index is
/// far below that.
#[must_use]
pub fn transition_table(dfa: &Determinized) -> Vec<u32> {
    debug_assert!(
        row_index(state_row(dfa.states.len())).unwrap_or(u32::MAX) <= TRANS_ROW_MASK,
        "a table row index must fit the entry's row field"
    );
    // Row 0 is the dead row and stays all zeroes, which is what makes a dead
    // transition self-loop.
    let mut table = vec![0u32; state_row(dfa.states.len()) * TRANS_TABLE_ROW];
    for (index, state) in dfa.states.iter().enumerate() {
        let accept = if state.accept { TRANS_ACCEPT } else { 0 };
        let row = state_row(index);
        let Some(entries) = table.get_mut(row * TRANS_TABLE_ROW..row * TRANS_TABLE_ROW + TRANS_TABLE_ROW)
        else {
            continue;
        };
        for (byte, entry) in entries.iter_mut().enumerate() {
            let target = state
                .transitions
                .get(byte)
                .copied()
                .flatten()
                .map_or(0, |next| row_index(state_row(next)).unwrap_or(0));
            *entry = accept | target;
        }
    }
    table
}

/// The index of the table row `state` owns.
///
/// Row 0 belongs to the dead row, so the states are numbered from one.
const fn state_row(state: usize) -> usize {
    state + 1
}

/// The flat table index of row `row`, or `None` when it would not fit an
/// entry's row field.
const fn row_index(row: usize) -> Option<u32> {
    match row.checked_mul(TRANS_TABLE_ROW) {
        Some(flat) if flat <= TRANS_ROW_MASK as usize => Some(flat as u32),
        _ => None,
    }
}

/// Whether `entry` was read from an accepting state.
#[must_use]
#[inline]
pub const fn entry_accepts(entry: u32) -> bool {
    entry & TRANS_ACCEPT != 0
}

/// The flat index of the row `entry` leads to.
///
/// Zero names the dead row, so a missing transition needs no separate flag:
/// reading a row that points back at itself consumes the input until it ends.
#[must_use]
#[inline]
pub const fn entry_row(entry: u32) -> u32 {
    entry & TRANS_ROW_MASK
}

/// Lowers `dfa` to an IR module with one entry-point function that
/// implements DFA matching against the input buffer in a `MatchItem`.
///
/// The generated function is anchored at the first input byte and records
/// the *longest* accepting prefix: each accepting state stores its input
/// offset into the accepted-end slot, and the final state is checked once
/// more at end of input.  A match that is not extended by later bytes
/// therefore still wins over a later rejection.
///
/// # Errors
///
/// Returns an error when the DFA has no states or when the module cannot
/// be built.
pub fn lower_dfa(dfa: &Determinized) -> Result<Module, NegexError> {
    if dfa.states.is_empty() {
        return Err(NegexError::EmptyDfa);
    }

    log_debug!("lower: lowering DFA with {} states", dfa.states.len());

    let mut module = Module::new(Target::opencl());

    // Types.
    let void = module.void_ty();
    let i64 = module.int_ty(64, true);
    let u64 = module.int_ty(64, false);
    let i32 = module.int_ty(32, true);
    let i8 = module.int_ty(8, true);
    let ptr_i64 = module.ptr_ty(Storage::Function, i64);
    let ptr_i32 = module.ptr_ty(Storage::Function, i32);
    let ptr_byte = module.ptr_ty(Storage::Function, i8);
    // The accepted-end slot holds a byte pointer, so it is a pointer to one.
    let ptr_end = module.ptr_ty(Storage::Function, ptr_byte);
    let ptr_match_item = module.ptr_ty(Storage::Function, i64); // opaque pointer to MatchItem

    // Function takes a pointer to MatchItem (opaque i64 pointer)
    let sig = module.fn_ty(void, vec![ptr_match_item]);
    let func = module
        .add_function(MATCH_FN, sig, Linkage::External)
        .map_err(|source| NegexError::Module {
            operation: "add_function",
            source,
        })?;
    module
        .begin_body(func)
        .map_err(|source| NegexError::Module {
            operation: "begin_body",
            source,
        })?;

    let entry = push_block(&mut module, func, "entry")?;
    let loop_bb = push_block(&mut module, func, "loop")?;
    let rec = push_block(&mut module, func, "rec")?;
    let next = push_block(&mut module, func, "next")?;
    let tail = push_block(&mut module, func, "tail")?;
    let tail_hit = push_block(&mut module, func, "tail_hit")?;
    let tail_miss = push_block(&mut module, func, "tail_miss")?;
    let found = push_block(&mut module, func, "miss_found")?;
    let ret_bb = push_block(&mut module, func, "ret")?;

    let arg0 = function_arg(&module, func, 0).ok_or(NegexError::MissingMatchArgument)?;

    // ---- entry ----
    // The byte pointer, its row, and the bytes left are the loop's carried
    // values and travel as phis.  Only integers may be compared, so the bound
    // is a countdown rather than an end pointer: the pointer feeds the load
    // directly and the counter feeds the loop test, and neither needs an
    // address computed from a base each byte.  Every value that stays live
    // across the back edge costs the allocator a register it cannot give up,
    // so the loop carries three and pins one more (`table_base`).
    // The accepted end is a slot instead: `rec` writes it and only the tail
    // reads it, which keeps it out of the register file while the loop runs.
    let one = const_value(&mut module, u64, 1)?;
    let zero_u = const_value(&mut module, u64, 0)?;
    let zero32 = const_i32(&mut module, 0)?;
    let row_mask = const_i32(&mut module, u64::from(TRANS_ROW_MASK))?;

    let end_slot = alloc_slot(&mut module, func, entry, ptr_end)?;
    let all_ones = const_value(&mut module, i64, u64::MAX)?;
    let no_end = int_to_ptr(&mut module, func, entry, ptr_byte, all_ones)?;
    store(&mut module, func, entry, end_slot, no_end)?;

    // Clear the reported span; the tail overwrites it on a match.
    write_const_field(&mut module, func, entry, arg0, OFF_START, 0)?;
    write_const_field(&mut module, func, entry, arg0, OFF_END, 0)?;
    write_const_field(&mut module, func, entry, arg0, OFF_RESULT, 0)?;

    let input = load_field(&mut module, func, entry, arg0, OFF_INPUT_PTR, i64, ptr_i64)?;
    let pc0 = int_to_ptr(&mut module, func, entry, ptr_byte, input)?;
    let len = load_field(&mut module, func, entry, arg0, OFF_INPUT_LEN, i32, ptr_i32)?;
    let len_wide = zext(&mut module, func, entry, u64, len)?;
    let table = load_field(&mut module, func, entry, arg0, OFF_TRANS_TABLE, i64, ptr_i64)?;
    let table_base = int_to_ptr(&mut module, func, entry, ptr_i32, table)?;
    let start_row = const_value(
        &mut module,
        u64,
        u64::from(row_index(state_row(dfa.start)).unwrap_or(0)),
    )?;
    let row0 = access_chain(&mut module, func, entry, ptr_i32, table_base, &[start_row])?;

    let in_bounds = compare(&mut module, func, entry, CmpOp::ULessThan, zero_u, len_wide)?;
    branch_cond(&mut module, func, entry, in_bounds, loop_bb, tail)?;

    // ---- loop: the byte at `pc` indexes the current row ----
    // The values `next` defines are named here, before the phis that read
    // them, because a phi has to head its block.
    let pc_next = new_value(&mut module, func, next, ptr_byte);
    let row_next = new_value(&mut module, func, next, ptr_i32);
    let left_next = new_value(&mut module, func, next, u64);
    let pc = phi(
        &mut module,
        func,
        loop_bb,
        ptr_byte,
        &[(pc0, entry), (pc_next, next)],
    )?;
    let row = phi(
        &mut module,
        func,
        loop_bb,
        ptr_i32,
        &[(row0, entry), (row_next, next)],
    )?;
    let left = phi(
        &mut module,
        func,
        loop_bb,
        u64,
        &[(len_wide, entry), (left_next, next)],
    )?;
    let byte = load(&mut module, func, loop_bb, i8, pc)?;
    let byte_index = zext(&mut module, func, loop_bb, u64, byte)?;
    let cell = access_chain(&mut module, func, loop_bb, ptr_i32, row, &[byte_index])?;
    let entry_val = load(&mut module, func, loop_bb, i32, cell)?;
    // The accept flag is the entry's sign bit, so a signed compare against
    // zero is the whole test.
    let accepted = compare(&mut module, func, loop_bb, CmpOp::SLessThan, entry_val, zero32)?;
    branch_cond(&mut module, func, loop_bb, accepted, rec, next)?;

    // ---- rec: publish this byte position as the accepted end ----
    store(&mut module, func, rec, end_slot, pc)?;
    branch(&mut module, func, rec, next)?;

    // ---- next: step to the entry's row and move on ----
    // The entry already names a row by its flat index, so the row address is
    // one scaled index and the dead row's zero points back at itself: the
    // scan cannot die early, but it needs no death test either.
    let row_index = binary(
        &mut module,
        func,
        next,
        i32,
        BinOp::BitwiseAnd,
        entry_val,
        row_mask,
    )?;
    let row_flat = zext(&mut module, func, next, u64, row_index)?;
    access_chain_into(
        &mut module,
        func,
        next,
        row_next,
        ptr_i32,
        table_base,
        &[row_flat],
    )?;
    access_chain_into(&mut module, func, next, pc_next, ptr_byte, pc, &[one])?;
    binary_into(&mut module, func, next, left_next, u64, BinOp::ISub, left, one)?;
    let more = compare(&mut module, func, next, CmpOp::INotEqual, left_next, zero_u)?;
    branch_cond(&mut module, func, next, more, loop_bb, tail)?;

    // ---- tail: end of input ----
    // A row reached exactly at end of input has not been published yet, and
    // every entry of a row carries its own state's accept flag, so the row's
    // first entry answers whether the match ends here.
    let row_here = phi(
        &mut module,
        func,
        tail,
        ptr_i32,
        &[(row0, entry), (row_next, next)],
    )?;
    let final_entry = load(&mut module, func, tail, i32, row_here)?;
    let ends_here = compare(&mut module, func, tail, CmpOp::SLessThan, final_entry, zero32)?;
    branch_cond(&mut module, func, tail, ends_here, tail_hit, tail_miss)?;

    // ---- tail_hit: the last state accepts, so the whole tail matched ----
    // Each accepting path writes its own span, so no value has to cross
    // blocks to be chosen here.
    let len32 = trunc(&mut module, func, tail_hit, i32, len_wide)?;
    write_const_field(&mut module, func, tail_hit, arg0, OFF_RESULT, 1)?;
    write_field(&mut module, func, tail_hit, arg0, OFF_END, len32)?;
    branch(&mut module, func, tail_hit, ret_bb)?;

    // ---- tail_miss: the published end stands ----
    let end_ptr = load(&mut module, func, tail_miss, ptr_byte, end_slot)?;
    let end_addr = ptr_to_int(&mut module, func, tail_miss, i64, end_ptr)?;
    let matched = compare(&mut module, func, tail_miss, CmpOp::INotEqual, end_addr, all_ones)?;
    branch_cond(&mut module, func, tail_miss, matched, found, ret_bb)?;

    // ---- found: the accepted end is a span from the anchored byte ----
    let base_addr = ptr_to_int(&mut module, func, found, i64, pc0)?;
    let span = binary(&mut module, func, found, i64, BinOp::ISub, end_addr, base_addr)?;
    let end32 = trunc(&mut module, func, found, i32, span)?;
    write_const_field(&mut module, func, found, arg0, OFF_RESULT, 1)?;
    write_field(&mut module, func, found, arg0, OFF_END, end32)?;
    branch(&mut module, func, found, ret_bb)?;

    // ---- ret block ----
    emit(&mut module, func, ret_bb, Inst::none(Op::Return))?;
    Ok(module)
}

/// Declares a function-space slot (`alloca`) in `block` and returns its
/// pointer.
fn alloc_slot(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ptr_ty: TypeId,
) -> Result<ValueId, NegexError> {
    let slot = new_value(module, func, block, ptr_ty);
    emit(
        module,
        func,
        block,
        Inst::def(slot, ptr_ty, Op::Variable { init: None }),
    )?;
    Ok(slot)
}

/// Loads the value `ptr` points at.
fn load(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ty: TypeId,
    ptr: ValueId,
) -> Result<ValueId, NegexError> {
    let val = new_value(module, func, block, ty);
    emit(module, func, block, Inst::def(val, ty, Op::Load { ptr }))?;
    Ok(val)
}

/// Zero-extends `operand` to `ty`.
fn zext(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ty: TypeId,
    operand: ValueId,
) -> Result<ValueId, NegexError> {
    let val = new_value(module, func, block, ty);
    emit(
        module,
        func,
        block,
        Inst::def(
            val,
            ty,
            Op::Convert {
                op: ConvOp::UConvert,
                operand,
            },
        ),
    )?;
    Ok(val)
}

/// Addresses `base` plus `indices`, each scaled by the width of the level it
/// walks into, and writes the pointer to `dest`.
///
/// A back end that can scale an index folds one such index into a single
/// address, which is why the loop reads a table cell with this instead of a
/// shift and an add.
fn access_chain_into(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    dest: ValueId,
    ty: TypeId,
    base: ValueId,
    indices: &[ValueId],
) -> Result<(), NegexError> {
    emit(
        module,
        func,
        block,
        Inst::def(
            dest,
            ty,
            Op::PtrAccessChain {
                base,
                indices: indices.to_vec(),
            },
        ),
    )
}

/// [`access_chain_into`] with a fresh result.
fn access_chain(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ty: TypeId,
    base: ValueId,
    indices: &[ValueId],
) -> Result<ValueId, NegexError> {
    let val = new_value(module, func, block, ty);
    access_chain_into(module, func, block, val, ty, base, indices)?;
    Ok(val)
}

/// Compares `lhs` and `rhs` and names the boolean result.
fn compare(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    op: CmpOp,
    lhs: ValueId,
    rhs: ValueId,
) -> Result<ValueId, NegexError> {
    let bool = module.bool_ty();
    let val = new_value(module, func, block, bool);
    emit(
        module,
        func,
        block,
        Inst::def(val, bool, Op::Compare { op, lhs, rhs }),
    )?;
    Ok(val)
}

/// Applies `op` to `lhs` and `rhs` and names the result.
fn binary(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ty: TypeId,
    op: BinOp,
    lhs: ValueId,
    rhs: ValueId,
) -> Result<ValueId, NegexError> {
    let val = new_value(module, func, block, ty);
    binary_into(module, func, block, val, ty, op, lhs, rhs)?;
    Ok(val)
}

/// Applies `op` to `lhs` and `rhs`, writing the result into `dest`.
///
/// `dest` is named by the caller so a pre-created value — the one a phi in an
/// earlier block already refers to — can be defined where it is live.
fn binary_into(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    dest: ValueId,
    ty: TypeId,
    op: BinOp,
    lhs: ValueId,
    rhs: ValueId,
) -> Result<(), NegexError> {
    emit(
        module,
        func,
        block,
        Inst::def(dest, ty, Op::Binary { op, lhs, rhs }),
    )
}

/// Reinterprets the integer `operand` as a pointer of `ty`.
fn int_to_ptr(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ty: TypeId,
    operand: ValueId,
) -> Result<ValueId, NegexError> {
    let val = new_value(module, func, block, ty);
    emit(module, func, block, Inst::def(val, ty, Op::IntToPtr { operand }))?;
    Ok(val)
}

/// Reinterprets the pointer `operand` as an integer of `ty`.
fn ptr_to_int(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ty: TypeId,
    operand: ValueId,
) -> Result<ValueId, NegexError> {
    let val = new_value(module, func, block, ty);
    emit(module, func, block, Inst::def(val, ty, Op::PtrToInt { operand }))?;
    Ok(val)
}

/// Narrows `operand` to `ty`.
fn trunc(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ty: TypeId,
    operand: ValueId,
) -> Result<ValueId, NegexError> {
    let val = new_value(module, func, block, ty);
    emit(
        module,
        func,
        block,
        Inst::def(
            val,
            ty,
            Op::Convert {
                op: ConvOp::UConvert,
                operand,
            },
        ),
    )?;
    Ok(val)
}

/// Merges `incomings` at the head of `block`.
///
/// A phi has to lead its block, so the values a later block defines are named
/// before that block is built.
fn phi(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ty: TypeId,
    incomings: &[(ValueId, BlockId)],
) -> Result<ValueId, NegexError> {
    let val = new_value(module, func, block, ty);
    emit(
        module,
        func,
        block,
        Inst::def(
            val,
            ty,
            Op::Phi {
                incomings: incomings.to_vec(),
            },
        ),
    )?;
    Ok(val)
}

fn push_block(module: &mut Module, func: ValueId, name: &str) -> Result<BlockId, NegexError> {
    module
        .push_block(func, name)
        .map_err(|source| NegexError::Module {
            operation: "push_block",
            source,
        })
}

fn emit(module: &mut Module, func: ValueId, block: BlockId, inst: Inst) -> Result<(), NegexError> {
    module
        .emit(func, block, inst)
        .map_err(|source| NegexError::Module {
            operation: "emit",
            source,
        })
}

fn new_value(module: &mut Module, _func: ValueId, _block: BlockId, ty: TypeId) -> ValueId {
    module.new_inst_value(ty)
}

fn function_arg(module: &Module, func: ValueId, index: usize) -> Option<ValueId> {
    module
        .function(func)
        .and_then(|f| f.args.get(index).copied())
}

fn const_i32(module: &mut Module, value: u64) -> Result<ValueId, NegexError> {
    let i32 = module.int_ty(32, true);
    module
        .intern_const(i32, ConstValue::Int(value))
        .map_err(|source| NegexError::Module {
            operation: "intern_const",
            source,
        })
}

fn const_value(module: &mut Module, ty: TypeId, value: u64) -> Result<ValueId, NegexError> {
    module
        .intern_const(ty, ConstValue::Int(value))
        .map_err(|source| NegexError::Module {
            operation: "intern_const",
            source,
        })
}

fn store(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ptr: ValueId,
    value: ValueId,
) -> Result<(), NegexError> {
    emit(module, func, block, Inst::none(Op::Store { ptr, value }))
}

fn load_field(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ptr_arg: ValueId,
    offset: u64,
    load_ty: TypeId,
    ptr_ty: TypeId,
) -> Result<ValueId, NegexError> {
    let addr_ty = module.int_ty(64, true);
    // Convert pointer to integer first
    let base_int = new_value(module, func, block, addr_ty);
    emit(
        module,
        func,
        block,
        Inst::def(base_int, addr_ty, Op::PtrToInt { operand: ptr_arg }),
    )?;
    let off = const_value(module, addr_ty, offset)?;
    let addr = new_value(module, func, block, addr_ty);
    emit(
        module,
        func,
        block,
        Inst::def(
            addr,
            addr_ty,
            Op::Binary {
                op: BinOp::IAdd,
                lhs: base_int,
                rhs: off,
            },
        ),
    )?;
    let typed = new_value(module, func, block, ptr_ty);
    emit(
        module,
        func,
        block,
        Inst::def(typed, ptr_ty, Op::IntToPtr { operand: addr }),
    )?;
    let val = new_value(module, func, block, load_ty);
    emit(
        module,
        func,
        block,
        Inst::def(val, load_ty, Op::Load { ptr: typed }),
    )?;
    Ok(val)
}

fn branch(module: &mut Module, func: ValueId, block: BlockId, target: BlockId) -> Result<(), NegexError> {
    emit(module, func, block, Inst::none(Op::Branch { target }))
}

fn branch_cond(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    cond: ValueId,
    then: BlockId,
    other: BlockId,
) -> Result<(), NegexError> {
    emit(
        module,
        func,
        block,
        Inst::none(Op::BranchConditional { cond, then, other }),
    )
}

/// Stores a value into a 32-bit field of the `MatchItem` at `ptr_arg + offset`.
fn write_field(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ptr_arg: ValueId,
    offset: u64,
    value: ValueId,
) -> Result<(), NegexError> {
    let ptr = field_ptr(module, func, block, ptr_arg, offset)?;
    emit(module, func, block, Inst::none(Op::Store { ptr, value }))
}

/// Stores a constant into a 32-bit field of the `MatchItem`.
fn write_const_field(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ptr_arg: ValueId,
    offset: u64,
    value: u64,
) -> Result<(), NegexError> {
    let i32 = module.int_ty(32, true);
    let val = const_value(module, i32, value)?;
    let ptr = field_ptr(module, func, block, ptr_arg, offset)?;
    emit(module, func, block, Inst::none(Op::Store { ptr, value: val }))
}

/// Computes `ptr_arg + offset` as an `i32*`.
fn field_ptr(
    module: &mut Module,
    func: ValueId,
    block: BlockId,
    ptr_arg: ValueId,
    offset: u64,
) -> Result<ValueId, NegexError> {
    let addr_ty = module.int_ty(64, true);
    let i32 = module.int_ty(32, true);
    let ptr_i32 = module.ptr_ty(Storage::Function, i32);
    // Convert pointer to integer first
    let base_int = new_value(module, func, block, addr_ty);
    emit(
        module,
        func,
        block,
        Inst::def(base_int, addr_ty, Op::PtrToInt { operand: ptr_arg }),
    )?;
    let off = const_value(module, addr_ty, offset)?;
    let addr = new_value(module, func, block, addr_ty);
    emit(
        module,
        func,
        block,
        Inst::def(
            addr,
            addr_ty,
            Op::Binary {
                op: BinOp::IAdd,
                lhs: base_int,
                rhs: off,
            },
        ),
    )?;
    let ptr = new_value(module, func, block, ptr_i32);
    emit(
        module,
        func,
        block,
        Inst::def(ptr, ptr_i32, Op::IntToPtr { operand: addr }),
    )?;
    Ok(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dfa::Determinized;
    use crate::nfa::build;
    use codevar_negex_parse::parse;

    fn build_dfa(pattern: &str) -> Determinized {
        let output = parse(pattern);
        assert!(output.errors.is_empty(), "pattern `{pattern}` must parse cleanly");
        let (list, start) = build(&output.ast).expect("builds");
        Determinized::from_nfa(&list, start).expect("determinizes")
    }

    fn build_and_lower(pattern: &str) -> (Module, usize, usize) {
        let dfa = build_dfa(pattern);
        let module = lower_dfa(&dfa).expect("lowers");
        (
            module,
            dfa.state_count(),
            dfa.states.iter().filter(|s| s.accept).count(),
        )
    }

    fn block_count(module: &Module) -> usize {
        let func = module
            .find_function(MATCH_FN)
            .expect("negex_match function must exist");
        module
            .function(func)
            .and_then(|f| f.body.as_ref())
            .map(Vec::len)
            .expect("lowered function must have a body")
    }

    #[test]
    fn lower_literal_dfa() {
        let (module, states, accepts) = build_and_lower("a");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
        assert!(accepts > 0, "literal must have an accept state");
    }

    #[test]
    fn lower_star_dfa() {
        let (module, states, _accepts) = build_and_lower("a*");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
    }

    #[test]
    fn lower_plus_dfa() {
        let (module, states, _accepts) = build_and_lower("a+");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
    }

    #[test]
    fn lower_question_dfa() {
        let (module, states, _accepts) = build_and_lower("a?");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
    }

    #[test]
    fn lower_alternation_dfa() {
        let (module, states, _accepts) = build_and_lower("a|b");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
    }

    #[test]
    fn lower_range_dfa() {
        let (module, states, _accepts) = build_and_lower("a{2,4}");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
    }

    #[test]
    fn lower_exact_count_dfa() {
        let (module, states, _accepts) = build_and_lower("a{3}");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
    }

    #[test]
    fn lower_empty_dfa() {
        let (module, states, accepts) = build_and_lower("");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
        assert!(accepts > 0, "empty regex matches empty input");
    }

    #[test]
    fn lower_group_dfa() {
        let (module, states, _accepts) = build_and_lower("(ab)");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
    }

    #[test]
    fn lower_quantified_group_dfa() {
        let (module, states, _accepts) = build_and_lower("(ab)*");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
    }

    #[test]
    fn lower_char_class_dfa() {
        let (module, states, _accepts) = build_and_lower("[0-9]");
        assert!(!module.functions().is_empty());
        assert!(states > 0);
    }

    #[test]
    fn lower_rejects_empty_dfa() {
        let dfa = Determinized {
            states: Vec::new(),
            start: 0,
            matches_empty: false,
        };
        assert!(lower_dfa(&dfa).is_err());
    }

    #[test]
    fn state_count_bounded() {
        for pattern in ["a", "a*", "a+", "a?", "a|b", "a{2,4}", "(ab)", "[0-9]"] {
            let (_, states, _accepts) = build_and_lower(pattern);
            assert!(
                states <= crate::MAX_DFA_STATES,
                "`{pattern}`: {states} states exceeds cap"
            );
        }
    }

    #[test]
    fn function_is_negex_match() {
        let (module, _states, _accepts) = build_and_lower("a");
        let func = module
            .find_function("negex_match")
            .expect("negex_match function must exist");
        let name = &module.value(func).name;
        assert_eq!(name, "negex_match");
    }

    /// The table stepping loop has one shape for every DFA: no per-state or
    /// per-class block is emitted any more.
    #[test]
    fn lowering_does_not_grow_with_the_state_count() {
        for pattern in ["a", "a+", "[0-9]+", "(ab|cd)+", "a{2,4}b*", "\\w+\\d*"] {
            let dfa = build_dfa(pattern);
            let module = lower_dfa(&dfa).expect("lowers");
            assert_eq!(
                block_count(&module),
                9,
                "`{pattern}` has {} states but must still lower to nine blocks",
                dfa.state_count()
            );
        }
    }

    #[test]
    fn table_has_one_row_entry_per_byte() {
        for pattern in ["a", "[0-9]+", "(ab|cd)*"] {
            let dfa = build_dfa(pattern);
            let table = transition_table(&dfa);
            // One row per state, plus the dead row the missing transitions
            // point at.
            assert_eq!(
                table.len(),
                (dfa.states.len() + 1) * TRANS_TABLE_ROW,
                "`{pattern}`"
            );
        }
    }

    #[test]
    fn table_entry_names_the_target_row_and_carries_the_source_flag() {
        let dfa = build_dfa("a");
        let table = transition_table(&dfa);
        let row = state_row(dfa.start);
        let on_a = table[row * TRANS_TABLE_ROW + usize::from(b'a')];
        assert!(!entry_accepts(on_a), "the start state of `a` does not accept");
        let target = dfa.states[dfa.start].transitions[usize::from(b'a')].expect("target");
        assert_eq!(
            entry_row(on_a),
            row_index(state_row(target)).expect("row fits"),
            "the entry is the flat index of the target state's row"
        );
        let on_b = table[row * TRANS_TABLE_ROW + usize::from(b'b')];
        assert_eq!(entry_row(on_b), 0, "a byte outside `a` leads to the dead row");
        assert!(!entry_accepts(on_b));
        // The dead row is row zero and reads zero, so a dead transition
        // points back at itself and the loop needs no death test.
        for entry in table.iter().take(TRANS_TABLE_ROW) {
            assert_eq!(*entry, 0, "the dead row must self-loop without accepting");
        }
    }

    #[test]
    fn table_agrees_with_the_dfa_byte_for_byte() {
        for pattern in ["", "a", "a*", "[0-9]+", "(ab|cd)+", "\\w+\\d", "a{2,4}"] {
            let dfa = build_dfa(pattern);
            let table = transition_table(&dfa);
            for (index, state) in dfa.states.iter().enumerate() {
                let row = state_row(index);
                for byte in 0..TRANS_TABLE_ROW {
                    let entry = table[row * TRANS_TABLE_ROW + byte];
                    assert_eq!(entry_accepts(entry), state.accept, "`{pattern}` state {index}");
                    let expected = state
                        .transitions
                        .get(byte)
                        .copied()
                        .flatten()
                        .map_or(0, |target| row_index(state_row(target)).expect("row fits"));
                    assert_eq!(
                        entry_row(entry),
                        expected,
                        "`{pattern}` state {index} byte {byte}"
                    );
                }
            }
        }
    }

    /// The accept flag must not eat into the row field: a row is only
    /// representable while the last row's flat index stays below the mask.
    #[test]
    fn entry_flag_leaves_the_row_field_wide() {
        let last_row = row_index(state_row(crate::MAX_DFA_STATES as usize));
        assert!(last_row.expect("the last row fits the entry field") <= TRANS_ROW_MASK);
        assert_eq!(TRANS_ACCEPT & TRANS_ROW_MASK, 0);
        for row in [0u32, TRANS_TABLE_ROW as u32, last_row.expect("the last row fits")] {
            for flags in [0, TRANS_ACCEPT] {
                let entry = row | flags;
                assert_eq!(entry_row(entry), row, "the flag must not disturb the row");
                assert_eq!(entry_accepts(entry), flags & TRANS_ACCEPT != 0);
            }
        }
    }
}
