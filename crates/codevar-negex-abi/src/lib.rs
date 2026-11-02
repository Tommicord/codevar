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

//! The regex matching ABI, shared by the Negex engine and the native
//! machine code it compiles.
//!
//! One contract is shared by the two sides of a match: the engine, which
//! owns the input buffer and the match state, and the generated machine
//! code, which scans the input and writes the match result.  The contract
//! lives in its own crate so neither side can drift: the engine includes
//! this definition, and the generated code addresses the very same fields
//! by the offsets published here.
//!
//! # The call
//!
//! A compiled regex is an `unsafe extern "C" fn(*mut MatchItem)`: the
//! pointer names a per-match state block that lives on the launching
//! core's stack for the duration of the call.  The generated code reads
//! the input buffer and the DFA transition table from it and writes the
//! match result back.  The table is owned by the engine, which binds it
//! with [`MatchItem::with_table`]; it must stay alive for the call.
//!
//! # Example
//!
//! ```
//! use codevar_negex_abi::{MatchItem, MAX_CAPTURES};
//!
//! let mut item = MatchItem::new(b"hello world");
//! assert_eq!(item.input_len, 11);
//! assert_eq!(MAX_CAPTURES, 16);
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

use core::ffi::c_void;

/// The invocation type of a compiled regex — the single ABI every match
/// dispatches through.
///
/// # Safety
///
/// Implementations receive a fully initialized [`MatchItem`] whose
/// `input` points at `input_len` bytes of readable memory; they must not
/// escape the match item beyond the call.
pub type MatchFn = unsafe extern "C" fn(*mut MatchItem);

/// Maximum number of capture groups a compiled regex may report.
pub const MAX_CAPTURES: usize = 16;

/// Maximum number of input chunks the streaming matcher holds.
pub const MAX_STREAM_CHUNKS: usize = 8;

/// Entries per state in a DFA transition table: one per input byte.
pub const TRANS_TABLE_ROW: usize = 256;

/// Per-match state handed to a compiled regex.
///
/// `#[repr(C)]`, documented field by field because generated code and the
/// engine address these offsets directly — changing the order breaks them.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MatchItem {
    /// Pointer to the input buffer.
    pub input: *const u8,
    /// Length of the input buffer in bytes.
    pub input_len: u32,
    /// Match result: `1` when the input matches, `0` otherwise.
    pub result: u32,
    /// Start offset of the match (valid when `result == 1`).
    pub start: u32,
    /// End offset of the match (valid when `result == 1`).
    pub end: u32,
    /// Capture group offsets: pairs of `[start, end]` per capture.
    pub captures: [(u32, u32); MAX_CAPTURES],
    /// Number of captures that participated in the match.
    pub capture_count: u32,
    /// Reserved for future use (streaming state, etc.).
    pub reserved: [u32; 4],
    /// The DFA transition table a compiled matcher steps through.
    ///
    /// Row-major: the row for state `s` starts at
    /// `(s + 1) * TRANS_TABLE_ROW`, so the table holds
    /// `(state_count + 1) * TRANS_TABLE_ROW` entries.  Row 0 is the dead row
    /// that points at itself.  Each entry names the target state's row and
    /// carries the source state's accept flag; see the
    /// `transition_table` builder in `codevar_negex_codegen` for the exact
    /// encoding.
    ///
    /// A native matcher dereferences it once per input byte, so a match
    /// through generated code requires a non-null, live table.  It is null
    /// for an item built by [`MatchItem::new`], which only the interpreter
    /// — which never reads it — can run.
    pub trans_table: *const u32,
}

impl MatchItem {
    /// Builds a match item for `input`, with all result fields cleared and
    /// no transition table.
    ///
    /// Generated machine code dereferences [`MatchItem::trans_table`] once
    /// per byte, so an item built here is only a container for input and
    /// result: run it through the interpreter, never through a trampoline.
    /// Use [`MatchItem::with_table`] to bind a table.
    #[must_use]
    pub fn new(input: &[u8]) -> Self {
        Self {
            input: input.as_ptr(),
            input_len: u32::try_from(input.len()).unwrap_or(u32::MAX),
            result: 0,
            start: 0,
            end: 0,
            captures: [(0, 0); MAX_CAPTURES],
            capture_count: 0,
            reserved: [0; 4],
            trans_table: core::ptr::null(),
        }
    }

    /// Builds a match item for `input` that steps through `table`.
    ///
    /// `table` must be the transition table of the DFA the code was
    /// compiled from — `state_count * TRANS_TABLE_ROW` entries — and must
    /// outlive the item; the binding is a raw pointer because the ABI
    /// struct cannot own it.
    #[must_use]
    pub fn with_table(input: &[u8], table: &[u32]) -> Self {
        let mut item = Self::new(input);
        item.trans_table = table.as_ptr();
        item
    }

    /// Returns the input buffer as a byte slice.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `input` is valid for `input_len`
    /// reads for the duration of the call.
    #[must_use]
    #[inline]
    pub unsafe fn input_slice(&self) -> &[u8] {
        // SAFETY: the engine guarantees `input` points at `input_len`
        // readable bytes for the duration of the match call.
        unsafe { core::slice::from_raw_parts(self.input, self.input_len as usize) }
    }
}

impl Default for MatchItem {
    fn default() -> Self {
        Self {
            input: core::ptr::null(),
            input_len: 0,
            result: 0,
            start: 0,
            end: 0,
            captures: [(0, 0); MAX_CAPTURES],
            capture_count: 0,
            reserved: [0; 4],
            trans_table: core::ptr::null(),
        }
    }
}

/// A field of the per-match state a compiled regex can read or write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchItemField {
    /// The input buffer pointer.
    Input,
    /// The input length in bytes.
    InputLen,
    /// The match result (`1` = match, `0` = no match).
    Result,
    /// The match start offset.
    Start,
    /// The match end offset.
    End,
    /// The DFA transition table pointer.
    TransTable,
}

impl MatchItemField {
    /// Every field, in layout order.
    pub const ALL: [Self; 6] = [
        Self::Input,
        Self::InputLen,
        Self::Result,
        Self::Start,
        Self::End,
        Self::TransTable,
    ];

    /// Byte offset of the field inside [`MatchItem`].
    #[must_use]
    #[inline]
    pub const fn offset(self) -> u32 {
        match self {
            Self::Input => 0,
            Self::InputLen => core::mem::size_of::<*const u8>() as u32,
            Self::Result => (core::mem::size_of::<*const u8>() + core::mem::size_of::<u32>()) as u32,
            Self::Start => (core::mem::size_of::<*const u8>() + 2 * core::mem::size_of::<u32>()) as u32,
            Self::End => (core::mem::size_of::<*const u8>() + 3 * core::mem::size_of::<u32>()) as u32,
            Self::TransTable => TRANS_TABLE_OFFSET,
        }
    }
}

/// Byte offset of the captures array inside [`MatchItem`].
pub const CAPTURES_OFFSET: u32 = (core::mem::size_of::<*const u8>() + 5 * core::mem::size_of::<u32>()) as u32;

/// Byte offset of the transition table pointer inside [`MatchItem`].
///
/// Taken from the layout itself: the field sits past `reserved`, which
/// leaves four bytes of padding before it on a 64-bit target.
pub const TRANS_TABLE_OFFSET: u32 = core::mem::offset_of!(MatchItem, trans_table) as u32;

/// A null match-item pointer, used by the engine to mark an unbound match.
pub const NULL_MATCH_ITEM: *mut c_void = core::ptr::null_mut();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_item_layout_is_stable() {
        let item = MatchItem::new(b"hello");
        let base = &item as *const MatchItem as usize;
        let input = &item.input as *const *const u8 as usize;
        assert_eq!((input - base) as u32, MatchItemField::Input.offset());
        let len = &item.input_len as *const u32 as usize;
        assert_eq!((len - base) as u32, MatchItemField::InputLen.offset());
    }

    #[test]
    fn captures_offset_is_after_scalars() {
        assert_eq!(
            CAPTURES_OFFSET,
            (core::mem::size_of::<*const u8>() + 5 * core::mem::size_of::<u32>()) as u32
        );
    }

    #[test]
    fn new_clears_result_fields() {
        let item = MatchItem::new(b"abc");
        assert_eq!(item.result, 0);
        assert_eq!(item.start, 0);
        assert_eq!(item.end, 0);
        assert_eq!(item.capture_count, 0);
    }

    #[test]
    fn new_leaves_no_table_bound() {
        assert!(MatchItem::new(b"abc").trans_table.is_null());
        assert!(MatchItem::default().trans_table.is_null());
    }

    #[test]
    fn with_table_binds_the_table_pointer() {
        let table = [0u32; TRANS_TABLE_ROW];
        let item = MatchItem::with_table(b"abc", &table);
        assert_eq!(item.trans_table, table.as_ptr());
        assert_eq!(item.input_len, 3);
    }

    #[test]
    fn trans_table_offset_is_the_field_address() {
        let table = [0u32; TRANS_TABLE_ROW];
        let item = MatchItem::with_table(b"abc", &table);
        let base = &item as *const MatchItem as usize;
        let field = &item.trans_table as *const *const u32 as usize;
        assert_eq!((field - base) as u32, MatchItemField::TransTable.offset());
        assert!(
            MatchItemField::ALL
                .windows(2)
                .all(|pair| pair[0].offset() < pair[1].offset()),
            "every hot field must precede the table"
        );
    }
}
