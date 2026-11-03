// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

//! The operation queue: record layout, operation tags and the
//! reduction from an operation to a boundary row.
//!
//! A batch is a queue of fixed-size [`OpRecord`]s in the ops plane
//! plus the counts the control plane carries. Nothing about the
//! protocol assumes a queue entry is executed in order: the planning
//! pass turns the whole queue into one sorted [`BoundaryRow`] table,
//! which is why the tags here are split into classes. The classes say
//! what a planner does with an entry — whether it contributes a row to
//! the index map, moves a caret, expands into rows taken from retained
//! state, or fills the payload without editing — and every kernel
//! downstream only ever sees the reduced table.
//!
//! The reduction is the reason an edit is cheap on a device: an entry
//! that removes `len` elements and inserts `count` contributes exactly
//! one discontinuity with jump `count - len` at [`OpRecord::pos`], no
//! matter how long the document is.

use super::access::{get_word, put_word};
use super::fields::{BoundaryRow, FLAG_CAPTURE, NO_PAYLOAD};
use super::geometry::{Plane, record_word};
use crate::error::{WreditError, WreditResult};

// ---------------------------------------------------------------------------
// Operation record fields
// ---------------------------------------------------------------------------

/// Field holding the operation tag, one of the `OP_*` kind constants.
pub const OP_KIND: u32 = 0;
/// Field holding the element index the operation applies to, in source
/// plane coordinates.
pub const OP_POS: u32 = 1;
/// Field holding how many source elements the operation removes.
pub const OP_LEN: u32 = 2;
/// Field holding how many payload elements the operation inserts.
pub const OP_COUNT: u32 = 3;
/// Field holding the payload word the inserted run starts at, or
/// [`NO_PAYLOAD`].
pub const OP_PAYLOAD: u32 = 4;
/// Field holding the cursor row the operation is attached to.
pub const OP_CURSOR: u32 = 5;
/// Field holding per-operation flags, a union of the `OP_FLAG_*` bits.
pub const OP_FLAGS: u32 = 6;
/// Reserved field; always written zero.
pub const OP_RESERVED: u32 = 7;

/// Operation fields a kernel is expected to name, for the ABI-drift
/// test: the constant's spelling and the word offset it must carry.
pub const OP_FIELDS: [(&str, u32); 7] = [
    ("OP_KIND", OP_KIND),
    ("OP_POS", OP_POS),
    ("OP_LEN", OP_LEN),
    ("OP_COUNT", OP_COUNT),
    ("OP_PAYLOAD", OP_PAYLOAD),
    ("OP_CURSOR", OP_CURSOR),
    ("OP_FLAGS", OP_FLAGS),
];

// ---------------------------------------------------------------------------
// Operation tags
// ---------------------------------------------------------------------------

/// No operation. The queue is zero-initialised, so an entry the
/// planning pass never filled reads back with this tag, and a kernel
/// skips it without reading any other field.
pub const OP_NOP: u32 = 0;
/// Insert the payload run at [`OP_POS`]: the engine's `put` and
/// `put_shared`.
pub const OP_PUT: u32 = 1;
/// Remove [`OP_LEN`] elements at [`OP_POS`]: `delete`, `delete_range`
/// and `backspace`, which differ only in how the position is computed.
pub const OP_DELETE: u32 = 2;
/// Remove [`OP_LEN`] elements at [`OP_POS`] and insert the [`OP_COUNT`]
/// payload elements there: `replace` and `replace_range`.
pub const OP_REPLACE: u32 = 3;
/// Write the payload run over [`OP_COUNT`] existing elements without
/// changing the length: `overwrite` and `overwrite_slice`. It removes
/// and inserts the same number, so it contributes no jump — only the
/// payload words themselves differ, which is what keeps it a distinct
/// tag.
pub const OP_OVERWRITE: u32 = 4;
/// Copy [`OP_LEN`] source elements at [`OP_POS`] into the payload at
/// [`OP_PAYLOAD`] without editing the document. A delete that wants to
/// stay undoable is a capture followed by a delete.
pub const OP_CAPTURE: u32 = 5;
/// Commit the staging gap owned by [`OP_CURSOR`] as an insertion at
/// [`OP_POS`]: `flush_gap` and `flush_all`. The planner knows the gap's
/// fill and payload from the gap plane, so this tag carries no length.
pub const OP_GAP_COMMIT: u32 = 6;
/// Put the caret of cursor [`OP_CURSOR`] at [`OP_POS`] and its anchor at
/// [`OP_LEN`]: `move_cursor_to`.
pub const OP_CURSOR_SET: u32 = 7;
/// Displace the caret of cursor [`OP_CURSOR`] by the signed element
/// count in [`OP_POS`]: `move_cursor_forward` and `move_cursor_backward`.
/// [`OP_LEN`] is the column the vertical moves aim at, so a horizontal
/// move keeps it zero.
pub const OP_CURSOR_MOVE: u32 = 8;
/// Displace the caret of cursor [`OP_CURSOR`] by the signed *line* count
/// in [`OP_POS`] to column [`OP_LEN`]: `move_cursor_next_line` and
/// `move_cursor_prev_line`. Resolving it needs the newline positions,
/// which the scan pass derives from the source plane.
pub const OP_CURSOR_LINE: u32 = 9;
/// Select the whole document for cursor [`OP_CURSOR`]: `select_all`.
pub const OP_SELECT_ALL: u32 = 10;
/// Replay the retained log backwards. The planner expands it into the
/// inverse rows of the retained boundary table, so a device never
/// walks a history list.
pub const OP_UNDO: u32 = 11;
/// Replay the retained log forwards, after an [`OP_UNDO`].
pub const OP_REDO: u32 = 12;

/// One past the largest tag; a kind at or above this is not an
/// operation the protocol knows.
pub const OP_KIND_LIMIT: u32 = 13;

/// What a planner does with an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// Contributes nothing to any plane.
    Neutral,
    /// Contributes one discontinuity to the index map.
    Edit,
    /// Fills the payload plane; edits nothing.
    Capture,
    /// Contributes a movement to the cursor remap pass.
    Cursor,
    /// Expanded into rows taken from the retained log, not from the
    /// fields of the record itself.
    Replay,
}

impl OpClass {
    /// The class's spelling, for logging.
    #[must_use]
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Neutral => "Neutral",
            Self::Edit => "Edit",
            Self::Capture => "Capture",
            Self::Cursor => "Cursor",
            Self::Replay => "Replay",
        }
    }
}

/// Every tag the protocol knows, with its spelling and class. The
/// kernel re-declares the tags as file-scope constants, so this table
/// is what the ABI-drift test walks.
pub const OP_KINDS: [(&str, u32, OpClass); 13] = [
    ("OP_NOP", OP_NOP, OpClass::Neutral),
    ("OP_PUT", OP_PUT, OpClass::Edit),
    ("OP_DELETE", OP_DELETE, OpClass::Edit),
    ("OP_REPLACE", OP_REPLACE, OpClass::Edit),
    ("OP_OVERWRITE", OP_OVERWRITE, OpClass::Edit),
    ("OP_CAPTURE", OP_CAPTURE, OpClass::Capture),
    ("OP_GAP_COMMIT", OP_GAP_COMMIT, OpClass::Edit),
    ("OP_CURSOR_SET", OP_CURSOR_SET, OpClass::Cursor),
    ("OP_CURSOR_MOVE", OP_CURSOR_MOVE, OpClass::Cursor),
    ("OP_CURSOR_LINE", OP_CURSOR_LINE, OpClass::Cursor),
    ("OP_SELECT_ALL", OP_SELECT_ALL, OpClass::Cursor),
    ("OP_UNDO", OP_UNDO, OpClass::Replay),
    ("OP_REDO", OP_REDO, OpClass::Replay),
];

/// The class of `kind`, or `None` when the tag is unknown.
///
/// A kernel cannot call this — the dialect has no table lookups — so
/// it is the host-side guard that turns a malformed queue into an
/// error instead of a wrong document.
pub const fn class_of(kind: u32) -> Option<OpClass> {
    let mut index = 0;
    while index < OP_KINDS.len() {
        if OP_KINDS[index].1 == kind {
            return Some(OP_KINDS[index].2);
        }
        index += 1;
    }
    None
}

/// The spelling of `kind`, or `None` when the tag is unknown.
pub const fn kind_name(kind: u32) -> Option<&'static str> {
    let mut index = 0;
    while index < OP_KINDS.len() {
        if OP_KINDS[index].1 == kind {
            return Some(OP_KINDS[index].0);
        }
        index += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// Per-operation flags
// ---------------------------------------------------------------------------

/// Capture this operation's deleted span even when the batch does not
/// ask for capture.
pub const OP_FLAG_KEEP: u32 = 1 << 0;
/// Never capture this operation's deleted span, even when the batch
/// does ask: used for a removal too large to keep in the payload.
pub const OP_FLAG_SKIP_CAPTURE: u32 = 1 << 1;
/// The inserted run is a shared staging gap slot, named by
/// [`OP_CURSOR`], rather than a private payload allocation.
pub const OP_FLAG_SHARED: u32 = 1 << 2;
/// The selection anchor follows the caret, so the operation leaves a
/// collapsed selection behind.
pub const OP_FLAG_ANCHOR: u32 = 1 << 3;

/// Per-operation flag bits a kernel is expected to name, for the
/// ABI-drift test.
pub const OP_FLAG_BITS: [(&str, u32); 4] = [
    ("OP_FLAG_KEEP", OP_FLAG_KEEP),
    ("OP_FLAG_SKIP_CAPTURE", OP_FLAG_SKIP_CAPTURE),
    ("OP_FLAG_SHARED", OP_FLAG_SHARED),
    ("OP_FLAG_ANCHOR", OP_FLAG_ANCHOR),
];

/// Whether the deleted span of an operation must be copied to the
/// payload before it goes away.
///
/// A per-operation override always wins: [`FLAG_CAPTURE`] makes a
/// whole batch undoable, and [`OP_FLAG_SKIP_CAPTURE`] is how a single
/// oversized removal stays out of the retained state.
#[must_use]
#[inline]
pub const fn op_captures(batch_flags: u32, op_flags: u32) -> bool {
    if op_flags & OP_FLAG_KEEP != 0 {
        return true;
    }
    if op_flags & OP_FLAG_SKIP_CAPTURE != 0 {
        return false;
    }
    batch_flags & FLAG_CAPTURE != 0
}

// ---------------------------------------------------------------------------
// The operation record
// ---------------------------------------------------------------------------

/// One entry of the operation queue.
///
/// The fields are the whole contract: `pos` is in *source* coordinates,
/// because the planning pass builds the index map from the document the
/// apply kernel reads, not from the one it is producing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpRecord {
    /// The operation tag, one of the `OP_*` kind constants.
    pub kind: u32,
    /// Element index the operation applies to, in source coordinates.
    pub pos: u32,
    /// Source elements the operation removes.
    pub len: u32,
    /// Payload elements the operation inserts.
    pub count: u32,
    /// Payload word the inserted run starts at, [`NO_PAYLOAD`] if none.
    pub payload: u32,
    /// Cursor row the operation is attached to.
    pub cursor: u32,
    /// Per-operation flags, a union of the `OP_FLAG_*` bits.
    pub flags: u32,
}

impl OpRecord {
    /// An empty queue slot, which every kernel skips.
    ///
    /// Its payload word is the sentinel rather than zero, so a record
    /// the host built but never filled cannot address payload word 0 by
    /// accident; a slot the planning pass never wrote reads back
    /// zeroed, which is why the tag — not the payload — is what a
    /// kernel tests.
    #[must_use]
    #[inline]
    pub const fn none() -> Self {
        Self {
            kind: OP_NOP,
            pos: 0,
            len: 0,
            count: 0,
            payload: NO_PAYLOAD,
            cursor: 0,
            flags: 0,
        }
    }

    /// Reads operation `index` from `image`.
    ///
    /// # Errors
    ///
    /// [`crate::error::WreditError::OutOfRange`] when `index` escapes
    /// the queue or the image,
    /// [`crate::error::WreditError::InvalidAddress`] when `image` is
    /// not whole words.
    pub fn read(image: &[u8], index: u32) -> WreditResult<Self> {
        let word =
            |field: u32| record_word(Plane::Ops, index, field).and_then(|offset| get_word(image, offset));
        Ok(Self {
            kind: word(OP_KIND)?,
            pos: word(OP_POS)?,
            len: word(OP_LEN)?,
            count: word(OP_COUNT)?,
            payload: word(OP_PAYLOAD)?,
            cursor: word(OP_CURSOR)?,
            flags: word(OP_FLAGS)?,
        })
    }

    /// Writes operation `index` into `image`.
    ///
    /// The reserved word is written zero here rather than left to the
    /// caller, so a queue slot never carries a stale value into the
    /// next batch.
    ///
    /// # Errors
    ///
    /// As [`Self::read`].
    pub fn write(&self, image: &mut [u8], index: u32) -> WreditResult<()> {
        let mut put = |field: u32, value: u32| {
            record_word(Plane::Ops, index, field).and_then(|offset| put_word(image, offset, value))
        };
        put(OP_KIND, self.kind)?;
        put(OP_POS, self.pos)?;
        put(OP_LEN, self.len)?;
        put(OP_COUNT, self.count)?;
        put(OP_PAYLOAD, self.payload)?;
        put(OP_CURSOR, self.cursor)?;
        put(OP_FLAGS, self.flags)?;
        put(OP_RESERVED, 0)
    }

    /// The tag's class, or `None` when the record carries a tag the
    /// protocol does not know.
    ///
    /// # Errors
    ///
    /// [`crate::error::WreditError::InvalidOperation`] naming the
    /// unknown tag.
    pub const fn class(&self) -> WreditResult<OpClass> {
        match class_of(self.kind) {
            Some(class) => Ok(class),
            None => Err(WreditError::InvalidOperation {
                what: "unknown operation tag",
            }),
        }
    }

    /// The tag's spelling, or `None` for an unknown tag.
    #[must_use]
    #[inline]
    pub const fn name(&self) -> Option<&'static str> {
        kind_name(self.kind)
    }

    /// How much the index map jumps where this operation applies.
    ///
    /// Insertions minus removals is the whole effect of an edit on the
    /// destination offsets, which is why a bulk shift does not depend
    /// on the operation's length. Both counts are bounded by the text
    /// capacity, far under `2^31`, so the subtraction cannot overflow.
    #[must_use]
    #[inline]
    pub const fn jump(&self) -> i32 {
        self.count.cast_signed() - self.len.cast_signed()
    }

    /// The discontinuity this operation contributes to the index map.
    ///
    /// [`OpClass::Capture`] and [`OpClass::Cursor`] operations have no
    /// row: their payload and movement work is scheduled from the
    /// queue directly. An [`OpClass::Replay`] operation's rows come
    /// from the retained log, not from this record, so its reduction is
    /// not meaningful.
    #[must_use]
    pub const fn boundary_row(&self) -> BoundaryRow {
        BoundaryRow {
            pos: self.pos,
            delta: self.jump(),
            // The scan passes recompute the prefix sum; a fresh row is
            // always zero here.
            cum: 0,
            payload: self.payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptbp::geometry::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn image() -> Vec<u8> {
        vec![0u8; IMAGE_WORDS as usize * WORD_BYTES]
    }

    #[test]
    fn op_fields_are_distinct_and_inside_the_record() {
        for (name, field) in OP_FIELDS {
            assert!(field < OP_WORDS, "{name} escapes the operation record");
            for (other, value) in OP_FIELDS {
                if other != name {
                    assert_ne!(field, value, "{name} collides with {other}");
                }
            }
        }
        assert_eq!(OP_RESERVED, OP_WORDS - 1, "the spare word must stay last");
    }

    #[test]
    fn the_queue_holds_whole_op_records() {
        assert_eq!(Plane::Ops.words(), OP_WORDS * OP_CAPACITY);
        assert_eq!(Plane::Ops.record_words(), Some(OP_WORDS));
    }

    #[test]
    fn every_tag_has_one_class_and_a_distinct_spelling() {
        let mut seen = [0u32; OP_KIND_LIMIT as usize];
        for (name, kind, class) in OP_KINDS {
            assert!(kind < OP_KIND_LIMIT, "{name} is outside the tag range");
            seen[kind as usize] += 1;
            assert_eq!(class_of(kind), Some(class), "{name} disagrees with its class");
            assert_eq!(kind_name(kind), Some(name));
            for (other, _, _) in OP_KINDS {
                if other != name {
                    assert_ne!(name, other, "two tags share a spelling");
                }
            }
        }
        assert!(
            seen.iter().all(|hits| *hits == 1),
            "every tag in the range must appear exactly once"
        );
        assert_eq!(OP_KINDS.len(), OP_KIND_LIMIT as usize);
    }

    #[test]
    fn a_tag_outside_the_table_is_refused() {
        assert_eq!(class_of(OP_KIND_LIMIT), None);
        assert_eq!(kind_name(0xFFFF), None);
        let record = OpRecord {
            kind: OP_KIND_LIMIT,
            ..OpRecord::none()
        };
        assert_eq!(record.name(), None);
        assert!(matches!(
            record.class(),
            Err(WreditError::InvalidOperation { .. })
        ));
    }

    #[test]
    fn an_op_record_round_trips_through_the_queue() {
        let mut data = image();
        let record = OpRecord {
            kind: OP_REPLACE,
            pos: 1_024,
            len: 7,
            count: 3,
            payload: 512,
            cursor: CURSOR_CAPACITY - 1,
            flags: OP_FLAG_SHARED | OP_FLAG_ANCHOR,
        };
        record.write(&mut data, OP_CAPACITY - 1).unwrap();
        assert_eq!(
            OpRecord::read(&data, OP_CAPACITY - 1).unwrap(),
            record,
            "the last queue slot must be addressable"
        );
        assert!(matches!(
            OpRecord::read(&data, OP_CAPACITY),
            Err(WreditError::OutOfRange { .. })
        ));
        let blank = OpRecord::read(&data, 0).unwrap();
        assert_eq!(
            blank.kind, OP_NOP,
            "a slot the planning pass never wrote must read as no operation"
        );
        assert_eq!(blank.class().unwrap(), OpClass::Neutral);
        assert_eq!(blank.jump(), 0);
    }

    #[test]
    fn writing_an_operation_clears_its_reserved_word() {
        let mut data = image();
        put_word(
            &mut data,
            record_word(Plane::Ops, 3, OP_RESERVED).unwrap(),
            0xDEAD_BEEF,
        )
        .unwrap();
        OpRecord {
            kind: OP_PUT,
            pos: 9,
            ..OpRecord::none()
        }
        .write(&mut data, 3)
        .unwrap();
        assert_eq!(
            get_word(&data, record_word(Plane::Ops, 3, OP_RESERVED).unwrap()).unwrap(),
            0,
            "a queue slot must never carry a stale word into the next batch"
        );
    }

    #[test]
    fn an_edit_reduces_to_one_discontinuity() {
        let insert = OpRecord {
            kind: OP_PUT,
            pos: 40,
            len: 0,
            count: 5,
            payload: 512,
            cursor: 0,
            flags: 0,
        };
        assert_eq!(insert.jump(), 5);
        assert_eq!(
            insert.boundary_row(),
            BoundaryRow {
                pos: 40,
                delta: 5,
                cum: 0,
                payload: 512
            }
        );

        let remove = OpRecord {
            kind: OP_DELETE,
            pos: 40,
            len: 7,
            count: 0,
            payload: NO_PAYLOAD,
            cursor: 0,
            flags: 0,
        };
        assert_eq!(remove.jump(), -7);
        assert_eq!(remove.boundary_row().payload, NO_PAYLOAD);

        let replace = OpRecord {
            kind: OP_REPLACE,
            pos: 40,
            len: 7,
            count: 3,
            payload: 512,
            cursor: 0,
            flags: 0,
        };
        assert_eq!(replace.jump(), -4, "one row carries both halves of a replace");

        let overwrite = OpRecord {
            kind: OP_OVERWRITE,
            pos: 40,
            len: 3,
            count: 3,
            payload: 512,
            cursor: 0,
            flags: 0,
        };
        assert_eq!(overwrite.jump(), 0, "an overwrite cannot change the length");
    }

    #[test]
    fn only_the_edit_tags_move_the_index_map() {
        let edits = [OP_PUT, OP_DELETE, OP_REPLACE, OP_OVERWRITE, OP_GAP_COMMIT];
        for kind in edits {
            assert_eq!(
                class_of(kind),
                Some(OpClass::Edit),
                "{}",
                kind_name(kind).unwrap()
            );
        }
        for (name, kind, class) in OP_KINDS {
            if !edits.contains(&kind) {
                assert_ne!(class, OpClass::Edit, "{name} must not reduce to a row");
            }
        }
    }

    #[test]
    fn an_undo_replays_the_reduced_row() {
        let record = OpRecord {
            kind: OP_REPLACE,
            pos: 40,
            len: 7,
            count: 3,
            payload: 512,
            cursor: 0,
            flags: 0,
        };
        let back = record.boundary_row().checked_inverse().unwrap();
        assert_eq!(back.pos, record.pos, "the jump stays at its source index");
        assert_eq!(back.delta, -record.jump());
        assert_eq!(
            back.payload, record.payload,
            "the run an undo re-inserts is still addressed"
        );
        assert_eq!(back.cum, 0, "the inverse scan recomputes the prefix");
    }

    #[test]
    fn a_per_operation_flag_overrides_the_batch_capture() {
        assert!(op_captures(FLAG_CAPTURE, 0));
        assert!(!op_captures(0, 0));
        assert!(
            op_captures(0, OP_FLAG_KEEP),
            "a single undoable removal must survive a batch that does not capture"
        );
        assert!(
            !op_captures(FLAG_CAPTURE, OP_FLAG_SKIP_CAPTURE),
            "an oversized removal must be excludable from a capturing batch"
        );
        assert!(
            op_captures(FLAG_CAPTURE, OP_FLAG_SKIP_CAPTURE | OP_FLAG_KEEP),
            "keeping wins over skipping, so a caller cannot lose data silently"
        );
    }

    #[test]
    fn no_operation_edits_anything() {
        let none = OpRecord::none();
        assert_eq!(none.kind, OP_NOP);
        assert_eq!(none.class().unwrap(), OpClass::Neutral);
        assert_eq!(none.jump(), 0);
        assert_eq!(none.payload, NO_PAYLOAD);
    }
}
