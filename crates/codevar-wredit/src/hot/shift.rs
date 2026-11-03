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

//! Riding a boundary table: the position maps for cursors and gaps.
//!
//! An edit moves more than text. The CPU engine keeps its cursor plane
//! and its gap descriptors in step with every commit by shifting them
//! (`BaseWritable::shift_cursors`, `BaseWritable::shift_gap_bases` and
//! `BaseWritable::shift_shared_bases` in [`crate::edit`]), and it does
//! that once per operation. A batch shifts them all at once, from the
//! table [`Batch::rows`](super::sequentially_batch::Batch::rows)
//! produces, which is what this module implements.
//!
//! # The two rules differ at the boundary
//!
//! A caret and a gap base at the very position an edit starts behave
//! differently, and the difference is not an artefact: the caret stays
//! *before* text inserted at its own position (typing advances the
//! caret explicitly, the shift does not do it), while a gap base moves
//! *after* it, because the region it names has really grown. Removing
//! collapses both onto the removal start. So one function per kind —
//! [`caret_at`] and [`gap_base_at`] — and neither is a generalisation
//! of the other.
//!
//! # Scope
//!
//! These maps move positions the batch does not otherwise touch. A
//! cursor the queue *moves* is not remapped here: an
//! [`crate::ptbp::OpClass::Cursor`] record states a position in the
//! document the earlier operations of the batch already produced, and
//! resolving that is the dispatcher's job, not a shift.
//!
//! Every map takes the source length as well as the table, because the
//! caret saturates at the destination length exactly like the CPU shift
//! saturates at the logical end.

use crate::ptbp::BoundaryRow;

use super::sequentially_batch::{is_drop, is_insert};

/// Prefix sum through the first `reached` rows.
fn cum_of(rows: &[BoundaryRow], reached: usize) -> i32 {
    rows.get(reached.wrapping_sub(1))
        .map_or(0, |row| row.cum)
}

/// Prefix sum through the rows that start strictly before `pos`.
///
/// Rows are sorted and share a position only inside one anchor group,
/// so this is the shift in force *before* anything anchored at `pos`.
#[must_use]
pub fn cum_before(rows: &[BoundaryRow], pos: u32) -> i32 {
    cum_of(rows, rows.partition_point(|row| row.pos < pos))
}

/// Prefix sum through the rows that start at or before `pos`.
///
/// This is the shift a source element at `pos` sees: it sits after
/// every run anchored at `pos`.
#[must_use]
pub fn cum_through(rows: &[BoundaryRow], pos: u32) -> i32 {
    cum_of(rows, rows.partition_point(|row| row.pos <= pos))
}

/// Elements the table inserts exactly at `pos`.
#[must_use]
pub fn inserts_at(rows: &[BoundaryRow], pos: u32) -> u32 {
    let group = rows
        .get(rows.partition_point(|row| row.pos < pos)..rows.partition_point(|row| row.pos <= pos))
        .unwrap_or(&[]);
    group
        .iter()
        .filter(|row| is_insert(row))
        .fold(0u32, |total, row| total + row.delta.cast_unsigned())
}

/// Anchor of the drop row whose span reaches `pos`, if any.
///
/// Drop spans are disjoint and a row's anchor is its position, so only
/// the last row starting at or before `pos` can cover it. The span is
/// closed at both ends (`anchor..=anchor + removed`), which is the
/// range a removal collapses: the caret that sat just past the removed
/// text ends up at the start of it.
#[must_use]
pub fn drop_anchor(rows: &[BoundaryRow], pos: u32) -> Option<u32> {
    let reached = rows.partition_point(|row| row.pos <= pos);
    let row = *rows.get(reached.checked_sub(1)?)?;
    if !is_drop(&row) {
        return None;
    }
    if pos <= row.pos.saturating_add(row.delta.unsigned_abs()) {
        Some(row.pos)
    } else {
        None
    }
}

/// Destination length of a pass over `src_len` source elements.
#[must_use]
pub fn dst_len(rows: &[BoundaryRow], src_len: u32) -> u32 {
    let net = rows.last().map_or(0, |row| row.cum);
    u32::try_from(i64::from(src_len) + i64::from(net)).unwrap_or(0)
}

/// Saturates a destination against the length of the destination.
fn clamp(shifted: i64, end: u32) -> u32 {
    if shifted <= 0 {
        0
    } else {
        u32::try_from(shifted).unwrap_or(end).min(end)
    }
}

/// Destination of a caret at source position `pos`.
///
/// The anchor of a removal, and every position the removal covers,
/// collapses onto the anchor *before* the text the same anchor inserts;
/// any other position takes the shift in force strictly before its own
/// anchor, which leaves a caret at an insertion point in front of the
/// inserted run. The result saturates at [`dst_len`], the way the CPU
/// shift saturates at the logical end.
#[must_use]
pub fn caret_at(rows: &[BoundaryRow], pos: u32, src_len: u32) -> u32 {
    let shifted = match drop_anchor(rows, pos) {
        Some(anchor) if pos > anchor => i64::from(anchor) + i64::from(cum_before(rows, anchor)),
        _ => i64::from(pos) + i64::from(cum_before(rows, pos)),
    };
    clamp(shifted, dst_len(rows, src_len))
}

/// Destination of a gap base at source position `pos`.
///
/// A base inside a removal collapses onto the anchor and then moves
/// after the text inserted there; a base at an insertion point moves
/// after the inserted run without collapsing. A position away from any
/// anchor takes the shift through its own anchor, which is the
/// element-side rule [`dst_of`](super::sequentially_batch::dst_of)
/// uses.
#[must_use]
pub fn gap_base_at(rows: &[BoundaryRow], pos: u32, src_len: u32) -> u32 {
    let shifted = match drop_anchor(rows, pos) {
        Some(anchor) => {
            i64::from(anchor) + i64::from(cum_before(rows, anchor)) + i64::from(inserts_at(rows, anchor))
        }
        None => i64::from(pos) + i64::from(cum_through(rows, pos)),
    };
    clamp(shifted, dst_len(rows, src_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hot::sequentially_batch::Batch;
    use crate::ptbp::{NO_PAYLOAD, OP_DELETE, OP_PUT, OP_REPLACE, OpRecord};

    /// One operation as the queue spells it, with an insert always
    /// anchored at payload word zero.
    fn op(kind: u32, pos: u32, len: u32, count: u32) -> OpRecord {
        OpRecord {
            kind,
            pos,
            len,
            count,
            payload: if count == 0 { NO_PAYLOAD } else { 0 },
            cursor: 0,
            flags: 0,
        }
    }

    /// A batch over `src_len` source elements with `ops` folded.
    fn table(src_len: u32, ops: &[OpRecord]) -> Vec<BoundaryRow> {
        let mut batch = Batch::new(src_len);
        for record in ops {
            batch.fold(record).expect("the script folds");
        }
        batch.rows()
    }

    /// The engine's shift of one position under one operation, copied
    /// from `edit/writable.rs`: `logical_end` is the length *after* the
    /// operation, which is what the insertion clamps against.
    fn engine_shift(pos: u32, position: u32, delta: u32, removing: bool, logical_end: u32) -> u32 {
        if removing {
            let end = position.saturating_add(delta);
            if pos <= position {
                pos
            } else if pos <= end {
                position
            } else {
                pos - delta
            }
        } else if pos > position {
            pos.saturating_add(delta).min(logical_end)
        } else {
            pos
        }
    }

    /// How the engine moves one caret for one operation. A replace is
    /// a removal and then an insertion at the same position, which is
    /// exactly the order `commit_removal` and `commit_insertion` run
    /// `shift_cursors` in.
    fn engine_caret(record: &OpRecord, pos: u32, before: u32) -> u32 {
        let after_remove = if record.len == 0 {
            pos
        } else {
            engine_shift(pos, record.pos, record.len, true, before - record.len)
        };
        if record.count == 0 {
            after_remove
        } else {
            engine_shift(
                after_remove,
                record.pos,
                record.count,
                false,
                before - record.len + record.count,
            )
        }
    }

    #[test]
    fn a_removal_collapses_the_caret_onto_its_start() {
        let rows = table(8, &[op(OP_DELETE, 2, 3, 0)]);
        assert_eq!(caret_at(&rows, 1, 8), 1);
        assert_eq!(caret_at(&rows, 2, 8), 2);
        assert_eq!(caret_at(&rows, 3, 8), 2);
        assert_eq!(caret_at(&rows, 5, 8), 2);
        assert_eq!(caret_at(&rows, 6, 8), 3);
    }

    #[test]
    fn an_insertion_leaves_a_caret_at_its_own_position_behind() {
        let rows = table(8, &[op(OP_PUT, 2, 0, 3)]);
        assert_eq!(caret_at(&rows, 2, 8), 2);
        assert_eq!(caret_at(&rows, 3, 8), 6);
        assert_eq!(caret_at(&rows, 1, 8), 1);
    }

    #[test]
    fn a_gap_base_moves_after_the_text_inserted_at_it() {
        let rows = table(8, &[op(OP_PUT, 2, 0, 3)]);
        assert_eq!(gap_base_at(&rows, 2, 8), 5);
        assert_eq!(gap_base_at(&rows, 1, 8), 1);
        assert_eq!(gap_base_at(&rows, 3, 8), 6);
    }

    #[test]
    fn a_replace_collapses_then_reopens_around_the_caret() {
        // Remove three and insert two over them: a caret inside the
        // removed run lands before the new text, a caret past it lands
        // behind, which is how the two engine shifts compose.
        let rows = table(8, &[op(OP_REPLACE, 2, 3, 2)]);
        let record = op(OP_REPLACE, 2, 3, 2);
        for pos in 0..=8 {
            assert_eq!(
                caret_at(&rows, pos, 8),
                engine_caret(&record, pos, 8),
                "caret at {pos}"
            );
        }
        assert_eq!(gap_base_at(&rows, 2, 8), 4);
        assert_eq!(gap_base_at(&rows, 4, 8), 4);
        assert_eq!(gap_base_at(&rows, 5, 8), 4);
        assert_eq!(gap_base_at(&rows, 6, 8), 5);
    }

    #[test]
    fn the_maps_reproduce_the_engine_shift_for_every_single_edit() {
        let src_len = 12u32;
        let mut cases: Vec<OpRecord> = Vec::new();
        for position in 0..=src_len {
            for amount in 0..=4u32 {
                cases.push(op(OP_PUT, position, 0, amount));
                if position + amount <= src_len {
                    cases.push(op(OP_DELETE, position, amount, 0));
                    if amount > 0 {
                        cases.push(op(OP_REPLACE, position, amount, amount - 1));
                        cases.push(op(OP_REPLACE, position, amount, amount + 1));
                    }
                }
            }
        }
        for record in &cases {
            let rows = table(src_len, std::slice::from_ref(record));
            for pos in 0..=src_len {
                assert_eq!(
                    caret_at(&rows, pos, src_len),
                    engine_caret(record, pos, src_len),
                    "{} at {} of {}..{}, caret {pos}",
                    record.name().unwrap_or("op"),
                    record.pos,
                    record.len,
                    record.count,
                );
            }
        }
    }

    #[test]
    fn one_pass_shifts_a_caret_as_many_engine_shifts_do() {
        let src_len = 10u32;
        let script = [
            op(OP_PUT, 3, 0, 2),
            op(OP_DELETE, 1, 3, 0),
            op(OP_REPLACE, 6, 2, 3),
            op(OP_PUT, 0, 0, 1),
        ];
        let rows = table(src_len, &script);
        for start in 0..=src_len {
            let mut walked = start;
            let mut length = src_len;
            for record in &script {
                walked = engine_caret(record, walked, length);
                length = length - record.len + record.count;
            }
            assert_eq!(caret_at(&rows, start, src_len), walked, "caret from {start}");
        }
    }

    #[test]
    fn the_anchor_helpers_read_the_table_as_the_caret_rule_does() {
        let rows = table(8, &[op(OP_REPLACE, 2, 3, 3), op(OP_PUT, 7, 0, 2)]);
        assert_eq!(cum_before(&rows, 2), 0);
        assert_eq!(cum_through(&rows, 2), 0);
        assert_eq!(inserts_at(&rows, 2), 3);
        assert_eq!(inserts_at(&rows, 7), 2);
        assert_eq!(drop_anchor(&rows, 2), Some(2));
        assert_eq!(drop_anchor(&rows, 4), Some(2));
        assert_eq!(drop_anchor(&rows, 5), Some(2));
        assert_eq!(drop_anchor(&rows, 7), None);
        assert_eq!(dst_len(&rows, 8), 10);
    }

    #[test]
    fn a_caret_past_the_end_saturates_at_the_destination_length() {
        let rows = table(8, &[op(OP_DELETE, 0, 6, 0)]);
        assert_eq!(dst_len(&rows, 8), 2);
        // The engine shifts a caret beyond the removal left by its
        // whole length, and clamps the document end to the new length.
        assert_eq!(caret_at(&rows, 7, 8), 1);
        assert_eq!(caret_at(&rows, 8, 8), 2);
        assert_eq!(gap_base_at(&rows, 8, 8), 2);
    }

    #[test]
    fn an_empty_table_is_the_identity() {
        let rows: Vec<BoundaryRow> = Vec::new();
        for pos in 0..5 {
            assert_eq!(caret_at(&rows, pos, 5), pos);
            assert_eq!(gap_base_at(&rows, pos, 5), pos);
        }
        assert_eq!(dst_len(&rows, 5), 5);
        assert_eq!(drop_anchor(&rows, 0), None);
    }
}
