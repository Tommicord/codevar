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

//! Field offsets of the protocol records, and the boundary row.
//!
//! A record is a fixed run of words inside a plane, so a field is an
//! index `0..record_words` and the host and the kernel must agree on
//! every one of them. Each table here exists twice over: as constants
//! for the host, and as the named file-scope constants a kernel
//! declares — the drift test compares the two spellings, so an offset
//! that changes on one side and not the other fails the build instead
//! of corrupting a document.

use super::access::{get_word, put_word};
use super::geometry::{Plane, record_word};
use crate::error::WreditResult;

// ---------------------------------------------------------------------------
// Control plane fields
// ---------------------------------------------------------------------------

/// Image identity; must read [`MAGIC`] before anything else is trusted.
pub const CTRL_MAGIC: u32 = 0;
/// Protocol revision; must read [`VERSION`].
pub const CTRL_VERSION: u32 = 1;
/// Batch flags, a union of the `FLAG_*` bits.
pub const CTRL_FLAGS: u32 = 2;
/// Monotonic batch counter, incremented once per dispatched frame.
pub const CTRL_BATCH: u32 = 3;
/// Elements in the source text plane.
pub const CTRL_SRC_LEN: u32 = 4;
/// Elements written to the destination text plane; produced by the
/// apply kernel, never by the host.
pub const CTRL_DST_LEN: u32 = 5;
/// Elements a text plane can hold; the bounds guard of every kernel.
pub const CTRL_TEXT_CAPACITY: u32 = 6;
/// Records queued in the operation plane.
pub const CTRL_OP_COUNT: u32 = 7;
/// Rows filled in the boundary table.
pub const CTRL_BOUNDARY_COUNT: u32 = 8;
/// Cursor records the image holds.
pub const CTRL_CURSOR_COUNT: u32 = 9;
/// Staging gap descriptors filled by the host.
pub const CTRL_GAP_COUNT: u32 = 10;
/// Elements the payload plane can hold.
pub const CTRL_PAYLOAD_CAPACITY: u32 = 11;
/// Payload words consumed in this batch: the allocator cursor the
/// host and the gap-commit kernel share.
pub const CTRL_PAYLOAD_USED: u32 = 12;
/// Records retained in the operation log.
pub const CTRL_LOG_COUNT: u32 = 13;
/// Stride of the next boundary-scan pass; zero when the scan is done.
pub const CTRL_SCAN_STRIDE: u32 = 14;
/// Reserved for a future field; always written zero.
pub const CTRL_RESERVED: u32 = 15;

/// Control fields a kernel is expected to name, for the ABI-drift
/// test: the constant's spelling and the word offset it must carry.
pub const CTRL_FIELDS: [(&str, u32); 15] = [
    ("CTRL_MAGIC", CTRL_MAGIC),
    ("CTRL_VERSION", CTRL_VERSION),
    ("CTRL_FLAGS", CTRL_FLAGS),
    ("CTRL_BATCH", CTRL_BATCH),
    ("CTRL_SRC_LEN", CTRL_SRC_LEN),
    ("CTRL_DST_LEN", CTRL_DST_LEN),
    ("CTRL_TEXT_CAPACITY", CTRL_TEXT_CAPACITY),
    ("CTRL_OP_COUNT", CTRL_OP_COUNT),
    ("CTRL_BOUNDARY_COUNT", CTRL_BOUNDARY_COUNT),
    ("CTRL_CURSOR_COUNT", CTRL_CURSOR_COUNT),
    ("CTRL_GAP_COUNT", CTRL_GAP_COUNT),
    ("CTRL_PAYLOAD_CAPACITY", CTRL_PAYLOAD_CAPACITY),
    ("CTRL_PAYLOAD_USED", CTRL_PAYLOAD_USED),
    ("CTRL_LOG_COUNT", CTRL_LOG_COUNT),
    ("CTRL_SCAN_STRIDE", CTRL_SCAN_STRIDE),
];

/// The destination plane is text plane B and the source is A; when
/// set, the roles swap. The bit is the parity of the batch, so two
/// consecutive batches alternate planes without a copy.
pub const FLAG_PARITY: u32 = 1 << 0;
/// Apply the inverse of the retained boundary table — an undo pass.
pub const FLAG_INVERSE: u32 = 1 << 1;
/// Retain the applied records in the operation log.
pub const FLAG_LOG: u32 = 1 << 2;
/// Remap caret and anchor positions into the output cursor fields.
pub const FLAG_REMAP: u32 = 1 << 3;
/// Copy deleted spans into the payload before deleting them, so the
/// batch can be re-applied by a redo pass.
pub const FLAG_CAPTURE: u32 = 1 << 4;
/// Commit the staging gaps described by the gap table as insertions.
pub const FLAG_FLUSH_GAPS: u32 = 1 << 5;

/// Flag bits a kernel is expected to name, for the ABI-drift test.
pub const FLAG_BITS: [(&str, u32); 6] = [
    ("FLAG_PARITY", FLAG_PARITY),
    ("FLAG_INVERSE", FLAG_INVERSE),
    ("FLAG_LOG", FLAG_LOG),
    ("FLAG_REMAP", FLAG_REMAP),
    ("FLAG_CAPTURE", FLAG_CAPTURE),
    ("FLAG_FLUSH_GAPS", FLAG_FLUSH_GAPS),
];

/// Source text plane for the parity bit in `flags`.
#[must_use]
#[inline]
pub const fn src_plane(flags: u32) -> Plane {
    if flags & FLAG_PARITY == 0 {
        Plane::TextA
    } else {
        Plane::TextB
    }
}

/// Destination text plane for the parity bit in `flags`.
///
/// The pair always differs, so an apply pass is strictly out-of-place
/// and no kernel ever reads and writes the same word.
#[must_use]
#[inline]
pub const fn dst_plane(flags: u32) -> Plane {
    if flags & FLAG_PARITY == 0 {
        Plane::TextB
    } else {
        Plane::TextA
    }
}

// ---------------------------------------------------------------------------
// Boundary rows
// ---------------------------------------------------------------------------

/// Row field holding the first source index the discontinuity applies
/// to.
pub const BND_POS: u32 = 0;
/// Row field holding the signed change to the destination index, as a
/// two's-complement word.
pub const BND_DELTA: u32 = 1;
/// Row field holding the inclusive prefix sum of [`BND_DELTA`]
/// through this row; written by the scan passes, read by the apply
/// kernel.
pub const BND_CUM: u32 = 2;
/// Row field holding the payload word the inserted run starts at, or
/// [`NO_PAYLOAD`].
pub const BND_PAYLOAD: u32 = 3;

/// Payload sentinel: this discontinuity inserts nothing.
pub const NO_PAYLOAD: u32 = 0xFFFF_FFFF;

/// One discontinuity of the source→destination index map.
///
/// The map is piecewise-constant: a source element at index `i` lands
/// at `i + cum`, where `cum` is the prefix sum of the last row whose
/// [`BoundaryRow::pos`] is `<= i`. That is what makes the bulk shift
/// purely data-parallel — one binary search over a sorted row table
/// per element, no per-element scan and no atomics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryRow {
    /// First source index this row applies to.
    pub pos: u32,
    /// Signed jump applied to every index from [`Self::pos`] onward.
    pub delta: i32,
    /// Inclusive prefix sum through this row.
    pub cum: i32,
    /// Payload word of the run inserted here, [`NO_PAYLOAD`] if none.
    pub payload: u32,
}

impl BoundaryRow {
    /// A row that applies nowhere and inserts nothing.
    #[must_use]
    #[inline]
    pub const fn none() -> Self {
        Self {
            pos: 0,
            delta: 0,
            cum: 0,
            payload: NO_PAYLOAD,
        }
    }

    /// Reads row `index` from `image`.
    ///
    /// # Errors
    ///
    /// [`WreditError::OutOfRange`] when `index` escapes the table or
    /// the image, [`WreditError::InvalidAddress`] when `image` is not
    /// whole words.
    pub fn read(image: &[u8], index: u32) -> WreditResult<Self> {
        let word = |field: u32| {
            record_word(Plane::Boundaries, index, field).and_then(|offset| get_word(image, offset))
        };
        Ok(Self {
            pos: word(BND_POS)?,
            delta: signed(word(BND_DELTA)?),
            cum: signed(word(BND_CUM)?),
            payload: word(BND_PAYLOAD)?,
        })
    }

    /// Writes row `index` into `image`.
    ///
    /// # Errors
    ///
    /// As [`Self::read`].
    pub fn write(&self, image: &mut [u8], index: u32) -> WreditResult<()> {
        let mut put = |field: u32, value: u32| {
            record_word(Plane::Boundaries, index, field).and_then(|offset| put_word(image, offset, value))
        };
        put(BND_POS, self.pos)?;
        put(BND_DELTA, self.delta.cast_unsigned())?;
        put(BND_CUM, self.cum.cast_unsigned())?;
        put(BND_PAYLOAD, self.payload)
    }

    /// The row as an undo pass replays it: the same discontinuity at
    /// the same source index with the jump negated.
    ///
    /// The prefix sum is reset rather than copied, because the inverse
    /// scan recomputes it from the negated deltas — carrying a stale
    /// `cum` through would silently mis-place everything after the
    /// first row.
    ///
    /// `None` for a [`i32::MIN`] jump, which has no negation. A
    /// conforming image cannot carry one: a jump is bounded by the
    /// text plane, and the plane capacity is far below `2^31` elements.
    #[must_use]
    #[inline]
    pub const fn checked_inverse(&self) -> Option<Self> {
        match self.delta.checked_neg() {
            Some(delta) => Some(Self {
                pos: self.pos,
                delta,
                cum: 0,
                payload: self.payload,
            }),
            None => None,
        }
    }
}

/// Reinterprets a protocol word as its two's-complement value.
#[must_use]
#[inline]
pub const fn signed(word: u32) -> i32 {
    word as i32
}

// ---------------------------------------------------------------------------
// Cursor records
// ---------------------------------------------------------------------------

/// Field holding the caret, in source-plane element indices.
pub const CUR_POS: u32 = 0;
/// Field holding the selection anchor, in source indices.
pub const CUR_ANCHOR: u32 = 1;
/// Field holding the signed movement a cursor op applies to [`CUR_POS`].
pub const CUR_STEP: u32 = 2;
/// Field holding the cursor flags, a union of the `CURSOR_*` bits.
pub const CUR_FLAGS: u32 = 3;
/// Field the remap pass writes the new caret into.
pub const CUR_OUT_POS: u32 = 4;
/// Field the remap pass writes the new anchor into.
pub const CUR_OUT_ANCHOR: u32 = 5;
/// Field the remap pass copies [`CUR_FLAGS`] into, clearing
/// [`CURSOR_SELECTED`] when the anchor collapsed onto the caret.
pub const CUR_OUT_FLAGS: u32 = 6;
/// Reserved field; always written zero.
pub const CUR_RESERVED: u32 = 7;

/// Cursor fields a kernel is expected to name, for the ABI-drift test.
pub const CURSOR_FIELDS: [(&str, u32); 7] = [
    ("CUR_POS", CUR_POS),
    ("CUR_ANCHOR", CUR_ANCHOR),
    ("CUR_STEP", CUR_STEP),
    ("CUR_FLAGS", CUR_FLAGS),
    ("CUR_OUT_POS", CUR_OUT_POS),
    ("CUR_OUT_ANCHOR", CUR_OUT_ANCHOR),
    ("CUR_OUT_FLAGS", CUR_OUT_FLAGS),
];

/// The cursor has an active selection between [`CUR_POS`] and
/// [`CUR_ANCHOR`].
pub const CURSOR_SELECTED: u32 = 1 << 0;
/// The cursor record is live; when clear the row is free and every
/// kernel must skip it.
pub const CURSOR_LIVE: u32 = 1 << 1;

// ---------------------------------------------------------------------------
// Gap records
// ---------------------------------------------------------------------------

/// Field holding the payload word the staged run starts at.
pub const GAP_PAYLOAD: u32 = 0;
/// Field holding how many elements are staged.
pub const GAP_FILL: u32 = 1;
/// Field holding the commit base, in source indices.
pub const GAP_BASE: u32 = 2;
/// Field holding the owning cursor mask in bits `0..16` and the
/// shared-slot tag in bit 16.
pub const GAP_OWNER: u32 = 3;

/// Gap fields a kernel is expected to name, for the ABI-drift test.
pub const GAP_FIELDS: [(&str, u32); 4] = [
    ("GAP_PAYLOAD", GAP_PAYLOAD),
    ("GAP_FILL", GAP_FILL),
    ("GAP_BASE", GAP_BASE),
    ("GAP_OWNER", GAP_OWNER),
];

/// Bit 16 of [`GAP_OWNER`]: the descriptor names a shared gap slot
/// rather than a private per-cursor gap.
pub const GAP_SHARED: u32 = 1 << 16;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::WreditError;
    use crate::ptbp::geometry::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn image() -> Vec<u8> {
        vec![0u8; IMAGE_WORDS as usize * WORD_BYTES]
    }

    #[test]
    fn control_fields_are_distinct_and_inside_the_plane() {
        for (name, field) in CTRL_FIELDS {
            assert!(field < CONTROL_WORDS, "{name} escapes the control plane");
            for (other, value) in CTRL_FIELDS {
                if other != name {
                    assert_ne!(field, value, "{name} collides with {other}");
                }
            }
        }
        assert_eq!(
            CTRL_RESERVED,
            CONTROL_WORDS - 1,
            "the spare field must stay the last word"
        );
    }

    #[test]
    fn flag_bits_are_single_and_disjoint() {
        for (name, bit) in FLAG_BITS {
            assert_eq!(bit.count_ones(), 1, "{name} is not one bit");
            for (other, value) in FLAG_BITS {
                if other != name {
                    assert_eq!(bit & value, 0, "{name} overlaps {other}");
                }
            }
        }
    }

    #[test]
    fn field_tables_name_every_field_of_their_plane() {
        for (name, field) in CURSOR_FIELDS {
            assert!(field < CURSOR_WORDS, "{name} escapes a cursor record");
            for (other, value) in CURSOR_FIELDS {
                if other != name {
                    assert_ne!(field, value, "{name} collides with {other}");
                }
            }
        }
        assert_eq!(
            CURSOR_FIELDS.len() as u32 + 1,
            CURSOR_WORDS,
            "exactly one cursor word is a reserved spare"
        );
        for (name, field) in GAP_FIELDS {
            assert!(field < GAP_WORDS, "{name} escapes a gap record");
        }
        assert_eq!(GAP_FIELDS.len() as u32, GAP_WORDS, "every gap word is named");
    }

    #[test]
    fn parity_always_selects_two_different_text_planes() {
        for flags in [0, FLAG_PARITY, FLAG_PARITY | FLAG_INVERSE | FLAG_LOG] {
            let (src, dst) = (src_plane(flags), dst_plane(flags));
            assert_ne!(src, dst);
            assert!(matches!(
                (src, dst),
                (Plane::TextA, Plane::TextB) | (Plane::TextB, Plane::TextA)
            ));
        }
    }

    #[test]
    fn record_words_land_on_the_field_they_name() {
        assert_eq!(
            record_word(Plane::Boundaries, 0, BND_CUM),
            Ok(BOUNDARIES_OFFSET + BND_CUM)
        );
        assert_eq!(
            record_word(Plane::Cursors, 3, CUR_OUT_POS),
            Ok(CURSORS_OFFSET + 3 * CURSOR_WORDS + CUR_OUT_POS)
        );
        assert_eq!(
            record_word(Plane::Gaps, 19, GAP_OWNER),
            Ok(GAPS_OFFSET + 19 * GAP_WORDS + GAP_OWNER)
        );
        assert_eq!(
            record_word(Plane::Log, 5, BND_POS),
            Ok(LOG_OFFSET + 5 * LOG_WORDS + BND_POS)
        );
        assert_eq!(
            record_word(Plane::Payload, 0, 0),
            Err(WreditError::InvalidOperation {
                what: "plane is not record-structured"
            })
        );
    }

    #[test]
    fn record_word_rejects_a_record_or_field_past_its_plane() {
        assert!(
            matches!(
                record_word(Plane::Cursors, CURSOR_CAPACITY, CUR_POS),
                Err(WreditError::OutOfRange { start, end })
                    if start == CURSOR_CAPACITY as u64 && end == CURSOR_CAPACITY as u64
            ),
            "the record after the pool must be refused"
        );
        assert!(matches!(
            record_word(Plane::Gaps, 0, GAP_WORDS),
            Err(WreditError::OutOfRange { start, end }) if start == GAP_WORDS as u64 && end == GAP_WORDS as u64
        ));
    }

    #[test]
    fn boundary_rows_round_trip_including_negative_deltas() {
        let mut data = image();
        let rows = [
            BoundaryRow {
                pos: 0,
                delta: 0,
                cum: 0,
                payload: NO_PAYLOAD,
            },
            BoundaryRow {
                pos: 12,
                delta: -7,
                cum: -7,
                payload: 4_096,
            },
            BoundaryRow {
                pos: 900,
                delta: 1_024,
                cum: 1_017,
                payload: 77,
            },
            BoundaryRow {
                pos: u32::MAX,
                delta: i32::MIN,
                cum: i32::MAX,
                payload: 0,
            },
        ];
        for (index, row) in rows.iter().enumerate() {
            row.write(&mut data, index as u32).unwrap();
        }
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(BoundaryRow::read(&data, index as u32).unwrap(), *row);
        }
        assert!(
            matches!(
                BoundaryRow::read(&data, BOUNDARY_CAPACITY),
                Err(WreditError::OutOfRange { .. })
            ),
            "the row after the table must be refused"
        );
    }

    #[test]
    fn an_inverse_row_negates_the_jump_and_drops_the_prefix() {
        let row = BoundaryRow {
            pos: 40,
            delta: -9,
            cum: 12,
            payload: 512,
        };
        let back = row.checked_inverse().unwrap();
        assert_eq!(
            back.pos, 40,
            "the discontinuity stays where the forward pass put it"
        );
        assert_eq!(back.delta, 9);
        assert_eq!(
            back.payload, 512,
            "the run an undo must re-insert is still addressed"
        );
        assert_eq!(
            back.cum, 0,
            "a stale prefix must never survive into an inverse pass"
        );
    }

    #[test]
    fn an_unnegatable_jump_is_refused_rather_than_wrapped() {
        let row = BoundaryRow {
            pos: 1,
            delta: i32::MIN,
            cum: 7,
            payload: 3,
        };
        assert_eq!(
            row.checked_inverse(),
            None,
            "a jump with no negation must stop the undo pass, not wrap to itself"
        );
        assert_eq!(row.checked_inverse().map(|back| back.delta), None);
    }

    #[test]
    fn an_empty_boundary_row_inserts_nothing() {
        assert_eq!(BoundaryRow::none().payload, NO_PAYLOAD);
        assert_eq!(NO_PAYLOAD, u32::MAX);
        assert_eq!(BoundaryRow::none().delta, 0);
    }

    #[test]
    fn signed_views_use_twos_complement() {
        assert_eq!(signed(0xFFFF_FFF9), -7);
        assert_eq!(signed(0x8000_0000), i32::MIN);
        assert_eq!((-7i32).cast_unsigned(), 0xFFFF_FFF9);
    }
}
