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

//! Plane geometry of the Parallel Text Buffer Protocol image.
//!
//! Every value here is an index into one flat array of 32-bit words.
//! The kernel dialect has no struct parameters and no include
//! mechanism — a kernel receives `*mut uint` and can only address
//! memory by computed word offset — so the protocol is defined as word
//! offsets, and those same numbers are what a kernel source names with
//! file-scope constants. Bytes are only ever a host concern, so
//! [`WORD_BYTES`] is the single place the conversion lives.
//!
//! The image is one OpenCL buffer: the planes tile it with no gaps and
//! a kernel needs no argument besides the base pointer, which keeps
//! dispatch one buffer per frame and every access coalesced on 4-byte
//! elements. Plane order is fixed by the constants below; nothing about
//! the geometry is computed at run time.

use crate::edit::{MAX_CURSORS, MAX_GAPS, SHARED_GAP_SLOTS};
use crate::error::{WreditError, WreditResult};

/// Bytes in one protocol word.
pub const WORD_BYTES: usize = 4;

/// Magic word a conforming image holds at [`super::fields::CTRL_MAGIC`].
pub const MAGIC: u32 = 0x5054_4250;

/// Protocol revision a conforming image holds at [`super::fields::CTRL_VERSION`].
///
/// A kernel reads these two words first and returns without touching
/// the text planes when they disagree; the layout is versioned as a
/// whole because every field offset is part of the ABI.
pub const VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Plane geometry
// ---------------------------------------------------------------------------

/// Words in the control plane.
pub const CONTROL_WORDS: u32 = 0x10;

/// Words in one operation record.
pub const OP_WORDS: u32 = 8;

/// Operation records the pending queue holds.
///
/// One record per user-level edit in a frame; a frame that produces
/// more than this is split into two batches by the dispatch layer.
pub const OP_CAPACITY: u32 = 0x400;

/// Words in one retained (logged) operation record.
pub const LOG_WORDS: u32 = OP_WORDS;

/// Operation records the undo/redo log retains.
pub const LOG_CAPACITY: u32 = 0x1000;

/// Words in one boundary row.
pub const BOUNDARY_WORDS: u32 = 4;

/// Boundary rows the plan table holds.
///
/// Every queued operation contributes at most one discontinuity and
/// every staging gap at most one more, so twice the queue depth plus
/// the gap table is the closed-form worst case; the extra row is the
/// terminator the scan pass writes.
pub const BOUNDARY_CAPACITY: u32 = 2 * OP_CAPACITY + (MAX_GAPS + SHARED_GAP_SLOTS) as u32 + 1;

/// Words in one cursor record.
pub const CURSOR_WORDS: u32 = 8;

/// Cursor records the image holds — exactly the engine's cursor pool.
pub const CURSOR_CAPACITY: u32 = MAX_CURSORS as u32;

/// Words in one staging-gap record.
pub const GAP_WORDS: u32 = 4;

/// Gap records the image holds: the private gaps plus the shared slots.
pub const GAP_CAPACITY: u32 = (MAX_GAPS + SHARED_GAP_SLOTS) as u32;

/// Words in one payload element (a packed raw element).
pub const PAYLOAD_WORDS: u32 = 1;

/// Payload elements available for inserted and captured text runs.
pub const PAYLOAD_CAPACITY: u32 = 0x1_0000;

/// Words in one text element (a packed raw element).
pub const TEXT_WORDS: u32 = 1;

/// Elements each text plane can hold.
pub const TEXT_CAPACITY: u32 = 0x10_0000;

/// First word of the control plane.
pub const CONTROL_OFFSET: u32 = 0;

/// First word of the pending operation queue.
pub const OPS_OFFSET: u32 = CONTROL_OFFSET + CONTROL_WORDS;

/// First word of the retained operation log.
pub const LOG_OFFSET: u32 = OPS_OFFSET + OP_WORDS * OP_CAPACITY;

/// First word of the boundary table.
pub const BOUNDARIES_OFFSET: u32 = LOG_OFFSET + LOG_WORDS * LOG_CAPACITY;

/// First word of the cursor table.
pub const CURSORS_OFFSET: u32 = BOUNDARIES_OFFSET + BOUNDARY_WORDS * BOUNDARY_CAPACITY;

/// First word of the staging-gap table.
pub const GAPS_OFFSET: u32 = CURSORS_OFFSET + CURSOR_WORDS * CURSOR_CAPACITY;

/// First word of the payload plane.
pub const PAYLOAD_OFFSET: u32 = GAPS_OFFSET + GAP_WORDS * GAP_CAPACITY;

/// First word of text plane A.
pub const TEXT_A_OFFSET: u32 = PAYLOAD_OFFSET + PAYLOAD_WORDS * PAYLOAD_CAPACITY;

/// First word of text plane B.
pub const TEXT_B_OFFSET: u32 = TEXT_A_OFFSET + TEXT_WORDS * TEXT_CAPACITY;

/// Total words in a conforming image.
pub const IMAGE_WORDS: u32 = TEXT_B_OFFSET + TEXT_WORDS * TEXT_CAPACITY;

/// A plane of the protocol image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    /// Batch state: lengths, counts, flags.
    Control,
    /// The pending operation queue.
    Ops,
    /// Retained operation records, newest last.
    Log,
    /// The source→destination discontinuity table.
    Boundaries,
    /// Caret and selection state, one record per cursor.
    Cursors,
    /// Staging gap descriptors.
    Gaps,
    /// Inserted and captured text runs.
    Payload,
    /// Text plane A.
    TextA,
    /// Text plane B.
    TextB,
}

/// Every plane, in image order.
pub const PLANES: [Plane; 9] = [
    Plane::Control,
    Plane::Ops,
    Plane::Log,
    Plane::Boundaries,
    Plane::Cursors,
    Plane::Gaps,
    Plane::Payload,
    Plane::TextA,
    Plane::TextB,
];

impl Plane {
    /// First word of this plane.
    #[must_use]
    #[inline]
    pub const fn offset(self) -> u32 {
        match self {
            Self::Control => CONTROL_OFFSET,
            Self::Ops => OPS_OFFSET,
            Self::Log => LOG_OFFSET,
            Self::Boundaries => BOUNDARIES_OFFSET,
            Self::Cursors => CURSORS_OFFSET,
            Self::Gaps => GAPS_OFFSET,
            Self::Payload => PAYLOAD_OFFSET,
            Self::TextA => TEXT_A_OFFSET,
            Self::TextB => TEXT_B_OFFSET,
        }
    }

    /// Words this plane occupies in the image.
    #[must_use]
    #[inline]
    pub const fn words(self) -> u32 {
        match self {
            Self::Control => CONTROL_WORDS,
            Self::Ops => OP_WORDS * OP_CAPACITY,
            Self::Log => LOG_WORDS * LOG_CAPACITY,
            Self::Boundaries => BOUNDARY_WORDS * BOUNDARY_CAPACITY,
            Self::Cursors => CURSOR_WORDS * CURSOR_CAPACITY,
            Self::Gaps => GAP_WORDS * GAP_CAPACITY,
            Self::Payload => PAYLOAD_WORDS * PAYLOAD_CAPACITY,
            Self::TextA | Self::TextB => TEXT_WORDS * TEXT_CAPACITY,
        }
    }

    /// Words in one record of this plane, or `None` for the planes
    /// that hold a run of single-word elements.
    #[inline]
    pub const fn record_words(self) -> Option<u32> {
        match self {
            Self::Ops | Self::Log => Some(OP_WORDS),
            Self::Boundaries => Some(BOUNDARY_WORDS),
            Self::Cursors => Some(CURSOR_WORDS),
            Self::Gaps => Some(GAP_WORDS),
            Self::Control | Self::Payload | Self::TextA | Self::TextB => None,
        }
    }

    /// Records this plane holds, or `None` for the control plane,
    /// which has no repeating structure.
    #[inline]
    pub const fn capacity(self) -> Option<u32> {
        match self {
            Self::Ops => Some(OP_CAPACITY),
            Self::Log => Some(LOG_CAPACITY),
            Self::Boundaries => Some(BOUNDARY_CAPACITY),
            Self::Cursors => Some(CURSOR_CAPACITY),
            Self::Gaps => Some(GAP_CAPACITY),
            Self::Payload => Some(PAYLOAD_CAPACITY),
            Self::TextA | Self::TextB => Some(TEXT_CAPACITY),
            Self::Control => None,
        }
    }

    /// First byte of this plane, for host buffer regions.
    #[must_use]
    #[inline]
    pub const fn offset_bytes(self) -> usize {
        self.offset() as usize * WORD_BYTES
    }

    /// Bytes this plane occupies, for host buffer regions.
    #[must_use]
    #[inline]
    pub const fn size_bytes(self) -> usize {
        self.words() as usize * WORD_BYTES
    }
}

/// Absolute word of `field` in record `record` of `plane`.
///
/// This is the one arithmetic a kernel mirrors: plane base, plus
/// record index times the record stride, plus the field index.
///
/// # Errors
///
/// [`WreditError::InvalidOperation`] when the plane holds no records;
/// [`WreditError::OutOfRange`] when `record` escapes the plane or
/// `field` escapes one record.
pub const fn record_word(plane: Plane, record: u32, field: u32) -> WreditResult<u32> {
    match plane.record_words() {
        None => Err(WreditError::InvalidOperation {
            what: "plane is not record-structured",
        }),
        Some(stride) => match plane.capacity() {
            None => Err(WreditError::InvalidOperation {
                what: "plane has no capacity",
            }),
            Some(capacity) => {
                if record >= capacity {
                    Err(WreditError::OutOfRange {
                        start: record as u64,
                        end: capacity as u64,
                    })
                } else if field >= stride {
                    Err(WreditError::OutOfRange {
                        start: field as u64,
                        end: stride as u64,
                    })
                } else {
                    Ok(plane.offset() + record * stride + field)
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planes_tile_the_image_without_gaps_or_overlap() {
        let mut end = 0u32;
        for plane in PLANES {
            assert_eq!(plane.offset(), end, "{plane:?} does not follow its predecessor");
            end = plane.offset() + plane.words();
        }
        assert_eq!(end, IMAGE_WORDS, "the last plane must end the image exactly");
    }

    #[test]
    fn plane_byte_ranges_agree_with_the_word_geometry() {
        for plane in PLANES {
            assert_eq!(plane.offset_bytes(), plane.offset() as usize * WORD_BYTES);
            assert_eq!(plane.size_bytes(), plane.words() as usize * WORD_BYTES);
        }
    }

    #[test]
    fn record_planes_divide_into_whole_records() {
        for plane in PLANES {
            let Some(stride) = plane.record_words() else {
                continue;
            };
            assert_eq!(plane.words() % stride, 0, "{plane:?} ends mid-record");
            assert_eq!(
                plane.words() / stride,
                plane.capacity().unwrap_or(0),
                "{plane:?} capacity and its word count disagree"
            );
        }
    }

    #[test]
    fn capacities_match_the_engine_limits_they_mirror() {
        assert_eq!(CURSOR_CAPACITY, MAX_CURSORS as u32);
        assert_eq!(GAP_CAPACITY, (MAX_GAPS + SHARED_GAP_SLOTS) as u32);
        assert_eq!(
            BOUNDARY_CAPACITY,
            2 * OP_CAPACITY + GAP_CAPACITY + 1,
            "the boundary table must hold one row per queued op, one per gap, and the terminator"
        );
    }

    #[test]
    fn the_image_fits_well_inside_a_kernel_word_offset() {
        const {
            assert!(
                IMAGE_WORDS < u32::MAX / 2,
                "plane offsets must stay far from a u32 wrap"
            )
        };
    }

    #[test]
    fn the_payload_and_text_planes_start_where_the_offsets_say() {
        assert_eq!(PAYLOAD_OFFSET, GAPS_OFFSET + GAP_WORDS * GAP_CAPACITY);
        assert_eq!(TEXT_A_OFFSET, PAYLOAD_OFFSET + PAYLOAD_CAPACITY);
        assert_eq!(TEXT_B_OFFSET, TEXT_A_OFFSET + TEXT_CAPACITY);
        assert_eq!(IMAGE_WORDS, TEXT_B_OFFSET + TEXT_CAPACITY);
    }
}
