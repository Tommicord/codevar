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

//! Folding a sequential operation queue into one parallel pass.
//!
//! The queue is sequential by construction: operation `k` states its
//! [`crate::ptbp::OpRecord::pos`] in the document the first `k`
//! operations already produced. A device cannot apply that directly —
//! every work item would have to wait for the previous one — so the
//! queue is first *folded* into a description that lives entirely in
//! source coordinates, and then applied in one pass.
//!
//! # The piece list
//!
//! Folding needs a working model of the half-applied document, and
//! [`Batch`] keeps the cheapest one that is still exact: an ordered
//! list of [`Piece`] entries, each either a slice of the source text
//! or a run of the payload plane. A list of pieces is the document,
//! so applying an operation is a split-and-splice on the list and no
//! position ever has to be guessed.
//!
//! # The boundary table
//!
//! [`Batch::rows`] linearises the list into the [`Plane::Boundaries`]
//! table the kernels consume. Each hole in the source contributes one
//! **drop row** (`delta` is the negated length of the hole: those
//! source elements have no destination), and each payload piece
//! contributes one **insert row** (`delta` is the run length,
//! [`BoundaryRow::payload`] its first word). A replace therefore costs
//! two rows, not one: a four-word row cannot carry both the removed
//! length and the inserted length, which is why the geometry sizes the
//! table for twice the operation capacity.
//!
//! Rows are emitted in source order, and rows sharing a position are
//! emitted inserts-first. That ordering is what makes the destination
//! of an inserted run `pos + cum` of the previous row — the text the
//! batch removed is gone from the destination before the text
//! replacing it is written, so the two never collide and no pass needs
//! an atomic or a barrier.
//!
//! [`apply_rows`] is the scalar reading of that contract: one binary
//! search per source element, one copy per insert row. It is the CPU
//! fallback and the oracle every PTBP kernel is compared against.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::{WreditError, WreditResult};
use crate::ptbp::{
    BoundaryRow, NO_PAYLOAD, OP_CAPACITY, OP_OVERWRITE, OpClass, OpRecord, PAYLOAD_CAPACITY, TEXT_CAPACITY,
};

/// Piece budget of a batch: two pieces per queued operation, which is
/// also the number of boundary rows [`Batch::rows`] can emit for it.
pub const MAX_PIECES: usize = 2 * OP_CAPACITY as usize;

/// Row budget of a batch: every piece becomes at most one row, plus
/// one drop row for the hole the trailing payload pieces sit in.
pub const MAX_ROWS: usize = MAX_PIECES + 1;

/// What a [`Piece`] of a batch's working document points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceKind {
    /// A live slice of the source text plane, never edited again by
    /// this batch.
    Source,
    /// Text the batch inserts, held in the payload plane.
    Payload,
}

/// One run of the half-applied document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    /// Which plane [`Self::start`] indexes.
    pub kind: PieceKind,
    /// First element of the run: a source index, or a payload word
    /// index.
    pub start: u32,
    /// Elements in the run; never zero.
    pub len: u32,
}

impl Piece {
    /// A live slice of the source text.
    #[must_use]
    #[inline]
    pub const fn source(start: u32, len: u32) -> Self {
        Self {
            kind: PieceKind::Source,
            start,
            len,
        }
    }

    /// An inserted run read from the payload plane.
    #[must_use]
    #[inline]
    pub const fn payload(start: u32, len: u32) -> Self {
        Self {
            kind: PieceKind::Payload,
            start,
            len,
        }
    }
}

/// A document being edited by a queue of operations, held in pieces.
///
/// The batch owns no text: it names source slices and payload runs, so
/// folding an operation is bounded by the piece list and never by the
/// document length. That is the whole point of the representation —
/// the document-sized work happens once, in the apply pass, on the
/// device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    /// Elements in the source text plane.
    src_len: u32,
    /// The document, in current-space order.
    pieces: Vec<Piece>,
    /// Cached document length in current space.
    len: u32,
}

impl Batch {
    /// A batch over a source document of `src_len` elements, with no
    /// operation folded yet.
    #[must_use]
    pub fn new(src_len: u32) -> Self {
        let pieces = if src_len == 0 {
            Vec::new()
        } else {
            vec![Piece::source(0, src_len)]
        };
        Self {
            src_len,
            pieces,
            len: src_len,
        }
    }

    /// Elements in the source text plane.
    #[must_use]
    #[inline]
    pub const fn src_len(&self) -> u32 {
        self.src_len
    }

    /// Length of the document after the operations folded so far.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// Whether the folded document is empty.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The working document, in current-space order.
    #[must_use]
    #[inline]
    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    /// Folds one text-editing operation into the batch.
    ///
    /// Positions are read in current space — the document as the
    /// operations folded so far left it — which is how the queue
    /// states them.
    ///
    /// # Errors
    ///
    /// [`WreditError::OutOfRange`] when the range escapes the folded
    /// document or the payload run escapes the payload plane,
    /// [`WreditError::InvalidOperation`] for a tag that is not a plain
    /// text edit, for an insert with no payload, and for an overwrite
    /// that changes the length,
    /// [`WreditError::LimitExceeded`] when the piece list would grow
    /// past [`MAX_ROWS`], which means the queue must be cut into two
    /// passes.
    pub fn fold(&mut self, op: &OpRecord) -> WreditResult<()> {
        match op.class()? {
            OpClass::Neutral => Ok(()),
            OpClass::Edit => self.fold_edit(op),
            _ => Err(WreditError::InvalidOperation {
                what: "operation is not a plain text edit",
            }),
        }
    }

    /// Folds the longest editable prefix of `ops` and reports how many
    /// operations it took.
    ///
    /// A [`crate::ptbp::OpClass::Capture`],
    /// [`crate::ptbp::OpClass::Cursor`] or
    /// [`crate::ptbp::OpClass::Replay`] entry is not a text edit and
    /// stops the prefix rather than failing it: the caller dispatches
    /// the batch built so far, schedules that operation from the
    /// planes it belongs to, and starts a new batch with the operation
    /// still queued.
    pub fn fold_prefix(&mut self, ops: &[OpRecord]) -> WreditResult<u32> {
        let mut folded = 0u32;
        for op in ops {
            if !matches!(op.class()?, OpClass::Edit | OpClass::Neutral) {
                break;
            }
            self.fold(op)?;
            folded += 1;
        }
        Ok(folded)
    }

    /// Folds one edit, after the class check.
    fn fold_edit(&mut self, op: &OpRecord) -> WreditResult<()> {
        if op.kind == OP_OVERWRITE && op.len != op.count {
            return Err(WreditError::InvalidOperation {
                what: "overwrite changes the length",
            });
        }
        let end = op
            .pos
            .checked_add(op.len)
            .ok_or(WreditError::OutOfRange {
                start: op.pos as u64,
                end: u64::MAX,
            })?;
        if end > self.len {
            return Err(WreditError::OutOfRange {
                start: end as u64,
                end: self.len as u64,
            });
        }
        if op.count > 0 {
            if op.payload == NO_PAYLOAD {
                return Err(WreditError::InvalidOperation {
                    what: "insert with no payload run",
                });
            }
            let stop = u64::from(op.payload) + u64::from(op.count);
            if stop > u64::from(PAYLOAD_CAPACITY) {
                return Err(WreditError::OutOfRange {
                    start: stop,
                    end: u64::from(PAYLOAD_CAPACITY),
                });
            }
        }
        // Two splits plus the inserted run is the most the splice can
        // add, so the budget is checked before anything is mutated.
        if self.pieces.len() + 3 > MAX_PIECES {
            return Err(WreditError::LimitExceeded {
                what: "batch piece list",
            });
        }
        self.splice(op.pos, op.len, op.count, op.payload);
        self.len = self.len - op.len + op.count;
        Ok(())
    }

    /// Replaces the current-space range `[at, at + delete)` with an
    /// inserted run of `insert` payload words.
    ///
    /// The range is already validated, so the only work is naming the
    /// piece boundaries: a piece the range starts or stops inside is
    /// split, the pieces the range covers leave the list, and the new
    /// run takes their place.
    fn splice(&mut self, at: u32, delete: u32, insert: u32, payload: u32) {
        let first = self.split_before(at);
        let last = self.split_before(at.saturating_add(delete));
        if first != last {
            self.pieces.drain(first..last);
        }
        if insert > 0 {
            self.pieces
                .insert(first, Piece::payload(payload, insert));
        }
    }

    /// Returns the index of the piece that starts at current-space
    /// position `pos`, splitting a piece when `pos` falls inside one.
    ///
    /// `pos` must not exceed the document length; the length is
    /// cached, so the walk stops at the piece that answers it.
    fn split_before(&mut self, pos: u32) -> usize {
        if pos == self.len {
            return self.pieces.len();
        }
        let mut running = 0u32;
        for index in 0..self.pieces.len() {
            let piece = self.pieces[index];
            let next = running + piece.len;
            if pos < next {
                let offset = pos - running;
                if offset == 0 {
                    return index;
                }
                let left = Piece {
                    kind: piece.kind,
                    start: piece.start,
                    len: offset,
                };
                let right = Piece {
                    kind: piece.kind,
                    start: piece.start + offset,
                    len: piece.len - offset,
                };
                self.pieces[index] = left;
                self.pieces.insert(index + 1, right);
                return index + 1;
            }
            running = next;
        }
        self.pieces.len()
    }

    /// Linearises the batch into the boundary table of one apply pass.
    ///
    /// The table is sorted by source position, carries a running
    /// prefix sum in [`BoundaryRow::cum`], and is sized so that
    /// [`dst_of`] answers every element of the source plane. A batch
    /// that folded nothing yields an empty table, which is the honest
    /// answer: the destination is the source, and the pass can be
    /// skipped entirely.
    #[must_use]
    pub fn rows(&self) -> Vec<BoundaryRow> {
        let mut rows = Vec::with_capacity(self.pieces.len() + 1);
        let mut cum = 0i32;
        // Source elements already accounted for by the table.
        let mut consumed = 0u32;
        let mut index = 0usize;
        while index < self.pieces.len() {
            let group = index;
            while index < self.pieces.len() && self.pieces[index].kind == PieceKind::Payload {
                index += 1;
            }
            for piece in &self.pieces[group..index] {
                cum += piece.len.cast_signed();
                rows.push(BoundaryRow {
                    pos: consumed,
                    delta: piece.len.cast_signed(),
                    cum,
                    payload: piece.start,
                });
            }
            let next = self
                .pieces
                .get(index)
                .map_or(self.src_len, |piece| piece.start);
            let hole = next - consumed;
            if hole > 0 {
                cum -= hole.cast_signed();
                rows.push(BoundaryRow {
                    pos: consumed,
                    delta: hole.cast_signed().wrapping_neg(),
                    cum,
                    payload: NO_PAYLOAD,
                });
            }
            let Some(piece) = self.pieces.get(index) else {
                break;
            };
            consumed = piece.start + piece.len;
            index += 1;
        }
        rows
    }

    /// Destination of source element `source` under this batch, or
    /// `None` when the batch removes it.
    ///
    /// A convenience over [`dst_of`] that borrows the table once; the
    /// kernels use the row layout directly, so the two must agree and
    /// the drift is tested.
    #[must_use]
    pub fn destination_of(&self, source: u32) -> Option<u32> {
        dst_of(&self.rows(), source)
    }
}

/// Whether `row` removes source elements.
#[must_use]
#[inline]
pub const fn is_drop(row: &BoundaryRow) -> bool {
    row.delta < 0
}

/// Whether `row` writes a payload run.
#[must_use]
#[inline]
pub const fn is_insert(row: &BoundaryRow) -> bool {
    row.delta > 0 && row.payload != NO_PAYLOAD
}

/// Destination of source element `source` under `rows`, or `None` when
/// a drop row covers it.
///
/// This is the per-element rule a kernel runs: the last row that
/// starts at or before `source` carries both the answer (its prefix
/// sum added to `source`) and the only way `source` can be removed,
/// because drop spans are disjoint and the row directly at or before
/// `source` is the drop row exactly when `source` is inside it.
#[must_use]
pub fn dst_of(rows: &[BoundaryRow], source: u32) -> Option<u32> {
    let reached = rows.partition_point(|row| row.pos <= source);
    if reached == 0 {
        return Some(source);
    }
    let row = rows[reached - 1];
    if is_drop(&row) && source < row.pos.saturating_add(row.delta.unsigned_abs()) {
        return None;
    }
    let shifted = i64::from(source) + i64::from(row.cum);
    u32::try_from(shifted).ok()
}

/// Runs one apply pass over `src`, the source plane, and returns the
/// destination plane.
///
/// It is the scalar form of the two device passes: the text pass
/// scatters every source element that survives to [`dst_of`], and the
/// payload pass writes each insert row at `pos + cum` of the row
/// before it. The two writes are disjoint and together cover the
/// destination exactly — a table that leaves a slot unwritten is
/// rejected rather than copied out, because the destination plane is
/// reused between batches and a hole would read as stale text.
///
/// # Errors
///
/// [`WreditError::OutOfRange`] when a row addresses past either plane
/// or the destination does not fit the text plane,
/// [`WreditError::InvalidOperation`] when `rows` is not sorted, when a
/// drop span overlaps another, or when the pass leaves a slot
/// unwritten.
pub fn apply_rows(src: &[u32], rows: &[BoundaryRow], payload: &[u32]) -> WreditResult<Vec<u32>> {
    check_table(rows)?;
    let net = rows.last().map_or(0, |row| row.cum);
    let width = src.len() as i64 + i64::from(net);
    if width < 0 {
        return Err(WreditError::OutOfRange {
            start: 0,
            end: src.len() as u64,
        });
    }
    if width > i64::from(TEXT_CAPACITY) {
        return Err(WreditError::OutOfRange {
            start: width.cast_unsigned(),
            end: u64::from(TEXT_CAPACITY),
        });
    }
    // The unwrap is safe by the two checks above, and `no_std` has no
    // `try_into` for `i64 -> usize` that keeps the error type useful.
    let len = width as usize;
    let mut dst = vec![0u32; len];

    let mut written = 0usize;
    for (source, word) in src.iter().enumerate() {
        let Some(at) = dst_of(rows, source as u32) else {
            continue;
        };
        let cell = dst
            .get_mut(usize::try_from(at).unwrap_or(usize::MAX))
            .ok_or(WreditError::OutOfRange {
                start: at as u64,
                end: len as u64,
            })?;
        *cell = *word;
        written += 1;
    }

    let mut filled = 0usize;
    for index in 0..rows.len() {
        let row = rows[index];
        if !is_insert(&row) {
            continue;
        }
        let before = if index == 0 { 0 } else { rows[index - 1].cum };
        let base = i64::from(row.pos) + i64::from(before);
        let count = row.delta.cast_unsigned() as usize;
        let run = payload
            .get(row.payload as usize..row.payload as usize + count)
            .ok_or(WreditError::OutOfRange {
                start: row.payload as u64,
                end: payload.len() as u64,
            })?;
        let cells = dst
            .get_mut(
                usize::try_from(base).unwrap_or(usize::MAX)
                    ..usize::try_from(base + count as i64).unwrap_or(usize::MAX),
            )
            .ok_or(WreditError::OutOfRange {
                start: base.cast_unsigned(),
                end: len as u64,
            })?;
        cells.copy_from_slice(run);
        filled += count;
    }

    if written + filled != len {
        return Err(WreditError::InvalidOperation {
            what: "boundary table leaves an unwritten slot",
        });
    }
    Ok(dst)
}

/// Rejects a table a kernel could not follow: rows must be sorted by
/// position, an insert row must carry a positive run, a drop row a
/// positive removal, and no two drop spans may overlap — an overlap
/// would map one source element to two destinations.
fn check_table(rows: &[BoundaryRow]) -> WreditResult<()> {
    for pair in rows.windows(2) {
        if pair[1].pos < pair[0].pos {
            return Err(WreditError::InvalidOperation {
                what: "boundary rows are unsorted",
            });
        }
    }
    for row in rows {
        if row.payload != NO_PAYLOAD && row.delta <= 0 {
            return Err(WreditError::InvalidOperation {
                what: "payload row that inserts nothing",
            });
        }
        if is_drop(row)
            && row
                .pos
                .checked_add(row.delta.unsigned_abs())
                .is_none()
        {
            return Err(WreditError::OutOfRange {
                start: row.pos as u64,
                end: u64::MAX,
            });
        }
    }
    // Drop spans are emitted one per hole, in source order, so they
    // are disjoint exactly when each starts after the previous ends.
    let mut end = 0u32;
    for row in rows.iter().filter(|row| is_drop(row)) {
        if row.pos < end {
            return Err(WreditError::InvalidOperation {
                what: "overlapping drop rows",
            });
        }
        end = row.pos + row.delta.unsigned_abs();
    }
    Ok(())
}

