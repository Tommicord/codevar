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

//! The device image: staging a batch and running the pass it describes.
//!
//! [`Image`] owns one PTBP buffer — every plane, at the offsets
//! [`super::super::ptbp::geometry`] lays out — and it is the only place
//! the host writes protocol words. Two things follow from that, and
//! both matter here:
//!
//! * [`Image::apply`] reads the *boundary table out of the image*, not
//!   from the [`Batch`](super::sequentially_batch::Batch) that produced
//!   it. The CPU fallback therefore exercises the same words a kernel
//!   reads, so a host that fills a field a kernel never looks at, or
//!   forgets one it does, fails a test instead of diverging in the
//!   field.
//! * The staging calls are the dispatch list. Filling the planes and
//!   launching the passes are the same sequence of writes, so the
//!   device path replaces [`Image::apply`] with enqueue calls and keeps
//!   every staging step.
//!
//! # Planes and rounds
//!
//! [`Image::stage`] writes the source plane, the payload plane, the
//! operation queue and the boundary table; [`Image::apply`] writes the
//! destination plane, sets [`crate::ptbp::CTRL_DST_LEN`] and flips
//! [`crate::ptbp::FLAG_PARITY`], which makes the plane it wrote the
//! source of the next round. A round is therefore
//! stage-then-apply, and a document can pass through any number of
//! them without the host holding a second copy.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::{WreditError, WreditResult};
use crate::ptbp::fields::{
    CTRL_BATCH, CTRL_BOUNDARY_COUNT, CTRL_DST_LEN, CTRL_FLAGS, CTRL_OP_COUNT, CTRL_PAYLOAD_USED, CTRL_SRC_LEN,
};
use crate::ptbp::{
    BOUNDARY_CAPACITY, BoundaryRow, FLAG_PARITY, IMAGE_WORDS, OP_CAPACITY, OpRecord, PAYLOAD_CAPACITY, Plane,
    TEXT_CAPACITY, WORD_BYTES, control_at, dst_plane, init, read_run, set_control, src_plane, write_run,
};

use super::sequentially_batch::{Batch, apply_rows};

/// One PTBP image, sized for the full protocol geometry.
///
/// The buffer is `IMAGE_WORDS` words — the planes tile it with no gaps,
/// so every offset in the layout is addressable in every image. The
/// size is fixed by the protocol, not by the document: a kernel's
/// bounds guard reads [`crate::ptbp::CTRL_TEXT_CAPACITY`], and an image
/// that varied per document would vary that guard too.
#[derive(Debug, Clone)]
pub struct Image {
    /// The whole image, little-endian words.
    data: Vec<u8>,
}

impl Image {
    /// Allocates and stamps an image carrying `flags`.
    ///
    /// The text and payload planes are left zeroed by
    /// [`crate::ptbp::init`], which is what the protocol says to do:
    /// a document is uploaded before it is read.
    pub fn new(flags: u32) -> WreditResult<Self> {
        let mut data = vec![0u8; IMAGE_WORDS as usize * WORD_BYTES];
        init(&mut data, flags)?;
        Ok(Self { data })
    }

    /// The image bytes, as a device buffer sees them.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// The image bytes for an upload that bypasses the plane helpers.
    #[must_use]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// The batch flags currently stamped in the control plane.
    pub fn flags(&self) -> WreditResult<u32> {
        control_at(&self.data, CTRL_FLAGS)
    }

    /// Reads one control field.
    pub fn control(&self, field: u32) -> WreditResult<u32> {
        control_at(&self.data, field)
    }

    /// The plane the next pass reads.
    pub fn source_plane(&self) -> WreditResult<Plane> {
        Ok(src_plane(self.flags()?))
    }

    /// The plane the next pass writes.
    pub fn destination_plane(&self) -> WreditResult<Plane> {
        Ok(dst_plane(self.flags()?))
    }

    /// Uploads `text` as the source document.
    ///
    /// Only `text.len()` elements are written; the rest of the plane
    /// keeps whatever an earlier round left, which is why every read is
    /// bounded by [`crate::ptbp::CTRL_SRC_LEN`] and never by the
    /// capacity.
    pub fn upload_text(&mut self, text: &[u32]) -> WreditResult<()> {
        let plane = self.source_plane()?;
        let count = u32::try_from(text.len()).map_err(|_| WreditError::LimitExceeded { what: "text" })?;
        if count > TEXT_CAPACITY {
            return Err(WreditError::OutOfRange {
                start: count as u64,
                end: TEXT_CAPACITY as u64,
            });
        }
        write_run(&mut self.data, plane, 0, text)?;
        set_control(&mut self.data, CTRL_SRC_LEN, count)
    }

    /// Appends `run` to the payload plane and returns its first word.
    ///
    /// The payload cursor is the control field the gap-commit kernel
    /// shares, so an insert op can only name a run this call handed
    /// back.
    pub fn push_payload(&mut self, run: &[u32]) -> WreditResult<u32> {
        let base = self.control(CTRL_PAYLOAD_USED)?;
        let count =
            u32::try_from(run.len()).map_err(|_| WreditError::LimitExceeded { what: "payload run" })?;
        let end = base
            .checked_add(count)
            .ok_or(WreditError::OutOfRange {
                start: base as u64,
                end: u64::from(u32::MAX),
            })?;
        if end > PAYLOAD_CAPACITY {
            return Err(WreditError::OutOfRange {
                start: end as u64,
                end: PAYLOAD_CAPACITY as u64,
            });
        }
        write_run(&mut self.data, Plane::Payload, base, run)?;
        set_control(&mut self.data, CTRL_PAYLOAD_USED, end)?;
        Ok(base)
    }

    /// Clears the payload cursor for the next round.
    ///
    /// Payload runs are scratch: once the boundary table names them and
    /// a pass has written the destination plane, nothing downstream
    /// reads them.
    pub fn reset_payload(&mut self) -> WreditResult<()> {
        set_control(&mut self.data, CTRL_PAYLOAD_USED, 0)
    }

    /// Queues `ops` and returns how many fit the operation plane.
    ///
    /// The queue is written whole: a record the planner never filled
    /// would still be read by a kernel walking to
    /// [`crate::ptbp::CTRL_OP_COUNT`], so the count is the only bound
    /// and stale slots past it are never visited.
    pub fn queue_ops(&mut self, ops: &[OpRecord]) -> WreditResult<u32> {
        let count = u32::try_from(ops.len()).map_err(|_| WreditError::LimitExceeded {
            what: "operation queue",
        })?;
        if count > OP_CAPACITY {
            return Err(WreditError::LimitExceeded {
                what: "operation queue",
            });
        }
        for (index, op) in ops.iter().enumerate() {
            op.write(&mut self.data, index as u32)?;
        }
        set_control(&mut self.data, CTRL_OP_COUNT, count)?;
        Ok(count)
    }

    /// Writes `rows` as the boundary table of the next pass.
    pub fn write_rows(&mut self, rows: &[BoundaryRow]) -> WreditResult<u32> {
        let count = u32::try_from(rows.len()).map_err(|_| WreditError::LimitExceeded {
            what: "boundary table",
        })?;
        if count > BOUNDARY_CAPACITY {
            return Err(WreditError::LimitExceeded {
                what: "boundary table",
            });
        }
        for (index, row) in rows.iter().enumerate() {
            row.write(&mut self.data, index as u32)?;
        }
        set_control(&mut self.data, CTRL_BOUNDARY_COUNT, count)?;
        Ok(count)
    }

    /// Stages one whole round: the document, the queue, the payload
    /// runs the queue names, and the table the queue folds to.
    ///
    /// The payload words of every insert are appended in queue order
    /// and each operation's [`crate::ptbp::OpRecord::payload`] is
    /// rewritten to the word the image actually put them at, which is
    /// how a caller that staged a [`Batch`](super::sequentially_batch::Batch)
    /// by hand keeps the table and the plane in step.
    ///
    /// Returns the folded batch, so the caller can remap cursors with
    /// [`super::shift`] against the same table it just wrote.
    pub fn stage(&mut self, text: &[u32], ops: &[OpRecord], runs: &[&[u32]]) -> WreditResult<Batch> {
        self.reset_payload()?;
        self.upload_text(text)?;
        let mut staged: Vec<OpRecord> = Vec::with_capacity(ops.len());
        let mut batch = Batch::new(u32::try_from(text.len()).unwrap_or(u32::MAX));
        for op in ops {
            let mut record = *op;
            if record.count > 0 {
                let run = runs
                    .get(op.payload as usize)
                    .ok_or(WreditError::OutOfRange {
                        start: u64::from(op.payload),
                        end: runs.len() as u64,
                    })?;
                if u32::try_from(run.len()).unwrap_or(u32::MAX) != record.count {
                    return Err(WreditError::OutOfRange {
                        start: run.len() as u64,
                        end: u64::from(record.count),
                    });
                }
                record.payload = self.push_payload(run)?;
            }
            batch.fold(&record)?;
            staged.push(record);
        }
        self.queue_ops(&staged)?;
        let rows = batch.rows();
        self.write_rows(&rows)?;
        Ok(batch)
    }

    /// The boundary table currently staged, read back from the image.
    pub fn rows(&self) -> WreditResult<Vec<BoundaryRow>> {
        let count = self.control(CTRL_BOUNDARY_COUNT)?;
        let mut rows = Vec::with_capacity(count as usize);
        for index in 0..count {
            rows.push(BoundaryRow::read(&self.data, index)?);
        }
        Ok(rows)
    }

    /// The payload plane currently staged.
    pub fn payload(&self) -> WreditResult<Vec<u32>> {
        let used = self.control(CTRL_PAYLOAD_USED)?;
        let mut run = vec![0u32; used as usize];
        read_run(&self.data, Plane::Payload, 0, used, &mut run)?;
        Ok(run)
    }

    /// The document the next pass reads.
    pub fn text(&self) -> WreditResult<Vec<u32>> {
        let plane = self.source_plane()?;
        let len = self.control(CTRL_SRC_LEN)?;
        let mut text = vec![0u32; len as usize];
        read_run(&self.data, plane, 0, len, &mut text)?;
        Ok(text)
    }

    /// True when the staged table edits nothing, which is the dispatcher
    /// signalling that no pass is needed.
    ///
    /// Skipping a round must not flip the parity: the untouched plane
    /// stays the source, so the caller that skips simply does not
    /// advance the round.
    pub fn is_noop(&self) -> WreditResult<bool> {
        Ok(self.control(CTRL_BOUNDARY_COUNT)? == 0)
    }

    /// Runs the staged round on the CPU and advances to the next one.
    ///
    /// This is the fallback the error type promises: the same table,
    /// payload and source plane a kernel would read, applied through
    /// [`apply_rows`], written to the other text plane, after which the
    /// parity flips and the source length becomes the destination
    /// length.
    ///
    /// # Errors
    ///
    /// [`WreditError::OutOfRange`] when a staged row addresses past a
    /// plane, [`WreditError::InvalidOperation`] when the table is not a
    /// pass a kernel could follow, and the plane errors of
    /// [`read_run`] and [`write_run`].
    pub fn apply(&mut self) -> WreditResult<u32> {
        let rows = self.rows()?;
        let payload = self.payload()?;
        let text = self.text()?;
        let out = apply_rows(&text, &rows, &payload)?;
        let plane = self.destination_plane()?;
        write_run(&mut self.data, plane, 0, &out)?;
        let len = u32::try_from(out.len()).map_err(|_| WreditError::LimitExceeded { what: "text" })?;
        set_control(&mut self.data, CTRL_DST_LEN, len)?;
        set_control(&mut self.data, CTRL_SRC_LEN, len)?;
        let flags = self.flags()?;
        set_control(&mut self.data, CTRL_FLAGS, flags ^ FLAG_PARITY)?;
        let batch = self.control(CTRL_BATCH)?.wrapping_add(1);
        set_control(&mut self.data, CTRL_BATCH, batch)?;
        self.reset_payload()?;
        Ok(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptbp::fields::CTRL_LOG_COUNT;
    use crate::ptbp::{FLAG_LOG, NO_PAYLOAD, OP_DELETE, OP_PUT, OP_REPLACE, is_conforming};

    fn op(kind: u32, pos: u32, len: u32, run: Option<&[u32]>) -> (OpRecord, Vec<u32>) {
        let payload = run.map(<[u32]>::to_vec).unwrap_or_default();
        let record = OpRecord {
            kind,
            pos,
            len,
            count: payload.len() as u32,
            // The stage call rewrites this to the word the image used.
            payload: if payload.is_empty() { NO_PAYLOAD } else { 0 },
            cursor: 0,
            flags: 0,
        };
        (record, payload)
    }

    /// An operation list plus the runs it names, as `stage` wants them:
    /// operation `i` names run `i`, and `stage` rewrites the field to
    /// the word the payload plane put it at.
    fn script(items: &[(OpRecord, Vec<u32>)]) -> (Vec<OpRecord>, Vec<&[u32]>) {
        let runs: Vec<&[u32]> = items
            .iter()
            .map(|(_, run)| run.as_slice())
            .collect();
        let ops = items
            .iter()
            .enumerate()
            .map(|(index, (record, _))| {
                let mut staged = *record;
                if staged.count > 0 {
                    staged.payload = index as u32;
                }
                staged
            })
            .collect();
        (ops, runs)
    }

    #[test]
    fn a_new_image_is_conforming_and_empty() {
        let image = Image::new(FLAG_LOG).expect("allocate");
        assert!(is_conforming(image.as_bytes()));
        assert_eq!(image.control(CTRL_FLAGS).expect("flags"), FLAG_LOG);
        assert_eq!(image.control(CTRL_SRC_LEN).expect("length"), 0);
        assert_eq!(image.control(CTRL_OP_COUNT).expect("ops"), 0);
        assert_eq!(image.control(CTRL_BOUNDARY_COUNT).expect("rows"), 0);
        assert!(image.is_noop().expect("noop check"));
    }

    #[test]
    fn staging_writes_the_words_the_pass_reads() {
        let mut image = Image::new(0).expect("image");
        let items = [op(OP_PUT, 2, 0, Some(&[100, 101])), op(OP_DELETE, 0, 1, None)];
        let (ops, runs) = script(&items);
        let batch = image
            .stage(&[1, 2, 3, 4], &ops, &runs)
            .expect("stage");
        assert_eq!(image.control(CTRL_SRC_LEN).expect("src"), 4);
        assert_eq!(image.control(CTRL_OP_COUNT).expect("ops"), 2);
        assert_eq!(
            image.control(CTRL_BOUNDARY_COUNT).expect("rows") as usize,
            batch.rows().len()
        );
        // The queued records name the payload words the image holds.
        let queued = OpRecord::read(image.as_bytes(), 0).expect("read back");
        assert_eq!(queued.count, 2);
        assert_eq!(image.payload().expect("payload"), [100, 101]);
        assert_eq!(image.rows().expect("rows"), batch.rows());
    }

    #[test]
    fn a_staged_round_edits_like_the_same_ops_run_in_order() {
        let mut image = Image::new(0).expect("image");
        let items = [
            op(OP_REPLACE, 1, 2, Some(&[500, 501, 502])),
            op(OP_PUT, 0, 0, Some(&[400])),
            op(OP_DELETE, 4, 2, None),
        ];
        let (ops, runs) = script(&items);
        image
            .stage(&[1, 2, 3, 4, 5, 6], &ops, &runs)
            .expect("stage");
        let len = image.apply().expect("apply");
        // Replace 2,3 with three words, put one in front, then delete
        // two at position four of the document those produced.
        assert_eq!(len, 6);
        assert_eq!(image.text().expect("text"), [400, 1, 500, 501, 5, 6]);
        assert_eq!(image.control(CTRL_DST_LEN).expect("dst"), 6);
        assert_eq!(image.control(CTRL_BATCH).expect("batch"), 1);
        assert_eq!(image.control(CTRL_PAYLOAD_USED).expect("payload"), 0);
    }

    #[test]
    fn a_round_leaves_the_other_plane_ready_to_read() {
        let mut image = Image::new(0).expect("image");
        let first = image.destination_plane().expect("first dst");
        image.upload_text(&[7, 8, 9]).expect("upload");
        image.apply().expect("apply");
        assert_eq!(image.source_plane().expect("new src"), first);
        assert_eq!(
            image.destination_plane().expect("new dst"),
            crate::ptbp::Plane::TextA
        );
        assert_eq!(image.text().expect("text"), [7, 8, 9]);
    }

    #[test]
    fn rounds_compose_on_one_image() {
        let mut image = Image::new(0).expect("image");
        // Each round reads the plane the previous one wrote, so a
        // document is built without the host keeping a copy.
        let mut document: Vec<u32> = Vec::new();
        for word in [1u32, 2, 3, 4] {
            let at = document.len() as u32;
            let items = [op(OP_PUT, at, 0, Some(&[word]))];
            let (ops, runs) = script(&items);
            image
                .stage(&document, &ops, &runs)
                .expect("stage");
            image.apply().expect("apply");
            document.push(word);
            assert_eq!(image.text().expect("text"), document);
        }
        let items = [op(OP_DELETE, 0, 2, None), op(OP_PUT, 0, 0, Some(&[9]))];
        let (ops, runs) = script(&items);
        image
            .stage(&document, &ops, &runs)
            .expect("stage");
        image.apply().expect("apply");
        assert_eq!(image.text().expect("text"), [9, 3, 4]);
        assert_eq!(image.control(CTRL_BATCH).expect("batch"), 5);
    }

    #[test]
    fn an_empty_table_reports_no_pass_needed() {
        let mut image = Image::new(0).expect("image");
        let items = [op(OP_PUT, 2, 0, Some(&[9, 9])), op(OP_DELETE, 2, 2, None)];
        let (ops, runs) = script(&items);
        image
            .stage(&[1, 2, 3], &ops, &runs)
            .expect("stage");
        assert!(image.is_noop().expect("noop check"), "the batch folded away");
        assert_eq!(image.control(CTRL_OP_COUNT).expect("ops"), 2);
    }

    #[test]
    fn a_run_that_is_not_the_insert_length_is_refused() {
        let mut image = Image::new(0).expect("image");
        let (mut record, run) = op(OP_PUT, 0, 0, Some(&[1, 2]));
        record.count = 3;
        let runs: Vec<&[u32]> = vec![run.as_slice()];
        assert!(matches!(
            image.stage(&[1], &[record], &runs),
            Err(WreditError::OutOfRange { .. })
        ));
    }

    #[test]
    fn a_queue_past_the_operation_plane_is_refused() {
        let mut image = Image::new(0).expect("image");
        let ops: Vec<OpRecord> = core::iter::repeat_n(OpRecord::none(), OP_CAPACITY as usize + 1).collect();
        assert!(matches!(
            image.queue_ops(&ops),
            Err(WreditError::LimitExceeded {
                what: "operation queue"
            })
        ));
    }

    #[test]
    fn a_payload_that_would_overrun_the_plane_is_refused() {
        let mut image = Image::new(0).expect("image");
        image.reset_payload().expect("reset");
        let run = vec![0u32; PAYLOAD_CAPACITY as usize - 4];
        image
            .push_payload(&run)
            .expect("most of the plane");
        let over = vec![0u32; 5];
        let err = image
            .push_payload(&over)
            .expect_err("the last five words do not fit");
        assert!(matches!(err, WreditError::OutOfRange { .. }));
        // The cursor did not move: a refused append leaves the plane
        // usable for the next run.
        assert_eq!(
            image.control(CTRL_PAYLOAD_USED).expect("cursor"),
            run.len() as u32
        );
        image
            .push_payload(&[0u32; 4])
            .expect("exactly what fits");
        assert_eq!(image.control(CTRL_LOG_COUNT).expect("log"), 0);
    }
}
