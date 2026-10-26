//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Change tracking for the text pass.
//!
//! A text frame that did not change must not be re-recorded or re-uploaded.
//! The signal a producer raises and a consumer acknowledges carries that
//! decision as a monotonic epoch rather than a boolean, so a consumer that
//! misses several changes still redraws exactly once when it next runs.
//!
//! The second half of the module coalesces the atlas's dirty rectangles.
//! The atlas records one rectangle per glyph and per page allocation; the
//! pipe would copy each to the GPU separately. Merging the rectangles that
//! touch on the same page turns a line of freshly rasterized glyphs into a
//! handful of copies without reading a texel back from the GPU.

use alloc::vec::Vec;

use crate::atlas::DirtyRect;

/// Texels of gap that still count as touching when merging rectangles.
///
/// The shelf packer leaves one transparent texel between two glyphs, so two
/// glyphs that sit next to each other produce rectangles one texel apart.
const MERGE_SLACK: u32 = 1;

/// A producer-to-consumer dirty flag keyed by a monotonic epoch.
///
/// The producer calls [`DirtySignal::mark`] whenever the output changes and
/// the consumer calls [`DirtySignal::take`] when it has drawn the change. A
/// signal starts dirty so the first frame is never skipped.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DirtySignal {
    /// Epoch of the most recent change.
    epoch: u64,
    /// Epoch the consumer last drew.
    consumed: u64,
}

impl DirtySignal {
    /// A signal that is dirty, so a fresh consumer draws once.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            epoch: 1,
            consumed: 0,
        }
    }

    /// Records a change.
    ///
    /// The epoch wraps after `u64::MAX` marks; a wrap can only lose a
    /// redraw after roughly six hundred million years at sixty frames per
    /// second, so the wrap is left unguarded rather than panicking.
    #[inline]
    pub fn mark(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// The epoch of the most recent change.
    #[inline]
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Whether the signal holds a change the consumer has not drawn.
    #[inline]
    #[must_use]
    pub const fn is_dirty(&self) -> bool {
        self.epoch != self.consumed
    }

    /// Clears the signal and reports whether it was dirty.
    #[inline]
    pub fn take(&mut self) -> bool {
        let dirty = self.is_dirty();
        self.consumed = self.epoch;
        dirty
    }

    /// Marks the current epoch as drawn without reporting it.
    #[inline]
    pub fn acknowledge(&mut self) {
        self.consumed = self.epoch;
    }
}

/// Merges rectangles that overlap or touch on the same page.
///
/// Rectangles are written into `scratch`, which the caller reuses so that
/// coalescing a frame allocates nothing after its first call. On return
/// `scratch` holds the merged rectangles in `(page, y, x)` order.
///
/// The merge is a single sorted pass: a rectangle joins the last merged one
/// when their x and y extents meet. That is enough for an atlas, where a
/// frame's new glyphs land on one or two shelves, and it never grows a
/// rectangle past the union of the inputs.
pub fn coalesce(rects: &[DirtyRect], scratch: &mut Vec<DirtyRect>) {
    scratch.clear();
    scratch.extend_from_slice(rects);
    scratch.sort_unstable_by_key(|rect| (rect.page, rect.y, rect.x));

    let mut write = 0usize;
    for read in 0..scratch.len() {
        let next = scratch[read];
        let merged = if write > 0 {
            union(scratch[write - 1], next)
        } else {
            None
        };
        if let Some(merged) = merged {
            scratch[write - 1] = merged;
            continue;
        }
        scratch[write] = next;
        write += 1;
    }
    scratch.truncate(write);
}

/// The smallest rectangle covering `a` and `b` when they touch on one page.
///
/// Touching edges count as overlapping, and a texel of slack is allowed so
/// the one-texel padding the shelf packer leaves between glyphs does not
/// split a single upload in two.
fn union(a: DirtyRect, b: DirtyRect) -> Option<DirtyRect> {
    if a.page != b.page {
        return None;
    }

    let a_right = a.x.saturating_add(a.width);
    let b_right = b.x.saturating_add(b.width);
    let a_bottom = a.y.saturating_add(a.height);
    let b_bottom = b.y.saturating_add(b.height);

    let overlap_x = b.x <= a_right.saturating_add(MERGE_SLACK) && a.x <= b_right.saturating_add(MERGE_SLACK);
    let overlap_y =
        b.y <= a_bottom.saturating_add(MERGE_SLACK) && a.y <= b_bottom.saturating_add(MERGE_SLACK);
    if !overlap_x || !overlap_y {
        return None;
    }

    let x = a.x.min(b.x);
    let y = a.y.min(b.y);
    Some(DirtyRect {
        page: a.page,
        x,
        y,
        width: a_right.max(b_right) - x,
        height: a_bottom.max(b_bottom) - y,
    })
}

#[cfg(test)]
mod tests {
    //! Epoch accounting and rectangle merging.

    use super::*;
    use alloc::vec;

    fn rect(page: u32, x: u32, y: u32, width: u32, height: u32) -> DirtyRect {
        DirtyRect {
            page,
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn a_fresh_signal_is_dirty_for_the_first_frame() {
        let mut signal = DirtySignal::new();
        assert!(signal.is_dirty());
        assert!(signal.take());
        assert!(!signal.is_dirty());
        assert!(!signal.take(), "an idle frame stays clean");
    }

    #[test]
    fn a_mark_survives_until_it_is_taken() {
        let mut signal = DirtySignal::new();
        signal.take();
        signal.mark();
        signal.mark();
        assert!(signal.is_dirty());
        assert_eq!(signal.epoch(), 3);
        assert!(signal.take());
        assert!(!signal.is_dirty());
    }

    #[test]
    fn acknowledging_a_change_does_not_report_it() {
        let mut signal = DirtySignal::new();
        signal.take();
        signal.mark();
        signal.acknowledge();
        assert!(!signal.is_dirty());
        signal.mark();
        assert!(signal.is_dirty(), "a later change is still visible");
    }

    #[test]
    fn coalescing_keeps_disjoint_rectangles_apart() {
        let mut scratch = Vec::new();
        coalesce(&[rect(0, 0, 0, 4, 4), rect(0, 20, 20, 4, 4)], &mut scratch);
        assert_eq!(scratch.len(), 2);
    }

    #[test]
    fn overlapping_rectangles_merge_into_their_union() {
        let mut scratch = Vec::new();
        coalesce(&[rect(0, 0, 0, 4, 4), rect(0, 2, 2, 4, 4)], &mut scratch);
        assert_eq!(scratch, vec![rect(0, 0, 0, 6, 6)]);
    }

    #[test]
    fn a_one_texel_gap_still_merges() {
        let mut scratch = Vec::new();
        coalesce(&[rect(0, 0, 0, 4, 4), rect(0, 5, 0, 4, 4)], &mut scratch);
        assert_eq!(scratch, vec![rect(0, 0, 0, 9, 4)]);
    }

    #[test]
    fn rectangles_of_different_pages_never_merge() {
        let mut scratch = Vec::new();
        coalesce(&[rect(0, 0, 0, 4, 4), rect(1, 0, 0, 4, 4)], &mut scratch);
        assert_eq!(scratch.len(), 2);
        assert_eq!(scratch[0].page, 0);
        assert_eq!(scratch[1].page, 1);
    }

    #[test]
    fn a_chain_of_touching_rectangles_merges_in_one_pass() {
        let mut scratch = Vec::new();
        coalesce(
            &[rect(0, 0, 0, 2, 2), rect(0, 2, 0, 2, 2), rect(0, 4, 0, 2, 2)],
            &mut scratch,
        );
        assert_eq!(scratch, vec![rect(0, 0, 0, 6, 2)]);
    }

    #[test]
    fn coalescing_reuses_the_scratch_allocation() {
        let mut scratch = Vec::new();
        coalesce(&[rect(0, 0, 0, 4, 4)], &mut scratch);
        let capacity = scratch.capacity();
        coalesce(&[rect(0, 0, 0, 4, 4)], &mut scratch);
        assert_eq!(scratch.len(), 1);
        assert_eq!(scratch.capacity(), capacity);
    }
}
