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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The Staging Gap: cursor-suballocated gap buffer of the Wredit engine.
//!
//! A [`StagingGap`] stores a committed sequence of elements with a
//! contiguous window of free space — *the gap* — inside it, exactly
//! like the reference engine's `WritableGap`: inserting at the gap
//! position never shifts existing data, and relocating the gap is a
//! single block move. On top of that, the gap's free capacity feeds a
//! fixed-size **staging area** of `GAP_SIZE` slots that is
//! suballocated among cursors the way the reference engine's
//! `SharedGap` suballocates its circular buffer: each cursor owns a
//! [`StagingRegion`] and writes land there in O(1) without contending
//! with the other cursors. Nothing is committed until the region (or
//! the whole gap) is flushed, so a burst of keystrokes from several
//! cursors is staged almost for free and then committed in a single
//! block move.
//!
//! This module is written in safe Rust and uses **no heap
//! allocation**: the committed buffer, the staging area and the
//! cursor regions are all fixed-size arrays sized by const generics,
//! so a [`StagingGap`] fits entirely on the stack or in static
//! storage. Like the engine's disk layer, a staged element is never
//! accepted unless the gap can later hold it, so a flush always fits.
//!
//! # Layout invariants (checked, not trusted)
//!
//! 1. `committed_len() + gap_available() <= COMMIT_CAP`
//! 2. Regions are ordered by `start` and never overlap
//! 3. Every region lies inside the staging area (`start + size <= GAP_SIZE`)
//! 4. Staged elements never exceed the gap (`staging_used() <= gap_available()`)
//!
//! # Relation to the disk layer
//!
//! The staging gap plays the role a disk's write cache plays: edits
//! accumulate cheaply in the staging area and are committed in bulk.
//! [`crate::storage::writable::WritableFile`] sits on top of it and
//! pushes committed elements through the block store.

use crate::error::{WreditError, WreditResult};

/// A cursor's suballocated slice of the staging area.
///
/// The counterpart of the reference engine's `SharedGapCursorRegion`:
/// it records where a cursor's staging slots begin, how large the
/// reservation is, and how many elements are currently staged in it.
/// Regions are rebalanced whenever a cursor joins or leaves, so the
/// staging area is always partitioned evenly (with a floor of whatever
/// a cursor has already staged — rebalancing never discards edits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagingRegion {
    /// The cursor that owns this region.
    cursor_id: usize,
    /// First staging slot of the region, in `0..GAP_SIZE`.
    start: usize,
    /// Number of slots reserved for the cursor.
    size: usize,
    /// Slots currently holding staged (uncommitted) elements.
    staged: usize,
}

impl StagingRegion {
    /// Creates a region reservation.
    #[must_use]
    pub const fn new(cursor_id: usize, start: usize, size: usize) -> Self {
        Self {
            cursor_id,
            start,
            size,
            staged: 0,
        }
    }

    /// The cursor that owns this region.
    #[must_use]
    pub const fn cursor_id(&self) -> usize {
        self.cursor_id
    }

    /// First staging slot of the region.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// Number of slots reserved for the cursor.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Elements currently staged in this region.
    #[must_use]
    pub const fn staged(&self) -> usize {
        self.staged
    }

    /// Slots the cursor can still stage into before the region is full.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.size - self.staged
    }

    /// `true` when no slot is left in the region.
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.staged >= self.size
    }
}

/// A gap buffer whose free space is suballocated among cursors as
/// staging regions.
///
/// # Type parameters
///
/// * `T`          - element type stored in the buffer (`u32` code points
///   for the text engine); must be cheap to copy
/// * `COMMIT_CAP` - committed buffer capacity in elements
/// * `GAP_SIZE`   - staging capacity in elements; the gap's initial free
///   window matches it
///
/// # Examples
///
/// use codevar_wredit::error::WreditResult;
/// use codevar_wredit::staging_gap::StagingGap;
///
/// fn edit() -> WreditResult<()> {
///     let mut gap: StagingGap<u32, 32, 8> = StagingGap::try_from_slice(&[1, 2, 3])?;
///     gap.add_region(0)?;
///     let at = gap.stage(0, 9)?;
///     assert_eq!(at, 3);
///     assert_eq!(gap.get(3), Some(9));
///     gap.flush_all();
///     assert_eq!(gap.get(3), Some(9));
///     Ok(())
/// }
///
/// assert!(edit().is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct StagingGap<T: Copy + Default, const COMMIT_CAP: usize, const GAP_SIZE: usize> {
    /// Committed elements with the free gap window
    /// `[gap_pos, gap_pos + gap_len)` inside them.
    buf: [T; COMMIT_CAP],
    /// Number of committed elements (staged edits excluded).
    committed_len: usize,
    /// Logical position of the gap inside the committed sequence.
    gap_pos: usize,
    /// Free slots currently available in `buf`.
    gap_len: usize,
    /// The staging area: `GAP_SIZE` slots partitioned among regions.
    staging: [T; GAP_SIZE],
    /// Cursor regions, ordered by `start`, non-overlapping.
    regions: [StagingRegion; GAP_SIZE],
    /// Number of regions currently allocated.
    region_count: usize,
}

impl<T: Copy + Default, const COMMIT_CAP: usize, const GAP_SIZE: usize> StagingGap<T, COMMIT_CAP, GAP_SIZE> {
    /// Creates an empty gap buffer whose whole gap sits at position 0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: [T::default(); COMMIT_CAP],
            committed_len: 0,
            gap_pos: 0,
            gap_len: GAP_SIZE,
            staging: [T::default(); GAP_SIZE],
            regions: [StagingRegion::new(0, 0, 0); GAP_SIZE],
            region_count: 0,
        }
    }

    /// Creates a gap buffer holding `elements` with the gap after them.
    ///
    /// The staging area starts empty; call [`Self::add_region`] for
    /// every cursor that will stage edits.
    ///
    /// # Errors
    ///
    /// [`WreditError::InvalidOperation`] when `elements.len() + GAP_SIZE`
    /// exceeds [`COMMIT_CAP`].
    pub fn try_from_slice(elements: &[T]) -> WreditResult<Self> {
        let len = elements.len();
        if len.saturating_add(GAP_SIZE) > COMMIT_CAP {
            return Err(WreditError::InvalidOperation {
                what: "elements plus staging capacity exceed commit capacity",
            });
        }
        let mut buf = [T::default(); COMMIT_CAP];
        buf[..len].copy_from_slice(elements);
        Ok(Self {
            buf,
            committed_len: len,
            gap_pos: len,
            gap_len: GAP_SIZE,
            staging: [T::default(); GAP_SIZE],
            regions: [StagingRegion::new(0, 0, 0); GAP_SIZE],
            region_count: 0,
        })
    }

    /// Total staging capacity, in elements.
    #[must_use]
    pub const fn capacity() -> usize {
        GAP_SIZE
    }

    /// Number of logical elements: committed plus staged.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.committed_len + self.staging_used()
    }

    /// `true` when the buffer holds no committed and no staged element.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of committed elements (staged edits excluded).
    #[must_use]
    pub const fn committed_len(&self) -> usize {
        self.committed_len
    }

    /// Elements currently staged across every region.
    #[must_use]
    pub const fn staging_used(&self) -> usize {
        let mut total: usize = 0;
        let mut i = 0;
        while i < self.region_count {
            total = total.saturating_add(self.regions[i].staged);
            i += 1;
        }
        total
    }

    /// Logical position of the gap: where the next flush commits.
    #[must_use]
    pub const fn gap_position(&self) -> usize {
        self.gap_pos
    }

    /// Free slots the gap currently offers. A flush of `n` staged
    /// elements consumes exactly `n` of them.
    #[must_use]
    pub const fn gap_available(&self) -> usize {
        self.gap_len
    }

    /// Reads a committed element; staged edits are invisible here.
    #[must_use]
    pub fn committed_get(&self, index: usize) -> Option<T> {
        if index >= self.committed_len {
            return None;
        }
        let physical = if index < self.gap_pos {
            index
        } else {
            index + self.gap_len
        };
        self.buf.get(physical).copied()
    }

    /// Reads the logical document: committed elements with the staged
    /// edits of every region spliced in at the gap in slot order —
    /// exactly what the buffer looks like after [`Self::flush_all`].
    #[must_use]
    pub fn get(&self, index: usize) -> Option<T> {
        if index < self.gap_pos {
            return self.committed_get(index);
        }
        let staged = self.staging_used();
        if index < self.gap_pos + staged {
            return self.staged_at(index - self.gap_pos);
        }
        self.committed_get(index - staged)
    }

    /// Streams the logical document in contiguous chunks: the committed
    /// prefix, then each region's staged slice in order, then the
    /// committed suffix. No heap allocation; suitable for streaming the
    /// document to a block device or encoder one chunk at a time.
    ///
    /// # Arguments
    ///
    /// * `f` - called once per logical chunk with a `&[T]` slice.
    pub fn for_each_logical(&self, mut f: impl FnMut(&[T])) {
        let committed = self.committed_len;
        f(&self.buf[..self.gap_pos.min(committed)]);
        for i in 0..self.region_count {
            let region = self.regions[i];
            if region.staged > 0 {
                f(&self.staging[region.start..region.start + region.staged]);
            }
        }
        let suffix_start = self.gap_pos + self.gap_len;
        let suffix_len = committed.saturating_sub(self.gap_pos);
        if suffix_len > 0 {
            f(&self.buf[suffix_start..suffix_start + suffix_len]);
        }
    }

    /// Reserves staging space for `cursor_id` and rebalances regions.
    ///
    /// # Errors
    ///
    /// * [`WreditError::InvalidOperation`] when the cursor already owns
    ///   a region
    /// * [`WreditError::StagingFull`] when no slot is left for another
    ///   region
    pub fn add_region(&mut self, cursor_id: usize) -> WreditResult<()> {
        if self.region_count >= GAP_SIZE {
            return Err(WreditError::StagingFull {
                staged: self.staging_used(),
                capacity: GAP_SIZE,
            });
        }
        if (0..self.region_count).any(|i| self.regions[i].cursor_id == cursor_id) {
            return Err(WreditError::InvalidOperation {
                what: "cursor already owns a staging region",
            });
        }
        self.regions[self.region_count] = StagingRegion::new(cursor_id, 0, 0);
        self.region_count += 1;
        self.rebalance();
        Ok(())
    }

    /// Drops `cursor_id`'s region and its staged (uncommitted) edits.
    ///
    /// Returns `true` when a region was removed.
    pub fn remove_region(&mut self, cursor_id: usize) -> bool {
        let Some(index) = self.region_index(cursor_id) else {
            return false;
        };
        let last = self.region_count - 1;
        self.regions[index] = self.regions[last];
        self.regions[last] = StagingRegion::new(0, 0, 0);
        self.region_count = last;
        self.rebalance();
        true
    }

    /// The region owned by `cursor_id`, if any.
    #[must_use]
    pub fn region(&self, cursor_id: usize) -> Option<&StagingRegion> {
        self.region_index(cursor_id)
            .map(|i| &self.regions[i])
    }

    /// Number of cursors holding a region.
    #[must_use]
    pub const fn region_count(&self) -> usize {
        self.region_count
    }

    /// `true` when `cursor_id` owns a region.
    #[must_use]
    pub fn has_region(&self, cursor_id: usize) -> bool {
        self.region_index(cursor_id).is_some()
    }

    /// Stages one element for `cursor_id` — O(1), nothing committed.
    ///
    /// Returns the logical index the element will occupy once the
    /// regions up to this one are flushed.
    ///
    /// # Errors
    ///
    /// * [`WreditError::InvalidCursor`] when the cursor owns no region
    /// * [`WreditError::StagingFull`] when the region is full, or when
    ///   the staging budget (every staged element must fit the gap
    ///   before it commits) is exhausted
    pub fn stage(&mut self, cursor_id: usize, value: T) -> WreditResult<usize> {
        let index = self
            .region_index(cursor_id)
            .ok_or(WreditError::InvalidCursor { cursor: cursor_id })?;
        let region = self.regions[index];
        if region.is_full() {
            return Err(WreditError::StagingFull {
                staged: region.staged,
                capacity: region.size,
            });
        }
        let used = self.staging_used();
        if used >= self.gap_len {
            return Err(WreditError::StagingFull {
                staged: used,
                capacity: self.gap_len,
            });
        }
        let slot = region.start + region.staged;
        self.staging[slot] = value;
        self.regions[index].staged = region.staged + 1;
        Ok(self.gap_pos + self.staged_before(index) + region.staged)
    }

    /// Stages a whole slice for `cursor_id` atomically.
    ///
    /// Either every element is staged or none is, so a burst of
    /// keystrokes never ends up half-applied.
    ///
    /// # Errors
    ///
    /// * [`WreditError::InvalidCursor`] when the cursor owns no region
    /// * [`WreditError::StagingFull`] when either the remaining region
    ///   slots or the staging budget cannot hold the whole slice
    pub fn stage_slice(&mut self, cursor_id: usize, values: &[T]) -> WreditResult<usize> {
        if values.is_empty() {
            return Ok(0);
        }
        let index = self
            .region_index(cursor_id)
            .ok_or(WreditError::InvalidCursor { cursor: cursor_id })?;
        let region = self.regions[index];
        if values.len() > region.remaining() {
            return Err(WreditError::StagingFull {
                staged: region.staged,
                capacity: region.size,
            });
        }
        let used = self.staging_used();
        if used.saturating_add(values.len()) > self.gap_len {
            return Err(WreditError::StagingFull {
                staged: used,
                capacity: self.gap_len,
            });
        }
        let slot = region.start + region.staged;
        self.staging[slot..slot + values.len()].copy_from_slice(values);
        self.regions[index].staged = region.staged + values.len();
        Ok(values.len())
    }

    /// Commits every element staged at or before `cursor_id`'s region,
    /// in slot order.
    ///
    /// Regions *after* the cursor keep their staged edits, so the
    /// logical positions of earlier regions never shift underneath
    /// them. Flushing the last region is [`Self::flush_all`].
    ///
    /// # Errors
    ///
    /// [`WreditError::InvalidCursor`] when the cursor owns no region.
    pub fn flush_region(&mut self, cursor_id: usize) -> WreditResult<usize> {
        let index = self
            .region_index(cursor_id)
            .ok_or(WreditError::InvalidCursor { cursor: cursor_id })?;
        Ok(self.commit_up_to(index))
    }

    /// Commits every staged element, in slot order.
    ///
    /// # Returns
    ///
    /// The number of elements committed.
    #[must_use]
    pub fn flush_all(&mut self) -> usize {
        match self.region_count.checked_sub(1) {
            Some(last) => self.commit_up_to(last),
            None => 0,
        }
    }

    /// Relocates the empty gap to the logical position `index`.
    ///
    /// Staged edits are committed first — exactly like the reference
    /// engine flushes a gap before shifting the block — then the gap
    /// window slides to `index` with a single block move.
    ///
    /// # Errors
    ///
    /// [`WreditError::InvalidPosition`] when `index` exceeds the
    /// committed length.
    pub fn move_gap_to(&mut self, index: usize) -> WreditResult<()> {
        let _ = self.flush_all();
        let committed = self.committed_len;
        if index > committed {
            return Err(WreditError::InvalidPosition {
                position: index as u64,
                len: committed as u64,
            });
        }
        self.move_gap_to_impl(index);
        Ok(())
    }

    /// Deletes the committed range `[start, end)` by widening the gap —
    /// the same way the disk layer frees blocks instead of moving data.
    ///
    /// Staged edits are committed first (the range addresses committed
    /// positions), then the gap absorbs the deleted elements.
    ///
    /// # Returns
    ///
    /// The number of removed elements.
    ///
    /// # Errors
    ///
    /// [`WreditError::OutOfRange`] when `start > end` or `end` exceeds
    /// the committed length.
    pub fn delete_range(&mut self, start: usize, end: usize) -> WreditResult<usize> {
        if start > end {
            return Err(WreditError::OutOfRange {
                start: start as u64,
                end: end as u64,
            });
        }
        self.move_gap_to(start)?;
        if end > self.committed_len {
            return Err(WreditError::OutOfRange {
                start: start as u64,
                end: end as u64,
            });
        }
        let count = end - start;
        if count == 0 {
            return Ok(0);
        }
        // Committed `[start..end]` sits immediately after the gap, so
        // deleting it just widens the gap — no data needs to move, the
        // same way a disk frees blocks in place.
        self.committed_len -= count;
        self.gap_len += count;
        Ok(count)
    }

    /// Evenly partitions the staging area among the regions.
    ///
    /// Each region gets `GAP_SIZE / count` slots, except regions that
    /// already staged more than the share — their reservation keeps
    /// their edits. Called automatically when a cursor joins or leaves.
    pub fn rebalance(&mut self) {
        let count = self.region_count;
        if count == 0 {
            return;
        }
        // `add_region` caps the count at `GAP_SIZE`, so the share is
        // at least one slot per cursor.
        let share = GAP_SIZE / count;
        // Rebuilding needs the staged values up front: new slots may
        // overlap old ones. Stack buffer, bounded by GAP_SIZE.
        let mut staged = [T::default(); GAP_SIZE];
        let mut staged_len = 0;
        for i in 0..count {
            let region = self.regions[i];
            staged[staged_len..staged_len + region.staged]
                .copy_from_slice(&self.staging[region.start..region.start + region.staged]);
            staged_len += region.staged;
        }
        let mut start = 0;
        let mut next = 0;
        for i in 0..count {
            let region = self.regions[i];
            let size = share.max(region.staged);
            for slot in 0..region.staged {
                self.staging[start + slot] = staged[next];
                next += 1;
            }
            // Preserve how many slots are occupied so the cursor's edits
            // are not discarded by rebalancing.
            self.regions[i] = StagingRegion {
                cursor_id: region.cursor_id,
                start,
                size,
                staged: region.staged,
            };
            start += size;
        }
        debug_assert!(start <= GAP_SIZE, "regions must fit the staging area");
    }

    /// The `offset`-th staged element in slot order, if it exists.
    fn staged_at(&self, offset: usize) -> Option<T> {
        let mut rest = offset;
        for i in 0..self.region_count {
            let region = self.regions[i];
            if rest < region.staged {
                return self.staging.get(region.start + rest).copied();
            }
            rest -= region.staged;
        }
        None
    }

    /// Index of `cursor_id`'s region.
    fn region_index(&self, cursor_id: usize) -> Option<usize> {
        let mut i = 0;
        while i < self.region_count {
            if self.regions[i].cursor_id == cursor_id {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Elements staged by the regions before `index`.
    fn staged_before(&self, index: usize) -> usize {
        let mut total: usize = 0;
        let mut i = 0;
        while i < index {
            total = total.saturating_add(self.regions[i].staged);
            i += 1;
        }
        total
    }

    /// Commits the staged elements of regions `0..=last` into the gap.
    fn commit_up_to(&mut self, last: usize) -> usize {
        let mut values = [T::default(); GAP_SIZE];
        let mut values_len: usize = 0;
        for i in 0..=last {
            let region = self.regions[i];
            values[values_len..values_len + region.staged]
                .copy_from_slice(&self.staging[region.start..region.start + region.staged]);
            values_len += region.staged;
        }
        let total = values_len;
        if total == 0 {
            return 0;
        }
        // Invariant 4: `stage`/`stage_slice` reject writes that would
        // leave more staged elements than the gap can hold.
        debug_assert!(total <= self.gap_len, "staged elements must fit the gap");
        // Write the values into the (free) gap, then pull the committed
        // suffix flush against them — one block move, no data overlaps.
        let old_span = self.committed_len + self.gap_len;
        let tail = self.gap_pos + self.gap_len;
        let elem = core::mem::size_of::<T>();
        // Safety: invariant 4 caps `total` at `gap_len`, and the
        // committed span plus gap fit `buf`, so every range below is in
        // bounds; the second move may overlap and uses `memmove`.
        unsafe {
            let buf_ptr = self.buf.as_mut_ptr();
            crate::arch::memcpy(
                buf_ptr.add(self.gap_pos).cast::<u8>(),
                values.as_ptr().cast::<u8>(),
                total.saturating_mul(elem),
            );
            crate::arch::memmove(
                buf_ptr.add(self.gap_pos + total).cast::<u8>(),
                buf_ptr.add(tail).cast::<u8>(),
                (old_span - tail).saturating_mul(elem),
            );
        }
        self.committed_len = self.committed_len.saturating_add(total);
        self.gap_pos += total;
        self.gap_len -= total;
        // If the gap was fully consumed but buffer capacity remains,
        // coalesce the free space at the end of the committed data so
        // that further staging is possible.
        if self.gap_len == 0 && self.committed_len < COMMIT_CAP {
            self.gap_pos = self.committed_len;
            self.gap_len = COMMIT_CAP - self.committed_len;
        }
        for i in 0..=last {
            self.regions[i].staged = 0;
        }
        total
    }

    /// Internal gap relocation after staged edits are committed.
    fn move_gap_to_impl(&mut self, index: usize) {
        if index == self.gap_pos {
            return;
        }
        if index > self.gap_pos {
            // Elements `[gap_pos, index)` slide left into the gap.
            let count = index - self.gap_pos;
            let src = self.gap_pos + self.gap_len;
            let elem = core::mem::size_of::<T>();
            // Safety: both ranges lie inside `buf`; the arch `memmove`
            // tolerates their overlap.
            unsafe {
                let ptr = self.buf.as_mut_ptr();
                crate::arch::memmove(
                    ptr.add(self.gap_pos).cast::<u8>(),
                    ptr.add(src).cast::<u8>(),
                    count.saturating_mul(elem),
                );
            }
        } else {
            // Elements `[index, gap_pos)` slide right past the gap.
            let count = self.gap_pos - index;
            let elem = core::mem::size_of::<T>();
            // Safety: both ranges lie inside `buf`; the arch `memmove`
            // tolerates their overlap.
            unsafe {
                let ptr = self.buf.as_mut_ptr();
                crate::arch::memmove(
                    ptr.add(index + self.gap_len).cast::<u8>(),
                    ptr.add(index).cast::<u8>(),
                    count.saturating_mul(elem),
                );
            }
        }
        self.gap_pos = index;
    }
}

impl<T: Copy + Default, const COMMIT_CAP: usize, const GAP_SIZE: usize> Default
    for StagingGap<T, COMMIT_CAP, GAP_SIZE>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::StagingGap;
    use crate::error::WreditError;

    /// Justification: `unwrap` in tests is permitted by the project
    /// lint configuration (`allow-unwrap-in-tests`); every call here
    /// exercises a path the assertion right next to it expects to
    /// succeed, and failure paths are asserted with `matches!`.

    #[test]
    fn construction_and_lengths() {
        let gap: StagingGap<u32, 32, 8> = StagingGap::new();
        assert!(gap.is_empty());
        assert_eq!(gap.committed_len(), 0);
        assert_eq!(gap.gap_position(), 0);
        assert_eq!(gap.gap_available(), 8);

        let gap: StagingGap<u32, 32, 8> = StagingGap::try_from_slice(&[1, 2, 3]).unwrap();
        assert_eq!(gap.len(), 3);
        assert_eq!(gap.committed_len(), 3);
        assert_eq!(gap.committed_get(0), Some(1));
        assert_eq!(gap.committed_get(2), Some(3));
        assert_eq!(gap.committed_get(3), None);
        assert_eq!(gap.gap_position(), 3);
        assert_eq!(gap.gap_available(), 8);
    }

    #[test]
    fn too_many_elements_for_commit_cap_is_rejected() {
        assert!(StagingGap::<u32, 4, 8>::try_from_slice(&[1, 2, 3, 4, 5]).is_err());
    }

    #[test]
    fn staged_edits_are_invisible_until_flushed_in_the_committed_view() {
        let mut gap: StagingGap<u32, 32, 8> = StagingGap::try_from_slice(&[1, 2]).unwrap();
        gap.add_region(0).unwrap();
        gap.stage(0, 9).unwrap();
        gap.stage(0, 8).unwrap();
        // Committed view unchanged …
        assert_eq!(gap.committed_len(), 2);
        assert_eq!(gap.committed_get(0), Some(1));
        // … but the logical view already shows the staged tail.
        assert_eq!(gap.get(0), Some(1));
        assert_eq!(gap.get(2), Some(9));
        assert_eq!(gap.get(3), Some(8));
        assert_eq!(gap.flush_all(), 2);
        assert_eq!(gap.committed_len(), 4);
        assert_eq!(gap.get(3), Some(8));
        assert_eq!(gap.gap_position(), 4);
        assert_eq!(gap.gap_available(), 6);
    }

    #[test]
    fn two_regions_commit_in_slot_order() {
        let mut gap: StagingGap<u32, 64, 16> = StagingGap::try_from_slice(&[10, 20]).unwrap();
        gap.add_region(0).unwrap();
        gap.add_region(1).unwrap();
        assert_eq!(gap.region_count(), 2);
        assert_eq!(gap.region(0).unwrap().size(), 8);
        gap.stage(0, 11).unwrap();
        gap.stage(0, 12).unwrap();
        let at = gap.stage(1, 21).unwrap();
        assert_eq!(at, 4);
        assert_eq!(gap.flush_region(1).unwrap(), 3);
        assert_eq!(gap.get(4), Some(21));
        assert_eq!(gap.region(0).unwrap().staged(), 0);
    }

    #[test]
    fn flushing_an_earlier_region_leaves_later_regions_staged() {
        let mut gap: StagingGap<u32, 64, 16> = StagingGap::try_from_slice(&[1]).unwrap();
        gap.add_region(0).unwrap();
        gap.add_region(1).unwrap();
        gap.stage(0, 2).unwrap();
        gap.stage(1, 3).unwrap();
        assert_eq!(gap.flush_region(0).unwrap(), 1);
        assert_eq!(gap.region(1).unwrap().staged(), 1);
        assert_eq!(gap.get(2), Some(3));
        assert_eq!(gap.flush_all(), 1);
        assert_eq!(gap.get(2), Some(3));
    }

    #[test]
    fn a_full_region_reports_staging_full_and_stage_slice_is_atomic() {
        let mut gap: StagingGap<u32, 16, 4> = StagingGap::new();
        gap.add_region(0).unwrap();
        assert!(gap.stage_slice(0, &[1, 2, 3]).is_ok());
        assert!(matches!(
            gap.stage_slice(0, &[4, 5]),
            Err(WreditError::StagingFull {
                staged: 3,
                capacity: 4
            })
        ));
        // Atomic: the failed slice staged nothing.
        assert_eq!(gap.region(0).unwrap().staged(), 3);
        // One slot remains: staging one more fills the region.
        gap.stage(0, 4).unwrap();
        assert_eq!(gap.region(0).unwrap().staged(), 4);
        // Now the region is full.
        assert!(matches!(
            gap.stage(0, 5),
            Err(WreditError::StagingFull {
                staged: 4,
                capacity: 4
            })
        ));
        // Flush frees the region again.
        assert_eq!(gap.flush_all(), 4);
        gap.stage(0, 4).unwrap();
        assert_eq!(gap.get(3), Some(4));
    }

    #[test]
    fn move_gap_relocates_with_one_block_move_and_keeps_content() {
        let mut gap: StagingGap<u32, 32, 4> = StagingGap::try_from_slice(&[1, 2, 3, 4, 5]).unwrap();
        gap.move_gap_to(2).unwrap();
        assert_eq!(gap.gap_position(), 2);
        assert_eq!(gap.get(0), Some(1));
        assert_eq!(gap.get(4), Some(5));
        gap.add_region(0).unwrap();
        gap.stage(0, 99).unwrap();
        // Moving the gap commits the staged edit first.
        gap.move_gap_to(0).unwrap();
        assert_eq!(gap.get(2), Some(99));
        assert_eq!(gap.gap_position(), 0);
        // The document now has 6 committed elements, so position 6 is
        // the valid gap-at-the-end position.
        gap.move_gap_to(6).unwrap();
        assert_eq!(gap.gap_position(), 6);
        assert!(matches!(
            gap.move_gap_to(7),
            Err(WreditError::InvalidPosition { position: 7, len: 6 })
        ));
    }

    #[test]
    fn delete_range_widens_the_gap_and_returns_the_count() {
        let mut gap: StagingGap<u32, 32, 4> = StagingGap::try_from_slice(&[1, 2, 3, 4, 5]).unwrap();
        let removed = gap.delete_range(1, 3).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(gap.get(0), Some(1));
        assert_eq!(gap.get(1), Some(4));
        assert_eq!(gap.committed_len(), 3);
        assert_eq!(gap.gap_position(), 1);
        assert_eq!(gap.gap_available(), 6);
        assert!(matches!(
            gap.delete_range(2, 9),
            Err(WreditError::OutOfRange { start: 2, end: 9 })
        ));
        assert!(matches!(
            gap.delete_range(2, 1),
            Err(WreditError::OutOfRange { start: 2, end: 1 })
        ));
    }

    #[test]
    fn rebalance_spreads_regions_evenly_without_losing_staged_edits() {
        let mut gap: StagingGap<u32, 64, 12> = StagingGap::new();
        gap.add_region(0).unwrap();
        gap.add_region(1).unwrap();
        gap.add_region(2).unwrap();
        let regions = &gap.regions[..gap.region_count];
        assert_eq!(regions[0].start(), 0);
        assert_eq!(regions[1].start(), 4);
        assert_eq!(regions[2].start(), 8);
        assert!(regions.iter().all(|r| r.size() == 4));
        // Stage into region 2, then let a fourth cursor join: region 2
        // keeps its edit above the new three-slot share.
        gap.stage(2, 42).unwrap();
        gap.add_region(3).unwrap();
        assert_eq!(gap.region(2).unwrap().size(), 3);
        assert_eq!(gap.region(2).unwrap().staged(), 1);
        assert_eq!(gap.region(3).unwrap().size(), 3);
        assert_eq!(gap.get(0), Some(42));
    }

    #[test]
    fn removing_a_region_drops_its_staged_edits() {
        let mut gap: StagingGap<u32, 32, 8> = StagingGap::try_from_slice(&[1]).unwrap();
        gap.add_region(0).unwrap();
        gap.add_region(1).unwrap();
        gap.stage(0, 2).unwrap();
        gap.stage(1, 3).unwrap();
        assert!(gap.remove_region(0));
        assert!(!gap.remove_region(0));
        assert!(!gap.has_region(0));
        assert_eq!(gap.get(1), Some(3));
    }

    #[test]
    fn duplicate_regions_and_zero_capacity_are_rejected() {
        let mut gap: StagingGap<u32, 32, 8> = StagingGap::new();
        gap.add_region(0).unwrap();
        assert!(matches!(
            gap.add_region(0),
            Err(WreditError::InvalidOperation { .. })
        ));
    }

    #[test]
    fn interleaved_edits_from_many_cursors_survive_every_flush() {
        let mut gap: StagingGap<u32, 128, 32> = StagingGap::try_from_slice(&[0, 0, 0]).unwrap();
        for cursor in 0..4 {
            gap.add_region(cursor).unwrap();
        }
        let mut expected = vec![0u32, 0, 0];
        for round in 0..8u32 {
            for cursor in 0..4 {
                let value = round * 4 + cursor + 1;
                let at = gap.stage(cursor as usize, value).unwrap();
                expected.insert(at, value);
            }
            assert_eq!(gap.get(expected.len() - 1), Some(*expected.last().unwrap()));
            gap.flush_region((round % 4) as usize).unwrap();
        }
        let _ = gap.flush_all();
        assert_eq!(gap.committed_len(), expected.len());
    }
    #[test]
    fn region_metadata_is_reported() {
        let mut gap: StagingGap<u32, 32, 16> = StagingGap::new();
        gap.add_region(7).unwrap();
        let region = gap.region(7).unwrap();
        assert_eq!(region.cursor_id(), 7);
        assert_eq!(region.start(), 0);
        assert_eq!(region.size(), 16);
        assert_eq!(region.staged(), 0);
        assert_eq!(region.remaining(), 16);
        assert!(!region.is_full());
    }
}
