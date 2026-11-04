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

//! Shared gap slots: fixed 8-byte descriptors suballocated from a
//! circular table.
//!
//! A *shared gap* is a staging region inside the writable's shared
//! heap arena that several cursors can write to concurrently without
//! contending with the private per-cursor gaps. Its metadata lives in
//! a [`SharedGapSlot`] — an exactly 8-byte packed descriptor — and all
//! descriptors together live in a fixed [`SharedGapTable`]. Slots are
//! recycled through a 7-bit generation counter so a stale index held
//! by one subsystem can never be silently confused with a fresh
//! allocation at the same position (ABA safety).
//!
//! Bit layout of a slot (`u64`, `repr(transparent)`):
//!
//! | bits    | field                                             |
//! |---------|---------------------------------------------------|
//! | `0..16` | cursor membership mask (one bit per cursor, 16)   |
//! | `16..32`| region base offset in the shared arena (elements) |
//! | `32..48`| region size in elements currently staged          |
//! | `48`    | allocated flag                                    |
//! | `49..56`| generation (bumped on every free)                 |
//! | `56..64`| reserved, always zero                             |

/// Number of cursors that can share one gap (bits of the mask).
pub const SLOT_CURSOR_BITS: usize = 16;

const SHIFT_MASK: u32 = 0;
const SHIFT_BASE: u32 = 16;
const SHIFT_SIZE: u32 = 32;
const SHIFT_ALLOCATED: u32 = 48;
const SHIFT_GENERATION: u32 = 49;

const MASK_MASK: u64 = 0xFFFF;
const MASK_BASE: u64 = 0xFFFF << SHIFT_BASE;
const MASK_SIZE: u64 = 0xFFFF << SHIFT_SIZE;
const MASK_ALLOCATED: u64 = 1 << SHIFT_ALLOCATED;
const MASK_GENERATION: u64 = 0x7F << SHIFT_GENERATION;

/// Largest base offset addressable in a slot.
pub const SLOT_MAX_BASE: usize = u16::MAX as usize;
/// Largest region size addressable in a slot.
pub const SLOT_MAX_SIZE: usize = u16::MAX as usize;

/// An exactly 8-byte packed shared-gap slot descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct SharedGapSlot(u64);

impl SharedGapSlot {
    /// A free slot carrying `generation`.
    #[must_use]
    pub const fn free(generation: u8) -> Self {
        Self((generation as u64) << SHIFT_GENERATION)
    }

    /// An allocated slot for `cursor_mask` at region `base` with
    /// `size` elements staged, carrying `generation`.
    #[must_use]
    pub const fn allocated(cursor_mask: u16, base: u16, size: u16, generation: u8) -> Self {
        let generation = ((generation as u64) << SHIFT_GENERATION) & MASK_GENERATION;
        Self(
            ((cursor_mask as u64) << SHIFT_MASK)
                | ((base as u64) << SHIFT_BASE)
                | ((size as u64) << SHIFT_SIZE)
                | MASK_ALLOCATED
                | generation,
        )
    }

    /// `true` when the slot is currently allocated.
    #[must_use]
    pub const fn is_allocated(self) -> bool {
        self.0 & MASK_ALLOCATED != 0
    }

    /// Cursor membership mask of the slot.
    #[must_use]
    pub const fn cursor_mask(self) -> u16 {
        ((self.0 & MASK_MASK) >> SHIFT_MASK) as u16
    }

    /// Region base offset in the shared arena, in elements.
    #[must_use]
    pub const fn base(self) -> usize {
        ((self.0 & MASK_BASE) >> SHIFT_BASE) as usize
    }

    /// Region size in the shared arena, in elements.
    #[must_use]
    pub const fn size(self) -> usize {
        ((self.0 & MASK_SIZE) >> SHIFT_SIZE) as usize
    }

    /// The slot's generation counter.
    #[must_use]
    pub const fn generation(self) -> u8 {
        ((self.0 & MASK_GENERATION) >> SHIFT_GENERATION) as u8
    }

    /// `true` when `cursor_id` is a member of the slot.
    #[must_use]
    pub const fn contains_cursor(self, cursor_id: usize) -> bool {
        if cursor_id >= SLOT_CURSOR_BITS {
            return false;
        }
        self.cursor_mask() & (1u16 << cursor_id) != 0
    }

    /// The slot with `cursor_id` added to the membership mask.
    ///
    /// Returns `None` when the cursor is already a member or does not
    /// fit the mask.
    #[must_use]
    pub const fn with_cursor(self, cursor_id: usize) -> Option<Self> {
        if cursor_id >= SLOT_CURSOR_BITS {
            return None;
        }
        let bit = 1u16 << cursor_id;
        if self.cursor_mask() & bit != 0 {
            return None;
        }
        Some(Self((self.0 & !MASK_MASK) | ((self.cursor_mask() | bit) as u64)))
    }

    /// The slot with `cursor_id` removed from the membership mask.
    #[must_use]
    pub const fn without_cursor(self, cursor_id: usize) -> Self {
        if cursor_id >= SLOT_CURSOR_BITS {
            return self;
        }
        Self((self.0 & !MASK_MASK) | ((self.cursor_mask() & !(1u16 << cursor_id)) as u64))
    }

    /// The slot with its region fields replaced.
    #[must_use]
    pub const fn with_region(self, base: usize, size: usize) -> Self {
        let base_clamped = if base > SLOT_MAX_BASE { SLOT_MAX_BASE } else { base };
        let size_clamped = if size > SLOT_MAX_SIZE { SLOT_MAX_SIZE } else { size };
        let base = (base_clamped as u64) << SHIFT_BASE;
        let size = (size_clamped as u64) << SHIFT_SIZE;
        Self((self.0 & !(MASK_BASE | MASK_SIZE)) | base | size)
    }

    /// The raw descriptor word.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// A fixed circular table of [`SHARED SLOTS`](SharedGapTable).
///
/// Allocation scans the ring starting at the head and returns the
/// first free slot; freeing marks the slot free and bumps its
/// generation, so a reallocated slot at the same index is
/// distinguishable from the stale one.
pub struct SharedGapTable<const SLOTS: usize> {
    /// The fixed ring of packed descriptors.
    slots: [SharedGapSlot; SLOTS],
    /// Index the next allocation scan starts from.
    head: usize,
    /// Number of allocated slots.
    count: usize,
}

impl<const SLOTS: usize> SharedGapTable<SLOTS> {
    /// Creates a table with every slot free.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: [SharedGapSlot::free(0); SLOTS],
            head: 0,
            count: 0,
        }
    }

    /// Number of slots in the ring.
    #[must_use]
    pub const fn capacity() -> usize {
        SLOTS
    }

    /// Number of allocated slots.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// `true` when every slot is allocated.
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.count >= SLOTS
    }

    /// `true` when no slot is allocated.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The slot at `index`, if the index is in the ring.
    #[must_use]
    pub const fn slot(&self, index: usize) -> Option<SharedGapSlot> {
        if index < SLOTS {
            Some(self.slots[index])
        } else {
            None
        }
    }

    /// Allocates a slot for `cursor_mask`, scanning the ring from the
    /// head and reusing the first free slot.
    ///
    /// Returns the slot index, or `None` when the ring is full.
    pub fn alloc_slot(&mut self, cursor_mask: u16) -> Option<usize> {
        if SLOTS == 0 || self.count >= SLOTS {
            return None;
        }
        let mut offset = 0;
        while offset < SLOTS {
            let index = (self.head + offset) % SLOTS;
            let slot = self.slots[index];
            if !slot.is_allocated() {
                self.slots[index] = SharedGapSlot::allocated(cursor_mask, 0, 0, slot.generation());
                self.head = (index + 1) % SLOTS;
                self.count += 1;
                return Some(index);
            }
            offset += 1;
        }
        None
    }

    /// Frees the slot at `index`, bumping its generation.
    ///
    /// Returns `true` when a live slot was freed; freeing an
    /// out-of-range or already free slot returns `false`.
    pub fn free_slot(&mut self, index: usize) -> bool {
        if index >= SLOTS {
            return false;
        }
        let slot = self.slots[index];
        if !slot.is_allocated() {
            return false;
        }
        let generation = slot.generation().wrapping_add(1) & 0x7F;
        self.slots[index] = SharedGapSlot::free(generation);
        self.count = self.count.saturating_sub(1);
        true
    }

    /// Replaces the slot at `index`; returns `false` for a bad index.
    pub fn set_slot(&mut self, index: usize, slot: SharedGapSlot) -> bool {
        if index >= SLOTS {
            return false;
        }
        self.slots[index] = slot;
        true
    }

    /// Updates the region fields of the slot at `index`.
    ///
    /// Returns `false` for a bad index or a free slot.
    pub fn set_region(&mut self, index: usize, base: usize, size: usize) -> bool {
        if index >= SLOTS {
            return false;
        }
        let slot = self.slots[index];
        if !slot.is_allocated() {
            return false;
        }
        self.slots[index] = slot.with_region(base, size);
        true
    }

    /// `true` when the allocated slot at `index` includes `cursor_id`.
    #[must_use]
    pub const fn is_cursor_in(&self, index: usize, cursor_id: usize) -> bool {
        if index >= SLOTS {
            return false;
        }
        self.slots[index].contains_cursor(cursor_id)
    }

    /// Adds `cursor_id` to the membership mask of the slot at `index`.
    ///
    /// Returns `false` for a bad index, a free slot, an existing
    /// member, or a cursor id outside the mask.
    pub fn join_cursor(&mut self, index: usize, cursor_id: usize) -> bool {
        if index >= SLOTS {
            return false;
        }
        let Some(slot) = self.slots[index].with_cursor(cursor_id) else {
            return false;
        };
        self.slots[index] = slot;
        true
    }

    /// Removes `cursor_id` from the membership mask of the slot at
    /// `index`. Returns `false` when the cursor was not a member.
    pub fn leave_cursor(&mut self, index: usize, cursor_id: usize) -> bool {
        if !self.is_cursor_in(index, cursor_id) {
            return false;
        }
        self.slots[index] = self.slots[index].without_cursor(cursor_id);
        true
    }

    /// The first allocated slot index that includes `cursor_id`.
    #[must_use]
    pub fn slot_for_cursor(&self, cursor_id: usize) -> Option<usize> {
        if cursor_id >= SLOT_CURSOR_BITS {
            return None;
        }
        let mut index = 0;
        while index < SLOTS {
            if self.slots[index].contains_cursor(cursor_id) {
                return Some(index);
            }
            index += 1;
        }
        None
    }

    /// The ring head (next allocation position).
    #[must_use]
    pub const fn head(&self) -> usize {
        self.head
    }
}

impl<const SLOTS: usize> Default for SharedGapTable<SLOTS> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{SharedGapSlot, SharedGapTable};

    /// Cursor ids at or above the mask width never fit a slot.
    const SLOT_CURSOR_LIMIT: usize = super::SLOT_CURSOR_BITS;

    #[test]
    fn slot_is_exactly_eight_bytes() {
        assert_eq!(core::mem::size_of::<SharedGapSlot>(), 8);
    }

    #[test]
    fn slot_fields_roundtrip() {
        let slot = SharedGapSlot::allocated(0b1011, 42, 7, 3);
        assert!(slot.is_allocated());
        assert_eq!(slot.cursor_mask(), 0b1011);
        assert_eq!(slot.base(), 42);
        assert_eq!(slot.size(), 7);
        assert_eq!(slot.generation(), 3);
        assert!(slot.contains_cursor(0));
        assert!(slot.contains_cursor(1));
        assert!(!slot.contains_cursor(2));
        assert!(slot.contains_cursor(3));
        assert!(!slot.contains_cursor(SLOT_CURSOR_LIMIT));
    }

    #[test]
    fn join_and_leave_cursors() {
        let slot = SharedGapSlot::allocated(0, 0, 0, 0);
        let joined = slot.with_cursor(5).unwrap();
        assert!(joined.contains_cursor(5));
        assert!(joined.with_cursor(5).is_none()); // already a member
        assert!(joined.with_cursor(SLOT_CURSOR_LIMIT).is_none());
        let left = joined.without_cursor(5);
        assert!(!left.contains_cursor(5));
        // Leaving a non-member is a no-op.
        assert_eq!(left.without_cursor(5), left);
    }

    #[test]
    fn region_is_clamped_to_addressable_bits() {
        let slot = SharedGapSlot::allocated(1, 0, 0, 0).with_region(usize::MAX, usize::MAX);
        assert_eq!(slot.base(), super::SLOT_MAX_BASE);
        assert_eq!(slot.size(), super::SLOT_MAX_SIZE);
    }

    #[test]
    fn table_allocates_and_frees_slots() {
        let mut table: SharedGapTable<4> = SharedGapTable::new();
        assert!(table.is_empty());
        let a = table.alloc_slot(0b1).unwrap();
        let b = table.alloc_slot(0b10).unwrap();
        assert_ne!(a, b);
        assert_eq!(table.count(), 2);
        assert!(table.is_cursor_in(a, 0));
        assert_eq!(table.slot_for_cursor(1), Some(b));
        assert_eq!(table.slot_for_cursor(2), None);
        assert!(table.free_slot(a));
        assert!(!table.free_slot(a)); // double free is a no-op
        assert!(!table.free_slot(99)); // out of range
        assert_eq!(table.count(), 1);
    }

    #[test]
    fn freed_slots_are_recycled_with_a_new_generation() {
        let mut table: SharedGapTable<1> = SharedGapTable::new();
        let index = table.alloc_slot(0).unwrap();
        let first_generation = table
            .slot(index)
            .unwrap_or_else(|| SharedGapSlot::free(0))
            .generation();
        assert!(table.free_slot(index));
        let freed = table.slot(index).unwrap();
        assert!(!freed.is_allocated());
        assert_eq!(freed.generation(), first_generation.wrapping_add(1) & 0x7F);
        // The single slot is recycled: same index, bumped generation.
        assert_eq!(table.alloc_slot(0b1), Some(index));
        assert_eq!(table.slot(index).unwrap().generation(), freed.generation());
        assert!(table.is_full());
        assert!(table.alloc_slot(0).is_none());
    }

    #[test]
    fn zero_slot_tables_never_allocate() {
        let mut table: SharedGapTable<0> = SharedGapTable::new();
        assert!(table.alloc_slot(1).is_none());
        assert!(!table.free_slot(0));
        assert!(!table.set_region(0, 0, 0));
    }

    #[test]
    fn join_and_leave_manage_membership() {
        let mut table: SharedGapTable<2> = SharedGapTable::new();
        let index = table.alloc_slot(0).unwrap();
        assert!(table.join_cursor(index, 2));
        assert!(!table.join_cursor(index, 2)); // duplicate
        assert!(!table.join_cursor(7, 0)); // bad index
        assert!(table.is_cursor_in(index, 2));
        assert!(table.leave_cursor(index, 2));
        assert!(!table.leave_cursor(index, 2));
        assert!(table.set_region(index, 3, 4));
        let slot = table.slot(index).unwrap();
        assert_eq!((slot.base(), slot.size()), (3, 4));
        assert!(table.set_region(index, 0, 0));
    }

    #[test]
    fn generation_wraps_instead_of_overflowing() {
        let slot = SharedGapSlot::free(0x7F);
        let bumped = slot.generation().wrapping_add(1) & 0x7F;
        assert_eq!(bumped, 0);
    }
}
