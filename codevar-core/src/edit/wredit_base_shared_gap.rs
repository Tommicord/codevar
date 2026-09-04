//! Copyright 2026 Codevar
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

//! # Shared Gap Buffer Allocation Module
//!
//! This module implements shared gap buffer allocation for multi-cursor text editing,
//! enabling efficient memory usage when multiple cursors are editing the same document.
//!
//! ## Architecture
//!
//! The shared gap system provides virtual partitioning of a single gap buffer among multiple cursors:
//!
//! - **Region Suballocation**: Each cursor gets a virtual region within the shared gap
//! - **Dynamic Region Management**: Regions can be moved and resized as needed
//! - **Proximity Detection**: Cursors in proximity share gap space for efficiency
//! - **Memory Efficiency**: Reduces memory overhead compared to per-cursor gap buffers
//!
//! ## Safety Considerations
//!
//! This module uses unsafe code extensively for:
//!
//! - **Raw Pointer Operations**: Direct gap buffer manipulation for performance
//! - **UnsafeCell Usage**: Interior mutability for shared state management
//! - **Pointer Arithmetic**: Offset calculations for region positioning
//! - **Memory Management**: Manual allocation and deallocation of gap buffers
//!
//! All unsafe operations maintain strict invariants:
//! - Gap buffers are always properly aligned and allocated
//! - Region offsets remain within valid gap boundaries
//! - Cursor IDs are always valid before dereferencing
//! - No concurrent access to the same gap buffer regions
//!
//! ## Key Invariants
//!
//! 1. **Region Containment**: All cursor regions must be within the shared gap boundaries
//! 2. **Non-overlapping Regions**: Cursor regions cannot overlap within the same gap
//! 3. **Cursor ID Validity**: All cursor IDs must correspond to existing cursors
//! 4. **Gap Allocation**: Shared gaps must be properly allocated before use
//! 5. **Offset Consistency**: Region offsets must be consistent with gap position

use super::WritableGap;
use std::cell::UnsafeCell;

/// `pub struct SharedGapCursorRegion`
///
/// Metadata structure for tracking a cursor's suballocated region within a shared gap.
///
/// This struct represents a virtual partition within a shared gap buffer that is
/// reserved for a specific cursor. Instead of allocating separate memory for each
/// cursor, the shared gap buffer is partitioned into regions, with each cursor
/// having its own reserved space tracked by this metadata. The actual memory
/// allocation is shared among all cursors in the gap, but each cursor operates
/// within its designated region.
///
/// The suballocation approach reduces memory overhead by avoiding separate gap
/// buffer allocations for each cursor when cursors are in proximity. When
/// a cursor's region fills up, it can be moved within the shared gap or the gap
/// itself can be moved, maintaining the virtual partitioning without additional
/// heap allocations.
///
/// # Fields
///
/// * `cursor_id` - The ID of the cursor that owns this region. This identifies
///   which cursor this suballocation belongs to and is used for region lookup
///   and management operations.
/// * `region_start` - The start offset of this cursor's region within the shared
///   gap buffer. This is a virtual offset into the gap buffer where the cursor's
///   reserved space begins. The offset is relative to the gap's gap_ptr position.
/// * `region_size` - The total size of this cursor's reserved region. This
///   determines how much space the cursor can use before needing to move or
///   flush. The size is determined at region creation based on cursor proximity.
/// * `remaining_space` - The amount of unused space remaining in this cursor's
///   region. This tracks how many more characters can be written before the region
///   is full and needs to be moved. When this reaches zero, the region must be
///   relocated.
/// * `cursor_position` - The current write position within this cursor's region.
///   This tracks where the next character will be written within the cursor's
///   reserved space and is used for calculating relative positions.
#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct SharedGapCursorRegion {
    /// The ID of the cursor that owns this region. This uniquely identifies
    /// which cursor this suballocation belongs to and is used for region
    /// lookup during cursor operations and gap management.
    cursor_id: usize,
    /// The start offset of this cursor's region within the shared gap buffer.
    /// This is a virtual offset (not a physical pointer) that marks where
    /// the cursor's reserved space begins relative to the gap's current position.
    region_start: usize,
    /// The total size of this cursor's reserved region in the shared gap buffer.
    /// This determines the capacity available to the cursor before the region
    /// must be moved or flushed. Larger regions reduce the frequency of moves.
    region_size: usize,
    /// The amount of unused space remaining in this cursor's region. This
    /// counter decrements as characters are written and is checked before
    /// each write operation. When it reaches zero, the region must be moved.
    remaining_space: usize,
    /// The current write position within this cursor's region. This offset
    /// tracks where the next character will be written relative to the
    /// region's start and is used for calculating absolute positions in the gap.
    cursor_pos: usize,
}

impl SharedGapCursorRegion {
    /// Creates a new cursor region with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor that will own this region
    /// * `region_start` - The start offset of the region within the gap buffer
    /// * `region_size` - The total size of the region
    ///
    /// # Returns
    ///
    /// A new SharedGapCursorRegion with remaining_space initialized to region_size
    /// and cursor_position initialized to region_start.

    pub fn new(cursor_id: usize, region_start: usize, region_size: usize) -> Self {
        Self {
            cursor_id,
            region_start,
            region_size,
            remaining_space: region_size,
            cursor_pos: region_start,
        }
    }

    /// Returns the cursor ID that owns this region.

    pub fn cursor_id(&self) -> usize {
        self.cursor_id
    }

    /// Returns the start offset of this region within the gap buffer.

    pub fn region_start(&self) -> usize {
        self.region_start
    }

    /// Returns the total size of this region.

    pub fn region_size(&self) -> usize {
        self.region_size
    }

    /// Sets the total size of this region.

    pub fn set_region_size(&mut self, size: usize) {
        self.region_size = size;
        self.remaining_space = size;
    }

    /// Returns the remaining unused space in this region.

    pub fn remaining_space(&self) -> usize {
        self.remaining_space
    }

    /// Returns the current cursor position within this region.

    pub fn cursor_position(&self) -> usize {
        self.cursor_pos
    }

    /// Sets the cursor position within this region.
    ///
    /// # Arguments
    ///
    /// * `position` - The new cursor position

    pub fn set_cursor_position(&mut self, position: usize) {
        self.cursor_pos = position;
    }

    /// Decrements the remaining space by the specified amount.
    ///
    /// # Arguments
    ///
    /// * `amount` - The amount to decrement

    pub fn decrement_remaining(&mut self, amount: usize) {
        self.remaining_space = self.remaining_space.saturating_sub(amount);
    }

    /// Moves this region to a new start position within the gap buffer.
    ///
    /// This updates both region_start and cursor_position to the new offset,
    /// effectively relocating the entire region without changing its size.
    ///
    /// # Arguments
    ///
    /// * `new_start` - The new start offset for the region

    pub fn move_to(&mut self, new_start: usize) {
        let old_start = self.region_start;
        self.region_start = new_start;

        // Calculate delta using signed arithmetic to handle backward movement
        let delta = new_start as isize - old_start as isize;

        if delta >= 0 {
            self.cursor_pos = self.cursor_pos.saturating_add(delta as usize);
        } else {
            self.cursor_pos = self.cursor_pos.saturating_sub((-delta) as usize);
        }

        let region_end = self.region_start.saturating_add(self.region_size);
        // Ensure cursor_pos stays within region bounds
        if self.cursor_pos > region_end {
            self.cursor_pos = region_end;
        }
    }

    /// Checks if this region has remaining space for writing.
    ///
    /// # Returns
    ///
    /// `true` if remaining_space > 0, `false` otherwise

    pub fn has_space(&self) -> bool {
        self.remaining_space > 0
    }
}

/// `pub struct SharedGap<T, const GAP_SIZE: usize>`
///
/// Shared gap structure for multi-cursor gap buffer management with suballocation.
///
/// This struct extends the concept of a gap buffer to support multiple cursors
/// sharing the same physical gap buffer through virtual suballocation. Instead of
/// allocating separate gap buffers for each cursor, cursors that are in proximity
/// (within GAP_SIZE range) share a single gap buffer, with each cursor
/// having a reserved region tracked by metadata.
///
/// The shared gap uses composition to include a WritableGap, providing all the
/// underlying gap buffer functionality while adding cursor region management.
/// When two or more cursors are within the gap's range (gap_ptr_start to
/// gap_ptr_start + GAP_SIZE), the gap becomes shared and regions are allocated
/// for each cursor.
///
/// The suballocation system allows efficient memory usage by avoiding duplicate
/// allocations for nearby cursors. When a cursor's region fills up, the region
/// can be moved within the shared gap, or the entire gap can be moved. If moving
/// the gap would collide with another gap, a new shared gap is created.
///
/// # Type Parameters
///
/// * `T` - The type of elements in the gap buffer (typically u32 for packed data)
/// * `GAP_SIZE` - The size of the gap buffer (must be a power of 2)
///
/// # Fields
///
/// * `gap` - The underlying WritableGap that provides the physical gap buffer.
///   This composed struct handles the actual memory allocation, gap bounds,
///   and circular buffer semantics. The shared gap delegates buffer operations
///   to this underlying gap.
/// * `cursor_regions` - Vector of cursor regions currently allocated in this
///   shared gap. Each region tracks a cursor's reserved space within the gap
///   buffer. The vector is wrapped in UnsafeCell for interior mutability,
///   allowing regions to be modified through immutable references.
/// * `is_shared` - Flag indicating whether this gap is currently shared by
///   multiple cursors. A gap becomes shared when 2+ cursors are within its
///   range. This flag is used to determine whether suballocation logic should
///   be applied during write operations.
#[repr(C)]
pub struct SharedGap<T, const GAP_SIZE: usize> {
    /// The underlying WritableGap that provides the physical gap buffer.
    /// This composed struct handles memory allocation, gap bounds management,
    /// and circular buffer semantics. The shared gap extends this functionality
    /// with cursor region management and suballocation logic.
    gap: WritableGap<T, GAP_SIZE>,
    /// Vector of cursor regions currently allocated in this shared gap.
    /// Each region represents a virtual partition of the gap buffer reserved
    /// for a specific cursor. The vector tracks all active cursors in the gap
    /// and their respective reserved spaces. Wrapped in UnsafeCell for interior
    /// mutability to allow modifications through immutable references.
    cursor_regions: UnsafeCell<Vec<SharedGapCursorRegion>>,
    /// Flag indicating whether this gap is currently shared by multiple cursors.
    /// A gap becomes shared when two or more cursors are within its range
    /// (gap_ptr_start to gap_ptr_start + GAP_SIZE). This flag determines whether
    /// suballocation logic should be applied during write operations and whether
    /// cursor region management is active.
    is_shared: UnsafeCell<bool>,
}

impl<T, const GAP_SIZE: usize> SharedGap<T, GAP_SIZE> {
    /// Creates a new shared gap with the specified gap size.
    ///
    /// # Arguments
    ///
    /// * `gap_size` - The size of the gap buffer
    ///
    /// # Returns
    ///
    /// A new SharedGap with an empty cursor regions vector and is_shared set to false.

    pub fn new() -> Self {
        Self {
            gap: WritableGap::new(),
            cursor_regions: UnsafeCell::new(Vec::new()),
            is_shared: UnsafeCell::new(false),
        }
    }

    /// Returns a reference to the underlying gap.

    pub fn gap(&self) -> &WritableGap<T, GAP_SIZE> {
        &self.gap
    }

    /// Returns a mutable reference to the underlying gap.

    pub fn gap_mut(&mut self) -> &mut WritableGap<T, GAP_SIZE> {
        &mut self.gap
    }

    /// Checks if this gap is currently shared by multiple cursors.

    pub fn is_shared(&self) -> bool {
        unsafe { *self.is_shared.get() }
    }

    /// Sets the shared flag for this gap.
    ///
    /// # Arguments
    ///
    /// * `shared` - The new shared flag value

    pub fn set_shared(&self, shared: bool) {
        unsafe {
            *self.is_shared.get() = shared;
        }
    }

    /// Adds a cursor region to this shared gap.
    ///
    /// This method allocates a new region for the specified cursor within the
    /// shared gap buffer. The region size is calculated based on the number of
    /// cursors in the gap to ensure fair space distribution.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to add
    ///
    /// # Safety
    ///
    /// The cursor_id must be unique within this gap

    pub unsafe fn add_cursor_region(&self, cursor_id: usize) {
        let regions = &mut *self.cursor_regions.get();

        // Add a temporary region with size 0, will be set by rebalance
        regions.push(SharedGapCursorRegion::new(cursor_id, 0, 0));

        // Rebalance to redistribute space among all cursors
        self.rebalance(regions);

        if regions.len() >= 2 {
            *self.is_shared.get() = true;
        }
    }

    /// Removes a cursor region from this shared gap.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to remove
    ///
    /// # Returns
    ///
    /// `true` if the cursor was found and removed, `false` otherwise

    pub unsafe fn remove_cursor_region(&self, cursor_id: usize) -> bool {
        let regions = &mut *self.cursor_regions.get();
        if let Some(pos) = regions.iter().position(|r| r.cursor_id() == cursor_id) {
            regions.remove(pos);

            if regions.len() < 2 {
                *self.is_shared.get() = false;
            }

            true
        } else {
            false
        }
    }

    /// Returns the cursor region for the specified cursor ID.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to look up
    ///
    /// # Returns
    ///
    /// `Some(&SharedGapCursorRegion)` if found, `None` otherwise

    pub unsafe fn cursor_region(
        &self,
        cursor_id: usize,
    ) -> Option<&SharedGapCursorRegion> {
        let regions = &*self.cursor_regions.get();
        regions.iter().find(|r| r.cursor_id() == cursor_id)
    }

    /// Returns the cursor region for the specified cursor ID (mutable).
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to look up
    ///
    /// # Returns
    ///
    /// `Some(&mut SharedGapCursorRegion)` if found, `None` otherwise

    pub unsafe fn cursor_region_mut(
        &self,
        cursor_id: usize,
    ) -> Option<&mut SharedGapCursorRegion> {
        let regions = &mut *self.cursor_regions.get();
        regions.iter_mut().find(|r| r.cursor_id() == cursor_id)
    }

    /// Returns the number of cursor regions in this shared gap.

    pub unsafe fn cursor_region_count(&self) -> usize {
        let regions = &*self.cursor_regions.get();
        regions.len()
    }

    /// Checks if a cursor with the specified ID is in this shared gap.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor to check
    ///
    /// # Returns
    ///
    /// `true` if the cursor is in this gap, `false` otherwise

    pub unsafe fn has_cursor(&self, cursor_id: usize) -> bool {
        self.cursor_region(cursor_id).is_some()
    }

    /// Moves a cursor's region to the right within the shared gap.
    ///
    /// When a cursor's region fills up (remaining_space == 0), this method
    /// moves the region to the right by its size. If there's not enough space
    /// in the current gap, the entire gap needs to be moved (handled by caller).
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The ID of the cursor whose region should be moved
    ///
    /// # Returns
    ///
    /// `true` if the region was moved successfully, `false` if there's not enough space

    pub unsafe fn move_cursor_region_right(&self, cursor_id: usize) -> bool {
        let regions = &mut *self.cursor_regions.get();
        let (current_start, region_size) = {
            if let Some(region) = regions.iter().find(|r| r.cursor_id() == cursor_id) {
                (region.region_start(), region.region_size())
            } else {
                return false;
            }
        };
        let next_region_start = regions
            .iter()
            .filter(|r| r.region_start() > current_start)
            .map(|r| r.region_start())
            .min();

        let new_start = current_start.saturating_add(region_size);
        let new_end = new_start.saturating_add(region_size);

        if let Some(next_start) = next_region_start {
            if new_start >= next_start {
                return false; // Not enough space, gap needs to move
            }
        } else if new_end > GAP_SIZE {
            return false; // Would exceed gap size
        }

        // Now move the region using mutable borrow
        if let Some(region) = regions.iter_mut().find(|r| r.cursor_id() == cursor_id) {
            region.move_to(new_start);
            true
        } else {
            false
        }
    }

    pub unsafe fn cursor_regions(&self) -> &Vec<SharedGapCursorRegion> {
        &*self.cursor_regions.get()
    }

    pub unsafe fn cursor_regions_mut(&self) -> &mut Vec<SharedGapCursorRegion> {
        &mut *self.cursor_regions.get()
    }

    /// Rebalances cursor regions when the gap size changes or cursors are added/removed.
    ///
    /// This redistributes the gap space among all cursor regions to ensure fair
    /// allocation. It recalculates region sizes based on the current cursor count
    /// and updates all region start positions accordingly.
    ///
    /// # Arguments
    ///
    /// * `regions`: The shared gap cursor regions

    pub fn rebalance(&self, regions: &mut Vec<SharedGapCursorRegion>) {
        let cursor_count = regions.len();
        if cursor_count == 0 {
            return;
        }
        let region_size = GAP_SIZE / cursor_count;
        let mut current_start = 0;

        for region in regions.iter_mut() {
            region.set_region_size(region_size);
            region.move_to(current_start);
            current_start = current_start.saturating_add(region_size);
        }
    }

    /// Calculates the total used space across all cursor regions.
    ///
    /// # Returns
    ///
    /// The sum of all region sizes

    pub unsafe fn used(&self) -> usize {
        let regions = &*self.cursor_regions.get();
        regions.iter().map(|r| r.region_size()).sum()
    }

    /// Calculates the remaining space across all cursor regions.
    ///
    /// # Returns
    ///
    /// The sum of all remaining_space values

    pub unsafe fn remaining(&self) -> usize {
        let regions = &*self.cursor_regions.get();
        regions.iter().map(|r| r.remaining_space()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::{SharedGap, SharedGapCursorRegion};

    fn creation() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        assert!(!shared_gap.is_shared());
    }

    fn add_first_cursor() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            assert!(!shared_gap.is_shared()); // Still not shared with only 1 cursor
            assert_eq!(shared_gap.cursor_region_count(), 1);
        }
    }

    fn add_second_cursor() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            shared_gap.add_cursor_region(1);
            assert!(shared_gap.is_shared()); // Now shared with 2+ cursors
            assert_eq!(shared_gap.cursor_region_count(), 2);
        }
    }

    fn remove_cursor() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            shared_gap.add_cursor_region(1);
            assert!(shared_gap.is_shared());

            let removed = shared_gap.remove_cursor_region(0);
            assert!(removed);
            assert!(!shared_gap.is_shared()); // No longer shared with only 1 cursor
            assert_eq!(shared_gap.cursor_region_count(), 1);
        }
    }

    fn remove_nonexistent_cursor() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            let removed = shared_gap.remove_cursor_region(999);
            assert!(!removed);
        }
    }

    fn get_cursor_region() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(5);
            let region = shared_gap.cursor_region(5);
            assert!(region.is_some());
            assert_eq!(region.unwrap().cursor_id(), 5);
        }
    }

    fn get_cursor_region_mut() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(5);
            let region = shared_gap.cursor_region_mut(5);
            assert!(region.is_some());
            assert_eq!(region.unwrap().cursor_id(), 5);
        }
    }

    fn has_cursor() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            assert!(!shared_gap.has_cursor(0));
            shared_gap.add_cursor_region(0);
            assert!(shared_gap.has_cursor(0));
        }
    }

    fn region_allocation_size() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            let region = shared_gap.cursor_region(0).unwrap();
            // With 1 cursor, region size should be GAP_SIZE / 1 = 256
            assert_eq!(region.region_size(), 256);
        }
    }

    fn multiple_cursors_region_size() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            shared_gap.add_cursor_region(1);
            shared_gap.add_cursor_region(2);

            // With 3 cursors, each region should be GAP_SIZE / 3
            let region0 = shared_gap.cursor_region(0).unwrap();
            let region1 = shared_gap.cursor_region(1).unwrap();
            let region2 = shared_gap.cursor_region(2).unwrap();

            assert_eq!(region0.region_size(), 256 / 3);
            assert_eq!(region1.region_size(), 256 / 3);
            assert_eq!(region2.region_size(), 256 / 3);
        }
    }

    fn region_start_positions() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            shared_gap.add_cursor_region(1);

            let region0 = shared_gap.cursor_region(0).unwrap();
            let region1 = shared_gap.cursor_region(1).unwrap();

            assert_eq!(region0.region_start(), 0);
            assert_eq!(region1.region_start(), 128); // 256 / 2
        }
    }

    fn move_cursor_region_right() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            shared_gap.add_cursor_region(1);

            // With 2 cursors, each gets 128 space
            // Region 0 at 0, Region 1 at 128
            // Region 0 cannot move right without colliding with Region 1
            let moved = shared_gap.move_cursor_region_right(0);
            assert!(!moved);

            let region0 = shared_gap.cursor_region(0).unwrap();
            assert_eq!(region0.region_start(), 0);
        }
    }

    fn move_cursor_region_right_no_space() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            shared_gap.add_cursor_region(1);

            // Try to move region1 right (no space after it)
            let moved = shared_gap.move_cursor_region_right(1);
            assert!(!moved);
        }
    }

    fn rebalance_regions() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            shared_gap.add_cursor_region(1);
            shared_gap.add_cursor_region(2);

            let regions = shared_gap.cursor_regions_mut();
            shared_gap.rebalance(regions);

            let region0 = shared_gap.cursor_region(0).unwrap();
            let region1 = shared_gap.cursor_region(1).unwrap();
            let region2 = shared_gap.cursor_region(2).unwrap();

            // After rebalance, regions should be evenly distributed
            assert_eq!(region0.region_start(), 0);
            assert_eq!(region1.region_start(), 256 / 3);
            assert_eq!(region2.region_start(), 2 * (256 / 3));
        }
    }

    fn total_used_space() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            shared_gap.add_cursor_region(1);

            let total = shared_gap.used();
            assert_eq!(total, 256); // 128 + 128
        }
    }

    fn total_remaining_space() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            shared_gap.add_cursor_region(0);
            shared_gap.add_cursor_region(1);

            let total = shared_gap.remaining();
            assert_eq!(total, 256); // All space initially remaining
        }
    }

    fn set_shared_flag() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        assert!(!shared_gap.is_shared());

        shared_gap.set_shared(true);
        assert!(shared_gap.is_shared());

        shared_gap.set_shared(false);
        assert!(!shared_gap.is_shared());
    }

    fn gap_access() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        let gap = shared_gap.gap();
        assert_eq!(gap.gap_size(), 256);
    }

    fn gap_mut_access() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        let gap = shared_gap.gap();
        assert_eq!(gap.gap_size(), 256);
    }

    fn empty_regions() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            assert_eq!(shared_gap.cursor_region_count(), 0);
            let total = shared_gap.used();
            assert_eq!(total, 0);
        }
    }

    fn rebalance_empty() {
        let shared_gap: SharedGap<u32, 256> = SharedGap::new();
        unsafe {
            let regions = shared_gap.cursor_regions_mut();
            shared_gap.rebalance(regions); // Should not panic
            assert_eq!(shared_gap.cursor_region_count(), 0);
        }
    }

    fn large_cursor_count() {
        let shared_gap: SharedGap<u32, 1024> = SharedGap::new();
        unsafe {
            for i in 0..10 {
                shared_gap.add_cursor_region(i);
            }
            assert_eq!(shared_gap.cursor_region_count(), 10);
            assert!(shared_gap.is_shared());
        }
    }

    fn region_creation() {
        let region = SharedGapCursorRegion::new(0, 0, 100);
        assert_eq!(region.cursor_id(), 0);
        assert_eq!(region.region_start(), 0);
        assert_eq!(region.region_size(), 100);
        assert_eq!(region.remaining_space(), 100);
        assert_eq!(region.cursor_position(), 0);
    }

    fn region_with_offset() {
        let region = SharedGapCursorRegion::new(5, 50, 200);
        assert_eq!(region.cursor_id(), 5);
        assert_eq!(region.region_start(), 50);
        assert_eq!(region.region_size(), 200);
        assert_eq!(region.remaining_space(), 200);
        assert_eq!(region.cursor_position(), 50);
    }

    fn region_has_space() {
        let region = SharedGapCursorRegion::new(0, 0, 100);
        assert!(region.has_space());
    }

    fn region_no_space() {
        let mut region = SharedGapCursorRegion::new(0, 0, 1);
        region.decrement_remaining(1);
        assert!(!region.has_space());
    }

    fn decrement_remaining() {
        let mut region = SharedGapCursorRegion::new(0, 0, 100);
        region.decrement_remaining(10);
        assert_eq!(region.remaining_space(), 90);
    }

    fn decrement_remaining_clamps() {
        let mut region = SharedGapCursorRegion::new(0, 0, 10);
        region.decrement_remaining(20);
        assert_eq!(region.remaining_space(), 0);
    }

    fn set_cursor_position() {
        let mut region = SharedGapCursorRegion::new(0, 0, 100);
        region.set_cursor_position(50);
        assert_eq!(region.cursor_position(), 50);
    }

    fn move_to() {
        let mut region = SharedGapCursorRegion::new(0, 0, 100);
        region.move_to(50);
        assert_eq!(region.region_start(), 50);
        assert_eq!(region.cursor_position(), 50);
    }

    fn move_to_updates_cursor_position() {
        let mut region = SharedGapCursorRegion::new(0, 10, 100);
        region.set_cursor_position(20);
        region.move_to(50);
        assert_eq!(region.region_start(), 50);
        assert_eq!(region.cursor_position(), 60); // 20 + (50 - 10)
    }

    fn move_to_zero() {
        let mut region = SharedGapCursorRegion::new(0, 100, 100);
        region.move_to(0);
        assert_eq!(region.region_start(), 0);
        assert_eq!(region.cursor_position(), 0);
    }

    fn region_clone() {
        let region = SharedGapCursorRegion::new(1, 10, 50);
        let cloned = region.clone();
        assert_eq!(cloned.cursor_id(), 1);
        assert_eq!(cloned.region_start(), 10);
        assert_eq!(cloned.region_size(), 50);
    }

    fn multiple_decrements() {
        let mut region = SharedGapCursorRegion::new(0, 0, 100);
        region.decrement_remaining(1);
        region.decrement_remaining(1);
        region.decrement_remaining(1);
        assert_eq!(region.remaining_space(), 97);
    }

    fn region_size_zero() {
        let region = SharedGapCursorRegion::new(0, 0, 0);
        assert_eq!(region.region_size(), 0);
        assert_eq!(region.remaining_space(), 0);
        assert!(!region.has_space());
    }

    fn large_region() {
        let region = SharedGapCursorRegion::new(0, 0, 1_000_000);
        assert_eq!(region.region_size(), 1_000_000);
        assert!(region.has_space());
    }
}
