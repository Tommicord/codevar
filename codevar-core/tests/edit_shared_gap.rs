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

//! Integration tests for shared gap functionality.

use codevar_core::edit::{StreamWritable, TextEditablePut, Writable};

#[test]
fn shared_gap_integration_two_cursors_proximity() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let cursor_id2 = writable.append_cursor();

    unsafe {
        // Both cursors start at same position, should be in proximity
        assert!(writable.proximity(cursor_id1, cursor_id2));

        // Create shared gap for cursors in proximity
        writable.new_shared_gap(cursor_id1);

        // Both cursors should now be in a shared gap
        assert!(writable.is_cursor_in_shared_gap(cursor_id1));
        assert!(writable.is_cursor_in_shared_gap(cursor_id2));

        // They should be in the same shared gap
        let gap_idx1 = writable.shared_gap_index(cursor_id1);
        let gap_idx2 = writable.shared_gap_index(cursor_id2);
        assert_eq!(gap_idx1, gap_idx2);
    }
}

#[test]
fn shared_gap_integration_cursor_leaves_shared_gap() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let cursor_id2 = writable.append_cursor();

    unsafe {
        writable.new_shared_gap(cursor_id1);

        // Move cursor1 far away from cursor2
        if let Some(cursor) = writable.cursor_from_id_mut(cursor_id1) {
            cursor.move_forward(5000); // Move beyond GAP_SIZE
        }
        // Cursor1 should no longer be in proximity with cursor2
        assert!(!writable.proximity(cursor_id1, cursor_id2));
    }
}

#[test]
fn shared_gap_integration_three_cursors() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let cursor_id2 = writable.append_cursor();
    let cursor_id3 = writable.append_cursor();

    unsafe {
        writable.new_shared_gap(cursor_id1);
        writable.new_shared_gap(cursor_id2);
        writable.new_shared_gap(cursor_id3);

        // All three cursors should be in the same shared gap
        let gap_idx1 = writable.shared_gap_index(cursor_id1);
        let gap_idx2 = writable.shared_gap_index(cursor_id2);
        let gap_idx3 = writable.shared_gap_index(cursor_id3);

        assert_eq!(gap_idx1, gap_idx2);
        assert_eq!(gap_idx2, gap_idx3);
    }
}

#[test]
fn shared_gap_integration_stream_writable_put() {
    let mut writable: StreamWritable<u32, u8, 4096> =
        StreamWritable::with_capacity("test", 1024);

    let cursor_id1 = writable.base_mut().append_cursor();
    let cursor_id2 = writable.base_mut().append_cursor();

    unsafe {
        writable.base_mut().new_shared_gap(cursor_id1);

        // Put characters using shared gap
        writable.put(0x41, false, cursor_id1);
        writable.put(0x42, false, cursor_id2);

        // Both cursors should still be in shared gap
        assert!(writable.base().is_cursor_in_shared_gap(cursor_id1));
        assert!(writable.base().is_cursor_in_shared_gap(cursor_id2));
    }
}

#[test]
fn shared_gap_integration_remove_cursor_from_shared_gap() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let cursor_id2 = writable.append_cursor();

    unsafe {
        writable.new_shared_gap(cursor_id1);

        // Both cursors in shared gap
        assert!(writable.is_cursor_in_shared_gap(cursor_id1));
        assert!(writable.is_cursor_in_shared_gap(cursor_id2));

        // Remove cursor2
        writable.delete_cursor(cursor_id2);

        // Cursor1 should still be in shared gap (but no longer shared)
        let gap_idx = writable.shared_gap_index(cursor_id1);
        assert_ne!(gap_idx, usize::MAX);
    }
}

#[test]
fn shared_gap_integration_region_space_tracking() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let _cursor_id2 = writable.append_cursor();

    unsafe {
        writable.new_shared_gap(cursor_id1);

        let gap_idx = writable.shared_gap_index(cursor_id1);
        let shared_gaps = &*writable.shared_gaps().get();

        if let Some(shared_gap) = shared_gaps.get(gap_idx) {
            let total_space = shared_gap.used();
            assert!(total_space > 0);

            let remaining = shared_gap.remaining();
            assert!(remaining > 0);
        }
    }
}

#[test]
fn shared_gap_integration_cursor_not_in_proximity() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let _cursor_id2 = writable.append_cursor();

    unsafe {
        // Move cursor2 far away
        if let Some(cursor) = writable.cursor_from_id_mut(_cursor_id2) {
            cursor.move_forward(10000);
        }

        // Should not be in proximity
        assert!(!writable.proximity(cursor_id1, _cursor_id2));

        writable.new_shared_gap(cursor_id1);

        // Cursor1 should not be in shared gap (no other cursor in proximity)
        assert!(!writable.is_cursor_in_shared_gap(cursor_id1));
    }
}

#[test]
fn shared_gap_integration_multiple_shared_gaps() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let _cursor_id2 = writable.append_cursor();
    let cursor_id3 = writable.append_cursor();
    let cursor_id4 = writable.append_cursor();

    unsafe {
        // Move cursor3 and cursor4 far away
        if let Some(cursor) = writable.cursor_from_id_mut(cursor_id3) {
            cursor.move_forward(10000);
        }
        if let Some(cursor) = writable.cursor_from_id_mut(cursor_id4) {
            cursor.move_forward(10000);
        }

        // Create shared gap for cursor1 and cursor2
        writable.new_shared_gap(cursor_id1);

        // Create shared gap for cursor3 and cursor4
        writable.new_shared_gap(cursor_id3);

        // Should have two different shared gaps
        let gap_idx1 = writable.shared_gap_index(cursor_id1);
        let gap_idx3 = writable.shared_gap_index(cursor_id3);

        assert_ne!(gap_idx1, gap_idx3);
    }
}

#[test]
fn shared_gap_integration_cursor_reassignment() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let _cursor_id2 = writable.append_cursor();
    let cursor_id3 = writable.append_cursor();

    unsafe {
        writable.new_shared_gap(cursor_id1);

        // Move cursor3 close to cursor1
        if let Some(cursor) = writable.cursor_from_id_mut(cursor_id3) {
            cursor.move_forward(10);
        }

        // Cursor3 should join the existing shared gap
        writable.new_shared_gap(cursor_id3);

        let gap_idx1 = writable.shared_gap_index(cursor_id1);
        let gap_idx3 = writable.shared_gap_index(cursor_id3);

        assert_eq!(gap_idx1, gap_idx3);
    }
}

#[test]
fn shared_gap_integration_newline_handling() {
    let mut writable: StreamWritable<u32, u8, 4096> =
        StreamWritable::with_capacity("test", 1024);

    let cursor_id1 = writable.base_mut().append_cursor();
    let _cursor_id2 = writable.base_mut().append_cursor();

    unsafe {
        writable.base_mut().new_shared_gap(cursor_id1);

        let initial_space_y = writable.base().v_space();

        // Put newline in shared gap
        writable.put(0x0A, false, cursor_id1);

        // space_y should increment
        assert_eq!(writable.base().v_space(), initial_space_y + 1);
    }
}

#[test]
fn shared_gap_integration_region_movement() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let _cursor_id2 = writable.append_cursor();

    unsafe {
        writable.new_shared_gap(cursor_id1);

        let gap_idx = writable.shared_gap_index(cursor_id1);
        let shared_gaps = &mut *writable.shared_gaps().get();

        if let Some(shared_gap) = shared_gaps.get(gap_idx) {
            // With two cursors the gap is fully partitioned; region 0 cannot move right.
            let moved = shared_gap.move_cursor_region_right(cursor_id1);
            assert!(!moved);
        }
    }
}

#[test]
fn shared_gap_integration_rebalance_after_cursor_add() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let _cursor_id2 = writable.append_cursor();

    unsafe {
        writable.new_shared_gap(cursor_id1);

        let gap_idx = writable.shared_gap_index(cursor_id1);
        let shared_gaps = &mut *writable.shared_gaps().get();

        if let Some(shared_gap) = shared_gaps.get(gap_idx) {
            let initial_region_count = shared_gap.cursor_region_count();
            let regions = shared_gap.cursor_regions_mut();
            // Rebalance regions
            shared_gap.rebalance(regions);

            // Region count should remain the same
            assert_eq!(shared_gap.cursor_region_count(), initial_region_count);
        }
    }
}

#[test]
fn shared_gap_integration_cursor_to_gap_mapping() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let cursor_id2 = writable.append_cursor();
    let cursor_id3 = writable.append_cursor();

    unsafe {
        writable.new_shared_gap(cursor_id1);

        // Check cursor_to_gap mapping
        let gap_idx1 = writable.shared_gap_index(cursor_id1);
        let gap_idx2 = writable.shared_gap_index(cursor_id2);
        let gap_idx3 = writable.shared_gap_index(cursor_id3);

        assert_ne!(gap_idx1, usize::MAX);
        assert_ne!(gap_idx2, usize::MAX);
        assert_eq!(gap_idx3, usize::MAX); // cursor3 not in shared gap
    }
}

#[test]
fn shared_gap_integration_shared_gaps_vector_growth() {
    let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();

    let cursor_id1 = writable.append_cursor();
    let _cursor_id2 = writable.append_cursor();
    let cursor_id3 = writable.append_cursor();
    let cursor_id4 = writable.append_cursor();

    unsafe {
        writable.new_shared_gap(cursor_id1);

        let shared_gaps = &*writable.shared_gaps().get();
        let initial_count = shared_gaps.len();

        // Move cursor3 far away and create another shared gap
        if let Some(cursor) = writable.cursor_from_id_mut(cursor_id3) {
            cursor.move_forward(10000);
        }
        if let Some(cursor) = writable.cursor_from_id_mut(cursor_id4) {
            cursor.move_forward(10000);
        }

        writable.new_shared_gap(cursor_id3);

        let shared_gaps = &*writable.shared_gaps().get();
        assert!(shared_gaps.len() > initial_count);
    }
}
