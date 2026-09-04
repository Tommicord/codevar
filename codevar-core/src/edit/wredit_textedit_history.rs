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

use super::{History, StreamWritable, TextEditableHistory, TextEditablePut};
use crate::edit::wredit_history_txu::{HistoryTXU, HistoryTXUDelta, HistoryTXUDeltaType};
use crate::edit::wredit_history_txu_ppbuff::{ChangeType, PpbuffBuilder};
use crate::edit::wredit_writable_trait::Writable;
use std::cell::UnsafeCell;
use std::sync::Arc;

impl<Raw, Buf, const GAP_SIZE: usize> TextEditableHistory<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
{
    unsafe fn begin_edit_group(&mut self) {
        let history = self.base_mut().history();
        let history_mut = history.get();
        unsafe {
            (*history_mut).set_edit_group_active(true);
            (*history_mut).clear_pending_changes();
        }
    }

    unsafe fn end_edit_group(&mut self) {
        let history = self.base_mut().history();
        let mut history_mut = history.get();
        unsafe {
            (*history_mut).set_edit_group_active(false);
            for txu in (*history_mut).pending_changes().to_vec() {
                (*history_mut).add_txu(txu);
            }
            (*history_mut).clear_pending_changes();
        }
    }

    unsafe fn undo(&mut self) -> bool {
        let history = self.base_mut().history();
        let history_mut = history.get();

        let current_index = unsafe { (*history_mut).current_index() };
        if current_index == 0 {
            return false;
        }
        let new_index = current_index - 1;
        if unsafe { (*history_mut).ensure_txu_decompressed(new_index).is_err() } {
            return false;
        }
        let txu = if let Ok(txus) = unsafe { (*history_mut).txus().read() } {
            if new_index >= txus.len() {
                return false;
            }
            Some(txus[new_index].clone())
        } else {
            return false;
        };
        let mut txu = match txu {
            Some(t) => t,
            None => return false,
        };
        if txu.ensure_decompressed().is_err() {
            return false;
        }
        let delta = txu.delta();
        let start = delta.start() as usize;
        let end = delta.end() as usize;
        let ppbuff = delta.ppbuff().to_vec();
        let delta_type = delta.type_().clone();

        let (_changes, text_data) = match unsafe { Self::parse_ppbuff(&ppbuff) } {
            Ok(parsed) => parsed,
            Err(_) => return false,
        };
        // Release history borrow before text operations
        // Now we can safely use self for text operations
        match delta_type {
            HistoryTXUDeltaType::CodeAdd | HistoryTXUDeltaType::CodeAdd2 => {
                // Inverse of add is delete: remove the text that was added
                self.delete_raw_range_at(start, end.saturating_sub(start));
            }
            HistoryTXUDeltaType::CodeDelete | HistoryTXUDeltaType::CodeDelete2 => {
                // Inverse of delete is add: restore the deleted text
                if let Some(cursor) = self.base_mut().cursor_from_id_mut(0) {
                    cursor.set_array_pos(start);
                }
                for chunk in text_data.chunks_exact(4) {
                    let ch = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    self.put(ch, true, 0);
                }
            }
            HistoryTXUDeltaType::CodeReplace | HistoryTXUDeltaType::CodeReplace2 => {
                // Inverse of replace is to restore the original text
                self.delete_raw_range_at(start, end.saturating_sub(start));
                if let Some(cursor) = self.base_mut().cursor_from_id_mut(0) {
                    cursor.set_array_pos(start);
                }
                for chunk in text_data.chunks_exact(4) {
                    let ch = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    self.put(ch, true, 0);
                }
            }
        }
        unsafe { (*history_mut).set_current_index(new_index) };
        true
    }

    unsafe fn redo(&mut self) -> bool {
        let history = self.base_mut().history();
        let history_mut = history.get();

        let current_index = unsafe { (*history_mut).current_index() };
        if let Ok(txus) = unsafe { (*history_mut).txus().read() } {
            if current_index >= txus.len() {
                return false;
            }
        } else {
            return false;
        }
        let new_index = current_index + 1;
        let txu_index = new_index - 1;
        if unsafe { (*history_mut).ensure_txu_decompressed(txu_index).is_err() } {
            return false;
        }
        let txu = if let Ok(txus) = unsafe { (*history_mut).txus().read() } {
            if new_index > txus.len() {
                return false;
            }
            Some(txus[txu_index].clone())
        } else {
            return false;
        };
        let mut txu = match txu {
            Some(t) => t,
            None => return false,
        };
        if txu.ensure_decompressed().is_err() {
            return false;
        }
        let delta = txu.delta();
        let start = delta.start() as usize;
        let end = delta.end() as usize;
        let ppbuff = delta.ppbuff().to_vec();
        let delta_type = delta.type_().clone();
        let (_changes, text_data) = match unsafe { Self::parse_ppbuff(&ppbuff) } {
            Ok(parsed) => parsed,
            Err(_) => return false,
        };

        match delta_type {
            HistoryTXUDeltaType::CodeAdd | HistoryTXUDeltaType::CodeAdd2 => {
                // Re-apply the add operation
                if let Some(cursor) = self.base_mut().cursor_from_id_mut(0) {
                    cursor.set_array_pos(start);
                }
                for chunk in text_data.chunks_exact(4) {
                    let ch = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    self.put(ch, true, 0);
                }
            }
            HistoryTXUDeltaType::CodeDelete | HistoryTXUDeltaType::CodeDelete2 => {
                // Re-apply the delete operation
                self.delete_raw_range_at(start, end.saturating_sub(start));
            }
            HistoryTXUDeltaType::CodeReplace | HistoryTXUDeltaType::CodeReplace2 => {
                // Re-apply the replace operation
                self.delete_raw_range_at(start, end.saturating_sub(start));
                if let Some(cursor) = self.base_mut().cursor_from_id_mut(0) {
                    cursor.set_array_pos(start);
                }
                for chunk in text_data.chunks_exact(4) {
                    let ch = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    self.put(ch, true, 0);
                }
            }
        }

        unsafe { (*history_mut).set_current_index(new_index) };
        true
    }

    fn can_undo(&self) -> bool {
        let history = self.base().history();
        unsafe { (*history.get()).current_index() > 0 }
    }

    fn can_redo(&self) -> bool {
        let history = self.base().history();
        unsafe {
            if let Ok(txus) = (*history.get()).txus().read() {
                (*history.get()).current_index() < txus.len()
            } else {
                false
            }
        }
    }

    unsafe fn clear_history(&mut self) {
        let history = self.base_mut().history();
        let history_mut = history.get();
        unsafe {
            (*history_mut).set_current_index(0);
            (*history_mut).set_redo_start_index(0);
            (*history_mut).clear_pending_changes();

            if let Ok(mut txus) = (*history_mut).txus_mut().write() {
                *txus = Arc::new(Box::new([]));
            }
        }
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> StreamWritable<Raw, Buf, GAP_SIZE> {
    /// Parses the preprocessed buffer (ppbuff) to extract change information and text data.
    ///
    /// # Arguments
    ///
    /// * `ppbuff` - The preprocessed buffer bytecode to parse
    ///
    /// # Returns
    ///
    /// A tuple containing (changes vector, text data vector) or an error
    unsafe fn parse_ppbuff(
        ppbuff: &[u8],
    ) -> Result<
        (
            Vec<crate::edit::wredit_history_txu_ppbuff::KeyValue>,
            Vec<u8>,
        ),
        (),
    > {
        let mut offset = 0;

        // Read start
        if offset + std::mem::size_of::<usize>() > ppbuff.len() {
            return Err(());
        }
        let _start = usize::from_le_bytes(
            ppbuff[offset..offset + std::mem::size_of::<usize>()]
                .try_into()
                .map_err(|_| ())?,
        );
        offset += std::mem::size_of::<usize>();

        // Read end
        if offset + std::mem::size_of::<usize>() > ppbuff.len() {
            return Err(());
        }
        let _end = usize::from_le_bytes(
            ppbuff[offset..offset + std::mem::size_of::<usize>()]
                .try_into()
                .map_err(|_| ())?,
        );
        offset += std::mem::size_of::<usize>();

        // Read change count
        if offset + std::mem::size_of::<u32>() > ppbuff.len() {
            return Err(());
        }
        let change_count = u32::from_le_bytes(
            ppbuff[offset..offset + std::mem::size_of::<u32>()]
                .try_into()
                .map_err(|_| ())?,
        ) as usize;
        offset += std::mem::size_of::<u32>();

        // Read changes
        let mut changes = Vec::with_capacity(change_count);
        for _ in 0..change_count {
            // Read line index
            if offset + std::mem::size_of::<usize>() > ppbuff.len() {
                return Err(());
            }
            let line_index = usize::from_le_bytes(
                ppbuff[offset..offset + std::mem::size_of::<usize>()]
                    .try_into()
                    .map_err(|_| ())?,
            );
            offset += std::mem::size_of::<usize>();

            // Read line length
            if offset + std::mem::size_of::<usize>() > ppbuff.len() {
                return Err(());
            }
            let line_length = usize::from_le_bytes(
                ppbuff[offset..offset + std::mem::size_of::<usize>()]
                    .try_into()
                    .map_err(|_| ())?,
            );
            offset += std::mem::size_of::<usize>();

            // Read change type
            if offset >= ppbuff.len() {
                return Err(());
            }
            let change_type_byte = ppbuff[offset];
            offset += 1;

            let change_type =
                crate::edit::wredit_history_txu_ppbuff::ChangeType::from_byte(
                    change_type_byte,
                )
                .map_err(|_| ())?;

            changes.push(crate::edit::wredit_history_txu_ppbuff::KeyValue::new(
                line_index,
                line_length,
                change_type,
            ));
        }

        // Read separator (0xFFFFFFFF)
        if offset + std::mem::size_of::<u32>() > ppbuff.len() {
            return Err(());
        }
        let _separator = u32::from_le_bytes(
            ppbuff[offset..offset + std::mem::size_of::<u32>()]
                .try_into()
                .map_err(|_| ())?,
        );
        offset += std::mem::size_of::<u32>();

        // Read text data length
        if offset + std::mem::size_of::<u32>() > ppbuff.len() {
            return Err(());
        }
        let text_data_len = u32::from_le_bytes(
            ppbuff[offset..offset + std::mem::size_of::<u32>()]
                .try_into()
                .map_err(|_| ())?,
        ) as usize;
        offset += std::mem::size_of::<u32>();

        // Read text data
        if offset + text_data_len > ppbuff.len() {
            return Err(());
        }
        let text_data = ppbuff[offset..offset + text_data_len].to_vec();

        Ok((changes, text_data))
    }

    unsafe fn record(history: &UnsafeCell<History>, builder: &PpbuffBuilder) {
        if let Ok(delta) = HistoryTXUDelta::from_ppbuff_builder(
            HistoryTXUDeltaType::CodeReplace,
            &builder,
            0, // UTF-8 encoding
        ) {
            let txu = HistoryTXU::new::<Raw, Buf, GAP_SIZE>(delta);
            unsafe {
                if (*history.get()).is_edit_group_active() {
                    (*history.get()).pending_changes_mut().push(txu);
                } else {
                    (*history.get()).add_txu(txu);
                }
            }
        }
    }
    /// Records an insertion operation in the history.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID that performed the insertion
    /// * `text` - The text that was inserted
    /// * `position` - The position where the insertion occurred
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    pub unsafe fn record_insert(
        &mut self,
        _cursor_id: usize,
        text: &[u32],
        position: usize,
    ) {
        let history = self.base_mut().history();
        let mut builder = PpbuffBuilder::new(position, position + text.len());
        builder.add_change(0, text.len(), ChangeType::Insert);
        builder.set_text_data(text.iter().flat_map(|&c| c.to_le_bytes()).collect());
        unsafe { Self::record(history, &builder) };
    }

    /// Records a deletion operation in the history.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID that performed the deletion
    /// * `deleted_text` - The text that was deleted
    /// * `position` - The position where the deletion occurred
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    pub unsafe fn record_delete(
        &mut self,
        _cursor_id: usize,
        deleted_text: &[u32],
        position: usize,
    ) {
        let base = self.base_mut();
        let history = base.history();

        // Create ppbuff for the deletion
        let mut builder = PpbuffBuilder::new(position, position + deleted_text.len());
        builder.add_change(0, deleted_text.len(), ChangeType::Delete);
        builder
            .set_text_data(deleted_text.iter().flat_map(|&c| c.to_le_bytes()).collect());

        if let Ok(delta) = HistoryTXUDelta::from_ppbuff_builder(
            HistoryTXUDeltaType::CodeDelete,
            &builder,
            base.encoding(),
        ) {
            let txu = HistoryTXU::new::<Raw, Buf, GAP_SIZE>(delta);
            let history_mut = history.get();
            unsafe {
                if (*history_mut).is_edit_group_active() {
                    (*history_mut).pending_changes_mut().push(txu);
                } else {
                    (*history_mut).add_txu(txu);
                }
            }
        }
        unsafe { Self::record(history, &builder) };
    }

    /// Records a replacement operation in the history.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID that performed the replacement
    /// * `old_text` - The text that was replaced
    /// * `new_text` - The new text that replaced it
    /// * `position` - The position where the replacement occurred
    ///
    /// # Safety
    ///
    /// The cursor_id must be valid.
    pub unsafe fn record_replace(
        &mut self,
        _cursor_id: usize,
        old_text: &[u32],
        new_text: &[u32],
        position: usize,
    ) {
        let history = self.base_mut().history();
        let mut builder = PpbuffBuilder::new(position, position + old_text.len());
        builder.add_change(0, new_text.len(), ChangeType::Replace);
        builder.set_text_data(new_text.iter().flat_map(|&c| c.to_le_bytes()).collect());
        unsafe { Self::record(history, &builder) };
    }
}

#[cfg(test)]
mod tests {
    use crate::edit::{StreamWritable, TextEditableHistory, TextEditablePut};

    fn undo_empty_history() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);
        unsafe {
            let result = writable.undo();
            assert!(!result, "Undo should return false when history is empty");
        }
    }

    fn redo_empty_history() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);
        unsafe {
            let result = writable.redo();
            assert!(!result, "Redo should return false when history is empty");
        }
    }

    fn can_undo_empty_history() {
        let writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);
        assert!(
            !writable.can_undo(),
            "can_undo should return false when history is empty"
        );
    }

    fn can_redo_empty_history() {
        let writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);
        assert!(
            !writable.can_redo(),
            "can_redo should return false when history is empty"
        );
    }

    fn undo_after_single_insertion() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 0); // Insert 'A'
            let text_before_undo = writable.snapshot_text();
            assert_eq!(text_before_undo.len(), 1);
            assert_eq!(text_before_undo[0], 0x41);

            let undo_result = writable.undo();
            assert!(undo_result, "Undo should succeed after insertion");

            let text_after_undo = writable.snapshot_text();
            assert_eq!(text_after_undo.len(), 0, "Text should be empty after undo");
        }
    }

    fn redo_after_undo() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 0); // Insert 'A'
            writable.undo();

            let redo_result = writable.redo();
            assert!(redo_result, "Redo should succeed after undo");

            let text_after_redo = writable.snapshot_text();
            assert_eq!(text_after_redo.len(), 1);
            assert_eq!(
                text_after_redo[0], 0x41,
                "Text should be restored after redo"
            );
        }
    }

    fn multiple_undo_operations() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            // Insert multiple characters
            writable.put(0x41, false, 0); // 'A'
            writable.put(0x42, false, 0); // 'B'
            writable.put(0x43, false, 0); // 'C'

            let text_initial = writable.snapshot_text();
            assert_eq!(text_initial.len(), 3);

            // Undo one operation
            writable.undo();
            let text_after_first_undo = writable.snapshot_text();
            assert_eq!(text_after_first_undo.len(), 2);

            // Undo another operation
            writable.undo();
            let text_after_second_undo = writable.snapshot_text();
            assert_eq!(text_after_second_undo.len(), 1);

            // Undo the last operation
            writable.undo();
            let text_after_third_undo = writable.snapshot_text();
            assert_eq!(text_after_third_undo.len(), 0);
        }
    }

    fn multiple_redo_operations() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            // Insert multiple characters
            writable.put(0x41, false, 0); // 'A'
            writable.put(0x42, false, 0); // 'B'
            writable.put(0x43, false, 0); // 'C'

            // Undo all operations
            writable.undo();
            writable.undo();
            writable.undo();

            let text_after_all_undo = writable.snapshot_text();
            assert_eq!(text_after_all_undo.len(), 0);

            // Redo all operations
            writable.redo();
            writable.redo();
            writable.redo();

            let text_after_all_redo = writable.snapshot_text();
            assert_eq!(text_after_all_redo.len(), 3);
            assert_eq!(text_after_all_redo[0], 0x41);
            assert_eq!(text_after_all_redo[1], 0x42);
            assert_eq!(text_after_all_redo[2], 0x43);
        }
    }

    fn undo_at_history_boundary() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 0); // Insert 'A'

            // Undo the only operation
            let first_undo = writable.undo();
            assert!(first_undo);

            // Try to undo again (should fail)
            let second_undo = writable.undo();
            assert!(!second_undo, "Undo should fail when at history boundary");
        }
    }

    fn redo_at_history_boundary() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 0); // Insert 'A'

            // Try to redo without any undo (should fail)
            let redo_without_undo = writable.redo();
            assert!(
                !redo_without_undo,
                "Redo should fail when at history boundary"
            );
        }
    }

    fn clear_history() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 0); // Insert 'A'
            writable.put(0x42, false, 0); // Insert 'B'

            assert!(writable.can_undo(), "Should be able to undo before clear");

            writable.clear_history();

            assert!(
                !writable.can_undo(),
                "Should not be able to undo after clear"
            );
            assert!(
                !writable.can_redo(),
                "Should not be able to redo after clear"
            );
        }
    }

    fn begin_end_edit_group() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.begin_edit_group();
            writable.put(0x41, false, 0); // 'A'
            writable.put(0x42, false, 0); // 'B'
            writable.put(0x43, false, 0); // 'C'
            writable.end_edit_group();

            let text_after_group = writable.snapshot_text();
            assert_eq!(text_after_group.len(), 3);

            // Undo should undo the entire group
            writable.undo();
            let text_after_undo = writable.snapshot_text();
            assert_eq!(text_after_undo.len(), 0);
        }
    }

    fn undo_with_newlines() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 0); // 'A'
            writable.put(0x0A, false, 0); // newline
            writable.put(0x42, false, 0); // 'B'

            let text_before = writable.snapshot_text();
            assert_eq!(text_before.len(), 3);

            writable.undo();

            let text_after = writable.snapshot_text();
            assert_eq!(text_after.len(), 2); // Should undo only the last character
        }
    }

    fn interleaved_undo_redo() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 0); // 'A'
            writable.put(0x42, false, 0); // 'B'
            writable.put(0x43, false, 0); // 'C'

            // Undo one
            writable.undo();
            let text_after_undo = writable.snapshot_text();
            assert_eq!(text_after_undo.len(), 2);

            // Redo
            writable.redo();
            let text_after_redo = writable.snapshot_text();
            assert_eq!(text_after_redo.len(), 3);

            // Undo again
            writable.undo();
            let text_after_second_undo = writable.snapshot_text();
            assert_eq!(text_after_second_undo.len(), 2);
        }
    }

    fn redo_after_new_operation() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 0); // 'A'
            writable.put(0x42, false, 0); // 'B'

            writable.undo();

            // New operation should clear redo stack
            writable.put(0x43, false, 0); // 'C'

            let text_after_new_op = writable.snapshot_text();
            assert_eq!(text_after_new_op.len(), 2);
            assert_eq!(text_after_new_op[0], 0x41);
            assert_eq!(text_after_new_op[1], 0x43);

            // Redo should fail
            let redo_result = writable.redo();
            assert!(!redo_result, "Redo should fail after new operation");
        }
    }

    fn can_undo_can_redo_state_tracking() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            assert!(!writable.can_undo());
            assert!(!writable.can_redo());

            writable.put(0x41, false, 0);
            assert!(writable.can_undo());
            assert!(!writable.can_redo());

            writable.undo();
            assert!(!writable.can_undo());
            assert!(writable.can_redo());

            writable.redo();
            assert!(writable.can_undo());
            assert!(!writable.can_redo());
        }
    }

    fn undo_preserves_cursor_position() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 0); // 'A'
            writable.put(0x42, false, 0); // 'B'
            writable.put(0x43, false, 0); // 'C'

            let cursor_before = writable.base().cursor_from_id(0);
            let pos_before = cursor_before.unwrap().array_pos();

            writable.undo();

            let cursor_after = writable.base().cursor_from_id(0);
            let pos_after = cursor_after.unwrap().array_pos();

            // Cursor should be at the position after undo
            assert!(pos_after < pos_before);
        }
    }
}
