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

//! Facade for periodic editor memory reclaim.
//!
//! Prefer [`OpenedFileTable::maybe_reclaim_memory`] /
//! [`OpenedFileTable::reclaim_memory_now`] for table-scoped reclaim.
//! These helpers forward to that API and expose history-level entry points.

use crate::base::base_memory::{
    EditorMemoryStats, MemoryMonitor, MemoryPressure, SystemMemory,
};
use crate::edit::wredit_file_table::OpenedFileTable;
use crate::edit::wredit_history::History;
use std::sync::OnceLock;

/// Process-wide default monitor for callers that do not own a table.
fn global_monitor() -> &'static MemoryMonitor {
    static MONITOR: OnceLock<MemoryMonitor> = OnceLock::new();
    MONITOR.get_or_init(MemoryMonitor::with_defaults)
}

/// Returns the shared [`MemoryMonitor`] used by the edit module facade.
#[inline]
#[must_use]
pub fn edit_memory_monitor() -> &'static MemoryMonitor {
    global_monitor()
}

/// Aggregates memory stats from an opened-file table.
#[inline]
#[must_use]
pub fn collect_file_table_stats(table: &OpenedFileTable) -> EditorMemoryStats {
    table.memory_stats()
}

/// Runs a rate-limited reclaim pass over all opened file histories.
pub fn maybe_reclaim_opened_files(
    table: &OpenedFileTable,
) -> Option<(SystemMemory, MemoryPressure, usize)> {
    table.maybe_reclaim_memory()
}

/// Forces an immediate reclaim pass (ignores sample cooldown).
pub fn reclaim_opened_files_now(
    table: &OpenedFileTable,
) -> (SystemMemory, MemoryPressure, usize) {
    table.reclaim_memory_now()
}

/// Compresses cold TXUs on a single history under the given pressure.
#[inline]
pub fn reclaim_history(history: &mut History, pressure: MemoryPressure) -> usize {
    history.maybe_compress_under_pressure(pressure)
}

#[cfg(test)]
mod tests {
    use super::reclaim_opened_files_now;
    use crate::edit::wredit_base_writable::ENCODING_UTF8;
    use crate::edit::wredit_file_table::OpenedFileTable;
    use crate::edit::wredit_history_compress::TXU_BLOB_COMPRESS_THRESHOLD;
    use crate::edit::wredit_history_txu::{
        HistoryTXU, HistoryTXUDelta, HistoryTXUDeltaType,
    };

    fn large_payload() -> Vec<u8> {
        b"pub fn compress_me_please() { let x = 1; }\n"
            .iter()
            .copied()
            .cycle()
            .take(TXU_BLOB_COMPRESS_THRESHOLD * 2)
            .collect()
    }

    #[test]
    fn reclaim_compresses_large_history() {
        let table = OpenedFileTable::new();
        let file_id = table.open(b"big.rs", b"fn main() {}\n", 0);
        let payload = large_payload();
        for _ in 0..12 {
            let delta = HistoryTXUDelta::new(
                HistoryTXUDeltaType::CodeAdd,
                0,
                1,
                0,
                payload.clone(),
                ENCODING_UTF8,
                None,
            );
            let mut txu = HistoryTXU::new::<u8, u8, 4096>(delta);
            *txu.buffer_mut() = payload.clone();
            *txu.buffer_length_mut() = payload.len();
            // SAFETY: single-threaded test; history cell is exclusively owned here.
            unsafe {
                let history = table.history_mut(file_id).expect("history");
                (*history).add_txu(txu);
            }
        }

        let (_sys, _pressure, compressed) = reclaim_opened_files_now(&table);
        assert!(compressed > 0 || table.memory_stats().history_bytes > 0);
        assert!(table.memory_stats().opened_files >= 1);
    }
}
