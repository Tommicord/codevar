//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Opened-file table storing disk pointers and eagerly loaded writables.

use crate::base::base_memory::{
    EditorMemoryStats, MemoryMonitor, MemoryPressure, SystemMemory,
};
use crate::edit::wredit_base_writable::BaseWritable;
use crate::edit::wredit_history::History;
use crate::edit::wredit_observer::{UserActionEvent, UserActionObserver, UserActionType};
use crate::edit::wredit_writable_trait::Writable;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

fn file_table_monitor() -> &'static MemoryMonitor {
    static MONITOR: OnceLock<MemoryMonitor> = OnceLock::new();
    MONITOR.get_or_init(MemoryMonitor::with_defaults)
}

/// Opaque pointer to an opened file on disk (or a virtual path on WASM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct WritableFileHandle {
    /// Stable file identifier assigned by this table.
    pub file_id: u64,
    /// Operating-system handle when available; zero otherwise.
    pub os_handle: u64,
    /// Byte offset of the interned path in [`OpenedFileTable::path_arena`].
    pub path_offset: u32,
    /// Length of the interned path in bytes.
    pub path_len: u32,
}

/// Handle pairing a disk pointer with an eagerly loaded writable.
pub struct OpenedFile {
    /// Disk pointer (path intern + OS handle).
    pub pointer: WritableFileHandle,
    /// Eagerly loaded buffer for collaborative packing.
    pub writable: BaseWritable<u8, u8, 4096>,
}

/// Table of opened files. Only interned path bytes plus OS handles are retained
/// as the disk mapping; buffer contents live in the writable.
pub struct OpenedFileTable {
    next_id: AtomicU64,
    path_arena: Mutex<Vec<u8>>,
    files: Mutex<HashMap<u64, OpenedFile>>,
}

impl OpenedFileTable {
    /// Creates an empty table.
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            path_arena: Mutex::new(Vec::new()),
            files: Mutex::new(HashMap::new()),
        }
    }

    /// Interns `path`, allocates a file id, and eagerly loads `bytes` into a writable.
    pub fn open(&self, path: &[u8], bytes: &[u8], os_handle: u64) -> u64 {
        let file_id = self.next_id.fetch_add(1, Ordering::AcqRel);
        let (path_offset, path_len) = {
            let mut arena = self
                .path_arena
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let offset = arena.len() as u32;
            arena.extend_from_slice(path);
            (offset, path.len() as u32)
        };
        let mut writable = BaseWritable::<u8, u8, 4096>::with_capacity(
            core::str::from_utf8(path).unwrap_or("file"),
            bytes.len().max(1),
        );
        writable.load_bytes(bytes);
        writable.notify_action_observers(&UserActionEvent::new(
            UserActionType::FileOpened,
            0,
            None,
            1,
            file_id as u32,
        ));
        let opened = OpenedFile {
            pointer: WritableFileHandle {
                file_id,
                os_handle,
                path_offset,
                path_len,
            },
            writable,
        };
        self.files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(file_id, opened);
        let _ = self.maybe_reclaim_memory();
        file_id
    }

    /// Registers an observer on every currently opened writable.
    pub fn bind_observer(&self, observer: Arc<dyn UserActionObserver>) {
        let files = self
            .files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for file in files.values() {
            file.writable
                .register_action_observer(Arc::clone(&observer));
        }
    }

    /// Registers an observer on a single opened writable.
    pub fn bind_observer_for(
        &self,
        file_id: u64,
        observer: Arc<dyn UserActionObserver>,
    ) -> bool {
        let files = self
            .files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(file) = files.get(&file_id) {
            file.writable.register_action_observer(observer);
            true
        } else {
            false
        }
    }

    /// Closes a file and emits `FileClosed`.
    pub fn close(&self, file_id: u64) -> bool {
        let mut files = self
            .files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(file) = files.remove(&file_id) {
            file.writable.notify_action_observers(&UserActionEvent::new(
                UserActionType::FileClosed,
                0,
                None,
                1,
                file_id as u32,
            ));
            drop(files);
            let _ = self.maybe_reclaim_memory();
            true
        } else {
            false
        }
    }

    /// Copies the interned path for `file_id` into `out`.
    pub fn path_bytes(&self, file_id: u64, out: &mut Vec<u8>) -> bool {
        let files = self
            .files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(file) = files.get(&file_id) else {
            return false;
        };
        let arena = self
            .path_arena
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let start = file.pointer.path_offset as usize;
        let end = start.saturating_add(file.pointer.path_len as usize);
        if end > arena.len() {
            return false;
        }
        out.clear();
        out.extend_from_slice(&arena[start..end]);
        true
    }

    /// Returns disk pointers for every opened file.
    pub fn pointers(&self) -> Vec<WritableFileHandle> {
        self.files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .map(|f| f.pointer)
            .collect()
    }

    /// Snapshot of raw bytes for packing (eager writable contents).
    pub fn snapshot_bytes(&self, file_id: u64) -> Option<Vec<u8>> {
        let files = self
            .files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let file = files.get(&file_id)?;
        let len = file.writable.raw_length();
        let mut out = vec![0u8; len];
        unsafe {
            core::ptr::copy_nonoverlapping(
                file.writable.raw_ptr(),
                out.as_mut_ptr(),
                len,
            );
        }
        Some(out)
    }

    /// Number of opened files.
    pub fn len(&self) -> usize {
        self.files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Whether the table has no opened files.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Aggregates opened-file, writable, and history memory totals.
    #[must_use]
    pub fn memory_stats(&self) -> EditorMemoryStats {
        let files = self
            .files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut writable_bytes = 0u64;
        let mut history_bytes = 0u64;
        let mut construct_bytes = 0u64;
        for file in files.values() {
            writable_bytes =
                writable_bytes.saturating_add(file.writable.raw_length() as u64);
            // SAFETY: the table lock serializes access; history is only mutated
            // through this table or exclusive writable borrows.
            let history = unsafe { &*file.writable.history().get() };
            history_bytes =
                history_bytes.saturating_add(history.estimated_resident_bytes() as u64);
            construct_bytes =
                construct_bytes.saturating_add(history.construct().len() as u64);
        }
        EditorMemoryStats {
            opened_files: files.len(),
            writable_bytes,
            history_bytes,
            construct_bytes,
        }
    }

    /// Compresses cold TXUs on every opened file under `pressure`.
    ///
    /// Returns the number of TXUs newly compressed across all files.
    pub fn compress_histories(&self, pressure: MemoryPressure) -> usize {
        let files = self
            .files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut compressed = 0usize;
        for file in files.values() {
            // SAFETY: table lock held; no concurrent history mutation.
            let history = unsafe { &mut *file.writable.history().get() };
            compressed = compressed
                .saturating_add(history.maybe_compress_under_pressure(pressure));
        }
        compressed
    }

    /// Returns a raw pointer to the history cell for `file_id`, if present.
    ///
    /// # Safety
    ///
    /// Callers must ensure exclusive access while the returned pointer is used
    /// and must not hold the internal files lock across conflicting borrows.
    pub fn history_mut(&self, file_id: u64) -> Option<*mut History> {
        let files = self
            .files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let file = files.get(&file_id)?;
        Some(file.writable.history().get())
    }

    /// Rate-limited reclaim: samples memory and compresses cold TXUs when needed.
    pub fn maybe_reclaim_memory(&self) -> Option<(SystemMemory, MemoryPressure, usize)> {
        let stats = self.memory_stats();
        let (system, pressure) = file_table_monitor().maybe_sample(&stats)?;
        let compressed = self.compress_histories(pressure);
        crate::debug!(
            "OpenedFileTable reclaim: pressure={:?} files={} writable={} history={} compressed={}",
            pressure,
            stats.opened_files,
            stats.writable_bytes,
            stats.history_bytes,
            compressed
        );
        Some((system, pressure, compressed))
    }

    /// Immediate reclaim pass (ignores the monitor cooldown).
    pub fn reclaim_memory_now(&self) -> (SystemMemory, MemoryPressure, usize) {
        let stats = self.memory_stats();
        let (system, pressure) = file_table_monitor().sample_now(&stats);
        let compressed = self.compress_histories(pressure);
        (system, pressure, compressed)
    }
}

impl Default for OpenedFileTable {
    fn default() -> Self {
        Self::new()
    }
}
