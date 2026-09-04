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

use crate::edit::wredit_history_txu::HistoryTXU;
use sha2::{Digest, Sha256};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

/// Maximum size for the history buffer
pub const MAX_SIZE: usize = 256;

/// Serializable bytecode for history structs
pub trait HistorySerializable {
    /// Returns the serialized history struct or TXU struct
    /// Into structured bytecode for storage the history.
    fn to_bytecode(&self) -> BytecodeResult<Vec<u8>>;
}

/// Deserializable bytecode for history structs
pub trait HistoryDeserializable: Sized {
    /// Creates a history struct or TXU struct from bytecode.
    ///
    /// # Arguments
    ///
    /// * `bytecode` - The bytecode to deserialize
    ///
    /// # Returns
    ///
    /// A BytecodeResult containing the deserialized struct or an error.
    fn from_bytecode(bytecode: &[u8]) -> BytecodeResult<Self>;
}

/// Errors during history or TXU serialization due corrupted or invalid data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BytecodeError {
    /// Invalid encoding value
    InvalidEncoding,
    /// Lock acquisition failed (poisoned lock)
    LockError,
    /// Buffer overflow during serialization
    BufferOverflow,
    /// Invalid delta data structure
    InvalidDelta,
    /// Hash mismatch during verification
    HashMismatch,
    /// Invalid timestamp
    InvalidTimestamp,
}

/// Result type for serialization operations
pub type BytecodeResult<T> = Result<T, BytecodeError>;

/// History tracking for writable data structures.
///
/// This struct maintains a record of all changes made to a writable buffer,
/// enabling undo/redo functionality and change tracking. It stores text units
/// (TXUs) that represent individual changes, along with metadata like timestamps
/// and hashes for integrity verification.
pub struct History {
    /// The construct buffer for storing change data
    construct: Vec<u8>,
    /// The timestamp of the last change
    timestamp: u64,
    /// The hash for integrity verification
    hash: Box<[u8]>,
    /// The current index in the history timeline
    current_index: usize,
    /// The text units (TXUs) representing individual changes
    txus: RwLock<Arc<Box<[HistoryTXU]>>>,
    /// Whether an edit group is currently active
    edit_group_active: bool,
    /// Pending changes for the current edit group
    pending_changes: Vec<HistoryTXU>,
    /// The index where redo history starts (for truncating after new edits)
    redo_start_index: usize,
}

impl History {
    /// Generates a timestamp for the current history state.
    ///
    /// # Returns
    ///
    /// The current time in seconds since UNIX epoch, or 0 if time retrieval fails.
    pub fn gen_history_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs()
    }
    /// Generates a hash for the current history state.
    ///
    /// # Returns
    ///
    /// A SHA-256 hash of the current history state.
    pub fn gen_history_hash(&self) -> Box<[u8]> {
        let mut hasher = Sha256::new();

        // Hash some fields to get a unique hash
        let hasher_mut = &mut hasher;
        Digest::update(hasher_mut, &self.construct.as_slice());
        Digest::update(hasher_mut, self.timestamp.to_le_bytes());
        Digest::update(hasher_mut, self.current_index.to_le_bytes());

        if let Ok(txus) = self.txus.read() {
            for txu in txus.iter() {
                Digest::update(hasher_mut, &txu.timestamp().to_le_bytes());
                Digest::update(hasher_mut, txu.buffer());
                Digest::update(hasher_mut, txu.hash());
            }
        }
        hasher.finalize().to_vec().into_boxed_slice()
    }

    /// Creates a new history instance.
    ///
    /// # Returns
    ///
    /// A new History instance with initial timestamp and hash.
    pub fn new() -> Self {
        let history = Self {
            construct: Vec::new(),
            timestamp: 0,
            hash: Box::new([0u8; 32]),
            current_index: 0,
            txus: RwLock::new(Arc::new(Box::new([]))),
            edit_group_active: false,
            pending_changes: Vec::new(),
            redo_start_index: 0,
        };
        // Generate initial hash
        let hash = history.gen_history_hash();
        let timestamp = history.gen_history_timestamp();
        Self {
            hash,
            timestamp,
            ..history
        }
    }

    /// Returns the current history index.
    ///
    /// # Returns
    ///
    /// The current position in the history timeline.
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Sets the current history index.
    ///
    /// # Arguments
    ///
    /// * `index` - The new history index.
    pub fn set_current_index(&mut self, index: usize) {
        self.current_index = index;
    }

    /// Sets the timestamp for the history.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - The new timestamp value.
    pub fn set_timestamp(&mut self, timestamp: u64) {
        self.timestamp = timestamp;
    }

    /// Sets the hash for the history.
    ///
    /// # Arguments
    ///
    /// * `hash` - The new hash value.
    pub fn set_hash(&mut self, hash: Box<[u8]>) {
        self.hash = hash;
    }

    /// Returns a reference to the construct buffer.
    ///
    /// # Returns
    ///
    /// A slice containing the construct data.
    pub fn construct(&self) -> &[u8] {
        &self.construct
    }

    /// Returns a mutable reference to the construct buffer.
    ///
    /// # Returns
    ///
    /// A mutable reference to the construct vector.
    pub fn construct_mut(&mut self) -> &mut Vec<u8> {
        &mut self.construct
    }

    /// Returns a reference to the text units (TXUs).
    ///
    /// # Returns
    ///
    /// A reference to the RwLock containing the TXUs.
    pub fn txus(&self) -> &RwLock<Arc<Box<[HistoryTXU]>>> {
        &self.txus
    }

    /// Returns a mutable reference to the text units (TXUs).
    ///
    /// # Returns
    ///
    /// A mutable reference to the RwLock containing the TXUs.
    pub fn txus_mut(&mut self) -> &mut RwLock<Arc<Box<[HistoryTXU]>>> {
        &mut self.txus
    }

    /// Returns whether an edit group is currently active.
    ///
    /// # Returns
    ///
    /// True if an edit group is active, false otherwise.
    pub fn is_edit_group_active(&self) -> bool {
        self.edit_group_active
    }

    /// Sets the edit group active state.
    ///
    /// # Arguments
    ///
    /// * `active` - The new active state.
    pub fn set_edit_group_active(&mut self, active: bool) {
        self.edit_group_active = active;
    }

    /// Returns a reference to the pending changes.
    ///
    /// # Returns
    ///
    /// A slice of pending TXUs for the current edit group.
    pub fn pending_changes(&self) -> &[HistoryTXU] {
        &self.pending_changes
    }

    /// Returns a mutable reference to the pending changes.
    ///
    /// # Returns
    ///
    /// A mutable reference to the pending changes vector.
    pub fn pending_changes_mut(&mut self) -> &mut Vec<HistoryTXU> {
        &mut self.pending_changes
    }

    /// Clears the pending changes.
    pub fn clear_pending_changes(&mut self) {
        self.pending_changes.clear();
    }

    /// Returns the redo start index.
    ///
    /// # Returns
    ///
    /// The index where redo history starts.
    pub fn redo_start_index(&self) -> usize {
        self.redo_start_index
    }

    /// Sets the redo start index.
    ///
    /// # Arguments
    ///
    /// * `index` - The new redo start index.
    pub fn set_redo_start_index(&mut self, index: usize) {
        self.redo_start_index = index;
    }

    /// Adds a TXU to the history.
    ///
    /// # Arguments
    ///
    /// * `txu` - The TXU to add.
    pub fn add_txu(&mut self, txu: HistoryTXU) {
        if let Ok(mut txus) = self.txus.write() {
            let mut vec: Vec<HistoryTXU> = (*txus).to_vec();
            vec.push(txu);
            let new_index = vec.len();
            *txus = Arc::new(vec.into_boxed_slice());
            self.current_index = new_index;
        }
        self.update_hash_and_timestamp();
        // Reclaim memory when history grows large (no-op under Normal + small totals).
        let _ = self.maybe_compress_under_pressure(
            crate::base::base_memory::MemoryPressure::Normal,
        );
    }

    /// Updates the hash and timestamp after a change.
    fn update_hash_and_timestamp(&mut self) {
        self.hash = self.gen_history_hash();
        self.timestamp = self.gen_history_timestamp();
    }
}

impl HistorySerializable for History {
    /// Converts the history to bytecode representation.
    ///
    /// # Returns
    ///
    /// A SerializationResult containing the bytecode vector or an error.
    fn to_bytecode(&self) -> BytecodeResult<Vec<u8>> {
        let mut bytes = Vec::<u8>::new();

        let timestamp_size = std::mem::size_of::<u64>();
        bytes.extend_from_slice(&(timestamp_size as u8).to_le_bytes());

        bytes.extend_from_slice(&self.timestamp.to_le_bytes());

        let hash_size = self.hash.len();
        bytes.extend_from_slice(&(hash_size as u8).to_le_bytes());
        bytes.extend_from_slice(&*self.hash);
        bytes.extend_from_slice(&self.current_index.to_le_bytes());

        let construct_len = self.construct.len();
        bytes.extend_from_slice(&(construct_len as u32).to_le_bytes());
        bytes.extend_from_slice(&self.construct);

        let txu_read = self.txus.read().map_err(|_| BytecodeError::LockError)?;
        let txu_count = txu_read.len();
        bytes.extend_from_slice(&(txu_count as u32).to_le_bytes());

        for txu in txu_read.iter() {
            let txu_bytes = txu.to_bytecode()?;
            let txu_len = txu_bytes.len();
            bytes.extend_from_slice(&(txu_len as u32).to_le_bytes());
            bytes.extend_from_slice(&txu_bytes);
        }

        Ok(bytes)
    }
}

impl HistoryDeserializable for History {
    /// Creates a History instance from bytecode with validation.
    ///
    /// # Arguments
    ///
    /// * `bytecode` - The bytecode to deserialize
    ///
    /// # Returns
    ///
    /// A BytecodeResult containing the deserialized History or an error.
    fn from_bytecode(bytecode: &[u8]) -> BytecodeResult<Self> {
        let mut offset = 0;

        // Read timestamp size
        if offset >= bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let timestamp_size = bytecode[offset] as usize;
        offset += 1;

        if timestamp_size != std::mem::size_of::<u64>() {
            return Err(BytecodeError::InvalidTimestamp);
        }

        // Read timestamp
        if offset + timestamp_size > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let timestamp = u64::from_le_bytes(
            bytecode[offset..offset + timestamp_size]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        );
        offset += timestamp_size;

        // Read hash size
        if offset >= bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let hash_size = bytecode[offset] as usize;
        offset += 1;

        // Read hash
        if offset + hash_size > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let hash = bytecode[offset..offset + hash_size]
            .to_vec()
            .into_boxed_slice();
        offset += hash_size;

        // Read current index
        if offset + std::mem::size_of::<usize>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let current_index = usize::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<usize>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        );
        offset += std::mem::size_of::<usize>();

        // Read construct length
        if offset + std::mem::size_of::<u32>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let construct_len = u32::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<u32>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        ) as usize;
        offset += std::mem::size_of::<u32>();

        // Read construct
        if offset + construct_len > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let construct = bytecode[offset..offset + construct_len].to_vec();
        offset += construct_len;

        // Read TXU count
        if offset + std::mem::size_of::<u32>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let txu_count = u32::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<u32>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        ) as usize;
        offset += std::mem::size_of::<u32>();

        // Read TXUs
        let mut txus = Vec::with_capacity(txu_count);
        for _ in 0..txu_count {
            // Read TXU length
            if offset + std::mem::size_of::<u32>() > bytecode.len() {
                return Err(BytecodeError::BufferOverflow);
            }
            let txu_len = u32::from_le_bytes(
                bytecode[offset..offset + std::mem::size_of::<u32>()]
                    .try_into()
                    .map_err(|_| BytecodeError::BufferOverflow)?,
            ) as usize;
            offset += std::mem::size_of::<u32>();

            // Read TXU bytes
            if offset + txu_len > bytecode.len() {
                return Err(BytecodeError::BufferOverflow);
            }
            let txu_bytes = &bytecode[offset..offset + txu_len];
            offset += txu_len;

            let txu = HistoryTXU::from_bytecode(txu_bytes)?;
            txus.push(txu);
        }

        // Create History instance
        let history = Self {
            construct,
            timestamp,
            hash: hash.clone(),
            current_index,
            txus: RwLock::new(Arc::new(txus.into_boxed_slice())),
            edit_group_active: false,
            pending_changes: Vec::new(),
            redo_start_index: 0,
        };

        // Validate hash
        let computed_hash = history.gen_history_hash();
        if computed_hash != hash {
            return Err(BytecodeError::HashMismatch);
        }

        Ok(history)
    }
}
