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

//! UserView data structure for complete user view representation.
//!
//! This module provides the central UserView data structure that aggregates
//! all components of a user's collaborative editing view:
//! - File system state
//! - Writable buffer data
//! - User actions (compressed)
//! - Transaction units (TXUs)
//! - Cursor positions
//! - User metadata

use super::view_action_cache::{ActionCache, PackedAction};
use super::view_compression::{LzmaParams, LzmaResult};
use super::view_packing_decoder::BytecodeDecoder;
use super::view_packing_encoder::{BytecodeEncoder, SectionType};
use std::time::{SystemTime, UNIX_EPOCH};

/// UserView represents the complete state of a user's collaborative editing view.
///
/// This structure aggregates all components needed to represent and synchronize
/// a user's view in a collaborative editing environment.
#[derive(Debug, Clone)]
pub struct UserView {
    /// User identifier
    user_id: u64,
    /// Session identifier
    session_id: u64,
    /// File system data (serialized)
    filesystem_data: Option<Vec<u8>>,
    /// Writable buffer data (serialized)
    writable_data: Option<Vec<u8>>,
    /// User actions cache
    action_cache: ActionCache,
    /// Direct packed actions storage (bypasses cache compression)
    direct_actions: Vec<PackedAction>,
    /// Transaction units data (serialized)
    txu_data: Option<Vec<u8>>,
    /// Cursor positions data (serialized)
    cursor_data: Option<Vec<u8>>,
    /// User opened filesystem tracking data
    opened_fs_data: Option<Vec<u8>>,
    /// Creation timestamp
    created_at: u64,
    /// Last modified timestamp
    modified_at: u64,
    /// View version for synchronization
    version: u32,
    /// Compression parameters
    compression_params: LzmaParams,
}

impl UserView {
    /// Magic number for UserView serialization
    const MAGIC: u32 = 0x55534552; // "USER"

    /// Current version of the UserView format
    const VERSION: u16 = 1;

    /// Creates a new UserView with default settings.
    ///
    /// # Arguments
    ///
    /// * `user_id` - User identifier
    /// * `session_id` - Session identifier
    ///
    /// # Returns
    ///
    /// A new UserView instance.
    pub fn new(user_id: u64, session_id: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            user_id,
            session_id,
            filesystem_data: None,
            writable_data: None,
            action_cache: ActionCache::new(),
            direct_actions: Vec::new(),
            txu_data: None,
            cursor_data: None,
            opened_fs_data: None,
            created_at: now,
            modified_at: now,
            version: 1,
            compression_params: LzmaParams::default(),
        }
    }

    /// Creates a new UserView with custom compression parameters.
    ///
    /// # Arguments
    ///
    /// * `user_id` - User identifier
    /// * `session_id` - Session identifier
    /// * `params` - Compression parameters
    ///
    /// # Returns
    ///
    /// A new UserView instance with custom compression.
    pub fn with_compression(user_id: u64, session_id: u64, params: LzmaParams) -> Self {
        let mut view = Self::new(user_id, session_id);
        view.compression_params = params;
        view
    }

    /// Serializes the UserView to bytecode.
    ///
    /// # Returns
    ///
    /// Result containing the bytecode or an error.
    pub fn to_bytecode(&mut self) -> LzmaResult<Vec<u8>> {
        let mut encoder =
            BytecodeEncoder::with_compression(self.compression_params.clone());

        // Encode filesystem data if present
        if let Some(ref data) = self.filesystem_data {
            encoder.encode_filesystem(data)?;
        }

        // Encode writable data if present
        if let Some(ref data) = self.writable_data {
            encoder.encode_writable(data)?;
        }
        let actions = self.action_cache.flush();
        if !actions.is_empty() {
            encoder.encode_actions(&actions)?;
        }
        if !self.direct_actions.is_empty() {
            encoder.encode_actions(&self.direct_actions)?;
        }
        if let Some(ref data) = self.txu_data {
            encoder.encode_txus(data)?;
        }
        if let Some(ref data) = self.cursor_data {
            encoder.encode_cursors(data)?;
        }
        encoder.finalize()
    }

    /// Deserializes a UserView from bytecode.
    ///
    /// # Arguments
    ///
    /// * `bytecode` - Bytecode data to deserialize
    /// * `user_id` - User identifier for the view
    /// * `session_id` - Session identifier for the view
    ///
    /// # Returns
    ///
    /// Result containing the UserView or an error.
    pub fn from_bytecode(
        bytecode: &[u8],
        user_id: u64,
        session_id: u64,
    ) -> LzmaResult<Self> {
        let mut decoder = BytecodeDecoder::new();
        decoder.load(bytecode)?;

        let mut view = Self::new(user_id, session_id);
        if decoder.has_section(SectionType::SystemFs) {
            view.filesystem_data = Some(decoder.decode_filesystem()?);
        }
        if decoder.has_section(SectionType::SystemWritable) {
            view.writable_data = Some(decoder.decode_writable()?);
        }
        if decoder.has_section(SectionType::SystemActions) {
            let actions = decoder.decode_actions()?;
            view.direct_actions = actions;
        }
        if decoder.has_section(SectionType::SystemTransactionUnits) {
            view.txu_data = Some(decoder.decode_txus()?);
        }
        if decoder.has_section(SectionType::SystemCursor) {
            view.cursor_data = Some(decoder.decode_cursors()?);
        }
        view.modified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Ok(view)
    }

    /// Sets the file system data.
    ///
    /// # Arguments
    ///
    /// * `data` - File system data
    pub fn set_filesystem(&mut self, data: Vec<u8>) {
        self.filesystem_data = Some(data);
        self.touch();
    }

    /// Sets the writable buffer data.
    ///
    /// # Arguments
    ///
    /// * `data` - Writable buffer data
    pub fn set_writable(&mut self, data: Vec<u8>) {
        self.writable_data = Some(data);
        self.touch();
    }

    /// Sets the transaction units data.
    ///
    /// # Arguments
    ///
    /// * `data` - TXU data
    pub fn set_txus(&mut self, data: Vec<u8>) {
        self.txu_data = Some(data);
        self.touch();
    }

    /// Sets the cursor positions data.
    ///
    /// # Arguments
    ///
    /// * `data` - Cursor position data
    pub fn set_cursors(&mut self, data: Vec<u8>) {
        self.cursor_data = Some(data);
        self.touch();
    }

    /// Sets the opened filesystem tracking data.
    ///
    /// # Arguments
    ///
    /// * `data` - Opened filesystem data
    pub fn set_opened_fs(&mut self, data: Vec<u8>) {
        self.opened_fs_data = Some(data);
        self.touch();
    }

    /// Adds a packed action to the direct actions storage.
    ///
    /// # Arguments
    ///
    /// * `action` - Packed action to add
    pub fn add_packed_action(&mut self, action: PackedAction) {
        self.touch();
        self.direct_actions.push(action);
    }

    /// Flushes all cached actions and returns them.
    ///
    /// # Returns
    ///
    /// Vector of all packed actions from the cache.
    pub fn flush_actions(&mut self) -> Vec<PackedAction> {
        self.touch();
        self.action_cache.flush()
    }

    /// Returns all direct actions.
    ///
    /// # Returns
    ///
    /// Vector of all direct packed actions.
    pub fn direct_actions(&self) -> &[PackedAction] {
        &self.direct_actions
    }

    /// Clears all direct actions.
    pub fn clear_direct_actions(&mut self) {
        self.touch();
        self.direct_actions.clear();
    }
    /// Updates the modified timestamp and increments version.
    fn touch(&mut self) {
        self.modified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.version = self.version.wrapping_add(1);
    }
    /// Returns the user ID.
    pub fn user_id(&self) -> u64 {
        self.user_id
    }

    /// Returns the session ID.
    pub fn session_id(&self) -> u64 {
        self.session_id
    }

    /// Returns the creation timestamp.
    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    /// Returns the last modified timestamp.
    pub fn modified_at(&self) -> u64 {
        self.modified_at
    }

    /// Returns the view version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the number of cached actions.
    pub fn action_count(&self) -> usize {
        self.action_cache.len()
    }

    /// Returns the number of direct actions.
    pub fn direct_action_count(&self) -> usize {
        self.direct_actions.len()
    }

    /// Returns a reference to the action cache.
    pub fn action_cache(&self) -> &ActionCache {
        &self.action_cache
    }

    /// Returns the Writable Data.
    pub fn writable_data(&self) -> &Option<Vec<u8>> {
        &self.writable_data
    }

    /// Returns a mutable reference to the action cache.
    pub fn action_cache_mut(&mut self) -> &mut ActionCache {
        self.touch();
        &mut self.action_cache
    }
}

impl Default for UserView {
    fn default() -> Self {
        Self::new(0, 0)
    }
}
