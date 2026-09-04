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

//! UserView builder for constructing complete user view bytecodes.
//!
//! This module provides a high-level builder for constructing complete
//! UserView bytecodes from various components without coupling to specific modules.

use super::view_action_cache::PackedAction;
use super::view_compression::{LzmaParams, LzmaResult};
use super::view_packing_encoder::{BytecodeEncoder, SectionType};

/// UserView builder for constructing complete user view representations.
pub struct UserViewBuilder {
    /// The underlying bytecode encoder
    encoder: BytecodeEncoder,
    /// Collected actions
    actions: Vec<PackedAction>,
    /// File system data
    filesystem_data: Option<Vec<u8>>,
    /// Writable data
    writable_data: Option<Vec<u8>>,
    /// TXU data
    txu_data: Option<Vec<u8>>,
    /// Cursor data
    cursor_data: Option<Vec<u8>>,
}

impl UserViewBuilder {
    /// Creates a new UserView builder with default settings.
    pub fn new() -> Self {
        Self {
            encoder: BytecodeEncoder::new(),
            actions: Vec::new(),
            filesystem_data: None,
            writable_data: None,
            txu_data: None,
            cursor_data: None,
        }
    }

    /// Creates a new UserView builder with custom compression parameters.
    ///
    /// # Arguments
    ///
    /// * `params` - Compression parameters
    pub fn with_compression(params: LzmaParams) -> Self {
        Self {
            encoder: BytecodeEncoder::with_compression(params),
            actions: Vec::new(),
            filesystem_data: None,
            writable_data: None,
            txu_data: None,
            cursor_data: None,
        }
    }

    /// Adds user actions to the view.
    ///
    /// # Arguments
    ///
    /// * `actions` - Slice of packed actions
    pub fn add_actions(&mut self, actions: &[PackedAction]) -> &mut Self {
        self.actions.extend_from_slice(actions);
        self
    }

    /// Sets the file system data.
    ///
    /// # Arguments
    ///
    /// * `data` - File system data
    pub fn set_filesystem(&mut self, data: Vec<u8>) -> &mut Self {
        self.filesystem_data = Some(data);
        self
    }

    /// Sets the writable data.
    ///
    /// # Arguments
    ///
    /// * `data` - Writable buffer data
    pub fn set_writable(&mut self, data: Vec<u8>) -> &mut Self {
        self.writable_data = Some(data);
        self
    }

    /// Sets the TXU data.
    ///
    /// # Arguments
    ///
    /// * `data` - Transaction unit data
    pub fn set_txus(&mut self, data: Vec<u8>) -> &mut Self {
        self.txu_data = Some(data);
        self
    }

    /// Sets the cursor data.
    ///
    /// # Arguments
    ///
    /// * `data` - Cursor position data
    pub fn set_cursors(&mut self, data: Vec<u8>) -> &mut Self {
        self.cursor_data = Some(data);
        self
    }

    /// Builds the complete UserView bytecode.
    ///
    /// # Returns
    ///
    /// Result containing the complete bytecode or an error
    pub fn build(mut self) -> LzmaResult<Vec<u8>> {
        // Encode actions if present
        if !self.actions.is_empty() {
            self.encoder.encode_actions(&self.actions)?;
        }

        // Encode filesystem data if present
        if let Some(ref data) = self.filesystem_data {
            self.encoder.encode_filesystem(data)?;
        }

        // Encode writable data if present
        if let Some(ref data) = self.writable_data {
            self.encoder.encode_writable(data)?;
        }

        // Encode TXU data if present
        if let Some(ref data) = self.txu_data {
            self.encoder.encode_txus(data)?;
        }

        // Encode cursor data if present
        if let Some(ref data) = self.cursor_data {
            self.encoder.encode_cursors(data)?;
        }

        // Finalize and return bytecode
        self.encoder.finalize()
    }

    /// Returns the number of actions currently added.
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Clears all data from the builder.
    pub fn clear(&mut self) {
        self.actions.clear();
        self.filesystem_data = None;
        self.writable_data = None;
        self.txu_data = None;
        self.cursor_data = None;
    }
}

impl Default for UserViewBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Token types for the bytecode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// Header token
    Header { magic: u32, version: u16 },
    /// Section header token
    SectionHeader {
        section_type: SectionType,
        size: u32,
    },
    /// Section data token
    SectionData {
        section_type: SectionType,
        data: Vec<u8>,
    },
    /// End marker token
    EndMarker,
    /// Error token
    Error(String),
}

/// UserView tokenizer for breaking down bytecode into tokens.
pub struct UserViewTokenizer {
    /// Bytecode data
    bytecode: Vec<u8>,
    /// Current position
    position: usize,
}

impl UserViewTokenizer {
    /// Creates a new tokenizer from bytecode.
    ///
    /// # Arguments
    ///
    /// * `bytecode` - Bytecode data to tokenize
    pub fn new(bytecode: Vec<u8>) -> Self {
        Self {
            bytecode,
            position: 0,
        }
    }

    /// Returns the next token from the bytecode.
    ///
    /// # Returns
    ///
    /// The next token or None if end of bytecode
    pub fn next_token(&mut self) -> Option<Token> {
        if self.position >= self.bytecode.len() {
            return None;
        }
        if self.position == 0 {
            return self.read_header_token();
        }
        // Check for end marker
        if self.bytecode[self.position] == 255 {
            self.position += 1;
            return Some(Token::EndMarker);
        }
        // Try to read section header
        if self.read_section_header_token().is_some() {
            // Read section data next
            return self.read_section_data_token();
        }

        None
    }

    /// Reads the header token.
    fn read_header_token(&mut self) -> Option<Token> {
        if self.position + 4 > self.bytecode.len() {
            return Some(Token::Error("Insufficient data for header".to_string()));
        }
        let magic = u32::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
            self.bytecode[self.position + 2],
            self.bytecode[self.position + 3],
        ]);
        self.position += 4;

        if self.position + 2 > self.bytecode.len() {
            return Some(Token::Error("Insufficient data for version".to_string()));
        }

        let version = u16::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
        ]);
        self.position += 2;

        // Skip to end of header (32 bytes total)
        self.position = 32;

        Some(Token::Header { magic, version })
    }

    /// Reads a section header token.
    fn read_section_header_token(&mut self) -> Option<Token> {
        if self.position + 12 > self.bytecode.len() {
            return None;
        }

        let section_type_byte = self.bytecode[self.position];
        let section_type = match section_type_byte {
            0 => SectionType::SystemFs,
            1 => SectionType::SystemWritable,
            2 => SectionType::SystemActions,
            3 => SectionType::SystemTransactionUnits,
            4 => SectionType::SystemCursor,
            _ => return None,
        };
        self.position += 1;
        let size = u32::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
            self.bytecode[self.position + 2],
            self.bytecode[self.position + 3],
        ]);
        self.position += 4;

        // Skip rest of section header
        self.position += 7;

        Some(Token::SectionHeader { section_type, size })
    }

    /// Reads section data token.
    fn read_section_data_token(&mut self) -> Option<Token> {
        // Need to go back to get the section type and size
        let current_pos = self.position;
        self.position -= 12; // Go back to section header start

        let section_type_byte = self.bytecode[self.position];
        let section_type = match section_type_byte {
            0 => SectionType::SystemFs,
            1 => SectionType::SystemWritable,
            2 => SectionType::SystemActions,
            3 => SectionType::SystemTransactionUnits,
            4 => SectionType::SystemCursor,
            _ => return None,
        };
        self.position += 1;

        let compressed_size = u32::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
            self.bytecode[self.position + 2],
            self.bytecode[self.position + 3],
        ]);
        self.position += 4;

        // Skip rest of section header
        self.position += 7;
        if self.position + compressed_size as usize > self.bytecode.len() {
            return Some(Token::Error("Insufficient data for section".to_string()));
        }
        let data = self.bytecode[self.position..self.position + compressed_size as usize]
            .to_vec();
        self.position += compressed_size as usize;

        Some(Token::SectionData { section_type, data })
    }

    /// Resets the tokenizer to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Returns the current position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns the total bytecode length.
    pub fn len(&self) -> usize {
        self.bytecode.len()
    }

    /// Returns true if at end of bytecode.
    pub fn is_finished(&self) -> bool {
        self.position >= self.bytecode.len()
    }
}
