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

//! Bytecode packing decoder for UserView data structures.
//!
//! This module provides the decoder for unpacking UserView bytecode
//! back into its component parts (File System, Writable, Actions, TXUs).

use super::view_action_cache::PackedAction;
use super::view_compression::{LzmaDecompressor, LzmaError, LzmaParams, LzmaResult};
use super::view_packing_encoder::{BytecodeHeader, SectionHeader, SectionType};
use std::collections::HashMap;

/// Bytecode decoder for unpacking UserView data.
pub struct BytecodeDecoder {
    /// Compression parameters for decompression
    compression_params: LzmaParams,
    /// Parsed header
    header: Option<BytecodeHeader>,
    /// Parsed section headers
    section_headers: Vec<SectionHeader>,
    /// Section data offsets
    section_offsets: HashMap<SectionType, usize>,
    /// Raw bytecode data
    bytecode: Vec<u8>,
    /// Current position in bytecode
    position: usize,
}

impl BytecodeDecoder {
    /// Creates a new bytecode decoder with default settings.
    pub fn new() -> Self {
        Self {
            compression_params: LzmaParams::default(),
            header: None,
            section_headers: Vec::new(),
            section_offsets: HashMap::new(),
            bytecode: Vec::new(),
            position: 0,
        }
    }

    /// Creates a new bytecode decoder with custom compression parameters.
    ///
    /// # Arguments
    ///
    /// * `params` - Compression parameters
    pub fn with_compression(params: LzmaParams) -> Self {
        Self {
            compression_params: params,
            header: None,
            section_headers: Vec::new(),
            section_offsets: HashMap::new(),
            bytecode: Vec::new(),
            position: 0,
        }
    }

    /// Loads and parses bytecode data.
    ///
    /// # Arguments
    ///
    /// * `bytecode` - Bytecode data to decode
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn load(&mut self, bytecode: &[u8]) -> LzmaResult<()> {
        self.bytecode = bytecode.to_vec();
        self.position = 0;

        // Parse header
        self.header = Some(self.read_header()?);

        // Parse section headers
        let section_count = self.header.as_ref().unwrap().section_count as usize;
        self.section_headers = Vec::with_capacity(section_count);

        for _ in 0..section_count {
            let section_header = self.read_section_header()?;
            self.section_headers.push(section_header);
        }

        // Calculate section data offsets
        let mut data_offset = self.position;
        for section_header in &self.section_headers {
            self.section_offsets
                .insert(section_header.section_type, data_offset);
            data_offset += section_header.compressed_size as usize;
        }

        Ok(())
    }

    /// Reads the main header from bytecode.
    ///
    /// # Returns
    ///
    /// Parsed header or error
    fn read_header(&mut self) -> LzmaResult<BytecodeHeader> {
        if self.position + 32 > self.bytecode.len() {
            return Err(LzmaError::CorruptedData);
        }

        let magic = u32::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
            self.bytecode[self.position + 2],
            self.bytecode[self.position + 3],
        ]);
        self.position += 4;

        if magic != 0x43564356 {
            return Err(LzmaError::CorruptedData);
        }

        let version = u16::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
        ]);
        self.position += 2;

        let flags = u16::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
        ]);
        self.position += 2;

        let uncompressed_size = u64::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
            self.bytecode[self.position + 2],
            self.bytecode[self.position + 3],
            self.bytecode[self.position + 4],
            self.bytecode[self.position + 5],
            self.bytecode[self.position + 6],
            self.bytecode[self.position + 7],
        ]);
        self.position += 8;

        let compressed_size = u64::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
            self.bytecode[self.position + 2],
            self.bytecode[self.position + 3],
            self.bytecode[self.position + 4],
            self.bytecode[self.position + 5],
            self.bytecode[self.position + 6],
            self.bytecode[self.position + 7],
        ]);
        self.position += 8;

        let section_count = u32::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
            self.bytecode[self.position + 2],
            self.bytecode[self.position + 3],
        ]);
        self.position += 4;

        let mut reserved = [0u8; 16];
        reserved.copy_from_slice(&self.bytecode[self.position..self.position + 16]);
        self.position += 16;

        Ok(BytecodeHeader {
            magic,
            version,
            flags,
            uncompressed_size,
            compressed_size,
            section_count,
            reserved,
        })
    }

    /// Reads a section header from bytecode.
    ///
    /// # Returns
    ///
    /// Parsed section header or error
    fn read_section_header(&mut self) -> LzmaResult<SectionHeader> {
        if self.position + 12 > self.bytecode.len() {
            return Err(LzmaError::CorruptedData);
        }

        let section_type_byte = self.bytecode[self.position];
        let section_type = match section_type_byte {
            0 => SectionType::SystemFs,
            1 => SectionType::SystemWritable,
            2 => SectionType::SystemActions,
            3 => SectionType::SystemTransactionUnits,
            4 => SectionType::SystemCursor,
            255 => SectionType::EndMarker,
            _ => return Err(LzmaError::CorruptedData),
        };
        self.position += 1;

        let uncompressed_size = u32::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
            self.bytecode[self.position + 2],
            self.bytecode[self.position + 3],
        ]);
        self.position += 4;

        let compressed_size = u32::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
            self.bytecode[self.position + 2],
            self.bytecode[self.position + 3],
        ]);
        self.position += 4;

        let flags = u16::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
        ]);
        self.position += 2;

        let reserved = u16::from_le_bytes([
            self.bytecode[self.position],
            self.bytecode[self.position + 1],
        ]);
        self.position += 2;

        Ok(SectionHeader {
            section_type,
            uncompressed_size,
            compressed_size,
            flags,
            reserved,
        })
    }

    /// Decodes a specific section from the bytecode.
    ///
    /// # Arguments
    ///
    /// * `section_type` - Type of section to decode
    ///
    /// # Returns
    ///
    /// Decompressed section data or error
    pub fn decode_section(&self, section_type: SectionType) -> LzmaResult<Vec<u8>> {
        let offset = *self
            .section_offsets
            .get(&section_type)
            .ok_or(LzmaError::CorruptedData)?;

        let section_header = self
            .section_headers
            .iter()
            .find(|h| h.section_type == section_type)
            .ok_or(LzmaError::CorruptedData)?;

        let compressed_data =
            &self.bytecode[offset..offset + section_header.compressed_size as usize];
        let mut decompressed = vec![0u8; section_header.uncompressed_size as usize];

        let mut decompressor =
            LzmaDecompressor::with_params(self.compression_params.clone());
        let decompressed_size =
            decompressor.decompress(compressed_data, &mut decompressed)?;

        decompressed.truncate(decompressed_size);
        Ok(decompressed)
    }

    /// Decodes actions from the bytecode.
    ///
    /// # Returns
    ///
    /// Vector of packed actions or error
    pub fn decode_actions(&self) -> LzmaResult<Vec<PackedAction>> {
        let action_data = self.decode_section(SectionType::SystemActions)?;

        if action_data.len() & 7 != 0 {
            return Err(LzmaError::CorruptedData);
        }

        let mut actions = Vec::with_capacity(action_data.len() / 8);
        for chunk in action_data.chunks(8) {
            let value = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                chunk[7],
            ]);
            actions.push(PackedAction::from_u64(value));
        }

        Ok(actions)
    }

    /// Decodes file system data from the bytecode.
    ///
    /// # Returns
    ///
    /// File system data or error
    pub fn decode_filesystem(&self) -> LzmaResult<Vec<u8>> {
        self.decode_section(SectionType::SystemFs)
    }

    /// Decodes writable data from the bytecode.
    ///
    /// # Returns
    ///
    /// Writable data or error
    pub fn decode_writable(&self) -> LzmaResult<Vec<u8>> {
        self.decode_section(SectionType::SystemWritable)
    }

    /// Decodes TXU data from the bytecode.
    ///
    /// # Returns
    ///
    /// TXU data or error
    pub fn decode_txus(&self) -> LzmaResult<Vec<u8>> {
        self.decode_section(SectionType::SystemTransactionUnits)
    }

    /// Decodes cursor data from the bytecode.
    ///
    /// # Returns
    ///
    /// Cursor data or error
    pub fn decode_cursors(&self) -> LzmaResult<Vec<u8>> {
        self.decode_section(SectionType::SystemCursor)
    }

    /// Returns the parsed header if available.
    pub fn header(&self) -> Option<&BytecodeHeader> {
        self.header.as_ref()
    }

    /// Returns the section headers.
    pub fn section_headers(&self) -> &[SectionHeader] {
        &self.section_headers
    }

    /// Checks if a specific section type is present.
    pub fn has_section(&self, section_type: SectionType) -> bool {
        self.section_offsets.contains_key(&section_type)
    }
}

impl Default for BytecodeDecoder {
    fn default() -> Self {
        Self::new()
    }
}
