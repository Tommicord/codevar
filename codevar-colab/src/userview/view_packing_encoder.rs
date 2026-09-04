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

//! Bytecode packing encoder for UserView data structures.
//!
//! This module provides the encoder for packing various UserView components
//! (File System, Writable, Actions, TXUs) into a compressed bytecode format.

use super::view_action_cache::PackedAction;
use super::view_compression::{LzmaCompressor, LzmaParams, LzmaResult};

/// Bytecode section types for different data components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SectionType {
    /// File system metadata and structure
    SystemFs = 0,
    /// Writable buffer data
    SystemWritable = 1,
    /// User actions (compressed)
    SystemActions = 2,
    /// Text units (TXUs)
    SystemTransactionUnits = 3,
    /// Cursor positions
    SystemCursor = 4,
    /// End marker
    EndMarker = 255,
}

/// Bytecode header for the packed data structure.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct BytecodeHeader {
    /// Magic number for validation (0x43564356 = "CVCV")
    pub magic: u32,
    /// Version number
    pub version: u16,
    /// Flags for encoding options
    pub flags: u16,
    /// Total size of uncompressed data
    pub uncompressed_size: u64,
    /// Total size of compressed data
    pub compressed_size: u64,
    /// Number of sections
    pub section_count: u32,
    /// Reserved for future use
    pub reserved: [u8; 16],
}

impl Default for BytecodeHeader {
    fn default() -> Self {
        Self {
            magic: 0x43564356, // "CVCV"
            version: 1,
            flags: 0,
            uncompressed_size: 0,
            compressed_size: 0,
            section_count: 0,
            reserved: [0; 16],
        }
    }
}

/// Section header for individual data sections.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct SectionHeader {
    /// Section type
    pub section_type: SectionType,
    /// Uncompressed size of this section
    pub uncompressed_size: u32,
    /// Compressed size of this section
    pub compressed_size: u32,
    /// Section-specific flags
    pub flags: u16,
    /// Reserved
    pub reserved: u16,
}

impl SectionHeader {
    /// Creates a new section header.
    pub fn new(section_type: SectionType) -> Self {
        Self {
            section_type,
            uncompressed_size: 0,
            compressed_size: 0,
            flags: 0,
            reserved: 0,
        }
    }
}

/// Bytecode encoder for packing UserView data.
pub struct BytecodeEncoder {
    /// Output buffer
    buffer: Vec<u8>,
    /// Compression parameters
    compression_params: LzmaParams,
    /// Current header
    header: BytecodeHeader,
    /// Section headers
    section_headers: Vec<SectionHeader>,
}

impl BytecodeEncoder {
    /// Creates a new bytecode encoder with default settings.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            compression_params: LzmaParams::default(),
            header: BytecodeHeader::default(),
            section_headers: Vec::new(),
        }
    }

    /// Creates a new bytecode encoder with custom compression parameters.
    ///
    /// # Arguments
    ///
    /// * `params` - Compression parameters
    pub fn with_compression(params: LzmaParams) -> Self {
        Self {
            buffer: Vec::new(),
            compression_params: params,
            header: BytecodeHeader::default(),
            section_headers: Vec::new(),
        }
    }

    /// Encodes a section of data.
    ///
    /// # Arguments
    ///
    /// * `section_type` - Type of the section
    /// * `data` - Raw data to encode
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn encode_section(
        &mut self,
        section_type: SectionType,
        data: &[u8],
    ) -> LzmaResult<()> {
        let mut section_header = SectionHeader::new(section_type);
        section_header.uncompressed_size = data.len() as u32;

        let mut compressed = vec![0u8; data.len() * 2]; // Allocate space for compression
        let mut compressor = LzmaCompressor::with_params(self.compression_params.clone());
        let compressed_size = compressor.compress(data, &mut compressed)?;

        section_header.compressed_size = compressed_size as u32;
        compressed.truncate(compressed_size);
        self.section_headers.push(section_header);
        self.buffer.extend_from_slice(&compressed);

        Ok(())
    }

    /// Encodes packed actions as a section.
    ///
    /// # Arguments
    ///
    /// * `actions` - Slice of packed actions
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn encode_actions(&mut self, actions: &[PackedAction]) -> LzmaResult<()> {
        // Convert actions to raw bytes
        let mut action_data = Vec::with_capacity(actions.len() * 8);
        for action in actions {
            action_data.extend_from_slice(&action.as_u64().to_le_bytes());
        }

        self.encode_section(SectionType::SystemActions, &action_data)
    }

    /// Encodes raw file system data as a section.
    ///
    /// # Arguments
    ///
    /// * `fs_data` - File system data
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn encode_filesystem(&mut self, fs_data: &[u8]) -> LzmaResult<()> {
        self.encode_section(SectionType::SystemFs, fs_data)
    }

    /// Encodes writable buffer data as a section.
    ///
    /// # Arguments
    ///
    /// * `writable_data` - Writable buffer data
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn encode_writable(&mut self, writable_data: &[u8]) -> LzmaResult<()> {
        self.encode_section(SectionType::SystemWritable, writable_data)
    }

    /// Encodes TXU data as a section.
    ///
    /// # Arguments
    ///
    /// * `txu_data` - TXU data
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn encode_txus(&mut self, txu_data: &[u8]) -> LzmaResult<()> {
        self.encode_section(SectionType::SystemTransactionUnits, txu_data)
    }

    /// Encodes cursor positions as a section.
    ///
    /// # Arguments
    ///
    /// * `cursor_data` - Cursor position data
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn encode_cursors(&mut self, cursor_data: &[u8]) -> LzmaResult<()> {
        self.encode_section(SectionType::SystemCursor, cursor_data)
    }

    /// Finalizes the encoding and returns the complete bytecode.
    ///
    /// # Returns
    ///
    /// Result containing the complete bytecode or an error
    pub fn finalize(mut self) -> LzmaResult<Vec<u8>> {
        // Update header
        self.header.section_count = self.section_headers.len() as u32;
        self.header.uncompressed_size = self
            .section_headers
            .iter()
            .map(|h| h.uncompressed_size as u64)
            .sum();
        self.header.compressed_size = self.buffer.len() as u64;

        let mut output = Vec::new();
        self.write_header(&mut output, &self.header)?;
        for section_header in &self.section_headers {
            self.write_section_header(&mut output, section_header)?;
        }
        output.extend_from_slice(&self.buffer);
        output.push(SectionType::EndMarker as u8);

        Ok(output)
    }

    /// Writes the main header to the output buffer.
    ///
    /// # Arguments
    ///
    /// * `output` - Output buffer
    /// * `header` - Header to write
    fn write_header(
        &self,
        output: &mut Vec<u8>,
        header: &BytecodeHeader,
    ) -> LzmaResult<()> {
        output.extend_from_slice(&header.magic.to_le_bytes());
        output.extend_from_slice(&header.version.to_le_bytes());
        output.extend_from_slice(&header.flags.to_le_bytes());
        output.extend_from_slice(&header.uncompressed_size.to_le_bytes());
        output.extend_from_slice(&header.compressed_size.to_le_bytes());
        output.extend_from_slice(&header.section_count.to_le_bytes());
        output.extend_from_slice(&header.reserved);
        Ok(())
    }

    /// Writes a section header to the output buffer.
    ///
    /// # Arguments
    ///
    /// * `output` - Output buffer
    /// * `header` - Section header to write
    fn write_section_header(
        &self,
        output: &mut Vec<u8>,
        header: &SectionHeader,
    ) -> LzmaResult<()> {
        output.push(header.section_type as u8);
        output.extend_from_slice(&header.uncompressed_size.to_le_bytes());
        output.extend_from_slice(&header.compressed_size.to_le_bytes());
        output.extend_from_slice(&header.flags.to_le_bytes());
        output.extend_from_slice(&header.reserved.to_le_bytes());
        Ok(())
    }

    /// Returns the current buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the number of sections encoded so far.
    pub fn section_count(&self) -> usize {
        self.section_headers.len()
    }
}

impl Default for BytecodeEncoder {
    fn default() -> Self {
        Self::new()
    }
}
