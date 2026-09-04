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

//! FIFO encoding for efficient serialization of merge operations.
//!
//! This module provides encoding utilities for converting FIFO structures
//! into compact binary representations suitable for network transmission
//! and persistent storage.

use crate::userclient::mergen::mergen_binary_fifo::Fifo;
use std::io::{self, Write};
use std::mem;

pub const FIFO_VERSION: u8 = 1;
/// Magic bytes for format identification
pub const FIFO_MAGIC: &[u8; 4] = b"MGFE";

/// Default buffer size for encoding operations.
const DEFAULT_BUFFER_SIZE: usize = 8192;

/// FIFO encoder for serializing merge operations.
pub struct FifoEncoder {
    /// Buffer for encoding operations
    buffer: Vec<u8>,
    /// Current encoding format version
    version: u8,
}

impl FifoEncoder {
    /// Creates a new FIFO encoder with default buffer size.
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(DEFAULT_BUFFER_SIZE),
            version: FIFO_VERSION,
        }
    }

    /// Creates a new FIFO encoder with specified buffer size.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            version: FIFO_VERSION,
        }
    }

    /// Encodes a FIFO structure into binary format.
    pub fn encode_fifo(&mut self, fifo: &Fifo) -> io::Result<&[u8]> {
        self.buffer.clear();
        self.write_header()?;
        self.write_fifo_data(fifo)?;
        self.write_footer()?;
        Ok(&self.buffer)
    }

    /// Writes the format header.
    fn write_header(&mut self) -> io::Result<()> {
        self.buffer.write_all(FIFO_MAGIC)?;
        self.buffer.write_all(&[self.version])?;
        Ok(())
    }

    /// Writes the FIFO data.
    fn write_fifo_data(&mut self, fifo: &Fifo) -> io::Result<()> {
        let len = fifo.len() as u32;
        self.buffer.write_all(&len.to_le_bytes())?;

        for i in 0..fifo.len() {
            let char_val = fifo.characters[i];
            let hash_val = fifo.hashes[i];
            let change_count = fifo.change_counts[i];
            let source_pos = fifo.source_positions[i];

            self.buffer.write_all(&char_val.to_le_bytes())?;
            self.buffer.write_all(&hash_val.to_le_bytes())?;
            self.buffer.write_all(&change_count.to_le_bytes())?;
            self.buffer.write_all(&source_pos.to_le_bytes())?;
        }

        Ok(())
    }

    /// Writes the format footer with checksum.
    fn write_footer(&mut self) -> io::Result<()> {
        let checksum = self.checksum();
        self.buffer.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }

    /// Calculates a simple checksum for data integrity.
    fn checksum(&self) -> u32 {
        let mut sum: u32 = 0;
        for (i, &byte) in self.buffer.iter().enumerate() {
            sum = sum.wrapping_add(byte as u32).wrapping_add(i as u32);
        }
        sum
    }

    /// Returns the current buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Clears the internal buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

impl Default for FifoEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Encoded FIFO data with metadata.
#[derive(Debug, Clone)]
pub struct EncodedFifo {
    /// Raw encoded data
    pub data: Vec<u8>,
    /// Format version used for encoding
    pub version: u8,
    /// Checksum for data integrity
    pub checksum: u32,
}

impl EncodedFifo {
    /// Creates a new encoded FIFO from raw data.
    pub fn new(data: Vec<u8>, version: u8, checksum: u32) -> Self {
        Self {
            data,
            version,
            checksum,
        }
    }

    /// Validates the encoded data integrity.
    pub fn validate(&self) -> bool {
        if self.data.len() < FIFO_MAGIC.len() + 1 {
            return false;
        }
        if &self.data[..FIFO_MAGIC.len()] != FIFO_MAGIC {
            return false;
        }
        if self.data[FIFO_MAGIC.len()] != self.version {
            return false;
        }
        let calculated_checksum = self.checksum();
        calculated_checksum == self.checksum
    }

    /// Calculates checksum for validation.
    fn checksum(&self) -> u32 {
        let mut sum: u32 = 0;
        let data_len = self.data.len() - mem::size_of::<u32>();
        for (i, &byte) in self.data[..data_len].iter().enumerate() {
            sum = sum.wrapping_add(byte as u32).wrapping_add(i as u32);
        }
        sum
    }

    /// Returns the size of the encoded data.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Streaming FIFO encoder for large datasets.
pub struct StreamingFifoEncoder<W> {
    /// Writer for streaming output
    writer: W,
    /// Buffer for chunked writing
    buffer: Vec<u8>,
    /// Current position in the stream
    position: u64,
}

impl<W: Write> StreamingFifoEncoder<W> {
    /// Creates a new streaming FIFO encoder.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            buffer: Vec::with_capacity(DEFAULT_BUFFER_SIZE),
            position: 0,
        }
    }

    /// Creates a streaming encoder with custom buffer size.
    pub fn with_buffer_size(writer: W, buffer_size: usize) -> Self {
        Self {
            writer,
            buffer: Vec::with_capacity(buffer_size),
            position: 0,
        }
    }

    /// Encodes and writes a FIFO structure to the stream.
    pub fn encode_fifo(&mut self, fifo: &Fifo) -> io::Result<u64> {
        let start_pos = self.position;
        self.write_header()?;
        self.write_fifo_data(fifo)?;
        self.write_footer()?;
        self.flush_buffer()?;
        Ok(self.position - start_pos)
    }

    /// Writes the format header to the stream.
    fn write_header(&mut self) -> io::Result<()> {
        self.buffer.write_all(FIFO_MAGIC)?;
        self.buffer.write_all(&[FIFO_VERSION])?;
        self.position += (FIFO_MAGIC.len() + 1) as u64;
        Ok(())
    }

    /// Writes FIFO data to the stream.
    fn write_fifo_data(&mut self, fifo: &Fifo) -> io::Result<()> {
        let len = fifo.len() as u32;
        self.buffer.write_all(&len.to_le_bytes())?;
        self.position += std::mem::size_of::<u32>() as u64;

        for i in 0..fifo.len() {
            let char_val = fifo.characters[i];
            let hash_val = fifo.hashes[i];
            let change_count = fifo.change_counts[i];
            let source_pos = fifo.source_positions[i];

            self.buffer.write_all(&char_val.to_le_bytes())?;
            self.buffer.write_all(&hash_val.to_le_bytes())?;
            self.buffer.write_all(&change_count.to_le_bytes())?;
            self.buffer.write_all(&source_pos.to_le_bytes())?;

            self.position += (mem::size_of::<u32>() * 4) as u64;
            if self.buffer.len() >= self.buffer.capacity() {
                self.flush_buffer()?;
            }
        }

        Ok(())
    }

    /// Writes the format footer to the stream.
    fn write_footer(&mut self) -> io::Result<()> {
        let checksum = self.checksum();
        self.buffer.write_all(&checksum.to_le_bytes())?;
        self.position += std::mem::size_of::<u32>() as u64;
        Ok(())
    }

    /// Flushes the buffer to the underlying writer.
    fn flush_buffer(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            self.writer.write_all(&self.buffer)?;
            self.buffer.clear();
        }
        Ok(())
    }

    /// Calculates checksum for the current buffer.
    fn checksum(&self) -> u32 {
        let mut sum: u32 = 0;
        for (i, &byte) in self.buffer.iter().enumerate() {
            sum = sum.wrapping_add(byte as u32).wrapping_add(i as u32);
        }
        sum
    }

    /// Returns the current stream position.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Flushes any remaining data and returns the writer.
    pub fn into_inner(mut self) -> io::Result<W> {
        self.flush_buffer()?;
        Ok(self.writer)
    }
}

/// Compression level for encoded data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// No compression
    None,
    /// Fast compression
    Fast,
    /// Balanced compression
    Balanced,
    /// Maximum compression
    Max,
}

impl Default for CompressionLevel {
    fn default() -> Self {
        CompressionLevel::Balanced
    }
}

/// Compressed FIFO encoder for reducing data size.
pub struct CompressedFifoEncoder {
    /// Base encoder
    encoder: FifoEncoder,
    /// Compression level
    compression: CompressionLevel,
}

impl CompressedFifoEncoder {
    /// Creates a new compressed FIFO encoder.
    pub fn new(compression: CompressionLevel) -> Self {
        Self {
            encoder: FifoEncoder::new(),
            compression,
        }
    }

    /// Encodes and compresses a FIFO structure.
    pub fn encode_fifo(&mut self, fifo: &Fifo) -> io::Result<EncodedFifo> {
        let raw_data = self.encoder.encode_fifo(fifo)?;
        let compressed_data = match self.compression {
            CompressionLevel::None => raw_data.to_vec(),
            CompressionLevel::Fast
            | CompressionLevel::Balanced
            | CompressionLevel::Max => Self::compress(raw_data),
        };
        let checksum = self.checksum(&compressed_data);
        Ok(EncodedFifo::new(
            compressed_data,
            self.encoder.version,
            checksum,
        ))
    }

    /// Compresses data using a simple run-length encoding.
    fn compress(data: &[u8]) -> Vec<u8> {
        let mut compressed = Vec::with_capacity(data.len() / 2);
        let mut i = 0;

        while i < data.len() {
            let current = data[i];
            let mut count = 1;
            while i + count < data.len() && data[i + count] == current && count < 255 {
                count += 1;
            }
            if count > 2 {
                compressed.push(0xFF);
                compressed.push(count as u8);
                compressed.push(current);
            } else {
                for _ in 0..count {
                    compressed.push(current);
                }
            }
            i += count;
        }
        compressed
    }

    /// Calculates checksum for data validation.
    fn checksum(&self, data: &[u8]) -> u32 {
        let mut sum: u32 = 0;
        for (i, &byte) in data.iter().enumerate() {
            sum = sum.wrapping_add(byte as u32).wrapping_add(i as u32);
        }
        sum
    }

    /// Returns the current compression level.
    pub fn compression(&self) -> CompressionLevel {
        self.compression
    }
}

impl Default for CompressedFifoEncoder {
    fn default() -> Self {
        Self::new(CompressionLevel::default())
    }
}
