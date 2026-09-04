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

//! FIFO decoding for deserializing merge operations.
//!
//! This module provides decoding utilities for converting binary encoded
//! FIFO structures back into their in-memory representation.

use crate::userclient::mergen::mergen_binary_fifo::{Fifo, FifoEntry};
use crate::userclient::mergen::mergen_fifo_encode::{FIFO_MAGIC, FIFO_VERSION};
use std::io::{self, Read};

/// Error types for FIFO decoding operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Invalid magic bytes in header
    InvalidMagic,
    /// Unsupported format version
    UnsupportedVersion { expected: u8, found: u8 },
    /// Checksum validation failed
    ChecksumMismatch { expected: u32, calculated: u32 },
    /// Insufficient data for complete decode
    InsufficientData { required: usize, available: usize },
    /// Invalid data format
    InvalidFormat(String),
    /// I/O error during decoding
    IoError(String),
}

impl From<io::Error> for DecodeError {
    fn from(err: io::Error) -> Self {
        DecodeError::IoError(err.to_string())
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::InvalidMagic => write!(f, "Invalid magic bytes in header"),
            DecodeError::UnsupportedVersion { expected, found } => {
                write!(
                    f,
                    "Unsupported version: expected {}, found {}",
                    expected, found
                )
            }
            DecodeError::ChecksumMismatch {
                expected,
                calculated,
            } => {
                write!(
                    f,
                    "Checksum mismatch: expected {}, calculated {}",
                    expected, calculated
                )
            }
            DecodeError::InsufficientData {
                required,
                available,
            } => {
                write!(
                    f,
                    "Insufficient data: required {}, available {}",
                    required, available
                )
            }
            DecodeError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            DecodeError::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for DecodeError {}

/// FIFO decoder for deserializing merge operations.
pub struct FifoDecoder {
    /// Buffer for reading operations
    buffer: Vec<u8>,
    /// Expected format version
    expected_version: u8,
}

impl FifoDecoder {
    /// Creates a new FIFO decoder with default settings.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            expected_version: FIFO_VERSION,
        }
    }

    /// Creates a decoder with a specific expected version.
    pub fn with_version(version: u8) -> Self {
        Self {
            buffer: Vec::new(),
            expected_version: version,
        }
    }

    /// Decodes binary data into a FIFO structure.
    pub fn decode_fifo(&mut self, data: &[u8]) -> Result<Fifo, DecodeError> {
        self.buffer.clear();
        self.buffer.extend_from_slice(data);

        self.validate_header()?;
        let fifo = self.read_fifo_data()?;
        self.validate_checksum()?;

        Ok(fifo)
    }

    /// Validates the format header.
    fn validate_header(&self) -> Result<(), DecodeError> {
        if self.buffer.len() < FIFO_MAGIC.len() + 1 {
            return Err(DecodeError::InsufficientData {
                required: FIFO_MAGIC.len() + 1,
                available: self.buffer.len(),
            });
        }

        if &self.buffer[..FIFO_MAGIC.len()] != FIFO_MAGIC {
            return Err(DecodeError::InvalidMagic);
        }

        let version = self.buffer[FIFO_MAGIC.len()];
        if version != self.expected_version {
            return Err(DecodeError::UnsupportedVersion {
                expected: self.expected_version,
                found: version,
            });
        }

        Ok(())
    }

    /// Reads the FIFO data from the buffer.
    fn read_fifo_data(&self) -> Result<Fifo, DecodeError> {
        let header_size = FIFO_MAGIC.len() + 1;
        let footer_size = size_of::<u32>();

        if self.buffer.len() < header_size + std::mem::size_of::<u32>() + footer_size {
            return Err(DecodeError::InsufficientData {
                required: header_size + std::mem::size_of::<u32>() + footer_size,
                available: self.buffer.len(),
            });
        }

        let len = u32::from_le_bytes([
            self.buffer[header_size],
            self.buffer[header_size + 1],
            self.buffer[header_size + 2],
            self.buffer[header_size + 3],
        ]) as usize;

        let entry_size = size_of::<u32>() * 4;
        let data_size = len * entry_size;

        if self.buffer.len() < header_size + size_of::<u32>() + data_size + footer_size {
            return Err(DecodeError::InsufficientData {
                required: header_size + size_of::<u32>() + data_size + footer_size,
                available: self.buffer.len(),
            });
        }
        let mut fifo = Fifo::with_capacity(len.max(1));
        for i in 0..len {
            let offset = header_size + std::mem::size_of::<u32>() + i * entry_size;

            let char_val = u32::from_le_bytes([
                self.buffer[offset],
                self.buffer[offset + 1],
                self.buffer[offset + 2],
                self.buffer[offset + 3],
            ]);

            let hash_val = u64::from_le_bytes([
                self.buffer[offset + 4],
                self.buffer[offset + 5],
                self.buffer[offset + 6],
                self.buffer[offset + 7],
                self.buffer[offset + 8],
                self.buffer[offset + 9],
                self.buffer[offset + 10],
                self.buffer[offset + 11],
            ]);

            let change_count = u32::from_le_bytes([
                self.buffer[offset + 12],
                self.buffer[offset + 13],
                self.buffer[offset + 14],
                self.buffer[offset + 15],
            ]);

            let source_pos = u32::from_le_bytes([
                self.buffer[offset + 16],
                self.buffer[offset + 17],
                self.buffer[offset + 18],
                self.buffer[offset + 19],
            ]);

            let ch = char::from_u32(char_val).unwrap_or(char::REPLACEMENT_CHARACTER);
            let entry = FifoEntry::new(u32::from(ch), hash_val, change_count, source_pos);
            fifo.insert(entry);
        }

        Ok(fifo)
    }

    /// Validates the checksum in the footer.
    fn validate_checksum(&self) -> Result<(), DecodeError> {
        let footer_offset = self.buffer.len() - std::mem::size_of::<u32>();
        let stored_checksum = u32::from_le_bytes([
            self.buffer[footer_offset],
            self.buffer[footer_offset + 1],
            self.buffer[footer_offset + 2],
            self.buffer[footer_offset + 3],
        ]);

        let calculated_checksum = self.checksum();

        if stored_checksum != calculated_checksum {
            return Err(DecodeError::ChecksumMismatch {
                expected: stored_checksum,
                calculated: calculated_checksum,
            });
        }

        Ok(())
    }

    /// Calculates checksum for validation.
    fn checksum(&self) -> u32 {
        let data_len = self.buffer.len() - std::mem::size_of::<u32>();
        let mut sum: u32 = 0;
        for (i, &byte) in self.buffer[..data_len].iter().enumerate() {
            sum = sum.wrapping_add(byte as u32).wrapping_add(i as u32);
        }
        sum
    }

    /// Returns the expected format version.
    pub fn expected_version(&self) -> u8 {
        self.expected_version
    }
}

impl Default for FifoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming FIFO decoder for large datasets.
pub struct StreamingFifoDecoder<R> {
    /// Reader for streaming input
    reader: R,
    /// Buffer for reading operations
    buffer: Vec<u8>,
    /// Expected format version
    expected_version: u8,
}

impl<R: Read> StreamingFifoDecoder<R> {
    /// Creates a new streaming FIFO decoder.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: Vec::new(),
            expected_version: FIFO_VERSION,
        }
    }

    /// Creates a decoder with specific expected version.
    pub fn with_version(reader: R, version: u8) -> Self {
        Self {
            reader,
            buffer: Vec::new(),
            expected_version: version,
        }
    }

    /// Decodes a FIFO structure from the stream.
    pub fn decode_fifo(&mut self) -> Result<Fifo, DecodeError> {
        self.read_header()?;
        let fifo = self.read_fifo_data()?;
        self.read_and_validate_checksum()?;
        Ok(fifo)
    }

    /// Reads and validates the format header.
    fn read_header(&mut self) -> Result<(), DecodeError> {
        let mut header = vec![0u8; FIFO_MAGIC.len() + 1];
        self.reader.read_exact(&mut header)?;

        if &header[..FIFO_MAGIC.len()] != FIFO_MAGIC {
            return Err(DecodeError::InvalidMagic);
        }

        let version = header[FIFO_MAGIC.len()];
        if version != self.expected_version {
            return Err(DecodeError::UnsupportedVersion {
                expected: self.expected_version,
                found: version,
            });
        }

        Ok(())
    }

    /// Reads the FIFO data from the stream.
    fn read_fifo_data(&mut self) -> Result<Fifo, DecodeError> {
        let mut len_bytes = [0u8; size_of::<u32>()];
        self.reader.read_exact(&mut len_bytes)?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        let entry_size = size_of::<u32>() * 4;
        let mut fifo = Fifo::with_capacity(len.max(1));

        for _ in 0..len {
            let mut entry_data = vec![0u8; entry_size];
            self.reader.read_exact(&mut entry_data)?;

            let char_val = u32::from_le_bytes([
                entry_data[0],
                entry_data[1],
                entry_data[2],
                entry_data[3],
            ]);

            let hash_val = u64::from_le_bytes([
                entry_data[4],
                entry_data[5],
                entry_data[6],
                entry_data[7],
                entry_data[8],
                entry_data[9],
                entry_data[10],
                entry_data[11],
            ]);

            let change_count = u32::from_le_bytes([
                entry_data[12],
                entry_data[13],
                entry_data[14],
                entry_data[15],
            ]);

            let source_pos = u32::from_le_bytes([
                entry_data[16],
                entry_data[17],
                entry_data[18],
                entry_data[19],
            ]);

            let ch = char::from_u32(char_val).unwrap_or(char::REPLACEMENT_CHARACTER);
            let entry = FifoEntry::new(u32::from(ch), hash_val, change_count, source_pos);
            fifo.insert(entry);
        }

        Ok(fifo)
    }

    /// Reads and validates the checksum from the stream.
    fn read_and_validate_checksum(&mut self) -> Result<(), DecodeError> {
        let mut checksum_bytes = [0u8; std::mem::size_of::<u32>()];
        self.reader.read_exact(&mut checksum_bytes)?;
        let stored_checksum = u32::from_le_bytes(checksum_bytes);

        let calculated_checksum = self.stream_checksum();

        if stored_checksum != calculated_checksum {
            return Err(DecodeError::ChecksumMismatch {
                expected: stored_checksum,
                calculated: calculated_checksum,
            });
        }

        Ok(())
    }

    /// Calculates checksum for the stream data.
    fn stream_checksum(&self) -> u32 {
        let mut sum: u32 = 0;
        for (i, &byte) in self.buffer.iter().enumerate() {
            sum = sum.wrapping_add(byte as u32).wrapping_add(i as u32);
        }
        sum
    }

    /// Returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.reader
    }
}

/// Result of a FIFO decode operation.
#[derive(Debug, Clone)]
pub struct DecodeResult {
    /// The decoded FIFO structure
    pub fifo: Fifo,
    /// The format version that was decoded
    pub version: u8,
    /// Whether the checksum validation passed
    pub checksum_valid: bool,
}

impl DecodeResult {
    /// Creates a new decode result.
    pub fn new(fifo: Fifo, version: u8, checksum_valid: bool) -> Self {
        Self {
            fifo,
            version,
            checksum_valid,
        }
    }

    /// Returns true if the decode was successful and validated.
    pub fn is_valid(&self) -> bool {
        self.checksum_valid
    }
}
