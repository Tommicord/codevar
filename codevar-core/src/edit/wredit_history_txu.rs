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

use crate::edit::wredit_base_writable::{ENCODING_CUSTOM, ENCODING_UTF8, ENCODING_UTF32};
use crate::edit::wredit_history::{
    BytecodeError, BytecodeResult, HistoryDeserializable, HistorySerializable,
};
use crate::edit::wredit_history_txu_ppbuff::{KeyValue, PpbuffBuilder};
use std::time::{Duration, SystemTime};

/// The text unit delta hash size in bytes
pub const TXU_HASH_SIZE: usize = 8;

/// The type of delta operation for history tracking.
///
/// This enum defines the different types of text changes that can be
/// recorded in the history, such as insertions, deletions, and replacements.
#[derive(Clone, Copy)]
pub enum HistoryTXUDeltaType {
    /// Defines single-line text replacement
    CodeReplace,
    /// Defines multiline replacement of two points
    CodeReplace2,
    /// Defines single-line text deletion
    CodeDelete,
    /// Defines multiline two point deletion
    CodeDelete2,
    /// Defines single-line add of code
    CodeAdd,
    /// Defines multiline add of code
    CodeAdd2,
}

/// Stores a delta of a text history.
///
/// This struct represents a single change in the history, containing information
/// about the type of change, its location, and metadata about the change.
#[derive(Clone)]
pub struct HistoryTXUDelta {
    /// The TXU delta type
    pub type_: HistoryTXUDeltaType,
    /// The start offset of row, column of the changes (8-bits row, 8-bits column)
    pub start: u32,
    /// Pre-processed buffer
    pub ppbuff: Vec<u8>,
    /// The end offset of row, column of the changes (8-bits row, 8-bits column)
    pub end: u32,
    /// The text encoding
    pub encoding: u16,
    /// The next delta in memory
    pub next: Option<Box<HistoryTXUDelta>>,
    /// 1-bit of TXU delta type (0 = single-line, 1 = multiline)
    /// 63-bits count of changes in lines of text
    info: u64,
}

impl HistoryTXUDelta {
    /// Creates a new history TXU delta.
    ///
    /// # Arguments
    ///
    /// * `type_` - The type of delta operation.
    /// * `start` - The start offset of the change.
    /// * `end` - The end offset of the change.
    /// * `info` - Additional information about the change.
    /// * `encoding` - The text encoding used.
    ///
    /// # Returns
    ///
    /// A new HistoryTXUDelta instance.
    pub fn new(
        type_: HistoryTXUDeltaType,
        start: u32,
        end: u32,
        info: u64,
        ppbuff: Vec<u8>,
        encoding: u16,
        next: Option<Box<HistoryTXUDelta>>,
    ) -> Self {
        Self {
            type_,
            start,
            end,
            ppbuff,
            info,
            encoding,
            next,
        }
    }

    /// Creates a new history TXU delta from a PpbuffBuilder.
    ///
    /// # Arguments
    ///
    /// * `type_` - The type of delta operation.
    /// * `builder` - The PpbuffBuilder containing change data.
    /// * `encoding` - The text encoding used.
    ///
    /// # Returns
    ///
    /// A SerializationResult containing the new HistoryTXUDelta or an error.
    pub fn from_ppbuff_builder(
        type_: HistoryTXUDeltaType,
        builder: &PpbuffBuilder,
        encoding: u16,
    ) -> BytecodeResult<Self> {
        let ppbuff = builder.build()?;
        Ok(Self {
            type_,
            start: builder.start() as u32,
            end: builder.end() as u32,
            ppbuff,
            info: 0,
            encoding,
            next: None,
        })
    }

    /// Creates a new history TXU delta from KeyValue array and text data.
    ///
    /// # Arguments
    ///
    /// * `type_` - The type of delta operation.
    /// * `start` - The start position of the changes.
    /// * `end` - The end position of the changes.
    /// * `changes` - Array of KeyValue pairs describing line changes.
    /// * `text_data` - The encoded bytes of text lines that changed.
    /// * `encoding` - The text encoding used.
    ///
    /// # Returns
    ///
    /// A SerializationResult containing the new HistoryTXUDelta or an error.
    pub fn from_changes(
        type_: HistoryTXUDeltaType,
        start: usize,
        end: usize,
        changes: &[KeyValue],
        text_data: Vec<u8>,
        encoding: u16,
    ) -> BytecodeResult<Self> {
        let mut builder = PpbuffBuilder::new(start, end);
        builder.add_changes(changes);
        builder.set_text_data(text_data);
        Self::from_ppbuff_builder(type_, &builder, encoding)
    }

    /// Sets the type of the delta operation.
    ///
    /// # Arguments
    ///
    /// * `type_` - The new type of delta operation.
    pub fn set_type(&mut self, type_: HistoryTXUDeltaType) {
        self.type_ = type_;
    }

    /// Returns a reference to the type of the delta operation.
    ///
    /// # Returns
    ///
    /// A reference to the HistoryTXUDeltaType.
    pub fn type_(&self) -> &HistoryTXUDeltaType {
        &self.type_
    }

    /// Sets the next delta in the linked list.
    ///
    /// # Arguments
    ///
    /// * `next` - An optional boxed HistoryTXUDelta to set as the next
    ///            delta in the linked list.
    ///
    pub fn set_next(&mut self, next: Option<Box<HistoryTXUDelta>>) {
        self.next = next;
    }

    /// Returns a reference to the next delta of the current instance
    /// in the linked list.
    ///
    /// # Returns
    ///
    /// An optional reference to the next boxed HistoryTXUDelta.
    pub fn next(&self) -> Option<&Box<HistoryTXUDelta>> {
        self.next.as_ref()
    }

    /// Sets the start offset of the delta.
    ///
    /// # Arguments
    ///
    /// * `start` - The new start offset as a u32.
    pub fn set_start(&mut self, start: u32) {
        self.start = start;
    }

    /// Returns the start offset of the delta.
    ///
    /// # Returns
    ///
    /// The start offset as a u32.
    pub fn start(&self) -> u32 {
        self.start
    }

    /// Sets the end offset of the delta.
    ///
    /// # Arguments
    ///
    /// * `end` - The new end offset as a u32.
    pub fn set_end(&mut self, end: u32) {
        self.end = end;
    }

    /// Returns the end offset of the delta.
    ///
    /// # Returns
    ///
    /// The end offset as a u32.
    pub fn end(&self) -> u32 {
        self.end
    }

    /// Sets the encoding of the delta (eg. UTF8, UTF16).
    ///
    /// # Arguments
    ///
    /// * `encoding` - The new encoding value as a u8.
    pub fn set_encoding(&mut self, encoding: u16) {
        self.encoding = encoding;
    }

    /// Returns the encoding of the delta (eg. UTF8, UTF16).
    ///
    /// # Returns
    ///
    /// The encoding value as a u16.
    pub fn encoding(&self) -> u16 {
        self.encoding
    }

    /// Returns a reference to the pre-processed buffer.
    ///
    /// # Returns
    ///
    /// A reference to the pre-processed buffer vector.
    pub fn ppbuff(&self) -> &Vec<u8> {
        &self.ppbuff
    }

    /// Returns a mutable reference to the pre-processed buffer.
    ///
    /// # Returns
    ///
    /// A mutable reference to the pre-processed buffer vector.
    pub fn ppbuff_mut(&mut self) -> &mut Vec<u8> {
        &mut self.ppbuff
    }

    /// Returns the info field of the delta.
    ///
    /// # Returns
    ///
    /// The info value containing change metadata.
    pub fn info(&self) -> u64 {
        self.info
    }

    /// Sets the info field of the delta.
    ///
    /// # Arguments
    ///
    /// * `info` - The new info value.
    pub fn set_info(&mut self, info: u64) {
        self.info = info;
    }

    fn is_type(&self) -> u64 {
        (self.info >> 63) & 1
    }

    /// Checks if the delta is a multi-line operation.
    ///
    /// # Returns
    ///
    /// `true` if the delta spans multiple lines, `false` otherwise.
    pub fn is_multi_line(&self) -> bool {
        self.is_type() == 1
    }

    /// Checks if the delta is a single-line operation.
    ///
    /// # Returns
    ///
    /// `true` if the delta is on a single line, `false` otherwise.
    pub fn is_single_line(&self) -> bool {
        self.is_type() == 0
    }
}

/// Defines a text unit (TXU) for the file history.
///
/// This struct contains information about the HistoryBuffer and stores
/// delta information for efficient change tracking (only stores changed lines).
/// Large cold payloads may be FcWare-compressed in memory; see
/// [`crate::edit::wredit_history_compress`].
#[derive(Clone)]
pub struct HistoryTXU {
    buffer: Vec<u8>,
    delta_offset: Vec<usize>,
    delta: Box<HistoryTXUDelta>,
    buffer_length: usize,
    timestamp: u64,
    hash: Box<[u8]>,
    /// Whether `buffer` and/or `ppbuff` currently hold FcWare frames.
    payload_compressed: bool,
}
/// Hash computation utility for history TXU deltas.
///
/// This struct provides methods for computing hashes from deltas,
/// used for unique identification and integrity verification.
pub struct HistoryTXUDeltaHash;

impl HistoryTXUDeltaHash {
    /// Computes a hash from the history text unit delta with optional seed.
    ///
    /// # Arguments
    ///
    /// * `delta` - The delta to hash.
    /// * `seed_bytes` - Optional seed bytes for the hash.
    ///
    /// # Returns
    ///
    /// A hash of the delta.
    pub fn hash_from_delta_mixed(
        delta: &HistoryTXUDelta,
        seed_bytes: Option<&[u8]>,
    ) -> Box<[u8]> {
        let time = Self::hash_from_time(delta);
        let mut output = [0u8; TXU_HASH_SIZE];

        if let Some(seed) = seed_bytes {
            for (i, chunk) in seed.chunks(TXU_HASH_SIZE).enumerate() {
                for (byte_idx, &byte) in chunk.iter().enumerate() {
                    let slot = (i.saturating_add(byte_idx)) % TXU_HASH_SIZE;
                    output[slot] = output[slot].wrapping_add(byte);
                }
            }
        }
        for (i, &byte) in time.iter().enumerate() {
            let slot = i % TXU_HASH_SIZE;
            output[slot] = output[slot].wrapping_add(byte);
        }
        Box::new(output)
    }
    /// Computes a hash from the history text unit delta.
    ///
    /// # Arguments
    ///
    /// * `delta` - The delta to hash.
    ///
    /// # Returns
    ///
    /// A hash of the delta.
    pub fn hash_from_delta(delta: &HistoryTXUDelta) -> Box<[u8]> {
        Self::hash_from_delta_mixed(delta, None)
    }

    /// Computes a hash from the current time in nanoseconds.
    ///
    /// This is used as a fallback when getting the current time fails.
    ///
    /// # Arguments
    ///
    /// * `delta` - The delta to use for time-based hash generation.
    ///
    /// # Returns
    ///
    /// A hash based on time and delta information.
    fn hash_from_time(delta: &HistoryTXUDelta) -> Box<[u8]> {
        let time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| {
                // Random math only for robustness
                let mask = 0x530;
                let mut millis = (mask & delta.end) as u64;
                millis <<= 5;
                millis &= 0x985597;
                // Join bits of delta.start and delta.end and apply a random mask
                let nano_mask = 0xcabbffdd;
                let nanos =
                    ((delta.start as u64) << u32::BITS) | (delta.end as u64) & nano_mask;
                return Duration::new(millis, nanos as u32);
            })
            .as_nanos()
            .to_le_bytes();
        Box::new(time)
    }
}

impl HistoryTXU {
    /// Creates a new history TXU.
    ///
    /// # Arguments
    ///
    /// * `delta` - The delta to store in this TXU.
    ///
    /// # Returns
    ///
    /// A new HistoryTXU instance.
    pub fn new<Raw, Buf, const GAP_SIZE: usize>(delta: HistoryTXUDelta) -> Self {
        let hash = HistoryTXUDeltaHash::hash_from_delta(&delta);
        Self {
            buffer: Vec::new(),
            delta_offset: Vec::new(),
            delta: Box::new(delta),
            buffer_length: 0,
            timestamp: 0,
            hash,
            payload_compressed: false,
        }
    }

    /// Returns whether payload blobs are FcWare-compressed in memory.
    #[inline]
    #[must_use]
    pub(crate) fn payload_compressed(&self) -> bool {
        self.payload_compressed
    }

    /// Sets the in-memory FcWare compression flag.
    #[inline]
    pub(crate) fn set_payload_compressed(&mut self, compressed: bool) {
        self.payload_compressed = compressed;
    }

    /// Returns a reference to the buffer.
    ///
    /// # Returns
    ///
    /// A slice containing the buffer data.
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Returns a mutable reference to the buffer.
    ///
    /// # Returns
    ///
    /// A mutable reference to the buffer vector.
    pub fn buffer_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }

    /// Returns a reference to the delta offsets.
    ///
    /// # Returns
    ///
    /// A slice containing the delta offset indices.
    pub fn delta_offset(&self) -> &[usize] {
        &self.delta_offset
    }

    /// Returns a mutable reference to the delta offsets.
    ///
    /// # Returns
    ///
    /// A mutable reference to the delta offset vector.
    pub fn delta_offset_mut(&mut self) -> &mut Vec<usize> {
        &mut self.delta_offset
    }

    /// Returns a reference to the delta.
    ///
    /// # Returns
    ///
    /// A reference to the HistoryTXUDelta.
    pub fn delta(&self) -> &HistoryTXUDelta {
        &self.delta
    }

    /// Returns a mutable reference to the delta.
    ///
    /// # Returns
    ///
    /// A mutable reference to the HistoryTXUDelta.
    pub fn delta_mut(&mut self) -> &mut HistoryTXUDelta {
        &mut self.delta
    }

    /// Returns the buffer length.
    ///
    /// # Returns
    ///
    /// The length of the buffer in bytes.
    pub fn buffer_length(&self) -> usize {
        self.buffer_length
    }

    /// Returns a mutable reference to the buffer length.
    ///
    /// # Returns
    ///
    /// A mutable reference to the buffer length.
    pub fn buffer_length_mut(&mut self) -> &mut usize {
        &mut self.buffer_length
    }

    /// Returns the timestamp of this TXU.
    ///
    /// # Returns
    ///
    /// The timestamp value.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Returns the hash of this TXU.
    ///
    /// # Returns
    ///
    /// A slice containing the hash.
    pub fn hash(&self) -> &[u8] {
        &self.hash
    }

    /// Returns a mutable reference to the hash.
    ///
    /// # Returns
    ///
    /// A mutable reference to the hash slice.
    pub fn hash_mut(&mut self) -> &mut [u8] {
        &mut self.hash
    }
}

impl HistorySerializable for HistoryTXU {
    /// Converts the TXU to bytecode representation.
    ///
    /// # Returns
    ///
    /// A SerializationResult containing the bytecode vector or an error.
    fn to_bytecode(&self) -> BytecodeResult<Vec<u8>> {
        let encoding = self.delta.encoding();
        if encoding < ENCODING_UTF8 || encoding > ENCODING_UTF32 {
            return Err(BytecodeError::InvalidEncoding);
        }

        // Wire format always stores plain payloads.
        let mut plain = self.clone();
        if plain.payload_compressed {
            plain
                .ensure_decompressed()
                .map_err(|_| BytecodeError::InvalidDelta)?;
        }

        let mut bytes = Vec::<u8>::new();
        let buffer_len = plain.buffer_length;
        bytes.extend_from_slice(&buffer_len.to_le_bytes());
        bytes.extend_from_slice(&plain.buffer);

        let offset_count = plain.delta_offset.len();
        bytes.extend_from_slice(&(offset_count as u32).to_le_bytes());

        for offset in &plain.delta_offset {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }

        bytes.extend_from_slice(&plain.serialize_delta()?);

        bytes.extend_from_slice(&plain.timestamp.to_le_bytes());

        let hash_len = plain.hash.len();
        bytes.extend_from_slice(&(hash_len as u8).to_le_bytes());
        bytes.extend_from_slice(&*plain.hash);

        Ok(bytes)
    }
}

impl HistoryTXU {
    /// Serializes the delta to bytecode.
    ///
    /// # Returns
    ///
    /// A SerializationResult containing the serialized delta or an error.
    fn serialize_delta(&self) -> BytecodeResult<Vec<u8>> {
        let mut bytes = Vec::<u8>::new();
        let delta = &self.delta;

        // Serialize delta type
        let type_byte = match delta.type_() {
            HistoryTXUDeltaType::CodeReplace => 0,
            HistoryTXUDeltaType::CodeReplace2 => 1,
            HistoryTXUDeltaType::CodeDelete => 2,
            HistoryTXUDeltaType::CodeDelete2 => 3,
            HistoryTXUDeltaType::CodeAdd => 4,
            HistoryTXUDeltaType::CodeAdd2 => 5,
        };
        bytes.push(type_byte);

        // Serialize start position
        bytes.extend_from_slice(&delta.start().to_le_bytes());

        // Serialize end position
        bytes.extend_from_slice(&delta.end().to_le_bytes());

        // Serialize info
        bytes.extend_from_slice(&delta.info().to_le_bytes());

        // Serialize pre-processed buffer
        let ppbuff_len = delta.ppbuff().len();
        bytes.extend_from_slice(&(ppbuff_len as u32).to_le_bytes());
        bytes.extend_from_slice(delta.ppbuff());

        // Serialize encoding
        bytes.extend_from_slice(&delta.encoding().to_le_bytes());

        // Serialize next delta (if exists)
        match delta.next() {
            Some(next_delta) => {
                bytes.push(1); // Has next
                let next_txu = HistoryTXU::new::<u8, u8, 4096>((**next_delta).clone());
                bytes.extend_from_slice(&next_txu.to_bytecode()?);
            }
            None => {
                bytes.push(0); // No next
            }
        }

        Ok(bytes)
    }
}

impl HistoryDeserializable for HistoryTXU {
    /// Creates a HistoryTXU instance from bytecode with validation.
    ///
    /// # Arguments
    ///
    /// * `bytecode` - The bytecode to deserialize
    ///
    /// # Returns
    ///
    /// A BytecodeResult containing the deserialized HistoryTXU or an error.
    fn from_bytecode(bytecode: &[u8]) -> BytecodeResult<Self> {
        let mut offset = 0;

        // Read buffer length
        if offset + std::mem::size_of::<usize>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let buffer_length = usize::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<usize>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        );
        offset += std::mem::size_of::<usize>();

        // Read buffer
        if offset + buffer_length > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let buffer = bytecode[offset..offset + buffer_length].to_vec();
        offset += buffer_length;

        // Read delta offset count
        if offset + std::mem::size_of::<u32>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let offset_count = u32::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<u32>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        ) as usize;
        offset += std::mem::size_of::<u32>();

        // Read delta offsets
        let mut delta_offset = Vec::with_capacity(offset_count);
        for _ in 0..offset_count {
            if offset + std::mem::size_of::<usize>() > bytecode.len() {
                return Err(BytecodeError::BufferOverflow);
            }
            let off = usize::from_le_bytes(
                bytecode[offset..offset + std::mem::size_of::<usize>()]
                    .try_into()
                    .map_err(|_| BytecodeError::BufferOverflow)?,
            );
            delta_offset.push(off);
            offset += std::mem::size_of::<usize>();
        }

        // Deserialize delta
        let (delta, delta_bytes_consumed) = Self::deserialize_delta(&bytecode[offset..])?;
        offset += delta_bytes_consumed;

        // Read timestamp
        if offset + std::mem::size_of::<u64>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let timestamp = u64::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<u64>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        );
        offset += std::mem::size_of::<u64>();

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

        // Create HistoryTXU instance
        let txu = Self {
            buffer,
            delta_offset,
            delta: Box::new(delta),
            buffer_length,
            timestamp,
            hash,
            payload_compressed: false,
        };

        Ok(txu)
    }
}

impl HistoryTXU {
    /// Deserializes a delta from bytecode.
    ///
    /// # Arguments
    ///
    /// * `bytecode` - The bytecode to deserialize
    ///
    /// # Returns
    ///
    /// A BytecodeResult containing the deserialized delta and bytes consumed.
    fn deserialize_delta(bytecode: &[u8]) -> BytecodeResult<(HistoryTXUDelta, usize)> {
        let mut offset = 0;

        // Read delta type
        if offset >= bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let type_byte = bytecode[offset];
        offset += 1;

        let type_ = match type_byte {
            0 => HistoryTXUDeltaType::CodeReplace,
            1 => HistoryTXUDeltaType::CodeReplace2,
            2 => HistoryTXUDeltaType::CodeDelete,
            3 => HistoryTXUDeltaType::CodeDelete2,
            4 => HistoryTXUDeltaType::CodeAdd,
            5 => HistoryTXUDeltaType::CodeAdd2,
            _ => return Err(BytecodeError::InvalidDelta),
        };

        // Read start
        if offset + std::mem::size_of::<u32>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let start = u32::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<u32>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        );
        offset += std::mem::size_of::<u32>();

        // Read end
        if offset + std::mem::size_of::<u32>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let end = u32::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<u32>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        );
        offset += std::mem::size_of::<u32>();

        // Read info
        if offset + std::mem::size_of::<u64>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let info = u64::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<u64>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        );
        offset += std::mem::size_of::<u64>();

        // Read ppbuff length
        if offset + std::mem::size_of::<u32>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let ppbuff_len = u32::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<u32>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        ) as usize;
        offset += std::mem::size_of::<u32>();

        // Read ppbuff
        if offset + ppbuff_len > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let ppbuff = bytecode[offset..offset + ppbuff_len].to_vec();
        offset += ppbuff_len;

        // Read encoding
        if offset + std::mem::size_of::<u16>() > bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let encoding = u16::from_le_bytes(
            bytecode[offset..offset + std::mem::size_of::<u16>()]
                .try_into()
                .map_err(|_| BytecodeError::BufferOverflow)?,
        );
        offset += std::mem::size_of::<u16>();

        if encoding < ENCODING_UTF8 || encoding > ENCODING_CUSTOM {
            return Err(BytecodeError::InvalidEncoding);
        }
        // Read next delta flag
        if offset >= bytecode.len() {
            return Err(BytecodeError::BufferOverflow);
        }
        let has_next = bytecode[offset];
        offset += 1;

        let next = if has_next == 1 {
            // Deserialize next delta recursively
            let (next_delta, bytes_consumed) =
                Self::deserialize_delta(&bytecode[offset..])?;
            offset += bytes_consumed;
            Some(Box::new(next_delta))
        } else {
            None
        };

        let delta = HistoryTXUDelta {
            type_,
            start,
            end,
            ppbuff,
            info,
            encoding,
            next,
        };

        Ok((delta, offset))
    }
}
