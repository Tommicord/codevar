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

use crate::edit::wredit_history::{BytecodeError, BytecodeResult};

/// The type of change for a line in the preprocessed buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// Line was inserted
    Insert,
    /// Line was deleted
    Delete,
    /// Line was replaced/modified
    Replace,
}

impl ChangeType {
    /// Converts ChangeType to byte for serialization.
    pub fn to_byte(self) -> u8 {
        match self {
            ChangeType::Insert => 0,
            ChangeType::Delete => 1,
            ChangeType::Replace => 2,
        }
    }

    /// Converts byte to ChangeType for deserialization.
    pub fn from_byte(byte: u8) -> BytecodeResult<Self> {
        match byte {
            0 => Ok(ChangeType::Insert),
            1 => Ok(ChangeType::Delete),
            2 => Ok(ChangeType::Replace),
            _ => Err(BytecodeError::InvalidDelta),
        }
    }
}

/// A key-value pair for line change data.
#[derive(Debug, Clone)]
pub struct KeyValue {
    /// The line index
    pub line_index: usize,
    /// The length of the line
    pub line_length: usize,
    /// The type of change
    pub change_type: ChangeType,
}

impl KeyValue {
    /// Creates a new KeyValue pair.
    ///
    /// # Arguments
    ///
    /// * `line_index` - The index of the line
    /// * `line_length` - The length of the line
    /// * `change_type` - The type of change
    pub fn new(line_index: usize, line_length: usize, change_type: ChangeType) -> Self {
        Self {
            line_index,
            line_length,
            change_type,
        }
    }
}

/// Builder for creating preprocessed buffers (ppbuff) for history deltas.
///
/// This builder constructs a preprocessed buffer that contains metadata about
/// line changes and the actual text data that changed.
pub struct PpbuffBuilder {
    start: usize,
    end: usize,
    changes: Vec<KeyValue>,
    text_data: Vec<u8>,
}

impl PpbuffBuilder {
    /// Creates a new PpbuffBuilder.
    ///
    /// # Arguments
    ///
    /// * `start` - The start position of the changes
    /// * `end` - The end position of the changes
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            changes: Vec::new(),
            text_data: Vec::new(),
        }
    }

    /// Adds a line change to the builder.
    ///
    /// # Arguments
    ///
    /// * `line_index` - The index of the line
    /// * `line_length` - The length of the line
    /// * `change_type` - The type of change
    pub fn add_change(
        &mut self,
        line_index: usize,
        line_length: usize,
        change_type: ChangeType,
    ) {
        self.changes
            .push(KeyValue::new(line_index, line_length, change_type));
    }

    /// Adds multiple line changes from an array of KeyValue pairs.
    ///
    /// # Arguments
    ///
    /// * `changes` - A slice of KeyValue pairs to add
    pub fn add_changes(&mut self, changes: &[KeyValue]) {
        self.changes.extend_from_slice(changes);
    }

    /// Sets the text data for the changes.
    ///
    /// # Arguments
    ///
    /// * `text_data` - The encoded bytes of text lines that changed
    pub fn set_text_data(&mut self, text_data: Vec<u8>) {
        self.text_data = text_data;
    }

    /// Builds the preprocessed buffer bytecode.
    ///
    /// # Returns
    ///
    /// A SerializationResult containing the serialized ppbuff or an error.
    pub fn build(&self) -> BytecodeResult<Vec<u8>> {
        let mut bytes = Vec::<u8>::new();
        bytes.extend_from_slice(&self.start.to_le_bytes());
        bytes.extend_from_slice(&self.end.to_le_bytes());
        let change_count = self.changes.len();
        bytes.extend_from_slice(&(change_count as u32).to_le_bytes());

        // Serialize each change: line index -> length -> change type
        for change in &self.changes {
            bytes.extend_from_slice(&change.line_index.to_le_bytes());
            bytes.extend_from_slice(&change.line_length.to_le_bytes());
            bytes.push(change.change_type.to_byte());
        }

        bytes.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes());
        bytes.extend_from_slice(&(self.text_data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.text_data);

        Ok(bytes)
    }

    /// Returns the start position.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Returns the end position.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Returns the number of changes.
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    /// Returns a reference to the changes.
    pub fn changes(&self) -> &[KeyValue] {
        &self.changes
    }

    /// Returns a reference to the text data.
    pub fn text_data(&self) -> &[u8] {
        &self.text_data
    }
}

#[cfg(test)]
mod tests {
    use super::{ChangeType, KeyValue, PpbuffBuilder};

    fn creation() {
        let kv = KeyValue::new(0, 10, ChangeType::Insert);
        assert_eq!(kv.line_index, 0);
        assert_eq!(kv.line_length, 10);
        assert_eq!(kv.change_type, ChangeType::Insert);
    }

    fn change_type_to_byte() {
        assert_eq!(ChangeType::Insert.to_byte(), 0);
        assert_eq!(ChangeType::Delete.to_byte(), 1);
        assert_eq!(ChangeType::Replace.to_byte(), 2);
    }

    fn change_type_from_byte() {
        assert_eq!(ChangeType::from_byte(0).unwrap(), ChangeType::Insert);
        assert_eq!(ChangeType::from_byte(1).unwrap(), ChangeType::Delete);
        assert_eq!(ChangeType::from_byte(2).unwrap(), ChangeType::Replace);
        assert!(ChangeType::from_byte(3).is_err());
    }

    fn ppbuff_builder_creation() {
        let builder = PpbuffBuilder::new(0, 100);
        assert_eq!(builder.start(), 0);
        assert_eq!(builder.end(), 100);
        assert_eq!(builder.change_count(), 0);
    }

    fn ppbuff_add_change() {
        let mut builder = PpbuffBuilder::new(0, 100);
        builder.add_change(0, 10, ChangeType::Insert);
        assert_eq!(builder.change_count(), 1);
        assert_eq!(builder.changes()[0].line_index, 0);
        assert_eq!(builder.changes()[0].line_length, 10);
        assert_eq!(builder.changes()[0].change_type, ChangeType::Insert);
    }

    fn ppbuff_add_changes() {
        let mut builder = PpbuffBuilder::new(0, 100);
        let changes = vec![
            KeyValue::new(0, 10, ChangeType::Insert),
            KeyValue::new(1, 15, ChangeType::Delete),
        ];
        builder.add_changes(&changes);
        assert_eq!(builder.change_count(), 2);
    }

    fn ppbuff_set_text_data() {
        let mut builder = PpbuffBuilder::new(0, 100);
        let text_data = vec![1u8, 2, 3, 4];
        builder.set_text_data(text_data.clone());
        assert_eq!(builder.text_data(), &text_data[..]);
    }

    fn ppbuff_build_empty() {
        let builder = PpbuffBuilder::new(0, 100);
        let result = builder.build();
        assert!(result.is_ok());
        let bytes = result.unwrap();

        // Should contain: start (8 bytes) + end (8 bytes) + count (4 bytes) + end marker (4 bytes) + text_len (4 bytes)
        assert_eq!(bytes.len(), 28);
    }

    fn ppbuff_build_with_changes() {
        let mut builder = PpbuffBuilder::new(0, 100);
        builder.add_change(0, 10, ChangeType::Insert);
        builder.add_change(1, 15, ChangeType::Delete);
        builder.set_text_data(vec![1u8, 2, 3]);

        let result = builder.build();
        assert!(result.is_ok());
        let bytes = result.unwrap();

        // Should contain: start (8) + end (8) + count (4) + 2 changes (8+8+1 each = 34) + end marker (4) + text_len (4) + text_data (3)
        assert_eq!(bytes.len(), 65);
    }

    fn ppbuff_build_serialization_format() {
        let mut builder = PpbuffBuilder::new(50, 150);
        builder.add_change(5, 20, ChangeType::Replace);
        builder.set_text_data(vec![65u8, 66, 67]); // "ABC"

        let bytes = builder.build().unwrap();

        // Check start position (50)
        let start = usize::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(start, 50);

        // Check end position (150)
        let end = usize::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(end, 150);

        // Check change count (1)
        let count = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(count, 1);

        // Check line index (5)
        let line_index = usize::from_le_bytes(bytes[20..28].try_into().unwrap());
        assert_eq!(line_index, 5);

        // Check line length (20)
        let line_length = usize::from_le_bytes(bytes[28..36].try_into().unwrap());
        assert_eq!(line_length, 20);

        // Check change type (Replace = 2)
        let change_type = bytes[36];
        assert_eq!(change_type, 2);

        // Check end marker (0xFFFFFFFF)
        let end_marker = u32::from_le_bytes(bytes[37..41].try_into().unwrap());
        assert_eq!(end_marker, 0xFFFFFFFF);

        // Check text data length (3)
        let text_len = u32::from_le_bytes(bytes[41..45].try_into().unwrap());
        assert_eq!(text_len, 3);

        // Check text data
        assert_eq!(&bytes[45..], &[65u8, 66, 67]);
    }
}
