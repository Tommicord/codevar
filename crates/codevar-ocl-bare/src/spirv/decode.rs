//! Copyright 2026 Codevar Project
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

//! Bounds-checked decoding of the SPIR-V word stream.
//!
//! The binary format is a five-word header followed by instructions whose
//! first word packs the 16-bit word count with the 16-bit opcode. Nothing
//! in the format is self-describing enough to trust, so every step here
//! re-checks the remaining length before reading; a malformed module can
//! therefore fail with [`Error::InvalidBinary`] and never index out of
//! range.

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{Error, Result};

/// The SPIR-V magic number, `0x07230203` in the first word.
pub(crate) const MAGIC: u32 = 0x0723_0203;

/// Number of words in the module header.
pub(crate) const HEADER_WORDS: usize = 5;

/// One decoded instruction: its opcode and its operand words.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Instr {
    /// The 16-bit opcode.
    pub opcode: u16,
    /// Index of the first operand word in the module's word vector.
    pub start: u32,
    /// Number of operand words (the instruction's word count minus one).
    pub len: u16,
}

impl Instr {
    /// The operand words of this instruction inside `words`.
    #[must_use]
    #[inline]
    pub(crate) fn words(self, words: &[u32]) -> &[u32] {
        let start = self.start as usize;
        let end = start + self.len as usize;
        // The decoder established `start + len <= words.len()` when it
        // created this instruction.
        &words[start..end]
    }

    /// The operand word at `index`, or `None` when out of range.
    #[must_use]
    #[inline]
    pub(crate) fn word(self, words: &[u32], index: usize) -> Option<u32> {
        self.words(words).get(index).copied()
    }

    /// The `<id>` at operand `index`, validated below the module bound.
    #[must_use]
    #[inline]
    pub(crate) fn id(self, words: &[u32], index: usize, bound: u32) -> Option<u32> {
        let value = self.word(words, index)?;
        ((1..bound).contains(&value)).then_some(value)
    }
}

/// The validated header of a module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Header {
    /// The `<id>` bound — one above the largest id used.
    pub bound: u32,
}

/// Decodes the header and every instruction of `words`.
///
/// # Errors
///
/// [`Error::InvalidBinary`] when the word vector is shorter than the
/// header, the magic number or schema are wrong, the bound is zero, or
/// any instruction claims more words than remain.
pub(crate) fn decode(words: &[u32]) -> Result<(Header, Vec<Instr>)> {
    if words.len() < HEADER_WORDS {
        return Err(Error::InvalidBinary);
    }
    if words[0] != MAGIC || words[4] != 0 {
        return Err(Error::InvalidBinary);
    }
    let bound = words[3];
    if bound == 0 {
        return Err(Error::InvalidBinary);
    }
    let mut instrs = Vec::new();
    let mut index = HEADER_WORDS;
    while index < words.len() {
        let packed = words[index];
        let count = (packed >> 16) as usize;
        let opcode = packed as u16;
        if count == 0 || count > words.len() - index {
            return Err(Error::InvalidBinary);
        }
        let operands = index + 1;
        let len = count - 1;
        if len > u16::MAX as usize {
            return Err(Error::InvalidBinary);
        }
        instrs.push(Instr {
            opcode,
            start: operands as u32,
            len: len as u16,
        });
        index += count;
    }
    // A header without a single instruction is not a module.
    if instrs.is_empty() {
        return Err(Error::InvalidBinary);
    }
    Ok((Header { bound }, instrs))
}

/// Reads the NUL-terminated, 4-byte-padded UTF-8 string starting at
/// `words[start..]`.
///
/// Returns [`Error::InvalidBinary`] when the string runs past the end of
/// the word vector or is not valid UTF-8, and [`Error::OutOfMemory`]
/// when the copy cannot be reserved.
pub(crate) fn read_string(words: &[u32], start: usize) -> Result<(String, usize)> {
    let mut bytes: Vec<u8> = Vec::new();
    let mut offset = start;
    loop {
        let word = *words.get(offset).ok_or(Error::InvalidBinary)?;
        let raw = word.to_le_bytes();
        bytes.try_reserve(4).map_err(|_| Error::OutOfMemory)?;
        bytes.extend_from_slice(&raw);
        if raw.contains(&0) {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            bytes.truncate(end);
            let text = String::from_utf8(bytes).map_err(|_| Error::InvalidBinary)?;
            return Ok((text, offset - start + 1));
        }
        offset += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(header: &[u32; 5], body: &[u32]) -> Vec<u32> {
        let mut all = header.to_vec();
        all.extend_from_slice(body);
        all
    }

    #[test]
    fn decodes_instruction_boundaries() {
        // `2 << 16 | 17` = two-word OpCapability; `1 << 16 | 14` = one-word.
        let header = [MAGIC, 0x0001_0000, 0, 3, 0];
        let body = [(2 << 16) | 17, 6, (1 << 16) | 14];
        let (head, instrs) = decode(&words(&header, &body)).expect("decodes");
        assert_eq!(head.bound, 3);
        assert_eq!(instrs.len(), 2);
        assert_eq!(instrs[0].opcode, 17);
        assert_eq!(instrs[0].words(&words(&header, &body)), &[6]);
        assert_eq!(instrs[1].opcode, 14);
        assert!(instrs[1].words(&words(&header, &body)).is_empty());
    }

    #[test]
    fn rejects_truncated_module() {
        let header = [MAGIC, 0x0001_0000, 0, 3, 0];
        assert_eq!(decode(&header), Err(Error::InvalidBinary));
        // Word count claims three words, only one remains.
        let short = [MAGIC, 0x0001_0000, 0, 3, 0, (3 << 16) | 17, 6];
        assert_eq!(decode(&short), Err(Error::InvalidBinary));
        // Zero word count.
        let zero = [MAGIC, 0x0001_0000, 0, 3, 0, 17u32];
        assert_eq!(decode(&zero), Err(Error::InvalidBinary));
        // Wrong magic.
        let bad_magic = [0u32; 5];
        assert_eq!(decode(&bad_magic), Err(Error::InvalidBinary));
        // Zero bound.
        let no_bound = [MAGIC, 0x0001_0000, 0, 0, 0];
        assert_eq!(decode(&no_bound), Err(Error::InvalidBinary));
    }

    #[test]
    fn reads_padded_strings() {
        // "ba" packed little-endian with its NUL and pad in one word.
        let one = [u32::from_le_bytes([b'b', b'a', 0, 0])];
        let (text, used) = read_string(&one, 0).expect("reads");
        assert_eq!(text, "ba");
        assert_eq!(used, 1);
        // A string whose terminator falls in the second word.
        let two = [u32::from_le_bytes([b'b', b'a', b'c', b'd']), 0];
        let (text, used) = read_string(&two, 0).expect("reads across words");
        assert_eq!(text, "bacd");
        assert_eq!(used, 2);
        assert!(read_string(&[], 0).is_err());
        // Truncated run (no NUL before the end).
        let truncated = [u32::from_le_bytes([0x61, 0x62, 0x63, 0x64])];
        assert!(read_string(&truncated, 0).is_err());
        // Bytes before the NUL are not valid UTF-8.
        let broken = [u32::from_le_bytes([0xff, 0x00, 0x00, 0x00])];
        assert!(read_string(&broken, 0).is_err());
        // Out of range.
        assert!(read_string(&[0u32], 1).is_err());
    }
}
