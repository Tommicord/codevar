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

//! ELF structural parser (`e_ident` + program/section header tables).
//!
//! Validates the System V ABI identification bytes and emits header /
//! table spans. Does not relocate or load segments.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, ValidationInfo,
    read_u16_le, read_u32_le,
};

const ELFMAG: [u8; 4] = [0x7F, b'E', b'L', b'F'];

/// ELF format parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct ElfParser;

impl ElfParser {
    /// Creates an ELF parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl FormatParser for ElfParser {
    fn name() -> &'static str {
        "elf"
    }

    fn detect(data: &[u8]) -> bool {
        data.len() >= 4 && data.starts_with(&ELFMAG)
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        if data.len() < 16 {
            return Err(ParserError::InsufficientData);
        }
        if !Self::detect(data) {
            return Ok(ValidationInfo::err("bad ELF magic"));
        }
        let class = data[4];
        let data_enc = data[5];
        let version = data[6];
        if class != 1 && class != 2 {
            return Ok(ValidationInfo::err("invalid ELF class"));
        }
        if data_enc != 1 && data_enc != 2 {
            return Ok(ValidationInfo::err("invalid ELF data encoding"));
        }
        if version != 1 {
            return Ok(ValidationInfo::err("invalid ELF version"));
        }
        Ok(ValidationInfo::ok(format!(
            "valid ELF{} ident",
            if class == 1 { "32" } else { "64" }
        )))
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
        let info = Self::validate(data)?;
        if !info.valid {
            return Err(ParserError::InvalidHeader(info.message));
        }
        let class = data[4];
        let mut out = vec![Segment::new(SegmentKind::Signature, 0, 16, 0)];

        // ELF header remainder.
        let ehsize = if class == 1 { 52usize } else { 64usize };
        if data.len() < ehsize {
            return Err(ParserError::InsufficientData);
        }
        out.push(Segment::new(
            SegmentKind::Header,
            16,
            ehsize - 16,
            u32::from(class),
        ));

        // Section header table (little-endian hosts are the common case; we
        // only emit the table span when e_shoff / e_shentsize / e_shnum parse).
        if let Some(table) = section_table_span(data, class) {
            out.push(table);
        }
        if let Some(table) = program_table_span(data, class) {
            out.push(table);
        }
        Ok(out)
    }
}

fn section_table_span(data: &[u8], class: u8) -> Option<Segment> {
    let (shoff, shentsize, shnum) = if class == 1 {
        (
            u64::from(read_u32_le(data, 32)?),
            read_u16_le(data, 46)?,
            read_u16_le(data, 48)?,
        )
    } else {
        let off = data.get(40..48)?;
        let shoff = u64::from_le_bytes([
            off[0], off[1], off[2], off[3], off[4], off[5], off[6], off[7],
        ]);
        (shoff, read_u16_le(data, 58)?, read_u16_le(data, 60)?)
    };
    if shoff == 0 || shentsize == 0 || shnum == 0 {
        return None;
    }
    let start = usize::try_from(shoff).ok()?;
    let len = usize::from(shentsize).checked_mul(usize::from(shnum))?;
    if start.checked_add(len)? > data.len() {
        return None;
    }
    Some(Segment::new(
        SegmentKind::Table,
        start,
        len,
        u32::from(shnum),
    ))
}

fn program_table_span(data: &[u8], class: u8) -> Option<Segment> {
    let (phoff, phentsize, phnum) = if class == 1 {
        (
            u64::from(read_u32_le(data, 28)?),
            read_u16_le(data, 42)?,
            read_u16_le(data, 44)?,
        )
    } else {
        let off = data.get(32..40)?;
        let phoff = u64::from_le_bytes([
            off[0], off[1], off[2], off[3], off[4], off[5], off[6], off[7],
        ]);
        (phoff, read_u16_le(data, 54)?, read_u16_le(data, 56)?)
    };
    if phoff == 0 || phentsize == 0 || phnum == 0 {
        return None;
    }
    let start = usize::try_from(phoff).ok()?;
    let len = usize::from(phentsize).checked_mul(usize::from(phnum))?;
    if start.checked_add(len)? > data.len() {
        return None;
    }
    Some(Segment::new(
        SegmentKind::Table,
        start,
        len,
        u32::from(phnum),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_elf_magic() {
        let mut data = vec![0u8; 52];
        data[..4].copy_from_slice(&ELFMAG);
        data[4] = 1; // ELFCLASS32
        data[5] = 1; // ELFDATA2LSB
        data[6] = 1; // EV_CURRENT
        assert!(ElfParser::detect(&data));
        assert!(
            ElfParser::validate(&data)
                .unwrap_or(ValidationInfo::err("x"))
                .valid
        );
    }
}
