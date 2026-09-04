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

//! Windows PE / DLL structural parser.
//!
//! Detects the MZ stub + `PE\0\0` signature and emits DOS header, PE signature,
//! COFF header, and optional header spans. Shared by `.dll`, `.exe`, `.sys`.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, ValidationInfo,
    read_u16_le, read_u32_le,
};

/// PE / DLL format parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct DllParser;

impl DllParser {
    /// Creates a PE/DLL parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Returns the PE signature file offset from the MZ `e_lfanew` field.
    #[must_use]
    pub fn pe_offset(data: &[u8]) -> Option<usize> {
        if data.len() < 0x40 || data[0] != b'M' || data[1] != b'Z' {
            return None;
        }
        let off = read_u32_le(data, 0x3C)?;
        usize::try_from(off).ok()
    }
}

impl FormatParser for DllParser {
    fn name() -> &'static str {
        "dll"
    }

    fn detect(data: &[u8]) -> bool {
        let Some(pe) = Self::pe_offset(data) else {
            return false;
        };
        data.get(pe..pe + 4) == Some(&b"PE\0\0"[..])
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        let Some(pe) = Self::pe_offset(data) else {
            return Ok(ValidationInfo::err("missing MZ header"));
        };
        if data.len() < pe + 24 {
            return Err(ParserError::InsufficientData);
        }
        if data.get(pe..pe + 4) != Some(&b"PE\0\0"[..]) {
            return Ok(ValidationInfo::err("missing PE signature"));
        }
        let machine = read_u16_le(data, pe + 4).ok_or(ParserError::InsufficientData)?;
        let characteristics =
            read_u16_le(data, pe + 22).ok_or(ParserError::InsufficientData)?;
        let kind = if characteristics & 0x2000 != 0 {
            "DLL"
        } else if characteristics & 0x0002 != 0 {
            "EXE"
        } else {
            "PE"
        };
        Ok(ValidationInfo::ok(format!(
            "valid {kind} (machine=0x{machine:04X})"
        )))
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
        let info = Self::validate(data)?;
        if !info.valid {
            return Err(ParserError::InvalidHeader(info.message));
        }
        let pe = Self::pe_offset(data).ok_or(ParserError::InsufficientData)?;
        let mut out = Vec::new();
        out.push(Segment::new(
            SegmentKind::Signature,
            0,
            2,
            u32::from_be_bytes(*b"MZ\0\0"),
        ));
        out.push(Segment::new(SegmentKind::Header, 0, pe.min(data.len()), 0));
        out.push(Segment::new(
            SegmentKind::Signature,
            pe,
            4,
            u32::from_be_bytes(*b"PE\0\0"),
        ));

        // COFF file header is 20 bytes after PE signature.
        let coff = pe + 4;
        if data.len() < coff + 20 {
            return Err(ParserError::InsufficientData);
        }
        out.push(Segment::new(SegmentKind::Header, coff, 20, 0));

        let opt_size =
            read_u16_le(data, coff + 16).ok_or(ParserError::InsufficientData)? as usize;
        let opt_start = coff + 20;
        if opt_size > 0 {
            if data.len() < opt_start + opt_size {
                return Err(ParserError::InsufficientData);
            }
            out.push(Segment::new(SegmentKind::Header, opt_start, opt_size, 0));
        }

        let sections =
            read_u16_le(data, coff + 2).ok_or(ParserError::InsufficientData)? as usize;
        let sec_start = opt_start + opt_size;
        let sec_len = sections.checked_mul(40).ok_or_else(|| {
            ParserError::TokenizationError("section table overflow".into())
        })?;
        if data.len() < sec_start + sec_len {
            return Err(ParserError::InsufficientData);
        }
        if sec_len > 0 {
            out.push(Segment::new(
                SegmentKind::Table,
                sec_start,
                sec_len,
                u32::try_from(sections).unwrap_or(u32::MAX),
            ));
        }
        Ok(out)
    }
}
