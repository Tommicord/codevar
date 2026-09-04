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

//! Android APK parser (ZIP container with Android packaging cues).
//!
//! APKs are ZIP archives. This module validates the ZIP local/EOCD signatures
//! and looks for packaging hints (`AndroidManifest.xml`, `classes.dex`) in
//! local-file names without inflating entries.

use crate::parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, ValidationInfo,
    read_u16_le, read_u32_le,
};

const ZIP_LOCAL: [u8; 4] = [0x50, 0x4B, 0x03, 0x04];
const ZIP_EOCD: [u8; 4] = [0x50, 0x4B, 0x05, 0x06];
const ZIP_CENTRAL: [u8; 4] = [0x50, 0x4B, 0x01, 0x02];

/// APK (ZIP-based) format parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct ApkParser;

impl ApkParser {
    /// Creates an APK parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Returns `true` when local headers mention typical APK entries.
    #[must_use]
    pub fn looks_like_apk(data: &[u8]) -> bool {
        if !data.starts_with(&ZIP_LOCAL) && !data.starts_with(&ZIP_EOCD) {
            return false;
        }
        // Scan a bounded window of local headers for Android cues.
        let mut i = 0usize;
        let mut saw_zip = false;
        let mut cues = 0u32;
        while i + 30 <= data.len() && i < 1_048_576 {
            if data.get(i..i + 4) != Some(&ZIP_LOCAL) {
                i += 1;
                continue;
            }
            saw_zip = true;
            let name_len = read_u16_le(data, i + 26).unwrap_or(0) as usize;
            let extra_len = read_u16_le(data, i + 28).unwrap_or(0) as usize;
            let comp = read_u32_le(data, i + 18).unwrap_or(0) as usize;
            let name_at = i + 30;
            if let Some(name) = data.get(name_at..name_at + name_len)
                && (name == b"AndroidManifest.xml"
                    || name == b"classes.dex"
                    || name.starts_with(b"META-INF/")
                    || name.starts_with(b"res/"))
            {
                cues += 1;
            }
            let next = name_at
                .checked_add(name_len)
                .and_then(|n| n.checked_add(extra_len))
                .and_then(|n| n.checked_add(comp));
            match next {
                Some(n) if n > i => i = n,
                _ => break,
            }
        }
        saw_zip && cues > 0
    }
}

impl FormatParser for ApkParser {
    fn name() -> &'static str {
        "apk"
    }

    fn detect(data: &[u8]) -> bool {
        Self::looks_like_apk(data)
            || (data.starts_with(&ZIP_LOCAL) && data.windows(4).any(|w| w == ZIP_EOCD))
    }

    fn validate(data: &[u8]) -> ParserResult<ValidationInfo> {
        if !(data.starts_with(&ZIP_LOCAL)
            || data.starts_with(&ZIP_EOCD)
            || data.starts_with(&ZIP_CENTRAL))
        {
            return Ok(ValidationInfo::err("not a ZIP/APK container"));
        }
        let apk = Self::looks_like_apk(data);
        if apk {
            Ok(ValidationInfo::ok("valid APK ZIP with Android cues"))
        } else {
            Ok(ValidationInfo::ok(
                "valid ZIP container (APK cues not found)",
            ))
        }
    }

    fn segments(data: &[u8]) -> ParserResult<Vec<Segment>> {
        if data.len() < 4 {
            return Err(ParserError::InsufficientData);
        }
        let mut out = Vec::new();
        let mut i = 0usize;
        while i + 4 <= data.len() {
            let sig = &data[i..i + 4];
            if sig == ZIP_LOCAL {
                i = push_local(data, i, &mut out)?;
            } else if sig == ZIP_CENTRAL {
                i = push_central(data, i, &mut out)?;
            } else if sig == ZIP_EOCD {
                push_eocd(data, i, &mut out)?;
                break;
            } else {
                i += 1;
            }
        }
        if out.is_empty() {
            return Err(ParserError::InvalidHeader("no ZIP records found".into()));
        }
        Ok(out)
    }
}

fn push_local(data: &[u8], i: usize, out: &mut Vec<Segment>) -> ParserResult<usize> {
    if i + 30 > data.len() {
        return Err(ParserError::InsufficientData);
    }
    let name_len =
        read_u16_le(data, i + 26).ok_or(ParserError::InsufficientData)? as usize;
    let extra_len =
        read_u16_le(data, i + 28).ok_or(ParserError::InsufficientData)? as usize;
    let comp = read_u32_le(data, i + 18).ok_or(ParserError::InsufficientData)? as usize;
    let header_end = (i + 30)
        .checked_add(name_len + extra_len)
        .ok_or_else(|| ParserError::TokenizationError("local header overflow".into()))?;
    let entry_end = header_end
        .checked_add(comp)
        .ok_or_else(|| ParserError::TokenizationError("local entry overflow".into()))?;
    if entry_end > data.len() {
        return Err(ParserError::InsufficientData);
    }
    out.push(Segment::new(
        SegmentKind::Header,
        i,
        header_end - i,
        0x0403_4B50,
    ));
    if comp > 0 {
        out.push(Segment::new(SegmentKind::Payload, header_end, comp, 0));
    }
    Ok(entry_end)
}

fn push_central(data: &[u8], i: usize, out: &mut Vec<Segment>) -> ParserResult<usize> {
    if i + 46 > data.len() {
        return Err(ParserError::InsufficientData);
    }
    let name_len =
        read_u16_le(data, i + 28).ok_or(ParserError::InsufficientData)? as usize;
    let extra_len =
        read_u16_le(data, i + 30).ok_or(ParserError::InsufficientData)? as usize;
    let comment_len =
        read_u16_le(data, i + 32).ok_or(ParserError::InsufficientData)? as usize;
    let end = (i + 46)
        .checked_add(name_len + extra_len + comment_len)
        .ok_or_else(|| ParserError::TokenizationError("central overflow".into()))?;
    if end > data.len() {
        return Err(ParserError::InsufficientData);
    }
    out.push(Segment::new(SegmentKind::Table, i, end - i, 0x0201_4B50));
    Ok(end)
}

fn push_eocd(data: &[u8], i: usize, out: &mut Vec<Segment>) -> ParserResult<usize> {
    if i + 22 > data.len() {
        return Err(ParserError::InsufficientData);
    }
    let comment_len =
        read_u16_le(data, i + 20).ok_or(ParserError::InsufficientData)? as usize;
    let end = (i + 22)
        .checked_add(comment_len)
        .ok_or_else(|| ParserError::TokenizationError("eocd overflow".into()))?
        .min(data.len());
    out.push(Segment::new(SegmentKind::Trailer, i, end - i, 0x0605_4B50));
    Ok(end)
}
