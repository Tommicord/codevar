//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License.  You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing
//! permissions and limitations under the License.

//! Character-to-glyph mapping: the `cmap` table and its subtables.
//!
//! Port of `ttcmap.c` from FreeType 2.6.  [`parse_charmaps`] is
//! `tt_face_build_cmaps` (record walk plus per-format validation),
//! the [`CmapSubtable`] accessors are the `tt_cmapNN_char_index` /
//! `tt_cmapNN_char_next` callbacks and [`sfnt_find_encoding`] is the
//! helper of the same name from `sfobjs.c`.  [`SFNT_CMAP_CLASS`] is the
//! single static [`CMapClass`] that installs a validated subtable into a
//! [`CharMap`] of `codevar_truetype_core`.
//!
//! | FreeType item | Item here |
//! |---------------|-----------|
//! | `tt_cmap0_*`  | format 0 of [`CmapSubtable::char_index`] |
//! | `tt_cmap2_*`  | format 2 |
//! | `tt_cmap4_*`  | format 4 ([`CMAP_FLAG_UNSORTED`], [`CMAP_FLAG_OVERLAPPING`]) |
//! | `tt_cmap6_*`  | format 6 |
//! | `tt_cmap8_*`  | format 8 |
//! | `tt_cmap10_*` | format 10 |
//! | `tt_cmap12_*` | format 12 |
//! | `tt_face_build_cmaps` | [`parse_charmaps`] |
//! | `sfnt_find_encoding` | [`sfnt_find_encoding`] |
//!
//! ## Deviations from FreeType 2.6
//!
//! * Formats 13 and 14 are not ported (`tt_cmap13_*`, `tt_cmap14_*`);
//!   such subtables are reported by [`parse_charmaps`] as unsupported
//!   and skipped, exactly like an unknown format.
//! * Validation runs at `FT_VALIDATE_DEFAULT` level, the level
//!   `tt_face_build_cmaps` uses.  The `PARANOID`/`TIGHT` checks (which
//!   also need `face->max_profile.numGlyphs`) are therefore not
//!   performed.
//! * Format 4 keeps no iteration state (`TT_CMap4Rec`): every lookup is
//!   stateless.  Validation guarantees that `startCount` and `endCount`
//!   are non-decreasing unless [`CMAP_FLAG_UNSORTED`] is set, so the
//!   C state machine of `tt_cmap4_next`, the `TT_CMAP_FLAG_OVERLAPPING`
//!   fixup of `tt_cmap4_char_map_binary` and a binary search for the
//!   first segment with `endCount >= char_code` all select the same
//!   segment.  Only [`CMAP_FLAG_UNSORTED`] subtables keep the linear
//!   scan of `tt_cmap4_char_map_linear`.
//! * `char_next` writes the caller's character code only when a glyph
//!   index was found.  C stores `0` on exhaustion (formats 0, 2, 6, 8)
//!   or the code just past the covered range (format 10), which the
//!   core `CharMap::char_next` wrapper would report as a hit.
//! * The `limit` used by the last-segment fixup of the format 4
//!   lookups is the end of the whole `cmap` table
//!   (`face->cmap_table + face->cmap_size`); every
//!   [`CmapSubtable::data`] slice ends exactly there.

use alloc::boxed::Box;
use alloc::vec::Vec;
use bytes::Buf;
use codevar_truetype_core::{CMapClass, CharMap, Encoding, TtError, TtResult};
use core::any::Any;
use core::mem::size_of;

use crate::tags::{
    MAC_ID_ROMAN, MS_ID_BIG_5, MS_ID_JOHAB, MS_ID_PRC, MS_ID_SHIFT_JIS, MS_ID_SYMBOL_CS, MS_ID_UCS_4,
    MS_ID_UNICODE_CS, MS_ID_WANSUNG, PLATFORM_ISO, PLATFORM_MACINTOSH, PLATFORM_MICROSOFT, PLATFORM_UNICODE,
};

/// `TT_CMAP_FLAG_UNSORTED`: the segments of a format 4 subtable are not
/// ordered, so lookups must scan linearly (`ttcmap.h`, set by
/// `tt_cmap4_validate`).
pub const CMAP_FLAG_UNSORTED: u32 = 1;

/// `TT_CMAP_FLAG_OVERLAPPING`: the segments of a format 4 subtable
/// overlap in ascending order (`ttcmap.h`, set by `tt_cmap4_validate`).
pub const CMAP_FLAG_OVERLAPPING: u32 = 2;

/// One validated `cmap` subtable: the format number, the format 4 flags
/// and the bytes of the subtable (`TT_CMapRec`'s `data` plus `flags`).
///
/// `data` starts at the subtable header and runs to the end of the
/// whole `cmap` table, which is the validator limit
/// (`ft_validator_init(cmap, limit, ...)`) that `tt_face_build_cmaps`
/// passes to every `validate` function.  Instances are produced by
/// [`parse_charmaps`]; the slices are bounds-checked on every access, so
/// a hand-built value can never panic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CmapSubtable<'a> {
    /// The subtable format (`0`, `2`, `4`, `6`, `8`, `10` or `12`).
    pub format: u16,
    /// [`CMAP_FLAG_UNSORTED`] / [`CMAP_FLAG_OVERLAPPING`] for format 4,
    /// `0` for every other format (the `flags` member of `TT_CMapRec`).
    pub flags: u32,
    /// The subtable bytes, extending to the end of the `cmap` table.
    pub data: &'a [u8],
}

impl<'a> CmapSubtable<'a> {
    /// `FT_Get_Char_Index` (`tt_cmapNN_char_index`): the glyph index of
    /// `char_code`, or `0` when the code is not mapped.
    #[inline]
    pub fn char_index(&self, char_code: u32) -> u32 {
        subtable_char_index(self.format, self.flags, self.data, char_code)
    }

    /// `FT_Get_Next_Char` (`tt_cmapNN_char_next`): the first mapped
    /// character *after* `char_code` with its glyph index, or `None`
    /// when the map is exhausted.
    ///
    /// Like the C callbacks, `char_code` is advanced with wrapping
    /// arithmetic; the caller's value is only updated when a glyph was
    /// found (see the module docs).
    #[inline]
    pub fn char_next(&self, char_code: u32) -> Option<(u32, u32)> {
        subtable_char_next(self.format, self.flags, self.data, char_code)
    }
}

/// One encoding record of the `cmap` table plus its validated subtable
/// (`FT_CharMapRec` without the owning face).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SfntCMap<'a> {
    /// The `platformID` of the encoding record.
    pub platform_id: u16,
    /// The `encodingID` of the encoding record.
    pub encoding_id: u16,
    /// The [`Encoding`] derived from the identifiers
    /// ([`sfnt_find_encoding`]).
    pub encoding: Encoding,
    /// The subtable the record points at.
    pub subtable: CmapSubtable<'a>,
}

impl<'a> SfntCMap<'a> {
    /// [`CmapSubtable::char_index`] through this record's subtable.
    #[inline]
    pub fn char_index(&self, char_code: u32) -> u32 {
        self.subtable.char_index(char_code)
    }

    /// [`CmapSubtable::char_next`] through this record's subtable.
    #[inline]
    pub fn char_next(&self, char_code: u32) -> Option<(u32, u32)> {
        self.subtable.char_next(char_code)
    }
}

/// `sfnt_find_encoding` (`sfobjs.c`): the [`Encoding`] of an encoding
/// record.
///
/// The table is scanned in order and the first entry with a matching
/// platform ID whose encoding ID matches or is the wildcard `-1` wins;
/// records the table does not describe report [`Encoding::NONE`].
#[inline]
pub fn sfnt_find_encoding(platform_id: i32, encoding_id: i32) -> Encoding {
    const TT_ENCODINGS: [(i32, i32, Encoding); 11] = [
        (PLATFORM_ISO as i32, -1, Encoding::UNICODE),
        (PLATFORM_UNICODE as i32, -1, Encoding::UNICODE),
        (
            PLATFORM_MACINTOSH as i32,
            MAC_ID_ROMAN as i32,
            Encoding::APPLE_ROMAN,
        ),
        (
            PLATFORM_MICROSOFT as i32,
            MS_ID_SYMBOL_CS as i32,
            Encoding::MS_SYMBOL,
        ),
        (PLATFORM_MICROSOFT as i32, MS_ID_UCS_4 as i32, Encoding::UNICODE),
        (
            PLATFORM_MICROSOFT as i32,
            MS_ID_UNICODE_CS as i32,
            Encoding::UNICODE,
        ),
        (PLATFORM_MICROSOFT as i32, MS_ID_SHIFT_JIS as i32, Encoding::SJIS),
        (PLATFORM_MICROSOFT as i32, MS_ID_PRC as i32, Encoding::GB2312),
        (PLATFORM_MICROSOFT as i32, MS_ID_BIG_5 as i32, Encoding::BIG5),
        (PLATFORM_MICROSOFT as i32, MS_ID_WANSUNG as i32, Encoding::WANSUNG),
        (PLATFORM_MICROSOFT as i32, MS_ID_JOHAB as i32, Encoding::JOHAB),
    ];
    TT_ENCODINGS
        .iter()
        .find(|(platform, encoding, _)| {
            *platform == platform_id && (*encoding == encoding_id || *encoding == -1)
        })
        .map_or(Encoding::NONE, |(_, _, encoding)| *encoding)
}

/// `tt_face_build_cmaps` (`ttcmap.c`): parses the `cmap` table and
/// returns every subtable that validates.
///
/// The version field must be `0` and the table at least 4 bytes long
/// ([`TtError::INVALID_TABLE`]); encoding records with an offset of
/// `0` or beyond `table.len() - 2`, records whose subtable fails its
/// format validator and records naming an unsupported format are
/// skipped rather than failing the whole table — matching C, where only
/// the header checks can make `tt_face_build_cmaps` fail.
pub fn parse_charmaps(table: &[u8]) -> TtResult<Vec<SfntCMap<'_>>> {
    if table.len() < 4 {
        return Err(TtError::INVALID_TABLE);
    }
    if u16_at(table, 0).ok_or(TtError::INVALID_TABLE)? != 0 {
        return Err(TtError::INVALID_TABLE);
    }
    let num_cmaps = u16_at(table, 2).ok_or(TtError::INVALID_TABLE)?;
    let mut charmaps = Vec::new();
    let mut pos = 4usize;
    for _ in 0..num_cmaps {
        let Some(end) = pos.checked_add(8) else {
            break;
        };
        let Some(record) = table.get(pos..end) else {
            break;
        };
        let mut record: &[u8] = record;
        let platform_id = record
            .try_get_u16()
            .map_err(|_| TtError::INVALID_TABLE)?;
        let encoding_id = record
            .try_get_u16()
            .map_err(|_| TtError::INVALID_TABLE)?;
        let offset = record
            .try_get_u32()
            .map_err(|_| TtError::INVALID_TABLE)? as usize;
        pos = end;
        if offset == 0 {
            continue;
        }
        let Some(data) = table.get(offset..) else {
            continue;
        };
        if data.len() < 2 {
            continue;
        }
        let format = match u16_at(data, 0) {
            Some(format) => format,
            None => continue,
        };
        let Ok(flags) = validate_subtable(format, data) else {
            continue;
        };
        charmaps.push(SfntCMap {
            platform_id,
            encoding_id,
            encoding: sfnt_find_encoding(i32::from(platform_id), i32::from(encoding_id)),
            subtable: CmapSubtable { format, flags, data },
        });
    }
    Ok(charmaps)
}

/// The heap-allocated payload of [`SFNT_CMAP_CLASS`]: a validated
/// subtable owned by the [`CharMap`] (`TT_CMapRec`'s `data` member
/// points at the table bytes in C).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedCmapSubtable {
    /// The subtable format.
    pub format: u16,
    /// The format 4 flags (see [`CMAP_FLAG_UNSORTED`]).
    pub flags: u32,
    /// The bytes of the subtable, copied out of the input buffer.
    pub data: Vec<u8>,
}

impl OwnedCmapSubtable {
    /// [`CmapSubtable::char_index`] on the owned copy.
    #[inline]
    pub fn char_index(&self, char_code: u32) -> u32 {
        subtable_char_index(self.format, self.flags, &self.data, char_code)
    }

    /// [`CmapSubtable::char_next`] on the owned copy.
    #[inline]
    pub fn char_next(&self, char_code: u32) -> Option<(u32, u32)> {
        subtable_char_next(self.format, self.flags, &self.data, char_code)
    }
}

/// The static [`CMapClass`] installed by
/// [`crate::SfntFont::install_charmaps`].
///
/// It is format-agnostic: the format travels inside the
/// [`OwnedCmapSubtable`] payload of [`CharMap::data`], so one class
/// serves every subtable `tt_face_build_cmaps` accepts.
pub static SFNT_CMAP_CLASS: CMapClass = CMapClass {
    size: size_of::<OwnedCmapSubtable>() as u64,
    init: sfnt_cmap_init,
    done: sfnt_cmap_done,
    char_index: sfnt_cmap_char_index,
    char_next: sfnt_cmap_char_next,
    char_var_index: None,
    char_var_default: None,
    variant_list: None,
    charvariant_list: None,
    variantchar_list: None,
};

/// `init` of [`SFNT_CMAP_CLASS`]: clones the [`OwnedCmapSubtable`]
/// handed over as `init_data` into [`CharMap::data`].
fn sfnt_cmap_init(cmap: &CharMap, init_data: Option<&(dyn Any + Send + Sync)>) -> TtResult<()> {
    let payload = init_data.ok_or(TtError::INVALID_ARGUMENT)?;
    let owned = payload
        .downcast_ref::<OwnedCmapSubtable>()
        .ok_or(TtError::INVALID_ARGUMENT)?;
    let _previous = cmap.data.replace(Some(Box::new(owned.clone())))?;
    Ok(())
}

/// `done` of [`SFNT_CMAP_CLASS`]: releases the payload.
fn sfnt_cmap_done(cmap: &CharMap) {
    let _previous = cmap.data.replace(None);
}

/// `char_index` of [`SFNT_CMAP_CLASS`]: dispatches through the stored
/// payload; `0` when the payload is missing or borrowed.
fn sfnt_cmap_char_index(cmap: &CharMap, char_code: u32) -> u32 {
    let Ok(guard) = cmap.data.borrow() else {
        return 0;
    };
    let Some(payload) = guard.as_ref() else {
        return 0;
    };
    let Some(owned) = payload.downcast_ref::<OwnedCmapSubtable>() else {
        return 0;
    };
    owned.char_index(char_code)
}

/// `char_next` of [`SFNT_CMAP_CLASS`]: dispatches through the stored
/// payload; `0` (leaving `*pchar_code` untouched) when the payload is
/// missing or the map is exhausted.
fn sfnt_cmap_char_next(cmap: &CharMap, pchar_code: &mut u32) -> u32 {
    let Ok(guard) = cmap.data.borrow() else {
        return 0;
    };
    let Some(payload) = guard.as_ref() else {
        return 0;
    };
    let Some(owned) = payload.downcast_ref::<OwnedCmapSubtable>() else {
        return 0;
    };
    match owned.char_next(*pchar_code) {
        Some((char_code, gindex)) => {
            *pchar_code = char_code;
            gindex
        }
        None => 0,
    }
}

/// Dispatches [`CmapSubtable::char_index`] to the format reader.
fn subtable_char_index(format: u16, flags: u32, data: &[u8], char_code: u32) -> u32 {
    match format {
        0 => format0_char_index(data, char_code),
        2 => format2_char_index(data, char_code),
        4 => format4_char_index(data, flags, char_code),
        6 => format6_char_index(data, char_code),
        8 => format8_char_index(data, char_code),
        10 => format10_char_index(data, char_code),
        12 => format12_char_index(data, char_code),
        _ => 0,
    }
}

/// Dispatches [`CmapSubtable::char_next`] to the format reader.
fn subtable_char_next(format: u16, flags: u32, data: &[u8], char_code: u32) -> Option<(u32, u32)> {
    match format {
        0 => format0_char_next(data, char_code),
        2 => format2_char_next(data, char_code),
        4 => format4_char_next(data, flags, char_code),
        6 => format6_char_next(data, char_code),
        8 => format8_char_next(data, char_code),
        10 => format10_char_next(data, char_code),
        12 => format12_char_next(data, char_code),
        _ => None,
    }
}

/// Validates `data` as the subtable of `format`, returning the format 4
/// flags (`0` for the other formats); the `validate` callback of the
/// matching `TT_CMap_ClassRec`.
fn validate_subtable(format: u16, data: &[u8]) -> TtResult<u32> {
    match format {
        0 => validate_format0(data),
        2 => validate_format2(data),
        4 => validate_format4(data),
        6 => validate_format6(data),
        8 => validate_format8(data),
        10 => validate_format10(data),
        12 => validate_format12(data),
        _ => Err(TtError::INVALID_CHARMAP_FORMAT),
    }
}

#[inline]
fn u16_at(data: &[u8], pos: usize) -> Option<u16> {
    data.get(pos..pos.checked_add(2)?)?
        .try_get_u16()
        .ok()
}

#[inline]
fn i16_at(data: &[u8], pos: usize) -> Option<i16> {
    data.get(pos..pos.checked_add(2)?)?
        .try_get_i16()
        .ok()
}

#[inline]
fn u32_at(data: &[u8], pos: usize) -> Option<u32> {
    data.get(pos..pos.checked_add(4)?)?
        .try_get_u32()
        .ok()
}

/// `tt_cmap0_validate` (`ttcmap.c`): format 0 needs its 262-byte
/// header, of which the validator only reads the `length` field at the
/// default level.
fn validate_format0(data: &[u8]) -> TtResult<u32> {
    if data.len() < 4 {
        return Err(TtError::INVALID_TABLE);
    }
    let length = usize::from(u16_at(data, 2).ok_or(TtError::INVALID_TABLE)?);
    if length > data.len() || length < 262 {
        return Err(TtError::INVALID_TABLE);
    }
    Ok(0)
}

/// `tt_cmap0_char_index`: the byte at `6 + char_code`.
fn format0_char_index(data: &[u8], char_code: u32) -> u32 {
    if char_code >= 256 {
        return 0;
    }
    data.get(6 + char_code as usize)
        .map_or(0, |byte| u32::from(*byte))
}

/// `tt_cmap0_char_next`: the first non-zero byte after `char_code`.
fn format0_char_next(data: &[u8], char_code: u32) -> Option<(u32, u32)> {
    let mut code = char_code.wrapping_add(1);
    while code < 256 {
        match data.get(6 + code as usize) {
            Some(&gindex) if gindex != 0 => return Some((code, u32::from(gindex))),
            Some(_) => code += 1,
            None => return None,
        }
    }
    None
}

/// `tt_cmap2_validate` (`ttcmap.c`): checks `length`, derives the
/// sub-header count from the key table and validates every non-empty
/// sub-header's glyph-id offset against the table.
fn validate_format2(data: &[u8]) -> TtResult<u32> {
    if data.len() < 4 {
        return Err(TtError::INVALID_TABLE);
    }
    let length = usize::from(u16_at(data, 2).ok_or(TtError::INVALID_TABLE)?);
    if length > data.len() || length < 518 {
        return Err(TtError::INVALID_TABLE);
    }
    let mut max_subs = 0usize;
    for index in 0..256usize {
        let key = u16_at(data, 6 + index * 2).ok_or(TtError::INVALID_TABLE)?;
        max_subs = max_subs.max(usize::from(key >> 3));
    }
    let glyph_ids = 518 + (max_subs + 1) * 8;
    if glyph_ids > data.len() {
        return Err(TtError::INVALID_TABLE);
    }
    for sub_index in 0..=max_subs {
        let sub = 518 + sub_index * 8;
        let count = u16_at(data, sub + 2).ok_or(TtError::INVALID_TABLE)?;
        let offset = u16_at(data, sub + 6).ok_or(TtError::INVALID_TABLE)?;
        if count == 0 || offset == 0 {
            continue;
        }
        let ids = sub + 6 + usize::from(offset);
        let span = usize::from(count) * 2;
        if ids < glyph_ids
            || ids
                .checked_add(span)
                .is_none_or(|end| end > length)
        {
            return Err(TtError::INVALID_TABLE);
        }
    }
    Ok(0)
}

/// `tt_cmap2_get_subheader`: the sub-header of `char_code`, or `None`
/// for a code the key table rejects (an 8-bit code whose key is not
/// `0`, or a 16-bit code whose key selects sub-header 0).
fn format2_subheader(data: &[u8], char_code: u32) -> Option<usize> {
    if char_code >= 0x1_0000 {
        return None;
    }
    let char_lo = usize::from(char_code as u8);
    let char_hi = usize::from((char_code >> 8) as u8);
    if char_hi == 0 {
        if u16_at(data, 6 + char_lo * 2)? != 0 {
            return None;
        }
        Some(518)
    } else {
        let key = u16_at(data, 6 + char_hi * 2)?;
        let sub = 518 + (usize::from(key) & !7);
        if sub == 518 {
            return None;
        }
        Some(sub)
    }
}

/// `tt_cmap2_char_index`: resolves the sub-header, then maps the low
/// byte through its `(first, count, delta, offset)` record.
fn format2_char_index(data: &[u8], char_code: u32) -> u32 {
    let Some(sub) = format2_subheader(data, char_code) else {
        return 0;
    };
    let (Some(start), Some(count), Some(delta), Some(offset)) = (
        u16_at(data, sub),
        u16_at(data, sub + 2),
        i16_at(data, sub + 4),
        u16_at(data, sub + 6),
    ) else {
        return 0;
    };
    let mut index = u32::from(char_code as u8);
    index = index.wrapping_sub(u32::from(start));
    if index >= u32::from(count) || offset == 0 {
        return 0;
    }
    let Some(pos) = (sub + 6)
        .checked_add(usize::from(offset))
        .and_then(|base| base.checked_add(usize::try_from(index).ok()? * 2))
    else {
        return 0;
    };
    let Some(gindex) = u16_at(data, pos) else {
        return 0;
    };
    if gindex == 0 {
        return 0;
    }
    ((i32::from(gindex) + i32::from(delta)) as u16).into()
}

/// `tt_cmap2_char_next`: walks the sub-headers by high byte, scanning
/// the covered low bytes of each one.
fn format2_char_next(data: &[u8], char_code: u32) -> Option<(u32, u32)> {
    let mut code = char_code.wrapping_add(1);
    while code < 0x1_0000 {
        let Some(sub) = format2_subheader(data, code) else {
            code = (code & !0xFF) + 256;
            continue;
        };
        let (Some(start), Some(count), Some(delta), Some(offset)) = (
            u16_at(data, sub),
            u16_at(data, sub + 2),
            i16_at(data, sub + 4),
            u16_at(data, sub + 6),
        ) else {
            code = (code & !0xFF) + 256;
            continue;
        };
        if offset == 0 {
            code = (code & !0xFF) + 256;
            continue;
        }
        let base = sub + 6;
        let ids = base.checked_add(usize::from(offset))?;
        let char_lo = u32::from(code as u8);
        let (char_lo, mut pos) = if char_lo < u32::from(start) {
            (u32::from(start), 0u32)
        } else {
            (char_lo, char_lo - u32::from(start))
        };
        code = (code & !0xFF) + char_lo;
        while pos < u32::from(count) {
            let gindex = usize::try_from(pos)
                .ok()
                .and_then(|pos| ids.checked_add(pos * 2))
                .and_then(|pos| u16_at(data, pos))
                .unwrap_or(0);
            if gindex != 0 {
                let mapped = ((i32::from(gindex) + i32::from(delta)) as u16).into();
                if mapped != 0 {
                    return Some((code, mapped));
                }
            }
            pos += 1;
            code += 1;
        }
        code = (code & !0xFF) + 256;
    }
    None
}

/// The segment records of a format 4 subtable (`tt_cmap4_set_range`'s
/// `cur_*` members, without the iteration state).
#[derive(Clone, Copy, Debug)]
struct Format4Segment {
    /// `startCount` of the segment.
    start: u16,
    /// `endCount` of the segment.
    end: u16,
    /// `idDelta` of the segment (after the last-segment fixup).
    delta: i16,
    /// `idRangeOffset` of the segment (after the last-segment fixup).
    offset: u16,
    /// Absolute position of the `idRangeOffset` field inside the
    /// subtable.
    offset_field: usize,
}

/// The byte offset of the `startCount` array (`16 + segCountX2`).
#[inline]
fn format4_n2(data: &[u8]) -> Option<usize> {
    Some(usize::from(u16_at(data, 6)?) & !1)
}

/// Reads segment `index`, applying the last-segment fixup of
/// `tt_cmap4_set_range` (broken `0xFFFF` segments of real-world fonts
/// get `idDelta = 1`, `idRangeOffset = 0`).
fn format4_segment(data: &[u8], n2: usize, index: usize) -> Option<Format4Segment> {
    let num_segs = n2 / 2;
    if index >= num_segs {
        return None;
    }
    let starts = 16 + n2;
    let deltas = starts + n2;
    let offsets = deltas + n2;
    let start = u16_at(data, starts + index * 2)?;
    let end = u16_at(data, 14 + index * 2)?;
    let mut delta = i16_at(data, deltas + index * 2)?;
    let offset_field = offsets + index * 2;
    let mut offset = u16_at(data, offset_field)?;
    if index + 1 >= num_segs && start == 0xFFFF && end == 0xFFFF && offset != 0 {
        let limit = offset_field
            .checked_add(usize::from(offset))
            .and_then(|ids| ids.checked_add(2));
        if limit.is_none_or(|limit| limit > data.len()) {
            delta = 1;
            offset = 0;
        }
    }
    Some(Format4Segment {
        start,
        end,
        delta,
        offset,
        offset_field,
    })
}

/// The first segment whose `endCount` is `>= code` (the binary search
/// of `tt_cmap4_char_map_binary`; `num_segs` when there is none).
fn format4_lower_bound(data: &[u8], code: u32, num_segs: usize) -> usize {
    let mut low = 0usize;
    let mut high = num_segs;
    while low < high {
        let mid = low + (high - low) / 2;
        match u16_at(data, 14 + mid * 2) {
            Some(end) if u32::from(end) < code => low = mid + 1,
            _ => high = mid,
        }
    }
    low
}

/// The glyph index of `char_code` inside `segment`
/// (`idRangeOffset == 0` adds `idDelta` to the code, otherwise to the
/// stored glyph id).
fn format4_gindex(data: &[u8], segment: &Format4Segment, char_code: u32) -> u32 {
    if segment.offset == 0xFFFF {
        return 0;
    }
    if segment.offset != 0 {
        let index = usize::try_from(char_code.wrapping_sub(u32::from(segment.start))).unwrap_or(0);
        let Some(pos) = segment
            .offset_field
            .checked_add(usize::from(segment.offset))
            .and_then(|ids| ids.checked_add(index * 2))
        else {
            return 0;
        };
        let Some(gindex) = u16_at(data, pos) else {
            return 0;
        };
        if gindex == 0 {
            return 0;
        }
        return ((i32::from(gindex) + i32::from(segment.delta)) as u16).into();
    }
    ((char_code as i32).wrapping_add(i32::from(segment.delta)) as u16).into()
}

/// `tt_cmap4_validate` (`ttcmap.c`): checks `length` (clamped at the
/// default level), the segment arrays and every `idRangeOffset`, and
/// reports [`CMAP_FLAG_UNSORTED`] / [`CMAP_FLAG_OVERLAPPING`] for
/// overlapping segments.
fn validate_format4(data: &[u8]) -> TtResult<u32> {
    if data.len() < 4 {
        return Err(TtError::INVALID_TABLE);
    }
    let mut length = usize::from(u16_at(data, 2).ok_or(TtError::INVALID_TABLE)?);
    if length > data.len() {
        length = data.len();
    }
    if length < 16 {
        return Err(TtError::INVALID_TABLE);
    }
    let n2 = usize::from(u16_at(data, 6).ok_or(TtError::INVALID_TABLE)?) & !1;
    let num_segs = n2 / 2;
    if length < 16 + num_segs * 8 {
        return Err(TtError::INVALID_TABLE);
    }
    let starts = 16 + n2;
    let deltas = starts + n2;
    let offsets = deltas + n2;
    let glyph_ids = offsets + n2;
    let mut flags = 0u32;
    let (mut last_start, mut last_end) = (0u16, 0u16);
    for index in 0..num_segs {
        let start = u16_at(data, starts + index * 2).ok_or(TtError::INVALID_TABLE)?;
        let end = u16_at(data, 14 + index * 2).ok_or(TtError::INVALID_TABLE)?;
        let _delta = i16_at(data, deltas + index * 2).ok_or(TtError::INVALID_TABLE)?;
        let offset = u16_at(data, offsets + index * 2).ok_or(TtError::INVALID_TABLE)?;
        if start > end {
            return Err(TtError::INVALID_TABLE);
        }
        if index > 0 && start <= last_end {
            if last_start > start || last_end > end {
                flags |= CMAP_FLAG_UNSORTED;
            } else {
                flags |= CMAP_FLAG_OVERLAPPING;
            }
        }
        let sentinel = index + 1 == num_segs && start == 0xFFFF && end == 0xFFFF;
        if offset != 0 && offset != 0xFFFF {
            let ids = (offsets + index * 2)
                .checked_add(usize::from(offset))
                .ok_or(TtError::INVALID_TABLE)?;
            let span = (usize::from(end) - usize::from(start) + 1) * 2;
            let outside = ids < glyph_ids
                || ids
                    .checked_add(span)
                    .is_none_or(|end| end > data.len());
            if outside && !sentinel {
                return Err(TtError::INVALID_TABLE);
            }
        } else if offset == 0xFFFF && !sentinel {
            return Err(TtError::INVALID_TABLE);
        }
        last_start = start;
        last_end = end;
    }
    Ok(flags)
}

/// `tt_cmap4_char_index`: the linear scan of `tt_cmap4_char_map_linear`
/// for [`CMAP_FLAG_UNSORTED`] subtables, otherwise the binary search
/// of `tt_cmap4_char_map_binary`.
fn format4_char_index(data: &[u8], flags: u32, char_code: u32) -> u32 {
    if char_code >= 0x1_0000 {
        return 0;
    }
    let Some(n2) = format4_n2(data) else {
        return 0;
    };
    let num_segs = n2 / 2;
    if num_segs == 0 {
        return 0;
    }
    if flags & CMAP_FLAG_UNSORTED != 0 {
        for index in 0..num_segs {
            let Some(segment) = format4_segment(data, n2, index) else {
                return 0;
            };
            if char_code >= u32::from(segment.start) && char_code <= u32::from(segment.end) {
                return format4_gindex(data, &segment, char_code);
            }
        }
        return 0;
    }
    let index = format4_lower_bound(data, char_code, num_segs);
    let Some(segment) = format4_segment(data, n2, index) else {
        return 0;
    };
    if char_code < u32::from(segment.start) {
        return 0;
    }
    format4_gindex(data, &segment, char_code)
}

/// `tt_cmap4_char_next` without the `TT_CMap4Rec` iteration state: the
/// first mapped code after `char_code`, scanning segment by segment
/// (and code by code for [`CMAP_FLAG_UNSORTED`] subtables).
fn format4_char_next(data: &[u8], flags: u32, char_code: u32) -> Option<(u32, u32)> {
    if char_code >= 0xFFFF {
        return None;
    }
    let n2 = format4_n2(data)?;
    let num_segs = n2 / 2;
    if num_segs == 0 {
        return None;
    }
    let mut code = char_code + 1;
    if flags & CMAP_FLAG_UNSORTED != 0 {
        while code <= 0xFFFF {
            for index in 0..num_segs {
                let segment = format4_segment(data, n2, index)?;
                if code >= u32::from(segment.start) && code <= u32::from(segment.end) {
                    let gindex = format4_gindex(data, &segment, code);
                    if gindex != 0 {
                        return Some((code, gindex));
                    }
                    break;
                }
            }
            code += 1;
        }
        return None;
    }
    let mut index = format4_lower_bound(data, code, num_segs);
    loop {
        let segment = format4_segment(data, n2, index)?;
        if code < u32::from(segment.start) {
            code = u32::from(segment.start);
        }
        if code > u32::from(segment.end) {
            index += 1;
            continue;
        }
        while code <= u32::from(segment.end) {
            let gindex = format4_gindex(data, &segment, code);
            if gindex != 0 {
                return Some((code, gindex));
            }
            code += 1;
        }
        index += 1;
    }
}

/// `tt_cmap6_validate` (`ttcmap.c`): checks `length` against the
/// `count` of the trailing glyph-id array.
fn validate_format6(data: &[u8]) -> TtResult<u32> {
    if data.len() < 10 {
        return Err(TtError::INVALID_TABLE);
    }
    let length = usize::from(u16_at(data, 2).ok_or(TtError::INVALID_TABLE)?);
    let count = usize::from(u16_at(data, 8).ok_or(TtError::INVALID_TABLE)?);
    if length > data.len() || length < 10 + count * 2 {
        return Err(TtError::INVALID_TABLE);
    }
    Ok(0)
}

/// `tt_cmap6_char_index`: the entry at `char_code - firstCode`.
fn format6_char_index(data: &[u8], char_code: u32) -> u32 {
    let (Some(start), Some(count)) = (u16_at(data, 6), u16_at(data, 8)) else {
        return 0;
    };
    let index = char_code.wrapping_sub(u32::from(start));
    if index >= u32::from(count) {
        return 0;
    }
    u16_at(data, 10 + index as usize * 2).map_or(0, u32::from)
}

/// `tt_cmap6_char_next`: the first non-zero entry from
/// `char_code + 1` (clamped to `firstCode`).
fn format6_char_next(data: &[u8], char_code: u32) -> Option<(u32, u32)> {
    let mut code = char_code.wrapping_add(1);
    if code >= 0x1_0000 {
        return None;
    }
    let (Some(start), Some(count)) = (u16_at(data, 6), u16_at(data, 8)) else {
        return None;
    };
    if code < u32::from(start) {
        code = u32::from(start);
    }
    let mut index = code - u32::from(start);
    while index < u32::from(count) {
        let gindex = u16_at(data, 10 + index as usize * 2).map_or(0, u32::from);
        if gindex != 0 {
            return Some((code, gindex));
        }
        index += 1;
        code += 1;
    }
    None
}

/// `tt_cmap8_validate` (`ttcmap.c`): checks the 8208-byte header, the
/// group count and the ascending group order (the `is32` bitmap is only
/// inspected at the tight level).
fn validate_format8(data: &[u8]) -> TtResult<u32> {
    if data.len() < 8208 {
        return Err(TtError::INVALID_TABLE);
    }
    let length = u32_at(data, 4).ok_or(TtError::INVALID_TABLE)? as usize;
    if length > data.len() {
        return Err(TtError::INVALID_TABLE);
    }
    if length < 8208 {
        return Err(TtError::INVALID_TABLE);
    }
    let num_groups = u32_at(data, 8204).ok_or(TtError::INVALID_TABLE)? as usize;
    if num_groups > (data.len() - 8208) / 12 {
        return Err(TtError::INVALID_TABLE);
    }
    let mut last = 0u32;
    for index in 0..num_groups {
        let group = format8_group(data, index).ok_or(TtError::INVALID_TABLE)?;
        let (start, end, _start_id) = group;
        if start > end {
            return Err(TtError::INVALID_TABLE);
        }
        if index > 0 && start <= last {
            return Err(TtError::INVALID_TABLE);
        }
        last = end;
    }
    Ok(0)
}

/// The `index`-th `(start, end, startGlyphID)` group of a format 8
/// subtable (the groups start at byte 8208).
fn format8_group(data: &[u8], index: usize) -> Option<(u32, u32, u32)> {
    let pos = index.checked_mul(12)?.checked_add(8208)?;
    let start = u32_at(data, pos)?;
    let end = u32_at(data, pos + 4)?;
    let start_id = u32_at(data, pos + 8)?;
    Some((start, end, start_id))
}

/// The number of groups of a format 8 subtable (byte 8204).
#[inline]
fn format8_num_groups(data: &[u8]) -> usize {
    u32_at(data, 8204).map_or(0, u32::from) as usize
}

/// `tt_cmap8_char_index`: the linear group scan of `ttcmap.c`.
fn format8_char_index(data: &[u8], char_code: u32) -> u32 {
    for index in 0..format8_num_groups(data) {
        let Some((start, end, start_id)) = format8_group(data, index) else {
            return 0;
        };
        if char_code < start {
            return 0;
        }
        if char_code <= end {
            return start_id.wrapping_add(char_code.wrapping_sub(start));
        }
    }
    0
}

/// `tt_cmap8_char_next`: the first mapped code after `char_code`,
/// group by group.
fn format8_char_next(data: &[u8], char_code: u32) -> Option<(u32, u32)> {
    let mut code = char_code.wrapping_add(1);
    for index in 0..format8_num_groups(data) {
        let (start, end, start_id) = format8_group(data, index)?;
        if code < start {
            code = start;
        }
        if code <= end {
            let gindex = start_id.wrapping_add(code.wrapping_sub(start));
            if gindex != 0 {
                return Some((code, gindex));
            }
        }
    }
    None
}

/// `tt_cmap10_validate` (`ttcmap.c`): checks `length` against the
/// `numChars` glyph-id array.
fn validate_format10(data: &[u8]) -> TtResult<u32> {
    if data.len() < 20 {
        return Err(TtError::INVALID_TABLE);
    }
    let length = u32_at(data, 4).ok_or(TtError::INVALID_TABLE)?;
    let count = u32_at(data, 16).ok_or(TtError::INVALID_TABLE)?;
    if length as usize > data.len() || length < 20 || (length - 20) / 2 < count {
        return Err(TtError::INVALID_TABLE);
    }
    Ok(0)
}

/// `tt_cmap10_char_index`: the entry at `char_code - startCharCode`.
fn format10_char_index(data: &[u8], char_code: u32) -> u32 {
    let (Some(start), Some(count)) = (u32_at(data, 12), u32_at(data, 16)) else {
        return 0;
    };
    let index = char_code.wrapping_sub(start);
    if index >= count {
        return 0;
    }
    u16_at(data, 20 + index as usize * 2).map_or(0, u32::from)
}

/// `tt_cmap10_char_next`: the first non-zero entry from
/// `char_code + 1` (clamped to `startCharCode`).
fn format10_char_next(data: &[u8], char_code: u32) -> Option<(u32, u32)> {
    let mut code = char_code.wrapping_add(1);
    let (Some(start), Some(count)) = (u32_at(data, 12), u32_at(data, 16)) else {
        return None;
    };
    if code < start {
        code = start;
    }
    let mut index = code.wrapping_sub(start);
    while index < count {
        let gindex = u16_at(data, 20 + index as usize * 2).map_or(0, u32::from);
        if gindex != 0 {
            return Some((code, gindex));
        }
        index += 1;
        code += 1;
    }
    None
}

/// `tt_cmap12_validate` (`ttcmap.c`): checks `length`, the group count
/// and the ascending, non-overlapping group order.
fn validate_format12(data: &[u8]) -> TtResult<u32> {
    if data.len() < 16 {
        return Err(TtError::INVALID_TABLE);
    }
    let length = u32_at(data, 4).ok_or(TtError::INVALID_TABLE)?;
    let num_groups = u32_at(data, 12).ok_or(TtError::INVALID_TABLE)?;
    if length as usize > data.len() || length < 16 || (length - 16) / 12 < num_groups {
        return Err(TtError::INVALID_TABLE);
    }
    let mut last = 0u32;
    for index in 0..num_groups as usize {
        let (start, end, _start_id) = format12_group(data, index).ok_or(TtError::INVALID_TABLE)?;
        if start > end {
            return Err(TtError::INVALID_TABLE);
        }
        if index > 0 && start <= last {
            return Err(TtError::INVALID_TABLE);
        }
        last = end;
    }
    Ok(0)
}

/// The `index`-th `(start, end, startGlyphID)` group of a format 12
/// subtable (the groups start at byte 16).
fn format12_group(data: &[u8], index: usize) -> Option<(u32, u32, u32)> {
    let pos = index.checked_mul(12)?.checked_add(16)?;
    let start = u32_at(data, pos)?;
    let end = u32_at(data, pos + 4)?;
    let start_id = u32_at(data, pos + 8)?;
    Some((start, end, start_id))
}

/// The number of groups of a format 12 subtable (byte 12).
#[inline]
fn format12_num_groups(data: &[u8]) -> Option<u32> {
    u32_at(data, 12)
}

/// `tt_cmap12_char_map_binary`: the binary group search of `ttcmap.c`.
fn format12_char_index(data: &[u8], char_code: u32) -> u32 {
    let Some(num_groups) = format12_num_groups(data) else {
        return 0;
    };
    let mut low = 0u32;
    let mut high = num_groups;
    while low < high {
        let mid = low + (high - low) / 2;
        let Some((start, end, start_id)) = format12_group(data, mid as usize) else {
            return 0;
        };
        if char_code < start {
            high = mid;
        } else if char_code > end {
            low = mid + 1;
        } else {
            return start_id.wrapping_add(char_code.wrapping_sub(start));
        }
    }
    0
}

/// `tt_cmap12_char_next` without the `TT_CMap12Rec` state: the first
/// mapped code after `char_code`, group by group.
fn format12_char_next(data: &[u8], char_code: u32) -> Option<(u32, u32)> {
    let num_groups = format12_num_groups(data)?;
    if num_groups == 0 {
        return None;
    }
    let mut code = char_code.wrapping_add(1);
    let mut low = 0u32;
    let mut high = num_groups;
    while low < high {
        let mid = low + (high - low) / 2;
        match format12_group(data, mid as usize) {
            Some((_, end, _)) if end < code => low = mid + 1,
            _ => high = mid,
        }
    }
    let mut index = low;
    while index < num_groups {
        let (start, end, start_id) = format12_group(data, index as usize)?;
        if code < start {
            code = start;
        }
        if code <= end {
            while code <= end {
                let gindex = start_id.wrapping_add(code.wrapping_sub(start));
                if gindex != 0 {
                    return Some((code, gindex));
                }
                if code == u32::MAX {
                    break;
                }
                code += 1;
            }
        }
        index += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{cmap_table, push_i16, push_u16, push_u32};
    use alloc::sync::Weak;
    use codevar_truetype_core::LibCell;

    struct Segment {
        start: u16,
        end: u16,
        delta: i16,
        ids: Vec<u16>,
    }

    fn segment(start: u16, end: u16, delta: i16, ids: &[u16]) -> Segment {
        Segment {
            start,
            end,
            delta,
            ids: ids.to_vec(),
        }
    }

    fn format4_table(segments: &[Segment]) -> Vec<u8> {
        let n2 = segments.len() * 2;
        let offsets_at = 14 + n2 + 2 + n2 + n2;
        let glyph_ids_at = offsets_at + n2;
        let mut out = Vec::new();
        push_u16(&mut out, 4);
        push_u16(&mut out, 0);
        push_u16(&mut out, 0);
        push_u16(&mut out, n2 as u16);
        push_u16(&mut out, 0);
        push_u16(&mut out, 0);
        push_u16(&mut out, 0);
        for entry in segments {
            push_u16(&mut out, entry.end);
        }
        push_u16(&mut out, 0);
        for entry in segments {
            push_u16(&mut out, entry.start);
        }
        for entry in segments {
            push_i16(&mut out, entry.delta);
        }
        let mut ids_at = glyph_ids_at;
        for (index, entry) in segments.iter().enumerate() {
            if entry.ids.is_empty() {
                push_u16(&mut out, 0);
            } else {
                push_u16(&mut out, (ids_at - (offsets_at + index * 2)) as u16);
                ids_at += entry.ids.len() * 2;
            }
        }
        for entry in segments {
            for id in &entry.ids {
                push_u16(&mut out, *id);
            }
        }
        let length = out.len() as u16;
        out[2..4].copy_from_slice(&length.to_be_bytes());
        out
    }

    fn ordered_segments() -> Vec<Segment> {
        vec![
            segment(0x41, 0x42, -64, &[]),
            segment(0x61, 0x61, 0, &[3]),
            segment(0xFFFF, 0xFFFF, 1, &[]),
        ]
    }

    fn format0_table() -> Vec<u8> {
        let mut out = Vec::with_capacity(262);
        push_u16(&mut out, 0);
        push_u16(&mut out, 262);
        push_u16(&mut out, 0);
        let mut ids = [0u8; 256];
        ids[0x41] = 5;
        ids[0x42] = 7;
        ids[0xFF] = 9;
        out.extend_from_slice(&ids);
        out
    }

    fn format2_table() -> Vec<u8> {
        let mut out = Vec::with_capacity(544);
        push_u16(&mut out, 2);
        push_u16(&mut out, 544);
        push_u16(&mut out, 0);
        let mut keys = vec![0u16; 256];
        keys[0x81] = 8;
        for key in keys {
            push_u16(&mut out, key);
        }
        push_u16(&mut out, 0x41);
        push_u16(&mut out, 3);
        push_i16(&mut out, -64);
        push_u16(&mut out, 10);
        push_u16(&mut out, 0x40);
        push_u16(&mut out, 2);
        push_i16(&mut out, 100);
        push_u16(&mut out, 8);
        for id in [65u16, 66, 0, 1, 2] {
            push_u16(&mut out, id);
        }
        out
    }

    fn format6_table(ids: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 6);
        push_u16(&mut out, (10 + ids.len() * 2) as u16);
        push_u16(&mut out, 0);
        push_u16(&mut out, 0x41);
        push_u16(&mut out, ids.len() as u16);
        for id in ids {
            push_u16(&mut out, *id);
        }
        out
    }

    fn format8_table(groups: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 8);
        push_u16(&mut out, 0);
        push_u32(&mut out, (8208 + groups.len() * 12) as u32);
        push_u32(&mut out, 0);
        out.extend_from_slice(&[0u8; 8192]);
        push_u32(&mut out, groups.len() as u32);
        for &(start, end, start_id) in groups {
            push_u32(&mut out, start);
            push_u32(&mut out, end);
            push_u32(&mut out, start_id);
        }
        out
    }

    fn format10_table(ids: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 10);
        push_u16(&mut out, 0);
        push_u32(&mut out, (20 + ids.len() * 2) as u32);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0x41);
        push_u32(&mut out, ids.len() as u32);
        for id in ids {
            push_u16(&mut out, *id);
        }
        out
    }

    fn format12_table(groups: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 12);
        push_u16(&mut out, 0);
        push_u32(&mut out, (16 + groups.len() * 12) as u32);
        push_u32(&mut out, 0);
        push_u32(&mut out, groups.len() as u32);
        for &(start, end, start_id) in groups {
            push_u32(&mut out, start);
            push_u32(&mut out, end);
            push_u32(&mut out, start_id);
        }
        out
    }

    fn subtable(format: u16, data: &[u8]) -> CmapSubtable<'_> {
        CmapSubtable {
            format,
            flags: validate_subtable(format, data).unwrap(),
            data,
        }
    }

    fn test_charmap() -> CharMap {
        CharMap {
            face: Weak::new(),
            encoding: Encoding::UNICODE,
            platform_id: 3,
            encoding_id: 1,
            clazz: &SFNT_CMAP_CLASS,
            data: LibCell::new(None),
        }
    }

    #[test]
    fn format0_maps_byte_values() {
        let data = format0_table();
        assert!(validate_subtable(0, &data).is_ok());
        let sub = subtable(0, &data);
        assert_eq!(sub.flags, 0);
        assert_eq!(sub.char_index(0x41), 5);
        assert_eq!(sub.char_index(0x42), 7);
        assert_eq!(sub.char_index(0x43), 0);
        assert_eq!(sub.char_index(0xFF), 9);
        assert_eq!(sub.char_index(0x100), 0);
        assert_eq!(sub.char_next(0), Some((0x41, 5)));
        assert_eq!(sub.char_next(0x41), Some((0x42, 7)));
        assert_eq!(sub.char_next(0x42), Some((0xFF, 9)));
        assert_eq!(sub.char_next(0xFF), None);
    }

    #[test]
    fn format0_rejects_short_headers() {
        assert!(validate_format0(&[]).is_err());
        assert!(validate_format0(&[0, 0, 1, 0]).is_err());
        let mut data = format0_table();
        data.pop();
        assert!(validate_format0(&data).is_err());
        assert!(validate_format0(&format0_table()).is_ok());
    }

    #[test]
    fn format2_maps_low_and_high_bytes() {
        let data = format2_table();
        assert!(validate_subtable(2, &data).is_ok());
        let sub = subtable(2, &data);
        assert_eq!(sub.char_index(0x41), 1);
        assert_eq!(sub.char_index(0x42), 2);
        assert_eq!(sub.char_index(0x43), 0);
        assert_eq!(sub.char_index(0x40), 0);
        assert_eq!(sub.char_index(0x81), 0);
        assert_eq!(sub.char_index(0x8140), 101);
        assert_eq!(sub.char_index(0x8141), 102);
        assert_eq!(sub.char_index(0x8240), 0);
        assert_eq!(sub.char_index(0x1_0000), 0);
        assert_eq!(sub.char_next(0), Some((0x41, 1)));
        assert_eq!(sub.char_next(0x41), Some((0x42, 2)));
        assert_eq!(sub.char_next(0x42), Some((0x8140, 101)));
    }

    #[test]
    fn format2_rejects_short_and_dangling_tables() {
        let mut short = format2_table();
        short.truncate(517);
        assert!(validate_format2(&short).is_err());
        assert!(validate_format2(&format2_table()).is_ok());

        let mut dangling = format2_table();
        dangling[524..526].copy_from_slice(&0x0FF0u16.to_be_bytes());
        assert!(validate_format2(&dangling).is_err());
    }

    #[test]
    fn format4_ordered_maps_by_delta_and_range_offset() {
        let data = format4_table(&ordered_segments());
        let sub = subtable(4, &data);
        assert_eq!(sub.flags, 0);
        assert_eq!(sub.char_index(0x41), 1);
        assert_eq!(sub.char_index(0x42), 2);
        assert_eq!(sub.char_index(0x43), 0);
        assert_eq!(sub.char_index(0x61), 3);
        assert_eq!(sub.char_index(0x62), 0);
        assert_eq!(sub.char_index(0xFFFF), 0);
        assert_eq!(sub.char_index(0x1_0000), 0);
        assert_eq!(sub.char_next(0), Some((0x41, 1)));
        assert_eq!(sub.char_next(0x41), Some((0x42, 2)));
        assert_eq!(sub.char_next(0x42), Some((0x61, 3)));
        assert_eq!(sub.char_next(0x61), None);
        assert_eq!(sub.char_next(0xFFFF), None);
    }

    #[test]
    fn format4_unsorted_segments_scan_linearly() {
        let data = format4_table(&[
            segment(0x61, 0x61, 0, &[3]),
            segment(0x41, 0x42, -64, &[]),
            segment(0xFFFF, 0xFFFF, 1, &[]),
        ]);
        let sub = subtable(4, &data);
        assert_eq!(sub.flags, CMAP_FLAG_UNSORTED);
        assert_eq!(sub.flags & CMAP_FLAG_OVERLAPPING, 0);
        assert_eq!(sub.char_index(0x41), 1);
        assert_eq!(sub.char_index(0x42), 2);
        assert_eq!(sub.char_index(0x61), 3);
        assert_eq!(sub.char_next(0), Some((0x41, 1)));
        assert_eq!(sub.char_next(0x42), Some((0x61, 3)));
    }

    #[test]
    fn format4_overlapping_segments_report_the_flag() {
        let data = format4_table(&[
            segment(0x41, 0x50, -64, &[]),
            segment(0x48, 0x60, -71, &[]),
            segment(0xFFFF, 0xFFFF, 1, &[]),
        ]);
        let sub = subtable(4, &data);
        assert_eq!(sub.flags, CMAP_FLAG_OVERLAPPING);
        assert_eq!(sub.char_index(0x48), 8);
        assert_eq!(sub.char_index(0x50), 16);
        assert_eq!(sub.char_index(0x51), 10);
        assert_eq!(sub.char_index(0x61), 0);
    }

    #[test]
    fn format4_fixes_a_broken_sentinel_segment() {
        let mut data = format4_table(&[segment(0x41, 0x42, -64, &[]), segment(0xFFFF, 0xFFFF, 0, &[])]);
        data[30..32].copy_from_slice(&4u16.to_be_bytes());
        assert!(validate_subtable(4, &data).is_ok());
        let segment = format4_segment(&data, 4, 1).unwrap();
        assert_eq!(segment.delta, 1);
        assert_eq!(segment.offset, 0);
        assert_eq!(format4_char_index(&data, 0, 0xFFFF), 0);
    }

    #[test]
    fn format4_rejects_malformed_segments() {
        let reversed = format4_table(&[segment(0x50, 0x41, 0, &[]), segment(0xFFFF, 0xFFFF, 1, &[])]);
        assert!(validate_format4(&reversed).is_err());

        let mut dangling = format4_table(&[
            segment(0x41, 0x42, -64, &[]),
            segment(0x50, 0x51, 0, &[]),
            segment(0xFFFF, 0xFFFF, 1, &[]),
        ]);
        dangling[36..38].copy_from_slice(&4u16.to_be_bytes());
        assert!(validate_format4(&dangling).is_err());

        let mut far_offset = format4_table(&[
            segment(0x41, 0x42, -64, &[]),
            segment(0x50, 0x51, 0, &[]),
            segment(0xFFFF, 0xFFFF, 1, &[]),
        ]);
        far_offset[36..38].copy_from_slice(&0xFFFFu16.to_be_bytes());
        assert!(validate_format4(&far_offset).is_err());

        assert!(validate_format4(&[4, 0, 10, 0, 0, 0, 0, 0]).is_err());

        let mut too_many = vec![0u8; 16];
        too_many[0..2].copy_from_slice(&4u16.to_be_bytes());
        too_many[2..4].copy_from_slice(&16u16.to_be_bytes());
        too_many[6..8].copy_from_slice(&8u16.to_be_bytes());
        assert!(validate_format4(&too_many).is_err());
    }

    #[test]
    fn format6_maps_a_truncated_array() {
        let data = format6_table(&[5, 0, 7]);
        let sub = subtable(6, &data);
        assert_eq!(sub.char_index(0x41), 5);
        assert_eq!(sub.char_index(0x42), 0);
        assert_eq!(sub.char_index(0x43), 7);
        assert_eq!(sub.char_index(0x40), 0);
        assert_eq!(sub.char_index(0x1_0000), 0);
        assert_eq!(sub.char_next(0x40), Some((0x41, 5)));
        assert_eq!(sub.char_next(0x41), Some((0x43, 7)));
        assert_eq!(sub.char_next(0x43), None);
        assert_eq!(sub.char_next(0xFFFF), None);
    }

    #[test]
    fn format6_rejects_mismatched_counts() {
        let mut data = format6_table(&[5, 0, 7]);
        data[8..10].copy_from_slice(&100u16.to_be_bytes());
        assert!(validate_format6(&data).is_err());
        assert!(validate_format6(&[0u8; 9]).is_err());
        assert!(validate_format6(&format6_table(&[5, 0, 7])).is_ok());
    }

    #[test]
    fn format8_maps_groups() {
        let data = format8_table(&[(0x41, 0x42, 10), (0x1000, 0x1001, 20)]);
        let sub = subtable(8, &data);
        assert_eq!(sub.flags, 0);
        assert_eq!(sub.char_index(0x41), 10);
        assert_eq!(sub.char_index(0x42), 11);
        assert_eq!(sub.char_index(0x43), 0);
        assert_eq!(sub.char_index(0x1000), 20);
        assert_eq!(sub.char_index(0x1001), 21);
        assert_eq!(sub.char_index(0x1002), 0);
        assert_eq!(sub.char_next(0), Some((0x41, 10)));
        assert_eq!(sub.char_next(0x42), Some((0x1000, 20)));
        assert_eq!(sub.char_next(0x1001), None);
    }

    #[test]
    fn format8_rejects_broken_headers_and_groups() {
        assert!(validate_format8(&[0u8; 100]).is_err());

        let mut short_length = format8_table(&[(0x41, 0x42, 10)]);
        short_length[4..8].copy_from_slice(&100u32.to_be_bytes());
        assert!(validate_format8(&short_length).is_err());

        let mut too_many = format8_table(&[(0x41, 0x42, 10)]);
        too_many[8204..8208].copy_from_slice(&100u32.to_be_bytes());
        assert!(validate_format8(&too_many).is_err());

        let descending = format8_table(&[(0x1000, 0x1001, 1), (0x41, 0x42, 5)]);
        assert!(validate_format8(&descending).is_err());

        let crossed = format8_table(&[(0x41, 0x42, 10), (0x42, 0x41, 5)]);
        assert!(validate_format8(&crossed).is_err());
    }

    #[test]
    fn format10_maps_a_truncated_array() {
        let data = format10_table(&[5, 0, 7]);
        let sub = subtable(10, &data);
        assert_eq!(sub.char_index(0x41), 5);
        assert_eq!(sub.char_index(0x42), 0);
        assert_eq!(sub.char_index(0x43), 7);
        assert_eq!(sub.char_index(0x40), 0);
        assert_eq!(sub.char_next(0x40), Some((0x41, 5)));
        assert_eq!(sub.char_next(0x41), Some((0x43, 7)));
        assert_eq!(sub.char_next(0x43), None);
        assert_eq!(sub.char_next(0xFFFF), None);
    }

    #[test]
    fn format10_rejects_mismatched_counts() {
        let mut data = format10_table(&[5, 0, 7]);
        data[16..20].copy_from_slice(&100u32.to_be_bytes());
        assert!(validate_format10(&data).is_err());

        let mut long_length = format10_table(&[5, 0, 7]);
        long_length[4..8].copy_from_slice(&100u32.to_be_bytes());
        assert!(validate_format10(&long_length).is_err());

        assert!(validate_format10(&[0u8; 19]).is_err());
        assert!(validate_format10(&format10_table(&[5, 0, 7])).is_ok());
    }

    #[test]
    fn format12_maps_groups_by_binary_search() {
        let data = format12_table(&[(0x41, 0x42, 1), (0x10000, 0x10001, 7)]);
        let sub = subtable(12, &data);
        assert_eq!(sub.char_index(0x40), 0);
        assert_eq!(sub.char_index(0x41), 1);
        assert_eq!(sub.char_index(0x42), 2);
        assert_eq!(sub.char_index(0x43), 0);
        assert_eq!(sub.char_index(0x10000), 7);
        assert_eq!(sub.char_index(0x10001), 8);
        assert_eq!(sub.char_index(0x10002), 0);
        assert_eq!(sub.char_next(0x40), Some((0x41, 1)));
        assert_eq!(sub.char_next(0x42), Some((0x10000, 7)));
        assert_eq!(sub.char_next(0x10001), None);
    }

    #[test]
    fn format12_skips_groups_that_map_to_glyph_zero() {
        let data = format12_table(&[(0x41, 0x42, 0)]);
        let sub = subtable(12, &data);
        assert_eq!(sub.char_index(0x41), 0);
        assert_eq!(sub.char_index(0x42), 1);
        assert_eq!(sub.char_next(0x40), Some((0x42, 1)));
        assert_eq!(sub.char_next(0x42), None);
    }

    #[test]
    fn format12_rejects_broken_groups() {
        assert!(validate_format12(&[0u8; 15]).is_err());

        let mut long_length = format12_table(&[(0x41, 0x42, 1)]);
        long_length[4..8].copy_from_slice(&100u32.to_be_bytes());
        assert!(validate_format12(&long_length).is_err());

        let descending = format12_table(&[(0x50, 0x60, 1), (0x41, 0x42, 5)]);
        assert!(validate_format12(&descending).is_err());

        let crossed = format12_table(&[(0x41, 0x42, 1), (0x42, 0x41, 5)]);
        assert!(validate_format12(&crossed).is_err());

        assert!(validate_format12(&format12_table(&[(0x41, 0x42, 1)])).is_ok());
    }

    #[test]
    fn unsupported_formats_are_rejected_and_never_map() {
        let data = [0u8; 4];
        assert!(matches!(
            validate_subtable(13, &data),
            Err(TtError::INVALID_CHARMAP_FORMAT)
        ));
        assert!(matches!(
            validate_subtable(14, &data),
            Err(TtError::INVALID_CHARMAP_FORMAT)
        ));
        let sub = CmapSubtable {
            format: 13,
            flags: 0,
            data: &data,
        };
        assert_eq!(sub.char_index(0x41), 0);
        assert!(sub.char_next(0x41).is_none());
    }

    #[test]
    fn parse_charmaps_reads_every_valid_record() {
        let table = cmap_table();
        let charmaps = parse_charmaps(&table).unwrap();
        assert_eq!(charmaps.len(), 2);
        assert_eq!(charmaps[0].platform_id, 3);
        assert_eq!(charmaps[0].encoding_id, 1);
        assert_eq!(charmaps[0].encoding, Encoding::UNICODE);
        assert_eq!(charmaps[0].subtable.format, 4);
        assert_eq!(charmaps[0].subtable.flags, 0);
        assert_eq!(charmaps[0].char_index(0x41), 1);
        assert_eq!(charmaps[0].char_index(0x43), 0);
        assert_eq!(charmaps[0].char_next(0x41), Some((0x42, 2)));
        assert_eq!(charmaps[1].platform_id, 3);
        assert_eq!(charmaps[1].encoding_id, 10);
        assert_eq!(charmaps[1].encoding, Encoding::UNICODE);
        assert_eq!(charmaps[1].subtable.format, 12);
        assert_eq!(charmaps[1].char_index(0x42), 2);
        assert_eq!(charmaps[1].char_index(0x40), 0);
    }

    #[test]
    fn parse_charmaps_rejects_a_bad_header() {
        assert!(matches!(parse_charmaps(&[0, 0, 0]), Err(TtError::INVALID_TABLE)));
        assert!(matches!(
            parse_charmaps(&[1, 0, 0, 0]),
            Err(TtError::INVALID_TABLE)
        ));
        assert!(parse_charmaps(&[0, 0, 0, 0]).unwrap().is_empty());
    }

    #[test]
    fn parse_charmaps_skips_broken_records() {
        let format6 = format6_table(&[5, 0, 7]);
        let mut table = Vec::new();
        push_u16(&mut table, 0);
        push_u16(&mut table, 6);
        push_u16(&mut table, 3);
        push_u16(&mut table, 1);
        push_u32(&mut table, 0);
        push_u16(&mut table, 3);
        push_u16(&mut table, 1);
        push_u32(&mut table, 84);
        push_u16(&mut table, 3);
        push_u16(&mut table, 1);
        push_u32(&mut table, 100);
        push_u16(&mut table, 3);
        push_u16(&mut table, 10);
        push_u32(&mut table, 68);
        push_u16(&mut table, 3);
        push_u16(&mut table, 1);
        push_u32(&mut table, 52);
        push_u16(&mut table, 3);
        push_u16(&mut table, 1);
        push_u32(&mut table, 76);
        table.extend_from_slice(&format6);
        push_u16(&mut table, 13);
        push_u16(&mut table, 0);
        push_u32(&mut table, 0);
        push_u16(&mut table, 4);
        push_u16(&mut table, 8);
        push_u16(&mut table, 0);
        push_u16(&mut table, 0);
        let charmaps = parse_charmaps(&table).unwrap();
        assert_eq!(charmaps.len(), 1);
        assert_eq!(charmaps[0].subtable.format, 6);
        assert_eq!(charmaps[0].char_index(0x41), 5);
    }

    #[test]
    fn parse_charmaps_skips_unsupported_formats() {
        let mut table = Vec::new();
        push_u16(&mut table, 0);
        push_u16(&mut table, 2);
        push_u16(&mut table, 3);
        push_u16(&mut table, 10);
        push_u32(&mut table, 20);
        push_u16(&mut table, 3);
        push_u16(&mut table, 10);
        push_u32(&mut table, 26);
        push_u16(&mut table, 13);
        push_u16(&mut table, 0);
        push_u32(&mut table, 0);
        push_u16(&mut table, 14);
        push_u16(&mut table, 0);
        push_u32(&mut table, 0);
        assert!(parse_charmaps(&table).unwrap().is_empty());
    }

    #[test]
    fn parse_charmaps_stops_at_a_truncated_record_array() {
        let mut table = Vec::new();
        push_u16(&mut table, 0);
        push_u16(&mut table, 3);
        push_u16(&mut table, 3);
        push_u16(&mut table, 1);
        push_u32(&mut table, 12);
        table.extend_from_slice(&format6_table(&[5, 0, 7]));
        let charmaps = parse_charmaps(&table).unwrap();
        assert_eq!(charmaps.len(), 1);
        assert_eq!(charmaps[0].subtable.format, 6);
    }

    #[test]
    fn sfnt_find_encoding_matches_the_fttruetype_table() {
        assert_eq!(sfnt_find_encoding(0, 4), Encoding::UNICODE);
        assert_eq!(sfnt_find_encoding(0, 3), Encoding::UNICODE);
        assert_eq!(sfnt_find_encoding(2, 1), Encoding::UNICODE);
        assert_eq!(sfnt_find_encoding(2, 0), Encoding::UNICODE);
        assert_eq!(sfnt_find_encoding(1, 0), Encoding::APPLE_ROMAN);
        assert_eq!(sfnt_find_encoding(1, 1), Encoding::NONE);
        assert_eq!(sfnt_find_encoding(3, 0), Encoding::MS_SYMBOL);
        assert_eq!(sfnt_find_encoding(3, 1), Encoding::UNICODE);
        assert_eq!(sfnt_find_encoding(3, 10), Encoding::UNICODE);
        assert_eq!(sfnt_find_encoding(3, 2), Encoding::SJIS);
        assert_eq!(sfnt_find_encoding(3, 3), Encoding::GB2312);
        assert_eq!(sfnt_find_encoding(3, 4), Encoding::BIG5);
        assert_eq!(sfnt_find_encoding(3, 5), Encoding::WANSUNG);
        assert_eq!(sfnt_find_encoding(3, 6), Encoding::JOHAB);
        assert_eq!(sfnt_find_encoding(3, 7), Encoding::NONE);
        assert_eq!(sfnt_find_encoding(4, 4), Encoding::NONE);
    }

    #[test]
    fn owned_subtables_delegate_to_the_readers() {
        let owned = OwnedCmapSubtable {
            format: 6,
            flags: 0,
            data: format6_table(&[5, 0, 7]),
        };
        assert_eq!(owned.char_index(0x41), 5);
        assert_eq!(owned.char_index(0x42), 0);
        assert_eq!(owned.char_index(0x43), 7);
        assert_eq!(owned.char_next(0x40), Some((0x41, 5)));
        assert_eq!(owned.char_next(0x43), None);
    }

    #[test]
    fn sfnt_cmap_class_installs_and_dispatches_the_payload() {
        let cmap = test_charmap();
        let payload = OwnedCmapSubtable {
            format: 6,
            flags: 0,
            data: format6_table(&[5, 0, 7]),
        };
        (SFNT_CMAP_CLASS.init)(&cmap, Some(&payload)).unwrap();
        assert_eq!(cmap.char_index(0x41), 5);
        assert_eq!(cmap.char_index(0x42), 0);
        assert_eq!(cmap.char_next(0x41), Some((0x43, 7)));
        assert_eq!(cmap.char_next(0x43), None);
        (SFNT_CMAP_CLASS.done)(&cmap);
        assert_eq!(cmap.char_index(0x41), 0);
        assert_eq!(cmap.char_next(0x41), None);
    }

    #[test]
    fn sfnt_cmap_class_init_rejects_a_missing_or_foreign_payload() {
        let cmap = test_charmap();
        assert!(matches!(
            (SFNT_CMAP_CLASS.init)(&cmap, None),
            Err(TtError::INVALID_ARGUMENT)
        ));
        assert!(matches!(
            (SFNT_CMAP_CLASS.init)(&cmap, Some(&42u32)),
            Err(TtError::INVALID_ARGUMENT)
        ));
        assert_eq!(cmap.char_index(0x41), 0);
        assert_eq!(cmap.char_next(0x41), None);
    }
}
