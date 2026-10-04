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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing
//! permissions and limitations under the License.

//! SFNT tags, platform identifiers and container version tags.
//!
//! Port of the constants from FreeType 2.6's `fttruetype.h`,
//! `include/freetype/internal/tttypes.h` and `ttload.c` (the `TTTAG_*`
//! macros).  Every tag is built with [`make_tag`](codevar_truetype_core::make_tag)
//! in `A`, `B`, `C`, `D` byte order, matching the big-endian layout of
//! the file.

use codevar_truetype_core::{Tag, make_tag};

/// `'head'`: font header (`tt_face_load_head`).
pub const TAG_HEAD: Tag = make_tag(b'h', b'e', b'a', b'd');
/// `'bhed'`: Apple bitmap font header (same layout as `head`).
pub const TAG_BHED: Tag = make_tag(b'b', b'h', b'e', b'd');
/// `'maxp'`: maximum profile (`tt_face_load_max_profile`).
pub const TAG_MAXP: Tag = make_tag(b'm', b'a', b'x', b'p');
/// `'hhea'`: horizontal header (`tt_face_load_hhea`).
pub const TAG_HHEA: Tag = make_tag(b'h', b'h', b'e', b'a');
/// `'hmtx'`: horizontal metrics (`tt_face_load_hmtx`).
pub const TAG_HMTX: Tag = make_tag(b'h', b'm', b't', b'x');
/// `'vhea'`: vertical header.
pub const TAG_VHEA: Tag = make_tag(b'v', b'h', b'e', b'a');
/// `'vmtx'`: vertical metrics.
pub const TAG_VMTX: Tag = make_tag(b'v', b'm', b't', b'x');
/// `'loca'`: glyph location offsets (`tt_face_load_loca`).
pub const TAG_LOCA: Tag = make_tag(b'l', b'o', b'c', b'a');
/// `'glyf'`: glyph outlines (consumed by the TrueType driver).
pub const TAG_GLYF: Tag = make_tag(b'g', b'l', b'y', b'f');
/// `'cmap'`: character to glyph index mapping (`tt_face_build_cmaps`).
pub const TAG_CMAP: Tag = make_tag(b'c', b'm', b'a', b'p');
/// `'kern'`: legacy kerning (`tt_face_load_kern`).
pub const TAG_KERN: Tag = make_tag(b'k', b'e', b'r', b'n');
/// `'CFF '`: Compact Font Format outlines.
pub const TAG_CFF: Tag = make_tag(b'C', b'F', b'F', b' ');
/// `'name'`: naming table.
pub const TAG_NAME: Tag = make_tag(b'n', b'a', b'm', b'e');
/// `'post'`: PostScript metrics and names.
pub const TAG_POST: Tag = make_tag(b'p', b'o', b's', b't');
/// `'OS/2'`: OS/2 and Windows metrics.
pub const TAG_OS2: Tag = make_tag(b'O', b'S', b'/', b'2');
/// `'cvt '`: control value table (bytecode interpreter).
pub const TAG_CVT: Tag = make_tag(b'c', b'v', b't', b' ');
/// `'fpgm'`: font program (bytecode interpreter).
pub const TAG_FPGM: Tag = make_tag(b'f', b'p', b'g', b'm');
/// `'prep'`: CVT program (bytecode interpreter).
pub const TAG_PREP: Tag = make_tag(b'p', b'r', b'e', b'p');
/// `'gasp'`: grid-fitting and scan-conversion procedure.
pub const TAG_GASP: Tag = make_tag(b'g', b'a', b's', b'p');
/// `'PCLT'`: PCL 5 font header.
pub const TAG_PCLT: Tag = make_tag(b'P', b'C', b'L', b'T');
/// `'fvar'`: font variations (multiple masters).
pub const TAG_FVAR: Tag = make_tag(b'f', b'v', b'a', b'r');
/// `'gvar'`: glyph variations (multiple masters).
pub const TAG_GVAR: Tag = make_tag(b'g', b'v', b'a', b'r');
/// `'EBLC'`: embedded bitmap location.
pub const TAG_EBLC: Tag = make_tag(b'E', b'B', b'L', b'C');
/// `'EBDT'`: embedded bitmap data.
pub const TAG_EBDT: Tag = make_tag(b'E', b'B', b'D', b'T');
/// `'bloc'`: Apple embedded bitmap location.
pub const TAG_BLOC: Tag = make_tag(b'b', b'l', b'o', b'c');
/// `'bdat'`: Apple embedded bitmap data.
pub const TAG_BDAT: Tag = make_tag(b'b', b'd', b'a', b't');
/// `'sbix'`: Apple color bitmap strikes.
pub const TAG_SBIX: Tag = make_tag(b's', b'b', b'i', b'x');
/// `'COLR'`: color glyph layer table.
pub const TAG_COLR: Tag = make_tag(b'C', b'O', b'L', b'R');
/// `'CPAL'`: color palette table.
pub const TAG_CPAL: Tag = make_tag(b'C', b'P', b'A', b'L');
/// `'GPOS'`: glyph positioning table.
pub const TAG_GPOS: Tag = make_tag(b'G', b'P', b'O', b'S');
/// `'GSUB'`: glyph substitution table.
pub const TAG_GSUB: Tag = make_tag(b'G', b'S', b'U', b'B');
/// `'ttcf'`: TrueType collection header tag (`sfnt_open_font`).
pub const TAG_TTCF: Tag = make_tag(b't', b't', b'c', b'f');
/// `'wOFF'`: Web Open Font Format container (not ported).
pub const TAG_WOFF: Tag = make_tag(b'w', b'O', b'F', b'F');
/// `'true'`: legacy Apple TrueType SFNT version tag.
pub const TAG_TRUE: Tag = make_tag(b't', b'r', b'u', b'e');
/// `'typ1'`: Adobe Type 1 SFNT version tag.
pub const TAG_TYP1: Tag = make_tag(b't', b'y', b'p', b'1');
/// `'OTTO'`: OpenType (CFF outlines) SFNT version tag.
pub const TAG_OTTO: Tag = make_tag(b'O', b'T', b'T', b'O');
/// `'SING'`: single-glyph font table (bitmap-only SFNT fonts).
pub const TAG_SING: Tag = make_tag(b'S', b'I', b'N', b'G');
/// `'meta'`: metadata table shipped together with `SING`.
pub const TAG_META: Tag = make_tag(b'm', b'e', b't', b'a');

/// The original TrueType SFNT version (`0x00010000`).
pub const SFNT_VERSION_1_0: u32 = 0x0001_0000;
/// The OpenType SFNT version (`0x00020000`, used by rare fonts).
pub const SFNT_VERSION_2_0: u32 = 0x0002_0000;
/// The synthesized TTC version of a plain font (`1 << 16`, see
/// `sfnt_open_font`).
pub const SFNT_TTC_VERSION_SYNTHESIZED: u32 = 0x0001_0000;

/// The `head.magic_number` constant (`0x5F0F3CF5`).
pub const HEAD_MAGIC_NUMBER: u32 = 0x5F0F_3CF5;

/// `TT_PLATFORM_APPLE_UNICODE`: platform ID 0 (Unicode).
pub const PLATFORM_UNICODE: u16 = 0;
/// `TT_PLATFORM_MACINTOSH`: platform ID 1.
pub const PLATFORM_MACINTOSH: u16 = 1;
/// `TT_PLATFORM_ISO`: platform ID 2 (deprecated ISO/Unicode).
pub const PLATFORM_ISO: u16 = 2;
/// `TT_PLATFORM_MICROSOFT`: platform ID 3.
pub const PLATFORM_MICROSOFT: u16 = 3;

/// `TT_MAC_ID_ROMAN`: Macintosh Roman encoding.
pub const MAC_ID_ROMAN: u16 = 0;
/// `TT_MS_ID_SYMBOL_CS`: Microsoft symbol encoding.
pub const MS_ID_SYMBOL_CS: u16 = 0;
/// `TT_MS_ID_UNICODE_CS`: Microsoft UCS-2 (BMP) encoding.
pub const MS_ID_UNICODE_CS: u16 = 1;
/// `TT_MS_ID_SHIFT_JIS`: Microsoft Shift JIS.
pub const MS_ID_SHIFT_JIS: u16 = 2;
/// `TT_MS_ID_PRC`: Microsoft Simplified Chinese (PRC).
pub const MS_ID_PRC: u16 = 3;
/// `TT_MS_ID_BIG_5`: Microsoft Traditional Chinese (Big 5).
pub const MS_ID_BIG_5: u16 = 4;
/// `TT_MS_ID_WANSUNG`: Microsoft Wansung.
pub const MS_ID_WANSUNG: u16 = 5;
/// `TT_MS_ID_JOHAB`: Microsoft Johab.
pub const MS_ID_JOHAB: u16 = 6;
/// `TT_MS_ID_UCS_4`: Microsoft UCS-4 (full Unicode).
pub const MS_ID_UCS_4: u16 = 10;

/// Returns `true` when `version` is an SFNT version tag accepted by
/// `check_table_dir` (`ttload.c`): `0x00010000`, `'OTTO'`, `'true'`,
/// `'typ1'` or `0x00020000`.
///
/// `'ttcf'` and `'wOFF'` are rejected: a collection header is handled
/// before the directory is parsed (see [`crate::SfntContainer::open`])
/// and this port does not synthesize an SFNT out of WOFF
/// (`woff_open_font` is not ported).
#[inline]
pub const fn is_sfnt_version(version: u32) -> bool {
    matches!(
        version,
        SFNT_VERSION_1_0 | SFNT_VERSION_2_0 | TAG_OTTO | TAG_TRUE | TAG_TYP1
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_tag_is_big_endian_byte_order() {
        assert_eq!(TAG_HEAD, 0x6865_6164);
        assert_eq!(TAG_CMAP, 0x636D_6170);
        assert_eq!(TAG_TTCF, 0x7474_6366);
        assert_eq!(make_tag(b'O', b'S', b'/', b'2'), 0x4F53_2F32);
        assert_eq!(make_tag(b'C', b'F', b'F', b' '), 0x4346_4620);
    }

    #[test]
    fn is_sfnt_version_accepts_every_documented_tag() {
        assert!(is_sfnt_version(SFNT_VERSION_1_0));
        assert!(is_sfnt_version(SFNT_VERSION_2_0));
        assert!(is_sfnt_version(TAG_OTTO));
        assert!(is_sfnt_version(TAG_TRUE));
        assert!(is_sfnt_version(TAG_TYP1));
    }

    #[test]
    fn is_sfnt_version_rejects_unknown_ttcf_and_woff() {
        assert!(!is_sfnt_version(TAG_TTCF));
        assert!(!is_sfnt_version(TAG_WOFF));
        assert!(!is_sfnt_version(0x0003_0000));
        assert!(!is_sfnt_version(0));
        assert!(!is_sfnt_version(TAG_HEAD));
    }

    #[test]
    fn platform_and_encoding_ids_match_fttruetype() {
        assert_eq!(PLATFORM_UNICODE, 0);
        assert_eq!(PLATFORM_MACINTOSH, 1);
        assert_eq!(PLATFORM_ISO, 2);
        assert_eq!(PLATFORM_MICROSOFT, 3);
        assert_eq!(MAC_ID_ROMAN, 0);
        assert_eq!(MS_ID_SYMBOL_CS, 0);
        assert_eq!(MS_ID_UNICODE_CS, 1);
        assert_eq!(MS_ID_UCS_4, 10);
        assert_eq!(MS_ID_SHIFT_JIS, 2);
        assert_eq!(MS_ID_PRC, 3);
        assert_eq!(MS_ID_BIG_5, 4);
        assert_eq!(MS_ID_WANSUNG, 5);
        assert_eq!(MS_ID_JOHAB, 6);
    }
}
