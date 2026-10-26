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

//! # Codevar SFNT container
//!
//! A faithful Rust port of the FreeType 2.6 `sfnt` module
//! (`src/sfnt/`): the OpenType/TrueType container (offset table, table
//! directory, TrueType collections), the core table readers and the
//! character-map machinery.  Everything parses directly out of an
//! in-memory `&[u8]` — no file paths, no `std::fs` — so a font can be
//! decoded from a byte buffer, a WASM asset or a memory-mapped file
//! alike.
//!
//! | FreeType file | Section below |
//! |---------------|---------------|
//! | `fttruetype.h`/`tttypes.h` tags | [`tags`] |
//! | `ttload.c` (`tt_face_load_font_dir`, `check_table_dir`, `tt_face_lookup_table`) | [`container`] |
//! | `sfobjs.c` (`sfnt_open_font`) | [`container`] ([`SfntContainer`]) |
//! | `ttload.c` (`tt_face_load_head`, `tt_face_load_max_profile`) | [`tables`] |
//! | `ttmtx.c` (`tt_face_load_hhea`, `tt_face_get_metrics`) | [`tables`] ([`HmtxTable`]) |
//! | `ttpload.c` (`tt_face_load_loca`, `tt_face_get_location`) | [`tables`] ([`LocaTable`]), [`font`] |
//! | `ttkern.c` (`tt_face_load_kern`, `tt_face_get_kerning`) | [`tables`] ([`KernTable`]) |
//! | `ttcmap.c` (`tt_face_build_cmaps`, `tt_cmapNN_*`) | [`cmap`] |
//! | `sfobjs.c` (`sfnt_find_encoding`, `sfnt_init_face`, `sfnt_load_face`) | [`cmap`], [`font`] |
//!
//! ## Typical use
//!
//! ```ignore
//! let font = SfntFont::open(bytes, 0)?;
//! let glyph = font.char_index('A' as u32);
//! let (advance, lsb) = font.metrics_for_glyph(glyph);
//! font.apply_face_info(&mut face);
//! font.install_charmaps(&face)?;
//! ```
//!
//! [`SfntFont`] is the driver-facing façade: the TrueType driver's
//! `init_face` consumes [`SfntFont::open`], [`SfntFont::apply_face_info`]
//! and [`SfntFont::install_charmaps`]; `load_glyph` consumes
//! [`SfntFont::glyph_location`], [`SfntFont::advance_for_glyph`] and the
//! raw [`SfntFont::table`] views.
//!
//! ## Deviations from FreeType 2.6
//!
//! * A truncated table directory is rejected ([`TtError::INVALID_TABLE`])
//!   instead of being silently clamped to the readable entries.
//! * A font without a `maxp` table is rejected
//!   ([`TtError::TABLE_MISSING`]); FreeType would load it with zero
//!   glyphs because later `LOAD_` calls overwrite the error.
//! * `head` must be present and at least 54 bytes
//!   (`check_table_dir`), `units_per_em` must be non-zero
//!   (`sfnt_load_face`); `bhed`/SING/META fonts are not accepted.
//! * `cmap` construction errors are swallowed during [`SfntFont::open`]
//!   (yielding an empty charmap list), mirroring `sfnt_load_face`,
//!   while the standalone [`cmap::parse_charmaps`] reports them.
//! * WOFF (`wOFF`) containers are not synthesized into SFNT
//!   (`woff_open_font` is not ported); such files are reported as
//!   [`TtError::UNKNOWN_FILE_FORMAT`].
//!
//! All offsets, lengths and indices are bounds-checked before every
//! read; malformed bytes produce [`TtError`]s and never panic.
#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]
extern crate alloc;

pub mod cmap;
pub mod container;
pub mod font;
pub mod tables;
pub mod tags;

pub use cmap::{
    CMAP_FLAG_OVERLAPPING, CMAP_FLAG_UNSORTED, CmapSubtable, OwnedCmapSubtable, SFNT_CMAP_CLASS, SfntCMap,
    parse_charmaps, sfnt_find_encoding,
};
pub use container::{SfntContainer, SfntDirectory, TableRecord, TtcHeader};
pub use font::SfntFont;
pub use tables::{HeadTable, HheaTable, HmtxTable, KernTable, LocaTable, MaxpTable};

#[cfg(test)]
pub(crate) mod test_util;
