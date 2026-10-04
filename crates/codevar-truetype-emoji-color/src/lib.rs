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

//! Color font support: `CPAL`, `COLR` v0 layers, `CBLC`/`CBDT`
//! embedded PNG strikes and the glue that composites them into
//! premultiplied BGRA bitmaps.
//!
//! This crate is a port of the color machinery of FreeType 2.13.3
//! (`src/sfnt/ttcpal.c`, `ttcolr.c`, `ttsbit.c`, `pngshim.c` and the
//! layer loop of `FT_Render_Glyph_Internal` in `src/base/ftobjs.c`),
//! built on the `codevar-truetype-*` crates already in this workspace:
//!
//! | FreeType 2.13 | this crate |
//! |---|---|
//! | `tt_face_load_cpal` / `TT_Palette_Data` | [`cpal::CpalTable`] |
//! | `tt_face_load_colr` / `tt_face_get_colr_layer` | [`colr::ColrTable`] |
//! | `tt_face_load_sbit` / `tt_sbit_decoder_*` | [`sbit::SbitStrikes`] |
//! | `Load_SBit_Png` (minus libpng) | [`png::decode`] |
//! | `FT_Render_Glyph_Internal` + `tt_face_colr_blend_layer` | [`render::ColorFont`] |
//!
//! # Pipeline
//!
//! [`render::ColorFont::open`] takes a parsed [`SfntFont`] and keeps
//! the (optional) `CPAL`, `COLR` and `CBLC`+`CBDT` tables.
//! [`render::ColorFont::render_glyph`] then resolves one glyph in
//! this order, mirroring `FT_Load_Glyph` + `FT_Render_Glyph` with
//! `FT_LOAD_COLOR | FT_LOAD_RENDER`:
//!
//! 1. an embedded `CBLC`/`CBDT` PNG strike (nearest strike, then a
//!    box resample to the requested ppem),
//! 2. `COLR` v0 layers, each rendered as a gray outline and blended
//!    with its `CPAL` color into one BGRA canvas,
//! 3. the plain `glyf` outline as a gray bitmap.
//!
//! # Notes:
//!
//! * `COLR` v1 *paints* (`PaintColrLayers`, gradients, transforms,
//!   clips) are not executed; the v1 header is validated and ignored,
//!   and only v0 base-glyph/layer records composite.  Fonts that are
//!   v1-only fall back to the outline.
//! * `sbix` and monochrome `EBLC`/`EBDT` strikes are excluded; only
//!   `CBLC` strikes with `bitDepth == 32` and PNG image formats
//!   17/18/19 are decoded.  Other image formats report
//!   [`TtError::UNIMPLEMENTED_FEATURE`] (FreeType renders them gray).
//! * FreeType only uses a strike whose ppem matches the requested size
//!   exactly; this crate picks the nearest strike and resamples, which
//!   makes single-strike CBDT fonts usable at any size.
//! * For a font carrying both `CBDT` and `COLR`, FreeType blends the
//!   layers on top of the strike bitmap; this crate returns the strike
//!   image directly.
//! * A PNG whose `IHDR` size disagrees with the strike metrics is an
//!   [`TtError::INVALID_FILE_FORMAT`] here; FreeType silently keeps the
//!   zeroed buffer.
//! * Vertical metrics/layout (`FT_LOAD_VERTICAL_LAYOUT`) are not
//!   implemented.
//!
//! [`SfntFont`]: codevar_truetype_sfnt::SfntFont
//! [`TtError::UNIMPLEMENTED_FEATURE`]: codevar_truetype_core::TtError::UNIMPLEMENTED_FEATURE
//! [`TtError::INVALID_FILE_FORMAT`]: codevar_truetype_core::TtError::INVALID_FILE_FORMAT

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]
extern crate alloc;

/// Bounds-checked big-endian byte reader shared by the table parsers.
pub(crate) mod cursor;

pub mod colr;
pub mod cpal;
pub mod png;
pub mod render;
pub mod sbit;

#[cfg(test)]
mod test_util;
