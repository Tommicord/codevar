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

//! # Codevar TrueType `glyf` decoder
//!
//! A faithful Rust port of FreeType 2.6's TrueType outline loader
//! (`src/truetype/ttgload.c`): it turns the `glyf`/`loca` byte ranges of
//! an [`SfntFont`] into a ready-to-render [`Outline`] together with the
//! glyph's metrics, recursing into composite glyphs as it goes.
//!
//! | FreeType function | Rust equivalent |
//! |-------------------|-----------------|
//! | `TT_Access_Glyph_Frame` / `TT_Forget_Glyph_Frame` | `TtLoader` frame helpers |
//! | `TT_Load_Glyph_Header` | `TtLoader` header reader |
//! | `TT_Load_Simple_Glyph` | `TtLoader` simple-glyph reader |
//! | `TT_Load_Composite_Glyph` | `TtLoader` composite reader |
//! | `TT_Process_Simple_Glyph` | `TtLoader` simple-glyph processor |
//! | `TT_Process_Composite_Component` | `TtLoader` component processor |
//! | `TT_Hint_Glyph` | `TtLoader` phantom rounding |
//! | `tt_loader_set_pp`, `tt_get_metrics` | `TtLoader` phantom/metric helpers |
//! | `load_truetype_glyph` | `TtLoader` recursion driver |
//! | `compute_glyph_metrics` | `TtLoader` metric computer |
//! | `TT_Load_Glyph` | [`load_glyph`] |
//!
//! ## Typical use
//!
//! ```ignore
//! let font = SfntFont::open(bytes, 0)?;
//! let mut slot = /* a `GlyphSlot` owned by the face */;
//! load_glyph(&font, &mut slot, &size_metrics, glyph_index, LoadFlags::DEFAULT)?;
//! ```
//!
//! ## Deviations from FreeType 2.6
//!
//! * **No bytecode interpreter.** `TT_CONFIG_OPTION_BYTECODE_INTERPRETER`
//!   and `TT_CONFIG_OPTION_GX_VAR_SUPPORT` are treated as disabled:
//!   glyph instructions are bounds-checked and skipped, `TT_Hint_Glyph`
//!   reduces to rounding the four phantom points, `slot.control_data`
//!   stays empty and no `gvar` deltas are applied.  The hinting guard
//!   (`IS_HINTED`) is preserved so the rounding only runs when the
//!   caller asked for hinting.
//! * **No `hdmx`, `OS/2` or `vmtx`.** `compute_glyph_metrics` always
//!   takes FreeType's `hhea` fallback for the vertical metrics, and the
//!   device-width adjustment (`tt_face_get_device_metrics`) is skipped
//!   because no `hdmx` reader exists yet.  `slot.linear_*_advance` keep
//!   their font-unit values; the base layer scales them to 16.16
//!   (`FT_Load_Glyph`).
//! * **No embedded bitmaps.** `FT_LOAD_SBITS_ONLY` is rejected with
//!   [`TtError::INVALID_ARGUMENT`] and the `load_sbit_image` /
//!   `header_only` paths are not ported, so there is no "bbox only"
//!   loading mode.
//! * **Phantom points live on the loader.** FreeType writes `pp1..pp4`
//!   into the outline arrays just past `n_points` and reads them back;
//!   this port keeps them as `Vector` fields on `TtLoader`, which is
//!   equivalent because nothing but the interpreter ever observes that
//!   out-of-range region and it dodges the glyph loader's storage
//!   truncation.
//! * **No incremental interface** (`FT_CONFIG_OPTION_INCREMENTAL`), so
//!   `tt_get_metrics_incr_overrides` and the alternate stream handling
//!   of `load_truetype_glyph` are gone; `ttmetrics.valid` is not
//!   re-checked either because the caller always supplies a prepared
//!   [`SizeMetrics`].
//! * `FT_CONFIG_OPTION_COMPONENT_OFFSET_SCALED` is undefined (FreeType's
//!   default), so composite offsets are scaled only when the component
//!   explicitly sets `SCALED_COMPONENT_OFFSET`.
//!
//! [`Outline`]: codevar_truetype_core::Outline
//! [`SizeMetrics`]: codevar_truetype_core::SizeMetrics
//! [`TtError::INVALID_ARGUMENT`]: codevar_truetype_core::TtError::INVALID_ARGUMENT
#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]
extern crate alloc;

pub mod loader;

#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod test_util;

pub use loader::load_glyph;

pub use codevar_truetype_core::{
    SUBGLYPH_FLAG_2X2, SUBGLYPH_FLAG_ARGS_ARE_WORDS, SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES,
    SUBGLYPH_FLAG_ROUND_XY_TO_GRID, SUBGLYPH_FLAG_SCALE, SUBGLYPH_FLAG_USE_MY_METRICS,
    SUBGLYPH_FLAG_XY_SCALE,
};

/// `MORE_COMPONENTS`: another component follows this one in the
/// composite glyph's component list.
pub const SUBGLYPH_FLAG_MORE_COMPONENTS: u16 = 0x0020;

/// `WE_HAVE_INSTR`: the component list is followed by bytecode
/// instructions for the composite glyph.
///
/// Kept for format-level completeness; without the bytecode interpreter
/// the instruction bytes are skipped rather than executed.
pub const SUBGLYPH_FLAG_WE_HAVE_INSTR: u16 = 0x0100;

/// `OVERLAP_COMPOUND`: the components of this composite overlap and
/// need special handling by the rasterizer.
///
/// FreeType 2.6 parses but ignores this bit; it is exposed so that a
/// caller inspecting [`codevar_truetype_core::SubGlyph`] data can still
/// see it.
pub const SUBGLYPH_FLAG_OVERLAP_COMPOUND: u16 = 0x0400;

/// `SCALED_COMPONENT_OFFSET`: the component's x/y offset is scaled by
/// the component's own transform before it is applied.
///
/// This is the branch FreeType compiles in when
/// `TT_CONFIG_OPTION_COMPONENT_OFFSET_SCALED` is undefined.
pub const SUBGLYPH_FLAG_SCALED_COMPONENT_OFFSET: u16 = 0x0800;

/// `UNSCALED_COMPONENT_OFFSET`: the component's x/y offset stays in
/// font units regardless of the component's transform.
///
/// Only consulted when `TT_CONFIG_OPTION_COMPONENT_OFFSET_SCALED` is
/// defined, which this port does not do; the bit is exposed for
/// completeness so callers can round-trip the raw flags.
pub const SUBGLYPH_FLAG_UNSCALED_COMPONENT_OFFSET: u16 = 0x1000;
