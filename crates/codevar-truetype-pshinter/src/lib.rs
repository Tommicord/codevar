//! Copyright 2026 Codevar Project
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

//! PostScript hinter — Rust port of FreeType 2.6.5's PS hinter (pshinter).
//!
//! This crate provides the PostScript hinting engine for Type 1 / CID / CFF fonts.
//! It mirrors the C implementation's structure:
//! - [`globals`] — font-level globals (blue zones, standard widths, scaling)
//! - [`recorder`] — per-glyph hint recording (stems, masks, counters)
//! - [`algorithm`] — hinting algorithm (alignment, interpolation, grid-fitting)
//! - [`service`] — service function table for module integration
//!
//! The API is standalone and depends only on `codevar_truetype_core`.

#![cfg_attr(not(test), no_std)]
extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;

use codevar_truetype_core::{
    Fixed, Pos, Vector, Outline, TtError, TtResult,
    OUTLINE_CONTOURS_MAX, OUTLINE_POINTS_MAX,
};

use log::{trace, debug};

#[cfg(test)]
mod tests;

pub mod globals;
pub mod recorder;
pub mod algorithm;
pub mod service;

pub use globals::{
    Globals, Blues, BlueZone, BlueTable, Dimension, Width, Widths,
    Alignment, BLUE_ALIGN_NONE, BLUE_ALIGN_TOP, BLUE_ALIGN_BOT,
    MAX_BLUE_ZONES, MAX_STD_WIDTHS,
};
pub use recorder::{
    GlyphHints, Hint, HintTable, Mask, MaskTable, Dimension as RecorderDimension,
    HintType, HintFlags,
};
pub use algorithm::{
    Glyph, Hint as AlgoHint, HintTable as AlgoHintTable, Zone, Point, Contour,
    GlyphFlags, PointFlags, PointFlags2, Dir,
};
pub use service::PshinterService;