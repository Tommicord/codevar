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

//! Latin writing system global metrics (`aflatin.h` types).
//!
//! The glyph analysis routines of `aflatin.c` (and the alternate
//! `aflatin2.c` hinter) are ported in the [`super::latin`] module.

use crate::metrics::StyleMetricsRec;
use crate::{BLUE_STRINGSET_MAX_LEN, DIMENSION_MAX, LATIN_MAX_WIDTHS, Width};

/// `AF_LATIN_IS_TOP_BLUE` (`aflatin.h`): true for a top blue zone.
pub const fn latin_is_top_blue(properties: u16) -> bool {
    properties & crate::BLUE_PROPERTY_LATIN_TOP != 0
}

/// `AF_LATIN_IS_NEUTRAL_BLUE` (`aflatin.h`): true for a neutral blue.
pub const fn latin_is_neutral_blue(properties: u16) -> bool {
    properties & crate::BLUE_PROPERTY_LATIN_NEUTRAL != 0
}

/// `AF_LATIN_IS_X_HEIGHT_BLUE` (`aflatin.h`): true for an x-height blue.
pub const fn latin_is_x_height_blue(properties: u16) -> bool {
    properties & crate::BLUE_PROPERTY_LATIN_X_HEIGHT != 0
}

/// `AF_LATIN_IS_LONG_BLUE` (`aflatin.h`): true for a long blue zone.
pub const fn latin_is_long_blue(properties: u16) -> bool {
    properties & crate::BLUE_PROPERTY_LATIN_LONG != 0
}

/// `AF_LATIN_BLUE_ACTIVE` (`aflatin.h`): zone height is <= 3/4px.
pub const LATIN_BLUE_ACTIVE: u32 = 1 << 0;
/// `AF_LATIN_BLUE_TOP` (`aflatin.h`): we have a top blue zone.
pub const LATIN_BLUE_TOP: u32 = 1 << 1;
/// `AF_LATIN_BLUE_NEUTRAL` (`aflatin.h`): we have a neutral blue zone.
pub const LATIN_BLUE_NEUTRAL: u32 = 1 << 2;
/// `AF_LATIN_BLUE_ADJUSTMENT` (`aflatin.h`): used for the scale
/// adjustment optimization.
pub const LATIN_BLUE_ADJUSTMENT: u32 = 1 << 3;

/// `AF_LatinBlueRec` (`aflatin.h`): one blue zone of a Latin style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LatinBlue {
    /// Reference edge of the zone.
    pub r#ref: Width,
    /// Overshoot edge of the zone.
    pub shoot: Width,
    /// `AF_LATIN_BLUE_*` bit mask.
    pub flags: u32,
}

/// `AF_LATIN_HINTS_HORZ_SNAP` (`aflatin.h`): horizontal stem width
/// snapping.
pub const LATIN_HINTS_HORZ_SNAP: u32 = 1 << 0;
/// `AF_LATIN_HINTS_VERT_SNAP` (`aflatin.h`): vertical stem height
/// snapping.
pub const LATIN_HINTS_VERT_SNAP: u32 = 1 << 1;
/// `AF_LATIN_HINTS_STEM_ADJUST` (`aflatin.h`): stem width/height
/// adjustment.
pub const LATIN_HINTS_STEM_ADJUST: u32 = 1 << 2;
/// `AF_LATIN_HINTS_MONO` (`aflatin.h`): monochrome rendering.
pub const LATIN_HINTS_MONO: u32 = 1 << 3;

/// `AF_LatinAxisRec` (`aflatin.h`): global horizontal or vertical
/// metrics of a Latin style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LatinAxis {
    /// Scale from font units to 1/64th device pixels.
    pub scale: crate::Fixed,
    /// Added delta in 1/64th device pixels.
    pub delta: crate::Pos,
    /// Number of used widths.
    pub width_count: usize,
    /// Standard stem widths found in the face.
    pub widths: [Width; LATIN_MAX_WIDTHS],
    /// Used when creating edges.
    pub edge_distance_threshold: crate::Pos,
    /// The default stem thickness.
    pub standard_width: crate::Pos,
    /// Is the standard width very light?
    pub extra_light: bool,
    /// Number of used blue zones (ignored for vertical metrics).
    pub blue_count: usize,
    /// The blue zones of the style.
    pub blues: [LatinBlue; BLUE_STRINGSET_MAX_LEN],
    /// Original scale, for the scale adjustment optimization.
    pub org_scale: crate::Fixed,
    /// Original delta, for the scale adjustment optimization.
    pub org_delta: crate::Pos,
}

/// `AF_LatinMetricsRec` (`aflatin.h`): Latin global metrics.
#[derive(Clone, Debug)]
pub struct LatinMetrics {
    /// The style metrics base record.
    pub root: StyleMetricsRec,
    /// `face->units_per_EM`.
    pub units_per_em: u32,
    /// Global metrics for both dimensions.
    pub axis: [LatinAxis; DIMENSION_MAX],
}
