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

//! CJK writing system global metrics (`afcjk.h` types).
//!
//! The glyph analysis routines of `afcjk.c` are ported in the
//! [`super::cjk`] module; the Indic writing system (`afindic.c`)
//! reuses these metrics as well.

use crate::metrics::StyleMetricsRec;
use crate::{BLUE_STRINGSET_MAX_LEN, DIMENSION_MAX, Width};

/// `AF_CJK_MAX_WIDTHS` (`afcjk.h`): slots in a CJK standard-width
/// table.
pub const CJK_MAX_WIDTHS: usize = 16;

/// `AF_CJK_IS_TOP_BLUE` (`afcjk.h`): true for a top/right blue zone.
pub const fn cjk_is_top_blue(properties: u16) -> bool {
    properties & crate::BLUE_PROPERTY_CJK_TOP != 0
}

/// `AF_CJK_IS_HORIZ_BLUE` (`afcjk.h`): true for a horizontal blue zone.
pub const fn cjk_is_horiz_blue(properties: u16) -> bool {
    properties & crate::BLUE_PROPERTY_CJK_HORIZ != 0
}

/// `AF_CJK_BLUE_ACTIVE` (`afcjk.h`): zone height is <= 3/4px.
pub const CJK_BLUE_ACTIVE: u32 = 1 << 0;
/// `AF_CJK_BLUE_TOP` (`afcjk.h`): result of [`cjk_is_top_blue`].
pub const CJK_BLUE_TOP: u32 = 1 << 1;
/// `AF_CJK_BLUE_ADJUSTMENT` (`afcjk.h`): used for the scale
/// adjustment optimization.
pub const CJK_BLUE_ADJUSTMENT: u32 = 1 << 2;

/// `AF_CJKBlueRec` (`afcjk.h`): one blue zone of a CJK style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CjkBlue {
    /// Reference edge of the zone.
    pub r#ref: Width,
    /// Overshoot edge of the zone.
    pub shoot: Width,
    /// `AF_CJK_BLUE_*` bit mask.
    pub flags: u32,
}

/// `AF_CJKAxisRec` (`afcjk.h`): global horizontal or vertical metrics
/// of a CJK style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CjkAxis {
    /// Scale from font units to 1/64th device pixels.
    pub scale: crate::Fixed,
    /// Added delta in 1/64th device pixels.
    pub delta: crate::Pos,
    /// Number of used widths.
    pub width_count: usize,
    /// Standard stem widths found in the face.
    pub widths: [Width; CJK_MAX_WIDTHS],
    /// Used when creating edges.
    pub edge_distance_threshold: crate::Pos,
    /// The default stem thickness.
    pub standard_width: crate::Pos,
    /// Is the standard width very light?
    pub extra_light: bool,
    /// Control overshoots also for horizontal metrics (CJK).
    pub control_overshoot: bool,
    /// Number of used blue zones (used for both dimensions).
    pub blue_count: usize,
    /// The blue zones of the style.
    pub blues: [CjkBlue; BLUE_STRINGSET_MAX_LEN],
    /// Original scale, for the scale adjustment optimization.
    pub org_scale: crate::Fixed,
    /// Original delta, for the scale adjustment optimization.
    pub org_delta: crate::Pos,
}

/// `AF_CJKMetricsRec` (`afcjk.h`): CJK global metrics (also used by
/// the Indic writing system).
#[derive(Clone, Debug)]
pub struct CjkMetrics {
    /// The style metrics base record.
    pub root: StyleMetricsRec,
    /// `face->units_per_EM`.
    pub units_per_em: u32,
    /// Global metrics for both dimensions.
    pub axis: [CjkAxis; DIMENSION_MAX],
}
