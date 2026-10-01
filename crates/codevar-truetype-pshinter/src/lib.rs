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

//! # Codevar PostScript hinter
//!
//! Rus Port of FreeType 2.6's PostScript hinter
//! (`src/pshinter/`): the Type 1 / Type 2 glyph hint recorder
//! (`pshrec.c`), the font-wide blue-zone and snap-width data
//! (`pshglob.c`), the hinting algorithm (`pshalgo.c`) and the module
//! facade (`pshmod.c`, `pshinter.c`, `pshpic.c`, `pshints.h`) that the
//! `type1`, `cid` and `cff` drivers consume through `PSHinter_Service`.
//!
//! | FreeType file | Rust items |
//! |---------------|------------|
//! | `pshrec.c/h` | [`PsHints`], [`T1Hints`], [`T2Hints`] |
//! | `pshglob.c/h` | [`PshGlobals`], [`T1Private`], [`psh_globals_new`] |
//! | `pshalgo.c/h` | [`ps_hints_apply`] |
//! | `pshmod.c`, `pshints.h` | [`PshinterModule`], [`T1HintsFuncs`], [`T2HintsFuncs`], [`PshGlobalsFuncs`] |
//! | `pshnterr.h` | [`TtError`] / [`TtResult`] (re-exported from core) |
//!
//! # Example
//!
//! ```
//! use codevar_truetype_pshinter::{
//!     PshGlobals, PshinterModule, RenderMode, T1Private, T2HintsFuncs,
//! };
//!
//! let private = T1Private::default();
//! let globals = PshGlobals::new(Default::default(), &private)?;
//! let mut globals = globals;
//! globals.set_scale(0x4000, 0x4000, 0, 0);
//!
//! let mut module = PshinterModule::new();
//! let funcs: &T2HintsFuncs = module.t2_funcs();
//! let mut hints = module.t2_hints();
//! (funcs.open)(&mut hints.recorder());
//! # Ok::<(), codevar_truetype_pshinter::TtError>(())
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]
extern crate alloc;

use alloc::vec::Vec;

use codevar_truetype_core::{CURVE_TAG_ON, abs_pos, corner_is_flat, div_fix, mul_div, mul_fix, round_fix};
pub use codevar_truetype_core::{Fixed, Memory, Outline, Pos, RenderMode, TtError, TtResult, Vector};

/// `PS_GLOBALS_MAX_BLUE_ZONES` (`pshglob.h`): maximum number of blue zones
/// in a font's global hinting structure.
pub const PS_GLOBALS_MAX_BLUE_ZONES: usize = 16;

/// `PS_GLOBALS_MAX_STD_WIDTHS` (`pshglob.h`): maximum number of standard
/// and snap widths (or heights) in either direction.
pub const PS_GLOBALS_MAX_STD_WIDTHS: usize = 16;

/// `PS_HINT_FLAG_GHOST` (`pshrec.h`): the hint is a Type 1 ghost stem.
pub const PS_HINT_FLAG_GHOST: u32 = 1;

/// `PS_HINT_FLAG_BOTTOM` (`pshrec.h`): the ghost stem is a bottom ghost
/// stem (Type 1 length `-21`), i.e. its position has already been shifted.
pub const PS_HINT_FLAG_BOTTOM: u32 = 2;

/// `PSH_HINT_ACTIVE` (`pshalgo.h`): the hint is part of the active set of
/// the current hint mask.
const PSH_HINT_ACTIVE: u32 = 4;

/// `PSH_HINT_FITTED` (`pshalgo.h`): the hint has been grid-fitted.
const PSH_HINT_FITTED: u32 = 8;

/// `PSH_POINT_OFF` (`pshalgo.h`): the point is off the curve.
const PSH_POINT_OFF: u32 = 1;

/// `PSH_POINT_SMOOTH` (`pshalgo.h`): the point is smooth.
const PSH_POINT_SMOOTH: u32 = 2;

/// `PSH_POINT_INFLEX` (`pshalgo.h`): the point is an inflection point.
const PSH_POINT_INFLEX: u32 = 4;

/// `PSH_POINT_STRONG` (`pshalgo.h`): the point is strong.
const PSH_POINT_STRONG: u32 = 16;

/// `PSH_POINT_FITTED` (`pshalgo.h`): the point is already fitted.
const PSH_POINT_FITTED: u32 = 32;

/// `PSH_POINT_EXTREMUM` (`pshalgo.h`): the point is a local extremum.
const PSH_POINT_EXTREMUM: u32 = 64;

/// `PSH_POINT_POSITIVE` (`pshalgo.h`): the extremum has a positive
/// contour flow.
const PSH_POINT_POSITIVE: u32 = 128;

/// `PSH_POINT_NEGATIVE` (`pshalgo.h`): the extremum has a negative
/// contour flow.
const PSH_POINT_NEGATIVE: u32 = 256;

/// `PSH_POINT_EDGE_MIN` (`pshalgo.h`): the point is aligned to the
/// left/bottom edge of a stem.
const PSH_POINT_EDGE_MIN: u32 = 512;

/// `PSH_POINT_EDGE_MAX` (`pshalgo.h`): the point is aligned to the
/// top/right edge of a stem.
const PSH_POINT_EDGE_MAX: u32 = 1024;

/// `PSH_DIR_NONE` (`pshalgo.h`): the point has no preferred direction.
const PSH_DIR_NONE: i32 = 4;

/// `PSH_DIR_UP` (`pshalgo.h`): a near-vertical segment pointing up.
const PSH_DIR_UP: i32 = -1;

/// `PSH_DIR_DOWN` (`pshalgo.h`): a near-vertical segment pointing down.
const PSH_DIR_DOWN: i32 = 1;

/// `PSH_DIR_LEFT` (`pshalgo.h`): a near-horizontal segment pointing left.
const PSH_DIR_LEFT: i32 = -2;

/// `PSH_DIR_RIGHT` (`pshalgo.h`): a near-horizontal segment pointing right.
const PSH_DIR_RIGHT: i32 = 2;

/// `PSH_DIR_HORIZONTAL` (`pshalgo.h`): magnitude of the horizontal
/// direction (compared with `PSH_DIR_COMPARE`).
const PSH_DIR_HORIZONTAL: i32 = 2;

/// `PSH_DIR_VERTICAL` (`pshalgo.h`): magnitude of the vertical direction.
const PSH_DIR_VERTICAL: i32 = 1;

/// `PSH_BLUE_ALIGN_NONE` (`pshglob.h`): a stem snaps to no blue zone.
pub const PSH_BLUE_ALIGN_NONE: u8 = 0;

/// `PSH_BLUE_ALIGN_TOP` (`pshglob.h`): the top of the stem snaps to a
/// blue zone.
pub const PSH_BLUE_ALIGN_TOP: u8 = 1;

/// `PSH_BLUE_ALIGN_BOT` (`pshglob.h`): the bottom of the stem snaps to a
/// blue zone.
pub const PSH_BLUE_ALIGN_BOT: u8 = 2;

/// `PSH_BLUE_ALIGN_TOP | PSH_BLUE_ALIGN_BOT` (`pshglob.h`): both edges of
/// the stem snap to blue zones.
pub const PSH_BLUE_ALIGN_BOTH: u8 = 3;

/// `PSH_STRONG_THRESHOLD` (`pshalgo.c`): the accepted shift for strong
/// points, in 1/64 of a pixel.
const PSH_STRONG_THRESHOLD: i64 = 32;

/// `PSH_STRONG_THRESHOLD_MAXIMUM` (`pshalgo.c`): the same threshold
/// expressed in font units.
const PSH_STRONG_THRESHOLD_MAXIMUM: i32 = 30;

/// `PSH_MAX_STRONG_INTERNAL` (`pshalgo.c`): the number of strong points
/// that fit into the stack buffer of
/// `psh_glyph_interpolate_normal_points`.
const PSH_MAX_STRONG_INTERNAL: usize = 16;

/// `FT_PIX_FLOOR(x)` (`ftobjs.h`): rounds a 26.6 value down to the whole
/// pixel.
#[inline]
pub const fn ft_pix_floor(x: Pos) -> Pos {
    x & !63
}

/// `FT_PIX_ROUND(x)` (`ftobjs.h`): rounds a 26.6 value to the nearest
/// pixel, implemented exactly as FreeType's `FT_PIX_FLOOR(x + 32)`.
///
/// # Porting note
///
/// The core crate also exports a `pix_round` helper, but its formula
/// `(x + 32 + 63) & !63` does not match FreeType 2.6's macro, so this
/// crate defines its own to stay bit-compatible with the C hinter.
#[inline]
pub const fn ft_pix_round(x: Pos) -> Pos {
    (x + 32) & !63
}

/// `FIXED_TO_INT(x)` (`ftcalc.h`): `FT_RoundFix(x) >> 16`, the integer
/// (font-unit) part of a 16.16 value, truncated to `FT_Int` as in C.
#[inline]
fn fixed_to_int(x: Fixed) -> i32 {
    (round_fix(x) >> 16) as i32
}

/// `PSH_DIR_COMPARE(d1, d2)` (`pshalgo.h`): `true` when both directions
/// are equal or opposite.
#[inline]
fn dir_compare(d1: i32, d2: i32) -> bool {
    d1 == d2 || d1 == -d2
}

/// `T1_Private` (an alias of `PS_PrivateRec` in `t1tables.h`), reduced to
/// the fields read by [`psh_globals_new`].
///
/// The field list, types and array sizes are the real ones of
/// `PS_PrivateRec`; dictionary keys that the PostScript globals never
/// consume (`unique_id`, `lenIV`, `force_bold`, `expansion_factor`,
/// `language_group`, `password`, `min_feature`, ...) are not part of this
/// port's struct.
///
/// The [`Default`] implementation uses the values the Type 1 driver
/// installs before parsing a font (`t1load.c`): `blue_shift == 7`,
/// `blue_fuzz == 1` and `blue_scale == 0.039625 * 0x10000 * 1000`, with
/// every other field cleared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct T1Private {
    /// `num_blue_values`: number of entries in `blue_values` (0..14).
    pub num_blue_values: u8,
    /// `num_other_blues`: number of entries in `other_blues` (0..10).
    pub num_other_blues: u8,
    /// `num_family_blues`: number of entries in `family_blues` (0..14).
    pub num_family_blues: u8,
    /// `num_family_other_blues`: number of entries in
    /// `family_other_blues` (0..10).
    pub num_family_other_blues: u8,
    /// `blue_values[14]`: the `BlueValues` array, font-unit pairs.
    pub blue_values: [i16; 14],
    /// `other_blues[10]`: the `OtherBlues` array, font-unit pairs.
    pub other_blues: [i16; 10],
    /// `family_blues[14]`: the `FamilyBlues` array, font-unit pairs.
    pub family_blues: [i16; 14],
    /// `family_other_blues[10]`: the `FamilyOtherBlues` array.
    pub family_other_blues: [i16; 10],
    /// `blue_scale`: the `BlueScale` value, stored 1000 times its real
    /// value as FreeType does.
    pub blue_scale: Fixed,
    /// `blue_shift`: the `BlueShift` value, in font units.
    pub blue_shift: i32,
    /// `blue_fuzz`: the `BlueFuzz` value, in font units.
    pub blue_fuzz: i32,
    /// `standard_width[1]`: `StdHW`, the standard stem width.
    pub standard_width: [u16; 1],
    /// `standard_height[1]`: `StdVW`, the standard stem height.
    pub standard_height: [u16; 1],
    /// `num_snap_widths`: number of entries in `snap_widths` (0..13).
    pub num_snap_widths: u8,
    /// `num_snap_heights`: number of entries in `snap_heights` (0..13).
    pub num_snap_heights: u8,
    /// `snap_widths[13]`: the `StemSnapH` array.
    pub snap_widths: [i16; 13],
    /// `snap_heights[13]`: the `StemSnapV` array.
    pub snap_heights: [i16; 13],
}

impl Default for T1Private {
    fn default() -> Self {
        T1Private {
            num_blue_values: 0,
            num_other_blues: 0,
            num_family_blues: 0,
            num_family_other_blues: 0,
            blue_values: [0; 14],
            other_blues: [0; 10],
            family_blues: [0; 14],
            family_other_blues: [0; 10],
            // (FT_Fixed)( 0.039625 * 0x10000L * 1000 )
            blue_scale: 2_596_750,
            blue_shift: 7,
            blue_fuzz: 1,
            standard_width: [0; 1],
            standard_height: [0; 1],
            num_snap_widths: 0,
            num_snap_heights: 0,
            snap_widths: [0; 13],
            snap_heights: [0; 13],
        }
    }
}

/***************************************************************************/
/***************************************************************************/
/*****                                                                 *****/
/*****                    STANDARD WIDTHS (pshglob.c)                 *****/
/*****                                                                 *****/
/***************************************************************************/
/***************************************************************************/

/// `PSH_WidthRec` (`pshglob.h`): one standard or snap width.
///
/// `org` is in font units, `cur`/`fit` in 26.6 device coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PshWidth {
    /// `org`: the raw font-unit value.
    org: i32,
    /// `cur`: the scaled value (`FT_MulFix(org, scale_mult)`).
    cur: Pos,
    /// `fit`: `FT_PIX_ROUND(cur)`, the pixel-aligned value.
    fit: Pos,
}

/// `PSH_WidthsRec` (`pshglob.h`): the standard and snap width table of one
/// dimension. The first entry is the standard width/height itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PshWidths {
    /// `count`: the number of meaningful entries in `widths`.
    count: usize,
    /// `widths[PS_GLOBALS_MAX_STD_WIDTHS]`: the entries.
    widths: [PshWidth; PS_GLOBALS_MAX_STD_WIDTHS],
}

/// `PSH_DimensionRec` (`pshglob.h`): per-dimension scale data.
///
/// Dimension 0 holds the X coordinates / vertical stems data, dimension 1
/// the Y coordinates / horizontal stems data, exactly as the C comment
/// states.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PshDimension {
    /// `stdw`: the standard/snap width table of this dimension.
    stdw: PshWidths,
    /// `scale_mult`: the font-units to 26.6 scale (`FT_Fixed`).
    scale_mult: Fixed,
    /// `scale_delta`: the extra translation applied with the scale.
    scale_delta: Pos,
}

/// `psh_globals_scale_widths` (`pshglob.c`): scales the widths/heights
/// table of `direction` with the dimension's current scale.
///
/// The first entry (the standard width) is scaled and rounded; every
/// other entry that ends up within 128/64 = 2 pixels of the standard one
/// collapses onto it, so that snapping cannot produce a width that is
/// nearly indistinguishable from the standard width.
///
/// # Performance
///
/// Runs in `O(count)` with `count <= 16`; it is only executed when
/// `set_scale` actually changes the scale.
fn psh_globals_scale_widths(globals: &mut PshGlobals, direction: usize) {
    let Some(dim) = globals.dimension.get_mut(direction) else {
        return;
    };
    let scale = dim.scale_mult;
    let count = dim.stdw.count;
    if count == 0 || count > PS_GLOBALS_MAX_STD_WIDTHS {
        return;
    }

    let stand = &mut dim.stdw.widths[0];
    stand.cur = mul_fix(i64::from(stand.org), scale);
    stand.fit = ft_pix_round(stand.cur);
    let stand_cur = stand.cur;

    for width in dim.stdw.widths.iter_mut().take(count).skip(1) {
        let mut w = mul_fix(i64::from(width.org), scale);
        let dist = (w - stand_cur).wrapping_abs();

        if dist < 128 {
            w = stand_cur;
        }

        width.cur = w;
        width.fit = ft_pix_round(w);
    }
}

/// `PSH_Blue_ZoneRec` (`pshglob.h`): one blue zone.
///
/// The `org_*` fields are font units, the `cur_*` fields are the scaled
/// 26.6 values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PshBlueZone {
    /// `org_ref`: the font-unit reference (flat) edge of the zone.
    org_ref: i32,
    /// `org_delta`: the font-unit thickness of the zone.
    org_delta: i32,
    /// `org_top`: the font-unit top edge of the zone.
    org_top: i32,
    /// `org_bottom`: the font-unit bottom edge of the zone.
    org_bottom: i32,
    /// `cur_ref`: the scaled reference edge, pixel-rounded.
    cur_ref: Pos,
    /// `cur_delta`: the scaled thickness of the zone.
    cur_delta: Pos,
    /// `cur_bottom`: the scaled bottom edge of the zone.
    cur_bottom: Pos,
    /// `cur_top`: the scaled top edge of the zone.
    cur_top: Pos,
}

/// `PSH_Blue_TableRec` (`pshglob.h`): a sorted table of blue zones (top
/// zones, bottom zones, family top zones or family bottom zones).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PshBlueTable {
    /// `count`: the number of meaningful entries in `zones`.
    count: usize,
    /// `zones[PS_GLOBALS_MAX_BLUE_ZONES]`: the zones, sorted by
    /// `org_ref`.
    zones: [PshBlueZone; PS_GLOBALS_MAX_BLUE_ZONES],
}

impl PshBlueTable {
    /// Returns the meaningful (allocated) part of the zone array.
    #[inline]
    fn zones(&self) -> &[PshBlueZone] {
        &self.zones[..self.count.min(PS_GLOBALS_MAX_BLUE_ZONES)]
    }

    /// Returns the meaningful (allocated) part of the zone array for
    /// mutation.
    #[inline]
    fn zones_mut(&mut self) -> &mut [PshBlueZone] {
        let count = self.count.min(PS_GLOBALS_MAX_BLUE_ZONES);
        &mut self.zones[..count]
    }
}

/// `PSH_BluesRec` (`pshglob.h`): all blue zones of a font plus the
/// `BlueScale`, `BlueShift`, `BlueFuzz` and overshoot-suppression state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PshBlues {
    /// `normal_top`: the top zones of the `BlueValues`/`OtherBlues`
    /// arrays.
    normal_top: PshBlueTable,
    /// `normal_bottom`: the bottom zones of the `BlueValues`/`OtherBlues`
    /// arrays.
    normal_bottom: PshBlueTable,
    /// `family_top`: the top zones of the `FamilyBlues` arrays.
    family_top: PshBlueTable,
    /// `family_bottom`: the bottom zones of the `FamilyBlues` arrays.
    family_bottom: PshBlueTable,
    /// `blue_scale`: `BlueScale`, 1000 times the real value.
    blue_scale: Fixed,
    /// `blue_shift`: `BlueShift`, in font units.
    blue_shift: i32,
    /// `blue_threshold`: the largest font-unit distance whose scaled
    /// value still rounds below half a pixel (computed by
    /// `psh_blues_scale_zones`).
    blue_threshold: i32,
    /// `blue_fuzz`: `BlueFuzz`, in font units.
    blue_fuzz: i32,
    /// `no_overshoots`: whether overshoots are suppressed at the current
    /// scale.
    no_overshoots: bool,
}

/// `PSH_AlignmentRec` (`pshglob.h`): the result of snapping a stem to one
/// or two blue zones.
///
/// `align` is a bit mask of [`PSH_BLUE_ALIGN_NONE`],
/// [`PSH_BLUE_ALIGN_TOP`] and [`PSH_BLUE_ALIGN_BOT`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PshAlignment {
    /// `align`: which edges are aligned (`PSH_BLUE_ALIGN_*`).
    pub align: u8,
    /// `align_top`: the scaled reference of the top blue zone.
    pub align_top: Pos,
    /// `align_bot`: the scaled reference of the bottom blue zone.
    pub align_bot: Pos,
}

impl PshAlignment {
    /// `true` when the top edge of the stem snaps to a blue zone.
    #[inline]
    pub fn is_top(self) -> bool {
        self.align & PSH_BLUE_ALIGN_TOP != 0
    }

    /// `true` when the bottom edge of the stem snaps to a blue zone.
    #[inline]
    pub fn is_bottom(self) -> bool {
        self.align & PSH_BLUE_ALIGN_BOT != 0
    }
}

/// `psh_blues_set_zones_0` (`pshglob.c`): reads `read_count` font-unit
/// values (pairs) and inserts them into the sorted top/bottom zone
/// tables.
///
/// The first pair of `blue_values`, and every pair of `other_blues`
/// (signalled by `is_others`), describes a bottom zone; the remaining
/// pairs of `blue_values` describe top zones. Zones sharing the same
/// reference position are merged, keeping the largest delta.
///
/// # Safety note
///
/// FreeType writes into a fixed 16-entry array without checking for
/// overflow; this port drops the insertion when the target table is
/// already full instead of writing past its end.
fn psh_blues_set_zones_0(
    is_others: bool,
    read_count: usize,
    read: &[i16],
    top_table: &mut PshBlueTable,
    bot_table: &mut PshBlueTable,
) {
    let read_count = read_count.min(read.len());
    let mut count_top = top_table.count.min(PS_GLOBALS_MAX_BLUE_ZONES);
    let mut count_bot = bot_table.count.min(PS_GLOBALS_MAX_BLUE_ZONES);
    let mut first = true;
    let mut index = 0usize;

    while read_count > 1 && index + 1 < read_count {
        let r0 = i32::from(read[index]);
        let r1 = i32::from(read[index + 1]);

        // read blue zone entry, and select target top/bottom zone
        let (reference, delta, top) = if first || is_others {
            first = false;
            (r1, r0 - r1, false)
        } else {
            (r0, r1 - r0, true)
        };

        let (zones, count) = if top {
            (&mut top_table.zones, &mut count_top)
        } else {
            (&mut bot_table.zones, &mut count_bot)
        };
        let count_in = *count;

        // insert into the sorted table
        let mut zone = 0usize;
        let mut skip = false;
        while zone < count_in {
            let entry = &zones[zone];
            if reference < entry.org_ref {
                break;
            }
            if reference == entry.org_ref {
                // two zones on the same reference position: only keep
                // the largest one
                let delta0 = entry.org_delta;
                if delta < 0 {
                    if delta < delta0 {
                        zones[zone].org_delta = delta;
                    }
                } else if delta > delta0 {
                    zones[zone].org_delta = delta;
                }
                skip = true;
                break;
            }
            zone += 1;
        }

        // a skipped zone keeps the larger of the two equal-reference
        // entries; a full table is ignored even though FreeType would
        // overflow its fixed-size array here
        if !skip && count_in < PS_GLOBALS_MAX_BLUE_ZONES {
            // shift the tail of the table one slot to the right
            for k in (1..=(count_in - zone)).rev() {
                zones[zone + k] = zones[zone + k - 1];
            }
            zones[zone].org_ref = reference;
            zones[zone].org_delta = delta;
            *count = count_in + 1;
        }

        index += 2;
    }

    top_table.count = count_top;
    bot_table.count = count_bot;
}

/// `psh_blues_set_zones` (`pshglob.c`): builds the sorted, sanitized and
/// fuzz-expanded blue zone tables of one family of a font.
///
/// `family` selects the `FamilyBlues`/`FamilyOtherBlues` tables instead of
/// the `BlueValues`/`OtherBlues` tables.
fn psh_blues_set_zones(
    target: &mut PshBlues,
    count: usize,
    blues: &[i16],
    count_others: usize,
    other_blues: &[i16],
    fuzz: i32,
    family: bool,
) {
    let (top_table, bot_table) = if family {
        (&mut target.family_top, &mut target.family_bottom)
    } else {
        (&mut target.normal_top, &mut target.normal_bottom)
    };

    // read the input blue zones and build two sorted tables
    top_table.count = 0;
    bot_table.count = 0;

    psh_blues_set_zones_0(false, count, blues, top_table, bot_table);
    psh_blues_set_zones_0(true, count_others, other_blues, top_table, bot_table);

    let count_top = top_table.count.min(PS_GLOBALS_MAX_BLUE_ZONES);
    let count_bot = bot_table.count.min(PS_GLOBALS_MAX_BLUE_ZONES);

    // sanitize the top table
    for i in 0..count_top {
        if i + 1 < count_top {
            let delta = top_table.zones[i + 1].org_ref - top_table.zones[i].org_ref;
            if top_table.zones[i].org_delta > delta {
                top_table.zones[i].org_delta = delta;
            }
        }
        let org_ref = top_table.zones[i].org_ref;
        let org_delta = top_table.zones[i].org_delta;
        top_table.zones[i].org_bottom = org_ref;
        top_table.zones[i].org_top = org_ref + org_delta;
    }

    // sanitize the bottom table
    for i in 0..count_bot {
        if i + 1 < count_bot {
            let delta = bot_table.zones[i].org_ref - bot_table.zones[i + 1].org_ref;
            if bot_table.zones[i].org_delta < delta {
                bot_table.zones[i].org_delta = delta;
            }
        }
        let org_ref = bot_table.zones[i].org_ref;
        let org_delta = bot_table.zones[i].org_delta;
        bot_table.zones[i].org_top = org_ref;
        bot_table.zones[i].org_bottom = org_ref + org_delta;
    }

    // expand the top and bottom tables with blue fuzz
    psh_blues_expand_fuzz(&mut top_table.zones, count_top, fuzz);
    psh_blues_expand_fuzz(&mut bot_table.zones, count_bot, fuzz);
}

/// The fuzz-expansion pass of `psh_blues_set_zones` (`pshglob.c`),
/// applied to one zone table.
///
/// The bottom edge of the lowest zone is lowered by `fuzz`, the top edge
/// of the highest zone is raised by `fuzz`, and the two edges of two
/// neighbouring zones are pushed apart by `fuzz` unless the gap between
/// them is smaller than twice the fuzz, in which case they simply meet in
/// the middle.
fn psh_blues_expand_fuzz(zones: &mut [PshBlueZone; PS_GLOBALS_MAX_BLUE_ZONES], count: usize, fuzz: i32) {
    if count == 0 || count > PS_GLOBALS_MAX_BLUE_ZONES {
        return;
    }

    // expand the bottom of the lowest zone normally
    zones[0].org_bottom -= fuzz;
    let mut top = zones[0].org_top;

    for k in 1..count {
        let bot = zones[k].org_bottom;
        let delta = bot - top;

        if delta / 2 < fuzz {
            zones[k - 1].org_top = top + delta / 2;
            zones[k].org_bottom = top + delta / 2;
        } else {
            zones[k - 1].org_top = top + fuzz;
            zones[k].org_bottom = bot - fuzz;
        }

        top = zones[k].org_top;
    }

    // expand the top of the highest zone normally
    zones[count - 1].org_top = top + fuzz;
}

/// `psh_blues_scale_zones` (`pshglob.c`): re-scales every blue zone when
/// the device transform changes.
///
/// The comment block of the C function explains the overshoot test: an
/// overshoot is suppressed when `scale < blue_scale`, shortened to that
/// comparison because a Type 1 `EM` is 1000 units.
fn psh_blues_scale_zones(blues: &mut PshBlues, scale: Fixed, delta: Pos) {
    // 1000 / 64 = 125 / 8
    blues.no_overshoots = if scale >= 0x20C49BA {
        scale < blues.blue_scale * 8 / 125
    } else {
        scale * 125 < blues.blue_scale * 8
    };
    // The blue threshold is the font-unit distance under which
    // overshoots are suppressed due to BlueShift even if the scale is
    // greater than BlueScale: the smallest distance with
    // `dist <= BlueShift && dist * scale <= 0.5 pixels`.
    let mut threshold = blues.blue_shift;
    while threshold > 0 && mul_fix(i64::from(threshold), scale) > 32 {
        threshold -= 1;
    }
    blues.blue_threshold = threshold;

    // scale the four tables
    for table in [
        &mut blues.normal_top,
        &mut blues.normal_bottom,
        &mut blues.family_top,
        &mut blues.family_bottom,
    ] {
        for zone in table
            .zones
            .iter_mut()
            .take(table.count.min(PS_GLOBALS_MAX_BLUE_ZONES))
        {
            zone.cur_top = mul_fix(i64::from(zone.org_top), scale) + delta;
            zone.cur_bottom = mul_fix(i64::from(zone.org_bottom), scale) + delta;
            zone.cur_ref = mul_fix(i64::from(zone.org_ref), scale) + delta;
            zone.cur_delta = mul_fix(i64::from(zone.org_delta), scale);

            // round the scaled reference position
            zone.cur_ref = ft_pix_round(zone.cur_ref);
        }
    }

    // process the families now: a normal zone whose reference sits less
    // than one pixel away from a family zone adopts the family zone's
    // scaled values
    psh_blues_adopt_family(&mut blues.normal_top, &blues.family_top, scale);
    psh_blues_adopt_family(&mut blues.normal_bottom, &blues.family_bottom, scale);
}

/// `psh_blues_scale_zones` (family matching part): one normal zone whose
/// reference is less than one pixel away from a family zone adopts the
/// family zone's already scaled values.
///
/// This is a helper so that `normal` and `family` are borrowed as two
/// disjoint fields of the surrounding `PshBlues` struct.
fn psh_blues_adopt_family(normal: &mut PshBlueTable, family: &PshBlueTable, scale: Fixed) {
    for zone1 in normal.zones_mut() {
        for zone2 in family.zones() {
            let mut delta = zone1.org_ref - zone2.org_ref;
            if delta < 0 {
                delta = -delta;
            }

            if mul_fix(i64::from(delta), scale) < 64 {
                zone1.cur_top = zone2.cur_top;
                zone1.cur_bottom = zone2.cur_bottom;
                zone1.cur_ref = zone2.cur_ref;
                zone1.cur_delta = zone2.cur_delta;
                break;
            }
        }
    }
}

/// `psh_calc_max_height` (`pshglob.c`): returns the largest height of the
/// blue zone pairs in `values` (or `cur_max` when it is larger).
///
/// `values` is walked two entries at a time, exactly like the C loop
/// (`count < num`, so a trailing odd entry reads `values[num]`, the next
/// element of the font's private dictionary array). When that element
/// falls outside the slice — which FreeType can only reach with a
/// dictionary that fills the array completely and reports an odd count —
/// the entry is skipped instead of reading out of bounds.
fn psh_calc_max_height(num: usize, values: &[i16], cur_max: i16) -> i16 {
    let num = num.min(values.len());
    let mut max = cur_max;
    let mut count = 0usize;

    while count < num {
        if let Some(&next) = values.get(count + 1) {
            let cur_height = next - values[count];
            if cur_height > max {
                max = cur_height;
            }
        }
        count += 2;
    }

    max
}

/// `psh_blues_snap_stem` (`pshglob.c`): snaps a stem to one or two blue
/// zones.
///
/// `stem_top` and `stem_bot` are font-unit coordinates of the stem. The
/// top edge is looked up in the top zones (from the lowest zone up) and
/// the bottom edge in the bottom zones (from the highest zone down).
///
/// This is the internal helper behind [`PshGlobals::snap_stem`]; it takes
/// [`PshBlues`] directly, which is why it is crate-private.
fn psh_blues_snap_stem(blues: &PshBlues, stem_top: i32, stem_bot: i32, alignment: &mut PshAlignment) {
    alignment.align = PSH_BLUE_ALIGN_NONE;
    alignment.align_top = 0;
    alignment.align_bot = 0;

    let no_shoots = blues.no_overshoots;
    let fuzz = i64::from(blues.blue_fuzz);
    let threshold = i64::from(blues.blue_threshold);

    // look up the stem top in the top zones table
    let count = blues
        .normal_top
        .count
        .min(PS_GLOBALS_MAX_BLUE_ZONES);
    for zone in blues.normal_top.zones[..count].iter() {
        let delta = i64::from(stem_top) - i64::from(zone.org_bottom);
        if delta < -fuzz {
            break;
        }

        if i64::from(stem_top) <= i64::from(zone.org_top) + fuzz {
            if no_shoots || delta <= threshold {
                alignment.align |= PSH_BLUE_ALIGN_TOP;
                alignment.align_top = zone.cur_ref;
            }
            break;
        }
    }

    // look up the stem bottom in the bottom zones table
    let count = blues
        .normal_bottom
        .count
        .min(PS_GLOBALS_MAX_BLUE_ZONES);
    let mut index = count;
    while index > 0 {
        index -= 1;
        let zone = &blues.normal_bottom.zones[index];

        let delta = i64::from(zone.org_top) - i64::from(stem_bot);
        if delta < -fuzz {
            break;
        }

        if i64::from(stem_bot) >= i64::from(zone.org_bottom) - fuzz {
            if no_shoots || delta < threshold {
                alignment.align |= PSH_BLUE_ALIGN_BOT;
                alignment.align_bot = zone.cur_ref;
            }
            break;
        }
    }
}

/// `PSH_GlobalsRec` (`pshglob.h`): the per-font global hinting data used
/// by the hinter (`pshalgo.c`) while fitting a glyph.
///
/// Create one with [`PshGlobals::new`] (C: `psh_globals_new`) and rescale
/// it with [`PshGlobals::set_scale`] whenever the size changes.
#[derive(Clone, Copy, Debug, Default)]
pub struct PshGlobals {
    /// `memory`: the allocator handle, as in the C record.
    memory: Memory,
    /// `dimension[2]`: scale and standard width data per dimension.
    dimension: [PshDimension; 2],
    /// `blues`: the blue zones of the font.
    blues: PshBlues,
}

impl PshGlobals {
    /// `psh_globals_new` (`pshglob.c`): builds the global hinting data
    /// from a font's private dictionary.
    ///
    /// The standard width/height and their snap tables are copied into
    /// dimension 1 and 0 respectively (exactly like the C code, which
    /// stores the widths in `dimension[1]` and the heights in
    /// `dimension[0]`), the blue zones are built with
    /// `psh_blues_set_zones`, and `blue_scale` is clamped to
    /// `1000 / max(blue zone heights)` so that no blue zone becomes
    /// thinner than one pixel.
    ///
    /// # Errors
    ///
    /// Never fails with the current allocator: [`Memory::alloc_vec`]
    /// reports [`TtError::OUT_OF_MEMORY`] only for absurd sizes, which
    /// the bounded private-dictionary counters cannot reach. The
    /// `Result` is kept because the C signature returns `FT_Error`.
    pub fn new(memory: Memory, private: &T1Private) -> TtResult<Self> {
        psh_globals_new(memory, private)
    }

    /// `psh_globals_set_scale` (`pshglob.c`): installs a new device
    /// transform.
    ///
    /// `x_scale`/`y_scale` convert font units into 1/64th of a pixel,
    /// `x_delta`/`y_delta` are extra translations. The widths of a
    /// dimension and (for `y_scale`) the blue zones are only recomputed
    /// when the corresponding values actually change.
    pub fn set_scale(&mut self, x_scale: Fixed, y_scale: Fixed, x_delta: Pos, y_delta: Pos) {
        psh_globals_set_scale(self, x_scale, y_scale, x_delta, y_delta);
    }

    /// `psh_globals_destroy` (`pshglob.c`): releases the global hinting
    /// data.
    ///
    /// The C function zeroes the table counters and frees the record;
    /// here the same counters are cleared and the value is dropped.
    pub fn destroy(self) {
        psh_globals_destroy(self);
    }

    /// Snaps the stem `[stem_bot, stem_top]` (font units) to the font's
    /// blue zones; this is the public wrapper of `psh_blues_snap_stem`.
    pub fn snap_stem(&self, stem_top: i32, stem_bot: i32) -> PshAlignment {
        let mut alignment = PshAlignment::default();
        psh_blues_snap_stem(&self.blues, stem_top, stem_bot, &mut alignment);
        alignment
    }

    /// Returns `(org, cur, fit)` for entry `index` of the standard/snap
    /// width table of `dimension` (0 = heights, 1 = widths), or `None`
    /// when the dimension or the index is out of range.
    ///
    /// This mirrors the fields of `PSH_WidthRec`; `fit` is computed by
    /// `psh_globals_scale_widths` and is otherwise only observable
    /// through this accessor.
    pub fn std_width(&self, dimension: usize, index: usize) -> Option<(i32, Pos, Pos)> {
        let dim = self.dimension.get(dimension)?;
        if index >= dim.stdw.count || index >= PS_GLOBALS_MAX_STD_WIDTHS {
            return None;
        }
        let w = dim.stdw.widths[index];
        Some((w.org, w.cur, w.fit))
    }

    /// Returns `(org_ref, org_delta, org_top, org_bottom, cur_ref,
    /// cur_delta, cur_bottom, cur_top)` for zone `index` of `table`
    /// (`0` = normal top, `1` = normal bottom, `2` = family top,
    /// `3` = family bottom), or `None` when out of range.
    pub fn blue_zone(&self, table: usize, index: usize) -> Option<PshBlueZoneView> {
        let zone = match table {
            0 => self.blues.normal_top.zones.get(index)?,
            1 => self.blues.normal_bottom.zones.get(index)?,
            2 => self.blues.family_top.zones.get(index)?,
            3 => self.blues.family_bottom.zones.get(index)?,
            _ => return None,
        };
        Some(PshBlueZoneView {
            org_ref: zone.org_ref,
            org_delta: zone.org_delta,
            org_top: zone.org_top,
            org_bottom: zone.org_bottom,
            cur_ref: zone.cur_ref,
            cur_delta: zone.cur_delta,
            cur_bottom: zone.cur_bottom,
            cur_top: zone.cur_top,
        })
    }

    /// Returns the number of meaningful zones in `table`
    /// (`0` = normal top, `1` = normal bottom, `2` = family top,
    /// `3` = family bottom).
    pub fn blue_zone_count(&self, table: usize) -> usize {
        match table {
            0 => self.blues.normal_top.count,
            1 => self.blues.normal_bottom.count,
            2 => self.blues.family_top.count,
            3 => self.blues.family_bottom.count,
            _ => 0,
        }
    }

    /// Returns `(scale_mult, scale_delta)` of `dimension`
    /// (0 = X, 1 = Y).
    pub fn scale(&self, dimension: usize) -> Option<(Fixed, Pos)> {
        let dim = self.dimension.get(dimension)?;
        Some((dim.scale_mult, dim.scale_delta))
    }

    /// Returns `true` when overshoots are suppressed at the current
    /// scale (`PSH_BluesRec.no_overshoots`).
    pub fn no_overshoots(&self) -> bool {
        self.blues.no_overshoots
    }

    /// Returns the current `blue_threshold` (font units).
    pub fn blue_threshold(&self) -> i32 {
        self.blues.blue_threshold
    }

    /// Returns the allocator handle stored in the record
    /// (`PSH_GlobalsRec.memory`), which the algorithm module copies into
    /// the glyph record while hinting.
    pub fn memory(&self) -> Memory {
        self.memory
    }
}

/// A read-only view of one blue zone, as returned by
/// [`PshGlobals::blue_zone`].
///
/// The `org_*` fields are font units, the `cur_*` fields are scaled 26.6
/// coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PshBlueZoneView {
    /// `org_ref`: the font-unit reference edge of the zone.
    pub org_ref: i32,
    /// `org_delta`: the font-unit thickness of the zone.
    pub org_delta: i32,
    /// `org_top`: the font-unit top edge of the zone.
    pub org_top: i32,
    /// `org_bottom`: the font-unit bottom edge of the zone.
    pub org_bottom: i32,
    /// `cur_ref`: the scaled, pixel-rounded reference edge.
    pub cur_ref: Pos,
    /// `cur_delta`: the scaled thickness of the zone.
    pub cur_delta: Pos,
    /// `cur_bottom`: the scaled bottom edge of the zone.
    pub cur_bottom: Pos,
    /// `cur_top`: the scaled top edge of the zone.
    pub cur_top: Pos,
}

/// `psh_globals_new` (`pshglob.c`): see [`PshGlobals::new`].
///
/// # Errors
///
/// Returns [`TtError::OUT_OF_MEMORY`] when the backing allocation
/// reported by [`Memory::alloc_vec`] fails (the same error code the C
/// function would return from `FT_NEW`).
pub fn psh_globals_new(memory: Memory, private: &T1Private) -> TtResult<PshGlobals> {
    let mut globals = PshGlobals {
        memory,
        dimension: [PshDimension::default(), PshDimension::default()],
        blues: PshBlues::default(),
    };

    // copy the standard widths (dimension[1]) and heights (dimension[0])
    copy_std_widths(
        &mut globals.dimension[1],
        i32::from(private.standard_width[0]),
        &private.snap_widths,
        usize::from(private.num_snap_widths),
    );
    copy_std_widths(
        &mut globals.dimension[0],
        i32::from(private.standard_height[0]),
        &private.snap_heights,
        usize::from(private.num_snap_heights),
    );

    // copy the blue zones
    let num_blue_values = usize::from(private.num_blue_values);
    let num_other_blues = usize::from(private.num_other_blues);
    let num_family_blues = usize::from(private.num_family_blues);
    let num_family_other_blues = usize::from(private.num_family_other_blues);
    let fuzz = private.blue_fuzz;

    psh_blues_set_zones(
        &mut globals.blues,
        num_blue_values,
        &private.blue_values,
        num_other_blues,
        &private.other_blues,
        fuzz,
        false,
    );
    psh_blues_set_zones(
        &mut globals.blues,
        num_family_blues,
        &private.family_blues,
        num_family_other_blues,
        &private.family_other_blues,
        fuzz,
        true,
    );

    // limit the BlueScale value to `1 / max_of_blue_zone_heights`
    let mut max_height: i16 = 1;
    max_height = psh_calc_max_height(num_blue_values, &private.blue_values, max_height);
    max_height = psh_calc_max_height(num_other_blues, &private.other_blues, max_height);
    max_height = psh_calc_max_height(num_family_blues, &private.family_blues, max_height);
    max_height = psh_calc_max_height(num_family_other_blues, &private.family_other_blues, max_height);

    // BlueScale is scaled 1000 times
    let max_scale = div_fix(1000, i64::from(max_height));
    globals.blues.blue_scale = if private.blue_scale < max_scale {
        private.blue_scale
    } else {
        max_scale
    };

    globals.blues.blue_shift = private.blue_shift;
    globals.blues.blue_fuzz = private.blue_fuzz;

    globals.dimension[0].scale_mult = 0;
    globals.dimension[0].scale_delta = 0;
    globals.dimension[1].scale_mult = 0;
    globals.dimension[1].scale_delta = 0;

    Ok(globals)
}

/// Copies the standard width/height plus its snap table into `dim`,
/// the shared body of the two blocks of `psh_globals_new`.
fn copy_std_widths(dim: &mut PshDimension, std: i32, snaps: &[i16], num_snaps: usize) {
    dim.stdw.widths[0].org = std;

    let mut count = 1usize;
    let limit = num_snaps
        .min(snaps.len())
        .min(PS_GLOBALS_MAX_STD_WIDTHS.saturating_sub(1));
    for value in snaps.iter().take(limit) {
        dim.stdw.widths[count].org = i32::from(*value);
        count += 1;
    }

    dim.stdw.count = count;
}

/// `psh_globals_set_scale` (`pshglob.c`): see
/// [`PshGlobals::set_scale`].
pub fn psh_globals_set_scale(
    globals: &mut PshGlobals,
    x_scale: Fixed,
    y_scale: Fixed,
    x_delta: Pos,
    y_delta: Pos,
) {
    let (mult, delta) = {
        let dim = &globals.dimension[0];
        (dim.scale_mult, dim.scale_delta)
    };
    if x_scale != mult || x_delta != delta {
        globals.dimension[0].scale_mult = x_scale;
        globals.dimension[0].scale_delta = x_delta;
        psh_globals_scale_widths(globals, 0);
    }

    let (mult, delta) = {
        let dim = &globals.dimension[1];
        (dim.scale_mult, dim.scale_delta)
    };
    if y_scale != mult || y_delta != delta {
        globals.dimension[1].scale_mult = y_scale;
        globals.dimension[1].scale_delta = y_delta;
        psh_globals_scale_widths(globals, 1);
        psh_blues_scale_zones(&mut globals.blues, y_scale, y_delta);
    }
}

/// `psh_globals_destroy` (`pshglob.c`): see [`PshGlobals::destroy`].
pub fn psh_globals_destroy(mut globals: PshGlobals) {
    globals.dimension[0].stdw.count = 0;
    globals.dimension[1].stdw.count = 0;

    globals.blues.normal_top.count = 0;
    globals.blues.normal_bottom.count = 0;
    globals.blues.family_top.count = 0;
    globals.blues.family_bottom.count = 0;

    // C frees the record here; in Rust the counters are cleared and the
    // value (which is `Copy`) is explicitly ignored afterwards.
    let _ = globals;
}

/// `FT_PAD_CEIL( value, align )` (`ftimage.h`): rounds `value` up to the
/// next multiple of `align` (saturating, so a hostile size cannot wrap).
#[inline]
fn pad_ceil(value: usize, align: usize) -> usize {
    value.saturating_add(align - 1) & !(align - 1)
}

/// `PS_Hint_Type` (`pshrec.h`): the charstring dialect of a recording
/// session. `None` mirrors the zeroed C record, which accepts neither the
/// Type 1 nor the Type 2 operations until [`PsHints::open`] is called.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PsHintType {
    /// `PS_HINT_TYPE_1` (`pshrec.h`): Type 1 / CID charstrings.
    Type1 = 1,
    /// `PS_HINT_TYPE_2` (`pshrec.h`): Type 2 (CFF) charstrings.
    Type2 = 2,
}

/// `PS_HintRec` (`pshrec.h`): one recorded stem hint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PsHint {
    /// `pos`: font-unit position of the stem's lower edge.
    pos: i32,
    /// `len`: font-unit length of the stem (`0` for ghost stems).
    len: i32,
    /// `flags`: `PS_HINT_FLAG_GHOST` / `PS_HINT_FLAG_BOTTOM`.
    flags: u32,
}

/// `PS_Hint_TableRec` (`pshrec.h`): the growable stem table of one
/// dimension.
///
/// `hints.len()` is the C `num_hints`; the C `max_hints` capacity is the
/// `Vec` capacity, which only decides how often the table reallocates.
#[derive(Clone, Debug, Default)]
struct PsHintTable {
    /// `hints`: the recorded stems, in recording order.
    hints: Vec<PsHint>,
}

impl PsHintTable {
    /// `ps_hint_table_ensure` (`pshrec.c`): makes room for `count` stems,
    /// growing in multiples of eight exactly like `FT_PAD_CEIL( count, 8 )`.
    ///
    /// # Errors
    ///
    /// Returns [`TtError::OUT_OF_MEMORY`] when the reservation fails, the
    /// same code the C function returns from `FT_RENEW_ARRAY`.
    fn ensure(&mut self, count: usize) -> TtResult<()> {
        if count > self.hints.len() {
            let new_max = pad_ceil(count, 8);
            let extra = new_max - self.hints.len();
            self.hints
                .try_reserve(extra)
                .map_err(|_| TtError::OUT_OF_MEMORY)?;
        }
        Ok(())
    }

    /// `ps_hint_table_alloc` (`pshrec.c`): appends a cleared stem record
    /// and returns its index.
    ///
    /// # Errors
    ///
    /// Returns [`TtError::OUT_OF_MEMORY`] when the table cannot grow.
    fn alloc(&mut self) -> TtResult<usize> {
        let count = self.hints.len().saturating_add(1);
        self.ensure(count)?;
        self.hints.push(PsHint::default());
        Ok(self.hints.len() - 1)
    }
}

/// `PS_MaskRec` (`pshrec.h`): one hint or counter mask.
///
/// `max_bits` of the C record is `bytes.len() * 8` here, so the two
/// counters cannot disagree.
#[derive(Clone, Debug, Default)]
struct PsMask {
    /// `num_bits`: the number of meaningful bits.
    num_bits: usize,
    /// `bytes`: the mask bytes, MSB-first as in the Type 2 charstring.
    bytes: Vec<u8>,
    /// `end_point`: the outline index the mask applies up to.
    end_point: usize,
}

impl PsMask {
    /// `ps_mask_ensure` (`pshrec.c`): makes room for `count` bits, growing
    /// the byte array in multiples of eight bytes.
    ///
    /// # Errors
    ///
    /// Returns [`TtError::OUT_OF_MEMORY`] when the reservation fails.
    fn ensure(&mut self, count: usize) -> TtResult<()> {
        let old_max = self.bytes.len();
        let new_max = pad_ceil(count.saturating_add(7) >> 3, 8);
        if new_max > old_max {
            self.bytes
                .try_reserve(new_max - old_max)
                .map_err(|_| TtError::OUT_OF_MEMORY)?;
            // FreeType relies on freshly renewed memory being zero; this
            // port zeroes the new bytes explicitly so that no stale bit
            // can ever be read back.
            self.bytes.resize(new_max, 0);
        }
        Ok(())
    }

    /// `ps_mask_test_bit` (`pshrec.c`): `true` when bit `idx` is set.
    #[inline]
    fn test_bit(&self, idx: usize) -> bool {
        if idx >= self.num_bits {
            return false;
        }
        match self.bytes.get(idx >> 3) {
            Some(&byte) => byte & (0x80 >> (idx & 7)) != 0,
            None => false,
        }
    }

    /// `ps_mask_clear_bit` (`pshrec.c`): clears bit `idx`, ignoring bits
    /// at or above `num_bits`.
    ///
    /// The only C call site is the growth loop of `ps_mask_table_merge`,
    /// where `num_bits` is still the old count and the loop therefore
    /// clears nothing; the guard is kept so that the port behaves
    /// bit-for-bit like FreeType.
    #[inline]
    fn clear_bit(&mut self, idx: usize) {
        if idx >= self.num_bits {
            return;
        }
        if let Some(byte) = self.bytes.get_mut(idx >> 3) {
            *byte &= !(0x80 >> (idx & 7));
        }
    }

    /// `ps_mask_set_bit` (`pshrec.c`): sets bit `idx`, growing the mask
    /// when needed.
    ///
    /// # Errors
    ///
    /// Returns [`TtError::OUT_OF_MEMORY`] when the mask cannot grow.
    fn set_bit(&mut self, idx: usize) -> TtResult<()> {
        if idx >= self.num_bits {
            self.ensure(idx.saturating_add(1))?;
            self.num_bits = idx + 1;
        }
        if let Some(byte) = self.bytes.get_mut(idx >> 3) {
            *byte |= 0x80 >> (idx & 7);
        }
        Ok(())
    }

    /// The body of `ps_mask_table_set_bits` (`pshrec.c`): replaces the
    /// mask contents with `bit_count` bits read from `source` starting at
    /// `bit_pos`.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_ARGUMENT`] when `source` is too short to hold
    ///   `bit_pos + bit_count` bits (FreeType would read out of bounds).
    /// * [`TtError::OUT_OF_MEMORY`] when the mask cannot grow.
    fn set_bits(&mut self, source: &[u8], bit_pos: usize, bit_count: usize) -> TtResult<()> {
        let need = bit_pos.saturating_add(bit_count);
        if need > source.len().saturating_mul(8) {
            return Err(TtError::INVALID_ARGUMENT);
        }
        self.ensure(bit_count)?;
        self.num_bits = bit_count;

        // own exactly `ceil(bit_count / 8)` bytes: zero them first so no
        // bit of a previous session survives in the tail of the last byte
        let nbytes = bit_count.saturating_add(7) >> 3;
        for byte in self.bytes.iter_mut().take(nbytes) {
            *byte = 0;
        }

        let mut i = 0usize;
        while i < bit_count {
            let src = source[(bit_pos + i) >> 3];
            if src & (0x80 >> ((bit_pos + i) & 7)) != 0
                && let Some(byte) = self.bytes.get_mut(i >> 3)
            {
                *byte |= 0x80 >> (i & 7);
            }
            i += 1;
        }
        Ok(())
    }
}

/// `PS_Mask_TableRec` (`pshrec.h`): a growable table of masks.
///
/// `masks.len()` is the C `max_masks` capacity and the first `num_masks`
/// slots are the live masks, exactly like the C array; slots beyond
/// `num_masks` keep their allocated bytes for reuse after a merge.
#[derive(Clone, Debug, Default)]
struct PsMaskTable {
    /// `masks`: the capacity array (`max_masks` entries).
    masks: Vec<PsMask>,
    /// `num_masks`: the number of live masks.
    num_masks: usize,
}

impl PsMaskTable {
    /// The live (allocated) part of the table, as `&ps_hints->dimension[i].masks`
    /// is used by the algorithm module.
    #[inline]
    fn live(&self) -> &[PsMask] {
        let n = self.num_masks.min(self.masks.len());
        &self.masks[..n]
    }

    /// `ps_mask_table_ensure` (`pshrec.c`): makes room for `count` masks,
    /// growing in multiples of eight entries.
    ///
    /// # Errors
    ///
    /// Returns [`TtError::OUT_OF_MEMORY`] when the reservation fails.
    fn ensure(&mut self, count: usize) -> TtResult<()> {
        if count > self.masks.len() {
            let new_max = pad_ceil(count, 8);
            let extra = new_max - self.masks.len();
            self.masks
                .try_reserve(extra)
                .map_err(|_| TtError::OUT_OF_MEMORY)?;
            while self.masks.len() < new_max {
                self.masks.push(PsMask::default());
            }
        }
        Ok(())
    }

    /// `ps_mask_table_alloc` (`pshrec.c`): returns the index of a new
    /// mask appended to the table, reusing the slot that
    /// `ps_mask_table_merge` moved to the end when one is available.
    ///
    /// # Errors
    ///
    /// Returns [`TtError::OUT_OF_MEMORY`] when the table cannot grow.
    fn alloc(&mut self) -> TtResult<usize> {
        let count = self.num_masks.saturating_add(1);
        if count > self.masks.len() {
            self.ensure(count)?;
        }
        let idx = count - 1;
        let mask = self
            .masks
            .get_mut(idx)
            .ok_or(TtError::OUT_OF_MEMORY)?;
        mask.num_bits = 0;
        mask.end_point = 0;
        self.num_masks = count;
        Ok(idx)
    }

    /// `ps_mask_table_last` (`pshrec.c`): returns the last mask, creating
    /// one when the table is empty.
    ///
    /// # Errors
    ///
    /// Returns [`TtError::OUT_OF_MEMORY`] when a new mask is required and
    /// cannot be allocated.
    fn last(&mut self) -> TtResult<usize> {
        if self.num_masks == 0 {
            self.alloc()
        } else {
            Ok(self.num_masks - 1)
        }
    }

    /// `ps_mask_table_set_bits` (`pshrec.c`): loads `bit_count` bits of
    /// `source` (starting at `bit_pos`) into a new last mask.
    ///
    /// # Errors
    ///
    /// Propagates [`TtError::OUT_OF_MEMORY`] and
    /// [`TtError::INVALID_ARGUMENT`] (see [`PsMask::set_bits`]).
    fn set_bits(&mut self, source: &[u8], bit_pos: usize, bit_count: usize) -> TtResult<()> {
        let idx = self.last()?;
        let mask = self
            .masks
            .get_mut(idx)
            .ok_or(TtError::INVALID_HANDLE)?;
        mask.set_bits(source, bit_pos, bit_count)
    }

    /// `ps_mask_table_test_intersect` (`pshrec.c`): `true` when two masks
    /// of the table share at least one bit.
    fn test_intersect(&self, index1: usize, index2: usize) -> bool {
        let (Some(mask1), Some(mask2)) = (self.masks.get(index1), self.masks.get(index2)) else {
            return false;
        };

        let count = mask1.num_bits.min(mask2.num_bits);
        let full_bytes = count >> 3;

        for k in 0..full_bytes {
            let b1 = mask1.bytes.get(k).copied().unwrap_or(0);
            let b2 = mask2.bytes.get(k).copied().unwrap_or(0);
            if b1 & b2 != 0 {
                return true;
            }
        }

        let rest = count & 7;
        if rest == 0 {
            return false;
        }
        let b1 = mask1.bytes.get(full_bytes).copied().unwrap_or(0);
        let b2 = mask2.bytes.get(full_bytes).copied().unwrap_or(0);
        (b1 & b2) & !(0xFF >> rest) != 0
    }

    /// `ps_mask_table_merge` (`pshrec.c`): unions mask `index2` into mask
    /// `index1` and removes `index2`, moving the recycled slot to the end
    /// of the capacity array so that the next `alloc` can reuse it.
    ///
    /// # Errors
    ///
    /// Returns [`TtError::OUT_OF_MEMORY`] when `index1` must grow.
    fn merge(&mut self, mut index1: usize, mut index2: usize) -> TtResult<()> {
        if index1 > index2 {
            core::mem::swap(&mut index1, &mut index2);
        }
        if index1 >= index2 || index2 >= self.num_masks {
            // FreeType logs "ignoring invalid indices" and returns.
            return Ok(());
        }

        let count1 = self.masks[index1].num_bits;
        let count2 = self.masks[index2].num_bits;

        if count2 > 0 {
            if count2 > count1 {
                self.masks[index1].ensure(count2)?;
                // FreeType clears the bits that mask2 is about to
                // contribute here; because `num_bits` is still `count1`
                // the C loop clears nothing, and neither does this one
                for pos in count1..count2 {
                    self.masks[index1].clear_bit(pos);
                }
            }

            let src = core::mem::take(&mut self.masks[index2].bytes);
            let nbytes = count2.saturating_add(7) >> 3;
            for k in 0..nbytes {
                let b1 = self.masks[index1]
                    .bytes
                    .get(k)
                    .copied()
                    .unwrap_or(0);
                let b2 = src.get(k).copied().unwrap_or(0);
                if let Some(byte) = self.masks[index1].bytes.get_mut(k) {
                    *byte = b1 | b2;
                }
            }
            self.masks[index2].bytes = src;
            // `num_bits` of mask1 keeps its old value, exactly as in C
        }

        // remove mask2: rotate the tail left so that the (already
        // cleared) record of index2 lands at the end of the live region
        let end = self.num_masks;
        self.masks[index2..end].rotate_left(1);
        if let Some(last) = self.masks.get_mut(end - 1) {
            last.num_bits = 0;
            last.end_point = 0;
        }
        self.num_masks -= 1;
        Ok(())
    }

    /// `ps_mask_table_merge_all` (`pshrec.c`): merges every pair of
    /// intersecting masks, which turns the counter masks into independent
    /// counter paths.
    ///
    /// # Errors
    ///
    /// Propagates [`TtError::OUT_OF_MEMORY`] from [`PsMaskTable::merge`].
    fn merge_all(&mut self) -> TtResult<()> {
        let mut index1 = self.num_masks as isize - 1;
        while index1 > 0 {
            let mut index2 = index1 - 1;
            while index2 >= 0 {
                let i1 = index1 as usize;
                // after a merge the slot that used to hold `index1` has
                // been recycled with `num_bits == 0`, so it can never
                // intersect anything: the C loop keeps scanning without
                // effect, this guard skips the dead iterations
                if i1 < self.num_masks && self.test_intersect(i1, index2 as usize) {
                    self.merge(index2 as usize, i1)?;
                    break;
                }
                index2 -= 1;
            }
            index1 -= 1;
        }
        Ok(())
    }
}

/// `PS_DimensionRec` (`pshrec.h`): the stems, hint masks and counter masks
/// of one dimension (0 = X / vertical stems, 1 = Y / horizontal stems).
#[derive(Clone, Debug, Default)]
struct PsDimension {
    /// `hints`: the stem table.
    hints: PsHintTable,
    /// `masks`: the hint masks.
    masks: PsMaskTable,
    /// `counters`: the counter masks.
    counters: PsMaskTable,
}

/// `PS_HintsRec` (`pshrec.h`): the glyph hints recorder consumed by the
/// algorithm module ([`ps_hints_apply`]).
///
/// Record a glyph by calling [`PsHints::open`], then the `stem`/`mask`
/// operations, and finally [`PsHints::close`]; the recorded data is
/// consumed by [`ps_hints_apply`]. The [`T1Hints`] and [`T2Hints`]
/// handles wrap a recorder with the operator set of one charstring
/// dialect.
///
/// # Session model
///
/// The recorder is sticky: the first failure is stored (like the C
/// `PS_HintsRec.error` field) and every later recording operation becomes
/// a no-op until the next [`PsHints::open`]. [`PsHints::close`] reports
/// the stored failure.
#[derive(Clone, Debug)]
pub struct PsHints {
    /// `memory`: the allocator handle.
    memory: Memory,
    /// `error`: the sticky session error (`PS_HintsRec.error`).
    error: TtError,
    /// `hint_type`: the dialect of the current session.
    hint_type: Option<PsHintType>,
    /// `dimension[2]`: the X and Y recording tables.
    dimension: [PsDimension; 2],
}

/// `dimension` argument normalisation of `ps_hints_stem`
/// (`pshrec.c`): anything above 1 is mapped to dimension 1.
#[inline]
fn clamp_dimension(dimension: u32) -> usize {
    if dimension == 0 { 0 } else { 1 }
}

impl PsHints {
    /// `ps_hints_init` (`pshrec.c`): creates an empty recorder bound to
    /// `memory`.
    #[inline]
    pub fn new(memory: Memory) -> Self {
        PsHints {
            memory,
            error: TtError::OK,
            hint_type: None,
            dimension: [PsDimension::default(), PsDimension::default()],
        }
    }

    /// `ps_hints_done` (`pshrec.c`): releases the recorder's tables and
    /// unbinds the allocator.
    pub fn done(&mut self) {
        self.dimension[0] = PsDimension::default();
        self.dimension[1] = PsDimension::default();
        self.error = TtError::OK;
        self.hint_type = None;
        self.memory = Memory;
    }

    /// Returns the sticky error of the current recording session
    /// (`PS_HintsRec.error`), [`TtError::OK`] when recording succeeded.
    #[inline]
    pub fn error(&self) -> TtError {
        self.error
    }

    /// Returns the allocator handle the recorder was created with.
    #[inline]
    pub fn memory(&self) -> Memory {
        self.memory
    }

    /// `ps_hints_open` (`pshrec.c`): starts a new recording session for
    /// `hint_type`, rewinding the two dimensions but keeping their
    /// allocated capacity, exactly like `ps_dimension_init`.
    pub fn open(&mut self, hint_type: PsHintType) {
        self.error = TtError::OK;
        self.hint_type = Some(hint_type);

        for dim in self.dimension.iter_mut() {
            dim.hints.hints.clear();
            dim.masks.num_masks = 0;
            dim.counters.num_masks = 0;
        }
    }

    /// `ps_hints_stem` (`pshrec.c`): records `stems` as
    /// `(position, length)` pairs of font units in `dimension`.
    ///
    /// The C signature takes an explicit stem count; here the count is
    /// derived from the slice length (a trailing odd entry is ignored),
    /// which makes the 16-bit pair alignment impossible to get wrong.
    /// Failures are stored in [`PsHints::error`] like in C.
    pub fn stem(&mut self, dimension: u32, stems: &[i32]) {
        if self.error != TtError::OK {
            return;
        }
        let dim = clamp_dimension(dimension);

        let mut i = 0usize;
        while i + 1 < stems.len() {
            let (pos, len) = (stems[i], stems[i + 1]);
            if let Err(error) = ps_dimension_add_t1stem(&mut self.dimension[dim], pos, len) {
                self.error = error;
                return;
            }
            i += 2;
        }
    }

    /// `t1_hints_stem` (`pshrec.c`): records a single Type 1 stem given
    /// as a `(position, length)` pair in 16.16 format.
    pub fn t1_stem(&mut self, dimension: u32, coords: &[Fixed; 2]) {
        let stems = [fixed_to_int(coords[0]), fixed_to_int(coords[1])];
        self.stem(dimension, &stems);
    }

    /// `ps_hints_t1stem3` (`pshrec.c`): records a counter-controlled
    /// triple of stems (`hstem3`/`vstem3`) and the counter mask that
    /// groups them.
    ///
    /// `stems` holds three `(position, length)` pairs in 16.16 format.
    ///
    /// # Errors
    ///
    /// Sets [`TtError::INVALID_ARGUMENT`] when the recorder session is
    /// not a Type 1 session, and [`TtError::OUT_OF_MEMORY`] when a table
    /// cannot grow; both are reported by [`PsHints::close`].
    pub fn t1_stem3(&mut self, dimension: u32, stems: &[Fixed; 6]) {
        if self.error != TtError::OK {
            return;
        }
        if self.hint_type != Some(PsHintType::Type1) {
            self.error = TtError::INVALID_ARGUMENT;
            return;
        }

        let dim = clamp_dimension(dimension);
        let mut idx = [0usize; 3];
        for (n, pair) in stems.chunks(2).enumerate().take(3) {
            let (Some(&coord0), Some(&coord1)) = (pair.first(), pair.last()) else {
                self.error = TtError::INVALID_ARGUMENT;
                return;
            };
            match ps_dimension_add_t1stem(
                &mut self.dimension[dim],
                fixed_to_int(coord0),
                fixed_to_int(coord1),
            ) {
                Ok(index) => idx[n] = index,
                Err(error) => {
                    self.error = error;
                    return;
                }
            }
        }

        if let Err(error) = ps_dimension_add_counter(&mut self.dimension[dim], &idx) {
            self.error = error;
        }
    }

    /// `ps_hints_t1reset` (`pshrec.c`): closes the current hint masks of
    /// both dimensions at `end_point` and opens a new empty one, the Type
    /// 1 `hintmask` equivalent.
    ///
    /// # Errors
    ///
    /// Sets [`TtError::INVALID_ARGUMENT`] for a non-Type 1 session and
    /// [`TtError::OUT_OF_MEMORY`] when a mask cannot be allocated; both
    /// are reported by [`PsHints::close`].
    pub fn t1_reset(&mut self, end_point: u32) {
        if self.error != TtError::OK {
            return;
        }
        if self.hint_type != Some(PsHintType::Type1) {
            self.error = TtError::INVALID_ARGUMENT;
            return;
        }

        for dim in self.dimension.iter_mut() {
            if let Err(error) = ps_dimension_reset_mask(dim, end_point) {
                self.error = error;
                return;
            }
        }
    }

    /// `t2_hints_stems` (`pshrec.c`): records the table of Type 2 stems
    /// of one dimension from the delta-encoded 16.16 `coords` of the
    /// `hstemhm`/`vstemhm` operators.
    ///
    /// # Porting note
    ///
    /// FreeType processes at most 16 stems per batch and — deliberately or
    /// not — restarts at `coords[0]` for every batch, so this port reads
    /// the same coordinates again; the running `y` accumulator is kept
    /// across batches as in C.
    pub fn t2_stems(&mut self, dimension: u32, count: usize, coords: &[Fixed]) {
        if self.error != TtError::OK {
            return;
        }

        let mut y: Fixed = 0;
        let mut total = count.min(coords.len() / 2);
        while total > 0 {
            let mut batch = total;
            if batch > 16 {
                batch = 16;
            }

            let mut stems = [0i32; 32];
            for (n, slot) in stems.iter_mut().enumerate().take(batch * 2) {
                y = y.wrapping_add(coords[n]);
                *slot = fixed_to_int(y);
            }
            for n in (0..batch * 2).step_by(2) {
                stems[n + 1] = stems[n + 1].wrapping_sub(stems[n]);
            }

            self.stem(dimension, &stems[..batch * 2]);
            if self.error != TtError::OK {
                return;
            }
            total -= batch;
        }
    }

    /// `ps_hints_t2mask` (`pshrec.c`): the Type 2 `hintmask` operator:
    /// stores `bit_count` bits of `bytes` as the new hint mask of both
    /// dimensions.
    ///
    /// The first `count2` bits are the horizontal (dimension 1) hints and
    /// the remaining `count1` bits the vertical (dimension 0) ones, as in
    /// C. `bit_count` must equal `count1 + count2`; a mismatch silently
    /// ignores the operator, exactly like FreeType.
    ///
    /// # Errors
    ///
    /// Sets [`TtError::INVALID_ARGUMENT`] when `bytes` is too short and
    /// [`TtError::OUT_OF_MEMORY`] when a mask cannot grow; both are
    /// reported by [`PsHints::close`].
    pub fn t2_mask(&mut self, end_point: u32, bit_count: usize, bytes: &[u8]) {
        if self.error != TtError::OK {
            return;
        }
        let count1 = self.dimension[0].hints.hints.len();
        let count2 = self.dimension[1].hints.hints.len();
        if bit_count != count1 + count2 {
            return;
        }

        if let Err(error) =
            ps_dimension_set_mask_bits(&mut self.dimension[0], bytes, count2, count1, end_point)
        {
            self.error = error;
            return;
        }
        if let Err(error) = ps_dimension_set_mask_bits(&mut self.dimension[1], bytes, 0, count2, end_point) {
            self.error = error;
        }
    }

    /// `ps_hints_t2counter` (`pshrec.c`): the Type 2 `cntrmask`
    /// operator: stores `bit_count` bits of `bytes` as the counter mask
    /// of both dimensions (with `end_point` 0, as in C).
    ///
    /// # Errors
    ///
    /// See [`PsHints::t2_mask`]; failures are reported by
    /// [`PsHints::close`].
    pub fn t2_counter(&mut self, bit_count: usize, bytes: &[u8]) {
        if self.error != TtError::OK {
            return;
        }
        let count1 = self.dimension[0].hints.hints.len();
        let count2 = self.dimension[1].hints.hints.len();
        if bit_count != count1 + count2 {
            return;
        }

        if let Err(error) = ps_dimension_set_mask_bits(&mut self.dimension[0], bytes, 0, count1, 0) {
            self.error = error;
            return;
        }
        if let Err(error) = ps_dimension_set_mask_bits(&mut self.dimension[1], bytes, count1, count2, 0) {
            self.error = error;
        }
    }

    /// `ps_hints_close` (`pshrec.c`): ends the recording session at
    /// `end_point` and merges the counter masks into independent paths.
    ///
    /// # Errors
    ///
    /// Returns the sticky [`PsHints::error`] of the session, or an
    /// [`TtError::OUT_OF_MEMORY`] raised while merging the counters.
    pub fn close(&mut self, end_point: u32) -> TtResult<()> {
        if self.error != TtError::OK {
            return Err(self.error);
        }
        ps_dimension_end(&mut self.dimension[0], end_point)?;
        ps_dimension_end(&mut self.dimension[1], end_point)?;
        Ok(())
    }

    /// The live hint masks of `dimension` (0 = X, 1 = Y), the slice the
    /// algorithm module records its active hint sets from.
    ///
    /// Crate-private because [`PsMask`] is not part of the public API.
    #[inline]
    fn hint_masks(&self, dimension: usize) -> &[PsMask] {
        match self.dimension.get(dimension) {
            Some(dim) => dim.masks.live(),
            None => &[],
        }
    }

    /// The recorded stems of `dimension` (0 = X, 1 = Y), crate-private
    /// for the same reason as [`PsHints::hint_masks`].
    #[inline]
    fn hints(&self, dimension: usize) -> &[PsHint] {
        match self.dimension.get(dimension) {
            Some(dim) => dim.hints.hints.as_slice(),
            None => &[],
        }
    }
}

/// `ps_dimension_end_mask` (`pshrec.c`): stores `end_point` in the last
/// hint mask of `dim`, if any.
#[inline]
fn ps_dimension_end_mask(dim: &mut PsDimension, end_point: u32) {
    let n = dim.masks.num_masks;
    if n > 0
        && let Some(mask) = dim.masks.masks.get_mut(n - 1)
    {
        mask.end_point = end_point as usize;
    }
}

/// `ps_dimension_reset_mask` (`pshrec.c`): ends the current hint mask of
/// `dim` at `end_point` and appends a fresh empty one.
///
/// # Errors
///
/// Returns [`TtError::OUT_OF_MEMORY`] when the new mask cannot be
/// allocated.
fn ps_dimension_reset_mask(dim: &mut PsDimension, end_point: u32) -> TtResult<()> {
    ps_dimension_end_mask(dim, end_point);
    dim.masks.alloc().map(|_| ())
}

/// `ps_dimension_set_mask_bits` (`pshrec.c`): the body of the Type 2
/// `hintmask`/`cntrmask` operators for one dimension: closes the current
/// mask at `end_point`, appends a new one and loads `source_bits` bits of
/// `source` starting at `source_pos` into it.
///
/// # Errors
///
/// See [`PsMaskTable::set_bits`].
fn ps_dimension_set_mask_bits(
    dim: &mut PsDimension,
    source: &[u8],
    source_pos: usize,
    source_bits: usize,
    end_point: u32,
) -> TtResult<()> {
    ps_dimension_reset_mask(dim, end_point)?;
    dim.masks
        .set_bits(source, source_pos, source_bits)
}

/// `ps_dimension_add_t1stem` (`pshrec.c`): records one `(pos, len)` stem
/// of `dim`, deduplicating stems that are already present, and sets its
/// bit in the current hint mask. Returns the stem index.
///
/// Ghost stems (a negative length, per the Type 1 specification) are
/// flagged with [`PS_HINT_FLAG_GHOST`]; a bottom ghost stem (`-21`) also
/// gets [`PS_HINT_FLAG_BOTTOM`] and has its position shifted by its
/// (negative) length.
///
/// # Errors
///
/// Returns [`TtError::OUT_OF_MEMORY`] when a table cannot grow.
fn ps_dimension_add_t1stem(dim: &mut PsDimension, pos: i32, len: i32) -> TtResult<usize> {
    let mut flags = 0u32;
    let (mut pos, mut len) = (pos, len);

    if len < 0 {
        flags |= PS_HINT_FLAG_GHOST;
        if len == -21 {
            flags |= PS_HINT_FLAG_BOTTOM;
            pos = pos.wrapping_add(len);
        }
        len = 0;
    }

    let existing = dim
        .hints
        .hints
        .iter()
        .position(|hint| hint.pos == pos && hint.len == len);
    let idx = match existing {
        Some(index) => index,
        None => {
            let index = dim.hints.alloc()?;
            let hint = dim
                .hints
                .hints
                .get_mut(index)
                .ok_or(TtError::INVALID_HANDLE)?;
            hint.pos = pos;
            hint.len = len;
            hint.flags = flags;
            index
        }
    };

    let mask_idx = dim.masks.last()?;
    let mask = dim
        .masks
        .masks
        .get_mut(mask_idx)
        .ok_or(TtError::INVALID_HANDLE)?;
    mask.set_bit(idx)?;
    Ok(idx)
}

/// `ps_dimension_add_counter` (`pshrec.c`): adds the counter mask that
/// groups the three stems of an `hstem3`/`vstem3`, reusing an existing
/// counter that already references one of them.
///
/// # Errors
///
/// Returns [`TtError::OUT_OF_MEMORY`] when a counter mask cannot be
/// allocated.
fn ps_dimension_add_counter(dim: &mut PsDimension, hints: &[usize; 3]) -> TtResult<()> {
    let mut found = None;
    let live = dim.counters.num_masks;
    for (index, counter) in dim.counters.masks.iter().take(live).enumerate() {
        if counter.test_bit(hints[0]) || counter.test_bit(hints[1]) || counter.test_bit(hints[2]) {
            found = Some(index);
            break;
        }
    }

    let counter_idx = match found {
        Some(index) => index,
        None => dim.counters.alloc()?,
    };

    for &hint in hints {
        let counter = dim
            .counters
            .masks
            .get_mut(counter_idx)
            .ok_or(TtError::INVALID_HANDLE)?;
        counter.set_bit(hint)?;
    }
    Ok(())
}

/// `ps_dimension_end` (`pshrec.c`): ends the hint mask table of `dim` at
/// `end_point` and merges its counter masks into independent paths.
///
/// # Errors
///
/// Propagates [`TtError::OUT_OF_MEMORY`] from the counter merge.
fn ps_dimension_end(dim: &mut PsDimension, end_point: u32) -> TtResult<()> {
    ps_dimension_end_mask(dim, end_point);
    dim.counters.merge_all()
}

/// A handle to a Type 1 hint recorder (`T1_Hints` in `pshints.h`).
///
/// Recording follows the scheme of the C documentation: [`T1Hints::open`],
/// then [`T1Hints::stem`] / [`T1Hints::stem3`] / [`T1Hints::reset`] for
/// every hint of the charstring, and finally [`T1Hints::close`]. The
/// recorded hints are applied to an outline with [`T1Hints::apply`].
///
/// The C `T1_Hints` value is an opaque pointer to the module's shared
/// `PS_HintsRec`; here every handle owns its own [`PsHints`] recorder,
/// which is equivalent because FreeType uses one session at a time and
/// the function records of [`T1HintsFuncs`] take the recorder explicitly
/// as their first argument.
#[derive(Clone, Debug)]
pub struct T1Hints {
    /// The shared recorder body (`PS_HintsRec`).
    recorder: PsHints,
}

impl T1Hints {
    /// Creates a Type 1 recorder bound to `memory`.
    #[inline]
    pub fn new(memory: Memory) -> Self {
        T1Hints {
            recorder: PsHints::new(memory),
        }
    }

    /// The wrapped recorder, so that a [`T1HintsFuncs`] function record
    /// can be invoked as `(funcs.stem)(&mut hints.recorder(), ...)`,
    /// mirroring C's `funcs->stem(hints, ...)`.
    #[inline]
    pub fn recorder(&mut self) -> &mut PsHints {
        &mut self.recorder
    }

    /// `t1_hints_open` (`pshrec.c`): starts a Type 1 session.
    #[inline]
    pub fn open(&mut self) {
        self.recorder.open(PsHintType::Type1);
    }

    /// `t1_hints_stem` (`pshrec.c`): records one Type 1 stem given as a
    /// 16.16 `(position, length)` pair.
    #[inline]
    pub fn stem(&mut self, dimension: u32, coords: &[Fixed; 2]) {
        self.recorder.t1_stem(dimension, coords);
    }

    /// `ps_hints_t1stem3` (`pshrec.c`): records three counter-controlled
    /// stems given by three 16.16 `(position, length)` pairs.
    #[inline]
    pub fn stem3(&mut self, dimension: u32, stems: &[Fixed; 6]) {
        self.recorder.t1_stem3(dimension, stems);
    }

    /// `ps_hints_t1reset` (`pshrec.c`): resets the stem masks at
    /// `end_point`.
    #[inline]
    pub fn reset(&mut self, end_point: u32) {
        self.recorder.t1_reset(end_point);
    }

    /// `ps_hints_close` (`pshrec.c`): ends the session at `end_point`.
    ///
    /// # Errors
    ///
    /// Reports every failure stored during the session.
    #[inline]
    pub fn close(&mut self, end_point: u32) -> TtResult<()> {
        self.recorder.close(end_point)
    }

    /// `ps_hints_apply` (`pshalgo.c`): grid-fits `outline` with the
    /// recorded hints.
    ///
    /// # Errors
    ///
    /// See [`ps_hints_apply`].
    #[inline]
    pub fn apply(
        &self,
        outline: &mut Outline,
        globals: &mut PshGlobals,
        hint_mode: RenderMode,
    ) -> TtResult<()> {
        ps_hints_apply(&self.recorder, outline, globals, hint_mode)
    }
}

/// A handle to a Type 2 hint recorder (`T2_Hints` in `pshints.h`).
///
/// Recording follows the scheme of the C documentation: [`T2Hints::open`],
/// then [`T2Hints::stems`] / [`T2Hints::hintmask`] /
/// [`T2Hints::counter`], and finally [`T2Hints::close`]. See
/// [`T1Hints`] for the ownership note about the C opaque handle.
#[derive(Clone, Debug)]
pub struct T2Hints {
    /// The shared recorder body (`PS_HintsRec`).
    recorder: PsHints,
}

impl T2Hints {
    /// Creates a Type 2 recorder bound to `memory`.
    #[inline]
    pub fn new(memory: Memory) -> Self {
        T2Hints {
            recorder: PsHints::new(memory),
        }
    }

    /// The wrapped recorder, so that a [`T2HintsFuncs`] function record
    /// can be invoked as `(funcs.open)(&mut hints.recorder())`, mirroring
    /// C's `funcs->open(hints)`.
    #[inline]
    pub fn recorder(&mut self) -> &mut PsHints {
        &mut self.recorder
    }

    /// `t2_hints_open` (`pshrec.c`): starts a Type 2 session.
    #[inline]
    pub fn open(&mut self) {
        self.recorder.open(PsHintType::Type2);
    }

    /// `t2_hints_stems` (`pshrec.c`): records the delta-encoded stem
    /// table of one dimension.
    #[inline]
    pub fn stems(&mut self, dimension: u32, count: usize, coords: &[Fixed]) {
        self.recorder.t2_stems(dimension, count, coords);
    }

    /// `ps_hints_t2mask` (`pshrec.c`): records a `hintmask`.
    #[inline]
    pub fn hintmask(&mut self, end_point: u32, bit_count: usize, bytes: &[u8]) {
        self.recorder.t2_mask(end_point, bit_count, bytes);
    }

    /// `ps_hints_t2counter` (`pshrec.c`): records a `cntrmask`.
    #[inline]
    pub fn counter(&mut self, bit_count: usize, bytes: &[u8]) {
        self.recorder.t2_counter(bit_count, bytes);
    }

    /// `ps_hints_close` (`pshrec.c`): ends the session at `end_point`.
    ///
    /// # Errors
    ///
    /// Reports every failure stored during the session.
    #[inline]
    pub fn close(&mut self, end_point: u32) -> TtResult<()> {
        self.recorder.close(end_point)
    }

    /// `ps_hints_apply` (`pshalgo.c`): grid-fits `outline` with the
    /// recorded hints.
    ///
    /// # Errors
    ///
    /// See [`ps_hints_apply`].
    #[inline]
    pub fn apply(
        &self,
        outline: &mut Outline,
        globals: &mut PshGlobals,
        hint_mode: RenderMode,
    ) -> TtResult<()> {
        ps_hints_apply(&self.recorder, outline, globals, hint_mode)
    }
}

/// `PSH_HintRec` (`pshalgo.h`): one stem hint as seen by the algorithm.
///
/// `parent` replaces the C `PSH_Hint` pointer (it is an index into the
/// same table, `NULL` becomes `None`); the write-only C field `order` is
/// omitted as documented in the crate header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PshHint {
    /// `org_pos`: original font-unit position of the stem's lower edge.
    org_pos: i32,
    /// `org_len`: original font-unit length of the stem.
    org_len: i32,
    /// `cur_pos`: fitted 26.6 position, valid once [`PshHint::is_fitted`].
    cur_pos: Pos,
    /// `cur_len`: fitted 26.6 length, valid once [`PshHint::is_fitted`].
    cur_len: Pos,
    /// `flags`: `PS_HINT_FLAG_GHOST`/`BOTTOM` plus `PSH_HINT_ACTIVE` and
    /// `PSH_HINT_FITTED`.
    flags: u32,
    /// `parent`: index of the already-recorded hint this one overlaps.
    parent: Option<usize>,
}

impl PshHint {
    /// `psh_hint_is_active` (`pshalgo.h`).
    #[inline]
    fn is_active(&self) -> bool {
        self.flags & PSH_HINT_ACTIVE != 0
    }

    /// `psh_hint_is_fitted` (`pshalgo.h`).
    #[inline]
    fn is_fitted(&self) -> bool {
        self.flags & PSH_HINT_FITTED != 0
    }

    /// `psh_hint_activate` (`pshalgo.h`).
    #[inline]
    fn activate(&mut self) {
        self.flags |= PSH_HINT_ACTIVE;
    }

    /// `psh_hint_deactivate` (`pshalgo.h`).
    #[inline]
    fn deactivate(&mut self) {
        self.flags &= !PSH_HINT_ACTIVE;
    }

    /// `psh_hint_set_fitted` (`pshalgo.h`).
    #[inline]
    fn set_fitted(&mut self) {
        self.flags |= PSH_HINT_FITTED;
    }
}

/// `psh_hint_overlap` (`pshalgo.c`): `true` when two stem hints overlap.
///
/// The additions are performed in `i64` so that hostile font-unit values
/// cannot wrap the `FT_Int` arithmetic of the C code.
#[inline]
fn psh_hint_overlap(hint1: &PshHint, hint2: &PshHint) -> bool {
    let (pos1, len1) = (i64::from(hint1.org_pos), i64::from(hint1.org_len));
    let (pos2, len2) = (i64::from(hint2.org_pos), i64::from(hint2.org_len));
    pos1 + len1 >= pos2 && pos2 + len2 >= pos1
}

/// `PSH_Hint_TableRec` (`pshalgo.h`): the per-dimension hint table of a
/// glyph being fitted.
///
/// `sort` holds the two index arrays of the C record in one allocation:
/// `sort[..max_hints]` is the active set (`table->sort`) and
/// `sort[max_hints..]` the global order built while recording
/// (`table->sort_global`). `num_hints` plays the role of the C
/// `num_hints` field, which counts the global order during
/// [`psh_hint_table_init`] and the active set afterwards — exactly as in
/// FreeType, where [`psh_hint_table_activate_mask`] overwrites it.
#[derive(Clone, Debug)]
struct PshHintTable<'a> {
    /// `hints`: every recorded hint of the dimension, indexed as in the
    /// recorder's table.
    hints: Vec<PshHint>,
    /// `sort` + `sort_global`: two `max_hints`-long index arrays.
    sort: Vec<usize>,
    /// `num_hints`: see the type-level documentation.
    num_hints: usize,
    /// `hint_masks`: the recorder's hint masks of this dimension.
    hint_masks: &'a [PsMask],
}

impl<'a> PshHintTable<'a> {
    /// Creates an empty table bound to the hint masks of `dimension`.
    #[inline]
    fn empty(hint_masks: &'a [PsMask]) -> Self {
        PshHintTable {
            hints: Vec::new(),
            sort: Vec::new(),
            num_hints: 0,
            hint_masks,
        }
    }

    /// `max_hints` (`pshalgo.h`): the number of hints of the table.
    #[inline]
    fn max_hints(&self) -> usize {
        self.hints.len()
    }
}

/// `psh_hint_table_deactivate` (`pshalgo.c`): clears the
/// `PSH_HINT_ACTIVE` flag of every hint of `table`.
fn psh_hint_table_deactivate(table: &mut PshHintTable<'_>) {
    for hint in table.hints.iter_mut() {
        hint.deactivate();
    }
}

/// `psh_hint_table_record` (`pshalgo.c`): activates hint `idx` and appends
/// it to the global order, remembering the first already-recorded hint it
/// overlaps as its parent.
fn psh_hint_table_record(table: &mut PshHintTable<'_>, idx: usize) {
    let max = table.max_hints();
    if idx >= max {
        return;
    }
    // ignore active hints
    if table.hints[idx].is_active() {
        return;
    }
    table.hints[idx].activate();

    // scan the current active hint set for an overlapping parent
    let mut parent = None;
    for n in 0..table.num_hints {
        let Some(&other) = table.sort.get(max + n) else {
            break;
        };
        if other >= max {
            continue;
        }
        if psh_hint_overlap(&table.hints[idx], &table.hints[other]) {
            parent = Some(other);
            break;
        }
    }
    table.hints[idx].parent = parent;

    if table.num_hints < max {
        let slot = max + table.num_hints;
        if slot < table.sort.len() {
            table.sort[slot] = idx;
            table.num_hints += 1;
        }
    }
}

/// `psh_hint_table_record_mask` (`pshalgo.c`): activates every hint whose
/// bit is set in `hint_mask`.
///
/// # Performance
///
/// One bit test per mask bit: `O(num_bits)` with no allocation.
fn psh_hint_table_record_mask(table: &mut PshHintTable<'_>, hint_mask: &PsMask) {
    for idx in 0..hint_mask.num_bits {
        if hint_mask.test_bit(idx) {
            psh_hint_table_record(table, idx);
        }
    }
}

/// `psh_hint_table_init` (`pshalgo.c`): copies the recorder's hints of one
/// dimension into `table` and builds the global hint order from the hint
/// masks (plus a linear pass for hints no mask covered).
///
/// # Errors
///
/// Returns [`TtError::OUT_OF_MEMORY`] when a table cannot grow, the same
/// error the C function returns from `FT_NEW_ARRAY`.
fn psh_hint_table_init<'a>(
    table: &mut PshHintTable<'a>,
    hints: &[PsHint],
    hint_masks: &'a [PsMask],
) -> TtResult<()> {
    let count = hints.len();

    table.hints = Vec::new();
    table
        .hints
        .try_reserve(count)
        .map_err(|_| TtError::OUT_OF_MEMORY)?;
    for hint in hints {
        table.hints.push(PshHint {
            org_pos: hint.pos,
            org_len: hint.len,
            cur_pos: 0,
            cur_len: 0,
            flags: hint.flags,
            parent: None,
        });
    }

    table.sort = Vec::new();
    let slots = count.saturating_mul(2);
    table
        .sort
        .try_reserve(slots)
        .map_err(|_| TtError::OUT_OF_MEMORY)?;
    table.sort.resize(slots, 0);
    table.num_hints = 0;
    table.hint_masks = hint_masks;

    // determine the initial `parent` stems from the hint masks
    for mask in hint_masks {
        psh_hint_table_record_mask(table, mask);
    }

    // finally, do a linear parse in case some hints were left alone
    if table.num_hints != table.max_hints() {
        for idx in 0..count {
            psh_hint_table_record(table, idx);
        }
    }

    Ok(())
}

/// `psh_hint_table_activate_mask` (`pshalgo.c`): makes the hints of
/// `hint_mask` the active set of `table`, sorted by their original
/// position.
fn psh_hint_table_activate_mask(table: &mut PshHintTable<'_>, hint_mask: &PsMask) {
    let max = table.max_hints();

    psh_hint_table_deactivate(table);

    let mut count = 0usize;
    for idx in 0..hint_mask.num_bits {
        if hint_mask.test_bit(idx) && idx < max {
            // the C code reads `table->hints[idx]` without bounds check;
            // this port ignores bits that address a non-existent hint
            if !table.hints[idx].is_active() {
                table.hints[idx].activate();
                if count < max {
                    table.sort[count] = idx;
                    count += 1;
                }
            }
        }
    }
    table.num_hints = count;

    // a simple insertion sort: the hints are guaranteed not to overlap,
    // so `org_pos` can be compared directly (ties are reversed, exactly
    // like FreeType's loop)
    for i1 in 1..count {
        let hint1 = table.sort[i1];
        let pos1 = table.hints[hint1].org_pos;
        let mut i2 = i1 as isize - 1;
        while i2 >= 0 {
            let slot = i2 as usize;
            let hint2 = table.sort[slot];
            if table.hints[hint2].org_pos < pos1 {
                break;
            }
            table.sort[slot + 1] = hint2;
            table.sort[slot] = hint1;
            i2 -= 1;
        }
    }
}

/// `psh_dimension_quantize_len` (`pshalgo.c`): rounds a fitted stem length
/// to a value that renders as crisply as possible.
///
/// # Performance
///
/// Constant time; only executed for stems wider than one pixel.
fn psh_dimension_quantize_len(globals: &PshGlobals, dimension: usize, len: Pos, do_snapping: bool) -> Pos {
    let mut len = len;

    if len <= 64 {
        len = 64;
    } else {
        // `dimension->stdw.widths[0].cur`, the scaled standard width
        let stdw = match globals.dimension.get(dimension) {
            Some(dim) => dim.stdw.widths[0].cur,
            None => 0,
        };

        let delta = len.wrapping_sub(stdw).wrapping_abs();
        if delta < 40 {
            len = stdw;
            if len < 48 {
                len = 48;
            }
        }

        if len < 3 * 64 {
            let frac = len & 63;
            len &= !63;

            if frac < 10 {
                len += frac;
            } else if frac < 32 {
                len += 10;
            } else if frac < 54 {
                len += 54;
            } else {
                len += frac;
            }
        } else {
            len = ft_pix_round(len);
        }
    }

    if do_snapping {
        len = ft_pix_round(len);
    }

    len
}

/// `psh_hint_snap_stem_side_delta` (`pshalgo.c`): the smallest shift that
/// snaps one of the two stem edges onto the pixel grid.
#[inline]
fn psh_hint_snap_stem_side_delta(pos: Pos, len: Pos) -> Pos {
    let delta1 = ft_pix_round(pos) - pos;
    let delta2 = ft_pix_round(pos.wrapping_add(len)) - pos - len;

    if abs_pos(delta1) <= abs_pos(delta2) {
        delta1
    } else {
        delta2
    }
}

/// How the point coordinates of a glyph are snapped to the pixel grid,
/// derived from the render mode (`PSH_GlyphRec.do_*_snapping`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Snapping {
    /// No axis is snapped (`FT_RENDER_MODE_NORMAL` / `LIGHT`).
    Neither,
    /// Only the X axis is snapped (`FT_RENDER_MODE_LCD`).
    Horizontal,
    /// Only the Y axis is snapped (`FT_RENDER_MODE_LCD_V`).
    Vertical,
    /// Both axes are snapped (`FT_RENDER_MODE_MONO`).
    Both,
}

impl Snapping {
    /// Builds the snapping policy of `hint_mode`, mirroring the
    /// `FT_RENDER_MODE_*` tests of `ps_hints_apply`.
    fn from_render_mode(hint_mode: RenderMode) -> Self {
        let horz = matches!(hint_mode, RenderMode::Mono | RenderMode::Lcd);
        let vert = matches!(hint_mode, RenderMode::Mono | RenderMode::LcdV);

        match (horz, vert) {
            (true, true) => Snapping::Both,
            (true, false) => Snapping::Horizontal,
            (false, true) => Snapping::Vertical,
            (false, false) => Snapping::Neither,
        }
    }

    /// `PSH_GlyphRec.do_horz_snapping`.
    #[inline]
    fn horizontal(self) -> bool {
        matches!(self, Snapping::Horizontal | Snapping::Both)
    }

    /// `PSH_GlyphRec.do_vert_snapping`.
    #[inline]
    fn vertical(self) -> bool {
        matches!(self, Snapping::Vertical | Snapping::Both)
    }
}

/// The grid-fitting policy of one `ps_hints_apply` run
/// (`PSH_GlyphRec.do_horz_hints`, `do_vert_hints`, `do_stem_adjust`).
///
/// `do_horz_hints` and `do_vert_hints` are hard-coded to `1` by
/// `ps_hints_apply` in FreeType 2.6, so they are not part of this record;
/// only the two flags that actually depend on the render mode are kept
/// (one boolean plus one enum, per the crate's struct rules).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FitOptions {
    /// `do_horz_snapping` / `do_vert_snapping`.
    snapping: Snapping,
    /// `do_stem_adjust`: `hint_mode != FT_RENDER_MODE_LIGHT`.
    stem_adjust: bool,
}

impl FitOptions {
    /// Builds the fitting options of `hint_mode`.
    fn new(hint_mode: RenderMode) -> Self {
        FitOptions {
            snapping: Snapping::from_render_mode(hint_mode),
            stem_adjust: hint_mode != RenderMode::Light,
        }
    }
}

/// `psh_hint_align` (`pshalgo.c`): grid-fits a single stem hint.
///
/// Hints are aligned in `table->hints` order; a hint that overlaps an
/// already fitted parent is positioned relative to it, and (unless light
/// hinting is selected) its width is snapped to a standard width.
///
/// # Performance
///
/// Recursion depth is bounded by the number of recorded stems, because a
/// parent is always a hint that was recorded earlier.
fn psh_hint_align(
    hints: &mut [PshHint],
    index: usize,
    globals: &PshGlobals,
    dimension: usize,
    options: &FitOptions,
) {
    let Some(hint) = hints.get(index) else {
        return;
    };
    if hint.is_fitted() {
        return;
    }

    let org_pos = i64::from(hint.org_pos);
    let org_len = i64::from(hint.org_len);
    let parent = hint.parent;

    let (scale, delta) = match globals.dimension.get(dimension) {
        Some(dim) => (dim.scale_mult, dim.scale_delta),
        None => return,
    };

    let mut pos = mul_fix(org_pos, scale) + delta;
    let mut len = mul_fix(org_len, scale);
    let fit_len = len;

    // `do_snapping = (dimension == 0 && glyph->do_horz_snapping) ||
    //                 (dimension == 1 && glyph->do_vert_snapping)`
    let do_snapping =
        (dimension == 0 && options.snapping.horizontal()) || (dimension == 1 && options.snapping.vertical());

    hints[index].cur_len = fit_len;

    // check blue zones for horizontal stems
    let mut align = PshAlignment::default();
    if dimension == 1 {
        psh_blues_snap_stem(
            &globals.blues,
            (org_pos + org_len) as i32,
            org_pos as i32,
            &mut align,
        );
    }

    match align.align {
        // the top of the stem is aligned against a blue zone
        PSH_BLUE_ALIGN_TOP => {
            hints[index].cur_pos = align.align_top - fit_len;
        }

        // the bottom of the stem is aligned against a blue zone
        PSH_BLUE_ALIGN_BOT => {
            hints[index].cur_pos = align.align_bot;
        }

        // both edges of the stem are aligned against blue zones
        PSH_BLUE_ALIGN_BOTH => {
            hints[index].cur_pos = align.align_bot;
            hints[index].cur_len = align.align_top - align.align_bot;
        }

        _ => {
            if let Some(parent) = parent {
                if let Some(par) = hints.get(parent) {
                    // ensure that the parent is already fitted
                    if !par.is_fitted() {
                        psh_hint_align(hints, parent, globals, dimension, options);
                    }
                }

                // keep the original relation between the hints: use the
                // scaled distance between their centers
                if let Some(par) = hints.get(parent) {
                    let par_org_center = i64::from(par.org_pos) + (i64::from(par.org_len) >> 1);
                    let par_cur_center = par.cur_pos + (par.cur_len >> 1);
                    let cur_org_center = org_pos + (org_len >> 1);

                    let cur_delta = mul_fix(cur_org_center - par_org_center, scale);
                    pos = par_cur_center + cur_delta - (len >> 1);
                }
            }

            hints[index].cur_pos = pos;
            hints[index].cur_len = fit_len;

            // stem adjustment snaps stem widths to standard ones to
            // prevent unpleasant rounding artefacts
            if options.stem_adjust {
                if len <= 64 {
                    if len >= 32 {
                        // widen the stem to one pixel, centred on the
                        // nearest pixel centre
                        pos = ft_pix_floor(pos + (len >> 1));
                        len = 64;
                    } else if len > 0 {
                        // a very small stem: align it to the pixel grid
                        // with the minimum displacement
                        let left_nearest = ft_pix_round(pos);
                        let right_nearest = ft_pix_round(pos.wrapping_add(len));
                        let left_disp = abs_pos(left_nearest - pos);
                        let right_disp = abs_pos(right_nearest - (pos + len));

                        pos = if left_disp <= right_disp {
                            left_nearest
                        } else {
                            right_nearest
                        };
                    } else {
                        // a ghost stem: simply round it
                        pos = ft_pix_round(pos);
                    }
                } else {
                    len = psh_dimension_quantize_len(globals, dimension, len, false);
                }
            }

            // now that we have a good hinted stem width, position the
            // stem on a pixel grid integer coordinate
            hints[index].cur_pos = pos + psh_hint_snap_stem_side_delta(pos, len);
            hints[index].cur_len = len;
        }
    }

    if do_snapping {
        pos = hints[index].cur_pos;
        len = hints[index].cur_len;

        if len < 64 {
            len = 64;
        } else {
            len = ft_pix_round(len);
        }

        match align.align {
            PSH_BLUE_ALIGN_TOP => {
                hints[index].cur_pos = align.align_top - len;
                hints[index].cur_len = len;
            }

            PSH_BLUE_ALIGN_BOT => {
                hints[index].cur_len = len;
            }

            // don't touch
            PSH_BLUE_ALIGN_BOTH => {}

            _ => {
                if len & 64 != 0 {
                    pos = ft_pix_floor(pos + (len >> 1)) + 32;
                } else {
                    pos = ft_pix_round(pos + (len >> 1));
                }

                hints[index].cur_pos = pos - (len >> 1);
                hints[index].cur_len = len;
            }
        }
    }

    hints[index].set_fitted();
}

/// `psh_hint_table_align_hints` (`pshalgo.c`): grid-fits every hint of
/// `table`.
///
/// # Performance
///
/// `O(max_hints)`; each hint is fitted exactly once because
/// [`psh_hint_align`] returns immediately for a fitted hint.
fn psh_hint_table_align_hints(
    table: &mut PshHintTable<'_>,
    globals: &PshGlobals,
    dimension: usize,
    options: &FitOptions,
) {
    for index in 0..table.hints.len() {
        psh_hint_align(&mut table.hints, index, globals, dimension, options);
    }
}

/// `PSH_PointRec` (`pshalgo.h`): one outline point as seen by the hinter.
///
/// The C record's doubly-linked `prev`/`next` pointers and the
/// write-only `contour` back-pointer become indices: `prev`/`next` are
/// positions in [`PshGlyph::points`] and `contour` is omitted because
/// FreeType never reads it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PshPoint {
    /// `prev`: index of the previous point of the contour.
    prev: usize,
    /// `next`: index of the next point of the contour.
    next: usize,
    /// `flags`: computed once for both dimensions (`PSH_POINT_OFF`,
    /// `PSH_POINT_SMOOTH`, `PSH_POINT_INFLEX`).
    flags: u32,
    /// `flags2`: recomputed for each dimension (`PSH_POINT_STRONG`,
    /// `PSH_POINT_FITTED`, `PSH_POINT_EXTREMUM`, `PSH_POINT_POSITIVE`,
    /// `PSH_POINT_NEGATIVE`, `PSH_POINT_EDGE_MIN`, `PSH_POINT_EDGE_MAX`).
    flags2: u32,
    /// `dir_in`: [`psh_compute_dir`] of the incoming segment.
    dir_in: i32,
    /// `dir_out`: [`psh_compute_dir`] of the outgoing segment.
    dir_out: i32,
    /// `hint`: index of the hint of the current dimension this point is
    /// attached to (`NULL` becomes `None`); cleared per dimension by
    /// [`psh_glyph_load_points`].
    hint: Option<usize>,
    /// `org_u`: the original coordinate along the fitted axis.
    org_u: Pos,
    /// `org_v`: the original coordinate across the fitted axis.
    org_v: Pos,
    /// `cur_u`: the fitted coordinate along the fitted axis.
    cur_u: Pos,
}

impl PshPoint {
    /// `psh_point_is_smooth` (`pshalgo.h`).
    #[inline]
    fn is_smooth(&self) -> bool {
        self.flags & PSH_POINT_SMOOTH != 0
    }

    /// `psh_point_is_inflex` (`pshalgo.h`).
    #[inline]
    fn is_inflex(&self) -> bool {
        self.flags & PSH_POINT_INFLEX != 0
    }

    /// `psh_point_set_inflex` (`pshalgo.h`).
    #[inline]
    fn set_inflex(&mut self) {
        self.flags |= PSH_POINT_INFLEX;
    }

    /// `psh_point_is_strong` (`pshalgo.h`).
    #[inline]
    fn is_strong(&self) -> bool {
        self.flags2 & PSH_POINT_STRONG != 0
    }

    /// `psh_point_is_fitted` (`pshalgo.h`).
    #[inline]
    fn is_fitted(&self) -> bool {
        self.flags2 & PSH_POINT_FITTED != 0
    }

    /// `psh_point_is_extremum` (`pshalgo.h`).
    #[inline]
    fn is_extremum(&self) -> bool {
        self.flags2 & PSH_POINT_EXTREMUM != 0
    }

    /// `psh_point_is_edge_min` (`pshalgo.h`).
    #[inline]
    fn is_edge_min(&self) -> bool {
        self.flags2 & PSH_POINT_EDGE_MIN != 0
    }

    /// `psh_point_is_edge_max` (`pshalgo.h`).
    #[inline]
    fn is_edge_max(&self) -> bool {
        self.flags2 & PSH_POINT_EDGE_MAX != 0
    }

    /// `psh_point_set_strong` (`pshalgo.h`).
    #[inline]
    fn set_strong(&mut self) {
        self.flags2 |= PSH_POINT_STRONG;
    }

    /// `psh_point_set_fitted` (`pshalgo.h`).
    #[inline]
    fn set_fitted(&mut self) {
        self.flags2 |= PSH_POINT_FITTED;
    }

    /// `psh_point_set_extremum` (`pshalgo.h`).
    #[inline]
    fn set_extremum(&mut self) {
        self.flags2 |= PSH_POINT_EXTREMUM;
    }

    /// `psh_point_set_positive` (`pshalgo.h`).
    #[inline]
    fn set_positive(&mut self) {
        self.flags2 |= PSH_POINT_POSITIVE;
    }

    /// `psh_point_set_negative` (`pshalgo.h`).
    #[inline]
    fn set_negative(&mut self) {
        self.flags2 |= PSH_POINT_NEGATIVE;
    }

    /// `psh_point_set_edge_min` (`pshalgo.h`).
    #[inline]
    fn set_edge_min(&mut self) {
        self.flags2 |= PSH_POINT_EDGE_MIN;
    }

    /// `psh_point_set_edge_max` (`pshalgo.h`).
    #[inline]
    fn set_edge_max(&mut self) {
        self.flags2 |= PSH_POINT_EDGE_MAX;
    }

    /// `psh_point_clear_smooth` (`pshalgo.c`): clears
    /// `PSH_POINT_SMOOTH`, done for smooth extrema that take part in the
    /// interpolation.
    #[inline]
    fn clear_smooth(&mut self) {
        self.flags &= !PSH_POINT_SMOOTH;
    }
}

/// `PSH_ContourRec` (`pshalgo.h`): a run of points of one glyph contour.
///
/// `start` replaces the C pointer into the point array.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PshContour {
    /// `start`: index of the contour's first point.
    start: usize,
    /// `count`: the number of points of the contour.
    count: usize,
}

/// `PSH_GlyphRec` (`pshalgo.h`): the working state of one
/// [`ps_hints_apply`] run.
///
/// The C record also stores `memory`, `outline` and `globals` pointers;
/// this port passes them as arguments instead, which keeps the glyph
/// free of self-referential pointers (the same trick the crate header
/// documents for the linked lists). The write-only C fields `vertical`,
/// `major_dir` and `minor_dir` are omitted.
#[derive(Debug)]
struct PshGlyph<'a> {
    /// `points`: every outline point.
    points: Vec<PshPoint>,
    /// `contours`: every outline contour.
    contours: Vec<PshContour>,
    /// `hint_tables[2]`: the hint tables of the X and Y dimensions.
    hint_tables: [PshHintTable<'a>; 2],
    /// `do_horz_snapping` / `do_vert_snapping` / `do_stem_adjust`.
    options: FitOptions,
}

/// `ft_corner_orientation` (`ftcalc.c`): returns the orientation of the
/// corner spanned by `(in_x, in_y)` and `(out_x, out_y)`.
///
/// # Porting note
///
/// The core crate also exports a `corner_orientation` helper, but it only
/// computes the sign of the cross product. FreeType 2.6's function treats
/// the axis-aligned cases separately and returns the raw coordinate (the
/// sign is all its callers look at), which yields the opposite sign of
/// the cross product when `out_x` or `out_y` is zero. This port keeps the
/// C branches verbatim so that inflection detection matches FreeType bit
/// for bit.
///
/// The result is truncated to `i32` like the C `FT_Int` return type;
/// every caller only tests it for zero or for its sign.
#[inline]
fn psh_corner_orientation(in_x: Pos, in_y: Pos, out_x: Pos, out_y: Pos) -> i32 {
    if in_y == 0 {
        let result = out_y as i32;
        if in_x >= 0 { result } else { result.wrapping_neg() }
    } else if in_x == 0 {
        let result = out_x as i32;
        if in_y >= 0 { result.wrapping_neg() } else { result }
    } else if out_y == 0 {
        let result = in_y as i32;
        if out_x >= 0 { result } else { result.wrapping_neg() }
    } else if out_x == 0 {
        let result = in_x as i32;
        if out_y >= 0 { result.wrapping_neg() } else { result }
    } else {
        let delta = in_x
            .wrapping_mul(out_y)
            .wrapping_sub(in_y.wrapping_mul(out_x));
        if delta == 0 {
            0
        } else if delta < 0 {
            -1
        } else {
            1
        }
    }
}

/// `psh_compute_dir` (`pshalgo.c`): classifies the segment `(dx, dy)` as
/// near-horizontal, near-vertical or direction-less.
///
/// # Porting note
///
/// The C code multiplies `FT_Pos` values without overflow checks; the
/// `wrapping_*` operations here keep the result identical for every
/// realistic outline coordinate while staying panic-free.
fn psh_compute_dir(dx: Pos, dy: Pos) -> i32 {
    let ax = abs_pos(dx);
    let ay = abs_pos(dy);

    if ay.wrapping_mul(12) < ax {
        // `|dy| <<< |dx|` means a near-horizontal segment
        if dx >= 0 { PSH_DIR_RIGHT } else { PSH_DIR_LEFT }
    } else if ax.wrapping_mul(12) < ay {
        // `|dx| <<< |dy|` means a near-vertical segment
        if dy >= 0 { PSH_DIR_UP } else { PSH_DIR_DOWN }
    } else {
        PSH_DIR_NONE
    }
}

/// `psh_glyph_load_points` (`pshalgo.c`): copies the outline coordinates
/// into the hinter glyph, mapping them onto the `org_u`/`org_v` axes of
/// `dimension` (X for 0, Y for 1).
///
/// The per-dimension `flags2` and the attached hint are cleared here,
/// exactly as in FreeType; `flags` (which holds the dimension-independent
/// `OFF`/`SMOOTH`/`INFLEX` bits) is kept.
///
/// # Performance
///
/// `O(n_points)` with a single linear pass and no allocation.
fn psh_glyph_load_points(glyph: &mut PshGlyph<'_>, dimension: usize, outline: &Outline) {
    for (point, vec) in glyph.points.iter_mut().zip(outline.points.iter()) {
        point.flags2 = 0;
        point.hint = None;
        if dimension == 0 {
            point.org_u = vec.x;
            point.org_v = vec.y;
        } else {
            point.org_u = vec.y;
            point.org_v = vec.x;
        }
    }
}

/// `psh_glyph_save_points` (`pshalgo.c`): copies the fitted coordinate
/// back into the outline and flags touched points in the tag byte.
///
/// # Performance
///
/// `O(n_points)` with a single linear pass.
fn psh_glyph_save_points(glyph: &PshGlyph<'_>, dimension: usize, outline: &mut Outline) {
    for (n, point) in glyph.points.iter().enumerate() {
        if let Some(vec) = outline.points.get_mut(n) {
            if dimension == 0 {
                vec.x = point.cur_u;
            } else {
                vec.y = point.cur_u;
            }
        }

        if point.is_strong()
            && let Some(tag) = outline.tags.get_mut(n)
        {
            *tag |= if dimension == 0 { 32 } else { 64 };
        }
    }
}

/// `psh_glyph_init` (`pshalgo.c`): fills `glyph` from `outline`,
/// `ps_hints` and the font's global data.
///
/// The point array is turned into the circular doubly-linked structure of
/// the C record, the per-point directions and smooth flags are computed,
/// the inflection points are marked, and both hint tables are built.
///
/// # Errors
///
/// * [`TtError::OUT_OF_MEMORY`] when a table cannot grow.
/// * [`TtError::INVALID_OUTLINE`] when `outline` does not satisfy
///   [`Outline::check`] (the C code would read out of bounds instead).
///
/// # Performance
///
/// `O(n_points + n_hints)`; every array is allocated exactly once.
fn psh_glyph_init<'a>(glyph: &mut PshGlyph<'a>, outline: &Outline, ps_hints: &'a PsHints) -> TtResult<()> {
    let num_points = outline.n_points.max(0) as usize;
    let num_contours = outline.n_contours.max(0) as usize;

    outline.check()?;

    glyph.points = Vec::new();
    glyph
        .points
        .try_reserve(num_points)
        .map_err(|_| TtError::OUT_OF_MEMORY)?;
    glyph
        .points
        .resize(num_points, PshPoint::default());

    glyph.contours = Vec::new();
    glyph
        .contours
        .try_reserve(num_contours)
        .map_err(|_| TtError::OUT_OF_MEMORY)?;
    glyph
        .contours
        .resize(num_contours, PshContour::default());

    // link the points of every contour into a circular list
    let mut first = 0usize;
    for n in 0..num_contours {
        let end = *outline
            .contours
            .get(n)
            .ok_or(TtError::INVALID_OUTLINE)?;
        let next = usize::try_from(end)
            .map_err(|_| TtError::INVALID_OUTLINE)?
            .saturating_add(1);
        if next > num_points || next <= first {
            return Err(TtError::INVALID_OUTLINE);
        }

        let contour = glyph
            .contours
            .get_mut(n)
            .ok_or(TtError::INVALID_HANDLE)?;
        contour.start = first;
        contour.count = next - first;

        for k in first..next {
            let Some(point) = glyph.points.get_mut(k) else {
                return Err(TtError::INVALID_OUTLINE);
            };
            point.prev = if k == first { next - 1 } else { k - 1 };
            point.next = if k + 1 == next { first } else { k + 1 };
        }

        first = next;
    }

    // tag the off-curve points and compute directions and smoothness
    for n in 0..num_points {
        let (n_prev, n_next) = match glyph.points.get(n) {
            Some(point) => (point.prev, point.next),
            None => break,
        };
        let Some(&tag) = outline.tags.get(n) else {
            break;
        };
        let (Some(vec), Some(vec_prev), Some(vec_next)) = (
            outline.points.get(n),
            outline.points.get(n_prev),
            outline.points.get(n_next),
        ) else {
            break;
        };

        // `FT_MEM_ZERO` above guarantees that `flags` starts cleared
        let off_curve = tag & CURVE_TAG_ON == 0;

        let dxi = vec.x.wrapping_sub(vec_prev.x);
        let dyi = vec.y.wrapping_sub(vec_prev.y);
        let dxo = vec_next.x.wrapping_sub(vec.x);
        let dyo = vec_next.y.wrapping_sub(vec.y);

        let dir_in = psh_compute_dir(dxi, dyi);
        let dir_out = psh_compute_dir(dxo, dyo);

        let Some(point) = glyph.points.get_mut(n) else {
            break;
        };
        if off_curve {
            point.flags = PSH_POINT_OFF;
        }
        point.dir_in = dir_in;
        point.dir_out = dir_out;

        // C nests this as `if (OFF) ... else if (dir_in == dir_out &&
        // (dir_out != NONE || flat)) ...`; both arms only set SMOOTH
        let smooth = point.flags & PSH_POINT_OFF != 0
            || (dir_in == dir_out && (dir_out != PSH_DIR_NONE || corner_is_flat(dxi, dyi, dxo, dyo)));
        if smooth {
            point.flags |= PSH_POINT_SMOOTH;
        }
    }

    psh_glyph_load_points(glyph, 0, outline);
    psh_glyph_compute_inflections(glyph);

    // now deal with the hint tables
    psh_hint_table_init(
        &mut glyph.hint_tables[0],
        ps_hints.hints(0),
        ps_hints.hint_masks(0),
    )?;
    psh_hint_table_init(
        &mut glyph.hint_tables[1],
        ps_hints.hints(1),
        ps_hints.hint_masks(1),
    )?;

    Ok(())
}

/// `psh_glyph_compute_inflections` (`pshalgo.c`): marks the points where
/// a contour changes its curvature, which lets the interpolator keep the
/// shape of glyphs like `S`.
///
/// Every `goto Skip`/`goto Next` of the C function becomes a
/// `continue`/`break` of a labelled loop over the contour.
///
/// # Performance
///
/// `O(n_points)` overall: each contour is walked twice at most.
fn psh_glyph_compute_inflections(glyph: &mut PshGlyph<'_>) {
    let num_contours = glyph.contours.len();

    'contour: for ci in 0..num_contours {
        let count = glyph.contours[ci].count;

        // we need at least 4 points to create an inflection point
        if count < 4 {
            continue;
        }

        let contour_start = glyph.contours[ci].start;
        let mut first = contour_start;
        let mut start = first;
        let mut end = first;

        // compute the first non-degenerate segment of the contour
        let mut in_x;
        let mut in_y;
        loop {
            end = glyph.points[end].next;
            if end == first {
                continue 'contour;
            }
            in_x = glyph.points[end].org_u - glyph.points[start].org_u;
            in_y = glyph.points[end].org_v - glyph.points[start].org_v;
            if in_x != 0 || in_y != 0 {
                break;
            }
        }

        // extend the segment start whenever possible
        let mut before = start;
        let mut out_x;
        let mut out_y;
        let mut orient_prev;
        loop {
            loop {
                start = before;
                before = glyph.points[start].prev;
                if before == first {
                    continue 'contour;
                }
                out_x = glyph.points[start].org_u - glyph.points[before].org_u;
                out_y = glyph.points[start].org_v - glyph.points[before].org_v;
                if out_x != 0 || out_y != 0 {
                    break;
                }
            }

            orient_prev = psh_corner_orientation(in_x, in_y, out_x, out_y);
            if orient_prev != 0 {
                break;
            }
        }

        first = start;
        in_x = out_x;
        in_y = out_y;

        // now, process all segments in the contour
        let mut finished = false;
        while !finished {
            // first, extend the current segment's end whenever possible
            let mut after = end;
            loop {
                loop {
                    end = after;
                    after = glyph.points[end].next;
                    if after == first {
                        finished = true;
                    }
                    out_x = glyph.points[after].org_u - glyph.points[end].org_u;
                    out_y = glyph.points[after].org_v - glyph.points[end].org_v;
                    if out_x != 0 || out_y != 0 {
                        break;
                    }
                }

                let orient_cur = psh_corner_orientation(in_x, in_y, out_x, out_y);
                if orient_cur != 0 {
                    if (orient_cur ^ orient_prev) < 0 {
                        loop {
                            glyph.points[start].set_inflex();
                            start = glyph.points[start].next;
                            if start == end {
                                break;
                            }
                        }
                        glyph.points[start].set_inflex();
                    }

                    start = end;
                    end = after;
                    orient_prev = orient_cur;
                    in_x = out_x;
                    in_y = out_y;
                    break;
                }
            }
        }
    }
}

/// `psh_glyph_compute_extrema` (`pshalgo.c`): marks every local extremum
/// of the glyph along both axes and decides, for each of them, whether it
/// belongs to a positive or a negative part of the contour.
///
/// # Performance
///
/// `O(n_points)`: each contour is scanned with a single forward walk.
fn psh_glyph_compute_extrema(glyph: &mut PshGlyph<'_>) {
    let num_contours = glyph.contours.len();

    // first of all, compute all local extrema
    'contour: for ci in 0..num_contours {
        if glyph.contours[ci].count == 0 {
            continue;
        }
        let contour_start = glyph.contours[ci].start;

        let mut first = contour_start;
        let mut point = first;
        let mut before = point;

        loop {
            before = glyph.points[before].prev;
            if before == first {
                continue 'contour;
            }
            if glyph.points[before].org_u != glyph.points[point].org_u {
                break;
            }
        }

        first = glyph.points[before].next;
        point = first;

        loop {
            let mut after = point;
            loop {
                after = glyph.points[after].next;
                if after == first {
                    // the contour has been fully scanned
                    continue 'contour;
                }
                if glyph.points[after].org_u != glyph.points[point].org_u {
                    break;
                }
            }

            let before_u = glyph.points[before].org_u;
            let point_u = glyph.points[point].org_u;
            let after_u = glyph.points[after].org_u;

            let local_maximum = before_u < point_u && after_u < point_u;
            let local_minimum = before_u > point_u && after_u > point_u;

            if local_maximum || local_minimum {
                loop {
                    glyph.points[point].set_extremum();
                    point = glyph.points[point].next;
                    if point == after {
                        break;
                    }
                }
            }

            before = glyph.points[after].prev;
            point = after;
        }
    }

    // for each extremum, determine its direction along the orthogonal
    // axis
    for n in 0..glyph.points.len() {
        let mut before = n;
        let mut after = n;
        let mut skip = false;

        if glyph.points[n].is_extremum() {
            loop {
                before = glyph.points[before].prev;
                if before == n {
                    skip = true;
                    break;
                }
                if glyph.points[before].org_v != glyph.points[n].org_v {
                    break;
                }
            }

            if !skip {
                loop {
                    after = glyph.points[after].next;
                    if after == n {
                        skip = true;
                        break;
                    }
                    if glyph.points[after].org_v != glyph.points[n].org_v {
                        break;
                    }
                }
            }
        }

        if skip {
            continue;
        }

        let before_v = glyph.points[before].org_v;
        let point_v = glyph.points[n].org_v;
        let after_v = glyph.points[after].org_v;

        if before_v < point_v && after_v > point_v {
            glyph.points[n].set_positive();
        } else if before_v > point_v && after_v < point_v {
            glyph.points[n].set_negative();
        }
    }
}

/// `psh_hint_table_find_strong_points` (`pshalgo.c`): marks the points of
/// `points[first..first + count]` that sit on the edge of an active hint
/// and whose tangent runs along the hint.
///
/// The active set of `table` must have been installed with
/// [`psh_hint_table_activate_mask`] beforehand.
///
/// # Performance
///
/// `O(count * num_active_hints)`; the inner scan stops at the first
/// matching hint, and hints never overlap within an active set.
fn psh_hint_table_find_strong_points(
    table: &PshHintTable<'_>,
    points: &mut [PshPoint],
    first: usize,
    count: usize,
    threshold: i32,
    major_dir: i32,
) {
    let num_hints = table.num_hints;
    let threshold = i64::from(threshold);

    for k in first..first.saturating_add(count) {
        let Some(point) = points.get_mut(k) else {
            break;
        };
        if point.is_strong() {
            continue;
        }

        let org_u = point.org_u;
        let mut point_dir = 0;

        if dir_compare(point.dir_in, major_dir) {
            point_dir = point.dir_in;
        } else if dir_compare(point.dir_out, major_dir) {
            point_dir = point.dir_out;
        }

        if point_dir != 0 {
            // the point sits either on the leading or on the trailing
            // edge of the hints
            let on_leading_edge = point_dir == major_dir;
            let on_trailing_edge = point_dir == -major_dir;

            if on_leading_edge || on_trailing_edge {
                for nn in 0..num_hints {
                    let Some(&hint_idx) = table.sort.get(nn) else {
                        break;
                    };
                    let Some(hint) = table.hints.get(hint_idx) else {
                        continue;
                    };

                    let hint_edge = if on_leading_edge {
                        i64::from(hint.org_pos)
                    } else {
                        i64::from(hint.org_pos) + i64::from(hint.org_len)
                    };
                    let d = org_u - hint_edge;

                    if d < threshold && -d < threshold {
                        point.set_strong();
                        if on_leading_edge {
                            point.set_edge_min();
                        } else {
                            point.set_edge_max();
                        }
                        point.hint = Some(hint_idx);
                        break;
                    }
                }
            }
        } else if point.is_extremum() {
            // treat extrema as special cases for stem edge alignment
            let (min_flag, max_flag) = if major_dir == PSH_DIR_HORIZONTAL {
                (PSH_POINT_POSITIVE, PSH_POINT_NEGATIVE)
            } else {
                (PSH_POINT_NEGATIVE, PSH_POINT_POSITIVE)
            };

            let flags2 = point.flags2;
            let on_min = flags2 & min_flag != 0;
            let on_max = !on_min && flags2 & max_flag != 0;

            if on_min || on_max {
                for nn in 0..num_hints {
                    let Some(&hint_idx) = table.sort.get(nn) else {
                        break;
                    };
                    let Some(hint) = table.hints.get(hint_idx) else {
                        continue;
                    };

                    let hint_edge = if on_min {
                        i64::from(hint.org_pos)
                    } else {
                        i64::from(hint.org_pos) + i64::from(hint.org_len)
                    };
                    let d = org_u - hint_edge;

                    if d < threshold && -d < threshold {
                        if on_min {
                            point.set_edge_min();
                        } else {
                            point.set_edge_max();
                        }
                        point.hint = Some(hint_idx);
                        point.set_strong();
                        break;
                    }
                }
            }

            // otherwise, fall back to the hint that contains the point
            if point.hint.is_none() {
                for nn in 0..num_hints {
                    let Some(&hint_idx) = table.sort.get(nn) else {
                        break;
                    };
                    let Some(hint) = table.hints.get(hint_idx) else {
                        continue;
                    };

                    let hint_pos = i64::from(hint.org_pos);
                    if org_u >= hint_pos && org_u <= hint_pos + i64::from(hint.org_len) {
                        point.hint = Some(hint_idx);
                        break;
                    }
                }
            }
        }
    }
}

/// `psh_glyph_find_strong_points` (`pshalgo.c`): looks for the points of
/// the glyph that sit on a stem edge of `dimension`.
///
/// # Performance
///
/// `O(n_points * num_hints)` worst case, one mask activation per hint
/// mask; masks are only re-activated when the outline range they cover
/// actually grows.
fn psh_glyph_find_strong_points(glyph: &mut PshGlyph<'_>, dimension: usize, globals: &PshGlobals) {
    let num_points = glyph.points.len();
    let PshGlyph {
        points, hint_tables, ..
    } = glyph;
    let Some(table) = hint_tables.get_mut(dimension) else {
        return;
    };

    let masks = table.hint_masks;
    let num_masks = masks.len();

    // a point is `strong' if it is located on a stem edge and has an
    // `in' or `out' tangent parallel to the hint's direction
    let major_dir = if dimension == 0 {
        PSH_DIR_VERTICAL
    } else {
        PSH_DIR_HORIZONTAL
    };

    let scale = globals
        .scale(dimension)
        .map_or(0, |(mult, _delta)| mult);
    let mut threshold = div_fix(PSH_STRONG_THRESHOLD, scale) as i32;
    if threshold > PSH_STRONG_THRESHOLD_MAXIMUM {
        threshold = PSH_STRONG_THRESHOLD_MAXIMUM;
    }

    // process secondary hints to `selected' points
    if num_masks > 1 && num_points > 0 {
        // the `endchar' op can reduce the number of points
        let mut first = masks[0].end_point.min(num_points);

        for mask in masks.iter().skip(1) {
            let next = mask.end_point.min(num_points);

            if next > first {
                psh_hint_table_activate_mask(table, mask);
                psh_hint_table_find_strong_points(table, points, first, next - first, threshold, major_dir);
            }
            first = next;
        }
    }

    // process primary hints for all points
    if num_masks == 1 {
        psh_hint_table_activate_mask(table, &masks[0]);
        psh_hint_table_find_strong_points(table, points, 0, num_points, threshold, major_dir);
    }

    // now, certain points may have been attached to a hint and not
    // marked as strong; update their flags then
    for point in points.iter_mut() {
        if point.hint.is_some() && !point.is_strong() {
            point.set_strong();
        }
    }
}

/// `psh_glyph_find_blue_points` (`pshalgo.c`): snaps the points that sit
/// in a blue zone and whose tangent runs horizontally to that zone,
/// marking them strong and fitted.
///
/// # Performance
///
/// `O(n_points * num_blue_zones)`; both zone tables are sorted, so the
/// top zones are scanned upwards and the bottom zones downwards.
fn psh_glyph_find_blue_points(blues: &PshBlues, glyph: &mut PshGlyph<'_>) {
    let fuzz = i64::from(blues.blue_fuzz);
    let threshold = i64::from(blues.blue_threshold);
    let no_overshoots = blues.no_overshoots;

    let top_count = blues
        .normal_top
        .count
        .min(PS_GLOBALS_MAX_BLUE_ZONES);
    let bot_count = blues
        .normal_bottom
        .count
        .min(PS_GLOBALS_MAX_BLUE_ZONES);

    for point in glyph.points.iter_mut() {
        // check tangents
        if !dir_compare(point.dir_in, PSH_DIR_HORIZONTAL) && !dir_compare(point.dir_out, PSH_DIR_HORIZONTAL) {
            continue;
        }

        // skip strong points
        if point.is_strong() {
            continue;
        }

        let y = point.org_u;

        // look up top zones
        for zone in blues.normal_top.zones[..top_count].iter() {
            let delta = y - i64::from(zone.org_bottom);

            if delta < -fuzz {
                break;
            }

            if y <= i64::from(zone.org_top) + fuzz && (no_overshoots || delta <= threshold) {
                point.cur_u = zone.cur_bottom;
                point.set_strong();
                point.set_fitted();
            }
        }

        // look up bottom zones
        let mut index = bot_count;
        while index > 0 {
            index -= 1;
            let zone = &blues.normal_bottom.zones[index];

            let delta = i64::from(zone.org_top) - y;

            if delta < -fuzz {
                break;
            }

            if y >= i64::from(zone.org_bottom) - fuzz && (no_overshoots || delta < threshold) {
                point.cur_u = zone.cur_top;
                point.set_strong();
                point.set_fitted();
            }
        }
    }
}

/// `psh_glyph_interpolate_strong_points` (`pshalgo.c`): positions the
/// points attached to a hint relative to that hint's fitted edges.
///
/// # Performance
///
/// `O(n_points)`; points without a hint are skipped in constant time.
fn psh_glyph_interpolate_strong_points(glyph: &mut PshGlyph<'_>, dimension: usize, globals: &PshGlobals) {
    let scale = globals
        .scale(dimension)
        .map_or(0, |(mult, _delta)| mult);
    let PshGlyph {
        points, hint_tables, ..
    } = glyph;
    let table = match hint_tables.get(dimension) {
        Some(table) => table,
        None => return,
    };

    for point in points.iter_mut() {
        let Some(hint_index) = point.hint else {
            continue;
        };
        let Some(hint) = table.hints.get(hint_index) else {
            continue;
        };

        if point.is_edge_min() {
            point.cur_u = hint.cur_pos;
        } else if point.is_edge_max() {
            point.cur_u = hint.cur_pos + hint.cur_len;
        } else {
            let org_len = i64::from(hint.org_len);
            let delta = point.org_u - i64::from(hint.org_pos);

            if delta <= 0 {
                point.cur_u = hint.cur_pos + mul_fix(delta, scale);
            } else if delta >= org_len {
                point.cur_u = hint.cur_pos + hint.cur_len + mul_fix(delta - org_len, scale);
            } else {
                // `org_len > delta > 0`, so the divisor is never zero
                point.cur_u = hint.cur_pos + mul_div(delta, hint.cur_len, org_len);
            }
        }

        point.set_fitted();
    }
}

/// `psh_glyph_interpolate_normal_points` (`pshalgo.c`): interpolates the
/// points that are not attached to any hint between the strong points
/// (the local extrema of the glyph) of the same dimension.
///
/// # Errors
///
/// Returns [`TtError::OUT_OF_MEMORY`] when the sorted list of strong
/// points does not fit into [`PSH_MAX_STRONG_INTERNAL`] entries and
/// cannot grow. FreeType silently skips the interpolation in that case;
/// the error is propagated here so that no partial result is written
/// back to the outline.
///
/// # Performance
///
/// `O(n_points * log n_points)` dominated by the insertion sort of the
/// strong points, which is linear for the usual case of an already
/// ascending outline.
fn psh_glyph_interpolate_normal_points(
    glyph: &mut PshGlyph<'_>,
    dimension: usize,
    globals: &PshGlobals,
) -> TtResult<()> {
    let scale = globals
        .scale(dimension)
        .map_or(0, |(mult, _delta)| mult);
    let num_points = glyph.points.len();

    // first count the number of strong points
    let mut num_strongs = 0usize;
    for point in glyph.points.iter() {
        if point.is_strong() {
            num_strongs += 1;
        }
    }

    // nothing to do here
    if num_strongs == 0 {
        return Ok(());
    }

    // collect the strong points in increasing `org_u` order: the stack
    // buffer mirrors `strongs_0[PSH_MAX_STRONG_INTERNAL]`, the `Vec`
    // mirrors the `FT_NEW_ARRAY` fallback of the C code
    let mut internal = [0usize; PSH_MAX_STRONG_INTERNAL];
    let mut heap = Vec::new();
    let strongs: &mut [usize] = if num_strongs <= PSH_MAX_STRONG_INTERNAL {
        &mut internal[..num_strongs]
    } else {
        heap.try_reserve(num_strongs)
            .map_err(|_| TtError::OUT_OF_MEMORY)?;
        heap.resize(num_strongs, 0);
        heap.as_mut_slice()
    };

    let mut count = 0usize;
    for n in 0..num_points {
        if !glyph.points[n].is_strong() {
            continue;
        }

        // insertion sort by `org_u`, keeping equal entries in outline
        // order exactly like the C loop
        let mut insert = count;
        while insert > 0 {
            let previous = strongs[insert - 1];
            if glyph.points[previous].org_u <= glyph.points[n].org_u {
                break;
            }
            strongs[insert] = previous;
            insert -= 1;
        }
        strongs[insert] = n;
        count += 1;
    }

    // now try to interpolate all normal points
    for n in 0..num_points {
        if glyph.points[n].is_strong() {
            continue;
        }

        // sometimes, some local extrema are smooth points
        let clear_smooth = {
            let point = &glyph.points[n];
            if point.is_smooth() {
                if point.dir_in == PSH_DIR_NONE || point.dir_in != point.dir_out {
                    continue;
                }
                if !point.is_extremum() && !point.is_inflex() {
                    continue;
                }
                true
            } else {
                false
            }
        };
        if clear_smooth {
            glyph.points[n].clear_smooth();
        }

        // find the best enclosing point coordinates then interpolate
        let org_u = glyph.points[n].org_u;

        let mut nn = 0usize;
        while nn < count && glyph.points[strongs[nn]].org_u <= org_u {
            nn += 1;
        }

        let cur_u = if nn == 0 {
            // point before the first strong point
            let after = strongs[0];
            let after_org = glyph.points[after].org_u;
            let after_cur = glyph.points[after].cur_u;
            after_cur + mul_fix(org_u - after_org, scale)
        } else {
            let before = strongs[nn - 1];

            let mut back = count;
            while back > 0 && glyph.points[strongs[back - 1]].org_u >= org_u {
                back -= 1;
            }

            if back == count {
                // the point is after the last strong point
                let last = strongs[count - 1];
                let last_org = glyph.points[last].org_u;
                let last_cur = glyph.points[last].cur_u;
                last_cur + mul_fix(org_u - last_org, scale)
            } else {
                let after = strongs[back];
                let after_org = glyph.points[after].org_u;
                let after_cur = glyph.points[after].cur_u;
                let before_org = glyph.points[before].org_u;
                let before_cur = glyph.points[before].cur_u;

                if org_u == before_org {
                    before_cur
                } else if org_u == after_org {
                    after_cur
                } else {
                    // the divisor is non-zero: `before` lies strictly
                    // left of `org_u`, `after` at or right of it
                    before_cur + mul_div(org_u - before_org, after_cur - before_cur, after_org - before_org)
                }
            }
        };

        glyph.points[n].cur_u = cur_u;
        glyph.points[n].set_fitted();
    }

    Ok(())
}

/// `psh_glyph_interpolate_other_points` (`pshalgo.c`): interpolates the
/// points of every contour between its already fitted points, or scales
/// the whole contour when it carries fewer than two of them.
///
/// # Porting note
///
/// The C local `delta` is declared outside of the contour loop, so a
/// contour without any fitted point keeps the translation that the
/// preceding one-fitted-point contour computed; this port preserves that
/// behaviour.
///
/// # Performance
///
/// `O(n_points)`: every contour is walked once with `next` pointers that
/// were built by [`psh_glyph_init`].
fn psh_glyph_interpolate_other_points(glyph: &mut PshGlyph<'_>, dimension: usize, globals: &PshGlobals) {
    let scale = globals
        .scale(dimension)
        .map_or(0, |(mult, _delta)| mult);
    let mut delta = globals
        .scale(dimension)
        .map_or(0, |(_mult, scale_delta)| scale_delta);
    let num_contours = glyph.contours.len();

    'contour: for ci in 0..num_contours {
        let contour_start = glyph.contours[ci].start;
        let contour_end = contour_start.saturating_add(glyph.contours[ci].count);

        // count the number of fitted points in this contour
        let mut fit_count = 0usize;
        let mut first: Option<usize> = None;
        for k in contour_start..contour_end {
            if glyph
                .points
                .get(k)
                .is_some_and(PshPoint::is_fitted)
            {
                if first.is_none() {
                    first = Some(k);
                }
                fit_count += 1;
            }
        }

        // with fewer than two fitted points the contour is simply
        // scaled (and eventually translated)
        if fit_count < 2 {
            if fit_count == 1 {
                let fitted = first.unwrap_or(contour_start);
                let cur_u = glyph.points[fitted].cur_u;
                let org_u = glyph.points[fitted].org_u;
                delta = cur_u - mul_fix(org_u, scale);
            }

            for k in contour_start..contour_end {
                if first != Some(k) {
                    let org_u = glyph.points[k].org_u;
                    let cur_u = mul_fix(org_u, scale) + delta;
                    glyph.points[k].cur_u = cur_u;
                }
            }

            continue 'contour;
        }

        // more than two fitted points: interpolate the weak points
        let start = match first {
            Some(start) => start,
            None => continue 'contour,
        };
        let mut anchor = start;

        loop {
            // skip consecutive fitted points
            let mut next;
            loop {
                next = glyph.points[anchor].next;
                if next == start {
                    continue 'contour;
                }
                if !glyph.points[next].is_fitted() {
                    break;
                }
                anchor = next;
            }

            // find the next fitted point after the unfitted one
            loop {
                next = glyph.points[next].next;
                if glyph.points[next].is_fitted() {
                    break;
                }
            }

            // now interpolate between them
            let org_a0 = glyph.points[anchor].org_u;
            let cur_a0 = glyph.points[anchor].cur_u;
            let org_b0 = glyph.points[next].org_u;
            let cur_b0 = glyph.points[next].cur_u;

            let (org_a, cur_a, org_ab, cur_ab) = if org_a0 <= org_b0 {
                (org_a0, cur_a0, org_b0 - org_a0, cur_b0 - cur_a0)
            } else {
                (org_b0, cur_b0, org_a0 - org_b0, cur_a0 - cur_b0)
            };

            let scale_ab = if org_ab > 0 {
                div_fix(cur_ab, org_ab)
            } else {
                0x1_0000
            };

            let mut point = glyph.points[anchor].next;
            loop {
                let org_c = glyph.points[point].org_u;
                let org_ac = org_c - org_a;

                let cur_c = if org_ac <= 0 {
                    // on the left of the interpolation zone
                    cur_a + mul_fix(org_ac, scale)
                } else if org_ac >= org_ab {
                    // on the right of the interpolation zone
                    cur_a + cur_ab + mul_fix(org_ac - org_ab, scale)
                } else {
                    // within the interpolation zone
                    cur_a + mul_fix(org_ac, scale_ab)
                };

                glyph.points[point].cur_u = cur_c;

                point = glyph.points[point].next;
                if point == next {
                    break;
                }
            }

            // keep going until all points of the contour are processed
            anchor = next;
            if anchor == start {
                break;
            }
        }
    }
}

/// `ps_hints_apply` (`pshalgo.c`): grid-fits `outline` with the hints
/// recorded in `ps_hints` and the font-wide data of `globals`.
///
/// `hint_mode` selects how the fitted coordinates are snapped: light
/// hinting keeps the scaled positions, normal hinting additionally
/// adjusts the stem widths, and monochrome/LCD hinting snaps the
/// corresponding axis to the pixel grid.
///
/// Both axes are fitted in turn (`do_horz_hints`/`do_vert_hints` are
/// hard-coded to `1` in FreeType 2.6), and the `y_scale` is temporarily
/// adjusted so that the top of non-capital letters lands on a pixel
/// boundary whenever possible — restored after the first dimension, as
/// in the C code.
///
/// # Errors
///
/// * [`TtError::INVALID_OUTLINE`] when `outline` fails
///   [`Outline::check`].
/// * [`TtError::OUT_OF_MEMORY`] when a working table cannot grow.
///
/// # Performance
///
/// `O(n_points * n_hints)` in the worst case, `O(n_points + n_hints)`
/// for typical outlines; the function allocates exactly five arrays per
/// call (points, contours, and one hint table plus sort array per
/// dimension).
pub fn ps_hints_apply(
    ps_hints: &PsHints,
    outline: &mut Outline,
    globals: &mut PshGlobals,
    hint_mode: RenderMode,
) -> TtResult<()> {
    // something to do?
    if outline.n_points == 0 || outline.n_contours == 0 {
        return Ok(());
    }

    let mut glyph = PshGlyph {
        points: Vec::new(),
        contours: Vec::new(),
        hint_tables: [
            PshHintTable::empty(ps_hints.hint_masks(0)),
            PshHintTable::empty(ps_hints.hint_masks(1)),
        ],
        options: FitOptions::new(hint_mode),
    };

    psh_glyph_init(&mut glyph, outline, ps_hints)?;

    // try to optimize the `y_scale` so that the top of non-capital
    // letters is aligned on a pixel boundary whenever possible
    let (old_x_scale, old_y_scale) = match (globals.scale(0), globals.scale(1)) {
        (Some((x_scale, _)), Some((y_scale, _))) => (x_scale, y_scale),
        _ => return Ok(()),
    };
    let mut x_scale = old_x_scale;
    let mut y_scale = old_y_scale;

    let org_ref = globals.blues.normal_top.zones[0].org_ref;
    let scaled = mul_fix(i64::from(org_ref), y_scale);
    let fitted = ft_pix_round(scaled);

    let mut rescale = false;
    if fitted != 0 && scaled != fitted {
        rescale = true;

        y_scale = mul_div(y_scale, fitted, scaled);
        if fitted < scaled {
            x_scale -= x_scale / 50;
        }

        globals.set_scale(x_scale, y_scale, 0, 0);
    }

    for dimension in 0..2 {
        // load outline coordinates into the glyph
        psh_glyph_load_points(&mut glyph, dimension, outline);

        // compute local extrema
        psh_glyph_compute_extrema(&mut glyph);

        // compute aligned stem/hint positions
        psh_hint_table_align_hints(
            &mut glyph.hint_tables[dimension],
            globals,
            dimension,
            &glyph.options,
        );

        // find strong points, align them, then interpolate the others
        psh_glyph_find_strong_points(&mut glyph, dimension, globals);
        if dimension == 1 {
            psh_glyph_find_blue_points(&globals.blues, &mut glyph);
        }
        psh_glyph_interpolate_strong_points(&mut glyph, dimension, globals);
        psh_glyph_interpolate_normal_points(&mut glyph, dimension, globals)?;
        psh_glyph_interpolate_other_points(&mut glyph, dimension, globals);

        // save hinted coordinates back to the outline
        psh_glyph_save_points(&glyph, dimension, outline);

        if rescale {
            globals.set_scale(old_x_scale, old_y_scale, 0, 0);
        }
    }

    Ok(())
}

/***************************************************************************/
/***************************************************************************/
/*****                                                                 *****/
/*****        PUBLIC INTERFACES AND MODULE FACADE                      *****/
/*****            (pshints.h, pshmod.c, pshpic.c)                     *****/
/*****                                                                 *****/
/***************************************************************************/
/***************************************************************************/

/// `PSH_Globals_FuncsRec` (`pshints.h`): the function record with which
/// the font drivers create, scale and destroy the font-wide globals.
///
/// Its members are exactly the three functions that
/// `psh_globals_funcs_init` installs in FreeType: [`psh_globals_new`],
/// [`psh_globals_set_scale`] and [`psh_globals_destroy`].
#[derive(Clone, Copy, Debug)]
pub struct PshGlobalsFuncs {
    /// `create`: `PSH_Globals_NewFunc`, builds a [`PshGlobals`] from a
    /// font's private dictionary.
    pub create: fn(Memory, &T1Private) -> TtResult<PshGlobals>,
    /// `set_scale`: `PSH_Globals_SetScaleFunc`, installs a device
    /// transform.
    pub set_scale: fn(&mut PshGlobals, Fixed, Fixed, Pos, Pos),
    /// `destroy`: `PSH_Globals_DestroyFunc`, releases the globals.
    pub destroy: fn(PshGlobals),
}

impl PshGlobalsFuncs {
    /// `psh_globals_funcs_init` (`pshglob.c`): fills the record with
    /// this crate's globals functions.
    pub fn new() -> Self {
        PshGlobalsFuncs {
            create: psh_globals_new,
            set_scale: psh_globals_set_scale,
            destroy: psh_globals_destroy,
        }
    }
}

impl Default for PshGlobalsFuncs {
    fn default() -> Self {
        Self::new()
    }
}

/// `T1_Hints_FuncsRec` (`pshints.h`): the Type 1 hints interface.
///
/// The C record also carries a `hints` handle to the module's single
/// `PS_HintsRec`, so that C call sites read
/// `funcs->stem(funcs.hints, ...)`. This port hands every caller its own
/// recorder (a [`T1Hints`] or a bare [`PsHints`]), which leaves the
/// record a plain function table whose first argument is always the
/// recorder — the very signature the C function pointers have.
///
/// # Example
///
/// ```
/// use codevar_truetype_pshinter::{Memory, PshinterModule, PsHints};
///
/// let module = PshinterModule::new();
/// let funcs = module.t1_funcs();
/// let mut hints = module.t1_hints();
/// (funcs.open)(&mut hints.recorder());
/// let recorder: &mut PsHints = hints.recorder();
/// (funcs.stem)(recorder, 1, &[100 << 16, 30 << 16]);
/// # let _ = Memory::default();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct T1HintsFuncs {
    /// `open`: `T1_Hints_OpenFunc`, starts a recording session.
    pub open: fn(&mut PsHints),
    /// `close`: `T1_Hints_CloseFunc`, ends a recording session.
    pub close: fn(&mut PsHints, u32) -> TtResult<()>,
    /// `stem`: `T1_Hints_SetStemFunc`, records one `(position, length)`
    /// stem; `dimension` is `0` for `vstem` (X) and `1` for `hstem` (Y).
    pub stem: fn(&mut PsHints, u32, &[Fixed; 2]),
    /// `stem3`: `T1_Hints_SetStem3Func`, records three
    /// counter-controlled stems.
    pub stem3: fn(&mut PsHints, u32, &[Fixed; 6]),
    /// `reset`: `T1_Hints_ResetFunc`, resets the stem masks at a point.
    pub reset: fn(&mut PsHints, u32),
    /// `apply`: `T1_Hints_ApplyFunc`, grid-fits an outline with the
    /// recorded hints.
    pub apply: fn(&PsHints, &mut Outline, &mut PshGlobals, RenderMode) -> TtResult<()>,
}

impl T1HintsFuncs {
    /// `t1_hints_funcs_init` (`pshrec.c`): fills the record with the
    /// Type 1 recorder functions.
    pub fn new() -> Self {
        T1HintsFuncs {
            open: t1_hints_open,
            close: PsHints::close,
            stem: PsHints::t1_stem,
            stem3: PsHints::t1_stem3,
            reset: PsHints::t1_reset,
            apply: ps_hints_apply,
        }
    }
}

impl Default for T1HintsFuncs {
    fn default() -> Self {
        Self::new()
    }
}

/// `T2_Hints_FuncsRec` (`pshints.h`): the Type 2 hints interface; see
/// [`T1HintsFuncs`] for the ownership note about the `hints` handle.
#[derive(Clone, Copy, Debug)]
pub struct T2HintsFuncs {
    /// `open`: `T2_Hints_OpenFunc`, starts a recording session.
    pub open: fn(&mut PsHints),
    /// `close`: `T2_Hints_CloseFunc`, ends a recording session.
    pub close: fn(&mut PsHints, u32) -> TtResult<()>,
    /// `stems`: `T2_Hints_StemsFunc`, records the delta-encoded stem
    /// table of one dimension (`0` for `vstem`, `1` for `hstem`).
    pub stems: fn(&mut PsHints, u32, usize, &[Fixed]),
    /// `hintmask`: `T2_Hints_MaskFunc`, records a `hintmask` operator.
    pub hintmask: fn(&mut PsHints, u32, usize, &[u8]),
    /// `counter`: `T2_Hints_CounterFunc`, records a `cntrmask`
    /// operator.
    pub counter: fn(&mut PsHints, usize, &[u8]),
    /// `apply`: `T2_Hints_ApplyFunc`, grid-fits an outline with the
    /// recorded hints.
    pub apply: fn(&PsHints, &mut Outline, &mut PshGlobals, RenderMode) -> TtResult<()>,
}

impl T2HintsFuncs {
    /// `t2_hints_funcs_init` (`pshrec.c`): fills the record with the
    /// Type 2 recorder functions.
    pub fn new() -> Self {
        T2HintsFuncs {
            open: t2_hints_open,
            close: PsHints::close,
            stems: PsHints::t2_stems,
            hintmask: PsHints::t2_mask,
            counter: PsHints::t2_counter,
            apply: ps_hints_apply,
        }
    }
}

impl Default for T2HintsFuncs {
    fn default() -> Self {
        Self::new()
    }
}

/// `t1_hints_open` (`pshrec.c`): the `open` member of [`T1HintsFuncs`],
/// which the C code passes as a plain function pointer.
#[inline]
fn t1_hints_open(hints: &mut PsHints) {
    hints.open(PsHintType::Type1);
}

/// `t2_hints_open` (`pshrec.c`): the `open` member of [`T2HintsFuncs`].
#[inline]
fn t2_hints_open(hints: &mut PsHints) {
    hints.open(PsHintType::Type2);
}

/// `PS_Hinter_ModuleRec` (`pshmod.c`): the PostScript hinter module.
///
/// FreeType builds one module per library, initializes its shared
/// `PS_HintsRec` in `ps_hinter_init`, and serves the three interface
/// records through the `PSHinter_Service` request. This port keeps the
/// allocator and the three (stateless) function tables, but hands out a
/// fresh recorder handle per session with [`PshinterModule::t1_hints`]
/// and [`PshinterModule::t2_hints`] instead of sharing one recorder —
/// equivalent here because a font driver never runs two recording
/// sessions at once.
///
/// # Example
///
/// ```
/// use codevar_truetype_pshinter::{PshinterModule, T1HintsFuncs, T2HintsFuncs};
///
/// let module = PshinterModule::new();
/// let t1: &T1HintsFuncs = module.t1_funcs();
/// let t2: &T2HintsFuncs = module.t2_funcs();
///
/// let mut hints = module.t2_hints();
/// (t2.open)(&mut hints.recorder());
/// (t1.open)(&mut hints.recorder());
/// ```
#[derive(Clone, Debug)]
pub struct PshinterModule {
    /// `root.memory`: the allocator every handle is created with.
    memory: Memory,
    /// `globals_funcs`: the globals interface of the module.
    globals_funcs: PshGlobalsFuncs,
    /// `t1_funcs`: the Type 1 recorder interface of the module.
    t1_funcs: T1HintsFuncs,
    /// `t2_funcs`: the Type 2 recorder interface of the module.
    t2_funcs: T2HintsFuncs,
}

impl PshinterModule {
    /// `ps_hinter_init` (`pshmod.c`): creates the module and installs
    /// its three interfaces.
    ///
    /// The C constructor can only return `FT_Err_Ok`, so this port
    /// builds the module infallibly.
    pub fn new() -> Self {
        PshinterModule {
            memory: Memory,
            globals_funcs: PshGlobalsFuncs::new(),
            t1_funcs: T1HintsFuncs::new(),
            t2_funcs: T2HintsFuncs::new(),
        }
    }

    /// `pshinter_get_globals_funcs` (`pshmod.c`): the globals interface
    /// of the module.
    pub fn globals_funcs(&self) -> &PshGlobalsFuncs {
        &self.globals_funcs
    }

    /// `pshinter_get_t1_funcs` (`pshmod.c`): the Type 1 hints interface
    /// of the module.
    pub fn t1_funcs(&self) -> &T1HintsFuncs {
        &self.t1_funcs
    }

    /// `pshinter_get_t2_funcs` (`pshmod.c`): the Type 2 hints interface
    /// of the module.
    pub fn t2_funcs(&self) -> &T2HintsFuncs {
        &self.t2_funcs
    }

    /// Creates a Type 1 recorder bound to the module's allocator, for
    /// one recording session (the C `T1_Hints` handle).
    pub fn t1_hints(&self) -> T1Hints {
        T1Hints::new(self.memory)
    }

    /// Creates a Type 2 recorder bound to the module's allocator, for
    /// one recording session (the C `T2_Hints` handle).
    pub fn t2_hints(&self) -> T2Hints {
        T2Hints::new(self.memory)
    }
}

impl Default for PshinterModule {
    fn default() -> Self {
        Self::new()
    }
}

/***************************************************************************/
/***************************************************************************/
/*****                                                                 *****/
/*****                            TESTS                                *****/
/*****                                                                 *****/
/***************************************************************************/
/***************************************************************************/

#[cfg(test)]
mod tests {
    //! Behaviour tests that pin the FreeType 2.6 quirks this port has to
    //! reproduce (delta decoding, 16-stem batches, the Type 2 mask bit
    //! layout) and the end-to-end hinting of a simple rectangle.

    use super::*;

    /// Builds a four-point rectangle whose points are all on-curve and
    /// whose contour ends at the last point (`Outline::check` shape).
    fn rectangle(x0: Pos, y0: Pos, x1: Pos, y1: Pos) -> Outline {
        let mut outline = Outline::with_capacity(4, 1);
        outline.points.push(Vector::new(x0, y0));
        outline.points.push(Vector::new(x1, y0));
        outline.points.push(Vector::new(x1, y1));
        outline.points.push(Vector::new(x0, y1));
        outline.tags = vec![CURVE_TAG_ON; 4];
        outline.contours.push(3);
        outline.n_points = 4;
        outline.n_contours = 1;
        outline
    }

    /// Builds globals scaled by `scale` (16.16 26.6-units per font
    /// unit); the default private dictionary never fails to build, so the
    /// `Result` is unwrapped here only.
    fn scaled_globals(scale: Fixed) -> PshGlobals {
        // justified: `psh_globals_new` cannot fail for `T1Private::default`
        let mut globals = PshGlobals::new(Memory, &T1Private::default()).unwrap();
        globals.set_scale(scale, scale, 0, 0);
        globals
    }

    /// `t2_hints_stems` (`pshrec.c`) decodes the delta-encoded stem
    /// table and, after a batch of sixteen stems, keeps reading the
    /// coordinate array from its start while the accumulator continues.
    #[test]
    fn t2_stems_are_delta_decoded_and_restart_after_sixteen_stems() {
        let mut hints = PsHints::new(Memory);
        hints.open(PsHintType::Type2);

        // two stems, each `+10` font units wide and `+10` long
        let coords = [10 << 16, 10 << 16, 10 << 16, 10 << 16];
        hints.t2_stems(1, 2, &coords);
        assert_eq!(hints.error(), TtError::OK);

        let stems = hints.hints(1);
        assert_eq!(stems.len(), 2);
        assert_eq!((stems[0].pos, stems[0].len), (10, 10));
        assert_eq!((stems[1].pos, stems[1].len), (30, 10));

        // 17 stems: batch one consumes 32 coordinates (`y` ends at 320),
        // batch two restarts at `coords[0]`, so stem 16 begins at 330
        // instead of 320 — the FreeType 2.6 batch quirk.
        let mut batched = PsHints::new(Memory);
        batched.open(PsHintType::Type2);
        batched.t2_stems(1, 17, &[10 << 16; 34]);

        let stems = batched.hints(1);
        assert_eq!(stems.len(), 17);
        assert_eq!((stems[15].pos, stems[15].len), (310, 10));
        assert_eq!((stems[16].pos, stems[16].len), (330, 10));
        assert_eq!(batched.error(), TtError::OK);
    }

    /// Ghost stems are flagged, a bottom ghost (`-21`) also moves its
    /// position, and identical stems are recorded only once.
    #[test]
    fn t1_ghost_stems_are_flagged_and_deduplicated() {
        let mut hints = PsHints::new(Memory);
        hints.open(PsHintType::Type1);

        hints.t1_stem(0, &[200 << 16, -21 << 16]);
        hints.t1_stem(0, &[400 << 16, -20 << 16]);
        assert_eq!(hints.error(), TtError::OK);

        let stems = hints.hints(0);
        assert_eq!(stems.len(), 2);
        assert_eq!((stems[0].pos, stems[0].len), (179, 0));
        assert_eq!(stems[0].flags & PS_HINT_FLAG_GHOST, PS_HINT_FLAG_GHOST);
        assert_eq!(stems[0].flags & PS_HINT_FLAG_BOTTOM, PS_HINT_FLAG_BOTTOM);
        assert_eq!((stems[1].pos, stems[1].len), (400, 0));
        assert_eq!(stems[1].flags & PS_HINT_FLAG_GHOST, PS_HINT_FLAG_GHOST);
        assert_eq!(stems[1].flags & PS_HINT_FLAG_BOTTOM, 0);

        // a repeated stem reuses the existing record but keeps its mask bit
        hints.t1_stem(0, &[400 << 16, -20 << 16]);
        assert_eq!(hints.hints(0).len(), 2);
        let masks = hints.hint_masks(0);
        assert_eq!(masks.len(), 1);
        assert!(masks[0].test_bit(0));
        assert!(masks[0].test_bit(1));
    }

    /// The Type 2 `hintmask` gives the first `count2` bits to the
    /// horizontal (dimension 1) hints and the remaining `count1` bits to
    /// the vertical (dimension 0) ones.
    #[test]
    fn t2_hintmask_splits_bits_between_dimensions() {
        let mut hints = PsHints::new(Memory);
        hints.open(PsHintType::Type2);

        let coords = [10 << 16, 10 << 16, 10 << 16, 10 << 16];
        hints.t2_stems(1, 2, &coords);
        hints.t2_stems(0, 2, &coords);

        // bits `10 10`: keep hint 0 of each dimension, drop hint 1
        hints.t2_mask(3, 4, &[0xA0]);
        assert_eq!(hints.error(), TtError::OK);

        let horizontal = hints.hint_masks(1);
        assert_eq!(horizontal.len(), 2);
        assert_eq!(horizontal[0].end_point, 3);
        assert!(horizontal[0].test_bit(0));
        assert!(horizontal[0].test_bit(1));
        assert_eq!(horizontal[1].num_bits, 2);
        assert_eq!(horizontal[1].end_point, 0);
        assert!(horizontal[1].test_bit(0));
        assert!(!horizontal[1].test_bit(1));

        let vertical = hints.hint_masks(0);
        assert_eq!(vertical.len(), 2);
        assert_eq!(vertical[0].end_point, 3);
        assert_eq!(vertical[1].num_bits, 2);
        assert!(vertical[1].test_bit(0));
        assert!(!vertical[1].test_bit(1));

        // a wrong bit count silently ignores the operator, as in C
        hints.t2_mask(4, 3, &[0xE0]);
        assert_eq!(hints.error(), TtError::OK);
        assert_eq!(hints.hint_masks(1).len(), 2);
    }

    /// `hstem3` records three counter-controlled stems plus the counter
    /// mask that groups them (`ps_hints_t1stem3`).
    #[test]
    fn t1_stem3_records_a_counter_group() {
        let module = PshinterModule::new();
        let funcs: &T1HintsFuncs = module.t1_funcs();
        let mut hints = module.t1_hints();

        (funcs.open)(hints.recorder());
        (funcs.stem3)(
            hints.recorder(),
            0,
            &[11 << 16, 20 << 16, 41 << 16, 20 << 16, 71 << 16, 20 << 16],
        );
        (funcs.close)(hints.recorder(), 3).unwrap_or_else(|error| panic!("close failed: {error:?}"));

        let recorder = hints.recorder();
        assert_eq!(recorder.error(), TtError::OK);
        assert_eq!(recorder.hints(0).len(), 3);
        assert_eq!(recorder.hints(0)[1].pos, 41);
        assert_eq!(recorder.hints(1).len(), 0);

        let counters = &recorder.dimension[0].counters;
        assert_eq!(counters.num_masks, 1);
        let counter = counters
            .live()
            .first()
            .unwrap_or_else(|| panic!("missing counter"));
        assert!(counter.test_bit(0));
        assert!(counter.test_bit(1));
        assert!(counter.test_bit(2));
        // `ps_dimension_end_mask` only stamps the hint masks, so a
        // counter keeps the end point it was created with
        assert_eq!(counter.end_point, 0);
    }

    /// The module's function records dispatch into the recorder: `open`
    /// selects the dialect of the session.
    #[test]
    fn module_function_records_select_the_recorder_dialect() {
        let module = PshinterModule::new();

        let t2: &T2HintsFuncs = module.t2_funcs();
        let mut hints = module.t2_hints();
        (t2.open)(hints.recorder());
        assert_eq!(hints.recorder().hint_type, Some(PsHintType::Type2));

        let t1: &T1HintsFuncs = module.t1_funcs();
        (t1.open)(hints.recorder());
        assert_eq!(hints.recorder().hint_type, Some(PsHintType::Type1));

        // a stem recorded through the Type 1 record reaches dimension 0
        (t1.stem)(hints.recorder(), 0, &[11 << 16, 100 << 16]);
        assert_eq!(hints.recorder().hints(0).len(), 1);
        assert_eq!(hints.recorder().error(), TtError::OK);

        // the globals record builds and releases globals
        let funcs: PshGlobalsFuncs = *module.globals_funcs();
        let mut globals = (funcs.create)(Memory, &T1Private::default())
            .unwrap_or_else(|error| panic!("globals failed: {error:?}"));
        (funcs.set_scale)(&mut globals, 0x4000, 0x4000, 0, 0);
        assert_eq!(globals.scale(0), Some((0x4000, 0)));
        (funcs.destroy)(globals);

        module.t2_funcs();
    }

    /// `ps_hints_apply` grid-fits the two vertical stem edges of a
    /// rectangle onto the pixel grid while leaving the unhinted axis
    /// merely scaled.
    #[test]
    fn apply_grid_fits_the_stem_edges_of_a_rectangle() {
        // 16.16 scale of 4.0: 4 (26.6) units per font unit, i.e. a 16 px
        // em of 1024 font units
        let mut globals = scaled_globals(4 << 16);

        let mut hints = PsHints::new(Memory);
        hints.open(PsHintType::Type1);
        // one vertical stem (dimension 0) spanning both sides of the glyph
        hints.t1_stem(0, &[11 << 16, 100 << 16]);
        hints
            .close(3)
            .unwrap_or_else(|error| panic!("close failed: {error:?}"));

        let mut outline = rectangle(11, 0, 111, 137);
        ps_hints_apply(&hints, &mut outline, &mut globals, RenderMode::Normal)
            .unwrap_or_else(|error| panic!("apply failed: {error:?}"));

        let x: Vec<Pos> = outline
            .points
            .iter()
            .map(|point| point.x)
            .collect();
        let y: Vec<Pos> = outline
            .points
            .iter()
            .map(|point| point.y)
            .collect();

        // the stem spans 0.6875..6.9375 px and is fitted to 1..7 px
        assert_eq!(x, [64, 448, 448, 64]);
        // the unhinted axis keeps the plain scaled coordinates
        assert_eq!(y, [0, 0, 548, 548]);
    }

    /// An outline whose counters disagree with its slices, and an empty
    /// one, are the two early-outs of [`ps_hints_apply`].
    #[test]
    fn apply_rejects_invalid_and_accepts_empty_outlines() {
        let mut globals = scaled_globals(4 << 16);
        let hints = PsHints::new(Memory);

        let mut empty = Outline::new();
        assert_eq!(
            ps_hints_apply(&hints, &mut empty, &mut globals, RenderMode::Normal),
            Ok(())
        );

        let mut broken = rectangle(0, 0, 100, 100);
        broken.n_points = 5;
        assert_eq!(
            ps_hints_apply(&hints, &mut broken, &mut globals, RenderMode::Normal),
            Err(TtError::INVALID_OUTLINE)
        );
    }

    /// `psh_globals_new` + `psh_globals_set_scale` build the blue zones
    /// and the standard/snap widths of the font and scale them.
    #[test]
    fn globals_build_and_scale_blue_zones_and_widths() {
        let private = T1Private {
            num_blue_values: 4,
            blue_values: [0, 30, 700, 750, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            standard_width: [50],
            num_snap_widths: 2,
            snap_widths: [40, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ..T1Private::default()
        };

        // justified: a bounded private dictionary never fails to build
        let mut globals = PshGlobals::new(Memory, &private).unwrap();
        globals.set_scale(4 << 16, 4 << 16, 0, 0);

        // the first pair becomes a bottom zone, the second a top zone
        assert_eq!(globals.blue_zone_count(0), 1);
        assert_eq!(globals.blue_zone_count(1), 1);
        let top = globals.blue_zone(0, 0).unwrap_or_default();
        assert_eq!((top.org_ref, top.org_delta), (700, 50));
        assert_eq!((top.org_bottom, top.org_top), (699, 751));
        assert_eq!(top.cur_ref, 2816);
        assert_eq!((top.cur_bottom, top.cur_top), (2796, 3004));

        let bottom = globals.blue_zone(1, 0).unwrap_or_default();
        assert_eq!((bottom.org_ref, bottom.org_delta), (30, -30));
        assert_eq!((bottom.org_bottom, bottom.org_top), (-1, 31));

        // the standard width and its snaps are scaled (4 units per
        // font unit) and quantized to the pixel grid
        assert_eq!(globals.std_width(1, 0), Some((50, 200, 192)));
        assert_eq!(globals.std_width(1, 1), Some((40, 200, 192)));
        assert_eq!(globals.std_width(1, 2), Some((60, 200, 192)));

        // `BlueScale` is clamped to `1000 / max(blue zone heights)`
        assert_eq!(globals.blues.blue_scale, 1_310_720);
        assert_eq!(globals.blue_threshold(), 7);
        assert!(!globals.no_overshoots());
        assert_eq!(globals.scale(0), Some((4 << 16, 0)));
    }
}
