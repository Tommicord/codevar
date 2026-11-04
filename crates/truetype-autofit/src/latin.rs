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

//! Latin writing system: global metrics initialization, glyph
//! analysis and grid fitting (`aflatin.c` of FreeType 2.6).
//!
//! The record types declared above mirror `aflatin.h`; the routines
//! below port the functions of `aflatin.c` that fill and consume
//! them.

use codevar_truetype_core::{CURVE_TAG_ON, Outline, RenderMode, TtResult, div_fix, mul_div, mul_fix};

use crate::face::{Encoding, Face, STYLE_FLAG_ITALIC};
use crate::hints::{
    EDGE_DONE, EDGE_NEUTRAL, EDGE_NORMAL, EDGE_ROUND, EDGE_SERIF, FLAG_CONTROL, GlyphHints, do_blues,
};
use crate::metrics::{PROP_INCREASE_X_HEIGHT_MIN, StyleMetrics, StyleMetricsRec};
use crate::ranges::SCRIPT_CLASSES;
use crate::{
    BLUE_STRING_MAX_LEN, BLUE_STRINGS, BLUE_STRINGSET_MAX_LEN, BLUE_STRINGSETS, DIMENSION_MAX, Dimension,
    Direction, LATIN_MAX_WIDTHS, Pos, SCALER_FLAG_NO_HORIZONTAL, SCALER_FLAG_NO_WARPER, Scaler, Width,
    sort_and_quantize_widths, sort_positions,
};

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

/// `AF_LATIN_CONSTANT` (`aflatin.h`): scales the heuristic constant
/// `c` (given for `units_per_em == 2048`) to the face's
/// `units_per_em`.
#[inline]
fn latin_constant(units_per_em: u32, c: i64) -> Pos {
    c * i64::from(units_per_em) / 2048
}

/// The `units_per_em` cached by [`GlyphHints::rescale`], used where
/// FreeType reads `AF_LATIN_CONSTANT(hints->metrics, ...)`.
#[inline]
fn hints_units_per_em(hints: &GlyphHints) -> u32 {
    u32::from(hints.units_per_em)
}

/// `FT_CURVE_TAG(...) == FT_CURVE_TAG_ON`: true when the point at
/// `index` of `outline` lies on the curve.  Out of range indices
/// (including negative ones) are never on the curve.
#[inline]
fn point_is_on(outline: &Outline, index: i32) -> bool {
    outline
        .tags
        .get(index.max(0) as usize)
        .is_some_and(|tag| tag & 3 == CURVE_TAG_ON)
}

/// True when both directions lie along the same axis, mirroring the
/// `FT_ABS(dir) == FT_ABS(major_dir)` tests of `aflatin.c`.
#[inline]
fn same_axis_dir(a: Direction, b: Direction) -> bool {
    a.code().unsigned_abs() == b.code().unsigned_abs()
}

/// `FT_PIX_ROUND` (`ftcalc.h`): rounds `x` to the nearest multiple
/// of 64.
#[inline]
fn pix_round(x: Pos) -> Pos {
    x.wrapping_add(32) & !63
}

/// `af_latin_hints_compute_segments` (`aflatin.c`): walks every
/// contour of the reloaded hint record and collects the point runs
/// whose outward direction lies along the axis' major direction.
///
/// The C function reports `FT_Err_Out_Of_Memory`; the port's segment
/// table is a `Vec`, so this function is infallible.
pub fn compute_segments(hints: &mut GlyphHints, dim: Dimension) {
    for point in &mut hints.points {
        if dim == Dimension::Hort {
            point.u = i64::from(point.fx);
            point.v = i64::from(point.fy);
        } else {
            point.u = i64::from(point.fy);
            point.v = i64::from(point.fx);
        }
    }

    let axis = &mut hints.axis[dim.index()];
    let major_dir = axis.major_dir;
    axis.segments.clear();

    let mut segment_dir = Direction::None;
    let mut segment: Option<usize> = None;
    let mut segment_first = 0usize;
    let points_len = hints.points.len();

    for &contour_start in &hints.contours {
        let mut point = contour_start;
        if point >= points_len {
            continue;
        }
        let mut last = hints.points[point].prev;
        if last >= points_len {
            continue;
        }
        let mut on_edge = false;
        let mut min_pos: Pos = 32000;
        let mut max_pos: Pos = -32000;
        let mut passed;

        if point == last {
            continue;
        }

        if same_axis_dir(hints.points[last].out_dir, major_dir)
            && same_axis_dir(hints.points[point].out_dir, major_dir)
        {
            last = point;
            loop {
                point = hints.points[point].prev;
                if !same_axis_dir(hints.points[point].out_dir, major_dir) {
                    point = hints.points[point].next;
                    break;
                }
                if point == last {
                    break;
                }
            }
        }

        last = point;
        passed = false;

        loop {
            if on_edge {
                let u = hints.points[point].u;
                if u < min_pos {
                    min_pos = u;
                }
                if u > max_pos {
                    max_pos = u;
                }

                if hints.points[point].out_dir != segment_dir || point == last {
                    if let Some(seg_idx) = segment {
                        let seg = &mut axis.segments[seg_idx];
                        seg.last = Some(point);
                        seg.pos = ((min_pos + max_pos) >> 1) as i16;

                        let flags = hints.points[segment_first].flags | hints.points[point].flags;
                        if flags & FLAG_CONTROL != 0 {
                            seg.flags |= EDGE_ROUND;
                        }

                        let point_v = hints.points[point].v;
                        min_pos = point_v;
                        max_pos = point_v;
                        let first_v = hints.points[segment_first].v;
                        if first_v < min_pos {
                            min_pos = first_v;
                        }
                        if first_v > max_pos {
                            max_pos = first_v;
                        }
                        let min_coord = min_pos as i16;
                        let max_coord = max_pos as i16;
                        seg.min_coord = min_coord;
                        seg.max_coord = max_coord;
                        seg.height = (i32::from(max_coord) - i32::from(min_coord)) as i16;
                    }
                    on_edge = false;
                    segment = None;
                }
            }

            if point == last {
                if passed {
                    break;
                }
                passed = true;
            }

            if !on_edge && same_axis_dir(hints.points[point].out_dir, major_dir) {
                segment_dir = hints.points[point].out_dir;
                let idx = axis.new_segment();
                let seg = &mut axis.segments[idx];
                seg.score = 32000;
                seg.flags = EDGE_NORMAL;
                seg.dir = segment_dir;
                min_pos = hints.points[point].u;
                max_pos = min_pos;
                seg.first = Some(point);
                seg.last = Some(point);

                segment = Some(idx);
                segment_first = point;
                on_edge = true;
            }

            if point >= points_len {
                break;
            }
            point = hints.points[point].next;
        }
    }

    for seg in axis.segments.iter_mut() {
        let (Some(first), Some(last)) = (seg.first, seg.last) else {
            continue;
        };
        let (Some(first_point), Some(last_point)) = (hints.points.get(first), hints.points.get(last)) else {
            continue;
        };
        let first_v = first_point.v;
        let last_v = last_point.v;
        let mut height = seg.height;
        if first_v < last_v {
            if let Some(point) = hints.points.get(first_point.prev)
                && point.v < first_v
            {
                height = (i64::from(height) + ((first_v - point.v) >> 1)) as i16;
            }
            if let Some(point) = hints.points.get(last_point.next)
                && point.v > last_v
            {
                height = (i64::from(height) + ((point.v - last_v) >> 1)) as i16;
            }
        } else {
            if let Some(point) = hints.points.get(first_point.prev)
                && point.v > first_v
            {
                height = (i64::from(height) + ((point.v - first_v) >> 1)) as i16;
            }
            if let Some(point) = hints.points.get(last_point.next)
                && point.v < last_v
            {
                height = (i64::from(height) + ((last_v - point.v) >> 1)) as i16;
            }
        }
        seg.height = height;
    }
}

/// `af_latin_hints_link_segments` (`aflatin.c`): pairs segments of
/// opposite direction that face each other to form stems, and demotes
/// unmatched pair members to serifs.
///
/// When `width_count` and `widths` are non-zero (zero and empty in
/// `af_latin_metrics_init_widths`) the stem scoring is fine-tuned
/// with the known standard widths.
pub fn link_segments(hints: &mut GlyphHints, width_count: usize, widths: &[Width], dim: Dimension) {
    let units_per_em = hints_units_per_em(hints);
    let mut len_threshold = latin_constant(units_per_em, 8);
    if len_threshold == 0 {
        len_threshold = 1;
    }
    let len_score = latin_constant(units_per_em, 6000);
    let dist_score: Pos = 3000;
    let max_width = if width_count > 0 {
        widths
            .get(width_count - 1)
            .map_or(0, |width| width.org)
    } else {
        0
    };

    let axis = &mut hints.axis[dim.index()];
    let num_segments = axis.segments.len();

    for seg1 in 0..num_segments {
        if axis.segments[seg1].dir != axis.major_dir {
            continue;
        }
        for seg2 in 0..num_segments {
            let (first, second) = (&axis.segments[seg1], &axis.segments[seg2]);
            if first.dir.code() + second.dir.code() != 0 {
                continue;
            }
            let pos1 = i64::from(first.pos);
            let pos2 = i64::from(second.pos);
            if pos2 <= pos1 {
                continue;
            }

            let mut min = first.min_coord;
            if second.min_coord < min {
                min = second.min_coord;
            }
            let mut max = first.max_coord;
            if second.max_coord > max {
                max = second.max_coord;
            }
            let len = i64::from(max) - i64::from(min);
            if len < len_threshold {
                continue;
            }

            let dist = pos2 - pos1;
            let dist_demerit = if max_width != 0 {
                let delta = ((dist << 10) / max_width) - (1 << 10);
                if delta > 10000 {
                    32000
                } else if delta > 0 {
                    delta * delta / dist_score
                } else {
                    0
                }
            } else {
                dist
            };
            let score = dist_demerit + len_score / len;

            if score < axis.segments[seg1].score {
                axis.segments[seg1].score = score;
                axis.segments[seg1].link = Some(seg2);
            }
            if score < axis.segments[seg2].score {
                axis.segments[seg2].score = score;
                axis.segments[seg2].link = Some(seg1);
            }
        }
    }

    for seg1 in 0..num_segments {
        let Some(seg2) = axis.segments[seg1].link else {
            continue;
        };
        let Some(linked) = axis
            .segments
            .get(seg2)
            .map(|segment| segment.link)
        else {
            continue;
        };
        if linked != Some(seg1) {
            axis.segments[seg1].link = None;
            axis.segments[seg1].serif = linked;
        }
    }
}

/// Port of the per-character extremum search of
/// `af_latin_metrics_init_blues` (`aflatin.c`): finds the highest
/// (top blue) or lowest (otherwise) point of the outline, decides
/// whether the segment through it is flat or round, and applies the
/// additional constraints of a long blue zone.
///
/// Returns the blue zone coordinate and the `round` flag, or `None`
/// when the glyph must be skipped: a degenerate extremum direction,
/// or a round segment on a neutral zone.
///
/// # Porting notes
///
/// * `y_offset` is always zero: this port has no HarfBuzz shim.
/// * C's `continue` statements jump to the enclosing `while (*p)`
///   condition, which this port reproduces with `return None`.
/// * The innermost update of the long-blue search reads the stale
///   `points[next]` and `dist` variables of the earlier scans in the
///   C source; the stale values are reproduced bit for bit.
/// * A malformed outline is rejected through [`Outline::check`]
///   instead of being indexed blindly.
fn blue_extremum(
    outline: &Outline,
    is_top: bool,
    is_neutral: bool,
    is_long: bool,
    units_per_em: u32,
) -> Option<(Pos, bool)> {
    if outline.check().is_err() {
        return None;
    }
    let points = &outline.points;
    let mut best_point: i32 = -1;
    let mut best_y: Pos = 0;
    let mut best_contour_first = 0i32;
    let mut best_contour_last = 0i32;

    let mut first = 0i32;
    for &contour_end in &outline.contours {
        let last = i32::from(contour_end);
        if last > first {
            let old_best_point = best_point;
            for pp in first..=last {
                let y = points[pp as usize].y;
                let better = if is_top { y > best_y } else { y < best_y };
                if best_point < 0 || better {
                    best_point = pp;
                    best_y = y;
                }
            }
            if best_point != old_best_point {
                best_contour_first = first;
                best_contour_last = last;
            }
        }
        first = last + 1;
    }

    if best_point < 0 {
        return Some((best_y, false));
    }

    let best_x = points[best_point as usize].x;
    let mut best_segment_first = best_point;
    let mut best_segment_last = best_point;
    let (mut on_point_first, mut on_point_last);
    if point_is_on(outline, best_point) {
        on_point_first = best_point;
        on_point_last = best_point;
    } else {
        on_point_first = -1;
        on_point_last = -1;
    }

    let mut dist: u64;
    let mut prev = best_point;
    let mut next = best_point;

    loop {
        prev = if prev > best_contour_first {
            prev - 1
        } else {
            best_contour_last
        };
        dist = points[prev as usize]
            .y
            .wrapping_sub(best_y)
            .unsigned_abs();
        if dist > 5
            && points[prev as usize]
                .x
                .wrapping_sub(best_x)
                .unsigned_abs()
                <= dist.wrapping_mul(20)
        {
            break;
        }
        best_segment_first = prev;
        if point_is_on(outline, prev) {
            on_point_first = prev;
            if on_point_last < 0 {
                on_point_last = prev;
            }
        }
        if prev == best_point {
            break;
        }
    }

    loop {
        next = if next < best_contour_last {
            next + 1
        } else {
            best_contour_first
        };
        dist = points[next as usize]
            .y
            .wrapping_sub(best_y)
            .unsigned_abs();
        if dist > 5
            && points[next as usize]
                .x
                .wrapping_sub(best_x)
                .unsigned_abs()
                <= dist.wrapping_mul(20)
        {
            break;
        }
        best_segment_last = next;
        if point_is_on(outline, next) {
            on_point_last = next;
            if on_point_first < 0 {
                on_point_first = next;
            }
        }
        if next == best_point {
            break;
        }
    }

    if is_long {
        let length_threshold = u64::from(units_per_em) / 25;
        dist = points[best_segment_last as usize]
            .x
            .wrapping_sub(points[best_segment_first as usize].x)
            .unsigned_abs();

        if dist < length_threshold
            && best_segment_last - best_segment_first + 2 <= best_contour_last - best_contour_first
        {
            let height_threshold = u64::from(units_per_em) / 4;

            prev = best_point;
            loop {
                prev = if prev > best_contour_first {
                    prev - 1
                } else {
                    best_contour_last
                };
                if points[prev as usize].x != best_x {
                    break;
                }
                if prev == best_point {
                    break;
                }
            }
            if prev == best_point {
                return None;
            }

            let left2right = points[prev as usize].x < points[best_point as usize].x;
            let mut first_point = best_segment_last;
            let mut last = first_point;
            let mut hit = false;
            let mut p_first = 0i32;
            let mut p_last = 0i32;

            loop {
                if !hit {
                    first_point = last;
                    if point_is_on(outline, first_point) {
                        p_first = first_point;
                        p_last = first_point;
                    } else {
                        p_first = -1;
                        p_last = -1;
                    }
                    hit = true;
                }
                last = if last < best_contour_last {
                    last + 1
                } else {
                    best_contour_first
                };

                let mut skip_rest = false;
                let vertical = points[first_point as usize]
                    .y
                    .wrapping_sub(best_y)
                    .unsigned_abs();
                if vertical > height_threshold {
                    hit = false;
                    skip_rest = true;
                } else {
                    dist = points[last as usize]
                        .y
                        .wrapping_sub(points[first_point as usize].y)
                        .unsigned_abs();
                    if dist > 5
                        && points[last as usize]
                            .x
                            .wrapping_sub(points[first_point as usize].x)
                            .unsigned_abs()
                            <= dist.wrapping_mul(20)
                    {
                        hit = false;
                        skip_rest = true;
                    }
                }

                if !skip_rest {
                    if point_is_on(outline, last) {
                        p_last = last;
                        if p_first < 0 {
                            p_first = last;
                        }
                    }
                    let l2r = points[first_point as usize].x < points[last as usize].x;
                    let d = points[last as usize]
                        .x
                        .wrapping_sub(points[first_point as usize].x)
                        .unsigned_abs();

                    if l2r == left2right && d >= length_threshold {
                        loop {
                            last = if last < best_contour_last {
                                last + 1
                            } else {
                                best_contour_first
                            };
                            let d_inner = points[last as usize]
                                .y
                                .wrapping_sub(points[first_point as usize].y)
                                .unsigned_abs();
                            if d_inner > 5
                                && points[next as usize]
                                    .x
                                    .wrapping_sub(points[first_point as usize].x)
                                    .unsigned_abs()
                                    <= dist.wrapping_mul(20)
                            {
                                last = if last > best_contour_first {
                                    last - 1
                                } else {
                                    best_contour_last
                                };
                                break;
                            }
                            p_last = last;
                            if point_is_on(outline, last) {
                                p_last = last;
                                if p_first < 0 {
                                    p_first = last;
                                }
                            }
                            if last == best_segment_first {
                                break;
                            }
                        }

                        best_y = points[first_point as usize].y;
                        best_segment_first = first_point;
                        best_segment_last = last;
                        on_point_first = p_first;
                        on_point_last = p_last;
                        break;
                    }
                }

                if last == best_segment_first {
                    break;
                }
            }
        }
    }

    let wide = on_point_first >= 0
        && on_point_last >= 0
        && points[on_point_last as usize]
            .x
            .wrapping_sub(points[on_point_first as usize].x)
            .unsigned_abs()
            > u64::from(units_per_em) / 8;
    let round =
        !wide && (!point_is_on(outline, best_segment_first) || !point_is_on(outline, best_segment_last));

    if round && is_neutral {
        return None;
    }
    Some((best_y, round))
}

/// `af_latin_metrics_init_widths` (`aflatin.c`): finds the standard
/// stem widths of the face by analysing the glyph of the script's
/// standard character, then derives the standard width and the edge
/// distance threshold of both dimensions.
///
/// The analysis is skipped (but the fallback widths are still set)
/// when the standard character is missing or its glyph cannot be
/// loaded, exactly as FreeType's `goto Exit` does.
fn init_widths(metrics: &mut LatinMetrics, face: &mut dyn Face) {
    let mut hints = GlyphHints::default();
    metrics.axis[Dimension::Hort.index()].width_count = 0;
    metrics.axis[Dimension::Vert.index()].width_count = 0;

    let script_class = &SCRIPT_CLASSES[metrics.root.style_class.script.index()];

    let mut glyph_index = face.char_index(script_class.standard_char1);
    if glyph_index == 0 && script_class.standard_char2 != 0 {
        glyph_index = face.char_index(script_class.standard_char2);
        if glyph_index == 0 && script_class.standard_char3 != 0 {
            glyph_index = face.char_index(script_class.standard_char3);
        }
    }

    if glyph_index != 0
        && let Ok(slot) = face.load_glyph(glyph_index)
        && slot.outline.n_points > 0
    {
        let units_per_em = metrics.units_per_em;
        hints.units_per_em = units_per_em as u16;
        hints.scaler_flags = 0;
        hints.x_scale = 1 << 16;
        hints.y_scale = 1 << 16;
        hints.x_delta = 0;
        hints.y_delta = 0;
        hints.reload(&slot.outline);

        for dim in [Dimension::Hort, Dimension::Vert] {
            compute_segments(&mut hints, dim);
            link_segments(&mut hints, 0, &[], dim);

            let mut num_widths = 0usize;
            let segments = &hints.axis[dim.index()].segments;
            for (index, segment) in segments.iter().enumerate() {
                let Some(link) = segment.link else {
                    continue;
                };
                let Some(linked) = segments.get(link) else {
                    continue;
                };
                if linked.link == Some(index) && link > index {
                    let dist = (i32::from(segment.pos) - i32::from(linked.pos)).abs();
                    if num_widths < LATIN_MAX_WIDTHS {
                        metrics.axis[dim.index()].widths[num_widths].org = i64::from(dist);
                        num_widths += 1;
                    }
                }
            }

            let threshold = i64::from(metrics.units_per_em / 100);
            sort_and_quantize_widths(&mut num_widths, &mut metrics.axis[dim.index()].widths, threshold);
            metrics.axis[dim.index()].width_count = num_widths;
        }
    }

    let units_per_em = metrics.units_per_em;
    for dim in [Dimension::Hort, Dimension::Vert] {
        let axis = &mut metrics.axis[dim.index()];
        let stdw = if axis.width_count > 0 {
            axis.widths[0].org
        } else {
            latin_constant(units_per_em, 50)
        };
        axis.edge_distance_threshold = stdw / 5;
        axis.standard_width = stdw;
        axis.extra_light = false;
    }
}

/// `af_latin_metrics_init_blues` (`aflatin.c`): walks the blue zone
/// character strings of the style, measures the extremum of every
/// mapped glyph and stores the median of each zone in the vertical
/// axis.
///
/// Styles without a blue stringset (FreeType passes
/// `AF_Blue_Stringset` 0) never scan blue strings and return
/// immediately.
fn init_blues(metrics: &mut LatinMetrics, face: &mut dyn Face) {
    let Some(bss) = metrics.root.style_class.blue_stringset else {
        return;
    };
    let units_per_em = metrics.units_per_em;
    let mut flats = [0; BLUE_STRING_MAX_LEN];
    let mut rounds = [0; BLUE_STRING_MAX_LEN];

    let axis = &mut metrics.axis[Dimension::Vert.index()];

    for entry in BLUE_STRINGSETS.iter().skip(bss) {
        let Some(bstring) = entry.string else {
            break;
        };
        let properties = entry.properties;
        let is_top = latin_is_top_blue(properties);
        let is_neutral = latin_is_neutral_blue(properties);
        let is_long = latin_is_long_blue(properties);

        let mut num_flats = 0usize;
        let mut num_rounds = 0usize;

        for ch in BLUE_STRINGS[bstring as usize].chars() {
            let glyph_index = face.char_index(ch as u32);
            if glyph_index == 0 {
                continue;
            }
            let Ok(slot) = face.load_glyph(glyph_index) else {
                continue;
            };
            let outline = &slot.outline;
            if outline.n_points <= 2 {
                continue;
            }
            let Some((best_y, round)) = blue_extremum(outline, is_top, is_neutral, is_long, units_per_em)
            else {
                continue;
            };

            if round {
                if num_rounds < BLUE_STRING_MAX_LEN {
                    rounds[num_rounds] = best_y;
                    num_rounds += 1;
                }
            } else if num_flats < BLUE_STRING_MAX_LEN {
                flats[num_flats] = best_y;
                num_flats += 1;
            }
        }

        if num_flats == 0 && num_rounds == 0 {
            continue;
        }

        sort_positions(num_rounds, &mut rounds);
        sort_positions(num_flats, &mut flats);

        let index = axis.blue_count;
        if index >= BLUE_STRINGSET_MAX_LEN {
            break;
        }
        axis.blue_count += 1;
        let blue = &mut axis.blues[index];

        if num_flats == 0 {
            let median = rounds[num_rounds / 2];
            blue.r#ref.org = median;
            blue.shoot.org = median;
        } else if num_rounds == 0 {
            let median = flats[num_flats / 2];
            blue.r#ref.org = median;
            blue.shoot.org = median;
        } else {
            blue.r#ref.org = flats[num_flats / 2];
            blue.shoot.org = rounds[num_rounds / 2];
        }

        if blue.shoot.org != blue.r#ref.org {
            let reference = blue.r#ref.org;
            let shoot = blue.shoot.org;
            let over_ref = shoot > reference;
            if is_top != over_ref {
                let mean = shoot.wrapping_add(reference) / 2;
                blue.r#ref.org = mean;
                blue.shoot.org = mean;
            }
        }

        blue.flags = 0;
        if is_top {
            blue.flags |= LATIN_BLUE_TOP;
        }
        if is_neutral {
            blue.flags |= LATIN_BLUE_NEUTRAL;
        }
        if latin_is_x_height_blue(properties) {
            blue.flags |= LATIN_BLUE_ADJUSTMENT;
        }
    }
}

/// `af_latin_metrics_check_digits` (`aflatin.c`): records whether all
/// ASCII digits of the face share one advance width.  Digits that are
/// not mapped or cannot be loaded are ignored; with no digit at all
/// the answer stays "same width", as in C.
fn check_digits(metrics: &mut LatinMetrics, face: &mut dyn Face) {
    let mut started = false;
    let mut same_width = true;
    let mut old_advance: Pos = 0;

    for ch in 0x30u32..=0x39 {
        let glyph_index = face.char_index(ch);
        if glyph_index == 0 {
            continue;
        }
        let Ok(advance) = face.advance(glyph_index) else {
            continue;
        };
        if started {
            if advance != old_advance {
                same_width = false;
                break;
            }
        } else {
            old_advance = advance;
            started = true;
        }
    }

    metrics.root.digits_have_same_width = same_width;
}

/// `af_latin_metrics_init` (`aflatin.c`): initializes the global
/// metrics of a Latin style.
///
/// The face is switched to the Unicode charmap for the standard
/// character, blue zone and digit lookups and switched back
/// afterwards, exactly as FreeType does.
///
/// FreeType only reports failures of the final `FT_Set_Charmap`,
/// which it ignores; this port does the same, so no error can escape
/// today and the signature matches the writing system dispatch.
pub fn metrics_init(metrics: &mut LatinMetrics, face: &mut dyn Face) -> TtResult<()> {
    let old_charmap = face.charmap();
    metrics.units_per_em = u32::from(face.units_per_em());

    if face.select_charmap(Encoding::Unicode).is_ok() {
        init_widths(metrics, face);
        init_blues(metrics, face);
        check_digits(metrics, face);
    }

    let _ = face.set_charmap(old_charmap);
    Ok(())
}

/// `af_latin_metrics_scale_dim` (`aflatin.c`): scales the standard
/// widths, blue zones and scale fields of one dimension, correcting
/// the vertical scale so that the top of the small letters lands on
/// the pixel grid.
fn scale_dim(metrics: &mut LatinMetrics, scaler: &Scaler, dim: Dimension) {
    let (scale_in, delta_in) = if dim == Dimension::Hort {
        (scaler.x_scale, scaler.x_delta)
    } else {
        (scaler.y_scale, scaler.y_delta)
    };

    let (org_scale, org_delta) = {
        let axis = &metrics.axis[dim.index()];
        (axis.org_scale, axis.org_delta)
    };
    if org_scale == scale_in && org_delta == delta_in {
        return;
    }

    let mut scale = scale_in;

    let vert = &metrics.axis[Dimension::Vert.index()];
    let adjustment = vert
        .blues
        .iter()
        .take(vert.blue_count)
        .find(|blue| blue.flags & LATIN_BLUE_ADJUSTMENT != 0)
        .map(|blue| blue.shoot.org);
    if let Some(shoot_org) = adjustment {
        let scaled = mul_fix(shoot_org, scaler.y_scale);
        let limit = metrics.root.globals.increase_x_height();
        let mut threshold: Pos = 40;
        if limit != 0 && scaler.x_ppem <= limit && scaler.x_ppem >= PROP_INCREASE_X_HEIGHT_MIN {
            threshold = 52;
        }
        let fitted = (scaled + threshold) & !63;
        if scaled != fitted && dim == Dimension::Vert {
            scale = mul_div(scale, fitted, scaled);
        }
    }

    let axis = &mut metrics.axis[dim.index()];
    axis.org_scale = scale_in;
    axis.org_delta = delta_in;
    axis.scale = scale;
    axis.delta = delta_in;

    let root = &mut metrics.root.scaler;
    if dim == Dimension::Hort {
        root.x_scale = scale;
        root.x_delta = delta_in;
    } else {
        root.y_scale = scale;
        root.y_delta = delta_in;
    }

    let width_count = axis.width_count;
    for width in axis.widths.iter_mut().take(width_count) {
        width.cur = mul_fix(width.org, scale);
        width.fit = width.cur;
    }

    axis.extra_light = mul_fix(axis.standard_width, scale) < 32 + 8;

    if dim == Dimension::Vert {
        let blue_count = axis.blue_count;
        for blue in axis.blues.iter_mut().take(blue_count) {
            blue.r#ref.cur = mul_fix(blue.r#ref.org, scale).wrapping_add(delta_in);
            blue.r#ref.fit = blue.r#ref.cur;
            blue.shoot.cur = mul_fix(blue.shoot.org, scale).wrapping_add(delta_in);
            blue.shoot.fit = blue.shoot.cur;
            blue.flags &= !LATIN_BLUE_ACTIVE;

            let dist = mul_fix(blue.r#ref.org.wrapping_sub(blue.shoot.org), scale);
            if (-48..=48).contains(&dist) {
                let mut delta2 = dist;
                if dist < 0 {
                    delta2 = -delta2;
                }
                if delta2 < 32 {
                    delta2 = 0;
                } else if delta2 < 48 {
                    delta2 = 32;
                } else {
                    delta2 = 64;
                }
                if dist < 0 {
                    delta2 = -delta2;
                }

                blue.r#ref.fit = pix_round(blue.r#ref.cur);
                blue.shoot.fit = blue.r#ref.fit.wrapping_sub(delta2);
                blue.flags |= LATIN_BLUE_ACTIVE;
            }
        }
    }
}

/// `af_latin_metrics_scale` (`aflatin.c`): applies the target size
/// `scaler` to the global metrics of both dimensions.
pub fn metrics_scale(metrics: &mut LatinMetrics, scaler: &Scaler) {
    let root = &mut metrics.root.scaler;
    root.render_mode = scaler.render_mode;
    root.flags = scaler.flags;
    root.x_ppem = scaler.x_ppem;

    scale_dim(metrics, scaler, Dimension::Hort);
    scale_dim(metrics, scaler, Dimension::Vert);
}

/// `af_latin_hints_init` (`aflatin.c`): attaches `metrics` to the
/// hint record, restores the (possibly corrected) Latin scales and
/// derives the scaler and style flags from the render mode and the
/// face properties.
///
/// `face` provides the `FT_STYLE_FLAG_ITALIC` bit that FreeType
/// reads from `metrics->root.scaler.face`.
pub fn hints_init(hints: &mut GlyphHints, metrics: &StyleMetrics, face: &dyn Face) {
    hints.rescale(metrics);
    let Some(latin) = metrics.latin() else {
        return;
    };

    hints.x_scale = latin.axis[Dimension::Hort.index()].scale;
    hints.x_delta = latin.axis[Dimension::Hort.index()].delta;
    hints.y_scale = latin.axis[Dimension::Vert.index()].scale;
    hints.y_delta = latin.axis[Dimension::Vert.index()].delta;

    let mode = metrics.scaler().render_mode;
    let mut scaler_flags = hints.scaler_flags;
    let mut other_flags = 0u32;

    if mode == RenderMode::Mono || mode == RenderMode::Lcd {
        other_flags |= LATIN_HINTS_HORZ_SNAP;
    }
    if mode == RenderMode::Mono || mode == RenderMode::LcdV {
        other_flags |= LATIN_HINTS_VERT_SNAP;
    }
    if mode != RenderMode::Light {
        other_flags |= LATIN_HINTS_STEM_ADJUST;
    }
    if mode == RenderMode::Mono {
        other_flags |= LATIN_HINTS_MONO;
    }

    if mode == RenderMode::Light || face.style_flags() & STYLE_FLAG_ITALIC != 0 {
        scaler_flags |= SCALER_FLAG_NO_HORIZONTAL;
    }
    if !metrics.root().globals.warping() {
        scaler_flags |= SCALER_FLAG_NO_WARPER;
    }

    hints.scaler_flags = scaler_flags;
    hints.other_flags = other_flags;
}

/// `af_latin_hints_compute_edges` (`aflatin.c`): groups the segments
/// of one dimension into a position-sorted table of edges, then
/// derives each edge's round/serif flags and stem links from the
/// segments it contains.
///
/// Pass one inserts every segment into an edge whose position lies
/// within `edge_distance_threshold` (at most 1/4px), pass two stores
/// the owning edge back into every segment, and pass three computes
/// the edge properties.
///
/// # Porting note
///
/// FreeType reports `FT_Err_Out_Of_Memory` when the embedded edge
/// array overflows; the port's table is a `Vec`, so this function is
/// infallible.  FreeType dereferences `hints->metrics`
/// unconditionally; this port skips the analysis when the Latin
/// metrics are not attached, which only happens if [`hints_init`]
/// was never called.
pub fn compute_edges(hints: &mut GlyphHints, dim: Dimension) {
    let Some(edge_distance_threshold_org) = hints
        .metrics
        .as_ref()
        .and_then(StyleMetrics::latin)
        .map(|latin| latin.axis[dim.index()].edge_distance_threshold)
    else {
        return;
    };

    let scale = if dim == Dimension::Hort {
        hints.x_scale
    } else {
        hints.y_scale
    };

    // Segments shorter than one pixel are ignored to avoid many
    // problems with serif fonts; the threshold is computed in font
    // units for horizontal segments only.
    let segment_length_threshold = if dim == Dimension::Hort {
        div_fix(64, hints.y_scale)
    } else {
        0
    };

    // The edge distance threshold is at most 0.25px.
    let mut edge_distance_threshold = mul_fix(edge_distance_threshold_org, scale);
    if edge_distance_threshold > 64 / 4 {
        edge_distance_threshold = 64 / 4;
    }
    let edge_distance_threshold = div_fix(edge_distance_threshold, scale);

    let axis = &mut hints.axis[dim.index()];
    axis.edges.clear();

    // The segment table is moved out so that pass one can grow both
    // tables through a single `axis` borrow.
    let mut segments = core::mem::take(&mut axis.segments);

    // Pass one: collect the segments into position-sorted edges.
    let mut index = 0;
    while index < segments.len() {
        let seg_index = index;
        let seg = segments[seg_index];
        index += 1;

        if i64::from(seg.height) < segment_length_threshold {
            continue;
        }

        // A special case for serif edges: below 1.5 pixels we ignore
        // them.
        if seg.serif.is_some() && 2 * i64::from(seg.height) < 3 * segment_length_threshold {
            continue;
        }

        // Look for an edge that corresponds to the segment's position.
        let mut found = None;
        for (edge_index, edge) in axis.edges.iter().enumerate() {
            let mut dist = i64::from(seg.pos) - i64::from(edge.fpos);
            if dist < 0 {
                dist = -dist;
            }
            if dist < edge_distance_threshold && edge.dir == seg.dir {
                found = Some(edge_index);
                break;
            }
        }

        match found {
            // Insert a new edge; the table stays sorted by position.
            None => {
                let edge_index = axis.new_edge(seg.pos, seg.dir);
                let edge = &mut axis.edges[edge_index];
                edge.first = Some(seg_index);
                edge.last = Some(seg_index);
                edge.dir = seg.dir;
                edge.fpos = seg.pos;
                edge.opos = mul_fix(i64::from(seg.pos), scale);
                edge.pos = edge.opos;
                segments[seg_index].edge_next = Some(seg_index);
            }
            // Append the segment to the edge's circular list.
            Some(edge_index) => {
                let first = axis.edges[edge_index].first;
                let last = axis.edges[edge_index].last;
                segments[seg_index].edge_next = first;
                if let Some(last) = last {
                    segments[last].edge_next = Some(seg_index);
                }
                axis.edges[edge_index].last = Some(seg_index);
            }
        }
    }

    // Pass two: store the owning edge in every segment; this is
    // required in order to compute the edge links below.
    for (edge_index, edge) in axis.edges.iter().enumerate() {
        let Some(first) = edge.first else {
            continue;
        };
        let mut current = Some(first);
        let mut steps = 0usize;
        while let Some(seg_index) = current {
            let Some(segment) = segments.get_mut(seg_index) else {
                break;
            };
            segment.edge = Some(edge_index);
            current = segment.edge_next;
            if current == Some(first) {
                break;
            }
            // Defensive: a corrupted ring must not hang the analysis.
            steps += 1;
            if steps > segments.len() {
                break;
            }
        }
    }

    // Pass three: compute each edge's properties from its segments —
    // the round/serif flags plus link and serif edges.
    let num_edges = axis.edges.len();
    let mut edge_index = 0;
    while edge_index < num_edges {
        let mut is_round = 0i32;
        let mut is_straight = 0i32;

        let Some(first) = axis.edges[edge_index].first else {
            edge_index += 1;
            continue;
        };
        let mut current = Some(first);
        let mut steps = 0usize;
        while let Some(seg_index) = current {
            let Some(seg) = segments.get(seg_index).copied() else {
                break;
            };

            if seg.flags & EDGE_ROUND != 0 {
                is_round += 1;
            } else {
                is_straight += 1;
            }

            // If `seg.serif` is set, then `seg.link` must be ignored.
            let serif_edge = seg
                .serif
                .and_then(|serif| segments.get(serif))
                .and_then(|serif| serif.edge);
            let is_serif = serif_edge.is_some_and(|serif_edge| serif_edge != edge_index);
            let link_edge = seg
                .link
                .and_then(|link| segments.get(link))
                .and_then(|link| link.edge);

            if link_edge.is_some() || is_serif {
                let seg2_index = if is_serif { seg.serif } else { seg.link };
                let seg2 = seg2_index
                    .and_then(|seg2| segments.get(seg2))
                    .copied();
                let seg2_edge = seg2.and_then(|seg2| seg2.edge);
                let mut edge2 = if is_serif {
                    axis.edges[edge_index].serif
                } else {
                    axis.edges[edge_index].link
                };

                // Prefer the segment's own edge when the two segments
                // are closer to each other than the two edges are.
                if let Some(existing) = edge2 {
                    let mut edge_delta =
                        i64::from(axis.edges[edge_index].fpos) - i64::from(axis.edges[existing].fpos);
                    if edge_delta < 0 {
                        edge_delta = -edge_delta;
                    }
                    let seg_delta = seg2.map_or(0, |seg2| {
                        let mut delta = i64::from(seg.pos) - i64::from(seg2.pos);
                        if delta < 0 {
                            delta = -delta;
                        }
                        delta
                    });
                    if seg_delta < edge_delta {
                        edge2 = seg2_edge;
                    }
                } else {
                    edge2 = seg2_edge;
                }

                if is_serif {
                    axis.edges[edge_index].serif = edge2;
                    if let Some(edge2) = edge2 {
                        axis.edges[edge2].flags |= EDGE_SERIF;
                    }
                } else {
                    axis.edges[edge_index].link = edge2;
                }
            }

            current = seg.edge_next;
            if current == Some(first) {
                break;
            }
            // Defensive: a corrupted ring must not hang the analysis.
            steps += 1;
            if steps > segments.len() {
                break;
            }
        }

        // Set the round/straight flags.  The assignment also clears a
        // `AF_EDGE_SERIF` flag possibly inherited from an edge
        // processed earlier; FreeType behaves the same way.
        let mut flags = EDGE_NORMAL;
        if is_round > 0 && is_round >= is_straight {
            flags |= EDGE_ROUND;
        }
        axis.edges[edge_index].flags = flags;

        // Get rid of serifs if a link is set; this gets rid of many
        // unpleasant artefacts (FreeType cites the `c' in cour.pfa).
        if axis.edges[edge_index].serif.is_some() && axis.edges[edge_index].link.is_some() {
            axis.edges[edge_index].serif = None;
        }

        edge_index += 1;
    }

    axis.segments = segments;
}

/// `af_latin_hints_detect_features` (`aflatin.c`): computes the
/// segments, their stem links, and the edges of one dimension.
///
/// # Porting note
///
/// FreeType stops when the segment analysis reports an error; all
/// three steps are infallible in this port.
pub fn detect_features(hints: &mut GlyphHints, width_count: usize, widths: &[Width], dim: Dimension) {
    compute_segments(hints, dim);
    link_segments(hints, width_count, widths, dim);
    compute_edges(hints, dim);
}

/// `af_latin_hints_compute_blue_edges` (`aflatin.c`): assigns the
/// closest active blue zone of the vertical dimension to every
/// horizontal edge and flags edges aligned to a neutral zone.
///
/// # Porting note
///
/// FreeType passes the metrics as a second argument (obtained from
/// `hints->metrics` by the caller); the port reads them from the
/// hints directly and skips the analysis when the Latin metrics are
/// not attached.
pub fn compute_blue_edges(hints: &mut GlyphHints) {
    let Some((units_per_em, latin)) = hints
        .metrics
        .as_ref()
        .and_then(StyleMetrics::latin)
        .map(|metrics| (metrics.units_per_em, metrics.axis[Dimension::Vert.index()]))
    else {
        return;
    };
    let scale = latin.scale;

    let vert = &mut hints.axis[Dimension::Vert.index()];
    let major_dir = vert.major_dir;

    for edge in vert.edges.iter_mut() {
        // Initial threshold: a fraction of the EM size (40 is
        // heuristic), clamped to at least 0.5px.
        let mut best_dist = mul_fix(i64::from(units_per_em / 40), scale);
        if best_dist > 64 / 2 {
            best_dist = 64 / 2;
        }
        let mut best_blue: Option<Width> = None;
        let mut best_blue_is_neutral = false;

        for blue in latin.blues.iter().take(latin.blue_count) {
            // Skip inactive zones (those whose scaled size is too
            // large).
            if blue.flags & LATIN_BLUE_ACTIVE == 0 {
                continue;
            }

            // A top zone pairs with edges against the major
            // direction, a bottom zone with edges in the major
            // direction; neutral zones apply to both.
            let is_top_blue = blue.flags & LATIN_BLUE_TOP != 0;
            let is_neutral_blue = blue.flags & LATIN_BLUE_NEUTRAL != 0;
            let is_major_dir = edge.dir == major_dir;

            if is_top_blue != is_major_dir || is_neutral_blue {
                let mut dist = i64::from(edge.fpos) - blue.r#ref.org;
                if dist < 0 {
                    dist = -dist;
                }
                let dist = mul_fix(dist, scale);
                if dist < best_dist {
                    best_dist = dist;
                    best_blue = Some(blue.r#ref);
                    best_blue_is_neutral = is_neutral_blue;
                }

                // A rounded edge also compares against the overshoot
                // when it lies over the reference of a top zone or
                // under the reference of a bottom zone.
                if edge.flags & EDGE_ROUND != 0 && dist != 0 && !is_neutral_blue {
                    let is_under_ref = i64::from(edge.fpos) < blue.r#ref.org;
                    if is_top_blue != is_under_ref {
                        let mut dist = i64::from(edge.fpos) - blue.shoot.org;
                        if dist < 0 {
                            dist = -dist;
                        }
                        let dist = mul_fix(dist, scale);
                        if dist < best_dist {
                            best_dist = dist;
                            best_blue = Some(blue.shoot);
                            best_blue_is_neutral = is_neutral_blue;
                        }
                    }
                }
            }
        }

        if let Some(blue) = best_blue {
            edge.blue_edge = Some(blue);
            if best_blue_is_neutral {
                edge.flags |= EDGE_NEUTRAL;
            }
        }
    }
}

/// `AF_LATIN_HINTS_DO_HORZ_SNAP` (`aflatin.h`): true when horizontal
/// stem widths are snapped to the pixel grid.
#[inline]
pub fn hints_do_horz_snap(hints: &GlyphHints) -> bool {
    hints.test_other(LATIN_HINTS_HORZ_SNAP)
}

/// `AF_LATIN_HINTS_DO_VERT_SNAP` (`aflatin.h`): true when vertical
/// stem heights are snapped to the pixel grid.
#[inline]
pub fn hints_do_vert_snap(hints: &GlyphHints) -> bool {
    hints.test_other(LATIN_HINTS_VERT_SNAP)
}

/// `AF_LATIN_HINTS_DO_STEM_ADJUST` (`aflatin.h`): true when stem
/// widths and heights are adjusted at all.
#[inline]
pub fn hints_do_stem_adjust(hints: &GlyphHints) -> bool {
    hints.test_other(LATIN_HINTS_STEM_ADJUST)
}

/// `AF_LATIN_HINTS_DO_MONO` (`aflatin.h`): true for monochrome
/// rendering.
#[inline]
pub fn hints_do_mono(hints: &GlyphHints) -> bool {
    hints.test_other(LATIN_HINTS_MONO)
}

/// `af_latin_snap_width` (`aflatin.c`): moves `width` towards the
/// closest of the first `count` standard widths unless the difference
/// is more than 48/64 pixel.
///
/// # Porting note
///
/// FreeType passes the whole `widths` array together with its used
/// `count`; both arguments are kept here because the caller owns the
/// fixed size array of the metrics record.
#[inline]
pub fn snap_width(widths: &[Width], count: usize, width: Pos) -> Pos {
    let mut best: Pos = 64 + 32 + 2;
    let mut reference = width;

    for w in widths.iter().take(count) {
        let mut dist = width - w.cur;
        if dist < 0 {
            dist = -dist;
        }
        if dist < best {
            best = dist;
            reference = w.cur;
        }
    }

    let scaled = pix_round(reference);

    if width >= reference {
        if width < scaled + 48 {
            return reference;
        }
    } else if width > scaled - 48 {
        return reference;
    }

    width
}

/// `af_latin_compute_stem_width` (`aflatin.c`): computes the snapped
/// width of a given stem, ignoring very thin ones.
///
/// `base_flags` and `stem_flags` are the [`EDGE_*`] flags of the two
/// edges forming the stem; `width` is their original distance, which
/// may be negative for a reversed stem.
///
/// # Porting note
///
/// FreeType casts `hints->metrics` to `AF_LatinMetrics` unconditionally.
/// This port returns `width` unchanged when the attached metrics are not
/// Latin metrics, which cannot happen for the Latin writing system but
/// keeps the cast honest.
///
/// [`EDGE_*`]: crate::hints::EDGE_ROUND
pub fn compute_stem_width(
    hints: &GlyphHints,
    dim: Dimension,
    width: Pos,
    base_flags: u8,
    stem_flags: u8,
) -> Pos {
    let Some(axis) = hints
        .metrics
        .as_ref()
        .and_then(StyleMetrics::latin)
        .map(|metrics| &metrics.axis[dim.index()])
    else {
        return width;
    };

    if !hints_do_stem_adjust(hints) || axis.extra_light {
        return width;
    }

    let mut dist = width;
    let mut sign = false;
    if dist < 0 {
        dist = -width;
        sign = true;
    }

    let vertical = dim == Dimension::Vert;
    let snap = if vertical {
        hints_do_vert_snap(hints)
    } else {
        hints_do_horz_snap(hints)
    };

    let adjusted;
    if !snap {
        // Smooth hinting process: very lightly quantize the stem
        // width.  Leave the widths of serifs alone.
        if stem_flags & EDGE_SERIF != 0 && vertical && dist < 3 * 64 {
            adjusted = dist;
        } else {
            let mut dist = if base_flags & EDGE_ROUND != 0 {
                if dist < 80 { 64 } else { dist }
            } else if dist < 56 {
                56
            } else {
                dist
            };

            if axis.width_count > 0 {
                // Compare to the standard width.
                let mut delta = dist - axis.widths[0].cur;
                if delta < 0 {
                    delta = -delta;
                }

                if delta < 40 {
                    dist = axis.widths[0].cur;
                    if dist < 48 {
                        dist = 48;
                    }
                } else if dist < 3 * 64 {
                    let low = dist & 63;
                    dist &= -64;

                    if low < 10 {
                        dist += low;
                    } else if low < 32 {
                        dist += 10;
                    } else if low < 54 {
                        dist += 54;
                    } else {
                        dist += low;
                    }
                } else {
                    dist = (dist + 32) & !63;
                }
            }

            adjusted = dist;
        }
    } else {
        // Strong hinting process: snap the stem width to integer
        // pixels.
        let org_dist = dist;
        let dist = snap_width(&axis.widths, axis.width_count, dist);

        if vertical {
            // Vertical stem heights always round to integer pixels.
            if dist >= 64 {
                adjusted = (dist + 16) & !63;
            } else {
                adjusted = 64;
            }
        } else if hints_do_mono(hints) {
            // Monochrome horizontal hinting snaps widths to integer
            // pixels with a different threshold.
            if dist < 64 {
                adjusted = 64;
            } else {
                adjusted = (dist + 32) & !63;
            }
        } else if dist < 48 {
            // Strengthen small stems.
            adjusted = (dist + 64) >> 1;
        } else if dist < 128 {
            // Round stems of one to two pixels to an integer only if
            // the corresponding distortion is less than 1/4 pixel.
            let mut rounded = (dist + 22) & !63;
            let mut delta = rounded - org_dist;
            if delta < 0 {
                delta = -delta;
            }

            if delta >= 16 {
                let mut keep = org_dist;
                if keep < 48 {
                    keep = (keep + 64) >> 1;
                }
                rounded = keep;
            }
            adjusted = rounded;
        } else {
            // Round otherwise to prevent color fringes in LCD mode.
            adjusted = (dist + 32) & !63;
        }
    }

    if sign { -adjusted } else { adjusted }
}

/// `af_latin_align_linked_edge` (`aflatin.c`): aligns one stem edge
/// relative to the previous stem edge.
///
/// Out of range edge indices (possible for a corrupted edge table) are
/// ignored instead of triggering the assertion of the C build.
pub fn align_linked_edge(hints: &mut GlyphHints, dim: Dimension, base_index: usize, stem_index: usize) {
    let (Some(base), Some(stem)) = (
        hints.axis[dim.index()].edges.get(base_index),
        hints.axis[dim.index()].edges.get(stem_index),
    ) else {
        return;
    };

    let dist = stem.opos - base.opos;
    let fitted_width = compute_stem_width(hints, dim, dist, base.flags, stem.flags);
    let pos = base.pos + fitted_width;

    if let Some(stem) = hints.axis[dim.index()].edges.get_mut(stem_index) {
        stem.pos = pos;
    }
}

/// `af_latin_align_serif_edge` (`aflatin.c`): shifts the coordinates
/// of the `serif` edge by the same amount as the corresponding `base`
/// edge has been moved already.
///
/// # Porting note
///
/// FreeType ignores the hint record in this function (`FT_UNUSED`); the
/// port still takes it because the edge tables live inside it.
pub fn align_serif_edge(hints: &mut GlyphHints, dim: Dimension, base_index: usize, serif_index: usize) {
    let (Some(base), Some(serif)) = (
        hints.axis[dim.index()].edges.get(base_index),
        hints.axis[dim.index()].edges.get(serif_index),
    ) else {
        return;
    };

    let pos = base.pos + (serif.opos - base.opos);

    if let Some(serif) = hints.axis[dim.index()].edges.get_mut(serif_index) {
        serif.pos = pos;
    }
}

/// `af_latin_hint_edges` (`aflatin.c`): the main grid-fitting
/// routine.
///
/// All stems are first aligned relative to the blue zones (horizontal
/// dimension only), the remaining stems are then aligned relative to
/// each other, and finally the serifs and single edges are moved
/// along.
///
/// # Porting note
///
/// The C code indexes the edge table with pointers (`edge - edges`,
/// `edge[-1]`, `edge[1]`); this port uses indices and ignores the
/// positions of a stem whose edge index is out of range.  The trace
/// output of the C build is dropped.
pub fn hint_edges(hints: &mut GlyphHints, dim: Dimension) {
    let axis_index = dim.index();
    let num_edges = hints.axis[axis_index].num_edges();
    let mut anchor: Option<usize> = None;
    let mut has_serifs = 0usize;

    // Align all stems relative to the blue zones; this is only done
    // for horizontal edges.
    if dim == Dimension::Vert && do_blues() {
        for edge_index in 0..num_edges {
            if hints.axis[axis_index].edges[edge_index].flags & EDGE_DONE != 0 {
                continue;
            }

            let mut edge2 = hints.axis[axis_index].edges[edge_index]
                .link
                .filter(|index| *index < num_edges);

            // If a stem contains both a neutral and a non-neutral
            // blue zone, skip the neutral one.  If we have two
            // neutral blue zones, skip one of them.
            let both_blue = hints.axis[axis_index].edges[edge_index]
                .blue_edge
                .is_some()
                && edge2.is_some_and(|index| {
                    hints.axis[axis_index].edges[index]
                        .blue_edge
                        .is_some()
                });
            if both_blue {
                let neutral = hints.axis[axis_index].edges[edge_index].flags & EDGE_NEUTRAL != 0;
                let neutral2 =
                    edge2.is_some_and(|index| hints.axis[axis_index].edges[index].flags & EDGE_NEUTRAL != 0);

                if neutral2 {
                    if let Some(index) = edge2 {
                        hints.axis[axis_index].edges[index].blue_edge = None;
                        hints.axis[axis_index].edges[index].flags &= !EDGE_NEUTRAL;
                    }
                } else if neutral {
                    hints.axis[axis_index].edges[edge_index].blue_edge = None;
                    hints.axis[axis_index].edges[edge_index].flags &= !EDGE_NEUTRAL;
                }
            }

            // Prefer the edge that is aligned to a blue zone; flip
            // the pair when only the other edge has one.
            let mut edge1 = None;
            let mut blue = None;
            if let Some(value) = hints.axis[axis_index].edges[edge_index].blue_edge {
                blue = Some(value);
                edge1 = Some(edge_index);
            } else if let (Some(index), Some(value)) = (
                edge2,
                edge2.and_then(|index| {
                    hints.axis[axis_index]
                        .edges
                        .get(index)
                        .and_then(|edge| edge.blue_edge)
                }),
            ) {
                blue = Some(value);
                edge1 = Some(index);
                edge2 = Some(edge_index);
            }

            let (Some(edge1), Some(blue)) = (edge1, blue) else {
                continue;
            };

            hints.axis[axis_index].edges[edge1].pos = blue.fit;
            hints.axis[axis_index].edges[edge1].flags |= EDGE_DONE;

            if let Some(edge2) = edge2
                && hints.axis[axis_index].edges[edge2]
                    .blue_edge
                    .is_none()
            {
                align_linked_edge(hints, dim, edge1, edge2);
                hints.axis[axis_index].edges[edge2].flags |= EDGE_DONE;
            }

            if anchor.is_none() {
                anchor = Some(edge_index);
            }
        }
    }

    // Align all other stem edges, trying to maintain the relative
    // order of stems in the glyph.
    for edge_index in 0..num_edges {
        if hints.axis[axis_index].edges[edge_index].flags & EDGE_DONE != 0 {
            continue;
        }

        // Skip all non-stem edges.
        let Some(edge2) = hints.axis[axis_index].edges[edge_index]
            .link
            .filter(|index| *index < num_edges)
        else {
            has_serifs += 1;
            continue;
        };

        // This should not happen, but it's better to be safe.
        if hints.axis[axis_index].edges[edge2]
            .blue_edge
            .is_some()
        {
            align_linked_edge(hints, dim, edge2, edge_index);
            hints.axis[axis_index].edges[edge_index].flags |= EDGE_DONE;
            continue;
        }

        match anchor {
            // If we reach this arm, no stem has been aligned yet.
            None => {
                let org_len =
                    hints.axis[axis_index].edges[edge2].opos - hints.axis[axis_index].edges[edge_index].opos;
                let cur_len = compute_stem_width(
                    hints,
                    dim,
                    org_len,
                    hints.axis[axis_index].edges[edge_index].flags,
                    hints.axis[axis_index].edges[edge2].flags,
                );

                // Some voodoo to specially round edges for small
                // stem widths: align the center of the stem, then
                // shift the stem edges to suitable positions.
                let (u_off, d_off) = if cur_len <= 64 { (32, 32) } else { (38, 26) };

                if cur_len < 96 {
                    let org_center = hints.axis[axis_index].edges[edge_index].opos + (org_len >> 1);
                    let mut cur_pos1 = pix_round(org_center);

                    let mut error1 = org_center - (cur_pos1 - u_off);
                    if error1 < 0 {
                        error1 = -error1;
                    }
                    let mut error2 = org_center - (cur_pos1 + d_off);
                    if error2 < 0 {
                        error2 = -error2;
                    }

                    if error1 < error2 {
                        cur_pos1 -= u_off;
                    } else {
                        cur_pos1 += d_off;
                    }

                    let pos = cur_pos1 - cur_len / 2;
                    hints.axis[axis_index].edges[edge_index].pos = pos;
                    hints.axis[axis_index].edges[edge2].pos = pos + cur_len;
                } else {
                    hints.axis[axis_index].edges[edge_index].pos =
                        pix_round(hints.axis[axis_index].edges[edge_index].opos);
                }

                anchor = Some(edge_index);
                hints.axis[axis_index].edges[edge_index].flags |= EDGE_DONE;
                align_linked_edge(hints, dim, edge_index, edge2);
            }
            Some(anchor_index) => {
                let org_pos = hints.axis[axis_index].edges[anchor_index].pos
                    + (hints.axis[axis_index].edges[edge_index].opos
                        - hints.axis[axis_index].edges[anchor_index].opos);
                let org_len =
                    hints.axis[axis_index].edges[edge2].opos - hints.axis[axis_index].edges[edge_index].opos;
                let org_center = org_pos + (org_len >> 1);

                let cur_len = compute_stem_width(
                    hints,
                    dim,
                    org_len,
                    hints.axis[axis_index].edges[edge_index].flags,
                    hints.axis[axis_index].edges[edge2].flags,
                );

                if hints.axis[axis_index].edges[edge2].flags & EDGE_DONE != 0 {
                    // The stem's other edge has already been moved
                    // by an earlier anchor: follow it.
                    hints.axis[axis_index].edges[edge_index].pos =
                        hints.axis[axis_index].edges[edge2].pos - cur_len;
                } else if cur_len < 96 {
                    let mut cur_pos1 = pix_round(org_center);
                    let (u_off, d_off) = if cur_len <= 64 { (32, 32) } else { (38, 26) };

                    let mut delta1 = org_center - (cur_pos1 - u_off);
                    if delta1 < 0 {
                        delta1 = -delta1;
                    }
                    let mut delta2 = org_center - (cur_pos1 + d_off);
                    if delta2 < 0 {
                        delta2 = -delta2;
                    }

                    if delta1 < delta2 {
                        cur_pos1 -= u_off;
                    } else {
                        cur_pos1 += d_off;
                    }

                    hints.axis[axis_index].edges[edge_index].pos = cur_pos1 - cur_len / 2;
                    hints.axis[axis_index].edges[edge2].pos = cur_pos1 + cur_len / 2;
                } else {
                    let cur_pos1 = pix_round(org_pos);
                    let mut delta1 = cur_pos1 + (cur_len >> 1) - org_center;
                    if delta1 < 0 {
                        delta1 = -delta1;
                    }

                    let cur_pos2 = pix_round(org_pos + org_len) - cur_len;
                    let mut delta2 = cur_pos2 + (cur_len >> 1) - org_center;
                    if delta2 < 0 {
                        delta2 = -delta2;
                    }

                    let pos = if delta1 < delta2 { cur_pos1 } else { cur_pos2 };
                    hints.axis[axis_index].edges[edge_index].pos = pos;
                    hints.axis[axis_index].edges[edge2].pos = pos + cur_len;
                }

                hints.axis[axis_index].edges[edge_index].flags |= EDGE_DONE;
                hints.axis[axis_index].edges[edge2].flags |= EDGE_DONE;

                if edge_index > 0
                    && hints.axis[axis_index].edges[edge_index].pos
                        < hints.axis[axis_index].edges[edge_index - 1].pos
                {
                    hints.axis[axis_index].edges[edge_index].pos =
                        hints.axis[axis_index].edges[edge_index - 1].pos;
                }
            }
        }
    }

    // Make sure that lowercase m's maintain their symmetry; the
    // assumption is six vertical edges for a sans serif m and twelve
    // for a serif one.  Horizontal edges are not handled since the
    // third stem cannot easily be aligned with the base line.
    if dim == Dimension::Hort && (num_edges == 6 || num_edges == 12) {
        let (index1, index2, index3) = if num_edges == 6 { (0, 2, 4) } else { (1, 5, 9) };

        let dist1 = hints.axis[axis_index].edges[index2].opos - hints.axis[axis_index].edges[index1].opos;
        let dist2 = hints.axis[axis_index].edges[index3].opos - hints.axis[axis_index].edges[index2].opos;

        let mut span = dist1 - dist2;
        if span < 0 {
            span = -span;
        }

        if span < 8 {
            let delta = hints.axis[axis_index].edges[index3].pos
                - (2 * hints.axis[axis_index].edges[index2].pos - hints.axis[axis_index].edges[index1].pos);

            hints.axis[axis_index].edges[index3].pos -= delta;
            if let Some(link) = hints.axis[axis_index].edges[index3]
                .link
                .filter(|index| *index < num_edges)
            {
                hints.axis[axis_index].edges[link].pos -= delta;
            }

            // Move the serifs along with the stem.
            if num_edges == 12 {
                hints.axis[axis_index].edges[8].pos -= delta;
                hints.axis[axis_index].edges[11].pos -= delta;
            }

            hints.axis[axis_index].edges[index3].flags |= EDGE_DONE;
            if let Some(link) = hints.axis[axis_index].edges[index3]
                .link
                .filter(|index| *index < num_edges)
            {
                hints.axis[axis_index].edges[link].flags |= EDGE_DONE;
            }
        }
    }

    if has_serifs > 0 || anchor.is_none() {
        // Now hint the remaining edges (serifs and single ones) in
        // order to complete our processing.
        for edge_index in 0..num_edges {
            if hints.axis[axis_index].edges[edge_index].flags & EDGE_DONE != 0 {
                continue;
            }

            let serif = hints.axis[axis_index].edges[edge_index]
                .serif
                .filter(|index| *index < num_edges);
            let mut delta = 1000;
            if let Some(serif_index) = serif {
                delta = hints.axis[axis_index].edges[serif_index].opos
                    - hints.axis[axis_index].edges[edge_index].opos;
                if delta < 0 {
                    delta = -delta;
                }
            }

            if delta < 64 + 16 {
                if let Some(serif_index) = serif {
                    align_serif_edge(hints, dim, serif_index, edge_index);
                }
            } else if anchor.is_none() {
                hints.axis[axis_index].edges[edge_index].pos =
                    pix_round(hints.axis[axis_index].edges[edge_index].opos);
                anchor = Some(edge_index);
            } else {
                let anchor_index = match anchor {
                    Some(anchor_index) => anchor_index,
                    None => continue,
                };

                // Look for the closest hinted edges before and
                // after the current one.
                let mut before = None;
                for index in (0..edge_index).rev() {
                    if hints.axis[axis_index].edges[index].flags & EDGE_DONE != 0 {
                        before = Some(index);
                        break;
                    }
                }
                let mut after = None;
                for index in (edge_index + 1)..num_edges {
                    if hints.axis[axis_index].edges[index].flags & EDGE_DONE != 0 {
                        after = Some(index);
                        break;
                    }
                }

                if let (Some(before), Some(after)) = (before, after) {
                    if hints.axis[axis_index].edges[after].opos == hints.axis[axis_index].edges[before].opos {
                        hints.axis[axis_index].edges[edge_index].pos =
                            hints.axis[axis_index].edges[before].pos;
                    } else {
                        hints.axis[axis_index].edges[edge_index].pos = hints.axis[axis_index].edges[before]
                            .pos
                            + mul_div(
                                hints.axis[axis_index].edges[edge_index].opos
                                    - hints.axis[axis_index].edges[before].opos,
                                hints.axis[axis_index].edges[after].pos
                                    - hints.axis[axis_index].edges[before].pos,
                                hints.axis[axis_index].edges[after].opos
                                    - hints.axis[axis_index].edges[before].opos,
                            );
                    }
                } else {
                    let pos = hints.axis[axis_index].edges[anchor_index].pos
                        + ((hints.axis[axis_index].edges[edge_index].opos
                            - hints.axis[axis_index].edges[anchor_index].opos
                            + 16)
                            & !31);
                    hints.axis[axis_index].edges[edge_index].pos = pos;
                }
            }

            hints.axis[axis_index].edges[edge_index].flags |= EDGE_DONE;

            if edge_index > 0
                && hints.axis[axis_index].edges[edge_index].pos
                    < hints.axis[axis_index].edges[edge_index - 1].pos
            {
                hints.axis[axis_index].edges[edge_index].pos =
                    hints.axis[axis_index].edges[edge_index - 1].pos;
            }

            if let Some(next) = hints.axis[axis_index]
                .edges
                .get(edge_index + 1)
                .filter(|edge| edge.flags & EDGE_DONE != 0)
            {
                let pos = next.pos;
                if hints.axis[axis_index].edges[edge_index].pos > pos {
                    hints.axis[axis_index].edges[edge_index].pos = pos;
                }
            }
        }
    }
}

/// `af_latin_hints_apply` (`aflatin.c`): applies the complete
/// hinting algorithm to a Latin glyph.
///
/// # Porting note
///
/// FreeType's reload step reports `FT_Outline_Too_Small` or an
/// allocation error; [`GlyphHints::reload`] validates the outline
/// and clears the analysis instead, so this function cannot fail and
/// always returns `Ok(())`.  When `metrics` carries no Latin metrics
/// the hinting is skipped, which the C cast would simply misread.
pub fn hints_apply(hints: &mut GlyphHints, outline: &mut Outline, metrics: &StyleMetrics) -> TtResult<()> {
    hints.reload(outline);

    let Some(latin) = metrics.latin() else {
        return Ok(());
    };
    let render_mode = metrics.scaler().render_mode;

    // Analyze the glyph outline.
    if (render_mode == RenderMode::Light && hints.do_warp()) || hints.do_horizontal() {
        let axis = &latin.axis[Dimension::Hort.index()];
        detect_features(hints, axis.width_count, &axis.widths, Dimension::Hort);
    }

    if hints.do_vertical() {
        let axis = &latin.axis[Dimension::Vert.index()];
        detect_features(hints, axis.width_count, &axis.widths, Dimension::Vert);
        compute_blue_edges(hints);
    }

    // Grid-fit the outline.
    for dim in [Dimension::Hort, Dimension::Vert] {
        if dim == Dimension::Hort && render_mode == RenderMode::Light && hints.do_warp() {
            let (scale, delta) = crate::warp::warper_compute(hints, dim);
            hints.scale_dim(dim, scale, delta);
            continue;
        }

        let enabled = if dim == Dimension::Hort {
            hints.do_horizontal()
        } else {
            hints.do_vertical()
        };
        if enabled {
            hint_edges(hints, dim);
            hints.align_edge_points(dim);
            hints.align_strong_points(dim);
            hints.align_weak_points(dim);
        }
    }

    hints.save(outline);

    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use codevar_truetype_core::Vector;

    use super::*;
    use crate::face::{GlyphSlot, MockFace};
    use crate::hints::{Edge, Segment};
    use crate::metrics::GlobalsShared;
    use crate::ranges::{STYLE_CLASSES, Style};
    use crate::{BLUE_PROPERTY_LATIN_LONG, BLUE_PROPERTY_LATIN_TOP};

    /// A counter-clockwise (PostScript) rectangle from `(0, 0)` to
    /// `(500, 700)` with every point on the curve.
    fn rect_outline() -> Outline {
        let mut outline = Outline::with_capacity(4, 1);
        outline.points = vec![
            Vector::new(0, 0),
            Vector::new(500, 0),
            Vector::new(500, 700),
            Vector::new(0, 700),
        ];
        outline.tags = vec![CURVE_TAG_ON; 4];
        outline.contours = vec![3];
        outline.n_points = 4;
        outline.n_contours = 1;
        outline
    }

    /// The rectangle as a glyph slot of a face with `upem` font units.
    fn rect_slot() -> GlyphSlot {
        GlyphSlot {
            outline: rect_outline(),
            ..GlyphSlot::default()
        }
    }

    /// Latin metrics of a face with the given `units_per_EM`.
    fn latin_metrics(units_per_em: u32) -> LatinMetrics {
        let globals = Rc::new(GlobalsShared::new(units_per_em as u16));
        LatinMetrics {
            root: StyleMetricsRec::new(&STYLE_CLASSES[Style::LatnDflt.index()], globals),
            units_per_em,
            axis: Default::default(),
        }
    }

    /// Hint record reloaded from [`rect_outline`] at identity scale,
    /// with the Latin metrics attached as after [`hints_init`].
    fn rect_hints(units_per_em: u16) -> GlyphHints {
        let metrics = StyleMetrics::new(
            &STYLE_CLASSES[Style::LatnDflt.index()],
            Rc::new(GlobalsShared::new(units_per_em)),
        );
        let mut hints = GlyphHints::default();
        hints.rescale(&metrics);
        hints.x_scale = 1 << 16;
        hints.y_scale = 1 << 16;
        hints.reload(&rect_outline());
        hints.metrics = Some(metrics);
        hints
    }

    /// Hints with one vertical edge at `fpos` and a single vertical
    /// blue zone at reference 700 with shoot 716.
    fn blue_hints(fpos: i16, dir: Direction, flags: u8, blue_flags: u32) -> GlyphHints {
        let mut hints = rect_hints(1000);
        if let Some(StyleMetrics::Latin(latin)) = &mut hints.metrics {
            // Set by `metrics_init` in the production flow.
            latin.units_per_em = 1000;
            let vert = &mut latin.axis[Dimension::Vert.index()];
            vert.scale = 1 << 16;
            vert.blue_count = 1;
            vert.blues[0] = LatinBlue {
                r#ref: Width {
                    org: 700,
                    cur: 0,
                    fit: 0,
                },
                shoot: Width {
                    org: 716,
                    cur: 0,
                    fit: 0,
                },
                flags: blue_flags,
            };
        }
        let vert = &mut hints.axis[Dimension::Vert.index()];
        vert.edges = vec![Edge {
            fpos,
            dir,
            flags,
            ..Edge::default()
        }];
        hints
    }

    #[test]
    fn blue_properties_are_distinct_bits() {
        assert_eq!(crate::BLUE_PROPERTY_LATIN_TOP, 1);
        assert_eq!(crate::BLUE_PROPERTY_LATIN_NEUTRAL, 2);
        assert_eq!(crate::BLUE_PROPERTY_LATIN_X_HEIGHT, 4);
        assert_eq!(BLUE_PROPERTY_LATIN_LONG, 8);
    }

    #[test]
    fn latin_blue_predicates_test_their_own_bit() {
        assert!(latin_is_top_blue(BLUE_PROPERTY_LATIN_TOP));
        assert!(!latin_is_top_blue(0));
        assert!(!latin_is_top_blue(crate::BLUE_PROPERTY_LATIN_X_HEIGHT));

        assert!(latin_is_neutral_blue(crate::BLUE_PROPERTY_LATIN_NEUTRAL));
        assert!(!latin_is_neutral_blue(BLUE_PROPERTY_LATIN_TOP));

        assert!(latin_is_x_height_blue(crate::BLUE_PROPERTY_LATIN_X_HEIGHT));
        assert!(!latin_is_x_height_blue(crate::BLUE_PROPERTY_LATIN_NEUTRAL));

        assert!(latin_is_long_blue(BLUE_PROPERTY_LATIN_LONG));
        assert!(!latin_is_long_blue(crate::BLUE_PROPERTY_LATIN_NEUTRAL));

        let combined = crate::BLUE_PROPERTY_LATIN_TOP | crate::BLUE_PROPERTY_LATIN_X_HEIGHT;
        assert!(latin_is_top_blue(combined));
        assert!(latin_is_x_height_blue(combined));
        assert!(!latin_is_long_blue(combined));
    }

    #[test]
    fn flag_bits_are_distinct_powers_of_two() {
        for (flag, shift) in [
            (LATIN_BLUE_ACTIVE, 0),
            (LATIN_BLUE_TOP, 1),
            (LATIN_BLUE_NEUTRAL, 2),
            (LATIN_BLUE_ADJUSTMENT, 3),
        ] {
            assert_eq!(flag, 1 << shift);
        }
        for (flag, shift) in [
            (LATIN_HINTS_HORZ_SNAP, 0),
            (LATIN_HINTS_VERT_SNAP, 1),
            (LATIN_HINTS_STEM_ADJUST, 2),
            (LATIN_HINTS_MONO, 3),
        ] {
            assert_eq!(flag, 1 << shift);
        }
    }

    #[test]
    fn axis_default_is_zeroed_with_full_blue_slots() {
        let axis = LatinAxis::default();
        assert_eq!(axis.scale, 0);
        assert_eq!(axis.delta, 0);
        assert_eq!(axis.width_count, 0);
        assert_eq!(axis.standard_width, 0);
        assert!(!axis.extra_light);
        assert_eq!(axis.blue_count, 0);
        assert_eq!(axis.widths.len(), LATIN_MAX_WIDTHS);
        assert_eq!(axis.blues.len(), BLUE_STRINGSET_MAX_LEN);
        assert!(axis.widths.iter().all(|w| *w == Width::default()));
        assert!(
            axis.blues
                .iter()
                .all(|blue| *blue == LatinBlue::default())
        );
    }

    #[test]
    fn latin_constant_scales_the_heuristic_constants() {
        assert_eq!(latin_constant(2048, 8), 8);
        assert_eq!(latin_constant(1000, 8), 3);
        assert_eq!(latin_constant(1000, 6000), 2929);
        assert_eq!(latin_constant(1000, 50), 24);
        assert_eq!(latin_constant(0, 6000), 0);
    }

    #[test]
    fn compute_segments_and_link_segments_analyze_a_rectangle() {
        let mut hints = rect_hints(1000);

        // PostScript orientation of the rectangle selects the
        // downward/rightward major directions.
        assert_eq!(hints.axis[Dimension::Hort.index()].major_dir, Direction::Down);
        assert_eq!(hints.axis[Dimension::Vert.index()].major_dir, Direction::Right);

        compute_segments(&mut hints, Dimension::Hort);
        let segments = &hints.axis[Dimension::Hort.index()].segments;
        assert_eq!(segments.len(), 2);
        let (up, down) = (&segments[0], &segments[1]);
        assert_eq!(up.dir, Direction::Up);
        assert_eq!(up.pos, 500);
        assert_eq!((up.min_coord, up.max_coord), (0, 700));
        assert_eq!(up.height, 700);
        assert_eq!((up.first, up.last), (Some(1), Some(2)));
        assert_eq!(up.flags, EDGE_NORMAL);
        assert_eq!(up.score, 32000, "scoring happens in the link pass");
        assert_eq!(down.dir, Direction::Down);
        assert_eq!(down.pos, 0);
        assert_eq!((down.min_coord, down.max_coord), (0, 700));
        assert_eq!(down.height, 700);
        assert_eq!((down.first, down.last), (Some(3), Some(0)));

        link_segments(&mut hints, 0, &[], Dimension::Hort);
        let segments = &hints.axis[Dimension::Hort.index()].segments;
        // 500 units apart, 700 tall: dist demerit 500 plus
        // len_score 2929 / 700 = 4.
        assert_eq!(segments[0].score, 504);
        assert_eq!(segments[0].link, Some(1));
        assert_eq!(segments[0].serif, None);
        assert_eq!(segments[1].score, 504);
        assert_eq!(segments[1].link, Some(0));
        assert_eq!(segments[1].serif, None);

        compute_segments(&mut hints, Dimension::Vert);
        let segments = &hints.axis[Dimension::Vert.index()].segments;
        assert_eq!(segments.len(), 2);
        let (right, left) = (&segments[0], &segments[1]);
        assert_eq!(right.dir, Direction::Right);
        assert_eq!(right.pos, 0);
        assert_eq!((right.min_coord, right.max_coord), (0, 500));
        assert_eq!(right.height, 500);
        assert_eq!((right.first, right.last), (Some(0), Some(1)));
        assert_eq!(left.dir, Direction::Left);
        assert_eq!(left.pos, 700);
        assert_eq!((left.min_coord, left.max_coord), (0, 500));
        assert_eq!(left.height, 500);
        assert_eq!((left.first, left.last), (Some(2), Some(3)));

        link_segments(&mut hints, 0, &[], Dimension::Vert);
        let segments = &hints.axis[Dimension::Vert.index()].segments;
        // 700 units apart, 500 tall: 700 + 2929 / 500 = 705.
        assert_eq!(segments[0].score, 705);
        assert_eq!(segments[0].link, Some(1));
        assert_eq!(segments[1].score, 705);
        assert_eq!(segments[1].link, Some(0));
    }

    #[test]
    fn detect_features_builds_segments_links_and_edges_for_the_rectangle() {
        let mut hints = rect_hints(1000);
        if let Some(StyleMetrics::Latin(latin)) = &mut hints.metrics {
            // Effective threshold: 60 font units clamped to 0.25px
            // (16 font units at identity scale); the rectangle's
            // segments are far apart, so nothing merges.
            latin.axis[Dimension::Hort.index()].edge_distance_threshold = 60;
            latin.axis[Dimension::Vert.index()].edge_distance_threshold = 60;
        }

        detect_features(&mut hints, 0, &[], Dimension::Hort);
        let axis = &hints.axis[Dimension::Hort.index()];
        assert_eq!(axis.segments.len(), 2);
        assert_eq!(axis.segments[0].edge, Some(1));
        assert_eq!(axis.segments[0].edge_next, Some(0));
        assert_eq!(axis.segments[1].edge, Some(0));
        assert_eq!(axis.segments[1].edge_next, Some(1));
        assert_eq!(axis.edges.len(), 2);
        let (zero, five) = (&axis.edges[0], &axis.edges[1]);
        assert_eq!((zero.fpos, five.fpos), (0, 500));
        assert_eq!((zero.dir, five.dir), (Direction::Down, Direction::Up));
        assert_eq!((zero.opos, five.opos), (0, 500));
        assert_eq!((zero.pos, five.pos), (0, 500));
        assert_eq!((zero.first, zero.last), (Some(1), Some(1)));
        assert_eq!((five.first, five.last), (Some(0), Some(0)));
        // The linked segments pair the two edges.
        assert_eq!(zero.link, Some(1));
        assert_eq!(five.link, Some(0));
        assert_eq!(zero.serif, None);
        assert_eq!(five.serif, None);
        assert_eq!(zero.flags, EDGE_NORMAL);
        assert_eq!(five.flags, EDGE_NORMAL);
        assert_eq!(zero.blue_edge, None);

        detect_features(&mut hints, 0, &[], Dimension::Vert);
        let axis = &hints.axis[Dimension::Vert.index()];
        assert_eq!(axis.segments.len(), 2);
        assert_eq!(axis.segments[0].edge, Some(0));
        assert_eq!(axis.segments[1].edge, Some(1));
        assert_eq!(axis.edges.len(), 2);
        assert_eq!((axis.edges[0].fpos, axis.edges[1].fpos), (0, 700));
        assert_eq!(
            (axis.edges[0].dir, axis.edges[1].dir),
            (Direction::Right, Direction::Left)
        );
        assert_eq!((axis.edges[0].opos, axis.edges[1].opos), (0, 700));
        assert_eq!((axis.edges[0].first, axis.edges[0].last), (Some(0), Some(0)));
        assert_eq!((axis.edges[1].first, axis.edges[1].last), (Some(1), Some(1)));
        assert_eq!(axis.edges[0].link, Some(1));
        assert_eq!(axis.edges[1].link, Some(0));
        assert_eq!(axis.edges[0].flags, EDGE_NORMAL);
    }

    #[test]
    fn compute_edges_merges_nearby_segments_and_skips_small_serifs() {
        let mut hints = rect_hints(1000);
        if let Some(StyleMetrics::Latin(latin)) = &mut hints.metrics {
            latin.axis[Dimension::Hort.index()].edge_distance_threshold = 60;
        }
        {
            let axis = &mut hints.axis[Dimension::Hort.index()];
            axis.segments = vec![
                Segment {
                    dir: Direction::Down,
                    pos: 10,
                    height: 100,
                    ..Segment::default()
                },
                // 10 units away: merged into the first edge (10 < 16).
                Segment {
                    dir: Direction::Down,
                    pos: 20,
                    height: 100,
                    ..Segment::default()
                },
                // Serif segment below 1.5px: ignored (2 * 80 < 3 * 64).
                Segment {
                    dir: Direction::Down,
                    pos: 30,
                    height: 80,
                    serif: Some(0),
                    ..Segment::default()
                },
                // Other direction: gets its own edge despite proximity.
                Segment {
                    dir: Direction::Up,
                    pos: 12,
                    height: 100,
                    ..Segment::default()
                },
            ];
        }

        compute_edges(&mut hints, Dimension::Hort);
        let axis = &hints.axis[Dimension::Hort.index()];
        assert_eq!(axis.edges.len(), 2);
        let (first, second) = (&axis.edges[0], &axis.edges[1]);
        assert_eq!((first.fpos, first.dir), (10, Direction::Down));
        assert_eq!((first.first, first.last), (Some(0), Some(1)));
        assert_eq!(first.opos, 10);
        assert_eq!((second.fpos, second.dir), (12, Direction::Up));
        assert_eq!((second.first, second.last), (Some(3), Some(3)));
        assert_eq!(first.link, None, "the segments carry no links");
        assert_eq!(second.link, None);
        assert_eq!(first.flags, EDGE_NORMAL);
        assert_eq!(axis.segments[0].edge, Some(0));
        assert_eq!(axis.segments[1].edge, Some(0));
        assert_eq!(axis.segments[0].edge_next, Some(1));
        assert_eq!(axis.segments[1].edge_next, Some(0));
        assert_eq!(axis.segments[2].edge, None, "the small serif is ignored");
        assert_eq!(axis.segments[3].edge, Some(1));
        assert_eq!(axis.segments[3].edge_next, Some(3));
    }

    #[test]
    fn compute_blue_edges_assigns_the_closest_active_blue_zone() {
        let top = LATIN_BLUE_ACTIVE | LATIN_BLUE_TOP;

        // Against the major direction the edge pairs with the top
        // zone's reference; in the major direction the top zone does
        // not apply.
        let mut hints = blue_hints(700, Direction::Left, EDGE_NORMAL, top);
        compute_blue_edges(&mut hints);
        let edges = &hints.axis[Dimension::Vert.index()].edges;
        assert_eq!(
            edges[0].blue_edge,
            Some(Width {
                org: 700,
                cur: 0,
                fit: 0
            })
        );
        assert_eq!(edges[0].flags, EDGE_NORMAL);

        let mut hints = blue_hints(700, Direction::Right, EDGE_NORMAL, top);
        compute_blue_edges(&mut hints);
        let edges = &hints.axis[Dimension::Vert.index()].edges;
        assert_eq!(edges[0].blue_edge, None);

        // A rounded edge over the reference also compares against the
        // overshoot: 714 is 14 units past the reference but only 2
        // units short of the shoot (716).
        let mut hints = blue_hints(714, Direction::Left, EDGE_ROUND, top);
        compute_blue_edges(&mut hints);
        let edges = &hints.axis[Dimension::Vert.index()].edges;
        assert_eq!(
            edges[0].blue_edge,
            Some(Width {
                org: 716,
                cur: 0,
                fit: 0
            })
        );

        // Neutral zones apply to both directions and flag the edge.
        let neutral = LATIN_BLUE_ACTIVE | LATIN_BLUE_NEUTRAL;
        let mut hints = blue_hints(700, Direction::Right, EDGE_NORMAL, neutral);
        compute_blue_edges(&mut hints);
        let edges = &hints.axis[Dimension::Vert.index()].edges;
        assert_eq!(
            edges[0].blue_edge,
            Some(Width {
                org: 700,
                cur: 0,
                fit: 0
            })
        );
        assert_eq!(edges[0].flags, EDGE_NORMAL | EDGE_NEUTRAL);

        // Zones without `AF_LATIN_BLUE_ACTIVE` are skipped.
        let mut hints = blue_hints(700, Direction::Left, EDGE_NORMAL, LATIN_BLUE_TOP);
        compute_blue_edges(&mut hints);
        let edges = &hints.axis[Dimension::Vert.index()].edges;
        assert_eq!(edges[0].blue_edge, None);
    }

    #[test]
    fn metrics_init_measures_widths_and_blues_from_the_standard_character() {
        let mut face = MockFace::new()
            .with_units_per_em(1000)
            .with_charmap(Encoding::Unicode, &[(0x6F, 1)])
            .with_charmap(Encoding::AppleRoman, &[(0x6F, 1)])
            .select(Encoding::AppleRoman)
            .with_glyph(1, rect_slot());
        let mut metrics = latin_metrics(1000);

        metrics_init(&mut metrics, &mut face).expect("the Latin metrics initializer cannot fail today");

        assert_eq!(metrics.units_per_em, 1000);
        assert_eq!(
            face.charmap(),
            Some(Encoding::AppleRoman),
            "the charmap selected before the scan is restored"
        );

        let horz = &metrics.axis[Dimension::Hort.index()];
        assert_eq!(horz.width_count, 1);
        assert_eq!(horz.widths[0].org, 500);
        assert_eq!(horz.standard_width, 500);
        assert_eq!(horz.edge_distance_threshold, 100);
        assert!(!horz.extra_light);
        assert_eq!(horz.blue_count, 0, "blue zones live on the vertical axis");

        let vert = &metrics.axis[Dimension::Vert.index()];
        assert_eq!(vert.width_count, 1);
        assert_eq!(vert.widths[0].org, 700);
        assert_eq!(vert.standard_width, 700);
        assert_eq!(vert.edge_distance_threshold, 140);
        assert_eq!(vert.blue_count, 2);
        assert_eq!(vert.blues[0].r#ref.org, 700);
        assert_eq!(vert.blues[0].shoot.org, 700);
        assert_eq!(vert.blues[0].flags, LATIN_BLUE_TOP | LATIN_BLUE_ADJUSTMENT);
        assert_eq!(vert.blues[1].r#ref.org, 0);
        assert_eq!(vert.blues[1].shoot.org, 0);
        assert_eq!(vert.blues[1].flags, 0);

        assert!(
            metrics.root.digits_have_same_width,
            "an unmapped digit never breaks the match"
        );
    }

    #[test]
    fn metrics_init_falls_back_when_the_standard_character_is_missing() {
        let mut face = MockFace::new()
            .with_units_per_em(1000)
            .with_charmap(Encoding::Unicode, &[])
            .select(Encoding::Unicode);
        let mut metrics = latin_metrics(1000);

        metrics_init(&mut metrics, &mut face).expect("the Latin metrics initializer cannot fail today");

        // AF_LATIN_CONSTANT(50) at upem 1000: 50 * 1000 / 2048 = 24.
        for dim in [Dimension::Hort, Dimension::Vert] {
            let axis = &metrics.axis[dim.index()];
            assert_eq!(axis.width_count, 0);
            assert_eq!(axis.standard_width, 24);
            assert_eq!(axis.edge_distance_threshold, 4);
            assert!(!axis.extra_light);
        }
        assert_eq!(metrics.axis[Dimension::Vert.index()].blue_count, 0);
        assert!(metrics.root.digits_have_same_width);
        assert_eq!(face.charmap(), Some(Encoding::Unicode));
    }

    #[test]
    fn metrics_init_skips_the_analysis_without_a_unicode_charmap() {
        let mut face = MockFace::new()
            .with_units_per_em(1000)
            .with_charmap(Encoding::AppleRoman, &[(0x6F, 1)])
            .select(Encoding::AppleRoman)
            .with_glyph(1, rect_slot());
        let mut metrics = latin_metrics(1000);

        metrics_init(&mut metrics, &mut face).expect("the Latin metrics initializer cannot fail today");

        assert_eq!(face.charmap(), Some(Encoding::AppleRoman));
        // The still selected AppleRoman table maps `o` to the
        // rectangle, so the untouched axis fields prove the
        // Unicode-only analysis never ran.
        for dim in [Dimension::Hort, Dimension::Vert] {
            assert_eq!(metrics.axis[dim.index()].width_count, 0);
            assert_eq!(metrics.axis[dim.index()].standard_width, 0);
            assert_eq!(metrics.axis[dim.index()].edge_distance_threshold, 0);
        }
        assert_eq!(metrics.axis[Dimension::Vert.index()].blue_count, 0);
        assert!(
            !metrics.root.digits_have_same_width,
            "check_digits only runs after a successful charmap select"
        );
    }

    #[test]
    fn metrics_scale_snaps_the_x_height_and_scales_widths_and_blues() {
        let mut metrics = latin_metrics(2048);
        let horz = &mut metrics.axis[Dimension::Hort.index()];
        horz.width_count = 1;
        horz.widths[0].org = 200;
        horz.standard_width = 200;
        let vert = &mut metrics.axis[Dimension::Vert.index()];
        vert.blue_count = 2;
        vert.blues[0] = LatinBlue {
            r#ref: Width {
                org: 500,
                cur: 0,
                fit: 0,
            },
            shoot: Width {
                org: 500,
                cur: 0,
                fit: 0,
            },
            flags: LATIN_BLUE_ADJUSTMENT,
        };
        vert.blues[1] = LatinBlue {
            r#ref: Width {
                org: 1000,
                cur: 0,
                fit: 0,
            },
            shoot: Width {
                org: 1100,
                cur: 0,
                fit: 0,
            },
            flags: 0,
        };

        let scaler = Scaler {
            x_scale: 1 << 16,
            y_scale: 1 << 16,
            x_delta: 0,
            y_delta: 0,
            render_mode: RenderMode::Normal,
            flags: 0,
            x_ppem: 12,
        };
        metrics_scale(&mut metrics, &scaler);

        assert_eq!(metrics.root.scaler.render_mode, RenderMode::Normal);
        assert_eq!(metrics.root.scaler.x_ppem, 12);
        assert_eq!(metrics.root.scaler.x_scale, 65536);
        // The x-height zone scales to 500; the pixel grid wants 512,
        // so the vertical scale becomes 65536 * 512 / 500.
        assert_eq!(metrics.root.scaler.y_scale, 67109);

        let horz = &metrics.axis[Dimension::Hort.index()];
        assert_eq!(horz.org_scale, 65536);
        assert_eq!(horz.org_delta, 0);
        assert_eq!(horz.scale, 65536);
        assert_eq!(horz.delta, 0);
        assert_eq!(horz.widths[0].cur, 200);
        assert_eq!(horz.widths[0].fit, 200);
        assert!(!horz.extra_light, "200 units exceed 5/8 pixel");

        let vert = &metrics.axis[Dimension::Vert.index()];
        assert_eq!(vert.org_scale, 65536);
        assert_eq!(vert.scale, 67109);
        assert!(vert.extra_light, "a zero standard height stays extra light");

        let blue = &vert.blues[0];
        assert_eq!(blue.r#ref.cur, 512);
        assert_eq!(blue.r#ref.fit, 512);
        assert_eq!(blue.shoot.cur, 512);
        assert_eq!(blue.shoot.fit, 512);
        assert_eq!(blue.flags, LATIN_BLUE_ADJUSTMENT | LATIN_BLUE_ACTIVE);

        let blue = &vert.blues[1];
        assert_eq!(blue.r#ref.cur, 1024);
        assert_eq!(blue.r#ref.fit, 1024);
        assert_eq!(blue.shoot.cur, 1126);
        assert_eq!(blue.shoot.fit, 1126);
        assert_eq!(blue.flags, 0, "a 100 unit overshoot is not active");
    }

    #[test]
    fn metrics_scale_rounds_more_often_with_increase_x_height() {
        let mut metrics = latin_metrics(2048);
        let limit = PROP_INCREASE_X_HEIGHT_MIN + 6;
        metrics.root.globals.set_increase_x_height(limit);
        let vert = &mut metrics.axis[Dimension::Vert.index()];
        vert.blue_count = 1;
        vert.blues[0] = LatinBlue {
            r#ref: Width {
                org: 460,
                cur: 0,
                fit: 0,
            },
            shoot: Width {
                org: 460,
                cur: 0,
                fit: 0,
            },
            flags: LATIN_BLUE_ADJUSTMENT,
        };

        let scaler = Scaler {
            x_scale: 1 << 16,
            y_scale: 1 << 16,
            x_ppem: limit,
            ..Scaler::default()
        };
        metrics_scale(&mut metrics, &scaler);

        // Threshold 52 fits 460 up to 512: 65536 * 512 / 460 = 72944,
        // while the default threshold 40 would only reach 448
        // (65536 * 448 / 460 = 63826).
        assert_eq!(metrics.axis[Dimension::Vert.index()].scale, 72944);
    }

    #[test]
    fn metrics_scale_returns_early_for_an_unchanged_scaler() {
        let mut metrics = latin_metrics(2048);
        let scaler = Scaler {
            x_scale: 1 << 16,
            y_scale: 1 << 16,
            ..Scaler::default()
        };
        metrics_scale(&mut metrics, &scaler);

        metrics.axis[Dimension::Hort.index()].scale = 54321;
        metrics.axis[Dimension::Vert.index()].scale = 12345;
        metrics_scale(&mut metrics, &scaler);

        assert_eq!(metrics.axis[Dimension::Hort.index()].scale, 54321);
        assert_eq!(metrics.axis[Dimension::Vert.index()].scale, 12345);
    }

    #[test]
    fn hints_init_restores_the_scales_and_derives_the_flags() {
        let mut metrics = StyleMetrics::new(
            &STYLE_CLASSES[Style::LatnDflt.index()],
            Rc::new(GlobalsShared::new(1000)),
        );
        if let StyleMetrics::Latin(latin) = &mut metrics {
            latin.axis[Dimension::Hort.index()].scale = 1111;
            latin.axis[Dimension::Hort.index()].delta = 22;
            latin.axis[Dimension::Vert.index()].scale = 3333;
            latin.axis[Dimension::Vert.index()].delta = 44;
        }

        let face = MockFace::new();
        let mut hints = GlyphHints::default();

        metrics.scaler_mut().render_mode = RenderMode::Normal;
        hints_init(&mut hints, &metrics, &face);
        assert_eq!(hints.x_scale, 1111);
        assert_eq!(hints.x_delta, 22);
        assert_eq!(hints.y_scale, 3333);
        assert_eq!(hints.y_delta, 44);
        assert_eq!(hints.units_per_em, 1000);
        assert!(hints.metrics.is_some());
        assert_eq!(hints.scaler_flags, SCALER_FLAG_NO_WARPER);
        assert_eq!(hints.other_flags, LATIN_HINTS_STEM_ADJUST);

        metrics.scaler_mut().render_mode = RenderMode::Mono;
        hints_init(&mut hints, &metrics, &face);
        assert_eq!(
            hints.other_flags,
            LATIN_HINTS_HORZ_SNAP | LATIN_HINTS_VERT_SNAP | LATIN_HINTS_STEM_ADJUST | LATIN_HINTS_MONO
        );
        assert_eq!(hints.scaler_flags, SCALER_FLAG_NO_WARPER);

        let italic_face = MockFace::new().with_style_flags(STYLE_FLAG_ITALIC);
        metrics.scaler_mut().render_mode = RenderMode::Normal;
        hints_init(&mut hints, &metrics, &italic_face);
        assert_eq!(
            hints.scaler_flags,
            SCALER_FLAG_NO_HORIZONTAL | SCALER_FLAG_NO_WARPER
        );
        assert_eq!(hints.other_flags, LATIN_HINTS_STEM_ADJUST);

        metrics.scaler_mut().render_mode = RenderMode::Light;
        metrics.root().globals.set_warping(true);
        hints_init(&mut hints, &metrics, &face);
        assert_eq!(hints.scaler_flags, SCALER_FLAG_NO_HORIZONTAL);
        assert_eq!(hints.other_flags, 0, "light mode disables all adjustment");
    }

    /// Two linked vertical edges of a `700` unit wide stem, as
    /// [`detect_features`] leaves them for the rectangle.
    fn stem_hints(other_flags: u32) -> GlyphHints {
        let mut hints = rect_hints(1000);
        if let Some(StyleMetrics::Latin(latin)) = &mut hints.metrics {
            latin.axis[Dimension::Vert.index()].edge_distance_threshold = 60;
            latin.axis[Dimension::Hort.index()].edge_distance_threshold = 60;
        }
        detect_features(&mut hints, 0, &[], Dimension::Vert);
        hints.other_flags = other_flags;
        hints
    }

    #[test]
    fn snap_width_pulls_close_widths_to_the_standard() {
        let widths = [
            Width {
                org: 100,
                cur: 100,
                fit: 100,
            },
            Width {
                org: 300,
                cur: 300,
                fit: 300,
            },
        ];

        // Within 48/64 pixel of the rounded standard width the width
        // is snapped, from either side.
        assert_eq!(snap_width(&widths, 2, 110), 100);
        assert_eq!(snap_width(&widths, 2, 90), 100);
        assert_eq!(snap_width(&widths, 2, 275), 300);

        // Further away than the threshold the width is kept.
        assert_eq!(snap_width(&widths, 2, 180), 180);
        assert_eq!(snap_width(&widths, 2, 270), 270);

        // Only the first `count` entries are considered, and without
        // any standard width the input is returned unchanged.
        assert_eq!(snap_width(&widths, 1, 275), 275);
        assert_eq!(snap_width(&widths, 0, 123), 123);
    }

    #[test]
    fn compute_stem_width_keeps_widths_without_adjustment() {
        let mut hints = rect_hints(1000);
        assert_eq!(hints.other_flags, 0, "the fixture renders in light mode");

        assert_eq!(
            compute_stem_width(&hints, Dimension::Vert, 100, EDGE_NORMAL, EDGE_NORMAL),
            100
        );
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, -100, EDGE_ROUND, EDGE_NORMAL),
            -100
        );

        // Extra light faces keep every width as well.
        hints.other_flags = LATIN_HINTS_STEM_ADJUST | LATIN_HINTS_VERT_SNAP;
        if let Some(StyleMetrics::Latin(latin)) = &mut hints.metrics {
            latin.axis[Dimension::Vert.index()].extra_light = true;
        }
        assert_eq!(
            compute_stem_width(&hints, Dimension::Vert, 100, EDGE_NORMAL, EDGE_NORMAL),
            100
        );
    }

    #[test]
    fn compute_stem_width_quantizes_smooth_widths() {
        let mut hints = rect_hints(1000);
        hints.other_flags = LATIN_HINTS_STEM_ADJUST;

        // Neither dimension snaps, so the smooth process applies.
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, 40, EDGE_NORMAL, EDGE_NORMAL),
            56
        );
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, 70, EDGE_ROUND, EDGE_NORMAL),
            64,
            "a round stem below 80 units collapses to 64"
        );
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, 100, EDGE_ROUND, EDGE_NORMAL),
            100
        );
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, -40, EDGE_NORMAL, EDGE_NORMAL),
            -56,
            "the sign of a reversed stem is preserved"
        );

        // Serif widths of vertical stems stay untouched.
        assert_eq!(
            compute_stem_width(&hints, Dimension::Vert, 100, EDGE_NORMAL, EDGE_SERIF),
            100
        );

        // With a standard width the smooth process snaps to it.
        if let Some(StyleMetrics::Latin(latin)) = &mut hints.metrics {
            let axis = &mut latin.axis[Dimension::Hort.index()];
            axis.width_count = 1;
            axis.widths[0].cur = 100;
        }
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, 110, EDGE_NORMAL, EDGE_NORMAL),
            100
        );
    }

    #[test]
    fn compute_stem_width_snaps_in_the_strong_process() {
        let mut hints = rect_hints(1000);
        hints.other_flags = LATIN_HINTS_STEM_ADJUST | LATIN_HINTS_VERT_SNAP | LATIN_HINTS_HORZ_SNAP;

        // Vertical stems always round to integer pixels.
        assert_eq!(
            compute_stem_width(&hints, Dimension::Vert, 60, EDGE_NORMAL, EDGE_NORMAL),
            64
        );
        assert_eq!(
            compute_stem_width(&hints, Dimension::Vert, 70, EDGE_NORMAL, EDGE_NORMAL),
            64
        );
        assert_eq!(
            compute_stem_width(&hints, Dimension::Vert, 130, EDGE_NORMAL, EDGE_NORMAL),
            128
        );

        // Monochrome horizontal hinting snaps with its own
        // threshold.
        hints.other_flags |= LATIN_HINTS_MONO;
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, 60, EDGE_NORMAL, EDGE_NORMAL),
            64
        );
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, 100, EDGE_NORMAL, EDGE_NORMAL),
            128
        );

        // The anti-aliased horizontal process strengthens very small
        // stems, rounds the ones in the second pixel only when the
        // distortion stays below a quarter pixel, and rounds the
        // large ones.
        hints.other_flags = LATIN_HINTS_STEM_ADJUST | LATIN_HINTS_HORZ_SNAP;
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, 44, EDGE_NORMAL, EDGE_NORMAL),
            54
        );
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, 60, EDGE_NORMAL, EDGE_NORMAL),
            64
        );
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, 100, EDGE_NORMAL, EDGE_NORMAL),
            100,
            "rounding 100 to 64 would distort by more than 16 units"
        );
        assert_eq!(
            compute_stem_width(&hints, Dimension::Hort, 200, EDGE_NORMAL, EDGE_NORMAL),
            192
        );
    }

    #[test]
    fn align_linked_edge_moves_the_stem_relative_to_its_base() {
        let mut hints = stem_hints(LATIN_HINTS_STEM_ADJUST | LATIN_HINTS_VERT_SNAP);
        {
            let edges = &mut hints.axis[Dimension::Vert.index()].edges;
            assert_eq!(edges.len(), 2);
            edges[0].pos = 64;
        }

        align_linked_edge(&mut hints, Dimension::Vert, 0, 1);

        // The 700 unit stem snaps to 704 units.
        assert_eq!(hints.axis[Dimension::Vert.index()].edges[0].pos, 64);
        assert_eq!(hints.axis[Dimension::Vert.index()].edges[1].pos, 768);

        // Out of range edge indices are ignored.
        align_linked_edge(&mut hints, Dimension::Vert, 0, 9);
        assert_eq!(hints.axis[Dimension::Vert.index()].edges[1].pos, 768);
    }

    #[test]
    fn align_serif_edge_shifts_the_serif_by_the_base_movement() {
        let mut hints = rect_hints(1000);
        hints.axis[Dimension::Vert.index()].edges = vec![
            Edge {
                opos: 600,
                pos: 640,
                ..Edge::default()
            },
            Edge {
                opos: 700,
                pos: 999,
                ..Edge::default()
            },
        ];

        align_serif_edge(&mut hints, Dimension::Vert, 0, 1);
        assert_eq!(hints.axis[Dimension::Vert.index()].edges[1].pos, 740);

        align_serif_edge(&mut hints, Dimension::Vert, 1, 9);
        assert_eq!(hints.axis[Dimension::Vert.index()].edges[1].pos, 740);
    }

    #[test]
    fn hint_edges_snaps_the_stems_of_the_rectangle() {
        let mut hints = stem_hints(LATIN_HINTS_STEM_ADJUST | LATIN_HINTS_VERT_SNAP);
        hint_edges(&mut hints, Dimension::Vert);

        let edges = &hints.axis[Dimension::Vert.index()].edges;
        assert_eq!(edges.len(), 2);
        assert_eq!((edges[0].opos, edges[1].opos), (0, 700));
        assert_eq!(
            (edges[0].pos, edges[1].pos),
            (0, 704),
            "the first stem becomes the anchor, the second one follows it"
        );
        assert_eq!(edges[0].flags, EDGE_NORMAL | EDGE_DONE);
        assert_eq!(edges[1].flags, EDGE_NORMAL | EDGE_DONE);
    }

    #[test]
    fn hint_edges_anchors_a_stem_at_its_blue_zone() {
        let mut hints = stem_hints(0);
        {
            let vert = &mut hints.axis[Dimension::Vert.index()];
            // The left edge of the stem is aligned to a blue zone
            // whose fitted reference is 660 units.
            vert.edges[0].blue_edge = Some(Width {
                org: 660,
                cur: 660,
                fit: 660,
            });
        }

        hint_edges(&mut hints, Dimension::Vert);

        let edges = &hints.axis[Dimension::Vert.index()].edges;
        assert_eq!(edges[0].pos, 660, "the blue zone wins over rounding");
        assert_eq!(edges[1].pos, 660 + 700, "the linked edge follows");
        assert_eq!(edges[0].flags, EDGE_NORMAL | EDGE_DONE);
        assert_eq!(edges[1].flags, EDGE_NORMAL | EDGE_DONE);
    }

    #[test]
    fn hint_edges_moves_single_edges_between_their_neighbours() {
        let mut hints = rect_hints(1000);
        {
            let vert = &mut hints.axis[Dimension::Vert.index()];
            // Two stems that anchor the hinting plus two single
            // edges without a link or a serif.
            vert.edges = vec![
                Edge {
                    opos: 0,
                    pos: 0,
                    link: Some(2),
                    ..Edge::default()
                },
                Edge {
                    opos: 100,
                    pos: 0,
                    ..Edge::default()
                },
                Edge {
                    opos: 700,
                    pos: 0,
                    link: Some(0),
                    ..Edge::default()
                },
                Edge {
                    opos: 900,
                    pos: 0,
                    ..Edge::default()
                },
            ];
            vert.segments.clear();
        }
        hints.other_flags = 0;

        hint_edges(&mut hints, Dimension::Vert);

        let edges = &hints.axis[Dimension::Vert.index()].edges;
        assert_eq!((edges[0].pos, edges[2].pos), (0, 700), "the stem sets the anchor");
        // Interpolated between its neighbours: 0 + 100 * 700 / 700.
        assert_eq!(edges[1].pos, 100);
        // The last edge has no neighbour after it, so it snaps
        // relative to the anchor: (900 + 16) & ~31.
        assert_eq!(edges[3].pos, 896);
        for edge in edges {
            assert_eq!(edge.flags, EDGE_NORMAL | EDGE_DONE);
        }
    }

    #[test]
    fn hints_apply_round_trips_the_identity_scaled_rectangle() {
        let metrics = StyleMetrics::new(
            &STYLE_CLASSES[Style::LatnDflt.index()],
            Rc::new(GlobalsShared::new(1000)),
        );
        let mut hints = rect_hints(1000);
        let mut outline = rect_outline();
        let original = outline.points.clone();

        hints_apply(&mut hints, &mut outline, &metrics).expect("the Latin application cannot fail today");

        // Every edge of the rectangle sits exactly on a pixel at
        // identity scale, so the grid fitting does not move it.
        assert_eq!(outline.points, original);
        assert_eq!(hints.axis[Dimension::Hort.index()].edges.len(), 2);
        assert_eq!(hints.axis[Dimension::Vert.index()].edges.len(), 2);
    }

    #[test]
    fn hints_apply_grows_the_stem_when_stems_are_adjusted() {
        let metrics = StyleMetrics::new(
            &STYLE_CLASSES[Style::LatnDflt.index()],
            Rc::new(GlobalsShared::new(1000)),
        );
        let mut hints = rect_hints(1000);
        hints.other_flags = LATIN_HINTS_STEM_ADJUST | LATIN_HINTS_VERT_SNAP;
        let mut outline = rect_outline();

        hints_apply(&mut hints, &mut outline, &metrics).expect("the Latin application cannot fail today");

        // The vertical stem of the rectangle grows from 700 to 704
        // units, so the two upper points are lifted.
        assert_eq!(
            outline.points,
            vec![
                Vector::new(0, 0),
                Vector::new(500, 0),
                Vector::new(500, 704),
                Vector::new(0, 704),
            ]
        );
    }
}
