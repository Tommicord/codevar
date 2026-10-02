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

//! Warping algorithm (`afwarp.c`).
//!
//! The idea of the warping code is to slightly scale and shift a glyph
//! within a single dimension so that as much of its segments are
//! aligned (more or less) on the grid.  To find out the optimal scaling
//! and shifting value, various parameter combinations are tried and
//! scored.
//!
//! Warping is only used by the `light' rendering target of the Latin,
//! CJK and latin2 writing systems, and only when the `warping' module
//! property is enabled.

use codevar_truetype_core::{div_fix, mul_fix};

use crate::hints::GlyphHints;
use crate::{Dimension, Fixed, Pos};

/// `AF_WarpScore` (`afwarp.h`): the score type of the warper, an
/// `FT_Int32`.
type WarpScore = i32;

/// `AF_WARPER_FLOOR(x)` (`afwarp.h`): rounds `x` down to a multiple
/// of 64.
#[inline]
pub const fn warper_floor(x: i64) -> i64 {
    x & !63
}

/// `AF_WARPER_CEIL(x)` (`afwarp.h`): rounds `x` up to a multiple of
/// 64.
#[inline]
pub const fn warper_ceil(x: i64) -> i64 {
    warper_floor(x.wrapping_add(63))
}

/// `af_warper_weights[64]` (`afwarp.c`): the alignment weights.
///
/// The weights cover the range 0/64 - 63/64 of a pixel.  Obviously,
/// values around a half pixel (which means exactly between two grid
/// lines) gets the worst weight.
static WARPER_WEIGHTS: [WarpScore; 64] = [
    35, 32, 30, 25, 20, 15, 12, 10, 5, 1, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, -1, -2, -5, -8, -10, -10, -20, -20, -30, -30, //
    -30, -30, -20, -20, -10, -10, -8, -5, -2, -1, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 1, 5, 10, 12, 15, 20, 25, 30, 32,
];

/// `AF_WarperRec` (`afwarp.h`): the mutable state of one warping
/// computation.
#[derive(Debug)]
struct Warper {
    x1: Pos,
    x2: Pos,
    t1: Pos,
    /// FreeType writes `t2` but never reads it; kept so the record
    /// mirrors `AF_WarperRec` exactly.
    #[allow(dead_code)]
    t2: Pos,
    x1min: Pos,
    x1max: Pos,
    x2min: Pos,
    x2max: Pos,
    w0: Pos,
    wmin: Pos,
    wmax: Pos,
    best_scale: Fixed,
    best_delta: Pos,
    best_score: WarpScore,
    best_distort: WarpScore,
}

impl Warper {
    /// `af_warper_compute_line_best` (`afwarp.c`): scores segments for
    /// a given `scale` and `delta` in the range `xx1` to `xx2`, and
    /// stores the best result in the warper.  If the new best score is
    /// equal to the old one, prefer the value with a smaller distortion
    /// (around `base_distort`).
    fn compute_line_best(
        &mut self,
        scale: Fixed,
        delta: Pos,
        xx1: Pos,
        xx2: Pos,
        base_distort: WarpScore,
        segments: &[crate::hints::Segment],
    ) {
        let mut scores = [0 as WarpScore; 65];

        let idx0 = (xx1 - self.t1) as i32;

        let (idx_min, idx_max) = {
            let mut xx1min = self.x1min;
            let w = xx2 - xx1;

            if xx1min.wrapping_add(w) < self.x2min {
                xx1min = self.x2min.wrapping_sub(w);
            }

            let mut xx1max = self.x1max;
            if xx1max.wrapping_add(w) > self.x2max {
                xx1max = self.x2max.wrapping_sub(w);
            }

            ((xx1min - self.t1) as i32, (xx1max - self.t1) as i32)
        };

        if idx_min < 0 || idx_min > idx_max || idx_max > 64 {
            return;
        }

        for segment in segments {
            let len = i32::from(segment.max_coord) - i32::from(segment.min_coord);
            let y0 = mul_fix(i64::from(segment.pos), scale).wrapping_add(delta);
            let mut y = y0.wrapping_add(i64::from(idx_min.wrapping_sub(idx0)));

            for idx in idx_min..=idx_max {
                let weight = WARPER_WEIGHTS[(y & 63) as usize];
                scores[idx as usize] = scores[idx as usize].wrapping_add(weight.wrapping_mul(len));
                y = y.wrapping_add(1);
            }
        }

        for idx in idx_min..=idx_max {
            let score = scores[idx as usize];
            let distort = base_distort.wrapping_add(idx.wrapping_sub(idx0));

            if score > self.best_score || (score == self.best_score && distort < self.best_distort) {
                self.best_score = score;
                self.best_distort = distort;
                self.best_scale = scale;
                self.best_delta = delta.wrapping_add(i64::from(idx.wrapping_sub(idx0)));
            }
        }
    }
}

/// `af_warper_compute` (`afwarp.c`): computes the optimal scaling and
/// delta values for a glyph in one dimension.
///
/// Returns the `(scale, delta)` pair that best aligns the segments of
/// `hints` with the pixel grid, and stores the resulting overshoot
/// deltas in [`GlyphHints::xmin_delta`] and [`GlyphHints::xmax_delta`]
/// as FreeType does.
///
/// # Porting note
///
/// FreeType keeps an `AF_WarperRec` on the stack and writes the
/// results through two out parameters; this port returns them as a
/// tuple.  The original also reads `points[0]` before checking that
/// any point exists, which this port guards against.
pub fn warper_compute(hints: &mut GlyphHints, dim: Dimension) -> (Fixed, Pos) {
    let (org_scale, org_delta) = match dim {
        Dimension::Vert => (hints.y_scale, hints.y_delta),
        Dimension::Hort => (hints.x_scale, hints.x_delta),
    };

    let mut warper = Warper {
        x1: 0,
        x2: 0,
        t1: 0,
        t2: 0,
        x1min: 0,
        x1max: 0,
        x2min: 0,
        x2max: 0,
        w0: 0,
        wmin: 0,
        wmax: 0,
        best_scale: org_scale,
        best_delta: org_delta,
        best_score: WarpScore::MIN,
        best_distort: 0,
    };

    let axis = &hints.axis[dim.index()];
    let segments = &axis.segments;
    let num_segments = segments.len();
    let points = &hints.points;
    let num_points = points.len();

    if num_segments < 1 || num_points < 1 {
        return (org_scale, org_delta);
    }

    let mut x1 = i64::from(points[0].fx);
    let mut x2 = x1;
    for point in points.iter().skip(1) {
        let x = i64::from(point.fx);
        if x < x1 {
            x1 = x;
        }
        if x > x2 {
            x2 = x;
        }
    }

    if x1 >= x2 {
        return (org_scale, org_delta);
    }

    warper.x1 = mul_fix(x1, org_scale).wrapping_add(org_delta);
    warper.x2 = mul_fix(x2, org_scale).wrapping_add(org_delta);

    warper.t1 = warper_floor(warper.x1);
    warper.t2 = warper_ceil(warper.x2);

    warper.x1min = warper.x1 & !31;
    warper.x1max = warper.x1min.wrapping_add(32);
    warper.x2min = warper.x2 & !31;
    warper.x2max = warper.x2min.wrapping_add(32);

    if warper.x1max > warper.x2 {
        warper.x1max = warper.x2;
    }

    if warper.x2min < warper.x1 {
        warper.x2min = warper.x1;
    }

    warper.w0 = warper.x2 - warper.x1;

    if warper.w0 <= 64 {
        warper.x1max = warper.x1;
        warper.x2min = warper.x2;
    }

    warper.wmin = warper.x2min - warper.x1max;
    warper.wmax = warper.x2max - warper.x1min;

    let margin = {
        let mut margin = 16;
        if warper.w0 <= 128 {
            margin = 8;
            if warper.w0 <= 96 {
                margin = 4;
            }
        }
        margin
    };

    if warper.wmin < warper.w0 - margin {
        warper.wmin = warper.w0 - margin;
    }

    if warper.wmax > warper.w0 + margin {
        warper.wmax = warper.w0 + margin;
    }

    if warper.wmin < warper.w0 * 3 / 4 {
        warper.wmin = warper.w0 * 3 / 4;
    }

    if warper.wmax > warper.w0 * 5 / 4 {
        warper.wmax = warper.w0 * 5 / 4;
    }

    let mut w = warper.wmin;
    while w <= warper.wmax {
        let mut xx1 = warper.x1;
        let mut xx2 = warper.x2;

        if w >= warper.w0 {
            xx1 -= w - warper.w0;
            if xx1 < warper.x1min {
                xx2 += warper.x1min - xx1;
                xx1 = warper.x1min;
            }
        } else {
            xx1 -= w - warper.w0;
            if xx1 > warper.x1max {
                xx2 -= xx1 - warper.x1max;
                xx1 = warper.x1max;
            }
        }

        let mut base_distort;
        if xx1 < warper.x1 {
            base_distort = (warper.x1 - xx1) as WarpScore;
        } else {
            base_distort = (xx1 - warper.x1) as WarpScore;
        }

        if xx2 < warper.x2 {
            base_distort += (warper.x2 - xx2) as WarpScore;
        } else {
            base_distort += (xx2 - warper.x2) as WarpScore;
        }

        base_distort = base_distort.wrapping_mul(10);

        let new_scale = org_scale.wrapping_add(div_fix(w - warper.w0, x2 - x1));
        let new_delta = xx1.wrapping_sub(mul_fix(x1, new_scale));

        warper.compute_line_best(new_scale, new_delta, xx1, xx2, base_distort, segments);

        w = w.wrapping_add(1);
    }

    let best_scale = warper.best_scale;
    let best_delta = warper.best_delta;

    hints.xmin_delta = mul_fix(x1, best_scale - org_scale).wrapping_add(best_delta);
    hints.xmax_delta = mul_fix(x2, best_scale - org_scale).wrapping_add(best_delta);

    (best_scale, best_delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hints::{AxisHints, Point, Segment};

    #[test]
    fn warper_floor_and_ceil_round_to_pixel_boundaries() {
        assert_eq!(warper_floor(0), 0);
        assert_eq!(warper_floor(63), 0);
        assert_eq!(warper_floor(64), 64);
        assert_eq!(warper_floor(-1), -64);
        assert_eq!(warper_floor(-64), -64);
        assert_eq!(warper_floor(-65), -128);

        assert_eq!(warper_ceil(0), 0);
        assert_eq!(warper_ceil(1), 64);
        assert_eq!(warper_ceil(63), 64);
        assert_eq!(warper_ceil(64), 64);
        assert_eq!(warper_ceil(-1), 0);
        assert_eq!(warper_ceil(-64), -64);
    }

    #[test]
    fn weights_table_has_the_reference_values() {
        assert_eq!(WARPER_WEIGHTS.len(), 64);
        assert_eq!(WARPER_WEIGHTS[0], 35);
        assert_eq!(WARPER_WEIGHTS[9], 1);
        assert_eq!(WARPER_WEIGHTS[10], 0);
        assert_eq!(WARPER_WEIGHTS[22], -1);
        assert_eq!(WARPER_WEIGHTS[31], -30);
        assert_eq!(WARPER_WEIGHTS[32], -30);
        assert_eq!(WARPER_WEIGHTS[41], -1);
        assert_eq!(WARPER_WEIGHTS[42], 0);
        assert_eq!(WARPER_WEIGHTS[55], 1);
        assert_eq!(WARPER_WEIGHTS[63], 32);
        assert_eq!(WARPER_WEIGHTS.iter().copied().min(), Some(-30));
        assert!(WARPER_WEIGHTS[10..=21].iter().all(|w| *w == 0));
        assert!(WARPER_WEIGHTS[42..=54].iter().all(|w| *w == 0));
    }

    fn empty_hints() -> GlyphHints {
        GlyphHints {
            x_scale: 1 << 16,
            y_scale: 1 << 16,
            x_delta: 0,
            y_delta: 0,
            ..GlyphHints::default()
        }
    }

    #[test]
    fn no_segments_keeps_the_original_transformation() {
        let mut hints = empty_hints();
        let (scale, delta) = warper_compute(&mut hints, Dimension::Hort);
        assert_eq!(scale, 1 << 16);
        assert_eq!(delta, 0);
        assert_eq!(hints.xmin_delta, 0);
        assert_eq!(hints.xmax_delta, 0);
    }

    #[test]
    fn collinear_points_keep_the_original_transformation() {
        let mut hints = empty_hints();
        hints.points.push(Point {
            fx: 100,
            fy: 0,
            ..Point::default()
        });
        hints.points.push(Point {
            fx: 100,
            fy: 0,
            ..Point::default()
        });
        let mut axis = AxisHints::new();
        axis.segments.push(Segment::default());
        hints.axis[Dimension::Hort.index()] = axis;

        let (scale, delta) = warper_compute(&mut hints, Dimension::Hort);
        assert_eq!(scale, 1 << 16);
        assert_eq!(delta, 0);
    }

    #[test]
    fn a_single_vertical_stem_is_kept_on_the_grid() {
        let mut hints = empty_hints();
        for (fx, fy) in [(0, 0), (200, 0), (200, 900), (0, 900)] {
            hints.points.push(Point {
                fx,
                fy,
                ..Point::default()
            });
        }

        let mut axis = AxisHints::new();
        let idx = axis.new_segment();
        axis.segments[idx].pos = 100;
        axis.segments[idx].min_coord = 0;
        axis.segments[idx].max_coord = 900;
        axis.major_dir = crate::Direction::Up;
        hints.axis[Dimension::Hort.index()] = axis;

        let org_scale = hints.x_scale;
        let (scale, delta) = warper_compute(&mut hints, Dimension::Hort);

        assert!(scale >= org_scale - (1 << 16));
        assert!(scale <= org_scale + (1 << 16));
        let scaled = mul_fix(100, scale).wrapping_add(delta);
        assert_eq!(
            scaled.rem_euclid(64),
            0,
            "the warper snapped the stem onto the pixel grid"
        );

        let scaled_x1 = mul_fix(0, scale - org_scale).wrapping_add(delta);
        let scaled_x2 = mul_fix(200, scale - org_scale).wrapping_add(delta);
        assert_eq!(hints.xmin_delta, scaled_x1);
        assert_eq!(hints.xmax_delta, scaled_x2);
    }

    #[test]
    fn line_best_rejects_out_of_range_indices() {
        let mut warper = Warper {
            x1: 0,
            x2: 640,
            t1: 0,
            t2: 640,
            x1min: 0,
            x1max: 32,
            x2min: 608,
            x2max: 640,
            w0: 640,
            wmin: 576,
            wmax: 640,
            best_scale: 1 << 16,
            best_delta: 0,
            best_score: WarpScore::MIN,
            best_distort: 0,
        };
        let segments = [Segment {
            pos: 100,
            min_coord: 0,
            max_coord: 100,
            ..Segment::default()
        }];
        warper.compute_line_best(1 << 16, 0, -1000, 1000, 0, &segments);
        assert_eq!(warper.best_score, WarpScore::MIN);
        assert_eq!(warper.best_scale, 1 << 16);
        assert_eq!(warper.best_delta, 0);
    }
}
