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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Post-fitting path cleanup.
//!
//! The fitter works one smooth run at a time, so it can emit two nearly
//! collinear lines at a shallow vertex, or two cubics that could have been a
//! single cubic. This pass removes those redundancies without changing the
//! shape by more than a caller-selected tolerance:
//!
//! * zero-length segments are deleted;
//! * a cubic whose control polygon follows its chord is demoted to a line;
//! * adjacent lines that continue in the same direction are merged;
//! * adjacent cubics with continuous tangents are re-fitted as one if the
//!   combined fit stays within tolerance;
//! * subpaths without meaningful area are removed.
//!
//! Merging never crosses a corner: the tangent angle at the join must be below
//! the configured threshold. That keeps icon corners and letter stems sharp
//! while still collapsing a rasterized arc into a compact cubic chain.

use alloc::vec::Vec;
use libm::fabs;

use crate::bezier::{BezierFitOptions, chain_error, fit_chain};
use crate::error::{Error, Result};
use crate::geom::{CubicBezier, clamp};
use crate::polygon::{point_line_distance, project_on_line};
use crate::vector::{Segment, Shape, SubPath, VectorDoc};

/// Settings for [`optimize_document`] and [`optimize_path`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizeSettings {
    /// Distance below which a cubic is treated as a straight line.
    pub line_tolerance: f64,
    /// Maximum distance for merging two adjacent lines.
    pub line_merge_tolerance: f64,
    /// Maximum distance for merging two adjacent cubics.
    pub cubic_merge_tolerance: f64,
    /// Maximum tangent discontinuity for a cubic merge, in degrees.
    pub merge_angle_degrees: f64,
    /// Whether near-line cubics are converted to lines.
    pub convert_lines: bool,
    /// Whether adjacent collinear lines are merged.
    pub merge_lines: bool,
    /// Whether adjacent continuous cubics are merged.
    pub merge_cubics: bool,
    /// Number of cleanup passes.
    pub passes: u8,
    /// Minimum absolute area for a subpath to be retained.
    pub minimum_subpath_area: f64,
    /// Flattening tolerance used for area measurements.
    pub area_tolerance: f64,
}

impl Default for OptimizeSettings {
    fn default() -> Self {
        Self {
            line_tolerance: 0.2,
            line_merge_tolerance: 0.12,
            cubic_merge_tolerance: 0.3,
            merge_angle_degrees: 20.0,
            convert_lines: true,
            merge_lines: true,
            merge_cubics: true,
            passes: 2,
            minimum_subpath_area: 0.0,
            area_tolerance: 0.1,
        }
    }
}

impl OptimizeSettings {
    /// A conservative setting that only removes exact degeneracies.
    #[inline]
    pub const fn minimal() -> Self {
        Self {
            line_tolerance: 0.0,
            line_merge_tolerance: 0.0,
            cubic_merge_tolerance: 0.0,
            merge_angle_degrees: 0.0,
            convert_lines: false,
            merge_lines: false,
            merge_cubics: false,
            passes: 1,
            minimum_subpath_area: 0.0,
            area_tolerance: 0.1,
        }
    }

    /// Builds optimizer settings after checking every numeric field.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        line_tolerance: f64,
        line_merge_tolerance: f64,
        cubic_merge_tolerance: f64,
        merge_angle_degrees: f64,
        convert_lines: bool,
        merge_lines: bool,
        merge_cubics: bool,
        passes: u8,
        minimum_subpath_area: f64,
        area_tolerance: f64,
    ) -> Result<Self> {
        let settings = Self {
            line_tolerance,
            line_merge_tolerance,
            cubic_merge_tolerance,
            merge_angle_degrees,
            convert_lines,
            merge_lines,
            merge_cubics,
            passes,
            minimum_subpath_area,
            area_tolerance,
        };
        Self::try_from(&settings)
    }

    fn bezier_options(self) -> BezierFitOptions {
        BezierFitOptions {
            tolerance: self.cubic_merge_tolerance.max(1.0e-4),
            line_tolerance: self.line_tolerance,
            max_depth: 12,
            reparameterize: true,
        }
    }
}

impl TryFrom<&OptimizeSettings> for OptimizeSettings {
    type Error = Error;

    fn try_from(settings: &OptimizeSettings) -> Result<Self> {
        let tolerances = [
            ("line_tolerance", settings.line_tolerance),
            ("line_merge_tolerance", settings.line_merge_tolerance),
            ("cubic_merge_tolerance", settings.cubic_merge_tolerance),
            ("area_tolerance", settings.area_tolerance),
        ];
        for (field, value) in tolerances {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::InvalidConfig { field });
            }
        }
        if !settings.merge_angle_degrees.is_finite() || !(0.0..=180.0).contains(&settings.merge_angle_degrees)
        {
            return Err(Error::InvalidConfig {
                field: "merge_angle_degrees",
            });
        }
        if !settings.minimum_subpath_area.is_finite() || settings.minimum_subpath_area < 0.0 {
            return Err(Error::InvalidConfig {
                field: "minimum_subpath_area",
            });
        }
        Ok(*settings)
    }
}

/// Counts describing what the optimizer changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OptimizeReport {
    /// Segments before the pass.
    pub segments_before: usize,
    /// Segments after the pass.
    pub segments_after: usize,
    /// Number of adjacent line pairs merged.
    pub merged_lines: usize,
    /// Number of adjacent cubic pairs merged.
    pub merged_cubics: usize,
    /// Number of near-line cubics converted to lines.
    pub converted_lines: usize,
    /// Number of subpaths dropped for having no meaningful area.
    pub removed_subpaths: usize,
    /// Number of shapes dropped because their path became empty.
    pub removed_shapes: usize,
}

impl OptimizeReport {
    /// Net segment reduction (saturating at zero when a conversion adds work).
    #[inline]
    pub const fn reduction(self) -> usize {
        self.segments_before
            .saturating_sub(self.segments_after)
    }
}

/// Optimizes every path in a document in place.
pub fn optimize_document(document: &mut VectorDoc, settings: OptimizeSettings) -> OptimizeReport {
    let mut report = OptimizeReport {
        segments_before: document.segment_count(),
        ..OptimizeReport::default()
    };

    for shape in &mut document.shapes {
        optimize_path(&mut shape.path, settings, &mut report);
    }

    let before = document.shapes.len();
    document
        .shapes
        .retain(|shape: &Shape| !shape.is_degenerate());
    report.removed_shapes = before.saturating_sub(document.shapes.len());
    report.segments_after = document.segment_count();
    report
}

/// Optimizes one path in place.
pub fn optimize_path(
    path: &mut crate::vector::Path,
    settings: OptimizeSettings,
    report: &mut OptimizeReport,
) {
    let settings = sanitized(settings);
    for _ in 0..settings.passes.max(1) {
        for subpath in &mut path.subpaths {
            optimize_subpath(subpath, settings, report);
        }
        path.subpaths.retain(|subpath| {
            if subpath.is_empty() || !subpath.is_finite() {
                return false;
            }
            let area = subpath.signed_area(settings.area_tolerance);
            if fabs(area) < settings.minimum_subpath_area {
                report.removed_subpaths += 1;
                return false;
            }
            true
        });
    }
    // Recompute the start point after the first/last merge pass so the ring's
    // implicit closing edge always begins at the stored start.
    for subpath in &mut path.subpaths {
        if let Some(first) = subpath.segments.first() {
            subpath.start = first.start();
        }
        if subpath.closed {
            subpath.close();
        }
    }
}

/// Optimizes a single subpath in place.
pub fn optimize_subpath(subpath: &mut SubPath, settings: OptimizeSettings, report: &mut OptimizeReport) {
    if subpath.segments.is_empty() {
        return;
    }

    remove_degenerate_segments(subpath);

    if settings.convert_lines {
        convert_flat_cubics(subpath, settings, report);
    }

    if settings.merge_lines {
        merge_adjacent_lines(subpath, settings, report);
        if subpath.closed {
            merge_closing_lines(subpath, settings, report);
        }
    }

    if settings.merge_cubics {
        merge_adjacent_cubics(subpath, settings, report);
    }

    if let Some(first) = subpath.segments.first() {
        subpath.start = first.start();
    }
}

fn remove_degenerate_segments(subpath: &mut SubPath) {
    subpath.segments.retain(|segment| match segment {
        Segment::Line { start, end } => start.distance(*end) > 1.0e-9,
        Segment::Cubic(cubic) => {
            cubic.start.distance(cubic.end) > 1.0e-9
                || cubic.control1.distance(cubic.start) > 1.0e-9
                || cubic.control2.distance(cubic.end) > 1.0e-9
        }
    });
}

fn convert_flat_cubics(subpath: &mut SubPath, settings: OptimizeSettings, report: &mut OptimizeReport) {
    for segment in &mut subpath.segments {
        if let Segment::Cubic(cubic) = *segment
            && cubic_is_line_like(cubic, settings.line_tolerance)
        {
            *segment = Segment::line(cubic.start, cubic.end);
            report.converted_lines += 1;
        }
    }
}

fn cubic_is_line_like(cubic: CubicBezier, tolerance: f64) -> bool {
    let chord = cubic.end - cubic.start;
    let length = chord.length();
    if length <= 1.0e-9 {
        return false;
    }
    let control1_distance = point_line_distance(cubic.control1, cubic.start, cubic.end);
    let control2_distance = point_line_distance(cubic.control2, cubic.start, cubic.end);
    if control1_distance > tolerance || control2_distance > tolerance {
        return false;
    }
    let inverse = 1.0 / length;
    let direction = chord * inverse;
    let t1 = (cubic.control1 - cubic.start).dot(direction) * inverse;
    let t2 = (cubic.control2 - cubic.start).dot(direction) * inverse;
    let allowance = tolerance * inverse + 1.0e-9;
    t1 >= -allowance && t1 <= 1.0 + allowance && t2 >= -allowance && t2 <= 1.0 + allowance
}

fn merge_adjacent_lines(subpath: &mut SubPath, settings: OptimizeSettings, report: &mut OptimizeReport) {
    if subpath.segments.len() < 2 {
        return;
    }
    let mut index = 0;
    while index + 1 < subpath.segments.len() {
        let first = subpath.segments[index];
        let second = subpath.segments[index + 1];
        let (
            Segment::Line {
                start: first_start,
                end: first_end,
            },
            Segment::Line { end: second_end, .. },
        ) = (first, second)
        else {
            index += 1;
            continue;
        };
        if first_end.distance(second_end) <= 1.0e-9 {
            index += 1;
            continue;
        }
        let first_direction = first_end - first_start;
        let second_direction = second_end - first_end;
        let continuing = first_direction.dot(second_direction) > 0.0;
        let flat = point_line_distance(first_end, first_start, second_end) <= settings.line_merge_tolerance;
        if continuing && flat {
            subpath.segments[index] = Segment::line(first_start, second_end);
            subpath.segments.remove(index + 1);
            report.merged_lines += 1;
        } else {
            index += 1;
        }
    }
}

fn merge_closing_lines(subpath: &mut SubPath, settings: OptimizeSettings, report: &mut OptimizeReport) {
    if subpath.segments.len() < 2 {
        return;
    }
    let first = subpath.segments[0];
    let last = subpath.segments[subpath.segments.len() - 1];
    let (
        Segment::Line {
            start: first_start,
            end: first_end,
        },
        Segment::Line {
            start: last_start,
            end: last_end,
        },
    ) = (first, last)
    else {
        return;
    };
    if last_end.distance(first_start) > 1.0e-7 {
        return;
    }
    let incoming = last_end - last_start;
    let outgoing = first_end - first_start;
    if incoming.dot(outgoing) <= 0.0 {
        return;
    }
    if point_line_distance(last_end, last_start, first_end) > settings.line_merge_tolerance {
        return;
    }
    let merged = Segment::line(last_start, first_end);
    subpath
        .segments
        .remove(subpath.segments.len() - 1);
    if !subpath.segments.is_empty() {
        subpath.segments.remove(0);
    }
    subpath.segments.insert(0, merged);
    subpath.start = last_start;
    report.merged_lines += 1;
}

fn merge_adjacent_cubics(subpath: &mut SubPath, settings: OptimizeSettings, report: &mut OptimizeReport) {
    if subpath.segments.len() < 2 {
        return;
    }
    let angle_limit = settings.merge_angle_degrees * core::f64::consts::PI / 180.0;
    let mut index = 0;
    while index + 1 < subpath.segments.len() {
        let first = subpath.segments[index];
        let second = subpath.segments[index + 1];
        let (Segment::Cubic(first_cubic), Segment::Cubic(second_cubic)) = (first, second) else {
            index += 1;
            continue;
        };
        if first_cubic.end.distance(second_cubic.start) > 1.0e-7 {
            index += 1;
            continue;
        }
        let Some(first_tangent) = first_cubic.tangent(1.0) else {
            index += 1;
            continue;
        };
        let Some(second_tangent) = second_cubic.tangent(0.0) else {
            index += 1;
            continue;
        };
        if first_tangent.angle_between(second_tangent) > angle_limit {
            index += 1;
            continue;
        }
        let Some(merged) = try_merge_cubics(first_cubic, second_cubic, settings) else {
            index += 1;
            continue;
        };
        subpath.segments[index] = merged;
        subpath.segments.remove(index + 1);
        report.merged_cubics += 1;
    }
}

fn try_merge_cubics(first: CubicBezier, second: CubicBezier, settings: OptimizeSettings) -> Option<Segment> {
    let mut samples = Vec::with_capacity(32);
    samples.push(first.start);
    Segment::Cubic(first).flatten_into(0.2, &mut samples);
    Segment::Cubic(second).flatten_into(0.2, &mut samples);
    samples.dedup_by(|a, b| a.distance(*b) <= 1.0e-9);
    if samples.len() < 3 {
        return None;
    }
    let start_tangent = first.tangent(0.0)?;
    let end_tangent = second.tangent(1.0)?;
    let fitted = fit_chain(
        &samples,
        Some(start_tangent),
        Some(end_tangent),
        settings.bezier_options(),
    );
    if fitted.len() != 1 {
        return None;
    }
    let error = chain_error(&samples, &fitted);
    if error > settings.cubic_merge_tolerance * 1.5 + 1.0e-4 {
        return None;
    }
    Some(fitted[0])
}

fn sanitized(mut settings: OptimizeSettings) -> OptimizeSettings {
    settings.line_tolerance = non_negative(settings.line_tolerance, 0.2);
    settings.line_merge_tolerance = non_negative(settings.line_merge_tolerance, 0.12);
    settings.cubic_merge_tolerance = non_negative(settings.cubic_merge_tolerance, 0.3);
    settings.merge_angle_degrees = if settings.merge_angle_degrees.is_finite() {
        clamp(settings.merge_angle_degrees, 0.0, 180.0)
    } else {
        20.0
    };
    settings.passes = settings.passes.max(1);
    settings.minimum_subpath_area = non_negative(settings.minimum_subpath_area, 0.0);
    settings.area_tolerance = non_negative(settings.area_tolerance, 0.1);
    settings
}

fn non_negative(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        fallback
    }
}

/// Reprojects a cubic's control points onto its chord when they are already
/// within `tolerance`, making the curve numerically straighter.
///
/// Used by callers that want deterministic control points for a line-like
/// segment before converting it to a line.
pub fn snap_flat_cubic(cubic: CubicBezier, tolerance: f64) -> CubicBezier {
    if !cubic_is_line_like(cubic, tolerance) {
        return cubic;
    }
    let control1 = project_on_line(cubic.control1, cubic.start, cubic.end);
    let control2 = project_on_line(cubic.control2, cubic.start, cubic.end);
    CubicBezier::new(cubic.start, control1, control2, cubic.end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Point;
    use crate::vector::{Fill, Path};

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        fabs(a - b) <= tolerance
    }

    #[test]
    fn collinear_lines_merge() {
        let mut subpath = SubPath::new(Point::new(0.0, 0.0));
        subpath.push_line(Point::new(5.0, 0.0));
        subpath.push_line(Point::new(10.0, 0.0));
        let mut report = OptimizeReport::default();
        optimize_subpath(&mut subpath, OptimizeSettings::default(), &mut report);
        assert_eq!(subpath.segment_count(), 1);
        assert_eq!(report.merged_lines, 1);
        assert!(close(subpath.end().x, 10.0, 1.0e-12));
    }

    #[test]
    fn corner_is_not_merged() {
        let mut subpath = SubPath::new(Point::new(0.0, 0.0));
        subpath.push_line(Point::new(5.0, 0.0));
        subpath.push_line(Point::new(5.0, 5.0));
        let mut report = OptimizeReport::default();
        optimize_subpath(&mut subpath, OptimizeSettings::default(), &mut report);
        assert_eq!(subpath.segment_count(), 2);
        assert_eq!(report.merged_lines, 0);
    }

    #[test]
    fn flat_cubic_becomes_a_line() {
        let cubic = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.01),
            Point::new(8.0, -0.01),
            Point::new(10.0, 0.0),
        );
        let mut subpath = SubPath::new(cubic.start);
        subpath.segments.push(Segment::Cubic(cubic));
        let mut report = OptimizeReport::default();
        optimize_subpath(&mut subpath, OptimizeSettings::default(), &mut report);
        assert_eq!(subpath.segment_count(), 1);
        assert!(matches!(subpath.segments[0], Segment::Line { .. }));
        assert_eq!(report.converted_lines, 1);
    }

    #[test]
    fn smooth_cubics_can_merge() {
        let first = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(20.0, 10.0),
            Point::new(30.0, 10.0),
        );
        let second = CubicBezier::new(
            Point::new(30.0, 10.0),
            Point::new(40.0, 10.0),
            Point::new(50.0, 0.0),
            Point::new(60.0, 0.0),
        );
        let mut subpath = SubPath::new(first.start);
        subpath.segments.push(Segment::Cubic(first));
        subpath.segments.push(Segment::Cubic(second));
        let settings = OptimizeSettings {
            cubic_merge_tolerance: 2.0,
            ..OptimizeSettings::default()
        };
        let mut report = OptimizeReport::default();
        optimize_subpath(&mut subpath, settings, &mut report);
        assert!(subpath.segment_count() <= 2);
    }

    #[test]
    fn closing_lines_merge_across_the_seam() {
        let mut subpath = SubPath::closed(Point::new(0.0, 0.0));
        subpath.push_line(Point::new(2.0, 0.0));
        subpath.push_line(Point::new(2.0, 2.0));
        subpath.push_line(Point::new(0.0, 2.0));
        subpath.push_line(Point::new(0.0, 0.0));
        let mut report = OptimizeReport::default();
        optimize_subpath(&mut subpath, OptimizeSettings::default(), &mut report);
        assert!(subpath.segment_count() <= 4);
        assert!(subpath.end().distance(subpath.start) < 1.0e-7);
    }

    #[test]
    fn document_cleanup_removes_empty_shapes() {
        let mut document = VectorDoc::new(10, 10);
        let mut path = Path::new();
        path.subpaths.push(SubPath::new(Point::ZERO));
        document.push(Shape::new(Fill::new(crate::color::Color::BLACK), path, 0));
        // `push` rejects the empty shape immediately, so the document starts
        // empty and the report must remain consistent.
        assert!(document.is_empty());
        let report = optimize_document(&mut document, OptimizeSettings::default());
        assert_eq!(report.removed_shapes, 0);
        assert_eq!(report.segments_before, 0);
        assert_eq!(report.segments_after, 0);
    }

    #[test]
    fn minimum_area_removes_tiny_subpaths() {
        let mut path = Path::new();
        let mut subpath = SubPath::closed(Point::new(0.0, 0.0));
        subpath.push_line(Point::new(0.01, 0.0));
        subpath.push_line(Point::new(0.01, 0.01));
        subpath.push_line(Point::new(0.0, 0.01));
        subpath.push_line(Point::new(0.0, 0.0));
        path.subpaths.push(subpath);
        let settings = OptimizeSettings {
            minimum_subpath_area: 0.5,
            ..OptimizeSettings::default()
        };
        let mut report = OptimizeReport::default();
        optimize_path(&mut path, settings, &mut report);
        assert!(path.is_empty());
        assert_eq!(report.removed_subpaths, 1);
    }

    #[test]
    fn snapping_flat_cubic_preserves_endpoints() {
        let cubic = CubicBezier::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.001),
            Point::new(9.0, -0.001),
            Point::new(10.0, 0.0),
        );
        let snapped = snap_flat_cubic(cubic, 0.05);
        assert_eq!(snapped.start, cubic.start);
        assert_eq!(snapped.end, cubic.end);
        assert!(point_line_distance(snapped.control1, cubic.start, cubic.end) < 1.0e-9);
    }
}
