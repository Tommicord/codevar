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

//! Turns traced contours into fitted vector subpaths.
//!
//! The contract between contour extraction and curve fitting is a closed
//! polygon of lattice points. This module performs the remaining stages:
//!
//! 1. cyclic RDP simplification with a caller-selected line tolerance;
//! 2. corner classification by deflection angle;
//! 3. splitting at corners so each smooth run is fitted independently;
//! 4. least-squares cubic fitting inside each run;
//! 5. assembling the rings of one component into a single [`Path`].
//!
//! A component with no sharp corners is treated as a smooth ring: the seam
//! tangent is estimated from both sides, which avoids a visible join on circles
//! and rounded blobs. A component with corners keeps each corner as a hard
//! vertex, so rectangles, letters and icons stay crisp.

use alloc::vec::Vec;

use crate::bezier::{BezierFitOptions, fit_chain, fit_ring};
use crate::contour::Contour;
use crate::error::{Error, Result};
use crate::geom::Point;
use crate::simplify::{classify_corners, collect_run, simplify_ring, split_runs};
use crate::vector::{Path, Segment, SubPath};

/// How the fitter decides that a vertex is a corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CornerPolicy {
    /// Deflection above which a vertex is a corner, in degrees.
    ///
    /// A value of 70 keeps right angles and sharp icon vertices while treating
    /// the shallow vertices of a rasterized circle as a smooth run.
    pub threshold_degrees: f64,
    /// Minimum length of both adjacent polygon edges before a corner is
    /// considered. Zero preserves corners on one-pixel features.
    pub minimum_arm: f64,
}

impl Default for CornerPolicy {
    fn default() -> Self {
        Self {
            threshold_degrees: 70.0,
            minimum_arm: 0.0,
        }
    }
}

impl CornerPolicy {
    /// A policy that never classifies a corner.
    #[inline]
    pub const fn smooth_only() -> Self {
        Self {
            threshold_degrees: 180.0,
            minimum_arm: 0.0,
        }
    }

    /// A policy that treats every meaningful deflection as a corner.
    #[inline]
    pub const fn sharp() -> Self {
        Self {
            threshold_degrees: 25.0,
            minimum_arm: 0.0,
        }
    }

    /// Builds a corner policy after checking both parameters.
    ///
    /// Prefer this constructor over a struct literal when the values come from
    /// user input, a file or a UI control.
    pub fn try_new(threshold_degrees: f64, minimum_arm: f64) -> Result<Self> {
        let policy = Self {
            threshold_degrees,
            minimum_arm,
        };
        Self::try_from(&policy)
    }
}

impl TryFrom<&CornerPolicy> for CornerPolicy {
    type Error = Error;

    fn try_from(policy: &CornerPolicy) -> Result<Self> {
        if !policy.threshold_degrees.is_finite() || !(0.0..=180.0).contains(&policy.threshold_degrees) {
            return Err(Error::InvalidConfig {
                field: "corner_threshold_degrees",
            });
        }
        if !policy.minimum_arm.is_finite() || policy.minimum_arm < 0.0 {
            return Err(Error::InvalidConfig {
                field: "corner_minimum_arm",
            });
        }
        Ok(*policy)
    }
}

impl TryFrom<(f64, f64)> for CornerPolicy {
    type Error = Error;

    fn try_from((threshold_degrees, minimum_arm): (f64, f64)) -> Result<Self> {
        Self::try_new(threshold_degrees, minimum_arm)
    }
}

/// Settings for polygon simplification and curve fitting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitSettings {
    /// Maximum distance between the traced lattice boundary and the simplified
    /// polygon, in pixels.
    pub line_tolerance: f64,
    /// Corner classification policy.
    pub corners: CornerPolicy,
    /// Maximum distance between the fitted curve and the simplified polygon.
    pub curve_tolerance: f64,
    /// Flatness below which a cubic becomes a straight line.
    pub line_fit_tolerance: f64,
    /// Maximum recursive cubic subdivision depth.
    pub max_depth: u32,
    /// Whether to refine the parameterization before measuring the fit.
    pub reparameterize: bool,
    /// Subpaths whose absolute polygon area is below this value are discarded.
    pub minimum_area: f64,
}

impl Default for FitSettings {
    fn default() -> Self {
        Self {
            line_tolerance: 0.5,
            corners: CornerPolicy::default(),
            curve_tolerance: 0.45,
            line_fit_tolerance: 0.25,
            max_depth: 12,
            reparameterize: true,
            minimum_area: 0.0,
        }
    }
}

impl FitSettings {
    /// A setting tuned for line art: tighter tolerances and sharper corners.
    #[inline]
    pub const fn line_art() -> Self {
        Self {
            line_tolerance: 0.35,
            corners: CornerPolicy {
                threshold_degrees: 60.0,
                minimum_arm: 0.0,
            },
            curve_tolerance: 0.3,
            line_fit_tolerance: 0.15,
            max_depth: 14,
            reparameterize: true,
            minimum_area: 0.0,
        }
    }

    /// A setting tuned for photographs and painted artwork.
    #[inline]
    pub const fn painterly() -> Self {
        Self {
            line_tolerance: 0.9,
            corners: CornerPolicy {
                threshold_degrees: 85.0,
                minimum_arm: 0.0,
            },
            curve_tolerance: 0.8,
            line_fit_tolerance: 0.4,
            max_depth: 10,
            reparameterize: true,
            minimum_area: 0.0,
        }
    }

    /// Builds settings after checking every tolerance and the corner policy.
    ///
    /// The named presets ([`FitSettings::default`], [`FitSettings::line_art`]
    /// and [`FitSettings::painterly`]) already satisfy these invariants.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        line_tolerance: f64,
        corners: CornerPolicy,
        curve_tolerance: f64,
        line_fit_tolerance: f64,
        max_depth: u32,
        reparameterize: bool,
        minimum_area: f64,
    ) -> Result<Self> {
        let settings = Self {
            line_tolerance,
            corners,
            curve_tolerance,
            line_fit_tolerance,
            max_depth,
            reparameterize,
            minimum_area,
        };
        Self::try_from(&settings)
    }

    /// Converts the curve-related fields to the Bézier fitter's option type.
    #[inline]
    pub fn bezier_options(self) -> BezierFitOptions {
        BezierFitOptions {
            tolerance: self.curve_tolerance,
            line_tolerance: self.line_fit_tolerance,
            max_depth: self.max_depth,
            reparameterize: self.reparameterize,
        }
        .sanitized()
    }
}

impl TryFrom<&FitSettings> for FitSettings {
    type Error = Error;

    fn try_from(settings: &FitSettings) -> Result<Self> {
        if !settings.line_tolerance.is_finite() || settings.line_tolerance < 0.0 {
            return Err(Error::InvalidConfig {
                field: "line_tolerance",
            });
        }
        if !settings.curve_tolerance.is_finite() || settings.curve_tolerance <= 0.0 {
            return Err(Error::InvalidConfig {
                field: "curve_tolerance",
            });
        }
        if !settings.line_fit_tolerance.is_finite() || settings.line_fit_tolerance < 0.0 {
            return Err(Error::InvalidConfig {
                field: "line_fit_tolerance",
            });
        }
        if settings.max_depth == 0 {
            return Err(Error::InvalidConfig { field: "max_depth" });
        }
        if !settings.minimum_area.is_finite() || settings.minimum_area < 0.0 {
            return Err(Error::InvalidConfig {
                field: "minimum_area",
            });
        }
        CornerPolicy::try_from(&settings.corners)?;
        Ok(*settings)
    }
}

/// Fits a single closed contour.
///
/// Returns `None` when the contour is too small to produce a meaningful
/// subpath.
pub fn fit_contour(contour: &Contour, settings: FitSettings) -> Option<SubPath> {
    fit_contour_points(&contour.points, settings)
}

/// Fits a closed polygon into one subpath.
pub fn fit_contour_points(points: &[Point], settings: FitSettings) -> Option<SubPath> {
    if points.len() < 3 {
        return None;
    }
    let settings = sanitized(settings);

    let polygon = simplify_ring(points, settings.line_tolerance);
    if polygon.len() < 3 {
        return None;
    }
    let area = libm::fabs(crate::polygon::polygon_signed_area(&polygon));
    if area < settings.minimum_area {
        return None;
    }

    let corners = classify_corners(
        &polygon,
        settings.corners.threshold_degrees,
        settings.corners.minimum_arm,
    );
    let has_corner = corners.iter().any(|&corner| corner);
    let options = settings.bezier_options();

    let segments = if has_corner {
        let runs = split_runs(&corners);
        let mut fitted = Vec::new();
        for run in runs {
            let run_points = collect_run(&polygon, run);
            if run_points.len() < 2 {
                continue;
            }
            fitted.extend(fit_chain(&run_points, None, None, options));
        }
        fitted
    } else {
        fit_ring(&polygon, options)
    };

    if segments.is_empty() {
        return None;
    }
    assemble_subpath(&segments)
}

/// Fits every contour of a component into one compound path.
///
/// Outer rings and holes are kept together. The even-odd fill rule then treats
/// nested rings as holes without a separate containment step.
pub fn fit_contours(contours: &[Contour], settings: FitSettings) -> Path {
    let mut path = Path::new();
    for contour in contours {
        if let Some(subpath) = fit_contour(contour, settings) {
            path.subpaths.push(subpath);
        }
    }
    path
}

/// Fits contours and rejects a path that contains no meaningful geometry.
pub fn fit_contours_checked(contours: &[Contour], settings: FitSettings) -> Option<Path> {
    let path = fit_contours(contours, settings);
    if path.is_empty() { None } else { Some(path) }
}

fn assemble_subpath(segments: &[Segment]) -> Option<SubPath> {
    let first = *segments.first()?;
    if !first.start().is_finite() {
        return None;
    }
    let mut subpath = SubPath::closed(first.start());
    for &segment in segments {
        if !segment.is_finite() {
            return None;
        }
        if subpath.push_segment(segment) {
            continue;
        }
        // A fitted run can be separated from the previous run by a few ulps of
        // arithmetic. Bridging that gap with an explicit line is preferable to
        // emitting an invalid path; a genuinely discontinuous curve is rejected
        // because the bridge is only allowed to span an epsilon-sized gap.
        let gap = subpath.end().distance(segment.start());
        if gap > 1.0e-4 {
            return None;
        }
        subpath.push_line(segment.start());
        if !subpath.push_segment(segment) {
            return None;
        }
    }
    subpath.close();
    if subpath.segment_count() == 0 {
        return None;
    }
    Some(subpath)
}

fn sanitized(settings: FitSettings) -> FitSettings {
    let mut settings = settings;
    settings.line_tolerance = if settings.line_tolerance.is_finite() {
        settings.line_tolerance.max(0.0)
    } else {
        0.5
    };
    settings.curve_tolerance = if settings.curve_tolerance.is_finite() {
        settings.curve_tolerance.max(1.0e-4)
    } else {
        0.45
    };
    settings.line_fit_tolerance = if settings.line_fit_tolerance.is_finite() {
        settings.line_fit_tolerance.max(0.0)
    } else {
        0.25
    };
    settings.max_depth = settings.max_depth.clamp(1, 64);
    settings.corners.threshold_degrees = settings
        .corners
        .threshold_degrees
        .clamp(0.0, 180.0);
    settings.corners.minimum_arm = if settings.corners.minimum_arm.is_finite() {
        settings.corners.minimum_arm.max(0.0)
    } else {
        0.0
    };
    settings.minimum_area = if settings.minimum_area.is_finite() {
        settings.minimum_area.max(0.0)
    } else {
        0.0
    };
    settings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contour::Contour;

    fn square_points(size: f64) -> Vec<Point> {
        vec![
            Point::new(0.0, 0.0),
            Point::new(size, 0.0),
            Point::new(size, size),
            Point::new(0.0, size),
        ]
    }

    #[test]
    fn square_keeps_four_sharp_corners() {
        let contour = Contour::new(square_points(10.0));
        let subpath = fit_contour(&contour, FitSettings::default()).expect("fitted");
        assert!(subpath.closed);
        assert!(subpath.segment_count() >= 4);
        let area = subpath.signed_area(0.1);
        assert!(libm::fabs(area - 100.0) < 1.0);
        assert!(
            subpath
                .segments
                .iter()
                .all(|segment| matches!(segment, Segment::Line { .. }))
        );
    }

    #[test]
    fn circle_uses_cubics_and_closes() {
        let mut points = Vec::new();
        let sample_count = 48;
        for index in 0..sample_count {
            let angle = index as f64 * core::f64::consts::TAU / sample_count as f64;
            points.push(Point::new(
                libm::cos(angle) * 40.0 + 50.0,
                libm::sin(angle) * 40.0 + 50.0,
            ));
        }
        let contour = Contour::new(points);
        let subpath = fit_contour(&contour, FitSettings::default()).expect("fitted");
        assert!(subpath.segment_count() >= 2);
        assert!(
            subpath
                .segments
                .iter()
                .any(|segment| matches!(segment, Segment::Cubic(_)))
        );
        let area = subpath.signed_area(0.05);
        let expected = core::f64::consts::PI * 40.0 * 40.0;
        assert!(
            libm::fabs(area - expected) / expected < 0.03,
            "area {area} expected {expected}"
        );
        assert!(subpath.end().distance(subpath.start) < 1.0e-7);
    }

    #[test]
    fn contours_form_one_path_with_a_hole() {
        let outer = Contour::new(square_points(20.0));
        let mut hole_points = square_points(8.0);
        for point in &mut hole_points {
            *point += Point::new(6.0, 6.0);
        }
        hole_points.reverse();
        let hole = Contour::new(hole_points);
        let path = fit_contours(&[outer, hole], FitSettings::default());
        assert_eq!(path.len(), 2);
        let area = path.signed_area(0.05);
        assert!(libm::fabs(area - (400.0 - 64.0)) < 1.0);
    }

    #[test]
    fn degenerate_contours_are_dropped() {
        let contour = Contour {
            points: vec![Point::new(0.0, 0.0), Point::new(1.0, 0.0)],
            signed_area: 0.0,
            is_hole: false,
        };
        assert!(fit_contour(&contour, FitSettings::default()).is_none());
    }

    #[test]
    fn minimum_area_filters_small_rings() {
        let contour = Contour::new(square_points(1.0));
        let settings = FitSettings {
            minimum_area: 2.0,
            ..FitSettings::default()
        };
        assert!(fit_contour(&contour, settings).is_none());
    }

    #[test]
    fn sharp_policy_keeps_more_corners() {
        let mut points = Vec::new();
        for index in 0..16 {
            let angle = index as f64 * core::f64::consts::TAU / 16.0;
            let radius = if index % 2 == 0 { 20.0 } else { 10.0 };
            points.push(Point::new(
                libm::cos(angle) * radius + 30.0,
                libm::sin(angle) * radius + 30.0,
            ));
        }
        let sharp_flags = crate::simplify::classify_corners(
            &points,
            CornerPolicy::sharp().threshold_degrees,
            CornerPolicy::sharp().minimum_arm,
        );
        let smooth_flags = crate::simplify::classify_corners(
            &points,
            CornerPolicy::smooth_only().threshold_degrees,
            CornerPolicy::smooth_only().minimum_arm,
        );
        let sharp_corners = sharp_flags.iter().filter(|&&flag| flag).count();
        let smooth_corners = smooth_flags.iter().filter(|&&flag| flag).count();
        assert!(
            sharp_corners > smooth_corners,
            "sharp {sharp_corners}, smooth {smooth_corners}"
        );

        // Both policies must still produce a closed, finite subpath.
        let contour = Contour::new(points);
        for corners in [CornerPolicy::smooth_only(), CornerPolicy::sharp()] {
            let subpath = fit_contour(
                &contour,
                FitSettings {
                    corners,
                    ..FitSettings::default()
                },
            )
            .expect("fitted subpath");
            assert!(subpath.closed);
            assert!(subpath.is_finite());
        }
    }
}
