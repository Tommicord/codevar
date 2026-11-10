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

//! Segment fitters for the mosaic: fit one shared boundary segment into a
//! [`FittedGeom`](crate::fitter::FittedGeom), shared by both adjacent faces.

use crate::fitter::{CurveFitter, FitParams, FittedGeom, PixelFitter, PolygonFitter, SplineFitter};
use crate::ir::{Paint, RegionMask};
use crate::types::PointI32;

/// Fit a single shared boundary segment into vector geometry.
///
/// The mosaic strategy fits each shared segment exactly once and lets both
/// adjacent faces reference the result, so no seam ever appears between them.
pub trait SegmentFitter: Send + Sync {
    /// Fit `segment` (pixel-space points in boundary order) against `mask`,
    /// painting the result with `paint`. Returns one fitted geometry.
    fn fit_segment(&self, mask: &RegionMask, segment: &[PointI32], paint: Paint) -> FittedGeom;
}

/// Fit each segment with the pixel (point-sample) fitter: the boundary is
/// emitted as a polyline through every segment point.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PixelSegmentFitter;

impl SegmentFitter for PixelSegmentFitter {
    fn fit_segment(&self, mask: &RegionMask, segment: &[PointI32], _paint: Paint) -> FittedGeom {
        let fitter = PixelFitter;
        // Pixel fitter works on a mask; build a tiny mask covering the segment's
        // bounding box with just these pixels marked, then fit.
        let submask = segment_mask(mask, segment);
        let geoms = fitter.fit_region(&submask);
        geoms
            .into_iter()
            .next()
            .unwrap_or_else(|| FittedGeom::Polyline(Vec::new()))
    }
}

/// Fit each segment with the polygon fitter: approximate the boundary by a
/// closed polygon through the segment points.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PolygonSegmentFitter;

impl SegmentFitter for PolygonSegmentFitter {
    fn fit_segment(&self, mask: &RegionMask, segment: &[PointI32], _paint: Paint) -> FittedGeom {
        let fitter = PolygonFitter;
        let submask = segment_mask(mask, segment);
        let geoms = fitter.fit_region(&submask);
        geoms
            .into_iter()
            .next()
            .unwrap_or_else(|| FittedGeom::Polyline(Vec::new()))
    }
}

/// Fit each segment with the spline fitter: smooth the boundary into cubic
/// Bézier splines.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SplineSegmentFitter;

impl SegmentFitter for SplineSegmentFitter {
    fn fit_segment(&self, mask: &RegionMask, segment: &[PointI32], _paint: Paint) -> FittedGeom {
        let fitter = SplineFitter::new(FitParams::default());
        let submask = segment_mask(mask, segment);
        let geoms = fitter.fit_region(&submask);
        geoms
            .into_iter()
            .next()
            .unwrap_or_else(|| FittedGeom::Polyline(Vec::new()))
    }
}

/// Build a small mask (local to the segment's bounding box) marking exactly
/// the segment's pixels, so a region fitter can be reused for segment fitting.
fn segment_mask(mask: &RegionMask, segment: &[PointI32]) -> RegionMask {
    if segment.is_empty() {
        return RegionMask::new(
            crate::types::BinaryImage::new_w_h(0, 0),
            crate::types::PointI32::new(0, 0),
        );
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for p in segment {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    let width = (max_x - min_x + 1) as usize;
    let height = (max_y - min_y + 1) as usize;
    let mut image = crate::types::BinaryImage::new_w_h(width, height);
    let origin = mask.offset;
    for p in segment {
        // Convert absolute pixel to mask-local, then to submask-local.
        let lx = (p.x - origin.x) as usize;
        let ly = (p.y - origin.y) as usize;
        if lx < mask.image.width && ly < mask.image.height && mask.image.get_pixel(lx, ly) {
            let sx = (p.x - min_x) as usize;
            let sy = (p.y - min_y) as usize;
            image.set_pixel(sx, sy, true);
        }
    }
    RegionMask::new(image, crate::types::PointI32::new(min_x, min_y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BinaryImage, Color, PointI32};

    fn mask(w: usize, h: usize, on: &[(usize, usize)], ox: i32, oy: i32) -> RegionMask {
        let mut image = BinaryImage::new_w_h(w, h);
        for &(x, y) in on {
            image.set_pixel(x, y, true);
        }
        RegionMask::new(image, PointI32::new(ox, oy))
    }

    #[test]
    fn pixel_fitter_produces_geometry() {
        let fitter = PixelSegmentFitter;
        let mask = mask(4, 4, &[(1, 1), (2, 1), (1, 2), (2, 2)], 0, 0);
        let seg = &[
            PointI32::new(1, 1),
            PointI32::new(2, 1),
            PointI32::new(2, 2),
            PointI32::new(1, 2),
        ];
        let _geom = fitter.fit_segment(&mask, seg, Paint::Solid(Color::new(1, 2, 3)));
    }

    #[test]
    fn empty_segment_yields_default() {
        let fitter = PolygonSegmentFitter;
        let mask = mask(2, 2, &[], 0, 0);
        let _geom = fitter.fit_segment(&mask, &[], Paint::Solid(Color::new(0, 0, 0)));
    }

    #[test]
    fn spline_fitter_produces_geometry() {
        let fitter = SplineSegmentFitter;
        let mask = mask(4, 4, &[(1, 1), (2, 1), (1, 2), (2, 2)], 0, 0);
        let seg = &[PointI32::new(1, 1), PointI32::new(2, 1)];
        let _geom = fitter.fit_segment(&mask, seg, Paint::Solid(Color::new(9, 9, 9)));
    }
}
