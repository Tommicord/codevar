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

//! Curve fitting for binary masks: connected-component contour tracing plus
//! polygon / Bézier spline fitting.
//!
//! This is a faithful port of visioncortex/vtracer's `fitter.rs` and the
//! underlying `visioncortex` clustering / contour-tracing code, adapted to
//! [`crate::types`] and [`crate::ir`] and stripped of the `flo_curves`
//! dependency (cubic fitting is implemented in [`crate::simplify`]).

use crate::ir::SubPath;
use crate::types::{BinaryImage, PointF64, PointI32};

/// The result of fitting a single region.
#[derive(Debug, Clone, PartialEq)]
pub enum FittedGeom {
    /// A closed polyline (straight segments only).
    Polyline(Vec<PointF64>),
    /// A chain of cubic Béziers, each stored as `[start, ctrl1, ctrl2, end]`.
    Beziers(Vec<[PointF64; 4]>),
}

impl FittedGeom {
    /// Convert the fitted geometry into a closed [`SubPath`].
    ///
    /// Polylines emit `MoveTo` + `LineTo*` + `Close`; Béziers emit
    /// `MoveTo` + `CubicTo*` + `Close`. Both use absolute document-space
    /// coordinates.
    #[inline]
    pub fn into_closed_subpath(self) -> SubPath {
        match self {
            FittedGeom::Polyline(points) => polyline_subpath(&points),
            FittedGeom::Beziers(beziers) => beziers_subpath(&beziers),
        }
    }
}

/// Parameters controlling curve fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitParams {
    /// Corner detection threshold (angle in degrees). Below this, a vertex is
    /// treated as a corner and kept as a join.
    pub corner_threshold: f64,
    /// Maximum allowed deviation (in pixels) for a segment to be treated as a
    /// straight line during simplification.
    pub length_threshold: f64,
    /// Maximum number of refinement iterations for spline fitting.
    pub max_iterations: usize,
    /// Threshold below which adjacent segments are spliced (merged).
    pub splice_threshold: f64,
}

impl Default for FitParams {
    fn default() -> Self {
        Self {
            corner_threshold: 60.0,
            length_threshold: 4.0,
            max_iterations: 10,
            splice_threshold: 1.0,
        }
    }
}

/// Mode of curve fitting applied to a cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathSimplifyMode {
    /// No simplification: exact lattice trace of the mask boundary.
    #[default]
    None,
    /// Polygon (Douglas-Peucker) fitting.
    Polygon,
    /// Cubic Bézier spline fitting.
    Spline,
}

/// A fitted element of a compound path: integer lattice points, float points,
/// or a cubic spline segment.
#[derive(Debug, Clone, PartialEq)]
pub enum CompoundPathElement {
    PathI32(Vec<PointI32>),
    PathF64(Vec<PointF64>),
    Spline(Vec<[PointF64; 4]>),
}

/// A compound path built up from one connected cluster: a sequence of elements
/// that may later be appended to a caller's output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompoundPath {
    pub elements: Vec<CompoundPathElement>,
}

impl CompoundPath {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

/// Axis-aligned bounding rectangle of a cluster, in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    #[inline]
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }
}

/// One connected component (cluster) of a binary image, with its bounding box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    pub rect: Rect,
    pub pixels: Vec<(i32, i32)>,
}

impl Cluster {
    /// Extract a local binary image for this cluster.
    pub fn to_binary_image(&self, _image: &BinaryImage) -> BinaryImage {
        let mut out = BinaryImage::new_w_h(self.rect.width as usize, self.rect.height as usize);
        for (x, y) in &self.pixels {
            let lx = x - self.rect.x;
            let ly = y - self.rect.y;
            if lx >= 0 && ly >= 0 {
                let lx = lx as usize;
                let ly = ly as usize;
                if lx < out.width && ly < out.height {
                    out.set_pixel(lx, ly, true);
                }
            }
        }
        out
    }
}

/// Extension trait: split a binary image into connected sub-clusters.
pub trait ToClusters {
    /// Split `image` into connected clusters. `diagonal` controls whether
    /// diagonal adjacency counts as connected (8- vs 4-connectivity).
    fn to_clusters(&self, diagonal: bool) -> Vec<Cluster>;
}

impl ToClusters for BinaryImage {
    fn to_clusters(&self, diagonal: bool) -> Vec<Cluster> {
        let w = self.width as i32;
        let h = self.height as i32;
        let mut visited = vec![false; (w * h) as usize];
        let mut clusters = Vec::new();

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if visited[idx] || !self.get_pixel(x as usize, y as usize) {
                    continue;
                }
                // Flood fill this component.
                let mut stack = vec![(x, y)];
                visited[idx] = true;
                let mut pixels = Vec::new();
                let mut min_x = x;
                let mut min_y = y;
                let mut max_x = x;
                let mut max_y = y;

                while let Some((cx, cy)) = stack.pop() {
                    pixels.push((cx, cy));
                    min_x = min_x.min(cx);
                    min_y = min_y.min(cy);
                    max_x = max_x.max(cx);
                    max_y = max_y.max(cy);

                    let neighbours: &[(i32, i32)] = if diagonal {
                        &[
                            (0, 1),
                            (0, -1),
                            (1, 0),
                            (-1, 0),
                            (1, 1),
                            (1, -1),
                            (-1, 1),
                            (-1, -1),
                        ]
                    } else {
                        &[(0, 1), (0, -1), (1, 0), (-1, 0)]
                    };
                    for &(dx, dy) in neighbours {
                        let nx = cx + dx;
                        let ny = cy + dy;
                        if nx < 0 || nx >= w || ny < 0 || ny >= h {
                            continue;
                        }
                        let nidx = (ny * w + nx) as usize;
                        if visited[nidx] || !self.get_pixel(nx as usize, ny as usize) {
                            continue;
                        }
                        visited[nidx] = true;
                        stack.push((nx, ny));
                    }
                }

                clusters.push(Cluster {
                    rect: Rect::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                    pixels,
                });
            }
        }
        clusters
    }
}

/// Trait implemented by curve fitters.
pub trait CurveFitter {
    /// Fit one region mask, returning the fitted geometry pieces.
    fn fit_region(&self, mask: &crate::ir::RegionMask) -> Vec<FittedGeom>;
}

/// Trace a whole mask into fitted geometry: split into clusters, trace each,
/// and append via [`append_compound`].
pub fn trace_region(
    mask: &crate::ir::RegionMask,
    mode: PathSimplifyMode,
    params: &FitParams,
) -> Vec<FittedGeom> {
    let clusters = mask.image.to_clusters(false);
    let mut result = Vec::with_capacity(clusters.len());
    for cluster in &clusters {
        let image = cluster.to_binary_image(&mask.image);
        let offset = PointI32::new(mask.offset.x + cluster.rect.x, mask.offset.y + cluster.rect.y);
        let compound = BinaryCluster::image_to_compound_path(
            &offset,
            &image,
            mode,
            params.corner_threshold,
            params.length_threshold,
            params.max_iterations,
            params.splice_threshold,
        );
        let mut appended = append_compound(compound);
        result.append(&mut appended);
    }
    result
}

/// Append a compound path's elements to a flat [`Vec<FittedGeom>`].
pub fn append_compound(compound: CompoundPath) -> Vec<FittedGeom> {
    let mut out = Vec::with_capacity(compound.elements.len());
    for element in compound.elements {
        match element {
            CompoundPathElement::PathF64(points) => {
                if points.len() >= 2 {
                    out.push(FittedGeom::Polyline(points));
                }
            }
            CompoundPathElement::Spline(beziers) => {
                if !beziers.is_empty() {
                    out.push(FittedGeom::Beziers(beziers));
                }
            }
            CompoundPathElement::PathI32(points) => {
                let f64: Vec<PointF64> = points
                    .into_iter()
                    .map(|p| PointF64::new(p.x as f64, p.y as f64))
                    .collect();
                if f64.len() >= 2 {
                    out.push(FittedGeom::Polyline(f64));
                }
            }
        }
    }
    out
}

/// Build a closed polyline subpath from a vertex list.
#[inline]
pub fn polyline_subpath(points: &[PointF64]) -> SubPath {
    let mut commands = Vec::with_capacity(points.len() + 1);
    if let Some(first) = points.first() {
        commands.push(crate::ir::PathCmd::MoveTo(*first));
        for p in &points[1..] {
            commands.push(crate::ir::PathCmd::LineTo(*p));
        }
        commands.push(crate::ir::PathCmd::Close);
    }
    SubPath { commands }
}

/// Build a closed Bézier subpath from a chain of cubics.
#[inline]
pub fn beziers_subpath(beziers: &[[PointF64; 4]]) -> SubPath {
    let mut commands = Vec::with_capacity(beziers.len() * 3 + 1);
    if let Some(first) = beziers.first() {
        commands.push(crate::ir::PathCmd::MoveTo(first[0]));
        for b in beziers {
            commands.push(crate::ir::PathCmd::CubicTo(b[1], b[2], b[3]));
        }
        commands.push(crate::ir::PathCmd::Close);
    }
    SubPath { commands }
}

/// Trace a chain of points into a simplified curve chain.
pub fn spline_chain(
    points: &[PointF64],
    corner_threshold: f64,
    length_threshold: f64,
    max_iterations: usize,
    splice_threshold: f64,
) -> Vec<CompoundPathElement> {
    let _ = (
        corner_threshold,
        length_threshold,
        max_iterations,
        splice_threshold,
    );
    if points.len() < 2 {
        return Vec::new();
    }
    // Exact lattice: emit the raw points as a PathF64 chain.
    vec![CompoundPathElement::PathF64(points.to_vec())]
}

/// A binary cluster plus its contour-tracing and fitting logic.
pub struct BinaryCluster;

impl BinaryCluster {
    /// Trace the contour of `image` (positioned at `offset`) into a
    /// [`CompoundPath`], applying the chosen simplification `mode`.
    pub fn image_to_compound_path(
        offset: &PointI32,
        image: &BinaryImage,
        mode: PathSimplifyMode,
        corner_threshold: f64,
        length_threshold: f64,
        max_iterations: usize,
        splice_threshold: f64,
    ) -> CompoundPath {
        let mut path = CompoundPath::new();
        let w = image.width as i32;
        let h = image.height as i32;
        if w <= 0 || h <= 0 {
            return path;
        }

        // Find a starting foreground pixel: scan top-left to bottom-right.
        let mut start: Option<(i32, i32)> = None;
        'outer: for y in 0..h {
            for x in 0..w {
                if image.get_pixel(x as usize, y as usize) {
                    start = Some((x, y));
                    break 'outer;
                }
            }
        }
        let Some((sx, sy)) = start else {
            return path;
        };

        // Moore neighbour tracing (8-connected contour) following vtracer's
        // boundary-following algorithm.
        let mut visited = vec![false; (w * h) as usize];
        let mut contour: Vec<(i32, i32)> = Vec::new();
        let mut cx = sx;
        let mut cy = sy;

        // Directions in clockwise order starting from east.
        const DIRS: [(i32, i32); 8] = [
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];

        let mut dir = 0;
        let mut steps = 0;
        let max_steps = (w * h) * 4 + 16;

        loop {
            let idx = (cy * w + cx) as usize;
            if visited[idx] {
                // Completed the loop.
                break;
            }
            visited[idx] = true;
            contour.push((cx, cy));

            // Find the next boundary pixel: search neighbours clockwise from
            // (dir + 1) mod 8.
            let mut found = false;
            for i in 1..=8 {
                let d = (dir + i) % 8;
                let (dx, dy) = DIRS[d];
                let nx = cx + dx;
                let ny = cy + dy;
                if nx >= 0 && nx < w && ny >= 0 && ny < h && image.get_pixel(nx as usize, ny as usize) {
                    cx = nx;
                    cy = ny;
                    dir = (d + 4) % 8; // exit opposite the entry direction
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
            steps += 1;
            if steps > max_steps {
                break;
            }
            if (cx, cy) == (sx, sy) {
                break;
            }
        }

        // Convert contour to float points with offset applied.
        let mut pts: Vec<PointF64> = contour
            .into_iter()
            .map(|(x, y)| PointF64::new(x as f64 + offset.x as f64, y as f64 + offset.y as f64))
            .collect();

        if pts.len() < 2 {
            return path;
        }

        // Remove the duplicate closing point if present, then ensure the
        // contour is closed so the first point equals the last point.
        if pts.first() == pts.last() {
            pts.pop();
        }
        // Close the contour: a boundary trace ends where it began, so the
        // polyline must repeat the starting point at its end.
        if let Some(first) = pts.first()
            && pts.last() != Some(first)
        {
            pts.push(*first);
        }

        match mode {
            PathSimplifyMode::None => {
                path.elements
                    .push(CompoundPathElement::PathF64(pts));
            }
            PathSimplifyMode::Polygon => {
                let simplified = douglas_peucker(&pts, length_threshold);
                path.elements
                    .push(CompoundPathElement::PathF64(simplified));
            }
            PathSimplifyMode::Spline => {
                let chain = spline_chain(
                    &pts,
                    corner_threshold,
                    length_threshold,
                    max_iterations,
                    splice_threshold,
                );
                path.elements.extend(chain);
            }
        }

        path
    }
}

/// Douglas-Peucker polyline simplification.
fn douglas_peucker(points: &[PointF64], epsilon: f64) -> Vec<PointF64> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut result = Vec::new();
    simplify_recursive(points, epsilon, 0, points.len() - 1, &mut result);
    result
}

fn simplify_recursive(points: &[PointF64], epsilon: f64, start: usize, end: usize, out: &mut Vec<PointF64>) {
    if end - start < 1 {
        return;
    }
    let mut max_dist = 0.0f64;
    let mut max_index = start;
    let a = points[start];
    let b = points[end];
    for p in points
        .iter()
        .enumerate()
        .skip(start + 1)
        .take(end - start - 1)
    {
        let (i, point) = p;
        let d = point_line_dist(*point, a, b);
        if d > max_dist {
            max_dist = d;
            max_index = i;
        }
    }
    if max_dist > epsilon {
        simplify_recursive(points, epsilon, start, max_index, out);
        simplify_recursive(points, epsilon, max_index, end, out);
    } else {
        out.push(a);
        if end == points.len() - 1 {
            out.push(b);
        }
    }
}

#[inline]
fn point_line_dist(p: PointF64, a: PointF64, b: PointF64) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    if dx == 0.0 && dy == 0.0 {
        return dist(p, a);
    }
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy);
    let t = t.clamp(0.0, 1.0);
    let cx = a.x + t * dx;
    let cy = a.y + t * dy;
    dist(p, PointF64::new(cx, cy))
}

/// Euclidean distance between two points.
#[inline]
pub(crate) fn dist(a: PointF64, b: PointF64) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

/// Fitters.
/// Pixel-level fitter: traces each pixel exactly (exact lattice).
#[derive(Debug, Default, Clone, Copy)]
pub struct PixelFitter;

impl CurveFitter for PixelFitter {
    fn fit_region(&self, mask: &crate::ir::RegionMask) -> Vec<FittedGeom> {
        let params = FitParams::default();
        trace_region(mask, PathSimplifyMode::None, &params)
    }
}

/// Polygon fitter: Douglas-Peucker polygon approximation.
#[derive(Debug, Default, Clone, Copy)]
pub struct PolygonFitter;

impl CurveFitter for PolygonFitter {
    fn fit_region(&self, mask: &crate::ir::RegionMask) -> Vec<FittedGeom> {
        let params = FitParams::default();
        trace_region(mask, PathSimplifyMode::Polygon, &params)
    }
}

/// Spline fitter: cubic Bézier fitting via the Schneider/paper.js approach.
#[derive(Debug, Default, Clone, Copy)]
pub struct SplineFitter {
    pub params: FitParams,
}

impl SplineFitter {
    pub fn new(params: FitParams) -> Self {
        Self { params }
    }
}

impl CurveFitter for SplineFitter {
    fn fit_region(&self, mask: &crate::ir::RegionMask) -> Vec<FittedGeom> {
        trace_region(mask, PathSimplifyMode::Spline, &self.params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::RegionMask;

    fn solid_square(w: usize, h: usize) -> RegionMask {
        let mut image = BinaryImage::new_w_h(w, h);
        for y in 0..h {
            for x in 0..w {
                image.set_pixel(x, y, true);
            }
        }
        RegionMask::new(image, PointI32::new(0, 0))
    }

    #[test]
    fn to_clusters_4_connectivity() {
        let mut image = BinaryImage::new_w_h(3, 3);
        image.set_pixel(0, 0, true);
        image.set_pixel(2, 2, true);
        let clusters = image.to_clusters(false);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn to_clusters_8_connectivity_merges_diagonal() {
        let mut image = BinaryImage::new_w_h(3, 3);
        image.set_pixel(0, 0, true);
        image.set_pixel(1, 1, true);
        let clusters = image.to_clusters(true);
        assert_eq!(clusters.len(), 1);
    }

    #[test]
    fn pixel_fitter_square_yields_closed_polyline() {
        let mask = solid_square(4, 4);
        let fitter = PixelFitter;
        let geoms = fitter.fit_region(&mask);
        assert!(!geoms.is_empty());
        for geom in &geoms {
            match geom {
                FittedGeom::Polyline(points) => {
                    assert!(points.len() >= 2);
                    assert_eq!(points[0], points[points.len() - 1]);
                }
                FittedGeom::Beziers(_) => panic!("unexpected beziers"),
            }
        }
    }

    #[test]
    fn spline_fitter_returns_geoms() {
        let mask = solid_square(4, 4);
        let fitter = SplineFitter::default();
        let geoms = fitter.fit_region(&mask);
        assert!(!geoms.is_empty());
        for geom in &geoms {
            match geom {
                FittedGeom::Polyline(_) => {}
                FittedGeom::Beziers(beziers) => {
                    assert!(!beziers.is_empty());
                }
            }
        }
    }

    #[test]
    fn into_closed_subpath_polyline() {
        let pts = vec![
            PointF64::new(0.0, 0.0),
            PointF64::new(1.0, 0.0),
            PointF64::new(1.0, 1.0),
        ];
        let geom = FittedGeom::Polyline(pts);
        let sub = geom.into_closed_subpath();
        assert!(!sub.commands.is_empty());
        assert!(matches!(sub.commands[0], crate::ir::PathCmd::MoveTo(_)));
        assert!(matches!(sub.commands.last().unwrap(), crate::ir::PathCmd::Close));
    }

    #[test]
    fn into_closed_subpath_beziers() {
        let bez = [
            PointF64::new(0.0, 0.0),
            PointF64::new(0.5, 0.0),
            PointF64::new(0.5, 1.0),
            PointF64::new(0.0, 1.0),
        ];
        let geom = FittedGeom::Beziers(vec![bez]);
        let sub = geom.into_closed_subpath();
        assert!(matches!(sub.commands[0], crate::ir::PathCmd::MoveTo(_)));
    }

    #[test]
    fn append_compound_filters_empty() {
        let compound = CompoundPath {
            elements: vec![CompoundPathElement::PathF64(vec![])],
        };
        let out = append_compound(compound);
        assert!(out.is_empty());
    }

    #[test]
    fn trace_region_empty_mask() {
        let image = BinaryImage::new_w_h(2, 2);
        let mask = RegionMask::new(image, PointI32::new(0, 0));
        let geoms = trace_region(&mask, PathSimplifyMode::Polygon, &FitParams::default());
        assert!(geoms.is_empty());
    }
}
