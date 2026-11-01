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

//! Convex vertex bounds and the exact distance from a point to them.
//!
//! A bound is a convex polygon of at most [`MAX_VERTICES`] vertices in
//! logical pixels. Containment is a winding test, so the vertex order does
//! not matter; the distance is measured against the edge segments, which is
//! what makes a point beyond a corner get the corner distance rather than
//! the distance to the edge line.
//!
//! The vertex bound is what a rounded rectangle costs: [`CORNER_SEGMENTS`]
//! arc segments at each of the four corners is twenty edges, and a rounded
//! rectangle has no other edges, so twenty vertices describe the shape a
//! widget with rounded corners actually paints. A shape that needs more is
//! rejected rather than grown onto the heap, because the bound is embedded
//! in every widget's hit geometry and is copied by the picking code.
//!
//! Raw vertices handed to [`VertexBounds::new`] are put through a convex
//! hull first, so collinear and interior points drop out and a non-convex
//! list cannot make the winding test lie.

use libm::{cosf, sinf, sqrtf};

use crate::geometry::{Offset, Rect};
use crate::paint::CornerRadii;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use codevar_base::cpuid::{self, Feature};

/// The greatest number of vertices one shape may have.
///
/// Twenty is four corners times the five points a [`CORNER_SEGMENTS`] arc
/// contributes, which is exactly the rounded rectangle the widget tree
/// paints; a plain side between two corners is the edge the last arc point
/// and the next first point already form. Shapes needing more vertices are
/// rejected instead of heap-allocated.
pub const MAX_VERTICES: usize = 20;

/// The number of arc segments sampled at each rounded corner.
///
/// Four segments put the sampled arc within a fraction of a pixel of the
/// true circle for the corner radii widgets use, which is below the size of
/// a pointer target, and keeps a fully rounded rectangle inside
/// [`MAX_VERTICES`].
pub const CORNER_SEGMENTS: usize = 4;

/// The number of edges one batch holds, which is also the widest lane count
/// a kernel is given.
const LANES: usize = 8;

/// The relative tolerance below which a winding cross product is float noise.
///
/// [`VertexBounds::contains`] compares `side * side` against this times the
/// squared edge length and the squared distance to the point, which is the
/// same as calling a perpendicular distance under one hundred-thousandth of
/// the lever arm zero — far below a pixel at any widget size, and enough to
/// absorb the rounding of an arc sampled with `cosf` and `sinf`.
const CROSS_NOISE: f32 = 1e-10;

/// The shortest edge a shape keeps.
///
/// Two vertices closer than this form an edge whose direction is float noise
/// rather than geometry: the arcs of a fully rounded rectangle meet exactly
/// where one corner stops and the next begins, and a micro-edge like that
/// makes both the winding test and the segment distance unstable, because a
/// point far along its (arbitrary) line reads as being on the wrong side of
/// it. [`VertexBounds::from_vertices`] drops the second vertex of such a
/// pair, which is also what keeps `distance_sq` from measuring an edge no
/// renderer draws.
const MIN_EDGE: f32 = 1e-3;

/// One batch of edges and the point measured against them.
///
/// The coordinates are kept in parallel arrays so a kernel can load one
/// vector per coordinate instead of gathering from many `Offset`s. A batch
/// holds at most [`LANES`] edges; the lanes past the real ones repeat an edge
/// the batch already measures, which keeps the minimum unchanged and every
/// lane finite — `min` does not propagate `NaN` consistently across kernels.
struct Edges {
    /// The point the distance is measured from.
    point: Offset,
    /// First endpoint, horizontal.
    ax: [f32; LANES],
    /// First endpoint, vertical.
    ay: [f32; LANES],
    /// Second endpoint, horizontal.
    bx: [f32; LANES],
    /// Second endpoint, vertical.
    by: [f32; LANES],
    /// Number of real edges in the batch, from one up to [`LANES`].
    count: usize,
}

/// Identifier a widget uses to name one of its shapes.
///
/// The identifier is opaque to the rasterizer: it only travels back in a
/// hit so the widget can map the hit to its own handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoundId(u32);

impl BoundId {
    /// Creates an identifier from a widget-defined value.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// The widget-defined value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A convex polygon of up to [`MAX_VERTICES`] vertices, in logical pixels.
///
/// The vertices may be given clockwise or counter-clockwise; containment is
/// tested from the winding of the first non-collinear edge, so both orders
/// work. A quadrilateral gives an exact shape for a rotated or skewed box,
/// a rounded rectangle gives the shape a widget with rounded corners paints,
/// and a rectangle is the common case with a constructor and a fast path of
/// its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexBounds {
    /// Identifier returned when the pointer hits this shape.
    id: BoundId,
    /// The vertices, padded with the last real vertex.
    vertices: [Offset; MAX_VERTICES],
    /// Number of real vertices, from three up to [`MAX_VERTICES`].
    count: u8,
    /// Axis-aligned box that encloses every vertex.
    bound: Rect,
    /// Whether the shape is exactly the four corners of [`Self::bound`].
    ///
    /// A plain shape is the common case, and its containment is a rectangle
    /// test rather than a walk over the edges.
    plain: bool,
}

impl VertexBounds {
    /// A convex shape with `id` built from three or more vertices.
    ///
    /// The convex hull of `vertices` becomes the shape, so interior and
    /// collinear points drop out and a non-convex list is accepted. Returns
    /// `None` when the list has fewer than three vertices or more than
    /// [`MAX_VERTICES`], holds a non-finite coordinate, or has a hull with
    /// fewer than three corners, which is what an all-collinear or
    /// all-identical list produces.
    #[must_use]
    pub fn new(id: BoundId, vertices: &[Offset]) -> Option<Self> {
        if !(3..=MAX_VERTICES).contains(&vertices.len()) {
            return None;
        }
        if !vertices
            .iter()
            .all(|v| v.x.is_finite() && v.y.is_finite())
        {
            return None;
        }
        let (hull, count) = convex_hull(vertices)?;
        Some(Self::from_vertices(id, &hull[..count]))
    }

    /// A rectangular shape covering `rect`.
    #[must_use]
    pub fn rect(id: BoundId, rect: Rect) -> Self {
        let vertices = [
            rect.origin,
            Offset::new(rect.right(), rect.top()),
            Offset::new(rect.right(), rect.bottom()),
            Offset::new(rect.left(), rect.bottom()),
        ];
        Self {
            id,
            vertices: pad(&vertices),
            count: 4,
            bound: rect,
            plain: true,
        }
    }

    /// A triangular shape.
    ///
    /// The three vertices are taken as given; use [`Self::new`] to have a
    /// hull built when the list may be degenerate.
    #[must_use]
    pub fn triangle(id: BoundId, vertices: [Offset; 3]) -> Self {
        let mut padded = [vertices[2]; 4];
        padded[..3].copy_from_slice(&vertices);
        Self {
            id,
            vertices: pad(&padded),
            count: 3,
            bound: bounds_of(&vertices),
            plain: false,
        }
    }

    /// A quadrilateral shape.
    ///
    /// The four vertices are taken as given and must form a convex polygon;
    /// use [`Self::new`] when they may not. A quadrilateral that happens to
    /// be the axis-aligned box around itself keeps the rectangle fast path.
    #[must_use]
    pub fn quad(id: BoundId, vertices: [Offset; 4]) -> Self {
        let bound = bounds_of(&vertices);
        let plain = is_rect(&vertices, bound);
        Self {
            id,
            vertices: pad(&vertices),
            count: 4,
            bound,
            plain,
        }
    }

    /// A rectangle with rounded corners, the shape a widget with a rounded
    /// clip, background or border paints.
    ///
    /// Each corner is sampled with [`CORNER_SEGMENTS`] arc segments, so the
    /// shape has at most [`MAX_VERTICES`] vertices; a corner with no radius
    /// contributes its single corner point instead and makes the shape
    /// smaller. A radius larger than half the shorter side is scaled down the
    /// way a renderer scales it, so no arc crosses its neighbour. Returns
    /// `None` only when the rectangle has no area, because then there is no
    /// shape to hit.
    #[must_use]
    pub fn rounded_rect(id: BoundId, rect: Rect, radii: CornerRadii) -> Option<Self> {
        if rect.size.width <= 0.0 || rect.size.height <= 0.0 {
            return None;
        }
        let limit = (rect.size.width * 0.5)
            .min(rect.size.height * 0.5)
            .max(0.0);
        let radius = [
            radii.top_left.min(limit).max(0.0),
            radii.top_right.min(limit).max(0.0),
            radii.bottom_right.min(limit).max(0.0),
            radii.bottom_left.min(limit).max(0.0),
        ];
        if radius.iter().all(|value| *value <= 0.0) {
            return Some(Self::rect(id, rect));
        }

        // Each corner is its arc center and the angle the arc starts at, in
        // screen coordinates where the y axis points down. Walking the four
        // corners in this order goes around the rectangle clockwise, and the
        // straight side between two corners is the edge between the last
        // point of one arc and the first point of the next.
        let corners = [
            (
                radius[0],
                Offset::new(rect.left() + radius[0], rect.top() + radius[0]),
                core::f32::consts::PI,
            ),
            (
                radius[1],
                Offset::new(rect.right() - radius[1], rect.top() + radius[1]),
                3.0 * core::f32::consts::FRAC_PI_2,
            ),
            (
                radius[2],
                Offset::new(rect.right() - radius[2], rect.bottom() - radius[2]),
                0.0,
            ),
            (
                radius[3],
                Offset::new(rect.left() + radius[3], rect.bottom() - radius[3]),
                core::f32::consts::FRAC_PI_2,
            ),
        ];
        let sharp = [
            Offset::new(rect.left(), rect.top()),
            Offset::new(rect.right(), rect.top()),
            Offset::new(rect.right(), rect.bottom()),
            Offset::new(rect.left(), rect.bottom()),
        ];

        let mut vertices = [Offset::ZERO; MAX_VERTICES];
        let mut count = 0usize;
        let sweep = core::f32::consts::FRAC_PI_2 / CORNER_SEGMENTS as f32;
        for (corner, (radius, center, from)) in corners.iter().enumerate() {
            if *radius <= 0.0 {
                vertices[count] = sharp[corner];
                count += 1;
                continue;
            }
            for step in 0..=CORNER_SEGMENTS {
                let angle = from + step as f32 * sweep;
                // The sampled circle is clamped into the box: the rounding of
                // `cosf` and `sinf` can put an arc endpoint a fraction of an
                // ulp outside it, and the bound has to enclose every vertex.
                let x = (center.x + radius * cosf(angle)).clamp(rect.left(), rect.right());
                let y = (center.y + radius * sinf(angle)).clamp(rect.top(), rect.bottom());
                vertices[count] = Offset::new(x, y);
                count += 1;
            }
        }
        Some(Self::from_vertices(id, &vertices[..count]))
    }

    /// Builds a shape from the vertices of a polygon already known to be
    /// convex, in order around it.
    ///
    /// Vertices closer than [`MIN_EDGE`] collapse into one first, so the
    /// shape has no micro-edge, and a list that collapses down to a line or a
    /// point becomes the plain box around itself instead of a polygon nobody
    /// can hit.
    fn from_vertices(id: BoundId, vertices: &[Offset]) -> Self {
        debug_assert!(!vertices.is_empty() && vertices.len() <= MAX_VERTICES);
        let vertices = &vertices[..vertices.len().min(MAX_VERTICES)];
        let mut kept = [Offset::ZERO; MAX_VERTICES];
        let mut count = 0usize;
        for vertex in vertices {
            if count > 0 {
                let edge = vertex.sub(kept[count - 1]);
                if edge.x * edge.x + edge.y * edge.y < MIN_EDGE * MIN_EDGE {
                    continue;
                }
            }
            kept[count] = *vertex;
            count += 1;
        }
        if count > 1 {
            let closing = kept[0].sub(kept[count - 1]);
            if closing.x * closing.x + closing.y * closing.y < MIN_EDGE * MIN_EDGE {
                count -= 1;
            }
        }
        if count < 3 {
            return Self::rect(id, bounds_of(vertices));
        }
        let polygon = &kept[..count];
        let bound = bounds_of(polygon);
        let plain = is_rect(polygon, bound);
        let mut stored = [Offset::ZERO; MAX_VERTICES];
        stored[..count].copy_from_slice(polygon);
        let last = polygon[polygon.len() - 1];
        for slot in &mut stored[count..] {
            *slot = last;
        }
        Self {
            id,
            vertices: stored,
            count: count as u8,
            bound,
            plain,
        }
    }

    /// The identifier returned on a hit.
    #[must_use]
    pub const fn id(self) -> BoundId {
        self.id
    }

    /// The real vertices, without the padding.
    #[must_use]
    pub fn vertices(&self) -> &[Offset] {
        &self.vertices[..usize::from(self.count)]
    }

    /// The number of real vertices, and so of edges.
    #[must_use]
    pub const fn vertex_count(&self) -> usize {
        self.count as usize
    }

    /// The axis-aligned box that encloses the shape.
    #[must_use]
    pub const fn bound(self) -> Rect {
        self.bound
    }

    /// Whether the shape is exactly its own bounding box.
    ///
    /// A caller that only needs a yes-or-no hit can answer a plain shape with
    /// a rectangle test and never look at the edges, which is why the widget
    /// tree keeps the plain case on that fast path.
    #[must_use]
    pub const fn is_plain(self) -> bool {
        self.plain
    }

    /// Whether the point lies inside the shape or on its boundary.
    ///
    /// A plain shape is tested against its box; any other shape is tested by
    /// winding, which needs the vertices to form a convex polygon — the
    /// guarantee every constructor documents. An edge whose cross product is
    /// within [`CROSS_NOISE`] of zero counts as collinear with the point,
    /// which is what keeps a sampled arc whose endpoints `cosf` and `sinf`
    /// rounded onto each other from reading as a corner that turns away.
    #[must_use]
    pub fn contains(self, point: Offset) -> bool {
        if self.plain {
            return point.x >= self.bound.left()
                && point.x <= self.bound.right()
                && point.y >= self.bound.top()
                && point.y <= self.bound.bottom();
        }
        let vertices = self.vertices();
        let count = vertices.len();
        let mut winding = 0.0f32;
        for index in 0..count {
            let a = vertices[index];
            let b = vertices[(index + 1) % count];
            let edge = b.sub(a);
            let to_point = point.sub(a);
            let side = cross(edge, to_point);
            let len_sq = edge.x * edge.x + edge.y * edge.y;
            let reach_sq = to_point.x * to_point.x + to_point.y * to_point.y;
            if side * side <= CROSS_NOISE * len_sq * reach_sq {
                continue;
            }
            if winding == 0.0 {
                winding = side;
            } else if (side > 0.0) != (winding > 0.0) {
                return false;
            }
        }
        true
    }

    /// The distance from `point` to the shape, or zero when it is inside.
    #[must_use]
    pub fn distance(self, point: Offset) -> f32 {
        sqrtf(self.distance_sq(point))
    }

    /// The squared distance from `point` to the shape.
    ///
    /// An inside point is at distance zero. An outside point is measured
    /// against the nearest edge segment, including the segment endpoints, so
    /// a point beyond a corner gets the corner distance rather than the
    /// distance to the infinite edge line.
    ///
    /// The edges of a convex polygon are independent, so a kernel measures a
    /// batch of them at once and the batch minima are reduced; see
    /// [`Self::reduce`]. On x86-64 the widest kernel the CPU and OS support is
    /// chosen through [`codevar_base::cpuid`]; WebAssembly uses simd128 when
    /// it is compiled in, and every other target uses the scalar reference,
    /// which is the oracle the kernels are required to match.
    #[must_use]
    pub fn distance_sq(self, point: Offset) -> f32 {
        if self.contains(point) {
            return 0.0;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(kernel) = x86::kernel() {
                return self.reduce(point, kernel.lanes(), |edges| {
                    // SAFETY: the kernel was chosen by `x86::kernel`, which
                    // only returns one whose target feature the cpuid probe
                    // reported together with the OS-enabled register state.
                    unsafe { kernel.run(edges) }
                });
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if cpuid::has(Feature::Simd128) {
                return self.reduce(point, wasm::LANES, wasm::distance_sq);
            }
        }
        self.reduce(point, LANES, scalar::distance_sq)
    }

    /// Measures every edge of the shape in batches of `lanes` edges.
    ///
    /// `run` returns the smallest squared distance over one batch, and the
    /// minima of all the batches are reduced into the distance of the shape.
    /// A shape with a single edge group calls `run` once, which is the case
    /// the triangles and rectangles of a widget tree fall into.
    #[inline]
    fn reduce(self, point: Offset, lanes: usize, mut run: impl FnMut(&Edges) -> f32) -> f32 {
        let count = usize::from(self.count).max(1);
        let lanes = lanes.clamp(1, LANES);
        let mut best = f32::INFINITY;
        let mut start = 0;
        while start < count {
            let edges = self.batch(point, start, lanes);
            best = best.min(run(&edges));
            start += lanes;
        }
        best
    }

    /// The batch of at most `lanes` edges starting at edge `start`.
    ///
    /// Edge `i` runs from vertex `i` to vertex `i + 1`, wrapping at the last
    /// vertex. A trailing batch shorter than `lanes` repeats its own edges in
    /// the unused lanes, so the batch minimum is the shape minimum over the
    /// edges it covers and no lane holds a poison value.
    #[inline]
    fn batch(self, point: Offset, start: usize, lanes: usize) -> Edges {
        let count = usize::from(self.count).max(1);
        let lanes = lanes.clamp(1, LANES);
        let size = lanes.min(count - start).max(1);
        let mut ax = [0.0; LANES];
        let mut ay = [0.0; LANES];
        let mut bx = [0.0; LANES];
        let mut by = [0.0; LANES];
        for lane in 0..LANES {
            let edge = (start + lane % size) % count;
            let a = self.vertices[edge];
            let b = self.vertices[(edge + 1) % count];
            ax[lane] = a.x;
            ay[lane] = a.y;
            bx[lane] = b.x;
            by[lane] = b.y;
        }
        Edges {
            point,
            ax,
            ay,
            bx,
            by,
            count: size,
        }
    }
}

/// Copies up to four vertices into a full array, repeating the last one.
fn pad(vertices: &[Offset; 4]) -> [Offset; MAX_VERTICES] {
    let mut stored = [vertices[3]; MAX_VERTICES];
    stored[..4].copy_from_slice(vertices);
    stored
}

/// The axis-aligned box that encloses the vertices.
fn bounds_of(vertices: &[Offset]) -> Rect {
    let mut left = f32::INFINITY;
    let mut top = f32::INFINITY;
    let mut right = f32::NEG_INFINITY;
    let mut bottom = f32::NEG_INFINITY;
    for vertex in vertices {
        left = left.min(vertex.x);
        top = top.min(vertex.y);
        right = right.max(vertex.x);
        bottom = bottom.max(vertex.y);
    }
    Rect::from_min_size(Offset::new(left, top), right - left, bottom - top)
}

/// Whether the polygon is exactly the four corners of `bound`.
///
/// Both directions are checked, so a list that repeats one corner and omits
/// another is not called plain.
fn is_rect(vertices: &[Offset], bound: Rect) -> bool {
    if vertices.len() != 4 {
        return false;
    }
    let corners = [
        Offset::new(bound.left(), bound.top()),
        Offset::new(bound.right(), bound.top()),
        Offset::new(bound.right(), bound.bottom()),
        Offset::new(bound.left(), bound.bottom()),
    ];
    corners
        .iter()
        .all(|corner| vertices.contains(corner))
        && vertices
            .iter()
            .all(|vertex| corners.contains(vertex))
}

/// The two-dimensional cross product `a x b`.
#[inline]
fn cross(a: Offset, b: Offset) -> f32 {
    a.x * b.y - a.y * b.x
}

/// Andrew's monotone chain: the convex hull of `points` and its length.
///
/// The points are sorted by `x` then `y` and walked twice, once forward for
/// the lower chain and once backward for the upper one; a chain drops its
/// previous corner whenever the new one makes a right turn or goes straight,
/// so collinear points never survive as vertices. Returns `None` when fewer
/// than three corners survive, which is what an all-collinear, all-identical
/// or duplicate-heavy list produces: there is no area to hit.
///
/// The work is fixed-size — the input is bounded by [`MAX_VERTICES`] — so no
/// allocation happens on the hit-test path.
fn convex_hull(points: &[Offset]) -> Option<([Offset; MAX_VERTICES], usize)> {
    let count = points.len();
    if count < 3 {
        return None;
    }
    let mut sorted = [Offset::ZERO; MAX_VERTICES];
    sorted[..count].copy_from_slice(points);
    sorted[..count].sort_unstable_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));

    // A hull of `count` corners needs at most `count` slots per chain, and
    // the two chains share the first and last point.
    let mut hull = [Offset::ZERO; 2 * MAX_VERTICES];
    let mut size = 0usize;
    for point in sorted[..count].iter() {
        while size >= 2 && cross(hull[size - 1].sub(hull[size - 2]), point.sub(hull[size - 2])) <= 0.0 {
            size -= 1;
        }
        hull[size] = *point;
        size += 1;
    }
    let lower = size + 1;
    for point in sorted[..count].iter().rev().skip(1) {
        while size >= lower && cross(hull[size - 1].sub(hull[size - 2]), point.sub(hull[size - 2])) <= 0.0 {
            size -= 1;
        }
        hull[size] = *point;
        size += 1;
    }
    if size == 0 {
        return None;
    }
    // The last corner repeats the first one.
    size -= 1;
    if size < 3 {
        return None;
    }
    let mut result = [Offset::ZERO; MAX_VERTICES];
    result[..size].copy_from_slice(&hull[..size]);
    Some((result, size))
}

/// The scalar reference: one segment at a time.
///
/// Every target compiles it, and it is the oracle the vector kernels are
/// tested against.
mod scalar {
    use crate::geometry::Offset;

    use super::Edges;

    /// The smallest squared distance to the edges of the batch.
    #[inline]
    pub(super) fn distance_sq(edges: &Edges) -> f32 {
        let mut best = f32::INFINITY;
        for lane in 0..edges.count {
            let a = Offset::new(edges.ax[lane], edges.ay[lane]);
            let b = Offset::new(edges.bx[lane], edges.by[lane]);
            best = best.min(segment_distance_sq(edges.point, a, b));
        }
        best
    }

    /// The squared distance from `point` to the segment `a`-`b`.
    #[inline]
    fn segment_distance_sq(point: Offset, a: Offset, b: Offset) -> f32 {
        let ab = b.sub(a);
        let ap = point.sub(a);
        let length_sq = ab.x * ab.x + ab.y * ab.y;
        let t = if length_sq > 0.0 {
            ((ap.x * ab.x + ap.y * ab.y) / length_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let dx = ap.x - ab.x * t;
        let dy = ap.y - ab.y * t;
        dx * dx + dy * dy
    }
}

/// The x86-64 kernels, one edge per lane.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use codevar_base::cpuid::{self, Feature};

    use super::Edges;

    /// The widest kernel this CPU can run and the edges it takes per batch.
    #[derive(Debug, Clone, Copy)]
    pub(super) enum Kernel {
        /// Four edges per call, in two 128-bit registers.
        Sse2,
        /// Eight edges per call, in one 256-bit register.
        Avx2,
        /// Eight edges per call, broadcast into one 512-bit register.
        Avx512,
    }

    impl Kernel {
        /// The number of edges one batch holds for this kernel.
        pub(super) const fn lanes(self) -> usize {
            match self {
                Self::Sse2 => 4,
                Self::Avx2 | Self::Avx512 => 8,
            }
        }

        /// The smallest squared distance over one batch.
        ///
        /// # Safety
        ///
        /// The CPU must support the target feature the variant was built for.
        /// The batch is `lanes()` or fewer real edges with the remaining lanes
        /// repeating them, so every kernel reads only initialized lanes.
        pub(super) unsafe fn run(self, edges: &Edges) -> f32 {
            match self {
                // SAFETY: the caller upholds the per-variant precondition.
                Self::Sse2 => unsafe { distance_sse2(edges) },
                // SAFETY: the caller upholds the per-variant precondition.
                Self::Avx2 => unsafe { distance_avx2(edges) },
                // SAFETY: the caller upholds the per-variant precondition.
                Self::Avx512 => unsafe { distance_avx512(edges) },
            }
        }
    }

    /// The widest kernel the CPU and the OS support.
    ///
    /// Returns `None` when not even SSE2 is available, which leaves the
    /// scalar reference in charge.
    pub(super) fn kernel() -> Option<Kernel> {
        if cpuid::has(Feature::Avx512F) {
            return Some(Kernel::Avx512);
        }
        if cpuid::has(Feature::Avx2) {
            return Some(Kernel::Avx2);
        }
        if cpuid::has(Feature::Sse2) {
            return Some(Kernel::Sse2);
        }
        None
    }

    /// The edges of one batch, up to four of them in a 128-bit register.
    ///
    /// # Safety
    ///
    /// The caller must have verified SSE2 support. Every array has
    /// [`super::LANES`] elements, so the unaligned 16-byte loads of the first
    /// four stay inside it.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn distance_sse2(edges: &Edges) -> f32 {
        use core::arch::x86_64::{
            _mm_add_ps, _mm_cvtss_f32, _mm_div_ps, _mm_loadu_ps, _mm_max_ps, _mm_min_ps, _mm_mul_ps,
            _mm_set1_ps, _mm_setzero_ps, _mm_shuffle_ps, _mm_sub_ps,
        };

        // SAFETY: the caller guarantees SSE2; each load reads four
        // initialized floats and every operation stays in registers.
        unsafe {
            let px = _mm_set1_ps(edges.point.x);
            let py = _mm_set1_ps(edges.point.y);
            let ax = _mm_loadu_ps(edges.ax.as_ptr());
            let ay = _mm_loadu_ps(edges.ay.as_ptr());
            let bx = _mm_loadu_ps(edges.bx.as_ptr());
            let by = _mm_loadu_ps(edges.by.as_ptr());

            let abx = _mm_sub_ps(bx, ax);
            let aby = _mm_sub_ps(by, ay);
            let apx = _mm_sub_ps(px, ax);
            let apy = _mm_sub_ps(py, ay);

            // The denominator is a squared edge length. A degenerate edge (a
            // repeated vertex) has zero length; flooring it avoids 0/0 while
            // the clamp below still projects the point onto the vertex.
            let denominator = _mm_max_ps(
                _mm_add_ps(_mm_mul_ps(abx, abx), _mm_mul_ps(aby, aby)),
                _mm_set1_ps(f32::MIN_POSITIVE),
            );
            let t = _mm_min_ps(
                _mm_max_ps(
                    _mm_div_ps(
                        _mm_add_ps(_mm_mul_ps(apx, abx), _mm_mul_ps(apy, aby)),
                        denominator,
                    ),
                    _mm_setzero_ps(),
                ),
                _mm_set1_ps(1.0),
            );

            let closest_x = _mm_add_ps(ax, _mm_mul_ps(t, abx));
            let closest_y = _mm_add_ps(ay, _mm_mul_ps(t, aby));
            let dx = _mm_sub_ps(px, closest_x);
            let dy = _mm_sub_ps(py, closest_y);
            let distance_sq = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy));

            // Horizontal minimum: `0x4E` pairs lane `i` with lane `i + 2`,
            // then the second shuffle pairs lane 0 with lane 1.
            let pair_min = _mm_min_ps(
                distance_sq,
                _mm_shuffle_ps(distance_sq, distance_sq, 0b01_00_11_10),
            );
            let all_min = _mm_min_ps(pair_min, _mm_shuffle_ps(pair_min, pair_min, 0b00_00_00_01));
            _mm_cvtss_f32(all_min)
        }
    }

    /// The edges of one batch, up to eight of them in a 256-bit register.
    ///
    /// # Safety
    ///
    /// The caller must have verified AVX2 support. Every array has
    /// [`super::LANES`] elements, so the unaligned 32-byte loads stay inside
    /// it.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn distance_avx2(edges: &Edges) -> f32 {
        use core::arch::x86_64::{
            _mm_cvtss_f32, _mm_min_ps, _mm_shuffle_ps, _mm256_add_ps, _mm256_castps256_ps128, _mm256_div_ps,
            _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps,
            _mm256_set1_ps, _mm256_setzero_ps, _mm256_sub_ps,
        };

        // SAFETY: the caller guarantees AVX2; each load reads eight
        // initialized floats and every operation stays in registers.
        unsafe {
            let px = _mm256_set1_ps(edges.point.x);
            let py = _mm256_set1_ps(edges.point.y);
            let ax = _mm256_loadu_ps(edges.ax.as_ptr());
            let ay = _mm256_loadu_ps(edges.ay.as_ptr());
            let bx = _mm256_loadu_ps(edges.bx.as_ptr());
            let by = _mm256_loadu_ps(edges.by.as_ptr());

            let abx = _mm256_sub_ps(bx, ax);
            let aby = _mm256_sub_ps(by, ay);
            let apx = _mm256_sub_ps(px, ax);
            let apy = _mm256_sub_ps(py, ay);

            let denominator = _mm256_max_ps(
                _mm256_add_ps(_mm256_mul_ps(abx, abx), _mm256_mul_ps(aby, aby)),
                _mm256_set1_ps(f32::MIN_POSITIVE),
            );
            let t = _mm256_min_ps(
                _mm256_max_ps(
                    _mm256_div_ps(
                        _mm256_add_ps(_mm256_mul_ps(apx, abx), _mm256_mul_ps(apy, aby)),
                        denominator,
                    ),
                    _mm256_setzero_ps(),
                ),
                _mm256_set1_ps(1.0),
            );

            let closest_x = _mm256_add_ps(ax, _mm256_mul_ps(t, abx));
            let closest_y = _mm256_add_ps(ay, _mm256_mul_ps(t, aby));
            let dx = _mm256_sub_ps(px, closest_x);
            let dy = _mm256_sub_ps(py, closest_y);
            let distance_sq = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy));

            let low = _mm256_castps256_ps128(distance_sq);
            let high = _mm256_extractf128_ps::<1>(distance_sq);
            let combined = _mm_min_ps(low, high);
            // Horizontal minimum: `0x4E` pairs lane `i` with lane `i + 2`,
            // then the second shuffle pairs lane 0 with lane 1.
            let pair_min = _mm_min_ps(combined, _mm_shuffle_ps(combined, combined, 0b01_00_11_10));
            let all_min = _mm_min_ps(pair_min, _mm_shuffle_ps(pair_min, pair_min, 0b00_00_00_01));
            _mm_cvtss_f32(all_min)
        }
    }

    /// The edges of one batch, eight of them broadcast into a 512-bit
    /// register.
    ///
    /// # Safety
    ///
    /// The caller must have verified AVX-512F support and the OS-enabled ZMM
    /// state. The kernel loads the eight-element arrays with a 256-bit load
    /// and broadcasts that vector into both halves of the 512-bit register,
    /// so nothing outside the arrays is read.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn distance_avx512(edges: &Edges) -> f32 {
        use core::arch::x86_64::{
            _mm256_loadu_ps, _mm512_add_ps, _mm512_broadcast_f32x8, _mm512_div_ps, _mm512_max_ps,
            _mm512_min_ps, _mm512_mul_ps, _mm512_reduce_min_ps, _mm512_set1_ps, _mm512_setzero_ps,
            _mm512_sub_ps,
        };

        // SAFETY: the caller guarantees AVX-512F; each 256-bit load reads
        // eight initialized floats and every operation stays in registers.
        unsafe {
            let px = _mm512_set1_ps(edges.point.x);
            let py = _mm512_set1_ps(edges.point.y);
            let ax = _mm512_broadcast_f32x8(_mm256_loadu_ps(edges.ax.as_ptr()));
            let ay = _mm512_broadcast_f32x8(_mm256_loadu_ps(edges.ay.as_ptr()));
            let bx = _mm512_broadcast_f32x8(_mm256_loadu_ps(edges.bx.as_ptr()));
            let by = _mm512_broadcast_f32x8(_mm256_loadu_ps(edges.by.as_ptr()));

            let abx = _mm512_sub_ps(bx, ax);
            let aby = _mm512_sub_ps(by, ay);
            let apx = _mm512_sub_ps(px, ax);
            let apy = _mm512_sub_ps(py, ay);

            let denominator = _mm512_max_ps(
                _mm512_add_ps(_mm512_mul_ps(abx, abx), _mm512_mul_ps(aby, aby)),
                _mm512_set1_ps(f32::MIN_POSITIVE),
            );
            let t = _mm512_min_ps(
                _mm512_max_ps(
                    _mm512_div_ps(
                        _mm512_add_ps(_mm512_mul_ps(apx, abx), _mm512_mul_ps(apy, aby)),
                        denominator,
                    ),
                    _mm512_setzero_ps(),
                ),
                _mm512_set1_ps(1.0),
            );

            let closest_x = _mm512_add_ps(ax, _mm512_mul_ps(t, abx));
            let closest_y = _mm512_add_ps(ay, _mm512_mul_ps(t, aby));
            let dx = _mm512_sub_ps(px, closest_x);
            let dy = _mm512_sub_ps(py, closest_y);
            let distance_sq = _mm512_add_ps(_mm512_mul_ps(dx, dx), _mm512_mul_ps(dy, dy));

            _mm512_reduce_min_ps(distance_sq)
        }
    }
}

/// The WebAssembly simd128 kernel, one edge per lane.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        f32x4, f32x4_add, f32x4_div, f32x4_extract_lane, f32x4_max, f32x4_min, f32x4_mul, f32x4_splat,
        f32x4_sub, i32x4_shuffle,
    };

    use super::Edges;

    /// The number of edges one batch holds: the lanes of a 128-bit vector.
    pub(super) const LANES: usize = 4;

    /// The smallest squared distance to the edges of the batch.
    pub(super) fn distance_sq(edges: &Edges) -> f32 {
        let px = f32x4_splat(edges.point.x);
        let py = f32x4_splat(edges.point.y);
        let ax = f32x4(edges.ax[0], edges.ax[1], edges.ax[2], edges.ax[3]);
        let ay = f32x4(edges.ay[0], edges.ay[1], edges.ay[2], edges.ay[3]);
        let bx = f32x4(edges.bx[0], edges.bx[1], edges.bx[2], edges.bx[3]);
        let by = f32x4(edges.by[0], edges.by[1], edges.by[2], edges.by[3]);

        let abx = f32x4_sub(bx, ax);
        let aby = f32x4_sub(by, ay);
        let apx = f32x4_sub(px, ax);
        let apy = f32x4_sub(py, ay);

        let denominator = f32x4_max(
            f32x4_add(f32x4_mul(abx, abx), f32x4_mul(aby, aby)),
            f32x4_splat(f32::MIN_POSITIVE),
        );
        let t = f32x4_min(
            f32x4_max(
                f32x4_div(f32x4_add(f32x4_mul(apx, abx), f32x4_mul(apy, aby)), denominator),
                f32x4_splat(0.0),
            ),
            f32x4_splat(1.0),
        );

        let closest_x = f32x4_add(ax, f32x4_mul(t, abx));
        let closest_y = f32x4_add(ay, f32x4_mul(t, aby));
        let dx = f32x4_sub(px, closest_x);
        let dy = f32x4_sub(py, closest_y);
        let distance_sq = f32x4_add(f32x4_mul(dx, dx), f32x4_mul(dy, dy));

        let pair_min = f32x4_min(distance_sq, i32x4_shuffle::<2, 3, 0, 1>(distance_sq, distance_sq));
        let all_min = f32x4_min(pair_min, i32x4_shuffle::<1, 0, 1, 0>(pair_min, pair_min));
        f32x4_extract_lane::<0>(all_min)
    }
}

#[cfg(test)]
mod tests {
    use super::{BoundId, CORNER_SEGMENTS, MAX_VERTICES, VertexBounds};
    use crate::geometry::{Offset, Rect};
    use crate::paint::CornerRadii;

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use codevar_base::cpuid::{self, Feature};

    #[cfg(target_arch = "x86_64")]
    use codevar_base::cpuid::{self, Feature};

    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    use super::scalar;

    #[test]
    fn a_reversed_winding_still_contains_its_vertices() {
        let shape = VertexBounds::quad(
            BoundId::new(3),
            [
                Offset::new(0.0, 0.0),
                Offset::new(0.0, 10.0),
                Offset::new(10.0, 10.0),
                Offset::new(10.0, 0.0),
            ],
        );
        assert!(shape.contains(Offset::new(5.0, 5.0)));
        assert!(!shape.contains(Offset::new(15.0, 5.0)));
    }

    #[test]
    fn a_triangle_is_measured_against_its_edges() {
        let shape = VertexBounds::triangle(
            BoundId::new(4),
            [
                Offset::new(0.0, 0.0),
                Offset::new(40.0, 0.0),
                Offset::new(0.0, 40.0),
            ],
        );
        assert!(shape.contains(Offset::new(10.0, 10.0)));
        assert!(!shape.contains(Offset::new(30.0, 30.0)));
        let distance = shape.distance(Offset::new(30.0, 30.0));
        assert!((distance - (200.0f32).sqrt()).abs() < 0.001);
    }

    #[test]
    fn a_degenerate_list_is_rejected() {
        assert!(VertexBounds::new(BoundId::new(1), &[Offset::ZERO]).is_none());
        assert!(
            VertexBounds::new(
                BoundId::new(1),
                &[
                    Offset::ZERO,
                    Offset::new(1.0, 0.0),
                    Offset::new(2.0, 0.0),
                    Offset::new(3.0, 0.0),
                    Offset::new(4.0, 0.0),
                ]
            )
            .is_none(),
            "collinear vertices have no area"
        );
        assert!(
            VertexBounds::new(
                BoundId::new(1),
                &[
                    Offset::ZERO,
                    Offset::new(1.0, 0.0),
                    Offset::new(0.0, 1.0),
                    Offset::new(1.0, 1.0),
                    Offset::new(2.0, 2.0),
                    Offset::new(3.0, 3.0),
                    Offset::new(4.0, 4.0),
                    Offset::new(5.0, 5.0),
                    Offset::new(6.0, 6.0),
                    Offset::new(7.0, 7.0),
                    Offset::new(8.0, 8.0),
                    Offset::new(9.0, 9.0),
                    Offset::new(10.0, 10.0),
                    Offset::new(11.0, 11.0),
                    Offset::new(12.0, 12.0),
                    Offset::new(13.0, 13.0),
                    Offset::new(14.0, 14.0),
                    Offset::new(15.0, 15.0),
                    Offset::new(16.0, 16.0),
                    Offset::new(17.0, 17.0),
                    Offset::new(18.0, 18.0),
                    Offset::new(19.0, 19.0),
                    Offset::new(20.0, 20.0),
                    Offset::new(21.0, 21.0),
                ]
            )
            .is_none(),
            "a list past the vertex bound is rejected, not grown"
        );
        assert!(
            VertexBounds::new(
                BoundId::new(1),
                &[Offset::ZERO, Offset::new(f32::NAN, 1.0), Offset::new(1.0, 2.0),]
            )
            .is_none()
        );
    }

    #[test]
    fn the_hull_drops_interior_and_collinear_points() {
        // The same square with a point on each edge and one inside it.
        let shape = VertexBounds::new(
            BoundId::new(2),
            &[
                Offset::new(0.0, 0.0),
                Offset::new(10.0, 0.0),
                Offset::new(10.0, 10.0),
                Offset::new(0.0, 10.0),
                Offset::new(5.0, 0.0),
                Offset::new(5.0, 5.0),
            ],
        );
        let shape = shape.expect("the list has a real hull");
        assert_eq!(
            shape.vertex_count(),
            4,
            "the collinear and the interior point are gone"
        );
        assert!(shape.is_plain(), "what is left is the box itself");
    }

    #[test]
    fn a_shape_of_many_vertices_is_still_a_shape() {
        // A hexagon: convex, not axis-aligned, so it keeps the edge walk.
        let mut vertices = [Offset::ZERO; 6];
        for (index, vertex) in vertices.iter_mut().enumerate() {
            let angle = core::f32::consts::PI / 3.0 * index as f32;
            *vertex = Offset::new(50.0 + 40.0 * libm::cosf(angle), 50.0 + 40.0 * libm::sinf(angle));
        }
        let shape = VertexBounds::new(BoundId::new(6), &vertices).expect("a hexagon has a hull");
        assert_eq!(shape.vertex_count(), 6);
        assert!(!shape.is_plain());
        assert!(shape.contains(Offset::new(50.0, 50.0)));
        assert!(!shape.contains(Offset::new(2.0, 2.0)));
    }

    #[test]
    fn a_rounded_rectangle_fits_the_vertex_bound() {
        let rect = Rect::from_min_size(Offset::new(10.0, 20.0), 120.0, 60.0);
        let shape = VertexBounds::rounded_rect(BoundId::new(9), rect, CornerRadii::uniform(18.0))
            .expect("the rectangle has an area");
        // One corner is cut to the shorter half-height, so the four arcs of
        // five points each are the whole shape.
        assert_eq!(shape.vertex_count(), 4 * (CORNER_SEGMENTS + 1));
        assert!(shape.vertex_count() <= MAX_VERTICES);
        assert_eq!(shape.bound(), rect);
        assert!(!shape.is_plain());
        assert!(shape.contains(rect.center()));
    }

    #[test]
    fn a_rounded_corner_cuts_the_box_corner() {
        let rect = Rect::from_min_size(Offset::ZERO, 100.0, 100.0);
        let shape = VertexBounds::rounded_rect(BoundId::new(1), rect, CornerRadii::uniform(30.0))
            .expect("the rectangle has an area");
        assert!(shape.contains(Offset::new(50.0, 50.0)));
        assert!(shape.contains(Offset::new(50.0, 1.0)), "the straight side stays");
        assert!(
            !shape.contains(Offset::new(1.0, 1.0)),
            "the corner of the box is outside the arc"
        );
        let distance = shape.distance(Offset::new(0.0, 0.0));
        // The nearest point of the true rounded box is `r*sqrt(2) - r` away
        // from the box corner; the sampled arc cuts a little inside the
        // circle, so the polygon distance is a hair below that.
        let exact = 30.0f32 * (2.0f32).sqrt() - 30.0;
        assert!(
            (distance - exact).abs() < 0.5,
            "the box corner is one arc away from the shape: {distance} vs {exact}"
        );
    }

    #[test]
    fn a_rounded_rectangle_with_sharp_corners_is_plain() {
        let rect = Rect::from_min_size(Offset::ZERO, 40.0, 24.0);
        let shape = VertexBounds::rounded_rect(BoundId::new(2), rect, CornerRadii::ZERO)
            .expect("the rectangle has an area");
        assert!(shape.is_plain());
        assert_eq!(shape.vertex_count(), 4);
        assert_eq!(shape.bound(), rect);

        // Only two corners rounded: the plain fast path is gone but the
        // shape is smaller than the bound allows.
        let partly = VertexBounds::rounded_rect(BoundId::new(3), rect, CornerRadii::new(8.0, 8.0, 0.0, 0.0))
            .expect("the rectangle has an area");
        assert_eq!(partly.vertex_count(), 2 * (CORNER_SEGMENTS + 1) + 2);
        assert!(!partly.is_plain());
        assert!(partly.contains(Offset::new(20.0, 23.0)), "the sharp bottom stays");
        assert!(!partly.contains(Offset::new(0.5, 0.5)), "the top-left is cut");
    }

    #[test]
    fn a_rounded_rectangle_degrades_to_a_pill_when_the_radius_is_too_big() {
        let rect = Rect::from_min_size(Offset::ZERO, 40.0, 20.0);
        let shape = VertexBounds::rounded_rect(BoundId::new(4), rect, CornerRadii::uniform(100.0))
            .expect("the rectangle has an area");
        // Every radius is scaled to half the height, which is a full pill.
        assert_eq!(shape.bound(), rect);
        assert!(shape.contains(Offset::new(20.0, 10.0)));
        assert!(!shape.contains(Offset::new(0.5, 0.5)));
        assert!(!shape.contains(Offset::new(39.5, 19.5)));
    }

    #[test]
    fn an_empty_rectangle_has_no_shape() {
        assert!(
            VertexBounds::rounded_rect(
                BoundId::new(5),
                Rect::from_min_size(Offset::ZERO, 0.0, 20.0),
                CornerRadii::uniform(4.0)
            )
            .is_none()
        );
    }

    /// Compares a kernel result with the scalar oracle.
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn close(left: f32, right: f32) -> bool {
        (left - right).abs() <= 1e-3 * (1.0 + left.abs().max(right.abs()))
    }

    /// The sample shapes: a triangle, a rotated quad, a hexagon and a rounded
    /// rectangle whose four corners carry four different radii.
    ///
    /// Together they cover every vertex count between three and the bound, so
    /// a shape needs one batch, several full batches, and a trailing batch
    /// that has to repeat lanes.
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn sample_shapes() -> [VertexBounds; 4] {
        [
            VertexBounds::triangle(
                BoundId::new(7),
                [
                    Offset::new(4.0, 2.0),
                    Offset::new(46.0, 9.0),
                    Offset::new(12.0, 44.0),
                ],
            ),
            VertexBounds::quad(
                BoundId::new(8),
                [
                    Offset::new(6.0, 6.0),
                    Offset::new(30.0, 10.0),
                    Offset::new(26.0, 34.0),
                    Offset::new(2.0, 28.0),
                ],
            ),
            VertexBounds::new(
                BoundId::new(9),
                &[
                    Offset::new(10.0, 2.0),
                    Offset::new(38.0, 6.0),
                    Offset::new(44.0, 26.0),
                    Offset::new(30.0, 44.0),
                    Offset::new(8.0, 40.0),
                    Offset::new(2.0, 18.0),
                ],
            )
            .expect("the hexagon has a hull"),
            VertexBounds::rounded_rect(
                BoundId::new(10),
                Rect::from_min_size(Offset::new(3.0, 5.0), 44.0, 30.0),
                CornerRadii::new(9.0, 6.0, 12.0, 3.0),
            )
            .expect("the rectangle has an area"),
        ]
    }

    /// The sample points, on a grid that reaches inside each shape, outside
    /// it, past its corners and along its edge lines.
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn sample_points() -> [Offset; 169] {
        let mut grid = [Offset::ZERO; 169];
        let mut slot = 0;
        for row in 0..=12 {
            for column in 0..=12 {
                grid[slot] = Offset::new(column as f32 * 4.0 - 6.0, row as f32 * 4.0 - 6.0);
                slot += 1;
            }
        }
        grid
    }

    /// The scalar minimum over every edge of `shape`, measured in batches of
    /// `lanes` edges.
    ///
    /// Grouping cannot change a minimum, so this is the oracle for one batch
    /// and for the whole shape at once.
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn scalar_total(shape: &VertexBounds, point: Offset, lanes: usize) -> f32 {
        let mut best = f32::INFINITY;
        let mut start = 0;
        while start < shape.vertex_count() {
            best = best.min(scalar::distance_sq(&shape.batch(point, start, lanes)));
            start += lanes;
        }
        best
    }

    /// Runs one kernel over every batch of every sample and compares it with
    /// the scalar reference, then compares the dispatched distance with the
    /// scalar total.
    ///
    /// This is the parity test the vector kernels are held to: the batch
    /// check catches a kernel that misreads a padded lane or a batch
    /// boundary, and the dispatch check catches a `distance_sq` that forgets
    /// a batch or forgets that an inside point is at distance zero.
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn compare_kernels(lanes: usize, mut run: impl FnMut(&super::Edges) -> f32) {
        let mut compared = 0u32;
        for shape in sample_shapes() {
            for point in sample_points() {
                let expected = if shape.contains(point) {
                    0.0
                } else {
                    scalar_total(&shape, point, lanes)
                };
                let dispatched = shape.distance_sq(point);
                assert!(
                    close(dispatched, expected),
                    "dispatch {dispatched} != scalar {expected} at ({}, {})",
                    point.x,
                    point.y
                );
                let mut start = 0;
                while start < shape.vertex_count() {
                    let edges = shape.batch(point, start, lanes);
                    let reference = scalar::distance_sq(&edges);
                    let result = run(&edges);
                    assert!(
                        close(result, reference),
                        "kernel {result} != scalar {reference} for a batch at ({}, {})",
                        point.x,
                        point.y
                    );
                    compared += 1;
                    start += lanes;
                }
            }
        }
        assert!(compared > 0, "the kernel ran over no batch at all");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_x86_kernel_matches_the_scalar_reference() {
        assert!(super::x86::kernel().is_some(), "the host has a kernel");
        for kernel in [
            super::x86::Kernel::Sse2,
            super::x86::Kernel::Avx2,
            super::x86::Kernel::Avx512,
        ] {
            if !host_supports(kernel) {
                continue;
            }
            compare_kernels(kernel.lanes(), |edges| {
                // SAFETY: `host_supports` probed the target feature this
                // kernel was compiled for before any batch ran.
                unsafe { kernel.run(edges) }
            });
        }
    }

    /// Whether the host CPU can run `kernel`.
    #[cfg(target_arch = "x86_64")]
    fn host_supports(kernel: super::x86::Kernel) -> bool {
        match kernel {
            super::x86::Kernel::Sse2 => cpuid::has(Feature::Sse2),
            super::x86::Kernel::Avx2 => cpuid::has(Feature::Avx2),
            super::x86::Kernel::Avx512 => cpuid::has(Feature::Avx512F),
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn the_wasm_kernel_matches_the_scalar_reference() {
        assert!(cpuid::has(Feature::Simd128));
        compare_kernels(super::wasm::LANES, super::wasm::distance_sq);
    }
}
