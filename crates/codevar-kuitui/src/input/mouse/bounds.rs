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
//! A bound is three or four vertices in logical pixels. Containment is a
//! winding test, so the vertex order does not matter; the distance is
//! measured against the four edge segments, which is what makes a point
//! beyond a corner get the corner distance rather than the distance to the
//! edge line.

use libm::sqrtf;

use crate::geometry::{Offset, Rect};

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

/// A convex polygon of three or four vertices, in logical pixels.
///
/// The vertices may be given clockwise or counter-clockwise; containment is
/// tested from the winding of the first non-collinear edge, so both orders
/// work. A quadrilateral gives an exact shape for a rotated or skewed box;
/// a rectangle is the common case and has a constructor of its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexBounds {
    /// Identifier returned when the pointer hits this shape.
    id: BoundId,
    /// The vertices, padded with the last real vertex.
    vertices: [Offset; 4],
    /// Number of real vertices, three or four.
    count: u8,
    /// Axis-aligned box that encloses every vertex.
    bound: Rect,
}

impl VertexBounds {
    /// A shape with `id` from three or four vertices.
    ///
    /// Returns `None` when the list has fewer than three vertices, more
    /// than four, or is not finite.
    #[must_use]
    pub fn new(id: BoundId, vertices: &[Offset]) -> Option<Self> {
        if !(3..=4).contains(&vertices.len()) {
            return None;
        }
        let mut padded = [Offset::ZERO; 4];
        for (slot, vertex) in padded.iter_mut().zip(vertices) {
            if !vertex.x.is_finite() || !vertex.y.is_finite() {
                return None;
            }
            *slot = *vertex;
        }
        padded[3] = vertices[vertices.len() - 1];
        let count = u8::try_from(vertices.len()).unwrap_or(2);
        let bound = bounds_of(&padded[..vertices.len()]);
        Some(Self {
            id,
            vertices: padded,
            count,
            bound,
        })
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
            vertices,
            count: 4,
            bound: rect,
        }
    }

    /// A triangular shape.
    #[must_use]
    pub fn triangle(id: BoundId, vertices: [Offset; 3]) -> Self {
        let mut padded = [vertices[2]; 4];
        padded[..3].copy_from_slice(&vertices);
        Self {
            id,
            vertices: padded,
            count: 3,
            bound: bounds_of(&vertices),
        }
    }

    /// A quadrilateral shape.
    #[must_use]
    pub fn quad(id: BoundId, vertices: [Offset; 4]) -> Self {
        Self {
            id,
            vertices,
            count: 4,
            bound: bounds_of(&vertices),
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

    /// The axis-aligned box that encloses the shape.
    #[must_use]
    pub const fn bound(self) -> Rect {
        self.bound
    }

    /// Whether the point lies inside the shape or on its boundary.
    #[must_use]
    pub fn contains(self, point: Offset) -> bool {
        let vertices = self.vertices();
        let mut winding = 0.0f32;
        for index in 0..vertices.len() {
            let a = vertices[index];
            let b = vertices[(index + 1) % vertices.len()];
            let cross = cross(b.sub(a), point.sub(a));
            if cross == 0.0 {
                continue;
            }
            if winding == 0.0 {
                winding = cross;
            } else if (cross > 0.0) != (winding > 0.0) {
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
    #[must_use]
    pub fn distance_sq(self, point: Offset) -> f32 {
        if self.contains(point) {
            return 0.0;
        }
        let (ax, ay, bx, by) = self.edge_arrays();
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: SSE2 is part of the x86_64 baseline, and every array is
            // a local `[f32; 4]`, so the unaligned 16-byte loads stay inside
            // the array.
            unsafe { four_edge_distance_sq_sse2(point, &ax, &ay, &bx, &by) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut best = f32::INFINITY;
            for lane in 0..4 {
                let a = Offset::new(ax[lane], ay[lane]);
                let b = Offset::new(bx[lane], by[lane]);
                best = best.min(segment_distance_sq(point, a, b));
            }
            best
        }
    }

    /// The four edges as parallel coordinate arrays.
    ///
    /// A triangle repeats its first edge in the fourth lane; the minimum is
    /// unchanged because that edge is already measured.
    fn edge_arrays(self) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
        let count = usize::from(self.count);
        let mut ax = [0.0; 4];
        let mut ay = [0.0; 4];
        let mut bx = [0.0; 4];
        let mut by = [0.0; 4];
        for lane in 0..4 {
            let a = self.vertices[lane % count];
            let b = self.vertices[(lane + 1) % count];
            ax[lane] = a.x;
            ay[lane] = a.y;
            bx[lane] = b.x;
            by[lane] = b.y;
        }
        (ax, ay, bx, by)
    }
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

/// The two-dimensional cross product `a x b`.
#[inline]
fn cross(a: Offset, b: Offset) -> f32 {
    a.x * b.y - a.y * b.x
}

/// The squared distance from `point` to the segment `a`-`b`.
#[cfg(not(target_arch = "x86_64"))]
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

/// The smallest squared distance to four edges, one edge per SSE lane.
///
/// # Safety
///
/// SSE2 is guaranteed by the x86_64 target. Each input array has exactly four
/// `f32` elements and is read with an unaligned 16-byte load, so the loads do
/// not cross the array boundary.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn four_edge_distance_sq_sse2(
    point: Offset,
    ax: &[f32; 4],
    ay: &[f32; 4],
    bx: &[f32; 4],
    by: &[f32; 4],
) -> f32 {
    use core::arch::x86_64::{
        _mm_add_ps, _mm_cvtss_f32, _mm_div_ps, _mm_loadu_ps, _mm_max_ps, _mm_min_ps, _mm_mul_ps, _mm_set1_ps,
        _mm_setzero_ps, _mm_shuffle_ps, _mm_sub_ps,
    };

    // SAFETY: the caller guarantees SSE2 and valid four-element arrays; every
    // intrinsic below operates on registers or on those arrays.
    unsafe {
        let px = _mm_set1_ps(point.x);
        let py = _mm_set1_ps(point.y);
        let ax = _mm_loadu_ps(ax.as_ptr());
        let ay = _mm_loadu_ps(ay.as_ptr());
        let bx = _mm_loadu_ps(bx.as_ptr());
        let by = _mm_loadu_ps(by.as_ptr());

        let abx = _mm_sub_ps(bx, ax);
        let aby = _mm_sub_ps(by, ay);
        let apx = _mm_sub_ps(px, ax);
        let apy = _mm_sub_ps(py, ay);

        // The denominator is a squared edge length. A degenerate edge (a repeated
        // vertex) has zero length; flooring it avoids 0/0 while the clamp below
        // still projects the point onto the vertex.
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

        let pair_min = _mm_min_ps(
            distance_sq,
            _mm_shuffle_ps(distance_sq, distance_sq, 0b00_00_10_10),
        );
        let all_min = _mm_min_ps(pair_min, _mm_shuffle_ps(pair_min, pair_min, 0b00_00_00_01));
        _mm_cvtss_f32(all_min)
    }
}

#[cfg(test)]
mod tests {
    use super::{BoundId, VertexBounds};
    use crate::geometry::Offset;

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
                    Offset::new(0.0, 1.0),
                    Offset::new(1.0, 1.0),
                    Offset::new(2.0, 2.0),
                ],
            )
            .is_none()
        );
    }
}
