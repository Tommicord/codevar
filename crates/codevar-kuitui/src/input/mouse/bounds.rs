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

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use codevar_base::cpuid::{self, Feature};

/// The four edges of a shape and the point measured against them.
///
/// The coordinates are kept in parallel arrays so a kernel can load one
/// vector per coordinate instead of gathering from four `Offset`s. Lanes
/// four through seven repeat lanes zero through three, so an AVX2 or AVX-512
/// build can load wider registers without the padding changing the minimum.
/// Repeating real edges also keeps every lane finite, which matters because
/// `min` does not propagate `NaN` consistently across kernels.
struct Edges {
    /// The point the distance is measured from.
    point: Offset,
    /// First endpoint, horizontal.
    ax: [f32; 8],
    /// First endpoint, vertical.
    ay: [f32; 8],
    /// Second endpoint, horizontal.
    bx: [f32; 8],
    /// Second endpoint, vertical.
    by: [f32; 8],
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
    ///
    /// The four edges are independent, so one kernel measures all of them at
    /// once. On x86-64 the widest kernel the CPU and OS support is chosen
    /// through [`codevar_base::cpuid`]; WebAssembly uses simd128 when it is
    /// compiled in, and every other target uses the scalar reference.
    #[must_use]
    pub fn distance_sq(self, point: Offset) -> f32 {
        if self.contains(point) {
            return 0.0;
        }
        let edges = self.edges(point);
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(distance) = x86::distance_sq(&edges) {
                return distance;
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if cpuid::has(Feature::Simd128) {
                return wasm::distance_sq(&edges);
            }
        }
        scalar::distance_sq(&edges)
    }

    /// The four edges as parallel coordinate arrays.
    ///
    /// A triangle repeats its first edge in the fourth lane and every lane is
    /// repeated in the second half; the minimum is unchanged because those
    /// edges are already measured.
    fn edges(self, point: Offset) -> Edges {
        let count = usize::from(self.count);
        let mut ax = [0.0; 8];
        let mut ay = [0.0; 8];
        let mut bx = [0.0; 8];
        let mut by = [0.0; 8];
        for lane in 0..8 {
            let a = self.vertices[lane % count];
            let b = self.vertices[(lane + 1) % count];
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
        }
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

/// The scalar reference: one segment at a time.
///
/// Every target compiles it, and it is the oracle the vector kernels are
/// tested against.
mod scalar {
    use crate::geometry::Offset;

    use super::Edges;

    /// The smallest squared distance to the four edges.
    #[inline]
    pub(super) fn distance_sq(edges: &Edges) -> f32 {
        let mut best = f32::INFINITY;
        for lane in 0..4 {
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

    /// Runs the widest kernel the CPU and the OS support.
    ///
    /// Returns `None` when not even SSE2 is available, which leaves the
    /// scalar reference in charge.
    pub(super) fn distance_sq(edges: &Edges) -> Option<f32> {
        if cpuid::has(Feature::Avx512F) {
            // SAFETY: the probe reports AVX-512F together with the OS-enabled
            // ZMM state, which is exactly the precondition of the `avx512f`
            // target feature. The kernel reads only `edges`.
            return Some(unsafe { distance_avx512(edges) });
        }
        if cpuid::has(Feature::Avx2) {
            // SAFETY: the probe reports AVX2 together with the OS-enabled YMM
            // state, which is exactly the precondition of the `avx2` target
            // feature. The kernel reads only `edges`.
            return Some(unsafe { distance_avx2(edges) });
        }
        if cpuid::has(Feature::Sse2) {
            // SAFETY: the probe reports SSE2, the precondition of the `sse2`
            // target feature. The kernel reads only `edges`.
            return Some(unsafe { distance_sse2(edges) });
        }
        None
    }

    /// The four edges, one per lane of a 128-bit register pair.
    ///
    /// # Safety
    ///
    /// The caller must have verified SSE2 support. Every array has eight
    /// elements, so the unaligned 16-byte loads stay inside it.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn distance_sse2(edges: &Edges) -> f32 {
        use core::arch::x86_64::{
            _mm_add_ps, _mm_cvtss_f32, _mm_div_ps, _mm_loadu_ps, _mm_max_ps, _mm_min_ps, _mm_mul_ps,
            _mm_set1_ps, _mm_setzero_ps, _mm_shuffle_ps, _mm_sub_ps,
        };

        // SAFETY: the caller guarantees SSE2; each load reads eight
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

    /// The four edges, one per lane of a 256-bit register.
    ///
    /// # Safety
    ///
    /// The caller must have verified AVX2 support. Every array has eight
    /// elements, so the unaligned 32-byte loads stay inside it.
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

    /// The four edges, one per lane of a 512-bit register.
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

    /// The smallest squared distance to the four edges.
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
    use super::{BoundId, VertexBounds};
    use crate::geometry::Offset;

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
                    Offset::new(0.0, 1.0),
                    Offset::new(1.0, 1.0),
                    Offset::new(2.0, 2.0),
                ],
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

    /// Runs `compare` over a triangle and a rotated quad at every sample.
    ///
    /// The samples cover inside, outside, corner and edge-line regions, and
    /// both windings, so a kernel that mishandles any of them disagrees with
    /// the scalar reference.
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn compare_samples(mut compare: impl FnMut(&VertexBounds, Offset, &super::Edges, f32)) {
        let shapes = [
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
        ];
        for shape in shapes {
            for row in 0..=12 {
                for column in 0..=12 {
                    let point = Offset::new(column as f32 * 4.0 - 6.0, row as f32 * 4.0 - 6.0);
                    let edges = shape.edges(point);
                    let reference = scalar::distance_sq(&edges);
                    compare(&shape, point, &edges, reference);
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_x86_kernel_matches_the_scalar_reference() {
        let mut compared = 0u32;
        compare_samples(|shape, point, edges, reference| {
            // SAFETY: SSE2 is part of the x86_64 target.
            let sse2 = unsafe { super::x86::distance_sse2(edges) };
            assert!(
                close(sse2, reference),
                "sse2 {sse2} != scalar {reference} at ({}, {})",
                point.x,
                point.y
            );
            if cpuid::has(Feature::Avx2) {
                // SAFETY: the probe reported AVX2 and OS-enabled YMM state.
                let avx2 = unsafe { super::x86::distance_avx2(edges) };
                assert!(
                    close(avx2, reference),
                    "avx2 {avx2} != scalar {reference} at ({}, {})",
                    point.x,
                    point.y
                );
            }
            if cpuid::has(Feature::Avx512F) {
                // SAFETY: the probe reported AVX-512F and OS-enabled ZMM state.
                let avx512 = unsafe { super::x86::distance_avx512(edges) };
                assert!(
                    close(avx512, reference),
                    "avx512 {avx512} != scalar {reference} at ({}, {})",
                    point.x,
                    point.y
                );
            }
            let expected = if shape.contains(point) { 0.0 } else { reference };
            let dispatched = shape.distance_sq(point);
            assert!(
                close(dispatched, expected),
                "dispatch {dispatched} != {expected} at ({}, {})",
                point.x,
                point.y
            );
            compared += 1;
        });
        assert!(compared > 0);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn the_wasm_kernel_matches_the_scalar_reference() {
        assert!(cpuid::has(Feature::Simd128));
        let mut compared = 0u32;
        compare_samples(|shape, point, edges, reference| {
            let kernel = super::wasm::distance_sq(edges);
            assert!(
                close(kernel, reference),
                "wasm {kernel} != scalar {reference} at ({}, {})",
                point.x,
                point.y
            );
            let expected = if shape.contains(point) { 0.0 } else { reference };
            let dispatched = shape.distance_sq(point);
            assert!(
                close(dispatched, expected),
                "dispatch {dispatched} != {expected} at ({}, {})",
                point.x,
                point.y
            );
            compared += 1;
        });
        assert!(compared > 0);
    }
}
