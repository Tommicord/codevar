// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and
// limitations under the License.

// Virtual Geometry Rasterization: the narrow phase of pointer picking.
//
// The 8x8 index bitmap turns "which shape is under the pointer?" into a
// handful of candidate cells; the shapes those cells name still need an
// exact measurement, because the distance decides the winner and a distance
// of zero is what "inside the box" means. Measuring a long shape list is a
// pure map - every shape is independent - so the device takes it:
//
//   1. `pick_distances` - one work item per shape, writing that shape's
//                          squared distance to the pointer, or zero when the
//                          pointer is inside it.
//
// The host keeps the selection rule (nearest wins, the pointer's own cell
// and the topmost shape break ties) because it reads the bitmap; the device
// only pays for the math. Reading back `n_shapes` floats is cheaper than
// shipping the tie-breaking rule to the device.
//
// Shape layout - identical to `raster.rs`: nine floats per shape, the four
// vertex positions as (x, y) pairs followed by the number of real vertices
// stored as a float. `point` is two floats, the pointer position in logical
// pixels.
//
// Launch with a global size of at least `n_shapes` and a buffer of
// `4 * n_shapes` bytes for `dist_sq`; the guard discards extra items. The
// kernel measures every edge including its endpoints, so a point beyond a
// corner gets the corner distance rather than the distance to the infinite
// edge line - the same contract the CPU SIMD kernels implement.

/// Measures the squared distance from one point to every convex shape.
#[kernel]
fn pick_distances(
    shapes: *const float,
    n_shapes: int,
    point: *const float,
    dist_sq: *mut float,
) -> void {
    let gid = get_global_id(0) as int;
    if gid < n_shapes {
        let base = gid * 9;
        let count = shapes[base + 8] as int;
        let px = point[0];
        let py = point[1];

        // Containment first: the winding test fixes its sign from the first
        // non-collinear edge and any edge of the opposite sign leaves the
        // shape. An inside pointer is at distance zero.
        let mut winding = 0.0;
        let mut outside = 0;
        let mut edge = 0;
        while edge < count {
            let next = edge + 1;
            let wrapped = next - (next / count) * count;
            let a = base + edge * 2;
            let b = base + wrapped * 2;
            let edge_x = shapes[b] - shapes[a];
            let edge_y = shapes[b + 1] - shapes[a + 1];
            let point_x = px - shapes[a];
            let point_y = py - shapes[a + 1];
            let cross = edge_x * point_y - edge_y * point_x;
            if cross != 0.0 {
                if winding == 0.0 {
                    winding = cross;
                } else if (cross > 0.0 && winding < 0.0)
                    || (cross < 0.0 && winding > 0.0)
                {
                    outside = 1;
                    break;
                }
            }
            edge = edge + 1;
        }

        if outside == 0 {
            dist_sq[gid] = 0.0;
        } else {
            // Outside: the nearest point of any edge segment, clamped
            // projection and all, and the smallest squared distance wins.
            let mut best = 0.0;
            let mut first = 1;
            let mut segment = 0;
            while segment < count {
                let next = segment + 1;
                let wrapped = next - (next / count) * count;
                let a = base + segment * 2;
                let b = base + wrapped * 2;
                let ax = shapes[a];
                let ay = shapes[a + 1];
                let ab_x = shapes[b] - ax;
                let ab_y = shapes[b + 1] - ay;
                let ap_x = px - ax;
                let ap_y = py - ay;
                let length_sq = ab_x * ab_x + ab_y * ab_y;
                let mut t = 0.0;
                if length_sq > 0.0 {
                    t = (ap_x * ab_x + ap_y * ab_y) / length_sq;
                    if t < 0.0 {
                        t = 0.0;
                    }
                    if t > 1.0 {
                        t = 1.0;
                    }
                }
                let dx = ap_x - ab_x * t;
                let dy = ap_y - ab_y * t;
                let distance = dx * dx + dy * dy;
                if first != 0 || distance < best {
                    best = distance;
                    first = 0;
                }
                segment = segment + 1;
            }
            dist_sq[gid] = best;
        }
    }
}
