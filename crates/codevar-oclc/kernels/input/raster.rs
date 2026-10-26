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

// Virtual Geometry Rasterization: the 8x8 index bitmap and its distance
// field.
//
// The host side of Kuitui picks a widget by rasterizing its convex shapes
// into an 8x8 bitmap and then measuring the exact distance only for the
// shapes near the pointer's cell. Both halves are a pure map, so the device
// takes them when the shape list is long enough to pay for a launch:
//
//   1. `raster_cells`    - one work item per shape, sampling the 64 cell
//                          centers with the same winding test the CPU uses
//                          and writing that shape's occupancy row.
//   2. `chamfer_cells`   - one work item per region, running the two-pass
//                          Chebyshev fill over its packed bitmap.
//
// Neither kernel reads another work item's output, so there is no barrier
// and no local memory: the host packs `raster_cells` occupancy rows into
// the topmost-indexed cell list before launching `chamfer_cells`.
//
// Shape layout - `shapes` holds nine floats per shape: the four vertex
// positions (x, y pairs) followed by the number of real vertices, three or
// four, stored as a float. The padding vertex of a triangle repeats the last
// real vertex but is never read because every loop runs over `count`.
//
// Area layout - `area` holds four floats: origin x, origin y, width, height,
// in logical pixels. A cell center is `origin + (index + 0.5) * size / 8`,
// which is the same sample point the CPU rasterizer uses.
//
// Launch `raster_cells` with a global size of at least `n_shapes` and a
// buffer of `64 * n_shapes` bytes for `occ`; launch `chamfer_cells` with a
// global size of at least `regions` and buffers of `64 * regions` bytes.
// Both kernels guard their work item, so extra items touch nothing, and each
// item writes only its own row.

/// Rasterizes every shape into its own 64-byte occupancy row.
#[kernel]
fn raster_cells(
    shapes: *const float,
    n_shapes: int,
    area: *const float,
    occ: *mut uchar,
) -> void {
    let gid = get_global_id(0) as int;
    if gid < n_shapes {
        let base = gid * 9;
        let count = shapes[base + 8] as int;
        let origin_x = area[0];
        let origin_y = area[1];
        let step_x = area[2] / 8.0;
        let step_y = area[3] / 8.0;
        let out = gid * 64;
        let mut cell = 0;
        while cell < 64 {
            let row = cell / 8;
            let column = cell - row * 8;
            let sample_x = origin_x + ((column as float) + 0.5) * step_x;
            let sample_y = origin_y + ((row as float) + 0.5) * step_y;
            // The winding test is order agnostic: the first non-collinear
            // cross product fixes the expected sign and any later edge with
            // the opposite sign puts the sample outside.
            let mut winding = 0.0;
            let mut inside = 1;
            let mut edge = 0;
            while edge < count {
                let next = edge + 1;
                let wrapped = next - (next / count) * count;
                let a = base + edge * 2;
                let b = base + wrapped * 2;
                let edge_x = shapes[b] - shapes[a];
                let edge_y = shapes[b + 1] - shapes[a + 1];
                let point_x = sample_x - shapes[a];
                let point_y = sample_y - shapes[a + 1];
                let cross = edge_x * point_y - edge_y * point_x;
                if cross != 0.0 {
                    if winding == 0.0 {
                        winding = cross;
                    } else if (cross > 0.0 && winding < 0.0)
                        || (cross < 0.0 && winding > 0.0)
                    {
                        inside = 0;
                        break;
                    }
                }
                edge = edge + 1;
            }
            occ[out + cell] = inside as uchar;
            cell = cell + 1;
        }
    }
}

/// Fills each region's Chebyshev distance field with two chamfer passes.
///
/// An occupied cell is zero and an empty one starts at the unreachable
/// sentinel 15, which is above the largest distance an 8x8 grid can hold (7)
/// and never overflows when a pass adds one to it.
///
/// The passes run in `distance` itself: a work item only ever touches its own
/// 64 bytes, so the output buffer doubles as its scratch and no work item
/// reads another's data.
#[kernel]
fn chamfer_cells(
    cells: *const uchar,
    regions: int,
    distance: *mut uchar,
) -> void {
    let gid = get_global_id(0) as int;
    if gid < regions {
        let base = gid * 64;
        let mut seed = 0;
        while seed < 64 {
            if cells[base + seed] != 0 {
                distance[base + seed] = 0;
            } else {
                distance[base + seed] = 15;
            }
            seed = seed + 1;
        }

        let mut forward_y = 0;
        while forward_y < 8 {
            let mut forward_x = 0;
            while forward_x < 8 {
                let cell = base + forward_y * 8 + forward_x;
                let mut best = distance[cell] as int;
                if forward_x > 0 {
                    best = min(best, (distance[cell - 1] as int) + 1);
                }
                if forward_y > 0 {
                    let up = cell - 8;
                    best = min(best, (distance[up] as int) + 1);
                    if forward_x > 0 {
                        best = min(best, (distance[up - 1] as int) + 1);
                    }
                    if forward_x < 7 {
                        best = min(best, (distance[up + 1] as int) + 1);
                    }
                }
                distance[cell] = best as uchar;
                forward_x = forward_x + 1;
            }
            forward_y = forward_y + 1;
        }

        let mut back_y = 7;
        while back_y >= 0 {
            let mut back_x = 7;
            while back_x >= 0 {
                let cell = base + back_y * 8 + back_x;
                let mut best = distance[cell] as int;
                if back_x < 7 {
                    best = min(best, (distance[cell + 1] as int) + 1);
                }
                if back_y < 7 {
                    let down = cell + 8;
                    best = min(best, (distance[down] as int) + 1);
                    if back_x > 0 {
                        best = min(best, (distance[down - 1] as int) + 1);
                    }
                    if back_x < 7 {
                        best = min(best, (distance[down + 1] as int) + 1);
                    }
                }
                distance[cell] = best as uchar;
                back_x = back_x - 1;
            }
            back_y = back_y - 1;
        }
    }
}
