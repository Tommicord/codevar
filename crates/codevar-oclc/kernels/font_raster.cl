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
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

// Font rasterization: analytic anti-aliased coverage for one glyph.
//
// The host flattens the glyph outline (conics and cubics included) into
// a list of straight line segments and uploads them as `edges`: four
// floats per segment in contour order — x1, y1, x2, y2 — in pixel
// coordinates. Every contour must be closed explicitly by repeating its
// first point at the end.
//
// Each work item owns exactly one output pixel and samples it on a 2x2
// sub-pixel grid (four samples at 0.25/0.75 offsets). For every sample
// the winding number is accumulated across all edges with the classic
// PNPOLY crossing test, the fill rule picks inside/outside, and the
// four decisions merge into one 8-bit coverage value — the same
// `PixelMode::Gray` 0..=255 contract the CPU rasterizer produces.
//
// `even_odd` selects the fill rule: 0 applies the non-zero winding
// rule (the outline's default), 1 the even-odd rule
// (`OUTLINE_EVEN_ODD_FILL`).
//
// Launch with a global size covering at least width x height (extra
// work items are discarded by the bounds guard), and allocate
// width * height bytes for `coverage`. The kernel never reads or
// writes outside `edges[0 .. 4 * n_edges]` and `coverage[0 .. width *
// height]`, and it performs no synchronization, so a plain
// one-dimensional or two-dimensional dispatch is enough.
#[kernel]
fn font_raster(
    edges: *const float,
    n_edges: int,
    width: int,
    height: int,
    even_odd: int,
    coverage: *mut uchar,
) -> void {
    let px = get_global_id(0) as int;
    let py = get_global_id(1) as int;
    if px < width && py < height {
        // Sub-pixel hits out of the four 2x2 samples.
        let mut hits = 0;
        for sy in 0..2 {
            let sample_y = (py as float) + 0.25 + (sy as float) * 0.5;
            for sx in 0..2 {
                let sample_x = (px as float) + 0.25 + (sx as float) * 0.5;

                // Signed winding number (non-zero rule) and crossing
                // parity (even-odd rule) accumulated together.
                let mut wn = 0;
                let mut parity = 0;
                let mut e = 0;
                while e < n_edges {
                    let base = e * 4;
                    let x1 = edges[base];
                    let y1 = edges[base + 1];
                    let x2 = edges[base + 2];
                    let y2 = edges[base + 3];
                    // Which side of the directed edge the sample sits on.
                    let cross = (x2 - x1) * (sample_y - y1) - (sample_x - x1) * (y2 - y1);
                    if y1 <= sample_y {
                        if y2 > sample_y && cross > 0.0 {
                            wn += 1;
                            parity = 1 - parity;
                        }
                    } else {
                        if y2 <= sample_y && cross < 0.0 {
                            wn -= 1;
                            parity = 1 - parity;
                        }
                    }
                    e += 1;
                }

                let mut hit = 0;
                if even_odd != 0 {
                    if parity != 0 {
                        hit = 1;
                    }
                } else {
                    if wn != 0 {
                        hit = 1;
                    }
                }
                hits += hit;
            }
        }

        // Round hits * (255 / 4) to the nearest level: 0, 64, 128,
        // 191, 255.
        coverage[py * width + px] = ((hits * 255 + 2) / 4) as uchar;
    }
}
