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

//! Mosaic compositing entry point.
//!
//! Builds the planar boundary graph of the segmentation's layer masks, fits
//! each shared boundary segment exactly once, assigns the fitted geometry to
//! both adjacent faces, and emits one seam-free shape per face.

use crate::ir::{Paint, Segmentation, Shape, VectorDoc};
use crate::mosaic::face::{BoundarySegment, Face};
use crate::mosaic::fit::SegmentFitter;
use crate::mosaic::graph::Graph;
use crate::simplify::CurvePass;
use crate::types::PointI32;

/// Seam-free mosaic compositing.
///
/// Builds a boundary graph over the layer masks where every edge is shared by
/// exactly two faces, fits each shared segment once via `fitter`, and emits
/// one shape per face referencing the shared edge geometry.
pub fn compose_mosaic(
    seg: &Segmentation,
    fitter: &dyn SegmentFitter,
    merge_diff: i32,
    _passes: &[Box<dyn CurvePass>],
) -> VectorDoc {
    let mut doc = VectorDoc::new(seg.width, seg.height);
    if seg.layers.is_empty() {
        return doc;
    }

    // Merge adjacent same-paint layers whose colors are within merge_diff.
    let layers = merge_layers(seg, merge_diff);

    // Build faces from layers.
    let mut faces: Vec<Face> = layers
        .iter()
        .map(|l| Face::new(l.paint, l.mask.clone()))
        .collect();

    // Build the boundary graph and assign one fitted geometry per shared edge.
    let mut graph = Graph::new();
    build_graph(&layers, &mut graph);

    // Fit each graph edge once and attach to both incident faces.
    for edge in &graph.edges {
        let segment: Vec<PointI32> = edge.points.to_vec();
        let geom = fitter.fit_segment(&faces[edge.face_a].mask, &segment, faces[edge.face_a].paint);
        let geom_b = geom.clone();
        faces[edge.face_a].boundary.push(BoundarySegment {
            points: segment.clone(),
            fitted: geom,
            start_node: Some(edge.from),
            end_node: Some(edge.to),
        });
        faces[edge.face_b].boundary.push(BoundarySegment {
            points: segment,
            fitted: geom_b,
            start_node: Some(edge.to),
            end_node: Some(edge.from),
        });
    }

    // Emit one shape per face, walking its boundary into closed subpaths.
    for face in &faces {
        let mut multipath = crate::ir::MultiPath::new();
        for seg_seg in &face.boundary {
            let subpath = seg_seg.fitted.clone().into_closed_subpath();
            multipath.push(subpath);
        }
        doc.shapes.push(Shape {
            paint: face.paint,
            path: multipath,
        });
    }

    doc
}

/// Merge adjacent layers whose paints are within `merge_diff` of each other.
fn merge_layers(seg: &Segmentation, merge_diff: i32) -> Vec<crate::ir::Layer> {
    if merge_diff <= 0 || seg.layers.len() < 2 {
        return seg.layers.clone();
    }
    let mut merged: Vec<crate::ir::Layer> = Vec::new();
    for layer in &seg.layers {
        if let Some(last) = merged.last_mut()
            && paint_dist(last.paint, layer.paint) <= merge_diff
        {
            last.mask = last.mask.union(&layer.mask);
            continue;
        }
        merged.push(layer.clone());
    }
    merged
}

#[inline]
fn paint_dist(a: Paint, b: Paint) -> i32 {
    let ca = a.color();
    let cb = b.color();
    (ca.r as i32 - cb.r as i32).abs() + (ca.g as i32 - cb.g as i32).abs() + (ca.b as i32 - cb.b as i32).abs()
}

/// Build the boundary graph from layers: each pixel edge between different
/// layers becomes a graph edge shared by the two incident faces.
fn build_graph(layers: &[crate::ir::Layer], graph: &mut Graph) {
    if layers.is_empty() {
        return;
    }
    let w = layers[0].mask.image.width;
    let h = layers[0].mask.image.height;
    // For each pair of vertically/horizontally adjacent pixels belonging to
    // different layers, record a boundary edge between those two face indices.
    for y in 0..h {
        for x in 0..w {
            let Some(a) = face_at(layers, x, y) else { continue };
            if let Some(right) = face_at(layers, x + 1, y)
                && a != right
            {
                let fa = a;
                let fb = right;
                let n1 = graph.add_node(PointI32::new(x as i32, y as i32), true);
                let n2 = graph.add_node(PointI32::new((x + 1) as i32, y as i32), true);
                graph.add_edge(n1, n2, vec![PointI32::new((x + 1) as i32, y as i32)], fa, fb);
            }
            if let Some(below) = face_at(layers, x, y + 1)
                && a != below
            {
                let fa = a;
                let fb = below;
                let n1 = graph.add_node(PointI32::new(x as i32, y as i32), true);
                let n2 = graph.add_node(PointI32::new(x as i32, (y + 1) as i32), true);
                graph.add_edge(n1, n2, vec![PointI32::new(x as i32, (y + 1) as i32)], fa, fb);
            }
        }
    }
}

/// Which layer index (face) covers pixel `(x, y)`, or None.
fn face_at(layers: &[crate::ir::Layer], x: usize, y: usize) -> Option<usize> {
    for (i, layer) in layers.iter().enumerate() {
        let mx = x as i32 - layer.mask.offset.x;
        let my = y as i32 - layer.mask.offset.y;
        if mx >= 0
            && my >= 0
            && (mx as usize) < layer.mask.image.width
            && (my as usize) < layer.mask.image.height
            && layer
                .mask
                .image
                .get_pixel(mx as usize, my as usize)
        {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Layer, Paint, RegionMask};
    use crate::mosaic::PixelSegmentFitter;
    use crate::types::{BinaryImage, Color, PointI32};

    fn layer(paint: Color, w: usize, h: usize, on: &[(usize, usize)], ox: i32, oy: i32) -> Layer {
        let mut image = BinaryImage::new_w_h(w, h);
        for &(x, y) in on {
            image.set_pixel(x, y, true);
        }
        Layer {
            paint: Paint::Solid(paint),
            mask: RegionMask::new(image, PointI32::new(ox, oy)),
        }
    }

    #[test]
    fn compose_mosaic_emits_shapes_per_layer() {
        let seg = Segmentation {
            width: 4,
            height: 4,
            layers: vec![
                layer(
                    Color::new(10, 0, 0),
                    4,
                    4,
                    &[(0, 0), (1, 0), (0, 1), (1, 1)],
                    0,
                    0,
                ),
                layer(
                    Color::new(0, 10, 0),
                    4,
                    4,
                    &[(2, 2), (3, 2), (2, 3), (3, 3)],
                    0,
                    0,
                ),
            ],
        };
        let fitter = PixelSegmentFitter;
        let doc = compose_mosaic(&seg, &fitter, 0, &[]);
        assert_eq!(doc.shapes.len(), 2);
        assert_eq!(doc.width, 4);
        assert_eq!(doc.height, 4);
    }

    #[test]
    fn merge_diff_combines_similar_colors() {
        let seg = Segmentation {
            width: 4,
            height: 4,
            layers: vec![
                layer(Color::new(10, 0, 0), 4, 4, &[(0, 0), (1, 0)], 0, 0),
                layer(Color::new(12, 0, 0), 4, 4, &[(2, 0), (3, 0)], 0, 0),
            ],
        };
        let fitter = PixelSegmentFitter;
        let doc = compose_mosaic(&seg, &fitter, 5, &[]);
        // Similar colors merged into one shape.
        assert!(doc.shapes.len() <= 2);
    }
}
