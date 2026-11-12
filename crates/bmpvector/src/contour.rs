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

//! Pixel-edge contour extraction.
//!
//! A component's mask is converted into oriented boundary edges along the
//! lattice between pixels. Every inside pixel contributes the sides that face
//! an outside pixel, with the inside kept on the right of the direction of
//! travel. Following those edges with a right-hand rule produces one closed
//! loop per boundary:
//!
//! * outer boundaries wind clockwise on screen and have positive signed area;
//! * holes wind counter-clockwise and have negative signed area.
//!
//! Because the loops are traced from the component's own mask, holes are
//! discovered without a separate containment search. The output document uses
//! the even-odd fill rule, so the set of loops for a component is already a
//! complete description of its covered area.
//!
//! The implementation stores outgoing edges as an intrusive linked list per
//! lattice vertex. That avoids a hash map in the hot path and makes the walk
//! cache-friendly: following a contour touches one edge record at a time.

use alloc::vec;
use alloc::vec::Vec;

use crate::component::Component;
use crate::error::{Error, Result};
use crate::geom::Point;
use crate::polygon::polygon_signed_area;

/// A compact bitset over a rectangular grid.
///
/// Bits are stored row-major in 64-bit words. `width` and `height` include only
/// the component's bounding box, not the full image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitGrid {
    width: u32,
    height: u32,
    stride: usize,
    words: Vec<u64>,
}

impl BitGrid {
    /// An empty grid of the given size.
    pub fn new(width: u32, height: u32) -> Self {
        let stride = (width as usize).div_ceil(64);
        let word_count = stride.saturating_mul(height as usize);
        Self {
            width,
            height,
            stride,
            words: vec![0; word_count],
        }
    }

    /// Builds a local mask from a component's pixel list.
    ///
    /// `image_width` is the width of the source image, used to unpack the
    /// component's row-major pixel offsets. The returned grid is positioned at
    /// the component's bounding box origin.
    ///
    /// The component is checked before it is used: every pixel has to fall
    /// inside the declared bounds, and the declared area has to match the
    /// number of pixels. A component that fails either check describes a
    /// region that does not exist, and silently tracing it would produce
    /// geometry that does not match the source image.
    pub fn from_component(component: &Component, image_width: u32) -> Result<Self> {
        let width = component.bounds.width();
        let height = component.bounds.height();
        if width == 0 || height == 0 {
            return Err(Error::EmptyComponent {
                component: component.palette_index as usize,
            });
        }
        if component.pixels.len() as u64 != component.area {
            return Err(Error::InconsistentComponent {
                component: component.palette_index as usize,
            });
        }
        let mut grid = Self::new(width, height);
        for &packed in &component.pixels {
            let global_x = packed % image_width;
            let global_y = packed / image_width;
            if global_x < component.bounds.left
                || global_y < component.bounds.top
                || global_x >= component.bounds.right
                || global_y >= component.bounds.bottom
            {
                return Err(Error::InconsistentComponent {
                    component: component.palette_index as usize,
                });
            }
            let x = global_x - component.bounds.left;
            let y = global_y - component.bounds.top;
            grid.set(x, y, true);
        }
        Ok(grid)
    }

    /// Grid width in pixels.
    #[inline]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Grid height in pixels.
    #[inline]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Number of set bits.
    pub fn count_ones(&self) -> u64 {
        self.words
            .iter()
            .map(|word| word.count_ones() as u64)
            .sum()
    }

    /// Whether the grid has no set bits.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    /// Returns the bit at `(x, y)`, treating out-of-range coordinates as zero.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.get_unchecked(x, y)
    }

    /// Returns the bit at `(x, y)`.
    ///
    /// The caller must prove the coordinates are inside the grid.
    #[inline]
    pub(crate) fn get_unchecked(&self, x: u32, y: u32) -> bool {
        debug_assert!(x < self.width && y < self.height);
        let word = y as usize * self.stride + (x as usize / 64);
        (self.words[word] >> (x % 64)) & 1 != 0
    }

    /// Sets the bit at `(x, y)`.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, value: bool) {
        if x >= self.width || y >= self.height {
            return;
        }
        let word = y as usize * self.stride + (x as usize / 64);
        let mask = 1_u64 << (x % 64);
        if value {
            self.words[word] |= mask;
        } else {
            self.words[word] &= !mask;
        }
    }

    /// Clears the grid.
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Fills the grid with a value.
    pub fn fill(&mut self, value: bool) {
        let fill = if value { u64::MAX } else { 0 };
        self.words.fill(fill);
        self.clear_unused_bits();
    }

    fn clear_unused_bits(&mut self) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let used_bits = self.width % 64;
        if used_bits == 0 {
            return;
        }
        let mask = (1_u64 << used_bits) - 1;
        for y in 0..self.height as usize {
            let word = y * self.stride + self.stride - 1;
            self.words[word] &= mask;
        }
    }
}

/// One traced boundary ring.
#[derive(Clone, Debug, PartialEq)]
pub struct Contour {
    /// Vertices in document coordinates. The closing vertex is not repeated.
    pub points: Vec<Point>,
    /// Signed area in pixel units. Positive for an outer ring.
    pub signed_area: f64,
    /// Whether the ring encloses a hole.
    pub is_hole: bool,
}

impl Contour {
    /// Builds a contour from its vertices.
    pub fn new(points: Vec<Point>) -> Self {
        let signed_area = polygon_signed_area(&points);
        Self {
            points,
            signed_area,
            is_hole: signed_area < 0.0,
        }
    }

    /// Number of vertices.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.points.len()
    }

    /// Unsigned area in pixel units.
    #[inline]
    pub fn area(&self) -> f64 {
        libm::fabs(self.signed_area)
    }

    /// Reverses the ring orientation.
    pub fn reverse(&mut self) {
        self.points.reverse();
        self.signed_area = -self.signed_area;
        self.is_hole = self.signed_area < 0.0;
    }
}

/// Traces every boundary ring of a component.
///
/// `origin_x` and `origin_y` are the component's bounding-box origin in the
/// source image; vertices are translated by that amount to document space.
pub fn trace_component(component: &Component, image_width: u32, component_id: usize) -> Result<Vec<Contour>> {
    let grid = BitGrid::from_component(component, image_width)?;
    trace_grid(&grid, component.bounds.left, component.bounds.top, component_id)
}

/// Traces every boundary ring of a local bit grid.
pub fn trace_grid(grid: &BitGrid, origin_x: u32, origin_y: u32, component_id: usize) -> Result<Vec<Contour>> {
    if grid.is_empty() {
        return Err(Error::EmptyComponent {
            component: component_id,
        });
    }

    let vertex_width = grid.width as usize + 1;
    let vertex_height = grid.height as usize + 1;
    let vertex_count = vertex_width
        .checked_mul(vertex_height)
        .ok_or(Error::NumericalFailure {
            operation: "contour vertex grid",
        })?;

    let mut head: Vec<i32> = vec![-1; vertex_count];
    let mut next: Vec<i32> = Vec::new();
    let mut from: Vec<u32> = Vec::new();
    let mut to: Vec<u32> = Vec::new();
    let mut direction: Vec<u8> = Vec::new();
    let mut visited: Vec<bool> = Vec::new();

    for y in 0..grid.height {
        for x in 0..grid.width {
            if !grid.get_unchecked(x, y) {
                continue;
            }
            if !grid.get(x, y.wrapping_sub(1)) || y == 0 {
                add_edge(
                    (y as usize * vertex_width + x as usize) as u32,
                    (y as usize * vertex_width + x as usize + 1) as u32,
                    Direction::East,
                    &mut head,
                    &mut next,
                    &mut from,
                    &mut to,
                    &mut direction,
                    &mut visited,
                );
            }
            if x + 1 >= grid.width || !grid.get(x + 1, y) {
                add_edge(
                    (y as usize * vertex_width + x as usize + 1) as u32,
                    ((y as usize + 1) * vertex_width + x as usize + 1) as u32,
                    Direction::South,
                    &mut head,
                    &mut next,
                    &mut from,
                    &mut to,
                    &mut direction,
                    &mut visited,
                );
            }
            if y + 1 >= grid.height || !grid.get(x, y + 1) {
                add_edge(
                    ((y as usize + 1) * vertex_width + x as usize + 1) as u32,
                    ((y as usize + 1) * vertex_width + x as usize) as u32,
                    Direction::West,
                    &mut head,
                    &mut next,
                    &mut from,
                    &mut to,
                    &mut direction,
                    &mut visited,
                );
            }
            if x == 0 || !grid.get(x.wrapping_sub(1), y) {
                add_edge(
                    ((y as usize + 1) * vertex_width + x as usize) as u32,
                    (y as usize * vertex_width + x as usize) as u32,
                    Direction::North,
                    &mut head,
                    &mut next,
                    &mut from,
                    &mut to,
                    &mut direction,
                    &mut visited,
                );
            }
        }
    }

    let edge_count = next.len();
    let mut contours = Vec::new();
    for start_edge in 0..edge_count {
        if visited[start_edge] {
            continue;
        }
        let start_vertex = from[start_edge];
        let mut current_vertex = start_vertex;
        let mut current_edge = start_edge as i32;
        let mut points = Vec::new();
        let mut steps = 0_usize;

        loop {
            if steps > edge_count {
                return Err(Error::BrokenContour {
                    component: component_id,
                    vertex: vertex_to_xy(current_vertex, vertex_width),
                });
            }
            let edge = current_edge as usize;
            if visited[edge] {
                return Err(Error::BrokenContour {
                    component: component_id,
                    vertex: vertex_to_xy(current_vertex, vertex_width),
                });
            }
            visited[edge] = true;
            steps += 1;

            let (x, y) = vertex_to_xy(current_vertex, vertex_width);
            points.push(Point::new(x as f64 + origin_x as f64, y as f64 + origin_y as f64));

            current_vertex = to[edge];
            if current_vertex == start_vertex {
                break;
            }

            let incoming = direction[edge];
            let mut candidate = head[current_vertex as usize];
            let mut best_edge: Option<(u8, i32)> = None;
            while candidate >= 0 {
                let candidate_usize = candidate as usize;
                if !visited[candidate_usize] {
                    let outgoing = direction[candidate_usize];
                    let turn = (outgoing + 4 - incoming) % 4;
                    let rank = turn_rank(turn);
                    match best_edge {
                        None => best_edge = Some((rank, candidate)),
                        Some((best_rank, _)) if rank < best_rank => {
                            best_edge = Some((rank, candidate));
                        }
                        _ => {}
                    }
                }
                candidate = next[candidate_usize];
            }

            let Some((_, next_edge)) = best_edge else {
                return Err(Error::BrokenContour {
                    component: component_id,
                    vertex: vertex_to_xy(current_vertex, vertex_width),
                });
            };
            current_edge = next_edge;
        }

        // A loop with fewer than three vertices cannot enclose an area. The
        // edge construction never produces one for a non-empty grid, but the
        // guard keeps malformed caller-built grids from reaching the fitter.
        if points.len() >= 3 {
            contours.push(Contour::new(points));
        }
    }

    Ok(contours)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Direction {
    East = 0,
    South = 1,
    West = 2,
    North = 3,
}

impl Direction {
    #[inline]
    const fn value(self) -> u8 {
        self as u8
    }
}

/// Ranks outgoing directions for the right-hand rule.
///
/// A right turn is preferred, then straight, then left, and reversing is the
/// last resort. Lower values win.
#[inline]
const fn turn_rank(turn: u8) -> u8 {
    match turn {
        1 => 0,
        0 => 1,
        3 => 2,
        _ => 3,
    }
}

#[allow(clippy::too_many_arguments)]
fn add_edge(
    start: u32,
    end: u32,
    direction: Direction,
    head: &mut [i32],
    next: &mut Vec<i32>,
    from: &mut Vec<u32>,
    to: &mut Vec<u32>,
    directions: &mut Vec<u8>,
    visited: &mut Vec<bool>,
) {
    let edge = next.len();
    next.push(head[start as usize]);
    head[start as usize] = edge as i32;
    from.push(start);
    to.push(end);
    directions.push(direction.value());
    visited.push(false);
}

#[inline]
fn vertex_to_xy(vertex: u32, vertex_width: usize) -> (u32, u32) {
    let vertex = vertex as usize;
    ((vertex % vertex_width) as u32, (vertex / vertex_width) as u32)
}

/// A convenience wrapper that returns the outer and hole rings separately.
pub fn trace_component_split(
    component: &Component,
    image_width: u32,
    component_id: usize,
) -> Result<(Vec<Contour>, Vec<Contour>)> {
    let contours = trace_component(component, image_width, component_id)?;
    let mut outer = Vec::new();
    let mut holes = Vec::new();
    for contour in contours {
        if contour.is_hole {
            holes.push(contour);
        } else {
            outer.push(contour);
        }
    }
    Ok((outer, holes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::component::Connectivity;
    use crate::palette::{ColorMode, IndexedImage, NO_INDEX, Palette};
    use crate::quantize::quantize;

    fn grid_from_rows(rows: &[&str]) -> BitGrid {
        let width = rows
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0) as u32;
        let mut grid = BitGrid::new(width, rows.len() as u32);
        for (y, row) in rows.iter().enumerate() {
            for (x, character) in row.chars().enumerate() {
                grid.set(x as u32, y as u32, character == '#');
            }
        }
        grid
    }

    #[test]
    fn single_pixel_traces_a_unit_square() {
        let grid = grid_from_rows(&["#"]);
        let contours = trace_grid(&grid, 0, 0, 0).expect("trace");
        assert_eq!(contours.len(), 1);
        let contour = &contours[0];
        assert_eq!(contour.vertex_count(), 4);
        assert!(libm::fabs(contour.signed_area - 1.0) < 1.0e-12);
        assert!(!contour.is_hole);
    }

    #[test]
    fn filled_square_traces_one_ring() {
        let grid = grid_from_rows(&["####", "####", "####", "####"]);
        let contours = trace_grid(&grid, 0, 0, 0).expect("trace");
        assert_eq!(contours.len(), 1);
        assert!(libm::fabs(contours[0].signed_area - 16.0) < 1.0e-12);
        assert_eq!(contours[0].vertex_count(), 16);
    }

    #[test]
    fn ring_traces_outer_boundary_and_hole() {
        let grid = grid_from_rows(&["###", "#.#", "###"]);
        let contours = trace_grid(&grid, 0, 0, 0).expect("trace");
        assert_eq!(contours.len(), 2);
        let outer = contours
            .iter()
            .find(|contour| !contour.is_hole)
            .expect("outer");
        let hole = contours
            .iter()
            .find(|contour| contour.is_hole)
            .expect("hole");
        assert!(
            libm::fabs(outer.signed_area - 9.0) < 1.0e-12,
            "outer {} hole {}",
            outer.signed_area,
            hole.signed_area
        );
        assert!(libm::fabs(hole.signed_area + 1.0) < 1.0e-12);
    }

    #[test]
    fn two_separate_regions_trace_two_rings() {
        let grid = grid_from_rows(&["##..", "##..", "..##", "..##"]);
        let contours = trace_grid(&grid, 0, 0, 0).expect("trace");
        assert_eq!(contours.len(), 2);
        let total: f64 = contours
            .iter()
            .map(|contour| contour.signed_area)
            .sum();
        assert!(libm::fabs(total - 8.0) < 1.0e-12);
    }

    #[test]
    fn diagonal_pixels_trace_separate_loops() {
        let grid = grid_from_rows(&["#.", ".#"]);
        let contours = trace_grid(&grid, 0, 0, 0).expect("trace");
        assert_eq!(contours.len(), 2);
        assert!(
            contours
                .iter()
                .all(|contour| contour.signed_area > 0.0)
        );
    }

    #[test]
    fn origin_translation_is_applied() {
        let grid = grid_from_rows(&["#", "#"]);
        let contours = trace_grid(&grid, 5, 7, 0).expect("trace");
        assert_eq!(contours[0].points.first(), Some(&Point::new(5.0, 7.0)));
        assert_eq!(
            contours[0]
                .points
                .iter()
                .map(|point| point.y)
                .fold(f64::MIN, f64::max),
            9.0
        );
    }

    #[test]
    fn bit_grid_set_get_and_count() {
        let mut grid = BitGrid::new(70, 3);
        grid.set(0, 0, true);
        grid.set(64, 2, true);
        assert!(grid.get(0, 0));
        assert!(grid.get(64, 2));
        assert!(!grid.get(1, 0));
        assert!(!grid.get(64, 1));
        assert_eq!(grid.count_ones(), 2);
        grid.set(0, 0, false);
        assert_eq!(grid.count_ones(), 1);
        grid.fill(true);
        assert_eq!(grid.count_ones(), 70 * 3);
    }

    #[test]
    fn an_inconsistent_component_is_rejected() {
        let mut component = Component {
            palette_index: 0,
            bounds: crate::geom::PixelRect::new(0, 0, 2, 2),
            area: 4,
            pixels: vec![0, 1, 2, 3],
        };
        assert!(BitGrid::from_component(&component, 2).is_ok());

        // The declared area no longer matches the pixel list.
        component.area = 9;
        assert!(matches!(
            BitGrid::from_component(&component, 2),
            Err(Error::InconsistentComponent { component: 0 })
        ));

        // A pixel lies outside the declared bounds.
        component.area = 4;
        component.pixels = vec![0, 1, 2, 99];
        assert!(matches!(
            BitGrid::from_component(&component, 2),
            Err(Error::InconsistentComponent { component: 0 })
        ));
    }

    #[test]
    fn component_tracing_uses_absolute_coordinates() {
        let source = crate::image::ColorImage::from_luma8(
            4,
            4,
            &[0, 0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        )
        .expect("image");
        let indexed = quantize(&source, &ColorMode::Exact, None, 1).expect("quantize");
        let segmentation = crate::segment::label_components(&indexed, Connectivity::Four).expect("label");
        let component = segmentation
            .components
            .iter()
            .find(|component| component.color(&segmentation.palette) == Some(Color::WHITE))
            .expect("white component");
        let contours = trace_component(component, source.width(), 0).expect("trace component");
        assert!(!contours.is_empty());
        assert!(
            contours
                .iter()
                .flat_map(|contour| &contour.points)
                .all(|point| point.x >= 1.0 && point.x <= 3.0 && point.y >= 1.0 && point.y <= 3.0)
        );
    }

    #[test]
    fn split_returns_outer_and_holes() {
        let palette = Palette::new(vec![Color::BLACK]).expect("palette");
        let image = IndexedImage::from_parts(3, 3, vec![0, 0, 0, 0, NO_INDEX, 0, 0, 0, 0], palette);
        let segmentation = crate::segment::label_components(&image, Connectivity::Four).expect("label");
        let component = &segmentation.components[0];
        let (outer, holes) = trace_component_split(component, 3, 0).expect("trace");
        assert_eq!(outer.len(), 1);
        assert_eq!(holes.len(), 1);
    }

    #[test]
    fn empty_grid_is_rejected() {
        let grid = BitGrid::new(2, 2);
        assert!(matches!(
            trace_grid(&grid, 0, 0, 4),
            Err(Error::EmptyComponent { .. })
        ));
    }
}
