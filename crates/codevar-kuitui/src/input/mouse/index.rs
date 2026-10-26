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

//! The 8x8 index bitmap and pointer picking over it.
//!
//! Rasterizing samples each cell center against every shape, so the bitmap
//! costs at most 64 tests per shape and needs no coverage math. A cell stores
//! `index + 1` of the topmost shape, and a two-pass chamfer fill records the
//! Chebyshev distance from every cell to the nearest occupied one.

use crate::geometry::{Offset, Rect};

use super::bounds::{BoundId, VertexBounds};

/// Width and height of the rasterized index bitmap, in cells.
pub const GRID_SIDE: usize = 8;
/// Number of cells in the rasterized index bitmap.
pub const GRID_CELLS: usize = GRID_SIDE * GRID_SIDE;

/// The largest number of shapes a bitmap can name.
///
/// A cell stores `index + 1`, so it can name at most 255 shapes. Shapes past
/// the limit are ignored by the bitmap; a widget with that many interactive
/// shapes should split them across several regions.
pub const MAX_BOUNDS: usize = u8::MAX as usize;

/// The rasterized index bitmap of a shape list.
///
/// Cell `(x, y)` names the topmost shape whose center sample fell inside it,
/// encoded as `shape_index + 1`. Zero means the cell is empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryIndex {
    /// Encoded shape index per cell, row-major.
    cells: [u8; GRID_CELLS],
    /// Chebyshev distance from each cell to the nearest occupied cell.
    distance: [u8; GRID_CELLS],
    /// Number of occupied cells.
    occupied: u8,
}

impl GeometryIndex {
    /// An index bitmap with no occupied cells.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            cells: [0; GRID_CELLS],
            distance: [u8::MAX; GRID_CELLS],
            occupied: 0,
        }
    }

    /// Rasterizes `bounds` over `area`.
    ///
    /// A cell is occupied when its center lies inside a shape. Shapes are
    /// sampled in order, so a shape later in the list overwrites an earlier
    /// one in a shared cell, matching the painting order.
    #[must_use]
    pub fn rasterize(bounds: &[VertexBounds], area: Rect) -> Self {
        let mut index = Self::empty();
        if area.is_empty() || bounds.is_empty() {
            return index;
        }
        let step_x = area.size.width / GRID_SIDE as f32;
        let step_y = area.size.height / GRID_SIDE as f32;

        for (position, shape) in bounds.iter().take(MAX_BOUNDS).enumerate() {
            if !shape.bound().intersects(area) {
                continue;
            }
            let encoded = u8::try_from(position + 1).unwrap_or(u8::MAX);
            for y in 0..GRID_SIDE {
                let center_y = area.origin.y + (y as f32 + 0.5) * step_y;
                for x in 0..GRID_SIDE {
                    let center_x = area.origin.x + (x as f32 + 0.5) * step_x;
                    if shape.contains(Offset::new(center_x, center_y)) {
                        index.cells[y * GRID_SIDE + x] = encoded;
                    }
                }
            }
        }

        index.occupied = index
            .cells
            .iter()
            .filter(|cell| **cell != 0)
            .count()
            .try_into()
            .unwrap_or(GRID_CELLS as u8);
        index.fill_distance();
        index
    }

    /// The raw index bitmap, row-major, with `index + 1` per cell.
    #[must_use]
    pub const fn cells(&self) -> &[u8; GRID_CELLS] {
        &self.cells
    }

    /// The shape index of the cell at `(x, y)`.
    #[must_use]
    pub fn cell(&self, x: usize, y: usize) -> Option<usize> {
        let encoded = *self.cells.get(y * GRID_SIDE + x)?;
        (encoded != 0).then(|| usize::from(encoded) - 1)
    }

    /// The Chebyshev distance from the cell at `(x, y)` to the nearest
    /// occupied cell, or `None` when the coordinates are outside the grid.
    #[must_use]
    pub fn cell_distance(&self, x: usize, y: usize) -> Option<u32> {
        self.distance
            .get(y * GRID_SIDE + x)
            .map(|value| u32::from(*value))
    }

    /// Number of occupied cells.
    #[must_use]
    pub const fn occupied_cells(&self) -> u32 {
        self.occupied as u32
    }

    /// Whether no shape covered any cell center.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.occupied == 0
    }

    /// The cell that contains `point` in `area`, clamped to the grid.
    #[must_use]
    pub fn cell_of(area: Rect, point: Offset) -> (usize, usize) {
        if area.is_empty() {
            return (0, 0);
        }
        let x = libm::floorf((point.x - area.origin.x) / area.size.width * GRID_SIDE as f32);
        let y = libm::floorf((point.y - area.origin.y) / area.size.height * GRID_SIDE as f32);
        (
            x.clamp(0.0, (GRID_SIDE - 1) as f32) as usize,
            y.clamp(0.0, (GRID_SIDE - 1) as f32) as usize,
        )
    }

    /// Finds the shape under `point`.
    ///
    /// The bitmap narrows the work to the occupied cells within one cell of
    /// the nearest one; only those shapes are measured exactly. Returns `None`
    /// when no cell is occupied.
    #[must_use]
    pub fn pick(&self, bounds: &[VertexBounds], area: Rect, point: Offset) -> Option<Hit> {
        if self.is_empty() || bounds.is_empty() || area.is_empty() {
            return None;
        }
        let (mouse_x, mouse_y) = Self::cell_of(area, point);
        let reach = u32::from(self.distance[mouse_y * GRID_SIDE + mouse_x]).saturating_add(1);

        let mut best: Option<Hit> = None;
        for y in 0..GRID_SIDE {
            for x in 0..GRID_SIDE {
                let cell = y * GRID_SIDE + x;
                let encoded = self.cells[cell];
                if encoded == 0 || u32::from(self.distance[cell]) > reach {
                    continue;
                }
                let shape_index = usize::from(encoded) - 1;
                let Some(shape) = bounds.get(shape_index) else {
                    continue;
                };
                let distance = shape.distance(point);
                let candidate = Hit {
                    index: shape_index,
                    id: shape.id(),
                    cell: (x as u8, y as u8),
                    cell_distance: u32::from(self.distance[cell]),
                    distance,
                };
                let own_cell = x == mouse_x && y == mouse_y;
                let better = match best {
                    None => true,
                    Some(current) if distance != current.distance => distance < current.distance,
                    Some(current) => {
                        let current_own_cell =
                            current.cell.0 as usize == mouse_x && current.cell.1 as usize == mouse_y;
                        match (own_cell, current_own_cell) {
                            (true, false) => true,
                            (false, true) => false,
                            _ => {
                                candidate.cell_distance < current.cell_distance
                                    || (candidate.cell_distance == current.cell_distance
                                        && shape_index > current.index)
                            }
                        }
                    }
                };
                if better {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    /// Fills [`Self::distance`] with the Chebyshev distance to the nearest
    /// occupied cell via two chamfer passes.
    fn fill_distance(&mut self) {
        let mut distance = [u8::MAX; GRID_CELLS];
        for (cell, encoded) in self.cells.iter().enumerate() {
            if *encoded != 0 {
                distance[cell] = 0;
            }
        }

        for y in 0..GRID_SIDE {
            for x in 0..GRID_SIDE {
                let cell = y * GRID_SIDE + x;
                let mut best = distance[cell];
                if x > 0 {
                    best = best.min(distance[cell - 1].saturating_add(1));
                }
                if y > 0 {
                    let up = cell - GRID_SIDE;
                    best = best.min(distance[up].saturating_add(1));
                    if x > 0 {
                        best = best.min(distance[up - 1].saturating_add(1));
                    }
                    if x + 1 < GRID_SIDE {
                        best = best.min(distance[up + 1].saturating_add(1));
                    }
                }
                distance[cell] = best;
            }
        }

        for y in (0..GRID_SIDE).rev() {
            for x in (0..GRID_SIDE).rev() {
                let cell = y * GRID_SIDE + x;
                let mut best = distance[cell];
                if x + 1 < GRID_SIDE {
                    best = best.min(distance[cell + 1].saturating_add(1));
                }
                if y + 1 < GRID_SIDE {
                    let down = cell + GRID_SIDE;
                    best = best.min(distance[down].saturating_add(1));
                    if x > 0 {
                        best = best.min(distance[down - 1].saturating_add(1));
                    }
                    if x + 1 < GRID_SIDE {
                        best = best.min(distance[down + 1].saturating_add(1));
                    }
                }
                distance[cell] = best;
            }
        }

        self.distance = distance;
    }
}

/// The shape under the pointer and how far the pointer is from it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Index of the shape in the list that produced the bitmap.
    pub index: usize,
    /// Identifier of that shape.
    pub id: BoundId,
    /// Cell whose shape was measured.
    pub cell: (u8, u8),
    /// Chebyshev distance in bitmap cells from the pointer's cell.
    pub cell_distance: u32,
    /// Distance in logical pixels from the pointer to the shape.
    pub distance: f32,
}

impl Hit {
    /// Whether the pointer is inside the shape.
    ///
    /// Inside, on an edge and on a vertex all measure exactly zero, which is
    /// the condition the bitmap design uses to define "inside the box".
    #[must_use]
    pub fn is_inside(self) -> bool {
        self.distance <= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::{GRID_CELLS, GeometryIndex, Hit};
    use crate::geometry::{Offset, Rect, Size};
    use crate::input::mouse::bounds::{BoundId, VertexBounds};

    fn area() -> Rect {
        Rect::from_min_size(Offset::ZERO, 80.0, 80.0)
    }

    fn rect_shape(id: u32, x: f32, y: f32, width: f32, height: f32) -> VertexBounds {
        VertexBounds::rect(
            BoundId::new(id),
            Rect::from_min_size(Offset::new(x, y), width, height),
        )
    }

    #[test]
    fn a_rectangle_covers_the_cells_under_it() {
        let bounds = [rect_shape(1, 0.0, 0.0, 80.0, 80.0)];
        let index = GeometryIndex::rasterize(&bounds, area());
        assert_eq!(index.occupied_cells(), GRID_CELLS as u32);
        assert_eq!(index.cell(0, 0), Some(0));
        assert_eq!(index.cell(7, 7), Some(0));
    }

    #[test]
    fn the_pointer_inside_a_shape_has_zero_distance() {
        let bounds = [rect_shape(7, 20.0, 20.0, 40.0, 40.0)];
        let index = GeometryIndex::rasterize(&bounds, area());
        let hit = index
            .pick(&bounds, area(), Offset::new(40.0, 40.0))
            .expect("the center is inside");
        assert_eq!(hit.id, BoundId::new(7));
        assert_eq!(hit.distance, 0.0);
        assert!(hit.is_inside());
        assert_eq!(hit.cell_distance, 0);
    }

    #[test]
    fn an_outside_pointer_gets_the_distance_to_the_nearest_edge() {
        let bounds = [rect_shape(1, 20.0, 20.0, 20.0, 20.0)];
        let index = GeometryIndex::rasterize(&bounds, area());
        let hit = index
            .pick(&bounds, area(), Offset::new(60.0, 30.0))
            .expect("the shape is the nearest");
        assert!((hit.distance - 20.0).abs() < 0.001);
        assert!(!hit.is_inside());
    }

    #[test]
    fn a_later_shape_wins_the_shared_cell() {
        let bounds = [
            rect_shape(1, 20.0, 20.0, 40.0, 40.0),
            rect_shape(2, 30.0, 30.0, 40.0, 40.0),
        ];
        let index = GeometryIndex::rasterize(&bounds, area());
        assert_eq!(index.cell(4, 4), Some(1));
        let hit = index
            .pick(&bounds, area(), Offset::new(50.0, 50.0))
            .expect("the overlap has a shape");
        assert_eq!(hit.id, BoundId::new(2));
    }

    #[test]
    fn an_empty_list_has_no_hit() {
        let index = GeometryIndex::rasterize(&[], area());
        assert!(index.is_empty());
        assert!(index.pick(&[], area(), Offset::ZERO).is_none());
    }

    #[test]
    fn the_cell_distance_field_counts_chebyshev_steps() {
        let bounds = [rect_shape(1, 0.0, 0.0, 10.0, 10.0)];
        let index = GeometryIndex::rasterize(&bounds, area());
        assert_eq!(index.cell_distance(0, 0), Some(0));
        assert_eq!(index.cell_distance(1, 1), Some(1));
        assert_eq!(index.cell_distance(7, 7), Some(7));
    }

    #[test]
    fn a_hit_carries_the_measured_cell() {
        let bounds = [rect_shape(9, 60.0, 60.0, 20.0, 20.0)];
        let index = GeometryIndex::rasterize(&bounds, area());
        let hit: Hit = index
            .pick(&bounds, area(), Offset::new(70.0, 70.0))
            .expect("the corner is inside");
        assert_eq!(hit.cell, (7, 7));
        assert_eq!(hit.index, 0);
    }

    #[test]
    fn a_negative_extent_area_has_no_cells() {
        let bounds = [rect_shape(1, 0.0, 0.0, 10.0, 10.0)];
        let flat = Rect::new(Offset::ZERO, Size::new(0.0, 0.0));
        assert!(GeometryIndex::rasterize(&bounds, flat).is_empty());
    }

    #[test]
    fn picking_prefers_the_topmost_of_equal_shapes() {
        let bounds = [
            rect_shape(1, 20.0, 20.0, 40.0, 40.0),
            rect_shape(2, 20.0, 20.0, 40.0, 40.0),
        ];
        let index = GeometryIndex::rasterize(&bounds, area());
        let hit = index
            .pick(&bounds, area(), Offset::new(40.0, 40.0))
            .expect("both shapes cover the point");
        assert_eq!(hit.id, BoundId::new(2));
        assert_eq!(hit.cell_distance, 0);
    }
}
