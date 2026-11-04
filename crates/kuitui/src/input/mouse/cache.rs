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

//! The per-widget cache that keeps a bitmap across frames.
//!
//! The cache hashes the shape list and the raster area with FNV-1a. An
//! unchanged list returns the previous bitmap without touching a single
//! polygon; a moved vertex, a new shape or a resized area changes the hash
//! and triggers one rasterization.
//!
//! [`GeometryCache::covers`] is the form the widget tree hit test uses: it
//! answers *is the pointer on one of these shapes* with the cached bitmap
//! doing the culling and the polygons doing the deciding.

use crate::geometry::{Offset, Rect};

use super::bounds::VertexBounds;
use super::index::{GeometryIndex, Hit};

/// The cached index bitmap of one shape list.
///
/// The cache remembers a hash of the shapes and the area, as well as the
/// bitmap itself. Reusing it across frames turns picking into one hash pass
/// plus a few dozen array reads.
#[derive(Debug, Clone)]
pub struct GeometryCache {
    /// Hash of the shape list and area the bitmap was built from.
    signature: u64,
    /// Bitmap built from that list.
    index: GeometryIndex,
}

impl GeometryCache {
    /// A cache that has never been filled.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            signature: 0,
            index: GeometryIndex::empty(),
        }
    }

    /// The bitmap for the last list passed to [`Self::update`].
    #[must_use]
    pub const fn index(&self) -> &GeometryIndex {
        &self.index
    }

    /// The bitmap for `bounds` over `area`, rasterizing only when the list or
    /// the area changed since the last call.
    pub fn update(&mut self, bounds: &[VertexBounds], area: Rect) -> &GeometryIndex {
        let signature = signature_of(bounds, area);
        if signature != self.signature {
            self.index = GeometryIndex::rasterize(bounds, area);
            self.signature = signature;
        }
        &self.index
    }

    /// The hit under `point`, updating the bitmap first when needed.
    pub fn hit(&mut self, bounds: &[VertexBounds], area: Rect, point: Offset) -> Option<Hit> {
        self.update(bounds, area)
            .pick(bounds, area, point)
    }

    /// Whether `point` lies inside one of `bounds`.
    ///
    /// The yes-or-no form of [`Self::hit`] the widget tree needs: the
    /// bitmap culls the work, the polygons decide. A list of one shape that
    /// is its own box is answered by the rectangle test directly, which is the
    /// case the node hit test reaches for a widget that rounds or turns a
    /// single region and asks for nothing beyond it. A shape list whose bitmap
    /// has no occupied cell is measured directly, because a shape smaller
    /// than one cell center is exactly the case the 8x8 grid cannot answer,
    /// and a hit at zero distance is containment — the condition
    /// [`Hit::is_inside`] documents.
    #[must_use]
    pub fn covers(&mut self, bounds: &[VertexBounds], area: Rect, point: Offset) -> bool {
        if let [shape] = bounds
            && shape.is_plain()
        {
            return shape.contains(point);
        }
        let index = self.update(bounds, area);
        if index.is_empty() {
            return bounds.iter().any(|shape| shape.contains(point));
        }
        index
            .pick(bounds, area, point)
            .is_some_and(|hit| hit.is_inside())
    }
}

impl Default for GeometryCache {
    fn default() -> Self {
        Self::new()
    }
}

/// FNV-1a offset basis for the shape-list hash.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a prime for the shape-list hash.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hashes a shape list and the area it is rasterized over.
fn signature_of(bounds: &[VertexBounds], area: Rect) -> u64 {
    let mut hash = FNV_OFFSET;
    for value in [area.origin.x, area.origin.y, area.size.width, area.size.height] {
        hash = mix(hash, u64::from(value.to_bits()));
    }
    hash = mix(hash, bounds.len() as u64);
    for shape in bounds {
        hash = mix(hash, u64::from(shape.id().get()));
        for vertex in shape.vertices() {
            hash = mix(hash, u64::from(vertex.x.to_bits()));
            hash = mix(hash, u64::from(vertex.y.to_bits()));
        }
    }
    hash
}

/// Folds one value into a FNV-1a hash.
#[inline]
fn mix(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(FNV_PRIME)
}

#[cfg(test)]
mod tests {
    use super::GeometryCache;
    use crate::geometry::{Offset, Rect};
    use crate::input::mouse::bounds::{BoundId, VertexBounds};
    use crate::paint::CornerRadii;

    fn area() -> Rect {
        Rect::from_min_size(Offset::ZERO, 80.0, 80.0)
    }

    fn rect_shape(id: u32, x: f32, y: f32, width: f32, height: f32) -> VertexBounds {
        VertexBounds::rect(
            BoundId::new(id),
            Rect::from_min_size(Offset::new(x, y), width, height),
        )
    }

    fn rounded(id: u32, rect: Rect, radius: f32) -> VertexBounds {
        VertexBounds::rounded_rect(BoundId::new(id), rect, CornerRadii::uniform(radius))
            .expect("the rectangle has an area")
    }

    #[test]
    fn the_cache_reuses_an_unchanged_list_and_invalidates_a_change() {
        let mut cache = GeometryCache::new();
        let first = [rect_shape(1, 0.0, 0.0, 20.0, 20.0)];
        let initial = cache.update(&first, area()).clone();
        let again = cache.update(&first, area()).clone();
        assert_eq!(initial, again);
        assert_eq!(cache.index().occupied_cells(), initial.occupied_cells());

        let moved = [rect_shape(1, 40.0, 40.0, 20.0, 20.0)];
        let changed = cache.update(&moved, area()).clone();
        assert_eq!(changed.cell(0, 0), None);
        assert_eq!(changed.cell(5, 5), Some(0));
    }

    #[test]
    fn covering_stops_being_exact_on_the_cut_corner_of_a_shape() {
        let mut cache = GeometryCache::new();
        let shape = [rounded(4, area(), 30.0)];
        assert!(cache.covers(&shape, area(), Offset::new(40.0, 40.0)));
        assert!(cache.covers(&shape, area(), Offset::new(40.0, 1.0)));
        assert!(
            !cache.covers(&shape, area(), Offset::new(1.0, 1.0)),
            "the corner of the box is outside the arc the widget paints"
        );
        assert!(!cache.covers(&shape, area(), Offset::new(79.0, 1.0)));
        assert!(cache.covers(&shape, area(), Offset::new(40.0, 79.0)));
    }

    #[test]
    fn a_shape_too_small_for_the_grid_is_still_covered() {
        // Each cell of the 8x8 grid over an 80x80 area is ten pixels wide and
        // its centers sit at 5, 15, 25 and so on, so this shape covers no
        // center at all: the bitmap is empty and says nothing, and the
        // polygon is measured directly instead.
        let mut cache = GeometryCache::new();
        let sliver = [rounded(
            5,
            Rect::from_min_size(Offset::new(12.0, 12.0), 2.0, 2.0),
            0.5,
        )];
        assert!(
            cache.update(&sliver, area()).is_empty(),
            "no cell center of the grid falls inside the sliver"
        );
        assert!(cache.covers(&sliver, area(), Offset::new(13.0, 13.0)));
        assert!(!cache.covers(&sliver, area(), Offset::new(45.0, 45.0)));
    }

    #[test]
    fn an_empty_list_covers_nothing() {
        let mut cache = GeometryCache::new();
        assert!(!cache.covers(&[], area(), Offset::new(40.0, 40.0)));
    }

    #[test]
    fn a_single_plain_shape_is_covered_without_a_bitmap_being_built() {
        let mut cache = GeometryCache::new();
        let shape = [rect_shape(7, 10.0, 10.0, 20.0, 20.0)];
        assert!(cache.covers(&shape, area(), Offset::new(15.0, 15.0)));
        assert!(cache.covers(&shape, area(), Offset::new(30.0, 30.0)));
        assert!(!cache.covers(&shape, area(), Offset::new(35.0, 35.0)));
        assert_eq!(
            cache.index().occupied_cells(),
            0,
            "the rectangle test answered every one of them, so nothing was rasterized"
        );
    }

    #[test]
    fn the_plain_shortcut_answers_what_the_bitmap_would() {
        let shape = [rect_shape(8, 10.0, 10.0, 20.0, 20.0)];
        let mut cache = GeometryCache::new();
        // The point asked of the shortcut and of the bitmap it would have
        // used; a point on the edge of a shape is where the two paths could
        // disagree by an inclusive-versus-exclusive test, so the sweep holds
        // the corners of the box and one pixel past each of them.
        for point in [
            Offset::new(9.0, 15.0),
            Offset::new(10.0, 10.0),
            Offset::new(30.0, 20.0),
            Offset::new(20.0, 30.0),
            Offset::new(30.5, 30.5),
            Offset::new(0.0, 0.0),
        ] {
            let shortcut = cache.covers(&shape, area(), point);
            let bitmap = cache
                .update(&shape, area())
                .pick(&shape, area(), point)
                .is_some_and(|hit| hit.is_inside());
            assert_eq!(
                shortcut, bitmap,
                "the two paths disagree at ({}, {})",
                point.x, point.y
            );
        }
    }

    #[test]
    fn an_empty_area_still_leaves_the_polygons_to_decide() {
        let mut cache = GeometryCache::new();
        let shape = [rect_shape(6, 0.0, 0.0, 40.0, 40.0)];
        // A raster area with no extent leaves the bitmap empty, and a bitmap
        // that empty cannot answer anything, so `covers` measures the shapes
        // instead of trusting the grid.
        assert!(cache.covers(&shape, Rect::ZERO, Offset::new(10.0, 10.0)));
        assert!(!cache.covers(&shape, Rect::ZERO, Offset::new(60.0, 60.0)));
    }
}
