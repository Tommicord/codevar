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
}
