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

//! A mosaic face: one flat-color region plus the boundary segments that
//! enclose it. Each segment is shared with exactly one adjacent face.

use crate::fitter::FittedGeom;
use crate::ir::{Paint, RegionMask};
use crate::types::PointI32;

/// A segment of a face's boundary: a chain of pixel-space points and the
/// vector geometry fitted to it (shared with the adjacent face).
#[derive(Debug, Clone)]
pub struct BoundarySegment {
    /// Pixel-space points forming the segment, in boundary order.
    pub points: Vec<PointI32>,
    /// Fitted vector geometry for this segment (owned by this face; the
    /// adjacent face references a clone of the same fitted geometry).
    pub fitted: FittedGeom,
    /// The node index this segment starts at, if known.
    pub start_node: Option<usize>,
    /// The node index this segment ends at, if known.
    pub end_node: Option<usize>,
}

/// A face: a flat-color region and its enclosing boundary segments.
#[derive(Debug, Clone)]
pub struct Face {
    /// The paint (fill color) of this face.
    pub paint: Paint,
    /// The pixel mask covered by this face (local to this face's bounds).
    pub mask: RegionMask,
    /// The segments that enclose this face, in boundary-walk order.
    pub boundary: Vec<BoundarySegment>,
    /// Whether this face is a hole (inverted winding) of its parent region.
    pub hole: bool,
}

impl Face {
    /// A new face with no boundary segments yet.
    #[inline]
    pub fn new(paint: Paint, mask: RegionMask) -> Self {
        Self {
            paint,
            mask,
            boundary: Vec::new(),
            hole: false,
        }
    }

    /// The number of boundary segments.
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.boundary.len()
    }

    /// Total number of boundary points across all segments.
    #[inline]
    pub fn boundary_point_count(&self) -> usize {
        self.boundary.iter().map(|s| s.points.len()).sum()
    }
}
