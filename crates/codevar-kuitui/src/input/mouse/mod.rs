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

//! Virtual geometry rasterization and pointer picking.
//!
//! A widget describes its interactive shapes as [`VertexBounds`]: a
//! three- or four-vertex convex polygon plus an identifier. Rasterizing
//! the whole list into an 8x8 [index bitmap](GeometryIndex) costs at most
//! 64 containment tests per shape, and the bitmap then answers the common
//! question — *which shape is under the pointer?* — without touching the
//! polygons again.
//!
//! The bitmap stores `index + 1` per cell, so zero means "no shape" and a
//! later shape overwrites an earlier one. The later shape is the one drawn
//! on top, so the bitmap agrees with what the user sees.
//!
//! # From the bitmap to a distance
//!
//! Picking uses the bitmap as a coarse index: the pointer selects a cell,
//! the bitmap narrows the candidates to the cells around that one, and only
//! those shapes are measured exactly. [`Hit::distance`] is therefore the
//! real distance to the polygon, and it is zero exactly when the pointer is
//! inside the shape. [`Hit::cell_distance`] keeps the coarse, bitmap-space
//! distance so a caller can tell "inside" from "one cell away" without
//! measuring anything.
//!
//! # Caching
//!
//! [`GeometryCache`] keeps the bitmap for a shape list. It hashes the list
//! and the raster area, and rasterizes again only when either changes, so a
//! per-frame pick costs one hash pass and 64 lookups.

mod bounds;

pub use bounds::{BoundId, VertexBounds};
