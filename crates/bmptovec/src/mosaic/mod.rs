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

//! Seam-free mosaic compositing.
//!
//! Instead of tracing every layer independently (painter's algorithm), the
//! mosaic strategy segments the layer masks into a planar boundary graph
//! where every edge is shared by exactly two faces, fits each shared boundary
//! segment once, and emits one shape per face referencing the shared edge
//! geometry. This guarantees seam-free output with no overlapping strokes.

pub mod compose;
pub mod face;
pub mod fit;
pub mod graph;

pub use compose::compose_mosaic;
pub use fit::{PixelSegmentFitter, PolygonSegmentFitter, SegmentFitter, SplineSegmentFitter};
