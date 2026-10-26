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

//! Pointer state, the mouse hook, and hit testing.
//!
//! The layer mirrors the keyboard one: [`PointerState`] is a lock-free record
//! the pump writes and every handler reads through a permanent `'static`
//! reference, and [`MouseHandler`] adds a hook that runs after the state has
//! been updated.
//!
//! # The hit path
//!
//! A widget declares its interactive shapes as [`VertexBounds`] and the
//! rectangle they live in. [`MouseHandler::hit`] rasterizes those shapes into
//! the 8x8 index bitmap of [`GeometryIndex`], narrows the bitmap to the cells
//! the pointer can reach, and measures the exact distance to the remaining
//! shapes. A distance of zero means the pointer is inside the shape. The
//! bitmap lives in a [`GeometryCache`] the handler keeps across frames, so an
//! unchanged shape list costs one hash pass per pick.
//!
//! # Frame semantics
//!
//! Motion and scroll accumulate in the state. Once per frame the pump calls
//! [`PointerState::finish_frame`], which snapshots the accumulated motion and
//! restarts it from zero; the position and the counters survive.
//!
//! [`VertexBounds`]: crate::input::mouse::VertexBounds
//! [`GeometryIndex`]: crate::input::mouse::GeometryIndex
//! [`GeometryCache`]: crate::input::mouse::GeometryCache

mod snapshot;
mod state;

pub use snapshot::PointerSnapshot;
pub use state::{PointerButtons, PointerState};
