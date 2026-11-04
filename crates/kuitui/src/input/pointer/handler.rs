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

//! The pointer hook a widget implements, the permanent state reference it
//! holds, and the hit test built on the rasterized geometry.
//!
//! [`MouseHandler`] adds three things to [`Input`]: the `'static` state
//! reference, the vertex bounds of the shapes the widget reacts to, and the
//! hook. The default [`MouseHandler::hit`] runs the bounds through a
//! [`GeometryCache`], so a widget that returns its geometry gets Virtual
//! Geometry Rasterization and picking without writing any of it.

use alloc::vec::Vec;

use codevar_kuitui_control::{PointerButton, PointerEvent, Timestamp};

use crate::event::EventResult;
use crate::geometry::{Offset, Rect};
use crate::input::mouse::{GeometryCache, Hit, VertexBounds};
use crate::input::{Input, KeyboardModifiers};

use super::snapshot::PointerSnapshot;
use super::state::{PointerButtons, PointerState, to_offset};

/// What a pointer event did, as a handler sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerPhase {
    /// The pointer moved.
    Motion,
    /// A button changed state.
    Button,
    /// The pointer scrolled.
    Scroll,
    /// The pointer entered the application region.
    Enter,
    /// The pointer left the application region.
    Leave,
    /// The platform cancelled the pointer stream.
    Cancel,
    /// A pointer event this version does not know.
    Unknown,
}

impl PointerPhase {
    /// The phase of one control-layer event.
    #[must_use]
    pub fn of(event: &PointerEvent) -> Self {
        match event {
            PointerEvent::Motion { .. } => Self::Motion,
            PointerEvent::Button { .. } => Self::Button,
            PointerEvent::Scroll { .. } => Self::Scroll,
            PointerEvent::Enter { .. } => Self::Enter,
            PointerEvent::Leave => Self::Leave,
            PointerEvent::Cancel => Self::Cancel,
            _ => Self::Unknown,
        }
    }
}

/// One pointer event after the state has been updated.
///
/// The update carries the per-event delta and scroll rather than the frame
/// totals: a handler that needs the frame totals calls
/// [`PointerState::finish_frame`] once per frame and reads the snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerUpdate {
    /// When the backend observed the event.
    pub timestamp: Timestamp,
    /// What the event did.
    pub phase: PointerPhase,
    /// The position after the event, or `None` before the first report.
    pub position: Option<Offset>,
    /// Movement this event reported or that follows from the previous
    /// position.
    pub delta: Offset,
    /// Scroll this event reported.
    pub scroll: Offset,
    /// The button this event was about.
    pub button: Option<PointerButton>,
    /// The buttons held after the event.
    pub buttons: PointerButtons,
    /// The modifiers reported by the event.
    pub modifiers: KeyboardModifiers,
}

impl PointerUpdate {
    /// Applies `event` to `state` and describes the result.
    ///
    /// This is the single entry point the pump calls: the state is updated
    /// first, so a handler reading [`MouseHandler::pointer`] inside
    /// [`MouseHandler::on_pointer`] sees the event it was just handed.
    pub fn new(state: &PointerState, event: &PointerEvent, timestamp: Timestamp) -> Self {
        let previous = state.position();
        let delta = match event {
            PointerEvent::Motion { position, delta } => (*delta).map_or_else(
                || match (position, previous) {
                    (Some(position), Some(previous)) => to_offset(*position).sub(previous),
                    _ => Offset::ZERO,
                },
                to_offset,
            ),
            _ => Offset::ZERO,
        };
        let scroll = match event {
            PointerEvent::Scroll { delta, .. } => to_offset(*delta),
            _ => Offset::ZERO,
        };
        state.apply(event, timestamp);
        Self {
            timestamp,
            phase: PointerPhase::of(event),
            position: state.position(),
            delta,
            scroll,
            button: match event {
                PointerEvent::Button { button, .. } => Some(*button),
                _ => None,
            },
            buttons: state.buttons(),
            modifiers: state.modifiers(),
        }
    }
}

/// A widget that reacts to the pointing device.
///
/// The trait extends [`Input`] and adds the two pieces a pointer consumer
/// needs: a permanent reference to the shared state, and the list of convex
/// shapes the widget treats as interactive. [`MouseHandler::hit`] turns that
/// list into an 8x8 index bitmap and returns the shape under the pointer,
/// with distance zero when the pointer is inside it.
pub trait MouseHandler: Input {
    /// The pointer state this handler observes.
    ///
    /// The reference is `'static`: it points at [`POINTER`] or at another
    /// static of the application, so it outlives the handler and survives
    /// recomposition.
    fn pointer(&self) -> &'static PointerState;

    /// The convex shapes the pointer can hit.
    ///
    /// The default is no shape at all, which makes [`Self::hit`] return
    /// `None`. Shapes are painted in order, so a later shape wins a cell it
    /// shares with an earlier one.
    fn geometry(&self) -> &[VertexBounds] {
        &[]
    }

    /// The region the geometry is rasterized over.
    ///
    /// The default is the empty rectangle, which disables picking. A widget
    /// with geometry returns the rectangle that contains its interactive
    /// shapes.
    fn interactive_area(&self) -> Rect {
        Rect::ZERO
    }

    /// Receives one pointer event after the state has been updated.
    fn on_pointer(&mut self, update: &PointerUpdate) -> EventResult {
        let _ = update;
        EventResult::Ignored
    }

    /// The shape under the current pointer position.
    ///
    /// The cache is per handler: keep one across frames and the bitmap is
    /// only rebuilt when the shape list or the area changes. The returned
    /// [`Hit::distance`] is zero exactly when the pointer is inside the
    /// shape.
    fn hit(&self, cache: &mut GeometryCache) -> Option<Hit> {
        let point = self.pointer().position()?;
        cache.hit(self.geometry(), self.interactive_area(), point)
    }
}

/// A convenience holder for the permanent pointer reference and its geometry.
///
/// A widget that implements [`MouseHandler`] can keep one of these, forward
/// [`MouseHandler::pointer`] and [`MouseHandler::geometry`] to it, and let
/// [`MouseBinding::hit`] own the cache.
pub struct MouseBinding {
    /// The state the binding reads.
    state: &'static PointerState,
    /// The shapes the binding picks over.
    bounds: Vec<VertexBounds>,
    /// The region the shapes are rasterized over.
    area: Rect,
    /// The bitmap of the current shape list.
    cache: GeometryCache,
}

impl MouseBinding {
    /// Binds to `state` and picks over `area`.
    #[must_use]
    pub fn new(state: &'static PointerState, area: Rect) -> Self {
        Self {
            state,
            bounds: Vec::new(),
            area,
            cache: GeometryCache::new(),
        }
    }

    /// Binds to the process-wide [`POINTER`] state.
    #[must_use]
    pub fn global(area: Rect) -> Self {
        Self::new(&POINTER, area)
    }

    /// The bound state.
    #[must_use]
    pub const fn state(&self) -> &'static PointerState {
        self.state
    }

    /// The bound shapes.
    #[must_use]
    pub fn geometry(&self) -> &[VertexBounds] {
        &self.bounds
    }

    /// Replaces the shapes.
    ///
    /// The cache notices a changed vertex or id and rasterizes again, so a
    /// widget may move a shape without recreating the binding.
    pub fn set_geometry(&mut self, bounds: Vec<VertexBounds>) {
        self.bounds = bounds;
    }

    /// The region the shapes are rasterized over.
    #[must_use]
    pub const fn area(&self) -> Rect {
        self.area
    }

    /// Rasterizes over a new region on the next hit.
    pub fn set_area(&mut self, area: Rect) {
        self.area = area;
    }

    /// A consistent snapshot of the bound state.
    #[must_use]
    pub fn snapshot(&self) -> PointerSnapshot {
        self.state.snapshot()
    }

    /// The shape under the current pointer position.
    #[must_use]
    pub fn hit(&mut self) -> Option<Hit> {
        let point = self.state.position()?;
        self.cache.hit(&self.bounds, self.area, point)
    }
}

/// The pointer state of the process.
///
/// The `InputPump` writes it; handlers bind to it with
/// [`MouseBinding::global`] or by returning `&POINTER` from
/// [`MouseHandler::pointer`].
pub static POINTER: PointerState = PointerState::new();
