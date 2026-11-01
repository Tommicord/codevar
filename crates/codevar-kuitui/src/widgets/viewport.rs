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
//! CONDITIONS OF ANY KIND, either express or implied. See the
//! License for the specific language governing permissions and
//! limitations under the License.

//! The scrolling viewport, the region that reads its gestures and the
//! table the gestures are dispatched through.
//!
//! A viewport shows a part of content that is larger than the space it
//! has: it measures the content at the size the content wants along the
//! axes it scrolls, places it shifted by the scroll offset, and lets the
//! clip of its own modifier chain cut what falls outside the box. What
//! makes it a *reactive* viewport rather than a static scroller is that
//! the offset is state — a wheel, a drag or a fling writes it, and the
//! tree that reads it moves.
//!
//! # The state outlives the render
//!
//! [`ScrollState`] is a handle, like [`InteractionSource`](crate::effect::InteractionSource):
//! cloning it yields another view of the same offset, so the value a
//! gesture wrote survives the recomposition that rebuilt the widget. The
//! content closure therefore creates it once — with
//! [`use_state`](codevar_kuitui_rea::RenderCtx::use_state) — and every
//! later render passes the same handle to the viewport.
//!
//! The offset lives in a [`MutableState`], which is what lets a scroll
//! reach the layout pass: the frame loop asks the state whether a gesture
//! wrote it and lays the tree out again only when the answer is yes.
//!
//! [`MutableState`]: crate::state::MutableState

use alloc::rc::Rc;
use core::cell::RefCell;

use codevar_kuitui_control::Timestamp;

use crate::event::{Event, EventResult, PointerButton, PointerEvent, PointerEventKind, ScrollEvent};
use crate::geometry::{Constraints, Offset, Rect, Size};
use crate::state::MutableState;
use crate::widget::{EventScope, MeasureScope, PlaceScope, Widget};

/// The axes a viewport scrolls along.
///
/// A bit set rather than two flags, so a viewport that scrolls both ways
/// is one value and an axis a widget cannot scroll is a bit that is not
/// set — the same shape [`PointerButtons`] uses.
///
/// [`PointerButtons`]: crate::event::PointerButtons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ScrollAxes(u8);

impl ScrollAxes {
    /// Neither axis scrolls.
    pub const NONE: ScrollAxes = ScrollAxes(0);
    /// The vertical axis scrolls.
    pub const VERTICAL: ScrollAxes = ScrollAxes(1 << 0);
    /// The horizontal axis scrolls.
    pub const HORIZONTAL: ScrollAxes = ScrollAxes(1 << 1);
    /// Both axes scroll, the way a two dimensional canvas does.
    pub const BOTH: ScrollAxes = ScrollAxes(Self::VERTICAL.0 | Self::HORIZONTAL.0);

    /// Whether every axis of `other` scrolls.
    #[must_use]
    pub const fn contains(self, other: ScrollAxes) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no axis at all scrolls.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for ScrollAxes {
    type Output = ScrollAxes;

    fn bitor(self, rhs: ScrollAxes) -> ScrollAxes {
        ScrollAxes(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for ScrollAxes {
    fn bitor_assign(&mut self, rhs: ScrollAxes) {
        self.0 |= rhs.0;
    }
}

/// How far a drag has to travel before it stops being a press.
///
/// The slop is what keeps a click from turning into a scroll the moment
/// the hand shakes, and it is in logical pixels so it scales with the
/// window the way the platform's own slop does.
pub const TOUCH_SLOP: f32 = 4.0;

/// The window of the samples a fling is measured over, in microseconds.
///
/// Only the last moves decide the speed of a release: a pointer that
/// rested for a second before it lifted was not flicked. Android's
/// velocity tracker uses the same hundred millisecond window.
pub const VELOCITY_WINDOW: u64 = 100_000;

/// The most recent samples the tracker keeps.
const MAX_SAMPLES: usize = 20;

/// Pixels per second under which a fling is over.
///
/// Below this speed the remaining distance is under a pixel per frame,
/// so the animation would keep ticking to move nothing; Compose stops a
/// fling at the same threshold.
pub const FLING_STOP_VELOCITY: f32 = 50.0;

/// The per-second friction a fling decays through.
///
/// The model is viscous drag — the exponential decay of Compose's
/// `FloatExponentialDecaySpec`, whose default friction this is: the
/// velocity falls as `e^(-k t)` and the distance it still travels is
/// `v / k`, so a fast release carries the content further without the
/// deceleration ever becoming a hard stop.
pub const FLING_FRICTION: f32 = 1.3;

/// One position of the pointer, with the time it was reported at.
#[derive(Debug, Clone, Copy)]
struct Sample {
    /// When the backend observed the move.
    time: Timestamp,
    /// Where the pointer was.
    position: Offset,
}

/// The speeds a drag was moving at when it was released.
///
/// The tracker keeps the last [`MAX_SAMPLES`] moves and fits a straight
/// line through position over time across the trailing
/// [`VELOCITY_WINDOW`]: the slope of that fit is the velocity. A fit is
/// used instead of the last interval because a single move is one frame
/// of jitter, and a fling thrown by a jerk should not be thrown away by
/// the jerk's own last sample.
///
/// An axis whose events carry no time contributes nothing: the tracker
/// cannot invent a speed for a backend that does not report one.
#[derive(Debug, Clone)]
pub struct VelocityTracker {
    /// The samples, in a ring: `next` is where the newest goes.
    samples: [Sample; MAX_SAMPLES],
    /// Where the next sample is written.
    next: usize,
    /// How many of the ring slots hold a sample.
    filled: usize,
}

impl Default for VelocityTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl VelocityTracker {
    /// A tracker that has seen no move yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            samples: [Sample {
                time: Timestamp::UNSPECIFIED,
                position: Offset::ZERO,
            }; MAX_SAMPLES],
            next: 0,
            filled: 0,
        }
    }

    /// Records the pointer at `position`, reported at `time`.
    ///
    /// A move whose time is unspecified is dropped rather than recorded:
    /// keeping it would put an unmeasurable sample in the middle of the
    /// window and skew the fit around it.
    pub fn add(&mut self, position: Offset, time: Timestamp) {
        if !time.is_specified() {
            return;
        }
        self.samples[self.next] = Sample { time, position };
        self.next = (self.next + 1) % MAX_SAMPLES;
        if self.filled < MAX_SAMPLES {
            self.filled += 1;
        }
    }

    /// The velocity of the trailing window, in pixels per second.
    ///
    /// [`Offset::ZERO`] when fewer than two timed samples are left in the
    /// window, which is what a release after a single move is.
    #[must_use]
    pub fn velocity(&self) -> Offset {
        let Some(base) = self.window_base() else {
            return Offset::ZERO;
        };
        // The least squares fit over the window. The times are relative to
        // the oldest sample kept, which keeps their squares small enough
        // that the denominator does not lose the difference it divides by;
        // one microsecond is 1e-6 of the seconds the slope is asked for,
        // so the scaling happens once, at the end.
        let mut count = 0u64;
        let mut sum_t = 0u64;
        let mut sum_tt = 0u64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_tx = 0.0f64;
        let mut sum_ty = 0.0f64;
        for index in self.indices() {
            let sample = self.samples[index];
            let micros = sample.time.micros();
            if micros < base {
                continue;
            }
            let t = micros - base;
            count += 1;
            sum_t += t;
            sum_tt += t * t;
            sum_x += f64::from(sample.position.x);
            sum_y += f64::from(sample.position.y);
            sum_tx += t as f64 * f64::from(sample.position.x);
            sum_ty += t as f64 * f64::from(sample.position.y);
        }
        if count < 2 {
            return Offset::ZERO;
        }
        let n = count as f64;
        let sum_tf = sum_t as f64;
        let denominator = n * sum_tt as f64 - sum_tf * sum_tf;
        // Two samples at the same instant, or a fit that is a vertical
        // line, has no slope to report.
        if denominator.abs() < f64::EPSILON {
            return Offset::ZERO;
        }
        let x = (n * sum_tx - sum_tf * sum_x) / denominator;
        let y = (n * sum_ty - sum_tf * sum_y) / denominator;
        // The fit is in pixels per microsecond; a fling is asked for in
        // pixels per second.
        Offset::new((x * 1_000_000.0) as f32, (y * 1_000_000.0) as f32)
    }

    /// Forgets every sample, as a press and a release do.
    pub fn reset(&mut self) {
        self.next = 0;
        self.filled = 0;
    }

    /// The time of the oldest sample inside the trailing window.
    ///
    /// Samples older than [`VELOCITY_WINDOW`] before the newest are cut,
    /// which is also what drops the samples a resting pointer left behind.
    /// `None` when the ring holds less than two samples, since a single
    /// move has no slope either.
    fn window_base(&self) -> Option<u64> {
        if self.filled < 2 {
            return None;
        }
        let newest = self.samples[(self.next + MAX_SAMPLES - 1) % MAX_SAMPLES];
        let floor = newest
            .time
            .micros()
            .saturating_sub(VELOCITY_WINDOW);
        self.indices()
            .map(|index| self.samples[index].time.micros())
            .find(|micros| *micros >= floor)
    }

    /// The ring indices from the oldest sample kept to the newest.
    fn indices(&self) -> impl Iterator<Item = usize> + use<> {
        let start = (self.next + MAX_SAMPLES - self.filled) % MAX_SAMPLES;
        (0..self.filled).map(move |step| (start + step) % MAX_SAMPLES)
    }
}

/// What a viewport knows about the content it shows.
///
/// The sizes are not observable state: a layout pass writes them every
/// time it runs, and a scroll is the only thing an application should be
/// notified about. The velocity is here for the same reason — a fling is
/// a transient of the interaction, not a value the tree renders.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    /// The box the content is seen through.
    viewport: Size,
    /// The size of the content itself.
    content: Size,
    /// The velocity the fling currently running is decaying through.
    velocity: Offset,
    /// Whether a fling is running.
    flinging: bool,
}

impl Geometry {
    /// The largest offset the content can be scrolled to on each axis.
    fn max(&self) -> Offset {
        Offset::new(
            (self.content.width - self.viewport.width).max(0.0),
            (self.content.height - self.viewport.height).max(0.0),
        )
    }
}

/// The offset of a viewport, shared by handle.
///
/// A scroll writes the offset and marks the state dirty, which is how the
/// frame loop learns it has to lay the tree out again: the children of a
/// viewport are placed relative to the offset, so a scroll that never
/// reached the place pass would leave the content where it was.
///
/// The offset is never negative and never past the content: a viewport at
/// the start of a list reports zero, and one scrolled to the end reports
/// the difference between the content and the window. A gesture that asks
/// for more than that is a scroll to a bound, and the bound holds.
#[derive(Debug, Clone)]
pub struct ScrollState {
    /// The axes the viewport scrolls along.
    axes: ScrollAxes,
    /// The offset, as observable state.
    offset: MutableState<Offset>,
    /// The sizes and the fling, written by the layout and the gestures.
    geometry: Rc<RefCell<Geometry>>,
}

impl ScrollState {
    /// A state at the start of the content, scrolling along `axes`.
    #[must_use]
    pub fn new(axes: ScrollAxes) -> Self {
        Self {
            axes,
            offset: MutableState::new(Offset::ZERO),
            geometry: Rc::new(RefCell::new(Geometry {
                viewport: Size::ZERO,
                content: Size::ZERO,
                velocity: Offset::ZERO,
                flinging: false,
            })),
        }
    }

    /// The state with the content already scrolled to `offset`.
    ///
    /// The offset is pulled back into the content by the first layout,
    /// which is when the state learns how far the content reaches: a
    /// restored session that was scrolled past a shorter document starts
    /// at the end of it rather than in blank space.
    #[must_use]
    pub fn with_offset(offset: Offset, axes: ScrollAxes) -> Self {
        let state = Self::new(axes);
        state.offset.set(offset);
        state
    }

    /// The axes this state scrolls along.
    #[inline]
    #[must_use]
    pub const fn axes(&self) -> ScrollAxes {
        self.axes
    }

    /// The current offset, in the coordinate space of the content.
    #[inline]
    #[must_use]
    pub fn offset(&self) -> Offset {
        self.offset.get()
    }

    /// How far the content has scrolled along `axis`.
    ///
    /// An axis the state does not scroll reports zero, which is what makes
    /// a handler for one axis unable to move the other by accident.
    #[must_use]
    pub fn value(&self, axis: ScrollAxes) -> f32 {
        let offset = self.offset.get();
        if self.axes.contains(axis) {
            if axis.contains(ScrollAxes::VERTICAL) {
                offset.y
            } else {
                offset.x
            }
        } else {
            0.0
        }
    }

    /// The box the content is seen through, as the last layout reported it.
    #[inline]
    #[must_use]
    pub fn viewport(&self) -> Size {
        self.geometry.borrow().viewport
    }

    /// The size of the content, as the last layout reported it.
    #[inline]
    #[must_use]
    pub fn content(&self) -> Size {
        self.geometry.borrow().content
    }

    /// The largest offset each axis can reach.
    #[inline]
    #[must_use]
    pub fn max(&self) -> Offset {
        self.geometry.borrow().max()
    }

    /// Records the sizes the layout pass produced.
    ///
    /// Called by the viewport while it places its content. The offset is
    /// pulled back into the new bounds when the content shrank under it —
    /// a list that lost its last rows cannot stay scrolled past them —
    /// and a clamp that moved nothing does not mark the state dirty, so a
    /// layout that changed only the window does not ask for another frame.
    pub fn set_layout(&self, viewport: Size, content: Size) {
        let max = {
            let mut geometry = self.geometry.borrow_mut();
            geometry.viewport = viewport;
            geometry.content = content;
            geometry.max()
        };
        let current = self.offset.get();
        let clamped = Offset::new(current.x.max(0.0).min(max.x), current.y.max(0.0).min(max.y));
        if clamped != current {
            self.offset.set(clamped);
        }
    }

    /// Scrolls by `delta` and returns the delta nothing consumed.
    ///
    /// This is the contract a nested scroll needs: the viewport takes the
    /// part of the delta its own axes and bounds can use and hands back
    /// what is left, so a list inside a pager can pass the overflow to
    /// the pager. An axis that does not scroll consumes nothing, and a
    /// bound stops the consumption on that axis only.
    pub fn scroll_by(&self, delta: Offset) -> Offset {
        let max = self.max();
        let current = self.offset.get();
        let mut consumed = Offset::ZERO;
        let mut next = current;
        if self.axes.contains(ScrollAxes::HORIZONTAL) {
            let wanted = (current.x + delta.x).max(0.0).min(max.x);
            consumed.x = wanted - current.x;
            next.x = wanted;
        }
        if self.axes.contains(ScrollAxes::VERTICAL) {
            let wanted = (current.y + delta.y).max(0.0).min(max.y);
            consumed.y = wanted - current.y;
            next.y = wanted;
        }
        if consumed != Offset::ZERO {
            self.offset.set(next);
        }
        Offset::new(delta.x - consumed.x, delta.y - consumed.y)
    }

    /// Scrolls to `offset` and returns the offset it actually reached.
    ///
    /// A bound is a request, not an error: asking for the eleventh row of
    /// a list with three rows scrolls to the last one.
    pub fn scroll_to(&self, offset: Offset) -> Offset {
        let max = self.max();
        let wanted = Offset::new(offset.x.max(0.0).min(max.x), offset.y.max(0.0).min(max.y));
        let before = self.offset.get();
        if wanted != before {
            self.offset.set(wanted);
        }
        wanted
    }

    /// Whether the content is scrolled to the leading edge of `axis`.
    ///
    /// An axis the state does not scroll reports both edges, the same way
    /// [`Self::is_at_end`] does: there is nothing on it to reach.
    #[must_use]
    pub fn is_at_start(&self, axis: ScrollAxes) -> bool {
        !self.axes.contains(axis) || self.value(axis) <= 0.0
    }

    /// Whether the content is scrolled to the trailing edge of `axis`.
    ///
    /// A viewport whose content fits the window is at both edges: there is
    /// nothing between them, and a scroll that starts here belongs to
    /// whatever is outside the viewport.
    #[must_use]
    pub fn is_at_end(&self, axis: ScrollAxes) -> bool {
        if !self.axes.contains(axis) {
            return true;
        }
        let max = self.max();
        if axis.contains(ScrollAxes::VERTICAL) {
            self.value(axis) >= max.y || max.y <= 0.0
        } else {
            self.value(axis) >= max.x || max.x <= 0.0
        }
    }

    /// Starts a fling at `velocity` pixels per second.
    ///
    /// A velocity of zero — the one an untimed release reports — leaves
    /// the content where it is, which is the honest answer: nothing can be
    /// inferred about a speed the backend never measured.
    pub fn start_fling(&self, velocity: Offset) {
        let mut geometry = self.geometry.borrow_mut();
        geometry.velocity = velocity;
        geometry.flinging = velocity.x.abs() > FLING_STOP_VELOCITY || velocity.y.abs() > FLING_STOP_VELOCITY;
    }

    /// Whether a fling still carries the content.
    #[inline]
    #[must_use]
    pub fn is_flinging(&self) -> bool {
        self.geometry.borrow().flinging
    }

    /// Advances a running fling by `elapsed` and reports whether it moved.
    ///
    /// The decay is the viscous one [`FLING_FRICTION`] names: over the
    /// interval the velocity falls by `e^(-k t)` and the distance covered
    /// is the exact integral of that fall, `(v/k) · (1 - e^(-k t))`, so
    /// the frame is correct at any interval — a frame loop that dropped
    /// four frames loses four frames of distance rather than pretending
    /// they happened.
    ///
    /// An axis stops carrying the fling as soon as it has no speed left or
    /// the content is against one of its bounds — the leftover delta is
    /// the proof of that — and the fling is over when no axis is left.
    pub fn advance(&self, elapsed: Timestamp) -> bool {
        let seconds = elapsed.micros() as f32 / 1_000_000.0;
        if seconds <= 0.0 || !self.is_flinging() {
            return false;
        }
        let decay = libm::expf(-FLING_FRICTION * seconds);
        let velocity = self.geometry.borrow().velocity;
        // v0 · (1 - e^(-k t)) / k is the integral of v0 · e^(-k t).
        let distance = Offset::new(
            velocity.x * (1.0 - decay) / FLING_FRICTION,
            velocity.y * (1.0 - decay) / FLING_FRICTION,
        );
        let before = self.offset.get();
        self.scroll_by(distance);
        let max = self.max();
        let current = self.offset.get();
        // An axis is blocked when the content ended against the bound the
        // fling was travelling towards; the rounding of a clamped delta is
        // not evidence of that, so the bound is read from the offset.
        let blocked = |distance: f32, position: f32, limit: f32| {
            (distance > 0.0 && position >= limit) || (distance < 0.0 && position <= 0.0)
        };
        let vx = velocity.x * decay;
        let vy = velocity.y * decay;
        let running = |axis: ScrollAxes, speed: f32, distance: f32, position: f32, limit: f32| {
            self.axes.contains(axis)
                && speed.abs() > FLING_STOP_VELOCITY
                && !blocked(distance, position, limit)
        };
        let flinging = running(ScrollAxes::HORIZONTAL, vx, distance.x, current.x, max.x)
            || running(ScrollAxes::VERTICAL, vy, distance.y, current.y, max.y);
        let mut geometry = self.geometry.borrow_mut();
        geometry.velocity = Offset::new(vx, vy);
        geometry.flinging = flinging;
        current != before
    }

    /// Stops a fling, as a new press does.
    pub fn stop(&self) {
        let mut geometry = self.geometry.borrow_mut();
        geometry.velocity = Offset::ZERO;
        geometry.flinging = false;
    }

    /// Whether a scroll wrote the offset since the last call.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.offset.is_dirty()
    }

    /// Reports whether a scroll wrote the offset, and clears the flag.
    ///
    /// The frame loop calls this when it decides whether the tree has to
    /// be laid out again, which is what makes a scroll of a viewport that
    /// did not move cost nothing.
    pub fn take_dirty(&self) -> bool {
        self.offset.take_dirty()
    }
}

/// What the region of a widget recognized in an event.
///
/// The measurements are in the coordinate space of the window, because
/// that is the space the events arrive in. A handler that wants them
/// relative to its own box subtracts the origin the widget was placed at;
/// the region itself only needs the box to decide whether an event belongs
/// to it at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gesture {
    /// A button went down inside the region.
    Press {
        /// Where the press started.
        position: Offset,
    },
    /// The held pointer moved past the slop, so it is dragging the content.
    Drag {
        /// How far this move took the pointer.
        delta: Offset,
    },
    /// The pointer of a drag was released.
    Drop {
        /// The speed the release was measured at, in pixels per second.
        velocity: Offset,
    },
    /// A release fast enough to carry the content after the pointer left it.
    Fling {
        /// The speed the release was measured at, in pixels per second.
        velocity: Offset,
    },
    /// The pointer went down and up inside the region without dragging.
    Click {
        /// Where the release happened.
        position: Offset,
    },
    /// The wheel moved along the vertical axis.
    ScrollY {
        /// The distance the wheel reported.
        delta: f32,
    },
    /// The wheel moved along the horizontal axis.
    ScrollX {
        /// The distance the wheel reported.
        delta: f32,
    },
    /// The pointer entered or left the region with no button held.
    Hover {
        /// Whether the pointer is inside the region now.
        inside: bool,
    },
    /// The press is over and no gesture completed: its stream was cancelled,
    /// or its release fell outside the box the press was taken in. A widget
    /// that set a state on [`Press`](Self::Press) lets go of it here.
    Cancel,
}

/// The handler of one gesture of a [`ViewportDispatch`] table.
pub type GestureHandler = Rc<dyn Fn(&Gesture) -> EventResult>;

/// The gesture handlers of a widget that answers the pointer.
///
/// One slot per gesture, each of them optional: a slot that names no
/// handler leaves the gesture to the built-in behavior of the widget that
/// owns the region — for a scrolling viewport that is the [`ScrollState`]
/// it scrolls. Filling a slot observes
/// or overrides that one gesture and keeps the others, so a slider that
/// wants to report where a drag ended writes [`Self::on_drop`] and still
/// drags the way its widget drags.
///
/// The handlers are shared: cloning the table clones the `Rc` of each
/// handler, which is what lets a recomposition rebuild the widget without
/// rebuilding the closures or losing the state they captured.
#[derive(Clone, Default)]
pub struct ViewportDispatch {
    on_press: Option<GestureHandler>,
    on_drag: Option<GestureHandler>,
    on_drop: Option<GestureHandler>,
    on_fling: Option<GestureHandler>,
    on_click: Option<GestureHandler>,
    on_scroll_y: Option<GestureHandler>,
    on_scroll_x: Option<GestureHandler>,
    on_hover: Option<GestureHandler>,
    on_cancel: Option<GestureHandler>,
}

impl ViewportDispatch {
    /// A table that names no handler, leaving every gesture to the widget.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            on_press: None,
            on_drag: None,
            on_drop: None,
            on_fling: None,
            on_click: None,
            on_scroll_y: None,
            on_scroll_x: None,
            on_hover: None,
            on_cancel: None,
        }
    }

    /// The table that answers a press at `position`.
    #[must_use]
    pub fn on_press<H>(mut self, handler: H) -> Self
    where
        H: Fn(Offset) -> EventResult + 'static,
    {
        self.on_press = Some(Rc::new(move |gesture| match gesture {
            Gesture::Press { position } => handler(*position),
            _ => EventResult::Ignored,
        }));
        self
    }

    /// The table that answers a drag of `delta` pixels.
    #[must_use]
    pub fn on_drag<H>(mut self, handler: H) -> Self
    where
        H: Fn(Offset) -> EventResult + 'static,
    {
        self.on_drag = Some(Rc::new(move |gesture| match gesture {
            Gesture::Drag { delta } => handler(*delta),
            _ => EventResult::Ignored,
        }));
        self
    }

    /// The table that answers the release of a drag, given the speed the
    /// release was measured at.
    #[must_use]
    pub fn on_drop<H>(mut self, handler: H) -> Self
    where
        H: Fn(Offset) -> EventResult + 'static,
    {
        self.on_drop = Some(Rc::new(move |gesture| match gesture {
            Gesture::Drop { velocity } => handler(*velocity),
            _ => EventResult::Ignored,
        }));
        self
    }

    /// The table that answers a fling of `velocity` pixels per second.
    #[must_use]
    pub fn on_fling<H>(mut self, handler: H) -> Self
    where
        H: Fn(Offset) -> EventResult + 'static,
    {
        self.on_fling = Some(Rc::new(move |gesture| match gesture {
            Gesture::Fling { velocity } => handler(*velocity),
            _ => EventResult::Ignored,
        }));
        self
    }

    /// The table that answers a single click at `position`.
    #[must_use]
    pub fn on_click<H>(mut self, handler: H) -> Self
    where
        H: Fn(Offset) -> EventResult + 'static,
    {
        self.on_click = Some(Rc::new(move |gesture| match gesture {
            Gesture::Click { position } => handler(*position),
            _ => EventResult::Ignored,
        }));
        self
    }

    /// The table that answers the wheel moving `delta` along the vertical
    /// axis.
    #[must_use]
    pub fn on_scroll_y<H>(mut self, handler: H) -> Self
    where
        H: Fn(f32) -> EventResult + 'static,
    {
        self.on_scroll_y = Some(Rc::new(move |gesture| match gesture {
            Gesture::ScrollY { delta } => handler(*delta),
            _ => EventResult::Ignored,
        }));
        self
    }

    /// The table that answers the wheel moving `delta` along the horizontal
    /// axis.
    #[must_use]
    pub fn on_scroll_x<H>(mut self, handler: H) -> Self
    where
        H: Fn(f32) -> EventResult + 'static,
    {
        self.on_scroll_x = Some(Rc::new(move |gesture| match gesture {
            Gesture::ScrollX { delta } => handler(*delta),
            _ => EventResult::Ignored,
        }));
        self
    }

    /// The table that answers the pointer entering (`true`) or leaving
    /// (`false`) the region.
    #[must_use]
    pub fn on_hover<H>(mut self, handler: H) -> Self
    where
        H: Fn(bool) -> EventResult + 'static,
    {
        self.on_hover = Some(Rc::new(move |gesture| match gesture {
            Gesture::Hover { inside } => handler(*inside),
            _ => EventResult::Ignored,
        }));
        self
    }

    /// The table that answers a cancelled press.
    #[must_use]
    pub fn on_cancel<H>(mut self, handler: H) -> Self
    where
        H: Fn() -> EventResult + 'static,
    {
        self.on_cancel = Some(Rc::new(move |gesture| match gesture {
            Gesture::Cancel => handler(),
            _ => EventResult::Ignored,
        }));
        self
    }

    /// The handler the table names for `gesture`, if it names one.
    fn handler_for(&self, gesture: &Gesture) -> Option<&GestureHandler> {
        match gesture {
            Gesture::Press { .. } => self.on_press.as_ref(),
            Gesture::Drag { .. } => self.on_drag.as_ref(),
            Gesture::Drop { .. } => self.on_drop.as_ref(),
            Gesture::Fling { .. } => self.on_fling.as_ref(),
            Gesture::Click { .. } => self.on_click.as_ref(),
            Gesture::ScrollY { .. } => self.on_scroll_y.as_ref(),
            Gesture::ScrollX { .. } => self.on_scroll_x.as_ref(),
            Gesture::Hover { .. } => self.on_hover.as_ref(),
            Gesture::Cancel => self.on_cancel.as_ref(),
        }
    }
}

/// What a region made of an event.
enum Outcome {
    /// A gesture the table has to answer.
    Gesture(Gesture),
    /// The release of a drag: the content is put down and, when the release
    /// was fast enough, thrown.
    DropAndFling {
        /// The speed the release was measured at.
        velocity: Offset,
    },
    /// The wheel moved: every axis it moved along is a gesture of its own.
    Wheel {
        /// The distance the wheel reported.
        delta: Offset,
    },
    /// An event of a press the region already owns, with nothing to report
    /// yet — it is consumed so no ancestor starts a gesture of its own.
    Owned,
    /// The end of a press that completed no gesture. The widget is told so
    /// that it can let go of the states the press set, and the event stays
    /// taken: the press was the region's, so an ancestor must not answer a
    /// release it never held.
    TakenGesture(Gesture),
    /// An event the region has nothing to say about.
    None,
}

/// The state a region keeps between the events of one press.
///
/// It is shared through an `Rc` so that a recomposition in the middle of a
/// drag cannot restart it: the handle the new widget copies points at the
/// same state the old one was folding its events into.
#[derive(Debug, Default)]
struct Recognition {
    /// What the pointer is doing.
    phase: Phase,
    /// Whether the last hover report put the pointer inside the box.
    hovered: bool,
    /// The moves of the press in flight, for the speed of its release.
    tracker: VelocityTracker,
}

/// What the pointer of a region is doing.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
enum Phase {
    /// No button is held inside the region.
    #[default]
    Idle,
    /// A button is held and the pointer has not passed the slop, so the
    /// press is still a click candidate.
    Held {
        /// Where the press went down.
        origin: Offset,
        /// Where the pointer was last reported.
        last: Offset,
    },
    /// A button is held and the pointer has passed the slop, so its moves
    /// drag the content.
    Dragging {
        /// Where the pointer was last reported.
        last: Offset,
    },
}

/// The rectangle of the tree a widget answers the pointer in.
///
/// The region is the piece that turns the raw stream of pointer and wheel
/// events into the gestures of a [`ViewportDispatch`] table: it tracks the
/// press, decides when a move is a drag rather than a shake, measures the
/// speed of a release and hands the result to the table. It holds no
/// behavior of its own — the widget that owns it supplies the built-in
/// answer to each gesture, and the table overrides the ones it names.
///
/// Recognition follows the rules the platform gestures use, because a
/// gesture that fires against them reads as a bug:
///
/// * a press that has not travelled [`TOUCH_SLOP`] pixels is still a press,
///   so a click and a drag cannot both start from one down event;
/// * a release ends the drag *or* answers the click candidate — one of
///   them, never both;
/// * a fling needs the speed of the release, which needs the clock, so a
///   backend that reports no [`Timestamp`] drags but never flings;
/// * a cancel drops the press: no click and no drop is reported.
///
/// [`Timestamp`]: codevar_kuitui_control::Timestamp
#[derive(Clone, Default)]
pub struct GestureRegion {
    /// The table the gestures are dispatched through.
    dispatch: ViewportDispatch,
    /// The state of the press in flight.
    recognition: Rc<RefCell<Recognition>>,
}

impl GestureRegion {
    /// The region that dispatches through `dispatch`.
    #[must_use]
    pub fn new(dispatch: ViewportDispatch) -> Self {
        Self {
            dispatch,
            recognition: Rc::new(RefCell::new(Recognition::default())),
        }
    }

    /// The table the region dispatches through.
    #[inline]
    #[must_use]
    pub const fn dispatch(&self) -> &ViewportDispatch {
        &self.dispatch
    }

    /// The press the region is tracking, for a widget that draws it.
    #[inline]
    #[must_use]
    pub fn is_pressed(&self) -> bool {
        !matches!(self.recognition.borrow().phase, Phase::Idle)
    }

    /// Folds `event` into the gestures of the region and answers them.
    ///
    /// `bounds` is the box of the widget in the coordinate space of the
    /// window; an event outside it is not the region's business. `default`
    /// is the built-in answer of the widget, used for every gesture the
    /// table leaves unnamed.
    ///
    /// A press the region owns consumes its moves even when no handler
    /// answered them, which is what keeps an ancestor of the widget from
    /// starting a second gesture out of the same drag.
    pub fn on_event<F>(&self, event: &Event, bounds: Rect, mut default: F) -> EventResult
    where
        F: FnMut(&Gesture) -> EventResult,
    {
        let outcome = match event {
            Event::Pointer(pointer) => self.recognize_pointer(pointer, bounds),
            Event::Scroll(wheel) => self.recognize_scroll(wheel, bounds),
            _ => Outcome::None,
        };
        match outcome {
            Outcome::None => EventResult::Ignored,
            Outcome::Owned => EventResult::Consumed,
            Outcome::TakenGesture(gesture) => {
                // The answer of the widget is heard, but the event is the
                // region's to take either way: a release that completed
                // nothing still ends the press it held.
                combine(EventResult::Consumed, self.answer(&gesture, &mut default))
            }
            Outcome::Gesture(gesture) => self.answer(&gesture, &mut default),
            Outcome::DropAndFling { velocity } => {
                // A release both puts the content down and — if it was
                // moving — throws it, so the table can answer the drop and
                // the fling separately.
                let dropped = self.answer(&Gesture::Drop { velocity }, &mut default);
                let thrown = velocity.x.abs() > FLING_STOP_VELOCITY || velocity.y.abs() > FLING_STOP_VELOCITY;
                if !thrown {
                    return dropped;
                }
                combine(dropped, self.answer(&Gesture::Fling { velocity }, &mut default))
            }
            Outcome::Wheel { delta } => {
                let mut result = EventResult::Ignored;
                if delta.y != 0.0 {
                    result = self.answer(&Gesture::ScrollY { delta: delta.y }, &mut default);
                }
                if delta.x != 0.0 {
                    result = combine(
                        result,
                        self.answer(&Gesture::ScrollX { delta: delta.x }, &mut default),
                    );
                }
                result
            }
        }
    }

    /// The table's answer to `gesture`, or the widget's built-in one when
    /// the table names no handler for it.
    fn answer<F>(&self, gesture: &Gesture, default: &mut F) -> EventResult
    where
        F: FnMut(&Gesture) -> EventResult,
    {
        match self.dispatch.handler_for(gesture) {
            Some(handler) => handler(gesture),
            None => default(gesture),
        }
    }

    /// The gesture `event` completed, with the press state folded in.
    fn recognize_pointer(&self, event: &PointerEvent, bounds: Rect) -> Outcome {
        let inside = bounds.contains(event.position);
        let mut recognition = self.recognition.borrow_mut();
        match event.kind {
            PointerEventKind::Down => {
                if !inside || event.button != PointerButton::Primary {
                    return Outcome::None;
                }
                recognition.phase = Phase::Held {
                    origin: event.position,
                    last: event.position,
                };
                recognition.hovered = true;
                recognition.tracker.reset();
                recognition
                    .tracker
                    .add(event.position, event.time);
                Outcome::Gesture(Gesture::Press {
                    position: event.position,
                })
            }
            PointerEventKind::Move => match recognition.phase {
                Phase::Idle => {
                    if inside == recognition.hovered {
                        return Outcome::None;
                    }
                    recognition.hovered = inside;
                    Outcome::Gesture(Gesture::Hover { inside })
                }
                Phase::Held { origin, last } => {
                    recognition
                        .tracker
                        .add(event.position, event.time);
                    let delta = Offset::new(event.position.x - last.x, event.position.y - last.y);
                    if past_slop(event.position, origin) {
                        recognition.phase = Phase::Dragging { last: event.position };
                        Outcome::Gesture(Gesture::Drag { delta })
                    } else {
                        recognition.phase = Phase::Held {
                            origin,
                            last: event.position,
                        };
                        // The press is the region's, but it has not become
                        // anything yet.
                        Outcome::Owned
                    }
                }
                Phase::Dragging { last } => {
                    recognition
                        .tracker
                        .add(event.position, event.time);
                    recognition.phase = Phase::Dragging { last: event.position };
                    Outcome::Gesture(Gesture::Drag {
                        delta: Offset::new(event.position.x - last.x, event.position.y - last.y),
                    })
                }
            },
            PointerEventKind::Up => {
                let velocity = recognition.tracker.velocity();
                recognition.tracker.reset();
                match recognition.phase {
                    Phase::Dragging { .. } => {
                        recognition.phase = Phase::Idle;
                        recognition.hovered = inside;
                        Outcome::DropAndFling { velocity }
                    }
                    Phase::Held { origin, .. } => {
                        recognition.phase = Phase::Idle;
                        recognition.hovered = inside;
                        if past_slop(event.position, origin) || !inside {
                            // The press left the box without being a drag:
                            // neither a click nor a drag ended, so nothing
                            // is reported beyond the press being over.
                            return Outcome::TakenGesture(Gesture::Cancel);
                        }
                        Outcome::Gesture(Gesture::Click {
                            position: event.position,
                        })
                    }
                    Phase::Idle => {
                        if inside == recognition.hovered {
                            return Outcome::None;
                        }
                        recognition.hovered = inside;
                        Outcome::Gesture(Gesture::Hover { inside })
                    }
                }
            }
            PointerEventKind::Enter | PointerEventKind::Leave => {
                if recognition.phase != Phase::Idle {
                    return Outcome::Owned;
                }
                if inside == recognition.hovered {
                    return Outcome::None;
                }
                recognition.hovered = inside;
                Outcome::Gesture(Gesture::Hover { inside })
            }
            PointerEventKind::Cancel => {
                if matches!(recognition.phase, Phase::Idle) {
                    return Outcome::None;
                }
                recognition.phase = Phase::Idle;
                recognition.tracker.reset();
                Outcome::Gesture(Gesture::Cancel)
            }
        }
    }

    /// The wheel gestures of `event`, in the order the axes are reported.
    fn recognize_scroll(&self, event: &ScrollEvent, bounds: Rect) -> Outcome {
        if !bounds.contains(event.position) {
            return Outcome::None;
        }
        if event.delta == Offset::ZERO {
            return Outcome::None;
        }
        Outcome::Wheel { delta: event.delta }
    }
}

/// Whether either of two answers took the event.
fn combine(first: EventResult, second: EventResult) -> EventResult {
    if first.is_consumed() || second.is_consumed() {
        EventResult::Consumed
    } else {
        EventResult::Ignored
    }
}

/// Whether the pointer travelled far enough from `origin` to be dragging.
///
/// The slop is measured along each axis rather than as the distance, the
/// way a drag decides its direction: a shake that moves 2 pixels in both
/// axes is still a shake, and a move of 5 pixels along one is a drag.
fn past_slop(position: Offset, origin: Offset) -> bool {
    (position.x - origin.x).abs() > TOUCH_SLOP || (position.y - origin.y).abs() > TOUCH_SLOP
}

/// The widget that shows a part of a content larger than its box.
///
/// A viewport measures its content with the size the content asks for
/// along the axes it scrolls — an unbounded height for a vertical
/// viewport — and places it shifted by the offset its gestures write. It
/// holds one content, the way `verticalScroll` of Compose does: a list is
/// a [`Column`](crate::widgets::Column) inside the viewport, not several
/// children of it.
///
/// The viewport does not clip. Clipping belongs to the chain of the node
/// it sits in — `Viewport(modifier: Modifier::new().clip(), …)` — because
/// the framework only cuts the painting a modifier chain asks to be cut,
/// and a scroller that paints outside its own box is still a scroller.
///
/// # The gestures
///
/// Every pointer and wheel event that falls inside the box is folded into
/// gestures by the [`GestureRegion`] of the viewport and answered by its
/// [`ViewportDispatch`] table. For the gestures the table leaves empty the
/// viewport has a built-in answer:
///
/// | Gesture | What the viewport does |
/// | --- | --- |
/// | [`Drag`] | moves the offset *against* the pointer, so content follows the finger |
/// | [`Fling`] | starts [`ScrollState::start_fling`] with that same negated speed |
/// | [`ScrollY`], [`ScrollX`] | moves the offset *with* the wheel |
/// | the rest | nothing; the event keeps travelling |
///
/// A gesture that moved nothing is not consumed, so a viewport that is
/// already at its end lets the wheel reach its parent: nested scrolling
/// falls out of the leftover the state reports.
///
/// A fling needs a frame to run in. The loop calls [`ScrollState::advance`]
/// with the time since the frame before, and the content keeps moving until
/// the speed dies or an edge stops it.
///
/// [`Drag`]: Gesture::Drag
/// [`Fling`]: Gesture::Fling
/// [`ScrollY`]: Gesture::ScrollY
/// [`ScrollX`]: Gesture::ScrollX
#[derive(Clone)]
pub struct Viewport {
    /// The offset the gestures write and the layout reads.
    state: ScrollState,
    /// The region that turns the events of the box into gestures.
    region: GestureRegion,
    /// The box the viewport was last placed in, in window coordinates.
    ///
    /// A widget is handed no bounds when an event reaches it, so the
    /// viewport records the rectangle it was placed in: the region needs
    /// it to decide which events belong to it.
    bounds: Rect,
}

impl Viewport {
    /// A viewport that scrolls its content vertically.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: ScrollState::new(ScrollAxes::VERTICAL),
            region: GestureRegion::new(ViewportDispatch::default()),
            bounds: Rect::ZERO,
        }
    }

    /// The viewport that scrolls with `state`.
    ///
    /// The state is a handle, so the application keeps the offset across
    /// the recompositions that rebuild the widget: a content closure
    /// creates it once with
    /// [`use_state`](codevar_kuitui_rea::RenderCtx::use_state) and hands
    /// the same handle to the viewport every render.
    #[must_use]
    pub fn scroll(mut self, state: ScrollState) -> Self {
        self.state = state;
        self
    }

    /// The viewport that answers the gestures its table names.
    ///
    /// A slot of the table overrides the built-in answer of that one
    /// gesture and keeps the others, so a viewport that has to report
    /// where a drag ended writes [`on_drag`](ViewportDispatch::on_drag)
    /// and still scrolls.
    #[must_use]
    pub fn dispatch(mut self, table: ViewportDispatch) -> Self {
        self.region = GestureRegion::new(table);
        self
    }

    /// The state the viewport scrolls.
    #[inline]
    #[must_use]
    pub const fn state(&self) -> &ScrollState {
        &self.state
    }

    /// Whether a press is held inside the viewport.
    ///
    /// A widget that paints the pressed state of its own background reads
    /// this instead of keeping an [`InteractionSource`](crate::effect::InteractionSource)
    /// of its own, because the region already folds the press for it.
    #[must_use]
    pub fn is_pressed(&self) -> bool {
        self.region.is_pressed()
    }
}

impl Default for Viewport {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Viewport {
    /// Reports the size of the box, having measured the content at the
    /// size it asks for.
    ///
    /// The axes the viewport scrolls are relaxed for the content — both
    /// the minimum and the maximum of them — so content smaller than the
    /// box stays small and content larger than it is measured whole; the
    /// other axes keep the constraints of the parent, so a horizontal
    /// viewport is as tall as it was asked to be.
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let axes = self.state.axes();
        let bounded = |scrolls: bool, min: f32, max: f32| {
            if scrolls { (0.0, f32::INFINITY) } else { (min, max) }
        };
        let (min_width, max_width) = bounded(
            axes.contains(ScrollAxes::HORIZONTAL),
            constraints.min_width,
            constraints.max_width,
        );
        let (min_height, max_height) = bounded(
            axes.contains(ScrollAxes::VERTICAL),
            constraints.min_height,
            constraints.max_height,
        );
        let content = Constraints::new(min_width, max_width, min_height, max_height);
        let mut size = Size::ZERO;
        for index in 0..scope.child_count() {
            let child = scope.measure_child(index, content);
            size = Size::new(size.width.max(child.width), size.height.max(child.height));
        }
        let viewport = constraints.enforce(size);
        self.state.set_layout(viewport, size);
        viewport
    }

    /// Places the content shifted by the offset, and records the box the
    /// gestures are recognized inside.
    fn place(&mut self, scope: &mut PlaceScope<'_>, bounds: Rect) {
        self.bounds = bounds;
        let offset = self.state.offset();
        let shift = Offset::new(-offset.x, -offset.y);
        for index in 0..scope.child_count() {
            scope.place_child(index, shift);
        }
    }

    fn on_event(&mut self, scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        let _ = scope;
        let state = self.state.clone();
        self.region
            .on_event(event, self.bounds, move |gesture| match gesture {
                Gesture::Drag { delta } => scrolled(&state, Offset::new(-delta.x, -delta.y)),
                Gesture::Fling { velocity } => {
                    if *velocity == Offset::ZERO {
                        return EventResult::Ignored;
                    }
                    state.start_fling(Offset::new(-velocity.x, -velocity.y));
                    EventResult::Consumed
                }
                Gesture::ScrollY { delta } => scrolled(&state, Offset::new(0.0, *delta)),
                Gesture::ScrollX { delta } => scrolled(&state, Offset::new(*delta, 0.0)),
                _ => EventResult::Ignored,
            })
    }

    /// The viewport reads the pointer: an event inside its box is its
    /// gesture to make, even when none of its children takes it.
    fn hit_testable(&self) -> bool {
        true
    }

    /// Runs the fling the region started, so a release keeps carrying the
    /// content after the pointer left it.
    ///
    /// Reports whether the offset moved, which is what tells the frame
    /// loop that the tree has to be laid out again — and, once the decay
    /// has settled, that it no longer does.
    fn advance(&mut self, elapsed: Timestamp) -> bool {
        self.state.advance(elapsed)
    }
}

/// Whether an offset move of `asked` reached the state at all.
///
/// The state hands back what it could not consume, so a move that came
/// back whole is a move the viewport has no use for — and the event keeps
/// travelling to a parent that may.
fn scrolled(state: &ScrollState, asked: Offset) -> EventResult {
    if state.scroll_by(asked) == asked {
        EventResult::Ignored
    } else {
        EventResult::Consumed
    }
}

#[cfg(test)]
mod tests {
    use crate::event::PointerButtons;

    use super::*;

    /// A state over a window of `viewport` showing content of `content`.
    fn scrolled(axes: ScrollAxes, viewport: Size, content: Size) -> ScrollState {
        let state = ScrollState::new(axes);
        state.set_layout(viewport, content);
        state
    }

    #[test]
    fn the_axes_combine_with_the_bit_or_operator() {
        let axes = ScrollAxes::VERTICAL | ScrollAxes::HORIZONTAL;
        assert!(axes.contains(ScrollAxes::VERTICAL));
        assert!(axes.contains(ScrollAxes::BOTH));
        assert!(!ScrollAxes::VERTICAL.contains(ScrollAxes::HORIZONTAL));
        assert!(ScrollAxes::NONE.is_empty());
    }

    #[test]
    fn the_offset_is_bound_by_the_content() {
        let state = scrolled(
            ScrollAxes::VERTICAL,
            Size::new(100.0, 40.0),
            Size::new(100.0, 200.0),
        );
        assert_eq!(state.max().y, 160.0);
        assert_eq!(state.scroll_by(Offset::new(0.0, 20.0)), Offset::ZERO);
        assert_eq!(state.offset().y, 20.0);
        // Past the end: the content stops at the bound and the rest of the
        // delta is handed back for whoever is outside the viewport — the
        // 20 already scrolled plus the 400 asked for, minus the 160 there
        // is room for.
        assert_eq!(state.scroll_by(Offset::new(0.0, 400.0)).y, 260.0);
        assert_eq!(state.offset().y, 160.0);
        assert!(state.is_at_end(ScrollAxes::VERTICAL));
        state.scroll_to(Offset::ZERO);
        assert!(state.is_at_start(ScrollAxes::VERTICAL));
    }

    #[test]
    fn an_axis_that_does_not_scroll_consumes_nothing() {
        let state = scrolled(
            ScrollAxes::VERTICAL,
            Size::new(100.0, 40.0),
            Size::new(100.0, 200.0),
        );
        let left = state.scroll_by(Offset::new(15.0, 15.0));
        assert_eq!(left, Offset::new(15.0, 0.0));
        assert_eq!(state.offset(), Offset::new(0.0, 15.0));
        assert_eq!(state.value(ScrollAxes::HORIZONTAL), 0.0);
    }

    #[test]
    fn a_shrunken_content_pulls_the_offset_back() {
        let state = scrolled(
            ScrollAxes::VERTICAL,
            Size::new(100.0, 40.0),
            Size::new(100.0, 200.0),
        );
        state.scroll_by(Offset::new(0.0, 120.0));
        state.take_dirty();
        state.set_layout(Size::new(100.0, 40.0), Size::new(100.0, 100.0));
        assert_eq!(state.offset().y, 60.0);
        assert!(state.is_dirty());
    }

    #[test]
    fn a_layout_that_moved_nothing_leaves_the_state_clean() {
        let state = scrolled(
            ScrollAxes::VERTICAL,
            Size::new(100.0, 40.0),
            Size::new(100.0, 200.0),
        );
        state.take_dirty();
        state.set_layout(Size::new(100.0, 80.0), Size::new(100.0, 400.0));
        assert!(!state.is_dirty());
    }

    #[test]
    fn clones_share_the_offset() {
        let state = scrolled(ScrollAxes::BOTH, Size::new(50.0, 50.0), Size::new(400.0, 400.0));
        let other = state.clone();
        other.scroll_by(Offset::new(4.0, 6.0));
        assert_eq!(state.offset(), Offset::new(4.0, 6.0));
    }

    #[test]
    fn a_fling_decays_and_stops_at_its_bound() {
        let state = scrolled(
            ScrollAxes::VERTICAL,
            Size::new(100.0, 40.0),
            Size::new(100.0, 1_040.0),
        );
        state.take_dirty();
        state.start_fling(Offset::new(0.0, 1_000.0));
        assert!(state.is_flinging());
        // One second of decay travels the integral of the fall, not the
        // velocity times the interval, and leaves the speed multiplied by
        // e^(-k).
        let frame = Timestamp::from_micros(1_000_000);
        assert!(state.advance(frame));
        let expected = 1_000.0 * (1.0 - libm::expf(-FLING_FRICTION)) / FLING_FRICTION;
        assert!((state.offset().y - expected).abs() < 0.5);
        assert!(state.is_flinging());
        // A long interval runs the fling out: the speed falls under the
        // threshold and nothing moves any further.
        assert!(state.advance(Timestamp::from_micros(60_000_000)));
        assert!(!state.is_flinging());
        let settled = state.offset().y;
        assert!(!state.advance(Timestamp::from_micros(1_000_000)));
        assert_eq!(state.offset().y, settled);
    }

    #[test]
    fn an_untimed_release_never_flings() {
        let state = scrolled(
            ScrollAxes::VERTICAL,
            Size::new(100.0, 40.0),
            Size::new(100.0, 1_040.0),
        );
        state.start_fling(Offset::ZERO);
        assert!(!state.is_flinging());
        assert!(!state.advance(Timestamp::from_micros(16_000)));
    }

    #[test]
    fn the_velocity_is_the_slope_of_the_trailing_window() {
        let mut tracker = VelocityTracker::new();
        // A pointer moving 10 pixels per millisecond, sampled every
        // millisecond: the fit has to report 10_000 pixels per second.
        for step in 0..10u64 {
            tracker.add(
                Offset::new(0.0, step as f32 * 10.0),
                Timestamp::from_micros(step * 1_000),
            );
        }
        let velocity = tracker.velocity();
        assert!((velocity.y - 10_000.0).abs() < 1.0);
        assert!((velocity.x).abs() < 1.0);
    }

    #[test]
    fn a_resting_pointer_is_not_a_fling() {
        let mut tracker = VelocityTracker::new();
        tracker.add(Offset::new(0.0, 0.0), Timestamp::from_micros(10_000));
        // The pointer stopped: the last sample is far outside the window
        // of the ones that measure the release.
        tracker.add(Offset::new(0.0, 4.0), Timestamp::from_micros(900_000));
        tracker.add(Offset::new(0.0, 5.0), Timestamp::from_micros(910_000));
        let velocity = tracker.velocity();
        assert!(velocity.y.abs() < 200.0, "{velocity:?} is a resting pointer");
    }

    /// The box of the widget a test region answers inside.
    const BOX: Rect = Rect::new(Offset::new(10.0, 10.0), Size::new(100.0, 100.0));

    /// The name a test gives a gesture.
    fn name_of(gesture: &Gesture) -> &'static str {
        match gesture {
            Gesture::Press { .. } => "press",
            Gesture::Drag { .. } => "drag",
            Gesture::Drop { .. } => "drop",
            Gesture::Fling { .. } => "fling",
            Gesture::Click { .. } => "click",
            Gesture::ScrollY { .. } => "scroll-y",
            Gesture::ScrollX { .. } => "scroll-x",
            Gesture::Hover { .. } => "hover",
            Gesture::Cancel => "cancel",
        }
    }

    /// A logger of the gestures a region reports, shared by handle.
    #[derive(Clone, Default)]
    struct Log(Rc<RefCell<Vec<&'static str>>>);

    impl Log {
        /// The closure that writes the name of every gesture it is given.
        fn recorder(&self) -> impl Fn(&Gesture) -> EventResult {
            let seen = Rc::clone(&self.0);
            move |gesture| {
                seen.borrow_mut().push(name_of(gesture));
                EventResult::Ignored
            }
        }

        /// The gestures reported so far.
        fn seen(&self) -> Vec<&'static str> {
            self.0.borrow().clone()
        }
    }

    /// A pointer event of a press sequence, with the button held.
    fn pointer(kind: PointerEventKind, position: Offset, micros: u64) -> Event {
        let buttons = if kind == PointerEventKind::Up {
            PointerButtons::NONE
        } else {
            PointerButtons::PRIMARY
        };
        Event::Pointer(
            PointerEvent::new(kind, position)
                .with_buttons(buttons)
                .with_time(Timestamp::from_micros(micros)),
        )
    }

    /// The wheel moving `delta` under `position`.
    fn wheel(delta: Offset, position: Offset, micros: u64) -> Event {
        Event::Scroll(ScrollEvent::new(delta, position).with_time(Timestamp::from_micros(micros)))
    }

    #[test]
    fn a_press_that_never_left_the_slop_is_a_click_and_nothing_else() {
        let log = Log::default();
        let region = GestureRegion::new(ViewportDispatch::new());
        region.on_event(
            &pointer(PointerEventKind::Down, Offset::new(20.0, 20.0), 0),
            BOX,
            log.recorder(),
        );
        region.on_event(
            &pointer(PointerEventKind::Move, Offset::new(22.0, 21.0), 1_000),
            BOX,
            log.recorder(),
        );
        region.on_event(
            &pointer(PointerEventKind::Up, Offset::new(22.0, 21.0), 2_000),
            BOX,
            log.recorder(),
        );
        assert_eq!(log.seen(), ["press", "click"]);
    }

    #[test]
    fn a_move_past_the_slop_turns_the_press_into_a_drag() {
        let log = Log::default();
        let region = GestureRegion::new(ViewportDispatch::new());
        region.on_event(
            &pointer(PointerEventKind::Down, Offset::new(20.0, 20.0), 0),
            BOX,
            log.recorder(),
        );
        region.on_event(
            &pointer(PointerEventKind::Move, Offset::new(20.0, 40.0), 10_000),
            BOX,
            log.recorder(),
        );
        // The pointer then creeps: one pixel every 40 milliseconds is 25
        // pixels per second, under the speed a fling is asked for. The
        // trailing window of the tracker is measured from the newest
        // sample, so the fast move that started the drag has left it.
        region.on_event(
            &pointer(PointerEventKind::Move, Offset::new(20.0, 41.0), 250_000),
            BOX,
            log.recorder(),
        );
        region.on_event(
            &pointer(PointerEventKind::Move, Offset::new(20.0, 42.0), 290_000),
            BOX,
            log.recorder(),
        );
        region.on_event(
            &pointer(PointerEventKind::Up, Offset::new(20.0, 42.0), 300_000),
            BOX,
            log.recorder(),
        );
        assert_eq!(log.seen(), ["press", "drag", "drag", "drag", "drop"]);
    }

    #[test]
    fn a_fast_release_is_a_drop_and_a_fling() {
        let log = Log::default();
        let region = GestureRegion::new(ViewportDispatch::new());
        region.on_event(
            &pointer(PointerEventKind::Down, Offset::new(20.0, 20.0), 0),
            BOX,
            log.recorder(),
        );
        // A fling is the slope of a *sequence* of moves: one sample carries
        // no speed to measure. Three moves 40 pixels apart every 4
        // milliseconds is the 10_000 pixels per second a fling is asked for.
        for step in 1..=3u32 {
            region.on_event(
                &pointer(
                    PointerEventKind::Move,
                    Offset::new(20.0, 20.0 + step as f32 * 40.0),
                    u64::from(step) * 4_000,
                ),
                BOX,
                log.recorder(),
            );
        }
        region.on_event(
            &pointer(PointerEventKind::Up, Offset::new(20.0, 140.0), 15_000),
            BOX,
            log.recorder(),
        );
        assert_eq!(log.seen(), ["press", "drag", "drag", "drag", "drop", "fling"]);
    }

    #[test]
    fn a_drag_reports_the_move_it_was_given() {
        let travelled = Rc::new(RefCell::new(Offset::ZERO));
        let table = {
            let travelled = Rc::clone(&travelled);
            ViewportDispatch::new().on_drag(move |delta| {
                let mut sum = travelled.borrow_mut();
                *sum = Offset::new(sum.x + delta.x, sum.y + delta.y);
                EventResult::Ignored
            })
        };
        let region = GestureRegion::new(table);
        let log = Log::default();
        region.on_event(
            &pointer(PointerEventKind::Down, Offset::new(20.0, 20.0), 0),
            BOX,
            log.recorder(),
        );
        for step in 1..=10u32 {
            region.on_event(
                &pointer(
                    PointerEventKind::Move,
                    Offset::new(20.0, 20.0 + step as f32 * 20.0),
                    u64::from(step) * 10_000,
                ),
                BOX,
                log.recorder(),
            );
        }
        // The slop of the first move is not a scroll: only the moves that
        // followed the drag are counted.
        assert_eq!(travelled.borrow().y, 20.0 * 9.0 + (20.0 - 0.0));
    }

    #[test]
    fn a_press_that_ended_outside_the_box_is_neither_a_click_nor_a_drop() {
        let log = Log::default();
        let region = GestureRegion::new(ViewportDispatch::new());
        region.on_event(
            &pointer(PointerEventKind::Down, Offset::new(20.0, 20.0), 0),
            BOX,
            log.recorder(),
        );
        let result = region.on_event(
            &pointer(PointerEventKind::Up, Offset::new(400.0, 400.0), 1_000),
            BOX,
            log.recorder(),
        );
        assert_eq!(
            log.seen(),
            ["press", "cancel"],
            "the press ends, and the widget is told that it did"
        );
        // The press still belonged to the region, so the event is consumed.
        assert_eq!(result, EventResult::Consumed);
    }

    #[test]
    fn a_cancel_reports_the_cancel_and_no_release() {
        let log = Log::default();
        let region = GestureRegion::new(ViewportDispatch::new());
        region.on_event(
            &pointer(PointerEventKind::Down, Offset::new(20.0, 20.0), 0),
            BOX,
            log.recorder(),
        );
        region.on_event(
            &pointer(PointerEventKind::Cancel, Offset::new(20.0, 20.0), 1_000),
            BOX,
            log.recorder(),
        );
        assert_eq!(log.seen(), ["press", "cancel"]);
        assert!(!region.is_pressed());
    }

    #[test]
    fn the_wheel_moves_the_axis_it_moved_along() {
        let log = Log::default();
        let region = GestureRegion::new(ViewportDispatch::new());
        region.on_event(
            &wheel(Offset::new(0.0, 12.0), Offset::new(50.0, 50.0), 1),
            BOX,
            log.recorder(),
        );
        region.on_event(
            &wheel(Offset::new(5.0, 0.0), Offset::new(50.0, 50.0), 2),
            BOX,
            log.recorder(),
        );
        region.on_event(
            &wheel(Offset::new(5.0, 7.0), Offset::new(50.0, 50.0), 3),
            BOX,
            log.recorder(),
        );
        assert_eq!(log.seen(), ["scroll-y", "scroll-x", "scroll-y", "scroll-x"]);
    }

    #[test]
    fn a_wheel_outside_the_box_is_not_the_regions_business() {
        let log = Log::default();
        let region = GestureRegion::new(ViewportDispatch::new());
        let result = region.on_event(
            &wheel(Offset::new(0.0, 12.0), Offset::new(500.0, 500.0), 1),
            BOX,
            log.recorder(),
        );
        assert_eq!(result, EventResult::Ignored);
        assert!(log.seen().is_empty());
    }

    #[test]
    fn the_hover_of_a_pointer_is_reported_only_when_it_changes() {
        let log = Log::default();
        let region = GestureRegion::new(ViewportDispatch::new());
        for position in [Offset::new(50.0, 50.0), Offset::new(51.0, 51.0)] {
            region.on_event(&pointer(PointerEventKind::Move, position, 1), BOX, log.recorder());
        }
        region.on_event(
            &pointer(PointerEventKind::Move, Offset::new(500.0, 500.0), 2),
            BOX,
            log.recorder(),
        );
        assert_eq!(log.seen(), ["hover", "hover"]);
    }

    #[test]
    fn a_press_consumes_its_moves_before_it_becomes_a_drag() {
        let region = GestureRegion::new(ViewportDispatch::new());
        let log = Log::default();
        region.on_event(
            &pointer(PointerEventKind::Down, Offset::new(20.0, 20.0), 0),
            BOX,
            log.recorder(),
        );
        let result = region.on_event(
            &pointer(PointerEventKind::Move, Offset::new(21.0, 20.0), 1_000),
            BOX,
            log.recorder(),
        );
        assert_eq!(result, EventResult::Consumed);
        assert_eq!(log.seen(), ["press"]);
    }

    #[test]
    fn the_table_answers_the_gesture_it_names_and_the_default_the_rest() {
        let tabled = Log::default();
        let fallback = Log::default();
        let table = {
            let seen = Rc::clone(&tabled.0);
            ViewportDispatch::new().on_click(move |_| {
                seen.borrow_mut().push("click");
                EventResult::Consumed
            })
        };
        let region = GestureRegion::new(table);
        let answer = region.on_event(
            &pointer(PointerEventKind::Down, Offset::new(20.0, 20.0), 0),
            BOX,
            fallback.recorder(),
        );
        assert_eq!(answer, EventResult::Ignored);
        let answer = region.on_event(
            &pointer(PointerEventKind::Up, Offset::new(20.0, 20.0), 1_000),
            BOX,
            fallback.recorder(),
        );
        assert_eq!(answer, EventResult::Consumed);
        assert_eq!(tabled.seen(), ["click"]);
        assert_eq!(fallback.seen(), ["press"]);
    }

    #[test]
    fn an_event_of_another_kind_is_not_recognized() {
        let log = Log::default();
        let region = GestureRegion::new(ViewportDispatch::new());
        let result = region.on_event(&Event::Text('x'), BOX, log.recorder());
        assert_eq!(result, EventResult::Ignored);
        assert!(log.seen().is_empty());
    }

    #[test]
    fn a_press_of_another_button_is_not_a_gesture() {
        let log = Log::default();
        let region = GestureRegion::new(ViewportDispatch::new());
        let mut event = PointerEvent::new(PointerEventKind::Down, Offset::new(20.0, 20.0));
        event.button = PointerButton::Secondary;
        let result = region.on_event(&Event::Pointer(event), BOX, log.recorder());
        assert_eq!(result, EventResult::Ignored);
        assert!(log.seen().is_empty());
    }

    #[test]
    fn a_move_without_a_time_is_not_a_sample() {
        let mut tracker = VelocityTracker::new();
        tracker.add(Offset::new(0.0, 10.0), Timestamp::UNSPECIFIED);
        tracker.add(Offset::new(0.0, 20.0), Timestamp::UNSPECIFIED);
        assert_eq!(tracker.velocity(), Offset::ZERO);
    }

    #[test]
    fn two_samples_at_the_same_time_have_no_slope() {
        let mut tracker = VelocityTracker::new();
        tracker.add(Offset::new(0.0, 1.0), Timestamp::from_micros(500));
        tracker.add(Offset::new(0.0, 9.0), Timestamp::from_micros(500));
        assert_eq!(tracker.velocity(), Offset::ZERO);
    }

    #[test]
    fn the_ring_keeps_the_newest_samples() {
        let mut tracker = VelocityTracker::new();
        for step in 0..(MAX_SAMPLES as u64 + 8) {
            tracker.add(
                Offset::new(step as f32, 0.0),
                Timestamp::from_micros(step * 1_000),
            );
        }
        assert_eq!(tracker.filled, MAX_SAMPLES);
        // The eight oldest moves are gone from the ring, so the window
        // starts at the sample that displaced them.
        let oldest = tracker
            .indices()
            .next()
            .map(|index| tracker.samples[index].position.x);
        assert_eq!(oldest, Some(8.0));
        // One pixel per millisecond is the slope of every sample kept.
        let velocity = tracker.velocity();
        assert!((velocity.x - 1_000.0).abs() < 1.0, "{velocity:?}");
    }

    use crate::compose::Composition;
    use crate::node::Node;
    use crate::theme::Theme;
    use crate::widgets::Row;

    /// A page of content of a fixed size, to scroll inside a viewport.
    struct Page {
        size: Size,
    }

    impl Widget for Page {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(self.size)
        }
    }

    /// A page that takes the release of a press inside it, the way the
    /// check box of a scrolling list does.
    struct Taker {
        size: Size,
    }

    impl Widget for Taker {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(self.size)
        }

        fn on_event(&mut self, _scope: &mut EventScope<'_>, event: &Event) -> EventResult {
            if matches!(
                event,
                Event::Pointer(pointer) if pointer.kind == PointerEventKind::Up
            ) {
                return EventResult::Consumed;
            }
            EventResult::Ignored
        }

        fn hit_testable(&self) -> bool {
            true
        }
    }

    /// A laid-out composition of a viewport box showing a page of `content`.
    fn scrolling(state: ScrollState, box_size: Size, content: Size) -> Composition<'static> {
        let mut composition = Composition::new(Theme::light());
        composition.set_content(move |_| {
            Node::new(Viewport::new().scroll(state.clone())).child(Node::new(Page { size: content }))
        });
        composition.recompose(true);
        composition.layout(box_size, true);
        composition
    }

    /// The window a test viewport is laid out in.
    const WINDOW: Size = Size::new(100.0, 40.0);

    /// Throws a page of `composition` upward: `steps` moves of 8 pixels
    /// every 4 milliseconds, then the release that ends the drag.
    ///
    /// The sequence starts inside the 40-pixel box and stays inside it, so
    /// every event of the flick belongs to the viewport.
    fn flick(composition: &mut Composition<'_>, steps: u32) {
        composition.handle_event(&pointer(PointerEventKind::Down, Offset::new(50.0, 35.0), 1_000));
        for step in 1..=steps {
            composition.handle_event(&pointer(
                PointerEventKind::Move,
                Offset::new(50.0, 35.0 - step as f32 * 8.0),
                u64::from(step) * 4_000,
            ));
        }
        let end = 35.0 - steps as f32 * 8.0;
        composition.handle_event(&pointer(
            PointerEventKind::Up,
            Offset::new(50.0, end),
            u64::from(steps) * 4_000 + 1_000,
        ));
    }

    #[test]
    fn a_viewport_reports_its_box_and_measures_its_content_whole() {
        let state = ScrollState::new(ScrollAxes::VERTICAL);
        let composition = scrolling(state.clone(), WINDOW, Size::new(300.0, 400.0));
        assert_eq!(composition.root().size(), WINDOW);
        // The page is cut to the width the viewport was given, but is
        // measured at the height it asks for: that is what there is to
        // scroll through.
        assert_eq!(state.content(), Size::new(100.0, 400.0));
        assert_eq!(state.viewport(), WINDOW);
        assert_eq!(state.max(), Offset::new(0.0, 360.0));
    }

    #[test]
    fn a_wheel_inside_the_box_scrolls_the_content() {
        let state = ScrollState::new(ScrollAxes::VERTICAL);
        let mut composition = scrolling(state.clone(), WINDOW, Size::new(300.0, 400.0));
        let result = composition.handle_event(&wheel(Offset::new(0.0, 20.0), Offset::new(50.0, 20.0), 1_000));
        assert!(result.is_consumed());
        assert_eq!(state.offset().y, 20.0);
        composition.layout(WINDOW, true);
        // The page sits above the top of the box by the offset: the box
        // shows the part of it the offset reached.
        assert_eq!(
            composition.root().children()[0].bounds().origin,
            Offset::new(0.0, -20.0)
        );
    }

    #[test]
    fn a_wheel_outside_the_box_is_not_the_viewports_business() {
        let state = ScrollState::new(ScrollAxes::VERTICAL);
        let mut composition = scrolling(state.clone(), WINDOW, Size::new(300.0, 400.0));
        let result =
            composition.handle_event(&wheel(Offset::new(0.0, 20.0), Offset::new(500.0, 500.0), 1_000));
        assert!(!result.is_consumed());
        assert_eq!(state.offset(), Offset::ZERO);
    }

    #[test]
    fn a_drag_moves_the_content_with_the_finger() {
        let state = ScrollState::new(ScrollAxes::VERTICAL);
        let mut composition = scrolling(state.clone(), WINDOW, Size::new(300.0, 400.0));
        composition.handle_event(&pointer(PointerEventKind::Down, Offset::new(50.0, 30.0), 1_000));
        composition.handle_event(&pointer(PointerEventKind::Move, Offset::new(50.0, 20.0), 2_000));
        composition.handle_event(&pointer(PointerEventKind::Up, Offset::new(50.0, 20.0), 3_000));
        // The finger went up, so the content followed it and the offset
        // grew: the opposite of the pointer is the direction of a scroll.
        assert_eq!(state.offset().y, 10.0);
    }

    #[test]
    fn a_release_a_child_takes_ends_the_press_of_the_viewport() {
        let state = ScrollState::new(ScrollAxes::VERTICAL);
        let built = state.clone();
        let mut composition = Composition::new(Theme::light());
        composition.set_content(move |_| {
            Node::new(Viewport::new().scroll(built.clone())).child(Node::new(Taker {
                size: Size::new(100.0, 400.0),
            }))
        });
        composition.recompose(true);
        composition.layout(WINDOW, true);
        composition.handle_event(&pointer(PointerEventKind::Down, Offset::new(50.0, 30.0), 1_000));
        composition.handle_event(&pointer(PointerEventKind::Up, Offset::new(50.0, 30.0), 2_000));
        // The child consumed the release, so the region of the viewport was
        // never offered it; the cancel the dispatch gives back to the
        // ancestor is what ends the press it still held. Without it the
        // move below would drag the content instead of hovering it.
        composition.handle_event(&pointer(PointerEventKind::Move, Offset::new(50.0, 10.0), 3_000));
        assert_eq!(state.offset(), Offset::ZERO);
        // The region still scrolls: the press was forgotten, not the
        // viewport's hold on its own gestures.
        composition.handle_event(&pointer(PointerEventKind::Down, Offset::new(50.0, 30.0), 4_000));
        composition.handle_event(&pointer(PointerEventKind::Move, Offset::new(50.0, 10.0), 5_000));
        composition.handle_event(&pointer(PointerEventKind::Up, Offset::new(50.0, 10.0), 6_000));
        assert_eq!(state.offset().y, 20.0);
    }

    #[test]
    fn a_drag_past_the_end_of_the_content_is_left_to_the_parent() {
        let state = ScrollState::new(ScrollAxes::VERTICAL);
        let mut composition = scrolling(state.clone(), WINDOW, Size::new(300.0, 400.0));
        state.scroll_by(Offset::new(0.0, 360.0));
        composition.handle_event(&pointer(PointerEventKind::Down, Offset::new(50.0, 30.0), 1_000));
        let result =
            composition.handle_event(&pointer(PointerEventKind::Move, Offset::new(50.0, 10.0), 10_000));
        // Already at the end: the whole ask came back, so the viewport did
        // not take the event and whoever is outside it still can.
        assert!(!result.is_consumed());
        assert_eq!(state.offset().y, 360.0);
    }

    #[test]
    fn a_fling_throws_the_content_the_way_the_drag_was_moving() {
        let state = ScrollState::new(ScrollAxes::VERTICAL);
        let mut composition = scrolling(state.clone(), WINDOW, Size::new(300.0, 400.0));
        flick(&mut composition, 3);
        assert!(state.is_flinging());
        let after_drag = state.offset().y;
        assert!(state.advance(Timestamp::from_micros(8_000)));
        // Thrown the way the finger went: the content keeps rising after
        // the release.
        assert!(state.offset().y > after_drag, "{:?}", state.offset());
    }

    #[test]
    fn a_fling_runs_when_the_frame_advances() {
        let state = ScrollState::new(ScrollAxes::VERTICAL);
        let mut composition = scrolling(state.clone(), WINDOW, Size::new(300.0, 400.0));
        flick(&mut composition, 3);
        let after_drag = state.offset().y;
        // The hook, not an event, is what moves the content now: the
        // pointer is gone and the release still carries.
        assert!(
            composition
                .root_mut()
                .advance(Timestamp::from_micros(8_000))
        );
        assert!(state.offset().y > after_drag, "{:?}", state.offset());
        // Run the decay out, and the frame loop is told to stop laying the
        // tree out for a fling that has settled.
        composition
            .root_mut()
            .advance(Timestamp::from_micros(60_000_000));
        assert!(!state.is_flinging());
        assert!(
            !composition
                .root_mut()
                .advance(Timestamp::from_micros(8_000))
        );
    }

    #[test]
    fn a_horizontal_viewport_leaves_the_vertical_wheel_alone() {
        let state = ScrollState::new(ScrollAxes::HORIZONTAL);
        let mut composition = scrolling(state.clone(), WINDOW, Size::new(400.0, 300.0));
        let result = composition.handle_event(&wheel(Offset::new(0.0, 20.0), Offset::new(50.0, 20.0), 1_000));
        assert!(!result.is_consumed());
        assert_eq!(state.offset(), Offset::ZERO);
        let result = composition.handle_event(&wheel(Offset::new(20.0, 0.0), Offset::new(50.0, 20.0), 2_000));
        assert!(result.is_consumed());
        assert_eq!(state.offset().x, 20.0);
    }

    #[test]
    fn a_wheel_reaches_the_viewport_nested_in_a_container() {
        let state = ScrollState::new(ScrollAxes::VERTICAL);
        let built = state.clone();
        let mut composition = Composition::new(Theme::light());
        composition.set_content(move |_| {
            Node::new(Row::new()).child(Node::new(Viewport::new().scroll(built.clone())).child(Node::new(
                Page {
                    size: Size::new(300.0, 400.0),
                },
            )))
        });
        composition.recompose(true);
        composition.layout(WINDOW, true);
        // The wheel is not offered to the whole tree the way a key is: it
        // falls to the deepest node under the pointer, which here is the
        // viewport inside the row.
        let result = composition.handle_event(&wheel(Offset::new(0.0, 20.0), Offset::new(50.0, 20.0), 1_000));
        assert!(result.is_consumed());
        assert_eq!(state.offset().y, 20.0);
    }

    #[test]
    fn a_viewport_answers_the_gesture_its_table_names() {
        let clicks = Rc::new(core::cell::Cell::new(0_u32));
        let observed = Rc::clone(&clicks);
        let table = ViewportDispatch::new().on_click(move |_position| {
            observed.set(observed.get() + 1);
            EventResult::Consumed
        });
        let state = ScrollState::new(ScrollAxes::VERTICAL);
        let mut composition = Composition::new(Theme::light());
        composition.set_content(move |_| {
            Node::new(
                Viewport::new()
                    .scroll(state.clone())
                    .dispatch(table.clone()),
            )
            .child(Node::new(Page {
                size: Size::new(300.0, 400.0),
            }))
        });
        composition.recompose(true);
        composition.layout(WINDOW, true);
        composition.handle_event(&pointer(PointerEventKind::Down, Offset::new(50.0, 20.0), 1_000));
        let result = composition.handle_event(&pointer(PointerEventKind::Up, Offset::new(50.0, 20.0), 2_000));
        assert!(result.is_consumed());
        assert_eq!(clicks.get(), 1);
    }
}
