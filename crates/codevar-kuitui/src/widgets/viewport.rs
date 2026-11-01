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

//! The scroll state of a viewport, the gestures a pointer makes over it,
//! and the table those gestures are dispatched through.
//!
//! A viewport shows a part of content that is larger than the space it
//! has: it measures the content at the size the content wants, places it
//! shifted by the scroll offset, and clips it to its own box. What makes
//! it a *reactive* viewport rather than a static scroller is that the
//! offset is state — a wheel, a drag or a fling writes it, and the tree
//! that reads it moves.
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

use crate::event::EventResult;
use crate::geometry::{Offset, Size};
use crate::state::MutableState;

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
    /// The stream of the press was cancelled, so no gesture completed.
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
}

#[cfg(test)]
mod tests {
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
}
