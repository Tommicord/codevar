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

//! The single consumer of the input stream.
//!
//! [`InputPump`] owns one [`Subscription`] and is the only writer of the
//! keyboard and pointer states. It is a Reactive Streams consumer, not a
//! dispatcher: it never holds a list of handlers and never calls consumer
//! code from a backend thread. It drains what the stream has ready, applies
//! each event to the shared state, and stages the events of the frame so the
//! widget tree can hand them to the handlers it owns.
//!
//! # Where the design comes from
//!
//! | Decision | Reference design |
//! | --- | --- |
//! | Demand-granted delivery, bounded queue, non-terminal lag signal | Reactive Streams `Subscription.request(n)`; the batched replenishment is the same "request half the buffer when the buffer is half spent" rule that Project Reactor and `tokio::sync::mpsc` receivers use. |
//! | Per-frame double buffering, readers without registration | `bevy::ecs::event::Events`, which advances two buffers per update and lets each reader consume the frame without a callback list. |
//! | Application-driven, non-blocking drain | `winit`'s `EventLoop::pump_events` and `run_ontop_of`: the host loop decides when input is pulled, so the pump has no thread of its own. |
//! | Motion coalescing inside one frame | The browser `Pointer Events` coalesced-event model: many raw motions between two animation frames become one position with a summed delta, which is what a text cursor and a hover pass need. |
//!
//! # Contract
//!
//! The pump grants demand in batches, so a consumer that stops calling
//! [`pump`](Self::pump) applies backpressure by construction: the queue fills,
//! then the backend starts dropping and reports [`ControlError::Lagged`], and
//! the pump records the loss instead of growing memory.
//!
//! A frame runs in this order:
//!
//! 1. [`pump`](Self::pump) or [`poll_pump`](Self::poll_pump) — the states move
//!    and the frame's events are staged.
//! 2. [`finish_frame`](Self::finish_frame) — the staged events become the frame
//!    handlers read, and the pointer accumulators are folded into the returned
//!    [`PointerSnapshot`].
//! 3. [`frame`](Self::frame) — the tree walk hands the frame to each widget
//!    that implements a handler trait.
//!
//! Steps 1 and 2 belong to the application, before it composes; step 3 belongs
//! to the composition. Because the pump never calls a handler, a widget that
//! is not in the tree simply does not receive the frame, and removing or
//! rebuilding a widget cannot leave a dangling subscription behind.
//!
//! [`ControlError::Lagged`]: codevar_kuitui_control::ControlError::Lagged

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::poll_fn;
use core::task::{Context, Poll, Waker};

use codevar_kuitui_control::{
    Backend, ClassSet, Control, ControlError, DEFAULT_CAPACITY, Event, InputEvent, Subscription,
};
use codevar_logger::{log_debug, log_warn};

use crate::event::EventResult;
use crate::input::pointer::{PointerPhase, PointerSnapshot, PointerState, PointerUpdate};
use crate::input::{Input, KeyTransition, KeyboardHandler, KeyboardState, MouseHandler};

/// How many events the pump re-grants at a time, as a share of the queue.
///
/// Half of the capacity is the Reactive Streams convention: it keeps one half
/// of the buffer in flight while the consumer works, so a backend never waits
/// on a request that arrives after the queue is already full.
const DEMAND_DIVISOR: u64 = 2;

/// What one drain of the stream did.
///
/// The report is a value, not a callback: an application that only needs to
/// know whether to recompose reads
/// [`is_empty`](Self::is_empty), and a profiler reads the counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PumpReport {
    /// Events applied to the states and staged for the frame.
    pub delivered: u32,
    /// Raw pointer motions folded into an already staged motion.
    pub coalesced: u32,
    /// Events the bounded queue dropped while this drain ran.
    pub dropped: u64,
    /// Whether the stream has ended and will deliver nothing more.
    pub closed: bool,
}

impl PumpReport {
    /// Whether the drain applied no event at all.
    ///
    /// A frame with no events needs no recomposition, which is the common case
    /// for an idle editor.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.delivered == 0
    }
}

/// The events of one frame, as the widget tree receives them.
///
/// The frame borrows from the pump, so it cannot outlive the pump that built
/// it — which is also what keeps a handler from storing a frame across a
/// recomposition.
#[derive(Debug, Clone, Copy)]
pub struct InputFrame<'a> {
    /// Key transitions, oldest first.
    keys: &'a [KeyTransition],
    /// Pointer updates, oldest first, with the motions of the frame already
    /// coalesced.
    pointers: &'a [PointerUpdate],
    /// Events that belong to no state: text, gamepads and device changes.
    rest: &'a [Arc<Event>],
}

impl<'a> InputFrame<'a> {
    /// The key transitions of the frame.
    #[must_use]
    pub const fn keys(&self) -> &'a [KeyTransition] {
        self.keys
    }

    /// The pointer updates of the frame.
    #[must_use]
    pub const fn pointers(&self) -> &'a [PointerUpdate] {
        self.pointers
    }

    /// The events of the frame that no device state owns.
    #[must_use]
    pub const fn events(&self) -> &'a [Arc<Event>] {
        self.rest
    }

    /// Whether the frame carries nothing.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.keys.is_empty() && self.pointers.is_empty() && self.rest.is_empty()
    }

    /// How many separate delivery slots the frame holds.
    ///
    /// This counts the coalesced pointer updates, not the raw motion events
    /// that produced them.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len() + self.pointers.len() + self.rest.len()
    }

    /// Gives every key transition of the frame to `handler`.
    ///
    /// The state was already updated, so a handler that reads
    /// [`KeyboardHandler::keyboard`] inside the hook sees each transition.
    /// Delivery stops at the first [`EventResult::Consumed`].
    pub fn deliver_keys_to<H: KeyboardHandler + ?Sized>(&self, handler: &mut H) -> EventResult {
        for transition in self.keys {
            if handler.on_key(transition).is_consumed() {
                return EventResult::Consumed;
            }
        }
        EventResult::Ignored
    }

    /// Gives every pointer update of the frame to `handler`.
    pub fn deliver_pointers_to<H: MouseHandler + ?Sized>(&self, handler: &mut H) -> EventResult {
        for update in self.pointers {
            if handler.on_pointer(update).is_consumed() {
                return EventResult::Consumed;
            }
        }
        EventResult::Ignored
    }

    /// Gives every stateless event of the frame to `handler`.
    pub fn deliver_events_to<H: Input + ?Sized>(&self, handler: &mut H) -> EventResult {
        for event in self.rest {
            if handler
                .on_input(&event.payload, event.timestamp)
                .is_consumed()
            {
                return EventResult::Consumed;
            }
        }
        EventResult::Ignored
    }

    /// Gives the whole frame to a widget that handles both devices.
    ///
    /// The order is keyboard, pointer, then the events only [`Input`] sees, so
    /// a shortcut handler can consume a key before a text field reads the text
    /// the same press produced.
    pub fn deliver_to<H>(&self, handler: &mut H) -> EventResult
    where
        H: KeyboardHandler + MouseHandler + Input,
    {
        let keys = self.deliver_keys_to(handler);
        if keys.is_consumed() {
            return keys;
        }
        let pointers = self.deliver_pointers_to(handler);
        if pointers.is_consumed() {
            return pointers;
        }
        self.deliver_events_to(handler)
    }
}

/// The frame's staged events, before they become the frame handlers read.
#[derive(Debug, Default)]
struct Frame {
    /// Key transitions in arrival order.
    keys: Vec<KeyTransition>,
    /// Pointer updates in arrival order, with motions merged.
    pointers: Vec<PointerUpdate>,
    /// Everything else, as shared events so no payload is copied.
    rest: Vec<Arc<Event>>,
    /// Raw events staged, including the motions merged into a pointer update.
    total: u64,
    /// Motions merged into an existing pointer update.
    coalesced: u64,
}

impl Frame {
    /// Empties the frame without giving back the buffers.
    fn reset(&mut self) {
        self.keys.clear();
        self.pointers.clear();
        self.rest.clear();
        self.total = 0;
        self.coalesced = 0;
    }
}

/// The single consumer of the input stream and the only writer of the input
/// states.
///
/// One pump per application, owning one subscription opened from a
/// [`Control`]. The states it writes are borrowed, usually the process-wide
/// `KEYBOARD` and `POINTER` statics, so the handlers a widget binds to read
/// exactly what the pump wrote.
pub struct InputPump<'a> {
    /// The stream this pump consumes.
    subscription: Subscription,
    /// The keyboard state the pump writes.
    keyboard: &'a KeyboardState,
    /// The pointer state the pump writes.
    pointer: &'a PointerState,
    /// The frame being filled.
    staging: Frame,
    /// The frame the handlers read.
    delivered: Frame,
    /// Events to consume before demand is re-granted.
    batch: u64,
    /// Events consumed since demand was last granted.
    spent: u64,
    /// Events the queue dropped since the pump was created.
    dropped: u64,
    /// The error that ended the stream, if it ended by error.
    failure: Option<ControlError>,
}

impl<'a> InputPump<'a> {
    /// Wraps an already opened stream so that it writes `keyboard` and
    /// `pointer`.
    ///
    /// Use this when the application chose the device classes itself, for
    /// example to add a microphone stream. `capacity` must be the queue length
    /// `subscription` was opened with, because the prefetch it starts with —
    /// and therefore the batch the pump re-grants — is derived from it.
    #[must_use]
    pub fn new(
        subscription: Subscription,
        keyboard: &'a KeyboardState,
        pointer: &'a PointerState,
        capacity: usize,
    ) -> Self {
        let capacity = capacity.max(1) as u64;
        Self {
            subscription,
            keyboard,
            pointer,
            staging: Frame::default(),
            delivered: Frame::default(),
            batch: (capacity / DEMAND_DIVISOR).max(1),
            spent: 0,
            dropped: 0,
            failure: None,
        }
    }

    /// Opens the keyboard and pointer streams of `control` with the default
    /// queue length.
    pub fn open<B: Backend>(
        control: &mut Control<B>,
        keyboard: &'a KeyboardState,
        pointer: &'a PointerState,
    ) -> Result<Self, ControlError> {
        Self::open_with_capacity(control, keyboard, pointer, DEFAULT_CAPACITY)
    }

    /// Opens the keyboard and pointer streams of `control` with `capacity`.
    ///
    /// The classes are opened lazily by the control, so a platform with no
    /// pointer device returns an error here rather than in the frame loop.
    pub fn open_with_capacity<B: Backend>(
        control: &mut Control<B>,
        keyboard: &'a KeyboardState,
        pointer: &'a PointerState,
        capacity: usize,
    ) -> Result<Self, ControlError> {
        let subscription =
            control.open_many_with_capacity(ClassSet::KEYBOARD | ClassSet::POINTER, capacity)?;
        log_debug!("input: the pump is streaming keyboard and pointer events");
        Ok(Self::new(subscription, keyboard, pointer, capacity))
    }

    /// The keyboard state this pump writes.
    #[must_use]
    pub const fn keyboard(&self) -> &'a KeyboardState {
        self.keyboard
    }

    /// The pointer state this pump writes.
    #[must_use]
    pub const fn pointer(&self) -> &'a PointerState {
        self.pointer
    }

    /// Drains every event the stream has ready and returns what it did.
    ///
    /// Non-blocking: the call returns as soon as the queue is empty, so it
    /// belongs at the top of a frame loop that also has to draw. Demand is
    /// replenished in batches as the drain spends the prefetch, which is what
    /// keeps the queue fed without an unbounded request.
    pub fn pump(&mut self) -> PumpReport {
        // A drain without a task still has to hand the stream a waker; events
        // that arrive after it are picked up by the next call.
        let mut context = Context::from_waker(Waker::noop());
        self.drain(&mut context)
    }

    /// Drains the stream for a task, parking it only when nothing is queued.
    ///
    /// `Poll::Pending` means the queue is empty and the waker has been
    /// registered, so a frame loop built on an executor sleeps instead of
    /// spinning. A closed stream is reported as `Ready` with
    /// [`PumpReport::closed`] set, because it will never wake again.
    pub fn poll_pump(&mut self, context: &mut Context<'_>) -> Poll<PumpReport> {
        let report = self.drain(context);
        if report.delivered != 0 || report.closed {
            Poll::Ready(report)
        } else {
            Poll::Pending
        }
    }

    /// Waits until the stream delivers at least one event, then drains it.
    ///
    /// This is the async entry point for a task-driven frame loop; it never
    /// busy-waits because it parks on the subscription's waker.
    pub async fn next_events(&mut self) -> PumpReport {
        poll_fn(|context| self.poll_pump(context)).await
    }

    /// Ends the frame and returns the pointer totals it produced.
    ///
    /// The staged events become [`frame`](Self::frame), the buffers are
    /// swapped rather than reallocated, and the pointer accumulators are
    /// folded by [`PointerState::finish_frame`] — so the returned snapshot
    /// describes exactly the frame being handed to the handlers.
    pub fn finish_frame(&mut self) -> PointerSnapshot {
        core::mem::swap(&mut self.staging, &mut self.delivered);
        self.staging.reset();
        self.pointer.finish_frame()
    }

    /// The frame the widget tree should be given.
    #[must_use]
    pub fn frame(&self) -> InputFrame<'_> {
        InputFrame {
            keys: &self.delivered.keys,
            pointers: &self.delivered.pointers,
            rest: &self.delivered.rest,
        }
    }

    /// Releases the stream early.
    ///
    /// Idempotent, and also what dropping the pump does. The device classes
    /// stay registered with the [`Control`], which releases them on request.
    pub fn cancel(&mut self) {
        self.subscription.cancel();
        self.staging.reset();
        self.delivered.reset();
    }

    /// Whether the stream will deliver nothing more.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.subscription.is_cancelled()
    }

    /// The error that ended the stream, if one did.
    #[must_use]
    pub fn failure(&self) -> Option<&ControlError> {
        self.failure.as_ref()
    }

    /// How many events the bounded queue dropped since the pump was created.
    ///
    /// A growing count means the frame loop is slower than the device; the
    /// states stay correct because the queue keeps the newest events, but a
    /// click can be lost with the motion that preceded it.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Drains until the queue is empty, the stream parks, or it ends.
    fn drain(&mut self, context: &mut Context<'_>) -> PumpReport {
        let mut report = PumpReport::default();
        if self.is_closed() {
            report.closed = true;
            return report;
        }
        let coalesced_before = self.staging.coalesced;
        loop {
            if self.spent >= self.batch && self.grant_demand().is_err() {
                break;
            }
            match self.subscription.poll_next(context) {
                Poll::Pending => break,
                Poll::Ready(None) => {
                    report.closed = true;
                    break;
                }
                Poll::Ready(Some(Err(error))) => {
                    match error {
                        ControlError::Lagged { dropped } => {
                            self.dropped = self.dropped.saturating_add(dropped);
                            report.dropped = report.dropped.saturating_add(dropped);
                            log_warn!("input: the stream dropped {dropped} events");
                        }
                        terminal => {
                            log_warn!("input: the event stream ended: {terminal}");
                            self.failure = Some(terminal);
                            self.subscription.cancel();
                            report.closed = true;
                        }
                    }
                    if report.closed {
                        break;
                    }
                }
                Poll::Ready(Some(Ok(event))) => {
                    self.spent = self.spent.saturating_add(1);
                    self.stage(&event);
                    report.delivered = report.delivered.saturating_add(1);
                }
            }
        }
        report.coalesced = self
            .staging
            .coalesced
            .saturating_sub(coalesced_before) as u32;
        report
    }

    /// Re-grants the demand the last batch spent.
    fn grant_demand(&mut self) -> Result<(), ControlError> {
        let amount = core::mem::take(&mut self.spent);
        self.subscription.request(amount.max(1))
    }

    /// Applies one event to the states and stages it for the frame.
    fn stage(&mut self, event: &Arc<Event>) {
        self.staging.total = self.staging.total.saturating_add(1);
        match &event.payload {
            InputEvent::Key(key) => {
                self.keyboard.apply(key, event.timestamp);
                self.staging
                    .keys
                    .push(KeyTransition::new(key, event.timestamp));
            }
            InputEvent::Pointer(pointer) => {
                let update = PointerUpdate::new(self.pointer, pointer, event.timestamp);
                if merge_motion(&mut self.staging.pointers, update) {
                    self.staging.coalesced = self.staging.coalesced.saturating_add(1);
                } else {
                    self.staging.pointers.push(update);
                }
            }
            _ => self.staging.rest.push(Arc::clone(event)),
        }
    }
}

impl core::fmt::Debug for InputPump<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InputPump")
            .field("staged", &self.staging.total)
            .field("frame", &self.delivered.total)
            .field("coalesced", &self.staging.coalesced)
            .field("batch", &self.batch)
            .field("spent", &self.spent)
            .field("dropped", &self.dropped)
            .field("failure", &self.failure)
            .field("closed", &self.is_closed())
            .finish_non_exhaustive()
    }
}

/// Folds a motion into the staged motion of the same frame.
///
/// Only two consecutive motions merge: a button, scroll or lifecycle event in
/// between has to stay visible, or a click would be reported after the motion
/// that followed it.
fn merge_motion(pointers: &mut [PointerUpdate], update: PointerUpdate) -> bool {
    if update.phase != PointerPhase::Motion {
        return false;
    }
    match pointers.last_mut() {
        Some(previous) if previous.phase == PointerPhase::Motion => {
            let delta = previous.delta.add(update.delta);
            *previous = PointerUpdate {
                timestamp: update.timestamp,
                position: update.position.or(previous.position),
                delta,
                ..update
            };
            true
        }
        _ => false,
    }
}
