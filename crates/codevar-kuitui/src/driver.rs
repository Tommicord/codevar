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

//! The frame driver that runs the pump and the composition together.
//!
//! An application that owns an [`InputPump`] and a [`Composition`] has to call
//! them in the right order every frame: drain the stream before the states are
//! read, end the frame so the staged events become the frame the tree sees,
//! offer those events to the tree so a click can write a state, and only then
//! recompose and lay out. [`FrameDriver`] is that order, in one call.
//!
//! ```ignore
//! let mut driver = FrameDriver::open(&mut control, composition)?;
//! loop {
//!     let report = driver.frame(size, false);
//!     if report.closed {
//!         break;
//!     }
//!     backend.submit(driver.render());
//! }
//! ```
//!
//! # The order of a frame
//!
//! 1. [`InputPump::pump`] — the device states move and the frame's events are
//!    staged. Nothing may read the states before this, or a widget sees last
//!    frame's pointer.
//! 2. [`InputPump::finish_frame`] — the staged events become
//!    [`InputPump::frame`], and the pointer accumulators fold into the snapshot
//!    [`FrameDriver::pointer`] returns.
//! 3. [`bridge_frame`] — the staged events become the tree's [`Event`]s, in the
//!    order the frame delivers them to a handler. The first frame builds the
//!    tree before it offers anything to it, because a composition has no tree
//!    until its first recomposition.
//! 4. [`Composition::handle_event`] for each of them — a widget that writes a
//!    state marks the composition dirty here, which is why this step runs
//!    before the tree is rebuilt.
//! 5. [`Composition::recompose`] and [`Composition::layout`] — both skip their
//!    work when nothing is dirty, so an idle frame costs the drain alone.
//!
//! The driver never registers a callback with the pump and never holds a list
//! of widgets: it is a caller of the two layers, which keeps the "no observer
//! list" rule of the input module intact.

use alloc::vec::Vec;

use codevar_kuitui_control::{Backend, Control, ControlError};

use crate::compose::{Composition, Recomposition};
use crate::event::{Event, EventResult};
use crate::geometry::Size;
use crate::input::{InputPump, KEYBOARD, POINTER, PointerSnapshot, bridge_frame};
use crate::paint::DrawList;

/// What one frame of a [`FrameDriver`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameReport {
    /// Device events the pump applied to the states.
    pub delivered: u32,
    /// Raw pointer motions folded into a staged motion.
    pub coalesced: u32,
    /// Events the bounded queue dropped while this frame drained.
    pub dropped: u64,
    /// Tree events the bridge produced for the frame.
    pub events: u32,
    /// Tree events a widget consumed.
    pub consumed: u32,
    /// Whether the tree was rebuilt.
    pub recomposed: bool,
    /// Whether the input stream ended.
    pub closed: bool,
}

impl FrameReport {
    /// Whether the frame changed nothing at all.
    ///
    /// An application uses this to skip a redraw: no event arrived, no tree was
    /// rebuilt, and no widget consumed an event.
    #[must_use]
    pub const fn is_idle(&self) -> bool {
        self.delivered == 0 && !self.recomposed && self.consumed == 0
    }
}

/// The pump and the composition of one window, run as one frame.
///
/// The driver borrows the states through the pump it was given and owns the
/// composition, the display list and the buffer of bridged events, so the
/// buffers are allocated once for the life of the window.
pub struct FrameDriver<'device, 'content> {
    /// The stream and the states it writes.
    pump: InputPump<'device>,
    /// The tree the events are offered to and that is laid out each frame.
    composition: Composition<'content>,
    /// The tree events of the last frame, kept between frames to reuse it.
    events: Vec<Event>,
    /// The display list the tree renders into.
    draw: DrawList,
    /// The pointer totals of the last finished frame.
    pointer: Option<PointerSnapshot>,
    /// Whether the tree has been built, which the first frame has to do before
    /// it offers events to it.
    booted: bool,
}

impl<'device, 'content> FrameDriver<'device, 'content> {
    /// Drives `composition` with the events `pump` reads.
    #[must_use]
    pub const fn new(pump: InputPump<'device>, composition: Composition<'content>) -> Self {
        Self {
            pump,
            composition,
            events: Vec::new(),
            draw: DrawList::new(),
            pointer: None,
            booted: false,
        }
    }

    /// Opens the keyboard and pointer streams of `control` with the default
    /// queue length and drives `composition` with them.
    ///
    /// The pump writes the process-wide [`KEYBOARD`] and [`POINTER`] states,
    /// which is what a handler bound with
    /// [`KeyboardBinding::global`](crate::input::KeyboardBinding::global) or
    /// [`MouseBinding::global`](crate::input::MouseBinding::global) reads.
    pub fn open<B: Backend>(
        control: &mut Control<B>,
        composition: Composition<'content>,
    ) -> Result<Self, ControlError> {
        let pump = InputPump::open(control, &KEYBOARD, &POINTER)?;
        Ok(Self::new(pump, composition))
    }

    /// Opens the streams of `control` with `capacity` and drives `composition`.
    pub fn open_with_capacity<B: Backend>(
        control: &mut Control<B>,
        composition: Composition<'content>,
        capacity: usize,
    ) -> Result<Self, ControlError> {
        let pump = InputPump::open_with_capacity(control, &KEYBOARD, &POINTER, capacity)?;
        Ok(Self::new(pump, composition))
    }

    /// Runs one frame and returns what it did.
    ///
    /// `force` rebuilds the tree even when no state is dirty, which is what an
    /// application calls after it changed a value the composition cannot
    /// observe; it also lays the tree out again. The size is handed to the
    /// layout every frame, and the layout skips while the size is unchanged and
    /// the tree was not rebuilt.
    pub fn frame(&mut self, size: Size, force: bool) -> FrameReport {
        let drain = self.pump.pump();
        self.pointer = Some(self.pump.finish_frame());

        let booted = !self.booted;
        if booted {
            self.booted = true;
            self.composition.recompose(true);
        }

        self.events.clear();
        bridge_frame(self.pump.frame(), &mut self.events);
        let mut consumed = 0;
        for event in &self.events {
            if self.composition.handle_event(event).is_consumed() {
                consumed += 1;
            }
        }

        // The boot build above never sees the events of this frame, so the
        // tree is recomposed again here: a handler that wrote a state has to
        // be reflected in the same frame it was handled in.
        let rebuilt = matches!(self.composition.recompose(force), Recomposition::Recomposed);
        let recomposed = booted || rebuilt;
        self.composition.layout(size, force);

        FrameReport {
            delivered: drain.delivered,
            coalesced: drain.coalesced,
            dropped: drain.dropped,
            events: self.events.len() as u32,
            consumed,
            recomposed,
            closed: drain.closed,
        }
    }

    /// Records the tree into the display list and returns it.
    ///
    /// The list is emptied by [`Composition::render`], so it holds exactly the
    /// operations of this frame and a backend may keep borrowing it until the
    /// next call.
    pub fn render(&mut self) -> &DrawList {
        self.composition.render(&mut self.draw);
        &self.draw
    }

    /// Offers one event to the tree without touching the input stream.
    ///
    /// This is how a backend delivers what the stream does not carry: a resized
    /// or closed window, or a focus change the application decided on rather
    /// than the user. The tree exists after the first
    /// [`frame`](Self::frame) built it, so an event delivered before that finds
    /// no widget to handle it.
    pub fn deliver(&mut self, event: &Event) -> EventResult {
        self.composition.handle_event(event)
    }

    /// The pointer totals of the last finished frame.
    ///
    /// `None` before the first frame. The motion and scroll are the totals of
    /// that frame, already reset in the state, which is why the driver keeps
    /// them.
    #[must_use]
    pub const fn pointer(&self) -> Option<PointerSnapshot> {
        self.pointer
    }

    /// The stream the driver reads.
    #[must_use]
    pub const fn pump(&self) -> &InputPump<'device> {
        &self.pump
    }

    /// The stream, for an application that ends or reopens it.
    pub fn pump_mut(&mut self) -> &mut InputPump<'device> {
        &mut self.pump
    }

    /// The tree the driver runs.
    #[must_use]
    pub const fn composition(&self) -> &Composition<'content> {
        &self.composition
    }

    /// The tree, for an application that sets content, theme or focus.
    pub fn composition_mut(&mut self) -> &mut Composition<'content> {
        &mut self.composition
    }

    /// The display list of the last [`render`](Self::render).
    #[must_use]
    pub const fn draw_list(&self) -> &DrawList {
        &self.draw
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::rc::Rc;
    use alloc::string::ToString;
    use core::cell::RefCell;
    use std::sync::{Arc, Mutex};

    use codevar_kuitui_control::{
        ClassSet, ControlError, DeviceClass, DeviceId, DeviceInfo, EventSink, InputEvent, KeyCode, KeyEvent,
        KeyState, Modifiers, Point, PointerButton, PointerEvent, TextEvent,
    };

    use super::*;
    use crate::color::Color;
    use crate::event::{
        Key, KeyEvent as TreeKey, PointerButtons as TreeButtons, PointerEvent as TreePointer,
        PointerEventKind, WindowEvent,
    };
    use crate::geometry::{Constraints, Offset, Rect};
    use crate::input::{KeyboardState, PointerState};
    use crate::node::Node;
    use crate::state::MutableState;
    use crate::theme::Theme;
    use crate::widget::{DrawScope, EventScope, MeasureScope, Widget};

    /// A backend that only publishes, so a test decides what the stream holds.
    #[derive(Clone, Default)]
    struct TestBackend {
        sink: Arc<Mutex<Option<EventSink>>>,
    }

    impl TestBackend {
        fn publish(&self, event: InputEvent) {
            let sink = self.sink.lock().unwrap().clone();
            if let Some(sink) = sink {
                sink.emit(event);
            }
        }
    }

    impl Backend for TestBackend {
        fn start(&mut self, _class: DeviceClass, sink: EventSink) -> Result<(), ControlError> {
            let mut slot = self.sink.lock().unwrap();
            if slot.is_none() {
                *slot = Some(sink);
            }
            Ok(())
        }

        fn stop(&mut self, _class: DeviceClass) {}

        fn devices(&self) -> Vec<DeviceInfo> {
            Vec::new()
        }
    }

    /// The events the tree was offered, in the order it was offered them.
    #[derive(Clone, Default)]
    struct Log(Rc<RefCell<Vec<Event>>>);

    impl Log {
        /// The events recorded so far, leaving the log empty.
        fn take(&self) -> Vec<Event> {
            core::mem::take(&mut *self.0.borrow_mut())
        }
    }

    /// The widget of the tests: records every event and can write a state.
    struct Recorder {
        log: Log,
        state: Option<Rc<MutableState<u32>>>,
        consume: bool,
    }

    impl Widget for Recorder {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(10.0, 10.0))
        }

        fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
            scope.fill_rect(bounds, Color::RED);
        }

        fn on_event(&mut self, _scope: &mut EventScope<'_>, event: &Event) -> EventResult {
            self.log.0.borrow_mut().push(*event);
            if let (Some(state), Event::Key(_)) = (&self.state, event) {
                state.set(state.get() + 1);
            }
            if self.consume {
                EventResult::Consumed
            } else {
                EventResult::Ignored
            }
        }
    }

    /// A driver whose tree is one recorder, plus the stream that feeds it.
    ///
    /// The control is returned alive: dropping it ends every subscription, and a
    /// driver that began closed would report the frame as closed instead of
    /// delivering the events a test publishes.
    fn harness(
        log: Log,
        state: Option<Rc<MutableState<u32>>>,
        consume: bool,
    ) -> (TestBackend, Control<TestBackend>, FrameDriver<'static, 'static>) {
        let backend = TestBackend::default();
        let mut control = Control::new(backend.clone());
        let subscription = control
            .open_many_with_capacity(ClassSet::KEYBOARD | ClassSet::POINTER, 16)
            .unwrap();
        let keyboard = Box::leak(Box::new(KeyboardState::new()));
        let pointer = Box::leak(Box::new(PointerState::new()));
        let pump = InputPump::new(subscription, keyboard, pointer, 16);

        let mut composition = Composition::new(Theme::light());
        composition.set_content(move |composer| {
            if let Some(state) = &state {
                composer.observe(state);
            }
            Node::new(Recorder {
                log: log.clone(),
                state: state.clone(),
                consume,
            })
        });
        (backend, control, FrameDriver::new(pump, composition))
    }

    fn key(code: KeyCode) -> InputEvent {
        InputEvent::Key(KeyEvent {
            device: DeviceId::new(3),
            code,
            state: KeyState::Pressed,
            modifiers: Modifiers::NONE,
        })
    }

    fn text(body: &str) -> InputEvent {
        InputEvent::Text(TextEvent {
            device: DeviceId::new(3),
            text: body.to_string(),
        })
    }

    fn move_to(position: Point) -> InputEvent {
        InputEvent::Pointer(PointerEvent::Motion {
            position: Some(position),
            delta: None,
        })
    }

    fn motion(x: f32, y: f32) -> InputEvent {
        InputEvent::Pointer(PointerEvent::Motion {
            position: None,
            delta: Some(Point::new(x, y)),
        })
    }

    fn press(button: PointerButton) -> InputEvent {
        InputEvent::Pointer(PointerEvent::Button {
            button,
            state: KeyState::Pressed,
            position: None,
            modifiers: Modifiers::NONE,
        })
    }

    #[test]
    fn a_frame_bridges_the_stream_into_the_tree_in_delivery_order() {
        let log = Log::default();
        let (backend, _control, mut driver) = harness(log.clone(), None, false);
        backend.publish(key(KeyCode::Escape));
        backend.publish(move_to(Point::new(5.0, 5.0)));
        backend.publish(press(PointerButton::Primary));
        backend.publish(text("hi"));

        let report = driver.frame(Size::new(10.0, 10.0), false);
        assert_eq!(report.delivered, 4);
        assert_eq!(report.events, 5);
        assert_eq!(report.consumed, 0);
        assert_eq!(
            log.take(),
            [
                Event::Key(TreeKey::new(Key::Escape, true)),
                Event::Pointer(TreePointer::new(PointerEventKind::Move, Offset::new(5.0, 5.0))),
                Event::Pointer(
                    TreePointer::new(PointerEventKind::Down, Offset::new(5.0, 5.0))
                        .with_buttons(TreeButtons::PRIMARY),
                ),
                Event::Text('h'),
                Event::Text('i'),
            ]
        );
    }

    #[test]
    fn a_frame_with_nothing_to_do_reports_itself_idle() {
        let log = Log::default();
        let (_backend, _control, mut driver) = harness(log.clone(), None, false);
        driver.frame(Size::new(10.0, 10.0), true);
        log.take();
        let report = driver.frame(Size::new(10.0, 10.0), false);
        assert!(report.is_idle());
        assert!(log.take().is_empty());
    }

    #[test]
    fn a_state_written_by_a_handler_recomposes_the_same_frame() {
        let log = Log::default();
        let state = Rc::new(MutableState::new(0_u32));
        let (backend, _control, mut driver) = harness(log, Some(Rc::clone(&state)), false);
        backend.publish(key(KeyCode::Tab));

        let report = driver.frame(Size::new(10.0, 10.0), false);
        assert_eq!(state.get(), 1);
        assert!(report.recomposed);
        assert!(!driver.composition().is_dirty());
    }

    #[test]
    fn a_consumed_event_is_counted_and_a_closed_stream_is_reported() {
        let log = Log::default();
        let (backend, control, mut driver) = harness(log, None, true);
        backend.publish(text("ok"));

        let report = driver.frame(Size::new(10.0, 10.0), false);
        assert_eq!(report.events, 2);
        assert_eq!(report.consumed, 2);

        drop(control);
        assert!(driver.frame(Size::new(10.0, 10.0), false).closed);
        assert!(driver.pump().is_closed());
    }

    #[test]
    fn the_pointer_totals_of_a_frame_survive_the_state_reset() {
        let log = Log::default();
        let (backend, _control, mut driver) = harness(log, None, false);
        backend.publish(motion(4.0, 5.0));
        backend.publish(motion(1.0, 1.0));

        let report = driver.frame(Size::new(10.0, 10.0), false);
        assert_eq!(report.coalesced, 1);
        assert_eq!(report.events, 1);
        assert_eq!(driver.pointer().unwrap().delta, Offset::new(5.0, 6.0));
        assert_eq!(driver.pump().pointer().delta(), Offset::ZERO);
    }

    #[test]
    fn deliver_reaches_the_tree_without_the_stream() {
        let log = Log::default();
        let (_backend, _control, mut driver) = harness(log.clone(), None, false);
        driver.frame(Size::new(10.0, 10.0), false);
        log.take();

        let event = Event::Window(WindowEvent::CloseRequested);
        assert_eq!(driver.deliver(&event), EventResult::Ignored);
        assert_eq!(log.take(), [event]);
    }

    #[test]
    fn open_binds_the_process_wide_states() {
        let mut control = Control::new(TestBackend::default());
        let mut composition = Composition::new(Theme::light());
        composition.set_content(|_| {
            Node::new(Recorder {
                log: Log::default(),
                state: None,
                consume: false,
            })
        });

        let driver = FrameDriver::open(&mut control, composition).unwrap();
        assert!(core::ptr::eq(driver.pump().keyboard(), &KEYBOARD));
        assert!(core::ptr::eq(driver.pump().pointer(), &POINTER));
    }

    #[test]
    fn render_records_the_frame_that_the_layout_placed() {
        let log = Log::default();
        let (_backend, _control, mut driver) = harness(log, None, false);
        assert!(driver.render().is_empty());

        driver.frame(Size::new(10.0, 10.0), true);
        assert_eq!(driver.composition().size(), Size::new(10.0, 10.0));
        assert!(!driver.render().is_empty());
    }
}
