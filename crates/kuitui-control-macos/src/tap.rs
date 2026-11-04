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

//! CoreGraphics event tap and the pointer that routes its events.
//!
//! The tap is installed on the calling thread's run loop and stays
//! listen-only, so the focused application still receives every event. The
//! callback borrows a [`Router`] through the tap's user-data pointer; the
//! router only exists while the tap exists.

use core::ffi::c_void;

use alloc::boxed::Box;
use alloc::vec::Vec;

use codevar_kuitui_control::{DeviceClass, DeviceId, DeviceInfo, EventSink, KeyState};

use crate::decode::{self, KeyAction};
use crate::keymap;

/// Identifier reported for the keyboard stream.
///
/// A tap cannot attribute an event to one physical keyboard, so the whole
/// keyboard stream shares one identifier.
pub(crate) const KEYBOARD_DEVICE: DeviceId = DeviceId::new(1);

/// Identifier reported for the pointer stream.
pub(crate) const POINTER_DEVICE: DeviceId = DeviceId::new(2);

/// A CoreGraphics event tap installed on the caller's run loop.
pub(crate) struct Tap {
    /// The tap's mach port.
    port: CFMachPortRef,
    /// The run loop source wrapping `port`.
    source: CFRunLoopSourceRef,
    /// Routing state owned by the tap's user-data pointer.
    router: *mut Router,
}

impl Tap {
    /// Installs a listen-only tap for keyboard and pointer events.
    ///
    /// Fails when the process has not been granted Input Monitoring: a
    /// listen-only tap sees every keystroke, so macOS asks the user first.
    pub(crate) fn create() -> Result<Self, codevar_kuitui_control::ControlError> {
        // SAFETY: both calls only query and request the current process's tap
        // permission; neither takes an argument.
        let granted = unsafe { CGPreflightListenEventAccess() || CGRequestListenEventAccess() };
        if !granted {
            return Err(codevar_kuitui_control::ControlError::permission_denied(
                "Input Monitoring",
            ));
        }

        let router = Box::into_raw(Box::new(Router::new()));
        // SAFETY: `router` is a live allocation and the callback only
        // dereferences it while the run loop runs, which cannot overlap this
        // call or the teardown in `Drop`.
        let port = unsafe {
            CGEventTapCreate(
                KCG_HID_EVENT_TAP,
                KCG_HEAD_INSERT_EVENT_TAP,
                KCG_EVENT_TAP_OPTION_LISTEN_ONLY,
                event_mask(),
                event_callback,
                router.cast(),
            )
        };
        if port.is_null() {
            // SAFETY: the allocation was created above and was never shared.
            drop(unsafe { Box::from_raw(router) });
            return Err(codevar_kuitui_control::ControlError::backend(
                "CoreGraphics refused to create the event tap",
            ));
        }
        // SAFETY: `router` is exclusively owned here, and the tap callback
        // needs the port to re-enable a tap the system disabled.
        unsafe { (*router).port = port };

        // SAFETY: `port` is a valid tap and the default allocator is
        // requested with a null allocator reference.
        let source = unsafe { CFMachPortCreateRunLoopSource(core::ptr::null(), port, 0) };
        if source.is_null() {
            // SAFETY: both references are owned by this function.
            unsafe {
                CFRelease(port.cast());
                drop(Box::from_raw(router));
            }
            return Err(codevar_kuitui_control::ControlError::backend(
                "CoreGraphics refused to create the run loop source",
            ));
        }

        // SAFETY: the run loop is the current thread's; adding the source
        // retains it, and the tap is enabled after the source is reachable.
        unsafe {
            let run_loop = CFRunLoopGetCurrent();
            CFRunLoopAddSource(run_loop, source, kCFRunLoopCommonModes);
            CGEventTapEnable(port, true);
        }

        Ok(Self { port, source, router })
    }

    /// Services the events the run loop has ready and returns how many the
    /// router decoded.
    pub(crate) fn pump(&mut self) -> usize {
        // SAFETY: the source belongs to this thread's run loop, and a
        // zero-length run only services sources that are already ready.
        unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.0, true) };
        // SAFETY: the tap owns the only allocation behind `router`, and the
        // run loop has returned, so no callback is running.
        let router = unsafe { &mut *self.router };
        core::mem::take(&mut router.delivered)
    }

    /// Adds the stream of `class` to the tap.
    pub(crate) fn add(&mut self, class: DeviceClass, sink: EventSink, info: DeviceInfo) {
        // SAFETY: no run loop is running while the caller registers a class,
        // so no callback can read the router concurrently.
        unsafe { (*self.router).add(class, sink, info) };
    }

    /// Whether `class` already has a stream on this tap.
    pub(crate) fn has(&self, class: DeviceClass) -> bool {
        // SAFETY: the router is owned by the tap and this read does not race
        // with a callback, which only runs inside `pump`.
        unsafe {
            (*self.router)
                .sessions
                .iter()
                .any(|session| session.class == class)
        }
    }

    /// Removes the stream of `class` from the tap.
    pub(crate) fn remove(&mut self, class: DeviceClass) {
        // SAFETY: see `add`: registration changes never overlap a callback.
        unsafe { (*self.router).remove(class) };
    }

    /// The devices of the streams that are open.
    pub(crate) fn devices(&self) -> Vec<DeviceInfo> {
        // SAFETY: the router is owned by the tap and is only mutated between
        // pumps.
        unsafe {
            (*self.router)
                .sessions
                .iter()
                .map(|session| session.info.clone())
                .collect()
        }
    }
}

impl Drop for Tap {
    fn drop(&mut self) {
        // The run loop is thread-local, so the source must be removed from
        // the loop it was added to. `MacOsBackend` documents that it is
        // dropped on the thread that created it.
        // SAFETY: every reference here is owned by the tap; the run loop is
        // this thread's; and no callback can start because the tap is
        // disabled and its source removed before the router is freed.
        unsafe {
            CGEventTapEnable(self.port, false);
            let run_loop = CFRunLoopGetCurrent();
            CFRunLoopRemoveSource(run_loop, self.source, kCFRunLoopCommonModes);
            CFRelease(self.source.cast());
            CFRelease(self.port.cast());
            drop(Box::from_raw(self.router));
        }
    }
}

/// One registered class.
struct Session {
    /// Class the session serves.
    class: DeviceClass,
    /// Where its events go.
    sink: EventSink,
    /// Description published when the class opened.
    info: DeviceInfo,
    /// Keyboard decoding state.
    keyboard: decode::KeyboardState,
    /// Pointer decoding state.
    pointer: decode::PointerState,
}

/// State shared with the tap callback.
struct Router {
    /// One entry per class that is open.
    sessions: Vec<Session>,
    /// Number of events decoded since the last pump.
    delivered: usize,
    /// The tap's port, for re-enabling a disabled tap.
    port: CFMachPortRef,
}

impl Router {
    /// Creates an empty router before the tap's port exists.
    const fn new() -> Self {
        Self {
            sessions: Vec::new(),
            delivered: 0,
            port: core::ptr::null_mut(),
        }
    }

    /// Adds a stream.
    fn add(&mut self, class: DeviceClass, sink: EventSink, info: DeviceInfo) {
        let id = info.id;
        self.sessions.push(Session {
            class,
            sink,
            info,
            keyboard: decode::KeyboardState::new(id),
            pointer: decode::PointerState::new(),
        });
    }

    /// Removes a stream.
    fn remove(&mut self, class: DeviceClass) {
        self.sessions
            .retain(|session| session.class != class);
    }

    /// Re-enables the tap after the system disabled it.
    ///
    /// The system sends this notification with a null event when the callback
    /// ran too long or the user toggled the tap off, so it is handled before
    /// the event is decoded.
    fn recover(&self, event_type: CGEventType) {
        if event_type != KCG_EVENT_TAP_DISABLED_BY_TIMEOUT
            && event_type != KCG_EVENT_TAP_DISABLED_BY_USER_INPUT
        {
            return;
        }
        // SAFETY: the port is the tap this callback was installed for.
        unsafe { CGEventTapEnable(self.port, true) };
    }

    /// Decodes one event and emits its normalized form.
    fn handle(&mut self, event_type: u32, event: CGEventRef) {
        // SAFETY: the tap passes a live event reference with every callback.
        let flags = unsafe { CGEventGetFlags(event) };
        match event_type {
            KCG_EVENT_KEY_DOWN => {
                let action = if field(event, KCG_KEYBOARD_EVENT_AUTOREPEAT) != 0 {
                    KeyAction::Repeat
                } else {
                    KeyAction::Press
                };
                self.key(keycode(event), action, flags);
            }
            KCG_EVENT_KEY_UP => self.key(keycode(event), KeyAction::Release, flags),
            KCG_EVENT_FLAGS_CHANGED => self.key(keycode(event), KeyAction::FlagsChanged, flags),
            KCG_EVENT_MOUSE_MOVED
            | KCG_EVENT_LEFT_MOUSE_DRAGGED
            | KCG_EVENT_RIGHT_MOUSE_DRAGGED
            | KCG_EVENT_OTHER_MOUSE_DRAGGED => {
                // SAFETY: the tap passes a live event reference.
                let point = unsafe { CGEventGetLocation(event) };
                self.motion(point.x as f32, point.y as f32);
            }
            KCG_EVENT_LEFT_MOUSE_DOWN | KCG_EVENT_RIGHT_MOUSE_DOWN | KCG_EVENT_OTHER_MOUSE_DOWN => {
                self.button(event, KeyState::Pressed, flags)
            }
            KCG_EVENT_LEFT_MOUSE_UP | KCG_EVENT_RIGHT_MOUSE_UP | KCG_EVENT_OTHER_MOUSE_UP => {
                self.button(event, KeyState::Released, flags)
            }
            KCG_EVENT_SCROLL_WHEEL => {
                let continuous = field(event, KCG_SCROLL_WHEEL_EVENT_IS_CONTINUOUS) != 0;
                let (axis1, axis2) = if continuous {
                    (
                        field(event, KCG_SCROLL_WHEEL_EVENT_POINT_DELTA_AXIS_1),
                        field(event, KCG_SCROLL_WHEEL_EVENT_POINT_DELTA_AXIS_2),
                    )
                } else {
                    (
                        field(event, KCG_SCROLL_WHEEL_EVENT_DELTA_AXIS_1),
                        field(event, KCG_SCROLL_WHEEL_EVENT_DELTA_AXIS_2),
                    )
                };
                self.scroll(decode::scroll_delta(continuous, axis1, axis2), flags);
            }
            _ => {}
        }
    }

    /// Applies one keyboard transition.
    fn key(&mut self, keycode: u16, action: KeyAction, flags: u64) {
        let Some(session) = self
            .sessions
            .iter_mut()
            .find(|session| session.class == DeviceClass::Keyboard)
        else {
            return;
        };
        let sink = session.sink.clone();
        session
            .keyboard
            .key(keycode, action, flags, &mut |event| sink.emit(event));
        self.delivered += 1;
    }

    /// Applies one pointer position.
    fn motion(&mut self, x: f32, y: f32) {
        let Some(session) = self
            .sessions
            .iter_mut()
            .find(|session| session.class == DeviceClass::Pointer)
        else {
            return;
        };
        let sink = session.sink.clone();
        session
            .pointer
            .motion(x, y, &mut |event| sink.emit(event));
        self.delivered += 1;
    }

    /// Applies one pointer button transition.
    fn button(&mut self, event: CGEventRef, state: KeyState, flags: u64) {
        let number = field(event, KCG_MOUSE_EVENT_BUTTON_NUMBER);
        let Some(session) = self
            .sessions
            .iter_mut()
            .find(|session| session.class == DeviceClass::Pointer)
        else {
            return;
        };
        let sink = session.sink.clone();
        session.pointer.button(
            decode::button(number),
            state,
            keymap::modifiers(flags),
            &mut |event| sink.emit(event),
        );
        self.delivered += 1;
    }

    /// Applies one scroll step.
    fn scroll(&mut self, delta: codevar_kuitui_control::Point, flags: u64) {
        let Some(session) = self
            .sessions
            .iter_mut()
            .find(|session| session.class == DeviceClass::Pointer)
        else {
            return;
        };
        let sink = session.sink.clone();
        session
            .pointer
            .scroll(delta, keymap::modifiers(flags), &mut |event| sink.emit(event));
        self.delivered += 1;
    }
}

/// Entry point CoreGraphics calls for every event the tap receives.
///
/// # Safety
///
/// `user_info` must be the [`Router`] pointer passed to
/// [`CGEventTapCreate`]. `event` is a live event reference, or null for the
/// notification that reports a disabled tap. Both hold for calls the system
/// makes into the tap.
unsafe extern "C" fn event_callback(
    _proxy: CGEventTapProxy,
    event_type: CGEventType,
    event: CGEventRef,
    user_info: *mut c_void,
) -> CGEventRef {
    // SAFETY: `user_info` is the router pointer passed to `CGEventTapCreate`;
    // the callback only runs while the tap, and therefore the router, is alive.
    if let Some(router) = unsafe { user_info.cast::<Router>().as_mut() } {
        if event.is_null() {
            router.recover(event_type);
        } else {
            router.handle(event_type, event);
        }
    }
    event
}

/// Reads one integer field from an event.
fn field(event: CGEventRef, field: CGEventField) -> i64 {
    // SAFETY: the caller passes an event reference the tap owns.
    unsafe { CGEventGetIntegerValueField(event, field) }
}

/// The virtual keycode of a keyboard event.
fn keycode(event: CGEventRef) -> u16 {
    field(event, KCG_KEYBOARD_EVENT_KEYCODE) as u16
}

/// The mask of the events the tap subscribes to.
const fn event_mask() -> CGEventMask {
    bit(KCG_EVENT_LEFT_MOUSE_DOWN)
        | bit(KCG_EVENT_LEFT_MOUSE_UP)
        | bit(KCG_EVENT_RIGHT_MOUSE_DOWN)
        | bit(KCG_EVENT_RIGHT_MOUSE_UP)
        | bit(KCG_EVENT_MOUSE_MOVED)
        | bit(KCG_EVENT_LEFT_MOUSE_DRAGGED)
        | bit(KCG_EVENT_RIGHT_MOUSE_DRAGGED)
        | bit(KCG_EVENT_KEY_DOWN)
        | bit(KCG_EVENT_KEY_UP)
        | bit(KCG_EVENT_FLAGS_CHANGED)
        | bit(KCG_EVENT_SCROLL_WHEEL)
        | bit(KCG_EVENT_OTHER_MOUSE_DOWN)
        | bit(KCG_EVENT_OTHER_MOUSE_UP)
        | bit(KCG_EVENT_OTHER_MOUSE_DRAGGED)
}

/// The single-event mask of an event type.
const fn bit(event_type: CGEventType) -> CGEventMask {
    1 << event_type
}

type CFMachPortRef = *mut c_void;
type CFRunLoopSourceRef = *mut c_void;
type CFRunLoopRef = *mut c_void;
type CFStringRef = *const c_void;
type CFAllocatorRef = *const c_void;
type CFIndex = isize;
type CGEventRef = *mut c_void;
type CGEventTapProxy = *mut c_void;
type CGEventMask = u64;
type CGEventField = u32;
type CGEventFlags = u64;
type CGEventType = u32;
type CGEventTapLocation = u32;
type CGEventTapPlacement = u32;
type CGEventTapOptions = u32;

type CGEventTapCallBack =
    unsafe extern "C" fn(CGEventTapProxy, CGEventType, CGEventRef, *mut c_void) -> CGEventRef;

const KCG_HID_EVENT_TAP: CGEventTapLocation = 0;
const KCG_HEAD_INSERT_EVENT_TAP: CGEventTapPlacement = 0;
const KCG_EVENT_TAP_OPTION_LISTEN_ONLY: CGEventTapOptions = 1;

const KCG_EVENT_LEFT_MOUSE_DOWN: CGEventType = 1;
const KCG_EVENT_LEFT_MOUSE_UP: CGEventType = 2;
const KCG_EVENT_RIGHT_MOUSE_DOWN: CGEventType = 3;
const KCG_EVENT_RIGHT_MOUSE_UP: CGEventType = 4;
const KCG_EVENT_MOUSE_MOVED: CGEventType = 5;
const KCG_EVENT_LEFT_MOUSE_DRAGGED: CGEventType = 6;
const KCG_EVENT_RIGHT_MOUSE_DRAGGED: CGEventType = 7;
const KCG_EVENT_KEY_DOWN: CGEventType = 10;
const KCG_EVENT_KEY_UP: CGEventType = 11;
const KCG_EVENT_FLAGS_CHANGED: CGEventType = 12;
const KCG_EVENT_SCROLL_WHEEL: CGEventType = 22;
const KCG_EVENT_OTHER_MOUSE_DOWN: CGEventType = 25;
const KCG_EVENT_OTHER_MOUSE_UP: CGEventType = 26;
const KCG_EVENT_OTHER_MOUSE_DRAGGED: CGEventType = 27;
const KCG_EVENT_TAP_DISABLED_BY_TIMEOUT: CGEventType = 0xFFFF_FFFE;
const KCG_EVENT_TAP_DISABLED_BY_USER_INPUT: CGEventType = 0xFFFF_FFFF;

const KCG_MOUSE_EVENT_BUTTON_NUMBER: CGEventField = 3;
const KCG_KEYBOARD_EVENT_AUTOREPEAT: CGEventField = 8;
const KCG_KEYBOARD_EVENT_KEYCODE: CGEventField = 9;
const KCG_SCROLL_WHEEL_EVENT_DELTA_AXIS_1: CGEventField = 11;
const KCG_SCROLL_WHEEL_EVENT_DELTA_AXIS_2: CGEventField = 12;
const KCG_SCROLL_WHEEL_EVENT_IS_CONTINUOUS: CGEventField = 88;
const KCG_SCROLL_WHEEL_EVENT_POINT_DELTA_AXIS_1: CGEventField = 96;
const KCG_SCROLL_WHEEL_EVENT_POINT_DELTA_AXIS_2: CGEventField = 97;

/// An absolute point in the global display coordinate space.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CGPoint {
    /// Horizontal coordinate.
    x: f64,
    /// Vertical coordinate.
    y: f64,
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    /// Creates a tap and returns a reference the caller releases.
    fn CGEventTapCreate(
        tap: CGEventTapLocation,
        place: CGEventTapPlacement,
        options: CGEventTapOptions,
        events_of_interest: CGEventMask,
        callback: CGEventTapCallBack,
        user_info: *mut c_void,
    ) -> CFMachPortRef;
    /// Enables or disables a tap.
    fn CGEventTapEnable(tap: CFMachPortRef, enable: bool);
    /// Reads an integer field from an event.
    fn CGEventGetIntegerValueField(event: CGEventRef, field: CGEventField) -> i64;
    /// Reads the modifier and lock flags of an event.
    fn CGEventGetFlags(event: CGEventRef) -> CGEventFlags;
    /// Reads the global position of a pointer event.
    fn CGEventGetLocation(event: CGEventRef) -> CGPoint;
    /// Whether the process may install a listen-only tap.
    fn CGPreflightListenEventAccess() -> bool;
    /// Asks the user to grant listen-only tap access.
    fn CGRequestListenEventAccess() -> bool;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    /// Wraps a mach port in a run loop source.
    fn CFMachPortCreateRunLoopSource(
        allocator: CFAllocatorRef,
        port: CFMachPortRef,
        order: CFIndex,
    ) -> CFRunLoopSourceRef;
    /// The run loop of the calling thread.
    fn CFRunLoopGetCurrent() -> CFRunLoopRef;
    /// Attaches a source to a run loop mode.
    fn CFRunLoopAddSource(run_loop: CFRunLoopRef, source: CFRunLoopSourceRef, mode: CFStringRef);
    /// Detaches a source from a run loop mode.
    fn CFRunLoopRemoveSource(run_loop: CFRunLoopRef, source: CFRunLoopSourceRef, mode: CFStringRef);
    /// Runs a run loop until a source is handled or the timeout expires.
    fn CFRunLoopRunInMode(mode: CFStringRef, seconds: f64, return_after_source_handled: bool) -> i32;
    /// Releases a CoreFoundation reference the caller owns.
    fn CFRelease(reference: *const c_void);
    /// The mode that every common mode includes.
    static kCFRunLoopCommonModes: CFStringRef;
    /// The default run loop mode.
    static kCFRunLoopDefaultMode: CFStringRef;
}
