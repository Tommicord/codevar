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

//! DOM listeners attached while a class is registered.
//!
//! Every closure is kept in [`Listeners`] for as long as the class is open.
//! Dropping the value removes the listeners, which is what
//! [`Backend::stop`](codevar_kuitui_control::Backend::stop) relies on to leave
//! the page exactly as it found it.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use codevar_kuitui_control::{
    ControlError, DeviceClass, DeviceId, DeviceInfo, EventSink, InputEvent, KeyEvent, KeyState, Modifiers,
    Point, PointerButton, PointerEvent, Timestamp,
};
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use web_sys::{
    Event, EventTarget, GamepadEvent as DomGamepadEvent, KeyboardEvent, PointerEvent as DomPointerEvent,
    WheelEvent,
};

use crate::keymap;

/// Identifier reported for the browser's keyboard.
const KEYBOARD_DEVICE: DeviceId = DeviceId::new(1);
/// First identifier assigned to a gamepad, kept clear of the fixed devices.
const GAMEPAD_ID_BASE: u64 = 1 << 16;

/// The listeners of one registered class.
pub(crate) struct Listeners {
    /// Object the listeners were added to.
    target: EventTarget,
    /// Event name and closure of every attached listener.
    entries: Vec<(&'static str, Closure<dyn FnMut(Event)>)>,
}

impl Listeners {
    /// Attaches the listeners `class` needs to the window.
    pub(crate) fn attach(
        class: DeviceClass,
        devices: Rc<RefCell<Vec<DeviceInfo>>>,
        sink: EventSink,
    ) -> Result<Self, ControlError> {
        let window =
            web_sys::window().ok_or_else(|| ControlError::backend("the page has no window object"))?;
        let mut listeners = Self {
            target: window.unchecked_into(),
            entries: Vec::new(),
        };
        match class {
            DeviceClass::Keyboard => listeners.keyboard(sink)?,
            DeviceClass::Pointer => listeners.pointer(sink)?,
            DeviceClass::Gamepad => listeners.gamepad(devices, sink)?,
            _ => return Err(ControlError::Unsupported(class)),
        }
        Ok(listeners)
    }

    /// Listens for key transitions.
    fn keyboard(&mut self, sink: EventSink) -> Result<(), ControlError> {
        let down = sink.clone();
        self.on("keydown", move |event| {
            let Some(event) = event.dyn_ref::<KeyboardEvent>() else {
                return;
            };
            let state = if event.repeat() {
                KeyState::Repeat
            } else {
                KeyState::Pressed
            };
            down.emit_at(
                timestamp(event),
                InputEvent::Key(key_event(KEYBOARD_DEVICE, event, state)),
            );
        })?;
        self.on("keyup", move |event| {
            let Some(event) = event.dyn_ref::<KeyboardEvent>() else {
                return;
            };
            sink.emit_at(
                timestamp(event),
                InputEvent::Key(key_event(KEYBOARD_DEVICE, event, KeyState::Released)),
            );
        })
    }

    /// Listens for pointer motion, buttons, wheel and lifecycle changes.
    fn pointer(&mut self, sink: EventSink) -> Result<(), ControlError> {
        let moved = sink.clone();
        self.on("pointermove", move |event| {
            let Some(event) = event.dyn_ref::<DomPointerEvent>() else {
                return;
            };
            moved.emit_at(
                timestamp(event),
                InputEvent::Pointer(PointerEvent::Motion {
                    position: Some(position(event)),
                    delta: Some(Point::new(event.movement_x() as f32, event.movement_y() as f32)),
                }),
            );
        })?;

        let pressed = sink.clone();
        self.on("pointerdown", move |event| {
            let Some(event) = event.dyn_ref::<DomPointerEvent>() else {
                return;
            };
            pressed.emit_at(
                timestamp(event),
                InputEvent::Pointer(button_event(event, KeyState::Pressed)),
            );
        })?;

        let released = sink.clone();
        self.on("pointerup", move |event| {
            let Some(event) = event.dyn_ref::<DomPointerEvent>() else {
                return;
            };
            released.emit_at(
                timestamp(event),
                InputEvent::Pointer(button_event(event, KeyState::Released)),
            );
        })?;

        let entered = sink.clone();
        self.on("pointerenter", move |event| {
            let Some(event) = event.dyn_ref::<DomPointerEvent>() else {
                return;
            };
            entered.emit_at(
                timestamp(event),
                InputEvent::Pointer(PointerEvent::Enter {
                    position: Some(position(event)),
                }),
            );
        })?;

        let left = sink.clone();
        self.on("pointerleave", move |_event| {
            left.emit(InputEvent::Pointer(PointerEvent::Leave));
        })?;

        let cancelled = sink.clone();
        self.on("pointercancel", move |_event| {
            cancelled.emit(InputEvent::Pointer(PointerEvent::Cancel));
        })?;

        self.on("wheel", move |event| {
            let Some(event) = event.dyn_ref::<WheelEvent>() else {
                return;
            };
            sink.emit_at(
                timestamp(event),
                InputEvent::Pointer(PointerEvent::Scroll {
                    delta: Point::new(event.delta_x() as f32, event.delta_y() as f32),
                    position: Some(position(event)),
                    modifiers: mouse_modifiers(event),
                }),
            );
        })
    }

    /// Listens for gamepads connecting and disconnecting.
    fn gamepad(
        &mut self,
        devices: Rc<RefCell<Vec<DeviceInfo>>>,
        sink: EventSink,
    ) -> Result<(), ControlError> {
        let connected = sink.clone();
        let connected_devices = Rc::clone(&devices);
        self.on("gamepadconnected", move |event| {
            let Some(event) = event.dyn_ref::<DomGamepadEvent>() else {
                return;
            };
            let Some(gamepad) = event.gamepad() else {
                return;
            };
            let info = gamepad_info(&gamepad);
            let mut known = connected_devices.borrow_mut();
            known.retain(|device| device.id != info.id);
            known.push(info.clone());
            drop(known);
            connected.emit(InputEvent::DeviceAdded(info));
        })?;

        self.on("gamepaddisconnected", move |event| {
            let Some(event) = event.dyn_ref::<DomGamepadEvent>() else {
                return;
            };
            let Some(gamepad) = event.gamepad() else {
                return;
            };
            let id = gamepad_id(gamepad.index());
            devices
                .borrow_mut()
                .retain(|device| device.id != id);
            sink.emit(InputEvent::DeviceRemoved(id));
        })
    }

    /// Adds one listener and keeps its closure alive.
    fn on(&mut self, name: &'static str, handler: impl FnMut(Event) + 'static) -> Result<(), ControlError> {
        let closure = Closure::<dyn FnMut(Event)>::new(handler);
        self.target
            .add_event_listener_with_callback(name, closure.as_ref().unchecked_ref())
            .map_err(|error| ControlError::backend(describe(&error)))?;
        self.entries.push((name, closure));
        Ok(())
    }
}

impl Drop for Listeners {
    fn drop(&mut self) {
        for (name, closure) in &self.entries {
            let callback: &js_sys::Function = closure.as_ref().unchecked_ref();
            let _ = self
                .target
                .remove_event_listener_with_callback(name, callback);
        }
    }
}

/// Turns a key transition into the normalized event.
fn key_event(device: DeviceId, event: &KeyboardEvent, state: KeyState) -> KeyEvent {
    KeyEvent {
        device,
        code: keymap::key_code(event.code().as_str()),
        state,
        modifiers: modifiers(event),
    }
}

/// Reads the modifier and lock state the browser reports for a key event.
fn modifiers(event: &KeyboardEvent) -> Modifiers {
    let mut modifiers = Modifiers::NONE;
    if event.shift_key() {
        modifiers.insert(Modifiers::SHIFT);
    }
    if event.ctrl_key() {
        modifiers.insert(Modifiers::CONTROL);
    }
    if event.alt_key() {
        modifiers.insert(Modifiers::ALT);
    }
    if event.meta_key() {
        modifiers.insert(Modifiers::META);
    }
    if event.get_modifier_state("CapsLock") {
        modifiers.insert(Modifiers::CAPS_LOCK);
    }
    if event.get_modifier_state("NumLock") {
        modifiers.insert(Modifiers::NUM_LOCK);
    }
    modifiers
}

/// Reads the modifier state of a mouse or wheel event.
fn mouse_modifiers(event: &web_sys::MouseEvent) -> Modifiers {
    let mut modifiers = Modifiers::NONE;
    if event.shift_key() {
        modifiers.insert(Modifiers::SHIFT);
    }
    if event.ctrl_key() {
        modifiers.insert(Modifiers::CONTROL);
    }
    if event.alt_key() {
        modifiers.insert(Modifiers::ALT);
    }
    if event.meta_key() {
        modifiers.insert(Modifiers::META);
    }
    modifiers
}

/// The pointer position in the page's CSS pixels.
fn position(event: &web_sys::MouseEvent) -> Point {
    Point::new(event.client_x() as f32, event.client_y() as f32)
}

/// Wraps a pointer button transition.
fn button_event(event: &DomPointerEvent, state: KeyState) -> PointerEvent {
    PointerEvent::Button {
        button: pointer_button(event.button()),
        state,
        position: Some(position(event)),
        modifiers: mouse_modifiers(event),
    }
}

/// Maps the DOM button index to the portable button.
fn pointer_button(button: i16) -> PointerButton {
    match button {
        0 => PointerButton::Primary,
        1 => PointerButton::Middle,
        2 => PointerButton::Secondary,
        3 => PointerButton::Back,
        4 => PointerButton::Forward,
        other => PointerButton::Other(u16::try_from(other).unwrap_or_default()),
    }
}

/// The identifier of a gamepad slot.
fn gamepad_id(index: u32) -> DeviceId {
    DeviceId::new(GAMEPAD_ID_BASE.saturating_add(u64::from(index)))
}

/// Describes a connected gamepad.
fn gamepad_info(gamepad: &web_sys::Gamepad) -> DeviceInfo {
    DeviceInfo::new(gamepad_id(gamepad.index()), DeviceClass::Gamepad, gamepad.id())
}

/// The browser's event time in microseconds.
fn timestamp(event: &Event) -> Timestamp {
    let millis = event.time_stamp().max(0.0);
    Timestamp::from_micros((millis * 1_000.0) as u64)
}

/// Explains why the browser rejected a listener.
fn describe(error: &wasm_bindgen::JsValue) -> String {
    error
        .as_string()
        .unwrap_or_else(|| String::from("the browser rejected the event listener"))
}
