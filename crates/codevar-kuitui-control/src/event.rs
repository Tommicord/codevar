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

//! The normalized input event model shared by every backend.
//!
//! Backends translate their native records into these types once, at the
//! edge of the crate. Consumers therefore never depend on an evdev key code,
//! a Windows virtual-key value or a CoreGraphics event type.

use alloc::string::String;
use alloc::vec::Vec;

use crate::class::DeviceClass;
use crate::device::{DeviceId, DeviceInfo};

/// A monotonic timestamp in microseconds.
///
/// The origin is backend-defined (usually system boot, or
/// [`std::time::Instant`] on hosted targets) and must not be compared across
/// backends. [`Timestamp::UNSPECIFIED`] marks an event whose backend could
/// not timestamp it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timestamp(u64);

impl Timestamp {
    /// An event with no usable timestamp.
    pub const UNSPECIFIED: Timestamp = Timestamp(0);

    /// A timestamp of `micros` microseconds.
    #[must_use]
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros)
    }

    /// The timestamp in microseconds.
    #[must_use]
    pub const fn micros(self) -> u64 {
        self.0
    }

    /// Whether the backend provided a timestamp.
    #[must_use]
    pub const fn is_specified(self) -> bool {
        self.0 != 0
    }
}

/// An event together with the time the device reported it.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// When the backend observed the event.
    pub timestamp: Timestamp,
    /// The normalized event.
    pub payload: InputEvent,
}

impl Event {
    /// Pairs a payload with a timestamp.
    #[must_use]
    pub const fn new(timestamp: Timestamp, payload: InputEvent) -> Self {
        Self { timestamp, payload }
    }

    /// Wraps a payload whose backend has no timestamp.
    #[must_use]
    pub const fn untimed(payload: InputEvent) -> Self {
        Self {
            timestamp: Timestamp::UNSPECIFIED,
            payload,
        }
    }

    /// The class whose subscribers receive this event, or `None` when the
    /// event is a device removal that cannot be attributed to one class.
    #[must_use]
    pub fn class(&self) -> Option<DeviceClass> {
        self.payload.class()
    }
}

/// A normalized input event.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum InputEvent {
    /// A key transition on a keyboard.
    Key(KeyEvent),
    /// Text produced by the active keymap or input method.
    Text(TextEvent),
    /// Pointer motion, button, scroll or lifecycle event.
    Pointer(PointerEvent),
    /// A gamepad button or axis change.
    Gamepad(GamepadEvent),
    /// Captured audio samples from a microphone.
    Audio(AudioEvent),
    /// A device became available.
    DeviceAdded(DeviceInfo),
    /// A device disappeared.
    DeviceRemoved(DeviceId),
}

impl InputEvent {
    /// The class whose subscribers receive this event.
    #[must_use]
    pub fn class(&self) -> Option<DeviceClass> {
        match self {
            InputEvent::Key(_) | InputEvent::Text(_) => Some(DeviceClass::Keyboard),
            InputEvent::Pointer(_) => Some(DeviceClass::Pointer),
            InputEvent::Gamepad(_) => Some(DeviceClass::Gamepad),
            InputEvent::Audio(_) => Some(DeviceClass::Microphone),
            InputEvent::DeviceAdded(info) => Some(info.class),
            InputEvent::DeviceRemoved(_) => None,
        }
    }
}

/// Whether a key or button is down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyState {
    /// The key went down.
    Pressed,
    /// The key went up.
    Released,
    /// The key is held and the platform repeated it.
    Repeat,
}

/// A key on a keyboard.
///
/// The names follow the web `KeyboardEvent.code` values, so the WASM backend
/// needs no translation table and the desktop backends can share one. A key
/// the table does not cover keeps the backend's native code in
/// [`KeyCode::Native`] instead of being discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum KeyCode {
    /// The backend could not identify the key.
    Unidentified,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyA,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyB,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyC,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyD,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyE,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyF,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyG,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyH,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyI,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyJ,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyK,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyL,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyM,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyN,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyO,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyP,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyQ,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyR,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyS,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyT,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyU,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyV,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyW,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyX,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyY,
    /// A letter key, `KeyA` through `KeyZ`.
    KeyZ,
    /// A digit key, `Digit0` through `Digit9`.
    Digit0,
    /// A digit key, `Digit0` through `Digit9`.
    Digit1,
    /// A digit key, `Digit0` through `Digit9`.
    Digit2,
    /// A digit key, `Digit0` through `Digit9`.
    Digit3,
    /// A digit key, `Digit0` through `Digit9`.
    Digit4,
    /// A digit key, `Digit0` through `Digit9`.
    Digit5,
    /// A digit key, `Digit0` through `Digit9`.
    Digit6,
    /// A digit key, `Digit0` through `Digit9`.
    Digit7,
    /// A digit key, `Digit0` through `Digit9`.
    Digit8,
    /// A digit key, `Digit0` through `Digit9`.
    Digit9,
    /// The backquote key.
    Backquote,
    /// The minus key.
    Minus,
    /// The equals key.
    Equal,
    /// The left bracket key.
    BracketLeft,
    /// The right bracket key.
    BracketRight,
    /// The backslash key.
    Backslash,
    /// The semicolon key.
    Semicolon,
    /// The quote key.
    Quote,
    /// The comma key.
    Comma,
    /// The period key.
    Period,
    /// The slash key.
    Slash,
    /// The non-US backslash key.
    IntlBackslash,
    /// The Japanese Ro key.
    IntlRo,
    /// The Japanese Yen key.
    IntlYen,
    /// The escape key.
    Escape,
    /// The tab key.
    Tab,
    /// The caps lock key.
    CapsLock,
    /// The space bar.
    Space,
    /// The enter key.
    Enter,
    /// The backspace key.
    Backspace,
    /// The forward delete key.
    Delete,
    /// The insert key.
    Insert,
    /// The home key.
    Home,
    /// The end key.
    End,
    /// The page up key.
    PageUp,
    /// The page down key.
    PageDown,
    /// The up arrow key.
    ArrowUp,
    /// The down arrow key.
    ArrowDown,
    /// The left arrow key.
    ArrowLeft,
    /// The right arrow key.
    ArrowRight,
    /// The left shift key.
    ShiftLeft,
    /// The right shift key.
    ShiftRight,
    /// The left control key.
    ControlLeft,
    /// The right control key.
    ControlRight,
    /// The left alt or option key.
    AltLeft,
    /// The right alt or option key.
    AltRight,
    /// The left meta, command or Windows key.
    MetaLeft,
    /// The right meta, command or Windows key.
    MetaRight,
    /// The context menu key.
    ContextMenu,
    /// The print screen key.
    PrintScreen,
    /// The scroll lock key.
    ScrollLock,
    /// The pause key.
    Pause,
    /// The function key.
    Fn,
    /// The num lock key.
    NumLock,
    /// The keypad divide key.
    NumpadDivide,
    /// The keypad multiply key.
    NumpadMultiply,
    /// The keypad subtract key.
    NumpadSubtract,
    /// The keypad add key.
    NumpadAdd,
    /// The keypad enter key.
    NumpadEnter,
    /// The keypad decimal separator.
    NumpadDecimal,
    /// A keypad digit, `Numpad0` through `Numpad9`.
    Numpad0,
    /// A keypad digit, `Numpad0` through `Numpad9`.
    Numpad1,
    /// A keypad digit, `Numpad0` through `Numpad9`.
    Numpad2,
    /// A keypad digit, `Numpad0` through `Numpad9`.
    Numpad3,
    /// A keypad digit, `Numpad0` through `Numpad9`.
    Numpad4,
    /// A keypad digit, `Numpad0` through `Numpad9`.
    Numpad5,
    /// A keypad digit, `Numpad0` through `Numpad9`.
    Numpad6,
    /// A keypad digit, `Numpad0` through `Numpad9`.
    Numpad7,
    /// A keypad digit, `Numpad0` through `Numpad9`.
    Numpad8,
    /// A keypad digit, `Numpad0` through `Numpad9`.
    Numpad9,
    /// A function key, `F1` through `F12`.
    F1,
    /// A function key, `F1` through `F12`.
    F2,
    /// A function key, `F1` through `F12`.
    F3,
    /// A function key, `F1` through `F12`.
    F4,
    /// A function key, `F1` through `F12`.
    F5,
    /// A function key, `F1` through `F12`.
    F6,
    /// A function key, `F1` through `F12`.
    F7,
    /// A function key, `F1` through `F12`.
    F8,
    /// A function key, `F1` through `F12`.
    F9,
    /// A function key, `F1` through `F12`.
    F10,
    /// A function key, `F1` through `F12`.
    F11,
    /// A function key, `F1` through `F12`.
    F12,
    /// A backend-specific key code with no normalized name.
    Native(u32),
}

/// A modifier-key state.
///
/// The set is a bitset so adding a modifier does not change the size of every
/// key event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Modifiers(u8);

impl Modifiers {
    /// No modifier is down.
    pub const NONE: Modifiers = Modifiers(0);
    /// Shift is down.
    pub const SHIFT: Modifiers = Modifiers(1 << 0);
    /// Control is down.
    pub const CONTROL: Modifiers = Modifiers(1 << 1);
    /// Alt or option is down.
    pub const ALT: Modifiers = Modifiers(1 << 2);
    /// Meta, command or the Windows key is down.
    pub const META: Modifiers = Modifiers(1 << 3);
    /// Caps lock is active.
    pub const CAPS_LOCK: Modifiers = Modifiers(1 << 4);
    /// Num lock is active.
    pub const NUM_LOCK: Modifiers = Modifiers(1 << 5);

    /// Whether every modifier of `other` is set.
    #[must_use]
    pub const fn contains(self, other: Modifiers) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no modifier is set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Adds `other` to the set.
    pub fn insert(&mut self, other: Modifiers) {
        self.0 |= other.0;
    }

    /// Removes `other` from the set.
    pub fn remove(&mut self, other: Modifiers) {
        self.0 &= !other.0;
    }
}

impl core::ops::BitOr for Modifiers {
    type Output = Modifiers;

    fn bitor(self, rhs: Modifiers) -> Modifiers {
        Modifiers(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for Modifiers {
    fn bitor_assign(&mut self, rhs: Modifiers) {
        self.0 |= rhs.0;
    }
}

/// A key transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyEvent {
    /// The keyboard that reported the transition.
    pub device: DeviceId,
    /// The normalized key.
    pub code: KeyCode,
    /// Whether the key went down, up or repeated.
    pub state: KeyState,
    /// The modifiers held after the transition.
    pub modifiers: Modifiers,
}

/// Text produced by the active keymap or input method.
///
/// A key event reports the physical key; a text event reports what the user
/// actually composed. Input methods may emit a text event without a matching
/// key event, and repeated composition updates are ordinary text events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextEvent {
    /// The keyboard that produced the text.
    pub device: DeviceId,
    /// The UTF-8 text.
    pub text: String,
}

/// A point in the coordinate space the backend reports.
///
/// The origin and unit belong to the backend: a window-relative CSS pixel for
/// the browser, a global desktop pixel for Windows and macOS, and a relative
/// delta for evdev. Absolute positions are only comparable within one
/// backend.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: f32,
    /// Vertical coordinate.
    pub y: f32,
}

impl Point {
    /// Creates a point.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// The zero point.
    pub const ORIGIN: Point = Point::new(0.0, 0.0);
}

/// A pointer button.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PointerButton {
    /// The primary button, usually the left one.
    Primary,
    /// The secondary button, usually the right one.
    Secondary,
    /// The middle button.
    Middle,
    /// The back button.
    Back,
    /// The forward button.
    Forward,
    /// A button with a platform-specific index.
    Other(u16),
}

/// A pointer event.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PointerEvent {
    /// The pointer moved.
    Motion {
        /// Absolute position when the backend reports one.
        position: Option<Point>,
        /// Movement since the previous event when the backend reports it.
        delta: Option<Point>,
    },
    /// A pointer button changed state.
    Button {
        /// The button that changed.
        button: PointerButton,
        /// The new state.
        state: KeyState,
        /// Absolute position when the backend reports one.
        position: Option<Point>,
        /// Modifiers held when the button changed.
        modifiers: Modifiers,
    },
    /// The pointer scrolled.
    Scroll {
        /// Horizontal and vertical scroll in backend-defined units.
        delta: Point,
        /// Absolute position when the backend reports one.
        position: Option<Point>,
        /// Modifiers held while scrolling.
        modifiers: Modifiers,
    },
    /// The pointer entered the application's input region.
    Enter {
        /// Absolute position when the backend reports one.
        position: Option<Point>,
    },
    /// The pointer left the application's input region.
    Leave,
    /// The platform cancelled the pointer stream, for example on palm
    /// rejection.
    Cancel,
}

/// A gamepad event.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum GamepadEvent {
    /// A button changed state.
    Button {
        /// The gamepad that reported the change.
        device: DeviceId,
        /// The device-specific button index.
        button: u16,
        /// The new state.
        state: KeyState,
    },
    /// An axis moved.
    Axis {
        /// The gamepad that reported the change.
        device: DeviceId,
        /// The device-specific axis index.
        axis: u16,
        /// The normalized value in `-1.0..=1.0`.
        value: f32,
    },
}

/// The sample format of an [`AudioEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    /// Unsigned 8-bit samples.
    U8,
    /// Signed 16-bit little-endian samples.
    I16,
    /// 32-bit little-endian floating point samples.
    F32,
}

/// The layout of captured audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioFormat {
    /// Samples per second per channel.
    pub sample_rate: u32,
    /// Number of interleaved channels.
    pub channels: u16,
    /// How each sample is encoded.
    pub sample_format: SampleFormat,
}

/// A block of captured microphone samples.
///
/// The samples are raw bytes so a backend can move its capture buffer into
/// the event without an intermediate conversion. [`AudioFormat`] says how to
/// interpret them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioEvent {
    /// The microphone that captured the block.
    pub device: DeviceId,
    /// The sample layout.
    pub format: AudioFormat,
    /// Interleaved sample bytes.
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use super::{Event, InputEvent, KeyCode, KeyEvent, KeyState, Modifiers, Point, PointerEvent, Timestamp};
    use crate::class::DeviceClass;
    use crate::device::{DeviceId, DeviceInfo};

    fn key() -> InputEvent {
        InputEvent::Key(KeyEvent {
            device: DeviceId::new(1),
            code: KeyCode::KeyA,
            state: KeyState::Pressed,
            modifiers: Modifiers::SHIFT,
        })
    }

    #[test]
    fn payloads_route_to_their_class() {
        assert_eq!(key().class(), Some(DeviceClass::Keyboard));
        assert_eq!(
            InputEvent::Pointer(PointerEvent::Motion {
                position: Some(Point::new(3.0, 4.0)),
                delta: None,
            })
            .class(),
            Some(DeviceClass::Pointer)
        );
        assert_eq!(
            InputEvent::DeviceAdded(DeviceInfo::new(
                DeviceId::new(2),
                DeviceClass::Gamepad,
                String::from("pad")
            ))
            .class(),
            Some(DeviceClass::Gamepad)
        );
        assert_eq!(InputEvent::DeviceRemoved(DeviceId::new(2)).class(), None);
    }

    #[test]
    fn timestamps_distinguish_unspecified_from_zero_time() {
        assert!(!Timestamp::UNSPECIFIED.is_specified());
        assert!(Timestamp::from_micros(1).is_specified());
        assert_eq!(Timestamp::from_micros(9).micros(), 9);
        let event = Event::untimed(key());
        assert_eq!(event.timestamp, Timestamp::UNSPECIFIED);
    }

    #[test]
    fn modifiers_are_a_set() {
        let modifiers = Modifiers::SHIFT | Modifiers::CONTROL;
        assert!(modifiers.contains(Modifiers::SHIFT));
        assert!(modifiers.contains(Modifiers::CONTROL));
        assert!(!modifiers.contains(Modifiers::ALT));
        assert!(!modifiers.is_empty());
    }
}
