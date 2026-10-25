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

//! Input state and the handlers a widget implements.
//!
//! The operating system and browser code lives in `codevar-kuitui-control`:
//! it owns the devices and publishes [`InputEvent`]s through a reactive
//! [`Subscription`](codevar_kuitui_control::Subscription). This module is the
//! layer above it. It keeps the state a widget asks about — which keys are
//! down, where the pointer is, how loud the microphone is — in a small set of
//! global states, and it defines the traits a widget implements to receive
//! those events.
//!
//! # No observer list
//!
//! A handler is not registered with the event source. The application owns one
//! `InputPump`, which drains the subscription and updates the global states; a
//! widget is then handed the events of the frame through its handler trait.
//! The pump never calls consumer code from the backend thread, and a widget
//! that is not part of the current frame simply is not called.
//!
//! # A permanent state reference
//!
//! The three device traits share the same shape: a required method returns a
//! `&'static` reference to the state of that device, and an event hook
//! receives the event after the state has been updated. The reference is
//! permanent — it points into a `static`, never into the handler — so a
//! handler can be constructed once and survive recomposition, and so two
//! handlers reading the same state always agree.
//!
//! The states are read with atomic loads and updated by a single writer (the
//! pump) under a sequence counter, so a reader in another thread — a backend
//! worker, an audio callback, a profiler — never sees a half-updated
//! snapshot and never takes a lock.
//!
//! # The three capabilities
//!
//! | Trait | State | Hook |
//! | --- | --- | --- |
//! | `KeyboardHandler` | `KeyboardState` | `on_key` |
//! | `MouseHandler` | `PointerState` | `on_pointer` |
//! | `MicrophoneHandler` | `MicrophoneState` | `on_audio` |
//!
//! Every one of them extends [`Input`], the base trait that receives events
//! which carry no device-specific state (text, gamepads, device changes).

use codevar_kuitui_control::InputEvent;
pub use codevar_kuitui_control::{DeviceId, KeyCode, KeyState, Timestamp};

use crate::event::EventResult;

pub mod keyboard;

pub use keyboard::{
    KEYBOARD, KeyTransition, KeyboardBinding, KeyboardHandler, KeyboardModifiers, KeyboardSnapshot,
    KeyboardState,
};

/// A widget that consumes normalized input.
///
/// This is the base capability. Implementing it directly makes a widget
/// receive the events that do not belong to a keyboard, a pointer or a
/// microphone; a widget that reacts to a device implements the corresponding
/// device trait instead, which extends this one.
pub trait Input {
    /// Receives one event that no device-specific handler claimed.
    ///
    /// The global states have already been updated when this is called. The
    /// default ignores the event; returning [`EventResult::Consumed`] stops
    /// delivery to the other handlers of the frame.
    fn on_input(&mut self, event: &InputEvent, timestamp: Timestamp) -> EventResult {
        let _ = (event, timestamp);
        EventResult::Ignored
    }
}
