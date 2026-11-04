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

//! The key hook a widget implements and the permanent state reference it
//! holds.
//!
//! [`KeyboardHandler`] is the only trait a widget needs for key input: it
//! returns the `'static` reference it was built with and receives one
//! [`KeyTransition`] after the state has already been updated, so a handler
//! that reads [`KeyboardState::held_count`] sees the press it was just told
//! about.

use codevar_kuitui_control::{DeviceId, KeyCode, KeyEvent, KeyState, Timestamp};

use crate::event::EventResult;
use crate::input::Input;

use super::{KeyboardModifiers, KeyboardSnapshot, KeyboardState};

/// One key transition as a handler receives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyTransition {
    /// When the backend observed the transition.
    pub timestamp: Timestamp,
    /// The keyboard that reported it.
    pub device: DeviceId,
    /// The normalized key.
    pub code: KeyCode,
    /// Down, up or repeat.
    pub state: KeyState,
    /// The modifiers held after the transition.
    pub modifiers: KeyboardModifiers,
}

impl KeyTransition {
    /// Builds a transition from a control-layer event.
    #[must_use]
    pub fn new(event: &KeyEvent, timestamp: Timestamp) -> Self {
        Self {
            timestamp,
            device: event.device,
            code: event.code,
            state: event.state,
            modifiers: KeyboardModifiers::from(event.modifiers),
        }
    }
}

/// A widget that reacts to the keyboard.
///
/// The trait extends [`Input`] and adds the two pieces a keyboard consumer
/// needs: a permanent reference to the shared state, and the key hook. The
/// reference is obtained once, at construction, and never changes.
pub trait KeyboardHandler: Input {
    /// The keyboard state this handler observes.
    ///
    /// The reference is `'static`: it points at [`KEYBOARD`] or at another
    /// static of the application, so it outlives the handler and survives
    /// recomposition.
    fn keyboard(&self) -> &'static KeyboardState;

    /// Receives one key transition after the state has been updated.
    fn on_key(&mut self, transition: &KeyTransition) -> EventResult {
        let _ = transition;
        EventResult::Ignored
    }
}

/// A convenience holder for the permanent keyboard reference.
///
/// A widget that implements [`KeyboardHandler`] can keep one of these and
/// return its state from the trait method.
pub struct KeyboardBinding {
    /// The state the binding reads.
    state: &'static KeyboardState,
}

impl KeyboardBinding {
    /// Binds to `state`.
    #[must_use]
    pub const fn new(state: &'static KeyboardState) -> Self {
        Self { state }
    }

    /// Binds to the process-wide [`KEYBOARD`] state.
    #[must_use]
    pub const fn global() -> Self {
        Self::new(&KEYBOARD)
    }

    /// The bound state.
    #[must_use]
    pub const fn state(&self) -> &'static KeyboardState {
        self.state
    }

    /// A consistent snapshot of the bound state.
    #[must_use]
    pub fn snapshot(&self) -> KeyboardSnapshot {
        self.state.snapshot()
    }

    /// Whether `code` is held.
    #[must_use]
    pub fn is_pressed(&self, code: KeyCode) -> bool {
        self.state.is_pressed(code)
    }

    /// The held modifiers and active locks.
    #[must_use]
    pub fn modifiers(&self) -> KeyboardModifiers {
        self.state.modifiers()
    }

    /// Whether `code` is held together with every flag of `required`.
    #[must_use]
    pub fn is_shortcut(&self, code: KeyCode, required: KeyboardModifiers) -> bool {
        self.modifiers().contains(required) && self.is_pressed(code)
    }
}

/// The keyboard state of the process.
///
/// The `InputPump` writes it; handlers bind
/// to it with [`KeyboardBinding::global`].
pub static KEYBOARD: KeyboardState = KeyboardState::new();
