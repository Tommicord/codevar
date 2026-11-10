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

//! The state of a window decoration.
//!
//! A decoration reports which modifiers are active and which ones it is
//! allowed to change. The backend keeps the two in sync with the native
//! windowing system.

use crate::modifiers::{DecorationModifier, DecorationModifiers};

/// The complete decoration state of a window: what is active and what
/// may be changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DecorationState {
    /// The modifiers that are currently active.
    pub active: DecorationModifiers,
    /// The modifiers the backend is allowed to change.
    pub supported: DecorationModifiers,
}

impl DecorationState {
    /// Creates a state with no active and no supported modifier.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active: DecorationModifiers::NONE,
            supported: DecorationModifiers::NONE,
        }
    }

    /// Returns `true` when `modifier` is active.
    #[must_use]
    pub const fn is_active(self, modifier: DecorationModifier) -> bool {
        self.active.has(modifier)
    }

    /// Returns `true` when `modifier` is supported by the backend.
    #[must_use]
    pub const fn is_supported(self, modifier: DecorationModifier) -> bool {
        self.supported.has(modifier)
    }

    /// Returns the modifiers that are both active and supported.
    #[must_use]
    pub const fn active_and_supported(self) -> DecorationModifiers {
        DecorationModifiers::from_bits(self.active.bits() & self.supported.bits())
    }

    /// Returns the modifiers that are supported but not active.
    #[must_use]
    pub const fn available(self) -> DecorationModifiers {
        DecorationModifiers::from_bits(self.supported.bits() & !self.active.bits())
    }
}

/// A collection of decoration states, one per decorated window.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct DecorationStates {
    /// The state of every window, keyed by its decoration id.
    states: alloc::vec::Vec<(crate::types::DecorationId, DecorationState)>,
}

impl DecorationStates {
    /// Creates an empty collection.
    #[must_use]
    pub const fn new() -> Self {
        Self { states: alloc::vec::Vec::new() }
    }

    /// The state of the window identified by `id`, if any.
    #[must_use]
    pub fn state_of(&self, id: crate::types::DecorationId) -> Option<DecorationState> {
        self.states
            .iter()
            .find(|(window_id, _)| *window_id == id)
            .map(|(_, state)| *state)
    }

    /// Replaces or inserts the state of `id`.
    pub fn set(&mut self, id: crate::types::DecorationId, state: DecorationState) {
        if let Some(entry) = self
            .states
            .iter_mut()
            .find(|(window_id, _)| *window_id == id)
        {
            entry.1 = state;
        } else {
            self.states.push((id, state));
        }
    }

    /// Removes the state of `id`.
    pub fn remove(&mut self, id: crate::types::DecorationId) {
        if let Some(index) = self.states.iter().position(|(window_id, _)| *window_id == id) {
            self.states.swap_remove(index);
        }
    }

    /// Iterates over the states of every window.
    #[must_use]
    pub fn iter(&self) -> DecorationStateIter {
        DecorationStateIter {
            states: &self.states,
            index: 0,
        }
    }
}

/// An iterator over the decoration states of every window.
#[derive(Debug, Clone)]
pub struct DecorationStateIter<'a> {
    states: &'a [(crate::types::DecorationId, DecorationState)],
    index: usize,
}

impl<'a> Iterator for DecorationStateIter<'a> {
    type Item = (crate::types::DecorationId, DecorationState);

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.states.get(self.index)?;
        self.index += 1;
        Some(*entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_reports_active_and_supported_modifiers() {
        let state = DecorationState {
            active: DecorationModifier::Maximized.into(),
            supported: DecorationModifier::Maximized | DecorationModifier::Resizable,
        };
        assert!(state.is_active(DecorationModifier::Maximized));
        assert!(!state.is_active(DecorationModifier::Resizable));
        assert!(state.is_supported(DecorationModifier::Resizable));
        assert_eq!(
            state.available(),
            DecorationModifier::Resizable.into()
        );
    }

    #[test]
    fn states_collection_inserts_and_removes() {
        let mut states = DecorationStates::new();
        let id = crate::types::DecorationId::new(1);
        states.set(
            id,
            DecorationState {
                active: DecorationModifier::Fullscreen.into(),
                ..Default::default()
            },
        );
        assert!(states.state_of(id).unwrap().is_active(DecorationModifier::Fullscreen));
        states.remove(id);
        assert!(states.state_of(id).is_none());
    }
}
