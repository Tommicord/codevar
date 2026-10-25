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

//! The static keyboard state and the handler a widget implements.
//!
//! [`KeyboardState`] is a small, lock-free record of what the keyboard is
//! doing: the last key that went down, up or repeated, the last modifier, the
//! modifier and lock flags, and the set of keys currently held. It has no
//! allocation and no interior lock, so the whole state can live in a `static`
//! and a handler can hold one permanent `&'static KeyboardState` for its
//! entire lifetime.
//!
//! # Writers and readers
//!
//! `InputPump` is the only writer. It calls
//! [`apply`](KeyboardState::apply) once per key event, in the order the
//! backend reported them. Readers — a widget, a shortcut table, a profiler on
//! another thread — call [`snapshot`](KeyboardState::snapshot), which retries
//! until it observes a complete transition and therefore never returns a
//! half-updated combination of fields. Individual accessors such as
//! [`is_pressed`](KeyboardState::is_pressed) load one atomic and are cheap
//! enough for a per-frame query.

mod handler;
mod keys;
mod modifiers;
mod snapshot;
mod state;

pub use handler::{KEYBOARD, KeyTransition, KeyboardBinding, KeyboardHandler};
pub use modifiers::KeyboardModifiers;
pub use snapshot::KeyboardSnapshot;
pub use state::KeyboardState;
