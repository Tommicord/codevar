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

//! Buttons and the modifier decorations of Kuitui.
//!
//! This crate sits on top of [`codevar_kuitui`] and adds the widgets an
//! application clicks: a [`Button`](button::Button) trait whose
//! parameters are the parameters of Jetpack Compose's `Button`, the
//! text buttons that implement it, and the decorations a button needs
//! from the modifier chain — the rounded container, the ambient shadow,
//! the hover, press and selection tracking, and the canvas-backed
//! `concrete` decoration that paints a node from the outside.
//!
//! # Why a trait and not a function
//!
//! Compose spells `Button` as a function with defaulted parameters
//! because its composition model re-runs the call every frame. Kuitui
//! retains the tree, so the same eleven parameters are the methods of
//! the [`Button`](button::Button) trait: an application describes a
//! button once, and [`Button::node`](button::Button::node) turns that
//! description into the node that draws it. A custom button is a
//! `struct` that answers the questions the trait asks, which keeps the
//! parameter list honest — an implementation that cannot answer one of
//! them says so instead of silently defaulting.
//!
//! # Why the states are a bit set
//!
//! A button is enabled, hovered, pressed and selected at the same time,
//! so the flags of [`state::ButtonFlags`] and the states of
//! [`codevar_kuitui::InteractionState`] are combined with `|` instead
//! of being passed as booleans. The colors, the shadow and the border a
//! button draws are all resolved from that pair at draw time, which is
//! what lets the pressed look of a button change without the tree being
//! recomposed.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod defaults;
pub mod state;

pub use state::{BorderStroke, ButtonColors, ButtonElevation, ButtonFlags, ButtonPalette};
