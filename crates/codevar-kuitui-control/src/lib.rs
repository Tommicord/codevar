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

//! Device-agnostic input control for Kuitui.
//!
//! `codevar-kuitui-control` is the portable half of input handling. It owns
//! the device and event model, the lazy registration policy and the stream
//! delivery mechanism; a backend crate owns the operating-system calls.
//!
//! # Lazy registration
//!
//! Constructing a [`Control`] opens nothing. The first
//! [`Control::open`] for a [`DeviceClass`] calls the backend's
//! [`Backend::start`], and later opens of the same class share that
//! registration. [`Control::unregister`] is the explicit release: it stops
//! the backend and ends the live streams for the class. Opening the class
//! again registers it again. This keeps a microphone or a gamepad closed
//! until something actually consumes it, which matters both for battery and
//! for permission prompts on platforms that ask on first use.
//!
//! # Reactive streams instead of observers
//!
//! Input is published through [`Publisher`]/[`Subscription`], not through
//! callbacks registered on a list. A subscriber holds a bounded queue and a
//! demand counter: it calls [`Subscription::request`] to grant demand, and
//! the publisher never calls consumer code directly. Two consequences matter
//! for a UI:
//!
//! - A paused or slow consumer cannot make the producer allocate without
//!   bound. Events that do not fit the queue are dropped and the loss is
//!   reported once as [`ControlError::Lagged`], which is visible instead of
//!   silent.
//! - A subscription is a stream that composes with any executor. There is no
//!   runtime dependency; [`Subscription::poll_next`] is the primitive and
//!   [`Subscription::next`] is the async convenience.
//!
//! # Backends
//!
//! Backends live in sibling crates so a target only compiles the operating
//! system code it can run:
//!
//! | Crate | Native facility |
//! | --- | --- |
//! | `codevar-kuitui-control-linux` | evdev (`/dev/input/event*`) |
//! | `codevar-kuitui-control-windows` | Win32 Raw Input |
//! | `codevar-kuitui-control-macos` | CoreGraphics event tap |
//! | `codevar-kuitui-control-wasm` | DOM keyboard, pointer and wheel events |
//!
//! Microphone capture is part of the event model ([`AudioEvent`]) but is not
//! implemented by the initial set of backends; a backend reports
//! [`ControlError::Unsupported`] for a class it cannot provide.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod class;
pub mod device;
pub mod error;

pub use class::{ClassSet, DeviceCapability, DeviceClass};
pub use device::{DeviceId, DeviceInfo};
pub use error::ControlError;
