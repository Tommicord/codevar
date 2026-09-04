//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Android-specific functionality for Bluetooth HID input event handling.
//!
//! This module contains code that is specific to the Android platform,
//! including JNI bindings and platform-specific implementations for handling
//! Bluetooth HID (Human Interface Device) input events.
//!
//! # Platform Availability
//!
//! All modules in this crate are only available when targeting Android
//! (`target_os = "android"`). On other platforms, this module is empty.
//!
//! # Example
//!
//! ```rust,ignore
//! use codevar_core::android::bluetooth_callback::{InputCallback, CallbackManager};
//! use codevar_core::android::bluetooth_jni::register_callback;
//! use codevar_core::android::bluetooth_keyboard::{KeyEvent, KeyAction};
//!
//! struct MyCallback;
//!
//! impl InputCallback for MyCallback {
//!     fn on_key_pressed(&self, key_code: i32, modifiers: i32) {
//!         println!("Key pressed: {} (modifiers: {})", key_code, modifiers);
//!     }
//!
//!     fn on_key_released(&self, key_code: i32, modifiers: i32) {
//!         println!("Key released: {} (modifiers: {})", key_code, modifiers);
//!     }
//!
//!     fn on_mouse_move(&self, x: i32, y: i32) {
//!         println!("Mouse move: ({}, {})", x, y);
//!     }
//!
//!     fn on_button_press(&self, button: i32, x: i32, y: i32) {
//!         println!("Button press: {} at ({}, {})", button, x, y);
//!     }
//!
//!     fn on_button_release(&self, button: i32, x: i32, y: i32) {
//!         println!("Button release: {} at ({}, {})", button, x, y);
//!     }
//!
//!     fn on_mouse_scroll(&self, delta: i32, x: i32, y: i32) {
//!         println!("Mouse scroll: {} at ({}, {})", delta, x, y);
//!     }
//! }
//!
//! // Register the callback with a unique pointer
//! let ptr = 0x1234;
//! register_callback(ptr, Box::new(MyCallback));
//!
//! // Events will now be dispatched to MyCallback when Java calls JNI functions
//! ```
//!
//! # Thread Safety
//!
//! All operations are thread-safe. The `CallbackManager` uses `RwLock` for
//! efficient concurrent access, and JNI functions can be called from any Java thread.
//!
//! # Error Handling
//!
//! - JNI exceptions are checked and cleared before processing
//! - Invalid pointer values are handled gracefully (no-op)
//! - Callback panics are caught and logged without affecting other callbacks
//! - Statistics are tracked for monitoring callback health

#[cfg(target_os = "android")]
pub mod bluetooth_callback;

#[cfg(target_os = "android")]
pub mod bluetooth_input;

#[cfg(target_os = "android")]
pub mod bluetooth_jni;

#[cfg(target_os = "android")]
pub mod bluetooth_keyboard;

#[cfg(target_os = "android")]
pub mod bluetooth_mouse;

#[cfg(target_os = "android")]
pub use bluetooth_callback::{
    CallbackManager, CallbackPriority, CallbackStats, InputCallback,
};
#[cfg(target_os = "android")]
pub use bluetooth_input::{InputEvent, InputType};

#[cfg(target_os = "android")]
pub use bluetooth_keyboard::{KeyAction, KeyEvent};

#[cfg(target_os = "android")]
pub use bluetooth_mouse::{MouseEvent, MouseEventType};
