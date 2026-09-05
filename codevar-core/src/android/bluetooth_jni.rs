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

//! JNI bindings for Android Bluetooth HID input events.
//!
//! This module provides the JNI interface that Java calls to dispatch
//! input events to Rust callbacks. It acts as a thin layer that converts
//! JNI types to Rust types and forwards events to the CallbackManager.
//!
//! # Thread Safety
//!
//! All JNI functions are thread-safe and can be called from any Java thread.
//! The underlying `CallbackManager` uses `RwLock` for concurrent access.
//!
//! For example: `Java_net_codevar_bluetooth_NetBluetoothHandler_nativeOnKeyPressed`
//!
//! # Example
//!
//! ```rust,ignore
//! // From Rust code, register a callback
//! use codevar_core::android::bluetooth_jni::{register_callback, unregister_callback};
//! use codevar_core::android::bluetooth_callback::InputCallback;
//!
//! struct MyCallback;
//! impl InputCallback for MyCallback {
//!     fn on_key_pressed(&self, key_code: i32, modifiers: i32) {
//!         println!("Key pressed: {}", key_code);
//!     }
//!     // ... implement other methods
//! }
//!
//! let ptr = 0x1234;
//! register_callback(ptr, Box::new(MyCallback));
//!
//! // Later, unregister
//! unregister_callback(ptr);
//! ```

use crate::android::bluetooth_callback::{CallbackManager, CallbackStats};
use jni::JNIEnv;
use jni::objects::{JClass, JObject};
use jni::sys::{JNI_VERSION_1_6, jint, jlong};
use std::collections::Hashmap;
use std::sync::OnceLock;

/// Global callback manager instance.
/// This is initialized once and used by all JNI functions to dispatch events.
/// The `OnceLock` ensures thread-safe one-time initialization.
static CALLBACK_MANAGER: OnceLock<CallbackManager> = OnceLock::new();

/// Gets the global callback manager, initializing it if necessary.
///
/// This function uses `OnceLock` to ensure that the callback manager is
/// initialized exactly once, even if called from multiple threads concurrently.
///
/// # Returns
///
/// A reference to the global `CallbackManager` instance.
fn callback_manager() -> &'static CallbackManager {
    CALLBACK_MANAGER.or_init(|| {
        crate::debug!("Initializing global CallbackManager");
        CallbackManager::new()
    })
}

/// Checks if the JNI environment has a pending exception and clears it if present.
///
/// This is a safety measure to ensure that JNI operations don't fail
/// due to pending exceptions from previous operations.
///
/// # Arguments
///
/// * `env` - The JNI environment.
///
/// # Returns
///
/// `true` if an exception was pending and cleared, `false` otherwise.
fn check_and_clear_jni_exception(env: &JNIEnv) -> bool {
    if env.exception_check().unwrap_or(false) {
        crate::warn!("JNI exception pending, clearing");
        env.exception_clear().unwrap_or(());
        true
    } else {
        false
    }
}

/// JNI_OnLoad is called when the native library is loaded.
/// This is the entry point for the JNI library.
///
/// This function initializes the global callback manager and sets up
/// Android logging if running on Android.
///
/// # Arguments
///
/// * `vm` - The Java VM pointer.
/// * `_reserved` - Reserved pointer (unused).
///
/// # Returns
///
/// The JNI version required by this library (JNI_VERSION_1_6).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn JNI_OnLoad(
    vm: jni::sys::JavaVM,
    _reserved: *mut std::ffi::c_void,
) -> jint {
    crate::debug!("JNI_OnLoad called");
    let _manager = get_callback_manager();
    crate::info!("Codevar JNI library loaded successfully");
    JNI_VERSION_1_6
}

/// JNI_OnUnload is called when the native library is unloaded.
/// This is the cleanup point for the JNI library.
///
/// This function clears all registered callbacks to prevent memory leaks
/// and ensure proper cleanup.
///
/// # Arguments
///
/// * `_vm` - The Java VM pointer (unused).
/// * `_reserved` - Reserved pointer (unused).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn JNI_OnUnload(
    _vm: jni::sys::JavaVM,
    _reserved: *mut std::ffi::c_void,
) {
    crate::debug!("JNI_OnUnload called");
    if let Some(manager) = CALLBACK_MANAGER.get() {
        let count = manager.callback_count();
        manager.clear();
        crate::info!("Cleared {} callbacks during JNI unload", count);
    }
    crate::info!("JNI library unloaded");
}

/// JNI callback: Notifies Rust of a key press event.
///
/// This function is called from Java when a key is pressed. It validates
/// the JNI state, converts the pointer to a u64, and dispatches the event
/// to the registered callback.
///
/// # JNI Signature
///
/// `private native void nativeOnKeyPressed(long ptr, int keyCode, int modifiers);`
///
/// # Arguments
///
/// * `env` - The JNI environment.
/// * `_obj` - The Java object instance (unused).
/// * `ptr` - The callback pointer originally passed to nativeRegisterReceiver.
/// * `key_code` - The Android key code (e.g., KeyEvent.KEYCODE_A = 29).
/// * `modifiers` - Modifier flags (bitmask: 1=SHIFT, 2=ALT, 4=CTRL, 8=META).
///
/// # Safety
///
/// If a JNI exception is pending, this function clears it and returns early
/// without dispatching the event.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_net_codevar_bluetooth_NetBluetoothHandler_nativeOnKeyPressed(
    env: JNIEnv,
    _obj: JObject,
    ptr: jlong,
    key_code: jint,
    modifiers: jint,
) {
    if check_and_clear_jni_exception(&env) {
        return;
    }
    let manager = get_callback_manager();
    let ptr_u64 = ptr as u64;
    manager.on_key_pressed(ptr_u64, key_code, modifiers);
}

/// JNI callback: Notifies Rust of a key release event.
///
/// This function is called from Java when a key is released.
///
/// # JNI Signature
///
/// `private native void nativeOnKeyReleased(long ptr, int keyCode, int modifiers);`
///
/// # Arguments
///
/// * `env` - The JNI environment.
/// * `_obj` - The Java object instance (unused).
/// * `ptr` - The callback pointer originally passed to nativeRegisterReceiver.
/// * `key_code` - The Android key code (e.g., KeyEvent.KEYCODE_A = 29).
/// * `modifiers` - Modifier flags (bitmask: 1=SHIFT, 2=ALT, 4=CTRL, 8=META).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_net_codevar_bluetooth_NetBluetoothHandler_nativeOnKeyReleased(
    env: JNIEnv,
    _obj: JObject,
    ptr: jlong,
    key_code: jint,
    modifiers: jint,
) {
    if check_and_clear_jni_exception(&env) {
        return;
    }

    let manager = get_callback_manager();
    let ptr_u64 = ptr as u64;
    manager.on_key_released(ptr_u64, key_code, modifiers);
}

/// JNI callback: Notifies Rust of a mouse movement event.
///
/// This function is called from Java when the mouse moves.
///
/// # JNI Signature
///
/// `private native void nativeOnMouseMove(long ptr, int x, int y);`
///
/// # Arguments
///
/// * `env` - The JNI environment.
/// * `_obj` - The Java object instance (unused).
/// * `ptr` - The callback pointer originally passed to nativeRegisterReceiver.
/// * `x` - The X coordinate (relative to screen origin).
/// * `y` - The Y coordinate (relative to screen origin).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_net_codevar_bluetooth_NetBluetoothHandler_nativeOnMouseMove(
    env: JNIEnv,
    _obj: JObject,
    ptr: jlong,
    x: jint,
    y: jint,
) {
    if check_and_clear_jni_exception(&env) {
        return;
    }

    let manager = get_callback_manager();
    let ptr_u64 = ptr as u64;
    manager.on_mouse_move(ptr_u64, x, y);
}

/// JNI callback: Notifies Rust of a mouse button press event.
///
/// This function is called from Java when a mouse button is pressed.
///
/// # JNI Signature
///
/// `private native void nativeOnButtonPress(long ptr, int button, int x, int y);`
///
/// # Arguments
///
/// * `env` - The JNI environment.
/// * `_obj` - The Java object instance (unused).
/// * `ptr` - The callback pointer originally passed to nativeRegisterReceiver.
/// * `button` - The button identifier (1=left, 2=middle, 3=right, etc.).
/// * `x` - The X coordinate at the time of press.
/// * `y` - The Y coordinate at the time of press.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_net_codevar_bluetooth_NetBluetoothHandler_nativeOnButtonPress(
    env: JNIEnv,
    _obj: JObject,
    ptr: jlong,
    button: jint,
    x: jint,
    y: jint,
) {
    if check_and_clear_jni_exception(&env) {
        return;
    }

    let manager = get_callback_manager();
    let ptr_u64 = ptr as u64;
    manager.on_button_press(ptr_u64, button, x, y);
}

/// JNI callback: Notifies Rust of a mouse button release event.
///
/// This function is called from Java when a mouse button is released.
///
/// # JNI Signature
///
/// `private native void nativeOnButtonRelease(long ptr, int button, int x, int y);`
///
/// # Arguments
///
/// * `env` - The JNI environment.
/// * `_obj` - The Java object instance (unused).
/// * `ptr` - The callback pointer originally passed to nativeRegisterReceiver.
/// * `button` - The button identifier (1=left, 2=middle, 3=right, etc.).
/// * `x` - The X coordinate at the time of release.
/// * `y` - The Y coordinate at the time of release.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_net_codevar_bluetooth_NetBluetoothHandler_nativeOnButtonRelease(
    env: JNIEnv,
    _obj: JObject,
    ptr: jlong,
    button: jint,
    x: jint,
    y: jint,
) {
    if check_and_clear_jni_exception(&env) {
        return;
    }

    let manager = get_callback_manager();
    let ptr_u64 = ptr as u64;
    manager.on_button_release(ptr_u64, button, x, y);
}

/// JNI callback: Notifies Rust of a mouse scroll event.
///
/// This function is called from Java when the mouse wheel scrolls.
///
/// # JNI Signature
///
/// `private native void nativeOnMouseScroll(long ptr, int delta, int x, int y);`
///
/// # Arguments
///
/// * `env` - The JNI environment.
/// * `_obj` - The Java object instance (unused).
/// * `ptr` - The callback pointer originally passed to nativeRegisterReceiver.
/// * `delta` - The scroll delta (positive for scroll up, negative for scroll down).
/// * `x` - The X coordinate at the time of scroll.
/// * `y` - The Y coordinate at the time of scroll.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_net_codevar_bluetooth_NetBluetoothHandler_nativeOnMouseScroll(
    env: JNIEnv,
    _obj: JObject,
    ptr: jlong,
    delta: jint,
    x: jint,
    y: jint,
) {
    if check_and_clear_jni_exception(&env) {
        return;
    }
    let manager = get_callback_manager();
    let ptr_u64 = ptr as u64;
    manager.on_mouse_scroll(ptr_u64, delta, x, y);
}

/// Registers a Rust callback for receiving input events.
///
/// This function is called from Rust code (not Java) to register a callback
/// that will receive input events from the Java Bluetooth handler.
///
/// # Arguments
///
/// * `ptr` - A unique pointer identifier for this callback.
/// * `callback` - The callback implementation to register.
///
/// # Returns
///
/// `true` if the callback was registered successfully, `false` otherwise.
pub fn register_callback(
    ptr: u64,
    callback: std::sync::Arc<dyn crate::android::bluetooth_callback::InputCallback>,
) -> bool {
    let manager = get_callback_manager();
    manager.register(ptr, callback)
}

/// Registers a Rust callback with a specific priority.
///
/// # Arguments
///
/// * `ptr` - A unique pointer identifier for this callback.
/// * `callback` - The callback implementation to register.
/// * `priority` - The priority level for this callback.
///
/// # Returns
///
/// `true` if the callback was registered successfully, `false` otherwise.
pub fn register_callback_with_priority(
    ptr: u64,
    callback: std::sync::Arc<dyn crate::android::bluetooth_callback::InputCallback>,
    priority: crate::android::bluetooth_callback::CallbackPriority,
) -> bool {
    let manager = get_callback_manager();
    manager
        .register_with_priority(ptr, callback, priority)
        .is_ok()
}

/// Unregisters a previously registered callback.
///
/// This function is called from Rust code (not Java) to unregister a callback.
///
/// # Arguments
///
/// * `ptr` - The pointer identifier of the callback to unregister.
///
/// # Returns
///
/// `true` if the callback was found and unregistered, `false` otherwise.
pub fn unregister_callback(ptr: u64) -> bool {
    let manager = get_callback_manager();
    manager.unregister(ptr).is_ok()
}

/// Gets the number of currently registered callbacks.
///
/// This function is called from Rust code (not Java) to query the callback count.
///
/// # Returns
///
/// The number of registered callbacks.
pub fn callback_count() -> usize {
    let manager = get_callback_manager();
    manager.callback_count()
}

/// Checks if a callback is registered with the given pointer.
///
/// # Arguments
///
/// * `ptr` - The pointer identifier to check.
///
/// # Returns
///
/// `true` if a callback is registered with this pointer, `false` otherwise.
pub fn is_callback_registered(ptr: u64) -> bool {
    let manager = get_callback_manager();
    manager.is_registered(ptr)
}

/// Gets statistics for a specific callback.
///
/// # Arguments
///
/// * `ptr` - The pointer identifier of the callback.
///
/// # Returns
///
/// The callback statistics, or `None` if no callback is registered with this pointer.
pub fn callback_stats(ptr: u64) -> Option<CallbackStats> {
    let manager = get_callback_manager();
    manager.callback_stats(ptr)
}

/// Gets statistics for all registered callbacks.
///
/// # Returns
///
/// A map of pointer identifiers to their statistics.
pub fn all_callback_stats() -> HashMap<u64, CallbackStats> {
    let manager = get_callback_manager();
    manager.all_stats().unwrap_or_default()
}

/// Returns the total number of events dispatched across all callbacks.
///
/// # Returns
///
/// The total event count.
pub fn total_events_dispatched() -> u64 {
    let manager = get_callback_manager();
    manager.total_events_dispatched()
}

/// Returns the total number of panics across all callbacks.
///
/// # Returns
///
/// The total panic count.
pub fn total_panics() -> u64 {
    let manager = get_callback_manager();
    manager.total_panics()
}

/// Clears all registered callbacks.
///
/// This function is called from Rust code (not Java) to clear all callbacks.
/// Each callback's `on_unregister` method will be called before removal.
pub fn clear_callbacks() {
    let manager = get_callback_manager();
    manager.clear();
}

/// Resets statistics for a specific callback.
///
/// # Arguments
///
/// * `ptr` - The pointer identifier of the callback.
///
/// # Returns
///
/// `true` if the statistics were reset, `false` otherwise.
pub fn reset_callback_stats(ptr: u64) -> bool {
    let manager = get_callback_manager();
    manager.reset_callback_stats(ptr).unwrap_or(false)
}

/// Resets statistics for all callbacks.
pub fn reset_all_stats() {
    let manager = get_callback_manager();
    manager.reset_all_stats()
}

/// Returns a list of all registered callback pointers.
///
/// # Returns
///
/// A vector of pointer identifiers.
pub fn registered_pointers() -> Vec<u64> {
    let manager = get_callback_manager();
    manager.registered_pointers()
}

#[cfg(target_os = "android")]
#[cfg(test)]
mod tests {
    use super::{
        CallbackPriority, InputCallback, callback_count, clear_callbacks,
        get_all_callback_stats, get_callback_stats, is_callback_registered,
        register_callback, register_callback_with_priority, registered_pointers,
        reset_all_stats, reset_callback_stats, total_events_dispatched, total_panics,
        unregister_callback,
    };

    struct TestCallback;

    impl InputCallback for TestCallback {
        fn on_key_pressed(&self, _key_code: i32, _modifiers: i32) {}
        fn on_key_released(&self, _key_code: i32, _modifiers: i32) {}
        fn on_mouse_move(&self, _x: i32, _y: i32) {}
        fn on_button_press(&self, _button: i32, _x: i32, _y: i32) {}
        fn on_button_release(&self, _button: i32, _x: i32, _y: i32) {}
        fn on_mouse_scroll(&self, _delta: i32, _x: i32, _y: i32) {}
    }

    fn register_unregister() {
        let ptr = 0x1234;
        assert!(register_callback(ptr, Arc::new(TestCallback)));
        assert_eq!(callback_count(), 1);
        assert!(is_callback_registered(ptr));
        assert!(unregister_callback(ptr));
        assert_eq!(callback_count(), 0);
    }

    fn register_with_priority() {
        let ptr = 0x1234;
        assert!(register_callback_with_priority(
            ptr,
            Arc::new(TestCallback),
            CallbackPriority::High
        ));
        assert!(is_callback_registered(ptr));
        assert!(unregister_callback(ptr));
    }

    fn stats() {
        let ptr = 0x1234;
        register_callback(ptr, Arc::new(TestCallback));

        let stats = get_callback_stats(ptr);
        assert!(stats.is_some());

        let all_stats = get_all_callback_stats();
        assert_eq!(all_stats.len(), 1);

        assert!(reset_callback_stats(ptr));
        reset_all_stats();

        unregister_callback(ptr);
    }

    fn registered_pointers() {
        let ptr1 = 0x1234;
        let ptr2 = 0x5678;
        register_callback(ptr1, Arc::new(TestCallback));
        register_callback(ptr2, Arc::new(TestCallback));

        let pointers = registered_pointers();
        assert_eq!(pointers.len(), 2);
        assert!(pointers.contains(&ptr1));
        assert!(pointers.contains(&ptr2));

        clear_callbacks();
    }

    fn clear_callbacks() {
        register_callback(1, Arc::new(TestCallback));
        register_callback(2, Arc::new(TestCallback));
        assert_eq!(callback_count(), 2);

        clear_callbacks();
        assert_eq!(callback_count(), 0);
    }

    fn total_events() {
        let ptr = 0x1234;
        register_callback(ptr, Arc::new(TestCallback));

        let initial_events = total_events_dispatched();
        let initial_panics = total_panics();

        // These should be 0 since we haven't dispatched any events
        assert_eq!(initial_events, 0);
        assert_eq!(initial_panics, 0);

        unregister_callback(ptr);
    }

    fn is_callback_registered() {
        let ptr = 0x1234;
        assert!(!is_callback_registered(ptr));

        register_callback(ptr, Arc::new(TestCallback));
        assert!(is_callback_registered(ptr));

        unregister_callback(ptr);
        assert!(!is_callback_registered(ptr));
    }

    fn multiple_callbacks() {
        register_callback(1, Arc::new(TestCallback));
        register_callback(2, Arc::new(TestCallback));
        register_callback(3, Arc::new(TestCallback));

        assert_eq!(callback_count(), 3);
        assert_eq!(registered_pointers().len(), 3);

        clear_callbacks();
    }

    fn reset_stats_nonexistent() {
        // Should return false for non-existent callback
        assert!(!reset_callback_stats(9999));
    }

    fn stats_nonexistent() {
        // Should return None for non-existent callback
        assert!(get_callback_stats(9999).is_none());
    }
}
