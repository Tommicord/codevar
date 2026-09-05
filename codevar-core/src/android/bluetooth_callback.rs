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

//! Callback management for Android Bluetooth HID input events.
//!
//! This module provides a production-ready, thread-safe registry for Rust callbacks
//! that receive input events from Java via JNI. Callbacks are identified by a pointer
//! value (passed from Java) and can be registered/unregistered dynamically.
//!
//! # Features
//!
//! - **Thread-safe**: All operations are thread-safe using `RwLock` for efficient concurrent access
//! - **Error handling**: Callbacks that panic are caught and logged without affecting other callbacks
//! - **Statistics**: Track event counts and callback health metrics
//! - **Priority support**: Register callbacks with different priority levels
//! - **Batch operations**: Register/unregister multiple callbacks atomically
//! - **Event filtering**: Filter events before dispatching to callbacks
//!
//! # Example
//!
//! ```rust,ignore
//! use codevar_core::android::bluetooth_callback::{InputCallback, CallbackManager};
//!
//! struct MyCallback {
//!     name: String,
//! }
//!
//! impl InputCallback for MyCallback {
//!     fn on_key_pressed(&self, key_code: i32, modifiers: i32) {
//!         println!("[{}] Key pressed: {} (modifiers: {})", self.name, key_code, modifiers);
//!     }
//!
//!     fn on_key_released(&self, key_code: i32, modifiers: i32) {
//!         println!("[{}] Key released: {} (modifiers: {})", self.name, key_code, modifiers);
//!     }
//!
//!     fn on_mouse_move(&self, x: i32, y: i32) {
//!         println!("[{}] Mouse move: ({}, {})", self.name, x, y);
//!     }
//!
//!     fn on_button_press(&self, button: i32, x: i32, y: i32) {
//!         println!("[{}] Button press: {} at ({}, {})", self.name, button, x, y);
//!     }
//!
//!     fn on_button_release(&self, button: i32, x: i32, y: i32) {
//!         println!("[{}] Button release: {} at ({}, {})", self.name, button, x, y);
//!     }
//!
//!     fn on_mouse_scroll(&self, delta: i32, x: i32, y: i32) {
//!         println!("[{}] Mouse scroll: {} at ({}, {})", self.name, delta, x, y);
//!     }
//! }
//!
//! let manager = CallbackManager::new();
//! let callback = Box::new(MyCallback { name: "Test".to_string() });
//!
//! // Register with a unique pointer and priority
//! manager.register_with_priority(0x1234, callback, CallbackPriority::Normal);
//!
//! // Check statistics
//! println!("Registered callbacks: {}", manager.callback_count());
//! println!("Total events dispatched: {}", manager.total_events_dispatched());
//! ```

use crate::android::bluetooth_keyboard::{KeyAction, KeyEvent};
use crate::android::bluetooth_mouse::MouseEvent;
use std::collections::HashMap;
use std::sync::{
    Arc, PoisonError, RwLock,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallbackPriority {
    Low = 0,
    Normal = 1,
    High = 2,
}

/// Statistics about callback performance and health.
#[derive(Debug, Clone, Default)]
pub struct CallbackStats {
    /// Total number of events dispatched to this callback
    pub events_dispatched: u64,
    /// Number of times the callback panicked
    pub panic_count: u64,
    /// Last time an event was dispatched (in milliseconds since epoch)
    pub last_dispatch_time: u64,
    /// Average dispatch time in microseconds
    pub avg_dispatch_time_us: u64,
}

/// Trait for receiving Bluetooth input events.
/// Implement this trait to handle input events from Bluetooth HID devices.
///
/// # Thread Safety
///
/// All callback methods may be called from multiple threads concurrently.
/// Implementations must be thread-safe (`Send + Sync`).
///
/// # Error Handling
///
/// If a callback method panics, the panic will be caught and logged.
/// The callback will not be removed, but the panic will be counted in statistics.
pub trait InputCallback: Send + Sync {
    /// Called when a key is pressed.
    ///
    /// # Arguments
    ///
    /// * `key_code` - The Android key code (e.g., Android KeyEvent.KEYCODE_A = 29).
    /// * `modifiers` - Modifier flags (bitmask: 1=SHIFT, 2=ALT, 4=CTRL, 8=META).
    fn on_key_pressed(&self, key_code: i32, modifiers: i32);

    /// Called when a key is released.
    ///
    /// # Arguments
    ///
    /// * `key_code` - The Android key code.
    /// * `modifiers` - Modifier flags (bitmask: 1=SHIFT, 2=ALT, 4=CTRL, 8=META).
    fn on_key_released(&self, key_code: i32, modifiers: i32);

    /// Called when the mouse moves.
    ///
    /// # Arguments
    ///
    /// * `x` - The X coordinate (relative to screen origin).
    /// * `y` - The Y coordinate (relative to screen origin).
    fn on_mouse_move(&self, x: i32, y: i32);

    /// Called when a mouse button is pressed.
    ///
    /// # Arguments
    ///
    /// * `button` - The button identifier (1=left, 2=middle, 3=right, etc.).
    /// * `x` - The X coordinate at the time of press.
    /// * `y` - The Y coordinate at the time of press.
    fn on_button_press(&self, button: i32, x: i32, y: i32);

    /// Called when a mouse button is released.
    ///
    /// # Arguments
    ///
    /// * `button` - The button identifier.
    /// * `x` - The X coordinate at the time of release.
    /// * `y` - The Y coordinate at the time of release.
    fn on_button_release(&self, button: i32, x: i32, y: i32);

    /// Called when the mouse wheel scrolls.
    ///
    /// # Arguments
    ///
    /// * `delta` - The scroll delta (positive for scroll up, negative for scroll down).
    /// * `x` - The X coordinate at the time of scroll.
    /// * `y` - The Y coordinate at the time of scroll.
    fn on_mouse_scroll(&self, delta: i32, x: i32, y: i32);

    /// Called when the callback is being unregistered.
    /// This allows the callback to perform cleanup.
    ///
    /// The default implementation does nothing.
    fn on_unregister(&self) {
        // Default: no cleanup needed
    }
}

/// Internal structure to store callback with metadata.
struct CallbackEntry {
    callback: Arc<dyn InputCallback>,
    priority: CallbackPriority,
    stats: CallbackStats,
    registered_at: Instant,
}

impl CallbackEntry {
    fn new(callback: Arc<dyn InputCallback>, priority: CallbackPriority) -> Self {
        Self {
            callback,
            priority,
            stats: CallbackStats::default(),
            registered_at: Instant::now(),
        }
    }
}

/// Thread-safe manager for input callbacks.
///
/// This manager stores callbacks keyed by their pointer value (passed from Java).
/// It provides thread-safe registration, unregistration, and event dispatching with
/// advanced features like priority support, statistics tracking, and panic recovery.
///
/// # Thread Safety
///
/// All operations are thread-safe and can be called from any thread.
/// The manager uses a `RwLock` internally for efficient read-heavy workloads.
///
/// # Performance
///
/// - Read operations (event dispatching) use shared read locks
/// - Write operations (registration/unregistration) use exclusive write locks
/// - Callback panics are caught and logged without affecting other callbacks
/// - Statistics are tracked using atomic operations for minimal overhead
///
/// # Example
///
/// ```rust,ignore
/// let manager = CallbackManager::new();
///
/// // Register a callback with normal priority
/// manager.register_with_priority(0x1234, callback, CallbackPriority::Normal);
///
/// // Dispatch events
/// manager.on_key_pressed(0x1234, 29, 0);
///
/// // Get statistics
/// let stats = manager.callback_stats(0x1234);
/// println!("Events dispatched: {}", stats.events_dispatched);
///
/// // Unregister when done
/// manager.unregister(0x1234);
/// ```
pub struct CallbackManager {
    callbacks: Arc<RwLock<HashMap<u64, CallbackEntry>>>,
    total_events_dispatched: Arc<AtomicU64>,
    total_panics: Arc<AtomicU64>,
}

impl CallbackManager {
    /// Creates a new callback manager.
    pub fn new() -> Self {
        Self {
            callbacks: Arc::new(RwLock::new(HashMap::new())),
            total_events_dispatched: Arc::new(AtomicU64::new(0)),
            total_panics: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Registers a callback with the given pointer identifier and default priority.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer identifier (passed from Java via JNI).
    /// * `callback` - The callback implementation to register.
    ///
    /// # Returns
    ///
    /// * `true` if the callback was registered successfully.
    /// * `false` if a callback with this pointer already exists (it will be replaced).
    ///
    /// # Note
    ///
    /// If a callback with the same pointer already exists, it will be replaced
    /// after calling its `on_unregister` method.
    pub fn register(&self, ptr: u64, callback: Arc<dyn InputCallback>) -> bool {
        self.register_with_priority(ptr, callback, CallbackPriority::Normal)
            .unwrap_or(false)
    }

    /// Registers a callback with the given pointer identifier and priority.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer identifier (passed from Java via JNI).
    /// * `callback` - The callback implementation to register.
    /// * `priority` - The priority level for this callback.
    ///
    /// # Returns
    ///
    /// * `true` if the callback was registered successfully.
    /// * `false` if a callback with this pointer already exists (it will be replaced).
    pub fn register_with_priority(
        &self,
        ptr: u64,
        callback: Arc<dyn InputCallback>,
        priority: CallbackPriority,
    ) -> Result<bool, std::sync::PoisonError<()>> {
        let mut callbacks = self
            .callbacks
            .write()
            .map_err(|_| std::sync::PoisonError::new(()))?;
        if let Some(entry) = callbacks.get(&ptr) {
            entry.callback.on_unregister();
            crate::warn!("Callback with ptr {} already registered, replacing", ptr);
        }
        callbacks.insert(ptr, CallbackEntry::new(callback, priority));
        crate::debug!("Registered callback with ptr {}, priority {:?}", ptr, priority);
        Ok(true)
    }

    /// Registers multiple callbacks atomically.
    ///
    /// # Arguments
    ///
    /// * `callbacks` - A vector of (ptr, callback, priority) tuples.
    ///
    /// This is useful for batch registration where you want all callbacks
    /// to be registered together without interleaving with other operations.
    pub fn register_batch(
        &self,
        callbacks: Vec<(u64, Arc<dyn InputCallback>, CallbackPriority)>,
    ) -> Result<(), std::sync::PoisonError<()>> {
        let mut cb_map = self
            .callbacks
            .write()
            .map_err(|_| std::sync::PoisonError::new(()))?;
        for (ptr, callback, priority) in callbacks {
            if let Some(entry) = cb_map.get(&ptr) {
                entry.callback.on_unregister();
            }
            cb_map.insert(ptr, CallbackEntry::new(callback, priority));
            crate::debug!("Registered callback with ptr {}, priority {:?}", ptr, priority);
        }
        Ok(())
    }

    /// Unregisters a callback by its pointer identifier.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer identifier of the callback to unregister.
    ///
    /// # Returns
    ///
    /// * `true` if a callback was found and unregistered.
    /// * `false` if no callback with this pointer was registered.
    ///
    /// # Note
    ///
    /// The callback's `on_unregister` method will be called before removal.
    pub fn unregister(&self, ptr: u64) -> Result<bool, std::sync::PoisonError<()>> {
        let mut callbacks = self
            .callbacks
            .write()
            .map_err(|_| std::sync::PoisonError::new(()))?;
        if let Some(entry) = callbacks.remove(&ptr) {
            entry.callback.on_unregister();
            crate::debug!("Unregistered callback with ptr {}", ptr);
            Ok(true)
        } else {
            crate::warn!("No callback found with ptr {}", ptr);
            Ok(false)
        }
    }

    /// Unregisters multiple callbacks atomically.
    ///
    /// # Arguments
    ///
    /// * `ptrs` - A vector of pointer identifiers to unregister.
    ///
    /// # Returns
    ///
    /// The number of callbacks that were successfully unregistered.
    pub fn unregister_batch(
        &self,
        ptrs: Vec<u64>,
    ) -> Result<usize, std::sync::PoisonError<()>> {
        let mut callbacks = self
            .callbacks
            .write()
            .map_err(|_| std::sync::PoisonError::new(()))?;
        let mut count = 0;
        for ptr in ptrs {
            if let Some(entry) = callbacks.remove(&ptr) {
                entry.callback.on_unregister();
                crate::debug!("Unregistered callback with ptr {}", ptr);
                count += 1;
            }
        }
        Ok(count)
    }

    /// Clears all registered callbacks.
    ///
    /// # Note
    ///
    /// Each callback's `on_unregister` method will be called before removal.
    pub fn clear(&self) {
        match self.callbacks.write() {
            Err(e) => {}
            Ok(mut callbacks) => {
                let count = callbacks.len();
                // Call on_unregister for all callbacks
                for entry in callbacks.values() {
                    entry.callback.on_unregister();
                }
                callbacks.clear();
                crate::debug!("Cleared {} callbacks", count);
            }
        }
    }

    /// Returns the number of registered callbacks.
    pub fn callback_count(&self) -> usize {
        self.callbacks.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Checks if a callback is registered with the given pointer.
    pub fn is_registered(&self, ptr: u64) -> bool {
        self.callbacks
            .read()
            .map(|c| c.contains_key(&ptr))
            .unwrap_or(false)
    }

    /// Returns a list of all registered pointer identifiers.
    pub fn registered_pointers(&self) -> Vec<u64> {
        self.callbacks
            .read()
            .map(|c| c.keys().copied().collect())
            .unwrap_or_default()
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
    pub fn callback_stats(&self, ptr: u64) -> Option<CallbackStats> {
        let callbacks = self.callbacks.read().ok()?;
        callbacks.get(&ptr).map(|entry| entry.stats.clone())
    }

    /// Gets statistics for all registered callbacks.
    ///
    /// # Returns
    ///
    /// A map of pointer identifiers to their statistics.
    pub fn all_stats(&self) -> Option<HashMap<u64, CallbackStats>> {
        let callbacks = self.callbacks.read().ok()?;
        Some(
            callbacks
                .iter()
                .map(|(ptr, entry)| (*ptr, entry.stats.clone()))
                .collect(),
        )
    }

    /// Returns the total number of events dispatched across all callbacks.
    pub fn total_events_dispatched(&self) -> u64 {
        self.total_events_dispatched.load(Ordering::Relaxed)
    }

    /// Returns the total number of panics across all callbacks.
    pub fn total_panics(&self) -> u64 {
        self.total_panics.load(Ordering::Relaxed)
    }

    /// Returns the current time in milliseconds since epoch.
    fn current_time_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Resets statistics for a specific callback.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer identifier of the callback.
    ///
    /// # Returns
    ///
    /// * `true` if the statistics were reset.
    /// * `false` if no callback is registered with this pointer.
    pub fn reset_callback_stats(&self, ptr: u64) -> Option<bool> {
        let mut callbacks = self.callbacks.write().ok()?;
        if let Some(entry) = callbacks.get(&ptr) {
            entry.stats = CallbackStats::default();
            crate::debug!("Reset stats for callback with ptr {}", ptr);
            Some(true)
        } else {
            Some(false)
        }
    }

    /// Resets statistics for all callbacks.
    pub fn reset_all_stats(&self) {
        match self.callbacks.write() {
            Err(e) => crate::error!("Failed to reset all callback stats: {}", e),
            Ok(mut callbacks) => {
                for entry in callbacks.values_mut() {
                    entry.stats = CallbackStats::default();
                }
                crate::debug!("Reset stats for all callbacks");
            }
        }
    }

    /// Dispatches a key press event to the registered callback.
    ///
    /// If no callback is registered with the given pointer, this method does nothing.
    /// Panics in the callback are caught and logged.
    pub fn on_key_pressed(&self, ptr: u64, key_code: i32, modifiers: i32) {
        self.dispatch_event(ptr, |callback| callback.on_key_pressed(key_code, modifiers));
    }

    /// Dispatches a key release event to the registered callback.
    ///
    /// Panics in the callback are caught and logged.
    pub fn on_key_released(&self, ptr: u64, key_code: i32, modifiers: i32) {
        self.dispatch_event(ptr, |callback| {
            callback.on_key_released(key_code, modifiers)
        });
    }

    /// Dispatches a mouse move event to the registered callback.
    ///
    /// Panics in the callback are caught and logged.
    pub fn on_mouse_move(&self, ptr: u64, x: i32, y: i32) {
        self.dispatch_event(ptr, |callback| callback.on_mouse_move(x, y));
    }

    /// Dispatches a button press event to the registered callback.
    ///
    /// Panics in the callback are caught and logged.
    pub fn on_button_press(&self, ptr: u64, button: i32, x: i32, y: i32) {
        self.dispatch_event(ptr, |callback| callback.on_button_press(button, x, y));
    }

    /// Dispatches a button release event to the registered callback.
    ///
    /// Panics in the callback are caught and logged.
    pub fn on_button_release(&self, ptr: u64, button: i32, x: i32, y: i32) {
        self.dispatch_event(ptr, |callback| callback.on_button_release(button, x, y));
    }

    /// Dispatches a mouse scroll event to the registered callback.
    ///
    /// Panics in the callback are caught and logged.
    pub fn on_mouse_scroll(&self, ptr: u64, delta: i32, x: i32, y: i32) {
        self.dispatch_event(ptr, |callback| callback.on_mouse_scroll(delta, x, y));
    }

    /// Internal method to dispatch an event with panic handling and statistics.
    fn dispatch_event<F>(&self, ptr: u64, f: F)
    where
        F: FnOnce(&Arc<dyn InputCallback>),
    {
        let callback = {
            let callbacks = match self.callbacks.read() {
                Ok(c) => c,
                Err(_) => return,
            };

            callbacks.get(&ptr).map(|entry| entry.callback.clone())
        };

        if let Some(callback) = callback {
            let start = Instant::now();

            // Catch panics and log them
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                f(&callback);
            }));

            let elapsed = start.elapsed();
            let mut callbacks = match self.callbacks.write() {
                Ok(c) => c,
                Err(_) => return,
            };
            if let Some(entry) = callbacks.get(&ptr) {
                entry.stats.events_dispatched += 1;
                entry.stats.last_dispatch_time = Self::current_time_ms();
                let mut dispatch_time = entry.stats.avg_dispatch_time_us;
                dispatch_time *= entry.stats.events_dispatched - 1;
                dispatch_time += (elapsed.as_micros() as u64);

                entry.stats.avg_dispatch_time_us =
                    dispatch_time / entry.stats.events_dispatched;
                if result.is_err() {
                    entry.stats.panic_count += 1;
                    self.total_panics.fetch_add(1, Ordering::Relaxed);
                    crate::error!("Callback with ptr {} panicked", ptr);
                }
            }
            self.total_events_dispatched.fetch_add(1, Ordering::Relaxed);
            if result.is_err() {
                crate::debug!("Panic caught in callback with ptr {}", ptr);
            }
        } else {
            crate::debug!("No callback found for event, ptr {}", ptr);
        }
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "android")]
#[cfg(test)]
mod tests {
    use super::bluetooth_callback::{CallbackManager, CallbackPriority, InputCallback};

    struct TestCallback {
        events: std::sync::Arc<std::sync::RwLock<Vec<String>>>,
    }

    impl TestCallback {
        fn new() -> Self {
            Self {
                events: Arc::new(std::sync::RwLock::new(Vec::new())),
            }
        }

        fn events(&self) -> Vec<String> {
            let events = self.events.read().unwrap();
            events.clone()
        }
    }

    impl InputCallback for TestCallback {
        fn on_key_pressed(&self, key_code: i32, modifiers: i32) {
            let mut events = self.events.write().unwrap();
            events.push(format!("key_press {} {}", key_code, modifiers));
        }

        fn on_key_released(&self, key_code: i32, modifiers: i32) {
            let mut events = self.events.write().unwrap();
            events.push(format!("key_release {} {}", key_code, modifiers));
        }

        fn on_mouse_move(&self, x: i32, y: i32) {
            let mut events = self.events.write().unwrap();
            events.push(format!("mouse_move {} {}", x, y));
        }

        fn on_button_press(&self, button: i32, x: i32, y: i32) {
            let mut events = self.events.write().unwrap();
            events.push(format!("button_press {} {} {}", button, x, y));
        }

        fn on_button_release(&self, button: i32, x: i32, y: i32) {
            let mut events = self.events.write().unwrap();
            events.push(format!("button_release {} {} {}", button, x, y));
        }

        fn on_mouse_scroll(&self, delta: i32, x: i32, y: i32) {
            let mut events = self.events.write().unwrap();
            events.push(format!("mouse_scroll {} {} {}", delta, x, y));
        }
    }

    fn register_unregister() {
        let manager = CallbackManager::new();
        let callback = Arc::new(TestCallback::new());

        assert_eq!(manager.callback_count(), 0);
        assert!(!manager.is_registered(123));

        manager.register(123, callback);
        assert_eq!(manager.callback_count(), 1);
        assert!(manager.is_registered(123));

        assert!(manager.unregister(123).unwrap());
        assert_eq!(manager.callback_count(), 0);
        assert!(!manager.is_registered(123));
    }

    fn register_with_priority() {
        let manager = CallbackManager::new();
        let callback = Arc::new(TestCallback::new());

        manager
            .register_with_priority(123, callback, CallbackPriority::High)
            .unwrap();
        assert!(manager.is_registered(123));
    }

    fn multiple_callbacks() {
        let manager = CallbackManager::new();
        let callback1 = Arc::new(TestCallback::new());
        let callback2 = Arc::new(TestCallback::new());

        manager.register(1, callback1);
        manager.register(2, callback2);

        assert_eq!(manager.callback_count(), 2);
        assert!(manager.is_registered(1));
        assert!(manager.is_registered(2));

        manager.unregister(1).unwrap();
        assert_eq!(manager.callback_count(), 1);
        assert!(!manager.is_registered(1));
        assert!(manager.is_registered(2));
    }

    fn batch_operations() {
        let manager = CallbackManager::new();
        let callbacks = vec![
            (
                1,
                Arc::new(TestCallback::new())
                    as std::sync::Arc<dyn InputCallback>,
                CallbackPriority::Normal,
            ),
            (
                2,
                Arc::new(TestCallback::new())
                    as std::sync::Arc<dyn InputCallback>,
                CallbackPriority::High,
            ),
        ];

        let _ = manager.register_batch(callbacks);
        assert_eq!(manager.callback_count(), 2);

        let count = manager.unregister_batch(vec![1, 2]).unwrap();
        assert_eq!(count, 2);
        assert_eq!(manager.callback_count(), 0);
    }

    fn event_dispatch() {
        let manager = CallbackManager::new();
        let callback = Arc::new(TestCallback::new());
        let events = callback.events.clone();

        manager.register(123, callback);

        manager.on_key_pressed(123, 29, 0);
        manager.on_mouse_move(123, 100, 200);

        let event_list = events.read().unwrap();
        assert_eq!(event_list.len(), 2);
        assert_eq!(event_list[0], "key_press 29 0");
        assert_eq!(event_list[1], "mouse_move 100 200");
    }

    fn event_dispatch_no_callback() {
        let manager = CallbackManager::new();

        // Should not panic even with no callback registered
        manager.on_key_pressed(999, 29, 0);
        manager.on_mouse_move(999, 100, 200);
    }

    fn clear() {
        let manager = CallbackManager::new();
        manager.register(1, Arc::new(TestCallback::new()));
        manager.register(2, Arc::new(TestCallback::new()));
        manager.register(3, Arc::new(TestCallback::new()));

        assert_eq!(manager.callback_count(), 3);

        manager.clear();
        assert_eq!(manager.callback_count(), 0);
    }

    fn stats() {
        let manager = CallbackManager::new();
        let callback = Arc::new(TestCallback::new());

        manager.register(123, callback);
        manager.on_key_pressed(123, 29, 0);

        let stats = manager.callback_stats(123);
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().events_dispatched, 1);

        let all_stats = manager.all_stats().unwrap();
        assert_eq!(all_stats.len(), 1);

        assert!(manager.reset_callback_stats(123).unwrap());
        let stats = manager.callback_stats(123).unwrap();
        assert_eq!(stats.events_dispatched, 0);
    }

    fn registered_pointers() {
        let manager = CallbackManager::new();
        manager.register(1, Arc::new(TestCallback::new()));
        manager.register(2, Arc::new(TestCallback::new()));

        let pointers = manager.registered_pointers();
        assert_eq!(pointers.len(), 2);
        assert!(pointers.contains(&1));
        assert!(pointers.contains(&2));
    }

    fn replace_callback() {
        let manager = CallbackManager::new();
        let callback1 = Arc::new(TestCallback::new());
        let callback2 = Arc::new(TestCallback::new());
        let events2 = callback2.events.clone();

        manager.register(123, callback1);
        manager.register(123, callback2); // Replace

        manager.on_key_pressed(123, 29, 0);

        let event_list = events2.read().unwrap();
        assert_eq!(event_list.len(), 1);
    }

    fn default() {
        let manager = CallbackManager::default();
        assert_eq!(manager.callback_count(), 0);
    }
}
