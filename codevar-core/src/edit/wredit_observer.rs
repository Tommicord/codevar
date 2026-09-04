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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Observer pattern for tracking user actions on Writable structures.
//!
//! This module provides the observer pattern implementation for tracking
//! user actions on writable data structures, enabling collaborative features
//! by capturing and propagating user edits.

use std::sync::{Arc, Mutex};

/// Types of user actions that can be observed on a Writable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum UserActionType {
    /// Insert a character at cursor position
    InsertChar = 0,
    /// Delete character at cursor (forward)
    Delete = 1,
    /// Delete character before cursor (backward)
    Backspace = 2,
    /// Move cursor forward
    MoveForward = 3,
    /// Move cursor backward
    MoveBackward = 4,
    /// Move cursor to next line
    MoveNextLine = 5,
    /// Move cursor to previous line
    MovePrevLine = 6,
    /// Move cursor to line start
    MoveLineStart = 7,
    /// Move cursor to line end
    MoveLineEnd = 8,
    /// Move cursor forward by word
    MoveWordForward = 16,
    /// Move cursor backward by word
    MoveWordBackward = 17,
    /// Select all text
    SelectAll = 18,
    /// Copy selected text to clipboard
    Copy = 19,
    /// Paste text from clipboard
    Paste = 20,
    /// Cut selected text to clipboard
    Cut = 21,
    /// Undo last operation
    Undo = 9,
    /// Redo last undone operation
    Redo = 10,
    /// Begin edit group
    BeginEditGroup = 11,
    /// End edit group
    EndEditGroup = 12,
    /// File opened
    FileOpened = 13,
    /// File closed
    FileClosed = 14,
    /// Cursor selection changed
    SelectionChanged = 15,
}

/// A user action event with associated metadata.
#[derive(Debug, Clone)]
pub struct UserActionEvent {
    /// The type of action performed
    pub action_type: UserActionType,
    /// The cursor ID associated with this action
    pub cursor_id: usize,
    /// The character value (for insert actions)
    pub character: Option<u32>,
    /// The count/magnitude (for move actions)
    pub count: usize,
    /// Timestamp of the action
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: u32,
}

impl UserActionEvent {
    /// Creates a new user action event.
    pub fn new(
        action_type: UserActionType,
        cursor_id: usize,
        character: Option<u32>,
        count: usize,
        metadata: u32,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();
        Self {
            action_type,
            cursor_id,
            character,
            count,
            timestamp,
            metadata,
        }
    }
}

/// Observer trait for receiving user action notifications.
pub trait UserActionObserver: Send + Sync {
    /// Called when a user action occurs.
    ///
    /// # Arguments
    ///
    /// * `event` - The user action event
    fn on_action(&self, event: &UserActionEvent);

    /// Called when a batch of actions occurs.
    ///
    /// # Arguments
    ///
    /// * `events` - Slice of user action events
    fn on_action_batch(&self, events: &[UserActionEvent]) {
        for event in events {
            self.on_action(event);
        }
    }
}

/// Registry for managing user action observers.
#[derive(Clone)]
pub struct UserActionObserverRegistry {
    observers: Arc<Mutex<Vec<Arc<dyn UserActionObserver>>>>,
}

impl UserActionObserverRegistry {
    /// Creates a new observer registry.
    pub fn new() -> Self {
        Self {
            observers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Adds an observer to the registry.
    ///
    /// # Arguments
    ///
    /// * `observer` - The observer to add
    pub fn register(&self, observer: Arc<dyn UserActionObserver>) {
        let mut observers = self
            .observers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        observers.push(observer);
    }

    /// Notifies all observers of a user action.
    ///
    /// # Arguments
    ///
    /// * `event` - The user action event
    pub fn notify(&self, event: &UserActionEvent) {
        let observers = self
            .observers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for observer in observers.iter() {
            observer.on_action(event);
        }
    }

    /// Notifies all observers of a batch of user actions.
    ///
    /// # Arguments
    ///
    /// * `events` - Slice of user action events
    pub fn notify_batch(&self, events: &[UserActionEvent]) {
        let observers = self
            .observers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for observer in observers.iter() {
            observer.on_action_batch(events);
        }
    }

    /// Returns the number of registered observers.
    pub fn observer_count(&self) -> usize {
        self.observers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

impl Default for UserActionObserverRegistry {
    fn default() -> Self {
        Self::new()
    }
}
