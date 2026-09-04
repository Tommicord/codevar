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

//! Action queue for managing user action events.
//!
//! This module provides a queue-based system for managing user action events,
//! enabling batched processing and efficient action handling for collaborative features.

use codevar_core::edit::UserActionEvent;
use std::collections::VecDeque;

/// Queue for managing user action events.
///
/// This queue provides FIFO ordering for user actions, enabling batched
/// processing and efficient action management for collaborative features.
pub struct ActionQueue {
    /// The underlying deque storing actions
    queue: VecDeque<UserActionEvent>,
    /// Maximum queue size before forced flush
    max_size: usize,
}

impl ActionQueue {
    /// Creates a new action queue with default settings.
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            max_size: 1000,
        }
    }

    /// Creates a new action queue with specified max size.
    ///
    /// # Arguments
    ///
    /// * `max_size` - Maximum queue size before forced flush
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            max_size,
        }
    }

    /// Adds a user action event to the queue.
    ///
    /// # Arguments
    ///
    /// * `event` - The user action event to add
    ///
    /// # Returns
    ///
    /// Option containing the oldest action if the queue is full, None otherwise.
    pub fn push(&mut self, event: UserActionEvent) -> Option<UserActionEvent> {
        if self.queue.len() >= self.max_size {
            // Remove oldest action when queue is full
            let oldest = self.queue.pop_front();
            self.queue.push_back(event);
            oldest
        } else {
            self.queue.push_back(event);
            None
        }
    }

    /// Removes and returns the oldest action from the queue.
    ///
    /// # Returns
    ///
    /// Option containing the oldest action, or None if the queue is empty.
    pub fn pop(&mut self) -> Option<UserActionEvent> {
        self.queue.pop_front()
    }

    /// Peeks at the oldest action without removing it.
    ///
    /// # Returns
    ///
    /// Option containing a reference to the oldest action, or None if the queue is empty.
    pub fn peek(&self) -> Option<&UserActionEvent> {
        self.queue.front()
    }

    /// Peeks at the newest action without removing it.
    ///
    /// # Returns
    ///
    /// Option containing a reference to the newest action, or None if the queue is empty.
    pub fn peek_back(&self) -> Option<&UserActionEvent> {
        self.queue.back()
    }

    /// Returns the number of actions in the queue.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns true if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Clears all actions from the queue.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Drains all actions from the queue and returns them.
    ///
    /// # Returns
    ///
    /// A vector containing all actions that were in the queue.
    pub fn drain(&mut self) -> Vec<UserActionEvent> {
        self.queue.drain(..).collect()
    }

    /// Returns the maximum queue size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Sets the maximum queue size.
    ///
    /// # Arguments
    ///
    /// * `max_size` - The new maximum queue size
    pub fn set_max_size(&mut self, max_size: usize) {
        self.max_size = max_size;
    }

    /// Drains actions up to the specified count.
    ///
    /// # Arguments
    ///
    /// * `count` - Maximum number of actions to drain
    ///
    /// # Returns
    ///
    /// A vector containing up to `count` actions from the front of the queue.
    pub fn drain_up_to(&mut self, count: usize) -> Vec<UserActionEvent> {
        let drain_count = count.min(self.queue.len());
        self.queue.drain(0..drain_count).collect()
    }
}

impl Default for ActionQueue {
    fn default() -> Self {
        Self::new()
    }
}
