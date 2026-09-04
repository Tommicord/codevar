//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.
//!
use super::wredit_base_writable::{WritableEvent, WritableEventType};
use crate::base::Daemon;
use crate::base::base_comm::{Receiver, Sender};

/// `pub trait EventBridge`
///
/// Trait for event bridging between writable data structures and external systems
///
/// This trait defines the interface for event emission and reception,
/// allowing dependency injection and reducing coupling between components
pub trait EventBridge {
    /// Emits a writable event synchronously
    ///
    /// # Arguments
    ///
    /// * `event` - The event to emit
    ///
    /// # Returns
    ///
    /// A task that completes when the event is emitted
    fn emit_event(&self, event: WritableEvent) -> Daemon<()>;

    /// Checks if the bridge is active
    ///
    /// # Returns
    ///
    /// `true` if the bridge is active and can emit events
    fn is_active(&self) -> bool;
}

/// `pub struct ChannelEventBridge`
///
/// Channel-based event bridge implementation using manual channels
///
/// This implementation uses the existing Sender/Receiver channel system
/// for event emission, providing a concrete implementation of EventBridge
#[repr(C)]
pub struct ChannelEventBridge {
    /// Event sender
    sender: Option<Sender<WritableEvent>>,
    /// Event receiver
    receiver: Option<Receiver<WritableEvent>>,
}

impl ChannelEventBridge {
    /// Creates a new channel event bridge
    ///
    /// # Returns
    ///
    /// A new channel event bridge with a channel pair

    pub fn new() -> Self {
        let (sender, receiver) = crate::base::base_comm::channel(100);
        Self {
            sender: Some(sender),
            receiver: Some(receiver),
        }
    }

    /// Creates a new channel event bridge with the specified sender
    ///
    /// # Arguments
    ///
    /// * `sender` - The event sender to use
    ///
    /// # Returns
    ///
    /// A new channel event bridge with the provided sender

    pub fn with_sender(sender: Sender<WritableEvent>) -> Self {
        Self {
            sender: Some(sender),
            receiver: None,
        }
    }

    /// Returns the event sender

    pub fn sender(&self) -> Option<&Sender<WritableEvent>> {
        self.sender.as_ref()
    }

    /// Returns the event receiver

    pub fn receiver(&self) -> Option<&Receiver<WritableEvent>> {
        self.receiver.as_ref()
    }

    /// Returns the mutable event receiver

    pub fn receiver_mut(&mut self) -> Option<&mut Receiver<WritableEvent>> {
        self.receiver.as_mut()
    }

    /// Sets the event sender
    ///
    /// # Arguments
    ///
    /// * `sender` - The event sender to set

    pub fn set_sender(&mut self, sender: Sender<WritableEvent>) {
        self.sender = Some(sender);
    }

    /// Emits a buffer changed event
    ///
    /// # Arguments
    ///
    /// * `message` - The message to include in the event
    ///
    /// # Returns
    ///
    /// A task that completes when the event is emitted

    pub fn emit_buffer_changed(&self, message: String) -> Daemon<()> {
        if self.is_active() {
            let mut event = WritableEvent::new(WritableEventType::BufferChanged);
            event.message = message;
            self.emit_event(event)
        } else {
            Daemon::new(async move {})
        }
    }

    /// Emits a cursor moved event
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID
    /// * `row` - The row position
    /// * `col` - The column position
    ///
    /// # Returns
    ///
    /// A task that completes when the event is emitted

    pub fn emit_cursor_moved(
        &self,
        cursor_id: usize,
        row: usize,
        col: usize,
    ) -> Daemon<()> {
        if self.is_active() {
            let mut event = WritableEvent::new(WritableEventType::CursorMoved);
            event.cursor_id = cursor_id;
            event.row = row;
            event.col = col;
            self.emit_event(event)
        } else {
            Daemon::new(async move {})
        }
    }

    /// Emits a cursor selection changed event
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID
    /// * `row` - The row position
    /// * `col` - The column position
    /// * `start` - The selection start
    /// * `end` - The selection end
    ///
    /// # Returns
    ///
    /// A task that completes when the event is emitted

    pub fn emit_cursor_selection_changed(
        &self,
        cursor_id: usize,
        row: usize,
        col: usize,
        start: usize,
        end: usize,
    ) -> Daemon<()> {
        if self.is_active() {
            let mut event = WritableEvent::new(WritableEventType::CursorSelectionChanged);
            event.cursor_id = cursor_id;
            event.row = row;
            event.col = col;
            event.start = start;
            event.end = end;
            self.emit_event(event)
        } else {
            Daemon::new(async move {})
        }
    }

    /// Emits a cursor added event
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID
    ///
    /// # Returns
    ///
    /// A task that completes when the event is emitted

    pub fn emit_cursor_added(&self, cursor_id: usize) -> Daemon<()> {
        if self.is_active() {
            let mut event = WritableEvent::new(WritableEventType::CursorAdded);
            event.cursor_id = cursor_id;
            self.emit_event(event)
        } else {
            Daemon::new(async move {})
        }
    }

    /// Emits a cursor removed event
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID
    ///
    /// # Returns
    ///
    /// A task that completes when the event is emitted

    pub fn emit_cursor_removed(&self, cursor_id: usize) -> Daemon<()> {
        if self.is_active() {
            let mut event = WritableEvent::new(WritableEventType::CursorRemoved);
            event.cursor_id = cursor_id;
            self.emit_event(event)
        } else {
            Daemon::new(async move {})
        }
    }

    /// Emits a gap block shifted event
    ///
    /// # Arguments
    ///
    /// * `gap_id` - The gap ID
    /// * `offset` - The offset
    /// * `size` - The size
    ///
    /// # Returns
    ///
    /// A task that completes when the event is emitted

    pub fn emit_gap_block_shifted(
        &self,
        gap_id: usize,
        offset: usize,
        size: usize,
    ) -> Daemon<()> {
        if self.is_active() {
            let mut event = WritableEvent::new(WritableEventType::GapBlockShifted);
            event.cursor_id = gap_id;
            event.offset = offset;
            event.size = size;
            self.emit_event(event)
        } else {
            Daemon::new(async move {})
        }
    }

    /// Emits an error event
    ///
    /// # Arguments
    ///
    /// * `error` - The error message
    ///
    /// # Returns
    ///
    /// A task that completes when the event is emitted

    pub fn emit_error(&self, error: String) -> Daemon<()> {
        if self.is_active() {
            let mut event = WritableEvent::new(WritableEventType::Error);
            event.message = error;
            self.emit_event(event)
        } else {
            Daemon::new(async move {})
        }
    }
}

impl EventBridge for ChannelEventBridge {
    fn emit_event(&self, event: WritableEvent) -> Daemon<()> {
        if let Some(ref sender) = self.sender {
            let sender = sender.clone();
            Daemon::new(async move {
                let _ = sender.send(event);
            })
        } else {
            Daemon::new(async move {})
        }
    }

    fn is_active(&self) -> bool {
        self.sender.as_ref().map_or(false, |s| s.is_active())
    }
}

impl Default for ChannelEventBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// `pub struct NullEventBridge`
///
/// Null event bridge implementation that discards all events
///
/// This implementation is useful for testing or when event emission
/// is not required, providing a no-op implementation of EventBridge
#[repr(C)]
pub struct NullEventBridge;

impl NullEventBridge {
    /// Creates a new null event bridge

    pub const fn new() -> Self {
        Self
    }
}

impl EventBridge for NullEventBridge {
    fn emit_event(&self, _event: WritableEvent) -> Daemon<()> {
        Daemon::new(async move {})
    }

    fn is_active(&self) -> bool {
        false
    }
}

impl Default for NullEventBridge {
    fn default() -> Self {
        Self::new()
    }
}
