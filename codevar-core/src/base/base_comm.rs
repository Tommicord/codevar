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

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::task::Waker;

/// Error type for channel send operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SendError<T> {
    /// The channel is closed and no more messages can be sent.
    Closed(T),
    /// The channel is full and the send would block.
    Full(T),
}

/// Error type for channel receive operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryRecvError {
    /// The channel is empty.
    Empty,
    /// The channel is closed and no more messages will be sent.
    Disconnected,
}

/// Sender for channel communication.
#[repr(C)]
pub struct Sender<T> {
    inner: Arc<Mutex<ChannelInner<T>>>,
}

impl<T> Sender<T> {
    /// Creates a new sender from channel inner state.
    pub fn new(inner: Arc<Mutex<ChannelInner<T>>>) -> Self {
        Self { inner }
    }
}

impl<T: Clone> Sender<T> {
    /// Sends a value synchronously (returns error if channel is full).
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        let mut channel = self
            .inner
            .lock()
            .map_err(|_| SendError::Closed(value.clone()))?;
        if channel.is_closed {
            return Err(SendError::Closed(value));
        }
        if channel.buffer.len() >= channel.capacity {
            return Err(SendError::Full(value));
        }
        channel.buffer.push_back(value);
        if let Some(waker) = channel.receiver_waker.take() {
            waker.wake();
        }

        Ok(())
    }

    /// Attempts to send a value without blocking.
    pub fn try_send(&self, value: T) -> Result<(), std::sync::mpsc::TrySendError<T>> {
        let mut channel = self
            .inner
            .lock()
            .map_err(|_| std::sync::mpsc::TrySendError::Disconnected(value.clone()))?;
        if channel.is_closed {
            return Err(std::sync::mpsc::TrySendError::Disconnected(value));
        }
        if channel.buffer.len() >= channel.capacity {
            return Err(std::sync::mpsc::TrySendError::Full(value));
        }
        channel.buffer.push_back(value);
        if let Some(waker) = channel.receiver_waker.take() {
            waker.wake();
        }
        Ok(())
    }

    /// Checks if the sender is still connected to a receiver.
    pub fn is_active(&self) -> bool {
        let channel = self.inner.lock().ok();
        channel.map_or(false, |c| !c.is_closed)
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Receiver for channel communication.
#[repr(C)]
pub struct Receiver<T> {
    inner: Arc<Mutex<ChannelInner<T>>>,
}

impl<T> Receiver<T> {
    /// Creates a new receiver from channel inner state.
    pub fn new(inner: Arc<Mutex<ChannelInner<T>>>) -> Self {
        Self { inner }
    }

    /// Attempts to receive a value without blocking.
    pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
        let mut channel = self.inner.lock().map_err(|_| TryRecvError::Disconnected)?;
        if let Some(value) = channel.buffer.pop_front() {
            // Wake up any waiting sender
            if let Some(waker) = channel.sender_waker.take() {
                waker.wake();
            }
            Ok(value)
        } else if channel.is_closed {
            Err(TryRecvError::Disconnected)
        } else {
            Err(TryRecvError::Empty)
        }
    }

    /// Receives a value, blocking if necessary.
    pub fn recv(&mut self) -> Result<T, TryRecvError> {
        loop {
            match self.try_recv() {
                Ok(value) => return Ok(value),
                Err(TryRecvError::Disconnected) => {
                    return Err(TryRecvError::Disconnected);
                }
                Err(TryRecvError::Empty) => {
                    // Channel is empty but not closed, wait for data
                    std::thread::yield_now();
                }
            }
        }
    }
}

/// Inner channel state shared between sender and receiver.
struct ChannelInner<T> {
    /// Message buffer
    buffer: VecDeque<T>,
    /// Channel capacity
    capacity: usize,
    /// Whether the channel is closed
    is_closed: bool,
    /// Waker for waiting sender
    sender_waker: Option<Waker>,
    /// Waker for waiting receiver
    receiver_waker: Option<Waker>,
}

impl<T> ChannelInner<T> {
    /// Creates new channel inner state.
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            is_closed: false,
            sender_waker: None,
            receiver_waker: None,
        }
    }
}

/// Creates a new bounded channel for communication.
pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let inner = Arc::new(Mutex::new(ChannelInner::new(capacity)));
    (Sender::new(inner.clone()), Receiver::new(inner))
}

/// Creates a new bounded channel for communication using the project logger API.
pub fn bounded<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    channel(capacity)
}

/// Creates a new unbounded channel for communication using the project logger API.
pub fn unbounded<T>() -> (Sender<T>, Receiver<T>) {
    let inner = Arc::new(Mutex::new(ChannelInner {
        buffer: VecDeque::new(),
        capacity: usize::MAX,
        is_closed: false,
        sender_waker: None,
        receiver_waker: None,
    }));
    (Sender::new(inner.clone()), Receiver::new(inner))
}
/// Error type for broadcast send operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BroadcastSendError<T> {
    /// No receivers are currently subscribed.
    NoReceivers(T),
}

/// Error type for broadcast receive operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BroadcastRecvError {
    /// All senders have been dropped.
    Closed,
}

/// Broadcast sender for multiple subscribers.
#[repr(C)]
pub struct BroadcastSender<T> {
    inner: Arc<Mutex<BroadcastInner<T>>>,
}

impl<T: Clone> BroadcastSender<T> {
    /// Creates a new broadcast sender from inner state.
    pub fn new(inner: Arc<Mutex<BroadcastInner<T>>>) -> Self {
        Self { inner }
    }

    /// Broadcasts a value to all subscribers.
    pub fn send(&self, value: T) -> Result<usize, BroadcastSendError<T>> {
        let mut broadcast = self
            .inner
            .lock()
            .map_err(|_| BroadcastSendError::NoReceivers(value.clone()))?;
        if broadcast.receivers.is_empty() {
            return Err(BroadcastSendError::NoReceivers(value));
        }
        let mut receiver_count = 0;
        broadcast.receivers.retain(|receiver| {
            if let Ok(mut recv) = receiver.buffer.lock() {
                if recv.len() < receiver.capacity {
                    recv.push_back(value.clone());
                    receiver_count += 1;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        });
        broadcast.latest_value = Some(value);
        Ok(receiver_count)
    }

    /// Gets the number of active receivers.
    pub fn receiver_count(&self) -> usize {
        let broadcast = self.inner.lock().ok();
        broadcast.map_or(0, |b| b.receivers.len())
    }
}

impl<T: Clone> Clone for BroadcastSender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Broadcast receiver for subscribing to broadcasts.
#[repr(C)]
pub struct BroadcastReceiver<T> {
    buffer: Arc<Mutex<VecDeque<T>>>,
    sender: Arc<Mutex<BroadcastInner<T>>>,
}

impl<T: Clone> BroadcastReceiver<T> {
    /// Creates a new broadcast receiver.
    pub fn new(
        buffer: Arc<Mutex<VecDeque<T>>>,
        sender: Arc<Mutex<BroadcastInner<T>>>,
    ) -> Self {
        Self { buffer, sender }
    }

    /// Attempts to receive a broadcasted value without blocking.
    pub fn try_recv(&mut self) -> Result<T, BroadcastRecvError> {
        let mut buffer = self.buffer.lock().map_err(|_| BroadcastRecvError::Closed)?;

        if let Some(value) = buffer.pop_front() {
            Ok(value)
        } else {
            let sender = self.sender.lock().ok();
            if sender.is_none() || sender.map(|s| s.receivers.is_empty()).unwrap_or(true)
            {
                Err(BroadcastRecvError::Closed)
            } else {
                Err(BroadcastRecvError::Closed)
            }
        }
    }

    /// Receives a broadcasted value, blocking if necessary.
    pub fn recv(&mut self) -> Result<T, BroadcastRecvError> {
        loop {
            match self.try_recv() {
                Ok(value) => return Ok(value),
                Err(BroadcastRecvError::Closed) => {
                    std::thread::yield_now();
                }
            }
        }
    }
}

/// Receiver information for broadcast channels.
struct BroadcastReceiverInfo<T> {
    /// Message buffer
    buffer: Arc<Mutex<VecDeque<T>>>,
    /// Buffer capacity
    capacity: usize,
}

/// Inner broadcast state shared between senders and receivers.
struct BroadcastInner<T> {
    /// List of receiver information
    receivers: Vec<BroadcastReceiverInfo<T>>,
    /// Latest value for late subscribers
    latest_value: Option<T>,
}

impl<T> BroadcastInner<T> {
    /// Creates new broadcast inner state.
    fn new() -> Self {
        Self {
            receivers: Vec::new(),
            latest_value: None,
        }
    }

    /// Adds a new receiver.
    fn add_receiver(&mut self, buffer: Arc<Mutex<VecDeque<T>>>, capacity: usize) {
        self.receivers
            .push(BroadcastReceiverInfo { buffer, capacity });
    }
}

/// Creates a new broadcast channel for multiple subscribers.
pub fn broadcast_channel<T: Clone>(
    capacity: usize,
) -> (BroadcastSender<T>, BroadcastReceiver<T>) {
    let inner = Arc::new(Mutex::new(BroadcastInner::new()));
    let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(capacity)));

    {
        let mut broadcast = match inner.lock() {
            Ok(lock) => lock,
            Err(_) => {
                return (
                    BroadcastSender::new(inner.clone()),
                    BroadcastReceiver::new(buffer.clone(), inner.clone()),
                );
            }
        };
        broadcast.add_receiver(buffer.clone(), capacity);
    }
    (
        BroadcastSender::new(inner.clone()),
        BroadcastReceiver::new(buffer, inner.clone()),
    )
}
