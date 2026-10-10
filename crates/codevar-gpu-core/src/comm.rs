//! Copyright 2026 Codevar Project
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

//! Blocking-free messaging between the compositor and the presentation
//! side: bounded channels, the [`CompositorComm`] trait, its futures and
//! the two endpoints created by [`comm_pair`].

use alloc::collections::VecDeque;
use alloc::rc::Rc;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use spin::Mutex;

use crate::base::{Compositor, PipeHandoff};
use crate::types::{CommError, CompositorMessage};

/// Depth of each [`CompositorComm`] channel before a non-blocking send
/// reports [`CommError::Full`].
const CHANNEL_CAPACITY: usize = 64;

/// One end of the four channels connecting both sides of a frame.
///
/// Channels are bounded (see [`CHANNEL_CAPACITY`]) and their senders
/// never block: a full channel reports [`CommError::Full`] so the caller
/// can retry, which keeps the compositor's hot paths allocation-free.
struct Channel<T> {
    /// Maximum number of queued values.
    capacity: usize,
    /// Queue plus the parked task wakers.
    state: Mutex<ChannelState<T>>,
}

/// Contents of a [`Channel`] behind its mutex.
struct ChannelState<T> {
    /// Values waiting for the receiver.
    queue: VecDeque<T>,
    /// Set once the hub is closed; further sends fail.
    closed: bool,
    /// Receiver task to wake when a value arrives.
    recv_waker: Option<Waker>,
    /// Sender task to wake when capacity frees up.
    send_waker: Option<Waker>,
}

impl<T> Channel<T> {
    /// Creates an empty channel with the given capacity.
    const fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(ChannelState {
                queue: VecDeque::new(),
                closed: false,
                recv_waker: None,
                send_waker: None,
            }),
        }
    }

    /// Enqueues `value` if capacity allows, waking the receiver.
    fn try_send(&self, value: T) -> Result<(), CommError<T>> {
        let receiver = {
            let mut state = self.state.lock();
            if state.closed {
                return Err(CommError::Closed(value));
            }
            if state.queue.len() >= self.capacity {
                return Err(CommError::Full(value));
            }
            state.queue.push_back(value);
            state.recv_waker.take()
        };
        // Woken outside the lock: `spin::Mutex` is not reentrant, so a
        // waker that polls this channel again must not find it locked.
        if let Some(waker) = receiver {
            waker.wake();
        }
        Ok(())
    }

    /// Dequeues the oldest value, waking a parked sender.
    fn try_recv(&self) -> Option<T> {
        let (value, sender) = {
            let mut state = self.state.lock();
            let value = state.queue.pop_front();
            let sender = if value.is_some() {
                state.send_waker.take()
            } else {
                None
            };
            (value, sender)
        };
        if let Some(waker) = sender {
            waker.wake();
        }
        value
    }

    /// Parks the receiver until a value arrives or the hub closes.
    fn poll_recv(&self, cx: &mut Context<'_>) -> Poll<Option<T>> {
        let (outcome, sender) = {
            let mut state = self.state.lock();
            if let Some(value) = state.queue.pop_front() {
                (Poll::Ready(Some(value)), state.send_waker.take())
            } else if state.closed {
                (Poll::Ready(None), None)
            } else {
                state.recv_waker = Some(cx.waker().clone());
                (Poll::Pending, None)
            }
        };
        if let Some(waker) = sender {
            waker.wake();
        }
        outcome
    }

    /// Parks the sender while `value` is held for the next attempt.
    fn poll_send(&self, cx: &mut Context<'_>, value: &mut Option<T>) -> Poll<Result<(), CommError<T>>> {
        let (outcome, receiver) = {
            let mut state = self.state.lock();
            if state.closed {
                let result = match value.take() {
                    Some(held) => Err(CommError::Closed(held)),
                    None => Ok(()),
                };
                (Poll::Ready(result), None)
            } else if state.queue.len() < self.capacity {
                if let Some(held) = value.take() {
                    state.queue.push_back(held);
                }
                (Poll::Ready(Ok(())), state.recv_waker.take())
            } else {
                state.send_waker = Some(cx.waker().clone());
                (Poll::Pending, None)
            }
        };
        if let Some(waker) = receiver {
            waker.wake();
        }
        outcome
    }

    /// Closes the channel: receivers drain what is queued, senders get
    /// [`CommError::Closed`] and parked tasks are woken.
    fn close(&self) {
        let (receiver, sender) = {
            let mut state = self.state.lock();
            if state.closed {
                return;
            }
            state.closed = true;
            (state.recv_waker.take(), state.send_waker.take())
        };
        if let Some(waker) = receiver {
            waker.wake();
        }
        if let Some(waker) = sender {
            waker.wake();
        }
    }
}

/// The four channels shared by both endpoints.
///
/// `to_compositor` carries messages and pipes destined for the frame
/// loop, `to_presentation` the ones coming back from it.
pub(crate) struct CommHub {
    /// Messages the presentation side sends to the compositor.
    to_compositor_messages: Channel<CompositorMessage>,
    /// Pipes the presentation side registers with the compositor.
    to_compositor_pipes: Channel<PipeHandoff>,
    /// Messages the compositor sends to the presentation side.
    to_presentation_messages: Channel<CompositorMessage>,
    /// Pipes retired by the compositor, handed back for release.
    to_presentation_pipes: Channel<PipeHandoff>,
}

impl CommHub {
    /// Creates an open hub with [`CHANNEL_CAPACITY`] deep channels.
    const fn new() -> Self {
        Self {
            to_compositor_messages: Channel::new(CHANNEL_CAPACITY),
            to_compositor_pipes: Channel::new(CHANNEL_CAPACITY),
            to_presentation_messages: Channel::new(CHANNEL_CAPACITY),
            to_presentation_pipes: Channel::new(CHANNEL_CAPACITY),
        }
    }

    /// Closes every channel.
    fn close(&self) {
        self.to_compositor_messages.close();
        self.to_compositor_pipes.close();
        self.to_presentation_messages.close();
        self.to_presentation_pipes.close();
    }
}

/// Blocking-free messaging between the two ends of a frame.
///
/// Every method has a non-blocking form (`send_*`/`poll_recv_*`) and an
/// async form (`send_*_async`/`recv_*`); nothing ever blocks a thread.
/// Completed futures stay completed when polled again, so executors may
/// poll them after completion without side effects.
///
/// # Not `Send`
///
/// The endpoints own queued [`PipeHandoff`]s — type-erased renderers
/// bound to one Vulkan device — so they deliberately do not implement
/// `Send`/`Sync`: moving an endpoint to another thread would move those
/// renderers with it.
pub trait CompositorComm {
    /// Enqueues `message` without waiting.
    fn send_message(&self, message: CompositorMessage) -> Result<(), CommError<CompositorMessage>>;

    /// Future form of [`CompositorComm::send_message`].
    fn send_message_async(&self, message: CompositorMessage) -> SendFuture<'_, CompositorMessage>;

    /// Polls for a message without parking the task.
    fn poll_recv_message(&self, cx: &mut Context<'_>) -> Poll<Option<CompositorMessage>>;

    /// Future form of [`CompositorComm::poll_recv_message`].
    fn recv_message(&self) -> RecvFuture<'_, CompositorMessage>;

    /// Takes the next message without parking the task.
    fn try_recv_message(&self) -> Option<CompositorMessage>;

    /// Enqueues a pipe to register without waiting.
    fn send_pipe(&self, pipe: PipeHandoff) -> Result<(), CommError<PipeHandoff>>;

    /// Future form of [`CompositorComm::send_pipe`].
    fn send_pipe_async(&self, pipe: PipeHandoff) -> SendFuture<'_, PipeHandoff>;

    /// Polls for a pipe without parking the task.
    fn poll_recv_pipe(&self, cx: &mut Context<'_>) -> Poll<Option<PipeHandoff>>;

    /// Future form of [`CompositorComm::poll_recv_pipe`].
    fn recv_pipe(&self) -> RecvFuture<'_, PipeHandoff>;

    /// Takes the next pipe without parking the task.
    fn try_recv_pipe(&self) -> Option<PipeHandoff>;
}

/// Future of a non-blocking send that waits for capacity.
///
/// Yields while the channel is full and stays [`Poll::Ready`] with
/// `Ok(())` once the value was accepted (or the future was polled after
/// completion).
pub struct SendFuture<'a, T> {
    /// Channel the value is pushed into.
    channel: &'a Channel<T>,
    /// Value still waiting for capacity; `None` once sent.
    message: Option<T>,
}

impl<'a, T: Unpin> Future for SendFuture<'a, T> {
    type Output = Result<(), CommError<T>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: `T: Unpin` makes `Self` `Unpin`, so the pinned struct
        // may be accessed through the pin.
        let this = unsafe { self.get_unchecked_mut() };
        if this.message.is_none() {
            return Poll::Ready(Ok(()));
        }
        this.channel.poll_send(cx, &mut this.message)
    }
}

/// Future of a receive that waits for the next value.
///
/// Yields while the channel is empty and stays [`Poll::Ready`] with
/// `None` once the hub was closed and drained.
pub struct RecvFuture<'a, T> {
    /// Channel the value is taken from.
    channel: &'a Channel<T>,
    /// Set once the future resolved; further polls keep the result.
    done: bool,
}

impl<T> Future for RecvFuture<'_, T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: `Self` only holds a reference and a `bool`, both
        // `Unpin`, so the pin may be unwrapped safely.
        let this = unsafe { self.get_unchecked_mut() };
        if this.done {
            return Poll::Ready(None);
        }
        match this.channel.poll_recv(cx) {
            Poll::Ready(value) => {
                this.done = true;
                Poll::Ready(value)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Endpoint kept by the compositor: receives from the presentation side
/// and sends back to it.
///
/// Created by [`comm_pair`] or fetched from a running compositor with
/// [`Compositor::comm`](Compositor::comm).
#[derive(Clone)]
pub struct CompositorEndpoint {
    /// Shared channels.
    pub(crate) hub: Rc<CommHub>,
}

impl CompositorEndpoint {
    /// The messages addressed to the compositor.
    fn inbound_messages(&self) -> &Channel<CompositorMessage> {
        &self.hub.to_compositor_messages
    }

    /// The pipes addressed to the compositor.
    fn inbound_pipes(&self) -> &Channel<PipeHandoff> {
        &self.hub.to_compositor_pipes
    }

    /// The messages addressed to the presentation side.
    fn outbound_messages(&self) -> &Channel<CompositorMessage> {
        &self.hub.to_presentation_messages
    }

    /// The pipes addressed to the presentation side.
    fn outbound_pipes(&self) -> &Channel<PipeHandoff> {
        &self.hub.to_presentation_pipes
    }

    /// Closes every channel of the pair.
    pub(crate) fn close(&self) {
        self.hub.close();
    }
}

impl CompositorComm for CompositorEndpoint {
    fn send_message(&self, message: CompositorMessage) -> Result<(), CommError<CompositorMessage>> {
        self.outbound_messages().try_send(message)
    }

    fn send_message_async(&self, message: CompositorMessage) -> SendFuture<'_, CompositorMessage> {
        SendFuture {
            channel: self.outbound_messages(),
            message: Some(message),
        }
    }

    fn poll_recv_message(&self, cx: &mut Context<'_>) -> Poll<Option<CompositorMessage>> {
        self.inbound_messages().poll_recv(cx)
    }

    fn recv_message(&self) -> RecvFuture<'_, CompositorMessage> {
        RecvFuture {
            channel: self.inbound_messages(),
            done: false,
        }
    }

    fn try_recv_message(&self) -> Option<CompositorMessage> {
        self.inbound_messages().try_recv()
    }

    fn send_pipe(&self, pipe: PipeHandoff) -> Result<(), CommError<PipeHandoff>> {
        self.outbound_pipes().try_send(pipe)
    }

    fn send_pipe_async(&self, pipe: PipeHandoff) -> SendFuture<'_, PipeHandoff> {
        SendFuture {
            channel: self.outbound_pipes(),
            message: Some(pipe),
        }
    }

    fn poll_recv_pipe(&self, cx: &mut Context<'_>) -> Poll<Option<PipeHandoff>> {
        self.inbound_pipes().poll_recv(cx)
    }

    fn recv_pipe(&self) -> RecvFuture<'_, PipeHandoff> {
        RecvFuture {
            channel: self.inbound_pipes(),
            done: false,
        }
    }

    fn try_recv_pipe(&self) -> Option<PipeHandoff> {
        self.inbound_pipes().try_recv()
    }
}

/// Endpoint kept by the presentation side: mirrors
/// [`CompositorEndpoint`] in the opposite direction.
#[derive(Clone)]
pub struct RendererEndpoint {
    /// Shared channels.
    pub(crate) hub: Rc<CommHub>,
}

impl RendererEndpoint {
    /// The messages addressed to the compositor.
    fn outbound_messages(&self) -> &Channel<CompositorMessage> {
        &self.hub.to_compositor_messages
    }

    /// The pipes addressed to the compositor.
    fn outbound_pipes(&self) -> &Channel<PipeHandoff> {
        &self.hub.to_compositor_pipes
    }

    /// The messages addressed to the presentation side.
    fn inbound_messages(&self) -> &Channel<CompositorMessage> {
        &self.hub.to_presentation_messages
    }

    /// The pipes addressed to the presentation side.
    fn inbound_pipes(&self) -> &Channel<PipeHandoff> {
        &self.hub.to_presentation_pipes
    }
}

impl CompositorComm for RendererEndpoint {
    fn send_message(&self, message: CompositorMessage) -> Result<(), CommError<CompositorMessage>> {
        self.outbound_messages().try_send(message)
    }

    fn send_message_async(&self, message: CompositorMessage) -> SendFuture<'_, CompositorMessage> {
        SendFuture {
            channel: self.outbound_messages(),
            message: Some(message),
        }
    }

    fn poll_recv_message(&self, cx: &mut Context<'_>) -> Poll<Option<CompositorMessage>> {
        self.inbound_messages().poll_recv(cx)
    }

    fn recv_message(&self) -> RecvFuture<'_, CompositorMessage> {
        RecvFuture {
            channel: self.inbound_messages(),
            done: false,
        }
    }

    fn try_recv_message(&self) -> Option<CompositorMessage> {
        self.inbound_messages().try_recv()
    }

    fn send_pipe(&self, pipe: PipeHandoff) -> Result<(), CommError<PipeHandoff>> {
        self.outbound_pipes().try_send(pipe)
    }

    fn send_pipe_async(&self, pipe: PipeHandoff) -> SendFuture<'_, PipeHandoff> {
        SendFuture {
            channel: self.outbound_pipes(),
            message: Some(pipe),
        }
    }

    fn poll_recv_pipe(&self, cx: &mut Context<'_>) -> Poll<Option<PipeHandoff>> {
        self.inbound_pipes().poll_recv(cx)
    }

    fn recv_pipe(&self) -> RecvFuture<'_, PipeHandoff> {
        RecvFuture {
            channel: self.inbound_pipes(),
            done: false,
        }
    }

    fn try_recv_pipe(&self) -> Option<PipeHandoff> {
        self.inbound_pipes().try_recv()
    }
}

/// Creates a connected pair of endpoints with fresh channels.
#[must_use]
pub fn comm_pair() -> (CompositorEndpoint, RendererEndpoint) {
    let hub = Rc::new(CommHub::new());
    (
        CompositorEndpoint { hub: Rc::clone(&hub) },
        RendererEndpoint { hub },
    )
}

#[cfg(test)]
mod tests {
    //! Channel semantics, future completion and endpoint routing.

    use super::*;

    use crate::pipe::test_util::DummyRenderer;

    #[test]
    fn channel_reports_full_and_receives_in_order() {
        let channel = Channel::<u32>::new(2);
        channel.try_send(1).unwrap();
        channel.try_send(2).unwrap();
        match channel.try_send(3) {
            Err(CommError::Full(value)) => assert_eq!(value, 3),
            other => panic!("expected Full, got {other:?}"),
        }
        assert_eq!(channel.try_recv(), Some(1));
        assert_eq!(channel.try_recv(), Some(2));
        assert_eq!(channel.try_recv(), None);
    }

    #[test]
    fn channel_close_drains_for_receivers_and_rejects_senders() {
        let channel = Channel::<u32>::new(4);
        channel.try_send(7).unwrap();
        channel.close();
        assert!(matches!(channel.try_send(8), Err(CommError::Closed(8))));
        assert_eq!(channel.try_recv(), Some(7));
        assert_eq!(channel.try_recv(), None);
    }

    #[test]
    fn channel_futures_complete_under_a_noop_waker() {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);

        // A send into a full channel yields until a slot frees up, then
        // stays ready when polled again.
        let full = Channel::<u32>::new(1);
        full.try_send(1).unwrap();
        let mut send = SendFuture {
            channel: &full,
            message: Some(2),
        };
        assert!(matches!(Pin::new(&mut send).poll(&mut cx), Poll::Pending));
        assert_eq!(full.try_recv(), Some(1));
        assert!(matches!(Pin::new(&mut send).poll(&mut cx), Poll::Ready(Ok(()))));
        assert!(matches!(Pin::new(&mut send).poll(&mut cx), Poll::Ready(Ok(()))));
        assert_eq!(full.try_recv(), Some(2));

        // A receive on an empty channel yields until a value arrives,
        // then fuses to `None`.
        let channel = Channel::<u32>::new(1);
        let mut recv = RecvFuture {
            channel: &channel,
            done: false,
        };
        assert!(matches!(Pin::new(&mut recv).poll(&mut cx), Poll::Pending));
        channel.try_send(9).unwrap();
        assert!(matches!(Pin::new(&mut recv).poll(&mut cx), Poll::Ready(Some(9))));
        assert!(matches!(Pin::new(&mut recv).poll(&mut cx), Poll::Ready(None)));
    }

    #[test]
    fn comm_pair_routes_messages_and_pipes_both_ways() {
        let (compositor, renderer) = comm_pair();

        renderer
            .send_message(CompositorMessage::VisibilityChanged { visible: false })
            .unwrap();
        assert!(matches!(
            compositor.try_recv_message(),
            Some(CompositorMessage::VisibilityChanged { visible: false })
        ));

        compositor
            .send_message(CompositorMessage::FrameComposited { frame_index: 3 })
            .unwrap();
        assert!(matches!(
            renderer.try_recv_message(),
            Some(CompositorMessage::FrameComposited { frame_index: 3 })
        ));

        // Registration flows presentation → compositor; retirement
        // flows back compositor → presentation.
        renderer
            .send_pipe(PipeHandoff::from_renderer(DummyRenderer::empty()))
            .unwrap();
        assert!(compositor.try_recv_pipe().is_some());
        compositor
            .send_pipe(PipeHandoff::from_renderer(DummyRenderer::empty()))
            .unwrap();
        assert!(renderer.try_recv_pipe().is_some());
    }
}
