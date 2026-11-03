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

//! Getting device results back to the host without stalling the CPU.
//!
//! [`CommandQueue::read_buffer`](crate::CommandQueue::read_buffer) uses
//! the blocking form of `clEnqueueReadBuffer`: the call does not return
//! until the copy landed, which is one forced CPU stall per result. An
//! engine that enqueues many operations per frame cannot afford that, so
//! this module provides the two asynchronous forms OpenCL actually
//! defines:
//!
//! * **Non-blocking copy** — [`read_buffer_non_blocking`](crate::CommandQueue::read_buffer_non_blocking)
//!   submits `clEnqueueReadBuffer` with `CL_FALSE` into a caller-owned
//!   destination and hands back a [`Readback`] handle. The handle keeps
//!   the destination *mutably borrowed* for as long as the copy may be
//!   in flight: the data only becomes reachable through
//!   [`Readback::wait`] (or [`Readback::wait_all`] for a whole batch),
//!   both of which retire the backing event first, and dropping an
//!   unfinished `Readback` blocks until the driver is done with the
//!   pointer instead of handing back aliasable memory.
//! * **Mapped region** — see the map half of this module; zero-copy
//!   results through `clEnqueueMapBuffer` when the driver exposes it.
//!
//! Correctness depends on the returned [`crate::Event`], never on an
//! implicit wait: [`Readback::is_complete`] answers from
//! `clGetEventInfo`'s execution-status query, and batch completion is a
//! single [`crate::Event::wait_all`] over the whole set of outstanding
//! handles.
//!
//! # Examples
//!
//! ```
//! use codevar_ocl::{Buffer, Context, Runtime};
//!
//! let Ok(runtime) = Runtime::load() else {
//!     return;
//! };
//! let Some(platform) = runtime.platforms().first() else {
//!     return;
//! };
//! let Some(device) = platform.all_devices().ok().and_then(|d| d.into_iter().next()) else {
//!     return;
//! };
//! let Ok(context) = Context::new(&device) else {
//!     return;
//! };
//! let Ok(queue) = codevar_ocl::CommandQueue::new(&context, &device) else {
//!     return;
//! };
//! let Ok(buffer) = Buffer::from_slice(&context, &[1u32, 2, 3, 4]) else {
//!     return;
//! };
//! let mut out = [0u32; 4];
//! // Submit without blocking; the queue stays open for more work.
//! let Ok(readback) = queue.read_buffer_non_blocking(&buffer, 0, &mut out, &[]) else {
//!     return;
//! };
//! queue.flush().ok();
//! // The copy owns `out` until its event retired; `wait` returns the
//! // data only after that.
//! let Ok(data) = readback.wait() else {
//!     return;
//! };
//! assert_eq!(data, &[1, 2, 3, 4]);
//! ```

use alloc::vec::Vec;
use core::fmt;
use core::mem::size_of_val;
use core::ptr::null_mut;

use codevar_logger::log_warn;

use crate::buffer::{Buffer, check_range};
use crate::error::{self, Error, Result};
use crate::event::{CommandState, Event};
use crate::queue::CommandQueue;
use crate::sys;

/// Validates the wait list of an asynchronous command and collects the
/// raw event handles in submission order.
///
/// Every event must belong to the queue's context: an event of another
/// context (or another implementation entirely) in the list is undefined
/// at the driver level, so it is rejected here with
/// [`Error::ContextMismatch`].
///
/// # Errors
///
/// Also returns [`Error::InvalidArgument`] when the list is longer than
/// `u32::MAX` entries.
pub(crate) fn event_wait_handles(
    queue: &CommandQueue,
    wait_list: &[&Event],
) -> Result<Vec<sys::EventHandle>> {
    let count = wait_list.len();
    if count > u32::MAX as usize {
        return Err(Error::InvalidArgument {
            what: "an event wait list must hold at most u32::MAX events",
        });
    }
    for event in wait_list {
        queue.check_context(
            event.context(),
            "an event in the wait list belongs to a different context",
        )?;
    }
    Ok(wait_list
        .iter()
        .map(|event| event.raw())
        .collect())
}

/// The element-pointer of a wait-handle array for the FFI calls.
///
/// OpenCL ignores the list when the count is zero, so an empty vector
/// passes null — the same shape the queue's blocking transfers use.
pub(crate) fn wait_list_ptr(handles: &[sys::EventHandle]) -> *const sys::EventHandle {
    if handles.is_empty() {
        core::ptr::null()
    } else {
        handles.as_ptr()
    }
}

/// An outstanding non-blocking readback into caller-owned memory.
///
/// The handle is the *proof of in flight*: it mutably borrows the
/// destination slice until the backing command retires, so the copied
/// bytes cannot be read, written, moved or freed while the device still
/// owns the pointer. The destination becomes reachable only through
/// [`wait`](Self::wait) or [`wait_all`](Self::wait_all), both of which
/// wait on the event first, or — for the status-query path — after
/// [`is_complete`](Self::is_complete) reported `true` **and** the
/// command's event retired.
///
/// Dropping a `Readback` whose destination was never taken blocks on the
/// event (best effort; failures are logged) so an abandoned handle can
/// never leave the caller with memory the driver is still writing.
pub struct Readback<'a, T> {
    /// The read command's event; also keeps the queue and its context
    /// alive for as long as the copy may run.
    event: Event,
    /// `Some` while the destination is parked in the handle. `wait` and
    /// `wait_all` take it after the event retired; the `Drop` guard uses
    /// it to decide whether a wait is still owed.
    dest: Option<&'a mut [T]>,
}

// The auto trait derivations are the intended ones: `Readback` is `Send`
// when `T: Send` (the destination moves with it), and never `Sync`,
// because a shared `Readback` would hand two threads the same in-flight
// destination.

impl<'a, T> Readback<'a, T> {
    /// Returns the event that gates this copy.
    ///
    /// The event is the completion condition for the destination: the
    /// copied bytes are only well-defined after it reports
    /// [`CommandState::Complete`].
    #[must_use]
    pub fn event(&self) -> &Event {
        &self.event
    }

    /// Queries the driver for the copy's completion status
    /// (`clGetEventInfo`).
    ///
    /// This is a pure status query: it does **not** flush the queue, so
    /// a read that was never flushed (see
    /// [`CommandQueue::flush`](crate::CommandQueue::flush)) stays
    /// `Queued` and polls `false` forever. An errored command surfaces
    /// as [`Error::Status`] carrying the driver's failure code rather
    /// than a `CommandState`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the query fails or the command
    /// finished with an error status.
    pub fn is_complete(&self) -> Result<bool> {
        Ok(self.event.state()? == CommandState::Complete)
    }

    /// Waits on the read's event and then hands back the destination.
    ///
    /// Only the event, not the enqueue call, decides when the bytes are
    /// valid: this flushes the queue, blocks until the copy retired, and
    /// releases the borrow on the destination afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the flush or the wait fails
    /// (including a copy whose command ended with an error status). The
    /// command is retired when the wait returns at all, so a failed
    /// `wait` still leaves the destination free for the caller.
    pub fn wait(mut self) -> Result<&'a mut [T]> {
        let result = self.event.wait();
        // Park the destination out of the handle first: once the wait
        // returned (success or error) the command retired and taking it
        // is the caller's data back; `Drop` then sees `None` and skips
        // its guard wait.
        let dest = self.dest.take();
        result?;
        dest.ok_or(Error::InvalidArgument {
            what: "the readback destination was already taken",
        })
    }

    /// Waits for a whole batch of readbacks with one flush per queue and
    /// one `clWaitForEvents`, then returns every destination.
    ///
    /// This is the batching entry point the frame-loop use case needs:
    /// N enqueued reads, one CPU stall. On failure — a command ended
    /// with an error status — the remaining readbacks are dropped, and
    /// each drop guard retires its own copy, so no destination is ever
    /// released while in flight.
    ///
    /// # Errors
    ///
    /// See [`Event::wait_all`](crate::Event::wait_all) and
    /// [`Readback::wait`].
    pub fn wait_all(readbacks: Vec<Self>) -> Result<Vec<&'a mut [T]>> {
        let events: Vec<&Event> = readbacks.iter().map(Self::event).collect();
        Event::wait_all(&events)?;
        let mut data = Vec::with_capacity(readbacks.len());
        for mut readback in readbacks {
            // The wait above retired every command, so taking the
            // destination is sound here and neutralises the drop guard.
            if let Some(dest) = readback.dest.take() {
                data.push(dest);
            }
        }
        Ok(data)
    }
}

impl<T> Drop for Readback<'_, T> {
    fn drop(&mut self) {
        // The destination is still parked in the handle: the driver may
        // hold a live pointer into it, so the borrow may not end before
        // the command retired. Blocking here is the contract that makes
        // abandoning a readback safe; the wait itself can only fail
        // because a command errored, which also retires it.
        if self.dest.is_some()
            && let Err(error) = self.event.wait()
        {
            log_warn!("abandoned readback wait on drop failed: {error}");
        }
    }
}

impl<T> fmt::Debug for Readback<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Readback")
            .field("event", &self.event)
            .field("destination_parked", &self.dest.is_some())
            .finish_non_exhaustive()
    }
}

impl CommandQueue {
    /// Starts a non-blocking read of `buffer` into `data`
    /// (`clEnqueueReadBuffer` with `CL_FALSE`).
    ///
    /// The call returns as soon as the driver queued the copy — the CPU
    /// never stalls on the device — and ownership of `data` transfers to
    /// the returned [`Readback`] for as long as the copy may run. The
    /// read depends on every command in `wait_list` (typically the
    /// kernel launch that produced the results) rather than on queue
    /// position alone, so the submission remains correct on an
    /// out-of-order queue; pass `&[]` for in-order queues like the ones
    /// this crate creates.
    ///
    /// Submit one or more reads, [`flush`](Self::flush) once, then
    /// retire the batch with [`Readback::wait_all`] — N stalls become
    /// one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] when the buffer or a wait-list
    /// event belongs to another context, [`Error::InvalidArgument`] for
    /// an out-of-range region or an over-long wait list, and
    /// [`Error::Status`] / [`Error::NullHandle`] on driver failures.
    pub fn read_buffer_non_blocking<'a, T>(
        &self,
        buffer: &Buffer,
        offset_bytes: usize,
        data: &'a mut [T],
        wait_list: &[&Event],
    ) -> Result<Readback<'a, T>> {
        self.check_context(buffer.context(), "the read buffer belongs to a different context")?;
        let handles = event_wait_handles(self, wait_list)?;
        let size = size_of_val(data);
        check_range(buffer.size(), offset_bytes, size)?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        // SAFETY: all handles are live and belong to the same context
        // (checked above); `data` points at `size` writable bytes whose
        // borrow moves into the returned `Readback`, which cannot be
        // dropped or relinquished before the command retired — either
        // `wait`/`wait_all` took the destination after the event
        // completed, or the `Drop` guard blocked on the same event;
        // `handles` outlives the call and the list is null exactly when
        // empty; `event` is a valid out-pointer.
        let code = unsafe {
            (self.api().enqueue_read_buffer)(
                self.raw(),
                buffer.raw(),
                sys::NON_BLOCKING,
                offset_bytes,
                size,
                data.as_mut_ptr().cast(),
                handles.len() as u32,
                wait_list_ptr(&handles),
                &mut event,
            )
        };
        let raw = error::creation(event, code, "clEnqueueReadBuffer")?;
        Ok(Readback {
            event: Event::new(self, raw),
            dest: Some(data),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, Runtime};
    use alloc::vec;

    /// A queue over the built-in driver or the installed one, whichever
    /// answers first; `None` when the machine exposes no usable device.
    fn queue() -> Option<(Context, CommandQueue)> {
        let runtime = Runtime::load().ok()?;
        let device = runtime
            .platforms()
            .first()
            .and_then(|platform| platform.all_devices().ok())
            .and_then(|devices| devices.into_iter().next())?;
        let context = Context::new(&device).ok()?;
        let queue = CommandQueue::new(&context, &device).ok()?;
        Some((context, queue))
    }

    /// Whether the test is running on the built-in CPU driver, whose
    /// copies complete inside the enqueue call.
    fn is_bare(queue: &CommandQueue) -> bool {
        queue
            .context()
            .device()
            .name()
            .is_ok_and(|name| name == "codevar-cpu")
    }

    /// A non-blocking read whose data only appears after the event
    /// retired: the destination must contain the buffer contents once
    /// `wait` released it.
    #[test]
    fn non_blocking_read_delivers_after_the_event_retires() {
        let Some((context, queue)) = queue() else {
            return;
        };
        let source = vec![7u32; 32];
        let buffer = Buffer::from_slice(&context, &source).expect("upload");
        let mut out = vec![0u32; 32];
        let readback = queue
            .read_buffer_non_blocking(&buffer, 0, &mut out, &[])
            .expect("non-blocking read submits");
        queue.flush().expect("flush");
        let data = readback.wait().expect("the copy completes");
        assert_eq!(data, &source[..]);
    }

    /// The status-query path: after a flush and a wait the event must
    /// report completion, and polling before completion never hands out
    /// the data.
    #[test]
    fn is_complete_answers_from_the_event_status() {
        let Some((context, queue)) = queue() else {
            return;
        };
        let buffer = Buffer::from_slice(&context, &[1u16, 2, 3, 4]).expect("upload");
        let mut out = [0u16; 4];
        let readback = queue
            .read_buffer_non_blocking(&buffer, 0, &mut out, &[])
            .expect("submit");
        queue.flush().expect("flush");
        let complete = readback.is_complete().expect("status query");
        if is_bare(&queue) {
            // The CPU driver retires copies inside the enqueue call.
            assert!(complete, "the bare driver must report the read done");
        }
        let data = readback.wait().expect("complete");
        assert_eq!(data, &[1, 2, 3, 4]);
    }

    /// The batching contract: N reads, one `Event::wait_all`, and every
    /// destination comes back with its own slice of the buffer.
    #[test]
    fn wait_all_retires_a_batch_with_one_wait() {
        let Some((context, queue)) = queue() else {
            return;
        };
        let source: Vec<u8> = (0..96).map(|i| i as u8).collect();
        let buffer = Buffer::from_slice(&context, &source).expect("upload");
        let mut head = vec![0u8; 16];
        let mut middle = vec![0u8; 16];
        let mut tail = vec![0u8; 16];
        let r1 = queue
            .read_buffer_non_blocking(&buffer, 0, &mut head, &[])
            .expect("submit 1");
        let r2 = queue
            .read_buffer_non_blocking(&buffer, 32, &mut middle, &[])
            .expect("submit 2");
        let r3 = queue
            .read_buffer_non_blocking(&buffer, 80, &mut tail, &[])
            .expect("submit 3");
        let batch = Readback::wait_all(vec![r1, r2, r3]).expect("one wait for three reads");
        assert_eq!(batch[0], &source[0..16]);
        assert_eq!(batch[1], &source[32..48]);
        assert_eq!(batch[2], &source[80..96]);
    }

    /// Abandoning a readback must not leak in-flight memory back to the
    /// caller: the drop guard blocks on the event, so reading `out`
    /// afterwards observes the finished copy.
    #[test]
    fn dropping_an_unwaited_readback_still_completes_the_copy() {
        let Some((context, queue)) = queue() else {
            return;
        };
        let buffer = Buffer::from_slice(&context, &[9i32; 8]).expect("upload");
        let mut out = vec![0i32; 8];
        {
            let readback = queue
                .read_buffer_non_blocking(&buffer, 0, &mut out, &[])
                .expect("submit");
            drop(readback);
        }
        assert_eq!(out, vec![9i32; 8]);
    }

    /// A wait-list event from another context is rejected before the
    /// driver is asked to schedule an undefined dependency.
    #[test]
    fn wait_list_events_must_share_the_context() {
        let Some((context, queue)) = queue() else {
            return;
        };
        let other = Context::new(context.device()).expect("second context");
        let mut out = [0u8; 4];
        let Ok(other_buffer) = Buffer::new(&other, 4) else {
            return;
        };
        // Foreign buffer: rejected first.
        assert!(matches!(
            queue.read_buffer_non_blocking(&other_buffer, 0, &mut out, &[]),
            Err(Error::ContextMismatch { .. })
        ));
        // Foreign event in the wait list: rejected too, even for a
        // perfectly valid local buffer.
        let local = Buffer::new(&context, 4).expect("local buffer");
        let Ok(other_queue) = CommandQueue::new(&other, other.device()) else {
            return;
        };
        let foreign_event = other_queue
            .write_buffer(&other_buffer, 0, &[0u8; 4])
            .expect("foreign command");
        assert!(matches!(
            queue.read_buffer_non_blocking(&local, 0, &mut out, &[&foreign_event]),
            Err(Error::ContextMismatch { .. })
        ));
    }

    /// An out-of-range region never reaches the driver.
    #[test]
    fn ranges_are_checked_before_submitting() {
        let Some((context, queue)) = queue() else {
            return;
        };
        let buffer = Buffer::new(&context, 8).expect("buffer");
        let mut out = [0u8; 8];
        assert!(matches!(
            queue.read_buffer_non_blocking(&buffer, 4, &mut out, &[]),
            Err(Error::InvalidArgument { .. })
        ));
    }

    /// Waiting on an empty batch succeeds without touching the driver —
    /// the degenerate frame that enqueued nothing.
    #[test]
    fn waiting_on_an_empty_batch_is_a_no_op() {
        let batch: Vec<Readback<'_, u8>> = Vec::new();
        assert!(Readback::wait_all(batch).is_ok_and(|data| data.is_empty()));
        assert!(Event::wait_all(&[]).is_ok());
    }
}
