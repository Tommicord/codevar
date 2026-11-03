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

//! Enqueued-command events.
//!
//! Every enqueue call of [`CommandQueue`](crate::CommandQueue) returns
//! an [`Event`] that can be waited on, queried for its
//! [`CommandState`], or — when the queue was created with
//! [`QueueProperties::Profiling`](crate::QueueProperties::Profiling) —
//! asked for device timestamps with [`Event::profiling_timestamp`].
//!
//! An [`Event`] keeps its [`CommandQueue`](crate::CommandQueue) alive,
//! so waiting always flushes the queue the command was submitted to and
//! neither the queue nor its context can be released early.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use codevar_logger::log_warn;

use crate::api::Api;
use crate::context::Context;
use crate::error::{Error, Result, status};
use crate::query;
use crate::queue::CommandQueue;
use crate::sys;

/// Execution state of an enqueued command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandState {
    /// The command is in the command queue.
    Queued,
    /// The command has been submitted to the device.
    Submitted,
    /// The command is currently executing.
    Running,
    /// The command finished successfully.
    Complete,
}

impl CommandState {
    /// Maps a raw `cl_command_execution_status` value.
    fn from_code(code: i32) -> Option<Self> {
        match code {
            sys::QUEUED => Some(Self::Queued),
            sys::SUBMITTED => Some(Self::Submitted),
            sys::RUNNING => Some(Self::Running),
            sys::COMPLETE => Some(Self::Complete),
            _ => None,
        }
    }
}

/// Device timestamp requested from profiling-enabled queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfilingTimestamp {
    /// The command entered the queue.
    Queued,
    /// The command was submitted to the device.
    Submitted,
    /// The command started executing.
    Started,
    /// The command finished executing.
    Finished,
}

impl ProfilingTimestamp {
    /// Returns the `CL_PROFILING_COMMAND_*` parameter of this timestamp.
    const fn param(self) -> u32 {
        match self {
            Self::Queued => sys::PROFILING_COMMAND_QUEUED,
            Self::Submitted => sys::PROFILING_COMMAND_SUBMIT,
            Self::Started => sys::PROFILING_COMMAND_START,
            Self::Finished => sys::PROFILING_COMMAND_END,
        }
    }
}

/// A handle to one enqueued command.
pub struct Event {
    api: Arc<Api>,
    queue: CommandQueue,
    raw: sys::EventHandle,
}

// SAFETY: `Event` holds a shared `Arc<Api>` of function pointers, a
// `CommandQueue` (already `Send + Sync`) and an opaque driver handle
// whose queries are thread-safe per the OpenCL specification.
unsafe impl Send for Event {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for Event {}

impl Event {
    /// Wraps a raw handle produced by a successful enqueue call.
    pub(crate) fn new(queue: &CommandQueue, raw: sys::EventHandle) -> Self {
        Self {
            api: queue.api().clone(),
            queue: queue.clone(),
            raw,
        }
    }

    /// Blocks until the command completed, flushing its queue first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the flush or the wait fails (for
    /// example when the command itself failed).
    pub fn wait(&self) -> Result<()> {
        self.queue.flush()?;
        status(
            // SAFETY: `raw` is a live event of the driver that resolved
            // `api`; the one-element list is a valid array.
            unsafe { (self.api.wait_for_events)(1, &self.raw) },
            "clWaitForEvents",
        )
    }

    /// Waits for every command in `events` with a single
    /// `clWaitForEvents` call.
    ///
    /// This is the batch half of the non-blocking submission protocol:
    /// enqueue N commands, collect their events, and retire all of them
    /// with one CPU wait instead of N. Each *distinct* queue the events
    /// belong to is flushed once before the wait, so a never-flushed
    /// queue cannot stall the batch. An empty list succeeds without
    /// touching the driver.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when the events come from more
    /// than one OpenCL implementation or the list is longer than
    /// `u32::MAX` entries, [`Error::Status`] when a flush fails or the
    /// driver reports that one of the waited commands ended with an
    /// error status.
    pub fn wait_all(events: &[&Event]) -> Result<()> {
        let Some(first) = events.first() else {
            return Ok(());
        };
        if !events
            .iter()
            .all(|event| Arc::ptr_eq(&event.api, &first.api))
        {
            return Err(Error::InvalidArgument {
                what: "events of different OpenCL implementations cannot be waited on together",
            });
        }
        let mut handles = Vec::with_capacity(events.len());
        let mut flushed: Vec<usize> = Vec::with_capacity(events.len());
        for event in events {
            let queue = event.queue.raw().as_raw() as usize;
            if !flushed.contains(&queue) {
                event.queue.flush()?;
                flushed.push(queue);
            }
            handles.push(event.raw);
        }
        let count = u32::try_from(handles.len()).map_err(|_| Error::InvalidArgument {
            what: "an event wait list must hold at most u32::MAX events",
        })?;
        status(
            // SAFETY: `handles` is a live array of `count` event handles,
            // all belonging to the driver that resolved `first.api` (the
            // shared-table pointer check above guarantees they came from
            // one implementation).
            unsafe { (first.api.wait_for_events)(count, handles.as_ptr()) },
            "clWaitForEvents",
        )
    }

    /// Returns the current execution state of the command.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the query fails or the driver
    /// reports an unknown state code.
    pub fn state(&self) -> Result<CommandState> {
        let mut call = |size: usize, value: *mut core::ffi::c_void, size_ret: *mut usize| {
            // SAFETY: the raw handle belongs to the driver that resolved
            // `api`, and `query` provides valid buffers.
            unsafe {
                (self.api.get_event_info)(
                    self.raw,
                    sys::EVENT_COMMAND_EXECUTION_STATUS,
                    size,
                    value,
                    size_ret,
                )
            }
        };
        let code = query::query_i32(&mut call, "clGetEventInfo")?;
        CommandState::from_code(code).ok_or(Error::Status {
            code,
            context: "clGetEventInfo",
        })
    }

    /// Returns a device timestamp of this command.
    ///
    /// Requires a queue created with
    /// [`QueueProperties::Profiling`](crate::QueueProperties::Profiling);
    /// otherwise the driver reports
    /// `CL_PROFILING_INFO_NOT_AVAILABLE`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when profiling is disabled or the
    /// query fails.
    pub fn profiling_timestamp(&self, timestamp: ProfilingTimestamp) -> Result<u64> {
        let param = timestamp.param();
        let mut call = |size: usize, value: *mut core::ffi::c_void, size_ret: *mut usize| {
            // SAFETY: the raw handle belongs to the driver that resolved
            // `api`, and `query` provides valid buffers.
            unsafe { (self.api.get_event_profiling_info)(self.raw, param, size, value, size_ret) }
        };
        query::query_u64(&mut call, "clGetEventProfilingInfo")
    }

    /// Returns the raw driver handle.
    pub(crate) fn raw(&self) -> sys::EventHandle {
        self.raw
    }

    /// Returns the context of the queue this event was enqueued on.
    pub(crate) fn context(&self) -> &Context {
        self.queue.context()
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by an enqueue call through
        // this same `api`, and each `Event` value owns its handle
        // exactly once (`Event` is not `Clone`).
        let code = unsafe { (self.api.release_event)(self.raw) };
        if code != sys::SUCCESS {
            log_warn!("clReleaseEvent failed with {}", sys::error_name(code));
        }
    }
}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Event")
            .field("handle", &self.raw)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn event_is_send_and_sync() {
        _assert_send_sync::<Event>();
    }

    #[test]
    fn command_states_map_from_the_header_values() {
        assert_eq!(CommandState::from_code(0), Some(CommandState::Complete));
        assert_eq!(CommandState::from_code(1), Some(CommandState::Running));
        assert_eq!(CommandState::from_code(2), Some(CommandState::Submitted));
        assert_eq!(CommandState::from_code(3), Some(CommandState::Queued));
        assert_eq!(CommandState::from_code(-999), None);
    }

    #[test]
    fn profiling_timestamps_select_the_right_parameters() {
        assert_eq!(ProfilingTimestamp::Queued.param(), 0x1280);
        assert_eq!(ProfilingTimestamp::Submitted.param(), 0x1281);
        assert_eq!(ProfilingTimestamp::Started.param(), 0x1282);
        assert_eq!(ProfilingTimestamp::Finished.param(), 0x1283);
    }
}
