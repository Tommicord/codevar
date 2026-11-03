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
//! * **Mapped region** — [`map_buffer`](crate::CommandQueue::map_buffer)
//!   submits `clEnqueueMapBuffer` with the same non-blocking discipline
//!   and hands back a [`MappedRegion`]. On host-visible memory the result
//!   needs no copy at all; the [non-coherent](MappedRegion::flush) case
//!   is covered by the explicit `clFlushMemObject` flush and the
//!   mandatory [`unmap`](MappedRegion::unmap) — the two calls a coherent
//!   host-visible buffer would not need. When the implementation does not
//!   expose the map entry points ([`supports_buffer_mapping`](crate::CommandQueue::supports_buffer_mapping)
//!   answers `false`, e.g. the built-in CPU driver), callers must take
//!   the non-blocking copy path above instead of failing.
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

use crate::api::Api;
use crate::buffer::{Buffer, check_range};
use crate::error::{self, Error, Result, status};
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

/// The direction a buffer mapping is valid for.
///
/// The three `CL_MAP_*` flags are mutually exclusive per the OpenCL
/// specification (a region cannot be mapped for reading and writing at
/// once), so this is an enum rather than combinable bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapMode {
    /// The host reads the region after the device wrote it
    /// (`CL_MAP_READ`).
    Read,
    /// The host writes the region, preserving the rest of the buffer
    /// (`CL_MAP_WRITE`).
    Write,
    /// The host overwrites the region and the previous contents are
    /// undefined, letting the driver skip the device-to-host copy
    /// (`CL_MAP_WRITE_INVALIDATE_REGION`).
    WriteInvalidate,
}

impl MapMode {
    /// Returns the `cl_map_flags` bit of this mode.
    #[must_use]
    pub const fn bits(self) -> u64 {
        match self {
            Self::Read => sys::MAP_READ,
            Self::Write => sys::MAP_WRITE,
            Self::WriteInvalidate => sys::MAP_WRITE_INVALIDATE_REGION,
        }
    }
}

/// An outstanding non-blocking mapping of a buffer region into the host
/// address space.
///
/// The region is mapped by a `clEnqueueMapBuffer` command whose event
/// gates access: while the mapping command may still be in flight the
/// memory is unreachable — [`MappedView`], the only type that hands out
/// slices, is produced exclusively by [`wait`](Self::wait) after that
/// event retired. The region keeps the buffer borrowed, so the mapping
/// cannot outlive the [`Buffer`] it views.
///
/// On machines whose host memory is not coherent with the device, the
/// [`flush`](Self::flush) (write direction, without dropping the
/// mapping) and [`unmap`](Self::unmap) (final completion) calls are the
/// required non-coherent handling; a coherent host-visible buffer would
/// need neither.
///
/// Dropping a region that was never unmapped enqueues a blocking unmap,
/// so an abandoned mapping neither leaks the driver-side mapping nor
/// lets the buffer be released while still mapped.
pub struct MappedRegion<'a> {
    /// The queue the map command runs on; also keeps the driver alive.
    queue: CommandQueue,
    /// The borrowed buffer: the mapping is a view of its memory, and
    /// releasing the buffer while mapped would dangle `ptr`.
    buffer: &'a Buffer,
    /// The pointer `clEnqueueMapBuffer` returned for `[offset, offset +
    /// size)`; non-null by construction.
    ptr: *mut core::ffi::c_void,
    /// The mapped region size in bytes.
    size: usize,
    /// The direction the region was mapped for.
    mode: MapMode,
    /// The map command's event; its completion licenses host access.
    map_event: Event,
    /// `false` once the unmap was handed to the driver, so `Drop` never
    /// double-unmaps.
    mapped: bool,
}

// The raw `ptr` field makes `MappedRegion` (and `MappedView`) neither
// `Send` nor `Sync` automatically, which is exactly right: a host
// mapping of a queue object is thread-confined by the borrow rules of
// OpenCL.

impl<'a> MappedRegion<'a> {
    /// Returns the event that gates host access to the region.
    #[must_use]
    pub fn event(&self) -> &Event {
        &self.map_event
    }

    /// Returns the direction this region was mapped for.
    #[must_use]
    pub fn mode(&self) -> MapMode {
        self.mode
    }

    /// Returns the mapped region size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Queries the driver for the map command's completion status
    /// (`clGetEventInfo`).
    ///
    /// Like [`Readback::is_complete`] this never flushes the queue.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the query fails or the map command
    /// ended with an error status.
    pub fn is_complete(&self) -> Result<bool> {
        Ok(self.map_event.state()? == CommandState::Complete)
    }

    /// Enqueues `clFlushMemObject` for the mapped buffer
    /// (`CL_NON_BLOCKING` semantics through the returned event).
    ///
    /// On a non-coherent machine this pushes the mapping's current
    /// contents to the host-visible side while the caller keeps working
    /// on the region — the incremental flush that a coherent
    /// host-visible buffer does not need. The command runs after the map
    /// (dependency list). `CL_MAP_WRITE` regions must still be completed
    /// by [`unmap`](Self::unmap); the flush does not replace it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the implementation does not
    /// export `clFlushMemObject` — it has no ICD dispatch slot, so a
    /// vendor reached only through its dispatch table lacks it. The
    /// mandatory unmap-based flush in [`unmap`](Self::unmap) works
    /// regardless; treat this error as "unmap-only non-coherent
    /// handling", not a failure of the mapping.
    pub fn flush(&self) -> Result<Event> {
        let flush = Api::optional(self.queue.api().flush_mem_object, "clFlushMemObject")?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        let dependency = self.map_event.raw();
        // SAFETY: queue, buffer and event handles are live and from one
        // context; `dependency` is the map command's own event, so the
        // flush runs after the mapping exists; `event` is a valid
        // out-pointer.
        let code = unsafe { flush(self.queue.raw(), self.buffer.raw(), 1, &dependency, &mut event) };
        let raw = error::creation(event, code, "clFlushMemObject")?;
        Ok(Event::new(&self.queue, raw))
    }

    /// Blocks until the map command retired and returns the host-access
    /// view of the region.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the flush or the wait fails.
    pub fn wait(&mut self) -> Result<MappedView<'_, 'a>> {
        self.map_event.wait()?;
        Ok(MappedView { region: self })
    }

    /// Releases the mapping (`clEnqueueUnmapMemObject`, non-blocking)
    /// and returns the unmap command's event.
    ///
    /// For non-coherent host memory this is the completion step: the
    /// region's contents reach the device when the returned event
    /// retires. Chain later commands on that event rather than waiting
    /// on the CPU.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the unmap; the
    /// region then stays mapped and the drop guard retries a blocking
    /// unmap.
    pub fn unmap(mut self) -> Result<Event> {
        let unmap = Api::optional(
            self.queue.api().enqueue_unmap_mem_object,
            "clEnqueueUnmapMemObject",
        )?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        let dependency = self.map_event.raw();
        // SAFETY: the queue, buffer and pointer are the live mapping
        // created by this handle (`ptr` is exactly what the map call
        // returned, and the region is still mapped); `dependency` keeps
        // the unmap after the map on out-of-order queues; `event` is a
        // valid out-pointer.
        let code = unsafe {
            unmap(
                self.queue.raw(),
                self.buffer.raw(),
                self.ptr,
                1,
                &dependency,
                &mut event,
            )
        };
        status(code, "clEnqueueUnmapMemObject")?;
        self.mapped = false;
        let raw = error::creation(event, sys::SUCCESS, "clEnqueueUnmapMemObject")?;
        Ok(Event::new(&self.queue, raw))
    }
}

impl Drop for MappedRegion<'_> {
    fn drop(&mut self) {
        if !self.mapped {
            return;
        }
        self.mapped = false;
        let Some(unmap) = self.queue.api().enqueue_unmap_mem_object else {
            return;
        };
        // The blocking form: after this drop the caller may reuse or
        // release the buffer, so the mapping must be gone — not merely
        // queued — before the borrow ends. Failures are logged; a
        // half-released mapping is the driver's problem to report.
        // SAFETY: queue and buffer handles are live in one context and
        // `ptr` is the still-valid mapping pointer (mapped was `true`);
        // a null wait list and null event are permitted.
        let code = unsafe {
            unmap(
                self.queue.raw(),
                self.buffer.raw(),
                self.ptr,
                0,
                core::ptr::null(),
                null_mut(),
            )
        };
        if code != sys::SUCCESS {
            log_warn!("clEnqueueUnmapMemObject failed with {}", sys::error_name(code));
        }
    }
}

impl fmt::Debug for MappedRegion<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedRegion")
            .field("map_event", &self.map_event)
            .field("mode", &self.mode)
            .field("size", &self.size)
            .field("mapped", &self.mapped)
            .finish_non_exhaustive()
    }
}

/// A host-accessible window into a [`MappedRegion`] whose map command
/// has retired.
///
/// The only way to obtain one is [`MappedRegion::wait`], so holding this
/// type is the proof that the device-to-host transfer finished; the
/// mutable borrow on the region keeps the mapping alive and un-released
/// for exactly as long as the slices are handed out.
pub struct MappedView<'r, 'a> {
    region: &'r mut MappedRegion<'a>,
}

impl MappedView<'_, '_> {
    /// Returns the region contents visible to the host.
    ///
    /// The slice covers the mapped `[offset, offset + size)` bytes as
    /// `clEnqueueMapBuffer` reports them.
    #[must_use]
    pub fn slice(&self) -> &[u8] {
        // SAFETY: `MappedView` is only produced by `MappedRegion::wait`,
        // which blocked until the map command retired, so the driver is
        // done establishing the mapping; `ptr` was validated non-null
        // for `size` bytes at creation; the borrow on `region` prevents
        // unmap or drop while this view (and any slice from it) is
        // alive. `u8` needs no alignment.
        unsafe { core::slice::from_raw_parts(self.region.ptr.cast::<u8>(), self.region.size) }
    }

    /// Returns the region for host writing, or `None` when the region
    /// was mapped [`MapMode::Read`] — a read mapping is not writable.
    #[must_use]
    pub fn slice_mut(&mut self) -> Option<&mut [u8]> {
        (self.region.mode != MapMode::Read).then(|| {
            // SAFETY: as for `slice`, plus the write direction was
            // checked; writing is host-side and the mapping is exclusive
            // to this queue's context while mapped for writing per the
            // OpenCL specification.
            unsafe { core::slice::from_raw_parts_mut(self.region.ptr.cast::<u8>(), self.region.size) }
        })
    }

    /// Returns the mapping direction.
    #[must_use]
    pub fn mode(&self) -> MapMode {
        self.region.mode
    }

    /// Returns the mapped region size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.region.size
    }

    /// Forwards to [`MappedRegion::flush`] while the view keeps the
    /// region borrowed — the mid-write flush of non-coherent handling.
    ///
    /// # Errors
    ///
    /// See [`MappedRegion::flush`].
    pub fn flush(&self) -> Result<Event> {
        self.region.flush()
    }
}

impl fmt::Debug for MappedView<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedView")
            .field("mode", &self.region.mode)
            .field("size", &self.region.size)
            .finish_non_exhaustive()
    }
}

impl CommandQueue {
    /// Whether this queue's driver exposes the buffer-mapping commands
    /// (`clEnqueueMapBuffer` and `clEnqueueUnmapMemObject`).
    ///
    /// A `false` answer is the degradation signal: mapped readbacks are
    /// unavailable, but [`read_buffer_non_blocking`](Self::read_buffer_non_blocking)
    /// is not affected, so frame loops can pick the copy path without
    /// trying and failing per call.
    #[must_use]
    pub fn supports_buffer_mapping(&self) -> bool {
        let api = self.api();
        api.enqueue_map_buffer.is_some() && api.enqueue_unmap_mem_object.is_some()
    }

    /// Starts a non-blocking mapping of `buffer`'s region
    /// (`clEnqueueMapBuffer` with `CL_FALSE`).
    ///
    /// This is the zero-copy readback form: on host-visible memory the
    /// result needs no separate copy — [`MappedRegion::wait`] then reads
    /// the device's output straight out of the mapping. The returned
    /// [`MappedRegion`] borrows `buffer`, and host access is gated by
    /// the map event exactly like [`Readback`] gates a copy destination.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the driver lacks the map or
    /// unmap entry points — callers must then fall back to
    /// [`read_buffer_non_blocking`](Self::read_buffer_non_blocking), not
    /// treat it as fatal. Also returns [`Error::ContextMismatch`] for a
    /// foreign buffer or wait-list event, [`Error::InvalidArgument`] for
    /// an out-of-range region or over-long wait list, and
    /// [`Error::Status`] / [`Error::NullHandle`] on driver failures.
    pub fn map_buffer<'a>(
        &self,
        buffer: &'a Buffer,
        offset_bytes: usize,
        size: usize,
        mode: MapMode,
        wait_list: &[&Event],
    ) -> Result<MappedRegion<'a>> {
        self.check_context(
            buffer.context(),
            "the mapped buffer belongs to a different context",
        )?;
        // Both halves of the mapping protocol are required up front: a
        // region that could never be unmapped must not be created.
        let map = Api::optional(self.api().enqueue_map_buffer, "clEnqueueMapBuffer")?;
        Api::optional(self.api().enqueue_unmap_mem_object, "clEnqueueUnmapMemObject")?;
        let handles = event_wait_handles(self, wait_list)?;
        check_range(buffer.size(), offset_bytes, size)?;
        let mut errcode = sys::SUCCESS;
        let mut event = sys::EventHandle::from_raw(null_mut());
        // SAFETY: queue and buffer handles are live in one context
        // (checked above); `handles` outlives the call and the list is
        // null exactly when empty; `event` and `errcode` are valid
        // out-pointers. The returned pointer's usability is gated by
        // `MappedRegion`'s event discipline, not by this call.
        let ptr = unsafe {
            map(
                self.raw(),
                buffer.raw(),
                sys::NON_BLOCKING,
                mode.bits(),
                offset_bytes,
                size,
                handles.len() as u32,
                wait_list_ptr(&handles),
                &mut event,
                &mut errcode,
            )
        };
        status(errcode, "clEnqueueMapBuffer")?;
        if ptr.is_null() {
            // Success without a pointer is a driver contract violation;
            // the (possibly set) event leaks under the same policy
            // `error::creation` documents for failed creations.
            return Err(Error::NullHandle {
                context: "clEnqueueMapBuffer",
            });
        }
        let raw = error::creation(event, sys::SUCCESS, "clEnqueueMapBuffer")?;
        Ok(MappedRegion {
            queue: self.clone(),
            buffer,
            ptr,
            size,
            mode,
            map_event: Event::new(self, raw),
            mapped: true,
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

    /// The map flags are the ABI values from `CL/cl.h`; a wrong bit maps
    /// the region in the wrong direction.
    #[test]
    fn map_mode_bits_match_the_header() {
        assert_eq!(MapMode::Read.bits(), sys::MAP_READ);
        assert_eq!(MapMode::Write.bits(), sys::MAP_WRITE);
        assert_eq!(MapMode::WriteInvalidate.bits(), sys::MAP_WRITE_INVALIDATE_REGION);
        assert_eq!(MapMode::Read.bits(), 1);
        assert_eq!(MapMode::Write.bits(), 2);
        assert_eq!(MapMode::WriteInvalidate.bits(), 4);
    }

    /// Mapping follows the same event discipline as the copy path: on a
    /// driver without the map entry points the capability check answers
    /// `false` and `map_buffer` reports `Unsupported` — the degradation
    /// signal for callers, not a fatal error. Where the driver does map,
    /// a full read round-trip must return the uploaded bytes and a
    /// write mapping must land in the buffer once unmapped.
    #[test]
    fn mapped_regions_round_trip_or_degrade() {
        let Some((context, queue)) = queue() else {
            return;
        };
        if !queue.supports_buffer_mapping() {
            let buffer = Buffer::new(&context, 8).expect("buffer");
            assert!(matches!(
                queue.map_buffer(&buffer, 0, 8, MapMode::Read, &[]),
                Err(Error::Unsupported { .. })
            ));
            return;
        }

        let source: Vec<u8> = (0..40).map(|i| i as u8 * 3).collect();
        let read_buffer = Buffer::from_slice(&context, &source).expect("upload");
        let mut region = queue
            .map_buffer(&read_buffer, 4, 16, MapMode::Read, &[])
            .expect("map for reading");
        {
            let mut view = region.wait().expect("the map command retires");
            assert_eq!(view.slice(), &source[4..20]);
            assert_eq!(view.size(), 16);
            assert_eq!(view.mode(), MapMode::Read);
            assert!(view.slice_mut().is_none(), "a read mapping is not writable");
        }
        region
            .unmap()
            .expect("unmap")
            .wait()
            .expect("unmap retires");

        let write_buffer = Buffer::new(&context, 8).expect("buffer");
        let mut region = queue
            .map_buffer(&write_buffer, 0, 8, MapMode::Write, &[])
            .expect("map for writing");
        {
            let mut view = region.wait().expect("the map command retires");
            // A driver without the symbol-only flush entry answers
            // `Unsupported`; the unmap below is then the only flush, which
            // is the contract the mapped path keeps.
            if let Ok(flush) = view.flush() {
                flush.wait().expect("flush retires");
            }
            if let Some(slot) = view.slice_mut() {
                slot.copy_from_slice(&[0xaau8; 8]);
            }
        }
        region
            .unmap()
            .expect("unmap")
            .wait()
            .expect("unmap retires");
        let mut out = [0u8; 8];
        queue
            .read_buffer(&write_buffer, 0, &mut out)
            .expect("blocking read back");
        assert_eq!(out, [0xaau8; 8]);
    }

    /// A mapped region must validate its range against the buffer and
    /// its context before touching the driver, exactly like the copy
    /// path.
    #[test]
    fn mapping_validates_arguments_first() {
        let Some((context, queue)) = queue() else {
            return;
        };
        if !queue.supports_buffer_mapping() {
            return;
        }
        let buffer = Buffer::new(&context, 16).expect("buffer");
        assert!(matches!(
            queue.map_buffer(&buffer, 8, 16, MapMode::Read, &[]),
            Err(Error::InvalidArgument { .. })
        ));
        let other = Context::new(context.device()).expect("second context");
        let Ok(foreign) = Buffer::new(&other, 16) else {
            return;
        };
        assert!(matches!(
            queue.map_buffer(&foreign, 0, 16, MapMode::Read, &[]),
            Err(Error::ContextMismatch { .. })
        ));
    }
}
