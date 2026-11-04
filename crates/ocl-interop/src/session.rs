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

//! A share whose ownership window cannot be forgotten.
//!
//! Importing a Vulkan allocation into OpenCL does not hand it over: the
//! OpenCL commands of a queue may only read or write it between
//! `clEnqueueAcquireExternalMemObjectsKHR` and the matching release, and a
//! command that touches it outside that window races the Vulkan device that
//! still owns it. Neither driver detects the race, so the window is the
//! thing this module models: [`SharedBuffer::acquire`] opens it and returns
//! a [`Share`] that closes it again when the scope ends.
//!
//! A signal crosses the same border in both directions, and its invariant is
//! a balanced pair rather than a window: exactly one wait belongs to each
//! signal, on whichever side created the semaphore. [`SharedSemaphore`]
//! carries one OpenCL semaphore with the handle kind it crosses as, so the
//! export asks for that kind and the queue commands name that semaphore
//! instead of a list a caller has to keep straight.
//!
//! The descriptors themselves are never owned here. A handle exported by
//! [`Vulkan`](crate::Vulkan) stays owned by the exporter, which closes it
//! once [`import`](SharedBuffer::import) has read it, while the descriptor a
//! Vulkan semaphore import takes is given to that driver, as its
//! specification requires. Either way, no type in this crate holds an `i32`
//! it might have to close.

use core::slice;

use codevar_logger::log_warn;
use codevar_ocl::{Buffer, CommandQueue, Context, Event, Semaphore};

use crate::error::Result;
use crate::handle::{MemoryHandle, MemoryKind, SemaphoreHandle, SemaphoreKind};

/// A buffer of an OpenCL context that stands on memory a Vulkan device
/// allocated.
pub struct SharedBuffer {
    buffer: Buffer,
    kind: MemoryKind,
}

impl SharedBuffer {
    /// Imports the allocation `handle` names as a buffer of `context`.
    ///
    /// `handle` is what [`Vulkan::export_memory`](crate::Vulkan::export_memory)
    /// returned for the allocation, and `size` is the size of the buffer to
    /// create over it — the descriptor carries no length, so a `size`
    /// larger than the allocation is a driver-side error at best and out-
    /// of-bounds writes by OpenCL at worst.
    ///
    /// The descriptor stays owned by the exporter: this call only reads it,
    /// and closing it is the caller's job.
    ///
    /// # Errors
    ///
    /// - [`crate::Error::OpenCl`] when the driver refuses the import, which
    ///   is also how a context without `cl_khr_external_memory`, a handle
    ///   from another device and a size beyond the allocation are reported.
    pub fn import(context: &Context, size: usize, handle: MemoryHandle) -> Result<Self> {
        let buffer = Buffer::from_external(context, size, handle.as_opencl())?;
        Ok(Self {
            buffer,
            kind: handle.kind(),
        })
    }

    /// The OpenCL buffer, for the enqueue calls of a
    /// [`CommandQueue`].
    ///
    /// Reading or writing it is only legal while a [`Share`] of the same
    /// buffer is alive.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// The handle kind the allocation was imported as.
    #[must_use]
    pub const fn kind(&self) -> MemoryKind {
        self.kind
    }

    /// Takes ownership of the allocation for `queue`, in the same scope that
    /// gives it back.
    ///
    /// See [`Share`] for the release; the acquire event it carries is the
    /// command that every OpenCL read or writing to the buffer must wait on.
    ///
    /// # Errors
    ///
    /// - [`crate::Error::OpenCl`] for every refusal of this side, which is
    ///   also how a driver without `clEnqueueAcquireExternalMemObjectsKHR`
    ///   answers.
    pub fn acquire<'q>(&self, queue: &'q CommandQueue) -> Result<Share<'_, 'q>> {
        Share::new(slice::from_ref(&self.buffer), queue)
    }
}

/// The window in which one OpenCL queue owns a shared allocation.
///
/// The value borrows the buffers it guards, so no buffer can be dropped,
/// re-imported or moved to another context while the window is open, and
/// the window cannot outlive the queue that opened it.
pub struct Share<'b, 'q> {
    buffers: &'b [Buffer],
    queue: &'q CommandQueue,
    event: Event,
}

impl<'b, 'q> Share<'b, 'q> {
    /// Acquires `buffers` on `queue`.
    ///
    /// # Errors
    ///
    /// As [`SharedBuffer::acquire`], which is how a caller reaches this, plus
    /// [`crate::Error::OpenCl`] carrying the wrapper's own invalid-argument
    /// error for an empty list: `Share` is checked before the queue is asked,
    /// so a hand-built window over nothing never reaches the driver.
    pub fn new(buffers: &'b [Buffer], queue: &'q CommandQueue) -> Result<Self> {
        let event = queue.acquire_external_memory(buffers)?;
        Ok(Self {
            buffers,
            queue,
            event,
        })
    }

    /// The acquire command.
    #[must_use]
    pub fn event(&self) -> &Event {
        &self.event
    }

    /// The queue that holds the allocation.
    #[must_use]
    pub fn queue(&self) -> &CommandQueue {
        self.queue
    }

    /// The buffers under guard.
    #[must_use]
    pub fn buffers(&self) -> &[Buffer] {
        self.buffers
    }
}

impl Drop for Share<'_, '_> {
    fn drop(&mut self) {
        // Releasing is what hands the allocation back to Vulkan; a driver
        // that refuses it leaves the memory owned by nobody, which the
        // caller cannot see from a `Drop`, so it goes to the log.
        if let Err(error) = self.queue.release_external_memory(self.buffers) {
            log_warn!("clEnqueueReleaseExternalMemObjectsKHR failed with {error}");
        }
    }
}

/// One OpenCL semaphore that crosses to or from a Vulkan device.
///
/// The value holds a single semaphore, which is what makes the pair
/// checkable: a wait and the signal that releases it name the same object,
/// and the handle kind is fixed when the semaphore is made rather than
/// guessed at the export.
pub struct SharedSemaphore {
    semaphore: Semaphore,
    kind: SemaphoreKind,
}

impl SharedSemaphore {
    /// Imports the signal `handle` names as a semaphore of `context`.
    ///
    /// `handle` is what
    /// [`Vulkan::export_semaphore`](crate::Vulkan::export_semaphore)
    /// returned for a Vulkan semaphore; the export stays owned by the side
    /// that made it, and the caller closes it once this call has read it.
    ///
    /// # Errors
    ///
    /// - [`crate::Error::OpenCl`] for every refusal of this side, which is
    ///   also how a context without `cl_khr_external_semaphore` and a handle
    ///   the device cannot import are reported.
    pub fn import(context: &Context, handle: SemaphoreHandle) -> Result<Self> {
        let semaphore = Semaphore::from_external(context, handle.as_opencl())?;
        Ok(Self {
            semaphore,
            kind: handle.kind(),
        })
    }

    /// Creates an OpenCL semaphore that may be exported as `kind`.
    ///
    /// The type has to be named at creation:
    /// `clGetSemaphoreHandleForTypeKHR` refuses a type the semaphore was not
    /// built for, so a signal destined for a Vulkan submission is made here
    /// rather than converted afterwards. Ask
    /// [`Vulkan::can_import_semaphore`](crate::Vulkan::can_import_semaphore)
    /// which kind the other side takes before choosing it.
    ///
    /// # Errors
    ///
    /// - [`crate::Error::OpenCl`] when the driver refuses the creation, which
    ///   is also how a driver without `cl_khr_semaphore` answers.
    pub fn exportable(context: &Context, kind: SemaphoreKind) -> Result<Self> {
        let semaphore = Semaphore::exportable(context, &[kind.cl_token()])?;
        Ok(Self { semaphore, kind })
    }

    /// The OpenCL semaphore, for the enqueue calls of a [`CommandQueue`].
    #[must_use]
    pub fn semaphore(&self) -> &Semaphore {
        &self.semaphore
    }

    /// The handle kind this semaphore crosses as.
    #[must_use]
    pub const fn kind(&self) -> SemaphoreKind {
        self.kind
    }

    /// The handle a Vulkan device imports with
    /// [`Vulkan::import_semaphore_fd`](crate::Vulkan::import_semaphore_fd).
    ///
    /// The driver answers with a copy of what it holds, so a failed import
    /// leaves the descriptor here for the caller to close — but a successful
    /// one hands it to Vulkan, which takes ownership of the descriptor itself.
    ///
    /// # Errors
    ///
    /// - [`crate::Error::OpenCl`] when the driver does not export
    ///   `clGetSemaphoreHandleForTypeKHR` or refuses this kind, which cannot
    ///   happen for a semaphore built by
    ///   [`exportable`](Self::exportable) but does for one
    ///   [`import`](Self::import)ed from the other side — an imported
    ///   semaphore is not an exportable one.
    pub fn handle(&self) -> Result<SemaphoreHandle> {
        let token = self.kind.cl_token();
        let handle = match self.kind {
            SemaphoreKind::OpaqueFd => SemaphoreHandle::OpaqueFd(self.semaphore.export_descriptor(token)?),
            SemaphoreKind::SyncFd => SemaphoreHandle::SyncFd(self.semaphore.export_descriptor(token)?),
            SemaphoreKind::OpaqueWin32 => SemaphoreHandle::OpaqueWin32(self.semaphore.export_handle(token)?),
            SemaphoreKind::OpaqueWin32Kmt => {
                SemaphoreHandle::OpaqueWin32Kmt(self.semaphore.export_handle(token)?)
            }
            SemaphoreKind::D3D12Fence => SemaphoreHandle::D3D12Fence(self.semaphore.export_handle(token)?),
        };
        Ok(handle)
    }

    /// Runs the commands submitted before it, then blocks `queue` until this
    /// signal arrives.
    ///
    /// The wait consumes the signal, so exactly one belongs to each
    /// [`signal`](Self::signal); a second wait with no signal in between is
    /// undefined by the extension.
    ///
    /// # Errors
    ///
    /// - [`crate::Error::OpenCl`] when the queue refuses the command — a
    ///   driver without `clEnqueueWaitSemaphoresKHR` answers this way, and so
    ///   does a queue of another context.
    pub fn wait(&self, queue: &CommandQueue) -> Result<Event> {
        Ok(queue.wait_semaphores(slice::from_ref(&self.semaphore))?)
    }

    /// Signals this semaphore from `queue`, after the commands already
    /// submitted to it have run.
    ///
    /// # Errors
    ///
    /// As [`wait`](Self::wait), for `clEnqueueSignalSemaphoresKHR`.
    pub fn signal(&self, queue: &CommandQueue) -> Result<Event> {
        Ok(queue.signal_semaphores(slice::from_ref(&self.semaphore))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_ocl::{Device, Runtime};

    /// A context on the installed driver or the built-in CPU one, whichever
    /// answers first, with the device it was built from and a queue for it.
    fn queue() -> Option<(Device, Context, CommandQueue)> {
        let runtime = Runtime::load().ok()?;
        let device = runtime
            .platforms()
            .first()
            .and_then(|platform| platform.all_devices().ok())
            .and_then(|devices| devices.into_iter().next())?;
        let context = Context::new(&device).ok()?;
        let queue = CommandQueue::new(&context, &device).ok()?;
        Some((device, context, queue))
    }

    /// A driver without `cl_khr_external_memory` has to refuse an import,
    /// and a bogus descriptor has to be refused by the driver that does
    /// have it: neither may panic, and neither may pretend the buffer is
    /// shared. Both answers arrive as an OpenCL error, because the OpenCL
    /// side is the one that reads the handle.
    #[test]
    fn an_import_only_succeeds_on_a_driver_that_can_take_the_handle() {
        let Some((_, context, _)) = queue() else { return };
        let imported = SharedBuffer::import(&context, 64, MemoryHandle::DmaBuf(-1));
        assert!(
            matches!(imported, Err(crate::Error::OpenCl(_))),
            "a dma_buf descriptor of -1 cannot name an allocation"
        );
    }

    /// The guard only ever claims buffers of the queue's own context, which
    /// the wrapper checks by identity and not by device: a second context
    /// over the same GPU is still foreign, so a hand-built share cannot hand
    /// back memory another queue holds.
    #[test]
    fn a_guard_over_a_foreign_buffer_is_refused_before_the_driver() {
        let Some((device, _, queue)) = queue() else { return };
        let Ok(other) = Context::new(&device) else {
            return;
        };
        let Ok(buffer) = Buffer::new(&other, 64) else {
            return;
        };
        assert!(matches!(
            Share::new(slice::from_ref(&buffer), &queue),
            Err(crate::Error::OpenCl(codevar_ocl::Error::ContextMismatch { .. }))
        ));
    }

    /// An empty list never reaches the driver either: the guard is refused by
    /// the wrapper that checks it, so a hand-built window over nothing cannot
    /// leave a queue holding nothing.
    #[test]
    fn a_guard_over_no_buffers_is_refused_before_the_driver() {
        let Some((_, _, queue)) = queue() else { return };
        assert!(matches!(
            Share::new(&[], &queue),
            Err(crate::Error::OpenCl(codevar_ocl::Error::InvalidArgument { .. }))
        ));
    }

    /// The OpenCL side is the one that reads a semaphore handle, so a
    /// descriptor that names no signal is refused there rather than
    /// wrapping something that only looks shared.
    #[test]
    fn a_semaphore_imported_from_a_bogus_descriptor_is_refused() {
        let Some((_, context, _)) = queue() else { return };
        assert!(
            matches!(
                SharedSemaphore::import(&context, SemaphoreHandle::SyncFd(-1)),
                Err(crate::Error::OpenCl(_))
            ),
            "a sync_file descriptor of -1 cannot name a signal"
        );
    }

    /// A semaphore made for one kind is exported as that kind: the value and
    /// the enumerant come out of the same match, so a handle the Vulkan side
    /// would misread shows up here. A driver without `cl_khr_semaphore` never
    /// gets that far and says so as an OpenCL error, which is the only other
    /// legal answer.
    #[test]
    fn an_exportable_semaphore_is_exported_as_the_kind_it_was_made_for() {
        let Some((_, context, _)) = queue() else { return };
        match SharedSemaphore::exportable(&context, SemaphoreKind::OpaqueFd) {
            Ok(shared) => assert!(
                matches!(shared.handle(), Ok(SemaphoreHandle::OpaqueFd(_))),
                "the export must answer with the kind the semaphore was built for"
            ),
            Err(error) => assert!(
                matches!(error, crate::Error::OpenCl(_)),
                "only the OpenCL side can refuse the creation: {error}"
            ),
        }
    }
}
