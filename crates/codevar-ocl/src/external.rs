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

//! Memory that another API exported.
//!
//! An imported buffer is a *view* of somebody else's allocation, which
//! changes two things about it compared to [`Buffer::new`]: it is created
//! through a property list instead of a plain size, and the exporting API
//! keeps ownership of the underlying memory.
//!
//! [`ExternalMemoryHandle`] names the handle in the same vocabulary the
//! driver uses — the `cl_external_memory_handle_type_khr` token plus the
//! handle itself — and [`Buffer::from_external`] widens that pair into the
//! `(name, value)` words of a [`crate::properties::PropertyList`]. The
//! Khronos example for a single-device context is exactly one pair plus the
//! terminator; no device-handle list is needed because
//! [`Context`](crate::Context) binds one device, so the memory is usable
//! there by construction.
//!
//! Ownership is the second difference. Importing hands the driver a file
//! descriptor or a `HANDLE`, and this crate never closes or frees it: an
//! [`Buffer`] drop calls `clReleaseMemObject` and nothing else, so the
//! exporter keeps its own reference and decides when the allocation is
//! finally released. Ownership *of the memory for command execution* is a
//! separate, per-command question answered by
//! [`CommandQueue::acquire_external_memory`] and
//! [`CommandQueue::release_external_memory`].
//!
//! # Examples
//!
//! ```
//! use codevar_ocl::{Buffer, Context, ExternalMemoryHandle, Runtime, sys};
//!
//! /// Imports a compositor's dma-buf, or reports that this driver cannot.
//! fn import(context: &Context, dma_buf: i32, bytes: usize) -> Option<Buffer> {
//!     if !context.supports_external_memory() {
//!         return None;
//!     }
//!     Buffer::from_external(context, bytes, ExternalMemoryHandle::DmaBuf(dma_buf)).ok()
//! }
//!
//! let Ok(runtime) = Runtime::load() else {
//!     return;
//! };
//! let Some(platform) = runtime.platforms().first() else {
//!     return;
//! };
//! for device in platform.all_devices().unwrap_or_default() {
//!     // Only a handle type the device lists can be imported; asking first
//!     // turns a driver rejection into a capability answer.
//!     let dma_buf = device
//!         .external_memory_import_handle_types()
//!         .unwrap_or_default()
//!         .contains(&sys::EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR);
//!     if !dma_buf {
//!         continue;
//!     }
//!     let Ok(context) = Context::new(&device) else {
//!         continue;
//!     };
//!     // A descriptor this process does not own is refused, which is the
//!     // point of the negotiation rather than a failure of the program.
//!     let _ = import(&context, -1, 4096);
//! }
//! ```

use alloc::vec::Vec;
use core::ptr::null_mut;

use crate::api::Api;
use crate::buffer::Buffer;
use crate::context::Context;
use crate::error::{self, Error, Result};
use crate::event::Event;
use crate::properties::{self, PropertyList};
use crate::queue::CommandQueue;
use crate::sys;

/// A handle to memory exported by another API, in the driver's own
/// vocabulary: each variant is one `CL_EXTERNAL_MEMORY_HANDLE_*_KHR` token
/// and the handle it names.
///
/// The variants mirror the extension rather than the platform: a Windows
/// handle variant is constructible on Linux too, where the driver simply
/// never lists those handle types as importable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalMemoryHandle<'a> {
    /// A POSIX file descriptor from `vkGetMemoryFdKHR`
    /// (`CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR`).
    OpaqueFd(i32),
    /// A file descriptor holding a dma-buf
    /// (`CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR`), as a compositor or a
    /// video decoder exports one.
    DmaBuf(i32),
    /// A Win32 `HANDLE` to an NT handle
    /// (`CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR`), widened from the
    /// pointer-sized `HANDLE`.
    OpaqueWin32(usize),
    /// A Win32 `HANDLE` to a KMT handle
    /// (`CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR`).
    OpaqueWin32Kmt(usize),
    /// The global-object *name* of a Win32 section
    /// (`CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_NAME_KHR`), in UTF-16 code
    /// units without a trailing zero; the terminator is added for the
    /// driver call.
    OpaqueWin32Name(&'a [u16]),
}

impl ExternalMemoryHandle<'_> {
    /// Returns the `cl_external_memory_handle_type_khr` token of this
    /// handle — the value a device's importable-handle list is compared
    /// against (see
    /// [`Device::external_memory_import_handle_types`](crate::Device::external_memory_import_handle_types)).
    #[must_use]
    pub const fn handle_type(&self) -> sys::ExternalMemoryHandleType {
        match self {
            Self::OpaqueFd(_) => sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR,
            Self::DmaBuf(_) => sys::EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
            Self::OpaqueWin32(_) => sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR,
            Self::OpaqueWin32Kmt(_) => sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR,
            Self::OpaqueWin32Name(_) => sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_NAME_KHR,
        }
    }
}

impl Buffer {
    /// Creates a buffer that views `size` bytes of memory exported by
    /// another API.
    ///
    /// The import goes through `clCreateBufferWithProperties` with a
    /// single `(handle type, handle)` property, and the flags are
    /// `CL_MEM_READ_WRITE`: the exporter's allocation is shared, not
    /// copied, so nothing is read from a host pointer.
    ///
    /// `handle` is only read during this call and this crate never closes
    /// or frees it, so the exporter keeps its own reference for as long as
    /// it needs the memory. The buffer may not be used by a command queue
    /// until that queue has taken ownership of it with
    /// [`CommandQueue::acquire_external_memory`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `size` is zero or a Win32
    /// name is empty, [`Error::Unsupported`] when the driver does not
    /// export `clCreateBufferWithProperties`, and [`Error::Status`] /
    /// [`Error::NullHandle`] when the driver refuses the handle — which is
    /// how a handle type the device cannot import, or a descriptor the
    /// driver cannot open, is reported.
    pub fn from_external(context: &Context, size: usize, handle: ExternalMemoryHandle<'_>) -> Result<Self> {
        if size == 0 {
            return Err(Error::InvalidArgument {
                what: "an imported buffer must be at least one byte large",
            });
        }
        let create = Api::optional(
            context.api().create_buffer_with_properties,
            "clCreateBufferWithProperties",
        )?;
        // A Win32 object name reaches the driver as a wide string, so the
        // caller's slice gets its own terminated copy here; it must stay
        // alive until the driver call below returns.
        let mut wide: Vec<u16> = Vec::new();
        let mut list = PropertyList::new();
        match handle {
            ExternalMemoryHandle::OpaqueFd(fd) => list.pair(
                sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR.into(),
                properties::from_int(fd),
            ),
            ExternalMemoryHandle::DmaBuf(fd) => list.pair(
                sys::EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR.into(),
                properties::from_int(fd),
            ),
            ExternalMemoryHandle::OpaqueWin32(raw) => list.pair(
                sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR.into(),
                raw as sys::Properties,
            ),
            ExternalMemoryHandle::OpaqueWin32Kmt(raw) => list.pair(
                sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR.into(),
                raw as sys::Properties,
            ),
            ExternalMemoryHandle::OpaqueWin32Name(name) => {
                if name.is_empty() {
                    return Err(Error::InvalidArgument {
                        what: "a Win32 object name must not be empty",
                    });
                }
                wide.extend_from_slice(name);
                wide.push(0);
                list.pair(
                    sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_NAME_KHR.into(),
                    properties::from_ptr(wide.as_ptr()),
                );
            }
        }
        let properties = list.build();
        let mut errcode = sys::SUCCESS;
        // SAFETY: `context` is a live handle of the driver that resolved
        // `create`; `properties` is a zero-terminated `cl_properties`
        // array that outlives the call, and every value in it is a word
        // the driver reads during the call only; no host pointer is
        // passed, because the memory belongs to the exporter; `errcode` is
        // a valid out-pointer.
        let raw = unsafe {
            create(
                context.raw(),
                properties.as_ptr(),
                sys::MEM_READ_WRITE,
                size,
                null_mut(),
                &mut errcode,
            )
        };
        let raw = error::creation(raw, errcode, "clCreateBufferWithProperties")?;
        Ok(Self::from_raw(context, raw, size))
    }
}

/// The ownership half of sharing memory: an imported buffer may only be
/// read, written or bound to a kernel between an acquire and the matching
/// release on the queue running those commands.
impl CommandQueue {
    /// Validates `buffers` for an ownership command and collects their
    /// driver handles in the order the driver expects.
    fn imported_handles(&self, buffers: &[Buffer]) -> Result<Vec<sys::MemHandle>> {
        if buffers.is_empty() {
            return Err(Error::InvalidArgument {
                what: "an external-memory command must name at least one buffer",
            });
        }
        for buffer in buffers {
            self.check_context(
                buffer.context(),
                "an imported buffer belongs to a different context",
            )?;
        }
        Ok(buffers.iter().map(Buffer::raw).collect())
    }

    /// Takes ownership of every buffer in `buffers` for this queue's
    /// device (`clEnqueueAcquireExternalMemObjectsKHR`).
    ///
    /// The memory stays the exporter's until this command runs, and the
    /// command runs at its place in the queue — after everything submitted
    /// before it. Between [`Buffer::from_external`] and this call a kernel
    /// reading the buffer sees whatever the exporter left there.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `buffers` is empty,
    /// [`Error::ContextMismatch`] when a buffer came from another context,
    /// [`Error::Unsupported`] when the driver does not export the entry
    /// point, and [`Error::Status`] / [`Error::NullHandle`] on driver
    /// failures — the driver refuses a buffer that was not imported with
    /// `CL_INVALID_OPERATION`.
    pub fn acquire_external_memory(&self, buffers: &[Buffer]) -> Result<Event> {
        let handles = self.imported_handles(buffers)?;
        let acquire = Api::optional(
            self.context()
                .extensions()
                .acquire_external_mem_objects,
            "clEnqueueAcquireExternalMemObjectsKHR",
        )?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        // SAFETY: `handles` are the live `cl_mem` values of this queue's
        // own context, which `imported_handles` just checked, and the
        // vector outlives the call; the wait list is empty and `event` is
        // a valid out-pointer.
        let code = unsafe {
            acquire(
                self.raw(),
                handles.len() as u32,
                handles.as_ptr(),
                0,
                core::ptr::null(),
                &mut event,
            )
        };
        let raw = error::creation(event, code, "clEnqueueAcquireExternalMemObjectsKHR")?;
        Ok(Event::new(self, raw))
    }

    /// Hands every buffer in `buffers` back to its exporter
    /// (`clEnqueueReleaseExternalMemObjectsKHR`).
    ///
    /// Every command this queue still has to run on those buffers must be
    /// submitted *before* the release; after the released command completes
    /// the exporter may read or free the memory again. Releasing a buffer
    /// the queue never acquired is an error the driver reports as
    /// [`Error::Status`].
    ///
    /// # Errors
    ///
    /// Returns the same errors as
    /// [`acquire_external_memory`](Self::acquire_external_memory).
    pub fn release_external_memory(&self, buffers: &[Buffer]) -> Result<Event> {
        let handles = self.imported_handles(buffers)?;
        let release = Api::optional(
            self.context()
                .extensions()
                .release_external_mem_objects,
            "clEnqueueReleaseExternalMemObjectsKHR",
        )?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        // SAFETY: as for the acquire above — the handles are live, belong
        // to this queue's context and outlive the call, and `event` is a
        // valid out-pointer.
        let code = unsafe {
            release(
                self.raw(),
                handles.len() as u32,
                handles.as_ptr(),
                0,
                core::ptr::null(),
                &mut event,
            )
        };
        let raw = error::creation(event, code, "clEnqueueReleaseExternalMemObjectsKHR")?;
        Ok(Event::new(self, raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    /// A context on the built-in driver or the installed one, whichever
    /// answers first; `None` when the machine exposes no usable device at
    /// all, in which case the driver-facing tests have nothing to check.
    fn context() -> Option<Context> {
        let runtime = Runtime::load().ok()?;
        let device = runtime
            .platforms()
            .first()
            .and_then(|platform| platform.all_devices().ok())
            .and_then(|devices| devices.into_iter().next())?;
        Context::new(&device).ok()
    }

    /// Importing is a negotiation over one token; a handle that reported
    /// the wrong token would be checked against the wrong capability list.
    #[test]
    fn every_handle_names_the_token_it_is_imported_as() {
        let name: &[u16] = &[0x73, 0];
        assert_eq!(
            ExternalMemoryHandle::OpaqueFd(3).handle_type(),
            sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR
        );
        assert_eq!(
            ExternalMemoryHandle::DmaBuf(3).handle_type(),
            sys::EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR
        );
        assert_eq!(
            ExternalMemoryHandle::OpaqueWin32(1).handle_type(),
            sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR
        );
        assert_eq!(
            ExternalMemoryHandle::OpaqueWin32Kmt(1).handle_type(),
            sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR
        );
        assert_eq!(
            ExternalMemoryHandle::OpaqueWin32Name(name).handle_type(),
            sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_NAME_KHR
        );
    }

    /// The arguments are checked before the driver is called, so an invalid
    /// request never reaches a creation entry point.
    #[test]
    fn from_external_rejects_bad_arguments_before_the_driver() {
        let Some(context) = context() else {
            return;
        };
        assert!(matches!(
            Buffer::from_external(&context, 0, ExternalMemoryHandle::DmaBuf(3)),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            Buffer::from_external(&context, 64, ExternalMemoryHandle::OpaqueWin32Name(&[])),
            Err(Error::InvalidArgument { .. })
        ));
    }

    /// A handle this process never exported cannot be imported. Whether the
    /// driver lacks the entry point or rejects the descriptor, the answer is
    /// an error rather than a buffer aliasing somebody else's memory.
    #[test]
    fn from_external_never_succeeds_with_a_bogus_handle() {
        let Some(context) = context() else {
            return;
        };
        let imported = Buffer::from_external(&context, 4096, ExternalMemoryHandle::OpaqueFd(-1));
        assert!(imported.is_err(), "a closed descriptor must not import");
        let imported = Buffer::from_external(&context, 4096, ExternalMemoryHandle::DmaBuf(i32::MAX));
        assert!(imported.is_err(), "a descriptor out of range must not import");
    }

    /// The arguments of an ownership command are checked before the driver
    /// is asked, so a malformed request answers the same way on a driver
    /// that exports `cl_khr_external_memory` and on one that does not.
    #[test]
    fn ownership_commands_validate_their_arguments_first() {
        let Some(context) = context() else {
            return;
        };
        let Ok(queue) = CommandQueue::new(&context, context.device()) else {
            return;
        };
        assert!(matches!(
            queue.acquire_external_memory(&[]),
            Err(Error::InvalidArgument { .. })
        ));
        // A second context over the same device is still a foreign
        // context, because the driver tracks ownership per context.
        let Ok(other) = Context::new(context.device()) else {
            return;
        };
        let Ok(foreign) = Buffer::new(&other, 16) else {
            return;
        };
        assert!(matches!(
            queue.release_external_memory(&[foreign]),
            Err(Error::ContextMismatch { .. })
        ));
    }
}
