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

//! Semaphores that outlive one command queue.
//!
//! A [`Semaphore`] is a binary flag owned by a [`Context`], not by a
//! queue: one API signals it and another waits on it, which is what makes
//! it the synchronization half of Vulkan interop. The object itself comes
//! from `cl_khr_semaphore`; exporting and importing it as a handle comes
//! from `cl_khr_external_semaphore` (see
//! [`exportable`](Semaphore::exportable) and
//! [`from_external`](Semaphore::from_external)).
//!
//! Both creation paths speak in the property words of
//! [`crate::properties`], and they differ in one decisive way:
//!
//! * an *exportable* semaphore names the handle types it may ever be
//!   exported for when it is created. A driver refuses to export a type
//!   the object was not created for, so the list cannot be decided later
//!   — the caller has to know which endpoint it will talk to.
//! * an *imported* semaphore is named by the `(handle type, handle)` pair
//!   itself: the token doubles as the property name and the handle as its
//!   value, exactly as in external-memory import.
//!
//! A payload is what distinguishes a binary from a timeline semaphore,
//! and this extension defines only the binary type
//! (`CL_SEMAPHORE_TYPE_BINARY_KHR`), which has no payload to choose: it is
//! either unsignaled or signaled. The wait and signal commands therefore
//! take semaphores rather than semaphore/payload pairs, and pass a null
//! payload list — the form the specification allows for a semaphore that
//! requires no payload.
//!
//! # Examples
//!
//! ```
//! use codevar_ocl::{Context, Runtime, Semaphore, sys};
//!
//! let Ok(runtime) = Runtime::load() else {
//!     return;
//! };
//! let Some(platform) = runtime.platforms().first() else {
//!     return;
//! };
//! for device in platform.all_devices().unwrap_or_default() {
//!     let Ok(context) = Context::new(&device) else {
//!         continue;
//!     };
//!     if !context.supports_semaphores() {
//!         continue;
//!     }
//!     // A semaphore a Vulkan endpoint will wait on has to be created
//!     // with that handle type already in its property list.
//!     let created = Semaphore::exportable(
//!         &context,
//!         &[sys::SEMAPHORE_HANDLE_OPAQUE_FD_KHR],
//!     );
//!     let _ = created;
//! }
//! ```

use alloc::sync::Arc;
use alloc::vec::Vec;

use core::ffi::c_void;
use core::fmt;
use core::mem::size_of;
use core::ptr::null_mut;

use codevar_logger::log_warn;

use crate::api::Api;
use crate::context::Context;
use crate::error::{self, Error, Result, status};
use crate::event::Event;
use crate::properties::{self, PropertyList};
use crate::query;
use crate::queue::CommandQueue;
use crate::sys;

/// A handle to a semaphore exported by another API.
///
/// Each variant is one `CL_SEMAPHORE_HANDLE_*_KHR` token plus the handle
/// it names; a handle value is given as the `usize` the platform stores in
/// the `HANDLE` / `int` slot the driver reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalSemaphoreHandle {
    /// A POSIX file descriptor from `vkGetSemaphoreFdKHR`
    /// (`CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR`).
    OpaqueFd(i32),
    /// A `sync_file` descriptor, as a dma-fence-based exporter hands one
    /// out (`CL_SEMAPHORE_HANDLE_SYNC_FD_KHR`).
    SyncFd(i32),
    /// A Win32 `HANDLE` to an NT-handle semaphore
    /// (`CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR`).
    OpaqueWin32(usize),
    /// A Win32 `HANDLE` to a KMT-handle semaphore
    /// (`CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR`).
    OpaqueWin32Kmt(usize),
    /// A `HANDLE` to a D3D12 fence
    /// (`CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR`).
    D3D12Fence(usize),
}

impl ExternalSemaphoreHandle {
    /// Returns the `cl_external_semaphore_handle_type_khr` token of this
    /// handle, the value compared against a device's importable-handle
    /// list (see
    /// [`Device::semaphore_import_handle_types`](crate::Device::semaphore_import_handle_types)).
    #[must_use]
    pub const fn handle_type(&self) -> sys::ExternalSemaphoreHandleType {
        match self {
            Self::OpaqueFd(_) => sys::SEMAPHORE_HANDLE_OPAQUE_FD_KHR,
            Self::SyncFd(_) => sys::SEMAPHORE_HANDLE_SYNC_FD_KHR,
            Self::OpaqueWin32(_) => sys::SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR,
            Self::OpaqueWin32Kmt(_) => sys::SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR,
            Self::D3D12Fence(_) => sys::SEMAPHORE_HANDLE_D3D12_FENCE_KHR,
        }
    }

    /// Returns the property value word carrying this handle.
    const fn word(&self) -> sys::SemaphoreProperties {
        match self {
            Self::OpaqueFd(fd) | Self::SyncFd(fd) => properties::from_int(*fd),
            Self::OpaqueWin32(handle) | Self::OpaqueWin32Kmt(handle) | Self::D3D12Fence(handle) => {
                *handle as sys::SemaphoreProperties
            }
        }
    }
}

/// A binary semaphore owned by a [`Context`].
///
/// Cloning is cheap and shares the single driver-side object, which is
/// released when the last clone is dropped.
#[derive(Clone)]
pub struct Semaphore {
    inner: Arc<SemaphoreInner>,
}

struct SemaphoreInner {
    context: Context,
    raw: sys::SemaphoreHandle,
}

impl Drop for SemaphoreInner {
    fn drop(&mut self) {
        let Some(release) = self.context.extensions().release_semaphore else {
            // The object was created through this very entry point, so a
            // driver that no longer exports it cannot be reached anyway.
            return;
        };
        // SAFETY: the handle was returned by `clCreateSemaphoreWithPropertiesKHR`
        // through the driver this context belongs to, and this `Drop` runs
        // exactly once for the single shared `SemaphoreInner`.
        let code = unsafe { release(self.raw) };
        if code != sys::SUCCESS {
            log_warn!("clReleaseSemaphoreKHR failed with {}", sys::error_name(code));
        }
    }
}

// SAFETY: `SemaphoreInner` is an opaque driver handle plus the context it
// belongs to; semaphores are reference-counted by the driver and the
// release entry point is thread-safe per the extension specification.
unsafe impl Send for SemaphoreInner {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for SemaphoreInner {}

/// Builds the property list of a binary semaphore created with no export
/// handle types, which is the plain `cl_khr_semaphore` object.
fn binary_properties() -> Vec<sys::SemaphoreProperties> {
    let mut list = PropertyList::new();
    list.pair(sys::SEMAPHORE_TYPE_KHR, sys::SEMAPHORE_TYPE_BINARY_KHR);
    list.build()
}

/// Builds the property list of a binary semaphore that may be exported for
/// every token in `handle_types`.
fn exportable_properties(handle_types: &[sys::ExternalSemaphoreHandleType]) -> Vec<sys::SemaphoreProperties> {
    let mut list = PropertyList::new();
    list.pair(sys::SEMAPHORE_TYPE_KHR, sys::SEMAPHORE_TYPE_BINARY_KHR);
    let types: Vec<sys::SemaphoreProperties> = handle_types
        .iter()
        .map(|token| sys::SemaphoreProperties::from(*token))
        .collect();
    list.list(
        sys::SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,
        &types,
        sys::SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR,
    );
    list.build()
}

/// Builds the property list that imports `handle`: the handle-type token
/// names the property and the handle is its value.
fn import_properties(handle: ExternalSemaphoreHandle) -> Vec<sys::SemaphoreProperties> {
    let mut list = PropertyList::new();
    list.pair(
        sys::SemaphoreProperties::from(handle.handle_type()),
        handle.word(),
    );
    list.build()
}

impl Semaphore {
    /// Creates a binary semaphore usable on the context's device.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the driver does not export
    /// `clCreateSemaphoreWithPropertiesKHR`, and [`Error::Status`] /
    /// [`Error::NullHandle`] on driver failures.
    pub fn new(context: &Context) -> Result<Self> {
        Self::create(
            context,
            &binary_properties(),
            "clCreateSemaphoreWithPropertiesKHR",
        )
    }

    /// Creates a binary semaphore that may be exported to another API,
    /// naming in advance every handle type it may be exported for.
    ///
    /// Exporting asks the driver for one of these types afterwards
    /// (`clGetSemaphoreHandleForTypeKHR`); a type that is not in
    /// `handle_types` is refused, which is why the list is decided at
    /// creation rather than at the export.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `handle_types` is empty,
    /// [`Error::Unsupported`] when the driver does not export the entry
    /// point, and [`Error::Status`] / [`Error::NullHandle`] when the driver
    /// rejects a type it cannot export.
    pub fn exportable(context: &Context, handle_types: &[sys::ExternalSemaphoreHandleType]) -> Result<Self> {
        if handle_types.is_empty() {
            return Err(Error::InvalidArgument {
                what: "an exportable semaphore must name at least one handle type",
            });
        }
        Self::create(
            context,
            &exportable_properties(handle_types),
            "clCreateSemaphoreWithPropertiesKHR",
        )
    }

    /// Creates a semaphore that views the semaphore `handle` refers to in
    /// another API.
    ///
    /// Like external memory, the handle is only read during this call and
    /// this crate never closes or frees it, so the exporter keeps its own
    /// reference.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the driver does not export the
    /// entry point, and [`Error::Status`] / [`Error::NullHandle`] when the
    /// driver refuses the handle — which is also how a handle type the
    /// device cannot import is reported.
    pub fn from_external(context: &Context, handle: ExternalSemaphoreHandle) -> Result<Self> {
        Self::create(
            context,
            &import_properties(handle),
            "clCreateSemaphoreWithPropertiesKHR",
        )
    }

    /// Creates the object from one already-built property list.
    fn create(
        context: &Context,
        properties: &[sys::SemaphoreProperties],
        symbol: &'static str,
    ) -> Result<Self> {
        let create = Api::optional(
            context
                .extensions()
                .create_semaphore_with_properties,
            symbol,
        )?;
        let mut errcode = sys::SUCCESS;
        // SAFETY: `context` is a live handle of the driver that resolved
        // `create`, and `properties` is a zero-terminated
        // `cl_semaphore_properties_khr` array the driver reads during the
        // call only; `errcode` is a valid out-pointer.
        let raw = unsafe { create(context.raw(), properties.as_ptr(), &mut errcode) };
        let raw = error::creation(raw, errcode, symbol)?;
        Ok(Self {
            inner: Arc::new(SemaphoreInner {
                context: context.clone(),
                raw,
            }),
        })
    }

    /// Returns the context the semaphore was created from.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.inner.context
    }

    /// Returns the raw driver handle.
    pub(crate) fn raw(&self) -> sys::SemaphoreHandle {
        self.inner.raw
    }

    /// Reads one semaphore attribute through `clGetSemaphoreInfoKHR`.
    fn info<T>(
        &self,
        param: sys::SemaphoreInfo,
        read: impl FnOnce(&mut query::QueryCall<'_>, &'static str) -> Result<T>,
    ) -> Result<T> {
        let get_info = Api::optional(
            self.context().extensions().get_semaphore_info,
            "clGetSemaphoreInfoKHR",
        )?;
        let raw = self.inner.raw;
        let mut call = |size: usize, value: *mut c_void, size_ret: *mut usize| {
            // SAFETY: `raw` is a live semaphore of the driver that
            // resolved `get_info`, and the query helpers only pass a buffer
            // of the size the driver reported plus a valid size out-pointer.
            unsafe { get_info(raw, param, size, value, size_ret) }
        };
        read(&mut call, "clGetSemaphoreInfoKHR")
    }

    /// Returns the payload a timeline semaphore would carry
    /// (`CL_SEMAPHORE_PAYLOAD_KHR`).
    ///
    /// The only type this extension defines is binary, so the answer is
    /// `0` while the semaphore is unsignaled and `1` once it is signaled.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the driver does not export the
    /// entry point, and [`Error::Status`] on driver failures.
    pub fn payload(&self) -> Result<sys::SemaphorePayload> {
        self.info(sys::SEMAPHORE_PAYLOAD_KHR, query::query_pod)
    }

    /// Returns the driver-side reference count
    /// (`CL_SEMAPHORE_REFERENCE_COUNT_KHR`).
    ///
    /// This counts the driver's references, one of which this crate holds
    /// per shared object — clones of a [`Semaphore`] share that reference
    /// rather than retaining the handle again.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`payload`](Self::payload).
    pub fn reference_count(&self) -> Result<u32> {
        self.info(sys::SEMAPHORE_REFERENCE_COUNT_KHR, query::query_u32)
    }

    /// Whether the semaphore was created so that it may be exported to
    /// another API (`CL_SEMAPHORE_EXPORTABLE_KHR`).
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`payload`](Self::payload).
    pub fn is_exportable(&self) -> Result<bool> {
        self.info(sys::SEMAPHORE_EXPORTABLE_KHR, query::query_bool)
    }

    /// Reads the exported handle of this semaphore for `handle_type`
    /// (`clGetSemaphoreHandleForTypeKHR`) as the `T` the platform stores
    /// it in.
    ///
    /// The handle is a copy of what the exporter holds: this crate does not
    /// take ownership of it, so closing it is the caller's — or the other
    /// API's — job.
    fn export_as<T: Copy + Default>(&self, handle_type: sys::ExternalSemaphoreHandleType) -> Result<T> {
        let get_handle = Api::optional(
            self.context()
                .extensions()
                .get_semaphore_handle_for_type,
            "clGetSemaphoreHandleForTypeKHR",
        )?;
        let (raw, device) = (self.inner.raw, self.context().device().raw());
        let mut reported = 0usize;
        // SAFETY: `raw` and `device` are live handles of the driver that
        // resolved `get_handle` and belong together, because the device is
        // the one of the context the semaphore was created from; a null
        // buffer with a valid size out-pointer is the first half of the
        // two-call protocol.
        let code = unsafe { get_handle(raw, device, handle_type, 0, null_mut(), &mut reported) };
        status(code, "clGetSemaphoreHandleForTypeKHR")?;
        if reported != size_of::<T>() {
            return Err(Error::Status {
                code: sys::INVALID_VALUE,
                context: "clGetSemaphoreHandleForTypeKHR",
            });
        }
        let mut value = T::default();
        // SAFETY: `value` is a `T` and the driver is told to write exactly
        // `reported == size_of::<T>()` bytes into it.
        let code = unsafe {
            get_handle(
                raw,
                device,
                handle_type,
                reported,
                (&mut value as *mut T).cast(),
                null_mut(),
            )
        };
        status(code, "clGetSemaphoreHandleForTypeKHR")?;
        Ok(value)
    }

    /// Exports this semaphore as a POSIX file descriptor
    /// (`CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR`, `CL_SEMAPHORE_HANDLE_SYNC_FD_KHR`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the driver does not export
    /// `clGetSemaphoreHandleForTypeKHR` — which includes a semaphore
    /// created without the matching export handle type — and
    /// [`Error::Status`] when the driver refuses the handle or reports it
    /// with a size other than an `int`.
    pub fn export_descriptor(&self, handle_type: sys::ExternalSemaphoreHandleType) -> Result<i32> {
        self.export_as(handle_type)
    }

    /// Exports this semaphore as a platform handle value
    /// (`CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR` and friends, a D3D12 fence).
    ///
    /// # Errors
    ///
    /// Returns the same errors as
    /// [`export_descriptor`](Self::export_descriptor), with the size the
    /// platform stores a `HANDLE` in instead of an `int`.
    pub fn export_handle(&self, handle_type: sys::ExternalSemaphoreHandleType) -> Result<usize> {
        self.export_as(handle_type)
    }
}

/// The exchange of signals: a Vulkan endpoint signals a semaphore this
/// queue waits on, or an OpenCL command signals one Vulkan waits on.
impl CommandQueue {
    /// Validates `semaphores` for a semaphore command and collects their
    /// driver handles in the order the driver expects.
    fn semaphore_handles(&self, semaphores: &[Semaphore]) -> Result<Vec<sys::SemaphoreHandle>> {
        if semaphores.is_empty() {
            return Err(Error::InvalidArgument {
                what: "a semaphore command must name at least one semaphore",
            });
        }
        for semaphore in semaphores {
            self.check_context(
                semaphore.context(),
                "a signaled semaphore belongs to a different context",
            )?;
        }
        Ok(semaphores.iter().map(Semaphore::raw).collect())
    }

    /// Runs the commands submitted before it, then blocks this queue until
    /// every semaphore in `semaphores` is signaled
    /// (`clEnqueueWaitSemaphoresKHR`).
    ///
    /// The command consumes the signal: it does not return until each
    /// binary semaphore is back in the state that lets it be signaled
    /// again, so exactly one wait belongs to each signal. Waiting twice on
    /// one semaphore without a signal in between is undefined, which is
    /// why this is a command on a queue rather than a peek at a flag.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `semaphores` is empty,
    /// [`Error::ContextMismatch`] when a semaphore came from another
    /// context, [`Error::Unsupported`] when the driver does not export the
    /// entry point, and [`Error::Status`] / [`Error::NullHandle`] on driver
    /// failures.
    pub fn wait_semaphores(&self, semaphores: &[Semaphore]) -> Result<Event> {
        let handles = self.semaphore_handles(semaphores)?;
        let wait = Api::optional(
            self.context().extensions().wait_semaphores,
            "clEnqueueWaitSemaphoresKHR",
        )?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        // SAFETY: `handles` are the live `cl_semaphore_khr` values of this
        // queue's own context, which `semaphore_handles` just checked, and
        // the vector outlives the call; the payload list is null, which the
        // specification allows because a binary semaphore requires no
        // payload; the wait list is empty and `event` is a valid
        // out-pointer.
        let code = unsafe {
            wait(
                self.raw(),
                handles.len() as u32,
                handles.as_ptr(),
                core::ptr::null(),
                0,
                core::ptr::null(),
                &mut event,
            )
        };
        let raw = error::creation(event, code, "clEnqueueWaitSemaphoresKHR")?;
        Ok(Event::new(self, raw))
    }

    /// Runs the commands submitted before it, then signals every semaphore
    /// in `semaphores` (`clEnqueueSignalSemaphoresKHR`).
    ///
    /// The signal is a queue command, not an instant flag write: a Vulkan
    /// endpoint waiting on it is released when the *command* completes, so
    /// work enqueued on this queue earlier is finished by then. A semaphore
    /// that is already signaled stays signaled — the binary type has no
    /// count — so a signal without the matching wait on the other side
    /// leaves the flag set.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`wait_semaphores`](Self::wait_semaphores).
    pub fn signal_semaphores(&self, semaphores: &[Semaphore]) -> Result<Event> {
        let handles = self.semaphore_handles(semaphores)?;
        let signal = Api::optional(
            self.context().extensions().signal_semaphores,
            "clEnqueueSignalSemaphoresKHR",
        )?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        // SAFETY: as for the wait above — the handles are live, belong to
        // this queue's context and outlive the call, no semaphore requires
        // a payload, and `event` is a valid out-pointer.
        let code = unsafe {
            signal(
                self.raw(),
                handles.len() as u32,
                handles.as_ptr(),
                core::ptr::null(),
                0,
                core::ptr::null(),
                &mut event,
            )
        };
        let raw = error::creation(event, code, "clEnqueueSignalSemaphoresKHR")?;
        Ok(Event::new(self, raw))
    }
}

impl fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Semaphore")
            .field("handle", &self.inner.raw)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    /// A context on the installed driver or the built-in one; `None` when
    /// no device answers at all.
    fn context() -> Option<Context> {
        let runtime = Runtime::load().ok()?;
        let device = runtime
            .platforms()
            .first()
            .and_then(|platform| platform.all_devices().ok())
            .and_then(|devices| devices.into_iter().next())?;
        Context::new(&device).ok()
    }

    /// Creation is the one place a driver learns what a semaphore is for;
    /// a wrong word order would be a silent capability loss, not an error.
    #[test]
    fn binary_properties_name_the_one_defined_type() {
        assert_eq!(
            binary_properties(),
            [sys::SEMAPHORE_TYPE_KHR, sys::SEMAPHORE_TYPE_BINARY_KHR, 0]
        );
    }

    /// The export list carries its own terminator *inside* the outer list,
    /// so the driver can stop reading the types without losing the rest.
    #[test]
    fn exportable_properties_terminate_the_handle_type_list() {
        let types = [
            sys::SEMAPHORE_HANDLE_OPAQUE_FD_KHR,
            sys::SEMAPHORE_HANDLE_SYNC_FD_KHR,
        ];
        assert_eq!(
            exportable_properties(&types),
            [
                sys::SEMAPHORE_TYPE_KHR,
                sys::SEMAPHORE_TYPE_BINARY_KHR,
                sys::SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,
                sys::SEMAPHORE_HANDLE_OPAQUE_FD_KHR.into(),
                sys::SEMAPHORE_HANDLE_SYNC_FD_KHR.into(),
                sys::SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR,
                0
            ]
        );
    }

    /// Importing names the handle by its own token, the same shape
    /// external-memory import uses.
    #[test]
    fn import_properties_pair_the_token_with_the_handle() {
        assert_eq!(
            import_properties(ExternalSemaphoreHandle::OpaqueFd(-1)),
            [sys::SEMAPHORE_HANDLE_OPAQUE_FD_KHR.into(), u64::MAX, 0]
        );
        assert_eq!(
            import_properties(ExternalSemaphoreHandle::D3D12Fence(7)),
            [sys::SEMAPHORE_HANDLE_D3D12_FENCE_KHR.into(), 7, 0]
        );
    }

    /// Every handle reports the token it is imported as, or the capability
    /// list is checked against the wrong entry.
    #[test]
    fn every_handle_names_the_token_it_is_imported_as() {
        assert_eq!(
            ExternalSemaphoreHandle::OpaqueFd(1).handle_type(),
            sys::SEMAPHORE_HANDLE_OPAQUE_FD_KHR
        );
        assert_eq!(
            ExternalSemaphoreHandle::SyncFd(1).handle_type(),
            sys::SEMAPHORE_HANDLE_SYNC_FD_KHR
        );
        assert_eq!(
            ExternalSemaphoreHandle::OpaqueWin32(1).handle_type(),
            sys::SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR
        );
        assert_eq!(
            ExternalSemaphoreHandle::OpaqueWin32Kmt(1).handle_type(),
            sys::SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR
        );
        assert_eq!(
            ExternalSemaphoreHandle::D3D12Fence(1).handle_type(),
            sys::SEMAPHORE_HANDLE_D3D12_FENCE_KHR
        );
    }

    /// An exportable semaphore with no types could never be exported, so it
    /// is refused before the driver is called.
    #[test]
    fn exportable_needs_at_least_one_handle_type() {
        let Some(context) = context() else {
            return;
        };
        assert!(matches!(
            Semaphore::exportable(&context, &[]),
            Err(Error::InvalidArgument { .. })
        ));
    }

    /// A driver without `cl_khr_semaphore` answers with `Unsupported` and a
    /// driver with it answers with a live object; both are dropped without
    /// panicking, including the clones that share it.
    #[test]
    fn creating_a_semaphore_never_panics() {
        let Some(context) = context() else {
            return;
        };
        if let Ok(semaphore) = Semaphore::new(&context) {
            let clone = semaphore.clone();
            assert_eq!(semaphore.raw(), clone.raw());
            assert!(semaphore.context().same_as(clone.context()));
        } else {
            assert!(!context.supports_semaphores());
        }
    }

    /// The arguments of a semaphore command are checked before the driver
    /// is asked, so a malformed request answers the same way whether or not
    /// the driver exports `cl_khr_semaphore` at all.
    #[test]
    fn semaphore_commands_validate_their_arguments_first() {
        let Some(context) = context() else {
            return;
        };
        let Ok(queue) = CommandQueue::new(&context, context.device()) else {
            return;
        };
        assert!(matches!(
            queue.wait_semaphores(&[]),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            queue.signal_semaphores(&[]),
            Err(Error::InvalidArgument { .. })
        ));
        // A second context over the same device is still a foreign
        // context, and a semaphore of that context cannot be waited on
        // here — the driver tracks the signal per context.
        let Ok(other) = Context::new(context.device()) else {
            return;
        };
        let Ok(foreign) = Semaphore::new(&other) else {
            return;
        };
        assert!(matches!(
            queue.wait_semaphores(core::slice::from_ref(&foreign)),
            Err(Error::ContextMismatch { .. })
        ));
        assert!(matches!(
            queue.signal_semaphores(&[foreign]),
            Err(Error::ContextMismatch { .. })
        ));
    }

    /// The information queries describe the object that was just created:
    /// unsignaled, held once, and not exportable — which is also why no
    /// descriptor may be read from it.
    #[test]
    fn a_fresh_semaphore_is_unsignaled_and_not_exportable() {
        let Some(context) = context() else {
            return;
        };
        if !context.supports_semaphores() {
            return;
        }
        let Ok(semaphore) = Semaphore::new(&context) else {
            return;
        };
        assert_eq!(semaphore.payload(), Ok(0));
        assert_eq!(semaphore.reference_count(), Ok(1));
        assert_eq!(semaphore.is_exportable(), Ok(false));
        assert!(
            semaphore
                .export_descriptor(sys::SEMAPHORE_HANDLE_OPAQUE_FD_KHR)
                .is_err()
        );
    }

    /// A semaphore created for one handle type is reported as exportable
    /// and hands out that handle; a type it was not created for stays
    /// refused, because the creation list is the only thing the driver
    /// agrees to export later.
    #[test]
    fn an_exportable_semaphore_names_its_handle_type() {
        let Some(context) = context() else {
            return;
        };
        if !context.supports_semaphores() || !context.supports_semaphore_handles() {
            return;
        }
        let Ok(semaphore) = Semaphore::exportable(&context, &[sys::SEMAPHORE_HANDLE_SYNC_FD_KHR]) else {
            return;
        };
        assert_eq!(semaphore.is_exportable(), Ok(true));
        assert!(
            semaphore
                .export_descriptor(sys::SEMAPHORE_HANDLE_SYNC_FD_KHR)
                .is_ok()
        );
        assert!(
            semaphore
                .export_descriptor(sys::SEMAPHORE_HANDLE_OPAQUE_FD_KHR)
                .is_err()
        );
    }
}
