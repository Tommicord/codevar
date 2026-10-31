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
//! (`CL_SEMAPHORE_TYPE_BINARY_KHR`), whose payload is `0` for the unset
//! and `1` for the set side. The wait and signal commands therefore take
//! semaphores rather than semaphore/payload pairs.
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

use core::fmt;

use codevar_logger::log_warn;

use crate::api::Api;
use crate::context::Context;
use crate::error::{self, Error, Result};
use crate::properties::{self, PropertyList};
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
            assert!(semaphore.context().same_as(clone.context()));
        } else {
            assert!(!context.supports_semaphores());
        }
    }
}
