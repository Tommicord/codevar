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

//! The `cl_khr` entry points a context resolved for its device.
//!
//! Importing Vulkan memory and synchronizing with Vulkan semaphores needs
//! nine extension functions that are not part of any OpenCL core version:
//! `cl_khr_external_memory` contributes the acquire/release pair,
//! `cl_khr_semaphore` the create/retain/release/info quartet, and
//! `cl_khr_external_semaphore` the handle export plus the wait/signal pair.
//! None of them is exported by the ICD loader as a plain symbol, so they are
//! looked up per platform with `clGetExtensionFunctionAddressForPlatform`
//! (see [`crate::api::Api::extension_fn`]).
//!
//! A driver that lacks an extension simply returns null for its entry
//! points, which this module records as `None`; the safe wrappers then report
//! [`Error::Unsupported`](crate::error::Error::Unsupported) with the exact
//! entry-point name, and [`Extensions::has_semaphores`] /
//! [`Extensions::has_external_memory`] answer capability questions without
//! touching the driver again.
//!
//! Resolution happens once per [`Context`](crate::context::Context) because
//! each lookup is a driver call; the resulting [`Extensions`] is shared with
//! every queue, buffer and semaphore created from that context.

use crate::api::Api;
use crate::sys;

/// Declares one cached entry point per line: the field, its type and the
/// driver symbol it is resolved from.
macro_rules! define_extensions {
    ($($field:ident : $ty:ident => $symbol:expr),* $(,)?) => {
        /// Extension entry points resolved for one context's platform.
        ///
        /// Every field is `None` on a driver without the extension that
        /// provides it, so the whole struct is always constructible.
        pub(crate) struct Extensions {
            $(
                #[doc = concat!("`", stringify!($symbol), "` when the driver exports it.")]
                pub(crate) $field: Option<sys::$ty>,
            )*
        }

        impl Extensions {
            /// Resolves every entry point for `platform` through `api`.
            ///
            /// A platform that does not export a resolver at all yields an
            /// all-`None` set rather than an error: capability probing is
            /// expected to work on plain OpenCL 1.2 drivers.
            pub(crate) fn resolve(api: &Api, platform: sys::PlatformHandle) -> Self {
                Self {
                    $(
                        // SAFETY: `$ty` is the entry-point type `sys`
                        // declares for the driver function named by
                        // `$symbol`, and `platform` is the platform this
                        // `api` was resolved from, so the lookup reaches the
                        // implementation that owns both.
                        $field: unsafe { api.extension_fn::<sys::$ty>(platform, $symbol) },
                    )*
                }
            }
        }
    };
}

define_extensions! {
    acquire_external_mem_objects: EnqueueAcquireExternalMemObjects
        => c"clEnqueueAcquireExternalMemObjectsKHR",
    release_external_mem_objects: EnqueueReleaseExternalMemObjects
        => c"clEnqueueReleaseExternalMemObjectsKHR",
    create_semaphore_with_properties: CreateSemaphoreWithProperties
        => c"clCreateSemaphoreWithPropertiesKHR",
    retain_semaphore: RetainSemaphore => c"clRetainSemaphoreKHR",
    release_semaphore: ReleaseSemaphore => c"clReleaseSemaphoreKHR",
    get_semaphore_info: GetSemaphoreInfo => c"clGetSemaphoreInfoKHR",
    get_semaphore_handle_for_type: GetSemaphoreHandleForType
        => c"clGetSemaphoreHandleForTypeKHR",
    wait_semaphores: EnqueueWaitSemaphores => c"clEnqueueWaitSemaphoresKHR",
    signal_semaphores: EnqueueSignalSemaphores => c"clEnqueueSignalSemaphoresKHR",
}

/// Capability answers over the resolved set; [`crate::Context`] reports
/// them as its public support queries.
impl Extensions {
    /// Whether the driver exposes the complete `cl_khr_semaphore` surface,
    /// including the wait and signal commands.
    ///
    /// `cl_khr_external_semaphore` is additive: [`Self::semaphore_handle`]
    /// still answers whether handle exchange is possible.
    #[must_use]
    pub(crate) const fn has_semaphores(&self) -> bool {
        self.create_semaphore_with_properties.is_some()
            && self.retain_semaphore.is_some()
            && self.release_semaphore.is_some()
            && self.get_semaphore_info.is_some()
            && self.wait_semaphores.is_some()
            && self.signal_semaphores.is_some()
    }

    /// Whether the driver can import external memory and take ownership of
    /// it on a command queue.
    #[must_use]
    pub(crate) const fn has_external_memory(&self) -> bool {
        self.acquire_external_mem_objects.is_some() && self.release_external_mem_objects.is_some()
    }

    /// Whether a created semaphore can be exported to, or imported from, a
    /// handle of another API.
    #[must_use]
    pub(crate) const fn semaphore_handle(&self) -> bool {
        self.get_semaphore_handle_for_type.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::tests::{api_platform, api_with_fake_dispatch};

    /// The fake driver of `crate::api` exports four semaphore entry points
    /// and nothing else; the cached set must mirror exactly that.
    #[test]
    fn resolve_keeps_only_the_entry_points_a_driver_exports() {
        let api = api_with_fake_dispatch(true, false);
        let extensions = Extensions::resolve(&api, api_platform());
        assert!(
            extensions
                .create_semaphore_with_properties
                .is_some()
        );
        assert!(extensions.retain_semaphore.is_some());
        assert!(extensions.release_semaphore.is_some());
        assert!(extensions.wait_semaphores.is_some());
        assert!(extensions.get_semaphore_info.is_none());
        assert!(extensions.acquire_external_mem_objects.is_none());
    }

    /// A partial semaphore surface must not report the extension as usable:
    /// `has_semaphores` is the predicate the safe wrappers rely on.
    #[test]
    fn incomplete_semaphore_surface_is_not_semaphores() {
        let api = api_with_fake_dispatch(true, false);
        let extensions = Extensions::resolve(&api, api_platform());
        assert!(!extensions.has_semaphores());
        assert!(!extensions.has_external_memory());
        assert!(!extensions.semaphore_handle());
    }

    /// A driver without any resolver exports no extension entry points at
    /// all, which is a capability answer rather than an error.
    #[test]
    fn a_driver_without_resolvers_yields_an_empty_set() {
        let api = api_with_fake_dispatch(false, false);
        let extensions = Extensions::resolve(&api, api_platform());
        assert!(!extensions.has_semaphores());
        assert!(
            extensions
                .create_semaphore_with_properties
                .is_none()
        );
        assert!(extensions.signal_semaphores.is_none());
    }
}
