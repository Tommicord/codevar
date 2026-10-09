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

//! Execution contexts.
//!
//! A [`Context`] binds one [`Device`] and is the parent of every other
//! object in this crate: [`Buffer`](crate::Buffer),
//! [`CommandQueue`](crate::CommandQueue) and
//! [`Program`](crate::Program) are created from it and keep it alive.
//! Objects from different contexts can never be mixed — every operation
//! that takes two related objects compares context identity first and
//! fails with [`Error::ContextMismatch`](crate::Error::ContextMismatch).
//!
//! Cloning a [`Context`] is cheap (an `Arc` increment); all clones share
//! the single driver-side handle, which is released when the last clone
//! is dropped.

use alloc::sync::Arc;
use core::fmt;
use core::ptr::null;

use codevar_logger::log_warn;

use crate::api::Api;
use crate::error::{self, Result};
use crate::platform::Device;
use crate::sys;

/// A single-device OpenCL context.
#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

struct ContextInner {
    api: Arc<Api>,
    device: Device,
    raw: sys::ContextHandle,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by `clCreateContext` through
        // this same `api`, and this `Drop` runs exactly once for the
        // single shared `ContextInner`.
        let code = unsafe { (self.api.release_context)(self.raw) };
        if code != sys::SUCCESS {
            log_warn!("clReleaseContext failed with {}", sys::error_name(code));
        }
    }
}

// SAFETY: `ContextInner` is an opaque driver handle plus shared
// function pointers; the driver reference-counts contexts and all
// entry points used here are thread-safe per the OpenCL specification.
unsafe impl Send for ContextInner {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for ContextInner {}

impl Context {
    /// Creates a context bound to `device`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the request and
    /// [`Error::NullHandle`] when it reports success without a handle.
    pub fn new(device: &Device) -> Result<Self> {
        let api = device.api().clone();
        let handle = device.raw();
        let mut errcode = sys::SUCCESS;
        // SAFETY: `properties`/`user_data` are null as the API permits
        // (default platform, no callback), `handle` is a live device of
        // the driver that resolved `api`, and `errcode` is a valid
        // out-pointer.
        let raw =
            unsafe { (api.create_context)(null(), 1, &handle, None, core::ptr::null_mut(), &mut errcode) };
        let raw = error::creation(raw, errcode, "clCreateContext")?;
        Ok(Self {
            inner: Arc::new(ContextInner {
                api,
                device: device.clone(),
                raw,
            }),
        })
    }

    /// Returns the resolved entry points of the owning driver.
    pub(crate) fn api(&self) -> &Arc<Api> {
        &self.inner.api
    }

    /// Returns the raw driver handle.
    pub(crate) fn raw(&self) -> sys::ContextHandle {
        self.inner.raw
    }

    /// Returns the device this context is bound to.
    pub(crate) fn device(&self) -> &Device {
        &self.inner.device
    }

    /// Whether `self` and `other` are the same context (not merely
    /// contexts over equal devices).
    pub(crate) fn same_as(&self, other: &Context) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("handle", &self.inner.raw)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    /// Contexts must be shareable across threads; nothing to assert
    /// beyond compiling `Send + Sync` bounds.
    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn context_is_send_and_sync() {
        _assert_send_sync::<Context>();
    }

    /// On machines with OpenCL a context must be creatable, cloneable
    /// with shared identity, and releasable without panicking.
    #[test]
    fn new_context_shares_identity_between_clones_when_available() {
        let Ok(runtime) = Runtime::load() else {
            return;
        };
        let Some(device) = runtime
            .platforms()
            .first()
            .and_then(|p| p.all_devices().ok())
            .and_then(|d| d.into_iter().next())
        else {
            return;
        };
        let Ok(context) = Context::new(&device) else {
            return;
        };
        let clone = context.clone();
        assert!(context.same_as(&clone));
        assert_eq!(context.raw(), clone.raw());
        drop(clone);
        drop(context);
    }
}
