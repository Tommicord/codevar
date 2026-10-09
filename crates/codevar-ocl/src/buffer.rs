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

//! Device buffers.
//!
//! A [`Buffer`] is a chunk of device memory created from a
//! [`Context`](crate::Context). Transfers go through
//! [`CommandQueue`](crate::CommandQueue) methods; kernel arguments keep
//! their buffers alive automatically (see [`Kernel::set_arg`](crate::Kernel::set_arg)).
//!
//! Cloning a [`Buffer`] is cheap and shares the single driver-side
//! handle, which is released when the last clone is dropped.

use alloc::sync::Arc;
use core::fmt;
use core::mem::size_of_val;
use core::ptr::null_mut;

use codevar_logger::log_warn;

use crate::api::Api;
use crate::context::Context;
use crate::error::{self, Error, Result};
use crate::sys;

/// A device memory object.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

struct BufferInner {
    api: Arc<Api>,
    context: Context,
    raw: sys::MemHandle,
    size: usize,
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by `clCreateBuffer` through
        // this same `api`, and this `Drop` runs exactly once for the
        // single shared `BufferInner`.
        let code = unsafe { (self.api.release_mem_object)(self.raw) };
        if code != sys::SUCCESS {
            log_warn!("clReleaseMemObject failed with {}", sys::error_name(code));
        }
    }
}

// SAFETY: `BufferInner` is an opaque driver handle plus shared
// function pointers; buffers are reference-counted by the driver and
// all entry points used here are thread-safe per the specification.
unsafe impl Send for BufferInner {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for BufferInner {}

impl Buffer {
    /// Creates an uninitialized read/write buffer of `size` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `size` is zero, and
    /// [`Error::Status`] / [`Error::NullHandle`] on driver failures.
    pub fn new(context: &Context, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::InvalidArgument {
                what: "a buffer must be at least one byte large",
            });
        }
        let api = context.api().clone();
        let mut errcode = sys::SUCCESS;
        // SAFETY: `context` is a live handle of `api`; no host pointer
        // is passed for an uninitialized allocation; `errcode` is a
        // valid out-pointer.
        let raw = unsafe {
            (api.create_buffer)(context.raw(), sys::MEM_READ_WRITE, size, null_mut(), &mut errcode)
        };
        let raw = error::creation(raw, errcode, "clCreateBuffer")?;
        Ok(Self {
            inner: Arc::new(BufferInner {
                api,
                context: context.clone(),
                raw,
                size,
            }),
        })
    }

    /// Creates a read/write buffer initialised with a copy of `data`.
    ///
    /// The copy happens inside the driver call, so `data` may be dropped
    /// as soon as this function returns.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `data` is empty, and
    /// [`Error::Status`] / [`Error::NullHandle`] on driver failures.
    pub fn from_slice<T>(context: &Context, data: &[T]) -> Result<Self> {
        let size = size_of_val(data);
        if size == 0 {
            return Err(Error::InvalidArgument {
                what: "buffer data must not be empty",
            });
        }
        let api = context.api().clone();
        let mut errcode = sys::SUCCESS;
        // SAFETY: `context` is a live handle of `api`; `data` points at
        // `size` initialized bytes which `CL_MEM_COPY_HOST_PTR` copies
        // during the call; `errcode` is a valid out-pointer.
        let raw = unsafe {
            (api.create_buffer)(
                context.raw(),
                sys::MEM_READ_WRITE | sys::MEM_COPY_HOST_PTR,
                size,
                data.as_ptr().cast_mut().cast(),
                &mut errcode,
            )
        };
        let raw = error::creation(raw, errcode, "clCreateBuffer")?;
        Ok(Self {
            inner: Arc::new(BufferInner {
                api,
                context: context.clone(),
                raw,
                size,
            }),
        })
    }

    /// Returns the buffer size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.size
    }

    /// Returns the context the buffer was created from.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.inner.context
    }

    /// Returns the raw driver handle.
    pub(crate) fn raw(&self) -> sys::MemHandle {
        self.inner.raw
    }
}

/// Validates a transfer range against a buffer of `buffer_size` bytes.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when the range overflows,
/// exceeds the buffer, or requests zero bytes.
pub(crate) fn check_range(buffer_size: usize, offset: usize, size: usize) -> Result<()> {
    let end = offset
        .checked_add(size)
        .ok_or(Error::InvalidArgument {
            what: "buffer range overflows the address space",
        })?;
    if end > buffer_size {
        return Err(Error::InvalidArgument {
            what: "buffer range exceeds the buffer size",
        });
    }
    if size == 0 {
        return Err(Error::InvalidArgument {
            what: "transfer size must not be zero",
        });
    }
    Ok(())
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("handle", &self.inner.raw)
            .field("size", &self.inner.size)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn buffer_is_send_and_sync() {
        _assert_send_sync::<Buffer>();
    }

    #[test]
    fn range_validation_accepts_exact_fit_and_rejects_the_rest() {
        assert!(check_range(64, 0, 64).is_ok());
        assert!(check_range(64, 32, 32).is_ok());

        assert!(matches!(
            check_range(64, 32, 33),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            check_range(64, 64, 0),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            check_range(64, usize::MAX, 2),
            Err(Error::InvalidArgument { .. })
        ));
    }
}
