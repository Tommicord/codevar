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

//! Command queues and the operations enqueued on them.
//!
//! A [`CommandQueue`] belongs to one [`Context`] and executes commands
//! for its device. Buffer transfers use the *blocking* form of the
//! OpenCL API: the driver copies the caller's slice before returning,
//! so Rust slices can never outlive the transfer. Every enqueue returns
//! an [`Event`]; [`Event::wait`] flushes the queue first, so a plain
//! `wait` after an enqueue is always sufficient.

use alloc::sync::Arc;
use core::fmt;
use core::mem::size_of_val;
use core::ptr::null_mut;

use codevar_logger::log_warn;

use crate::api::Api;
use crate::buffer::{Buffer, check_range};
use crate::context::Context;
use crate::error::{self, Error, Result, status};
use crate::event::Event;
use crate::kernel::Kernel;
use crate::platform::Device;
use crate::sys;

/// Optional features enabled at queue creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QueueProperties {
    /// Default queue.
    #[default]
    None,
    /// Collect device timestamps for [`Event::profiling_timestamp`].
    Profiling,
}

impl QueueProperties {
    /// Returns the `cl_command_queue_properties` bits of this value.
    #[must_use]
    pub const fn bits(self) -> u64 {
        match self {
            Self::None => 0,
            Self::Profiling => sys::QUEUE_PROFILING_ENABLE,
        }
    }
}

/// An ordered queue of commands for one device.
#[derive(Clone)]
pub struct CommandQueue {
    inner: Arc<QueueInner>,
}

struct QueueInner {
    api: Arc<Api>,
    context: Context,
    raw: sys::CommandQueueHandle,
}

impl Drop for QueueInner {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by `clCreateCommandQueue`
        // through this same `api`, and this `Drop` runs exactly once
        // for the single shared `QueueInner`.
        let code = unsafe { (self.api.release_command_queue)(self.raw) };
        if code != sys::SUCCESS {
            log_warn!("clReleaseCommandQueue failed with {}", sys::error_name(code));
        }
    }
}

// SAFETY: `QueueInner` is an opaque driver handle plus shared
// function pointers; queues are reference-counted by the driver and
// all entry points used here are thread-safe per the specification.
unsafe impl Send for QueueInner {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for QueueInner {}

impl CommandQueue {
    /// Creates a queue for `device` with default properties.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] when `device` is not the
    /// context's device, and [`Error::Status`] / [`Error::NullHandle`]
    /// on driver failures.
    pub fn new(context: &Context, device: &Device) -> Result<Self> {
        Self::with_properties(context, device, QueueProperties::None)
    }

    /// Creates a queue for `device` with the given [`QueueProperties`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] when `device` is not the
    /// context's device, and [`Error::Status`] / [`Error::NullHandle`]
    /// on driver failures.
    pub fn with_properties(context: &Context, device: &Device, properties: QueueProperties) -> Result<Self> {
        if device.raw() != context.device().raw() {
            return Err(Error::ContextMismatch {
                what: "the queue device is not the device of the context",
            });
        }
        let api = context.api().clone();
        let mut errcode = sys::SUCCESS;
        // SAFETY: `context`/`device` are live handles of `api` and the
        // device check above guarantees they belong together;
        // `errcode` is a valid out-pointer.
        let raw = unsafe {
            (api.create_command_queue)(context.raw(), device.raw(), properties.bits(), &mut errcode)
        };
        let raw = error::creation(raw, errcode, "clCreateCommandQueue")?;
        Ok(Self {
            inner: Arc::new(QueueInner {
                api,
                context: context.clone(),
                raw,
            }),
        })
    }

    /// Submits pending commands to the device.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the flush.
    pub fn flush(&self) -> Result<()> {
        status(
            // SAFETY: the handle is live and owned by this queue.
            unsafe { (self.inner.api.flush)(self.inner.raw) },
            "clFlush",
        )
    }

    /// Waits until every submitted command finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the wait.
    pub fn finish(&self) -> Result<()> {
        status(
            // SAFETY: the handle is live and owned by this queue.
            unsafe { (self.inner.api.finish)(self.inner.raw) },
            "clFinish",
        )
    }

    /// Copies `data` into `buffer` starting at `offset_bytes`.
    ///
    /// The transfer is blocking: `data` is fully consumed before this
    /// function returns.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] when the buffer belongs to
    /// another context, [`Error::InvalidArgument`] when the range is
    /// invalid, and [`Error::Status`] / [`Error::NullHandle`] on
    /// driver failures.
    pub fn write_buffer<T>(&self, buffer: &Buffer, offset_bytes: usize, data: &[T]) -> Result<Event> {
        self.check_context(
            buffer.context(),
            "the written buffer belongs to a different context",
        )?;
        let size = size_of_val(data);
        check_range(buffer.size(), offset_bytes, size)?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        // SAFETY: all handles are live and belong to the same context;
        // `data` points at `size` initialized bytes which the blocking
        // form copies before returning; the wait-list is empty and
        // `event` is a valid out-pointer.
        let code = unsafe {
            (self.inner.api.enqueue_write_buffer)(
                self.inner.raw,
                buffer.raw(),
                sys::BLOCKING,
                offset_bytes,
                size,
                data.as_ptr().cast(),
                0,
                core::ptr::null(),
                &mut event,
            )
        };
        let raw = error::creation(event, code, "clEnqueueWriteBuffer")?;
        Ok(Event::new(self, raw))
    }

    /// Copies `buffer` data starting at `offset_bytes` into `data`.
    ///
    /// The transfer is blocking: `data` is fully written before this
    /// function returns.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] when the buffer belongs to
    /// another context, [`Error::InvalidArgument`] when the range is
    /// invalid, and [`Error::Status`] / [`Error::NullHandle`] on
    /// driver failures.
    pub fn read_buffer<T>(&self, buffer: &Buffer, offset_bytes: usize, data: &mut [T]) -> Result<Event> {
        self.check_context(buffer.context(), "the read buffer belongs to a different context")?;
        let size = size_of_val(data);
        check_range(buffer.size(), offset_bytes, size)?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        // SAFETY: all handles are live and belong to the same context;
        // `data` points at `size` writable bytes for the duration of
        // the blocking call; the wait-list is empty and `event` is a
        // valid out-pointer.
        let code = unsafe {
            (self.inner.api.enqueue_read_buffer)(
                self.inner.raw,
                buffer.raw(),
                sys::BLOCKING,
                offset_bytes,
                size,
                data.as_mut_ptr().cast(),
                0,
                core::ptr::null(),
                &mut event,
            )
        };
        let raw = error::creation(event, code, "clEnqueueReadBuffer")?;
        Ok(Event::new(self, raw))
    }

    /// Copies `size` bytes inside `src` (from `src_offset`) into `dst`
    /// (at `dst_offset`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] when either buffer belongs to
    /// another context, [`Error::InvalidArgument`] when a range is
    /// invalid, and [`Error::Status`] / [`Error::NullHandle`] on
    /// driver failures.
    pub fn copy_buffer(
        &self,
        src: &Buffer,
        dst: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<Event> {
        self.check_context(src.context(), "the source buffer belongs to a different context")?;
        self.check_context(
            dst.context(),
            "the destination buffer belongs to a different context",
        )?;
        check_range(src.size(), src_offset, size)?;
        check_range(dst.size(), dst_offset, size)?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        // SAFETY: all handles are live and belong to the same context;
        // both ranges were validated above; the wait-list is empty and
        // `event` is a valid out-pointer.
        let code = unsafe {
            (self.inner.api.enqueue_copy_buffer)(
                self.inner.raw,
                src.raw(),
                dst.raw(),
                src_offset,
                dst_offset,
                size,
                0,
                core::ptr::null(),
                &mut event,
            )
        };
        let raw = error::creation(event, code, "clEnqueueCopyBuffer")?;
        Ok(Event::new(self, raw))
    }

    /// Enqueues `kernel` over `global_work_size`.
    ///
    /// `local_work_size` lets the caller pin the work-group shape; when
    /// `None` the implementation chooses one. Both slices may have one
    /// to three entries (the three spatial dimensions); every entry
    /// must be non-zero, and a local shape must match the global shape
    /// dimension-for-dimension.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] when the kernel belongs to
    /// another context, [`Error::InvalidArgument`] for malformed work
    /// sizes, and [`Error::Status`] / [`Error::NullHandle`] on
    /// driver failures.
    pub fn enqueue_kernel(
        &self,
        kernel: &Kernel,
        global_work_size: &[usize],
        local_work_size: Option<&[usize]>,
    ) -> Result<Event> {
        self.check_context(
            kernel.context(),
            "the kernel belongs to a different context than the queue",
        )?;
        check_work_size(global_work_size, local_work_size)?;
        let mut event = sys::EventHandle::from_raw(null_mut());
        let local = local_work_size.map_or(core::ptr::null(), |sizes| sizes.as_ptr());
        // SAFETY: all handles are live and belong to the same context;
        // the work sizes were validated; the wait-list is empty and
        // `event` is a valid out-pointer.
        let code = unsafe {
            (self.inner.api.enqueue_nd_range_kernel)(
                self.inner.raw,
                kernel.raw(),
                global_work_size.len() as u32,
                core::ptr::null(),
                global_work_size.as_ptr(),
                local,
                0,
                core::ptr::null(),
                &mut event,
            )
        };
        let raw = error::creation(event, code, "clEnqueueNDRangeKernel")?;
        Ok(Event::new(self, raw))
    }

    /// Returns the context the queue was created from.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.inner.context
    }

    /// Returns the resolved entry points of the owning driver.
    pub(crate) fn api(&self) -> &Arc<Api> {
        &self.inner.api
    }

    /// Fails when `other` is not the queue's own context.
    fn check_context(&self, other: &Context, what: &'static str) -> Result<()> {
        if self.inner.context.same_as(other) {
            Ok(())
        } else {
            Err(Error::ContextMismatch { what })
        }
    }
}

/// Validates the global/local work sizes of an ND-range launch.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when the dimension count is
/// outside 1..=3, an entry is zero, or the local shape does not match
/// the global shape.
pub(crate) fn check_work_size(global: &[usize], local: Option<&[usize]>) -> Result<()> {
    if global.is_empty() || global.len() > 3 {
        return Err(Error::InvalidArgument {
            what: "a work size must have one to three dimensions",
        });
    }
    if global.contains(&0) {
        return Err(Error::InvalidArgument {
            what: "global work size entries must be non-zero",
        });
    }
    if let Some(local) = local {
        if local.len() != global.len() {
            return Err(Error::InvalidArgument {
                what: "local and global work sizes must have the same dimensions",
            });
        }
        if local.contains(&0) {
            return Err(Error::InvalidArgument {
                what: "local work size entries must be non-zero",
            });
        }
    }
    Ok(())
}

impl fmt::Debug for CommandQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandQueue")
            .field("handle", &self.inner.raw)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn queue_is_send_and_sync() {
        _assert_send_sync::<CommandQueue>();
    }

    #[test]
    fn profiling_bits_match_the_header() {
        assert_eq!(QueueProperties::None.bits(), 0);
        assert_eq!(QueueProperties::Profiling.bits(), 2);
        assert_eq!(QueueProperties::default(), QueueProperties::None);
    }

    #[test]
    fn work_size_validation_covers_every_shape_rule() {
        assert!(check_work_size(&[64], None).is_ok());
        assert!(check_work_size(&[64, 32], Some(&[8, 4])).is_ok());

        assert!(matches!(
            check_work_size(&[], None),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            check_work_size(&[1, 2, 3, 4], None),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            check_work_size(&[64, 0], None),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            check_work_size(&[64, 32], Some(&[8])),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            check_work_size(&[64], Some(&[0])),
            Err(Error::InvalidArgument { .. })
        ));
    }
}
