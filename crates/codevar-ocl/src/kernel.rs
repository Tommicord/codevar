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

//! Kernel entry points.
//!
//! A [`Kernel`] is created from a built [`Program`](crate::Program) and
//! configured with [`Kernel::set_arg`] before it is enqueued through
//! [`CommandQueue::enqueue_kernel`](crate::CommandQueue::enqueue_kernel).
//! Buffer arguments passed to `set_arg` are retained by the kernel, so
//! enqueues never risk the driver holding a freed buffer.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ffi::{c_char, c_void};
use core::fmt;
use core::mem::size_of_val;

use codevar_logger::log_warn;

use crate::api::Api;
use crate::buffer::Buffer;
use crate::error::{Error, Result, creation, status};
use crate::program::Program;
use crate::query;
use crate::sys;

/// One kernel argument value.
#[derive(Debug, Clone, Copy)]
pub enum Arg<'a> {
    /// A buffer (memory object) argument.
    Buffer(&'a Buffer),
    /// A work-group local memory allocation of the given byte size.
    Local(usize),
    /// A 32-bit unsigned integer.
    U32(u32),
    /// A 32-bit signed integer.
    I32(i32),
    /// A 64-bit unsigned integer.
    U64(u64),
    /// A 64-bit signed integer.
    I64(i64),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit double.
    F64(f64),
}

/// A kernel entry point bound to one [`Program`].
pub struct Kernel {
    api: Arc<Api>,
    program: Program,
    raw: sys::KernelHandle,
    held_args: Vec<Option<Buffer>>,
}

// SAFETY: `Kernel` holds a shared `Arc<Api>`, a `Program` (already
// `Send + Sync`) and an opaque driver handle; kernels may be used from
// any thread as long as one thread mutates the arguments at a time,
// which Rust's exclusive `&mut self` on `set_arg` guarantees.
unsafe impl Send for Kernel {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Creates a kernel for `name` from a built `program`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `name` is empty or
    /// contains a NUL byte, and [`Error::Status`] / [`Error::NullHandle`]
    /// when the program has no such entry point.
    pub fn new(program: &Program, name: &str) -> Result<Self> {
        if name.is_empty() || name.bytes().any(|byte| byte == 0) {
            return Err(Error::InvalidArgument {
                what: "a kernel name must be non-empty and NUL-free",
            });
        }
        let mut c_name = name.as_bytes().to_vec();
        c_name.push(0);
        let api = program.api().clone();
        let mut errcode = sys::SUCCESS;
        // SAFETY: `program` is a live handle of `api` and `c_name` is a
        // valid NUL-terminated string that is only read during the call;
        // `errcode` is a valid out-pointer.
        let raw =
            unsafe { (api.create_kernel)(program.raw(), c_name.as_ptr().cast::<c_char>(), &mut errcode) };
        let raw = creation(raw, errcode, "clCreateKernel")?;
        Ok(Self {
            api,
            program: program.clone(),
            raw,
            held_args: Vec::new(),
        })
    }

    /// Sets the argument at `index`.
    ///
    /// Argument indices follow the order of the kernel signature in the
    /// OpenCL C source. Buffer arguments are retained by the kernel
    /// until they are replaced or the kernel is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] when a buffer belongs to a
    /// different context, [`Error::InvalidArgument`] for a zero-sized
    /// local allocation, and [`Error::Status`] when the driver rejects
    /// the value (typically a type or index mismatch).
    pub fn set_arg(&mut self, index: usize, arg: Arg<'_>) -> Result<()> {
        match arg {
            Arg::Buffer(buffer) => self.set_buffer_arg(index, buffer),
            Arg::Local(size) => {
                if size == 0 {
                    return Err(Error::InvalidArgument {
                        what: "local argument size must not be zero",
                    });
                }
                self.call_set_arg(index, size, core::ptr::null())?;
                self.clear_held(index);
                Ok(())
            }
            Arg::U32(value) => self.set_scalar_arg(index, &value.to_ne_bytes()),
            Arg::I32(value) => self.set_scalar_arg(index, &value.to_ne_bytes()),
            Arg::U64(value) => self.set_scalar_arg(index, &value.to_ne_bytes()),
            Arg::I64(value) => self.set_scalar_arg(index, &value.to_ne_bytes()),
            Arg::F32(value) => self.set_scalar_arg(index, &value.to_ne_bytes()),
            Arg::F64(value) => self.set_scalar_arg(index, &value.to_ne_bytes()),
        }
    }

    /// Returns the kernel name as reported by the driver.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the query fails.
    pub fn name(&self) -> Result<alloc::string::String> {
        let mut call = |size: usize, value: *mut c_void, size_ret: *mut usize| {
            // SAFETY: `raw` is a live handle of the driver that
            // resolved `api`, and `query` provides valid buffers.
            unsafe { (self.api.get_kernel_info)(self.raw, sys::KERNEL_FUNCTION_NAME, size, value, size_ret) }
        };
        query::query_string(&mut call, "clGetKernelInfo")
    }

    /// Returns the number of arguments of the kernel.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the query fails.
    pub fn num_args(&self) -> Result<u32> {
        let mut call = |size: usize, value: *mut c_void, size_ret: *mut usize| {
            // SAFETY: `raw` is a live handle of the driver that
            // resolved `api`, and `query` provides valid buffers.
            unsafe { (self.api.get_kernel_info)(self.raw, sys::KERNEL_NUM_ARGS, size, value, size_ret) }
        };
        query::query_u32(&mut call, "clGetKernelInfo")
    }

    /// Returns the program the kernel was created from.
    #[must_use]
    pub const fn program(&self) -> &Program {
        &self.program
    }

    /// Returns the context of the kernel's program.
    pub(crate) fn context(&self) -> &crate::context::Context {
        self.program.context()
    }

    /// Returns the raw driver handle.
    pub(crate) const fn raw(&self) -> sys::KernelHandle {
        self.raw
    }

    /// Forwards one `clSetKernelArg` call.
    fn call_set_arg(&self, index: usize, size: usize, value: *const c_void) -> Result<()> {
        status(
            // SAFETY: `raw` is a live handle of the driver that
            // resolved `api`; `value` is either null (local storage)
            // or points at `size` readable bytes for the duration of
            // the call.
            unsafe { (self.api.set_kernel_arg)(self.raw, index as u32, size, value) },
            "clSetKernelArg",
        )
    }

    /// Sets a scalar (bit-pattern) argument.
    fn set_scalar_arg(&mut self, index: usize, bytes: &[u8]) -> Result<()> {
        self.call_set_arg(index, bytes.len(), bytes.as_ptr().cast())?;
        self.clear_held(index);
        Ok(())
    }

    /// Sets a buffer argument, retaining it on success.
    fn set_buffer_arg(&mut self, index: usize, buffer: &Buffer) -> Result<()> {
        if !buffer.context().same_as(self.context()) {
            return Err(Error::ContextMismatch {
                what: "the buffer belongs to a different context than the kernel",
            });
        }
        let handle = buffer.raw();
        self.call_set_arg(index, size_of_val(&handle), (&raw const handle).cast())?;
        // Only retain after the driver accepted the argument, so a
        // failed call never leaves a stale buffer in the kernel.
        self.held_args.resize_with(index + 1, || None);
        self.held_args[index] = Some(buffer.clone());
        Ok(())
    }

    /// Drops a previously retained buffer at `index`.
    fn clear_held(&mut self, index: usize) {
        if let Some(slot) = self.held_args.get_mut(index) {
            *slot = None;
        }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by `clCreateKernel` through
        // this same `api`, and each `Kernel` value owns its handle
        // exactly once (`Kernel` is not `Clone`).
        let code = unsafe { (self.api.release_kernel)(self.raw) };
        if code != sys::SUCCESS {
            log_warn!("clReleaseKernel failed with {}", sys::error_name(code));
        }
    }
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel")
            .field("handle", &self.raw)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn kernel_is_send_and_sync() {
        _assert_send_sync::<Kernel>();
    }
}
