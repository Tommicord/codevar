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

//! Compiled programs.
//!
//! A [`Program`] is created from OpenC (or OpenCL C) source and built
//! for the context's device with [`Program::build`]. Source and build
//! options are owned copies; on failure the driver's build log is
//! returned inside [`Error::BuildFailed`](crate::Error::BuildFailed).

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ffi::{c_char, c_void};
use core::fmt;
use core::ptr::null_mut;

use codevar_logger::log_warn;

use crate::api::Api;
use crate::context::Context;
use crate::error::{Error, Result, creation, status};
use crate::kernel::Kernel;
use crate::query;
use crate::sys;

/// A program compiled for one context's device.
#[derive(Clone)]
pub struct Program {
    inner: Arc<ProgramInner>,
}

struct ProgramInner {
    api: Arc<Api>,
    context: Context,
    raw: sys::ProgramHandle,
}

impl Drop for ProgramInner {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by `clCreateProgramWithSource`
        // through this same `api`, and this `Drop` runs exactly once for
        // the single shared `ProgramInner`.
        let code = unsafe { (self.api.release_program)(self.raw) };
        if code != sys::SUCCESS {
            log_warn!("clReleaseProgram failed with {}", sys::error_name(code));
        }
    }
}

// SAFETY: `ProgramInner` is an opaque driver handle plus shared
// function pointers; programs are reference-counted by the driver and
// all entry points used here are thread-safe per the specification.
unsafe impl Send for ProgramInner {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for ProgramInner {}

impl Program {
    /// Creates a program for `context` from one or more source units.
    ///
    /// The driver receives the sources as owned, NUL-terminated copies.
    /// The program must be built with [`Program::build`] before kernels
    /// can be created from it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `sources` is empty, and
    /// [`Error::Status`] / [`Error::NullHandle`] on driver failures.
    pub fn from_sources<'a, I>(context: &Context, sources: I) -> Result<Self>
    where
        I: IntoIterator<Item = &'a str>,
    {
        // Owned copies must outlive the driver call that reads them.
        let owned: Vec<Vec<u8>> = sources
            .into_iter()
            .map(|unit| {
                let mut bytes = unit.as_bytes().to_vec();
                bytes.push(0);
                bytes
            })
            .collect();
        if owned.is_empty() {
            return Err(Error::InvalidArgument {
                what: "at least one source unit is required",
            });
        }
        let pointers: Vec<*const u8> = owned.iter().map(|unit| unit.as_ptr()).collect();
        let api = context.api().clone();
        let mut errcode = sys::SUCCESS;
        // SAFETY: `context` is a live handle of `api`; `pointers` is a
        // valid array of `pointers.len()` NUL-terminated strings that
        // is only read during the call; `errcode` is a valid
        // out-pointer.
        let raw = unsafe {
            (api.create_program_with_source)(
                context.raw(),
                pointers.len() as u32,
                pointers.as_ptr().cast::<*const c_char>(),
                null_mut(),
                &mut errcode,
            )
        };
        let raw = creation(raw, errcode, "clCreateProgramWithSource")?;
        Ok(Self {
            inner: Arc::new(ProgramInner {
                api,
                context: context.clone(),
                raw,
            }),
        })
    }

    /// Creates a program from a single OpenCL C source string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] / [`Error::NullHandle`] on driver
    /// failures.
    pub fn from_source(context: &Context, source: &str) -> Result<Self> {
        Self::from_sources(context, core::iter::once(source))
    }

    /// Builds the program for the context's device.
    ///
    /// `options` is passed to the driver verbatim (for example
    /// `"-cl-std=CL3.0"` or `"-cl-fast-relaxed-math"`); an empty string
    /// requests the defaults.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BuildFailed`] carrying the driver's build log
    /// when compilation fails, and [`Error::Status`] when the driver
    /// rejects the request outright.
    pub fn build(&self, options: &str) -> Result<()> {
        if options.bytes().any(|byte| byte == 0) {
            return Err(Error::InvalidArgument {
                what: "build options must not contain NUL bytes",
            });
        }
        let mut c_options = options.as_bytes().to_vec();
        c_options.push(0);
        let device = self.inner.context.device().raw();
        // SAFETY: `raw`/`device` are live handles of the same `api`;
        // `c_options` is a valid NUL-terminated string that is only read
        // during the call and `None` selects synchronous building.
        let code = unsafe {
            (self.inner.api.build_program)(
                self.inner.raw,
                1,
                &device,
                core::ffi::CStr::from_bytes_with_nul_unchecked(&c_options).as_ptr(),
                None,
                core::ptr::null_mut(),
            )
        };
        if code == sys::SUCCESS {
            return Ok(());
        }
        let log = self.build_log().unwrap_or_default();
        if !log.is_empty() {
            return Err(Error::BuildFailed { log });
        }
        status(code, "clBuildProgram")
    }

    /// Returns the device build log of this program.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the query fails.
    pub fn build_log(&self) -> Result<alloc::string::String> {
        let mut call = |size: usize, value: *mut c_void, size_ret: *mut usize| {
            // SAFETY: `raw` is a live handle of the driver that
            // resolved `api`, and `query` provides valid buffers.
            unsafe {
                (self.inner.api.get_program_build_info)(
                    self.inner.raw,
                    self.inner.context.device().raw(),
                    sys::PROGRAM_BUILD_LOG,
                    size,
                    value,
                    size_ret,
                )
            }
        };
        query::query_string(&mut call, "clGetProgramBuildInfo")
    }

    /// Returns the context the program was created for.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.inner.context
    }

    /// Creates a kernel entry point from this built program.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] / [`Error::NullHandle`] when the
    /// program has no entry point with `name`.
    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        Kernel::new(self, name)
    }

    /// Returns the resolved entry points of the owning driver.
    pub(crate) fn api(&self) -> &Arc<Api> {
        &self.inner.api
    }

    /// Returns the raw driver handle.
    pub(crate) fn raw(&self) -> sys::ProgramHandle {
        self.inner.raw
    }
}

impl fmt::Debug for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Program")
            .field("handle", &self.inner.raw)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn program_is_send_and_sync() {
        _assert_send_sync::<Program>();
    }

    /// Source preparation must NUL-terminate each unit.
    #[test]
    fn source_units_are_nul_terminated() {
        let units: Vec<&str> = vec!["__kernel void k(void) {}", ""];
        let owned: Vec<Vec<u8>> = units
            .iter()
            .map(|unit| {
                let mut bytes = unit.as_bytes().to_vec();
                bytes.push(0);
                bytes
            })
            .collect();
        assert_eq!(owned.len(), 2);
        for unit in &owned {
            assert_eq!(unit.last(), Some(&0));
        }
    }
}
