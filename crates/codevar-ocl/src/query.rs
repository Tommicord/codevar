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

//! Two-call information queries shared by every OpenCL object.
//!
//! `clGetXxxInfo` follows the same protocol everywhere: call once with a
//! null value pointer to learn how many bytes the driver wants to write,
//! then call again with a buffer of that size. This module implements that
//! dance once — for strings ([`query_string`]) and for scalars
//! ([`query_pod`] and its typed wrappers) — so the individual object modules
//! only supply a closure that forwards to their entry point.

use alloc::string::String;
use alloc::vec;
use bsize::BSize;
use core::ffi::c_void;
use core::mem::size_of;
use core::ptr::null_mut;

use crate::error::{Error, Result, status};

/// Upper bound on the size a driver may report for an info query.
///
/// Real values are tiny (the largest are device-extension strings, a few
/// kilobytes); anything beyond this cap is treated as a driver bug instead
/// of being allocated.
const MAX_QUERY_BYTES: usize = BSize::mib(16).bytes();

/// Upper bound on the number of handles a driver may report when
/// enumerating objects (platforms, devices).
///
/// Real machines expose at most a handful of each; a larger count is
/// treated as a driver bug so a corrupt count cannot trigger an enormous
/// allocation.
pub(crate) const MAX_HANDLES: u32 = 4096;

/// Closure shape of every `clGetXxxInfo`-style entry point.
pub(crate) type QueryCall<'a> = dyn FnMut(usize, *mut c_void, *mut usize) -> i32 + 'a;

/// Queries a NUL-terminated (or raw byte) string from a driver.
///
/// The two-call protocol is retried up to three times when the driver
/// reports that the value grew between the calls
/// ([`sys::INVALID_VALUE`]). Trailing NUL bytes are stripped and invalid
/// UTF-8 is replaced lossily, because driver strings are not required to be
/// valid UTF-8 (they are almost always ASCII).
///
/// # Errors
///
/// Returns [`Error::Status`] when the driver rejects a call, and
/// [`Error::InvalidArgument`] when it reports a size beyond
/// [`MAX_QUERY_BYTES`].
pub(crate) fn query_string(call: &mut QueryCall<'_>, context: &'static str) -> Result<String> {
    let mut attempts = 0u8;
    loop {
        let mut size = 0usize;
        status(call(0, null_mut(), &mut size), context)?;
        if size == 0 {
            return Ok(String::new());
        }
        if size > MAX_QUERY_BYTES {
            return Err(Error::InvalidArgument { what: context });
        }
        let mut buffer = vec![0u8; size];
        let code = call(size, buffer.as_mut_ptr().cast(), null_mut());
        if code == crate::sys::SUCCESS {
            while buffer.last() == Some(&0u8) {
                buffer.pop();
            }
            return Ok(String::from_utf8_lossy(&buffer).into_owned());
        }
        if code != crate::sys::INVALID_VALUE || attempts >= 3 {
            return Err(Error::Status { code, context });
        }
        attempts += 1;
    }
}

/// Queries a scalar value of type `T` (one of `u32`, `u64`, `usize`, `i32`).
///
/// The reported size is validated against `size_of::<T>()` first, so a
/// driver writing a differently sized value can never overflow the stack
/// slot used for the result.
///
/// # Errors
///
/// Returns [`Error::Status`] when the driver rejects a call or reports a
/// size that does not fit `T`.
pub(crate) fn query_pod<T: Copy + Default>(call: &mut QueryCall<'_>, context: &'static str) -> Result<T> {
    let mut reported = 0usize;
    status(call(0, null_mut(), &mut reported), context)?;
    if reported == 0 || reported > size_of::<T>() {
        return Err(Error::Status {
            code: crate::sys::INVALID_VALUE,
            context,
        });
    }
    let mut value = T::default();
    status(call(reported, (&mut value as *mut T).cast(), null_mut()), context)?;
    Ok(value)
}

/// Queries a `u32` info value (e.g. `CL_DEVICE_MAX_COMPUTE_UNITS`).
///
/// # Errors
///
/// See [`query_pod`].
#[inline]
pub(crate) fn query_u32(call: &mut QueryCall<'_>, context: &'static str) -> Result<u32> {
    query_pod::<u32>(call, context)
}

/// Queries a `u64` info value (e.g. `CL_DEVICE_GLOBAL_MEM_SIZE`).
///
/// # Errors
///
/// See [`query_pod`].
#[inline]
pub(crate) fn query_u64(call: &mut QueryCall<'_>, context: &'static str) -> Result<u64> {
    query_pod::<u64>(call, context)
}

/// Queries a pointer-sized info value (e.g. `CL_DEVICE_MAX_WORK_GROUP_SIZE`).
///
/// # Errors
///
/// See [`query_pod`].
#[inline]
pub(crate) fn query_usize(call: &mut QueryCall<'_>, context: &'static str) -> Result<usize> {
    query_pod::<usize>(call, context)
}

/// Queries a signed `i32` info value (e.g. `CL_EVENT_COMMAND_EXECUTION_STATUS`).
///
/// # Errors
///
/// See [`query_pod`].
#[inline]
pub(crate) fn query_i32(call: &mut QueryCall<'_>, context: &'static str) -> Result<i32> {
    query_pod::<i32>(call, context)
}

/// Queries a `cl_bool` info value and converts it to a Rust `bool`.
///
/// # Errors
///
/// See [`query_pod`].
#[inline]
pub(crate) fn query_bool(call: &mut QueryCall<'_>, context: &'static str) -> Result<bool> {
    Ok(query_pod::<u32>(call, context)? != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives `query_string` with a scripted driver that hands out
    /// `"hello\0"` in the documented two-call pattern.
    #[test]
    fn query_string_reads_and_trims_a_two_call_value() {
        let value = b"hello\0";
        let mut seen: Vec<usize> = Vec::new();
        let mut call = |size: usize, out: *mut c_void, size_ret: *mut usize| {
            if out.is_null() {
                if !size_ret.is_null() {
                    // SAFETY: the caller passes a valid pointer to a usize.
                    unsafe { size_ret.write(value.len()) };
                }
                return crate::sys::SUCCESS;
            }
            seen.push(size);
            if size < value.len() {
                return crate::sys::INVALID_VALUE;
            }
            // SAFETY: `out` points at a buffer of `size` bytes provided by
            // `query_string`, which is large enough for `value`.
            unsafe { core::ptr::copy_nonoverlapping(value.as_ptr(), out.cast::<u8>(), value.len()) };
            crate::sys::SUCCESS
        };
        let text = query_string(&mut call, "clGetDeviceInfo");
        assert_eq!(text.ok(), Some(String::from("hello")));
        assert_eq!(seen, vec![value.len()]);
    }

    /// A driver whose value grows between the calls must be retried, not
    /// reported as a failure.
    #[test]
    fn query_string_retries_when_the_value_grows() {
        let mut grows = true;
        let mut call = |size: usize, out: *mut c_void, size_ret: *mut usize| {
            if out.is_null() {
                let len = if grows { 3 } else { 6 };
                grows = false;
                if !size_ret.is_null() {
                    // SAFETY: the caller passes a valid pointer to a usize.
                    unsafe { size_ret.write(len) };
                }
                return crate::sys::SUCCESS;
            }
            if size < 6 {
                return crate::sys::INVALID_VALUE;
            }
            let value = b"abcdef";
            // SAFETY: `out` points at a buffer of at least 6 bytes as
            // reported by the first call of this iteration.
            unsafe { core::ptr::copy_nonoverlapping(value.as_ptr(), out.cast::<u8>(), value.len()) };
            crate::sys::SUCCESS
        };
        let text = query_string(&mut call, "clGetPlatformInfo");
        assert_eq!(text.ok(), Some(String::from("abcdef")));
    }

    /// An empty value must produce an empty string without a second call.
    #[test]
    fn query_string_handles_empty_values() {
        let mut call = |size: usize, out: *mut c_void, size_ret: *mut usize| {
            let _ = size;
            if out.is_null() && !size_ret.is_null() {
                // SAFETY: the caller passes a valid pointer to a usize.
                unsafe { size_ret.write(0) };
            }
            crate::sys::SUCCESS
        };
        assert_eq!(
            query_string(&mut call, "clGetDeviceInfo").ok(),
            Some(String::new())
        );
    }

    /// A driver reporting a size that cannot fit the target type must be
    /// rejected instead of overflowing the result slot.
    #[test]
    fn query_pod_rejects_oversized_values() {
        let mut call = |size: usize, out: *mut c_void, size_ret: *mut usize| {
            let _ = (size, out);
            if !size_ret.is_null() {
                // SAFETY: the caller passes a valid pointer to a usize.
                unsafe { size_ret.write(16) };
            }
            crate::sys::SUCCESS
        };
        let result = query_pod::<u32>(&mut call, "clGetDeviceInfo");
        assert!(
            matches!(result, Err(Error::Status { code: -30, .. })),
            "got {result:?}"
        );
    }

    /// Scalars must come back with their driver-reported value.
    #[test]
    fn query_pod_returns_the_scalar_value() {
        let mut call = |size: usize, out: *mut c_void, size_ret: *mut usize| {
            if out.is_null() {
                if !size_ret.is_null() {
                    // SAFETY: the caller passes a valid pointer to a usize.
                    unsafe { size_ret.write(4) };
                }
                return crate::sys::SUCCESS;
            }
            let _ = size;
            // SAFETY: `out` is the caller's `u32` result slot, which is four
            // bytes as reported by the first call.
            unsafe { (out as *mut u32).write(7) };
            crate::sys::SUCCESS
        };
        assert_eq!(query_u32(&mut call, "clGetDeviceInfo").ok(), Some(7));
    }
}
