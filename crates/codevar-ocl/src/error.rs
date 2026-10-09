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

//! Error type of the OpenCL framework.
//!
//! Every fallible operation in this crate returns [`Result<T>`], which is a
//! thin alias over `core::result::Result<T, Error>`. The type deliberately
//! mixes loading-time failures (no library, missing entry points) with
//! driver-reported status codes so that callers only ever handle one error
//! family.
//!
//! # Examples
//!
//! ```
//! use codevar_ocl::{Error, sys};
//!
//! let error = Error::Status {
//!     code: sys::INVALID_KERNEL_ARGS,
//!     context: "clSetKernelArg",
//! };
//! assert_eq!(
//!     error.to_string(),
//!     "clSetKernelArg failed with CL_INVALID_KERNEL_ARGS (-52)"
//! );
//! ```

use alloc::string::String;
use core::fmt;

use crate::sys;

/// Result type used by every fallible operation in this crate.
pub type Result<T> = core::result::Result<T, Error>;

/// Why an OpenCL operation failed.
///
/// Variants fall into two groups: failures that happen before any driver code
/// runs ([`LibraryOpen`](Self::LibraryOpen),
/// [`MissingSymbol`](Self::MissingSymbol), [`NoLoader`](Self::NoLoader), …)
/// and status codes reported by the driver itself
/// ([`Status`](Self::Status), [`BuildFailed`](Self::BuildFailed), …).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A candidate library could not be opened by the dynamic loader.
    LibraryOpen {
        /// The library name or path that was tried.
        library: String,
        /// The platform loader's diagnostic message.
        message: String,
    },
    /// The library opened but does not export a required entry point.
    MissingSymbol {
        /// The exact C symbol that could not be resolved.
        symbol: &'static str,
    },
    /// The library opened but exposes no usable OpenCL entry points at all.
    NotOpenclLibrary {
        /// The library name or path that was tried.
        library: String,
    },
    /// Neither an ICD loader nor a vendor driver library could be loaded.
    NoLoader,
    /// The vendor-driver scan found no usable implementation.
    NoVendorLibraries,
    /// Driver enumeration succeeded but reported zero platforms.
    NoPlatforms,
    /// The driver returned a non-success status code.
    Status {
        /// The raw `cl_int` status code; see [`sys::error_name`].
        code: i32,
        /// Short description of the entry point that failed, e.g.
        /// `"clCreateBuffer"`.
        context: &'static str,
    },
    /// An optional entry point is not available in this implementation.
    Unsupported {
        /// The entry point the caller attempted to use.
        symbol: &'static str,
    },
    /// `clBuildProgram` failed and the per-device build log is attached.
    BuildFailed {
        /// The concatenation of the build logs of every built device.
        log: String,
    },
    /// An argument was rejected before it reached the driver.
    InvalidArgument {
        /// What was wrong with the argument.
        what: &'static str,
    },
    /// Two objects that must share a context were mixed.
    ContextMismatch {
        /// Which object disagreed with the queue or program.
        what: &'static str,
    },
    /// The driver reported success but returned a null handle.
    NullHandle {
        /// The `clCreate*` entry point that returned the null handle.
        context: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryOpen { library, message } => {
                write!(f, "failed to open the OpenCL library {library}: {message}")
            }
            Self::MissingSymbol { symbol } => {
                write!(f, "the OpenCL implementation does not export {symbol}")
            }
            Self::NotOpenclLibrary { library } => {
                write!(
                    f,
                    "the library {library} does not expose usable OpenCL entry points"
                )
            }
            Self::NoLoader => f.write_str("no OpenCL loader or vendor driver library could be loaded"),
            Self::NoVendorLibraries => f.write_str("no usable OpenCL vendor driver library was found"),
            Self::NoPlatforms => f.write_str("no OpenCL platform is available"),
            Self::Status { code, context } => {
                write!(f, "{context} failed with {} ({code})", sys::error_name(*code))
            }
            Self::Unsupported { symbol } => {
                write!(f, "this OpenCL driver does not support {symbol}")
            }
            Self::BuildFailed { log } => {
                write!(f, "the OpenCL program failed to build:\n{log}")
            }
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::ContextMismatch { what } => {
                write!(f, "objects from different OpenCL contexts were mixed: {what}")
            }
            Self::NullHandle { context } => {
                write!(f, "{context} reported success but returned a null handle")
            }
        }
    }
}

impl core::error::Error for Error {}

/// Converts an OpenCL status code into a [`Result`].
///
/// `sys::SUCCESS` maps to `Ok(())`; every other code becomes
/// [`Error::Status`] carrying the caller-provided `context` label.
#[inline]
pub(crate) fn status(code: i32, context: &'static str) -> Result<()> {
    if code == sys::SUCCESS {
        Ok(())
    } else {
        Err(Error::Status { code, context })
    }
}

/// Validates the `errcode_ret` out-parameter of a `clCreate*` entry point.
///
/// On failure the handle is assumed to be null, as required by the OpenCL
/// specification; a non-null handle accompanying an error status is leaked
/// rather than released, because releasing it would require knowing which
/// `clRelease*` function matches the (unknown-to-this-generic-function)
/// handle type.
///
/// # Errors
///
/// Returns [`Error::Status`] when the driver reported a failure code and
/// [`Error::NullHandle`] when the driver reported success but returned a
/// null handle.
#[inline]
pub(crate) fn creation<H: sys::Nullable>(handle: H, errcode: i32, context: &'static str) -> Result<H> {
    status(errcode, context)?;
    if handle.is_null() {
        return Err(Error::NullHandle { context });
    }
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys::ContextHandle;

    #[test]
    fn status_maps_success_and_failures() {
        assert!(status(sys::SUCCESS, "clFlush").is_ok());
        let error = status(sys::OUT_OF_RESOURCES, "clFinish").err();
        assert!(
            matches!(
                error,
                Some(Error::Status { code, context })
                    if code == sys::OUT_OF_RESOURCES && context == "clFinish"
            ),
            "expected Error::Status, got {error:?}"
        );
    }

    #[test]
    fn creation_accepts_success_with_a_real_handle() {
        let raw = ContextHandle::from_raw(0xdead_beef as *mut _);
        assert_eq!(creation(raw, sys::SUCCESS, "clCreateContext").ok(), Some(raw));
    }

    #[test]
    fn creation_rejects_error_codes_and_null_handles() {
        let null = ContextHandle::from_raw(core::ptr::null_mut());
        assert!(matches!(
            creation(null, sys::INVALID_DEVICE, "clCreateContext"),
            Err(Error::Status { code: -33, .. })
        ));
        assert!(matches!(
            creation(null, sys::SUCCESS, "clCreateContext"),
            Err(Error::NullHandle {
                context: "clCreateContext"
            })
        ));
    }

    #[test]
    fn display_messages_are_actionable() {
        let error = Error::Status {
            code: sys::INVALID_KERNEL_ARGS,
            context: "clSetKernelArg",
        };
        assert_eq!(
            error.to_string(),
            "clSetKernelArg failed with CL_INVALID_KERNEL_ARGS (-52)"
        );

        let error = Error::MissingSymbol {
            symbol: "clGetPlatformIDs",
        };
        assert_eq!(
            error.to_string(),
            "the OpenCL implementation does not export clGetPlatformIDs"
        );

        let error = Error::BuildFailed {
            log: String::from("kernel.cl:3:1: error: use of undeclared identifier 'x'"),
        };
        assert!(
            error
                .to_string()
                .contains("error: use of undeclared identifier")
        );
    }
}
