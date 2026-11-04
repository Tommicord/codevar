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

//! Why an exchange between the two APIs failed.
//!
//! A share crosses two drivers, so a failure says which side answered:
//! [`Error::OpenCl`] and [`Error::Vulkan`] carry the driver's own code,
//! while [`Error::Unsupported`] and [`Error::InvalidArgument`] are decided
//! before either driver is reached. [`Error::DeviceMismatch`] is the one
//! that has no counterpart in a single API: a handle exported by one GPU
//! simply is not importable by another, and no driver can tell the caller
//! which of its own devices was the wrong choice.

use core::fmt;

use ash::vk;
use codevar_ocl::Error as ClError;

/// Result type of every operation in this crate.
pub type Result<T> = core::result::Result<T, Error>;

/// Why a buffer or semaphore was not shared.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The OpenCL driver or wrapper refused the request.
    OpenCl(ClError),
    /// A Vulkan entry point returned an error result.
    Vulkan {
        /// The raw `VkResult`; [`codevar_ocl::sys::error_name`] has the
        /// OpenCL counterpart of the same reporting job.
        code: vk::Result,
        /// The entry point that failed, e.g. `"vkGetMemoryFdKHR"`.
        context: &'static str,
    },
    /// One side cannot do what the other offered.
    Unsupported {
        /// What is missing, naming the extension or capability.
        what: &'static str,
    },
    /// An argument was rejected before it reached either driver.
    InvalidArgument {
        /// What was wrong with the argument.
        what: &'static str,
    },
    /// The two devices are not the same device, so a handle of one cannot
    /// be imported by the other.
    DeviceMismatch {
        /// Which pairing failed.
        what: &'static str,
    },
}

impl From<ClError> for Error {
    fn from(error: ClError) -> Self {
        Self::OpenCl(error)
    }
}

impl Error {
    /// Reads the code one Vulkan call answered with.
    ///
    /// Only `VK_SUCCESS` is a success: the other positive codes belong to
    /// the `fenceGetStatus`-style queries, and none of the entry points here
    /// is allowed to return one.
    pub(crate) fn check(code: vk::Result, context: &'static str) -> Result<()> {
        if code == vk::Result::SUCCESS {
            Ok(())
        } else {
            Err(Self::Vulkan { code, context })
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenCl(error) => write!(f, "opencl: {error}"),
            Self::Vulkan { code, context } => {
                write!(f, "{context} failed with {code} ({})", code.as_raw())
            }
            Self::Unsupported { what } => write!(f, "unsupported: {what}"),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::DeviceMismatch { what } => write!(f, "device mismatch: {what}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_ocl::sys;

    /// A Vulkan code has to be readable as both a name and a number, since
    /// a driver error is only ever reported by its integer value.
    #[test]
    fn a_vulkan_failure_names_the_entry_point_and_code() {
        let error = Error::Vulkan {
            code: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            context: "vkGetMemoryFdKHR",
        };
        let text = format!("{error}");
        assert!(
            text.starts_with("vkGetMemoryFdKHR failed with "),
            "the entry point must be named: {text}"
        );
        assert!(text.contains("(-1)"), "the raw code must stay readable: {text}");
    }

    /// The OpenCL side keeps its own codes — losing them would hide which
    /// `cl_*` call answered.
    #[test]
    fn an_opencl_error_keeps_its_status_code() {
        let error = Error::from(ClError::Status {
            code: sys::INVALID_OPERATION,
            context: "clEnqueueAcquireExternalMemObjectsKHR",
        });
        assert!(matches!(error, Error::OpenCl(ClError::Status { .. })));
        assert!(
            format!("{error}").contains("CL_INVALID_OPERATION"),
            "the opencl code must stay readable: {error}"
        );
    }

    /// A positive code other than success is an error here: `VK_NOT_READY`
    /// would otherwise read as a share that worked.
    #[test]
    fn only_success_is_a_success() {
        assert_eq!(Error::check(vk::Result::SUCCESS, "vkGetMemoryFdKHR"), Ok(()));
        for code in [vk::Result::NOT_READY, vk::Result::INCOMPLETE] {
            assert_eq!(
                Error::check(code, "vkGetMemoryFdKHR"),
                Err(Error::Vulkan {
                    code,
                    context: "vkGetMemoryFdKHR"
                })
            );
        }
    }
}
