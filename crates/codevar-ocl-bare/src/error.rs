//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Error type of the bare-metal OpenCL driver.
//!
//! Every failure inside the driver is an [`Error`]; the [`crate::api`]
//! layer maps it to the matching `CL_*` status code at the C boundary.

use core::fmt;

/// Errors this driver can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The platform was not initialized with [`crate::init`].
    NotInitialized,
    /// A handle failed its magic-tag validation.
    InvalidHandle,
    /// The platform handle is invalid.
    InvalidPlatform,
    /// The device handle is invalid.
    InvalidDevice,
    /// The context handle is invalid.
    InvalidContext,
    /// The command-queue handle is invalid.
    InvalidQueue,
    /// The program handle is invalid.
    InvalidProgram,
    /// The kernel handle is invalid.
    InvalidKernel,
    /// The memory-object handle is invalid.
    InvalidMemObject,
    /// The buffer size was zero.
    InvalidBufferSize,
    /// A host pointer was required by the flags but missing (or forbidden
    /// but supplied).
    InvalidHostPtr,
    /// An argument was outside the range the entry point accepts.
    InvalidArgument,
    /// A `clSetKernelArg` index is past the kernel's declared arguments.
    InvalidArgIndex,
    /// A `clSetKernelArg` size does not match the declared argument.
    InvalidArgSize,
    /// A `clSetKernelArg` value was null (local-memory arguments are not
    /// supported by this driver).
    InvalidArgValue,
    /// A work launch was malformed (sizes, divisibility, dimensionality).
    InvalidLaunch,
    /// The requested device is not present or not of the requested type.
    DeviceNotFound,
    /// The kernel name is not in the registry.
    KernelNotFound,
    /// The kernel cannot execute on the queue's device (an accelerator
    /// opcode has no handler, for example).
    UnsupportedOnDevice,
    /// A required kernel argument has not been set.
    MissingKernelArg,
    /// The kernel takes more arguments than the device can marshal.
    TooManyArgs,
    /// An allocation failed.
    OutOfMemory,
    /// The device ran out of resources executing a command.
    OutOfResources,
    /// The operation is not supported (e.g. online compilation).
    Unsupported,
    /// A wait-list event completed with an error.
    EventFailed,
    /// An event/list handle combination was inconsistent.
    InvalidEvent,
    /// An event wait list was null or contained an invalid handle.
    InvalidWaitList,
    /// A SPIR-V program binary is malformed or uses a construct outside
    /// this driver's embedded subset.
    InvalidBinary,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::NotInitialized => "OpenCL platform is not initialized",
            Self::InvalidHandle => "object handle failed validation",
            Self::InvalidPlatform => "platform handle is invalid",
            Self::InvalidDevice => "device handle is invalid",
            Self::InvalidContext => "context handle is invalid",
            Self::InvalidQueue => "command-queue handle is invalid",
            Self::InvalidProgram => "program handle is invalid",
            Self::InvalidKernel => "kernel handle is invalid",
            Self::InvalidMemObject => "memory-object handle is invalid",
            Self::InvalidBufferSize => "buffer size is zero",
            Self::InvalidHostPtr => "host pointer does not match the flags",
            Self::InvalidArgument => "an argument was out of range",
            Self::InvalidArgIndex => "kernel argument index is out of range",
            Self::InvalidArgSize => "kernel argument size does not match the declaration",
            Self::InvalidArgValue => "kernel argument value is invalid",
            Self::InvalidLaunch => "work launch is malformed",
            Self::DeviceNotFound => "device was not found",
            Self::KernelNotFound => "kernel name is not registered",
            Self::UnsupportedOnDevice => "kernel cannot run on this device",
            Self::MissingKernelArg => "a required kernel argument is missing",
            Self::TooManyArgs => "kernel exceeds the device argument limit",
            Self::OutOfMemory => "an allocation failed",
            Self::OutOfResources => "device ran out of resources",
            Self::Unsupported => "operation is not supported",
            Self::EventFailed => "a waited event completed with an error",
            Self::InvalidEvent => "an event handle or wait list was invalid",
            Self::InvalidWaitList => "event wait list is invalid",
            Self::InvalidBinary => "SPIR-V program binary is invalid",
        };
        f.write_str(text)
    }
}

/// Result alias used throughout the driver.
pub type Result<T> = core::result::Result<T, Error>;

impl Error {
    /// Maps this error to the OpenCL `CL_*` status code it represents.
    #[must_use]
    pub const fn to_cl(self) -> i32 {
        match self {
            Self::NotInitialized => crate::sys::PLATFORM_NOT_FOUND_KHR,
            Self::InvalidHandle => crate::sys::INVALID_OPERATION,
            Self::InvalidPlatform => crate::sys::INVALID_PLATFORM,
            Self::InvalidDevice => crate::sys::INVALID_DEVICE,
            Self::InvalidContext => crate::sys::INVALID_CONTEXT,
            Self::InvalidQueue => crate::sys::INVALID_COMMAND_QUEUE,
            Self::InvalidProgram => crate::sys::INVALID_PROGRAM,
            Self::InvalidKernel => crate::sys::INVALID_KERNEL,
            Self::InvalidMemObject => crate::sys::INVALID_MEM_OBJECT,
            Self::InvalidBufferSize => crate::sys::INVALID_BUFFER_SIZE,
            Self::InvalidHostPtr => crate::sys::INVALID_HOST_PTR,
            Self::InvalidArgument => crate::sys::INVALID_VALUE,
            Self::InvalidArgIndex => crate::sys::INVALID_ARG_INDEX,
            Self::InvalidArgSize => crate::sys::INVALID_ARG_SIZE,
            Self::InvalidArgValue => crate::sys::INVALID_ARG_VALUE,
            Self::InvalidLaunch => crate::sys::INVALID_WORK_GROUP_SIZE,
            Self::DeviceNotFound => crate::sys::DEVICE_NOT_FOUND,
            Self::KernelNotFound => crate::sys::INVALID_KERNEL_NAME,
            Self::UnsupportedOnDevice => crate::sys::INVALID_KERNEL,
            Self::MissingKernelArg => crate::sys::INVALID_KERNEL_ARGS,
            Self::TooManyArgs => crate::sys::INVALID_KERNEL_ARGS,
            Self::OutOfMemory => crate::sys::OUT_OF_HOST_MEMORY,
            Self::OutOfResources => crate::sys::OUT_OF_RESOURCES,
            Self::Unsupported => crate::sys::COMPILER_NOT_AVAILABLE,
            Self::EventFailed => crate::sys::EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
            Self::InvalidEvent => crate::sys::INVALID_EVENT,
            Self::InvalidWaitList => crate::sys::INVALID_EVENT_WAIT_LIST,
            Self::InvalidBinary => crate::sys::INVALID_BINARY,
        }
    }
}
