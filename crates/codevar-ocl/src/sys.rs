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

//! Raw OpenCL C API types, constants and entry-point signatures.
//!
//! This module is a hand-written, `no_std` translation of the subset of the
//! Khronos `CL/cl_platform.h`, `CL/cl.h` and `CL/cl_ext.h` headers that the
//! safe layer in this crate uses. Nothing here links against `libOpenCL` at
//! build time: the entry points are function-pointer types that are resolved
//! at runtime from a vendor ICD driver or the Khronos ICD loader (see
//! [`crate::runtime`]).
//!
//! # Naming
//!
//! Rust is not C, so the C conventions are dropped instead of mirrored:
//!
//! * The redundant scalar aliases (`cl_int`, `cl_uint`, `cl_bool`, …) are
//!   removed in favour of the native Rust types they alias (`i32`, `u32`,
//!   …).
//! * Constants drop the `CL_` prefix, exactly as `ash` drops `VK_`:
//!   [`SUCCESS`] is Khronos `CL_SUCCESS`, [`DEVICE_TYPE_GPU`] is
//!   `CL_DEVICE_TYPE_GPU`, and so on. The strings reported by
//!   [`error_name`] keep the full `CL_` names because those are what
//!   developers see in driver logs and Khronos documentation.
//! * Entry-point types drop the `cl` prefix: [`GetPlatformIds`] is the type
//!   of the C symbol `clGetPlatformIDs`.
//! * Opaque handles are `#[repr(transparent)]` newtypes over
//!   `*mut c_void`, so a device handle cannot be passed where a context
//!   handle is expected.
//!
//! # Reference
//!
//! * <https://github.com/KhronosGroup/OpenCL-Headers> — `CL/cl.h`,
//!   `CL/cl_platform.h`, `CL/cl_ext.h`.
//! * <https://github.com/KhronosGroup/OpenCL-ICD-Loader> — the canonical
//!   client loader this crate interoperates with.
//!
//! # Safety
//!
//! Every type in this module is either a plain integer or an opaque handle
//! that must only be produced and consumed by the driver that created it.
//! Invoking the function-pointer types is `unsafe`; the safe layer in
//! [`crate::api`] is the only intended caller.
//!
//! # Examples
//!
//! ```
//! use codevar_ocl::sys;
//!
//! assert_eq!(sys::SUCCESS, 0);
//! assert_eq!(sys::DEVICE_TYPE_GPU, 1 << 2);
//! assert_eq!(sys::error_name(sys::INVALID_KERNEL_ARGS), "CL_INVALID_KERNEL_ARGS");
//! ```

use core::ffi::{c_char, c_void};

/// Implemented by the opaque handle newtypes of this module.
///
/// The safe layer uses this trait to detect the null handle that
/// `clCreate*` entry points return on failure without downcasting to
/// concrete handle types at every call site.
pub trait Nullable {
    /// Returns `true` when the handle is null.
    #[must_use]
    fn is_null(&self) -> bool;
}

macro_rules! handle {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        ///
        /// The newtype is `#[repr(transparent)]` over `*mut c_void`, so it has
        /// the same layout and FFI ABI as the C pointer type while remaining
        /// distinct from every other handle type.
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(*mut c_void);

        impl $name {
            /// Wraps a raw pointer returned by the driver.
            ///
            /// The value is only meaningful as an argument to the driver that
            /// produced it.
            #[must_use]
            pub const fn from_raw(raw: *mut c_void) -> Self {
                Self(raw)
            }

            /// Returns the raw pointer for use in an FFI call.
            #[must_use]
            pub const fn as_raw(self) -> *mut c_void {
                self.0
            }

            /// Returns `true` when the driver returned a null handle.
            #[must_use]
            pub const fn is_null(self) -> bool {
                self.0.is_null()
            }
        }

        impl Nullable for $name {
            #[inline]
            fn is_null(&self) -> bool {
                self.0.is_null()
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_tuple(stringify!($name)).field(&self.0).finish()
            }
        }
    };
}

handle!(
    /// Opaque handle to a platform (`cl_platform_id`).
    ///
    /// Platforms are not reference-counted by OpenCL 1.x; the handle stays
    /// valid for as long as the driver library that exposed it is loaded.
    PlatformHandle
);
handle!(
    /// Opaque handle to a device (`cl_device_id`).
    DeviceHandle
);
handle!(
    /// Opaque, reference-counted handle to a context (`cl_context`).
    ContextHandle
);
handle!(
    /// Opaque, reference-counted handle to a command queue
    /// (`cl_command_queue`).
    CommandQueueHandle
);
handle!(
    /// Opaque, reference-counted handle to a memory object (`cl_mem`).
    MemHandle
);
handle!(
    /// Opaque, reference-counted handle to a program (`cl_program`).
    ProgramHandle
);
handle!(
    /// Opaque, reference-counted handle to a kernel (`cl_kernel`).
    KernelHandle
);
handle!(
    /// Opaque, reference-counted handle to an event (`cl_event`).
    EventHandle
);

/// Success status returned by every OpenCL entry point (`CL_SUCCESS`).
pub const SUCCESS: i32 = 0;
/// `true` value for `cl_bool` queries and blocking flags (`CL_TRUE`).
pub const TRUE: u32 = 1;
/// `false` value for `cl_bool` queries and blocking flags (`CL_FALSE`).
pub const FALSE: u32 = 0;
/// Blocking form of an enqueue call (`CL_BLOCKING`, alias of [`TRUE`]).
pub const BLOCKING: u32 = TRUE;
/// Non-blocking form of an enqueue call (`CL_NON_BLOCKING`, alias of [`FALSE`]).
pub const NON_BLOCKING: u32 = FALSE;

/// The named device is not present (`CL_DEVICE_NOT_FOUND`).
pub const DEVICE_NOT_FOUND: i32 = -1;
/// The device is present but not currently available (`CL_DEVICE_NOT_AVAILABLE`).
pub const DEVICE_NOT_AVAILABLE: i32 = -2;
/// No online compiler is available for the device (`CL_COMPILER_NOT_AVAILABLE`).
pub const COMPILER_NOT_AVAILABLE: i32 = -3;
/// Allocation of device memory failed (`CL_MEM_OBJECT_ALLOCATION_FAILURE`).
pub const MEM_OBJECT_ALLOCATION_FAILURE: i32 = -4;
/// The device ran out of resources (`CL_OUT_OF_RESOURCES`).
pub const OUT_OF_RESOURCES: i32 = -5;
/// The host ran out of memory (`CL_OUT_OF_HOST_MEMORY`).
pub const OUT_OF_HOST_MEMORY: i32 = -6;
/// Profiling information is not available for the event
/// (`CL_PROFILING_INFO_NOT_AVAILABLE`).
pub const PROFILING_INFO_NOT_AVAILABLE: i32 = -7;
/// Source and destination of a copy overlap (`CL_MEM_COPY_OVERLAP`).
pub const MEM_COPY_OVERLAP: i32 = -8;
/// The image format does not match the descriptor (`CL_IMAGE_FORMAT_MISMATCH`).
pub const IMAGE_FORMAT_MISMATCH: i32 = -9;
/// The requested image format is not supported (`CL_IMAGE_FORMAT_NOT_SUPPORTED`).
pub const IMAGE_FORMAT_NOT_SUPPORTED: i32 = -10;
/// [`BuildProgram`] failed; the build log describes why
/// (`CL_BUILD_PROGRAM_FAILURE`).
pub const BUILD_PROGRAM_FAILURE: i32 = -11;
/// A memory map operation failed (`CL_MAP_FAILURE`).
pub const MAP_FAILURE: i32 = -12;
/// A sub-buffer offset is not properly aligned (`CL_MISALIGNED_SUB_BUFFER_OFFSET`).
pub const MISALIGNED_SUB_BUFFER_OFFSET: i32 = -13;
/// An event in the wait list finished with an error status
/// (`CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`).
pub const EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: i32 = -14;
/// The program failed to compile (`CL_COMPILE_PROGRAM_FAILURE`).
pub const COMPILE_PROGRAM_FAILURE: i32 = -15;
/// No linker is available (`CL_LINKER_NOT_AVAILABLE`).
pub const LINKER_NOT_AVAILABLE: i32 = -16;
/// The program failed to link (`CL_LINK_PROGRAM_FAILURE`).
pub const LINK_PROGRAM_FAILURE: i32 = -17;
/// Device partitioning failed (`CL_DEVICE_PARTITION_FAILED`).
pub const DEVICE_PARTITION_FAILED: i32 = -18;
/// Kernel argument info is not available (`CL_KERNEL_ARG_INFO_NOT_AVAILABLE`).
pub const KERNEL_ARG_INFO_NOT_AVAILABLE: i32 = -19;

/// An argument had an invalid value (`CL_INVALID_VALUE`).
pub const INVALID_VALUE: i32 = -30;
/// The device type is invalid (`CL_INVALID_DEVICE_TYPE`).
pub const INVALID_DEVICE_TYPE: i32 = -31;
/// The platform handle is invalid (`CL_INVALID_PLATFORM`).
pub const INVALID_PLATFORM: i32 = -32;
/// The device handle is invalid (`CL_INVALID_DEVICE`).
pub const INVALID_DEVICE: i32 = -33;
/// The context handle is invalid (`CL_INVALID_CONTEXT`).
pub const INVALID_CONTEXT: i32 = -34;
/// The queue properties are invalid (`CL_INVALID_QUEUE_PROPERTIES`).
pub const INVALID_QUEUE_PROPERTIES: i32 = -35;
/// The command-queue handle is invalid (`CL_INVALID_COMMAND_QUEUE`).
pub const INVALID_COMMAND_QUEUE: i32 = -36;
/// The host pointer is invalid for the given flags (`CL_INVALID_HOST_PTR`).
pub const INVALID_HOST_PTR: i32 = -37;
/// The sampler handle is invalid (`CL_INVALID_SAMPLER`).
pub const INVALID_SAMPLER: i32 = -41;
/// The program binary is invalid (`CL_INVALID_BINARY`).
pub const INVALID_BINARY: i32 = -42;
/// The build options string is invalid (`CL_INVALID_BUILD_OPTIONS`).
pub const INVALID_BUILD_OPTIONS: i32 = -43;
/// The program handle is invalid (`CL_INVALID_PROGRAM`).
pub const INVALID_PROGRAM: i32 = -44;
/// There is no successfully built executable for the program
/// (`CL_INVALID_PROGRAM_EXECUTABLE`).
pub const INVALID_PROGRAM_EXECUTABLE: i32 = -45;
/// The kernel name is invalid (`CL_INVALID_KERNEL_NAME`).
pub const INVALID_KERNEL_NAME: i32 = -46;
/// The kernel definition is invalid for the device (`CL_INVALID_KERNEL_DEFINITION`).
pub const INVALID_KERNEL_DEFINITION: i32 = -47;
/// The kernel handle is invalid (`CL_INVALID_KERNEL`).
pub const INVALID_KERNEL: i32 = -48;
/// The kernel argument index is out of range (`CL_INVALID_ARG_INDEX`).
pub const INVALID_ARG_INDEX: i32 = -49;
/// The kernel argument value is invalid (`CL_INVALID_ARG_VALUE`).
pub const INVALID_ARG_VALUE: i32 = -50;
/// The kernel argument size is invalid (`CL_INVALID_ARG_SIZE`).
pub const INVALID_ARG_SIZE: i32 = -51;
/// Required kernel arguments have not been set (`CL_INVALID_KERNEL_ARGS`).
pub const INVALID_KERNEL_ARGS: i32 = -52;
/// The work dimension is not in `1..=3` (`CL_INVALID_WORK_DIMENSION`).
pub const INVALID_WORK_DIMENSION: i32 = -53;
/// The work-group size is invalid (`CL_INVALID_WORK_GROUP_SIZE`).
pub const INVALID_WORK_GROUP_SIZE: i32 = -54;
/// A work-item size is invalid (`CL_INVALID_WORK_ITEM_SIZE`).
pub const INVALID_WORK_ITEM_SIZE: i32 = -55;
/// The global work offset is invalid (`CL_INVALID_GLOBAL_OFFSET`).
pub const INVALID_GLOBAL_OFFSET: i32 = -56;
/// The event wait list is invalid (`CL_INVALID_EVENT_WAIT_LIST`).
pub const INVALID_EVENT_WAIT_LIST: i32 = -57;
/// The event handle is invalid (`CL_INVALID_EVENT`).
pub const INVALID_EVENT: i32 = -58;
/// The operation is not valid for the given objects (`CL_INVALID_OPERATION`).
pub const INVALID_OPERATION: i32 = -59;
/// The OpenGL object is invalid (`CL_INVALID_GL_OBJECT`).
pub const INVALID_GL_OBJECT: i32 = -60;
/// The requested buffer size is invalid (`CL_INVALID_BUFFER_SIZE`).
pub const INVALID_BUFFER_SIZE: i32 = -61;
/// The mip level is invalid (`CL_INVALID_MIP_LEVEL`).
pub const INVALID_MIP_LEVEL: i32 = -62;
/// The global work size is invalid (`CL_INVALID_GLOBAL_WORK_SIZE`).
pub const INVALID_GLOBAL_WORK_SIZE: i32 = -63;
/// A property in a property list is invalid (`CL_INVALID_PROPERTY`).
pub const INVALID_PROPERTY: i32 = -64;
/// The image descriptor is invalid (`CL_INVALID_IMAGE_DESCRIPTOR`).
pub const INVALID_IMAGE_DESCRIPTOR: i32 = -65;
/// The compiler options are invalid (`CL_INVALID_COMPILER_OPTIONS`).
pub const INVALID_COMPILER_OPTIONS: i32 = -66;
/// The linker options are invalid (`CL_INVALID_LINKER_OPTIONS`).
pub const INVALID_LINKER_OPTIONS: i32 = -67;
/// The device partition count is invalid (`CL_INVALID_DEVICE_PARTITION_COUNT`).
pub const INVALID_DEVICE_PARTITION_COUNT: i32 = -68;
/// The requested pipe size is invalid (`CL_INVALID_PIPE_SIZE`).
pub const INVALID_PIPE_SIZE: i32 = -69;
/// The device queue is invalid (`CL_INVALID_DEVICE_QUEUE`).
pub const INVALID_DEVICE_QUEUE: i32 = -70;
/// No platform was found; returned by ICD loaders when no vendor ICD is
/// registered (`CL_PLATFORM_NOT_FOUND_KHR`).
pub const PLATFORM_NOT_FOUND_KHR: i32 = -1001;

/// Use every device on the platform (`CL_DEVICE_TYPE_ALL`).
pub const DEVICE_TYPE_ALL: u64 = 0xFFFF_FFFF;
/// The default device of the platform (`CL_DEVICE_TYPE_DEFAULT`).
pub const DEVICE_TYPE_DEFAULT: u64 = 1 << 0;
/// A CPU device (`CL_DEVICE_TYPE_CPU`).
pub const DEVICE_TYPE_CPU: u64 = 1 << 1;
/// A GPU device (`CL_DEVICE_TYPE_GPU`).
pub const DEVICE_TYPE_GPU: u64 = 1 << 2;
/// A dedicated accelerator device (`CL_DEVICE_TYPE_ACCELERATOR`).
pub const DEVICE_TYPE_ACCELERATOR: u64 = 1 << 3;
/// A custom, vendor-defined device (`CL_DEVICE_TYPE_CUSTOM`).
pub const DEVICE_TYPE_CUSTOM: u64 = 1 << 4;

/// The buffer may be read and written by the kernel and the host
/// (`CL_MEM_READ_WRITE`).
pub const MEM_READ_WRITE: u64 = 1 << 0;
/// The kernel may only write the buffer (`CL_MEM_WRITE_ONLY`).
pub const MEM_WRITE_ONLY: u64 = 1 << 1;
/// The kernel may only read the buffer (`CL_MEM_READ_ONLY`).
pub const MEM_READ_ONLY: u64 = 1 << 2;
/// Use the caller-provided host pointer as backing storage
/// (`CL_MEM_USE_HOST_PTR`).
pub const MEM_USE_HOST_PTR: u64 = 1 << 3;
/// Let the implementation allocate host-visible storage (`CL_MEM_ALLOC_HOST_PTR`).
pub const MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
/// Copy the caller-provided host pointer into the buffer at creation
/// (`CL_MEM_COPY_HOST_PTR`).
pub const MEM_COPY_HOST_PTR: u64 = 1 << 5;
/// The host may not write the buffer (`CL_MEM_HOST_WRITE_ONLY`).
pub const MEM_HOST_WRITE_ONLY: u64 = 1 << 7;
/// The host may not read the buffer (`CL_MEM_HOST_READ_ONLY`).
pub const MEM_HOST_READ_ONLY: u64 = 1 << 8;
/// The host may neither read nor write the buffer (`CL_MEM_HOST_NO_ACCESS`).
pub const MEM_HOST_NO_ACCESS: u64 = 1 << 9;

/// Commands complete out of order
/// (`CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`).
pub const QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE: u64 = 1 << 0;
/// The queue collects profiling timestamps for its events
/// (`CL_QUEUE_PROFILING_ENABLE`).
pub const QUEUE_PROFILING_ENABLE: u64 = 1 << 1;
/// Map for reading (`CL_MAP_READ`).
pub const MAP_READ: u64 = 1 << 0;
/// Map for writing (`CL_MAP_WRITE`).
pub const MAP_WRITE: u64 = 1 << 1;
/// Map for writing without preserving the previous contents
/// (`CL_MAP_WRITE_INVALIDATE_REGION`).
pub const MAP_WRITE_INVALIDATE_REGION: u64 = 1 << 2;

/// Full profile (`FULL_PROFILE` or `EMBEDDED_PROFILE`)
/// (`CL_PLATFORM_PROFILE`).
pub const PLATFORM_PROFILE: u32 = 0x0900;
/// OpenCL version string (`CL_PLATFORM_VERSION`).
pub const PLATFORM_VERSION: u32 = 0x0901;
/// Platform name (`CL_PLATFORM_NAME`).
pub const PLATFORM_NAME: u32 = 0x0902;
/// Platform vendor name (`CL_PLATFORM_VENDOR`).
pub const PLATFORM_VENDOR: u32 = 0x0903;
/// Space-separated list of platform extensions (`CL_PLATFORM_EXTENSIONS`).
pub const PLATFORM_EXTENSIONS: u32 = 0x0904;
/// Vendor-specific ICD suffix, `cl_khr_icd` (`CL_PLATFORM_ICD_SUFFIX_KHR`).
pub const PLATFORM_ICD_SUFFIX_KHR: u32 = 0x0920;

/// Device type bitfield (`CL_DEVICE_TYPE`).
pub const DEVICE_TYPE: u32 = 0x1000;
/// Maximum number of compute units (`CL_DEVICE_MAX_COMPUTE_UNITS`).
pub const DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
/// Dimensionality of the work-item grid (`CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS`).
pub const DEVICE_MAX_WORK_ITEM_DIMENSIONS: u32 = 0x1003;
/// Maximum work-group size (`CL_DEVICE_MAX_WORK_GROUP_SIZE`).
pub const DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;
/// Maximum work-item sizes per dimension (`CL_DEVICE_MAX_WORK_ITEM_SIZES`).
pub const DEVICE_MAX_WORK_ITEM_SIZES: u32 = 0x1005;
/// Address width in bits (`CL_DEVICE_ADDRESS_BITS`).
pub const DEVICE_ADDRESS_BITS: u32 = 0x100D;
/// Maximum single allocation size in bytes (`CL_DEVICE_MAX_MEM_ALLOC_SIZE`).
pub const DEVICE_MAX_MEM_ALLOC_SIZE: u32 = 0x1010;
/// Whether images are supported (`CL_DEVICE_IMAGE_SUPPORT`).
pub const DEVICE_IMAGE_SUPPORT: u32 = 0x1016;
/// Total global memory in bytes (`CL_DEVICE_GLOBAL_MEM_SIZE`).
pub const DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
/// Local (on-chip) memory in bytes (`CL_DEVICE_LOCAL_MEM_SIZE`).
pub const DEVICE_LOCAL_MEM_SIZE: u32 = 0x1023;
/// Whether the device is currently available (`CL_DEVICE_AVAILABLE`).
pub const DEVICE_AVAILABLE: u32 = 0x1027;
/// Whether an online compiler is available (`CL_DEVICE_COMPILER_AVAILABLE`).
pub const DEVICE_COMPILER_AVAILABLE: u32 = 0x1028;
/// Device name (`CL_DEVICE_NAME`).
pub const DEVICE_NAME: u32 = 0x102B;
/// Device vendor name (`CL_DEVICE_VENDOR`).
pub const DEVICE_VENDOR: u32 = 0x102C;
/// Driver version string (`CL_DRIVER_VERSION`).
pub const DRIVER_VERSION: u32 = 0x102D;
/// Device profile (`CL_DEVICE_PROFILE`).
pub const DEVICE_PROFILE: u32 = 0x102E;
/// Device version string (`CL_DEVICE_VERSION`).
pub const DEVICE_VERSION: u32 = 0x102F;
/// Space-separated list of device extensions (`CL_DEVICE_EXTENSIONS`).
pub const DEVICE_EXTENSIONS: u32 = 0x1030;
/// The platform the device belongs to (`CL_DEVICE_PLATFORM`).
pub const DEVICE_PLATFORM: u32 = 0x1031;

/// Reference count of the program (`CL_PROGRAM_REFERENCE_COUNT`).
pub const PROGRAM_REFERENCE_COUNT: u32 = 0x1160;
/// The context the program belongs to (`CL_PROGRAM_CONTEXT`).
pub const PROGRAM_CONTEXT: u32 = 0x1161;
/// Number of devices the program was built for (`CL_PROGRAM_NUM_DEVICES`).
pub const PROGRAM_NUM_DEVICES: u32 = 0x1162;
/// Devices the program was built for (`CL_PROGRAM_DEVICES`).
pub const PROGRAM_DEVICES: u32 = 0x1163;
/// Program source text (`CL_PROGRAM_SOURCE`).
pub const PROGRAM_SOURCE: u32 = 0x1164;
/// Build status for a device (`CL_PROGRAM_BUILD_STATUS`).
pub const PROGRAM_BUILD_STATUS: u32 = 0x1181;
/// Build options used for a device (`CL_PROGRAM_BUILD_OPTIONS`).
pub const PROGRAM_BUILD_OPTIONS: u32 = 0x1182;
/// Human-readable build log for a device (`CL_PROGRAM_BUILD_LOG`).
pub const PROGRAM_BUILD_LOG: u32 = 0x1183;
/// Build succeeded (`CL_BUILD_SUCCESS`).
pub const BUILD_SUCCESS: i32 = 0;
/// No build has been started (`CL_BUILD_NONE`).
pub const BUILD_NONE: i32 = -1;
/// The build failed (`CL_BUILD_ERROR`).
pub const BUILD_ERROR: i32 = -2;
/// The build is still running (`CL_BUILD_IN_PROGRESS`).
pub const BUILD_IN_PROGRESS: i32 = -3;

/// The kernel's entry-point function name (`CL_KERNEL_FUNCTION_NAME`).
pub const KERNEL_FUNCTION_NAME: u32 = 0x1190;
/// Number of arguments the kernel takes (`CL_KERNEL_NUM_ARGS`).
pub const KERNEL_NUM_ARGS: u32 = 0x1191;
/// The context the kernel belongs to (`CL_KERNEL_CONTEXT`).
pub const KERNEL_CONTEXT: u32 = 0x1193;
/// The program the kernel was created from (`CL_KERNEL_PROGRAM`).
pub const KERNEL_PROGRAM: u32 = 0x1194;

/// Maximum work-group size for the kernel on a device
/// (`CL_KERNEL_WORK_GROUP_SIZE`).
pub const KERNEL_WORK_GROUP_SIZE: u32 = 0x1100;
/// Preferred work-group size as a multiple of the device's warp size
/// (`CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`).
pub const KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: u32 = 0x1103;

/// The execution status of the command associated with the event
/// (`CL_EVENT_COMMAND_EXECUTION_STATUS`).
pub const EVENT_COMMAND_EXECUTION_STATUS: u32 = 0x11D3;

/// The command completed successfully (`CL_COMPLETE`).
pub const COMPLETE: i32 = 0x0;
/// The command is currently executing (`CL_RUNNING`).
pub const RUNNING: i32 = 0x1;
/// The command has been submitted to the device (`CL_SUBMITTED`).
pub const SUBMITTED: i32 = 0x2;
/// The command is in the command queue (`CL_QUEUED`).
pub const QUEUED: i32 = 0x3;

/// Host timestamp: queued (`CL_PROFILING_COMMAND_QUEUED`).
pub const PROFILING_COMMAND_QUEUED: u32 = 0x1280;
/// Host timestamp: submitted (`CL_PROFILING_COMMAND_SUBMIT`).
pub const PROFILING_COMMAND_SUBMIT: u32 = 0x1281;
/// Host timestamp: started on the device (`CL_PROFILING_COMMAND_START`).
pub const PROFILING_COMMAND_START: u32 = 0x1282;
/// Host timestamp: finished on the device (`CL_PROFILING_COMMAND_END`).
pub const PROFILING_COMMAND_END: u32 = 0x1283;

/// Context creation notification callback ([`CreateContext`]).
pub type ContextNotify = unsafe extern "system" fn(
    errinfo: *const c_char,
    private_info: *const c_void,
    cb: usize,
    user_data: *mut c_void,
);

/// Program build notification callback ([`BuildProgram`]).
pub type ProgramNotify = unsafe extern "system" fn(program: ProgramHandle, user_data: *mut c_void);

/// Enumerates OpenCL platforms (`clGetPlatformIDs`).
pub type GetPlatformIds = unsafe extern "system" fn(
    num_entries: u32,
    platforms: *mut PlatformHandle,
    num_platforms: *mut u32,
) -> i32;

/// Queries platform attributes (`clGetPlatformInfo`).
pub type GetPlatformInfo = unsafe extern "system" fn(
    platform: PlatformHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32;

/// Enumerates devices on a platform (`clGetDeviceIDs`).
pub type GetDeviceIds = unsafe extern "system" fn(
    platform: PlatformHandle,
    device_type: u64,
    num_entries: u32,
    devices: *mut DeviceHandle,
    num_devices: *mut u32,
) -> i32;

/// Queries device attributes (`clGetDeviceInfo`).
pub type GetDeviceInfo = unsafe extern "system" fn(
    device: DeviceHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32;

/// Creates a context for the given devices (`clCreateContext`).
pub type CreateContext = unsafe extern "system" fn(
    properties: *const isize,
    num_devices: u32,
    devices: *const DeviceHandle,
    pfn_notify: Option<ContextNotify>,
    user_data: *mut c_void,
    errcode_ret: *mut i32,
) -> ContextHandle;

/// Decrements a context's reference count (`clReleaseContext`).
pub type ReleaseContext = unsafe extern "system" fn(context: ContextHandle) -> i32;

/// Creates a command queue for a device (`clCreateCommandQueue`).
pub type CreateCommandQueue = unsafe extern "system" fn(
    context: ContextHandle,
    device: DeviceHandle,
    properties: u64,
    errcode_ret: *mut i32,
) -> CommandQueueHandle;

/// Decrements a command queue's reference count (`clReleaseCommandQueue`).
pub type ReleaseCommandQueue = unsafe extern "system" fn(command_queue: CommandQueueHandle) -> i32;

/// Submits queued commands to the device (`clFlush`).
pub type Flush = unsafe extern "system" fn(command_queue: CommandQueueHandle) -> i32;

/// Waits until all queued commands complete (`clFinish`).
pub type Finish = unsafe extern "system" fn(command_queue: CommandQueueHandle) -> i32;

/// Creates a buffer object (`clCreateBuffer`).
pub type CreateBuffer = unsafe extern "system" fn(
    context: ContextHandle,
    flags: u64,
    size: usize,
    host_ptr: *mut c_void,
    errcode_ret: *mut i32,
) -> MemHandle;

/// Decrements a memory object's reference count (`clReleaseMemObject`).
pub type ReleaseMemObject = unsafe extern "system" fn(memobj: MemHandle) -> i32;

/// Creates a program from OpenCL C source text (`clCreateProgramWithSource`).
pub type CreateProgramWithSource = unsafe extern "system" fn(
    context: ContextHandle,
    count: u32,
    strings: *const *const c_char,
    lengths: *const usize,
    errcode_ret: *mut i32,
) -> ProgramHandle;

/// Decrements a program's reference count (`clReleaseProgram`).
pub type ReleaseProgram = unsafe extern "system" fn(program: ProgramHandle) -> i32;

/// Builds a program for the given devices (`clBuildProgram`).
pub type BuildProgram = unsafe extern "system" fn(
    program: ProgramHandle,
    num_devices: u32,
    device_list: *const DeviceHandle,
    options: *const c_char,
    pfn_notify: Option<ProgramNotify>,
    user_data: *mut c_void,
) -> i32;

/// Queries program attributes such as [`PROGRAM_SOURCE`] (`clGetProgramInfo`).
pub type GetProgramInfo = unsafe extern "system" fn(
    program: ProgramHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32;

/// Queries the build log or status for one device (`clGetProgramBuildInfo`).
pub type GetProgramBuildInfo = unsafe extern "system" fn(
    program: ProgramHandle,
    device: DeviceHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32;

/// Creates a kernel from a program entry point (`clCreateKernel`).
pub type CreateKernel = unsafe extern "system" fn(
    program: ProgramHandle,
    kernel_name: *const c_char,
    errcode_ret: *mut i32,
) -> KernelHandle;

/// Decrements a kernel's reference count (`clReleaseKernel`).
pub type ReleaseKernel = unsafe extern "system" fn(kernel: KernelHandle) -> i32;

/// Binds a value or memory object to a kernel argument (`clSetKernelArg`).
pub type SetKernelArg = unsafe extern "system" fn(
    kernel: KernelHandle,
    arg_index: u32,
    arg_size: usize,
    arg_value: *const c_void,
) -> i32;

/// Queries kernel attributes (`clGetKernelInfo`).
pub type GetKernelInfo = unsafe extern "system" fn(
    kernel: KernelHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32;

/// Queries kernel work-group limits for a device (`clGetKernelWorkGroupInfo`).
pub type GetKernelWorkGroupInfo = unsafe extern "system" fn(
    kernel: KernelHandle,
    device: DeviceHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32;

/// Reads a buffer into host memory (`clEnqueueReadBuffer`).
pub type EnqueueReadBuffer = unsafe extern "system" fn(
    command_queue: CommandQueueHandle,
    buffer: MemHandle,
    blocking_read: u32,
    offset: usize,
    size: usize,
    ptr: *mut c_void,
    num_events_in_wait_list: u32,
    event_wait_list: *const EventHandle,
    event: *mut EventHandle,
) -> i32;

/// Writes host memory into a buffer (`clEnqueueWriteBuffer`).
pub type EnqueueWriteBuffer = unsafe extern "system" fn(
    command_queue: CommandQueueHandle,
    buffer: MemHandle,
    blocking_write: u32,
    offset: usize,
    size: usize,
    ptr: *const c_void,
    num_events_in_wait_list: u32,
    event_wait_list: *const EventHandle,
    event: *mut EventHandle,
) -> i32;

/// Copies between two buffers (`clEnqueueCopyBuffer`).
pub type EnqueueCopyBuffer = unsafe extern "system" fn(
    command_queue: CommandQueueHandle,
    src_buffer: MemHandle,
    dst_buffer: MemHandle,
    src_offset: usize,
    dst_offset: usize,
    size: usize,
    num_events_in_wait_list: u32,
    event_wait_list: *const EventHandle,
    event: *mut EventHandle,
) -> i32;

/// Runs a kernel over an N-dimensional range (`clEnqueueNDRangeKernel`).
pub type EnqueueNdRangeKernel = unsafe extern "system" fn(
    command_queue: CommandQueueHandle,
    kernel: KernelHandle,
    work_dim: u32,
    global_work_offset: *const usize,
    global_work_size: *const usize,
    local_work_size: *const usize,
    num_events_in_wait_list: u32,
    event_wait_list: *const EventHandle,
    event: *mut EventHandle,
) -> i32;

/// Waits for events to complete (`clWaitForEvents`).
pub type WaitForEvents = unsafe extern "system" fn(num_events: u32, event_list: *const EventHandle) -> i32;

/// Decrements an event's reference count (`clReleaseEvent`).
pub type ReleaseEvent = unsafe extern "system" fn(event: EventHandle) -> i32;

/// Queries event attributes (`clGetEventInfo`).
pub type GetEventInfo = unsafe extern "system" fn(
    event: EventHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32;

/// Queries profiling timestamps for a completed event
/// (`clGetEventProfilingInfo`).
pub type GetEventProfilingInfo = unsafe extern "system" fn(
    event: EventHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32;

/// Fills a buffer region with a repeated pattern (`clEnqueueFillBuffer`,
/// OpenCL 1.2).
pub type EnqueueFillBuffer = unsafe extern "system" fn(
    command_queue: CommandQueueHandle,
    buffer: MemHandle,
    pattern: *const c_void,
    pattern_size: usize,
    offset: usize,
    size: usize,
    num_events_in_wait_list: u32,
    event_wait_list: *const EventHandle,
    event: *mut EventHandle,
) -> i32;

/// Resolves an extension entry point by name (`clGetExtensionFunctionAddress`).
///
/// Core entry-point names intentionally return null on ICD-compliant
/// implementations; only `clXxxKHR`-style extension names resolve.
pub type GetExtensionFunctionAddress = unsafe extern "system" fn(func_name: *const c_char) -> *mut c_void;

/// ICD entry point used to enumerate a vendor implementation's platforms
/// directly (`clIcdGetPlatformIDsKHR`, `cl_khr_icd`).
pub type IcdGetPlatformIdsKhr = unsafe extern "system" fn(
    num_entries: u32,
    platforms: *mut PlatformHandle,
    num_platforms: *mut u32,
) -> i32;

/// Returns the canonical short name of an OpenCL status code.
///
/// The returned strings keep the full `CL_` prefix of the Khronos headers
/// because those are the names used in driver logs and Khronos documentation.
/// Unknown codes render as `CL_UNKNOWN_ERROR`. This function is a pure
/// constant lookup with no allocation.
///
/// # Examples
///
/// ```
/// assert_eq!(codevar_ocl::sys::error_name(0), "CL_SUCCESS");
/// assert_eq!(codevar_ocl::sys::error_name(-5), "CL_OUT_OF_RESOURCES");
/// ```
#[must_use]
pub const fn error_name(code: i32) -> &'static str {
    match code {
        SUCCESS => "CL_SUCCESS",
        DEVICE_NOT_FOUND => "CL_DEVICE_NOT_FOUND",
        DEVICE_NOT_AVAILABLE => "CL_DEVICE_NOT_AVAILABLE",
        COMPILER_NOT_AVAILABLE => "CL_COMPILER_NOT_AVAILABLE",
        MEM_OBJECT_ALLOCATION_FAILURE => "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        OUT_OF_RESOURCES => "CL_OUT_OF_RESOURCES",
        OUT_OF_HOST_MEMORY => "CL_OUT_OF_HOST_MEMORY",
        PROFILING_INFO_NOT_AVAILABLE => "CL_PROFILING_INFO_NOT_AVAILABLE",
        MEM_COPY_OVERLAP => "CL_MEM_COPY_OVERLAP",
        IMAGE_FORMAT_MISMATCH => "CL_IMAGE_FORMAT_MISMATCH",
        IMAGE_FORMAT_NOT_SUPPORTED => "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        BUILD_PROGRAM_FAILURE => "CL_BUILD_PROGRAM_FAILURE",
        MAP_FAILURE => "CL_MAP_FAILURE",
        MISALIGNED_SUB_BUFFER_OFFSET => "CL_MISALIGNED_SUB_BUFFER_OFFSET",
        EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST => "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
        COMPILE_PROGRAM_FAILURE => "CL_COMPILE_PROGRAM_FAILURE",
        LINKER_NOT_AVAILABLE => "CL_LINKER_NOT_AVAILABLE",
        LINK_PROGRAM_FAILURE => "CL_LINK_PROGRAM_FAILURE",
        DEVICE_PARTITION_FAILED => "CL_DEVICE_PARTITION_FAILED",
        KERNEL_ARG_INFO_NOT_AVAILABLE => "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
        INVALID_VALUE => "CL_INVALID_VALUE",
        INVALID_DEVICE_TYPE => "CL_INVALID_DEVICE_TYPE",
        INVALID_PLATFORM => "CL_INVALID_PLATFORM",
        INVALID_DEVICE => "CL_INVALID_DEVICE",
        INVALID_CONTEXT => "CL_INVALID_CONTEXT",
        INVALID_QUEUE_PROPERTIES => "CL_INVALID_QUEUE_PROPERTIES",
        INVALID_COMMAND_QUEUE => "CL_INVALID_COMMAND_QUEUE",
        INVALID_HOST_PTR => "CL_INVALID_HOST_PTR",
        INVALID_SAMPLER => "CL_INVALID_SAMPLER",
        INVALID_BINARY => "CL_INVALID_BINARY",
        INVALID_BUILD_OPTIONS => "CL_INVALID_BUILD_OPTIONS",
        INVALID_PROGRAM => "CL_INVALID_PROGRAM",
        INVALID_PROGRAM_EXECUTABLE => "CL_INVALID_PROGRAM_EXECUTABLE",
        INVALID_KERNEL_NAME => "CL_INVALID_KERNEL_NAME",
        INVALID_KERNEL_DEFINITION => "CL_INVALID_KERNEL_DEFINITION",
        INVALID_KERNEL => "CL_INVALID_KERNEL",
        INVALID_ARG_INDEX => "CL_INVALID_ARG_INDEX",
        INVALID_ARG_VALUE => "CL_INVALID_ARG_VALUE",
        INVALID_ARG_SIZE => "CL_INVALID_ARG_SIZE",
        INVALID_KERNEL_ARGS => "CL_INVALID_KERNEL_ARGS",
        INVALID_WORK_DIMENSION => "CL_INVALID_WORK_DIMENSION",
        INVALID_WORK_GROUP_SIZE => "CL_INVALID_WORK_GROUP_SIZE",
        INVALID_WORK_ITEM_SIZE => "CL_INVALID_WORK_ITEM_SIZE",
        INVALID_GLOBAL_OFFSET => "CL_INVALID_GLOBAL_OFFSET",
        INVALID_EVENT_WAIT_LIST => "CL_INVALID_EVENT_WAIT_LIST",
        INVALID_EVENT => "CL_INVALID_EVENT",
        INVALID_OPERATION => "CL_INVALID_OPERATION",
        INVALID_GL_OBJECT => "CL_INVALID_GL_OBJECT",
        INVALID_BUFFER_SIZE => "CL_INVALID_BUFFER_SIZE",
        INVALID_MIP_LEVEL => "CL_INVALID_MIP_LEVEL",
        INVALID_GLOBAL_WORK_SIZE => "CL_INVALID_GLOBAL_WORK_SIZE",
        INVALID_PROPERTY => "CL_INVALID_PROPERTY",
        INVALID_IMAGE_DESCRIPTOR => "CL_INVALID_IMAGE_DESCRIPTOR",
        INVALID_COMPILER_OPTIONS => "CL_INVALID_COMPILER_OPTIONS",
        INVALID_LINKER_OPTIONS => "CL_INVALID_LINKER_OPTIONS",
        INVALID_DEVICE_PARTITION_COUNT => "CL_INVALID_DEVICE_PARTITION_COUNT",
        INVALID_PIPE_SIZE => "CL_INVALID_PIPE_SIZE",
        INVALID_DEVICE_QUEUE => "CL_INVALID_DEVICE_QUEUE",
        PLATFORM_NOT_FOUND_KHR => "CL_PLATFORM_NOT_FOUND_KHR",
        _ => "CL_UNKNOWN_ERROR",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_names_keep_the_khronos_spelling() {
        assert_eq!(error_name(SUCCESS), "CL_SUCCESS");
        assert_eq!(error_name(OUT_OF_RESOURCES), "CL_OUT_OF_RESOURCES");
        assert_eq!(error_name(INVALID_KERNEL_ARGS), "CL_INVALID_KERNEL_ARGS");
        assert_eq!(error_name(PLATFORM_NOT_FOUND_KHR), "CL_PLATFORM_NOT_FOUND_KHR");
        assert_eq!(error_name(1234), "CL_UNKNOWN_ERROR");
        assert_eq!(error_name(-999), "CL_UNKNOWN_ERROR");
    }

    #[test]
    fn handle_newtypes_round_trip_raw_pointers() {
        let null = ContextHandle::from_raw(core::ptr::null_mut());
        assert!(null.is_null());
        assert!(Nullable::is_null(&null));
        assert_eq!(null.as_raw(), core::ptr::null_mut());

        let raw = 0x1234usize as *mut c_void;
        let handle = MemHandle::from_raw(raw);
        assert!(!handle.is_null());
        assert_eq!(handle.as_raw(), raw);
    }

    #[test]
    fn entry_point_types_have_pointer_sized_fn_pointers() {
        // `api.rs` transmutes `dlsym` results into these types; that is only
        // sound while function pointers are pointer-sized.
        assert_eq!(
            core::mem::size_of::<GetPlatformIds>(),
            core::mem::size_of::<*mut c_void>()
        );
        assert_eq!(
            core::mem::size_of::<CreateContext>(),
            core::mem::size_of::<*mut c_void>()
        );
        assert_eq!(
            core::mem::size_of::<EnqueueNdRangeKernel>(),
            core::mem::size_of::<*mut c_void>()
        );
    }

    #[test]
    fn status_and_flag_values_match_the_headers() {
        assert_eq!(SUCCESS, 0);
        assert_eq!(TRUE, 1);
        assert_eq!(FALSE, 0);
        assert_eq!(DEVICE_TYPE_ALL, 0xFFFF_FFFF);
        assert_eq!(DEVICE_TYPE_GPU, 4);
        assert_eq!(MEM_READ_WRITE, 1);
        assert_eq!(QUEUE_PROFILING_ENABLE, 2);
        assert_eq!(PLATFORM_NAME, 0x0902);
        assert_eq!(DEVICE_NAME, 0x102B);
        assert_eq!(BUILD_PROGRAM_FAILURE, -11);
        assert_eq!(PLATFORM_NOT_FOUND_KHR, -1001);
    }
}
