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

//! Raw OpenCL types, constants and handles for the bare-metal driver.
//!
//! A trimmed, self-contained translation of the subset of the Khronos
//! `CL/cl_platform.h`, `CL/cl.h` and `CL/cl_ext.h` headers that this driver
//! implements, following the naming conventions of [`codevar_ocl::sys`] on
//! the host side (constants drop the `CL_` prefix, entry points drop the
//! `cl` prefix, handles are `#[repr(transparent)]` newtypes over
//! `*mut c_void`). Unlike the host runtime, nothing here is resolved at
//! runtime: the `cl*` entry points of [`crate::api`] are ordinary `extern
//! "C"` functions compiled into the image, so no ICD loader, no `dlopen`
//! and no dynamic linker are involved at any point.
//!
//! The subset targets the OpenCL **embedded profile**: no images, no
//! samplers, no pipes, no device queues, no online compiler, no SVM.
//!
//! # Reference
//!
//! * <https://github.com/KhronosGroup/OpenCL-Headers>
//! * <https://registry.khronos.org/OpenCL/specs/3.0/html/OpenCL_API.html>
//!   (embedded profile requirements)

use core::ffi::{c_char, c_void};

/// Success status returned by every OpenCL entry point (`CL_SUCCESS`).
pub const SUCCESS: i32 = 0;
/// `true` value for `cl_bool` queries and blocking flags (`CL_TRUE`).
pub const TRUE: u32 = 1;
/// `false` value for `cl_bool` queries and blocking flags (`CL_FALSE`).
pub const FALSE: u32 = 0;
/// Blocking form of an enqueue call (`CL_BLOCKING`).
pub const BLOCKING: u32 = TRUE;
/// Non-blocking form of an enqueue call (`CL_NON_BLOCKING`).
pub const NON_BLOCKING: u32 = FALSE;

/// The named device is not present (`CL_DEVICE_NOT_FOUND`).
pub const DEVICE_NOT_FOUND: i32 = -1;
/// No online compiler is available (`CL_COMPILER_NOT_AVAILABLE`).
pub const COMPILER_NOT_AVAILABLE: i32 = -3;
/// Device-side allocation failed (`CL_MEM_OBJECT_ALLOCATION_FAILURE`).
pub const MEM_OBJECT_ALLOCATION_FAILURE: i32 = -4;
/// The device ran out of resources (`CL_OUT_OF_RESOURCES`).
pub const OUT_OF_RESOURCES: i32 = -5;
/// The host ran out of memory (`CL_OUT_OF_HOST_MEMORY`).
pub const OUT_OF_HOST_MEMORY: i32 = -6;
/// Profiling information is not available (`CL_PROFILING_INFO_NOT_AVAILABLE`).
pub const PROFILING_INFO_NOT_AVAILABLE: i32 = -7;
/// The build failed (`CL_BUILD_PROGRAM_FAILURE`).
pub const BUILD_PROGRAM_FAILURE: i32 = -11;
/// A wait-list event finished with an error
/// (`CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`).
pub const EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: i32 = -14;

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
/// The memory object is invalid (`CL_INVALID_MEM_OBJECT`).
pub const INVALID_MEM_OBJECT: i32 = -38;
/// The program binary is invalid (`CL_INVALID_BINARY`).
pub const INVALID_BINARY: i32 = -42;
/// The build options string is invalid (`CL_INVALID_BUILD_OPTIONS`).
pub const INVALID_BUILD_OPTIONS: i32 = -43;
/// The program handle is invalid (`CL_INVALID_PROGRAM`).
pub const INVALID_PROGRAM: i32 = -44;
/// No built executable exists for the program
/// (`CL_INVALID_PROGRAM_EXECUTABLE`).
pub const INVALID_PROGRAM_EXECUTABLE: i32 = -45;
/// The kernel name is invalid (`CL_INVALID_KERNEL_NAME`).
pub const INVALID_KERNEL_NAME: i32 = -46;
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
/// The operation is not valid for these objects (`CL_INVALID_OPERATION`).
pub const INVALID_OPERATION: i32 = -59;
/// The requested buffer size is invalid (`CL_INVALID_BUFFER_SIZE`).
pub const INVALID_BUFFER_SIZE: i32 = -61;
/// The global work size is invalid (`CL_INVALID_GLOBAL_WORK_SIZE`).
pub const INVALID_GLOBAL_WORK_SIZE: i32 = -63;
/// A property in a property list is invalid (`CL_INVALID_PROPERTY`).
pub const INVALID_PROPERTY: i32 = -64;
/// No platform was found (`CL_PLATFORM_NOT_FOUND_KHR`).
pub const PLATFORM_NOT_FOUND_KHR: i32 = -1001;

/// Use every device on the platform (`CL_DEVICE_TYPE_ALL`).
pub const DEVICE_TYPE_ALL: u64 = 0xFFFF_FFFF;
/// The default device of the platform (`CL_DEVICE_TYPE_DEFAULT`).
pub const DEVICE_TYPE_DEFAULT: u64 = 1 << 0;
/// A CPU device (`CL_DEVICE_TYPE_CPU`).
pub const DEVICE_TYPE_CPU: u64 = 1 << 1;
/// A GPU device (`CL_DEVICE_TYPE_GPU`).
pub const DEVICE_TYPE_GPU: u64 = 1 << 2;
/// A dedicated accelerator (`CL_DEVICE_TYPE_ACCELERATOR`).
pub const DEVICE_TYPE_ACCELERATOR: u64 = 1 << 3;
/// A custom device (`CL_DEVICE_TYPE_CUSTOM`).
pub const DEVICE_TYPE_CUSTOM: u64 = 1 << 4;

/// The buffer may be read and written (`CL_MEM_READ_WRITE`).
pub const MEM_READ_WRITE: u64 = 1 << 0;
/// The kernel may only write the buffer (`CL_MEM_WRITE_ONLY`).
pub const MEM_WRITE_ONLY: u64 = 1 << 1;
/// The kernel may only read the buffer (`CL_MEM_READ_ONLY`).
pub const MEM_READ_ONLY: u64 = 1 << 2;
/// Use the caller-provided host pointer as storage (`CL_MEM_USE_HOST_PTR`).
pub const MEM_USE_HOST_PTR: u64 = 1 << 3;
/// Let the implementation allocate host-visible storage
/// (`CL_MEM_ALLOC_HOST_PTR`).
pub const MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
/// Copy the host pointer into the buffer at creation
/// (`CL_MEM_COPY_HOST_PTR`).
pub const MEM_COPY_HOST_PTR: u64 = 1 << 5;

/// Commands complete out of order (`CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`).
pub const QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE: u64 = 1 << 0;
/// The queue collects profiling timestamps (`CL_QUEUE_PROFILING_ENABLE`).
pub const QUEUE_PROFILING_ENABLE: u64 = 1 << 1;

/// Full or embedded platform profile (`CL_PLATFORM_PROFILE`).
pub const PLATFORM_PROFILE: u32 = 0x0900;
/// OpenCL version string (`CL_PLATFORM_VERSION`).
pub const PLATFORM_VERSION: u32 = 0x0901;
/// Platform name (`CL_PLATFORM_NAME`).
pub const PLATFORM_NAME: u32 = 0x0902;
/// Platform vendor (`CL_PLATFORM_VENDOR`).
pub const PLATFORM_VENDOR: u32 = 0x0903;
/// Platform extensions (`CL_PLATFORM_EXTENSIONS`).
pub const PLATFORM_EXTENSIONS: u32 = 0x0904;

/// Device type bitfield (`CL_DEVICE_TYPE`).
pub const DEVICE_TYPE: u32 = 0x1000;
/// Vendor ID (`CL_DEVICE_VENDOR_ID`).
pub const DEVICE_VENDOR_ID: u32 = 0x1001;
/// Maximum compute units (`CL_DEVICE_MAX_COMPUTE_UNITS`).
pub const DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
/// Maximum work-item dimensions (`CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS`).
pub const DEVICE_MAX_WORK_ITEM_DIMENSIONS: u32 = 0x1003;
/// Maximum work-group size (`CL_DEVICE_MAX_WORK_GROUP_SIZE`).
pub const DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;
/// Maximum work-item sizes (`CL_DEVICE_MAX_WORK_ITEM_SIZES`).
pub const DEVICE_MAX_WORK_ITEM_SIZES: u32 = 0x1005;
/// Address width in bits (`CL_DEVICE_ADDRESS_BITS`).
pub const DEVICE_ADDRESS_BITS: u32 = 0x100D;
/// Maximum single allocation (`CL_DEVICE_MAX_MEM_ALLOC_SIZE`).
pub const DEVICE_MAX_MEM_ALLOC_SIZE: u32 = 0x1010;
/// Whether images are supported (`CL_DEVICE_IMAGE_SUPPORT`).
pub const DEVICE_IMAGE_SUPPORT: u32 = 0x1016;
/// Total global memory (`CL_DEVICE_GLOBAL_MEM_SIZE`).
pub const DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
/// Local memory size (`CL_DEVICE_LOCAL_MEM_SIZE`).
pub const DEVICE_LOCAL_MEM_SIZE: u32 = 0x1023;
/// Whether the device is available (`CL_DEVICE_AVAILABLE`).
pub const DEVICE_AVAILABLE: u32 = 0x1027;
/// Whether an online compiler exists (`CL_DEVICE_COMPILER_AVAILABLE`).
pub const DEVICE_COMPILER_AVAILABLE: u32 = 0x1028;
/// Device name (`CL_DEVICE_NAME`).
pub const DEVICE_NAME: u32 = 0x102B;
/// Device vendor (`CL_DEVICE_VENDOR`).
pub const DEVICE_VENDOR: u32 = 0x102C;
/// Driver version (`CL_DRIVER_VERSION`).
pub const DRIVER_VERSION: u32 = 0x102D;
/// Device profile (`CL_DEVICE_PROFILE`).
pub const DEVICE_PROFILE: u32 = 0x102E;
/// Device version (`CL_DEVICE_VERSION`).
pub const DEVICE_VERSION: u32 = 0x102F;
/// Device extensions (`CL_DEVICE_EXTENSIONS`).
pub const DEVICE_EXTENSIONS: u32 = 0x1030;
/// The platform of the device (`CL_DEVICE_PLATFORM`).
pub const DEVICE_PLATFORM: u32 = 0x1031;
/// Host and device share unified memory (`CL_DEVICE_HOST_UNIFIED_MEMORY`).
pub const DEVICE_HOST_UNIFIED_MEMORY: u32 = 0x1035;
/// Built-in kernel names (`CL_DEVICE_BUILTIN_KERNELS`).
pub const DEVICE_BUILTIN_KERNELS: u32 = 0x1036;
/// Maximum kernel argument block in bytes (`CL_DEVICE_MAX_PARAMETER_SIZE`).
pub const DEVICE_MAX_PARAMETER_SIZE: u32 = 0x1058;

/// Context reference count (`CL_CONTEXT_REFERENCE_COUNT`).
pub const CONTEXT_REFERENCE_COUNT: u32 = 0x1080;
/// Context devices (`CL_CONTEXT_DEVICES`).
pub const CONTEXT_DEVICES: u32 = 0x1081;
/// Number of context devices (`CL_CONTEXT_NUM_DEVICES`).
pub const CONTEXT_NUM_DEVICES: u32 = 0x1082;
/// Context property list (`CL_CONTEXT_PROPERTIES`).
pub const CONTEXT_PROPERTIES: u32 = 0x1083;
/// Platform property key in a context property list (`CL_CONTEXT_PLATFORM`).
pub const CONTEXT_PLATFORM: isize = 0x1084;

/// The queue's context (`CL_QUEUE_CONTEXT`).
pub const QUEUE_CONTEXT: u32 = 0x1090;
/// The queue's device (`CL_QUEUE_DEVICE`).
pub const QUEUE_DEVICE: u32 = 0x1091;
/// Queue reference count (`CL_QUEUE_REFERENCE_COUNT`).
pub const QUEUE_REFERENCE_COUNT: u32 = 0x1092;
/// Queue properties (`CL_QUEUE_PROPERTIES`).
pub const QUEUE_PROPERTIES: u32 = 0x1093;

/// Program reference count (`CL_PROGRAM_REFERENCE_COUNT`).
pub const PROGRAM_REFERENCE_COUNT: u32 = 0x1160;
/// The context of the program (`CL_PROGRAM_CONTEXT`).
pub const PROGRAM_CONTEXT: u32 = 0x1161;
/// Number of devices (`CL_PROGRAM_NUM_DEVICES`).
pub const PROGRAM_NUM_DEVICES: u32 = 0x1162;
/// Devices of the program (`CL_PROGRAM_DEVICES`).
pub const PROGRAM_DEVICES: u32 = 0x1163;
/// Program source (`CL_PROGRAM_SOURCE`).
pub const PROGRAM_SOURCE: u32 = 0x1164;
/// Number of associated kernels (`CL_PROGRAM_NUM_KERNELS`).
pub const PROGRAM_NUM_KERNELS: u32 = 0x1167;
/// Kernel names (`CL_PROGRAM_KERNEL_NAMES`).
pub const PROGRAM_KERNEL_NAMES: u32 = 0x1168;
/// Build status (`CL_PROGRAM_BUILD_STATUS`).
pub const PROGRAM_BUILD_STATUS: u32 = 0x1181;
/// Build options (`CL_PROGRAM_BUILD_OPTIONS`).
pub const PROGRAM_BUILD_OPTIONS: u32 = 0x1182;
/// Build log (`CL_PROGRAM_BUILD_LOG`).
pub const PROGRAM_BUILD_LOG: u32 = 0x1183;
/// Build succeeded (`CL_BUILD_SUCCESS`).
pub const BUILD_SUCCESS: i32 = 0;
/// No build has been started (`CL_BUILD_NONE`).
pub const BUILD_NONE: i32 = -1;
/// The build failed (`CL_BUILD_ERROR`).
pub const BUILD_ERROR: i32 = -2;

/// The kernel's function name (`CL_KERNEL_FUNCTION_NAME`).
pub const KERNEL_FUNCTION_NAME: u32 = 0x1190;
/// Number of arguments (`CL_KERNEL_NUM_ARGS`).
pub const KERNEL_NUM_ARGS: u32 = 0x1191;
/// The context of the kernel (`CL_KERNEL_CONTEXT`).
pub const KERNEL_CONTEXT: u32 = 0x1193;
/// The program of the kernel (`CL_KERNEL_PROGRAM`).
pub const KERNEL_PROGRAM: u32 = 0x1194;
/// Maximum work-group size for the kernel (`CL_KERNEL_WORK_GROUP_SIZE`).
pub const KERNEL_WORK_GROUP_SIZE: u32 = 0x1100;
/// Preferred work-group multiple (`CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`).
pub const KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: u32 = 0x1103;

/// Buffer reference count (`CL_MEM_REFERENCE_COUNT`).
pub const MEM_REFERENCE_COUNT: u32 = 0x1114;
/// Buffer context (`CL_MEM_CONTEXT`).
pub const MEM_CONTEXT: u32 = 0x1116;
/// Buffer size in bytes (`CL_MEM_SIZE`).
pub const MEM_SIZE: u32 = 0x1111;
/// Buffer flags (`CL_MEM_FLAGS`).
pub const MEM_FLAGS: u32 = 0x1110;

/// The command the event tracks (`CL_EVENT_COMMAND_TYPE`).
pub const EVENT_COMMAND_TYPE: u32 = 0x11D1;
/// Event reference count (`CL_EVENT_REFERENCE_COUNT`).
pub const EVENT_REFERENCE_COUNT: u32 = 0x11D2;
/// The execution status of the command (`CL_EVENT_COMMAND_EXECUTION_STATUS`).
pub const EVENT_COMMAND_EXECUTION_STATUS: u32 = 0x11D3;
/// The context of the event's command (`CL_EVENT_CONTEXT`).
pub const EVENT_CONTEXT: u32 = 0x11D4;
/// The queue of the event's command (`CL_EVENT_COMMAND_QUEUE`).
pub const EVENT_COMMAND_QUEUE: u32 = 0x11D0;
/// The command completed successfully (`CL_COMPLETE`).
pub const COMPLETE: i32 = 0x0;
/// The command is currently executing (`CL_RUNNING`).
pub const RUNNING: i32 = 0x1;
/// The command has been submitted (`CL_SUBMITTED`).
pub const SUBMITTED: i32 = 0x2;
/// The command is in the command queue (`CL_QUEUED`).
pub const QUEUED: i32 = 0x3;

/// `cl_command_type` for an NDRange kernel command (`CL_COMMAND_NDRANGE_KERNEL`).
pub const COMMAND_NDRANGE_KERNEL: u32 = 0x11F0;
/// `cl_command_type` for a buffer read (`CL_COMMAND_READ_BUFFER`).
pub const COMMAND_READ_BUFFER: u32 = 0x11F2;
/// `cl_command_type` for a buffer write (`CL_COMMAND_WRITE_BUFFER`).
pub const COMMAND_WRITE_BUFFER: u32 = 0x11F3;
/// `cl_command_type` for a buffer copy (`CL_COMMAND_COPY_BUFFER`).
pub const COMMAND_COPY_BUFFER: u32 = 0x11F4;

/// Profiling timestamp: queued (`CL_PROFILING_COMMAND_QUEUED`).
pub const PROFILING_COMMAND_QUEUED: u32 = 0x1280;
/// Profiling timestamp: submitted (`CL_PROFILING_COMMAND_SUBMIT`).
pub const PROFILING_COMMAND_SUBMIT: u32 = 0x1281;
/// Profiling timestamp: started (`CL_PROFILING_COMMAND_START`).
pub const PROFILING_COMMAND_START: u32 = 0x1282;
/// Profiling timestamp: ended (`CL_PROFILING_COMMAND_END`).
pub const PROFILING_COMMAND_END: u32 = 0x1283;

macro_rules! handle {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        ///
        /// `#[repr(transparent)]` over `*mut c_void`, so it has the C pointer
        /// layout while remaining distinct from every other handle type.
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(*mut c_void);

        impl $name {
            /// Wraps a raw pointer produced by the driver.
            #[must_use]
            pub const fn from_raw(raw: *mut c_void) -> Self {
                Self(raw)
            }

            /// Returns the raw pointer for use as a C handle.
            #[must_use]
            pub const fn as_raw(self) -> *mut c_void {
                self.0
            }

            /// Returns `true` when the handle is null.
            #[must_use]
            pub const fn is_null(self) -> bool {
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
    /// Opaque platform handle (`cl_platform_id`).
    PlatformHandle
);
handle!(
    /// Opaque device handle (`cl_device_id`).
    DeviceHandle
);
handle!(
    /// Opaque, reference-counted context handle (`cl_context`).
    ContextHandle
);
handle!(
    /// Opaque, reference-counted command-queue handle (`cl_command_queue`).
    CommandQueueHandle
);
handle!(
    /// Opaque, reference-counted memory-object handle (`cl_mem`).
    MemHandle
);
handle!(
    /// Opaque, reference-counted program handle (`cl_program`).
    ProgramHandle
);
handle!(
    /// Opaque, reference-counted kernel handle (`cl_kernel`).
    KernelHandle
);
handle!(
    /// Opaque, reference-counted event handle (`cl_event`).
    EventHandle
);

/// Writes `value`'s bytes into a caller buffer honoring the OpenCL
/// `param_value`/`param_value_size`/`param_value_size_ret` contract.
///
/// Returns [`SUCCESS`] when the query fits, [`INVALID_VALUE`] when the
/// caller's buffer is too small (as the spec requires), and always reports
/// the required size through `param_value_size_ret` when it is non-null.
///
/// # Safety
///
/// When non-null, `param_value` must point at `param_value_size` writable
/// bytes and `param_value_size_ret` at one writable `usize` — the OpenCL
/// caller contract of every `clGet*Info` entry point.
pub unsafe fn write_info(
    bytes: &[u8],
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    if !param_value_size_ret.is_null() {
        // SAFETY: the caller contract above guarantees a writable slot.
        unsafe { *param_value_size_ret = bytes.len() };
    }
    if param_value.is_null() {
        return SUCCESS;
    }
    if param_value_size < bytes.len() {
        return INVALID_VALUE;
    }
    // SAFETY: the size check above plus the caller's contract that
    // `param_value` points at `param_value_size` writable bytes.
    unsafe {
        core::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            param_value.cast::<u8>(),
            bytes.len(),
        );
    }
    SUCCESS
}

/// [`write_info`] for NUL-terminated strings, appending the terminator.
///
/// OpenCL string queries report the length *including* the trailing NUL.
///
/// # Safety
///
/// See [`write_info`].
pub unsafe fn write_info_str(
    value: &str,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let len = value.len() + 1;
    if !param_value_size_ret.is_null() {
        // SAFETY: the caller contract above guarantees a writable slot.
        unsafe { *param_value_size_ret = len };
    }
    if param_value.is_null() {
        return SUCCESS;
    }
    if param_value_size < len {
        return INVALID_VALUE;
    }
    // SAFETY: the size check plus the caller's writable-buffer contract;
    // `value.as_bytes()` contains no interior NUL by `str`'s invariant.
    unsafe {
        let dst = param_value.cast::<u8>();
        core::ptr::copy_nonoverlapping(value.as_ptr(), dst, value.len());
        *dst.add(value.len()) = 0;
    }
    SUCCESS
}

/// [`write_info`] for `u32` scalars.
///
/// # Safety
///
/// See [`write_info`].
pub unsafe fn write_info_u32(
    value: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    // SAFETY: forwarded unchanged to `write_info`, whose contract this
    // function inherits.
    unsafe { write_info(&value.to_ne_bytes(), param_value_size, param_value, param_value_size_ret) }
}

/// [`write_info`] for `u64` scalars.
///
/// # Safety
///
/// See [`write_info`].
pub unsafe fn write_info_u64(
    value: u64,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    // SAFETY: forwarded unchanged to `write_info`, whose contract this
    // function inherits.
    unsafe { write_info(&value.to_ne_bytes(), param_value_size, param_value, param_value_size_ret) }
}

/// [`write_info`] for `usize` scalars (handle-sized quantities).
///
/// # Safety
///
/// See [`write_info`].
pub unsafe fn write_info_usize(
    value: usize,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    // SAFETY: forwarded unchanged to `write_info`, whose contract this
    // function inherits.
    unsafe { write_info(&value.to_ne_bytes(), param_value_size, param_value, param_value_size_ret) }
}

/// [`write_info`] for a NUL-terminated `c_char` C string.
///
/// # Safety
///
/// `value` must be a valid NUL-terminated C string (every embedded byte
/// before the terminator is non-zero).
pub unsafe fn write_info_cstr(
    value: *const c_char,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let mut len = 1usize;
    // SAFETY: the caller guarantees `value` is NUL-terminated, so this walk
    // terminates before reading unmapped memory.
    while unsafe { *value.add(len - 1) } != 0 {
        len += 1;
    }
    if !param_value_size_ret.is_null() {
        // SAFETY: the caller contractually provided a writable slot.
        unsafe { *param_value_size_ret = len };
    }
    if param_value.is_null() {
        return SUCCESS;
    }
    if param_value_size < len {
        return INVALID_VALUE;
    }
    // SAFETY: the size check plus the caller's writable-buffer contract and
    // the NUL-termination guarantee that makes `len` in bounds.
    unsafe {
        core::ptr::copy_nonoverlapping(value.cast::<u8>(), param_value.cast::<u8>(), len);
    }
    SUCCESS
}
