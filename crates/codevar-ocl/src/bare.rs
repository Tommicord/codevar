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

//! The built-in fallback driver: `codevar-ocl-bare` behind the host API.
//!
//! A machine without `libOpenCL.so` (or `OpenCL.dll`) has no vendor
//! driver to talk to, but it still has a CPU. The bare driver of
//! [`codevar_ocl_bare`] is exactly that: a statically linked OpenCL
//! implementation with a CPU back end, an embedded-profile SPIR-V
//! interpreter and no runtime dependency whatsoever.
//!
//! This module publishes that driver as an ordinary [`Platform`] of this
//! crate, so the whole safe API — contexts, buffers, queues, kernels —
//! works unchanged on a host without any OpenCL installation.
//!
//! # How the two crates meet
//!
//! The bare driver exports the `cl*` entry points as ordinary functions
//! (see [`codevar_ocl_bare::api`]); this crate resolves them into an
//! [`Api`] the same way it resolves the symbols of a dynamic library —
//! into the `extern "system"` entry-point types of [`crate::sys`]. The
//! two crates describe their handles as `#[repr(transparent)]` newtypes
//! over `*mut c_void`, so the machine level signatures are identical —
//! the adapters below only exist for the handful of entry points whose
//! *Rust* signatures differ (the platform and device counters are
//! `size_t` in the bare driver and `cl_uint` in the OpenCL headers, and
//! the notification callbacks are typed differently).
//!
//! # Limits of the fallback
//!
//! The embedded profile is smaller than the full one: no online compiler
//! (programs are built from SPIR-V), no out-of-order queues, no event
//! profiling and no images. Those limits surface as ordinary OpenCL
//! status codes, not as missing entry points.

use alloc::boxed::Box;
use alloc::sync::Arc;
use core::ffi::{c_char, c_void};
use core::mem::size_of;

use codevar_logger::log_debug;
use codevar_ocl_bare::api as bare_api;
use codevar_ocl_bare::sys as bare_sys;
use codevar_ocl_bare::{Device as BareDevice, DeviceInfo, PlatformConfig, SmpDevice};

use crate::api::Api;
use crate::error::{Error, Result};
use crate::platform::Platform;
use crate::sys;

/// `CL_PLATFORM_NAME` of the built-in fallback platform.
const PLATFORM_NAME: &str = "Codevar bare OpenCL";

/// `CL_PLATFORM_VENDOR` of the built-in fallback platform.
const PLATFORM_VENDOR: &str = "Codevar";

/// `CL_DEVICE_NAME` of the built-in fallback device.
const DEVICE_NAME: &str = "codevar-cpu";

/// The description of the fallback CPU device.
///
/// The device is a single compute unit: the claim loop of the bare SMP
/// back end runs on the calling thread, so the host needs no worker
/// threads and the driver stays `no_std`.
static DEVICE_INFO: DeviceInfo = DeviceInfo::new(
    DEVICE_NAME,
    PLATFORM_VENDOR,
    bare_sys::DEVICE_TYPE_CPU,
    1,
    256,
    256 * 1024 * 1024,
    64 * 1024,
    1024,
);

/// Casts a bare entry point to the signature of this crate.
///
/// Both crates describe their handles as `#[repr(transparent)]` newtypes
/// over `*mut c_void` and call the entry points with the C calling
/// convention, so the machine level signatures of the two are identical;
/// only the Rust types differ.
macro_rules! entry {
    ($function:path, $ty:ty) => {{
        debug_assert_eq!(size_of::<$ty>(), size_of::<*const ()>());
        // SAFETY: `$ty` is a function pointer type, so it is
        // pointer-sized (checked above), and `$function` is an entry
        // point of the bare driver whose machine level signature is the
        // one `$ty` describes. The bit pattern of the function address
        // is therefore preserved unchanged.
        unsafe { core::mem::transmute::<*const (), $ty>($function as *const ()) }
    }};
}

/// Publishes the bare driver's platform and returns it as a [`Platform`].
///
/// The driver is initialized once per process: [`codevar_ocl_bare::init`]
/// keeps the first platform, so a later call observes the same platform
/// even if the application registered its own devices earlier.
///
/// # Errors
///
/// Returns [`Error::NoPlatforms`] when the bare driver could not be
/// initialized (an allocation failure) or does not expose a platform.
pub(crate) fn platform() -> Result<Platform> {
    let api = Arc::new(api());
    // The bare driver publishes exactly one platform; enumerating it
    // through the ordinary entry points keeps the discovery path
    // identical to the one used for a dynamic driver.
    let platforms = crate::icd::query_platforms(&api)?;
    let platform = platforms
        .into_iter()
        .next()
        .ok_or(Error::NoPlatforms)?;
    log_debug!("using the built-in bare driver: platform {PLATFORM_NAME}, device {DEVICE_NAME}");
    Ok(platform)
}

/// Initializes the bare driver (once) and returns its platform handle.
fn init() -> Result<()> {
    static INIT: spin::Once<core::result::Result<(), codevar_ocl_bare::Error>> = spin::Once::new();
    let result = INIT.call_once(|| {
        let device = SmpDevice::new(DEVICE_INFO.clone(), 1)?;
        // The platform keeps the device list for as long as the platform
        // lives, so the list is leaked together with the device.
        let devices: &'static [&'static dyn BareDevice] = Box::leak(Box::new([device as &dyn BareDevice]));
        codevar_ocl_bare::init(PlatformConfig {
            name: PLATFORM_NAME,
            vendor: PLATFORM_VENDOR,
            modules: &[],
            devices,
        })
    });
    result
        .as_ref()
        .map_err(|error| {
            log_debug!("the built-in bare driver could not be initialized: {error}");
            Error::NoPlatforms
        })
        .copied()
}

/// Builds the [`Api`] of the bare driver.
fn api() -> Api {
    // The bare driver must be initialized before its entry points are
    // used; a failure here leaves the driver reporting
    // `CL_PLATFORM_NOT_FOUND_KHR`, which the discovery path turns into
    // "no platform".
    let _ = init();
    Api {
        get_platform_ids,
        get_platform_info: entry!(bare_api::clGetPlatformInfo, sys::GetPlatformInfo),
        get_device_ids,
        get_device_info: entry!(bare_api::clGetDeviceInfo, sys::GetDeviceInfo),
        create_context,
        release_context: entry!(bare_api::clReleaseContext, sys::ReleaseContext),
        create_command_queue: entry!(bare_api::clCreateCommandQueue, sys::CreateCommandQueue),
        release_command_queue: entry!(bare_api::clReleaseCommandQueue, sys::ReleaseCommandQueue),
        flush: entry!(bare_api::clFlush, sys::Flush),
        finish: entry!(bare_api::clFinish, sys::Finish),
        create_buffer: entry!(bare_api::clCreateBuffer, sys::CreateBuffer),
        release_mem_object: entry!(bare_api::clReleaseMemObject, sys::ReleaseMemObject),
        create_program_with_source: entry!(bare_api::clCreateProgramWithSource, sys::CreateProgramWithSource),
        release_program: entry!(bare_api::clReleaseProgram, sys::ReleaseProgram),
        build_program,
        get_program_info: entry!(bare_api::clGetProgramInfo, sys::GetProgramInfo),
        get_program_build_info: entry!(bare_api::clGetProgramBuildInfo, sys::GetProgramBuildInfo),
        create_kernel: entry!(bare_api::clCreateKernel, sys::CreateKernel),
        release_kernel: entry!(bare_api::clReleaseKernel, sys::ReleaseKernel),
        set_kernel_arg: entry!(bare_api::clSetKernelArg, sys::SetKernelArg),
        get_kernel_info: entry!(bare_api::clGetKernelInfo, sys::GetKernelInfo),
        get_kernel_work_group_info: entry!(bare_api::clGetKernelWorkGroupInfo, sys::GetKernelWorkGroupInfo),
        enqueue_read_buffer: entry!(bare_api::clEnqueueReadBuffer, sys::EnqueueReadBuffer),
        enqueue_write_buffer: entry!(bare_api::clEnqueueWriteBuffer, sys::EnqueueWriteBuffer),
        enqueue_copy_buffer: entry!(bare_api::clEnqueueCopyBuffer, sys::EnqueueCopyBuffer),
        enqueue_nd_range_kernel: entry!(bare_api::clEnqueueNDRangeKernel, sys::EnqueueNdRangeKernel),
        wait_for_events: entry!(bare_api::clWaitForEvents, sys::WaitForEvents),
        release_event: entry!(bare_api::clReleaseEvent, sys::ReleaseEvent),
        get_event_info: entry!(bare_api::clGetEventInfo, sys::GetEventInfo),
        get_event_profiling_info: entry!(bare_api::clGetEventProfilingInfo, sys::GetEventProfilingInfo),
        create_program_with_il: Some(entry!(bare_api::clCreateProgramWithIL, sys::CreateProgramWithIL)),
        // The embedded profile has neither a fill command nor extension
        // entry points.
        enqueue_fill_buffer: None,
        get_extension_function_address: None,
    }
}

/// `clGetPlatformIDs` — the bare driver counts platforms in a `size_t`,
/// the OpenCL headers in a `cl_uint`.
unsafe extern "system" fn get_platform_ids(
    num_entries: u32,
    platforms: *mut sys::PlatformHandle,
    num_platforms: *mut u32,
) -> i32 {
    let mut count = 0usize;
    let slot = if num_platforms.is_null() {
        core::ptr::null_mut()
    } else {
        &raw mut count
    };
    // SAFETY: `platforms` is the caller's array of `num_entries` handles
    // and `slot` a writable `size_t`; the bare driver only writes what
    // the OpenCL contract allows.
    let code = unsafe { bare_api::clGetPlatformIDs(num_entries, platforms.cast(), slot) };
    if !num_platforms.is_null() && (code == sys::SUCCESS || code == bare_sys::PLATFORM_NOT_FOUND_KHR) {
        // SAFETY: the caller provided a writable `cl_uint` slot.
        unsafe { *num_platforms = count.min(u32::MAX as usize) as u32 };
    }
    code
}

/// `clGetDeviceIDs` — the bare driver counts devices in a `size_t`, the
/// OpenCL headers in a `cl_uint`.
unsafe extern "system" fn get_device_ids(
    platform: sys::PlatformHandle,
    device_type: u64,
    num_entries: u32,
    devices: *mut sys::DeviceHandle,
    num_devices: *mut u32,
) -> i32 {
    let mut count = 0usize;
    let slot = if num_devices.is_null() {
        core::ptr::null_mut()
    } else {
        &raw mut count
    };
    // SAFETY: `devices` is the caller's array of `num_entries` handles
    // and `slot` a writable `size_t`; the platform handle is the one the
    // bare driver published.
    let code = unsafe {
        bare_api::clGetDeviceIDs(
            bare_sys::PlatformHandle::from_raw(platform.as_raw()),
            device_type,
            num_entries,
            devices.cast(),
            slot,
        )
    };
    if !num_devices.is_null() && (code == sys::SUCCESS || code == sys::DEVICE_NOT_FOUND) {
        // SAFETY: the caller provided a writable `cl_uint` slot.
        unsafe { *num_devices = count.min(u32::MAX as usize) as u32 };
    }
    code
}

/// `clCreateContext` — the bare driver takes the notification callback
/// as an opaque pointer.
unsafe extern "system" fn create_context(
    properties: *const isize,
    num_devices: u32,
    devices: *const sys::DeviceHandle,
    pfn_notify: Option<sys::ContextNotify>,
    user_data: *mut c_void,
    errcode_ret: *mut i32,
) -> sys::ContextHandle {
    let notify = pfn_notify.map_or(core::ptr::null(), |callback| callback as *const c_void);
    // SAFETY: `devices` is the caller's array of `num_devices` handles,
    // `notify` is the callback the caller passed (the bare driver never
    // invokes it) and `errcode_ret` is a valid out-pointer.
    let handle = unsafe {
        bare_api::clCreateContext(
            properties,
            num_devices,
            devices.cast(),
            notify,
            user_data,
            errcode_ret,
        )
    };
    sys::ContextHandle::from_raw(handle.as_raw())
}

/// `clBuildProgram` — the bare driver takes the notification callback as
/// an opaque pointer.
unsafe extern "system" fn build_program(
    program: sys::ProgramHandle,
    num_devices: u32,
    device_list: *const sys::DeviceHandle,
    options: *const c_char,
    pfn_notify: Option<sys::ProgramNotify>,
    user_data: *mut c_void,
) -> i32 {
    let notify = pfn_notify.map_or(core::ptr::null(), |callback| callback as *const c_void);
    // SAFETY: `device_list` is the caller's array of `num_devices` handles,
    // `options` is the caller's NUL-terminated string and `notify` is the
    // callback the caller passed (the bare driver never invokes it).
    unsafe {
        bare_api::clBuildProgram(
            bare_sys::ProgramHandle::from_raw(program.as_raw()),
            num_devices,
            device_list.cast(),
            options,
            notify,
            user_data,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::{Arg, Buffer, CommandQueue, Context, Program};

    /// The kernel of the end-to-end fallback test: the dialect form of a
    /// vector add, compiled to SPIR-V by the host front end.
    const KERNEL: &str = "\
#[kernel]
fn vector_add(a: *const int, b: *const int, out: *mut int, n: int) -> void {
    let i = get_global_id(0);
    if i < n {
        out[i] = a[i] + b[i];
    }
}
";

    /// Elements of the vector add.
    const SIZE: usize = 64;

    /// The fallback must always publish a platform with one device: the
    /// whole point of the module is that a host without an OpenCL
    /// installation still has a usable device.
    #[test]
    fn the_fallback_publishes_a_platform_with_a_device() {
        let platform = platform().expect("the bare driver publishes a platform");
        assert_eq!(platform.name().ok().as_deref(), Some(PLATFORM_NAME));
        let devices = platform
            .all_devices()
            .expect("device enumeration");
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name().ok().as_deref(), Some(DEVICE_NAME));
    }

    /// The whole safe API must work on the fallback device: a program
    /// built from SPIR-V, a launch and the result read back.
    #[test]
    fn the_fallback_runs_a_kernel() {
        let platform = platform().expect("the bare driver publishes a platform");
        let device = platform
            .all_devices()
            .expect("device enumeration")
            .into_iter()
            .next()
            .expect("the fallback has one device");
        let context = Context::new(&device).expect("context");
        let queue = CommandQueue::new(&context, &device).expect("queue");

        let a: Vec<i32> = (0..SIZE as i32).collect();
        let b: Vec<i32> = vec![7; SIZE];
        let expected: Vec<i32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

        let a_buffer = Buffer::from_slice(&context, &a).expect("input buffer");
        let b_buffer = Buffer::from_slice(&context, &b).expect("input buffer");
        let out_buffer = Buffer::new(&context, SIZE * core::mem::size_of::<i32>()).expect("output buffer");

        let program = Program::from_source(&context, KERNEL).expect("the dialect kernel compiles");
        program.build("").expect("build");
        let mut kernel = program.kernel("vector_add").expect("kernel");
        kernel
            .set_arg(0, Arg::Buffer(&a_buffer))
            .expect("arg 0");
        kernel
            .set_arg(1, Arg::Buffer(&b_buffer))
            .expect("arg 1");
        kernel
            .set_arg(2, Arg::Buffer(&out_buffer))
            .expect("arg 2");
        kernel
            .set_arg(3, Arg::I32(SIZE as i32))
            .expect("arg 3");

        let event = queue
            .enqueue_kernel(&kernel, &[SIZE], None)
            .expect("launch");
        event.wait().expect("the launch completes");

        let mut out = vec![0i32; SIZE];
        let _read = queue
            .read_buffer(&out_buffer, 0, &mut out)
            .expect("read back");
        assert_eq!(out, expected);
    }

    /// The device counters of the two crates disagree on the width of
    /// the count slot; the adapter must translate, not truncate.
    #[test]
    fn the_platform_and_device_counts_survive_the_translation() {
        init().expect("the bare driver initializes");
        let mut platforms = 0u32;
        // SAFETY: the count slot is a valid out-pointer.
        let code = unsafe { get_platform_ids(0, core::ptr::null_mut(), &mut platforms) };
        assert_eq!(code, sys::SUCCESS);
        assert_eq!(platforms, 1);

        let mut devices = 0u32;
        // SAFETY: the count slot is a valid out-pointer.
        let code = unsafe {
            get_device_ids(
                sys::PlatformHandle::from_raw(core::ptr::null_mut()),
                sys::DEVICE_TYPE_ALL,
                0,
                core::ptr::null_mut(),
                &mut devices,
            )
        };
        assert!(code == sys::SUCCESS || code == sys::INVALID_PLATFORM);
    }
}
