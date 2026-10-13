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

//! The `cl*` entry points — the C boundary of the driver.
//!
//! Every function here is an `#[unsafe(no_mangle)] extern "C"` symbol the
//! application links against directly (no ICD loader, no `dlopen`). The
//! layer does exactly three things:
//!
//! 1. validate handles and arguments per the OpenCL spec, mapping each
//!    failure to its precise `CL_*` code (the [`crate::Error`] mapping
//!    covers driver-internal failures only);
//! 2. marshal buffers, launches and events onto the object layer;
//! 3. execute buffer commands inline (a single address space makes them
//!    `memcpy`s) and hand kernel launches to the queue's device.
//!
//! Error-code policy: validation that the spec distinguishes (work
//! dimension vs. global size vs. work-group size, argument index vs.
//! size) is checked here so each case reports its own `CL_*` value;
//! everything else funnels through [`crate::Error::to_cl`].
//!
//! The embedded profile executes commands in order: buffer commands run
//! inline before the call returns, and a kernel launch completes before
//! `Device::submit` returns on synchronous back ends (AMP/accelerator
//! launches may still be in flight — their events drive completion).

#![allow(non_snake_case)]
use core::ffi::{c_char, c_void};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::Error;
use crate::exec::Launch;
use crate::objects::{
    self, Context, DeviceObject, Event, Kernel, Mem, Program, Queue, RefCounted, release_ref,
};
use crate::platform::Platform;
use crate::registry::KernelDef;
use crate::sys::{
    self, CommandQueueHandle, ContextHandle, DeviceHandle, EventHandle, KernelHandle, MemHandle,
    PlatformHandle, ProgramHandle,
};

/// Stores `code` through a caller's `errcode_ret` slot when non-null.
fn set_err(errcode_ret: *mut i32, code: i32) {
    if !errcode_ret.is_null() {
        // SAFETY: the OpenCL caller contract of every `clCreate*`/`clBuild*`
        // entry point: a non-null `errcode_ret` points at one writable
        // `cl_int`.
        unsafe {
            *errcode_ret = code;
        }
    }
}

/// Wraps a driver object's address as a C handle.
fn handle_ptr<T>(object: &T) -> *mut c_void {
    core::ptr::from_ref(object).cast_mut().cast()
}

/// Reads a NUL-terminated C string.
///
/// # Safety
///
/// `pointer` must be a valid NUL-terminated string.
unsafe fn cstr<'a>(pointer: *const c_char) -> Result<&'a str, Error> {
    if pointer.is_null() {
        return Err(Error::InvalidArgument);
    }
    let mut len = 0usize;
    // SAFETY: the caller guarantees a NUL terminator exists, so the walk
    // stops inside the string.
    while unsafe { *pointer.add(len) } != 0 {
        len += 1;
    }
    // SAFETY: `len` bytes precede the terminator the walk just found.
    let bytes = unsafe { core::slice::from_raw_parts(pointer.cast::<u8>(), len) };
    core::str::from_utf8(bytes).map_err(|_| Error::InvalidArgument)
}

/// Validates the `num_events_in_wait_list`/`event_wait_list` pair and the
/// handles in it (`CL_INVALID_EVENT_WAIT_LIST`), then waits for every
/// event, propagating a failed command as
/// `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`.
fn wait_list(num_events: u32, event_wait_list: *const EventHandle) -> Result<(), i32> {
    if num_events == 0 {
        return if event_wait_list.is_null() {
            Ok(())
        } else {
            Err(sys::INVALID_EVENT_WAIT_LIST)
        };
    }
    if event_wait_list.is_null() {
        return Err(sys::INVALID_EVENT_WAIT_LIST);
    }
    for index in 0..num_events as usize {
        // SAFETY: the caller contract — `event_wait_list` points at
        // `num_events` readable `cl_event` slots.
        let handle = unsafe { *event_wait_list.add(index) };
        let event = objects::event(handle).map_err(|_| sys::INVALID_EVENT_WAIT_LIST)?;
        event.wait().map_err(Error::to_cl)?;
        if event.status() < 0 {
            return Err(sys::EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        }
    }
    Ok(())
}

/// Validates a context property list: null (no properties), or key/value
/// `isize` pairs terminated by zero. Only `CL_CONTEXT_PLATFORM` is known,
/// and its value must be this platform.
fn context_platform(properties: *const isize) -> Result<&'static Platform, i32> {
    let platform = crate::platform::state().map_err(Error::to_cl)?;
    if properties.is_null() {
        return Ok(platform);
    }
    let mut cursor = properties;
    loop {
        // SAFETY: OpenCL property lists are NUL/zero-terminated key-value
        // `intptr_t` pairs; the terminator check bounds every read.
        let key = unsafe { *cursor };
        if key == 0 {
            break;
        }
        // SAFETY: a key implies its value slot precedes the terminator.
        let value = unsafe { *cursor.add(1) };
        if key == sys::CONTEXT_PLATFORM {
            let raw = value as *const Platform;
            if raw.is_null() || !platform.matches(raw) {
                return Err(sys::INVALID_PLATFORM);
            }
        } else {
            return Err(sys::INVALID_PROPERTY);
        }
        // SAFETY: the pair read above advances past both slots.
        cursor = unsafe { cursor.add(2) };
    }
    Ok(platform)
}

/// Whether `device` is listed in `context`.
fn in_context(context: &Context, device: &DeviceObject) -> bool {
    context
        .devices()
        .iter()
        .any(|entry| core::ptr::eq(*entry, device))
}

/// The bytes of a `cl_device_id` array for an info query.
fn device_array(devices: &[&'static DeviceObject]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for device in devices {
        bytes.extend_from_slice(&(handle_ptr(*device) as usize).to_ne_bytes());
    }
    bytes
}

/// `clGetPlatformIDs` — reports the one platform of the image, or
/// `CL_PLATFORM_NOT_FOUND_KHR` before [`crate::init`].
///
/// # Safety
///
/// When non-null, `platforms` points at `num_entries` writable
/// `cl_platform_id` slots and `num_platforms` at one writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetPlatformIDs(
    num_entries: u32,
    platforms: *mut PlatformHandle,
    num_platforms: *mut usize,
) -> i32 {
    if num_platforms.is_null() {
        return sys::INVALID_VALUE;
    }
    let platform = match crate::platform::state() {
        Ok(platform) => platform,
        Err(error) => {
            // SAFETY: caller contract guarantees a writable slot.
            unsafe {
                *num_platforms = 0;
            }
            return error.to_cl();
        }
    };
    if !platforms.is_null() && num_entries == 0 {
        return sys::INVALID_VALUE;
    }
    // SAFETY: caller contract guarantees a writable slot.
    unsafe {
        *num_platforms = 1;
    }
    if !platforms.is_null() {
        // SAFETY: caller contract — room for `num_entries >= 1` handles.
        unsafe {
            *platforms = PlatformHandle::from_raw(handle_ptr(platform));
        }
    }
    sys::SUCCESS
}

/// `clGetPlatformInfo` — the platform's profile/version/name/vendor/
/// extensions strings.
///
/// # Safety
///
/// The `clGet*Info` caller contract for the out parameters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetPlatformInfo(
    platform: PlatformHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let raw = platform.as_raw().cast::<Platform>();
    if raw.is_null() {
        return sys::INVALID_PLATFORM;
    }
    let state = match crate::platform::state() {
        Ok(state) => state,
        Err(error) => return error.to_cl(),
    };
    if !state.matches(raw) {
        return sys::INVALID_PLATFORM;
    }
    let value = match param_name {
        sys::PLATFORM_PROFILE => state.profile(),
        sys::PLATFORM_VERSION => state.version(),
        sys::PLATFORM_NAME => state.name(),
        sys::PLATFORM_VENDOR => state.vendor(),
        sys::PLATFORM_EXTENSIONS => "",
        _ => return sys::INVALID_VALUE,
    };
    // SAFETY: forwarded to `write_info_str`, whose contract the entry-point
    // caller satisfies.
    unsafe { sys::write_info_str(value, param_value_size, param_value, param_value_size_ret) }
}

/// Whether `info` matches the requested `device_type` filter; `is_first`
/// marks the platform's default device for `CL_DEVICE_TYPE_DEFAULT`.
fn device_type_matches(info: &crate::DeviceInfo, requested: u64, is_first: bool) -> bool {
    if requested == sys::DEVICE_TYPE_ALL {
        return true;
    }
    if requested & sys::DEVICE_TYPE_DEFAULT != 0 && is_first {
        return true;
    }
    requested & info.device_type != 0
}

/// `clGetDeviceIDs` — the platform devices matching a type filter.
///
/// # Safety
///
/// When non-null, `devices` points at `num_entries` writable
/// `cl_device_id` slots and `num_devices` at one writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetDeviceIDs(
    platform: PlatformHandle,
    device_type: u64,
    num_entries: u32,
    devices: *mut DeviceHandle,
    num_devices: *mut usize,
) -> i32 {
    if num_devices.is_null() {
        return sys::INVALID_VALUE;
    }
    if device_type & !sys::DEVICE_TYPE_ALL != 0 {
        return sys::INVALID_DEVICE_TYPE;
    }
    if !devices.is_null() && num_entries == 0 {
        return sys::INVALID_VALUE;
    }
    let raw = platform.as_raw().cast::<Platform>();
    if raw.is_null() {
        return sys::INVALID_PLATFORM;
    }
    let state = match crate::platform::state() {
        Ok(state) => state,
        Err(error) => {
            // SAFETY: caller contract guarantees a writable slot.
            unsafe {
                *num_devices = 0;
            }
            return error.to_cl();
        }
    };
    if !state.matches(raw) {
        // SAFETY: caller contract guarantees a writable slot.
        unsafe {
            *num_devices = 0;
        }
        return sys::INVALID_PLATFORM;
    }
    let mut count = 0usize;
    for (index, device) in state.devices().iter().enumerate() {
        if !device_type_matches(device.device().info(), device_type, index == 0) {
            continue;
        }
        if !devices.is_null() && (count as u32) < num_entries {
            // SAFETY: caller contract — room for `num_entries` handles.
            unsafe {
                *devices.add(count) = DeviceHandle::from_raw(handle_ptr(*device));
            }
        }
        count += 1;
    }
    // SAFETY: caller contract guarantees a writable slot.
    unsafe {
        *num_devices = count;
    }
    if count == 0 {
        return sys::DEVICE_NOT_FOUND;
    }
    sys::SUCCESS
}

/// `clGetDeviceInfo` — the static description of a device.
///
/// # Safety
///
/// The `clGet*Info` caller contract for the out parameters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetDeviceInfo(
    device: DeviceHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let object = match objects::device_object(device) {
        Ok(object) => object,
        Err(error) => return error.to_cl(),
    };
    let info = object.device().info();
    // SAFETY: the `clGet*Info` caller contract — when `param_value` is
    // non-null it points at `param_value_size` writable bytes and
    // `param_value_size_ret` at one writable `usize`.
    unsafe {
        match param_name {
            sys::DEVICE_TYPE => sys::write_info_u64(
                info.device_type,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_VENDOR_ID => sys::write_info_u32(
                info.vendor_id,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_MAX_COMPUTE_UNITS => sys::write_info_u32(
                info.compute_units,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_MAX_WORK_ITEM_DIMENSIONS => {
                sys::write_info_u32(3, param_value_size, param_value, param_value_size_ret)
            }
            sys::DEVICE_MAX_WORK_GROUP_SIZE => sys::write_info_u32(
                info.max_work_group_size,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_MAX_WORK_ITEM_SIZES => sys::write_info(
                &info
                    .max_work_item_sizes
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect::<Vec<u8>>(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_ADDRESS_BITS => sys::write_info_u32(
                info.address_bits,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_MAX_MEM_ALLOC_SIZE => sys::write_info_u64(
                info.max_mem_alloc_size,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_GLOBAL_MEM_SIZE => sys::write_info_u64(
                info.global_mem_size,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_LOCAL_MEM_SIZE => sys::write_info_u64(
                info.local_mem_size,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_MAX_PARAMETER_SIZE => sys::write_info_u32(
                info.max_parameter_size,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_IMAGE_SUPPORT => {
                sys::write_info_u32(sys::FALSE, param_value_size, param_value, param_value_size_ret)
            }
            sys::DEVICE_AVAILABLE => {
                sys::write_info_u32(sys::TRUE, param_value_size, param_value, param_value_size_ret)
            }
            sys::DEVICE_COMPILER_AVAILABLE => {
                sys::write_info_u32(sys::FALSE, param_value_size, param_value, param_value_size_ret)
            }
            sys::DEVICE_HOST_UNIFIED_MEMORY => {
                sys::write_info_u32(sys::TRUE, param_value_size, param_value, param_value_size_ret)
            }
            sys::DEVICE_NAME => {
                sys::write_info_str(info.name, param_value_size, param_value, param_value_size_ret)
            }
            sys::DEVICE_VENDOR => {
                sys::write_info_str(info.vendor, param_value_size, param_value, param_value_size_ret)
            }
            sys::DRIVER_VERSION => sys::write_info_str(
                info.driver_version,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_PROFILE => {
                sys::write_info_str(info.profile, param_value_size, param_value, param_value_size_ret)
            }
            sys::DEVICE_VERSION => {
                sys::write_info_str(info.version, param_value_size, param_value, param_value_size_ret)
            }
            sys::DEVICE_EXTENSIONS => sys::write_info_str(
                info.extensions,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::DEVICE_PLATFORM => {
                let platform = match crate::platform::state() {
                    Ok(platform) => platform,
                    Err(error) => return error.to_cl(),
                };
                sys::write_info_usize(
                    core::ptr::from_ref(platform) as usize,
                    param_value_size,
                    param_value,
                    param_value_size_ret,
                )
            }
            sys::DEVICE_BUILTIN_KERNELS => {
                let names = match crate::platform::state() {
                    Ok(platform) => platform.registry().names_joined(),
                    Err(error) => return error.to_cl(),
                };
                sys::write_info_str(&names, param_value_size, param_value, param_value_size_ret)
            }
            _ => sys::INVALID_VALUE,
        }
    }
}

/// `clCreateContext` — a context over `num_devices` platform devices.
///
/// `pfn_notify`/`user_data` are accepted and ignored: creation is
/// synchronous, so the callback would have nothing to report.
///
/// # Safety
///
/// `properties` is null or a zero-terminated key/value list; `devices`
/// points at `num_devices` valid `cl_device_id` handles; `errcode_ret` is
/// null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clCreateContext(
    properties: *const isize,
    num_devices: u32,
    devices: *const DeviceHandle,
    _pfn_notify: *const c_void,
    _user_data: *mut c_void,
    errcode_ret: *mut i32,
) -> ContextHandle {
    let fail = |code: i32| {
        set_err(errcode_ret, code);
        ContextHandle::from_raw(core::ptr::null_mut())
    };
    if num_devices == 0 || devices.is_null() {
        return fail(sys::INVALID_VALUE);
    }
    let platform = match context_platform(properties) {
        Ok(platform) => platform,
        Err(code) => return fail(code),
    };
    let mut list: Vec<&'static DeviceObject> = Vec::new();
    for index in 0..num_devices as usize {
        // SAFETY: caller contract — `devices` holds `num_devices` handles.
        let handle = unsafe { *devices.add(index) };
        let device = match objects::device_object(handle) {
            Ok(device) => device,
            Err(_) => return fail(sys::INVALID_DEVICE),
        };
        if !platform
            .devices()
            .iter()
            .any(|entry| core::ptr::eq(*entry, device))
        {
            return fail(sys::INVALID_DEVICE);
        }
        list.push(device);
    }
    let context: &'static Context = Box::leak(Context::new(list));
    set_err(errcode_ret, sys::SUCCESS);
    ContextHandle::from_raw(handle_ptr(context))
}

/// `clRetainContext`.
///
/// # Safety
///
/// `context` must be null or a handle produced by this driver.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clRetainContext(context: ContextHandle) -> i32 {
    match objects::context(context) {
        Ok(object) => {
            object.header().retain();
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clReleaseContext`.
///
/// # Safety
///
/// `context` must be a handle produced by this driver and released no more
/// times than it was retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clReleaseContext(context: ContextHandle) -> i32 {
    match objects::context(context) {
        Ok(object) => {
            release_ref(object);
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clGetContextInfo` — reference count, device list, device count or
/// (empty) property list.
///
/// # Safety
///
/// The `clGet*Info` caller contract for the out parameters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetContextInfo(
    context: ContextHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let object = match objects::context(context) {
        Ok(object) => object,
        Err(error) => return error.to_cl(),
    };
    // SAFETY: the `clGet*Info` caller contract for the out parameters.
    unsafe {
        match param_name {
            sys::CONTEXT_REFERENCE_COUNT => sys::write_info_u32(
                object.header().refs(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::CONTEXT_NUM_DEVICES => sys::write_info_u32(
                object.num_devices() as u32,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::CONTEXT_DEVICES => {
                let bytes = device_array(object.devices());
                sys::write_info(&bytes, param_value_size, param_value, param_value_size_ret)
            }
            sys::CONTEXT_PROPERTIES => {
                let terminator = 0isize;
                sys::write_info(
                    &terminator.to_ne_bytes(),
                    param_value_size,
                    param_value,
                    param_value_size_ret,
                )
            }
            _ => sys::INVALID_VALUE,
        }
    }
}

/// `clCreateCommandQueue` — an in-order queue on `device`.
///
/// Out-of-order execution and profiling are not part of the embedded
/// profile: any property bit besides zero is `CL_INVALID_QUEUE_PROPERTIES`.
///
/// # Safety
///
/// `context`/`device` must be handles produced by this driver;
/// `errcode_ret` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clCreateCommandQueue(
    context: ContextHandle,
    device: DeviceHandle,
    properties: u64,
    errcode_ret: *mut i32,
) -> CommandQueueHandle {
    let fail = |code: i32| {
        set_err(errcode_ret, code);
        CommandQueueHandle::from_raw(core::ptr::null_mut())
    };
    if properties != 0 {
        return fail(sys::INVALID_QUEUE_PROPERTIES);
    }
    let context = match objects::context(context) {
        Ok(context) => context,
        Err(error) => return fail(error.to_cl()),
    };
    let device = match objects::device_object(device) {
        Ok(device) => device,
        Err(_) => return fail(sys::INVALID_DEVICE),
    };
    if !in_context(context, device) {
        return fail(sys::INVALID_DEVICE);
    }
    let queue: &'static Queue = Box::leak(Queue::new(context, device, properties));
    set_err(errcode_ret, sys::SUCCESS);
    CommandQueueHandle::from_raw(handle_ptr(queue))
}

/// `clRetainCommandQueue`.
///
/// # Safety
///
/// `command_queue` must be null or a handle produced by this driver.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clRetainCommandQueue(command_queue: CommandQueueHandle) -> i32 {
    match objects::queue(command_queue) {
        Ok(object) => {
            object.header().retain();
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clReleaseCommandQueue`.
///
/// # Safety
///
/// `command_queue` must be a handle produced by this driver and released
/// no more times than it was retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clReleaseCommandQueue(command_queue: CommandQueueHandle) -> i32 {
    match objects::queue(command_queue) {
        Ok(object) => {
            release_ref(object);
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clGetCommandQueueInfo` — context, device, reference count or
/// properties.
///
/// # Safety
///
/// The `clGet*Info` caller contract for the out parameters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetCommandQueueInfo(
    command_queue: CommandQueueHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let object = match objects::queue(command_queue) {
        Ok(object) => object,
        Err(error) => return error.to_cl(),
    };
    // SAFETY: the `clGet*Info` caller contract for the out parameters.
    unsafe {
        match param_name {
            sys::QUEUE_CONTEXT => sys::write_info_usize(
                handle_ptr(object.context()) as usize,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::QUEUE_DEVICE => sys::write_info_usize(
                handle_ptr(object.device_object()) as usize,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::QUEUE_REFERENCE_COUNT => sys::write_info_u32(
                object.header().refs(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::QUEUE_PROPERTIES => sys::write_info_u64(
                object.properties(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            _ => sys::INVALID_VALUE,
        }
    }
}

/// `clFlush` — polls asynchronous completions once.
///
/// # Safety
///
/// `command_queue` must be a handle produced by this driver.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clFlush(command_queue: CommandQueueHandle) -> i32 {
    match objects::queue(command_queue) {
        Ok(queue) => match queue.device().progress() {
            Ok(()) => sys::SUCCESS,
            Err(error) => error.to_cl(),
        },
        Err(error) => error.to_cl(),
    }
}

/// `clFinish` — blocks until every command of the queue has retired.
///
/// # Safety
///
/// `command_queue` must be a handle produced by this driver.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clFinish(command_queue: CommandQueueHandle) -> i32 {
    match objects::queue(command_queue) {
        Ok(queue) => match queue.device().wait_all() {
            Ok(()) => sys::SUCCESS,
            Err(error) => error.to_cl(),
        },
        Err(error) => error.to_cl(),
    }
}

/// `clCreateBuffer` — a buffer object in the context.
///
/// # Safety
///
/// `context` must be a handle produced by this driver; `host_ptr` obeys
/// the `CL_MEM_USE_HOST_PTR`/`CL_MEM_COPY_HOST_PTR` contract;
/// `errcode_ret` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clCreateBuffer(
    context: ContextHandle,
    flags: u64,
    size: usize,
    host_ptr: *mut c_void,
    errcode_ret: *mut i32,
) -> MemHandle {
    let fail = |code: i32| {
        set_err(errcode_ret, code);
        MemHandle::from_raw(core::ptr::null_mut())
    };
    let context = match objects::context(context) {
        Ok(context) => context,
        Err(error) => return fail(error.to_cl()),
    };
    match Mem::create(context, flags, size, host_ptr) {
        Ok(buffer) => {
            let buffer: &'static Mem = Box::leak(buffer);
            set_err(errcode_ret, sys::SUCCESS);
            MemHandle::from_raw(handle_ptr(buffer))
        }
        Err(error) => fail(error.to_cl()),
    }
}

/// `clRetainMemObject`.
///
/// # Safety
///
/// `buffer` must be null or a handle produced by this driver.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clRetainMemObject(buffer: MemHandle) -> i32 {
    match objects::mem(buffer) {
        Ok(object) => {
            object.header().retain();
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clReleaseMemObject`.
///
/// # Safety
///
/// `buffer` must be a handle produced by this driver and released no more
/// times than it was retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clReleaseMemObject(buffer: MemHandle) -> i32 {
    match objects::mem(buffer) {
        Ok(object) => {
            release_ref(object);
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clGetMemObjectInfo` — flags, size, context or reference count.
///
/// # Safety
///
/// The `clGet*Info` caller contract for the out parameters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetMemObjectInfo(
    buffer: MemHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let object = match objects::mem(buffer) {
        Ok(object) => object,
        Err(error) => return error.to_cl(),
    };
    // SAFETY: the `clGet*Info` caller contract for the out parameters.
    unsafe {
        match param_name {
            sys::MEM_FLAGS => sys::write_info_u64(
                object.flags(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::MEM_SIZE => {
                sys::write_info_usize(object.size(), param_value_size, param_value, param_value_size_ret)
            }
            sys::MEM_CONTEXT => sys::write_info_usize(
                handle_ptr(object.context()) as usize,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::MEM_REFERENCE_COUNT => sys::write_info_u32(
                object.header().refs(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            _ => sys::INVALID_VALUE,
        }
    }
}

/// `clCreateProgramWithSource` — always `CL_COMPILER_NOT_AVAILABLE`: the
/// embedded profile has no online compiler. The strings are validated but
/// not parsed — nothing would compile them.
///
/// # Safety
///
/// `context` must be a handle produced by this driver; the string array
/// obeys the OpenCL source-list contract; `errcode_ret` is null or
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clCreateProgramWithSource(
    context: ContextHandle,
    count: u32,
    strings: *const *const c_char,
    _lengths: *const usize,
    errcode_ret: *mut i32,
) -> ProgramHandle {
    let fail = |code: i32| {
        set_err(errcode_ret, code);
        ProgramHandle::from_raw(core::ptr::null_mut())
    };
    if let Err(error) = objects::context(context) {
        return fail(error.to_cl());
    }
    if count == 0 || strings.is_null() {
        return fail(sys::INVALID_VALUE);
    }
    fail(sys::COMPILER_NOT_AVAILABLE)
}

/// `clCreateProgramWithBuiltInKernels` — binds the named registry kernels
/// (`;`-separated) into a program.
///
/// # Safety
///
/// `context`/`device_list` must be handles produced by this driver;
/// `kernel_names` is a valid NUL-terminated `;`-separated list;
/// `errcode_ret` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clCreateProgramWithBuiltInKernels(
    context: ContextHandle,
    num_devices: u32,
    device_list: *const DeviceHandle,
    kernel_names: *const c_char,
    errcode_ret: *mut i32,
) -> ProgramHandle {
    let fail = |code: i32| {
        set_err(errcode_ret, code);
        ProgramHandle::from_raw(core::ptr::null_mut())
    };
    let context = match objects::context(context) {
        Ok(context) => context,
        Err(error) => return fail(error.to_cl()),
    };
    if num_devices == 0 || device_list.is_null() {
        return fail(sys::INVALID_VALUE);
    }
    for index in 0..num_devices as usize {
        // SAFETY: caller contract — `device_list` holds `num_devices`
        // handles.
        let handle = unsafe { *device_list.add(index) };
        match objects::device_object(handle) {
            Ok(device) if in_context(context, device) => {}
            _ => return fail(sys::INVALID_DEVICE),
        }
    }
    let names = match unsafe { cstr(kernel_names) } {
        Ok(names) => names,
        Err(error) => return fail(error.to_cl()),
    };
    let registry = match crate::platform::state() {
        Ok(platform) => platform.registry(),
        Err(error) => return fail(error.to_cl()),
    };
    let mut defs: Vec<KernelDef> = Vec::new();
    for name in names.split(';').filter(|part| !part.is_empty()) {
        match registry.find(name) {
            Some(def) => defs.push(def.clone()),
            None => return fail(sys::INVALID_KERNEL_NAME),
        }
    }
    if defs.is_empty() {
        return fail(sys::INVALID_VALUE);
    }
    let program: &'static Program = Box::leak(Program::new(context, defs));
    set_err(errcode_ret, sys::SUCCESS);
    ProgramHandle::from_raw(handle_ptr(program))
}

/// `clCreateProgramWithBinary` — loads a SPIR-V module as the program.
///
/// Only the first entry of `device_list`/`lengths`/`binaries` is used: a
/// program is created for the whole context.
///
/// # Safety
///
/// `context`/`device_list` must be handles produced by this driver;
/// `lengths`/`binaries` point at `num_devices` entries whose first binary
/// is `lengths[0]` readable bytes; `binary_status`/`errcode_ret` are null
/// or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clCreateProgramWithBinary(
    context: ContextHandle,
    num_devices: u32,
    device_list: *const DeviceHandle,
    lengths: *const usize,
    binaries: *const *const u8,
    binary_status: *mut i32,
    errcode_ret: *mut i32,
) -> ProgramHandle {
    let fail = |code: i32| {
        if !binary_status.is_null() {
            // SAFETY: caller contract — `binary_status` has `num_devices`
            // writable `cl_int` slots.
            unsafe {
                *binary_status = code;
            }
        }
        set_err(errcode_ret, code);
        ProgramHandle::from_raw(core::ptr::null_mut())
    };
    let context = match objects::context(context) {
        Ok(context) => context,
        Err(error) => return fail(error.to_cl()),
    };
    if num_devices == 0 || device_list.is_null() || lengths.is_null() || binaries.is_null() {
        return fail(sys::INVALID_VALUE);
    }
    for index in 0..num_devices as usize {
        // SAFETY: caller contract — `device_list` holds `num_devices`
        // handles.
        let handle = unsafe { *device_list.add(index) };
        match objects::device_object(handle) {
            Ok(device) if in_context(context, device) => {}
            _ => return fail(sys::INVALID_DEVICE),
        }
    }
    // SAFETY: caller contract — the first array entries are readable.
    let length = unsafe { *lengths };
    // SAFETY: caller contract — the first binary pointer has `length`
    // readable bytes.
    let bytes = unsafe { core::slice::from_raw_parts(*binaries, length) };
    match crate::spirv::load(bytes).and_then(|module| Program::from_spirv(context, module)) {
        Ok(program) => {
            let program: &'static Program = Box::leak(program);
            if !binary_status.is_null() {
                // SAFETY: caller contract — writable slot.
                unsafe {
                    *binary_status = sys::SUCCESS;
                }
            }
            set_err(errcode_ret, sys::SUCCESS);
            ProgramHandle::from_raw(handle_ptr(program))
        }
        Err(_) => fail(sys::INVALID_BINARY),
    }
}

/// `clBuildProgram` — a no-op that records the options: programs bind to
/// already-resolved definitions, so they are built at creation.
///
/// `pfn_notify`/`user_data` are accepted and ignored; the build completes
/// synchronously before the call returns.
///
/// # Safety
///
/// `program`/`device_list` must be handles produced by this driver;
/// `options` is null or a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clBuildProgram(
    program: ProgramHandle,
    num_devices: u32,
    device_list: *const DeviceHandle,
    options: *const c_char,
    _pfn_notify: *const c_void,
    _user_data: *mut c_void,
) -> i32 {
    let program = match objects::program(program) {
        Ok(program) => program,
        Err(error) => return error.to_cl(),
    };
    if num_devices > 0 {
        if device_list.is_null() {
            return sys::INVALID_VALUE;
        }
        for index in 0..num_devices as usize {
            // SAFETY: caller contract — `device_list` holds `num_devices`
            // handles.
            let handle = unsafe { *device_list.add(index) };
            match objects::device_object(handle) {
                Ok(device) if in_context(program.context(), device) => {}
                _ => return sys::INVALID_DEVICE,
            }
        }
    }
    let text = if options.is_null() {
        String::new()
    } else {
        match unsafe { cstr(options) } {
            Ok(text) => String::from(text),
            Err(error) => return error.to_cl(),
        }
    };
    program.set_options(text);
    sys::SUCCESS
}

/// `clRetainProgram`.
///
/// # Safety
///
/// `program` must be null or a handle produced by this driver.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clRetainProgram(program: ProgramHandle) -> i32 {
    match objects::program(program) {
        Ok(object) => {
            object.header().retain();
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clReleaseProgram`.
///
/// # Safety
///
/// `program` must be a handle produced by this driver and released no more
/// times than it was retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clReleaseProgram(program: ProgramHandle) -> i32 {
    match objects::program(program) {
        Ok(object) => {
            release_ref(object);
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clGetProgramInfo` — context, devices, kernel names, counts, source,
/// reference count or build fields.
///
/// # Safety
///
/// The `clGet*Info` caller contract for the out parameters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetProgramInfo(
    program: ProgramHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let object = match objects::program(program) {
        Ok(object) => object,
        Err(error) => return error.to_cl(),
    };
    // SAFETY: the `clGet*Info` caller contract for the out parameters.
    unsafe {
        match param_name {
            sys::PROGRAM_REFERENCE_COUNT => sys::write_info_u32(
                object.header().refs(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::PROGRAM_CONTEXT => sys::write_info_usize(
                handle_ptr(object.context()) as usize,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::PROGRAM_NUM_DEVICES => sys::write_info_u32(
                object.context().num_devices() as u32,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::PROGRAM_DEVICES => {
                let bytes = device_array(object.context().devices());
                sys::write_info(&bytes, param_value_size, param_value, param_value_size_ret)
            }
            sys::PROGRAM_SOURCE => {
                sys::write_info_str("", param_value_size, param_value, param_value_size_ret)
            }
            sys::PROGRAM_NUM_KERNELS => sys::write_info_usize(
                object.defs().len(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::PROGRAM_KERNEL_NAMES => {
                let mut names = String::new();
                for def in object.defs() {
                    if !names.is_empty() {
                        names.push(';');
                    }
                    names.push_str(&def.name);
                }
                sys::write_info_str(&names, param_value_size, param_value, param_value_size_ret)
            }
            sys::PROGRAM_BUILD_STATUS => sys::write_info_u32(
                sys::BUILD_SUCCESS as u32,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::PROGRAM_BUILD_OPTIONS => {
                let options = object.options();
                sys::write_info_str(&options, param_value_size, param_value, param_value_size_ret)
            }
            sys::PROGRAM_BUILD_LOG => {
                sys::write_info_str("", param_value_size, param_value, param_value_size_ret)
            }
            _ => sys::INVALID_VALUE,
        }
    }
}

/// `clGetProgramBuildInfo` — build status, options or log.
///
/// # Safety
///
/// The `clGet*Info` caller contract for the out parameters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetProgramBuildInfo(
    program: ProgramHandle,
    device: DeviceHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let object = match objects::program(program) {
        Ok(object) => object,
        Err(error) => return error.to_cl(),
    };
    match objects::device_object(device) {
        Ok(device) if in_context(object.context(), device) => {}
        _ => return sys::INVALID_DEVICE,
    }
    // SAFETY: the `clGet*Info` caller contract for the out parameters.
    unsafe {
        match param_name {
            sys::PROGRAM_BUILD_STATUS => sys::write_info_u32(
                sys::BUILD_SUCCESS as u32,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::PROGRAM_BUILD_OPTIONS => {
                let options = object.options();
                sys::write_info_str(&options, param_value_size, param_value, param_value_size_ret)
            }
            sys::PROGRAM_BUILD_LOG => {
                sys::write_info_str("", param_value_size, param_value, param_value_size_ret)
            }
            _ => sys::INVALID_VALUE,
        }
    }
}

/// `clCreateKernel` — an instance of the program's named kernel.
///
/// # Safety
///
/// `program` must be a handle produced by this driver; `kernel_name` is a
/// valid NUL-terminated string; `errcode_ret` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clCreateKernel(
    program: ProgramHandle,
    kernel_name: *const c_char,
    errcode_ret: *mut i32,
) -> KernelHandle {
    let fail = |code: i32| {
        set_err(errcode_ret, code);
        KernelHandle::from_raw(core::ptr::null_mut())
    };
    let program = match objects::program(program) {
        Ok(program) => program,
        Err(error) => return fail(error.to_cl()),
    };
    let name = match unsafe { cstr(kernel_name) } {
        Ok(name) => name,
        Err(error) => return fail(error.to_cl()),
    };
    match program.find(name) {
        Some(def) => {
            let kernel: &'static Kernel = Box::leak(Kernel::new(program, def));
            set_err(errcode_ret, sys::SUCCESS);
            KernelHandle::from_raw(handle_ptr(kernel))
        }
        None => fail(sys::INVALID_KERNEL_NAME),
    }
}

/// `clRetainKernel`.
///
/// # Safety
///
/// `kernel` must be null or a handle produced by this driver.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clRetainKernel(kernel: KernelHandle) -> i32 {
    match objects::kernel(kernel) {
        Ok(object) => {
            object.header().retain();
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clReleaseKernel`.
///
/// # Safety
///
/// `kernel` must be a handle produced by this driver and released no more
/// times than it was retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clReleaseKernel(kernel: KernelHandle) -> i32 {
    match objects::kernel(kernel) {
        Ok(object) => {
            release_ref(object);
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clGetKernelInfo` — function name, argument count, context or program.
///
/// # Safety
///
/// The `clGet*Info` caller contract for the out parameters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetKernelInfo(
    kernel: KernelHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let object = match objects::kernel(kernel) {
        Ok(object) => object,
        Err(error) => return error.to_cl(),
    };
    // SAFETY: the `clGet*Info` caller contract for the out parameters.
    unsafe {
        match param_name {
            sys::KERNEL_FUNCTION_NAME => sys::write_info_str(
                &object.def().name,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::KERNEL_NUM_ARGS => sys::write_info_u32(
                object.def().num_args(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::KERNEL_CONTEXT => sys::write_info_usize(
                handle_ptr(object.program().context()) as usize,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::KERNEL_PROGRAM => sys::write_info_usize(
                handle_ptr(object.program()) as usize,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            _ => sys::INVALID_VALUE,
        }
    }
}

/// `clGetKernelWorkGroupSizeInfo` — the device work-group cap and the
/// preferred launch multiple (1: the claim loop has no divisibility
/// preference).
///
/// # Safety
///
/// The `clGet*Info` caller contract for the out parameters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetKernelWorkGroupInfo(
    kernel: KernelHandle,
    device: DeviceHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    if objects::kernel(kernel).is_err() {
        return sys::INVALID_KERNEL;
    }
    let device = match objects::device_object(device) {
        Ok(device) => device,
        Err(error) => return error.to_cl(),
    };
    let info = device.device().info();
    // SAFETY: the `clGet*Info` caller contract for the out parameters.
    unsafe {
        match param_name {
            sys::KERNEL_WORK_GROUP_SIZE => sys::write_info_u32(
                info.max_work_group_size,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE => {
                sys::write_info_u32(1, param_value_size, param_value, param_value_size_ret)
            }
            _ => sys::INVALID_VALUE,
        }
    }
}

/// `clSetKernelArg` — binds argument `arg_index`.
///
/// # Safety
///
/// `kernel` must be a handle produced by this driver; `arg_value` points
/// at `arg_size` readable bytes (a `cl_mem` handle for buffer arguments).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clSetKernelArg(
    kernel: KernelHandle,
    arg_index: u32,
    arg_size: usize,
    arg_value: *const c_void,
) -> i32 {
    let kernel = match objects::kernel(kernel) {
        Ok(object) => object,
        Err(error) => return error.to_cl(),
    };
    match kernel.set_arg(arg_index as usize, arg_size, arg_value) {
        Ok(()) => sys::SUCCESS,
        Err(error) => error.to_cl(),
    }
}

/// Runs one inline buffer command: transition `event` through
/// `CL_SUBMITTED`/`CL_RUNNING`/`CL_COMPLETE` around `execute`, then either
/// hand the event to the caller or release it. The inline command needs no
/// device progress: it completes before the call returns.
fn run_inline<F>(command_type: u32, queue: &'static Queue, event_out: *mut EventHandle, execute: F) -> i32
where
    F: FnOnce(),
{
    let event: &'static Event = Box::leak(Event::new(command_type, queue.context(), queue, None));
    event.set_status(sys::SUBMITTED);
    event.set_status(sys::RUNNING);
    execute();
    event.finish(sys::COMPLETE);
    if event_out.is_null() {
        release_ref(event);
    } else {
        // SAFETY: caller contract — a non-null event out parameter points
        // at one writable `cl_event` slot.
        unsafe { *event_out = EventHandle::from_raw(handle_ptr(event)) };
    }
    sys::SUCCESS
}

/// Bounds-checks a buffer command region: `offset + size` must not
/// overflow and must fit the buffer.
fn region_ok(buffer: &Mem, offset: usize, size: usize) -> bool {
    offset
        .checked_add(size)
        .is_some_and(|end| end <= buffer.size())
}

/// `clEnqueueReadBuffer` — copies `size` bytes from the buffer to `ptr`.
///
/// The embedded single-address-space profile executes the copy inline, so
/// the command has completed when the call returns regardless of
/// `blocking_read`.
///
/// # Safety
///
/// `command_queue`/`buffer`/`event_wait_list`/`event` obey the OpenCL
/// caller contract; `ptr` points at `size` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clEnqueueReadBuffer(
    command_queue: CommandQueueHandle,
    buffer: MemHandle,
    _blocking_read: u32,
    offset: usize,
    size: usize,
    ptr: *mut c_void,
    num_events_in_wait_list: u32,
    event_wait_list: *const EventHandle,
    event: *mut EventHandle,
) -> i32 {
    let queue = match objects::queue(command_queue) {
        Ok(queue) => queue,
        Err(error) => return error.to_cl(),
    };
    let buffer = match objects::mem(buffer) {
        Ok(buffer) => buffer,
        Err(error) => return error.to_cl(),
    };
    if !core::ptr::eq(
        buffer.context() as *const Context,
        queue.context() as *const Context,
    ) {
        return sys::INVALID_CONTEXT;
    }
    if ptr.is_null() || !region_ok(buffer, offset, size) {
        return sys::INVALID_VALUE;
    }
    if let Err(code) = wait_list(num_events_in_wait_list, event_wait_list) {
        return code;
    }
    run_inline(sys::COMMAND_READ_BUFFER, queue, event, || {
        // SAFETY: the bounds check above proves `size` bytes at
        // `buffer.as_ptr() + offset` and at `ptr` are live and in range;
        // the single address space makes this a plain host copy.
        unsafe {
            core::ptr::copy_nonoverlapping(buffer.as_ptr().add(offset), ptr.cast::<u8>(), size);
        }
    })
}

/// `clEnqueueWriteBuffer` — copies `size` bytes from `ptr` into the
/// buffer, inline.
///
/// # Safety
///
/// `command_queue`/`buffer`/`event_wait_list`/`event` obey the OpenCL
/// caller contract; `ptr` points at `size` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clEnqueueWriteBuffer(
    command_queue: CommandQueueHandle,
    buffer: MemHandle,
    _blocking_write: u32,
    offset: usize,
    size: usize,
    ptr: *const c_void,
    num_events_in_wait_list: u32,
    event_wait_list: *const EventHandle,
    event: *mut EventHandle,
) -> i32 {
    let queue = match objects::queue(command_queue) {
        Ok(queue) => queue,
        Err(error) => return error.to_cl(),
    };
    let buffer = match objects::mem(buffer) {
        Ok(buffer) => buffer,
        Err(error) => return error.to_cl(),
    };
    if !core::ptr::eq(
        buffer.context() as *const Context,
        queue.context() as *const Context,
    ) {
        return sys::INVALID_CONTEXT;
    }
    if ptr.is_null() || !region_ok(buffer, offset, size) {
        return sys::INVALID_VALUE;
    }
    if let Err(code) = wait_list(num_events_in_wait_list, event_wait_list) {
        return code;
    }
    run_inline(sys::COMMAND_WRITE_BUFFER, queue, event, || {
        // SAFETY: the bounds check above proves the destination range is
        // live; the caller contract makes the source readable.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.cast::<u8>(), buffer.as_ptr().add(offset), size);
        }
    })
}

/// `clEnqueueCopyBuffer` — copies `size` bytes between two buffers of the
/// context, inline.
///
/// # Safety
///
/// `command_queue`/`src_buffer`/`dst_buffer`/`event_wait_list`/`event`
/// obey the OpenCL caller contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clEnqueueCopyBuffer(
    command_queue: CommandQueueHandle,
    src_buffer: MemHandle,
    dst_buffer: MemHandle,
    src_offset: usize,
    dst_offset: usize,
    size: usize,
    num_events_in_wait_list: u32,
    event_wait_list: *const EventHandle,
    event: *mut EventHandle,
) -> i32 {
    let queue = match objects::queue(command_queue) {
        Ok(queue) => queue,
        Err(error) => return error.to_cl(),
    };
    let src = match objects::mem(src_buffer) {
        Ok(buffer) => buffer,
        Err(error) => return error.to_cl(),
    };
    let dst = match objects::mem(dst_buffer) {
        Ok(buffer) => buffer,
        Err(error) => return error.to_cl(),
    };
    if !core::ptr::eq(src.context() as *const Context, queue.context() as *const Context)
        || !core::ptr::eq(dst.context() as *const Context, queue.context() as *const Context)
    {
        return sys::INVALID_CONTEXT;
    }
    if !region_ok(src, src_offset, size) || !region_ok(dst, dst_offset, size) {
        return sys::INVALID_VALUE;
    }
    if let Err(code) = wait_list(num_events_in_wait_list, event_wait_list) {
        return code;
    }
    run_inline(sys::COMMAND_COPY_BUFFER, queue, event, || {
        // SAFETY: both ranges passed the bounds check; `copy` (memmove)
        // tolerates the overlap the spec allows between the regions.
        unsafe {
            core::ptr::copy(src.as_ptr().add(src_offset), dst.as_ptr().add(dst_offset), size);
        }
    })
}

/// `clEnqueueNDRangeKernel` — launches `kernel` on the queue's device.
///
/// # Safety
///
/// `command_queue`/`kernel`/`event_wait_list`/`event` obey the OpenCL
/// caller contract; the size arrays point at `work_dim` readable
/// `size_t`s when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clEnqueueNDRangeKernel(
    command_queue: CommandQueueHandle,
    kernel: KernelHandle,
    work_dim: u32,
    global_work_offset: *const usize,
    global_work_size: *const usize,
    local_work_size: *const usize,
    num_events_in_wait_list: u32,
    event_wait_list: *const EventHandle,
    event: *mut EventHandle,
) -> i32 {
    let queue = match objects::queue(command_queue) {
        Ok(queue) => queue,
        Err(error) => return error.to_cl(),
    };
    let kernel = match objects::kernel(kernel) {
        Ok(kernel) => kernel,
        Err(error) => return error.to_cl(),
    };
    if !core::ptr::eq(
        kernel.program().context() as *const Context,
        queue.context() as *const Context,
    ) {
        return sys::INVALID_CONTEXT;
    }
    if !queue.device().supports(kernel.def()) {
        return sys::INVALID_KERNEL;
    }
    if let Err(code) = wait_list(num_events_in_wait_list, event_wait_list) {
        return code;
    }
    if !(1..=3).contains(&work_dim) {
        return sys::INVALID_WORK_DIMENSION;
    }
    if global_work_size.is_null() {
        return sys::INVALID_GLOBAL_WORK_SIZE;
    }
    let axis = work_dim as usize;
    let mut global = [1usize; 3];
    for (index, slot) in global.iter_mut().enumerate().take(axis) {
        // SAFETY: caller contract — `work_dim` readable entries.
        *slot = unsafe { *global_work_size.add(index) };
        if *slot == 0 || *slot > u32::MAX as usize {
            return sys::INVALID_GLOBAL_WORK_SIZE;
        }
    }
    let mut offset = [0usize; 3];
    if !global_work_offset.is_null() {
        for (index, slot) in offset.iter_mut().enumerate().take(axis) {
            // SAFETY: caller contract — `work_dim` readable entries.
            *slot = unsafe { *global_work_offset.add(index) };
            if *slot > u32::MAX as usize {
                return sys::INVALID_GLOBAL_OFFSET;
            }
        }
    }
    let info = queue.device().info();
    let mut local = [1usize; 3];
    if !local_work_size.is_null() {
        let mut product: u64 = 1;
        for (index, slot) in local.iter_mut().enumerate().take(axis) {
            // SAFETY: caller contract — `work_dim` readable entries.
            *slot = unsafe { *local_work_size.add(index) };
            if *slot == 0 || *slot > u32::MAX as usize {
                return sys::INVALID_WORK_GROUP_SIZE;
            }
            if *slot > info.max_work_item_sizes[index] as usize {
                return sys::INVALID_WORK_ITEM_SIZE;
            }
            if global[index] % *slot != 0 {
                return sys::INVALID_WORK_GROUP_SIZE;
            }
            product = product.saturating_mul(*slot as u64);
        }
        if product > u64::from(info.max_work_group_size) {
            return sys::INVALID_WORK_GROUP_SIZE;
        }
    }
    let (offset, global, local) = match crate::exec::validate_sizes(
        work_dim,
        Some(&offset),
        &global,
        Some(&local),
        info.max_work_group_size,
    ) {
        Ok(triple) => triple,
        Err(error) => return error.to_cl(),
    };
    let args = match kernel.marshal() {
        Ok(args) => args,
        Err(error) => return error.to_cl(),
    };
    let def = kernel.def();
    let launch = Launch {
        kernel: def.func,
        name: def.name.as_ref(),
        binding: def.binding,
        args: &args,
        dims: work_dim,
        offset,
        global,
        local,
    };
    let created: &'static Event = Box::leak(Event::new(
        sys::COMMAND_NDRANGE_KERNEL,
        queue.context(),
        queue,
        Some(queue.device()),
    ));
    // The command's reference, owned by `Device::submit`.
    created.header().retain();
    match queue.device().submit(&launch, created) {
        Ok(()) => {
            if event.is_null() {
                release_ref(created);
            } else {
                // SAFETY: caller contract — a writable `cl_event` slot.
                unsafe { *event = EventHandle::from_raw(handle_ptr(created)) };
            }
            sys::SUCCESS
        }
        Err(error) => {
            // `submit` released the command reference without completing
            // the event; finish it for observability, then drop the
            // caller's reference — a failed enqueue returns no handle.
            created.finish(error.to_cl());
            release_ref(created);
            error.to_cl()
        }
    }
}

/// `clWaitForEvents` — blocks until every listed command retires.
///
/// # Safety
///
/// `event_list` points at `num_events` valid `cl_event` handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clWaitForEvents(num_events: u32, event_list: *const EventHandle) -> i32 {
    if num_events == 0 {
        return sys::INVALID_VALUE;
    }
    match wait_list(num_events, event_list) {
        Ok(()) => sys::SUCCESS,
        Err(code) => code,
    }
}

/// `clGetEventInfo` — command type, execution status, context, queue or
/// reference count.
///
/// # Safety
///
/// The `clGet*Info` caller contract for the out parameters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clGetEventInfo(
    event: EventHandle,
    param_name: u32,
    param_value_size: usize,
    param_value: *mut c_void,
    param_value_size_ret: *mut usize,
) -> i32 {
    let object = match objects::event(event) {
        Ok(object) => object,
        Err(error) => return error.to_cl(),
    };
    // SAFETY: the `clGet*Info` caller contract for the out parameters.
    unsafe {
        match param_name {
            sys::EVENT_COMMAND_TYPE => sys::write_info_u32(
                object.command_type(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::EVENT_COMMAND_EXECUTION_STATUS => sys::write_info(
                &object.status().to_ne_bytes(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::EVENT_CONTEXT => sys::write_info_usize(
                handle_ptr(object.context()) as usize,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::EVENT_COMMAND_QUEUE => sys::write_info_usize(
                handle_ptr(object.queue()) as usize,
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            sys::EVENT_REFERENCE_COUNT => sys::write_info_u32(
                object.header().refs(),
                param_value_size,
                param_value,
                param_value_size_ret,
            ),
            _ => sys::INVALID_VALUE,
        }
    }
}

/// `clRetainEvent`.
///
/// # Safety
///
/// `event` must be null or a handle produced by this driver.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clRetainEvent(event: EventHandle) -> i32 {
    match objects::event(event) {
        Ok(object) => {
            object.header().retain();
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

/// `clReleaseEvent`.
///
/// # Safety
///
/// `event` must be a handle produced by this driver and released no more
/// times than it was retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clReleaseEvent(event: EventHandle) -> i32 {
    match objects::event(event) {
        Ok(object) => {
            release_ref(object);
            sys::SUCCESS
        }
        Err(error) => error.to_cl(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NULL32: *mut c_void = core::ptr::null_mut();

    #[test]
    fn platform_ids_requires_a_num_platforms_slot() {
        let code = unsafe { clGetPlatformIDs(0, core::ptr::null_mut(), core::ptr::null_mut()) };
        assert_eq!(code, sys::INVALID_VALUE);
    }

    #[test]
    fn platform_ids_reports_count_or_platform_not_found() {
        let mut count = usize::MAX;
        let code = unsafe { clGetPlatformIDs(0, core::ptr::null_mut(), &mut count) };
        assert!(code == sys::SUCCESS || code == sys::PLATFORM_NOT_FOUND_KHR);
        if code == sys::SUCCESS {
            assert_eq!(count, 1);
        } else {
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn info_queries_reject_null_handles() {
        let code = unsafe {
            clGetPlatformInfo(
                PlatformHandle::from_raw(NULL32),
                sys::PLATFORM_NAME,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_PLATFORM);
        let code = unsafe {
            clGetDeviceInfo(
                DeviceHandle::from_raw(NULL32),
                sys::DEVICE_NAME,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_DEVICE);
        let code = unsafe {
            clGetContextInfo(
                ContextHandle::from_raw(NULL32),
                sys::CONTEXT_NUM_DEVICES,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_CONTEXT);
        let code = unsafe {
            clGetCommandQueueInfo(
                CommandQueueHandle::from_raw(NULL32),
                sys::QUEUE_CONTEXT,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_COMMAND_QUEUE);
        let code = unsafe {
            clGetMemObjectInfo(
                MemHandle::from_raw(NULL32),
                sys::MEM_SIZE,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_MEM_OBJECT);
        let code = unsafe {
            clGetProgramInfo(
                ProgramHandle::from_raw(NULL32),
                sys::PROGRAM_NUM_KERNELS,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_PROGRAM);
        let code = unsafe {
            clGetKernelInfo(
                KernelHandle::from_raw(NULL32),
                sys::KERNEL_NUM_ARGS,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_KERNEL);
        let code = unsafe {
            clGetEventInfo(
                EventHandle::from_raw(NULL32),
                sys::EVENT_COMMAND_TYPE,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_EVENT);
    }

    #[test]
    fn retain_release_reject_null_handles() {
        assert_eq!(
            unsafe { clRetainContext(ContextHandle::from_raw(NULL32)) },
            sys::INVALID_CONTEXT
        );
        assert_eq!(
            unsafe { clReleaseContext(ContextHandle::from_raw(NULL32)) },
            sys::INVALID_CONTEXT
        );
        assert_eq!(
            unsafe { clRetainCommandQueue(CommandQueueHandle::from_raw(NULL32)) },
            sys::INVALID_COMMAND_QUEUE
        );
        assert_eq!(
            unsafe { clReleaseCommandQueue(CommandQueueHandle::from_raw(NULL32)) },
            sys::INVALID_COMMAND_QUEUE
        );
        assert_eq!(
            unsafe { clRetainMemObject(MemHandle::from_raw(NULL32)) },
            sys::INVALID_MEM_OBJECT
        );
        assert_eq!(
            unsafe { clReleaseMemObject(MemHandle::from_raw(NULL32)) },
            sys::INVALID_MEM_OBJECT
        );
        assert_eq!(
            unsafe { clRetainProgram(ProgramHandle::from_raw(NULL32)) },
            sys::INVALID_PROGRAM
        );
        assert_eq!(
            unsafe { clReleaseProgram(ProgramHandle::from_raw(NULL32)) },
            sys::INVALID_PROGRAM
        );
        assert_eq!(
            unsafe { clRetainKernel(KernelHandle::from_raw(NULL32)) },
            sys::INVALID_KERNEL
        );
        assert_eq!(
            unsafe { clReleaseKernel(KernelHandle::from_raw(NULL32)) },
            sys::INVALID_KERNEL
        );
        assert_eq!(
            unsafe { clRetainEvent(EventHandle::from_raw(NULL32)) },
            sys::INVALID_EVENT
        );
        assert_eq!(
            unsafe { clReleaseEvent(EventHandle::from_raw(NULL32)) },
            sys::INVALID_EVENT
        );
    }

    #[test]
    fn create_entry_points_reject_bad_prerequisites() {
        let mut errcode = 0i32;
        let handle = unsafe {
            clCreateContext(
                core::ptr::null(),
                0,
                core::ptr::null(),
                NULL32,
                NULL32,
                &mut errcode,
            )
        };
        assert!(handle.is_null());
        assert_eq!(errcode, sys::INVALID_VALUE);

        let mut errcode = 0i32;
        let handle = unsafe {
            clCreateCommandQueue(
                ContextHandle::from_raw(NULL32),
                DeviceHandle::from_raw(NULL32),
                sys::QUEUE_PROFILING_ENABLE,
                &mut errcode,
            )
        };
        assert!(handle.is_null());
        assert_eq!(errcode, sys::INVALID_QUEUE_PROPERTIES);

        let mut errcode = 0i32;
        let handle = unsafe {
            clCreateCommandQueue(
                ContextHandle::from_raw(NULL32),
                DeviceHandle::from_raw(NULL32),
                0,
                &mut errcode,
            )
        };
        assert!(handle.is_null());
        assert_eq!(errcode, sys::INVALID_CONTEXT);

        let mut errcode = 0i32;
        let handle = unsafe { clCreateBuffer(ContextHandle::from_raw(NULL32), 0, 64, NULL32, &mut errcode) };
        assert!(handle.is_null());
        assert_eq!(errcode, sys::INVALID_CONTEXT);

        let mut errcode = 0i32;
        let handle = unsafe {
            clCreateProgramWithSource(
                ContextHandle::from_raw(NULL32),
                1,
                core::ptr::null(),
                core::ptr::null(),
                &mut errcode,
            )
        };
        assert!(handle.is_null());
        assert_eq!(errcode, sys::INVALID_CONTEXT);

        let mut errcode = 0i32;
        let handle = unsafe {
            clCreateProgramWithBuiltInKernels(
                ContextHandle::from_raw(NULL32),
                1,
                core::ptr::null(),
                b"kernel\0".as_ptr().cast(),
                &mut errcode,
            )
        };
        assert!(handle.is_null());
        assert_eq!(errcode, sys::INVALID_CONTEXT);

        let mut errcode = 0i32;
        let handle = unsafe {
            clCreateProgramWithBinary(
                ContextHandle::from_raw(NULL32),
                1,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null_mut(),
                &mut errcode,
            )
        };
        assert!(handle.is_null());
        assert_eq!(errcode, sys::INVALID_CONTEXT);

        let mut errcode = 0i32;
        let handle = unsafe {
            clCreateKernel(
                ProgramHandle::from_raw(NULL32),
                b"k\0".as_ptr().cast(),
                &mut errcode,
            )
        };
        assert!(handle.is_null());
        assert_eq!(errcode, sys::INVALID_PROGRAM);
    }

    #[test]
    fn enqueue_entry_points_reject_bad_handles() {
        let code = unsafe {
            clEnqueueReadBuffer(
                CommandQueueHandle::from_raw(NULL32),
                MemHandle::from_raw(NULL32),
                sys::BLOCKING,
                0,
                4,
                NULL32,
                0,
                core::ptr::null(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_COMMAND_QUEUE);
        let code = unsafe {
            clEnqueueWriteBuffer(
                CommandQueueHandle::from_raw(NULL32),
                MemHandle::from_raw(NULL32),
                sys::BLOCKING,
                0,
                4,
                NULL32,
                0,
                core::ptr::null(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_COMMAND_QUEUE);
        let code = unsafe {
            clEnqueueCopyBuffer(
                CommandQueueHandle::from_raw(NULL32),
                MemHandle::from_raw(NULL32),
                MemHandle::from_raw(NULL32),
                0,
                0,
                4,
                0,
                core::ptr::null(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_COMMAND_QUEUE);
        let code = unsafe {
            clEnqueueNDRangeKernel(
                CommandQueueHandle::from_raw(NULL32),
                KernelHandle::from_raw(NULL32),
                1,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                0,
                core::ptr::null(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::INVALID_COMMAND_QUEUE);
    }

    #[test]
    fn wait_for_events_validates_its_list() {
        assert_eq!(
            unsafe { clWaitForEvents(0, core::ptr::null()) },
            sys::INVALID_VALUE
        );
        assert_eq!(
            unsafe { clWaitForEvents(1, core::ptr::null()) },
            sys::INVALID_EVENT_WAIT_LIST
        );
    }

    #[test]
    fn set_kernel_arg_rejects_invalid_kernel() {
        let code = unsafe { clSetKernelArg(KernelHandle::from_raw(NULL32), 0, 4, core::ptr::null()) };
        assert_eq!(code, sys::INVALID_KERNEL);
    }

    #[test]
    fn flush_and_finish_reject_invalid_queue() {
        assert_eq!(
            unsafe { clFlush(CommandQueueHandle::from_raw(NULL32)) },
            sys::INVALID_COMMAND_QUEUE
        );
        assert_eq!(
            unsafe { clFinish(CommandQueueHandle::from_raw(NULL32)) },
            sys::INVALID_COMMAND_QUEUE
        );
    }
}
