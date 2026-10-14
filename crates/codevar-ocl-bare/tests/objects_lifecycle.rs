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

//! Object-layer integration: contexts, queues, buffers, programs and
//! kernels created through the `cl*` boundary.
//!
//! The tests drive reference counts, the `clCreateBuffer` flag rules,
//! inline buffer commands with their events, wait-list validation,
//! cross-context rejection and the registry-backed program path — every
//! interaction between `api`, `objects`, `registry` and `platform` the
//! embedded profile exposes.

use core::ffi::c_void;
use std::borrow::Cow;

use codevar_ocl_bare::api::{
    clBuildProgram, clCreateBuffer, clCreateCommandQueue, clCreateContext, clCreateKernel,
    clCreateProgramWithBuiltInKernels, clCreateProgramWithSource, clEnqueueCopyBuffer, clEnqueueReadBuffer,
    clEnqueueWriteBuffer, clFinish, clFlush, clGetCommandQueueInfo, clGetContextInfo, clGetDeviceIDs, clGetEventInfo, clGetKernelInfo, clGetKernelWorkGroupInfo, clGetMemObjectInfo,
    clGetPlatformIDs, clGetProgramBuildInfo, clGetProgramInfo, clReleaseCommandQueue, clReleaseContext,
    clReleaseEvent, clReleaseKernel, clReleaseMemObject, clReleaseProgram, clRetainCommandQueue,
    clRetainContext, clRetainEvent, clRetainKernel, clRetainMemObject, clRetainProgram,
    clWaitForEvents,
};
use codevar_ocl_bare::sys::{
    self, CommandQueueHandle, ContextHandle, DeviceHandle, EventHandle, KernelHandle, MemHandle,
    ProgramHandle,
};
use codevar_ocl_bare::{
    AmpDevice, ArgKind, Device, DeviceInfo, KernelDef, KernelModule, PlatformConfig, ServiceMode, SmpDevice,
    SpirvBinding, WorkItem, init,
};

/// Registry kernel writing `global_id[0]` into its buffer argument.
const FILL: KernelDef = KernelDef {
    name: Cow::Borrowed("fill"),
    func: fill,
    kinds: Cow::Borrowed(&[ArgKind::Buffer]),
    binding: SpirvBinding::NONE,
};

/// Argument-less kernel: `clSetKernelArg` on it must always report an
/// out-of-range index.
const NOP: KernelDef = KernelDef {
    name: Cow::Borrowed("nop"),
    func: nop,
    kinds: Cow::Borrowed(&[]),
    binding: SpirvBinding::NONE,
};

static MODULE: KernelModule = KernelModule {
    name: "objects",
    kernels: &[FILL, NOP],
};

static MODULES: &[&KernelModule] = &[&MODULE];

unsafe extern "C" fn fill(item: *mut WorkItem) {
    // SAFETY: the claim loop hands every work-item a live argument block.
    let item = unsafe { &*item };
    let out = unsafe { item.arg_ptr(0) }.cast::<u32>();
    if out.is_null() {
        return;
    }
    // SAFETY: slot 0 is a buffer of at least `global_id[0] + 1` elements.
    unsafe { *out.add(item.global_id[0] as usize) = item.global_id[0] };
}

unsafe extern "C" fn nop(_item: *mut WorkItem) {}

static SMP_INFO: DeviceInfo = DeviceInfo::new(
    "lifecycle-cpu",
    "codevar-tests",
    sys::DEVICE_TYPE_CPU,
    2,
    64,
    1 << 20,
    4096,
    256,
);

static AMP_INFO: DeviceInfo = DeviceInfo::new(
    "lifecycle-amp",
    "codevar-tests",
    sys::DEVICE_TYPE_ACCELERATOR,
    1,
    64,
    1 << 20,
    4096,
    256,
);

fn ensure_init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let smp = SmpDevice::new(SMP_INFO.clone(), 2).unwrap();
        let amp = AmpDevice::simulated(AMP_INFO.clone(), ServiceMode::Cooperative).unwrap();
        let devices: &'static [&'static dyn Device] =
            Box::leak(vec![smp as &dyn Device, amp as &dyn Device].into_boxed_slice());
        init(PlatformConfig {
            name: "lifecycle-board",
            vendor: "codevar-tests",
            modules: MODULES,
            devices,
        })
        .expect("board registers");
    });
}

/// Both platform devices, in registration order.
fn device_handles() -> [DeviceHandle; 2] {
    ensure_init();
    let mut platform = codevar_ocl_bare::sys::PlatformHandle::from_raw(core::ptr::null_mut());
    let mut count = 0usize;
    let code = unsafe { clGetPlatformIDs(1, &mut platform, &mut count) };
    assert_eq!(code, sys::SUCCESS);
    let mut devices = [DeviceHandle::from_raw(core::ptr::null_mut()); 2];
    let code = unsafe {
        clGetDeviceIDs(
            platform,
            sys::DEVICE_TYPE_ALL,
            2,
            devices.as_mut_ptr(),
            &mut count,
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(count, 2);
    devices
}

fn make_context(devices: &[DeviceHandle]) -> ContextHandle {
    let mut errcode = sys::SUCCESS;
    let context = unsafe {
        clCreateContext(
            core::ptr::null(),
            devices.len() as u32,
            devices.as_ptr(),
            core::ptr::null(),
            core::ptr::null_mut(),
            &mut errcode,
        )
    };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!context.is_null());
    context
}

fn make_queue(context: ContextHandle, device: DeviceHandle) -> CommandQueueHandle {
    let mut errcode = sys::SUCCESS;
    let queue = unsafe { clCreateCommandQueue(context, device, 0, &mut errcode) };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!queue.is_null());
    queue
}

fn make_buffer(context: ContextHandle, flags: u64, size: usize, host: *mut c_void) -> MemHandle {
    let mut errcode = sys::SUCCESS;
    let buffer = unsafe { clCreateBuffer(context, flags, size, host, &mut errcode) };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!buffer.is_null());
    buffer
}

/// Runs an `clGet*Info` query through the size-probe protocol.
fn query_scalar(query: impl Fn(usize, *mut c_void, *mut usize) -> i32) -> Vec<u8> {
    let mut size = 0usize;
    assert_eq!(query(0, core::ptr::null_mut(), &mut size), sys::SUCCESS);
    assert!(size > 0);
    let mut buf = vec![0u8; size];
    assert_eq!(
        query(size, buf.as_mut_ptr().cast(), core::ptr::null_mut()),
        sys::SUCCESS
    );
    buf
}

fn query_string(query: impl Fn(usize, *mut c_void, *mut usize) -> i32) -> String {
    let bytes = query_scalar(query);
    let text = String::from_utf8(bytes).expect("info strings are UTF-8");
    text.trim_end_matches('\0').to_string()
}

fn query_u32(query: impl Fn(usize, *mut c_void, *mut usize) -> i32) -> u32 {
    let bytes = query_scalar(query);
    assert_eq!(bytes.len(), 4);
    u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn query_usize(query: impl Fn(usize, *mut c_void, *mut usize) -> i32) -> usize {
    let bytes = query_scalar(query);
    usize::from_ne_bytes(
        bytes
            .as_slice()
            .try_into()
            .expect("one native usize"),
    )
}

fn context_refs(context: ContextHandle) -> u32 {
    query_u32(|size, ptr, ret| unsafe {
        clGetContextInfo(context, sys::CONTEXT_REFERENCE_COUNT, size, ptr, ret)
    })
}

fn queue_refs(queue: CommandQueueHandle) -> u32 {
    query_u32(|size, ptr, ret| unsafe {
        clGetCommandQueueInfo(queue, sys::QUEUE_REFERENCE_COUNT, size, ptr, ret)
    })
}

fn mem_refs(buffer: MemHandle) -> u32 {
    query_u32(|size, ptr, ret| unsafe {
        clGetMemObjectInfo(buffer, sys::MEM_REFERENCE_COUNT, size, ptr, ret)
    })
}

fn program_refs(program: ProgramHandle) -> u32 {
    query_u32(|size, ptr, ret| unsafe {
        clGetProgramInfo(program, sys::PROGRAM_REFERENCE_COUNT, size, ptr, ret)
    })
}

fn event_status(event: EventHandle) -> i32 {
    let bytes = query_scalar(|size, ptr, ret| unsafe {
        clGetEventInfo(event, sys::EVENT_COMMAND_EXECUTION_STATUS, size, ptr, ret)
    });
    assert_eq!(bytes.len(), 4);
    i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn write_buffer(queue: CommandQueueHandle, buffer: MemHandle, offset: usize, data: &[u8]) -> i32 {
    unsafe {
        clEnqueueWriteBuffer(
            queue,
            buffer,
            sys::BLOCKING,
            offset,
            data.len(),
            data.as_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    }
}

fn read_buffer(queue: CommandQueueHandle, buffer: MemHandle, offset: usize, out: &mut [u8]) -> i32 {
    unsafe {
        clEnqueueReadBuffer(
            queue,
            buffer,
            sys::BLOCKING,
            offset,
            out.len(),
            out.as_mut_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    }
}

#[test]
fn context_lifecycle_and_info_round_trip() {
    let devices = device_handles();
    let context = make_context(&devices);
    assert_eq!(context_refs(context), 1);
    assert_eq!(unsafe { clRetainContext(context) }, sys::SUCCESS);
    assert_eq!(context_refs(context), 2);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
    assert_eq!(context_refs(context), 1);

    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetContextInfo(context, sys::CONTEXT_NUM_DEVICES, size, ptr, ret)
        }),
        2
    );

    // The device list is the raw handle array the caller passed in.
    let list = query_scalar(|size, ptr, ret| unsafe {
        clGetContextInfo(context, sys::CONTEXT_DEVICES, size, ptr, ret)
    });
    let width = core::mem::size_of::<usize>();
    assert_eq!(list.len(), 2 * width);
    for (index, device) in devices.iter().enumerate() {
        let chunk = &list[index * width..(index + 1) * width];
        let value = usize::from_ne_bytes(chunk.try_into().expect("one native usize"));
        assert_eq!(value, device.as_raw() as usize);
    }

    // No properties were requested: a single zero terminator.
    let props = query_scalar(|size, ptr, ret| unsafe {
        clGetContextInfo(context, sys::CONTEXT_PROPERTIES, size, ptr, ret)
    });
    assert_eq!(props.len(), core::mem::size_of::<isize>());
    assert!(props.iter().all(|byte| *byte == 0));

    let code = unsafe { clGetContextInfo(context, 0xFFFF, 0, core::ptr::null_mut(), core::ptr::null_mut()) };
    assert_eq!(code, sys::INVALID_VALUE);

    // Creation-time validation.
    let mut errcode = sys::SUCCESS;
    let bad = unsafe {
        clCreateContext(
            core::ptr::null(),
            0,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null_mut(),
            &mut errcode,
        )
    };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_VALUE);

    let unknown_key = [0x9999isize, 1, 0];
    let mut errcode = sys::SUCCESS;
    let bad = unsafe {
        clCreateContext(
            unknown_key.as_ptr(),
            1,
            devices.as_ptr(),
            core::ptr::null(),
            core::ptr::null_mut(),
            &mut errcode,
        )
    };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_PROPERTY);

    // The platform property must name *this* platform (no dereference of
    // the value happens — pointer equality only).
    let mut platform = codevar_ocl_bare::sys::PlatformHandle::from_raw(core::ptr::null_mut());
    let mut count = 0usize;
    assert_eq!(
        unsafe { clGetPlatformIDs(1, &mut platform, &mut count) },
        sys::SUCCESS
    );
    let wrong_platform = [sys::CONTEXT_PLATFORM, (platform.as_raw() as isize) ^ 1, 0];
    let mut errcode = sys::SUCCESS;
    let bad = unsafe {
        clCreateContext(
            wrong_platform.as_ptr(),
            1,
            devices.as_ptr(),
            core::ptr::null(),
            core::ptr::null_mut(),
            &mut errcode,
        )
    };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_PLATFORM);

    let right_platform = [sys::CONTEXT_PLATFORM, platform.as_raw() as isize, 0];
    let mut errcode = sys::SUCCESS;
    let extra = unsafe {
        clCreateContext(
            right_platform.as_ptr(),
            1,
            devices.as_ptr(),
            core::ptr::null(),
            core::ptr::null_mut(),
            &mut errcode,
        )
    };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!extra.is_null());
    assert_eq!(unsafe { clReleaseContext(extra) }, sys::SUCCESS);

    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

#[test]
fn queue_lifecycle_properties_and_membership() {
    let devices = device_handles();
    let context = make_context(&devices);

    // The embedded profile rejects every property but zero.
    let mut errcode = sys::SUCCESS;
    let bad = unsafe { clCreateCommandQueue(context, devices[0], sys::QUEUE_PROFILING_ENABLE, &mut errcode) };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_QUEUE_PROPERTIES);
    let mut errcode = sys::SUCCESS;
    let bad = unsafe {
        clCreateCommandQueue(
            context,
            devices[0],
            sys::QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            &mut errcode,
        )
    };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_QUEUE_PROPERTIES);

    // A device outside the context cannot join one of its queues. Both
    // handles are valid platform devices — only membership differs.
    let one_device = make_context(&devices[..1]);
    let mut errcode = sys::SUCCESS;
    let bad = unsafe { clCreateCommandQueue(one_device, devices[1], 0, &mut errcode) };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_DEVICE);
    assert_eq!(unsafe { clReleaseContext(one_device) }, sys::SUCCESS);

    let queue = make_queue(context, devices[0]);
    assert_eq!(queue_refs(queue), 1);
    assert_eq!(unsafe { clRetainCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(queue_refs(queue), 2);
    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(queue_refs(queue), 1);

    assert_eq!(
        query_usize(|size, ptr, ret| unsafe {
            clGetCommandQueueInfo(queue, sys::QUEUE_CONTEXT, size, ptr, ret)
        }),
        context.as_raw() as usize
    );
    assert_eq!(
        query_usize(|size, ptr, ret| unsafe {
            clGetCommandQueueInfo(queue, sys::QUEUE_DEVICE, size, ptr, ret)
        }),
        devices[0].as_raw() as usize
    );
    let properties = query_scalar(|size, ptr, ret| unsafe {
        clGetCommandQueueInfo(queue, sys::QUEUE_PROPERTIES, size, ptr, ret)
    });
    let mut word = [0u8; 8];
    word.copy_from_slice(&properties);
    assert_eq!(u64::from_ne_bytes(word), 0);
    let code =
        unsafe { clGetCommandQueueInfo(queue, 0xFFFF, 0, core::ptr::null_mut(), core::ptr::null_mut()) };
    assert_eq!(code, sys::INVALID_VALUE);

    assert_eq!(unsafe { clFlush(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clFinish(queue) }, sys::SUCCESS);

    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

#[test]
fn buffer_creation_obeys_the_flag_rules() {
    let devices = device_handles();
    let context = make_context(&devices);

    let expect_failure = |flags: u64, size: usize, host: *mut c_void, expected: i32| {
        let mut errcode = sys::SUCCESS;
        let buffer = unsafe { clCreateBuffer(context, flags, size, host, &mut errcode) };
        assert!(buffer.is_null());
        assert_eq!(errcode, expected);
    };
    expect_failure(0, 0, core::ptr::null_mut(), sys::INVALID_BUFFER_SIZE);
    expect_failure(1 << 10, 64, core::ptr::null_mut(), sys::INVALID_VALUE);
    expect_failure(
        sys::MEM_READ_ONLY | sys::MEM_WRITE_ONLY,
        64,
        core::ptr::null_mut(),
        sys::INVALID_VALUE,
    );
    expect_failure(
        sys::MEM_USE_HOST_PTR | sys::MEM_COPY_HOST_PTR,
        64,
        core::ptr::null_mut(),
        sys::INVALID_VALUE,
    );
    expect_failure(
        sys::MEM_USE_HOST_PTR,
        64,
        core::ptr::null_mut(),
        sys::INVALID_HOST_PTR,
    );
    expect_failure(
        sys::MEM_COPY_HOST_PTR,
        64,
        core::ptr::null_mut(),
        sys::INVALID_HOST_PTR,
    );

    // `COPY` seeds the buffer from the host bytes at creation time.
    let seed = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let buffer = make_buffer(
        context,
        sys::MEM_COPY_HOST_PTR,
        seed.len(),
        seed.as_ptr().cast_mut().cast(),
    );
    let mut queue = make_queue(context, devices[0]);
    let mut back = [0u8; 8];
    assert_eq!(read_buffer(queue, buffer, 0, &mut back), sys::SUCCESS);
    assert_eq!(back, seed);

    assert_eq!(
        query_scalar(|size, ptr, ret| unsafe { clGetMemObjectInfo(buffer, sys::MEM_SIZE, size, ptr, ret) })
            .len(),
        8
    );
    let flags =
        query_scalar(|size, ptr, ret| unsafe { clGetMemObjectInfo(buffer, sys::MEM_FLAGS, size, ptr, ret) });
    let mut word = [0u8; 8];
    word.copy_from_slice(&flags);
    assert_eq!(u64::from_ne_bytes(word), sys::MEM_COPY_HOST_PTR);
    assert_eq!(
        query_usize(|size, ptr, ret| unsafe { clGetMemObjectInfo(buffer, sys::MEM_CONTEXT, size, ptr, ret) }),
        context.as_raw() as usize
    );

    // Reference counting: one for the handle, one for the retain.
    assert_eq!(mem_refs(buffer), 1);
    assert_eq!(unsafe { clRetainMemObject(buffer) }, sys::SUCCESS);
    assert_eq!(mem_refs(buffer), 2);
    assert_eq!(unsafe { clReleaseMemObject(buffer) }, sys::SUCCESS);
    assert_eq!(mem_refs(buffer), 1);

    // `USE` aliases the application's storage: writes through either side
    // are visible to the other.
    let mut host = [9u8, 9, 9, 9, 9, 9, 9, 9];
    let alias = make_buffer(
        context,
        sys::MEM_USE_HOST_PTR,
        host.len(),
        host.as_mut_ptr().cast(),
    );
    host[0] = 42;
    let mut back = [0u8; 8];
    assert_eq!(read_buffer(queue, alias, 0, &mut back), sys::SUCCESS);
    assert_eq!(back[0], 42);
    assert_eq!(write_buffer(queue, alias, 0, &[7u8; 8]), sys::SUCCESS);
    assert_eq!(host, [7u8; 8]);

    // Every buffer retains the context it was created in; releasing the
    // buffers gives those references back.
    let before = context_refs(context);
    assert_eq!(before, 4); // context + queue + buffer + alias
    assert_eq!(unsafe { clReleaseMemObject(alias) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(buffer) }, sys::SUCCESS);
    assert_eq!(context_refs(context), before - 2);

    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    queue = CommandQueueHandle::from_raw(core::ptr::null_mut());
    let _ = queue;
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

#[test]
fn buffer_commands_copy_data_and_drive_events() {
    let devices = device_handles();
    let context = make_context(&devices);
    let queue = make_queue(context, devices[0]);
    let source = make_buffer(context, 0, 32, core::ptr::null_mut());
    let target = make_buffer(context, 0, 32, core::ptr::null_mut());

    // Write with an offset, copy a window, read it back.
    assert_eq!(write_buffer(queue, source, 4, &[10u8, 11, 12, 13]), sys::SUCCESS);
    let code = unsafe {
        clEnqueueCopyBuffer(
            queue,
            source,
            target,
            4,
            16,
            4,
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    let mut window = [0u8; 4];
    assert_eq!(read_buffer(queue, target, 16, &mut window), sys::SUCCESS);
    assert_eq!(window, [10, 11, 12, 13]);
    let mut whole = vec![0u8; 32];
    assert_eq!(read_buffer(queue, source, 0, &mut whole), sys::SUCCESS);
    assert_eq!(&whole[4..8], &[10, 11, 12, 13]);
    assert!(whole[..4].iter().all(|byte| *byte == 0));

    // An inline command's event is already complete when the call
    // returns, and it describes the command that produced it.
    let mut event = EventHandle::from_raw(core::ptr::null_mut());
    let code = unsafe {
        clEnqueueWriteBuffer(
            queue,
            source,
            sys::BLOCKING,
            0,
            4,
            whole.as_ptr().cast(),
            0,
            core::ptr::null(),
            &mut event,
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert!(!event.is_null());
    assert_eq!(event_status(event), sys::COMPLETE);
    let command =
        query_u32(|size, ptr, ret| unsafe { clGetEventInfo(event, sys::EVENT_COMMAND_TYPE, size, ptr, ret) });
    assert_eq!(command, sys::COMMAND_WRITE_BUFFER);
    assert_eq!(
        query_usize(|size, ptr, ret| unsafe { clGetEventInfo(event, sys::EVENT_CONTEXT, size, ptr, ret) }),
        context.as_raw() as usize
    );
    assert_eq!(
        query_usize(|size, ptr, ret| unsafe {
            clGetEventInfo(event, sys::EVENT_COMMAND_QUEUE, size, ptr, ret)
        }),
        queue.as_raw() as usize
    );
    let refs = query_u32(|size, ptr, ret| unsafe {
        clGetEventInfo(event, sys::EVENT_REFERENCE_COUNT, size, ptr, ret)
    });
    assert_eq!(refs, 1);
    assert_eq!(unsafe { clRetainEvent(event) }, sys::SUCCESS);
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetEventInfo(event, sys::EVENT_REFERENCE_COUNT, size, ptr, ret)
        }),
        2
    );
    assert_eq!(unsafe { clReleaseEvent(event) }, sys::SUCCESS);
    assert_eq!(unsafe { clWaitForEvents(1, &event) }, sys::SUCCESS);

    // The completed event can gate a later command.
    let mut out = [0u8; 4];
    let code = unsafe {
        clEnqueueReadBuffer(
            queue,
            source,
            sys::BLOCKING,
            0,
            4,
            out.as_mut_ptr().cast(),
            1,
            &event,
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseEvent(event) }, sys::SUCCESS);

    // Wait-list pair validation happens before anything else runs.
    let code = unsafe {
        clEnqueueReadBuffer(
            queue,
            source,
            sys::BLOCKING,
            0,
            4,
            out.as_mut_ptr().cast(),
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_EVENT_WAIT_LIST);
    let mut stale = EventHandle::from_raw(core::ptr::null_mut());
    let code = unsafe {
        clEnqueueReadBuffer(
            queue,
            source,
            sys::BLOCKING,
            0,
            4,
            out.as_mut_ptr().cast(),
            0,
            &stale,
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_EVENT_WAIT_LIST);
    let code = unsafe {
        clEnqueueReadBuffer(
            queue,
            source,
            sys::BLOCKING,
            0,
            4,
            out.as_mut_ptr().cast(),
            1,
            &stale,
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_EVENT_WAIT_LIST);

    // Region bounds: past the end, overflowing, or a null destination.
    let code = unsafe {
        clEnqueueReadBuffer(
            queue,
            source,
            sys::BLOCKING,
            30,
            8,
            out.as_mut_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_VALUE);
    let code = unsafe {
        clEnqueueReadBuffer(
            queue,
            source,
            sys::BLOCKING,
            usize::MAX - 4,
            8,
            out.as_mut_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_VALUE);
    let code = unsafe {
        clEnqueueReadBuffer(
            queue,
            source,
            sys::BLOCKING,
            0,
            4,
            core::ptr::null_mut(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_VALUE);

    // Objects from another context never mix with this queue.
    let other = make_context(&devices[..1]);
    let foreign = make_buffer(other, 0, 32, core::ptr::null_mut());
    let code = unsafe {
        clEnqueueReadBuffer(
            queue,
            foreign,
            sys::BLOCKING,
            0,
            4,
            out.as_mut_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_CONTEXT);
    let code = unsafe {
        clEnqueueWriteBuffer(
            queue,
            foreign,
            sys::BLOCKING,
            0,
            4,
            whole.as_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_CONTEXT);
    let code = unsafe {
        clEnqueueCopyBuffer(
            queue,
            foreign,
            source,
            0,
            0,
            4,
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_CONTEXT);

    // Null handles fail with their own codes.
    let code = unsafe {
        clEnqueueReadBuffer(
            CommandQueueHandle::from_raw(core::ptr::null_mut()),
            source,
            sys::BLOCKING,
            0,
            4,
            out.as_mut_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_COMMAND_QUEUE);
    let code = unsafe {
        clEnqueueReadBuffer(
            queue,
            MemHandle::from_raw(core::ptr::null_mut()),
            sys::BLOCKING,
            0,
            4,
            out.as_mut_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_MEM_OBJECT);

    assert_eq!(unsafe { clReleaseMemObject(foreign) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(other) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(target) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(source) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

#[test]
fn programs_bind_the_registry_and_track_reference_counts() {
    let devices = device_handles();
    let context = make_context(&devices);

    // No online compiler exists: source programs are refused outright,
    // but only after the context and the string list validate.
    let source = b"kernel void ghost() {}\0";
    let strings = [source.as_ptr().cast::<core::ffi::c_char>()];
    let mut errcode = sys::SUCCESS;
    let program =
        unsafe { clCreateProgramWithSource(context, 1, strings.as_ptr(), core::ptr::null(), &mut errcode) };
    assert!(program.is_null());
    assert_eq!(errcode, sys::COMPILER_NOT_AVAILABLE);
    let mut errcode = sys::SUCCESS;
    let program =
        unsafe { clCreateProgramWithSource(context, 0, core::ptr::null(), core::ptr::null(), &mut errcode) };
    assert!(program.is_null());
    assert_eq!(errcode, sys::INVALID_VALUE);

    // Built-in programs resolve their names against the registry.
    let names = b"fill;nop\0";
    let mut errcode = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBuiltInKernels(context, 1, devices.as_ptr(), names.as_ptr().cast(), &mut errcode)
    };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!program.is_null());

    assert_eq!(
        query_usize(|size, ptr, ret| unsafe {
            clGetProgramInfo(program, sys::PROGRAM_NUM_KERNELS, size, ptr, ret)
        }),
        2
    );
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetProgramInfo(program, sys::PROGRAM_KERNEL_NAMES, size, ptr, ret)
        }),
        "fill;nop"
    );
    assert_eq!(
        query_usize(|size, ptr, ret| unsafe {
            clGetProgramInfo(program, sys::PROGRAM_CONTEXT, size, ptr, ret)
        }),
        context.as_raw() as usize
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetProgramInfo(program, sys::PROGRAM_NUM_DEVICES, size, ptr, ret)
        }),
        2
    );
    let sources = query_scalar(|size, ptr, ret| unsafe {
        clGetProgramInfo(program, sys::PROGRAM_SOURCE, size, ptr, ret)
    });
    assert_eq!(sources, [0]);
    let status = query_u32(|size, ptr, ret| unsafe {
        clGetProgramInfo(program, sys::PROGRAM_BUILD_STATUS, size, ptr, ret)
    });
    assert_eq!(status as i32, sys::BUILD_SUCCESS);

    // `clBuildProgram` only records the options it was given.
    let options = b"-D WIDTH=8\0";
    let code = unsafe {
        clBuildProgram(
            program,
            0,
            core::ptr::null(),
            options.as_ptr().cast(),
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetProgramInfo(program, sys::PROGRAM_BUILD_OPTIONS, size, ptr, ret)
        }),
        "-D WIDTH=8"
    );
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetProgramBuildInfo(program, devices[0], sys::PROGRAM_BUILD_LOG, size, ptr, ret)
        }),
        ""
    );

    // Creation-time failures of the built-in path.
    let mut errcode = sys::SUCCESS;
    let bad = unsafe {
        clCreateProgramWithBuiltInKernels(
            context,
            1,
            devices.as_ptr(),
            b"ghost\0".as_ptr().cast(),
            &mut errcode,
        )
    };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_KERNEL_NAME);
    let mut errcode = sys::SUCCESS;
    let bad = unsafe {
        clCreateProgramWithBuiltInKernels(context, 1, devices.as_ptr(), b"\0".as_ptr().cast(), &mut errcode)
    };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_VALUE);
    let mut errcode = sys::SUCCESS;
    let bad = unsafe {
        clCreateProgramWithBuiltInKernels(context, 0, core::ptr::null(), names.as_ptr().cast(), &mut errcode)
    };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_VALUE);

    // The kernel hands its program a reference: creating a kernel bumps
    // the count, releasing the program first only drops the caller's.
    assert_eq!(program_refs(program), 1);
    assert_eq!(unsafe { clRetainProgram(program) }, sys::SUCCESS);
    assert_eq!(program_refs(program), 2);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(program_refs(program), 1);

    let mut errcode = sys::SUCCESS;
    let kernel = unsafe { clCreateKernel(program, b"fill\0".as_ptr().cast(), &mut errcode) };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!kernel.is_null());
    assert_eq!(program_refs(program), 2);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(program_refs(program), 1);

    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetKernelInfo(kernel, sys::KERNEL_FUNCTION_NAME, size, ptr, ret)
        }),
        "fill"
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe { clGetKernelInfo(kernel, sys::KERNEL_NUM_ARGS, size, ptr, ret) }),
        1
    );
    assert_eq!(
        query_usize(|size, ptr, ret| unsafe { clGetKernelInfo(kernel, sys::KERNEL_CONTEXT, size, ptr, ret) }),
        context.as_raw() as usize
    );
    assert_eq!(
        query_usize(|size, ptr, ret| unsafe { clGetKernelInfo(kernel, sys::KERNEL_PROGRAM, size, ptr, ret) }),
        program.as_raw() as usize
    );

    // Work-group limits come from the queue's device.
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetKernelWorkGroupInfo(kernel, devices[0], sys::KERNEL_WORK_GROUP_SIZE, size, ptr, ret)
        }),
        64
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetKernelWorkGroupInfo(
                kernel,
                devices[0],
                sys::KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                size,
                ptr,
                ret,
            )
        }),
        1
    );
    let code = unsafe {
        clGetKernelWorkGroupInfo(
            kernel,
            devices[0],
            0xFFFF,
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_VALUE);
    let code = unsafe {
        clGetKernelWorkGroupInfo(
            kernel,
            DeviceHandle::from_raw(core::ptr::null_mut()),
            sys::KERNEL_WORK_GROUP_SIZE,
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_DEVICE);
    let code = unsafe {
        clGetKernelWorkGroupInfo(
            KernelHandle::from_raw(core::ptr::null_mut()),
            devices[0],
            sys::KERNEL_WORK_GROUP_SIZE,
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_KERNEL);

    // Unknown kernel names and handles fail creation.
    let mut errcode = sys::SUCCESS;
    let bad = unsafe { clCreateKernel(program, b"ghost\0".as_ptr().cast(), &mut errcode) };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_KERNEL_NAME);
    let mut errcode = sys::SUCCESS;
    let bad = unsafe { clCreateKernel(program, core::ptr::null(), &mut errcode) };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_VALUE);
    let mut errcode = sys::SUCCESS;
    let bad = unsafe {
        clCreateKernel(
            ProgramHandle::from_raw(core::ptr::null_mut()),
            b"fill\0".as_ptr().cast(),
            &mut errcode,
        )
    };
    assert!(bad.is_null());
    assert_eq!(errcode, sys::INVALID_PROGRAM);

    // Balance: the kernel drops its program reference on release.
    assert_eq!(unsafe { clRetainKernel(kernel) }, sys::SUCCESS);
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe { clGetKernelInfo(kernel, sys::KERNEL_NUM_ARGS, size, ptr, ret) }),
        1
    );
    assert_eq!(unsafe { clReleaseKernel(kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(kernel) }, sys::SUCCESS);
    assert_eq!(program_refs(program), 1);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}
