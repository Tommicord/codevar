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

//! Kernel-launch integration: `clSetKernelArg` validation, the exact
//! `clEnqueueNDRangeKernel` error codes and their ordering, successful
//! launches with their events, an SMP worker thread and concurrent queues.
//!
//! The tests exercise `api` → `objects` → `exec` → `devices` as one unit:
//! a launch marshals through the kernel's argument slots, is validated
//! against the queue's device and executes on the SMP claim loop — with a
//! real secondary-core worker thread in one test.

// Integration test crate: AGENTS.md permits expect()/unwrap() in tests,
// but clippy.toml's allow-expect-in-tests only exempts `#[test]` bodies,
// not the fixtures shared by them.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use core::ffi::c_void;
use std::borrow::Cow;
use std::sync::{Once, OnceLock};

use codevar_ocl_bare::api::{
    clCreateBuffer, clCreateCommandQueue, clCreateContext, clCreateKernel, clCreateProgramWithBuiltInKernels,
    clEnqueueNDRangeKernel, clEnqueueReadBuffer, clEnqueueWriteBuffer, clFinish, clFlush, clGetDeviceIDs,
    clGetEventInfo, clGetMemObjectInfo, clGetPlatformIDs, clReleaseCommandQueue, clReleaseContext,
    clReleaseEvent, clReleaseKernel, clReleaseMemObject, clReleaseProgram, clSetKernelArg, clWaitForEvents,
};
use codevar_ocl_bare::sys::{
    self, CommandQueueHandle, ContextHandle, DeviceHandle, EventHandle, KernelHandle, MemHandle,
    ProgramHandle,
};
use codevar_ocl_bare::{
    ArgKind, Device, DeviceInfo, KernelDef, KernelModule, PlatformConfig, SmpDevice, SpirvBinding, WorkItem,
    init,
};

/// Registry kernel writing `global_id[0]` into `out[global_id[0]]`.
const FILL: KernelDef = KernelDef {
    name: Cow::Borrowed("fill"),
    func: fill,
    kinds: Cow::Borrowed(&[ArgKind::Buffer]),
    binding: SpirvBinding::NONE,
};

/// Registry kernel packing a 3-D global id into one word, indexed
/// row-major over the launch extent (the launches using it pass no
/// global offset).
const STORE_IDS: KernelDef = KernelDef {
    name: Cow::Borrowed("store_ids"),
    func: store_ids,
    kinds: Cow::Borrowed(&[ArgKind::Buffer]),
    binding: SpirvBinding::NONE,
};

/// Registry kernel multiplying `out[global_id[0]]` by a scalar — the
/// buffer-plus-scalar marshalling path.
const SCALE: KernelDef = KernelDef {
    name: Cow::Borrowed("scale"),
    func: scale,
    kinds: Cow::Borrowed(&[ArgKind::Buffer, ArgKind::Scalar(4)]),
    binding: SpirvBinding::NONE,
};

/// Argument-less kernel: every `clSetKernelArg` index is out of range and
/// an unbound launch fails marshal.
const NOP: KernelDef = KernelDef {
    name: Cow::Borrowed("nop"),
    func: nop,
    kinds: Cow::Borrowed(&[]),
    binding: SpirvBinding::NONE,
};

/// Kernel declaring a 20-byte scalar: wider than one argument slot, so a
/// correctly sized `clSetKernelArg` must still be refused.
const BIG: KernelDef = KernelDef {
    name: Cow::Borrowed("big"),
    func: nop,
    kinds: Cow::Borrowed(&[ArgKind::Scalar(20)]),
    binding: SpirvBinding::NONE,
};

static MODULE: KernelModule = KernelModule {
    name: "launch",
    kernels: &[FILL, STORE_IDS, SCALE, NOP, BIG],
};

static MODULES: &[&KernelModule] = &[&MODULE];

unsafe extern "C" fn fill(item: *mut WorkItem) {
    // SAFETY: the claim loop hands every work-item a live argument block.
    let item = unsafe { &*item };
    let out = unsafe { item.arg_ptr(0) }.cast::<u32>();
    if out.is_null() {
        return;
    }
    // SAFETY: the buffer covers at least `global_id[0] + 1` elements.
    unsafe { *out.add(item.global_id[0] as usize) = item.global_id[0] };
}

unsafe extern "C" fn store_ids(item: *mut WorkItem) {
    // SAFETY: the claim loop hands every work-item a live argument block.
    let item = unsafe { &*item };
    let out = unsafe { item.arg_ptr(0) }.cast::<u32>();
    if out.is_null() {
        return;
    }
    let width = item.local_size[0] * item.num_groups[0];
    let height = item.local_size[1] * item.num_groups[1];
    let index = item.global_id[0] + item.global_id[1] * width + item.global_id[2] * width * height;
    let packed = item.global_id[0] | (item.global_id[1] << 8) | (item.global_id[2] << 16);
    // SAFETY: the buffer covers the whole `width * height * local_size[2]`
    // index space of the launch.
    unsafe { *out.add(index as usize) = packed };
}

unsafe extern "C" fn scale(item: *mut WorkItem) {
    // SAFETY: the claim loop hands every work-item a live argument block.
    let item = unsafe { &*item };
    let out = unsafe { item.arg_ptr(0) }.cast::<u32>();
    if out.is_null() {
        return;
    }
    let factor = unsafe { item.arg_u32(1) };
    let slot = unsafe { out.add(item.global_id[0] as usize) };
    // SAFETY: the buffer covers at least `global_id[0] + 1` elements.
    unsafe { *slot = (*slot).wrapping_mul(factor) };
}

unsafe extern "C" fn nop(_item: *mut WorkItem) {}

static SMP_INFO: DeviceInfo = DeviceInfo::new(
    "launch-cpu",
    "codevar-tests",
    sys::DEVICE_TYPE_CPU,
    2,
    64,
    1 << 20,
    4096,
    256,
);

/// The concrete SMP device, kept for `worker_loop`/`wait_workers`.
static SMP: OnceLock<&'static SmpDevice> = OnceLock::new();

fn ensure_init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let smp = SmpDevice::new(SMP_INFO.clone(), 2).expect("smp device");
        let devices: &'static [&'static dyn Device] = Box::leak(vec![smp as &dyn Device].into_boxed_slice());
        init(PlatformConfig {
            name: "launch-board",
            vendor: "codevar-tests",
            modules: MODULES,
            devices,
        })
        .expect("board registers");
        let _ = SMP.set(smp);
    });
}

/// The platform's single device (the SMP CPU).
fn device_handle() -> DeviceHandle {
    ensure_init();
    let mut platform = sys::PlatformHandle::from_raw(core::ptr::null_mut());
    let mut count = 0usize;
    assert_eq!(
        unsafe { clGetPlatformIDs(1, &mut platform, &mut count) },
        sys::SUCCESS
    );
    let mut device = DeviceHandle::from_raw(core::ptr::null_mut());
    let code = unsafe { clGetDeviceIDs(platform, sys::DEVICE_TYPE_ALL, 1, &mut device, &mut count) };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(count, 1);
    device
}

fn make_context(device: DeviceHandle) -> ContextHandle {
    let mut errcode = sys::SUCCESS;
    let context = unsafe {
        clCreateContext(
            core::ptr::null(),
            1,
            &device,
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

fn make_buffer(context: ContextHandle, size: usize) -> MemHandle {
    let mut errcode = sys::SUCCESS;
    let buffer = unsafe { clCreateBuffer(context, 0, size, core::ptr::null_mut(), &mut errcode) };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!buffer.is_null());
    buffer
}

fn make_program(context: ContextHandle, device: DeviceHandle, names: &str) -> ProgramHandle {
    let mut c_name = names.as_bytes().to_vec();
    c_name.push(0);
    let mut errcode = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBuiltInKernels(context, 1, &device, c_name.as_ptr().cast(), &mut errcode)
    };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!program.is_null());
    program
}

fn make_kernel(program: ProgramHandle, name: &str) -> KernelHandle {
    let mut c_name = name.as_bytes().to_vec();
    c_name.push(0);
    let mut errcode = sys::SUCCESS;
    let kernel = unsafe { clCreateKernel(program, c_name.as_ptr().cast(), &mut errcode) };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!kernel.is_null());
    kernel
}

fn set_arg_raw(kernel: KernelHandle, index: u32, size: usize, value: *const c_void) -> i32 {
    unsafe { clSetKernelArg(kernel, index, size, value) }
}

fn set_buffer_arg(kernel: KernelHandle, index: u32, buffer: MemHandle) {
    let code = set_arg_raw(
        kernel,
        index,
        core::mem::size_of::<MemHandle>(),
        (&buffer as *const MemHandle).cast(),
    );
    assert_eq!(code, sys::SUCCESS);
}

fn set_scalar_arg(kernel: KernelHandle, index: u32, value: u32) {
    let code = set_arg_raw(kernel, index, 4, (&value as *const u32).cast());
    assert_eq!(code, sys::SUCCESS);
}

/// Enqueues with no wait list and no event out parameter.
#[allow(clippy::too_many_arguments)]
fn enqueue(
    queue: CommandQueueHandle,
    kernel: KernelHandle,
    work_dim: u32,
    global_work_offset: *const usize,
    global_work_size: *const usize,
    local_work_size: *const usize,
) -> i32 {
    unsafe {
        clEnqueueNDRangeKernel(
            queue,
            kernel,
            work_dim,
            global_work_offset,
            global_work_size,
            local_work_size,
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    }
}

/// Enqueues a 1-D launch, returning the command's event.
fn enqueue_event(
    queue: CommandQueueHandle,
    kernel: KernelHandle,
    global: &[usize],
    event: *mut EventHandle,
) -> i32 {
    unsafe {
        clEnqueueNDRangeKernel(
            queue,
            kernel,
            global.len() as u32,
            core::ptr::null(),
            global.as_ptr(),
            core::ptr::null(),
            0,
            core::ptr::null(),
            event,
        )
    }
}

fn read_buffer(queue: CommandQueueHandle, buffer: MemHandle, out: &mut [u8]) -> i32 {
    unsafe {
        clEnqueueReadBuffer(
            queue,
            buffer,
            sys::BLOCKING,
            0,
            out.len(),
            out.as_mut_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    }
}

fn write_buffer(queue: CommandQueueHandle, buffer: MemHandle, data: &[u8]) -> i32 {
    unsafe {
        clEnqueueWriteBuffer(
            queue,
            buffer,
            sys::BLOCKING,
            0,
            data.len(),
            data.as_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    }
}

/// Views a `u32` scratch slice as the byte window a read command fills.
fn bytes_mut(words: &mut [u32]) -> &mut [u8] {
    // SAFETY: `u32` has no padding; the reinterpreted slice addresses the
    // same bytes, `size_of::<u32>() * len` of them.
    unsafe { core::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), words.len() * 4) }
}

fn event_status(event: EventHandle) -> i32 {
    let mut size = 0usize;
    let code = unsafe {
        clGetEventInfo(
            event,
            sys::EVENT_COMMAND_EXECUTION_STATUS,
            0,
            core::ptr::null_mut(),
            &mut size,
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(size, 4);
    let mut word = [0u8; 4];
    let code = unsafe {
        clGetEventInfo(
            event,
            sys::EVENT_COMMAND_EXECUTION_STATUS,
            word.len(),
            word.as_mut_ptr().cast(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    i32::from_ne_bytes(word)
}

fn mem_refs(buffer: MemHandle) -> u32 {
    let mut size = 0usize;
    let code = unsafe {
        clGetMemObjectInfo(
            buffer,
            sys::MEM_REFERENCE_COUNT,
            0,
            core::ptr::null_mut(),
            &mut size,
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(size, 4);
    let mut word = [0u8; 4];
    let code = unsafe {
        clGetMemObjectInfo(
            buffer,
            sys::MEM_REFERENCE_COUNT,
            word.len(),
            word.as_mut_ptr().cast(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    u32::from_ne_bytes(word)
}

#[test]
fn ndrange_rejects_malformed_launches_with_exact_codes() {
    let device = device_handle();
    let context = make_context(device);
    let queue = make_queue(context, device);
    let program = make_program(context, device, "fill;nop");
    let fill_kernel = make_kernel(program, "fill");
    let nop_kernel = make_kernel(program, "nop");
    let buffer = make_buffer(context, 16 * 4);

    let global = [8usize];
    let local = [8usize];
    let zero_global = [0usize];
    let huge_global = [usize::MAX];
    let huge_offset = [usize::MAX];
    let zero_local = [0usize];
    let item_too_big = [128usize];
    let indivisible = [6usize];
    let four = [4usize];
    let cube_global = [8usize; 3];
    let cube_local = [8usize; 3];

    // A failed enqueue hands back no event: the caller's slot is
    // untouched no matter which rule rejects the launch.
    let mut event_slot = EventHandle::from_raw(core::ptr::null_mut());

    // Handle-level failures come first.
    assert_eq!(
        enqueue(
            CommandQueueHandle::from_raw(core::ptr::null_mut()),
            fill_kernel,
            1,
            core::ptr::null(),
            global.as_ptr(),
            core::ptr::null(),
        ),
        sys::INVALID_COMMAND_QUEUE
    );
    assert_eq!(
        enqueue(
            queue,
            KernelHandle::from_raw(core::ptr::null_mut()),
            1,
            core::ptr::null(),
            global.as_ptr(),
            core::ptr::null(),
        ),
        sys::INVALID_KERNEL
    );

    // A kernel never launches on a queue of another context — and that
    // check precedes every launch-shape rule (work_dim 0 below).
    let other_context = make_context(device);
    let other_queue = make_queue(other_context, device);
    assert_eq!(
        enqueue(
            other_queue,
            fill_kernel,
            0,
            core::ptr::null(),
            global.as_ptr(),
            core::ptr::null(),
        ),
        sys::INVALID_CONTEXT
    );
    assert_eq!(
        enqueue(
            other_queue,
            nop_kernel,
            0,
            core::ptr::null(),
            global.as_ptr(),
            core::ptr::null(),
        ),
        sys::INVALID_CONTEXT
    );

    // The wait list is validated before the launch shape: the malformed
    // dimension loses to the malformed wait list.
    let code = unsafe {
        clEnqueueNDRangeKernel(
            queue,
            fill_kernel,
            0,
            core::ptr::null(),
            global.as_ptr(),
            core::ptr::null(),
            1,
            core::ptr::null(),
            &mut event_slot,
        )
    };
    assert_eq!(code, sys::INVALID_EVENT_WAIT_LIST);

    // Dimension range.
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            0,
            core::ptr::null(),
            global.as_ptr(),
            core::ptr::null(),
        ),
        sys::INVALID_WORK_DIMENSION
    );
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            4,
            core::ptr::null(),
            global.as_ptr(),
            core::ptr::null(),
        ),
        sys::INVALID_WORK_DIMENSION
    );

    // Global work size: null, zero or wider than the device can index.
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            1,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
        ),
        sys::INVALID_GLOBAL_WORK_SIZE
    );
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            1,
            core::ptr::null(),
            zero_global.as_ptr(),
            core::ptr::null(),
        ),
        sys::INVALID_GLOBAL_WORK_SIZE
    );
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            1,
            core::ptr::null(),
            huge_global.as_ptr(),
            core::ptr::null(),
        ),
        sys::INVALID_GLOBAL_WORK_SIZE
    );

    // Global offset wider than a work-item id.
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            1,
            huge_offset.as_ptr(),
            global.as_ptr(),
            core::ptr::null(),
        ),
        sys::INVALID_GLOBAL_OFFSET
    );

    // Local work size: zero, beyond the device's work-item sizes, not
    // dividing the global size, or over the work-group cap.
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            1,
            core::ptr::null(),
            global.as_ptr(),
            zero_local.as_ptr(),
        ),
        sys::INVALID_WORK_GROUP_SIZE
    );
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            1,
            core::ptr::null(),
            item_too_big.as_ptr(),
            item_too_big.as_ptr(),
        ),
        sys::INVALID_WORK_ITEM_SIZE
    );
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            1,
            core::ptr::null(),
            indivisible.as_ptr(),
            four.as_ptr(),
        ),
        sys::INVALID_WORK_GROUP_SIZE
    );
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            3,
            core::ptr::null(),
            cube_global.as_ptr(),
            cube_local.as_ptr(),
        ),
        sys::INVALID_WORK_GROUP_SIZE
    );

    // Sizes fine, arguments unbound: the marshal step reports it last.
    assert_eq!(
        enqueue(
            queue,
            fill_kernel,
            1,
            core::ptr::null(),
            global.as_ptr(),
            local.as_ptr(),
        ),
        sys::INVALID_KERNEL_ARGS
    );

    // An argument-less kernel needs no arguments and launches.
    assert_eq!(
        enqueue(
            queue,
            nop_kernel,
            1,
            core::ptr::null(),
            global.as_ptr(),
            core::ptr::null(),
        ),
        sys::SUCCESS
    );

    // With arguments bound the same launch succeeds; the synchronous CPU
    // device has already completed the event by the time enqueue returns.
    set_buffer_arg(fill_kernel, 0, buffer);
    let mut event = EventHandle::from_raw(core::ptr::null_mut());
    assert_eq!(
        enqueue_event(queue, fill_kernel, &global, &mut event),
        sys::SUCCESS
    );
    assert!(event_slot.is_null());
    assert!(!event.is_null());
    assert_eq!(event_status(event), sys::COMPLETE);
    assert_eq!(unsafe { clWaitForEvents(1, &event) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseEvent(event) }, sys::SUCCESS);

    assert_eq!(unsafe { clReleaseKernel(fill_kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(nop_kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(buffer) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(other_queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(other_context) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

#[test]
fn set_kernel_arg_enforces_the_declaration_and_rebinds_buffers() {
    let device = device_handle();
    let context = make_context(device);
    let queue = make_queue(context, device);
    let program = make_program(context, device, "fill;scale;big");
    let fill_kernel = make_kernel(program, "fill");
    let scale_kernel = make_kernel(program, "scale");
    let big_kernel = make_kernel(program, "big");
    let first = make_buffer(context, 16 * 4);
    let second = make_buffer(context, 16 * 4);
    let slot: MemHandle = first;

    // The index is checked against the declaration before anything else.
    assert_eq!(
        set_arg_raw(
            fill_kernel,
            1,
            core::mem::size_of::<MemHandle>(),
            (&slot as *const MemHandle).cast(),
        ),
        sys::INVALID_ARG_INDEX
    );
    assert_eq!(
        set_arg_raw(
            fill_kernel,
            2,
            core::mem::size_of::<MemHandle>(),
            (&slot as *const MemHandle).cast(),
        ),
        sys::INVALID_ARG_INDEX
    );

    // A buffer slot demands exactly the handle size.
    assert_eq!(
        set_arg_raw(fill_kernel, 0, 4, (&slot as *const MemHandle).cast()),
        sys::INVALID_ARG_SIZE
    );
    assert_eq!(
        set_arg_raw(
            fill_kernel,
            0,
            core::mem::size_of::<MemHandle>() + 1,
            (&slot as *const MemHandle).cast(),
        ),
        sys::INVALID_ARG_SIZE
    );

    // A scalar slot demands exactly its declared size.
    let word = 7u32;
    assert_eq!(
        set_arg_raw(scale_kernel, 1, 8, (&word as *const u32).cast()),
        sys::INVALID_ARG_SIZE
    );
    assert_eq!(
        set_arg_raw(scale_kernel, 1, 3, (&word as *const u32).cast()),
        sys::INVALID_ARG_SIZE
    );

    // Null values are refused (local-memory arguments don't exist in this
    // profile), and so are null handles behind a valid pointer.
    assert_eq!(
        set_arg_raw(
            fill_kernel,
            0,
            core::mem::size_of::<MemHandle>(),
            core::ptr::null(),
        ),
        sys::INVALID_ARG_VALUE
    );
    let null_mem = MemHandle::from_raw(core::ptr::null_mut());
    assert_eq!(
        set_arg_raw(
            fill_kernel,
            0,
            core::mem::size_of::<MemHandle>(),
            (&null_mem as *const MemHandle).cast(),
        ),
        sys::INVALID_MEM_OBJECT
    );

    // The magic tag catches a live handle of the wrong object type — the
    // context is a readable object, so validation happens without any
    // undefined behavior.
    let as_mem = MemHandle::from_raw(context.as_raw());
    assert_eq!(
        set_arg_raw(
            fill_kernel,
            0,
            core::mem::size_of::<MemHandle>(),
            (&as_mem as *const MemHandle).cast(),
        ),
        sys::INVALID_MEM_OBJECT
    );

    // A null kernel handle fails before any argument logic.
    assert_eq!(
        set_arg_raw(
            KernelHandle::from_raw(core::ptr::null_mut()),
            0,
            core::mem::size_of::<MemHandle>(),
            (&slot as *const MemHandle).cast(),
        ),
        sys::INVALID_KERNEL
    );

    // A correctly sized scalar that still cannot live in one argument
    // slot is refused rather than truncated.
    let big = [0u8; 20];
    assert_eq!(
        set_arg_raw(big_kernel, 0, big.len(), big.as_ptr().cast()),
        sys::INVALID_ARG_SIZE
    );

    // Rebinding swaps the kernel's retained reference between buffers:
    // binding `first` retains it, rebinding to `second` releases it again.
    assert_eq!(mem_refs(first), 1);
    assert_eq!(mem_refs(second), 1);
    set_buffer_arg(fill_kernel, 0, first);
    assert_eq!(mem_refs(first), 2);
    assert_eq!(mem_refs(second), 1);
    set_buffer_arg(fill_kernel, 0, second);
    assert_eq!(mem_refs(first), 1);
    assert_eq!(mem_refs(second), 2);
    set_buffer_arg(fill_kernel, 0, first);
    assert_eq!(mem_refs(first), 2);
    assert_eq!(mem_refs(second), 1);

    // The bound arguments really execute: `first` starts zeroed, and the
    // scale kernel multiplies each element by the scalar in slot 1.
    let mut words = [0u32; 16];
    for (index, word) in words.iter_mut().enumerate() {
        *word = index as u32 + 1;
    }
    assert_eq!(write_buffer(queue, first, bytes_mut(&mut words)), sys::SUCCESS);
    set_buffer_arg(scale_kernel, 0, first);
    set_scalar_arg(scale_kernel, 1, 3);
    let global = [16usize];
    assert_eq!(
        enqueue_event(queue, scale_kernel, &global, core::ptr::null_mut()),
        sys::SUCCESS
    );
    let mut back = [0u32; 16];
    assert_eq!(read_buffer(queue, first, bytes_mut(&mut back)), sys::SUCCESS);
    for (index, value) in back.iter().enumerate() {
        assert_eq!(*value, (index as u32 + 1) * 3);
    }

    // Releasing the kernels returns the buffer references they owned:
    // `fill` and `scale` both hold `first` at this point.
    assert_eq!(mem_refs(first), 3);
    assert_eq!(unsafe { clReleaseKernel(fill_kernel) }, sys::SUCCESS);
    assert_eq!(mem_refs(first), 2);
    assert_eq!(unsafe { clReleaseKernel(scale_kernel) }, sys::SUCCESS);
    assert_eq!(mem_refs(first), 1);
    assert_eq!(unsafe { clReleaseKernel(big_kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(second) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(first) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

fn event_command_type(event: EventHandle) -> u32 {
    let mut size = 0usize;
    let code = unsafe {
        clGetEventInfo(
            event,
            sys::EVENT_COMMAND_TYPE,
            0,
            core::ptr::null_mut(),
            &mut size,
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(size, 4);
    let mut word = [0u8; 4];
    let code = unsafe {
        clGetEventInfo(
            event,
            sys::EVENT_COMMAND_TYPE,
            word.len(),
            word.as_mut_ptr().cast(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    u32::from_ne_bytes(word)
}

#[test]
fn successful_launches_cover_offsets_and_3d_geometry() {
    let device = device_handle();
    let context = make_context(device);
    let queue = make_queue(context, device);
    let program = make_program(context, device, "fill;store_ids");
    let fill_kernel = make_kernel(program, "fill");
    let store_kernel = make_kernel(program, "store_ids");

    // `fill` with a global offset: work-item ids start at the offset and
    // the returned event already describes the finished command.
    let fill_buffer = make_buffer(context, 16 * 4);
    set_buffer_arg(fill_kernel, 0, fill_buffer);
    let offset = [4usize];
    let global = [8usize];
    let local = [8usize];
    let mut event = EventHandle::from_raw(core::ptr::null_mut());
    let code = unsafe {
        clEnqueueNDRangeKernel(
            queue,
            fill_kernel,
            1,
            offset.as_ptr(),
            global.as_ptr(),
            local.as_ptr(),
            0,
            core::ptr::null(),
            &mut event,
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert!(!event.is_null());
    assert_eq!(event_status(event), sys::COMPLETE);
    assert_eq!(event_command_type(event), sys::COMMAND_NDRANGE_KERNEL);
    assert_eq!(unsafe { clWaitForEvents(1, &event) }, sys::SUCCESS);

    let mut words = [0u32; 16];
    assert_eq!(
        read_buffer(queue, fill_buffer, bytes_mut(&mut words)),
        sys::SUCCESS
    );
    for (index, value) in words.iter().enumerate() {
        let expected = if (4..12).contains(&index) { index as u32 } else { 0 };
        assert_eq!(*value, expected, "index {index}");
    }

    // The completed event can gate a later command through its wait list.
    let code = unsafe {
        clEnqueueNDRangeKernel(
            queue,
            fill_kernel,
            1,
            core::ptr::null(),
            global.as_ptr(),
            local.as_ptr(),
            1,
            &event,
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseEvent(event) }, sys::SUCCESS);

    // A 3-D launch over eight work-groups: ids are delivered row-major
    // and every work-item sees its exact coordinates.
    let store_buffer = make_buffer(context, 64 * 4);
    set_buffer_arg(store_kernel, 0, store_buffer);
    let dims = [4usize, 4, 4];
    let group = [2usize, 2, 2];
    let code = unsafe {
        clEnqueueNDRangeKernel(
            queue,
            store_kernel,
            3,
            core::ptr::null(),
            dims.as_ptr(),
            group.as_ptr(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    let mut words = [0u32; 64];
    assert_eq!(
        read_buffer(queue, store_buffer, bytes_mut(&mut words)),
        sys::SUCCESS
    );
    for gid2 in 0..4u32 {
        for gid1 in 0..4u32 {
            for gid0 in 0..4u32 {
                let index = (gid0 + gid1 * 4 + gid2 * 16) as usize;
                assert_eq!(words[index], gid0 | (gid1 << 8) | (gid2 << 16));
            }
        }
    }

    assert_eq!(unsafe { clFlush(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clFinish(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(store_kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(fill_kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(store_buffer) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(fill_buffer) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

#[test]
fn secondary_core_worker_claims_groups_of_every_launch() {
    let device = device_handle();
    let smp = *SMP.get().expect("smp device registered");
    // The secondary core enters its claim loop exactly once per process;
    // the thread never returns (it parks in `wfe` between epochs), so it
    // detaches when the process exits.
    static WORKER: Once = Once::new();
    WORKER.call_once(|| {
        std::thread::spawn(|| {
            smp.worker_loop(0);
        });
    });
    // The first enqueue cannot race the worker's registration.
    smp.wait_workers();

    let context = make_context(device);
    let queue = make_queue(context, device);
    let program = make_program(context, device, "fill;store_ids");
    let fill_kernel = make_kernel(program, "fill");
    let store_kernel = make_kernel(program, "store_ids");

    // Four work-groups per launch: the submitting core claims alongside
    // the worker, and repeated launches exercise the epoch handshake the
    // descriptor copy between them relies on.
    let fill_buffer = make_buffer(context, 64 * 4);
    set_buffer_arg(fill_kernel, 0, fill_buffer);
    let global = [64usize];
    let local = [16usize];
    for round in 0..3 {
        let code = unsafe {
            clEnqueueNDRangeKernel(
                queue,
                fill_kernel,
                1,
                core::ptr::null(),
                global.as_ptr(),
                local.as_ptr(),
                0,
                core::ptr::null(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, sys::SUCCESS);
        let mut words = [0u32; 64];
        assert_eq!(
            read_buffer(queue, fill_buffer, bytes_mut(&mut words)),
            sys::SUCCESS
        );
        for (index, value) in words.iter().enumerate() {
            assert_eq!(*value, index as u32, "round {round} index {index}");
        }
    }

    // An eight-group 3-D launch proves the geometry survives the hand-off
    // to the worker unchanged.
    let store_buffer = make_buffer(context, 64 * 4);
    set_buffer_arg(store_kernel, 0, store_buffer);
    let dims = [4usize, 4, 4];
    let group = [2usize, 2, 2];
    let code = unsafe {
        clEnqueueNDRangeKernel(
            queue,
            store_kernel,
            3,
            core::ptr::null(),
            dims.as_ptr(),
            group.as_ptr(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    let mut words = [0u32; 64];
    assert_eq!(
        read_buffer(queue, store_buffer, bytes_mut(&mut words)),
        sys::SUCCESS
    );
    for gid2 in 0..4u32 {
        for gid1 in 0..4u32 {
            for gid0 in 0..4u32 {
                let index = (gid0 + gid1 * 4 + gid2 * 16) as usize;
                assert_eq!(words[index], gid0 | (gid1 << 8) | (gid2 << 16));
            }
        }
    }

    assert_eq!(unsafe { clFinish(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(store_kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(fill_kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(store_buffer) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(fill_buffer) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

#[test]
fn four_threads_drive_independent_queues_without_cross_talk() {
    let handles: Vec<_> = (0..4u32)
        .map(|slot| {
            std::thread::spawn(move || {
                // Every thread owns its whole object graph: context,
                // queue, program, kernel and buffer. The device handle is
                // resolved here because raw handles are `!Send`.
                let device = device_handle();
                let context = make_context(device);
                let queue = make_queue(context, device);
                let program = make_program(context, device, "fill");
                let kernel = make_kernel(program, "fill");
                let buffer = make_buffer(context, 128 * 4);

                // A rejected launch must not disturb the queue: the same
                // kernel launches fine right after the malformed attempt.
                let probe = [8usize];
                assert_eq!(
                    enqueue(
                        queue,
                        kernel,
                        4,
                        core::ptr::null(),
                        probe.as_ptr(),
                        core::ptr::null(),
                    ),
                    sys::INVALID_WORK_DIMENSION
                );

                set_buffer_arg(kernel, 0, buffer);
                let offset = [(slot as usize) * 32];
                let global = [32usize];
                let local = [16usize];
                let code = unsafe {
                    clEnqueueNDRangeKernel(
                        queue,
                        kernel,
                        1,
                        offset.as_ptr(),
                        global.as_ptr(),
                        local.as_ptr(),
                        0,
                        core::ptr::null(),
                        core::ptr::null_mut(),
                    )
                };
                assert_eq!(code, sys::SUCCESS);
                assert_eq!(unsafe { clFinish(queue) }, sys::SUCCESS);

                // Only this thread's window was written; the rest of the
                // buffer proves no other thread's launch leaked into it.
                let mut words = [0u32; 128];
                assert_eq!(read_buffer(queue, buffer, bytes_mut(&mut words)), sys::SUCCESS);
                let base = slot * 32;
                for (index, value) in words.iter().enumerate() {
                    let inside = (base..base + 32).contains(&(index as u32));
                    let expected = if inside { index as u32 } else { 0 };
                    assert_eq!(*value, expected, "slot {slot} index {index}");
                }

                assert_eq!(unsafe { clReleaseKernel(kernel) }, sys::SUCCESS);
                assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
                assert_eq!(unsafe { clReleaseMemObject(buffer) }, sys::SUCCESS);
                assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
                assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("queue thread completes");
    }
}
