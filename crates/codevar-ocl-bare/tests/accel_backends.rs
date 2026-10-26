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

//! Back-end integration: the AMP descriptor ring and the accelerator's
//! MMIO register protocol, driven through the `cl*` boundary.
//!
//! The platform exposes a cooperative AMP device (the driver services its
//! ring inline) and a *remote* accelerator whose doorbell, opcode mirror
//! and completion interrupt are observable step by step: the tests play
//! the engine's firmware role by calling `service` themselves between
//! enqueue and reap. Every test serializes on one file-wide mutex — the
//! ring protocols assume a single engine consumer, so concurrent `service`
//! calls from parallel tests could interleave `head` advances — and each
//! test drains its device before releasing the guard, leaving the register
//! block in its reset state for whoever runs next.

// Integration test crate: AGENTS.md permits expect()/unwrap() in tests,
// but clippy.toml's allow-expect-in-tests only exempts `#[test]` bodies,
// not the fixtures shared by them.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use core::ffi::c_void;
use std::borrow::Cow;
use std::sync::{Once, OnceLock};

use codevar_ocl_bare::api::{
    clCreateBuffer, clCreateCommandQueue, clCreateContext, clCreateKernel, clCreateProgramWithBuiltInKernels,
    clEnqueueNDRangeKernel, clEnqueueReadBuffer, clFinish, clFlush, clGetDeviceIDs, clGetEventInfo,
    clGetPlatformIDs, clReleaseCommandQueue, clReleaseContext, clReleaseEvent, clReleaseKernel,
    clReleaseMemObject, clReleaseProgram, clSetKernelArg,
};
use codevar_ocl_bare::sys::{
    self, CommandQueueHandle, ContextHandle, DeviceHandle, EventHandle, KernelHandle, MemHandle,
    ProgramHandle,
};
use codevar_ocl_bare::{
    AccelDevice, AccelHandler, AmpDevice, ArgKind, Device, DeviceInfo, KernelDef, KernelModule,
    PlatformConfig, ServiceMode, SpirvBinding, WorkItem, default_accel_handler, init,
};

/// Registry kernel writing `global_id[0]` into `out[global_id[0]]` — the
/// one kernel the accelerator's opcode table also serves.
const FILL: KernelDef = KernelDef {
    name: Cow::Borrowed("fill"),
    func: fill,
    kinds: Cow::Borrowed(&[ArgKind::Buffer]),
    binding: SpirvBinding::NONE,
};

/// Argument-less kernel registered for the CPU/ring devices only: the
/// accelerator has no opcode for it, so `supports` must reject it.
const NOP: KernelDef = KernelDef {
    name: Cow::Borrowed("nop"),
    func: nop,
    kinds: Cow::Borrowed(&[]),
    binding: SpirvBinding::NONE,
};

static MODULE: KernelModule = KernelModule {
    name: "backends",
    kernels: &[FILL, NOP],
};

static MODULES: &[&KernelModule] = &[&MODULE];

/// The accelerator's opcode table: position + 1 is the opcode, so `fill`
/// runs as opcode 1 — the value the register block must mirror.
static ACCEL_HANDLERS: &[(&str, AccelHandler)] = &[("fill", default_accel_handler)];

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

unsafe extern "C" fn nop(_item: *mut WorkItem) {}

static AMP_INFO: DeviceInfo = DeviceInfo::new(
    "backends-amp",
    "codevar-tests",
    sys::DEVICE_TYPE_ACCELERATOR,
    1,
    64,
    1 << 20,
    4096,
    256,
);

static ACCEL_INFO: DeviceInfo = DeviceInfo::new(
    "backends-accel",
    "codevar-tests",
    sys::DEVICE_TYPE_CUSTOM,
    1,
    64,
    1 << 20,
    4096,
    256,
);

/// The AMP back end, for its service loop and in-flight count.
static AMP: OnceLock<&'static AmpDevice> = OnceLock::new();
/// The accelerator back end, whose `service` the tests drive directly.
static ACC: OnceLock<&'static AccelDevice> = OnceLock::new();

fn ensure_init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let amp = AmpDevice::simulated(AMP_INFO.clone(), ServiceMode::Cooperative).expect("amp device");
        let accel = AccelDevice::simulated(ACCEL_INFO.clone(), ACCEL_HANDLERS, ServiceMode::Remote)
            .expect("accel device");
        let devices: &'static [&'static dyn Device] =
            Box::leak(vec![amp as &dyn Device, accel as &dyn Device].into_boxed_slice());
        init(PlatformConfig {
            name: "backends-board",
            vendor: "codevar-tests",
            modules: MODULES,
            devices,
        })
        .expect("board registers");
        let _ = AMP.set(amp);
        let _ = ACC.set(accel);
    });
}

/// Serializes every test in this file on one file-wide mutex (see the
/// module docs). A poisoned lock is recovered from — the panic that
/// poisoned it already failed its test.
fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    match LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn amp_device() -> &'static AmpDevice {
    ensure_init();
    AMP.get().expect("amp registered")
}

fn accel_device() -> &'static AccelDevice {
    ensure_init();
    ACC.get().expect("accel registered")
}

/// Both platform devices in registration order: AMP first, accelerator
/// second (the order `clGetDeviceIDs` reports them in).
fn device_handles() -> [DeviceHandle; 2] {
    ensure_init();
    let mut platform = sys::PlatformHandle::from_raw(core::ptr::null_mut());
    let mut count = 0usize;
    assert_eq!(
        unsafe { clGetPlatformIDs(1, &mut platform, &mut count) },
        sys::SUCCESS
    );
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

fn make_buffer(context: ContextHandle, size: usize) -> MemHandle {
    let mut errcode = sys::SUCCESS;
    let buffer = unsafe { clCreateBuffer(context, 0, size, core::ptr::null_mut(), &mut errcode) };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!buffer.is_null());
    buffer
}

fn make_program(context: ContextHandle, devices: &[DeviceHandle], names: &str) -> ProgramHandle {
    let mut c_name = names.as_bytes().to_vec();
    c_name.push(0);
    let mut errcode = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBuiltInKernels(
            context,
            devices.len() as u32,
            devices.as_ptr(),
            c_name.as_ptr().cast(),
            &mut errcode,
        )
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

fn set_buffer_arg(kernel: KernelHandle, index: u32, buffer: MemHandle) {
    let code = unsafe {
        clSetKernelArg(
            kernel,
            index,
            core::mem::size_of::<MemHandle>(),
            (&buffer as *const MemHandle).cast::<c_void>(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
}

/// Enqueues a 1-D launch of `global` items with no wait list, returning
/// the command's event through `event` when non-null.
#[allow(clippy::too_many_arguments)]
fn enqueue_1d(
    queue: CommandQueueHandle,
    kernel: KernelHandle,
    global: usize,
    event: *mut EventHandle,
) -> i32 {
    let size = [global];
    unsafe {
        clEnqueueNDRangeKernel(
            queue,
            kernel,
            1,
            core::ptr::null(),
            size.as_ptr(),
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

/// The cooperative AMP device completes descriptors inline: by the time
/// `clEnqueueNDRangeKernel` returns, the event is finished, the ring slot
/// is back to `EMPTY` and nothing is in flight — yet every protocol
/// transition (PENDING → DONE → EMPTY) really happened.
#[test]
fn cooperative_amp_completes_the_descriptor_inline() {
    let _guard = serial_guard();
    let amp = amp_device();
    let devices = device_handles();
    let context = make_context(&devices);
    let queue = make_queue(context, devices[0]);
    let program = make_program(context, &devices, "fill;nop");
    let kernel = make_kernel(program, "fill");
    let buffer = make_buffer(context, 64 * 4);
    set_buffer_arg(kernel, 0, buffer);

    let mut event = EventHandle::from_raw(core::ptr::null_mut());
    assert_eq!(enqueue_1d(queue, kernel, 32, &mut event), sys::SUCCESS);
    assert!(!event.is_null());
    assert_eq!(event_status(event), sys::COMPLETE);
    assert_eq!(amp.inflight(), 0);

    let mut words = [0u32; 64];
    assert_eq!(read_buffer(queue, buffer, bytes_mut(&mut words)), sys::SUCCESS);
    for (index, value) in words.iter().enumerate() {
        let expected = if index < 32 { index as u32 } else { 0 };
        assert_eq!(*value, expected, "index {index}");
    }

    assert_eq!(unsafe { clFlush(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clFinish(queue) }, sys::SUCCESS);

    assert_eq!(unsafe { clReleaseEvent(event) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(buffer) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

/// The remote accelerator walks the full register protocol: the enqueue
/// rings the doorbell and leaves the event running, `service` (the
/// engine's firmware, played by the test) mirrors opcode and status and
/// raises the completion interrupt, and only the host's reap —
/// `clFlush`/`clFinish` — completes the event and clears the IRQ.
///
/// The register words are the private `AccelRegs` constants, so they are
/// asserted as raw values: `status` 0=IDLE 1=RUNNING 2=DONE 3=ERROR,
/// `control` bit 0 the doorbell, `irq_status` bit 0 the completion flag.
#[test]
fn remote_accel_walks_the_register_protocol_to_completion() {
    let _guard = serial_guard();
    let accel = accel_device();
    let regs = accel.regs();
    let devices = device_handles();
    let context = make_context(&devices);
    let queue = make_queue(context, devices[1]);
    let program = make_program(context, &devices, "fill");
    let kernel = make_kernel(program, "fill");
    let buffer = make_buffer(context, 64 * 4);
    set_buffer_arg(kernel, 0, buffer);

    // Reset state: no doorbell, engine idle, no interrupt pending.
    assert_eq!(regs.control(), 0);
    assert_eq!(regs.status(), 0);
    assert_eq!(regs.irq_status(), 0);

    // The enqueue publishes the descriptor and rings the doorbell; the
    // remote engine has not run, so the event is still in flight.
    let mut event = EventHandle::from_raw(core::ptr::null_mut());
    assert_eq!(enqueue_1d(queue, kernel, 32, &mut event), sys::SUCCESS);
    assert_eq!(event_status(event), sys::RUNNING);
    assert_eq!(regs.control(), 1, "doorbell raised by the host");
    assert_eq!(accel.inflight(), 1);

    // The kernel has not executed yet — the descriptor waits in the ring.
    let mut words = [0u8; 64 * 4];
    assert_eq!(read_buffer(queue, buffer, &mut words), sys::SUCCESS);
    assert!(words.iter().all(|byte| *byte == 0));

    // The engine's side of the protocol: consume the descriptor, mirror
    // the running opcode, execute, post DONE and raise the interrupt.
    accel.service();
    assert_eq!(regs.control(), 0, "the engine acked the doorbell");
    assert_eq!(regs.status(), 0, "engine drained its queue back to IDLE");
    assert_eq!(regs.opcode(), 1, "fill is opcode 1 in the handler table");
    assert_eq!(regs.irq_status(), 1, "completion interrupt raised");
    assert_eq!(event_status(event), sys::RUNNING, "host has not reaped yet");
    assert_eq!(accel.inflight(), 1);

    let mut words = [0u32; 64];
    assert_eq!(read_buffer(queue, buffer, bytes_mut(&mut words)), sys::SUCCESS);
    for (index, value) in words.iter().enumerate() {
        let expected = if index < 32 { index as u32 } else { 0 };
        assert_eq!(*value, expected, "index {index}");
    }

    // The host's reap completes the event, empties the in-flight list and
    // clears the interrupt — the register block is back in reset state.
    assert_eq!(unsafe { clFlush(queue) }, sys::SUCCESS);
    assert_eq!(event_status(event), sys::COMPLETE);
    assert_eq!(regs.irq_status(), 0, "reaper cleared the interrupt");
    assert_eq!(accel.inflight(), 0);

    assert_eq!(unsafe { clFinish(queue) }, sys::SUCCESS);
    assert_eq!(regs.control(), 0);
    assert_eq!(regs.status(), 0);
    assert_eq!(regs.irq_status(), 0);

    assert_eq!(unsafe { clReleaseEvent(event) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(buffer) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

/// `supports` gates kernels per back end at enqueue time: the accelerator
/// only accepts names in its opcode table, the AMP device (like every
/// CPU-style back end) accepts the whole registry.
#[test]
fn opcode_table_decides_which_kernels_each_backend_accepts() {
    let _guard = serial_guard();
    let accel = accel_device();
    let amp = amp_device();
    let devices = device_handles();

    // Opcode derivation: the table position plus one, `None` for names
    // the engine has no handler for.
    assert_eq!(accel.opcode_of("fill"), Some(1));
    assert_eq!(accel.opcode_of("nop"), None);
    assert_eq!(accel.opcode_of("ghost"), None);

    // Accelerator queue: `nop` has no opcode, so the enqueue is refused
    // before any launch shape is even considered (the zero global size
    // below would otherwise be the reported error).
    let accel_context = make_context(&devices);
    let accel_queue = make_queue(accel_context, devices[1]);
    let program = make_program(accel_context, &devices, "fill;nop");
    let fill_kernel = make_kernel(program, "fill");
    let nop_kernel = make_kernel(program, "nop");
    let buffer = make_buffer(accel_context, 64 * 4);
    set_buffer_arg(fill_kernel, 0, buffer);
    assert_eq!(
        enqueue_1d(accel_queue, nop_kernel, 0, core::ptr::null_mut()),
        sys::INVALID_KERNEL
    );

    // `fill` is in the table and runs; drain the engine fully.
    assert_eq!(
        enqueue_1d(accel_queue, fill_kernel, 16, core::ptr::null_mut()),
        sys::SUCCESS
    );
    accel.service();
    assert_eq!(unsafe { clFinish(accel_queue) }, sys::SUCCESS);
    assert_eq!(accel.inflight(), 0);
    assert_eq!(accel.regs().irq_status(), 0);

    // The AMP device accepts the very same `nop` kernel.
    let amp_context = make_context(&devices[..1]);
    let amp_queue = make_queue(amp_context, devices[0]);
    let amp_program = make_program(amp_context, &devices[..1], "nop");
    let amp_nop = make_kernel(amp_program, "nop");
    assert_eq!(
        enqueue_1d(amp_queue, amp_nop, 4, core::ptr::null_mut()),
        sys::SUCCESS
    );
    assert_eq!(unsafe { clFinish(amp_queue) }, sys::SUCCESS);
    assert_eq!(amp.inflight(), 0);

    assert_eq!(unsafe { clReleaseKernel(amp_nop) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseProgram(amp_program) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(amp_queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(amp_context) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(nop_kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(fill_kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(buffer) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(accel_queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(accel_context) }, sys::SUCCESS);
}

/// One thread per back end, running at the same time: the cooperative AMP
/// thread completes every descriptor inline while the remote-accelerator
/// thread walks enqueue → service → reap cycles. The two protocols never
/// share a ring, so both must observe correct results throughout. The
/// file-wide guard is held for the whole test so no *other* test races the
/// accelerator's engine role.
#[test]
fn both_backends_run_concurrently_without_interference() {
    let _guard = serial_guard();
    let mut handles = Vec::new();

    // The AMP peer: pure inline completion, many launches back to back.
    handles.push(std::thread::spawn(|| {
        let devices = device_handles();
        let context = make_context(&devices[..1]);
        let queue = make_queue(context, devices[0]);
        let program = make_program(context, &devices[..1], "fill");
        let kernel = make_kernel(program, "fill");
        let buffer = make_buffer(context, 64 * 4);
        set_buffer_arg(kernel, 0, buffer);
        for round in 0..24usize {
            let global = 8 + round % 48;
            assert_eq!(
                enqueue_1d(queue, kernel, global, core::ptr::null_mut()),
                sys::SUCCESS
            );
            let mut words = [0u32; 64];
            assert_eq!(read_buffer(queue, buffer, bytes_mut(&mut words)), sys::SUCCESS);
            for (index, value) in words.iter().enumerate() {
                let expected = if index < global { index as u32 } else { 0 };
                assert_eq!(*value, expected, "amp round {round} index {index}");
            }
        }
        assert_eq!(unsafe { clFinish(queue) }, sys::SUCCESS);
        assert_eq!(unsafe { clReleaseKernel(kernel) }, sys::SUCCESS);
        assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
        assert_eq!(unsafe { clReleaseMemObject(buffer) }, sys::SUCCESS);
        assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
        assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
    }));

    // The accelerator: every round is a full protocol walk, with the
    // thread itself playing the engine between enqueue and reap.
    handles.push(std::thread::spawn(|| {
        let accel = accel_device();
        let devices = device_handles();
        let context = make_context(&devices[1..]);
        let queue = make_queue(context, devices[1]);
        let program = make_program(context, &devices[1..], "fill");
        let kernel = make_kernel(program, "fill");
        let buffer = make_buffer(context, 64 * 4);
        set_buffer_arg(kernel, 0, buffer);
        for round in 0..24usize {
            let global = 8 + round % 48;
            let mut event = EventHandle::from_raw(core::ptr::null_mut());
            assert_eq!(enqueue_1d(queue, kernel, global, &mut event), sys::SUCCESS);
            // Remote mode: the event stays in flight until the engine
            // (this thread) services the ring and the host reaps.
            assert_eq!(event_status(event), sys::RUNNING);
            assert_eq!(accel.inflight(), 1);
            accel.service();
            assert_eq!(unsafe { clFinish(queue) }, sys::SUCCESS);
            assert_eq!(event_status(event), sys::COMPLETE);
            assert_eq!(accel.inflight(), 0);
            assert_eq!(accel.regs().irq_status(), 0);
            let mut words = [0u32; 64];
            assert_eq!(read_buffer(queue, buffer, bytes_mut(&mut words)), sys::SUCCESS);
            for (index, value) in words.iter().enumerate() {
                let expected = if index < global { index as u32 } else { 0 };
                assert_eq!(*value, expected, "accel round {round} index {index}");
            }
            assert_eq!(unsafe { clReleaseEvent(event) }, sys::SUCCESS);
        }
        assert_eq!(unsafe { clReleaseKernel(kernel) }, sys::SUCCESS);
        assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
        assert_eq!(unsafe { clReleaseMemObject(buffer) }, sys::SUCCESS);
        assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
        assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
    }));

    for handle in handles {
        handle.join().expect("backend thread completes");
    }

    // Both engines are quiescent and the register block is in reset state.
    assert_eq!(amp_device().inflight(), 0);
    assert_eq!(accel_device().inflight(), 0);
    let regs = accel_device().regs();
    assert_eq!(regs.control(), 0);
    assert_eq!(regs.status(), 0);
    assert_eq!(regs.irq_status(), 0);
}
