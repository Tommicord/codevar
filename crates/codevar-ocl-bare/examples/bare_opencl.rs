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

//! OpenCL on real bare metal: the embedded driver as one flat image on
//! QEMU's `mps2-an385` (a Cortex-M3), driven through the `cl*` entry points
//! and reporting through the semihosting channel.
//!
//! Run it with the runner `.cargo/config.toml` defines for
//! `cfg(target_arch = "arm", target_os = "none")`:
//!
//! ```text
//! cargo run -p codevar-ocl-bare --example bare_opencl \
//!     --target thumbv7m-none-eabi --release
//! ```
//!
//! The image boots from address 0, every console line leaves through
//! semihosting, and the emulation's exit status is the verdict of the
//! checks below: `0` once every check passed, `1` otherwise — a failing
//! check prints its reason on stderr first. What runs on the target:
//!
//! - platform and device discovery, including the `CL_PLATFORM_NAME` and
//!   `CL_DEVICE_NAME` strings the image publishes;
//! - the embedded profile's missing online compiler:
//!   `clCreateProgramWithSource` must answer `CL_COMPILER_NOT_AVAILABLE`;
//! - a registry program (`clCreateProgramWithBuiltInKernels`) with a
//!   buffer-only kernel and a buffer-plus-scalar kernel, launched over an
//!   ND-range, waited on through its event, and verified by reading the
//!   buffer back;
//! - the release path of every object the checklist created.
//!
//! On a hosted target the same checklist runs in-process: that is what the
//! `fn main` below is for, so an ordinary `cargo test --workspace` still
//! compiles this file, while the bare-metal half is the `_start` image the
//! QEMU runner boots.

#![cfg_attr(target_os = "none", no_std)]
#![cfg_attr(target_os = "none", no_main)]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::c_void;

use codevar_logger::{log_error, log_info, log_success};
use codevar_ocl_bare::api::{
    clCreateBuffer, clCreateCommandQueue, clCreateContext, clCreateKernel, clCreateProgramWithBuiltInKernels,
    clCreateProgramWithSource, clEnqueueNDRangeKernel, clEnqueueReadBuffer, clEnqueueWriteBuffer, clFinish,
    clGetDeviceIDs, clGetDeviceInfo, clGetEventInfo, clGetKernelInfo, clGetPlatformIDs, clGetPlatformInfo,
    clReleaseCommandQueue, clReleaseContext, clReleaseEvent, clReleaseKernel, clReleaseMemObject,
    clReleaseProgram, clSetKernelArg, clWaitForEvents,
};
use codevar_ocl_bare::sys::{self, CommandQueueHandle, DeviceHandle, EventHandle, KernelHandle, MemHandle};
use codevar_ocl_bare::{
    ArgKind, Device, DeviceInfo, KernelDef, KernelModule, PlatformConfig, SmpDevice, SpirvBinding, WorkItem,
    init,
};

/// Elements per launch: one `u32` per work-item of the ND-range.
const ELEMENTS: usize = 64;

/// Scalar argument of the `scale` kernel.
const FACTOR: u32 = 3;

/// Value every word holds after the host wrote it and before any kernel
/// ran — an element still carrying it after a launch was not covered.
const SENTINEL: u32 = 0xDEAD_BEEF;

/// `CL_PLATFORM_NAME` published by, and checked back from, the image.
const PLATFORM_NAME: &str = "qemu-mps2-an385";

/// `CL_DEVICE_NAME` of the single SMP back end.
const DEVICE_NAME: &str = "mps2-cortex-m3";

/// `CL_PLATFORM_VENDOR` and `CL_DEVICE_VENDOR`.
const VENDOR: &str = "codevar";

/// The board description of QEMU's single Cortex-M3: one compute unit, so
/// the claim loop runs entirely on the submitting core.
static SMP_INFO: DeviceInfo = DeviceInfo::new(
    DEVICE_NAME,
    VENDOR,
    sys::DEVICE_TYPE_CPU,
    1,
    64,
    1 << 20,
    4096,
    256,
);

/// Registry kernel writing `global_id[0]` into `out[global_id[0]]` — the
/// buffer-only argument path.
const FILL: KernelDef = KernelDef {
    name: Cow::Borrowed("fill"),
    func: fill,
    kinds: Cow::Borrowed(&[ArgKind::Buffer]),
    binding: SpirvBinding::NONE,
};

/// Registry kernel multiplying `out[global_id[0]]` by a scalar — the
/// buffer-plus-scalar marshaling path.
const SCALE: KernelDef = KernelDef {
    name: Cow::Borrowed("scale"),
    func: scale,
    kinds: Cow::Borrowed(&[ArgKind::Buffer, ArgKind::Scalar(4)]),
    binding: SpirvBinding::NONE,
};

/// The image's kernel module: what `clCreateProgramWithBuiltInKernels`
/// resolves `fill;scale` against.
static MODULE: KernelModule = KernelModule {
    name: "bare-opencl",
    kernels: &[FILL, SCALE],
};

/// The registry list the platform publishes.
static MODULES: &[&KernelModule] = &[&MODULE];

/// `out[gid] = gid`, for every work-item of the launch.
///
/// # Safety
///
/// The claim loop hands every work-item a live argument block whose slot 0
/// is the buffer bound by `clSetKernelArg`; the launch's index space is
/// the buffer's element count, which the bounds check below re-verifies so
/// a mis-shaped launch fails closed instead of writing past the buffer.
unsafe extern "C" fn fill(item: *mut WorkItem) {
    // SAFETY: the claim loop guarantees a live work-item pointer here.
    let item = unsafe { &*item };
    let index = item.global_id[0] as usize;
    if index >= ELEMENTS {
        return;
    }
    // SAFETY: `index < ELEMENTS`, and slot 0 holds the `ELEMENTS`-word
    // buffer argument of this launch.
    let out = unsafe { item.arg_ptr(0) }.cast::<u32>();
    if out.is_null() {
        return;
    }
    // SAFETY: as above — `out` points at `ELEMENTS` live words.
    unsafe { *out.add(index) = item.global_id[0] };
}

/// `out[gid] *= factor`, with `factor` read from the scalar argument.
///
/// # Safety
///
/// Same contract as [`fill`]: slot 0 is the `ELEMENTS`-word buffer and
/// slot 1 the four-byte scalar `clSetKernelArg` bound.
unsafe extern "C" fn scale(item: *mut WorkItem) {
    // SAFETY: the claim loop guarantees a live work-item pointer here.
    let item = unsafe { &*item };
    let index = item.global_id[0] as usize;
    if index >= ELEMENTS {
        return;
    }
    // SAFETY: `index < ELEMENTS` and slot 0 is the buffer argument.
    let out = unsafe { item.arg_ptr(0) }.cast::<u32>();
    if out.is_null() {
        return;
    }
    // SAFETY: slot 1 belongs to this work-item's argument block and holds
    // the `u32` scalar bound as argument 1.
    let factor = unsafe { item.arg_u32(1) };
    // SAFETY: `out` points at `ELEMENTS` live words and `index` is in
    // range; the multiply wraps, as OpenCL integer arithmetic does.
    unsafe { *out.add(index) = (*out.add(index)).wrapping_mul(factor) };
}

/// Turns an OpenCL status into the checklist's error type.
fn ok(step: &str, code: i32) -> Result<(), String> {
    if code == sys::SUCCESS {
        Ok(())
    } else {
        Err(format!("{step} returned CL status {code}"))
    }
}

/// Checks a creating `cl*` call — status `CL_SUCCESS` *and* a non-null
/// handle — and yields the handle, so neither half of a broken creation
/// slips through.
macro_rules! created {
    ($step:expr, $code:expr, $handle:expr) => {{
        ok($step, $code)?;
        if $handle.is_null() {
            return Err(format!("{} returned a null handle", $step));
        }
        $handle
    }};
}

/// Reads a NUL-terminated string property out of a `cl*Info` query.
///
/// The two-call protocol (size first, then bytes) is the whole `cl*Info`
/// contract; the buffer is sized from the first call, and a short or
/// non-UTF-8 answer becomes an error instead of a panic.
fn query_string(
    step: &str,
    mut query: impl FnMut(usize, *mut c_void, *mut usize) -> i32,
) -> Result<String, String> {
    let mut size = 0usize;
    ok(step, query(0, core::ptr::null_mut(), &mut size))?;
    if size == 0 {
        return Err(format!("{step} reported an empty property"));
    }
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(size)
        .map_err(|_| format!("{step}: out of memory"))?;
    buffer.resize(size, 0);
    let mut written = 0usize;
    ok(step, query(size, buffer.as_mut_ptr().cast(), &mut written))?;
    let end = written.min(size);
    let bytes = buffer
        .get(..end)
        .ok_or_else(|| format!("{step}: wrote {written} of {size} bytes"))?;
    let bytes = bytes.strip_suffix(&[0]).unwrap_or(bytes);
    core::str::from_utf8(bytes)
        .map(String::from)
        .map_err(|_| format!("{step}: non-UTF-8 string"))
}

/// Reads a `u32` property out of a `cl*Info` query.
fn query_u32(
    step: &str,
    mut query: impl FnMut(usize, *mut c_void, *mut usize) -> i32,
) -> Result<u32, String> {
    let mut size = 0usize;
    ok(step, query(0, core::ptr::null_mut(), &mut size))?;
    let mut word = [0u8; core::mem::size_of::<u32>()];
    if size != word.len() {
        return Err(format!("{step} reported {size} bytes, expected {}", word.len()));
    }
    let mut written = 0usize;
    ok(step, query(word.len(), word.as_mut_ptr().cast(), &mut written))?;
    if written != word.len() {
        return Err(format!("{step} wrote {written} of {} bytes", word.len()));
    }
    Ok(u32::from_ne_bytes(word))
}

/// Reads `CL_EVENT_COMMAND_EXECUTION_STATUS` out of an event.
fn event_status(event: EventHandle) -> Result<i32, String> {
    let mut word = [0u8; core::mem::size_of::<i32>()];
    ok("clGetEventInfo", unsafe {
        clGetEventInfo(
            event,
            sys::EVENT_COMMAND_EXECUTION_STATUS,
            word.len(),
            word.as_mut_ptr().cast(),
            core::ptr::null_mut(),
        )
    })?;
    Ok(i32::from_ne_bytes(word))
}

/// The byte window a read/write command copies over `words`.
fn as_bytes(words: &[u32]) -> &[u8] {
    // SAFETY: `u32` has no padding, so the reinterpreted slice addresses
    // the same initialized bytes — `words.len() * 4` of them.
    unsafe { core::slice::from_raw_parts(words.as_ptr().cast(), words.len() * core::mem::size_of::<u32>()) }
}

/// The mutable counterpart of [`as_bytes`].
fn as_bytes_mut(words: &mut [u32]) -> &mut [u8] {
    // SAFETY: as `as_bytes`; the exclusive borrow of `words` lasts exactly
    // as long as the returned window, so no aliasing can arise.
    unsafe {
        core::slice::from_raw_parts_mut(
            words.as_mut_ptr().cast(),
            words.len() * core::mem::size_of::<u32>(),
        )
    }
}

/// Copies `data` into `buffer` with a blocking write command.
fn write_buffer(queue: CommandQueueHandle, buffer: MemHandle, data: &[u8]) -> Result<(), String> {
    ok("clEnqueueWriteBuffer", unsafe {
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
    })
}

/// Copies `buffer` out into `out` with a blocking read command.
fn read_buffer(queue: CommandQueueHandle, buffer: MemHandle, out: &mut [u8]) -> Result<(), String> {
    ok("clEnqueueReadBuffer", unsafe {
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
    })
}

/// Binds `buffer` as argument `index` of `kernel`.
fn set_buffer_arg(kernel: KernelHandle, index: u32, buffer: MemHandle) -> Result<(), String> {
    ok("clSetKernelArg(buffer)", unsafe {
        clSetKernelArg(
            kernel,
            index,
            core::mem::size_of::<MemHandle>(),
            (&buffer as *const MemHandle).cast(),
        )
    })
}

/// Binds the `u32` `value` as argument `index` of `kernel`.
fn set_scalar_arg(kernel: KernelHandle, index: u32, value: u32) -> Result<(), String> {
    ok("clSetKernelArg(scalar)", unsafe {
        clSetKernelArg(
            kernel,
            index,
            core::mem::size_of::<u32>(),
            (&value as *const u32).cast(),
        )
    })
}

/// Enqueues a 1-D launch over `global`, optionally producing an event.
fn launch(
    queue: CommandQueueHandle,
    kernel: KernelHandle,
    global: &[usize],
    event: *mut EventHandle,
) -> Result<(), String> {
    ok("clEnqueueNDRangeKernel", unsafe {
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
    })
}

/// Checks `words` against `expect`, naming the first element that differs.
fn verify(step: &str, words: &[u32], expect: impl Fn(usize) -> u32) -> Result<(), String> {
    for (index, &value) in words.iter().enumerate() {
        let wanted = expect(index);
        if value != wanted {
            return Err(format!("{step}: out[{index}] = {value:#x}, expected {wanted:#x}"));
        }
    }
    Ok(())
}

/// Publishes the board and runs every check of the embedded profile.
///
/// # Errors
///
/// The first check that fails, named with the `cl*` entry point (and its
/// `CL_*` status) or the element that held the wrong value.
fn run() -> Result<(), String> {
    let smp = SmpDevice::new(SMP_INFO.clone(), 1).map_err(|error| format!("SMP device: {error}"))?;
    let devices: &'static [&'static dyn Device] = Box::leak(Box::new([smp as &dyn Device]));
    init(PlatformConfig {
        name: PLATFORM_NAME,
        vendor: VENDOR,
        modules: MODULES,
        devices,
    })
    .map_err(|error| format!("platform publication: {error}"))?;

    // Discovery: exactly one platform and one device, both carrying the
    // names the image publishes.
    let mut platform = sys::PlatformHandle::from_raw(core::ptr::null_mut());
    let mut found = 0usize;
    ok("clGetPlatformIDs", unsafe {
        clGetPlatformIDs(1, &mut platform, &mut found)
    })?;
    if found != 1 || platform.is_null() {
        return Err(format!("platform discovery reported {found} platform(s)"));
    }
    let platform_name = query_string(
        "clGetPlatformInfo(PLATFORM_NAME)",
        |size, value, written| unsafe {
            clGetPlatformInfo(platform, sys::PLATFORM_NAME, size, value, written)
        },
    )?;
    if platform_name != PLATFORM_NAME {
        return Err(format!(
            "platform name {platform_name:?}, expected {PLATFORM_NAME:?}"
        ));
    }

    let mut device = DeviceHandle::from_raw(core::ptr::null_mut());
    found = 0;
    ok("clGetDeviceIDs", unsafe {
        clGetDeviceIDs(platform, sys::DEVICE_TYPE_ALL, 1, &mut device, &mut found)
    })?;
    if found != 1 || device.is_null() {
        return Err(format!("device enumeration reported {found} device(s)"));
    }
    let device_name = query_string("clGetDeviceInfo(DEVICE_NAME)", |size, value, written| unsafe {
        clGetDeviceInfo(device, sys::DEVICE_NAME, size, value, written)
    })?;
    if device_name != DEVICE_NAME {
        return Err(format!("device name {device_name:?}, expected {DEVICE_NAME:?}"));
    }
    log_info!("discovered {platform_name} / {device_name}");

    // Context and queue: the in-order queue the embedded profile allows.
    let mut code = sys::SUCCESS;
    let context = unsafe {
        clCreateContext(
            core::ptr::null(),
            1,
            &device,
            core::ptr::null(),
            core::ptr::null_mut(),
            &mut code,
        )
    };
    let context = created!("clCreateContext", code, context);
    code = sys::SUCCESS;
    let queue = unsafe { clCreateCommandQueue(context, device, 0, &mut code) };
    let queue = created!("clCreateCommandQueue", code, queue);

    // The embedded profile has no online compiler: source programs must be
    // refused with CL_COMPILER_NOT_AVAILABLE rather than silently parsed.
    let source = b"void noop(void) {}\0";
    let sources = [source.as_ptr().cast::<core::ffi::c_char>()];
    code = sys::SUCCESS;
    let from_source =
        unsafe { clCreateProgramWithSource(context, 1, sources.as_ptr(), core::ptr::null(), &mut code) };
    if !from_source.is_null() || code != sys::COMPILER_NOT_AVAILABLE {
        return Err(format!(
            "clCreateProgramWithSource answered status {code}, expected \
             CL_COMPILER_NOT_AVAILABLE ({})",
            sys::COMPILER_NOT_AVAILABLE
        ));
    }

    // The registry-backed program the image was built with.
    code = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBuiltInKernels(context, 1, &device, b"fill;scale\0".as_ptr().cast(), &mut code)
    };
    let program = created!("clCreateProgramWithBuiltInKernels", code, program);

    code = sys::SUCCESS;
    let buffer = unsafe {
        clCreateBuffer(
            context,
            sys::MEM_READ_WRITE,
            ELEMENTS * 4,
            core::ptr::null_mut(),
            &mut code,
        )
    };
    let buffer = created!("clCreateBuffer", code, buffer);

    // The copy path first: what the host writes must read back untouched
    // before any kernel runs.
    let words = [SENTINEL; ELEMENTS];
    write_buffer(queue, buffer, as_bytes(&words))?;
    let mut readback = [0u32; ELEMENTS];
    read_buffer(queue, buffer, as_bytes_mut(&mut readback))?;
    verify("buffer round-trip", &readback, |_| SENTINEL)?;

    // The buffer-only kernel, waited on through its event: the queue runs
    // inline on this device, so the event must already read CL_COMPLETE
    // once clWaitForEvents returns.
    code = sys::SUCCESS;
    let fill_kernel = unsafe { clCreateKernel(program, b"fill\0".as_ptr().cast(), &mut code) };
    let fill_kernel = created!("clCreateKernel(fill)", code, fill_kernel);
    let declared = query_u32(
        "clGetKernelInfo(KERNEL_NUM_ARGS)",
        |size, value, written| unsafe {
            clGetKernelInfo(fill_kernel, sys::KERNEL_NUM_ARGS, size, value, written)
        },
    )?;
    if declared != 1 {
        return Err(format!("fill declares {declared} argument(s), expected 1"));
    }
    set_buffer_arg(fill_kernel, 0, buffer)?;
    let global = [ELEMENTS];
    let mut event = EventHandle::from_raw(core::ptr::null_mut());
    launch(queue, fill_kernel, &global, &mut event)?;
    ok("clWaitForEvents", unsafe { clWaitForEvents(1, &event) })?;
    let status = event_status(event)?;
    if status != sys::COMPLETE {
        return Err(format!("fill launch event retired with status {status}"));
    }
    read_buffer(queue, buffer, as_bytes_mut(&mut readback))?;
    verify("fill", &readback, |index| index as u32)?;

    // The buffer-plus-scalar kernel over the same buffer, then the queue's
    // barrier before the last read.
    code = sys::SUCCESS;
    let scale_kernel = unsafe { clCreateKernel(program, b"scale\0".as_ptr().cast(), &mut code) };
    let scale_kernel = created!("clCreateKernel(scale)", code, scale_kernel);
    set_buffer_arg(scale_kernel, 0, buffer)?;
    set_scalar_arg(scale_kernel, 1, FACTOR)?;
    launch(queue, scale_kernel, &global, core::ptr::null_mut())?;
    ok("clFinish", unsafe { clFinish(queue) })?;
    read_buffer(queue, buffer, as_bytes_mut(&mut readback))?;
    verify("scale", &readback, |index| (index as u32).wrapping_mul(FACTOR))?;

    // Release every object the checklist created; a ref-count leak or a
    // double free shows up here as a bad status or a fault in QEMU.
    ok("clReleaseEvent", unsafe { clReleaseEvent(event) })?;
    ok("clReleaseKernel(fill)", unsafe { clReleaseKernel(fill_kernel) })?;
    ok("clReleaseKernel(scale)", unsafe { clReleaseKernel(scale_kernel) })?;
    ok("clReleaseMemObject", unsafe { clReleaseMemObject(buffer) })?;
    ok("clReleaseProgram", unsafe { clReleaseProgram(program) })?;
    ok("clReleaseCommandQueue", unsafe { clReleaseCommandQueue(queue) })?;
    ok("clReleaseContext", unsafe { clReleaseContext(context) })?;
    Ok(())
}

/// Runs the checklist and reports the verdict on the console.
///
/// Returns the process status the runner must observe: `0` when every
/// check passed and `1` otherwise — the value the bare-metal entry point
/// hands to `codevar_target::semihosting::exit`, which turns it into
/// QEMU's own exit status.
fn run_and_report() -> i32 {
    match run() {
        Ok(()) => {
            log_success!("PASS: {ELEMENTS} elements verified through the cl* entry points");
            0
        }
        Err(reason) => {
            log_error!("FAIL: {reason}");
            1
        }
    }
}

// The vector table and `Reset` stub of the bare-metal image; expands to
// nothing on any other target, where the C runtime supplies both.
#[cfg(target_os = "none")]
codevar_rt::vector_table!();

/// Size of the static heap handed to the allocator: there is no `mmap` to
/// grow with, so the region is part of the image. 64 KiB covers the
/// driver's object graph and buffers with room to spare, and the board
/// has 4 MiB of SRAM to hold it.
#[cfg(target_os = "none")]
const HEAP_REGION_BYTES: usize = 0x1_0000;

/// Backing store of that heap.
#[cfg(target_os = "none")]
static mut HEAP_REGION: [u8; HEAP_REGION_BYTES] = [0; HEAP_REGION_BYTES];

/// The allocator the image runs on: a fixed TLSF region, the bare-metal
/// counterpart of the growable heap a Unix process gets.
#[cfg(target_os = "none")]
#[global_allocator]
static HEAP: codevar_tlsf_alloc::LockedTlsf = codevar_tlsf_alloc::LockedTlsf::new();

/// Registers the static heap region before the first allocation runs.
///
/// Must be the first thing the entry point does: `LockedTlsf` serves no
/// request until a region is contributed, and `run` allocates as soon as
/// the platform is published.
#[cfg(target_os = "none")]
fn init_heap() {
    let start = core::ptr::addr_of_mut!(HEAP_REGION).cast::<u8>();
    // SAFETY: `HEAP_REGION` exists only to back `HEAP`; nothing else
    // aliases it, the region spans the whole array, and this runs before
    // any allocation is attempted.
    let registered = unsafe { HEAP.lock().add_region(start, HEAP_REGION_BYTES) };
    if registered.is_err() {
        log_error!("heap region rejected by the allocator, cannot continue");
        codevar_target::halt();
    }
}

/// Freestanding entry: bring up the heap, run the checklist, and hand the
/// verdict to the debug agent — `SYS_EXIT` reporting `0` or `1`, which
/// QEMU turns into the status `cargo run` sees. Without an agent the call
/// degrades to halting the core, so the image stops either way.
#[cfg(target_os = "none")]
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    init_heap();
    codevar_target::semihosting::exit(run_and_report())
}

/// Hosted entry: the same checklist in process, so the example also runs
/// where there is no QEMU runner around (and compiles under
/// `cargo test --workspace`).
#[cfg(not(target_os = "none"))]
fn main() -> std::process::ExitCode {
    if run_and_report() == 0 {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}

/// Reports the panic through the logger and stops with a failed status.
///
/// The report itself is allocation-free, so this stays usable even when
/// the heap is unavailable or corrupted — and `exit` only touches the
/// stack, so the verdict reaches the runner from a broken state too.
#[cfg(target_os = "none")]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    codevar_logger::report_panic(info);
    codevar_target::semihosting::exit(1)
}

/// Link-time stand-in for the personality routine referenced by the
/// precompiled `alloc` landing pads.
///
/// With `-C panic=abort` no unwinding ever reaches it, so an empty body is
/// safe; it exists only to satisfy the linker (`DW.ref.rust_eh_personality`
/// is emitted by the code generation unit that owns the panic path).
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}
