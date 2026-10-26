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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Execution-path benchmarks for `codevar-ocl`.
//!
//! Three execution paths are measured on the same embedded kernels, all
//! of which come from the `codevar-olc` build-script bindings
//! (`codevar_olcc::kernels::KERNEL_MODULES`) — nothing is hardcoded:
//!
//! * `native/cpu` — the kernel's native machine code
//!   (`NativeKernel.code`, emitted by the `codevar-ocl-native-codegen-*`
//!   back ends) is copied into executable memory with
//!   `codevar_ocl_bare::Trampoline` and entered once per work-item,
//!   exactly like the driver's fast path.
//! * `spirv/interpreted` — the kernel's SPIR-V image
//!   (`KernelModule.spirv`) is loaded through `codevar-ocl`
//!   (`Program::from_il`) on the built-in bare CPU driver, which runs it
//!   through its embedded SPIR-V interpreter.
//! * `gpu/spirv` — the same SPIR-V image through `codevar-ocl` on a real
//!   GPU device; the group is absent when no GPU device exists.
//!
//! Each group loads its program once outside the timed loop and times
//! only the launch plus the blocking read-back. Inputs are re-written
//! before every iteration so the output is always freshly computed, and
//! the read-back is `black_box`ed. A host reference run verifies every
//! path before it is timed.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::ffi::c_void;
use std::hint::black_box;
use std::mem::size_of;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use codevar_ocl::{Arg, Buffer, CommandQueue, Context, Device, DeviceKind, Kernel, Program, Result, Runtime};
use codevar_ocl_bare::{Trampoline, WorkItem, exec::ArgBlock};
use codevar_oclc::kernels;

/// Input sizes (elements) every group is measured over.
const SIZES: &[usize] = &[256, 1024, 4096, 16384];

/// How long each criterion group warms up for.
const WARM_UP_TIME: Duration = Duration::from_secs(2);
/// How long each criterion group measures for.
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);

/// How one kernel's `codevar-ocl` arguments are bound for one launch over
/// `size` elements: the three buffers (`a`, `b`, `out`) followed by the
/// two scalars (`count`, `scale`).
type BindArgs = fn(&mut Kernel, &Buffer, &Buffer, &Buffer, i32, i32);

/// How one kernel's native argument block is marshaled: the first two
/// slices feed the `a`/`b` (or `x`/`y`) buffers, the third the output,
/// and the last two scalars are `count` then `scale`.
///
/// Every kernel owns its argument order — `vector_add` is
/// `(a, b, out, count)` while `saxpy` is `(scale, x, y, out, count)` — so
/// the binder writes *every* slot, including the buffer pointers, instead
/// of relying on a fixed slot layout.
type BindNative = fn(&mut ArgBlock, &[i32], &[i32], &mut [i32], i32, i32);

/// One benchmarkable kernel: how its arguments are bound and its output
/// checked, plus the metadata the bindings expose.
struct BenchKernel<'a> {
    /// Kernel name used in group labels.
    name: &'a str,
    /// The embedded module from `kernels::KERNEL_MODULES`.
    module: &'a kernels::KernelModule,
    /// The `#[kernel]` entry point inside `module`.
    entry: &'a str,
    /// Binds the kernel's `codevar-ocl` arguments.
    bind: BindArgs,
    /// Marshals the kernel's native argument block.
    bind_native: BindNative,
    /// The `scale` scalar argument (unused by `vector_add`).
    scale: i32,
    /// The host-computed expectation for one element:
    /// `expect(a[i], b[i], scale)`.
    expect: fn(i32, i32, i32) -> i32,
}

/// The embedded benchmark kernels, in the order they are measured.
fn bench_kernels() -> [BenchKernel<'static>; 2] {
    let vector_add = module("bench/vector_add").expect("the build script embeds bench/vector_add");
    let saxpy = module("bench/saxpy").expect("the build script embeds bench/saxpy");
    [
        BenchKernel {
            name: "vector_add",
            module: vector_add,
            entry: "vector_add",
            bind: bind_vector_add,
            bind_native: bind_native_vector_add,
            scale: 0,
            expect: |a, b, _| a.wrapping_add(b),
        },
        BenchKernel {
            name: "saxpy",
            module: saxpy,
            entry: "saxpy",
            bind: bind_saxpy,
            bind_native: bind_native_saxpy,
            scale: 3,
            expect: |x, y, scale| x.wrapping_mul(scale).wrapping_add(y),
        },
    ]
}

/// Finds the embedded module `name` in `kernels::KERNEL_MODULES`.
fn module(name: &str) -> Option<&'static kernels::KernelModule> {
    kernels::KERNEL_MODULES
        .iter()
        .find(|module| module.name == name)
}

/// Binds `vector_add`'s arguments: `(a, b, out, count)`.
fn bind_vector_add(kernel: &mut Kernel, a: &Buffer, b: &Buffer, out: &Buffer, count: i32, _scale: i32) {
    kernel
        .set_arg(0, Arg::Buffer(a))
        .expect("vector_add arg 0");
    kernel
        .set_arg(1, Arg::Buffer(b))
        .expect("vector_add arg 1");
    kernel
        .set_arg(2, Arg::Buffer(out))
        .expect("vector_add arg 2");
    kernel
        .set_arg(3, Arg::I32(count))
        .expect("vector_add arg 3");
}

/// Binds `saxpy`'s arguments: `(scale, x, y, out, count)`.
fn bind_saxpy(kernel: &mut Kernel, x: &Buffer, y: &Buffer, out: &Buffer, count: i32, scale: i32) {
    kernel
        .set_arg(0, Arg::I32(scale))
        .expect("saxpy arg 0");
    kernel
        .set_arg(1, Arg::Buffer(x))
        .expect("saxpy arg 1");
    kernel
        .set_arg(2, Arg::Buffer(y))
        .expect("saxpy arg 2");
    kernel
        .set_arg(3, Arg::Buffer(out))
        .expect("saxpy arg 3");
    kernel
        .set_arg(4, Arg::I32(count))
        .expect("saxpy arg 4");
}

/// Marshals `vector_add`'s native argument block: `(a, b, out, count)` in
/// slots `0..=3`.
fn bind_native_vector_add(
    block: &mut ArgBlock,
    a: &[i32],
    b: &[i32],
    out: &mut [i32],
    count: i32,
    _scale: i32,
) {
    block.set_ptr(0, a.as_ptr() as *mut c_void);
    block.set_ptr(1, b.as_ptr() as *mut c_void);
    block.set_ptr(2, out.as_mut_ptr() as *mut c_void);
    block
        .set_scalar(3, &count.to_ne_bytes())
        .expect("count scalar fits");
}

/// Marshals `saxpy`'s native argument block: `(scale, x, y, out, count)` in
/// slots `0..=4`.
fn bind_native_saxpy(block: &mut ArgBlock, x: &[i32], y: &[i32], out: &mut [i32], count: i32, scale: i32) {
    block
        .set_scalar(0, &scale.to_ne_bytes())
        .expect("scale scalar fits");
    block.set_ptr(1, x.as_ptr() as *mut c_void);
    block.set_ptr(2, y.as_ptr() as *mut c_void);
    block.set_ptr(3, out.as_mut_ptr() as *mut c_void);
    block
        .set_scalar(4, &count.to_ne_bytes())
        .expect("count scalar fits");
}

/// The host-side inputs and expected output for one launch over `size`
/// elements: `a[i] = i`, `b[i] = 7`, `expected[i] = expect(a[i], b[i], scale)`.
fn host_values(size: usize, kernel: &BenchKernel<'_>) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let a: Vec<i32> = (0..size as i32).collect();
    let b: Vec<i32> = vec![7i32; size];
    let expected: Vec<i32> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (kernel.expect)(*x, *y, kernel.scale))
        .collect();
    (a, b, expected)
}

/// Measures the native path: the kernel's machine code, entered through
/// a `Trampoline`, once per work-item over `0..size`.
fn bench_native(c: &mut Criterion, kernel: &BenchKernel<'_>) {
    let Some(native) = kernel
        .module
        .native
        .iter()
        .find(|native| native.name == kernel.entry)
    else {
        return;
    };
    let Ok(trampoline) = Trampoline::new(native.code, native.entry_offset) else {
        return;
    };

    let mut group = c.benchmark_group(format!("{}/native/cpu", kernel.name));
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);

    for &size in SIZES {
        let (a, b, expected) = host_values(size, kernel);
        let mut out = vec![0i32; size];

        // Sanity check before timing: the native code must agree with the host.
        launch_native(&trampoline, &a, &b, &mut out, size, kernel);
        assert_eq!(out, expected, "the native path must match the host reference");

        group.throughput(Throughput::Bytes((size * 3 * size_of::<i32>()) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, &size| {
            let mut a = (0..size as i32).collect::<Vec<_>>();
            let mut b = vec![7i32; size];
            let mut out = vec![0i32; size];
            bencher.iter(|| {
                for (i, value) in a.iter_mut().enumerate() {
                    *value = i as i32;
                }
                b.fill(7);
                out.fill(0);
                launch_native(&trampoline, &a, &b, &mut out, size, kernel);
                black_box(&out);
            });
        });
    }

    group.finish();
}

/// Launches the native kernel once per work-item over `0..size`.
fn launch_native(
    trampoline: &Trampoline,
    a: &[i32],
    b: &[i32],
    out: &mut [i32],
    size: usize,
    kernel: &BenchKernel<'_>,
) {
    let mut block = ArgBlock::new();
    // Every kernel owns its argument order, so the binder writes the
    // buffer pointers too (see [`BindNative`]).
    (kernel.bind_native)(&mut block, a, b, out, size as i32, kernel.scale);

    for gid in 0..size {
        let mut item = WorkItem {
            global_id: [gid as u32, 0, 0],
            local_id: [0, 0, 0],
            group_id: [0, 0, 0],
            global_offset: [0, 0, 0],
            num_groups: [0, 0, 0],
            local_size: [0, 0, 0],
            arg_base: block.as_ptr() as *mut u8,
        };
        // SAFETY: the trampoline holds position-independent machine code
        // for this kernel and the argument block is fully initialized.
        trampoline.call(&mut item as *mut WorkItem);
    }
}

/// One `codevar-ocl` handle over one device: the context, queue and the
/// program/kernel pair built once from the embedded SPIR-V.
struct OclHandle {
    context: Context,
    queue: CommandQueue,
    kernel: Kernel,
    /// The program the kernel retains; kept for clarity.
    _program: Program,
}

/// Loads the embedded kernel for `device` through `codevar-ocl`.
fn load_ocl(device: &Device, kernel: &BenchKernel<'_>) -> Result<OclHandle> {
    let context = Context::new(device)?;
    let queue = CommandQueue::new(&context, device)?;
    let program = Program::from_il(&context, kernel.module.spirv)?;
    program.build("")?;
    let kernel = program.kernel(kernel.entry)?;
    Ok(OclHandle {
        context,
        queue,
        kernel,
        _program: program,
    })
}

/// Measures `codevar-ocl` on `device`, reporting the group as
/// `{kernel.name}/{suffix}`.
fn bench_ocl(c: &mut Criterion, device: &Device, kernel: &BenchKernel<'_>, suffix: &str) {
    // The `Runtime` that owns `device` must outlive every object created
    // from it; it is held by the caller for the whole group.
    let Ok(mut handle) = load_ocl(device, kernel) else {
        return;
    };

    let mut group = c.benchmark_group(format!("{}/{}", kernel.name, suffix));
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);

    for &size in SIZES {
        let (a_data, b_data, expected) = host_values(size, kernel);
        let a_buf = Buffer::from_slice(&handle.context, &a_data).expect("input buffer a");
        let b_buf = Buffer::from_slice(&handle.context, &b_data).expect("input buffer b");
        let out_buf = Buffer::new(&handle.context, size * size_of::<i32>()).expect("output buffer");

        (kernel.bind)(
            &mut handle.kernel,
            &a_buf,
            &b_buf,
            &out_buf,
            size as i32,
            kernel.scale,
        );

        // Sanity check before timing: the driver must agree with the host.
        let mut readback = vec![0i32; size];
        run_ocl(&handle.queue, &handle.kernel, &out_buf, size, &mut readback);
        assert_eq!(
            readback, expected,
            "the {suffix} path must match the host reference"
        );

        group.throughput(Throughput::Bytes((size * 3 * size_of::<i32>()) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, &size| {
            let mut readback = vec![0i32; size];
            bencher.iter(|| {
                // Re-write the inputs so every iteration computes fresh output.
                let _ = handle
                    .queue
                    .write_buffer(&a_buf, 0, &a_data)
                    .expect("write a")
                    .wait();
                let _ = handle
                    .queue
                    .write_buffer(&b_buf, 0, &b_data)
                    .expect("write b")
                    .wait();
                let event = handle
                    .queue
                    .enqueue_kernel(&handle.kernel, &[size], None)
                    .expect("launch");
                event.wait().expect("launch completes");
                let _ = handle
                    .queue
                    .read_buffer(&out_buf, 0, &mut readback)
                    .expect("read back")
                    .wait();
                black_box(&readback);
            });
        });
    }

    group.finish();
}

/// Launches `kernel` over `0..size` and reads the output back.
fn run_ocl(queue: &CommandQueue, kernel: &Kernel, out_buf: &Buffer, size: usize, readback: &mut [i32]) {
    let event = queue
        .enqueue_kernel(kernel, &[size], None)
        .expect("launch");
    event.wait().expect("launch completes");
    let _ = queue
        .read_buffer(out_buf, 0, readback)
        .expect("read back")
        .wait();
}

/// The built-in bare CPU driver, if it can be initialized.
fn bare_cpu() -> Option<(Runtime, Device)> {
    let runtime = Runtime::load_bare().ok()?;
    let device = runtime
        .platforms()
        .iter()
        .flat_map(|platform| platform.all_devices().ok().into_iter().flatten())
        .next()?;
    Some((runtime, device))
}

/// A real GPU device through the host OpenCL installation, if any.
fn gpu_device() -> Option<(Runtime, Device)> {
    let runtime = Runtime::load().ok()?;
    let device = runtime
        .platforms()
        .iter()
        .flat_map(|platform| {
            platform
                .devices(DeviceKind::Gpu)
                .ok()
                .into_iter()
                .flatten()
        })
        .next()?;
    Some((runtime, device))
}

fn benchmarks(c: &mut Criterion) {
    let bare_cpu_handle = bare_cpu();
    let gpu_handle = gpu_device();

    for kernel in bench_kernels() {
        bench_native(c, &kernel);
        if let Some((_, device)) = &bare_cpu_handle {
            bench_ocl(c, device, &kernel, "spirv/interpreted");
        }
        if let Some((_, device)) = &gpu_handle {
            bench_ocl(c, device, &kernel, "gpu/spirv");
        }
    }
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
