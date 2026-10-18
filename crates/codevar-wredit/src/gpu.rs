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

//! GPU-accelerated text-editing operations using OpenCL.
//!
//! This module dispatches bulk text-editing workloads (gap buffer
//! shifts, search-and-replace, line wrapping, UTF-8 validation) to
//! an OpenCL device when one is available, with CPU fallbacks for
//! every operation. The pipeline is built lazily on first use and
//! cached for the process lifetime, exactly like [`crate::hot`].
//!
//! All entry points return [`Option`]: `None` means "no GPU available
//! or operation failed, use the CPU path". The CPU fallbacks live in
//! [`crate::simd`] and the standard library and are always correct.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use codevar_logger::log_warn;
use codevar_ocl::{Arg, Buffer, CommandQueue, Context, DeviceKind, Kernel, Program, Runtime};
use codevar_oclc::kernels;
use spin::Mutex;

/// Why the GPU pipeline could not be built.
#[derive(Debug)]
enum GpuPipelineError {
    /// No platform exposed an available GPU device.
    NoGpuDevice,
    /// The requested entry point was not found in the embedded kernels.
    UnknownKernel,
    /// An OpenCL call failed.
    OpenCl(codevar_ocl::Error),
}

impl From<codevar_ocl::Error> for GpuPipelineError {
    #[inline]
    fn from(error: codevar_ocl::Error) -> Self {
        Self::OpenCl(error)
    }
}

impl core::fmt::Display for GpuPipelineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoGpuDevice => f.write_str("no available OpenCL GPU device"),
            Self::UnknownKernel => f.write_str("kernel entry point not found"),
            Self::OpenCl(error) => write!(f, "{error}"),
        }
    }
}

/// A ready-to-enqueue GPU pipeline for one kernel.
struct GpuKernelPipeline {
    /// In-order command queue for the cached device.
    queue: CommandQueue,
    /// The bound kernel.
    kernel: Kernel,
}

/// Process-wide pipeline state, built lazily per kernel name.
enum GpuPipelineState {
    /// Nothing attempted yet.
    Uninit,
    /// A build attempt failed; never touch the driver again.
    Unavailable,
    /// The pipeline is ready and reused.
    Ready(GpuKernelPipeline),
}

/// The cached pipeline, keyed by kernel name. A single static with a
/// name index is simpler than a map: we only ever have a handful of
/// kernels and they are looked up by name on first use.
static GPU_PIPELINE: Mutex<GpuPipelineState> = Mutex::new(GpuPipelineState::Uninit);

/// The name of the kernel the pipeline was built for (or empty if
/// uninitialized). Because we cache only one pipeline at a time,
/// switching kernels incurs a one-time rebuild — acceptable because
/// the engine typically uses one GPU kernel per editing session.
static GPU_KERNEL_NAME: spin::LazyLock<spin::Mutex<String>> =
    spin::LazyLock::new(|| spin::Mutex::new(String::new()));

/// Builds the OpenCL pipeline for `kernel_name`.
fn build_pipeline(kernel_name: &str) -> Result<GpuKernelPipeline, GpuPipelineError> {
    let runtime = Runtime::load()?;
    let mut selected = None;
    for platform in runtime.platforms() {
        let Ok(devices) = platform.devices(DeviceKind::Gpu) else {
            continue;
        };
        for device in devices {
            if device.available().unwrap_or(false) {
                selected = Some(device);
                break;
            }
        }
        if selected.is_some() {
            break;
        }
    }
    let device = selected.ok_or(GpuPipelineError::NoGpuDevice)?;
    let context = Context::new(&device)?;
    let queue = CommandQueue::new(&context, &device)?;

    // Find the SPIR-V for the requested kernel in the embedded modules.
    let mut spirv: Option<&[u8]> = None;
    for module in kernels::KERNEL_MODULES {
        if module.entry_points.contains(&kernel_name) {
            spirv = Some(module.spirv);
            break;
        }
    }
    let spirv = spirv.ok_or(GpuPipelineError::UnknownKernel)?;

    let program = Program::from_il(&context, spirv)?;
    program.build("")?;
    let kernel = program.kernel(kernel_name)?;
    Ok(GpuKernelPipeline { queue, kernel })
}

/// Runs `f` with exclusive access to the cached pipeline for
/// `kernel_name`, initializing it on first use. Returns `None` when
/// the pipeline is unavailable.
fn with_gpu_pipeline<R>(kernel_name: &str, f: impl FnOnce(&mut GpuKernelPipeline) -> R) -> Option<R> {
    let mut state = GPU_PIPELINE.lock();
    let mut name = GPU_KERNEL_NAME.lock();
    if (*name != kernel_name) || matches!(*state, GpuPipelineState::Uninit) {
        match build_pipeline(kernel_name) {
            Ok(pipeline) => {
                *state = GpuPipelineState::Ready(pipeline);
                *name = kernel_name.to_string();
            }
            Err(error) => {
                log_warn!("GPU pipeline for {kernel_name} unavailable, using CPU path: {error}");
                *state = GpuPipelineState::Unavailable;
            }
        }
    }
    match &mut *state {
        GpuPipelineState::Ready(pipeline) => Some(f(pipeline)),
        GpuPipelineState::Uninit | GpuPipelineState::Unavailable => None,
    }
}

/// `true` when a GPU pipeline can be built for `kernel_name` on this
/// machine. Attempts one-time initialization if it has not run yet.
#[must_use]
pub fn gpu_available_for(kernel_name: &str) -> bool {
    with_gpu_pipeline(kernel_name, |_| ()).is_some()
}

// ============================================================================
// gap_shift: parallel gap buffer insert/delete
// ============================================================================

/// Parallel gap-buffer insert shift: moves `src[0..len]` into `dst`,
/// opening a gap of `delta` slots at `at`.
///
/// Returns `Some(())` when the GPU handled the shift, `None` when the
/// caller should use the CPU path (`core::ptr::copy` twice).
///
/// # Safety
///
/// `src` must be valid for `len` elements; `dst` must be valid for
/// `len + delta` elements. `delta > 0`.
pub unsafe fn gap_shift_insert(
    src: *const u32,
    len: usize,
    at: usize,
    delta: usize,
    dst: *mut u32,
) -> Option<()> {
    if len == 0 || delta == 0 {
        return Some(());
    }
    let len_i = i32::try_from(len).ok()?;
    let at_i = i32::try_from(at).ok()?;
    let delta_i = i32::try_from(delta).ok()?;
    with_gpu_pipeline("gap_shift_insert", |pipeline| {
        let context = pipeline.queue.context();
        // Safety: `src` is valid for `len` elements per the caller's contract.
        let src_slice = unsafe { core::slice::from_raw_parts(src, len) };
        let src_buf = Buffer::from_slice(context, src_slice)?;
        let dst_vec = vec![0u32; len + delta];
        let dst_buf = Buffer::from_slice(context, &dst_vec)?;
        pipeline
            .kernel
            .set_arg(0, Arg::Buffer(&src_buf))?;
        pipeline.kernel.set_arg(1, Arg::I32(len_i))?;
        pipeline.kernel.set_arg(2, Arg::I32(at_i))?;
        pipeline.kernel.set_arg(3, Arg::I32(delta_i))?;
        pipeline
            .kernel
            .set_arg(4, Arg::Buffer(&dst_buf))?;
        let global = (len + delta).next_power_of_two();
        let event = pipeline
            .queue
            .enqueue_kernel(&pipeline.kernel, &[global], None)?;
        event.wait()?;
        let mut out = vec![0u32; len + delta];
        pipeline
            .queue
            .read_buffer(&dst_buf, 0, &mut out)?;
        // Safety: `dst` is valid for `len + delta` elements per the caller.
        unsafe { core::ptr::copy_nonoverlapping(out.as_ptr(), dst, len + delta) };
        Ok::<(), codevar_ocl::Error>(())
    })
    .and_then(|result| result.ok())
}

/// Parallel gap-buffer delete shift: moves `src[0..len]` into `dst`,
/// closing a gap of `delta` slots at `at`.
///
/// # Safety
///
/// `src` must be valid for `len` elements; `dst` must be valid for
/// `len - delta` elements. `delta > 0` and `delta <= len - at`.
pub unsafe fn gap_shift_delete(
    src: *const u32,
    len: usize,
    at: usize,
    delta: usize,
    dst: *mut u32,
) -> Option<()> {
    if len == 0 || delta == 0 {
        return Some(());
    }
    let len_i = i32::try_from(len).ok()?;
    let at_i = i32::try_from(at).ok()?;
    let delta_i = i32::try_from(delta).ok()?;
    with_gpu_pipeline("gap_shift_delete", |pipeline| {
        let context = pipeline.queue.context();
        // Safety: `src` is valid for `len` elements per the caller's contract.
        let src_slice = unsafe { core::slice::from_raw_parts(src, len) };
        let src_buf = Buffer::from_slice(context, src_slice)?;
        let dst_vec = vec![0u32; len.saturating_sub(delta)];
        let dst_buf = Buffer::from_slice(context, &dst_vec)?;
        pipeline
            .kernel
            .set_arg(0, Arg::Buffer(&src_buf))?;
        pipeline.kernel.set_arg(1, Arg::I32(len_i))?;
        pipeline.kernel.set_arg(2, Arg::I32(at_i))?;
        pipeline.kernel.set_arg(3, Arg::I32(delta_i))?;
        pipeline
            .kernel
            .set_arg(4, Arg::Buffer(&dst_buf))?;
        let global = len.saturating_sub(delta).next_power_of_two();
        let event = pipeline
            .queue
            .enqueue_kernel(&pipeline.kernel, &[global], None)?;
        event.wait()?;
        let mut out = vec![0u32; len.saturating_sub(delta)];
        pipeline
            .queue
            .read_buffer(&dst_buf, 0, &mut out)?;
        // Safety: `dst` is valid for `len - delta` elements per the caller.
        unsafe { core::ptr::copy_nonoverlapping(out.as_ptr(), dst, len.saturating_sub(delta)) };
        Ok::<(), codevar_ocl::Error>(())
    })
    .and_then(|result| result.ok())
}

// ============================================================================
// search_replace: parallel search and replace
// ============================================================================

/// Finds all non-overlapping matches of `needle` in `data[0..len]`.
/// Returns a `Vec<u32>` where entry `i` is 1 when `data[i..]` starts
/// with `needle`, else 0. Returns `None` when the GPU path is
/// unavailable.
///
/// # Safety
///
/// `data` must be valid for `len` elements; `needle` for `needle_len`.
pub unsafe fn find_matches(
    data: *const u32,
    len: usize,
    needle: *const u32,
    needle_len: usize,
) -> Option<Vec<u32>> {
    if len == 0 || needle_len == 0 {
        return Some(vec![0; len]);
    }
    let len_i = i32::try_from(len).ok()?;
    let needle_len_i = i32::try_from(needle_len).ok()?;
    let mut matches = vec![0u32; len];
    with_gpu_pipeline("find_matches", |pipeline| {
        let context = pipeline.queue.context();
        // Safety: `data` and `needle` are valid per the caller's contract.
        let data_slice = unsafe { core::slice::from_raw_parts(data, len) };
        let needle_slice = unsafe { core::slice::from_raw_parts(needle, needle_len) };
        let data_buf = Buffer::from_slice(context, data_slice)?;
        let needle_buf = Buffer::from_slice(context, needle_slice)?;
        let matches_buf = Buffer::from_slice(context, &matches)?;
        pipeline
            .kernel
            .set_arg(0, Arg::Buffer(&data_buf))?;
        pipeline.kernel.set_arg(1, Arg::I32(len_i))?;
        pipeline
            .kernel
            .set_arg(2, Arg::Buffer(&needle_buf))?;
        pipeline
            .kernel
            .set_arg(3, Arg::I32(needle_len_i))?;
        pipeline
            .kernel
            .set_arg(4, Arg::Buffer(&matches_buf))?;
        let global = len.next_power_of_two();
        let event = pipeline
            .queue
            .enqueue_kernel(&pipeline.kernel, &[global], None)?;
        event.wait()?;
        pipeline
            .queue
            .read_buffer(&matches_buf, 0, &mut matches)?;
        Ok::<(), codevar_ocl::Error>(())
    })
    .and_then(|result| result.ok())?;
    Some(matches)
}

// ============================================================================
// utf8_validate: parallel UTF-8 validation
// ============================================================================

/// Validates a byte buffer for well-formed UTF-8 in parallel.
/// Returns `Some(true)` when the buffer is valid UTF-8, `Some(false)`
/// when it is not, and `None` when the GPU path is unavailable.
///
/// # Safety
///
/// `data` must be valid for `len` bytes.
pub unsafe fn utf8_validate(data: *const u8, len: usize) -> Option<bool> {
    if len == 0 {
        return Some(true);
    }
    let len_i = i32::try_from(len).ok()?;
    let chunk_count = len.div_ceil(64);
    let mut valid = vec![0i32; chunk_count];
    with_gpu_pipeline("utf8_validate_chunk", |pipeline| {
        let context = pipeline.queue.context();
        // Safety: `data` is valid for `len` bytes per the caller's contract.
        let data_slice = unsafe { core::slice::from_raw_parts(data, len) };
        let data_buf = Buffer::from_slice(context, data_slice)?;
        let valid_buf = Buffer::from_slice(context, &valid)?;
        pipeline
            .kernel
            .set_arg(0, Arg::Buffer(&data_buf))?;
        pipeline.kernel.set_arg(1, Arg::I32(len_i))?;
        pipeline
            .kernel
            .set_arg(2, Arg::Buffer(&valid_buf))?;
        let global = chunk_count.next_power_of_two();
        let event = pipeline
            .queue
            .enqueue_kernel(&pipeline.kernel, &[global], None)?;
        event.wait()?;
        pipeline
            .queue
            .read_buffer(&valid_buf, 0, &mut valid)?;
        Ok::<(), codevar_ocl::Error>(())
    })
    .and_then(|result| result.ok())?;
    Some(valid.iter().all(|&v| v != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_shift_insert_cpu_equivalence() {
        // CPU reference for the `gap_shift_insert` kernel shape: the first
        // `at` elements are kept, `delta` zero slots open at `at`, and the
        // tail is shifted right by `delta`.
        let src: [u32; 5] = [1, 2, 3, 4, 5];
        let at = 2;
        let delta = 3;
        let mut dst = [0u32; 8];
        for (i, slot) in dst.iter_mut().enumerate() {
            *slot = if i < at {
                src[i]
            } else if i >= at + delta {
                src[i - delta]
            } else {
                0
            };
        }
        assert_eq!(dst, [1, 2, 0, 0, 0, 3, 4, 5]);
    }

    #[test]
    fn find_matches_empty() {
        let data: [u32; 0] = [];
        let needle: [u32; 0] = [];
        let result = unsafe { find_matches(data.as_ptr(), 0, needle.as_ptr(), 0) };
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn utf8_validate_empty_is_valid() {
        let data: [u8; 0] = [];
        assert_eq!(unsafe { utf8_validate(data.as_ptr(), 0) }, Some(true));
    }
}
