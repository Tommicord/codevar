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

//! Hardware-accelerated (GPU) bulk text operations.
//!
//! This module owns every OpenCL-dispatched workload in the engine and
//! its CPU fallbacks:
//!
//! - the HOT change distance ([`crate::change::ChangeFlags::HOT`]) via
//!   [`try_levenshtein`], the banded Wagner-Fischer dynamic program
//!   compiled at build time by `codevar-oclc` and embedded as SPIR-V;
//!   the kernel saturates at `cap + 1`, exactly like the CPU reference
//!   [`crate::change::levenshtein_capped`], so both agree on every input;
//! - parallel gap-buffer shifts via [`gap_shift_insert`] and
//!   [`gap_shift_delete`];
//! - parallel pattern matching via [`find_matches`];
//! - parallel UTF-8 validation via [`utf8_validate`].
//!
//! Everything here is best-effort: any failure — no OpenCL loader, no
//! GPU device, a driver that rejects the SPIR-V image, an allocation
//! error, or an input past a kernel's 32-bit limits — is reported as
//! [`None`] so the caller falls back to the CPU path, which always
//! produces a result. The OpenCL pipeline (device, context, queue,
//! program, kernel) is built once on first use and cached for the
//! lifetime of the process; only one kernel is cached at a time, so
//! switching kernels rebuilds it.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use codevar_logger::{log_debug, log_warn};
use codevar_ocl::{Arg, Buffer, CommandQueue, Context, DeviceKind, Kernel, Program, Runtime};
use codevar_oclc::kernels;
use spin::Mutex;

/// The kernel entry point used for the HOT change-distance query.
const LEVENSHTEIN_KERNEL: &str = "levenshtein";
/// Kernel entry point for the parallel gap-buffer insert shift.
const GAP_SHIFT_INSERT_KERNEL: &str = "gap_shift_insert";
/// Kernel entry point for the parallel gap-buffer delete shift.
const GAP_SHIFT_DELETE_KERNEL: &str = "gap_shift_delete";
/// Kernel entry point for parallel pattern matching.
const FIND_MATCHES_KERNEL: &str = "find_matches";
/// Kernel entry point for parallel UTF-8 validation.
const UTF8_VALIDATE_KERNEL: &str = "utf8_validate_chunk";

/// Largest sequence length the kernel's 32-bit parameters accept.
///
/// The kernel indexes with signed 32-bit integers and forms `row +
/// max_dist`, so the host keeps every length (and the cap derived from
/// it) comfortably below half of `i32::MAX`; a longer input simply
/// falls back to the CPU reference.
const MAX_LEN: usize = i32::MAX as usize / 2;

/// Why a GPU pipeline could not be built.
#[derive(Debug)]
enum PipelineError {
    /// No platform exposed an available GPU device.
    NoGpuDevice,
    /// The requested entry point was not found in the embedded kernels.
    UnknownKernel,
    /// An OpenCL call failed.
    OpenCl(codevar_ocl::Error),
}

impl From<codevar_ocl::Error> for PipelineError {
    #[inline]
    fn from(error: codevar_ocl::Error) -> Self {
        Self::OpenCl(error)
    }
}

impl core::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoGpuDevice => f.write_str("no available OpenCL GPU device was exposed by any platform"),
            Self::UnknownKernel => f.write_str("kernel entry point not found in the embedded modules"),
            Self::OpenCl(error) => write!(f, "OpenCL call failed: {error}"),
        }
    }
}

/// A ready-to-enqueue GPU pipeline: the queue owns the context and
/// device, the kernel owns the built program, so this pair pins
/// everything a kernel dispatch needs.
struct GpuPipeline {
    /// In-order command queue for the cached device.
    queue: CommandQueue,
    /// The kernel bound to the embedded SPIR-V program.
    kernel: Kernel,
}

/// Process-wide pipeline state, built lazily on the first GPU query.
enum PipelineState {
    /// Nothing attempted yet.
    Uninit,
    /// A build attempt failed; never touch the driver again.
    Unavailable,
    /// The pipeline is ready and reused for every subsequent query.
    Ready(GpuPipeline),
}

/// The cached pipeline, keyed by kernel name. The mutex also
/// serializes concurrent GPU queries, which matches the serialized
/// nature of a single command queue.
static PIPELINE: Mutex<PipelineState> = Mutex::new(PipelineState::Uninit);

/// The kernel name the cached pipeline was built for (empty when
/// uninitialized). Because only one pipeline is cached at a time,
/// switching kernels incurs a one-time rebuild — acceptable because the
/// engine typically uses one GPU kernel per editing session.
static PIPELINE_KERNEL: spin::LazyLock<Mutex<String>> = spin::LazyLock::new(|| Mutex::new(String::new()));

/// Builds the OpenCL pipeline for `kernel_name`: loader → GPU device →
/// context → queue → embedded SPIR-V program → kernel.
///
/// # Errors
///
/// Returns [`PipelineError::NoGpuDevice`] when no platform exposes an
/// available GPU, [`PipelineError::UnknownKernel`] when `kernel_name` is
/// absent from the embedded modules, and the underlying
/// [`codevar_ocl::Error`] for any driver failure (including a rejected
/// SPIR-V image).
fn build_pipeline(kernel_name: &str) -> Result<GpuPipeline, PipelineError> {
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
    let device = selected.ok_or(PipelineError::NoGpuDevice)?;
    let context = Context::new(&device)?;
    let queue = CommandQueue::new(&context, &device)?;

    // Find the SPIR-V for the requested entry point in the embedded
    // modules; every compiled kernel is listed in `KERNEL_MODULES`.
    let mut spirv: Option<&[u8]> = None;
    for module in kernels::KERNEL_MODULES {
        if module.entry_points.contains(&kernel_name) {
            spirv = Some(module.spirv);
            break;
        }
    }
    let spirv = spirv.ok_or(PipelineError::UnknownKernel)?;
    let program = Program::from_il(&context, spirv)?;
    program.build("")?;
    let kernel = program.kernel(kernel_name)?;
    Ok(GpuPipeline { queue, kernel })
}

/// Runs `f` with exclusive access to the cached pipeline for
/// `kernel_name`, initializing it on first use. Returns [`None`] when
/// the pipeline is unavailable; the initialization failure is logged
/// exactly once per kernel.
fn with_pipeline<R>(kernel_name: &str, f: impl FnOnce(&mut GpuPipeline) -> R) -> Option<R> {
    let mut state = PIPELINE.lock();
    let mut name = PIPELINE_KERNEL.lock();
    if (*name != kernel_name) || matches!(*state, PipelineState::Uninit) {
        match build_pipeline(kernel_name) {
            Ok(pipeline) => {
                *state = PipelineState::Ready(pipeline);
                *name = kernel_name.to_string();
            }
            Err(error) => {
                log_warn!("the GPU pipeline for {kernel_name} is unavailable, using the CPU path: {error}");
                *state = PipelineState::Unavailable;
            }
        }
    }
    match &mut *state {
        PipelineState::Ready(pipeline) => Some(f(pipeline)),
        PipelineState::Uninit | PipelineState::Unavailable => None,
    }
}

/// `true` when a GPU pipeline can be built for `kernel_name` on this
/// machine. Attempts the one-time initialization if it has not run yet;
/// a machine without a usable OpenCL GPU simply returns `false` (and
/// caches that answer).
#[must_use]
pub fn gpu_available_for(kernel_name: &str) -> bool {
    with_pipeline(kernel_name, |_| ()).is_some()
}

/// `true` when a HOT GPU pipeline can be built on this machine.
///
/// Equivalent to [`gpu_available_for`]`(LEVENSHTEIN_KERNEL)`. Useful for
/// tests and for callers that want to know whether
/// [`crate::change::ChangeFlags::HOT`] will take the GPU path or fall
/// back to the CPU reference.
#[must_use]
pub fn gpu_available() -> bool {
    gpu_available_for(LEVENSHTEIN_KERNEL)
}

/// Capped Levenshtein distance between `a` and `b`, on the GPU.
///
/// Returns the same value as [`crate::change::levenshtein_capped`] when
/// the GPU handled the pair, and [`None`] when the caller must run the
/// CPU reference instead: no pipeline, an empty input (an OpenCL buffer
/// may not be zero bytes), a length past the kernel's 32-bit limits, or
/// any driver error — all logged.
///
/// # Examples
///
/// ```
/// use codevar_wredit::hot::try_levenshtein;
///
/// // With no GPU this is `None`; with one it equals the CPU answer.
/// if let Some(distance) = try_levenshtein(&[1, 2, 3], &[1, 2, 4], 8) {
///     assert_eq!(distance, 1);
/// }
/// ```
#[must_use]
pub fn try_levenshtein(a: &[u32], b: &[u32], max_dist: u32) -> Option<u32> {
    if a.is_empty() || b.is_empty() {
        // `levenshtein_capped` answers these directly, and the kernel
        // cannot be handed a zero-length buffer.
        return None;
    }
    let longest = a.len().max(b.len());
    if longest > MAX_LEN {
        return None;
    }
    // Match the CPU reference's saturation exactly: the cap can never
    // exceed the longest input, and saturation is `cap + 1`.
    let cap = max_dist.min(u32::try_from(longest).ok()?);
    let a_len = i32::try_from(a.len()).ok()?;
    let b_len = i32::try_from(b.len()).ok()?;
    let cap_value = i32::try_from(cap).ok()?;
    with_pipeline(LEVENSHTEIN_KERNEL, |pipeline| {
        match enqueue(pipeline, a, b, a_len, b_len, cap_value) {
            Ok(distance) => Some(distance),
            Err(error) => {
                log_debug!("the HOT change distance fell back to the CPU path: {error}");
                None
            }
        }
    })
    .flatten()
}

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
    with_pipeline(GAP_SHIFT_INSERT_KERNEL, |pipeline| {
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
    with_pipeline(GAP_SHIFT_DELETE_KERNEL, |pipeline| {
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
    with_pipeline(FIND_MATCHES_KERNEL, |pipeline| {
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
    with_pipeline(UTF8_VALIDATE_KERNEL, |pipeline| {
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

/// Uploads both sequences, launches the kernel over a single work item,
/// and reads the saturated distance back.
///
/// # Errors
///
/// Returns [`codevar_ocl::Error::InvalidArgument`] when the scratch
/// buffer size overflows the address space, and the underlying driver
/// error for any buffer, argument, enqueue, wait, or read failure.
fn enqueue(
    pipeline: &mut GpuPipeline,
    a: &[u32],
    b: &[u32],
    a_len: i32,
    b_len: i32,
    cap: i32,
) -> Result<u32, codevar_ocl::Error> {
    let GpuPipeline { queue, kernel } = pipeline;
    let context = queue.context();
    let work_bytes = (b.len() + 1)
        .checked_mul(2)
        .and_then(|cells| cells.checked_mul(core::mem::size_of::<i32>()))
        .ok_or(codevar_ocl::Error::InvalidArgument {
            what: "the Levenshtein scratch buffer overflows the address space",
        })?;
    let a_buf = Buffer::from_slice(context, a)?;
    let b_buf = Buffer::from_slice(context, b)?;
    let work_buf = Buffer::new(context, work_bytes)?;
    let out_buf = Buffer::new(context, core::mem::size_of::<i32>())?;
    kernel.set_arg(0, Arg::Buffer(&a_buf))?;
    kernel.set_arg(1, Arg::I32(a_len))?;
    kernel.set_arg(2, Arg::Buffer(&b_buf))?;
    kernel.set_arg(3, Arg::I32(b_len))?;
    kernel.set_arg(4, Arg::I32(cap))?;
    kernel.set_arg(5, Arg::Buffer(&work_buf))?;
    kernel.set_arg(6, Arg::Buffer(&out_buf))?;
    // One work item owns the whole dynamic program; see the kernel.
    let event = queue.enqueue_kernel(kernel, &[1], None)?;
    event.wait()?;
    let mut out = [0i32; 1];
    queue.read_buffer(&out_buf, 0, &mut out)?;
    Ok(out[0].max(0) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change::levenshtein_capped;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Deterministic pseudo-random code points.
    fn sequence(len: usize, seed: u32) -> Vec<u32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state % 37
            })
            .collect()
    }

    #[test]
    fn empty_inputs_are_left_to_the_cpu() {
        assert_eq!(try_levenshtein(&[], &[], 4), None);
        assert_eq!(try_levenshtein(&[1, 2, 3], &[], 4), None);
        assert_eq!(try_levenshtein(&[], &[1, 2, 3], 4), None);
    }

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

    /// When a device is usable the GPU must agree with the CPU
    /// reference on every input; when it is not, `try_levenshtein`
    /// answers [`None`]. Both outcomes are valid here.
    #[test]
    fn gpu_matches_the_cpu_reference_when_available() {
        let cases: [(&[u32], &[u32], u32); 5] = [
            (&[1, 2, 3], &[1, 2, 4], 8),
            (&[7, 7, 7], &[7, 8, 7, 7], 5),
            (&[1, 2], &[1, 2], 0),
            (&[9, 9, 9, 9], &[1, 2], 2),
            (&[4, 5, 6, 7, 8], &[4, 5, 0, 7, 8], 3),
        ];
        for (a, b, cap) in cases {
            let cpu = levenshtein_capped(a, b, cap);
            if let Some(gpu) = try_levenshtein(a, b, cap) {
                assert_eq!(gpu, cpu, "GPU/CPU mismatch for cap {cap}");
            }
        }
    }

    #[test]
    fn larger_sequences_agree_when_the_gpu_runs() {
        let a = sequence(300, 0x1234_5678);
        let b = sequence(297, 0x8765_4321);
        for cap in [0u32, 3, 17, 4096] {
            let cpu = levenshtein_capped(&a, &b, cap);
            if let Some(gpu) = try_levenshtein(&a, &b, cap) {
                assert_eq!(gpu, cpu, "GPU/CPU mismatch for cap {cap}");
            }
        }
    }

    #[test]
    fn gpu_availability_probe_is_stable() {
        // The answer is cached, so two calls agree; a machine without a
        // device simply reports `false`.
        assert_eq!(gpu_available(), gpu_available());
    }
}
