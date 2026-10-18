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

//! Hardware (HOT) GPU change-distance path.
//!
//! When [`crate::change::ChangeFlags::HOT`] is requested,
//! [`crate::change::change_distance`] first tries this module: the two
//! code-point sequences are uploaded to an OpenCL device and the
//! capped Levenshtein distance is computed by the `levenshtein` kernel
//! — the banded Wagner-Fischer dynamic program compiled at build time
//! by `codevar-oclc` and embedded as SPIR-V. The kernel saturates its
//! answer at `cap + 1`, exactly like the CPU reference
//! [`crate::change::levenshtein_capped`], so the two backends agree on
//! every input and the flag only selects *where* the work runs.
//!
//! Everything here is best-effort: any failure — no OpenCL loader, no
//! GPU device, a driver that rejects the SPIR-V image, an allocation
//! error, or an input past the kernel's 32-bit limits — is reported as
//! [`None`] so the caller falls back to the CPU reference, which always
//! produces a result. The OpenCL pipeline (device, context, queue,
//! program, kernel) is built once on first use and cached for the
//! lifetime of the process.

use codevar_logger::{log_debug, log_warn};
use codevar_ocl::{Arg, Buffer, CommandQueue, Context, DeviceKind, Kernel, Program, Runtime};
use codevar_oclc::kernels;
use spin::Mutex;

/// Largest sequence length the kernel's 32-bit parameters accept.
///
/// The kernel indexes with signed 32-bit integers and forms `row +
/// max_dist`, so the host keeps every length (and the cap derived from
/// it) comfortably below half of `i32::MAX`; a longer input simply
/// falls back to the CPU reference.
const MAX_LEN: usize = i32::MAX as usize / 2;

/// Why the HOT pipeline could not be built.
enum PipelineError {
    /// No platform exposed an available GPU device.
    NoGpuDevice,
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
            Self::NoGpuDevice => f.write_str("no available GPU device was exposed by any platform"),
            Self::OpenCl(error) => write!(f, "OpenCL call failed: {error}"),
        }
    }
}

/// A ready-to-enqueue HOT pipeline: the queue owns the context and
/// device, the kernel owns the built program, so this pair pins
/// everything the distance path needs.
struct GpuPipeline {
    /// In-order command queue for the cached device.
    queue: CommandQueue,
    /// The `levenshtein` kernel bound to the embedded SPIR-V program.
    kernel: Kernel,
}

/// Process-wide pipeline state, built lazily on the first HOT query.
enum PipelineState {
    /// Nothing attempted yet.
    Uninit,
    /// A build attempt failed; never touch the driver again.
    Unavailable,
    /// The pipeline is ready and reused for every subsequent query.
    Ready(GpuPipeline),
}

/// The cached pipeline; the mutex also serializes concurrent HOT
/// queries, which matches the serialized nature of a single queue.
static PIPELINE: Mutex<PipelineState> = Mutex::new(PipelineState::Uninit);

/// Builds the OpenCL pipeline: loader → GPU device → context → queue →
/// embedded `levenshtein` SPIR-V program → kernel.
///
/// # Errors
///
/// Returns [`PipelineError::NoGpuDevice`] when no platform exposes an
/// available GPU, and the underlying [`codevar_ocl::Error`] for any
/// driver failure (including a rejected SPIR-V image).
fn build_pipeline() -> Result<GpuPipeline, PipelineError> {
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
    let program = Program::from_il(&context, kernels::wredit::LEVENSHTEIN_SPIRV)?;
    program.build("")?;
    let kernel = program.kernel("levenshtein")?;
    Ok(GpuPipeline { queue, kernel })
}

/// Runs `f` with exclusive access to the cached pipeline, initializing
/// it on first use. Returns [`None`] when the pipeline is unavailable;
/// the initialization failure is logged exactly once.
fn with_pipeline<R>(f: impl FnOnce(&mut GpuPipeline) -> R) -> Option<R> {
    let mut state = PIPELINE.lock();
    if matches!(*state, PipelineState::Uninit) {
        match build_pipeline() {
            Ok(pipeline) => *state = PipelineState::Ready(pipeline),
            Err(error) => {
                log_warn!("the hot change distance is unavailable, using the CPU path: {error}");
                *state = PipelineState::Unavailable;
            }
        }
    }
    match &mut *state {
        PipelineState::Ready(pipeline) => Some(f(pipeline)),
        PipelineState::Uninit | PipelineState::Unavailable => None,
    }
}

/// `true` when a HOT GPU pipeline can be built on this machine.
///
/// Attempts the one-time initialization if it has not run yet; a
/// machine without a usable OpenCL GPU simply returns `false` (and
/// caches that answer). Useful for tests and for callers that want to
/// know whether [`crate::change::ChangeFlags::HOT`] will take the GPU
/// path or fall back to the CPU reference.
#[must_use]
pub fn gpu_available() -> bool {
    with_pipeline(|_| ()).is_some()
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
    with_pipeline(
        |pipeline| match enqueue(pipeline, a, b, a_len, b_len, cap_value) {
            Ok(distance) => Some(distance),
            Err(error) => {
                log_debug!("the HOT change distance fell back to the CPU path: {error}");
                None
            }
        },
    )
    .flatten()
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
