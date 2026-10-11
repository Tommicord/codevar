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

//! Hardware (HOT) GPU path of the Wredit engine.
//!
//! This module is the counterpart of `codevar-truetype-smooth`'s
//! `hot` module: it only runs when the caller passes the **HOT flag**
//! down to an operation. It never decides on its own — the CPU path
//! ([`crate::simd`] and the scalar loops) stays the default, and a
//! HOT-flagged call first tries one of the `try_*` functions here.
//!
//! Every `try_*` function returns [`None`] when the result could not
//! be produced on the GPU, which is the signal for the caller to run
//! the CPU path instead. Falling back covers, in order: no OpenCL
//! loader, no available GPU device, a driver that rejects the
//! embedded SPIR-V image, a length that does not fit the kernel's
//! 32-bit parameters, and any allocation, enqueue, or wait failure.
//! Each cause is logged exactly once per process.
//!
//! | HOT entry point | Kernel | CPU fallback |
//! |-----------------|--------|--------------|
//! | [`try_count_utf8_units`] | `char_count` | [`crate::simd::count_utf8_units`] |
//! | [`try_linear_checksum`] | `block_checksum` | [`crate::simd::checksum_scalar`] |
//! | [`try_history_checksum`] | `history_checksum` | [`crate::simd::checksum_scalar`] |
//! | [`try_count_newlines`] | `count_newlines` | scalar newline scan |
//! | [`try_gap_compact`] | `gap_compact` | scalar gap compaction |
//! | [`try_levenshtein`] | `levenshtein` | [`crate::change::levenshtein_capped`] |
//!
//! # Kernel embedding
//!
//! The kernels live in `crates/codevar-oclc/kernels/` and are
//! compiled to SPIR-V at build time by `codevar-oclc`, which turns
//! any kernel syntax error into a build failure, the Rust side below
//! only ever sees programs that already parsed, analyzed and lowered
//! cleanly. The pipeline (device, context, queue, programs, kernels)
//! is built once on first use and cached for the lifetime of the
//! process.

use alloc::vec;
use alloc::vec::Vec;

use codevar_logger::{log_debug, log_warn};
use codevar_ocl::{Arg, Buffer, CommandQueue, Context, DeviceKind, Kernel, Program, Runtime};
use codevar_oclc::kernels;
use spin::Mutex;

use crate::error::{WreditError, WreditResult};

/// Bytes (or code points) per work item in the scanning kernels.
///
/// Must match the chunk size documented in `block.cl`,
/// `text.cl` and `history.cl`.
pub(crate) const GPU_CHUNK: usize = 0x40;

/// Longest sequence the Levenshtein kernel accepts.
///
/// The kernel addresses two DP rows of `b_len + 1` integers in one
/// scratch buffer and adds `max_dist` to row indices, so lengths are
/// capped at half of `i32` minus a safety margin; everything stays
/// positive in 32-bit arithmetic. Larger inputs answer [`None`] and
/// run on the CPU path instead.
const GPU_MAX_SEQUENCE: usize = (i32::MAX as usize / 2) - 2;

/// A ready-to-enqueue HOT pipeline: one shared queue plus every
/// kernel module the crate embeds.
///
/// A [`None`] entry means that module failed to build (rejected
/// SPIR-V, driver quirk); the other kernels stay usable, so one
/// broken program degrades a single operation instead of the crate.
struct Pipeline {
    /// In-order command queue for the cached device.
    queue: CommandQueue,
    /// `char_count` from `block.cl`.
    char_count: Option<Kernel>,
    /// `block_checksum` from `block.cl`.
    block_checksum: Option<Kernel>,
    /// `gap_compact` from `gap.cl`.
    gap_compact: Option<Kernel>,
    /// `count_newlines` from `text.cl`.
    count_newlines: Option<Kernel>,
    /// `history_checksum` from `history.cl`.
    history_checksum: Option<Kernel>,
    /// `levenshtein` from `levenshtein.cl`.
    levenshtein: Option<Kernel>,
}

impl Pipeline {
    /// Builds the device, the queue and every embedded kernel.
    ///
    /// [`None`] means the machine has no usable OpenCL GPU at all;
    /// individual kernel failures are recorded as [`None`] entries.
    fn build() -> Option<Self> {
        let (context, queue) = open_device()?;
        let mut pipeline = Self {
            queue,
            char_count: None,
            block_checksum: None,
            gap_compact: None,
            count_newlines: None,
            history_checksum: None,
            levenshtein: None,
        };
        pipeline.char_count = build_kernel(&context, kernels::wredit::BLOCK_SPIRV, "char_count");
        pipeline.block_checksum = build_kernel(&context, kernels::wredit::BLOCK_SPIRV, "block_checksum");
        pipeline.gap_compact = build_kernel(&context, kernels::wredit::GAP_SPIRV, "gap_compact");
        pipeline.count_newlines = build_kernel(&context, kernels::wredit::TEXT_SPIRV, "count_newlines");
        pipeline.history_checksum =
            build_kernel(&context, kernels::wredit::HISTORY_SPIRV, "history_checksum");
        pipeline.levenshtein = build_kernel(&context, kernels::wredit::LEVENSHTEIN_SPIRV, "levenshtein");
        Some(pipeline)
    }

    /// `true` when at least one kernel is ready to launch.
    fn any_ready(&self) -> bool {
        self.char_count.is_some()
            || self.block_checksum.is_some()
            || self.gap_compact.is_some()
            || self.count_newlines.is_some()
            || self.history_checksum.is_some()
            || self.levenshtein.is_some()
    }
}

/// Process-wide pipeline state, built lazily on the first HOT call.
///
/// The `Ready` variant is far larger than the two markers, but the
/// whole enum sits in a process-wide static behind a mutex and is
/// never cloned or swapped by value, so boxing it would only add an
/// indirection to a struct that is built once.
#[allow(clippy::large_enum_variant)]
enum PipelineState {
    /// Nothing attempted yet.
    Uninit,
    /// A build attempt failed; never touch the driver again.
    Unavailable,
    /// The pipeline is ready and reused for every subsequent call.
    Ready(Pipeline),
}

/// The cached pipeline; the mutex also serializes concurrent HOT
/// launches, which matches the serialized nature of a single queue.
static PIPELINE: Mutex<PipelineState> = Mutex::new(PipelineState::Uninit);

/// Loads the OpenCL loader and picks the first available GPU device.
///
/// Returns [`None`] when the machine has no usable device — the common
/// case on CI and in containers — so HOT calls fall back to the CPU
/// without retrying the driver.
fn open_device() -> Option<(Context, CommandQueue)> {
    let runtime = match Runtime::load() {
        Ok(runtime) => runtime,
        Err(error) => {
            log_debug!("the Wredit HOT path is unavailable: {error}");
            return None;
        }
    };
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
    let device = selected?;
    let context = Context::new(&device).ok()?;
    let queue = CommandQueue::new(&context, &device).ok()?;
    Some((context, queue))
}

/// Compiles one embedded SPIR-V module and extracts `entry_point`.
///
/// Failures are logged once per kernel and reported as [`None`] so
/// the remaining kernels stay usable.
fn build_kernel(context: &Context, spirv: &'static [u8], entry_point: &'static str) -> Option<Kernel> {
    let built = (|| -> Result<Kernel, codevar_ocl::Error> {
        let program = Program::from_il(context, spirv)?;
        program.build("")?;
        program.kernel(entry_point)
    })();
    match built {
        Ok(kernel) => Some(kernel),
        Err(error) => {
            log_warn!("the `{entry_point}` kernel is unusable, its operation stays on the CPU: {error}");
            None
        }
    }
}

/// Runs `f` with exclusive access to the cached pipeline, initializing
/// it on first use. Returns [`None`] when the pipeline is unavailable;
/// the initialization failure is logged exactly once.
fn with_pipeline<R>(f: impl FnOnce(&mut Pipeline) -> Option<R>) -> Option<R> {
    let mut state = PIPELINE.lock();
    if matches!(*state, PipelineState::Uninit) {
        *state = match Pipeline::build() {
            Some(pipeline) if pipeline.any_ready() => PipelineState::Ready(pipeline),
            _ => PipelineState::Unavailable,
        };
        if matches!(*state, PipelineState::Unavailable) {
            log_debug!("no Wredit GPU kernel could be built, using the CPU path from now on");
        }
    }
    match &mut *state {
        PipelineState::Ready(pipeline) => f(pipeline),
        PipelineState::Uninit | PipelineState::Unavailable => None,
    }
}

/// Converts a length to the kernels' 32-bit parameter, or [`None`].
#[inline]
fn as_kernel_len(len: usize) -> Option<i32> {
    match i32::try_from(len) {
        Ok(value) => Some(value),
        Err(_) => {
            log_warn!("input of {len} elements exceeds the GPU kernel's 32-bit length parameter");
            None
        }
    }
}

/// Number of work items covering `len` elements (at least one,
/// because an OpenCL global size may not be zero).
#[inline]
fn work_items(len: usize) -> usize {
    len.div_ceil(GPU_CHUNK).max(1)
}

/// Launches `kernel` over `items` work items and waits for completion.
///
/// # Errors
///
/// Returns [`WreditError::GpuFailed`] when the enqueue or the wait
/// fails; callers log and fall back to the CPU path.
fn launch(queue: &CommandQueue, kernel: &Kernel, items: usize) -> WreditResult<()> {
    let event = queue
        .enqueue_kernel(kernel, &[items], None)
        .map_err(|_| WreditError::GpuFailed {
            what: "enqueueing the kernel failed",
        })?;
    event.wait().map_err(|_| WreditError::GpuFailed {
        what: "waiting for the kernel failed",
    })
}

/// Uploads `input`, runs `kernel`, and reads back `items` partials.
///
/// Returns [`None`] on any driver, allocation, or launch failure so
/// the caller falls back to the CPU path; the `configure` closure
/// sets the kernel's scalar arguments in signature order.
fn run_partial_kernel<T>(
    queue: &CommandQueue,
    kernel: &mut Kernel,
    input: &[T],
    items: usize,
    configure: impl FnOnce(&mut Kernel) -> Option<()>,
) -> Option<Vec<i32>> {
    let context = queue.context();
    let input_buffer = Buffer::from_slice(context, input).ok()?;
    let output = Buffer::new(context, items * core::mem::size_of::<i32>()).ok()?;
    kernel
        .set_arg(0, Arg::Buffer(&input_buffer))
        .ok()?;
    configure(kernel)?;
    kernel.set_arg(2, Arg::Buffer(&output)).ok()?;
    launch(queue, kernel, items).ok()?;
    let mut partials = vec![0i32; items];
    queue
        .read_buffer(&output, 0, &mut partials)
        .ok()?;
    Some(partials)
}

/// Counts UTF-8 code units on the GPU.
///
/// Callers holding the HOT flag try this first and fall back to
/// [`crate::simd::count_utf8_units`] when it answers [`None`].
///
/// # Examples
///
/// ```
/// use codevar_wredit::{hot, simd};
///
/// let bytes = "héllo".as_bytes();
/// let count = hot::try_count_utf8_units(bytes).unwrap_or_else(|| simd::count_utf8_units(bytes));
/// assert_eq!(count, 5);
/// ```
pub fn try_count_utf8_units(bytes: &[u8]) -> Option<u64> {
    if bytes.is_empty() {
        return Some(0);
    }
    let len = as_kernel_len(bytes.len())?;
    let items = work_items(bytes.len());
    with_pipeline(|pipeline| {
        let Pipeline {
            queue, char_count, ..
        } = pipeline;
        let kernel = char_count.as_mut()?;
        let partials = run_partial_kernel(queue, kernel, bytes, items, |kernel| {
            kernel.set_arg(1, Arg::I32(len)).ok().map(|_| ())
        })?;
        Some(
            partials
                .iter()
                .map(|&value| u64::from(value as u32))
                .sum(),
        )
    })
}

/// Which checksum kernel [`reduce_checksum`] should launch.
#[derive(Clone, Copy)]
enum ChecksumKind {
    /// `block_checksum`, used by the block storage layer.
    Block,
    /// `history_checksum`, used by the history layer.
    History,
}

/// Computes [`crate::simd::linear_checksum`] on the GPU.
///
/// The host reduces the per-chunk partials modulo 2^30 exactly the
/// way [`crate::simd::checksum_scalar`] does, so both paths agree on
/// every input.
///
/// # Examples
///
/// ```
/// use codevar_wredit::{hot, simd};
///
/// let bytes = b"AB";
/// let sum = hot::try_linear_checksum(bytes).unwrap_or_else(|| simd::linear_checksum(bytes));
/// assert_eq!(sum, 197);
/// ```
pub fn try_linear_checksum(bytes: &[u8]) -> Option<u32> {
    reduce_checksum(bytes, ChecksumKind::Block)
}

/// Computes the history integrity checksum on the GPU.
///
/// Uses `history_checksum`, which is byte-identical in
/// behaviour to the block checksum kernel.
pub fn try_history_checksum(bytes: &[u8]) -> Option<u32> {
    reduce_checksum(bytes, ChecksumKind::History)
}

/// Shared body of the two checksum kernels.
fn reduce_checksum(bytes: &[u8], kind: ChecksumKind) -> Option<u32> {
    if bytes.is_empty() {
        return Some(0);
    }
    let len = as_kernel_len(bytes.len())?;
    let items = work_items(bytes.len());
    with_pipeline(|pipeline| {
        let Pipeline {
            queue,
            block_checksum,
            history_checksum,
            ..
        } = pipeline;
        let kernel = match kind {
            ChecksumKind::Block => block_checksum,
            ChecksumKind::History => history_checksum,
        }
        .as_mut()?;
        let partials = run_partial_kernel(queue, kernel, bytes, items, |kernel| {
            kernel.set_arg(1, Arg::I32(len)).ok().map(|_| ())
        })?;
        let mut sum = 0u64;
        for &value in &partials {
            sum += u64::from(value as u32);
        }
        Some((sum & u64::from(u32::MAX >> 2)) as u32)
    })
}

/// Counts U+000A code points on the GPU.
///
/// Callers holding the HOT flag try this first and fall back to a
/// scalar newline scan when it answers [`None`].
pub fn try_count_newlines(code_points: &[u32]) -> Option<u64> {
    if code_points.is_empty() {
        return Some(0);
    }
    let len = as_kernel_len(code_points.len())?;
    let items = work_items(code_points.len());
    with_pipeline(|pipeline| {
        let Pipeline {
            queue,
            count_newlines,
            ..
        } = pipeline;
        let kernel = count_newlines.as_mut()?;
        let partials = run_partial_kernel(queue, kernel, code_points, items, |kernel| {
            kernel.set_arg(1, Arg::I32(len)).ok().map(|_| ())
        })?;
        Some(
            partials
                .iter()
                .map(|&value| u64::from(value as u32))
                .sum(),
        )
    })
}

/// Closes a gap in a code-point buffer on the GPU.
///
/// `src` holds `len + gap_len` code points with the free run
/// `[gap_start, gap_start + gap_len)`; the returned vector holds the
/// compacted `len` elements. Callers holding the HOT flag try this
/// first and fall back to scalar compaction when it answers [`None`].
///
/// # Examples
///
/// ```
/// use codevar_wredit::hot;
///
/// let src = [1u32, 2, 3, 4];
/// let compacted = hot::try_gap_compact(&src, 1, 2).unwrap_or_else(|| {
///     let mut out = src[..1].to_vec();
///     out.extend_from_slice(&src[3..]);
///     out
/// });
/// assert_eq!(compacted, vec![1, 4]);
/// ```
pub fn try_gap_compact(src: &[u32], gap_start: usize, gap_len: usize) -> Option<Vec<u32>> {
    let out_len = src.len().checked_sub(gap_len)?;
    if gap_start > out_len {
        return None;
    }
    if out_len == 0 {
        return Some(Vec::new());
    }
    let len = as_kernel_len(out_len)?;
    let gap_start = as_kernel_len(gap_start)?;
    let gap_len = as_kernel_len(gap_len)?;
    with_pipeline(|pipeline| {
        let Pipeline {
            queue, gap_compact, ..
        } = pipeline;
        let kernel = gap_compact.as_mut()?;
        let context = queue.context();
        let input = Buffer::from_slice(context, src).ok()?;
        let output = Buffer::new(context, out_len * core::mem::size_of::<u32>()).ok()?;
        kernel.set_arg(0, Arg::Buffer(&input)).ok()?;
        kernel.set_arg(1, Arg::I32(len)).ok()?;
        kernel.set_arg(2, Arg::I32(gap_start)).ok()?;
        kernel.set_arg(3, Arg::I32(gap_len)).ok()?;
        kernel.set_arg(4, Arg::Buffer(&output)).ok()?;
        launch(queue, kernel, out_len).ok()?;
        let mut out = vec![0u32; out_len];
        queue.read_buffer(&output, 0, &mut out).ok()?;
        Some(out)
    })
}

/// Computes the capped Levenshtein distance between two code-point
/// sequences on the GPU.
///
/// The result is `min(distance, max_dist + 1)`: the exact edit
/// distance while it does not exceed `max_dist`, and `max_dist + 1`
/// once the sequences are further apart — the caller asked only for a
/// yes/no decision ("similar enough to blend") and pays no cost for
/// precision beyond the cap. A band of `2 * max_dist + 1` cells is
/// evaluated per row instead of the full matrix.
///
/// Callers holding the HOT flag try this first and fall back to
/// [`crate::change::levenshtein_capped`] when it answers [`None`].
///
/// # Examples
///
/// ```
/// use codevar_wredit::{change, hot};
///
/// let a = [1u32, 2, 3];
/// let b = [1u32, 3, 3];
/// let distance = hot::try_levenshtein(&a, &b, 4).unwrap_or_else(|| {
///     change::levenshtein_capped(&a, &b, 4)
/// });
/// assert_eq!(distance, 1);
/// ```
pub fn try_levenshtein(a: &[u32], b: &[u32], max_dist: u32) -> Option<u32> {
    if a.len() > GPU_MAX_SEQUENCE || b.len() > GPU_MAX_SEQUENCE {
        log_warn!(
            "a sequence of {} / {} code points exceeds the Levenshtein kernel's limit, using the CPU path",
            a.len(),
            b.len()
        );
        return None;
    }
    // Clamp the cap to the longest possible distance: a band wider
    // than both sequences can never change the saturated answer.
    let longest = u32::try_from(a.len().max(b.len())).unwrap_or(u32::MAX);
    let cap = max_dist.min(longest);
    let sat = cap.saturating_add(1);
    // Empty inputs never touch the pipeline.
    if a.is_empty() {
        let b_len = u32::try_from(b.len()).unwrap_or(u32::MAX);
        return Some(b_len.min(sat));
    }
    if b.is_empty() {
        let a_len = u32::try_from(a.len()).unwrap_or(u32::MAX);
        return Some(a_len.min(sat));
    }
    let a_len = as_kernel_len(a.len())?;
    let b_len = as_kernel_len(b.len())?;
    let cap = as_kernel_len(cap as usize)?;
    let row_len = b.len().checked_add(1)?;
    let scratch_len = row_len.checked_mul(2)?;
    with_pipeline(|pipeline| {
        let Pipeline {
            queue, levenshtein, ..
        } = pipeline;
        let kernel = levenshtein.as_mut()?;
        let context = queue.context();
        let a_buffer = Buffer::from_slice(context, a).ok()?;
        let b_buffer = Buffer::from_slice(context, b).ok()?;
        let work = Buffer::new(context, scratch_len.checked_mul(core::mem::size_of::<i32>())?).ok()?;
        let out = Buffer::new(context, core::mem::size_of::<i32>()).ok()?;
        kernel.set_arg(0, Arg::Buffer(&a_buffer)).ok()?;
        kernel.set_arg(1, Arg::I32(a_len)).ok()?;
        kernel.set_arg(2, Arg::Buffer(&b_buffer)).ok()?;
        kernel.set_arg(3, Arg::I32(b_len)).ok()?;
        kernel.set_arg(4, Arg::I32(cap)).ok()?;
        kernel.set_arg(5, Arg::Buffer(&work)).ok()?;
        kernel.set_arg(6, Arg::Buffer(&out)).ok()?;
        launch(queue, kernel, 1).ok()?;
        let mut result = [0i32; 1];
        queue.read_buffer(&out, 0, &mut result).ok()?;
        u32::try_from(result[0]).ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{checksum_scalar, count_utf8_units};

    #[test]
    fn kernel_chunk_and_work_item_math() {
        assert_eq!(work_items(0), 1);
        assert_eq!(work_items(1), 1);
        assert_eq!(work_items(GPU_CHUNK), 1);
        assert_eq!(work_items(GPU_CHUNK + 1), 2);
        assert_eq!(as_kernel_len(4096), Some(4096));
        assert_eq!(as_kernel_len(usize::MAX), None);
    }

    #[test]
    fn hot_helpers_handle_empty_input_without_touching_the_driver() {
        assert_eq!(try_count_utf8_units(b""), Some(0));
        assert_eq!(try_linear_checksum(b""), Some(0));
        assert_eq!(try_count_newlines(&[]), Some(0));
        assert_eq!(try_gap_compact(&[], 0, 0), Some(Vec::new()));
        assert_eq!(try_history_checksum(b""), Some(0));
    }

    #[test]
    fn gap_geometry_is_validated_before_any_dispatch() {
        // `gap_start` past the output range is a caller bug, not a
        // GPU problem: report it without touching the pipeline.
        assert_eq!(try_gap_compact(&[1, 2, 3], 4, 1), None);
        // Deleting more than the buffer holds has no output.
        assert_eq!(try_gap_compact(&[1, 2, 3], 0, 4), None);
    }

    /// When a device *is* usable, every GPU result must equal the
    /// CPU reference; when it is not, the `try_*` functions answer
    /// [`None`] and the caller falls back. Both outcomes are valid
    /// here, so the assertions are conditional on success.
    #[test]
    fn hot_paths_agree_with_the_cpu_reference() {
        let bytes: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        if let Some(gpu) = try_count_utf8_units(&bytes) {
            assert_eq!(gpu, count_utf8_units(&bytes));
        }
        if let Some(gpu) = try_linear_checksum(&bytes) {
            assert_eq!(gpu, checksum_scalar(&bytes));
        }
        if let Some(gpu) = try_history_checksum(&bytes) {
            assert_eq!(gpu, checksum_scalar(&bytes));
        }
        let text: Vec<u32> = (0..1000u32)
            .map(|i| if i % 7 == 0 { 10 } else { 65 + i % 26 })
            .collect();
        let expected = u64::try_from(text.iter().filter(|&&cp| cp == 10).count()).unwrap_or(0);
        if let Some(gpu) = try_count_newlines(&text) {
            assert_eq!(gpu, expected);
        }
        let src: Vec<u32> = (0..300u32).collect();
        let (gap_start, gap_len) = (100usize, 40usize);
        let mut expected: Vec<u32> = src[..gap_start].to_vec();
        expected.extend_from_slice(&src[gap_start + gap_len..]);
        if let Some(gpu) = try_gap_compact(&src, gap_start, gap_len) {
            assert_eq!(gpu, expected);
        }
    }

    /// The fallback contract: whatever the GPU answers (or refuses
    /// to), the CPU path produces the reference value.
    #[test]
    fn fallback_contract_holds_with_and_without_a_device() {
        let bytes: Vec<u8> = (0..3000u32)
            .map(|i| (i * 7 % 253) as u8)
            .collect();
        let count = try_count_utf8_units(&bytes).unwrap_or_else(|| count_utf8_units(&bytes));
        assert_eq!(count, count_utf8_units(&bytes));
        let sum = try_linear_checksum(&bytes).unwrap_or_else(|| checksum_scalar(&bytes));
        assert_eq!(sum, checksum_scalar(&bytes));
    }
}
