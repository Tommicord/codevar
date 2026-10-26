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

//! The device side of the Packlow codec.
//!
//! The entropy coder is a serial pipeline and stays on the host, but the
//! two expensive stages of the packer - the block signatures the shuffle
//! table keys on, and the classification of the difference between a block
//! and its candidate - are a pure map over independent work items and run
//! on the device.
//!
//! Every entry point here is best effort: any failure - no OpenCL loader,
//! no GPU device, a driver that rejects the SPIR-V image, an allocation
//! error - yields [`None`] so the caller keeps using the CPU path, which
//! always produces the same result.

use alloc::vec;
use alloc::vec::Vec;

use codevar_logger::log_warn;
use codevar_ocl::{Arg, Buffer, CommandQueue, Context, DeviceKind, Program, Runtime};
use codevar_oclc::kernels;
use spin::Mutex;

/// The kernel the signatures come from.
const SIGNATURE_KERNEL: &str = "block_signatures";
/// The kernel the prefix lengths come from.
const PREFIX_KERNEL: &str = "common_prefix";
/// The kernel the corrections come from.
const CLASSIFY_KERNEL: &str = "classify_deltas";

/// The device pipeline, cached for the lifetime of the process.
///
/// The program is kept rather than the kernels: a kernel handle holds the
/// arguments of one launch, and a launch wants its own handle.
struct GpuPipeline {
    queue: CommandQueue,
    program: Program,
}

/// The state of the cached pipeline.
enum PipelineState {
    Uninit,
    Unavailable,
    Ready(GpuPipeline),
}

static PIPELINE: Mutex<PipelineState> = Mutex::new(PipelineState::Uninit);

/// Why the device path could not be set up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineError {
    NoGpuDevice,
    OpenCl,
}

/// Builds the device pipeline.
fn build_pipeline() -> Result<GpuPipeline, PipelineError> {
    let runtime = Runtime::load().map_err(|_| PipelineError::OpenCl)?;
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
    let context = Context::new(&device).map_err(|_| PipelineError::OpenCl)?;
    let queue = CommandQueue::new(&context, &device).map_err(|_| PipelineError::OpenCl)?;
    let program =
        Program::from_il(&context, kernels::packlow::PACKLOW_SPIRV).map_err(|_| PipelineError::OpenCl)?;
    program
        .build("")
        .map_err(|_| PipelineError::OpenCl)?;
    Ok(GpuPipeline { queue, program })
}

/// Runs `f` with the device pipeline, or reports that it is unavailable.
fn with_pipeline<R>(f: impl FnOnce(&mut GpuPipeline) -> R) -> Option<R> {
    let mut state = PIPELINE.lock();
    if matches!(*state, PipelineState::Uninit) {
        match build_pipeline() {
            Ok(pipeline) => *state = PipelineState::Ready(pipeline),
            Err(error) => {
                log_warn!("the Packlow device pipeline is unavailable, using the CPU path: {error:?}");
                *state = PipelineState::Unavailable;
            }
        }
    }
    match &mut *state {
        PipelineState::Ready(pipeline) => Some(f(pipeline)),
        PipelineState::Uninit | PipelineState::Unavailable => None,
    }
}

/// A contiguous arena of blocks the device can read.
///
/// The kernels index into one buffer with an offset table, so the blocks
/// have to live next to each other in memory.
#[derive(Debug, Clone, Default)]
pub struct BlockSet {
    data: Vec<u8>,
    offsets: Vec<i32>,
    lengths: Vec<i32>,
}

impl BlockSet {
    /// Copies `blocks` into a contiguous arena.
    #[must_use]
    pub fn new(blocks: &[&[u8]]) -> Self {
        let mut set = Self::default();
        for block in blocks {
            set.offsets
                .push(i32::try_from(set.data.len()).unwrap_or(i32::MAX));
            set.lengths
                .push(i32::try_from(block.len()).unwrap_or(i32::MAX));
            set.data.extend_from_slice(block);
        }
        set
    }

    /// How many blocks the set holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// The corrections the device computed for a set of pairs.
#[derive(Debug, Clone, Default)]
pub struct DeltaClassification {
    /// The scale exponent of every byte, two bytes per pair and byte.
    pub exponent: Vec<u8>,
    /// The biased signed offset of every byte.
    pub number: Vec<u8>,
    /// How many leading bytes of every pair are reachable.
    pub representable: Vec<i32>,
}

/// Computes the block signatures on the device.
///
/// Returns [`None`] when the device path is unavailable, in which case
/// the caller uses [`crate::simd::block_signature`].
#[must_use]
pub fn try_block_signatures(blocks: &BlockSet) -> Option<Vec<u64>> {
    if blocks.is_empty() {
        return Some(Vec::new());
    }
    with_pipeline(|pipeline| {
        let GpuPipeline { queue, program } = pipeline;
        let mut kernel = program.kernel(SIGNATURE_KERNEL).ok()?;
        let context = queue.context();
        let count = i32::try_from(blocks.len()).unwrap_or(i32::MAX);
        let data = Buffer::from_slice(context, &blocks.data).ok()?;
        let offsets = Buffer::from_slice(context, &blocks.offsets).ok()?;
        let lengths = Buffer::from_slice(context, &blocks.lengths).ok()?;
        let mut out = vec![0i64; blocks.len()];
        let signatures_out = Buffer::new(context, core::mem::size_of_val(&out)).ok()?;
        kernel.set_arg(0, Arg::Buffer(&data)).ok()?;
        kernel.set_arg(1, Arg::Buffer(&offsets)).ok()?;
        kernel.set_arg(2, Arg::Buffer(&lengths)).ok()?;
        kernel.set_arg(3, Arg::I32(count)).ok()?;
        kernel
            .set_arg(4, Arg::Buffer(&signatures_out))
            .ok()?;
        let event = queue
            .enqueue_kernel(&kernel, &[blocks.len()], None)
            .ok()?;
        event.wait().ok()?;
        queue
            .read_buffer(&signatures_out, 0, &mut out)
            .ok()?;
        Some(
            out.into_iter()
                .map(|value| value as u64)
                .collect(),
        )
    })?
}

/// Measures how far each target agrees with its candidate on the device.
///
/// `pairs` names the target and the candidate of every pair.
#[must_use]
pub fn try_common_prefix(targets: &BlockSet, sources: &BlockSet, pairs: &[(i32, i32)]) -> Option<Vec<i32>> {
    if pairs.is_empty() {
        return Some(Vec::new());
    }
    with_pipeline(|pipeline| {
        let GpuPipeline { queue, program } = pipeline;
        let mut kernel = program.kernel(PREFIX_KERNEL).ok()?;
        let context = queue.context();
        let count = i32::try_from(pairs.len()).unwrap_or(i32::MAX);
        let target_data = Buffer::from_slice(context, &targets.data).ok()?;
        let target_offsets = Buffer::from_slice(context, &targets.offsets).ok()?;
        let target_lengths = Buffer::from_slice(context, &targets.lengths).ok()?;
        let source_data = Buffer::from_slice(context, &sources.data).ok()?;
        let source_offsets = Buffer::from_slice(context, &sources.offsets).ok()?;
        let pair_target: Vec<i32> = pairs.iter().map(|(t, _)| *t).collect();
        let pair_source: Vec<i32> = pairs.iter().map(|(_, s)| *s).collect();
        let pair_target_buf = Buffer::from_slice(context, &pair_target).ok()?;
        let pair_source_buf = Buffer::from_slice(context, &pair_source).ok()?;
        let mut out = vec![0i32; pairs.len()];
        let out_buf = Buffer::new(context, core::mem::size_of_val(&out)).ok()?;
        kernel
            .set_arg(0, Arg::Buffer(&target_data))
            .ok()?;
        kernel
            .set_arg(1, Arg::Buffer(&target_offsets))
            .ok()?;
        kernel
            .set_arg(2, Arg::Buffer(&target_lengths))
            .ok()?;
        kernel
            .set_arg(3, Arg::Buffer(&source_data))
            .ok()?;
        kernel
            .set_arg(4, Arg::Buffer(&source_offsets))
            .ok()?;
        kernel
            .set_arg(5, Arg::Buffer(&pair_target_buf))
            .ok()?;
        kernel
            .set_arg(6, Arg::Buffer(&pair_source_buf))
            .ok()?;
        kernel.set_arg(7, Arg::I32(count)).ok()?;
        kernel.set_arg(8, Arg::Buffer(&out_buf)).ok()?;
        let event = queue
            .enqueue_kernel(&kernel, &[pairs.len()], None)
            .ok()?;
        event.wait().ok()?;
        queue.read_buffer(&out_buf, 0, &mut out).ok()?;
        Some(out)
    })?
}

/// Classifies the difference of every pair on the device.
///
/// `width` is the longest block the host asked about; the device writes
/// `pairs.len() * width` bytes of corrections.
#[must_use]
pub fn try_classify_deltas(
    targets: &BlockSet,
    sources: &BlockSet,
    pairs: &[(i32, i32)],
    width: usize,
) -> Option<DeltaClassification> {
    if pairs.is_empty() {
        return Some(DeltaClassification::default());
    }
    with_pipeline(|pipeline| {
        let GpuPipeline { queue, program } = pipeline;
        let mut kernel = program.kernel(CLASSIFY_KERNEL).ok()?;
        let context = queue.context();
        let count = i32::try_from(pairs.len()).unwrap_or(i32::MAX);
        let width_i32 = i32::try_from(width).unwrap_or(i32::MAX);
        let target_data = Buffer::from_slice(context, &targets.data).ok()?;
        let target_offsets = Buffer::from_slice(context, &targets.offsets).ok()?;
        let target_lengths = Buffer::from_slice(context, &targets.lengths).ok()?;
        let source_data = Buffer::from_slice(context, &sources.data).ok()?;
        let source_offsets = Buffer::from_slice(context, &sources.offsets).ok()?;
        let pair_target: Vec<i32> = pairs.iter().map(|(t, _)| *t).collect();
        let pair_source: Vec<i32> = pairs.iter().map(|(_, s)| *s).collect();
        let pair_target_buf = Buffer::from_slice(context, &pair_target).ok()?;
        let pair_source_buf = Buffer::from_slice(context, &pair_source).ok()?;
        let cells = pairs.len().checked_mul(width)?;
        let mut exponent = vec![0u8; cells];
        let mut number = vec![0u8; cells];
        let mut representable = vec![0i32; pairs.len()];
        let exponent_buf = Buffer::new(context, cells).ok()?;
        let number_buf = Buffer::new(context, cells).ok()?;
        let representable_buf = Buffer::new(context, core::mem::size_of_val(&representable)).ok()?;
        kernel
            .set_arg(0, Arg::Buffer(&target_data))
            .ok()?;
        kernel
            .set_arg(1, Arg::Buffer(&target_offsets))
            .ok()?;
        kernel
            .set_arg(2, Arg::Buffer(&target_lengths))
            .ok()?;
        kernel
            .set_arg(3, Arg::Buffer(&source_data))
            .ok()?;
        kernel
            .set_arg(4, Arg::Buffer(&source_offsets))
            .ok()?;
        kernel
            .set_arg(5, Arg::Buffer(&pair_target_buf))
            .ok()?;
        kernel
            .set_arg(6, Arg::Buffer(&pair_source_buf))
            .ok()?;
        kernel.set_arg(7, Arg::I32(count)).ok()?;
        kernel.set_arg(8, Arg::I32(width_i32)).ok()?;
        kernel
            .set_arg(9, Arg::Buffer(&exponent_buf))
            .ok()?;
        kernel
            .set_arg(10, Arg::Buffer(&number_buf))
            .ok()?;
        kernel
            .set_arg(11, Arg::Buffer(&representable_buf))
            .ok()?;
        let event = queue
            .enqueue_kernel(&kernel, &[pairs.len()], None)
            .ok()?;
        event.wait().ok()?;
        queue
            .read_buffer(&exponent_buf, 0, &mut exponent)
            .ok()?;
        queue
            .read_buffer(&number_buf, 0, &mut number)
            .ok()?;
        queue
            .read_buffer(&representable_buf, 0, &mut representable)
            .ok()?;
        Some(DeltaClassification {
            exponent,
            number,
            representable,
        })
    })?
}

/// Whether the device path can be used at all.
///
/// The probe builds and caches the pipeline, so a later call does not pay
/// for the setup again.
#[must_use]
pub fn gpu_available() -> bool {
    with_pipeline(|_| ()).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;

    /// A deterministic pseudo random byte generator.
    fn pseudo_random(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect()
    }

    #[test]
    fn a_block_set_keeps_the_blocks_where_they_belong() {
        let blocks = [pseudo_random(10, 1), pseudo_random(20, 2), pseudo_random(30, 3)];
        let refs: Vec<&[u8]> = blocks
            .iter()
            .map(|block| block.as_slice())
            .collect();
        let set = BlockSet::new(&refs);
        assert_eq!(set.len(), 3);
        assert!(!set.is_empty());
        let mut offset = 0usize;
        for (index, block) in blocks.iter().enumerate() {
            assert_eq!(set.offsets[index], i32::try_from(offset).unwrap_or(i32::MAX));
            assert_eq!(set.lengths[index], i32::try_from(block.len()).unwrap_or(i32::MAX));
            assert_eq!(&set.data[offset..offset + block.len()], block.as_slice());
            offset += block.len();
        }
    }

    #[test]
    fn an_empty_block_set_is_empty() {
        let set = BlockSet::new(&[]);
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn the_device_path_never_panics() {
        // There may or may not be a device; either way the calls have to
        // answer, and the answer has to agree with the host.
        let blocks = [pseudo_random(4096, 5), pseudo_random(4096, 6)];
        let refs: Vec<&[u8]> = blocks
            .iter()
            .map(|block| block.as_slice())
            .collect();
        let set = BlockSet::new(&refs);
        let host: Vec<u64> = blocks
            .iter()
            .map(|block| simd::block_signature(block))
            .collect();
        if let Some(device) = try_block_signatures(&set) {
            assert_eq!(device, host, "the device must agree with the host");
        }
    }

    #[test]
    fn the_device_classification_agrees_with_the_host() {
        let blocks = [pseudo_random(1024, 7), pseudo_random(1024, 8)];
        let mut near = blocks.clone();
        near[0][0] = near[0][0].wrapping_add(1);
        let block_refs: Vec<&[u8]> = blocks
            .iter()
            .map(|block| block.as_slice())
            .collect();
        let near_refs: Vec<&[u8]> = near
            .iter()
            .map(|block| block.as_slice())
            .collect();
        let sources = BlockSet::new(&block_refs);
        let targets = BlockSet::new(&near_refs);
        let pairs = vec![(0i32, 0i32), (1i32, 1i32)];
        let host: Vec<simd::DeltaPlan> = pairs
            .iter()
            .map(|(target, source)| {
                simd::plan_delta(block_refs[*source as usize], near_refs[*target as usize])
            })
            .collect();
        if let Some(classification) = try_classify_deltas(&targets, &sources, &pairs, 1024) {
            for (index, plan) in host.iter().enumerate() {
                assert_eq!(
                    classification.representable[index],
                    i32::try_from(plan.representable).unwrap_or(i32::MAX)
                );
                for byte in 0..plan.representable {
                    assert_eq!(classification.exponent[index * 1024 + byte], plan.exponent[byte]);
                    assert_eq!(classification.number[index * 1024 + byte], plan.number[byte]);
                }
            }
        }
    }
}
