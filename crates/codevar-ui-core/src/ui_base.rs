//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Compositor-based rendering: pipes, offscreen framebuffers and the
//! fragment-shader mix pass.
//!
//! This module provides a *pipe* model: every
//! contributor to a frame is a [`PipeSource`] whose
//! [`PipeSource::pipe_entry`] returns an async [`PipeFuture`]. The
//! compositor drives those futures from [`Compositor::poll_composite`],
//! submits each pipe as its own queue submission chained with binary
//! semaphores, and finally folds every offscreen framebuffer registered
//! through [`PipeSupplyTraits`] into the presentation target with a
//! fullscreen `mix()` pass (`shaders/mix.frag`).
//!
//! # Frame lifecycle
//!
//! 1. [`Compositor::begin_frame`] retires the previous submission,
//!    drains the [`CompositorComm`] channels, refreshes the framebuffer
//!    snapshot and creates the frame's wait semaphore.
//! 2. [`Compositor::poll_composite`] walks three phases: the *layout*
//!    pass transitions every framebuffer to
//!    `COLOR_ATTACHMENT_OPTIMAL`, the *pipe* phase polls and submits the
//!    pipe futures in queue order, and the *mix* pass composites the
//!    framebuffers into the presentation target and hands it off to
//!    `GENERAL`.
//! 3. [`Compositor::end_frame`] exports GPU completion of the mix pass as
//!    a `sync_file` descriptor for the presentation layer.
//!
//! [`Compositor::composite_frame`] and [`Compositor::draw_frame`] chain
//! the steps for async callers; [`block_on`] drives them from
//! synchronous code.
//!
//! # Synchronization
//!
//! Submissions are chained GPU-to-GPU with binary semaphores: the layout
//! pass signals `chain[0]`, pipe *k* waits on the previous signal and
//! signals `chain[k + 1]`, and the mix pass waits on the last signal and
//! signals an exportable semaphore. A single fence covers the whole
//! chain; it is waited on at the start of the next frame (GPU-to-CPU),
//! which also returns every chain semaphore to the unsignaled state so
//! the pools can be reused.
//!
//! The compositor owns the image layouts: a pipe records draws but never
//! transitions the framebuffer it is attached to (see
//! [`PipeCtx::begin_render`]).

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use ash::vk;
use core::fmt;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use spin::Mutex;

use crate::ui_pipeline::{OwnedFd, PipelineContext};

/// Number of offscreen framebuffers the mix pass folds into one frame.
///
/// Matches the `u_frames[8]` array of `shaders/mix.frag`.
pub const MAX_MIX_TARGETS: usize = 8;

/// Largest number of pipes that may be registered at once.
///
/// One chain semaphore is needed for the layout pass plus one per pipe,
/// so the pipe count shares [`MAX_MIX_TARGETS`]'s bound.
pub const MAX_PIPES: usize = MAX_MIX_TARGETS;

/// Depth of each [`CompositorComm`] channel before a non-blocking send
/// reports [`CommError::Full`].
const CHANNEL_CAPACITY: usize = 64;

/// Embedded vertex shader of the mix pass.
static MIX_VERT_SPIRV: &[u8] = include_bytes!("shaders/mix.vert.spv");

/// Embedded fragment shader of the mix pass.
static MIX_FRAG_SPIRV: &[u8] = include_bytes!("shaders/mix.frag.spv");

/// SPIR-V magic number as stored in a little-endian module.
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Lifecycle state of a [`Compositor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompositorState {
    /// No frame resources exist and no frame can be begun.
    Stopped,
    /// The compositor accepts frames (see [`Compositor::begin_frame`]).
    Running,
    /// Rendering is suspended; registrations and resources are kept.
    Paused,
}

impl fmt::Display for CompositorState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stopped => f.write_str("stopped"),
            Self::Running => f.write_str("running"),
            Self::Paused => f.write_str("paused"),
        }
    }
}

/// Progress of the frame currently driven through [`Compositor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompositorPhase {
    /// No frame in progress; the next call may be
    /// [`Compositor::begin_frame`].
    Idle,
    /// [`Compositor::begin_frame`] succeeded: the command pool was reset
    /// and the frame's synchronization objects exist.
    Recording,
    /// The layout pass is on the queue; pipe futures are being driven.
    Pipes,
    /// Every pipe is on the queue; the mix pass is not recorded yet.
    Mixing,
    /// The mix pass is on the queue and its completion has not been
    /// exported by [`Compositor::end_frame`] yet.
    Submitted,
}

impl fmt::Display for CompositorPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => f.write_str("idle"),
            Self::Recording => f.write_str("recording"),
            Self::Pipes => f.write_str("pipes"),
            Self::Mixing => f.write_str("mixing"),
            Self::Submitted => f.write_str("submitted"),
        }
    }
}

/// Error returned by compositor lifecycle, frame, pipe and comm
/// operations.
#[derive(Debug)]
pub enum CompositorError {
    /// An operation was attempted in the wrong lifecycle state.
    InvalidState {
        /// The operation that was attempted (for example `begin_frame`).
        operation: &'static str,
        /// The state the compositor was in.
        state: CompositorState,
    },
    /// A frame operation was attempted while the frame was in another
    /// phase than the one it requires.
    FramePhase {
        /// The operation that was attempted.
        operation: &'static str,
        /// The phase the operation requires.
        expected: CompositorPhase,
        /// The phase the compositor was actually in.
        actual: CompositorPhase,
    },
    /// Allocating or freeing a command buffer failed.
    CommandAllocation(vk::Result),
    /// Resetting the frame command pool failed.
    CommandPoolReset(vk::Result),
    /// Recording a command buffer failed.
    CommandRecord(vk::Result),
    /// Creating a binary semaphore failed.
    SemaphoreCreate(vk::Result),
    /// Importing a `sync_file` fd into a wait semaphore failed.
    SemaphoreImport(vk::Result),
    /// Exporting a signalled semaphore as a `sync_file` fd failed.
    SemaphoreExport(vk::Result),
    /// Creating a fence failed.
    FenceCreate(vk::Result),
    /// Waiting for a fence failed (for example device loss).
    FenceWait(vk::Result),
    /// Queue submission failed.
    Submit(vk::Result),
    /// Reading or validating an embedded SPIR-V module failed.
    ShaderLoad(String),
    /// `vkCreateShaderModule` failed.
    ShaderModuleCreate(vk::Result),
    /// `vkCreateDescriptorSetLayout` failed.
    DescriptorSetLayoutCreate(vk::Result),
    /// `vkCreateDescriptorPool` failed.
    DescriptorPoolCreate(vk::Result),
    /// Allocating the mix descriptor set failed.
    DescriptorSetAllocate(vk::Result),
    /// `vkCreateSampler` failed.
    SamplerCreate(vk::Result),
    /// `vkCreatePipelineLayout` failed.
    PipelineLayoutCreate(vk::Result),
    /// `vkCreateGraphicsPipelines` failed.
    PipelineCreate(vk::Result),
    /// `vkCreateImage` failed for an offscreen target.
    ImageCreate(vk::Result),
    /// `vkCreateImageView` failed for an offscreen target.
    ImageViewCreate(vk::Result),
    /// No memory type both allowed by the offscreen image and
    /// device-local could be found.
    NoSuitableMemoryType,
    /// `vkAllocateMemory` failed.
    MemoryAllocate(vk::Result),
    /// `vkBindImageMemory` failed.
    MemoryBind(vk::Result),
    /// More pipes were registered than [`MAX_PIPES`] allows.
    TooManyPipes {
        /// The pipe count the compositor accepts.
        max: usize,
    },
    /// A pipe could not be started or recorded.
    Pipe {
        /// Zero-based position of the failing pipe in the queue.
        index: usize,
        /// The failure reported by the pipe or the driver.
        source: Box<CompositorError>,
    },
    /// An internal invariant was violated; this indicates a bug in this
    /// module.
    Internal(&'static str),
}

impl fmt::Display for CompositorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidState { operation, state } => {
                write!(f, "{operation} is not allowed while the compositor is {state}")
            }
            Self::FramePhase {
                operation,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{operation} requires the frame to be {expected} but it is {actual}"
                )
            }
            Self::CommandAllocation(err) => write!(f, "command buffer allocation failed: {err:?}"),
            Self::CommandPoolReset(err) => write!(f, "command pool reset failed: {err:?}"),
            Self::CommandRecord(err) => write!(f, "command buffer recording failed: {err:?}"),
            Self::SemaphoreCreate(err) => write!(f, "semaphore creation failed: {err:?}"),
            Self::SemaphoreImport(err) => write!(f, "sync_file semaphore import failed: {err:?}"),
            Self::SemaphoreExport(err) => write!(f, "sync_file semaphore export failed: {err:?}"),
            Self::FenceCreate(err) => write!(f, "fence creation failed: {err:?}"),
            Self::FenceWait(err) => write!(f, "waiting for a fence failed: {err:?}"),
            Self::Submit(err) => write!(f, "queue submission failed: {err:?}"),
            Self::ShaderLoad(err) => write!(f, "failed to load embedded SPIR-V: {err}"),
            Self::ShaderModuleCreate(err) => write!(f, "shader module creation failed: {err:?}"),
            Self::DescriptorSetLayoutCreate(err) => {
                write!(f, "descriptor set layout creation failed: {err:?}")
            }
            Self::DescriptorPoolCreate(err) => {
                write!(f, "descriptor pool creation failed: {err:?}")
            }
            Self::DescriptorSetAllocate(err) => {
                write!(f, "descriptor set allocation failed: {err:?}")
            }
            Self::SamplerCreate(err) => write!(f, "sampler creation failed: {err:?}"),
            Self::PipelineLayoutCreate(err) => write!(f, "pipeline layout creation failed: {err:?}"),
            Self::PipelineCreate(err) => write!(f, "graphics pipeline creation failed: {err:?}"),
            Self::ImageCreate(err) => write!(f, "offscreen image creation failed: {err:?}"),
            Self::ImageViewCreate(err) => write!(f, "offscreen image view creation failed: {err:?}"),
            Self::NoSuitableMemoryType => f.write_str("no suitable memory type for an offscreen image"),
            Self::MemoryAllocate(err) => write!(f, "offscreen memory allocation failed: {err:?}"),
            Self::MemoryBind(err) => write!(f, "binding offscreen memory failed: {err:?}"),
            Self::TooManyPipes { max } => write!(f, "at most {max} pipes may be registered"),
            Self::Pipe { index, source } => write!(f, "pipe {index} failed: {source}"),
            Self::Internal(msg) => write!(f, "internal compositor error: {msg}"),
        }
    }
}

impl core::error::Error for CompositorError {}

/// Error reported by the non-blocking [`CompositorComm`] senders.
///
/// The rejected value is handed back so the caller can retry it later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommError<T> {
    /// The channel is at capacity; the value was not enqueued.
    Full(T),
    /// The channel is closed; the value was not enqueued.
    Closed(T),
}

impl<T> fmt::Display for CommError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full(_) => f.write_str("the channel is full"),
            Self::Closed(_) => f.write_str("the channel is closed"),
        }
    }
}

impl<T: core::fmt::Debug> core::error::Error for CommError<T> {}

/// Decodes a little-endian SPIR-V blob into shader words.
///
/// This is the `no_std` replacement for `ash::util::read_spv` (which is
/// only available with ash's `std` feature): it validates the file length
/// and the module magic, then widens each 4-byte group into a `u32`.
///
/// # Errors
///
/// * [`CompositorError::ShaderLoad`] — the byte length is not a multiple
///   of 4, or the SPIR-V magic number does not match.
pub fn load_spir_v(bytes: &[u8]) -> Result<Vec<u32>, CompositorError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(CompositorError::ShaderLoad(format!(
            "SPIR-V byte length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let words: Vec<u32> = chunks
        .map(|group| u32::from_le_bytes([group[0], group[1], group[2], group[3]]))
        .collect();
    match words.first() {
        Some(&magic) if magic == SPIRV_MAGIC => Ok(words),
        Some(&magic) => Err(CompositorError::ShaderLoad(format!(
            "bad SPIR-V magic {magic:#010x}"
        ))),
        None => Err(CompositorError::ShaderLoad(String::from(
            "the SPIR-V module is empty",
        ))),
    }
}

/// Builds a shader module from an embedded SPIR-V blob.
///
/// # Errors
///
/// * [`CompositorError::ShaderLoad`] — the embedded module is malformed.
/// * [`CompositorError::ShaderModuleCreate`] — `vkCreateShaderModule`
///   failed.
fn shader_module(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule, CompositorError> {
    let words = load_spir_v(bytes)?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    // SAFETY: `words` is a validated, 4-byte-aligned SPIR-V module that
    // outlives the call.
    unsafe { device.create_shader_module(&info, None) }.map_err(CompositorError::ShaderModuleCreate)
}

/// Push constant payload of the mix pass.
///
/// The layout mirrors `MixPush` in `shaders/mix.frag`: `count` at offset
/// 0, then two `vec4` weight groups at offsets 16 and 32. The explicit
/// padding keeps the block identical under the std140, std430 and scalar
/// layout rules, so the byte range pushed by the compositor matches what
/// the shader reads.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MixPush {
    /// Number of framebuffers actually mixed (0 clears the target).
    pub count: u32,
    /// Explicit padding up to offset 16 (matches `layout(offset = 16)`).
    pub pad: [u32; 3],
    /// Mix weights of framebuffers 0..4.
    pub weights0: [f32; 4],
    /// Mix weights of framebuffers 4..8.
    pub weights1: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<MixPush>() == 48);
const _: () = assert!(core::mem::offset_of!(MixPush, weights0) == 16);
const _: () = assert!(core::mem::offset_of!(MixPush, weights1) == 32);

/// View of one offscreen color target exposed to the mix pass.
///
/// A renderer implementing [`PipeSupplyTraits`] hands these to the
/// compositor every frame; the compositor owns neither the image nor the
/// view, it only binds them for sampling.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct OffscreenFramebuffer {
    /// Color image backing the target.
    pub image: vk::Image,
    /// Image view bound as a sampled image by the mix pass.
    pub view: vk::ImageView,
    /// Format of `image` (must match the mix pipeline's attachment
    /// format for the render pass that writes it).
    pub format: vk::Format,
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
}

impl OffscreenFramebuffer {
    /// Placeholder used for unused slots of [`PipeCtx`]'s snapshot.
    pub const EMPTY: Self = Self {
        image: vk::Image::null(),
        view: vk::ImageView::null(),
        format: vk::Format::UNDEFINED,
        width: 0,
        height: 0,
    };

    /// Whether the view refers to a real, non-empty image.
    #[inline]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.view != vk::ImageView::null()
    }
}

/// Supplies the compositor with the offscreen targets a pipe draws into.
///
/// Implement this on the renderer registered with
/// [`Compositor::add_pipe`]; the compositor collects the targets once per
/// frame (in queue order) and folds them into the presentation target
/// with the mix pass.
pub trait PipeSupplyTraits {
    /// Number of framebuffers this pipe contributes to the frame.
    fn offscreen_framebuffer_count(&self) -> usize;

    /// The framebuffer at `index`, or `None` when the pipe contributed
    /// fewer targets than advertised.
    #[must_use]
    fn offscreen_framebuffer(&self, index: usize) -> Option<OffscreenFramebuffer>;

    /// Mix weight of framebuffer `index` in `[0.0, 1.0]`.
    ///
    /// The compositor sanitizes the value with
    /// [`sanitize_weight`] before pushing it; framebuffer 0 of the whole
    /// frame is the base layer the shader starts from, so its weight is
    /// ignored by the mix itself.
    #[must_use]
    fn mix_weight(&self, index: usize) -> f32 {
        let _ = index;
        1.0
    }
}

/// Clamps a mix weight to a finite value in `[0.0, 1.0]`.
///
/// Weights come from pipe implementations and travel to the GPU as push
/// constants, so NaN, infinities and out-of-range values are rejected
/// here rather than in the shader.
#[inline]
fn sanitize_weight(weight: f32) -> f32 {
    if weight.is_finite() {
        weight.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// What a pipe asks the compositor to do once its future resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipeOutcome {
    /// Stay registered; the compositor requests a new future for the
    /// next frame.
    Keep,
    /// Drop the pipe after the frame completes. Its renderer is released
    /// at the next [`Compositor::begin_frame`].
    Remove,
}

/// Result of one poll of a pipe future.
pub type PipeResult = Result<PipeOutcome, CompositorError>;

/// Async body of one pipe for the lifetime of a single frame.
///
/// The lifetime is the one borrowed from the pipe's renderer and
/// [`PipeCtx`]; it is erased once in [`trampoline`] (see the safety
/// comment there), so the compositor can store the future next to — and
/// drop it before — the data it borrows.
pub type PipeFuture<'a> = Pin<Box<dyn Future<Output = PipeResult> + 'a>>;

/// Type-erased entry point a [`Pipe`] dispatches to.
pub type PipeEntry = fn(PipeEnv) -> PipeFuture<'static>;

/// Raw arguments handed to a [`PipeEntry`].
///
/// The pointers are the pipe's renderer (of the concrete `PipeSource`
/// type) and the frame's [`PipeCtx`]; both stay valid for as long as the
/// returned future exists.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PipeEnv {
    /// `*mut <T as PipeSource>` for the renderer `T` of this pipe.
    pub renderer: *mut (),
    /// Frame context owned by the compositor for this pipe.
    pub ctx: *mut PipeCtx,
}

/// Per-frame recording state handed to one pipe.
///
/// A pipe draws into its own offscreen targets: [`PipeCtx::begin_render`]
/// opens a dynamic-rendering scope on one of them and
/// [`PipeCtx::end_render`] closes it. Layout transitions are recorded by
/// the compositor before any pipe runs, so a pipe must never transition
/// the image it renders into (see the module documentation).
///
/// The struct is lifetime-free: the compositor builds it from a
/// [`PipelineContext`] that outlives every pipe and hands a pointer to
/// it through [`PipeEnv`], which keeps the self-referential pipe futures
/// borrow-checkable at the cost of one documented lifetime erasure.
pub struct PipeCtx {
    /// Logical device of the owning pipeline; valid for `'p`.
    device: *const ash::Device,
    /// Primary command buffer being recorded for this pipe this frame.
    command_buffer: vk::CommandBuffer,
    /// Presentation target width in pixels (pipe viewport hint).
    width: u32,
    /// Presentation target height in pixels (pipe viewport hint).
    height: u32,
    /// Format of the presentation target.
    format: vk::Format,
    /// Number of frames composited so far (0 for the first frame).
    frame_index: u64,
    /// Global index of this pipe's first framebuffer inside
    /// [`PipeCtx::framebuffers`].
    base: usize,
    /// Snapshot of the frame's framebuffers; slots at or after
    /// [`PipeCtx::framebuffer_count`] are [`OffscreenFramebuffer::EMPTY`].
    framebuffers: [OffscreenFramebuffer; MAX_MIX_TARGETS],
    /// Number of valid entries in [`PipeCtx::framebuffers`].
    framebuffer_count: usize,
}

impl PipeCtx {
    /// Builds the context for one pipe of one frame.
    #[allow(
        clippy::too_many_arguments,
        reason = "raw Vulkan handle plus the frame geometry this context snapshots"
    )]
    fn new(
        device: *const ash::Device,
        command_buffer: vk::CommandBuffer,
        width: u32,
        height: u32,
        format: vk::Format,
        frame_index: u64,
        base: usize,
        framebuffers: &[OffscreenFramebuffer],
    ) -> Self {
        let mut snapshot = [OffscreenFramebuffer::EMPTY; MAX_MIX_TARGETS];
        let count = framebuffers.len().min(MAX_MIX_TARGETS);
        snapshot[..count].copy_from_slice(&framebuffers[..count]);
        Self {
            device,
            command_buffer,
            width,
            height,
            format,
            frame_index,
            base,
            framebuffers: snapshot,
            framebuffer_count: count,
        }
    }

    /// The logical device owning this pipe's command buffer.
    ///
    /// # Panics
    ///
    /// Never panics: the compositor only builds contexts whose device
    /// pointer comes from a live [`PipelineContext`].
    #[inline]
    #[must_use]
    pub fn device(&self) -> &ash::Device {
        // SAFETY: the device pointer is copied once from the
        // `PipelineContext` that owns this compositor and is never
        // mutated; that context outlives every pipe (enforced by `'p`),
        // so the pointee stays valid while the context exists.
        unsafe { &*self.device }
    }

    /// The primary command buffer being recorded for this pipe.
    ///
    /// It is in the recording state between [`PipeCtx::begin_render`]
    /// and [`PipeCtx::end_render`] while the pipe's future runs.
    #[inline]
    #[must_use]
    pub const fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    /// Presentation target width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Presentation target height in pixels.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Vulkan format of the presentation target.
    #[inline]
    #[must_use]
    pub const fn color_format(&self) -> vk::Format {
        self.format
    }

    /// Number of frames composited before this one.
    #[inline]
    #[must_use]
    pub const fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Global index of this pipe's first framebuffer.
    #[inline]
    #[must_use]
    pub const fn base(&self) -> usize {
        self.base
    }

    /// Number of framebuffers collected for this frame.
    #[inline]
    #[must_use]
    pub const fn framebuffer_count(&self) -> usize {
        self.framebuffer_count
    }

    /// The pipe-local framebuffer at `local_index` (its global slot is
    /// [`PipeCtx::base`] + `local_index`).
    #[must_use]
    pub fn framebuffer(&self, local_index: usize) -> Option<OffscreenFramebuffer> {
        self.framebuffers
            .get(self.base + local_index)
            .copied()
    }

    /// Opens a dynamic-rendering scope on the pipe-local framebuffer at
    /// `local_index`.
    ///
    /// `clear` selects the attachment load operation: `Some(color)`
    /// clears the target first (always pass this for a pipe's first
    /// frame, when the image still holds `UNDEFINED` contents) and
    /// `None` loads the previous contents. The viewport and scissor are
    /// set to the framebuffer's extent.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Internal`] — no framebuffer was collected at
    ///   that index, so the pipe cannot render.
    ///
    /// # Safety of use
    ///
    /// Only call this while the pipe's command buffer is recording and
    /// after the compositor's layout pass ran for the frame (the
    /// framebuffer is then in `COLOR_ATTACHMENT_OPTIMAL`).
    pub fn begin_render(
        &mut self,
        local_index: usize,
        clear: Option<[f32; 4]>,
    ) -> Result<(), CompositorError> {
        let framebuffer = self
            .framebuffer(local_index)
            .filter(OffscreenFramebuffer::is_valid)
            .ok_or(CompositorError::Internal(
                "begin_render: no framebuffer collected at this index",
            ))?;
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(framebuffer.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(if clear.is_some() {
                vk::AttachmentLoadOp::CLEAR
            } else {
                vk::AttachmentLoadOp::LOAD
            })
            .store_op(vk::AttachmentStoreOp::STORE);
        let attachment = match clear {
            Some(color) => attachment.clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            }),
            None => attachment,
        };
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: framebuffer.width,
                height: framebuffer.height,
            },
        };
        let rendering_info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(core::slice::from_ref(&attachment));
        // SAFETY: the command buffer is recording (checked by the
        // caller's contract), the image view is compatible with the
        // `COLOR_ATTACHMENT_OPTIMAL` layout recorded by the layout pass,
        // and the render area fits the image; dynamic rendering needs no
        // render pass or framebuffer object.
        unsafe {
            self.device()
                .cmd_begin_rendering(self.command_buffer, &rendering_info)
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: framebuffer.width as f32,
            height: framebuffer.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        // SAFETY: the rendering scope opened above has not been ended and
        // the framebuffer extent is non-zero (validated by
        // `OffscreenFramebuffer::is_valid`).
        unsafe {
            self.device()
                .cmd_set_viewport(self.command_buffer, 0, core::slice::from_ref(&viewport));
            self.device()
                .cmd_set_scissor(self.command_buffer, 0, core::slice::from_ref(&area));
        }
        Ok(())
    }

    /// Closes the dynamic-rendering scope opened by
    /// [`PipeCtx::begin_render`].
    ///
    /// Recording a mismatched begin/end pair corrupts the command
    /// buffer, so pair every successful `begin_render` with exactly one
    /// call to this method — including on the pipe's error paths.
    pub fn end_render(&mut self) {
        // SAFETY: the command buffer is recording inside a rendering
        // scope opened by `begin_render` (the pipe's pairing contract).
        unsafe {
            self.device()
                .cmd_end_rendering(self.command_buffer)
        };
    }
}

/// One registered pipe: an async body plus the renderer it drives.
///
/// The layout is part of the module's contract — the entry pointer sits
/// at byte offset 0 and the erased renderer pointer at offset 8 on
/// 64-bit targets — and is asserted by the `const` checks below.
#[repr(C)]
pub struct Pipe {
    /// Type-erased entry point called once per frame.
    entry: PipeEntry,
    /// `*mut T` for the `PipeSource`/`PipeSupplyTraits` type `T` this
    /// pipe was created from.
    renderer: *mut (),
}

const _: () = assert!(core::mem::offset_of!(Pipe, entry) == 0);
const _: () = assert!(core::mem::offset_of!(Pipe, renderer) == core::mem::size_of::<PipeEntry>());
const _: () = assert!(core::mem::size_of::<Pipe>() == 2 * core::mem::size_of::<PipeEntry>());

impl Pipe {
    /// Builds a pipe for `renderer`, erasing its type behind
    /// [`PipeEntry`].
    ///
    /// The renderer must outlive the pipe; [`PipeHandoff`] upholds this
    /// by taking ownership of the renderer, which is why this
    /// constructor is not public.
    fn new<T: PipeSource>(renderer: &mut T) -> Self {
        Self {
            entry: trampoline::<T>,
            renderer: core::ptr::from_mut(renderer).cast::<()>(),
        }
    }

    /// The type-erased entry point of this pipe.
    #[inline]
    #[must_use]
    pub const fn entry(&self) -> PipeEntry {
        self.entry
    }

    /// The erased renderer pointer (only valid for the pipe's lifetime).
    #[inline]
    #[must_use]
    pub const fn renderer(&self) -> *mut () {
        self.renderer
    }
}

impl fmt::Debug for Pipe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipe")
            .field("entry", &self.entry)
            .field("renderer", &self.renderer)
            .finish()
    }
}

/// Per-frame async body of a registered pipe.
///
/// The compositor calls this once per frame with the pipe's renderer and
/// the frame's [`PipeCtx`]; the returned future records the pipe's
/// commands and may yield (its waker decides when it is polled again)
/// before returning a [`PipeOutcome`].
pub trait PipeSource {
    /// Builds this pipe's future for one frame.
    fn pipe_entry<'a>(renderer: &'a mut Self, ctx: &'a mut PipeCtx) -> PipeFuture<'a>
    where
        Self: Sized;
}

/// Erases a pipe's type and its future's lifetime for the compositor.
///
/// # Safety
///
/// `env` must come from [`Pipe::new`] for the concrete `T`, and both
/// pointees must outlive the returned future: the compositor stores the
/// renderer in a [`PipeHandoff`] and the context in the pipe's
/// `ActivePipe`, and it drops the future before either of them (the
/// renderer is only released at the next [`Compositor::begin_frame`]).
fn trampoline<T: PipeSource>(env: PipeEnv) -> PipeFuture<'static> {
    // SAFETY: see the function's safety contract above; the two pointers
    // are not aliased elsewhere while the future is alive because the
    // compositor keeps exactly one active future per pipe.
    let renderer = unsafe { &mut *env.renderer.cast::<T>() };
    // SAFETY: same contract; the context is exclusively owned by the
    // pipe's active slot for the whole frame.
    let ctx = unsafe { &mut *env.ctx };
    let future: PipeFuture<'_> = T::pipe_entry(renderer, ctx);
    // SAFETY: the future borrows only `renderer` and `ctx`, which the
    // compositor keeps alive until the future is dropped; erasing the
    // lifetime here moves that obligation from the compiler to the drop
    // order documented in the safety contract above.
    unsafe { core::mem::transmute(future) }
}

/// Collects one pipe's framebuffers into the frame's mix inputs.
///
/// Arguments: the erased renderer, the frame's framebuffer list and the
/// per-slot mix weights.
type CollectFn = fn(*const (), &mut Vec<OffscreenFramebuffer>, &mut [f32; MAX_MIX_TARGETS]);

/// Releases the renderer owned by a [`PipeHandoff`].
type DropFn = fn(*mut ());

/// A pipe queued with the compositor together with everything the frame
/// loop needs from it.
///
/// The handoff owns its renderer: dropping it releases the renderer
/// through the type-erased drop function, so a pipe registered with
/// [`Compositor::add_pipe`] keeps working after the caller's value would
/// have gone out of scope.
#[derive(Debug)]
pub struct PipeHandoff {
    /// Entry point and erased renderer of the pipe.
    pipe: Pipe,
    /// Per-frame framebuffer/weight collector of the renderer.
    collect: CollectFn,
    /// Releases the renderer stored in `pipe`.
    drop_fn: DropFn,
}

impl PipeHandoff {
    /// Builds a handoff that takes ownership of `renderer`.
    #[must_use]
    pub fn from_renderer<T: PipeSource + PipeSupplyTraits>(renderer: T) -> Self {
        // The pipe records the address of the renderer, so the value
        // must live on the heap for as long as the handoff owns it;
        // `drop_trampoline` releases exactly this allocation.
        let mut boxed = Box::new(renderer);
        let pipe = Pipe::new(&mut *boxed);
        // Taking the raw pointer transfers ownership out of the `Box`,
        // so the local scope does not free what the pipe now points to.
        let renderer = Box::into_raw(boxed);
        debug_assert_eq!(renderer.cast::<()>(), pipe.renderer());
        Self {
            pipe,
            collect: collect_trampoline::<T>,
            drop_fn: drop_trampoline::<T>,
        }
    }

    /// The pipe registered with the compositor.
    #[inline]
    #[must_use]
    pub const fn pipe(&self) -> &Pipe {
        &self.pipe
    }

    /// Collects this pipe's framebuffers and weights for one frame.
    fn collect(&self, framebuffers: &mut Vec<OffscreenFramebuffer>, weights: &mut [f32; MAX_MIX_TARGETS]) {
        (self.collect)(self.pipe.renderer, framebuffers, weights);
    }
}

impl Drop for PipeHandoff {
    fn drop(&mut self) {
        (self.drop_fn)(self.pipe.renderer);
    }
}

/// Body of [`PipeHandoff::collect`] for the concrete renderer type.
fn collect_trampoline<T: PipeSupplyTraits>(
    renderer: *const (),
    framebuffers: &mut Vec<OffscreenFramebuffer>,
    weights: &mut [f32; MAX_MIX_TARGETS],
) {
    // SAFETY: the pointer is the renderer `PipeHandoff::from_renderer`
    // boxed for `T`; the handoff owns it and outlives the synchronous
    // collection below, which takes a shared borrow only.
    let renderer = unsafe { &*renderer.cast::<T>() };
    for index in 0..renderer.offscreen_framebuffer_count() {
        if framebuffers.len() >= MAX_MIX_TARGETS {
            break;
        }
        let Some(framebuffer) = renderer.offscreen_framebuffer(index) else {
            continue;
        };
        if !framebuffer.is_valid() {
            continue;
        }
        let slot = framebuffers.len();
        weights[slot] = sanitize_weight(renderer.mix_weight(index));
        framebuffers.push(framebuffer);
    }
}

/// Body of [`PipeHandoff`]'s `Drop` for the concrete renderer type.
fn drop_trampoline<T>(renderer: *mut ()) {
    // SAFETY: the pointer is the `Box<T>` allocation that
    // `PipeHandoff::from_renderer` moved into the handoff, and this
    // function runs exactly once — from `PipeHandoff::drop`.
    drop(unsafe { Box::from_raw(renderer.cast::<T>()) });
}

/// One queued pipe plus its pending-removal flag.
struct QueueEntry {
    /// Pipe and owned renderer.
    handoff: PipeHandoff,
    /// Set by [`PipeQueue::mark_removed`]; the entry is dropped by
    /// [`PipeQueue::drain_removed`] once the GPU finished the frame that
    /// still referenced it.
    remove: bool,
}

/// Ordered registry of pipes: queue order is mix order.
///
/// The queue only changes between frames (registration and removal are
/// rejected while a frame is in progress), which is what lets the frame
/// loop index it positionally when it wires up the semaphore chain.
#[derive(Default)]
pub struct PipeQueue {
    /// Queued pipes in mix order.
    entries: Vec<QueueEntry>,
}

impl PipeQueue {
    /// Creates an empty queue.
    #[must_use]
    pub const fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Appends `handoff` and returns its index.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::TooManyPipes`] — the queue already holds
    ///   [`MAX_PIPES`] pipes; the handoff is returned untouched inside
    ///   the error path by the caller.
    pub fn push(&mut self, handoff: PipeHandoff) -> Result<usize, CompositorError> {
        if self.entries.len() >= MAX_PIPES {
            return Err(CompositorError::TooManyPipes { max: MAX_PIPES });
        }
        self.entries.push(QueueEntry {
            handoff,
            remove: false,
        });
        Ok(self.entries.len() - 1)
    }

    /// Number of queued pipes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no pipe is queued.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The handoff queued at `index`.
    #[must_use]
    pub fn handoff(&self, index: usize) -> Option<&PipeHandoff> {
        self.entries
            .get(index)
            .map(|entry| &entry.handoff)
    }

    /// The pipe queued at `index`.
    #[must_use]
    pub fn pipe(&self, index: usize) -> Option<&Pipe> {
        self.handoff(index).map(PipeHandoff::pipe)
    }

    /// Marks the pipe at `index` for removal at the next frame boundary.
    ///
    /// Returns `false` when no such pipe exists or it was already
    /// marked.
    fn mark_removed(&mut self, index: usize) -> bool {
        let Some(entry) = self.entries.get_mut(index) else {
            return false;
        };
        let fresh = !entry.remove;
        entry.remove = true;
        fresh
    }

    /// Whether any pipe is waiting to be removed.
    fn has_removed(&self) -> bool {
        self.entries.iter().any(|entry| entry.remove)
    }

    /// Drops every pipe marked for removal and returns them.
    ///
    /// Only called after the frame that still referenced them retired
    /// (its fence was waited on), so releasing the renderers is safe.
    fn drain_removed(&mut self) -> Vec<PipeHandoff> {
        let mut retired = Vec::new();
        let mut index = 0;
        while index < self.entries.len() {
            if self.entries[index].remove {
                let entry = self.entries.remove(index);
                retired.push(entry.handoff);
            } else {
                index += 1;
            }
        }
        retired
    }

    /// Collects every pipe's framebuffers into `framebuffers` and its
    /// mix weights into `weights`, returning each pipe's first slot.
    ///
    /// The returned array is indexed like the queue; slots of unused
    /// entries hold the current length of `framebuffers`.
    fn collect_framebuffers(
        &self,
        framebuffers: &mut Vec<OffscreenFramebuffer>,
        weights: &mut [f32; MAX_MIX_TARGETS],
    ) -> [usize; MAX_PIPES] {
        let mut bases = [0; MAX_PIPES];
        for (index, entry) in self.entries.iter().enumerate() {
            if index >= MAX_PIPES {
                break;
            }
            bases[index] = framebuffers.len();
            entry.handoff.collect(framebuffers, weights);
        }
        bases
    }
}

/// Message exchanged between the compositor and the presentation side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositorMessage {
    /// The sender's offscreen framebuffers changed (resize, recreate);
    /// the compositor rebuilds its snapshot at the next
    /// [`Compositor::begin_frame`].
    FramebuffersChanged,
    /// Override the mix weight of framebuffer `slot`.
    MixWeightChanged {
        /// Framebuffer slot, in `[0, MAX_MIX_TARGETS)`.
        slot: u8,
        /// Requested weight; sanitized with [`sanitize_weight`].
        weight: f32,
    },
    /// Pause or resume pipe execution. The mix pass keeps running while
    /// invisible, so the presentation target keeps its last content.
    VisibilityChanged {
        /// `true` to drive the pipes again.
        visible: bool,
    },
    /// A frame finished and its `sync_file` was exported.
    FrameComposited {
        /// Frame index of the completed frame.
        frame_index: u64,
    },
    /// Application-defined notification forwarded unchanged.
    Custom(u32),
}

impl fmt::Display for CompositorMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FramebuffersChanged => f.write_str("framebuffers changed"),
            Self::MixWeightChanged { slot, weight } => write!(f, "mix weight {slot} = {weight}"),
            Self::VisibilityChanged { visible } => write!(f, "visible = {visible}"),
            Self::FrameComposited { frame_index } => {
                write!(f, "frame {frame_index} composited")
            }
            Self::Custom(id) => write!(f, "custom message {id}"),
        }
    }
}

/// One end of the four channels connecting both sides of a frame.
///
/// Channels are bounded (see [`CHANNEL_CAPACITY`]) and their senders
/// never block: a full channel reports [`CommError::Full`] so the caller
/// can retry, which keeps the compositor's hot paths allocation-free.
struct Channel<T> {
    /// Maximum number of queued values.
    capacity: usize,
    /// Queue plus the parked task wakers.
    state: Mutex<ChannelState<T>>,
}

/// Contents of a [`Channel`] behind its mutex.
struct ChannelState<T> {
    /// Values waiting for the receiver.
    queue: VecDeque<T>,
    /// Set once the hub is closed; further sends fail.
    closed: bool,
    /// Receiver task to wake when a value arrives.
    recv_waker: Option<Waker>,
    /// Sender task to wake when capacity frees up.
    send_waker: Option<Waker>,
}

impl<T> Channel<T> {
    /// Creates an empty channel with the given capacity.
    const fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(ChannelState {
                queue: VecDeque::new(),
                closed: false,
                recv_waker: None,
                send_waker: None,
            }),
        }
    }

    /// Enqueues `value` if capacity allows, waking the receiver.
    fn try_send(&self, value: T) -> Result<(), CommError<T>> {
        let receiver = {
            let mut state = self.state.lock();
            if state.closed {
                return Err(CommError::Closed(value));
            }
            if state.queue.len() >= self.capacity {
                return Err(CommError::Full(value));
            }
            state.queue.push_back(value);
            state.recv_waker.take()
        };
        // Woken outside the lock: `spin::Mutex` is not reentrant, so a
        // waker that polls this channel again must not find it locked.
        if let Some(waker) = receiver {
            waker.wake();
        }
        Ok(())
    }

    /// Dequeues the oldest value, waking a parked sender.
    fn try_recv(&self) -> Option<T> {
        let (value, sender) = {
            let mut state = self.state.lock();
            let value = state.queue.pop_front();
            let sender = if value.is_some() {
                state.send_waker.take()
            } else {
                None
            };
            (value, sender)
        };
        if let Some(waker) = sender {
            waker.wake();
        }
        value
    }

    /// Parks the receiver until a value arrives or the hub closes.
    fn poll_recv(&self, cx: &mut Context<'_>) -> Poll<Option<T>> {
        let (outcome, sender) = {
            let mut state = self.state.lock();
            if let Some(value) = state.queue.pop_front() {
                (Poll::Ready(Some(value)), state.send_waker.take())
            } else if state.closed {
                (Poll::Ready(None), None)
            } else {
                state.recv_waker = Some(cx.waker().clone());
                (Poll::Pending, None)
            }
        };
        if let Some(waker) = sender {
            waker.wake();
        }
        outcome
    }

    /// Parks the sender while `value` is held for the next attempt.
    fn poll_send(&self, cx: &mut Context<'_>, value: &mut Option<T>) -> Poll<Result<(), CommError<T>>> {
        let (outcome, receiver) = {
            let mut state = self.state.lock();
            if state.closed {
                let result = match value.take() {
                    Some(held) => Err(CommError::Closed(held)),
                    None => Ok(()),
                };
                (Poll::Ready(result), None)
            } else if state.queue.len() < self.capacity {
                if let Some(held) = value.take() {
                    state.queue.push_back(held);
                }
                (Poll::Ready(Ok(())), state.recv_waker.take())
            } else {
                state.send_waker = Some(cx.waker().clone());
                (Poll::Pending, None)
            }
        };
        if let Some(waker) = receiver {
            waker.wake();
        }
        outcome
    }

    /// Closes the channel: receivers drain what is queued, senders get
    /// [`CommError::Closed`] and parked tasks are woken.
    fn close(&self) {
        let (receiver, sender) = {
            let mut state = self.state.lock();
            if state.closed {
                return;
            }
            state.closed = true;
            (state.recv_waker.take(), state.send_waker.take())
        };
        if let Some(waker) = receiver {
            waker.wake();
        }
        if let Some(waker) = sender {
            waker.wake();
        }
    }
}

/// The four channels shared by both endpoints.
///
/// `to_compositor` carries messages and pipes destined for the frame
/// loop, `to_presentation` the ones coming back from it.
struct CommHub {
    /// Messages the presentation side sends to the compositor.
    to_compositor_messages: Channel<CompositorMessage>,
    /// Pipes the presentation side registers with the compositor.
    to_compositor_pipes: Channel<PipeHandoff>,
    /// Messages the compositor sends to the presentation side.
    to_presentation_messages: Channel<CompositorMessage>,
    /// Pipes retired by the compositor, handed back for release.
    to_presentation_pipes: Channel<PipeHandoff>,
}

impl CommHub {
    /// Creates an open hub with [`CHANNEL_CAPACITY`] deep channels.
    const fn new() -> Self {
        Self {
            to_compositor_messages: Channel::new(CHANNEL_CAPACITY),
            to_compositor_pipes: Channel::new(CHANNEL_CAPACITY),
            to_presentation_messages: Channel::new(CHANNEL_CAPACITY),
            to_presentation_pipes: Channel::new(CHANNEL_CAPACITY),
        }
    }

    /// Closes every channel.
    fn close(&self) {
        self.to_compositor_messages.close();
        self.to_compositor_pipes.close();
        self.to_presentation_messages.close();
        self.to_presentation_pipes.close();
    }
}

/// Blocking-free messaging between the two ends of a frame.
///
/// Every method has a non-blocking form (`send_*`/`poll_recv_*`) and an
/// async form (`send_*_async`/`recv_*`); nothing ever blocks a thread.
/// Completed futures stay completed when polled again, so executors may
/// poll them after completion without side effects.
///
/// # Not `Send`
///
/// The endpoints own queued [`PipeHandoff`]s — type-erased renderers
/// bound to one Vulkan device — so they deliberately do not implement
/// `Send`/`Sync`: moving an endpoint to another thread would move those
/// renderers with it.
pub trait CompositorComm {
    /// Enqueues `message` without waiting.
    fn send_message(&self, message: CompositorMessage) -> Result<(), CommError<CompositorMessage>>;

    /// Future form of [`CompositorComm::send_message`].
    fn send_message_async(&self, message: CompositorMessage) -> SendFuture<'_, CompositorMessage>;

    /// Polls for a message without parking the task.
    fn poll_recv_message(&self, cx: &mut Context<'_>) -> Poll<Option<CompositorMessage>>;

    /// Future form of [`CompositorComm::poll_recv_message`].
    fn recv_message(&self) -> RecvFuture<'_, CompositorMessage>;

    /// Takes the next message without parking the task.
    fn try_recv_message(&self) -> Option<CompositorMessage>;

    /// Enqueues a pipe to register without waiting.
    fn send_pipe(&self, pipe: PipeHandoff) -> Result<(), CommError<PipeHandoff>>;

    /// Future form of [`CompositorComm::send_pipe`].
    fn send_pipe_async(&self, pipe: PipeHandoff) -> SendFuture<'_, PipeHandoff>;

    /// Polls for a pipe without parking the task.
    fn poll_recv_pipe(&self, cx: &mut Context<'_>) -> Poll<Option<PipeHandoff>>;

    /// Future form of [`CompositorComm::poll_recv_pipe`].
    fn recv_pipe(&self) -> RecvFuture<'_, PipeHandoff>;

    /// Takes the next pipe without parking the task.
    fn try_recv_pipe(&self) -> Option<PipeHandoff>;
}

/// Future of a non-blocking send that waits for capacity.
///
/// Yields while the channel is full and stays [`Poll::Ready`] with
/// `Ok(())` once the value was accepted (or the future was polled after
/// completion).
pub struct SendFuture<'a, T> {
    /// Channel the value is pushed into.
    channel: &'a Channel<T>,
    /// Value still waiting for capacity; `None` once sent.
    message: Option<T>,
}

impl<'a, T: Unpin> Future for SendFuture<'a, T> {
    type Output = Result<(), CommError<T>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: `T: Unpin` makes `Self` `Unpin`, so the pinned struct
        // may be accessed through the pin.
        let this = unsafe { self.get_unchecked_mut() };
        if this.message.is_none() {
            return Poll::Ready(Ok(()));
        }
        this.channel.poll_send(cx, &mut this.message)
    }
}

/// Future of a receive that waits for the next value.
///
/// Yields while the channel is empty and stays [`Poll::Ready`] with
/// `None` once the hub was closed and drained.
pub struct RecvFuture<'a, T> {
    /// Channel the value is taken from.
    channel: &'a Channel<T>,
    /// Set once the future resolved; further polls keep the result.
    done: bool,
}

impl<T> Future for RecvFuture<'_, T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: `Self` only holds a reference and a `bool`, both
        // `Unpin`, so the pin may be unwrapped safely.
        let this = unsafe { self.get_unchecked_mut() };
        if this.done {
            return Poll::Ready(None);
        }
        match this.channel.poll_recv(cx) {
            Poll::Ready(value) => {
                this.done = true;
                Poll::Ready(value)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Endpoint kept by the compositor: receives from the presentation side
/// and sends back to it.
///
/// Created by [`comm_pair`] or fetched from a running compositor with
/// [`Compositor::comm`].
#[derive(Clone)]
pub struct CompositorEndpoint {
    /// Shared channels.
    hub: Rc<CommHub>,
}

impl CompositorEndpoint {
    /// The messages addressed to the compositor.
    fn inbound_messages(&self) -> &Channel<CompositorMessage> {
        &self.hub.to_compositor_messages
    }

    /// The pipes addressed to the compositor.
    fn inbound_pipes(&self) -> &Channel<PipeHandoff> {
        &self.hub.to_compositor_pipes
    }

    /// The messages addressed to the presentation side.
    fn outbound_messages(&self) -> &Channel<CompositorMessage> {
        &self.hub.to_presentation_messages
    }

    /// The pipes addressed to the presentation side.
    fn outbound_pipes(&self) -> &Channel<PipeHandoff> {
        &self.hub.to_presentation_pipes
    }

    /// Closes every channel of the pair.
    fn close(&self) {
        self.hub.close();
    }
}

impl CompositorComm for CompositorEndpoint {
    fn send_message(&self, message: CompositorMessage) -> Result<(), CommError<CompositorMessage>> {
        self.outbound_messages().try_send(message)
    }

    fn send_message_async(&self, message: CompositorMessage) -> SendFuture<'_, CompositorMessage> {
        SendFuture {
            channel: self.outbound_messages(),
            message: Some(message),
        }
    }

    fn poll_recv_message(&self, cx: &mut Context<'_>) -> Poll<Option<CompositorMessage>> {
        self.inbound_messages().poll_recv(cx)
    }

    fn recv_message(&self) -> RecvFuture<'_, CompositorMessage> {
        RecvFuture {
            channel: self.inbound_messages(),
            done: false,
        }
    }

    fn try_recv_message(&self) -> Option<CompositorMessage> {
        self.inbound_messages().try_recv()
    }

    fn send_pipe(&self, pipe: PipeHandoff) -> Result<(), CommError<PipeHandoff>> {
        self.outbound_pipes().try_send(pipe)
    }

    fn send_pipe_async(&self, pipe: PipeHandoff) -> SendFuture<'_, PipeHandoff> {
        SendFuture {
            channel: self.outbound_pipes(),
            message: Some(pipe),
        }
    }

    fn poll_recv_pipe(&self, cx: &mut Context<'_>) -> Poll<Option<PipeHandoff>> {
        self.inbound_pipes().poll_recv(cx)
    }

    fn recv_pipe(&self) -> RecvFuture<'_, PipeHandoff> {
        RecvFuture {
            channel: self.inbound_pipes(),
            done: false,
        }
    }

    fn try_recv_pipe(&self) -> Option<PipeHandoff> {
        self.inbound_pipes().try_recv()
    }
}

/// Endpoint kept by the presentation side: mirrors
/// [`CompositorEndpoint`] in the opposite direction.
#[derive(Clone)]
pub struct RendererEndpoint {
    /// Shared channels.
    hub: Rc<CommHub>,
}

impl RendererEndpoint {
    /// The messages addressed to the compositor.
    fn outbound_messages(&self) -> &Channel<CompositorMessage> {
        &self.hub.to_compositor_messages
    }

    /// The pipes addressed to the compositor.
    fn outbound_pipes(&self) -> &Channel<PipeHandoff> {
        &self.hub.to_compositor_pipes
    }

    /// The messages addressed to the presentation side.
    fn inbound_messages(&self) -> &Channel<CompositorMessage> {
        &self.hub.to_presentation_messages
    }

    /// The pipes addressed to the presentation side.
    fn inbound_pipes(&self) -> &Channel<PipeHandoff> {
        &self.hub.to_presentation_pipes
    }
}

impl CompositorComm for RendererEndpoint {
    fn send_message(&self, message: CompositorMessage) -> Result<(), CommError<CompositorMessage>> {
        self.outbound_messages().try_send(message)
    }

    fn send_message_async(&self, message: CompositorMessage) -> SendFuture<'_, CompositorMessage> {
        SendFuture {
            channel: self.outbound_messages(),
            message: Some(message),
        }
    }

    fn poll_recv_message(&self, cx: &mut Context<'_>) -> Poll<Option<CompositorMessage>> {
        self.inbound_messages().poll_recv(cx)
    }

    fn recv_message(&self) -> RecvFuture<'_, CompositorMessage> {
        RecvFuture {
            channel: self.inbound_messages(),
            done: false,
        }
    }

    fn try_recv_message(&self) -> Option<CompositorMessage> {
        self.inbound_messages().try_recv()
    }

    fn send_pipe(&self, pipe: PipeHandoff) -> Result<(), CommError<PipeHandoff>> {
        self.outbound_pipes().try_send(pipe)
    }

    fn send_pipe_async(&self, pipe: PipeHandoff) -> SendFuture<'_, PipeHandoff> {
        SendFuture {
            channel: self.outbound_pipes(),
            message: Some(pipe),
        }
    }

    fn poll_recv_pipe(&self, cx: &mut Context<'_>) -> Poll<Option<PipeHandoff>> {
        self.inbound_pipes().poll_recv(cx)
    }

    fn recv_pipe(&self) -> RecvFuture<'_, PipeHandoff> {
        RecvFuture {
            channel: self.inbound_pipes(),
            done: false,
        }
    }

    fn try_recv_pipe(&self) -> Option<PipeHandoff> {
        self.inbound_pipes().try_recv()
    }
}

/// Creates a connected pair of endpoints with fresh channels.
#[must_use]
pub fn comm_pair() -> (CompositorEndpoint, RendererEndpoint) {
    let hub = Rc::new(CommHub::new());
    (
        CompositorEndpoint { hub: Rc::clone(&hub) },
        RendererEndpoint { hub },
    )
}

/// Creates the color image of an [`OffscreenTarget`].
fn create_color_image(
    context: &PipelineContext,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<vk::Image, CompositorError> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: `info` describes a plain 2D color target and the device
    // outlives the call.
    unsafe { context.device().create_image(&info, None) }.map_err(CompositorError::ImageCreate)
}

/// Allocates and binds device-local memory for `image`.
fn allocate_image_memory(
    context: &PipelineContext,
    image: vk::Image,
) -> Result<vk::DeviceMemory, CompositorError> {
    let device = context.device();
    // SAFETY: `image` was created by [`create_color_image`] on this
    // device, so the driver fills a valid requirements struct.
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    // SAFETY: the instance and physical device outlive the context.
    let properties = unsafe {
        context
            .instance()
            .get_physical_device_memory_properties(context.physical_device())
    };
    let memory_type = crate::ui_pipeline::find_memory_type(
        &properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or(CompositorError::NoSuitableMemoryType)?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    // SAFETY: the allocation matches an advertised memory type and size.
    let memory =
        unsafe { device.allocate_memory(&allocate_info, None) }.map_err(CompositorError::MemoryAllocate)?;
    // SAFETY: `memory` was just allocated for this device and `image` is
    // a valid unbound image.
    if let Err(err) = unsafe { device.bind_image_memory(image, memory, 0) } {
        // SAFETY: nothing else references the allocation yet.
        unsafe { device.free_memory(memory, None) };
        return Err(CompositorError::MemoryBind(err));
    }
    Ok(memory)
}

/// Creates the sampled view of `image`.
fn create_color_view(
    context: &PipelineContext,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, CompositorError> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: the image exists on this device and the subresource range
    // covers its only level and layer.
    unsafe { context.device().create_image_view(&info, None) }.map_err(CompositorError::ImageViewCreate)
}

/// A pipe-owned offscreen color target: image, memory and sampled view.
///
/// The target is created in `UNDEFINED` layout and is transitioned by
/// the compositor's layout pass; a renderer exposes it through
/// [`PipeSupplyTraits`]. The `'p` lifetime keeps it from outliving the
/// [`PipelineContext`] that owns the device it releases on `Drop`.
pub struct OffscreenTarget<'p> {
    /// Device that owns every handle below (`'p`).
    device: &'p ash::Device,
    /// Color image.
    image: vk::Image,
    /// Device memory bound to `image`.
    memory: vk::DeviceMemory,
    /// View bound as a sampled image by the mix pass.
    view: vk::ImageView,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Format of `image`.
    format: vk::Format,
}

impl<'p> OffscreenTarget<'p> {
    /// Creates an offscreen target of `width` × `height` in `format`.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::ImageCreate`], [`CompositorError::NoSuitableMemoryType`],
    ///   [`CompositorError::MemoryAllocate`], [`CompositorError::MemoryBind`],
    ///   [`CompositorError::ImageViewCreate`] — the target could not be
    ///   created; every partially created object is released first.
    pub fn new(
        context: &'p PipelineContext,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<Self, CompositorError> {
        let device = context.device();
        let image = create_color_image(context, width, height, format)?;
        let memory = match allocate_image_memory(context, image) {
            Ok(memory) => memory,
            Err(err) => {
                // SAFETY: `image` is bound to nothing yet.
                unsafe { device.destroy_image(image, None) };
                return Err(err);
            }
        };
        let view = match create_color_view(context, image, format) {
            Ok(view) => view,
            Err(err) => {
                // SAFETY: both objects were created above and nothing
                // else references them.
                unsafe {
                    device.free_memory(memory, None);
                    device.destroy_image(image, None);
                }
                return Err(err);
            }
        };
        Ok(Self {
            device,
            image,
            memory,
            view,
            width,
            height,
            format,
        })
    }

    /// The target as seen by the compositor.
    #[must_use]
    pub const fn framebuffer(&self) -> OffscreenFramebuffer {
        OffscreenFramebuffer {
            image: self.image,
            view: self.view,
            format: self.format,
            width: self.width,
            height: self.height,
        }
    }

    /// The color image.
    #[inline]
    #[must_use]
    pub const fn image(&self) -> vk::Image {
        self.image
    }

    /// Width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Format of the image.
    #[inline]
    #[must_use]
    pub const fn format(&self) -> vk::Format {
        self.format
    }
}

impl Drop for OffscreenTarget<'_> {
    fn drop(&mut self) {
        // SAFETY: the view, memory and image were created by
        // `OffscreenTarget::new` on this device, nothing else references
        // them, and `'p` keeps the device alive for the target's whole
        // lifetime.
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// GPU resources of the mix pass, created by [`Compositor::start`] and
/// released when dropped (which [`Compositor::stop`] triggers).
///
/// The descriptor set always covers [`MAX_MIX_TARGETS`] slots; the
/// compositor aliases unused slots to the first framebuffer so every
/// entry of `u_frames` stays statically bound (see `shaders/mix.frag`).
struct MixResources<'p> {
    /// Device that owns every handle below.
    device: &'p ash::Device,
    /// Set layout with one `u_frames[8]` binding at binding 0.
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Pool backing the single mix descriptor set.
    descriptor_pool: vk::DescriptorPool,
    /// The mix descriptor set, rewritten every frame.
    descriptor_set: vk::DescriptorSet,
    /// Linear sampler used for the framebuffer reads.
    sampler: vk::Sampler,
    /// Pipeline layout: the set layout plus the `MixPush` range.
    pipeline_layout: vk::PipelineLayout,
    /// Fullscreen-triangle graphics pipeline of the mix pass.
    pipeline: vk::Pipeline,
}

impl<'p> MixResources<'p> {
    /// All-null resources for `device`: [`Drop`] is a no-op on them.
    fn empty(device: &'p ash::Device) -> Self {
        Self {
            device,
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            sampler: vk::Sampler::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        }
    }

    /// Creates every resource of the mix pass.
    ///
    /// # Errors
    ///
    /// Propagates the [`CompositorError`] of the first step that fails;
    /// dropping the partially built resources releases everything that
    /// was already created.
    fn create(context: &'p PipelineContext) -> Result<Self, CompositorError> {
        let mut resources = Self::empty(context.device());
        resources.create_descriptors(context)?;
        resources.create_pipeline(context)?;
        Ok(resources)
    }

    /// Creates the descriptor set layout, pool, set and sampler.
    fn create_descriptors(&mut self, context: &PipelineContext) -> Result<(), CompositorError> {
        let device = context.device();
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_MIX_TARGETS as u32)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = [binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: the binding array is well-formed and outlives the call.
        self.descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
            .map_err(CompositorError::DescriptorSetLayoutCreate)?;

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_MIX_TARGETS as u32);
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: one set of `MAX_MIX_TARGETS` samplers matches the pool.
        self.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(CompositorError::DescriptorPoolCreate)?;

        let layouts = [self.descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and layout are valid and the pool has room.
        let sets = unsafe { device.allocate_descriptor_sets(&allocate_info) }
            .map_err(CompositorError::DescriptorSetAllocate)?;
        self.descriptor_set = sets
            .first()
            .copied()
            .ok_or(CompositorError::Internal(
                "the driver allocated no mix descriptor set",
            ))?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .min_lod(0.0)
            .max_lod(0.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false);
        // SAFETY: every field of `sampler_info` is within spec limits.
        self.sampler =
            unsafe { device.create_sampler(&sampler_info, None) }.map_err(CompositorError::SamplerCreate)?;
        Ok(())
    }

    /// Creates the pipeline layout and the mix graphics pipeline.
    fn create_pipeline(&mut self, context: &PipelineContext) -> Result<(), CompositorError> {
        let device = context.device();
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(core::mem::size_of::<MixPush>() as u32);
        let push_ranges = [push_range];
        let set_layouts = [self.descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: the layout references this pass's set layout and push
        // range, both well-formed and alive for the call.
        self.pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(CompositorError::PipelineLayoutCreate)?;

        let vertex_module = shader_module(device, MIX_VERT_SPIRV)?;
        let fragment_module = match shader_module(device, MIX_FRAG_SPIRV) {
            Ok(module) => module,
            Err(err) => {
                // SAFETY: the vertex module was just created.
                unsafe { device.destroy_shader_module(vertex_module, None) };
                return Err(err);
            }
        };
        let pipeline = create_mix_graphics_pipeline(
            device,
            self.pipeline_layout,
            vertex_module,
            fragment_module,
            PipelineContext::color_format(),
        );
        // SAFETY: both modules are no longer needed once the pipeline
        // was created (or failed to be).
        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(fragment_module, None);
        }
        self.pipeline = pipeline?;
        Ok(())
    }
}

impl Drop for MixResources<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created by `create` on this device,
        // is not referenced by any in-flight command buffer (the
        // compositor waits for the device before releasing these
        // resources), and the descriptor set goes away with its pool.
        // Null handles are never passed to the driver.
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.sampler != vk::Sampler::null() {
                self.device.destroy_sampler(self.sampler, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
        }
    }
}

/// Creates the fullscreen-triangle pipeline of the mix pass.
fn create_mix_graphics_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    color_format: vk::Format,
) -> Result<vk::Pipeline, CompositorError> {
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachment = vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A,
    };
    let blend_attachments = [blend_attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(core::slice::from_ref(&color_format));
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .push_next(&mut rendering_info);
    // SAFETY: every create-info is well-formed, the rendering info
    // declares the single color attachment format of dynamic rendering,
    // and the modules are alive for the call.
    let pipelines = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            core::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, err)| CompositorError::PipelineCreate(err))?;
    pipelines
        .first()
        .copied()
        .ok_or(CompositorError::Internal("the driver returned no mix pipeline"))
}

/// Builds the push-constant payload of the mix pass.
fn mix_push(count: usize, weights: &[f32; MAX_MIX_TARGETS]) -> MixPush {
    MixPush {
        count: u32::try_from(count).unwrap_or(0),
        pad: [0; 3],
        weights0: [weights[0], weights[1], weights[2], weights[3]],
        weights1: [weights[4], weights[5], weights[6], weights[7]],
    }
}

const _: () = assert!(MAX_MIX_TARGETS == 8);

/// Per-frame lifecycle of one pipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipeState {
    /// Its future has not resolved yet.
    Recording,
    /// Resolved and its command buffer was closed; waiting for its turn
    /// in the submission chain.
    Ready,
    /// On the queue; nothing more is polled for it this frame.
    Submitted,
}

/// One pipe's state for the frame in progress.
///
/// Fields drop in declaration order, so the future — which borrows the
/// context and the renderer — is released before anything it references.
struct ActivePipe {
    /// Future built on first poll; `None` after it resolved.
    future: Option<PipeFuture<'static>>,
    /// Frame context handed to the future.
    ctx: PipeCtx,
    /// Command buffer recorded by the future (the queue slot's buffer).
    command_buffer: vk::CommandBuffer,
    /// Progress through the frame.
    state: PipeState,
}

/// Everything [`Compositor::start`] allocates, released on drop.
///
/// `quiesce` must have run before this is dropped so no handle is still
/// referenced by the queue.
struct StartResources<'p> {
    /// Pipeline backing every handle below.
    context: &'p PipelineContext,
    /// One primary command buffer per queue slot.
    pipe_command_buffers: [vk::CommandBuffer; MAX_PIPES],
    /// Command buffer of the frame's layout pass.
    layout_command_buffer: vk::CommandBuffer,
    /// Command buffer of the frame's mix pass.
    mix_command_buffer: vk::CommandBuffer,
    /// Number of command buffers allocated (0 or `MAX_PIPES + 2`).
    command_buffer_count: usize,
    /// Semaphore chain: `chain_sems[0]` is signalled by the layout pass,
    /// pipe *k* waits on `chain_sems[k]` and signals `chain_sems[k + 1]`.
    chain_sems: [vk::Semaphore; MAX_PIPES + 1],
    /// Fence of the whole chain, signalled by the mix submission.
    frame_fence: vk::Fence,
    /// Descriptor set, sampler and pipeline of the mix pass.
    mix: MixResources<'p>,
}

impl<'p> StartResources<'p> {
    /// Allocates the command buffers, the semaphore chain, the frame
    /// fence and the mix resources.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::CommandAllocation`], [`CompositorError::SemaphoreCreate`],
    ///   [`CompositorError::FenceCreate`] — a resource could not be
    ///   created; everything allocated before the failure is released by
    ///   this struct's `Drop`.
    fn create(context: &'p PipelineContext) -> Result<Self, CompositorError> {
        let device = context.device();
        let mut resources = Self {
            context,
            pipe_command_buffers: [vk::CommandBuffer::null(); MAX_PIPES],
            layout_command_buffer: vk::CommandBuffer::null(),
            mix_command_buffer: vk::CommandBuffer::null(),
            command_buffer_count: 0,
            chain_sems: [vk::Semaphore::null(); MAX_PIPES + 1],
            frame_fence: vk::Fence::null(),
            mix: MixResources::create(context)?,
        };

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(context.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count((MAX_PIPES + 2) as u32);
        // SAFETY: the pool belongs to `context` and is alive for `'p`.
        let allocated = unsafe { device.allocate_command_buffers(&allocate_info) }
            .map_err(CompositorError::CommandAllocation)?;
        let expected = MAX_PIPES + 2;
        if allocated.len() != expected {
            if !allocated.is_empty() {
                // SAFETY: the buffers were just allocated from this pool.
                unsafe { device.free_command_buffers(context.command_pool(), &allocated) };
            }
            return Err(CompositorError::Internal(
                "the driver allocated an unexpected number of command buffers",
            ));
        }
        resources
            .pipe_command_buffers
            .copy_from_slice(&allocated[..MAX_PIPES]);
        resources.layout_command_buffer = allocated[MAX_PIPES];
        resources.mix_command_buffer = allocated[MAX_PIPES + 1];
        resources.command_buffer_count = expected;

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        for semaphore in resources.chain_sems.iter_mut() {
            // SAFETY: a plain, well-formed create info.
            *semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }
                .map_err(CompositorError::SemaphoreCreate)?;
        }

        // SAFETY: a plain, well-formed create info; the fence starts
        // unsignaled and is waited on by [`Compositor::begin_frame`].
        resources.frame_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(CompositorError::FenceCreate)?;
        Ok(resources)
    }
}

impl Drop for StartResources<'_> {
    fn drop(&mut self) {
        let context = self.context;
        let device = context.device();
        // SAFETY: `quiesce` waits for the device before this struct is
        // dropped, so no fence, semaphore or command buffer is still
        // referenced by the queue; null handles are never passed on.
        unsafe {
            if self.frame_fence != vk::Fence::null() {
                device.destroy_fence(self.frame_fence, None);
            }
            for semaphore in &self.chain_sems {
                if *semaphore != vk::Semaphore::null() {
                    device.destroy_semaphore(*semaphore, None);
                }
            }
            if self.command_buffer_count > 0 {
                device.free_command_buffers(context.command_pool(), &self.pipe_command_buffers);
                device.free_command_buffers(
                    context.command_pool(),
                    core::slice::from_ref(&self.layout_command_buffer),
                );
                device.free_command_buffers(
                    context.command_pool(),
                    core::slice::from_ref(&self.mix_command_buffer),
                );
            }
        }
    }
}

/// Synchronization objects of the frame in progress.
struct FrameSync<'p> {
    /// Device that owns both semaphores.
    device: &'p ash::Device,
    /// Wait semaphore importing the caller's `sync_file`; null when the
    /// frame waits on nothing.
    import_sem: vk::Semaphore,
    /// Signal semaphore exported as this frame's `sync_file`.
    present_sem: vk::Semaphore,
}

impl Drop for FrameSync<'_> {
    fn drop(&mut self) {
        // SAFETY: the compositor destroys the frame sync either after
        // the frame fence was waited on or after `device_wait_idle`, so
        // neither semaphore is still referenced by the queue; null
        // handles are skipped.
        unsafe {
            if self.import_sem != vk::Semaphore::null() {
                self.device
                    .destroy_semaphore(self.import_sem, None);
            }
            if self.present_sem != vk::Semaphore::null() {
                self.device
                    .destroy_semaphore(self.present_sem, None);
            }
        }
    }
}

/// Clear color used when a pipe cannot supply one (its target still
/// holds `UNDEFINED` contents).
const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Compositor-based renderer: drives pipes, then mixes their offscreen
/// framebuffers into the presentation target.
///
/// A compositor is created stopped; call [`Compositor::start`], register
/// pipes with [`Compositor::add_pipe`] and drive frames through
/// [`Compositor::begin_frame`] / [`Compositor::poll_composite`] /
/// [`Compositor::end_frame`] (or the [`Compositor::draw_frame`] helper).
/// Exactly one frame may be in flight at a time.
///
/// The compositor borrows its [`PipelineContext`] for `'p` and must be
/// dropped before it: dropping waits for the device and releases the
/// renderers of every registered pipe.
pub struct Compositor<'p> {
    /// Device-level state borrowed for the compositor's lifetime.
    context: &'p PipelineContext,
    /// Lifecycle state.
    state: CompositorState,
    /// Phase of the frame in progress.
    phase: CompositorPhase,
    /// Pipes of the frame in progress; dropped before `queue` so their
    /// futures release the renderers they borrow.
    active: Vec<ActivePipe>,
    /// Registered pipes in mix order.
    queue: PipeQueue,
    /// Channels to the presentation side.
    comm: CompositorEndpoint,
    /// Framebuffer snapshot of the current frame.
    framebuffers: Vec<OffscreenFramebuffer>,
    /// First mix slot of each queued pipe (see [`PipeCtx::base`]).
    bases: [usize; MAX_PIPES],
    /// Per-slot mix weights, sanitized by the collector.
    weights: [f32; MAX_MIX_TARGETS],
    /// Whether the snapshot must be rebuilt at the next `begin_frame`.
    framebuffers_dirty: bool,
    /// Whether the pipes are driven (see [`CompositorMessage::VisibilityChanged`]).
    visible: bool,
    /// Visibility change queued for the next `begin_frame`.
    pending_visible: Option<bool>,
    /// Tracked layout of every composited image; empty means "contents
    /// unknown" (the next layout pass discards them).
    layouts: Vec<(vk::Image, vk::ImageLayout)>,
    /// Bit *i* set when framebuffer *i* still holds `UNDEFINED` content.
    undefined_mask: u32,
    /// Frames whose `sync_file` was exported so far.
    frame_index: u64,
    /// Whether the frame fence still has to be waited on.
    in_flight: bool,
    /// Frame synchronization of the frame in progress.
    frame_sync: Option<FrameSync<'p>>,
    /// Resources allocated by [`Compositor::start`].
    start_resources: Option<StartResources<'p>>,
}

impl<'p> Compositor<'p> {
    /// Creates a stopped compositor for `context` with no pipes.
    ///
    /// Fetch the presentation-side endpoint with
    /// [`Compositor::renderer_endpoint`] before handing frames out.
    #[must_use]
    pub fn new(context: &'p PipelineContext) -> Self {
        let (endpoint, _) = comm_pair();
        Self {
            context,
            state: CompositorState::Stopped,
            phase: CompositorPhase::Idle,
            active: Vec::new(),
            queue: PipeQueue::new(),
            comm: endpoint,
            framebuffers: Vec::new(),
            bases: [0; MAX_PIPES],
            weights: [1.0; MAX_MIX_TARGETS],
            framebuffers_dirty: true,
            visible: true,
            pending_visible: None,
            layouts: Vec::new(),
            undefined_mask: 0,
            frame_index: 0,
            in_flight: false,
            frame_sync: None,
            start_resources: None,
        }
    }

    /// Current lifecycle state.
    #[inline]
    #[must_use]
    pub const fn state(&self) -> CompositorState {
        self.state
    }

    /// Current frame phase (see [`CompositorPhase`]).
    #[inline]
    #[must_use]
    pub const fn phase(&self) -> CompositorPhase {
        self.phase
    }

    /// Number of frames whose completion `sync_file` was exported.
    #[inline]
    #[must_use]
    pub const fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Whether the pipes are currently driven each frame.
    #[inline]
    #[must_use]
    pub const fn visible(&self) -> bool {
        self.visible
    }

    /// Requests a visibility change for the next frame.
    ///
    /// The change is applied by [`Compositor::begin_frame`] so a frame
    /// already in progress never mixes a half-recorded state.
    pub fn set_visible(&mut self, visible: bool) {
        self.pending_visible = Some(visible);
    }

    /// The compositor's own endpoint (messages and retired pipes flow
    /// out through it, registration flows in).
    #[inline]
    #[must_use]
    pub fn comm(&self) -> &CompositorEndpoint {
        &self.comm
    }

    /// A fresh endpoint for the presentation side of the same channels.
    #[must_use]
    pub fn renderer_endpoint(&self) -> RendererEndpoint {
        RendererEndpoint {
            hub: Rc::clone(&self.comm.hub),
        }
    }

    /// Number of registered pipes.
    #[must_use]
    pub fn pipe_count(&self) -> usize {
        self.queue.len()
    }

    /// The pipe registry (order equals mix order).
    #[must_use]
    pub fn queue(&self) -> &PipeQueue {
        &self.queue
    }

    /// Framebuffer snapshot collected for the current frame.
    #[must_use]
    pub fn framebuffers(&self) -> &[OffscreenFramebuffer] {
        &self.framebuffers
    }

    /// Marks the framebuffer snapshot stale; it is rebuilt by the next
    /// [`Compositor::begin_frame`].
    pub fn invalidate_framebuffers(&mut self) {
        self.framebuffers_dirty = true;
    }

    /// Allocates the frame resources and enters the running state.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::InvalidState`] — the compositor is not
    ///   stopped.
    /// * [`CompositorError::CommandAllocation`], [`CompositorError::SemaphoreCreate`],
    ///   [`CompositorError::FenceCreate`] — frame resources could not be
    ///   created; everything allocated first is released again.
    pub fn start(&mut self) -> Result<(), CompositorError> {
        if self.state != CompositorState::Stopped {
            return Err(CompositorError::InvalidState {
                operation: "start",
                state: self.state,
            });
        }
        self.start_resources = Some(StartResources::create(self.context)?);
        self.state = CompositorState::Running;
        self.phase = CompositorPhase::Idle;
        self.framebuffers_dirty = true;
        Ok(())
    }

    /// Suspends frame operations, keeping pipes and resources.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::InvalidState`] — the compositor is not
    ///   running.
    /// * [`CompositorError::FramePhase`] — a frame is in progress; finish
    ///   it first (or stop the compositor).
    pub fn pause(&mut self) -> Result<(), CompositorError> {
        if self.state != CompositorState::Running {
            return Err(CompositorError::InvalidState {
                operation: "pause",
                state: self.state,
            });
        }
        self.require_phase(CompositorPhase::Idle, "pause")?;
        self.state = CompositorState::Paused;
        Ok(())
    }

    /// Resumes a paused compositor.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::InvalidState`] — the compositor is not
    ///   paused.
    pub fn resume(&mut self) -> Result<(), CompositorError> {
        if self.state != CompositorState::Paused {
            return Err(CompositorError::InvalidState {
                operation: "resume",
                state: self.state,
            });
        }
        self.state = CompositorState::Running;
        Ok(())
    }

    /// Waits for the device, releases the frame resources and returns to
    /// the stopped state.
    ///
    /// Registered pipes are kept (and dropped with the compositor). A
    /// frame that was begun but not finished is discarded.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::InvalidState`] — the compositor is already
    ///   stopped.
    pub fn stop(&mut self) -> Result<(), CompositorError> {
        if self.state == CompositorState::Stopped {
            return Err(CompositorError::InvalidState {
                operation: "stop",
                state: self.state,
            });
        }
        self.quiesce();
        self.start_resources = None;
        self.state = CompositorState::Stopped;
        Ok(())
    }

    /// Checks that the frame is in `expected` phase.
    fn require_phase(
        &self,
        expected: CompositorPhase,
        operation: &'static str,
    ) -> Result<(), CompositorError> {
        if self.phase == expected {
            Ok(())
        } else {
            Err(CompositorError::FramePhase {
                operation,
                expected,
                actual: self.phase,
            })
        }
    }

    /// Waits for the device (ignoring device loss), retires the frame's
    /// synchronization, discards any recorded frame and resets the
    /// command pool.
    fn quiesce(&mut self) {
        let context = self.context;
        // SAFETY: the caller has exclusive access, so no frame can be
        // recording or submitting concurrently; a failed wait (device
        // lost) is tolerated because teardown must still run.
        unsafe {
            let _ = context.device().device_wait_idle();
        }
        self.frame_sync = None;
        self.in_flight = false;
        self.active.clear();
        if self.start_resources.is_some() {
            // The device is idle, so resetting the pool cannot discard
            // in-flight commands.
            // SAFETY: the pool belongs to `context`.
            unsafe {
                let _ = context
                    .device()
                    .reset_command_pool(context.command_pool(), vk::CommandPoolResetFlags::empty());
            }
        }
        self.phase = CompositorPhase::Idle;
    }

    /// Waits for the frame fence, retires the frame's semaphores and
    /// resets the command pool for a fresh frame.
    fn retire_previous_frame(&mut self) -> Result<(), CompositorError> {
        let context = self.context;
        let device = context.device();
        let fence = self
            .start_resources
            .as_ref()
            .map(|resources| resources.frame_fence)
            .ok_or(CompositorError::Internal(
                "the compositor is running without frame resources",
            ))?;
        if self.in_flight {
            // SAFETY: the fence was created by `StartResources::create`,
            // `&mut self` excludes concurrent use, and `u64::MAX` means
            // the only failure is device loss (which is reported).
            unsafe { device.wait_for_fences(core::slice::from_ref(&fence), true, u64::MAX) }
                .map_err(CompositorError::FenceWait)?;
            // SAFETY: the fence just signaled, so it is in the
            // signaled state and may be reset.
            unsafe { device.reset_fences(core::slice::from_ref(&fence)) }
                .map_err(CompositorError::FenceWait)?;
            self.in_flight = false;
        }
        // The frame completed (or never reached the queue), so its
        // semaphores are no longer referenced: drop them.
        self.frame_sync = None;
        // SAFETY: no submission is pending — either the fence above was
        // waited on, or the frame was discarded after `device_wait_idle`.
        unsafe { device.reset_command_pool(context.command_pool(), vk::CommandPoolResetFlags::empty()) }
            .map_err(CompositorError::CommandPoolReset)
    }

    /// Creates this frame's semaphores: an optional wait semaphore
    /// importing `wait_sync_file`, plus the exportable signal semaphore.
    fn create_frame_sync(&self, wait_sync_file: Option<OwnedFd>) -> Result<FrameSync<'p>, CompositorError> {
        let device = self.context.device();
        let import_sem = if let Some(sync_file) = wait_sync_file {
            let semaphore = create_binary_semaphore(device, false)?;
            match import_sync_fd(self.context, semaphore, sync_file) {
                Ok(()) => semaphore,
                Err(err) => {
                    // SAFETY: the semaphore was just created and never
                    // submitted.
                    unsafe { device.destroy_semaphore(semaphore, None) };
                    return Err(err);
                }
            }
        } else {
            vk::Semaphore::null()
        };
        let present_sem = match create_binary_semaphore(device, true) {
            Ok(semaphore) => semaphore,
            Err(err) => {
                if import_sem != vk::Semaphore::null() {
                    // SAFETY: created above, never submitted.
                    unsafe { device.destroy_semaphore(import_sem, None) };
                }
                return Err(err);
            }
        };
        Ok(FrameSync {
            device,
            import_sem,
            present_sem,
        })
    }
}

/// Creates a binary semaphore, optionally with SYNC_FD export enabled.
fn create_binary_semaphore(
    device: &ash::Device,
    export_sync_fd: bool,
) -> Result<vk::Semaphore, CompositorError> {
    let mut export_info =
        vk::ExportSemaphoreCreateInfo::default().handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    // The export info must outlive the create info that chains it, so it
    // is built before the `if` rather than inside a branch.
    let semaphore_info = if export_sync_fd {
        vk::SemaphoreCreateInfo::default().push_next(&mut export_info)
    } else {
        vk::SemaphoreCreateInfo::default()
    };
    // SAFETY: the device is valid and, when exporting, the
    // `VkExportSemaphoreCreateInfo` chain is alive for the call.
    unsafe { device.create_semaphore(&semaphore_info, None) }.map_err(CompositorError::SemaphoreCreate)
}

/// Imports a `sync_file` fd into `semaphore` as a temporary SYNC_FD
/// payload.
///
/// On success ownership of `sync_file` transfers to the implementation;
/// on failure the descriptor is closed here.
fn import_sync_fd(
    context: &PipelineContext,
    semaphore: vk::Semaphore,
    sync_file: OwnedFd,
) -> Result<(), CompositorError> {
    let raw_fd = sync_file.into_raw();
    let import_info = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(semaphore)
        // VUID-VkImportSemaphoreFdInfoKHR-handleType-07307: copy-
        // transference handle types (SYNC_FD) must be TEMPORARY.
        .flags(vk::SemaphoreImportFlags::TEMPORARY)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        .fd(raw_fd);
    // SAFETY: `semaphore` is a fresh binary semaphore (VUID-
    // vkImportSemaphoreFdKHR-semaphore-01142), SYNC_FD import was
    // verified as supported when the pipeline context was created, and
    // the import info is well-formed.
    let result = unsafe {
        context
            .external_semaphore_fd()
            .import_semaphore_fd(&import_info)
    };
    if let Err(err) = result {
        // The implementation only takes ownership on success; rebuild an
        // `OwnedFd` so the descriptor closes on this error path.
        // SAFETY: the fd was moved out above and not consumed because
        // the import failed.
        drop(unsafe { OwnedFd::from_raw(raw_fd) });
        return Err(CompositorError::SemaphoreImport(err));
    }
    Ok(())
}

/// Exports a signaled (or signaling) binary semaphore as a `sync_file`
/// fd representing GPU completion.
fn export_sync_fd(context: &PipelineContext, semaphore: vk::Semaphore) -> Result<OwnedFd, CompositorError> {
    let get_info = vk::SemaphoreGetFdInfoKHR::default()
        .semaphore(semaphore)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    // SAFETY: the semaphore was created with SYNC_FD in
    // `VkExportSemaphoreCreateInfo` and is signalled or has a pending
    // signal operation (it was just submitted), which satisfies
    // VUID-VkSemaphoreGetFdInfoKHR-handleType-01132/01135.
    let raw_fd = unsafe {
        context
            .external_semaphore_fd()
            .get_semaphore_fd(&get_info)
    }
    .map_err(CompositorError::SemaphoreExport)?;
    if raw_fd < 0 {
        return Err(CompositorError::Internal(
            "vkGetSemaphoreFdKHR completed without producing a sync_file fd",
        ));
    }
    // SAFETY: on success ownership of the new descriptor transfers to
    // the caller.
    Ok(unsafe { OwnedFd::from_raw(raw_fd) })
}

/// One image layout transition recorded by [`record_image_transition`].
#[derive(Clone, Copy)]
struct ImageTransition {
    /// Image whose whole subresource is transitioned.
    image: vk::Image,
    /// Layout before the barrier.
    old_layout: vk::ImageLayout,
    /// Layout after the barrier.
    new_layout: vk::ImageLayout,
    /// Pipeline stages whose prior work the barrier waits for.
    src_stage: vk::PipelineStageFlags,
    /// Accesses of `src_stage` made available by the barrier.
    src_access: vk::AccessFlags,
    /// Pipeline stages that observe the transition.
    dst_stage: vk::PipelineStageFlags,
    /// Accesses of `dst_stage` made visible by the barrier.
    dst_access: vk::AccessFlags,
}

impl ImageTransition {
    /// Builds a transition for `image`.
    const fn new(
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_stage: vk::PipelineStageFlags,
        dst_access: vk::AccessFlags,
    ) -> Self {
        Self {
            image,
            old_layout,
            new_layout,
            src_stage,
            src_access,
            dst_stage,
            dst_access,
        }
    }
}

/// Records an image layout transition over the whole subresource.
fn record_image_transition(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    transition: &ImageTransition,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(transition.src_access)
        .dst_access_mask(transition.dst_access)
        .old_layout(transition.old_layout)
        .new_layout(transition.new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(transition.image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: the command buffer is recording, the barrier references a
    // live image of the frame, and both stage and access masks are
    // well-formed for the queue that runs it.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            transition.src_stage,
            transition.dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

impl<'p> Compositor<'p> {
    /// Registers `renderer` as a pipe in the next frame.
    ///
    /// The compositor takes ownership of `renderer`; it is released when
    /// the pipe is removed (see [`Compositor::remove_pipe`]) or when the
    /// compositor is dropped.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::FramePhase`] — a frame is in progress; the
    ///   queue may only grow between frames so slot indices stay stable.
    /// * [`CompositorError::TooManyPipes`] — [`MAX_PIPES`] pipes are
    ///   already registered; `renderer` is released again.
    pub fn add_pipe<T: PipeSource + PipeSupplyTraits>(
        &mut self,
        renderer: T,
    ) -> Result<usize, CompositorError> {
        self.add_handoff(PipeHandoff::from_renderer(renderer))
    }

    /// Registers a pre-built [`PipeHandoff`]; see [`Compositor::add_pipe`].
    ///
    /// # Errors
    ///
    /// * [`CompositorError::FramePhase`] — a frame is in progress.
    /// * [`CompositorError::TooManyPipes`] — the queue is full; the
    ///   handoff is released again.
    pub fn add_handoff(&mut self, handoff: PipeHandoff) -> Result<usize, CompositorError> {
        if self.phase != CompositorPhase::Idle {
            return Err(CompositorError::FramePhase {
                operation: "add_pipe",
                expected: CompositorPhase::Idle,
                actual: self.phase,
            });
        }
        let index = self.queue.push(handoff)?;
        self.framebuffers_dirty = true;
        Ok(index)
    }

    /// Marks the pipe at `index` for removal; its renderer is released
    /// at the next [`Compositor::begin_frame`], once the frame that
    /// still referenced it has retired.
    ///
    /// Returns `false` when no pipe is registered at that index or it
    /// was already marked.
    pub fn remove_pipe(&mut self, index: usize) -> bool {
        self.queue.mark_removed(index)
    }

    /// The layout currently tracked for `image` (`None` = unknown, i.e.
    /// the next transition discards its contents).
    fn layout_of(&self, image: vk::Image) -> Option<vk::ImageLayout> {
        self.layouts
            .iter()
            .find(|(tracked, _)| *tracked == image)
            .map(|(_, layout)| *layout)
    }

    /// Records the layout an image ends this frame in.
    fn set_layout(&mut self, image: vk::Image, layout: vk::ImageLayout) {
        if let Some(entry) = self
            .layouts
            .iter_mut()
            .find(|(tracked, _)| *tracked == image)
        {
            entry.1 = layout;
        } else {
            self.layouts.push((image, layout));
        }
    }

    /// Applies the presentation side's messages, registers the pipes it
    /// queued, rebuilds the framebuffer snapshot when needed and applies
    /// pending weight and visibility changes.
    fn drain_comm(&mut self) {
        let mut weight_updates: Vec<(usize, f32)> = Vec::new();
        while let Some(message) = self.comm.try_recv_message() {
            match message {
                CompositorMessage::FramebuffersChanged => self.framebuffers_dirty = true,
                CompositorMessage::MixWeightChanged { slot, weight } => {
                    weight_updates.push((usize::from(slot), weight));
                }
                CompositorMessage::VisibilityChanged { visible } => self.pending_visible = Some(visible),
                // Presentation-side notifications have no effect on the
                // frame loop; they are accepted so both endpoints share
                // one message type.
                CompositorMessage::FrameComposited { .. } | CompositorMessage::Custom(_) => {}
            }
        }
        while let Some(handoff) = self.comm.try_recv_pipe() {
            if self.queue.push(handoff).is_err() {
                break;
            }
            self.framebuffers_dirty = true;
        }
        if self.framebuffers_dirty {
            self.recollect();
            self.framebuffers_dirty = false;
        }
        for (slot, weight) in weight_updates {
            if slot < self.framebuffers.len() {
                self.weights[slot] = sanitize_weight(weight);
            }
        }
        if let Some(visible) = self.pending_visible.take() {
            self.visible = visible;
        }
    }

    /// Hands pipes marked for removal back to the presentation side.
    ///
    /// Only runs after [`Compositor::retire_previous_frame`] waited for
    /// the fence, so releasing the renderers no longer races the GPU.
    fn release_removed_pipes(&mut self) {
        if !self.queue.has_removed() {
            return;
        }
        for handoff in self.queue.drain_removed() {
            self.framebuffers_dirty = true;
            // On `Full`/`Closed` the handoff is released here instead:
            // it can no longer be rendered and nobody is listening.
            let _ = self.comm.send_pipe(handoff);
        }
    }

    /// Rebuilds the framebuffer snapshot and mix weights from the queue.
    fn recollect(&mut self) {
        self.framebuffers.clear();
        self.weights = [1.0; MAX_MIX_TARGETS];
        self.bases = self
            .queue
            .collect_framebuffers(&mut self.framebuffers, &mut self.weights);
    }

    /// Flags the framebuffers whose contents the layout pass has to
    /// discard (unknown tracked layout), so pipes clear them instead of
    /// loading garbage.
    fn update_undefined_mask(&mut self) {
        let mut mask = 0_u32;
        for (index, framebuffer) in self.framebuffers.iter().enumerate() {
            if index < MAX_MIX_TARGETS && self.layout_of(framebuffer.image).is_none() {
                mask |= 1_u32 << index;
            }
        }
        self.undefined_mask = mask;
    }

    /// Creates one active slot per queued pipe for this frame.
    fn build_active_pipes(&mut self) {
        let context = self.context;
        let Some(resources) = self.start_resources.as_ref() else {
            return;
        };
        let format = PipelineContext::color_format();
        self.active.clear();
        for index in 0..self.queue.len() {
            let command_buffer = resources
                .pipe_command_buffers
                .get(index)
                .copied()
                .unwrap_or(vk::CommandBuffer::null());
            let base = self.bases.get(index).copied().unwrap_or(0);
            let ctx = PipeCtx::new(
                context.device(),
                command_buffer,
                context.width(),
                context.height(),
                format,
                self.frame_index,
                base,
                &self.framebuffers,
            );
            self.active.push(ActivePipe {
                future: None,
                ctx,
                command_buffer,
                state: PipeState::Recording,
            });
        }
    }

    /// Starts one frame: retires the previous submission, drains the
    /// communication channels, refreshes the framebuffer snapshot and
    /// creates the frame's synchronization objects.
    ///
    /// `wait_sync_file` is an optional `sync_file` fd (for example a DRM
    /// release point) to wait on before rendering; ownership is consumed
    /// — on success the driver takes the descriptor, on failure it is
    /// closed here. The first frame passes `None`.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::InvalidState`] — the compositor is not
    ///   running.
    /// * [`CompositorError::FramePhase`] — another frame is still in
    ///   progress.
    /// * [`CompositorError::FenceWait`], [`CompositorError::CommandPoolReset`],
    ///   [`CompositorError::SemaphoreCreate`], [`CompositorError::SemaphoreImport`]
    ///   — the previous frame could not be retired or this frame's
    ///   synchronization could not be created.
    pub fn begin_frame(&mut self, wait_sync_file: Option<OwnedFd>) -> Result<(), CompositorError> {
        if self.state != CompositorState::Running {
            return Err(CompositorError::InvalidState {
                operation: "begin_frame",
                state: self.state,
            });
        }
        self.require_phase(CompositorPhase::Idle, "begin_frame")?;
        self.retire_previous_frame()?;
        // Release retired pipes before draining: the recollect below
        // must not snapshot framebuffers of a pipe that just left.
        self.release_removed_pipes();
        self.drain_comm();
        self.update_undefined_mask();
        let frame_sync = self.create_frame_sync(wait_sync_file)?;
        self.frame_sync = Some(frame_sync);
        self.build_active_pipes();
        self.phase = CompositorPhase::Recording;
        Ok(())
    }
}

/// Source stage and access mask of a layout transition leaving `layout`.
///
/// The masks cover the accesses that produced the current contents; the
/// previous submission's fence was waited on in
/// [`Compositor::begin_frame`], so matching the producing stage keeps
/// every barrier valid without over-synchronizing.
fn transition_source(layout: vk::ImageLayout) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match layout {
        vk::ImageLayout::UNDEFINED => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::GENERAL => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_READ),
        _ => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
    }
}

/// Records a clear-only rendering scope that defines one framebuffer's
/// contents after its transition into `COLOR_ATTACHMENT_OPTIMAL`.
///
/// The compositor clears unknown contents itself so correctness never
/// depends on a pipe cooperating with [`PipeCtx::begin_render`].
fn record_framebuffer_clear(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    framebuffer: &OffscreenFramebuffer,
) {
    let attachment = vk::RenderingAttachmentInfo::default()
        .image_view(framebuffer.view)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: DEFAULT_CLEAR_COLOR,
            },
        });
    let area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: framebuffer.width,
            height: framebuffer.height,
        },
    };
    let rendering_info = vk::RenderingInfo::default()
        .render_area(area)
        .layer_count(1)
        .color_attachments(core::slice::from_ref(&attachment));
    // SAFETY: the command buffer is recording, the caller transitioned
    // the image to `COLOR_ATTACHMENT_OPTIMAL` first, the view is
    // compatible with that layout, and the render area fits the image.
    unsafe {
        device.cmd_begin_rendering(command_buffer, &rendering_info);
        device.cmd_end_rendering(command_buffer);
    }
}

/// Submits one recorded command buffer with optional wait and signal
/// semaphores (the frame's GPU-to-GPU chain).
///
/// # Errors
///
/// * [`CompositorError::Submit`] — the queue rejected the submission.
fn submit_commands(
    context: &PipelineContext,
    command_buffer: vk::CommandBuffer,
    wait: &[vk::Semaphore],
    wait_stages: &[vk::PipelineStageFlags],
    signal: &[vk::Semaphore],
    fence: vk::Fence,
) -> Result<(), CompositorError> {
    debug_assert_eq!(wait.len(), wait_stages.len());
    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(wait)
        .wait_dst_stage_mask(wait_stages)
        .command_buffers(&command_buffers)
        .signal_semaphores(signal);
    // SAFETY: all handles belong to this frame and are alive; wait
    // semaphores are fresh binary semaphores (or the frame's temporary
    // SYNC_FD import) and signal semaphores and the fence are
    // unsignaled, as required by `vkQueueSubmit`.
    unsafe {
        context
            .device()
            .queue_submit(context.queue(), core::slice::from_ref(&submit_info), fence)
    }
    .map_err(CompositorError::Submit)
}

impl<'p> Compositor<'p> {
    /// Drives the frame one step closer to completion.
    ///
    /// Called repeatedly — usually from an async executor through
    /// [`Compositor::composite_frame`] — until it resolves:
    ///
    /// * `Recording` → records and submits the layout pass.
    /// * `Pipes` → polls every pipe future with `cx`; [`Poll::Pending`]
    ///   means at least one future yielded, and the compositor is woken
    ///   through the waker that future stored.
    /// * `Mixing` → records and submits the mix pass; the frame is then
    ///   on the queue and [`Compositor::end_frame`] may export it.
    ///
    /// A failure discards the half-recorded frame (see
    /// [`Compositor::discard_frame`]) and reports the error. Calling
    /// this outside the driving phases reports
    /// [`CompositorError::FramePhase`] without touching the frame.
    ///
    /// # Errors
    ///
    /// Propagates the recording, pipe and submission errors of the
    /// phase it drives; the frame is discarded first.
    pub fn poll_composite(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), CompositorError>> {
        loop {
            match self.phase {
                CompositorPhase::Recording => {
                    if let Err(err) = self.record_and_submit_layout() {
                        return self.fail_frame(err);
                    }
                    self.phase = CompositorPhase::Pipes;
                }
                CompositorPhase::Pipes => match self.drive_pipes(cx) {
                    Ok(true) => self.phase = CompositorPhase::Mixing,
                    Ok(false) => return Poll::Pending,
                    Err(err) => return self.fail_frame(err),
                },
                CompositorPhase::Mixing => {
                    if let Err(err) = self.record_and_submit_mix() {
                        return self.fail_frame(err);
                    }
                    self.phase = CompositorPhase::Submitted;
                    return Poll::Ready(Ok(()));
                }
                CompositorPhase::Idle | CompositorPhase::Submitted => {
                    return Poll::Ready(Err(CompositorError::FramePhase {
                        operation: "poll_composite",
                        expected: CompositorPhase::Recording,
                        actual: self.phase,
                    }));
                }
            }
        }
    }

    /// Discards the frame in progress and wraps `error` as its result.
    fn fail_frame(&mut self, error: CompositorError) -> Poll<Result<(), CompositorError>> {
        self.discard_frame();
        Poll::Ready(Err(error))
    }

    /// Records the layout pass: parks every framebuffer in the layout
    /// the pipes (or a frozen mix) read or write it in, defines the
    /// contents of unknown targets and submits the pass as the head of
    /// the frame's semaphore chain.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Internal`] — the frame resources or the
    ///   chain semaphores are missing.
    /// * [`CompositorError::CommandRecord`] — the layout command buffer
    ///   could not be opened or closed.
    /// * [`CompositorError::Submit`] — the submission was rejected.
    fn record_and_submit_layout(&mut self) -> Result<(), CompositorError> {
        let context = self.context;
        let device = context.device();
        let (layout_command_buffer, chain_first) = match self.start_resources.as_ref() {
            Some(resources) => (resources.layout_command_buffer, resources.chain_sems[0]),
            None => {
                return Err(CompositorError::Internal(
                    "a frame was driven without start resources",
                ));
            }
        };
        if chain_first == vk::Semaphore::null() {
            return Err(CompositorError::Internal(
                "the chain semaphore pool is incomplete",
            ));
        }
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the pool was reset when this frame began, the buffer
        // was allocated from it, and `&mut self` excludes any other
        // recording.
        unsafe { device.begin_command_buffer(layout_command_buffer, &begin_info) }
            .map_err(CompositorError::CommandRecord)?;

        let visible = self.visible;
        let framebuffer_count = self.framebuffers.len().min(MAX_MIX_TARGETS);
        // Park every framebuffer in the layout the rest of the frame
        // reads or writes it in.
        for index in 0..framebuffer_count {
            let framebuffer = self.framebuffers[index];
            let old_layout = self
                .layout_of(framebuffer.image)
                .unwrap_or(vk::ImageLayout::UNDEFINED);
            if old_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
                continue;
            }
            let (src_stage, src_access) = transition_source(old_layout);
            record_image_transition(
                device,
                layout_command_buffer,
                &ImageTransition::new(
                    framebuffer.image,
                    old_layout,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    src_stage,
                    src_access,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
            );
        }
        // Define the contents of every framebuffer whose tracked layout
        // is unknown: a pipe loading them — and a frozen frame, which
        // runs no pipe at all — would otherwise read garbage.
        for index in 0..framebuffer_count {
            if self.undefined_mask & (1_u32 << index) == 0 {
                continue;
            }
            let framebuffer = self.framebuffers[index];
            record_framebuffer_clear(device, layout_command_buffer, &framebuffer);
        }
        if !visible {
            // Pipes are skipped while hidden: the mix pass samples the
            // framebuffers directly, so park them in the sampled layout.
            for index in 0..framebuffer_count {
                let framebuffer = self.framebuffers[index];
                record_image_transition(
                    device,
                    layout_command_buffer,
                    &ImageTransition::new(
                        framebuffer.image,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::AccessFlags::SHADER_READ,
                    ),
                );
            }
        }
        let end_layout = if visible {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        for index in 0..framebuffer_count {
            let image = self.framebuffers[index].image;
            self.set_layout(image, end_layout);
        }

        // SAFETY: the buffer is still recording; everything recorded
        // above targets images of this frame.
        unsafe { device.end_command_buffer(layout_command_buffer) }
            .map_err(CompositorError::CommandRecord)?;
        let import_wait_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let (wait, wait_stages): (&[vk::Semaphore], &[vk::PipelineStageFlags]) =
            match self.frame_sync.as_ref() {
                Some(frame_sync) if frame_sync.import_sem != vk::Semaphore::null() => (
                    core::slice::from_ref(&frame_sync.import_sem),
                    core::slice::from_ref(&import_wait_stage),
                ),
                _ => (&[], &[]),
            };
        let signal = [chain_first];
        submit_commands(
            context,
            layout_command_buffer,
            wait,
            wait_stages,
            &signal,
            vk::Fence::null(),
        )
    }

    /// Polls the recording pipe futures and submits every resolved
    /// prefix of the chain in queue order.
    ///
    /// Returns `true` once all pipes of the frame are on the queue.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Pipe`] — a future failed; the index is the
    ///   pipe's position in the queue.
    /// * [`CompositorError::CommandRecord`] — a pipe command buffer
    ///   could not be opened or closed.
    /// * [`CompositorError::Submit`] — the submission was rejected.
    /// * [`CompositorError::Internal`] — the queue diverged from the
    ///   frame slots or the frame resources are missing.
    fn drive_pipes(&mut self, cx: &mut Context<'_>) -> Result<bool, CompositorError> {
        let context = self.context;
        let device = context.device();
        for index in 0..self.active.len() {
            if self.active[index].state != PipeState::Recording {
                continue;
            }
            let (entry, renderer) = match self.queue.pipe(index) {
                Some(pipe) => (pipe.entry(), pipe.renderer()),
                None => {
                    return Err(CompositorError::Internal(
                        "the pipe queue diverged from the frame slots",
                    ));
                }
            };
            if self.active[index].future.is_none() {
                let begin_info =
                    vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                let command_buffer = self.active[index].command_buffer;
                // SAFETY: the pool was reset when this frame began and
                // no other code records this slot's buffer.
                unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
                    .map_err(CompositorError::CommandRecord)?;
                let env = PipeEnv {
                    renderer,
                    ctx: core::ptr::from_mut(&mut self.active[index].ctx),
                };
                self.active[index].future = Some(entry(env));
            }
            let poll_result = {
                let active = &mut self.active[index];
                let Some(future) = active.future.as_mut() else {
                    return Err(CompositorError::Internal("a recording pipe has no future"));
                };
                future.as_mut().poll(cx)
            };
            match poll_result {
                Poll::Pending => {}
                Poll::Ready(Ok(outcome)) => {
                    // Release the future before closing the buffer it
                    // recorded: it borrows the renderer and the context.
                    let active = &mut self.active[index];
                    active.future = None;
                    // SAFETY: the future recorded into this buffer and
                    // has just been dropped, so recording is finished.
                    unsafe { device.end_command_buffer(active.command_buffer) }
                        .map_err(CompositorError::CommandRecord)?;
                    active.state = PipeState::Ready;
                    if outcome == PipeOutcome::Remove {
                        self.queue.mark_removed(index);
                    }
                }
                Poll::Ready(Err(source)) => {
                    return Err(CompositorError::Pipe {
                        index,
                        source: Box::new(source),
                    });
                }
            }
        }
        // Submit the resolved prefix: pipe *k* waits on `chain[k]` (the
        // layout pass signals `chain[0]`) and signals `chain[k + 1]`.
        let resources = self
            .start_resources
            .as_ref()
            .ok_or(CompositorError::Internal(
                "a frame was driven without start resources",
            ))?;
        let mut index = 0;
        while index < self.active.len() && self.active[index].state == PipeState::Ready {
            let wait = resources.chain_sems[index];
            let signal = resources.chain_sems[index + 1];
            if wait == vk::Semaphore::null() || signal == vk::Semaphore::null() {
                return Err(CompositorError::Internal(
                    "the chain semaphore pool is incomplete",
                ));
            }
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            submit_commands(
                context,
                self.active[index].command_buffer,
                core::slice::from_ref(&wait),
                &wait_stages,
                core::slice::from_ref(&signal),
                vk::Fence::null(),
            )?;
            self.active[index].state = PipeState::Submitted;
            index += 1;
        }
        Ok(self
            .active
            .iter()
            .all(|slot| slot.state == PipeState::Submitted))
    }

    /// Rewrites the mix descriptor set for this frame.
    ///
    /// Slots past the frame's framebuffer count alias the first
    /// framebuffer so every entry of `u_frames[8]` stays statically
    /// bound (see `shaders/mix.frag`). The write is safe because exactly
    /// one frame is in flight and [`Compositor::begin_frame`] waited for
    /// the previous submission's fence, so the set is never in use.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Internal`] — the frame resources are missing.
    fn update_mix_descriptors(&self) -> Result<(), CompositorError> {
        let resources = self
            .start_resources
            .as_ref()
            .ok_or(CompositorError::Internal(
                "a frame was driven without start resources",
            ))?;
        let count = self.framebuffers.len().min(MAX_MIX_TARGETS);
        if count == 0 {
            return Ok(());
        }
        let sampler = resources.mix.sampler;
        let base_view = self.framebuffers[0].view;
        let mut image_infos = [vk::DescriptorImageInfo::default(); MAX_MIX_TARGETS];
        for (index, image_info) in image_infos.iter_mut().enumerate() {
            let view = if index < count {
                self.framebuffers[index].view
            } else {
                base_view
            };
            *image_info = vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(resources.mix.descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos)];
        // SAFETY: the set was allocated from this pool against a layout
        // declaring `MAX_MIX_TARGETS` combined image samplers at binding
        // 0, every view is a valid color target collected for this
        // frame, and the set is not referenced by pending commands.
        unsafe {
            self.context
                .device()
                .update_descriptor_sets(&writes, &[])
        };
        Ok(())
    }

    /// Records and submits the mix pass: samples every framebuffer into
    /// the presentation target and hands it off in `GENERAL` for the
    /// external compositor.
    ///
    /// With no framebuffers the pass only clears the target, which is
    /// what the zero-pipe path of the launcher relies on.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::Internal`] — the frame resources, the chain
    ///   semaphore or the frame's synchronization are missing.
    /// * [`CompositorError::CommandRecord`] — the mix command buffer
    ///   could not be opened or closed.
    /// * [`CompositorError::Submit`] — the submission was rejected.
    fn record_and_submit_mix(&mut self) -> Result<(), CompositorError> {
        let context = self.context;
        let device = context.device();
        let (mix_command_buffer, frame_fence) = match self.start_resources.as_ref() {
            Some(resources) => (resources.mix_command_buffer, resources.frame_fence),
            None => {
                return Err(CompositorError::Internal(
                    "a frame was driven without start resources",
                ));
            }
        };
        let chain_wait = self
            .start_resources
            .as_ref()
            .map_or(vk::Semaphore::null(), |resources| {
                resources.chain_sems[self.active.len()]
            });
        if chain_wait == vk::Semaphore::null() {
            return Err(CompositorError::Internal(
                "the chain semaphore pool is incomplete",
            ));
        }
        let present_sem = match self.frame_sync.as_ref() {
            Some(frame_sync) => frame_sync.present_sem,
            None => {
                return Err(CompositorError::Internal(
                    "the frame was begun without synchronization",
                ));
            }
        };
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the pool was reset when this frame began and no other
        // code records this buffer.
        unsafe { device.begin_command_buffer(mix_command_buffer, &begin_info) }
            .map_err(CompositorError::CommandRecord)?;

        let count = self.framebuffers.len().min(MAX_MIX_TARGETS);
        if self.visible {
            // The pipes left their targets in the attachment layout; the
            // mix shader samples them. A hidden frame skips this (the
            // layout pass already parked them in the sampled layout).
            for index in 0..count {
                let framebuffer = self.framebuffers[index];
                record_image_transition(
                    device,
                    mix_command_buffer,
                    &ImageTransition::new(
                        framebuffer.image,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::AccessFlags::SHADER_READ,
                    ),
                );
            }
        }
        // The pass clears the presentation target below, so its previous
        // contents are discarded; the closing transition leaves it in
        // `GENERAL`, the conservative state for non-Vulkan consumers.
        record_image_transition(
            device,
            mix_command_buffer,
            &ImageTransition::new(
                context.image(),
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
        );
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: context.width(),
                height: context.height(),
            },
        };
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(context.image_view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: DEFAULT_CLEAR_COLOR,
                },
            });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(core::slice::from_ref(&attachment));
        // SAFETY: the command buffer is recording, the presentation view
        // matches the `COLOR_ATTACHMENT_OPTIMAL` layout recorded above,
        // and dynamic rendering needs no render pass object.
        unsafe { device.cmd_begin_rendering(mix_command_buffer, &rendering_info) };

        if count > 0 {
            self.update_mix_descriptors()?;
            let resources = self
                .start_resources
                .as_ref()
                .ok_or(CompositorError::Internal(
                    "a frame was driven without start resources",
                ))?;
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: context.width() as f32,
                height: context.height() as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            // SAFETY: the rendering scope is open, the pipeline declares
            // viewport and scissor dynamic, and every handle belongs to
            // this frame.
            unsafe {
                device.cmd_set_viewport(mix_command_buffer, 0, core::slice::from_ref(&viewport));
                device.cmd_set_scissor(mix_command_buffer, 0, core::slice::from_ref(&area));
                device.cmd_bind_pipeline(
                    mix_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    resources.mix.pipeline,
                );
                let sets = [resources.mix.descriptor_set];
                device.cmd_bind_descriptor_sets(
                    mix_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    resources.mix.pipeline_layout,
                    0,
                    &sets,
                    &[],
                );
            }
            let push = mix_push(count, &self.weights);
            // SAFETY: `MixPush` is `repr(C)` plain data with no padding
            // the shader could observe, so reading it as bytes is sound.
            let push_bytes = unsafe {
                core::slice::from_raw_parts(
                    core::ptr::from_ref(&push).cast::<u8>(),
                    core::mem::size_of::<MixPush>(),
                )
            };
            // SAFETY: the push range declared when the pipeline layout
            // was created is exactly `size_of::<MixPush>()` at
            // fragment-stage offset 0, and the bytes outlive the call.
            unsafe {
                device.cmd_push_constants(
                    mix_command_buffer,
                    resources.mix.pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    push_bytes,
                );
                // Fullscreen triangle generated by `mix.vert`.
                device.cmd_draw(mix_command_buffer, 3, 1, 0, 0);
            }
        }
        // SAFETY: the rendering scope opened above has not been ended.
        unsafe { device.cmd_end_rendering(mix_command_buffer) };
        record_image_transition(
            device,
            mix_command_buffer,
            &ImageTransition::new(
                context.image(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::empty(),
            ),
        );
        // SAFETY: the buffer is still recording; everything recorded
        // above targets images of this frame.
        unsafe { device.end_command_buffer(mix_command_buffer) }.map_err(CompositorError::CommandRecord)?;

        let wait = [chain_wait];
        let wait_stages =
            [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::FRAGMENT_SHADER];
        let signal = [present_sem];
        submit_commands(
            context,
            mix_command_buffer,
            &wait,
            &wait_stages,
            &signal,
            frame_fence,
        )?;
        self.in_flight = true;
        Ok(())
    }
}

impl<'p> Compositor<'p> {
    /// Finishes the frame: exports GPU completion of the mix pass as a
    /// `sync_file` descriptor for the presentation layer.
    ///
    /// The frame's semaphores stay alive until the next
    /// [`Compositor::begin_frame`] waits for the frame fence; the
    /// exported descriptor remains valid after they are destroyed.
    ///
    /// Calling this while a frame is still being recorded discards that
    /// frame first (so the compositor never wedges); calling it when no
    /// frame was submitted only reports the phase error.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::FramePhase`] — no frame is on the queue.
    /// * [`CompositorError::SemaphoreExport`] — the driver refused to
    ///   export the signal semaphore; the frame is discarded first.
    pub fn end_frame(&mut self) -> Result<OwnedFd, CompositorError> {
        if self.phase != CompositorPhase::Submitted {
            if self.phase != CompositorPhase::Idle {
                self.discard_frame();
            }
            return Err(CompositorError::FramePhase {
                operation: "end_frame",
                expected: CompositorPhase::Submitted,
                actual: self.phase,
            });
        }
        let context = self.context;
        let present_sem = match self.frame_sync.as_ref() {
            Some(frame_sync) => frame_sync.present_sem,
            None => {
                self.discard_frame();
                return Err(CompositorError::Internal(
                    "the frame was submitted without synchronization",
                ));
            }
        };
        match export_sync_fd(context, present_sem) {
            Ok(sync_file) => {
                // The presentation side may not be listening (it may
                // have no endpoint yet); the export above succeeded.
                let _ = self
                    .comm
                    .send_message(CompositorMessage::FrameComposited {
                        frame_index: self.frame_index,
                    });
                self.frame_index += 1;
                self.phase = CompositorPhase::Idle;
                Ok(sync_file)
            }
            Err(err) => {
                self.discard_frame();
                Err(err)
            }
        }
    }

    /// Returns to the idle phase after a failed frame: waits for the
    /// device, resets the chain fence, command pool and semaphores and
    /// forgets the layouts of every image the abandoned frame touched.
    ///
    /// Cleanup is best-effort — the original error is reported by the
    /// caller, and a resource that cannot be reset leaves the compositor
    /// idle with the frame's submissions no longer in flight.
    fn discard_frame(&mut self) {
        let context = self.context;
        let device = context.device();
        // SAFETY: `&mut self` excludes concurrent use; a failed wait
        // (device lost) is tolerated because the state below must be
        // reset regardless.
        let _ = unsafe { device.device_wait_idle() };
        if let Some(resources) = self.start_resources.as_ref() {
            let fence = resources.frame_fence;
            if fence != vk::Fence::null() {
                // SAFETY: the device is idle, so the fence has no pending
                // signal operation; resetting a signalled fence returns
                // it to the unsignaled state the next submission needs.
                let _ = unsafe { device.reset_fences(core::slice::from_ref(&fence)) };
            }
            // SAFETY: the device is idle, so resetting the pool cannot
            // discard in-flight commands; it releases the buffers the
            // abandoned frame recorded.
            let _ = unsafe {
                device.reset_command_pool(context.command_pool(), vk::CommandPoolResetFlags::empty())
            };
        }
        // Futures first (they borrow the renderers and the contexts),
        // then the frame's synchronization and layout tracking.
        self.active.clear();
        self.frame_sync = None;
        self.in_flight = false;
        self.layouts.clear();
        self.undefined_mask = 0;
        self.phase = CompositorPhase::Idle;
        self.reset_chain_semaphores();
    }

    /// Destroys and recreates the chain semaphores after a discarded
    /// frame.
    ///
    /// A binary semaphore left signalled by the abandoned submission
    /// cannot be signalled again, so the whole chain is rebuilt; a
    /// failure leaves slots null and the frame loop rejects the
    /// incomplete pool with [`CompositorError::Internal`] instead of
    /// handing a null handle to the driver.
    fn reset_chain_semaphores(&mut self) {
        let Some(resources) = self.start_resources.as_mut() else {
            return;
        };
        let device = resources.context.device();
        for semaphore in resources.chain_sems.iter_mut() {
            if *semaphore != vk::Semaphore::null() {
                // SAFETY: the device is idle, so no submission still
                // references a chain semaphore, and each handle was
                // created on this device.
                unsafe { device.destroy_semaphore(*semaphore, None) };
                *semaphore = vk::Semaphore::null();
            }
        }
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        for semaphore in resources.chain_sems.iter_mut() {
            // SAFETY: a plain, well-formed create info; a failure leaves
            // the slot null, which the frame loop detects.
            *semaphore =
                unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap_or(vk::Semaphore::null());
        }
    }

    /// Runs one whole frame asynchronously: begins it, drives every
    /// phase to submission and exports completion.
    ///
    /// `wait_sync_file` is the optional `sync_file` to wait on before
    /// rendering (ownership is consumed); the returned descriptor
    /// signals when the mix pass finished.
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`Compositor::begin_frame`],
    /// [`Compositor::poll_composite`] and [`Compositor::end_frame`].
    pub async fn composite_frame(
        &mut self,
        wait_sync_file: Option<OwnedFd>,
    ) -> Result<OwnedFd, CompositorError> {
        self.begin_frame(wait_sync_file)?;
        core::future::poll_fn(|cx| self.poll_composite(cx)).await?;
        self.end_frame()
    }

    /// Drives one frame to submission without exporting it, leaving the
    /// compositor in the `Submitted` phase for a custom
    /// [`Compositor::end_frame`].
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`Compositor::begin_frame`] and
    /// [`Compositor::poll_composite`].
    pub async fn draw_frame(&mut self, wait_sync_file: Option<OwnedFd>) -> Result<(), CompositorError> {
        self.begin_frame(wait_sync_file)?;
        core::future::poll_fn(|cx| self.poll_composite(cx)).await
    }
}

impl Drop for Compositor<'_> {
    fn drop(&mut self) {
        self.quiesce();
        // Unpark the presentation side: its receive futures resolve to
        // `None` once the channels close.
        self.comm.close();
        // Field order keeps this implicit step documented: `active` (the
        // pipe futures) drops before `queue` (the renderers they borrow)
        // and `start_resources` (whose `Drop` assumes an idle device)
        // drops last.
    }
}

/// Drives `future` to completion on the current thread.
///
/// A minimal single-threaded executor: the future is polled with a
/// no-op waker and every [`Poll::Pending`] is retried after a short
/// sleep, so pipe futures that wait on the communication channels make
/// progress through re-polling alone. Use it from synchronous code (the
/// display loop); executors should await
/// [`Compositor::composite_frame`] directly instead.
#[must_use]
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = core::pin::pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => {
                let delay = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 1_000_000,
                };
                // SAFETY: `delay` is a valid timespec, the remaining
                // time output is null (unused), and an `EINTR` just
                // retries on the next loop iteration.
                unsafe { libc::nanosleep(&delay, core::ptr::null_mut()) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the GPU-independent parts of the compositor: pipe
    //! registry, channels, framebuffer collection, SPIR-V loading,
    //! push-constant layout and the display impls.

    use super::*;
    use ash::vk::Handle;

    /// Renderer contributing at most one framebuffer with a given weight.
    struct DummyRenderer {
        /// The frame it hands to the mix pass, if any.
        framebuffer: Option<OffscreenFramebuffer>,
        /// Weight reported for that framebuffer.
        weight: f32,
    }

    impl DummyRenderer {
        /// A pipe whose fabricated (but structurally valid) 64×64 target
        /// passes the collector's checks without a live Vulkan device.
        fn with_framebuffer(weight: f32) -> Self {
            Self {
                framebuffer: Some(OffscreenFramebuffer {
                    image: vk::Image::from_raw(1),
                    view: vk::ImageView::from_raw(1),
                    format: vk::Format::B8G8R8A8_UNORM,
                    width: 64,
                    height: 64,
                }),
                weight,
            }
        }

        /// A pipe that contributes no framebuffers.
        fn empty() -> Self {
            Self {
                framebuffer: None,
                weight: 1.0,
            }
        }
    }

    impl PipeSource for DummyRenderer {
        fn pipe_entry<'a>(_renderer: &'a mut Self, _ctx: &'a mut PipeCtx) -> PipeFuture<'a> {
            Box::pin(core::future::ready(Ok(PipeOutcome::Keep)))
        }
    }

    impl PipeSupplyTraits for DummyRenderer {
        fn offscreen_framebuffer_count(&self) -> usize {
            usize::from(self.framebuffer.is_some())
        }

        fn offscreen_framebuffer(&self, index: usize) -> Option<OffscreenFramebuffer> {
            if index == 0 { self.framebuffer } else { None }
        }

        fn mix_weight(&self, _index: usize) -> f32 {
            self.weight
        }
    }

    #[test]
    fn pipe_dispatches_to_the_registered_type() {
        let handoff = PipeHandoff::from_renderer(DummyRenderer::empty());
        let expected: PipeEntry = trampoline::<DummyRenderer>;
        assert!(core::ptr::fn_addr_eq(handoff.pipe().entry(), expected));
        assert_eq!(core::mem::offset_of!(Pipe, entry), 0);
        assert_eq!(
            core::mem::offset_of!(Pipe, renderer),
            core::mem::size_of::<PipeEntry>()
        );
    }

    #[test]
    fn pipe_queue_registers_and_retires_in_order() {
        let mut queue = PipeQueue::new();
        let first = queue
            .push(PipeHandoff::from_renderer(DummyRenderer::empty()))
            .unwrap();
        let second = queue
            .push(PipeHandoff::from_renderer(DummyRenderer::empty()))
            .unwrap();
        assert_eq!((first, second), (0, 1));
        assert_eq!(queue.len(), 2);

        assert!(queue.mark_removed(0));
        assert!(!queue.mark_removed(0), "already marked");
        assert!(!queue.mark_removed(7), "no such pipe");
        assert!(queue.has_removed());

        let retired = queue.drain_removed();
        assert_eq!(retired.len(), 1);
        assert_eq!(queue.len(), 1);
        assert!(!queue.has_removed());
        assert!(queue.pipe(0).is_some());
        assert!(queue.pipe(1).is_none());
    }

    #[test]
    fn pipe_queue_enforces_capacity() {
        let mut queue = PipeQueue::new();
        for _ in 0..MAX_PIPES {
            queue
                .push(PipeHandoff::from_renderer(DummyRenderer::empty()))
                .unwrap();
        }
        match queue.push(PipeHandoff::from_renderer(DummyRenderer::empty())) {
            Err(CompositorError::TooManyPipes { max }) => assert_eq!(max, MAX_PIPES),
            other => panic!("expected TooManyPipes, got {other:?}"),
        }
        assert_eq!(queue.len(), MAX_PIPES);
    }

    #[test]
    fn collect_framebuffers_fills_slots_and_sanitizes_weights() {
        let mut queue = PipeQueue::new();
        queue
            .push(PipeHandoff::from_renderer(DummyRenderer::with_framebuffer(0.25)))
            .unwrap();
        queue
            .push(PipeHandoff::from_renderer(DummyRenderer::with_framebuffer(
                f32::NAN,
            )))
            .unwrap();
        queue
            .push(PipeHandoff::from_renderer(DummyRenderer::with_framebuffer(2.0)))
            .unwrap();
        queue
            .push(PipeHandoff::from_renderer(DummyRenderer::empty()))
            .unwrap();

        let mut framebuffers = Vec::new();
        let mut weights = [1.0; MAX_MIX_TARGETS];
        let bases = queue.collect_framebuffers(&mut framebuffers, &mut weights);

        assert_eq!(framebuffers.len(), 3);
        assert_eq!(&bases[..4], &[0, 1, 2, 3]);
        assert!(
            framebuffers
                .iter()
                .all(OffscreenFramebuffer::is_valid)
        );
        assert_eq!(weights[0], 0.25);
        assert_eq!(weights[1], 0.0, "NaN is rejected");
        assert_eq!(weights[2], 1.0, "out-of-range weights clamp to 1.0");
        assert!(!OffscreenFramebuffer::EMPTY.is_valid());
    }

    #[test]
    fn channel_reports_full_and_receives_in_order() {
        let channel = Channel::<u32>::new(2);
        channel.try_send(1).unwrap();
        channel.try_send(2).unwrap();
        match channel.try_send(3) {
            Err(CommError::Full(value)) => assert_eq!(value, 3),
            other => panic!("expected Full, got {other:?}"),
        }
        assert_eq!(channel.try_recv(), Some(1));
        assert_eq!(channel.try_recv(), Some(2));
        assert_eq!(channel.try_recv(), None);
    }

    #[test]
    fn channel_close_drains_for_receivers_and_rejects_senders() {
        let channel = Channel::<u32>::new(4);
        channel.try_send(7).unwrap();
        channel.close();
        assert!(matches!(channel.try_send(8), Err(CommError::Closed(8))));
        assert_eq!(channel.try_recv(), Some(7));
        assert_eq!(channel.try_recv(), None);
    }

    #[test]
    fn channel_futures_complete_under_a_noop_waker() {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);

        // A send into a full channel yields until a slot frees up, then
        // stays ready when polled again.
        let full = Channel::<u32>::new(1);
        full.try_send(1).unwrap();
        let mut send = SendFuture {
            channel: &full,
            message: Some(2),
        };
        assert!(matches!(Pin::new(&mut send).poll(&mut cx), Poll::Pending));
        assert_eq!(full.try_recv(), Some(1));
        assert!(matches!(Pin::new(&mut send).poll(&mut cx), Poll::Ready(Ok(()))));
        assert!(matches!(Pin::new(&mut send).poll(&mut cx), Poll::Ready(Ok(()))));
        assert_eq!(full.try_recv(), Some(2));

        // A receive on an empty channel yields until a value arrives,
        // then fuses to `None`.
        let channel = Channel::<u32>::new(1);
        let mut recv = RecvFuture {
            channel: &channel,
            done: false,
        };
        assert!(matches!(Pin::new(&mut recv).poll(&mut cx), Poll::Pending));
        channel.try_send(9).unwrap();
        assert!(matches!(Pin::new(&mut recv).poll(&mut cx), Poll::Ready(Some(9))));
        assert!(matches!(Pin::new(&mut recv).poll(&mut cx), Poll::Ready(None)));
    }

    #[test]
    fn comm_pair_routes_messages_and_pipes_both_ways() {
        let (compositor, renderer) = comm_pair();

        renderer
            .send_message(CompositorMessage::VisibilityChanged { visible: false })
            .unwrap();
        assert!(matches!(
            compositor.try_recv_message(),
            Some(CompositorMessage::VisibilityChanged { visible: false })
        ));

        compositor
            .send_message(CompositorMessage::FrameComposited { frame_index: 3 })
            .unwrap();
        assert!(matches!(
            renderer.try_recv_message(),
            Some(CompositorMessage::FrameComposited { frame_index: 3 })
        ));

        // Registration flows presentation → compositor; retirement
        // flows back compositor → presentation.
        renderer
            .send_pipe(PipeHandoff::from_renderer(DummyRenderer::empty()))
            .unwrap();
        assert!(compositor.try_recv_pipe().is_some());
        compositor
            .send_pipe(PipeHandoff::from_renderer(DummyRenderer::empty()))
            .unwrap();
        assert!(renderer.try_recv_pipe().is_some());
    }

    #[test]
    fn mix_push_matches_the_shader_layout() {
        use core::mem::{offset_of, size_of};

        assert_eq!(size_of::<MixPush>(), 48);
        assert_eq!(offset_of!(MixPush, weights0), 16);
        assert_eq!(offset_of!(MixPush, weights1), 32);

        let mut weights = [1.0; MAX_MIX_TARGETS];
        weights[3] = 0.5;
        let push = mix_push(2, &weights);
        assert_eq!(push.count, 2);
        assert_eq!(push.weights0, [1.0, 1.0, 1.0, 0.5]);
        assert_eq!(push.weights1, [1.0; 4]);
        assert_eq!(push.pad, [0; 3]);
    }

    #[test]
    fn load_spir_v_accepts_embedded_modules_and_rejects_garbage() {
        let module = load_spir_v(include_bytes!("shaders/mix.vert.spv")).unwrap();
        assert_eq!(module[0], SPIRV_MAGIC);

        let fragment = load_spir_v(include_bytes!("shaders/mix.frag.spv")).unwrap();
        assert!(fragment.len() > module.len());

        assert!(load_spir_v(&[]).is_err(), "empty module");
        assert!(load_spir_v(&[0_u8; 8]).is_err(), "bad magic");
        let embedded = include_bytes!("shaders/mix.vert.spv");
        assert!(load_spir_v(&embedded[..6]).is_err(), "not a multiple of 4");
    }

    #[test]
    fn sanitize_weight_rejects_non_finite_and_clamps() {
        assert_eq!(sanitize_weight(f32::NAN), 0.0);
        assert_eq!(sanitize_weight(f32::INFINITY), 0.0);
        assert_eq!(sanitize_weight(f32::NEG_INFINITY), 0.0);
        assert_eq!(sanitize_weight(-1.0), 0.0);
        assert_eq!(sanitize_weight(0.5), 0.5);
        assert_eq!(sanitize_weight(4.0), 1.0);
    }

    #[test]
    fn display_impls_render_expected_text() {
        assert_eq!(CompositorState::Running.to_string(), "running");
        assert_eq!(CompositorPhase::Mixing.to_string(), "mixing");
        assert_eq!(
            CompositorMessage::FrameComposited { frame_index: 2 }.to_string(),
            "frame 2 composited"
        );
        assert_eq!(
            CompositorError::TooManyPipes { max: 8 }.to_string(),
            "at most 8 pipes may be registered"
        );
        assert_eq!(CommError::Full(1u32).to_string(), "the channel is full");
    }

    #[test]
    fn block_on_drives_pending_futures_to_completion() {
        let mut polls = 0_u32;
        let future = core::future::poll_fn(|_cx| {
            polls += 1;
            if polls < 3 {
                Poll::<u32>::Pending
            } else {
                Poll::Ready(polls)
            }
        });
        assert_eq!(block_on(future), 3);
    }
}
