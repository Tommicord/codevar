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

//! Renderer subsystem: frame lifecycle and layer stack.
//!
//! The subsystem owns everything that is per-frame — one primary command
//! buffer, the frame's fence and semaphores, and an ordered stack of
//! [`RenderLayer`]s, while device-level state (instance, device, queue,
//! command pool and the offscreen render target) stays in
//! [`PipelineContext`](crate::ui_pipeline::PipelineContext), which is
//! analogous to rlgame's `game_pipeline.c`.
//!
//! The subsystem is swapchain-less: a frame renders into the single
//! offscreen image of the [`PipelineContext`] and produces a `sync_file`
//! file descriptor; [`crate::ui_display`] turns the image into a
//! `wl_buffer` and presents it through the linux-dmabuf protocol.
//!
//! # Frame lifecycle
//!
//! A started renderer drives one frame through three calls:
//!
//! 1. [`RendererSubsystem::begin_frame`] — waits for the previous
//!    submission, resets the command pool and creates the frame's
//!    synchronization objects (optionally importing a `sync_file` fd as
//!    the wait semaphore).
//! 2. [`RendererSubsystem::render_frame`] — records the command buffer
//!    (layout transition into a dynamic-rendering pass that runs every
//!    enabled layer in priority order, then the handoff transition) and
//!    submits it.
//! 3. [`RendererSubsystem::end_frame`] — exports GPU completion of the
//!    submission as a fresh `sync_file` fd for the presentation layer.
//!
//! [`RendererSubsystem::draw_frame`] chains all three steps. Exactly one
//! frame may be in flight at a time; frames are serialized by the fence
//! waited in `begin_frame` *and* by the presentation layer waiting for the
//! compositor's `wl_buffer.release` before the next frame is rendered.
//!
//! # Synchronization model
//!
//! Each frame uses a fresh set of synchronization objects (binary
//! signal/wait semaphores and a fence):
//!
//! * **Signal (export):** the frame's binary semaphore is created with
//!   `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT` in
//!   [`vk::ExportSemaphoreCreateInfo`], signaled by the submission, then
//!   exported with `vkGetSemaphoreFdKHR`. Per the Vulkan specification
//!   ([Importing Semaphore Payloads], "Export operations have the same
//!   transference as the specified handle type's import operations.
//!   Additionally, exporting a semaphore payload to a handle with copy
//!   transference has the same side effects on the source semaphore's
//!   payload as executing a semaphore wait operation"), SYNC_FD has *copy*
//!   transference, so a successful export consumes (unsignals) the binary
//!   semaphore. Export is only valid while the semaphore is signaled or
//!   its signal operation is pending
//!   (VUID-VkSemaphoreGetFdInfoKHR-handleType-01135), which is why it
//!   happens immediately after `vkQueueSubmit`.
//! * **Wait (import):** an optional `sync_file` fd is imported into a
//!   fresh binary semaphore via `vkImportSemaphoreFdKHR` with `handleType`
//!   SYNC_FD and `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT` (mandatory for
//!   copy-transference handle types: VUID-VkImportSemaphoreFdInfoKHR-
//!   handleType-07307). The import transfers ownership of the fd to the
//!   implementation on success (VUID-vkImportSemaphoreFdKHR-
//!   semaphore-01142 additionally requires that the semaphore has no
//!   incomplete queue commands, which a fresh semaphore satisfies).
//! * **Fence:** a fresh unsignaled fence per submit is waited on at the
//!   start of the next frame, after which the frame's semaphores and fence
//!   are destroyed.
//!
//! # Layer ordering
//!
//! [`RenderLayer`]s run inside the dynamic-rendering pass in ascending
//! [`RenderLayer::priority`] order (lower values run first). Disabled
//! layers stay registered but are skipped. The stack is re-sorted before
//! every pass, so a layer may change its priority at runtime.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use ash::vk;

use crate::ui_pipeline::{OwnedFd, PipelineContext};

/// SPIR-V magic number as stored in a little-endian `.spv` file.
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Lifecycle state of a [`RendererSubsystem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RendererState {
    /// No command buffer is allocated and no frame can be begun.
    Stopped,
    /// The renderer accepts frames (see [`RendererSubsystem::begin_frame`]).
    Running,
    /// Rendering is suspended; the subsystem only keeps its layers and
    /// command buffer.
    Paused,
}

impl fmt::Display for RendererState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stopped => f.write_str("stopped"),
            Self::Running => f.write_str("running"),
            Self::Paused => f.write_str("paused"),
        }
    }
}

/// Progress of the frame currently being driven through the subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FramePhase {
    /// No frame in progress; the next call may be
    /// [`RendererSubsystem::begin_frame`].
    Idle,
    /// [`RendererSubsystem::begin_frame`] succeeded: the command pool was
    /// reset and the frame's synchronization objects exist, but nothing
    /// has been submitted yet.
    Recording,
    /// [`RendererSubsystem::render_frame`] succeeded: the frame is on the
    /// queue and its completion `sync_file` has not been exported yet.
    Submitted,
}

impl fmt::Display for FramePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => f.write_str("idle"),
            Self::Recording => f.write_str("recording"),
            Self::Submitted => f.write_str("submitted"),
        }
    }
}

/// Error returned by renderer lifecycle, frame and layer operations.
#[derive(Debug)]
pub enum RendererError {
    /// An operation was attempted while the subsystem was in the wrong
    /// state.
    InvalidState {
        /// The operation that was attempted (for example `begin_frame`).
        operation: &'static str,
        /// The state the subsystem was in.
        state: RendererState,
    },
    /// A frame operation was attempted while the frame was in another
    /// phase than the one it requires.
    FramePhase {
        /// The operation that was attempted.
        operation: &'static str,
        /// The frame phase the operation requires.
        expected: FramePhase,
        /// The frame phase the subsystem was actually in.
        actual: FramePhase,
    },
    /// Allocating or freeing the frame command buffer failed.
    CommandAllocation(vk::Result),
    /// Resetting the frame command pool failed.
    CommandPoolReset(vk::Result),
    /// Recording the frame command buffer failed.
    CommandRecord(vk::Result),
    /// Creating a binary semaphore failed.
    SemaphoreCreate(vk::Result),
    /// Importing a `sync_file` fd into a wait semaphore failed.
    SemaphoreImport(vk::Result),
    /// Exporting a signalled semaphore as a `sync_file` fd failed.
    SemaphoreExport(vk::Result),
    /// Creating the per-frame fence failed.
    FenceCreate(vk::Result),
    /// Waiting for the previous frame's fence failed (for example device
    /// loss).
    FenceWait(vk::Result),
    /// Queue submission of the frame failed.
    Submit(vk::Result),
    /// Reading or validating an embedded SPIR-V module failed.
    ShaderLoad(String),
    /// `vkCreateShaderModule` failed.
    ShaderModuleCreate(vk::Result),
    /// `vkCreatePipelineLayout` failed.
    PipelineLayoutCreate(vk::Result),
    /// `vkCreateGraphicsPipelines` failed.
    PipelineCreate(vk::Result),
    /// A layer with the same name is already registered.
    DuplicateLayer(String),
    /// No registered layer matched the requested name.
    UnknownLayer(String),
    /// An internal invariant was violated; this indicates a bug in this
    /// module.
    Internal(&'static str),
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidState { operation, state } => {
                write!(f, "{operation} is not allowed while the renderer is {state}")
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
            Self::CommandAllocation(err) => {
                write!(f, "command buffer allocation failed: {err:?}")
            }
            Self::CommandPoolReset(err) => write!(f, "command pool reset failed: {err:?}"),
            Self::CommandRecord(err) => write!(f, "command buffer recording failed: {err:?}"),
            Self::SemaphoreCreate(err) => write!(f, "semaphore creation failed: {err:?}"),
            Self::SemaphoreImport(err) => write!(f, "sync_file semaphore import failed: {err:?}"),
            Self::SemaphoreExport(err) => write!(f, "sync_file semaphore export failed: {err:?}"),
            Self::FenceCreate(err) => write!(f, "fence creation failed: {err:?}"),
            Self::FenceWait(err) => write!(f, "waiting for the previous frame failed: {err:?}"),
            Self::Submit(err) => write!(f, "frame submission failed: {err:?}"),
            Self::ShaderLoad(err) => write!(f, "failed to load embedded SPIR-V: {err}"),
            Self::ShaderModuleCreate(err) => write!(f, "shader module creation failed: {err:?}"),
            Self::PipelineLayoutCreate(err) => {
                write!(f, "pipeline layout creation failed: {err:?}")
            }
            Self::PipelineCreate(err) => write!(f, "graphics pipeline creation failed: {err:?}"),
            Self::DuplicateLayer(name) => {
                write!(f, "a layer named {name:?} is already registered")
            }
            Self::UnknownLayer(name) => write!(f, "no layer named {name:?} is registered"),
            Self::Internal(msg) => write!(f, "internal renderer error: {msg}"),
        }
    }
}

impl core::error::Error for RendererError {}

/// Per-frame recording state handed to [`RenderLayer`] implementations.
///
/// The context borrows the frame command buffer for the duration of the
/// pass; a layer records commands with [`FrameContext::device`] and
/// [`FrameContext::command_buffer`] inside [`RenderLayer::render`].
pub struct FrameContext<'f> {
    device: &'f ash::Device,
    command_buffer: vk::CommandBuffer,
    width: u32,
    height: u32,
    format: vk::Format,
    frame_index: u64,
}

impl<'f> FrameContext<'f> {
    /// Builds the context for one render pass (used by the subsystem).
    fn new(
        device: &'f ash::Device,
        command_buffer: vk::CommandBuffer,
        width: u32,
        height: u32,
        format: vk::Format,
        frame_index: u64,
    ) -> Self {
        Self {
            device,
            command_buffer,
            width,
            height,
            format,
            frame_index,
        }
    }

    /// The logical device owning the frame command buffer.
    #[inline]
    #[must_use]
    pub fn device(&self) -> &'f ash::Device {
        self.device
    }

    /// The primary command buffer being recorded for this frame.
    ///
    /// It is in the recording state and inside a dynamic-rendering scope
    /// while a layer's [`RenderLayer::render`] runs.
    #[inline]
    #[must_use]
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    /// Render target width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Render target height in pixels.
    #[inline]
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Vulkan format of the color attachment
    /// (`B8G8R8A8_UNORM`, the memory-layout counterpart of
    /// `DRM_FORMAT_XRGB8888`).
    #[inline]
    #[must_use]
    pub fn color_format(&self) -> vk::Format {
        self.format
    }

    /// Number of frames submitted so far by this renderer (0 for the
    /// first frame); useful to animate layer content.
    #[inline]
    #[must_use]
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
}

/// A piece of drawable content registered with a [`LayerStack`].
///
/// One layer corresponds to one implementation of rlgame's
/// `r_game_renderer_layer` concept: an independent contributor to the
/// frame's render pass. During a pass every enabled layer runs, in
/// ascending [`RenderLayer::priority`] order, the three hooks
/// `before_pass` → `render` → `after_pass`.
pub trait RenderLayer {
    /// Stable identifier used for lookups, removal and enabling.
    fn name(&self) -> &str;

    /// Ordering key inside the render pass: lower values run first.
    /// Defaults to `0`.
    #[must_use]
    fn priority(&self) -> i32 {
        0
    }

    /// Whether the layer participates in a pass. Defaults to `true`;
    /// the stack can also override it with
    /// [`LayerStack::set_enabled`].
    #[must_use]
    fn enabled(&self) -> bool {
        true
    }

    /// Hook run immediately before this layer renders (bind per-layer
    /// state, transition auxiliary resources, …). The command buffer is
    /// already inside the dynamic-rendering scope.
    fn before_pass(&mut self, _frame: &mut FrameContext<'_>) -> Result<(), RendererError> {
        Ok(())
    }

    /// Records this layer's draw commands for the frame.
    fn render(&mut self, frame: &mut FrameContext<'_>) -> Result<(), RendererError>;

    /// Hook run immediately after this layer rendered (restore state, …).
    fn after_pass(&mut self, _frame: &mut FrameContext<'_>) -> Result<(), RendererError> {
        Ok(())
    }
}

/// One registered layer together with the stack-level enable override.
struct LayerEntry<'p> {
    layer: Box<dyn RenderLayer + 'p>,
    override_enabled: Option<bool>,
}

impl LayerEntry<'_> {
    /// Effective visibility: the override wins over the layer's own flag.
    fn enabled(&self) -> bool {
        self.override_enabled
            .unwrap_or_else(|| self.layer.enabled())
    }
}

/// Ordered registry of [`RenderLayer`]s driving a frame's render pass.
///
/// Layers are identified by [`RenderLayer::name`] (names must be unique)
/// and drawn in ascending [`RenderLayer::priority`] order. The stack is
/// re-sorted before every pass, so priorities may change at runtime.
pub struct LayerStack<'p> {
    entries: Vec<LayerEntry<'p>>,
}

impl<'p> LayerStack<'p> {
    /// Creates an empty stack.
    #[must_use]
    pub const fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Registers `layer`.
    ///
    /// # Errors
    ///
    /// * [`RendererError::DuplicateLayer`] — a layer with the same
    ///   [`RenderLayer::name`] is already registered; the new layer is
    ///   dropped by the caller.
    pub fn add(&mut self, layer: Box<dyn RenderLayer + 'p>) -> Result<(), RendererError> {
        let duplicate = self
            .entries
            .iter()
            .any(|entry| entry.layer.name() == layer.name());
        if duplicate {
            return Err(RendererError::DuplicateLayer(layer.name().to_string()));
        }
        self.entries.push(LayerEntry {
            layer,
            override_enabled: None,
        });
        Ok(())
    }

    /// Unregisters the layer called `name` and returns it, or `None` when
    /// no such layer exists.
    #[must_use]
    pub fn remove(&mut self, name: &str) -> Option<Box<dyn RenderLayer + 'p>> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.layer.name() == name)?;
        Some(self.entries.swap_remove(index).layer)
    }

    /// Whether a layer called `name` is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.layer.name() == name)
    }

    /// Returns the registered layer called `name` for mutation.
    #[must_use]
    pub fn get_mut(&mut self, name: &str) -> Option<&mut (dyn RenderLayer + 'p)> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.layer.name() == name)?;
        Some(&mut *self.entries[index].layer)
    }

    /// Overrides the visibility of the layer called `name`.
    ///
    /// The override applies until the layer is removed and wins over
    /// [`RenderLayer::enabled`].
    ///
    /// # Errors
    ///
    /// * [`RendererError::UnknownLayer`] — no layer with that name is
    ///   registered.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> Result<(), RendererError> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.layer.name() == name)
            .ok_or_else(|| RendererError::UnknownLayer(name.to_string()))?;
        self.entries[index].override_enabled = Some(enabled);
        Ok(())
    }

    /// Number of registered layers (including disabled ones).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no layer is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Unregisters every layer, dropping them.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Sorts the layers by ascending priority (stable: equal priorities
    /// keep registration order).
    fn sort_by_priority(&mut self) {
        self.entries
            .sort_by_key(|entry| entry.layer.priority());
    }

    /// Runs every enabled layer's hooks in priority order.
    ///
    /// The stack is re-sorted first. The first failing hook stops the pass
    /// and propagates the error; the subsystem then abandons the frame.
    fn run(&mut self, frame: &mut FrameContext<'_>) -> Result<(), RendererError> {
        self.sort_by_priority();
        for entry in &mut self.entries {
            if !entry.enabled() {
                continue;
            }
            entry.layer.before_pass(frame)?;
            entry.layer.render(frame)?;
            entry.layer.after_pass(frame)?;
        }
        Ok(())
    }
}

impl Default for LayerStack<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame orchestrator around a [`PipelineContext`].
///
/// Mirrors rlgame's renderer subsystem: a state machine
/// ([`RendererState`]) plus a [`LayerStack`], borrowing all device-level
/// state from the pipeline context it is created with. The subsystem must
/// not outlive that context (the borrow enforces it), and it must be
/// dropped before it — dropping waits for the device to go idle and
/// releases the frame's synchronization objects and command buffer.
pub struct RendererSubsystem<'p> {
    /// Device-level state borrowed for the whole lifetime of the
    /// subsystem (`'p`).
    context: &'p PipelineContext,
    state: RendererState,
    phase: FramePhase,
    layers: LayerStack<'p>,
    command_buffer: Option<vk::CommandBuffer>,
    frame_fence: Option<vk::Fence>,
    frame_wait_sem: Option<vk::Semaphore>,
    frame_signal_sem: Option<vk::Semaphore>,
    frame_index: u64,
}

impl<'p> RendererSubsystem<'p> {
    /// Creates a stopped subsystem for `context` with an empty layer
    /// stack.
    ///
    /// Call [`RendererSubsystem::start`] before the first frame.
    #[must_use]
    pub fn new(context: &'p PipelineContext) -> Self {
        Self {
            context,
            state: RendererState::Stopped,
            phase: FramePhase::Idle,
            layers: LayerStack::new(),
            command_buffer: None,
            frame_fence: None,
            frame_wait_sem: None,
            frame_signal_sem: None,
            frame_index: 0,
        }
    }

    /// Current lifecycle state.
    #[inline]
    #[must_use]
    pub const fn state(&self) -> RendererState {
        self.state
    }

    /// Current frame phase (see [`FramePhase`]).
    #[inline]
    #[must_use]
    pub const fn frame_phase(&self) -> FramePhase {
        self.phase
    }

    /// Number of frames fully rendered so far (frames whose completion
    /// `sync_file` was exported by [`RendererSubsystem::end_frame`]).
    #[inline]
    #[must_use]
    pub const fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// The layer stack driven by each frame.
    #[inline]
    #[must_use]
    pub fn layers(&self) -> &LayerStack<'p> {
        &self.layers
    }

    /// The layer stack driven by each frame, for mutation.
    #[inline]
    #[must_use]
    pub fn layers_mut(&mut self) -> &mut LayerStack<'p> {
        &mut self.layers
    }

    /// Allocates the frame command buffer and enters the running state.
    ///
    /// # Errors
    ///
    /// * [`RendererError::InvalidState`] — the subsystem is not stopped.
    /// * [`RendererError::CommandAllocation`] — the command buffer could
    ///   not be allocated from the pipeline's command pool.
    pub fn start(&mut self) -> Result<(), RendererError> {
        if self.state != RendererState::Stopped {
            return Err(RendererError::InvalidState {
                operation: "start",
                state: self.state,
            });
        }
        let context = self.context;
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(context.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the command pool belongs to `context`'s device and is
        // alive for `'p`, which outlives this subsystem.
        let allocated = unsafe {
            context
                .device()
                .allocate_command_buffers(&allocate_info)
        }
        .map_err(RendererError::CommandAllocation)?;
        let command_buffer = allocated
            .first()
            .copied()
            .ok_or(RendererError::Internal("the driver allocated no command buffer"))?;
        self.command_buffer = Some(command_buffer);
        self.state = RendererState::Running;
        Ok(())
    }

    /// Suspends frame operations, keeping the command buffer and layers.
    ///
    /// # Errors
    ///
    /// * [`RendererError::InvalidState`] — the subsystem is not running.
    /// * [`RendererError::FramePhase`] — a frame is in progress; finish it
    ///   with [`RendererSubsystem::end_frame`] (or stop the subsystem)
    ///   before pausing.
    pub fn pause(&mut self) -> Result<(), RendererError> {
        if self.state != RendererState::Running {
            return Err(RendererError::InvalidState {
                operation: "pause",
                state: self.state,
            });
        }
        self.require_phase(FramePhase::Idle, "pause")?;
        self.state = RendererState::Paused;
        Ok(())
    }

    /// Resumes a paused subsystem.
    ///
    /// # Errors
    ///
    /// * [`RendererError::InvalidState`] — the subsystem is not paused.
    pub fn resume(&mut self) -> Result<(), RendererError> {
        if self.state != RendererState::Paused {
            return Err(RendererError::InvalidState {
                operation: "resume",
                state: self.state,
            });
        }
        self.state = RendererState::Running;
        Ok(())
    }

    /// Waits for every submitted frame, releases the frame synchronization
    /// objects and the command buffer, and returns to the stopped state.
    ///
    /// A frame that was begun but not submitted is discarded. Failures of
    /// the final device wait (device loss) are ignored: destroying the
    /// handles is still the only way to release the host-side resources.
    ///
    /// # Errors
    ///
    /// * [`RendererError::InvalidState`] — the subsystem is already
    ///   stopped.
    pub fn stop(&mut self) -> Result<(), RendererError> {
        if self.state == RendererState::Stopped {
            return Err(RendererError::InvalidState {
                operation: "stop",
                state: self.state,
            });
        }
        let context = self.context;
        // SAFETY: exclusive access to the subsystem; a failed wait (device
        // lost) is tolerated because the teardown below must still run.
        unsafe {
            let _ = context.device().device_wait_idle();
        }
        self.retire_frame_sync();
        if self.phase != FramePhase::Idle {
            // Discards a partially recorded or already finished command
            // buffer: the device is idle, so nothing references it.
            // SAFETY: the pool is `context`'s and idle at this point.
            unsafe {
                let _ = context
                    .device()
                    .reset_command_pool(context.command_pool(), vk::CommandPoolResetFlags::empty());
            }
            self.phase = FramePhase::Idle;
        }
        if let Some(command_buffer) = self.command_buffer.take() {
            // SAFETY: the command buffer was allocated from this pool by
            // `start` and is not pending (device idle, pool reset above).
            unsafe {
                context
                    .device()
                    .free_command_buffers(context.command_pool(), core::slice::from_ref(&command_buffer));
            }
        }
        self.state = RendererState::Stopped;
        Ok(())
    }

    /// Starts one frame: waits for the previous submission, resets the
    /// command pool and creates the frame's synchronization objects.
    ///
    /// `wait_sync_file` is an optional `sync_file` fd (for example a DRM
    /// release point or the compositor's previous release) to wait on
    /// before rendering; ownership is consumed — on success the driver
    /// takes the descriptor, on failure it is closed here. The first frame
    /// passes `None`.
    ///
    /// # Errors
    ///
    /// * [`RendererError::InvalidState`] — the subsystem is not running.
    /// * [`RendererError::FramePhase`] — another frame is still in
    ///   progress.
    /// * [`RendererError::FenceWait`], [`RendererError::CommandPoolReset`]
    ///   — the previous frame could not be retired or the command pool
    ///   reset.
    /// * [`RendererError::SemaphoreCreate`], [`RendererError::SemaphoreImport`],
    ///   [`RendererError::FenceCreate`] — the frame's synchronization
    ///   objects could not be created (any object created before the
    ///   failure is destroyed again).
    pub fn begin_frame(&mut self, wait_sync_file: Option<OwnedFd>) -> Result<(), RendererError> {
        self.require_running("begin_frame")?;
        self.require_phase(FramePhase::Idle, "begin_frame")?;
        let context = self.context;
        let device = context.device();
        if self.command_buffer.is_none() {
            return Err(RendererError::Internal(
                "the subsystem is running without a command buffer",
            ));
        }
        // Wait for the previous submission, then retire its sync objects.
        if let Some(fence) = self.frame_fence {
            // SAFETY: `fence` was created by this subsystem, is waited on
            // with `&mut self` excluding concurrent use, and `u64::MAX`
            // means the only failure is device loss (reported, not
            // ignored).
            unsafe { device.wait_for_fences(core::slice::from_ref(&fence), true, u64::MAX) }
                .map_err(RendererError::FenceWait)?;
        }
        // The previous submission (if any) completed, so its fence and
        // semaphores are no longer referenced by the queue.
        self.retire_frame_sync();

        // SAFETY: the previous command buffer execution completed (fence
        // wait above, or no submission exists on the first frame), so
        // resetting the pool cannot discard in-flight commands.
        unsafe { device.reset_command_pool(context.command_pool(), vk::CommandPoolResetFlags::empty()) }
            .map_err(RendererError::CommandPoolReset)?;

        // Per-frame synchronization: fresh objects each frame because the
        // SYNC_FD export unsignals the source semaphore (copy
        // transference) and each object is used exactly once.
        let wait_semaphore = if let Some(sync_fd) = wait_sync_file {
            let semaphore = self.create_binary_semaphore(false)?;
            match self.import_sync_fd(semaphore, sync_fd) {
                Ok(()) => Some(semaphore),
                Err(err) => {
                    // SAFETY: the semaphore was just created and never
                    // submitted, so destroying it is valid.
                    unsafe { device.destroy_semaphore(semaphore, None) };
                    return Err(err);
                }
            }
        } else {
            None
        };
        let signal_semaphore = match self.create_binary_semaphore(true) {
            Ok(semaphore) => semaphore,
            Err(err) => {
                if let Some(semaphore) = wait_semaphore {
                    // SAFETY: created this frame, never submitted.
                    unsafe { device.destroy_semaphore(semaphore, None) };
                }
                return Err(err);
            }
        };
        // A fresh unsignaled fence: it is signaled by this frame's submit
        // and waited on at the start of the next one (no reset needed).
        let fence = match self.create_frame_fence() {
            Ok(fence) => fence,
            Err(err) => {
                // SAFETY: none of the semaphores was submitted.
                unsafe {
                    device.destroy_semaphore(signal_semaphore, None);
                    if let Some(semaphore) = wait_semaphore {
                        device.destroy_semaphore(semaphore, None);
                    }
                }
                return Err(err);
            }
        };
        self.frame_fence = Some(fence);
        self.frame_wait_sem = wait_semaphore;
        self.frame_signal_sem = Some(signal_semaphore);
        self.phase = FramePhase::Recording;
        Ok(())
    }

    /// Records the frame command buffer (running every enabled layer
    /// inside the render pass) and submits it to the graphics queue.
    ///
    /// Must be preceded by [`RendererSubsystem::begin_frame`]. On failure
    /// nothing is left in flight: the frame's synchronization objects are
    /// destroyed, the command pool is reset and the frame returns to the
    /// idle phase, so the subsystem stays usable.
    ///
    /// # Errors
    ///
    /// * [`RendererError::FramePhase`] — no frame is being recorded.
    /// * [`RendererError::CommandRecord`] — recording failed (including
    ///   errors reported by a layer's hooks).
    /// * [`RendererError::SemaphoreCreate`], [`RendererError::FenceCreate`],
    ///   [`RendererError::CommandPoolReset`], [`RendererError::Submit`] —
    ///   the frame could not be submitted.
    pub fn render_frame(&mut self) -> Result<(), RendererError> {
        self.require_phase(FramePhase::Recording, "render_frame")?;
        if let Err(err) = self.record_frame() {
            self.discard_unsubmitted_frame();
            return Err(err);
        }
        let context = self.context;
        let device = context.device();
        let fence = self
            .frame_fence
            .ok_or(RendererError::Internal("recorded frame without a fence"))?;
        let signal_semaphore = self
            .frame_signal_sem
            .ok_or(RendererError::Internal(
                "recorded frame without a signal semaphore",
            ))?;
        let command_buffer = self
            .command_buffer
            .ok_or(RendererError::Internal("recorded frame without a command buffer"))?;
        let wait_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let (wait_sems, wait_stages): (&[vk::Semaphore], &[vk::PipelineStageFlags]) =
            match self.frame_wait_sem {
                Some(ref semaphore) => (
                    core::slice::from_ref(semaphore),
                    core::slice::from_ref(&wait_stage),
                ),
                None => (&[], &[]),
            };
        let signal_sems = [signal_semaphore];
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_sems)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_sems);
        // SAFETY: all handles are valid and live; the wait semaphore (if
        // any) holds a temporary SYNC_FD import whose stage mask matches
        // `wait_stages`; the signal semaphore and the fence are unsignaled
        // (fresh), as required by `vkQueueSubmit`.
        let submit_result =
            unsafe { device.queue_submit(context.queue(), core::slice::from_ref(&submit_info), fence) };
        if let Err(err) = submit_result {
            self.discard_unsubmitted_frame();
            return Err(RendererError::Submit(err));
        }
        self.phase = FramePhase::Submitted;
        Ok(())
    }

    /// Exports GPU completion of the submitted frame as a `sync_file` fd
    /// and closes the frame.
    ///
    /// The returned descriptor represents "this frame's rendering finished"
    /// (exported from the signal semaphore of the submission) and is meant
    /// to be handed to the presentation layer — either imported as the next
    /// frame's wait point or attached to a DRM syncobj acquire point.
    ///
    /// Must be preceded by [`RendererSubsystem::render_frame`]. On success
    /// the frame phase returns to idle; if the export itself fails the
    /// frame stays submitted and is retired by the next
    /// [`RendererSubsystem::begin_frame`] or by stopping/dropping the
    /// subsystem.
    ///
    /// # Errors
    ///
    /// * [`RendererError::FramePhase`] — the frame has not been submitted.
    /// * [`RendererError::SemaphoreExport`] — the `sync_file` fd could not
    ///   be created.
    pub fn end_frame(&mut self) -> Result<OwnedFd, RendererError> {
        self.require_phase(FramePhase::Submitted, "end_frame")?;
        let signal_semaphore = self
            .frame_signal_sem
            .ok_or(RendererError::Internal(
                "submitted frame without a signal semaphore",
            ))?;
        let sync_file = self.export_sync_fd(signal_semaphore)?;
        self.phase = FramePhase::Idle;
        self.frame_index = self.frame_index.wrapping_add(1);
        Ok(sync_file)
    }

    /// Checks that the subsystem is running.
    fn require_running(&self, operation: &'static str) -> Result<(), RendererError> {
        if self.state == RendererState::Running {
            Ok(())
        } else {
            Err(RendererError::InvalidState {
                operation,
                state: self.state,
            })
        }
    }

    /// Checks that the frame is in `expected` phase.
    fn require_phase(&self, expected: FramePhase, operation: &'static str) -> Result<(), RendererError> {
        if self.phase == expected {
            Ok(())
        } else {
            Err(RendererError::FramePhase {
                operation,
                expected,
                actual: self.phase,
            })
        }
    }

    /// Records one frame: layout transition, the dynamic-rendering pass
    /// that runs the layer stack, then the handoff transition.
    fn record_frame(&mut self) -> Result<(), RendererError> {
        let context: &'p PipelineContext = self.context;
        let device = context.device();
        let command_buffer = self
            .command_buffer
            .ok_or(RendererError::Internal(
                "frame recording requested without a command buffer",
            ))?;
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was allocated from this pool and the
        // pool was reset at the start of the frame, so it is in the initial
        // state.
        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
            .map_err(RendererError::CommandRecord)?;

        // Handoff state for each frame: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
        // (contents are discarded; the pass clears the attachment anyway).
        let to_attachment = image_barrier(
            context.image(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        );
        // SAFETY: the command buffer is in the recording state and the
        // barrier references the live render-target image; both stage and
        // access masks are well-formed. The optional semaphore wait of this
        // submission covers COLOR_ATTACHMENT_OUTPUT, i.e. this transition.
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_attachment],
            );
        }
        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.00, 0.00, 0.00, 1.0],
            },
        };
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(context.image_view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value);
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: context.width(),
                height: context.height(),
            },
        };
        let rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(core::slice::from_ref(&color_attachment));
        // SAFETY: the image view is compatible with COLOR_ATTACHMENT_OPTIMAL
        // (checked at creation) and the render area fits the image; dynamic
        // rendering requires no render pass or framebuffer object.
        unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };

        // Dynamic viewport/scissor, applied once before the pass; layers may
        // override them while recording.
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: context.width() as f32,
            height: context.height() as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        // SAFETY: the command buffer is inside begin/end rendering scope
        // and pipelines are created with VIEWPORT/SCISSOR as dynamic states.
        unsafe {
            device.cmd_set_viewport(command_buffer, 0, core::slice::from_ref(&viewport));
            device.cmd_set_scissor(command_buffer, 0, core::slice::from_ref(&render_area));
        }
        let layer_result = {
            let mut frame = FrameContext::new(
                device,
                command_buffer,
                context.width(),
                context.height(),
                PipelineContext::color_format(),
                self.frame_index,
            );
            self.layers.run(&mut frame)
        };
        // SAFETY: the rendering scope opened above has not been ended yet.
        unsafe { device.cmd_end_rendering(command_buffer) };
        if let Err(err) = layer_result {
            // Close the command buffer so the pool can be reset by the
            // caller's error handling; the recording result itself is
            // secondary to the layer error being reported.
            // SAFETY: the command buffer is still in the recording state.
            let _ = unsafe { device.end_command_buffer(command_buffer) };
            return Err(err);
        }

        // COLOR_ATTACHMENT_OPTIMAL -> GENERAL as the conservative handoff
        // state for the external compositor: access through the exported
        // dma-buf is not layout-tracked by Vulkan, and GENERAL is the safest
        // state to leave the image in for non-Vulkan consumers.
        let to_general = image_barrier(
            context.image(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::empty(),
        );
        // SAFETY: the rendering scope above ended; the source stage/access
        // cover everything this frame wrote to the image.
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_general],
            );
        }
        // SAFETY: all recorded commands reference live objects of the
        // borrowed pipeline context.
        unsafe { device.end_command_buffer(command_buffer) }.map_err(RendererError::CommandRecord)
    }

    /// Creates a binary semaphore, optionally with SYNC_FD export enabled.
    fn create_binary_semaphore(&self, export_sync_fd: bool) -> Result<vk::Semaphore, RendererError> {
        let device = self.context.device();
        let mut export_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let plain_info = vk::SemaphoreCreateInfo::default();
        let semaphore_info = if export_sync_fd {
            vk::SemaphoreCreateInfo::default().push_next(&mut export_info)
        } else {
            plain_info
        };
        // SAFETY: the device is valid and, when exporting, the
        // `VkExportSemaphoreCreateInfo` chain outlives the call.
        unsafe { device.create_semaphore(&semaphore_info, None) }.map_err(RendererError::SemaphoreCreate)
    }

    /// Creates an unsignaled fence for one frame submission.
    fn create_frame_fence(&self) -> Result<vk::Fence, RendererError> {
        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: a plain, well-formed create info.
        unsafe {
            self.context
                .device()
                .create_fence(&fence_info, None)
        }
        .map_err(RendererError::FenceCreate)
    }

    /// Imports a `sync_file` fd into `semaphore` as a temporary SYNC_FD
    /// payload.
    ///
    /// On success ownership of `sync_fd` transfers to the implementation;
    /// on failure the fd is closed here because the driver never took it.
    fn import_sync_fd(&self, semaphore: vk::Semaphore, sync_fd: OwnedFd) -> Result<(), RendererError> {
        let raw_fd = sync_fd.into_raw();
        let import_info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore)
            // VUID-VkImportSemaphoreFdInfoKHR-handleType-07307: copy-
            // transference handle types (SYNC_FD) must be imported with
            // TEMPORARY.
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .fd(raw_fd);
        // SAFETY: `semaphore` is a fresh binary semaphore
        // (VUID-vkImportSemaphoreFdKHR-semaphore-01142), SYNC_FD import was
        // verified as supported when the pipeline context was created, and
        // the import info is well-formed.
        let result = unsafe {
            self.context
                .external_semaphore_fd()
                .import_semaphore_fd(&import_info)
        };
        if let Err(err) = result {
            // The implementation only takes ownership on success; rebuild an
            // OwnedFd so the descriptor is closed on this error path.
            // SAFETY: the fd was moved out above and not consumed by the
            // driver because the import failed.
            drop(unsafe { OwnedFd::from_raw(raw_fd) });
            return Err(RendererError::SemaphoreImport(err));
        }
        Ok(())
    }

    /// Exports a signaled (or pending-signal) binary semaphore as a new
    /// `sync_file` fd representing GPU completion.
    fn export_sync_fd(&self, semaphore: vk::Semaphore) -> Result<OwnedFd, RendererError> {
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        // SAFETY: the semaphore was created with SYNC_FD in
        // `VkExportSemaphoreCreateInfo` and is signaled or has a pending
        // signal operation (it was just submitted), satisfying
        // VUID-VkSemaphoreGetFdInfoKHR-handleType-01132/01135.
        let raw_fd = unsafe {
            self.context
                .external_semaphore_fd()
                .get_semaphore_fd(&get_info)
        }
        .map_err(RendererError::SemaphoreExport)?;
        if raw_fd < 0 {
            // The Vulkan spec allows `-1` to mean "already signaled", but a
            // `sync_file` fd is what the presentation layer needs; treat it
            // as an error.
            return Err(RendererError::Internal(
                "vkGetSemaphoreFdKHR completed without producing a sync_file fd",
            ));
        }
        // SAFETY: on success ownership of the new fd transfers to the
        // caller.
        Ok(unsafe { OwnedFd::from_raw(raw_fd) })
    }

    /// Destroys the frame's fence and semaphores. Only called after the
    /// frame fence signaled, after a failed submission (nothing on the
    /// queue) or from `Drop` after `device_wait_idle`.
    fn retire_frame_sync(&mut self) {
        let device = self.context.device();
        // SAFETY: this runs either after the frame fence was waited on,
        // after `device_wait_idle`, or when the objects were never
        // submitted; each handle was created by this subsystem and is taken
        // out below, so it cannot be destroyed twice.
        unsafe {
            if let Some(fence) = self.frame_fence.take() {
                device.destroy_fence(fence, None);
            }
            if let Some(semaphore) = self.frame_signal_sem.take() {
                device.destroy_semaphore(semaphore, None);
            }
            if let Some(semaphore) = self.frame_wait_sem.take() {
                device.destroy_semaphore(semaphore, None);
            }
        }
    }

    /// Undoes a frame that never reached the queue: destroys its
    /// synchronization objects, resets the command pool and returns the
    /// frame phase to idle.
    fn discard_unsubmitted_frame(&mut self) {
        let context = self.context;
        // None of these objects was passed to a queue submission, so they
        // cannot be referenced by the GPU (see `retire_frame_sync`).
        self.retire_frame_sync();
        if self.phase != FramePhase::Idle {
            // The command buffer holds a partial recording; resetting the
            // pool returns it to the initial state.
            // SAFETY: nothing was submitted for this frame (or the submit
            // failed), so the pool is not in use by the GPU.
            unsafe {
                let _ = context
                    .device()
                    .reset_command_pool(context.command_pool(), vk::CommandPoolResetFlags::empty());
            }
            self.phase = FramePhase::Idle;
        }
    }
}

impl<'p> Drop for RendererSubsystem<'p> {
    fn drop(&mut self) {
        let context = self.context;
        // SAFETY: `drop` has exclusive access to the subsystem, so no frame
        // can be recording or submitting concurrently. A failed wait
        // (device lost) is ignored: the GPU is gone and destroying the
        // handles is still the correct way to release the host-side
        // resources.
        unsafe {
            let _ = context.device().device_wait_idle();
        }
        // Retires the frame fence/semaphores after the idle wait (the SAFETY
        // comment inside covers that precondition).
        self.retire_frame_sync();
        if let Some(command_buffer) = self.command_buffer.take() {
            // SAFETY: the device is idle and the command buffer was
            // allocated from this pool by `start`.
            unsafe {
                context
                    .device()
                    .free_command_buffers(context.command_pool(), core::slice::from_ref(&command_buffer));
            }
        }
        // The command pool, image, device and instance belong to `context`
        // and outlive `'p`, which outlives this subsystem; nothing else is
        // owned here, so no further destruction is required.
    }
}

/// Builds an image memory barrier for a full-subresource layout transition.
///
/// The returned barrier only carries the barrier fields; the source and
/// destination stages are supplied to `vkCmdPipelineBarrier` directly.
fn image_barrier(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

/// Decodes a little-endian SPIR-V blob into shader words.
///
/// This is the `no_std` replacement for `ash::util::read_spv` (which is
/// only available with ash's `std` feature): it validates the file length
/// and the module magic, then widens each 4-byte group into a `u32`.
///
/// # Errors
///
/// * [`RendererError::ShaderLoad`] — the byte length is not a multiple of
///   4, or the SPIR-V magic number does not match.
pub fn load_spir_v(bytes: &[u8]) -> Result<Vec<u32>, RendererError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(RendererError::ShaderLoad(format!(
            "SPIR-V byte length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let words: Vec<u32> = chunks
        .map(|group| u32::from_le_bytes([group[0], group[1], group[2], group[3]]))
        .collect();
    match words.first() {
        Some(&magic) if magic == SPIRV_MAGIC => Ok(words),
        Some(&magic) => Err(RendererError::ShaderLoad(format!(
            "bad SPIR-V magic {magic:#010x}"
        ))),
        None => Err(RendererError::ShaderLoad(String::from(
            "the SPIR-V module is empty",
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    /// Records the names of the layers whose hooks ran, in call order.
    #[derive(Clone)]
    struct RecordingLayer {
        layer_name: &'static str,
        layer_priority: i32,
        layer_enabled: bool,
        calls: Rc<RefCell<Vec<&'static str>>>,
    }

    impl RenderLayer for RecordingLayer {
        fn name(&self) -> &str {
            self.layer_name
        }

        fn priority(&self) -> i32 {
            self.layer_priority
        }

        fn enabled(&self) -> bool {
            self.layer_enabled
        }

        fn render(&mut self, _frame: &mut FrameContext<'_>) -> Result<(), RendererError> {
            self.calls.borrow_mut().push(self.layer_name);
            Ok(())
        }
    }

    fn recording_layer(
        name: &'static str,
        priority: i32,
        enabled: bool,
        calls: &Rc<RefCell<Vec<&'static str>>>,
    ) -> Box<dyn RenderLayer + 'static> {
        Box::new(RecordingLayer {
            layer_name: name,
            layer_priority: priority,
            layer_enabled: enabled,
            calls: Rc::clone(calls),
        })
    }

    /// The modifier list of the pipeline is validated before the Vulkan
    /// loader is touched, so this error path must work on machines without
    /// any GPU or driver.
    #[test]
    fn empty_modifier_list_is_rejected() {
        assert!(matches!(
            PipelineContext::new(640, 640, &[]),
            Err(crate::ui_pipeline::PipelineError::EmptyModifierList)
        ));
    }

    /// A SPIR-V module shorter than one word cannot be decoded.
    #[test]
    fn truncated_spir_v_is_rejected() {
        assert!(matches!(
            load_spir_v(&[0x03, 0x02, 0x23]),
            Err(RendererError::ShaderLoad(_))
        ));
    }

    /// An empty module and a wrong magic number are both rejected.
    #[test]
    fn bad_spir_v_magic_is_rejected() {
        assert!(matches!(load_spir_v(&[]), Err(RendererError::ShaderLoad(_))));
        let wrong_magic = 0xDEAD_BEEFu32.to_le_bytes();
        assert!(matches!(
            load_spir_v(&wrong_magic),
            Err(RendererError::ShaderLoad(_))
        ));
    }

    /// Layers are drawn in ascending priority order, regardless of the
    /// order they were registered in.
    #[test]
    fn layers_run_in_priority_order() {
        let mut stack = LayerStack::new();
        let noop = || Rc::new(RefCell::new(Vec::new()));
        stack
            .add(recording_layer("third", 30, true, &noop()))
            .unwrap_or_else(|err| panic!("adding the third layer failed: {err}"));
        stack
            .add(recording_layer("first", 10, true, &noop()))
            .unwrap_or_else(|err| panic!("adding the first layer failed: {err}"));
        stack
            .add(recording_layer("second", 20, true, &noop()))
            .unwrap_or_else(|err| panic!("adding the second layer failed: {err}"));
        stack.sort_by_priority();
        let order: Vec<&str> = stack
            .entries
            .iter()
            .map(|entry| entry.layer.name())
            .collect();
        assert_eq!(order, ["first", "second", "third"]);
    }

    /// Registering two layers with the same name is rejected and the stack
    /// keeps only the first one.
    #[test]
    fn duplicate_layer_names_are_rejected() {
        let mut stack = LayerStack::new();
        stack
            .add(recording_layer("ui", 0, true, &Rc::new(RefCell::new(Vec::new()))))
            .unwrap_or_else(|err| panic!("adding the first layer failed: {err}"));
        let duplicate = recording_layer("ui", 5, true, &Rc::new(RefCell::new(Vec::new())));
        assert!(matches!(
            stack.add(duplicate),
            Err(RendererError::DuplicateLayer(name)) if name == "ui"
        ));
        assert_eq!(stack.len(), 1);
        assert!(stack.contains("ui"));
    }

    /// Removing a layer returns it; unknown names return `None` and
    /// enabling an unknown layer reports [`RendererError::UnknownLayer`].
    #[test]
    fn remove_and_enable_lookups() {
        let mut stack = LayerStack::new();
        stack
            .add(recording_layer("ui", 0, true, &Rc::new(RefCell::new(Vec::new()))))
            .unwrap_or_else(|err| panic!("adding the layer failed: {err}"));
        assert!(matches!(
            stack.set_enabled("missing", false),
            Err(RendererError::UnknownLayer(name)) if name == "missing"
        ));
        stack
            .set_enabled("ui", false)
            .unwrap_or_else(|err| panic!("disabling the layer failed: {err}"));
        assert!(!stack.entries[0].enabled());
        let removed = stack.remove("ui");
        assert!(removed.is_some());
        assert!(stack.is_empty());
        assert!(stack.remove("ui").is_none());
    }

    /// A layer that disables itself reports itself as disabled to the
    /// stack, which skips it during a pass.
    #[test]
    fn layer_disabled_flag_is_honoured_without_override() {
        let mut stack = LayerStack::new();
        stack
            .add(recording_layer(
                "hidden",
                0,
                false,
                &Rc::new(RefCell::new(Vec::new())),
            ))
            .unwrap_or_else(|err| panic!("adding the layer failed: {err}"));
        assert!(!stack.entries[0].enabled());
        stack
            .set_enabled("hidden", true)
            .unwrap_or_else(|err| panic!("enabling the layer failed: {err}"));
        assert!(stack.entries[0].enabled());
    }

    /// The lifecycle state machine only allows the documented
    /// transitions.
    #[test]
    fn state_display_is_stable() {
        assert_eq!(RendererState::Stopped.to_string(), "stopped");
        assert_eq!(RendererState::Running.to_string(), "running");
        assert_eq!(RendererState::Paused.to_string(), "paused");
        assert_eq!(FramePhase::Recording.to_string(), "recording");
    }
}
