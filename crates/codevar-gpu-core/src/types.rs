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

//! Lifecycle state, frame phase, error and message types shared by
//! every part of the compositor.

use crate::base::{Compositor, CompositorComm};
use alloc::boxed::Box;
use alloc::string::String;
use ash::vk;
use core::fmt;

/// Lifecycle state of a [`Compositor`](Compositor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompositorState {
    /// No frame resources exist and no frame can be begun.
    Stopped,
    /// The compositor accepts frames (see [`Compositor::begin_frame`](Compositor::begin_frame)).
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

/// Progress of the frame currently driven through [`Compositor`](Compositor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompositorPhase {
    /// No frame in progress; the next call may be
    /// [`Compositor::begin_frame`](Compositor::begin_frame).
    Idle,
    /// [`Compositor::begin_frame`](Compositor::begin_frame) succeeded: the command pool was reset
    /// and the frame's synchronization objects exist.
    Recording,
    /// The layout pass is on the queue; pipe futures are being driven.
    Pipes,
    /// Every pipe is on the queue; the mix pass is not recorded yet.
    Mixing,
    /// The mix pass is on the queue and its completion has not been
    /// exported by [`Compositor::end_frame`](Compositor::end_frame) yet.
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
            Self::Pipe { index, source } => write!(f, "pipe {index} failed: {source}"),
            Self::Internal(msg) => write!(f, "internal compositor error: {msg}"),
        }
    }
}

impl core::error::Error for CompositorError {}

/// Error reported by the non-blocking [`CompositorComm`](CompositorComm) senders.
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

/// Message exchanged between the compositor and the presentation side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositorMessage {
    /// The sender's offscreen framebuffers changed (resize, recreate);
    /// the compositor rebuilds its snapshot at the next
    /// [`Compositor::begin_frame`](Compositor::begin_frame).
    FramebuffersChanged,
    /// Override the mix weight of framebuffer `slot`.
    MixWeightChanged {
        /// Framebuffer slot index.
        slot: usize,
        /// Requested weight; sanitized with `sanitize_weight`.
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

#[cfg(test)]
mod tests {
    //! Display rendering of the public enums.

    use super::*;

    #[test]
    fn display_impls_render_expected_text() {
        assert_eq!(CompositorState::Running.to_string(), "running");
        assert_eq!(CompositorPhase::Mixing.to_string(), "mixing");
        assert_eq!(
            CompositorMessage::FrameComposited { frame_index: 2 }.to_string(),
            "frame 2 composited"
        );
        assert_eq!(CommError::Full(1u32).to_string(), "the channel is full");
    }
}
