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
//!
//! The vocabulary mirrors `codevar-gpu-vk` so a caller written against
//! the desktop compositor sees the same states on the web. The Vulkan
//! variants that name a `vk::Result` are replaced by
//! [`CompositorError::Graphics`], which carries the name of the failed
//! call and the driver's own message.

use alloc::boxed::Box;
use alloc::string::String;
use core::fmt;

/// Lifecycle state of a [`Compositor`](crate::compositor::Compositor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompositorState {
    /// No frame resources exist and no frame can be begun.
    Stopped,
    /// The compositor accepts frames.
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

/// Progress of the frame currently driven through the compositor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompositorPhase {
    /// No frame in progress; the next call may be `begin_frame`.
    Idle,
    /// `begin_frame` succeeded: the frame's targets and commands exist.
    Recording,
    /// The offscreen passes are being driven.
    Pipes,
    /// Every pipe is done; the mix pass is not recorded yet.
    Mixing,
    /// The mix pass is submitted and its completion not handed off yet.
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
    /// A graphics call failed: the name of the call and the driver's own
    /// message.
    Graphics {
        /// The call that failed (for example `create_buffer`).
        operation: &'static str,
        /// The driver's error message.
        detail: String,
    },
    /// Reading or compiling an embedded shader failed.
    ShaderLoad(String),
    /// A pipe could not be started or recorded.
    Pipe {
        /// Zero-based position of the failing pipe in the queue.
        index: usize,
        /// The failure reported by the pipe or the driver.
        source: Box<CompositorError>,
    },
    /// The operation is unavailable on this target.
    ///
    /// The WebGPU backend only opens a device from a `wasm32` build; on
    /// every other target the frame lifecycle reports this instead.
    UnsupportedTarget,
    /// An internal invariant was violated; this indicates a bug in this
    /// crate.
    Internal(&'static str),
}

impl CompositorError {
    /// Builds a [`CompositorError::Graphics`] from a driver error value.
    #[must_use]
    pub fn graphics(operation: &'static str, detail: impl fmt::Display) -> Self {
        Self::Graphics {
            operation,
            detail: alloc::format!("{detail}"),
        }
    }
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
            } => write!(
                f,
                "{operation} requires the frame to be {expected} but it is {actual}"
            ),
            Self::Graphics { operation, detail } => {
                write!(f, "{operation} failed: {detail}")
            }
            Self::ShaderLoad(err) => write!(f, "failed to load an embedded shader: {err}"),
            Self::Pipe { index, source } => write!(f, "pipe {index} failed: {source}"),
            Self::UnsupportedTarget => f.write_str("the WebGPU backend only runs on wasm32"),
            Self::Internal(msg) => write!(f, "internal compositor error: {msg}"),
        }
    }
}

impl core::error::Error for CompositorError {}

/// Error reported by the non-blocking comm senders.
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
    /// The sender's offscreen framebuffers changed; the compositor
    /// rebuilds its snapshot at the next `begin_frame`.
    FramebuffersChanged,
    /// Override the mix weight of framebuffer `slot`.
    MixWeightChanged {
        /// Framebuffer slot index.
        slot: usize,
        /// Requested weight; sanitized before it reaches the shader.
        weight: f32,
    },
    /// Pause or resume pipe execution.
    VisibilityChanged {
        /// `true` to drive the pipes again.
        visible: bool,
    },
    /// A frame finished and its presentation was handed off.
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

    #[test]
    fn a_graphics_error_names_the_call() {
        let error = CompositorError::graphics("create_buffer", "out of memory");
        assert_eq!(error.to_string(), "create_buffer failed: out of memory");
    }
}
