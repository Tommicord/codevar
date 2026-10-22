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

//! The errors reported by the WebGPU backend.

use core::fmt;

/// An error reported by the WebGPU backend.
///
/// The variants carry the operation that failed and the message the
/// driver reported, so a log keeps the provenance of the failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WgError {
    /// No adapter could render on this machine.
    NoAdapter,
    /// The adapter the application asked for is not available.
    AdapterNotFound(String),
    /// The device could not be created.
    DeviceCreate(String),
    /// The surface of the canvas could not be created.
    SurfaceCreate(String),
    /// The canvas was not found in the document.
    CanvasNotFound(String),
    /// A buffer of the renderer could not be created or resized.
    Buffer(String),
    /// A shader of the renderer could not be compiled.
    Shader(String),
    /// A pipeline of the renderer could not be created.
    Pipeline(String),
    /// The font atlas could not be built.
    Atlas(String),
    /// The frame could not be submitted.
    Submit(String),
}

impl fmt::Display for WgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => f.write_str("no WebGPU adapter can render on this machine"),
            Self::AdapterNotFound(name) => write!(f, "no WebGPU adapter named {name:?} is available"),
            Self::DeviceCreate(message) => write!(f, "the WebGPU device could not be created: {message}"),
            Self::SurfaceCreate(message) => {
                write!(f, "the WebGPU surface could not be created: {message}")
            }
            Self::CanvasNotFound(id) => write!(f, "no canvas named {id:?} was found"),
            Self::Buffer(message) => write!(f, "a WebGPU buffer could not be created: {message}"),
            Self::Shader(message) => write!(f, "a WebGPU shader could not be compiled: {message}"),
            Self::Pipeline(message) => write!(f, "a WebGPU pipeline could not be created: {message}"),
            Self::Atlas(message) => write!(f, "the glyph atlas could not be built: {message}"),
            Self::Submit(message) => write!(f, "the frame could not be submitted: {message}"),
        }
    }
}

impl core::error::Error for WgError {}
