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

//! The values every graphics backend agrees on: its identity, the
//! capabilities it reports, the description of a surface and the
//! completion handle of a frame.

use core::fmt;
use core::ops::{BitOr, BitOrAssign};

use crate::handle::OwnedFd;

/// The graphics API a backend implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// The Vulkan backend of the desktop build (`codevar-gpu-vk`).
    Vulkan,
    /// The WebGPU backend of the web build (`codevar-gpu-wg`).
    WebGpu,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan => f.write_str("Vulkan"),
            Self::WebGpu => f.write_str("WebGPU"),
        }
    }
}

/// One capability a backend may advertise.
///
/// The variants are single bits, so a set of them is a [`GpuFeatures`]
/// value. Keeping capabilities in a bit set means a struct never has to
/// carry one `bool` per capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GpuFeature {
    /// Dynamic rendering without a render-pass object.
    DynamicRendering = 1 << 0,
    /// The backend can export a render target as a dma-buf descriptor.
    DmaBufExport = 1 << 1,
    /// The backend can import and export `sync_file` semaphores.
    ExternalSemaphoreFd = 1 << 2,
    /// The backend can present an sRGB surface.
    SrgbSurface = 1 << 3,
    /// The backend can replay a display list directly on the device.
    DirectDisplayList = 1 << 4,
    /// The backend exposes a compute queue.
    Compute = 1 << 5,
}

impl GpuFeature {
    /// The bit of this capability.
    #[inline]
    #[must_use]
    pub const fn bit(self) -> u32 {
        self as u32
    }
}

/// A set of [`GpuFeature`]s.
///
/// The set is a bit mask with `BitOr`/`BitOrAssign` implementations, so
/// a backend can combine capabilities without storing a flag per
/// capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GpuFeatures(u32);

impl GpuFeatures {
    /// No capability.
    pub const NONE: Self = Self(0);

    /// Every capability known to this crate.
    pub const ALL: Self = Self((1 << 6) - 1);

    /// Builds a set from raw bits (unknown bits are preserved).
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// The raw bits of the set.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether every capability of `other` is present.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether the set is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Adds `feature` to the set.
    #[inline]
    #[must_use]
    pub const fn with(self, feature: GpuFeature) -> Self {
        Self(self.0 | feature.bit())
    }
}

impl BitOr for GpuFeatures {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for GpuFeatures {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// The fixed limits a device reports to its caller.
///
/// A backend fills the fields it can query; the remaining ones keep the
/// conservative values of [`GpuLimits::MINIMAL`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuLimits {
    /// Largest width or height of a 2D texture, in texels.
    pub max_texture_dimension_2d: u32,
    /// Largest number of vertex attributes of a pipeline.
    pub max_vertex_attributes: u32,
    /// Largest stride of a vertex buffer, in bytes.
    pub max_vertex_buffer_stride: u32,
    /// Largest number of bind groups of a pipeline.
    pub max_bind_groups: u32,
    /// Largest number of push-constant bytes, or `0` when the backend
    /// has no push constants.
    pub max_push_constant_bytes: u32,
}

impl GpuLimits {
    /// The smallest limits a conforming backend must support.
    pub const MINIMAL: Self = Self {
        max_texture_dimension_2d: 2048,
        max_vertex_attributes: 16,
        max_vertex_buffer_stride: 2048,
        max_bind_groups: 4,
        max_push_constant_bytes: 0,
    };
}

impl Default for GpuLimits {
    fn default() -> Self {
        Self::MINIMAL
    }
}

/// What a backend reports about itself: its identity, its capabilities
/// and its limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuCapabilities {
    /// The graphics API the backend implements.
    pub backend: Backend,
    /// The capabilities the backend supports.
    pub features: GpuFeatures,
    /// The fixed limits of the backend.
    pub limits: GpuLimits,
}

impl GpuCapabilities {
    /// Whether the backend supports `feature`.
    #[inline]
    #[must_use]
    pub const fn supports(&self, feature: GpuFeature) -> bool {
        self.features
            .contains(GpuFeatures::NONE.with(feature))
    }
}

/// The format of a surface's color attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceFormat {
    /// Blue, green, red and alpha, one byte each, linear.
    Bgra8Unorm,
    /// Blue, green, red and alpha, one byte each, sRGB.
    Bgra8UnormSrgb,
    /// Red, green, blue and alpha, one byte each, linear.
    Rgba8Unorm,
    /// Red, green, blue and alpha, one byte each, sRGB.
    Rgba8UnormSrgb,
}

impl SurfaceFormat {
    /// Whether the format encodes sRGB colors.
    #[inline]
    #[must_use]
    pub const fn is_srgb(self) -> bool {
        matches!(self, Self::Bgra8UnormSrgb | Self::Rgba8UnormSrgb)
    }

    /// The number of bytes one pixel occupies.
    #[inline]
    #[must_use]
    pub const fn bytes_per_pixel(self) -> u32 {
        4
    }
}

/// How a backend waits for the next frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PresentMode {
    /// Frames are presented in order on the vertical blank (vsync).
    #[default]
    Fifo,
    /// The newest frame replaces a queued frame; no tearing, no wait.
    Mailbox,
    /// A frame is presented as soon as it is ready (may tear).
    Immediate,
}

/// How the alpha channel of a surface is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AlphaMode {
    /// The surface is opaque; the alpha channel is ignored.
    #[default]
    Opaque,
    /// The color is premultiplied by the alpha channel.
    Premultiplied,
}

/// The description of a surface (and of the frames rendered into it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceConfig {
    /// The width of the surface, in device pixels.
    pub width: u32,
    /// The height of the surface, in device pixels.
    pub height: u32,
    /// The number of device pixels per logical pixel.
    pub scale: f32,
    /// The format of the color attachment.
    pub format: SurfaceFormat,
    /// How frames are presented.
    pub present_mode: PresentMode,
    /// How the alpha channel is interpreted.
    pub alpha_mode: AlphaMode,
}

impl SurfaceConfig {
    /// A configuration for a surface of `width` by `height` device
    /// pixels, with the default format and present mode.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            scale: 1.0,
            format: SurfaceFormat::Bgra8Unorm,
            present_mode: PresentMode::Fifo,
            alpha_mode: AlphaMode::Opaque,
        }
    }

    /// The configuration with a different scale factor.
    #[must_use]
    pub const fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// The configuration with a different color format.
    #[must_use]
    pub const fn with_format(mut self, format: SurfaceFormat) -> Self {
        self.format = format;
        self
    }

    /// The configuration with a different present mode.
    #[must_use]
    pub const fn with_present_mode(mut self, present_mode: PresentMode) -> Self {
        self.present_mode = present_mode;
        self
    }

    /// The configuration with a different alpha mode.
    #[must_use]
    pub const fn with_alpha_mode(mut self, alpha_mode: AlphaMode) -> Self {
        self.alpha_mode = alpha_mode;
        self
    }
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self::new(1280, 720)
    }
}

/// What a frame completion handle refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompletionKind {
    /// The backend serializes frames itself; there is no handle.
    Complete,
    /// A file descriptor that becomes readable when the frame completes.
    FileDescriptor,
    /// An opaque native handle understood by the backend that made it.
    Native,
}

/// The proof that a submitted frame finished on the device.
///
/// A Vulkan submission produces a `sync_file` descriptor, which is what
/// the Wayland presentation layer waits on. WebGPU serializes its
/// submissions itself, so it reports [`FrameCompletion::Complete`].
#[derive(Debug)]
pub enum FrameCompletion {
    /// No handle is needed: the backend's queue guarantees completion.
    Complete,
    /// A `sync_file` descriptor representing the completion of the frame.
    FileDescriptor(OwnedFd),
    /// An opaque native handle, only meaningful to the backend that
    /// produced it.
    Native(u64),
}

impl FrameCompletion {
    /// What kind of handle this completion carries.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> CompletionKind {
        match self {
            Self::Complete => CompletionKind::Complete,
            Self::FileDescriptor(_) => CompletionKind::FileDescriptor,
            Self::Native(_) => CompletionKind::Native,
        }
    }

    /// Takes the file descriptor out of a [`FrameCompletion::FileDescriptor`].
    #[must_use]
    pub fn into_file_descriptor(self) -> Option<OwnedFd> {
        match self {
            Self::FileDescriptor(fd) => Some(fd),
            Self::Complete | Self::Native(_) => None,
        }
    }
}

impl fmt::Display for CompletionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Complete => f.write_str("complete"),
            Self::FileDescriptor => f.write_str("file descriptor"),
            Self::Native => f.write_str("native handle"),
        }
    }
}

/// The error contract of a backend.
///
/// Every backend error is a `core::error::Error` that can cross a thread
/// boundary and is `'static`, which is what lets an application swap one
/// backend for another without touching its error handling.
pub trait GpuFailure: core::error::Error + Send + Sync + 'static {}

impl<T: core::error::Error + Send + Sync + 'static> GpuFailure for T {}

#[cfg(test)]
mod tests {
    //! Capability sets and surface descriptions.

    use super::*;

    #[test]
    fn features_combine_and_contain() {
        let features = GpuFeatures::NONE
            .with(GpuFeature::DynamicRendering)
            .with(GpuFeature::DmaBufExport);
        assert!(features.contains(GpuFeatures::NONE.with(GpuFeature::DynamicRendering)));
        assert!(features.contains(GpuFeatures::NONE.with(GpuFeature::DmaBufExport)));
        assert!(!features.contains(GpuFeatures::NONE.with(GpuFeature::Compute)));
        assert_eq!(
            features.bits(),
            GpuFeature::DynamicRendering.bit() | GpuFeature::DmaBufExport.bit()
        );
    }

    #[test]
    fn bitor_assign_keeps_every_feature() {
        let mut features = GpuFeatures::NONE.with(GpuFeature::SrgbSurface);
        features |= GpuFeatures::NONE.with(GpuFeature::Compute);
        assert!(features.contains(GpuFeatures::NONE.with(GpuFeature::SrgbSurface)));
        assert!(features.contains(GpuFeatures::NONE.with(GpuFeature::Compute)));
        assert_eq!(features.bits(), (1 << 3) | (1 << 5));
    }

    #[test]
    fn capabilities_report_a_single_feature() {
        let capabilities = GpuCapabilities {
            backend: Backend::Vulkan,
            features: GpuFeatures::NONE.with(GpuFeature::ExternalSemaphoreFd),
            limits: GpuLimits::MINIMAL,
        };
        assert!(capabilities.supports(GpuFeature::ExternalSemaphoreFd));
        assert!(!capabilities.supports(GpuFeature::SrgbSurface));
    }

    #[test]
    fn a_surface_description_is_built_with_defaults() {
        let config = SurfaceConfig::new(800, 600).with_scale(2.0);
        assert_eq!((config.width, config.height), (800, 600));
        assert_eq!(config.scale, 2.0);
        assert_eq!(config.format, SurfaceFormat::Bgra8Unorm);
        assert_eq!(config.present_mode, PresentMode::Fifo);
        assert_eq!(config.alpha_mode, AlphaMode::Opaque);
    }

    #[test]
    fn an_srgb_format_is_recognised() {
        assert!(SurfaceFormat::Bgra8UnormSrgb.is_srgb());
        assert!(!SurfaceFormat::Bgra8Unorm.is_srgb());
        assert_eq!(SurfaceFormat::Rgba8UnormSrgb.bytes_per_pixel(), 4);
    }

    #[test]
    fn a_completion_reports_its_kind() {
        assert_eq!(FrameCompletion::Complete.kind(), CompletionKind::Complete);
        assert_eq!(FrameCompletion::Native(7).kind(), CompletionKind::Native);
        assert!(
            FrameCompletion::Complete
                .into_file_descriptor()
                .is_none()
        );
    }
}
