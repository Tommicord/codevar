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

//! The Vulkan implementation of the backend-neutral platform traits.
//!
//! [`PipelineContext`] is the logical device, [`VkSurface`] is the
//! swapchain-less offscreen target it owns, and [`RendererSubsystem`] is
//! the frame lifecycle. The traits themselves live in `codevar-gpu-core`,
//! so a caller assembled for the desktop can be written against them and
//! later swapped for the WebGPU backend.
//!
//! The Vulkan target is created once, with the DRM modifiers the
//! compositor advertised. A surface therefore cannot change its size: a
//! resize creates a new [`PipelineContext`] and a new surface, which is
//! the same lifecycle the Wayland presentation layer already follows.

use alloc::vec::Vec;
use codevar_gpu_core::{
    Backend, FrameCompletion, GpuCapabilities, GpuDevice, GpuFeature, GpuFeatures, GpuFrame, GpuLimits,
    GpuPlatform, GpuRenderer, GpuSurface, SurfaceConfig, SurfaceFormat,
};

use crate::pipeline::{OwnedFd, PipelineContext, PipelineError};
use crate::renderer::{RendererError, RendererSubsystem};

/// The features the Vulkan backend requires from a device.
///
/// [`PipelineContext::new`] refuses a device without them, so their
/// presence is what a created context proves.
#[inline]
#[must_use]
pub const fn vulkan_features() -> GpuFeatures {
    GpuFeatures::NONE
        .with(GpuFeature::DynamicRendering)
        .with(GpuFeature::DmaBufExport)
        .with(GpuFeature::ExternalSemaphoreFd)
        .with(GpuFeature::DirectDisplayList)
}

/// The configuration of a Vulkan device.
///
/// `modifiers` is the list of DRM format modifiers advertised by the
/// compositor's linux-dmabuf feedback, exactly as
/// [`PipelineContext::new`] expects it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VkConfig {
    /// The width of the render target, in pixels.
    pub width: u32,
    /// The height of the render target, in pixels.
    pub height: u32,
    /// The DRM format modifiers the compositor accepts.
    pub modifiers: Vec<u64>,
}

impl VkConfig {
    /// A configuration for a target of `width` by `height` pixels.
    #[must_use]
    pub fn new(width: u32, height: u32, modifiers: impl Into<Vec<u64>>) -> Self {
        Self {
            width,
            height,
            modifiers: modifiers.into(),
        }
    }
}

/// The Vulkan platform: a stateless factory for [`PipelineContext`]s.
///
/// Creating a device opens the Vulkan loader and validates the required
/// extensions, external handle types and features. The platform itself
/// holds nothing, so it can be constructed anywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VkPlatform;

impl VkPlatform {
    /// Creates the platform.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl GpuPlatform for VkPlatform {
    type Config = VkConfig;
    type Device = PipelineContext;
    type Error = PipelineError;

    fn backend(&self) -> Backend {
        Backend::Vulkan
    }

    fn capabilities(&self) -> GpuCapabilities {
        GpuCapabilities {
            backend: Backend::Vulkan,
            features: vulkan_features(),
            limits: GpuLimits::MINIMAL,
        }
    }

    fn create_device(
        &self,
        config: &Self::Config,
    ) -> impl core::future::Future<Output = Result<Self::Device, Self::Error>> {
        core::future::ready(PipelineContext::new(
            config.width,
            config.height,
            &config.modifiers,
        ))
    }
}

impl GpuDevice for PipelineContext {
    type Surface<'a> = VkSurface<'a>;
    type Error = PipelineError;

    fn capabilities(&self) -> GpuCapabilities {
        // SAFETY: the physical device belongs to this context's instance and
        // the driver only fills the returned properties structure.
        let properties = unsafe {
            self.instance()
                .get_physical_device_properties(self.physical_device())
        };
        GpuCapabilities {
            backend: Backend::Vulkan,
            features: vulkan_features(),
            limits: GpuLimits {
                max_texture_dimension_2d: properties.limits.max_image_dimension2_d,
                max_vertex_attributes: properties.limits.max_vertex_input_attributes,
                max_vertex_buffer_stride: properties.limits.max_vertex_input_binding_stride,
                max_bind_groups: properties.limits.max_bound_descriptor_sets,
                max_push_constant_bytes: properties.limits.max_push_constants_size,
            },
        }
    }

    fn create_surface(&mut self, config: SurfaceConfig) -> Result<Self::Surface<'_>, Self::Error> {
        let actual = (self.width(), self.height());
        if config.width != actual.0 || config.height != actual.1 {
            return Err(PipelineError::SurfaceConfigUnsupported {
                requested: (config.width, config.height),
                actual,
            });
        }
        Ok(VkSurface {
            context: self,
            config: config.with_format(SurfaceFormat::Bgra8Unorm),
        })
    }

    fn wait_idle(&mut self) -> Result<(), Self::Error> {
        // SAFETY: the device belongs to this context and no frame is being
        // submitted through `&mut self`.
        unsafe { self.device().device_wait_idle() }.map_err(PipelineError::DeviceWait)
    }
}

/// The configuration and exported target of a Vulkan device.
///
/// The target itself belongs to the [`PipelineContext`]; the surface only
/// describes it and hands out frame handles that refer to it.
pub struct VkSurface<'a> {
    /// The device and offscreen target backing every frame.
    context: &'a PipelineContext,
    /// The configuration adopted for the frames of this surface.
    config: SurfaceConfig,
}

impl<'a> VkSurface<'a> {
    /// The pipeline context that owns the render target.
    #[inline]
    #[must_use]
    pub const fn context(&self) -> &'a PipelineContext {
        self.context
    }

    /// The exported render target the Wayland layer presents.
    #[inline]
    #[must_use]
    pub fn present_target(&self) -> &'a crate::pipeline::RenderTarget {
        self.context.present_target()
    }

    /// The Vulkan format of the color attachment.
    #[inline]
    #[must_use]
    pub const fn format(&self) -> SurfaceFormat {
        SurfaceFormat::Bgra8Unorm
    }
}

impl GpuSurface for VkSurface<'_> {
    type Frame = VkFrame;
    type Error = PipelineError;

    fn config(&self) -> SurfaceConfig {
        self.config
    }

    fn resize(&mut self, config: SurfaceConfig) -> Result<(), Self::Error> {
        let actual = (self.context.width(), self.context.height());
        if config.width != actual.0 || config.height != actual.1 {
            return Err(PipelineError::SurfaceConfigUnsupported {
                requested: (config.width, config.height),
                actual,
            });
        }
        self.config = config.with_format(SurfaceFormat::Bgra8Unorm);
        Ok(())
    }

    fn acquire(&mut self) -> Result<Self::Frame, Self::Error> {
        Ok(VkFrame {
            width: self.config.width,
            height: self.config.height,
        })
    }

    fn present(&mut self, _frame: Self::Frame) -> Result<FrameCompletion, Self::Error> {
        // The image is already the exported dma-buf; the Wayland layer
        // presents it once the renderer's `sync_file` signals completion.
        Ok(FrameCompletion::Complete)
    }
}

/// One acquired frame of a [`VkSurface`].
///
/// The frame carries no GPU handle: every frame of a context renders into
/// the same offscreen image, and the renderer's submission is what
/// reaches the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VkFrame {
    /// Width of the target, in pixels.
    width: u32,
    /// Height of the target, in pixels.
    height: u32,
}

impl GpuFrame for VkFrame {
    #[inline]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.height
    }
}

impl GpuRenderer for RendererSubsystem<'_> {
    type Error = RendererError;

    fn backend(&self) -> Backend {
        Backend::Vulkan
    }

    fn capabilities(&self) -> GpuCapabilities {
        self.context().capabilities()
    }

    fn begin_frame(&mut self, wait: Option<FrameCompletion>) -> Result<(), Self::Error> {
        let wait: Option<OwnedFd> = match wait {
            None => None,
            Some(FrameCompletion::FileDescriptor(fd)) => Some(fd),
            Some(other) => return Err(RendererError::UnsupportedCompletion(other.kind())),
        };
        RendererSubsystem::begin_frame(self, wait)
    }

    fn submit_frame(&mut self) -> Result<(), Self::Error> {
        RendererSubsystem::present_frame(self)
    }

    fn end_frame(&mut self) -> Result<FrameCompletion, Self::Error> {
        RendererSubsystem::end_frame(self).map(FrameCompletion::FileDescriptor)
    }

    fn frame_index(&self) -> u64 {
        RendererSubsystem::frame_index(self)
    }
}

#[cfg(test)]
mod tests {
    //! Platform metadata that does not need a live Vulkan device.

    use super::*;

    #[test]
    fn the_platform_reports_vulkan_capabilities() {
        let platform = VkPlatform::new();
        assert_eq!(platform.backend(), Backend::Vulkan);
        let capabilities = platform.capabilities();
        assert_eq!(capabilities.backend, Backend::Vulkan);
        assert!(capabilities.supports(GpuFeature::DynamicRendering));
        assert!(capabilities.supports(GpuFeature::DmaBufExport));
        assert!(capabilities.supports(GpuFeature::ExternalSemaphoreFd));
        assert!(!capabilities.supports(GpuFeature::SrgbSurface));
    }

    #[test]
    fn a_configuration_keeps_its_modifiers() {
        let config = VkConfig::new(640, 480, [1_u64, 2, 3]);
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.modifiers, [1, 2, 3]);
    }

    #[test]
    fn a_frame_reports_its_size() {
        let frame = VkFrame {
            width: 10,
            height: 20,
        };
        assert_eq!(frame.width(), 10);
        assert_eq!(frame.height(), 20);
    }
}
