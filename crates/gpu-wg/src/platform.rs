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

//! The WebGPU implementation of the backend-neutral platform traits.
//!
//! [`WgContext`] is the logical device, [`WgSurface`] is the canvas target
//! it presents to and [`crate::renderer`] is the frame lifecycle. The
//! traits themselves live in `codevar-gpu-core`, so a caller assembled for
//! the web can be written against them and later swapped for the Vulkan
//! backend.
//!
//! On `wasm32` the module drives `wgpu-hal`'s browser backend; on every
//! other target it compiles the stub at the bottom, which advertises the
//! backend but refuses to open a device.

#[cfg(target_arch = "wasm32")]
mod imp {
    use alloc::format;
    use core::fmt;

    use codevar_gpu_core::{
        Backend, FrameCompletion, GpuCapabilities, GpuDevice, GpuFeature, GpuFeatures, GpuFrame, GpuLimits,
        GpuPlatform, GpuSurface, SurfaceConfig, SurfaceFormat,
    };
    use wgpu_hal::gles::Texture;
    use wgpu_hal::{Queue as _, Surface as _};
    use wgpu_types as wgt;

    use crate::pipeline::{WgConfig, WgContext, WgContextError};

    /// The error type of the WebGPU platform.
    #[derive(Debug)]
    pub enum WgError {
        /// The pipeline context could not be created.
        Context(WgContextError),
        /// A graphics call failed: the name of the call and the driver message.
        Graphics(alloc::string::String),
    }

    impl fmt::Display for WgError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Context(err) => err.fmt(f),
                Self::Graphics(detail) => f.write_str(detail),
            }
        }
    }

    impl core::error::Error for WgError {}

    /// The features the WebGPU backend advertises.
    #[inline]
    #[must_use]
    pub const fn webgpu_features() -> GpuFeatures {
        GpuFeatures::NONE
            .with(GpuFeature::SrgbSurface)
            .with(GpuFeature::DirectDisplayList)
            .with(GpuFeature::Compute)
    }

    /// Maps a `wgpu-hal` surface format onto the backend-neutral one.
    #[inline]
    #[must_use]
    pub const fn surface_format_of(format: wgt::TextureFormat) -> SurfaceFormat {
        match format {
            wgt::TextureFormat::Bgra8Unorm => SurfaceFormat::Bgra8Unorm,
            wgt::TextureFormat::Bgra8UnormSrgb => SurfaceFormat::Bgra8UnormSrgb,
            wgt::TextureFormat::Rgba8Unorm => SurfaceFormat::Rgba8Unorm,
            _ => SurfaceFormat::Rgba8UnormSrgb,
        }
    }

    /// The WebGPU platform: a stateless factory for [`WgContext`]s.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct WgPlatform;

    impl WgPlatform {
        /// Creates the platform.
        #[must_use]
        pub const fn new() -> Self {
            Self
        }
    }

    impl GpuPlatform for WgPlatform {
        type Config = WgConfig;
        type Device = WgContext;
        type Error = WgError;

        fn backend(&self) -> Backend {
            Backend::WebGpu
        }

        fn capabilities(&self) -> GpuCapabilities {
            GpuCapabilities {
                backend: Backend::WebGpu,
                features: webgpu_features(),
                limits: GpuLimits::MINIMAL,
            }
        }

        fn create_device(
            &self,
            config: &Self::Config,
        ) -> impl core::future::Future<Output = Result<Self::Device, Self::Error>> {
            core::future::ready(WgContext::new(config).map_err(WgError::Context))
        }
    }

    impl GpuDevice for WgContext {
        type Surface<'a> = WgSurface<'a>;
        type Error = WgError;

        fn capabilities(&self) -> GpuCapabilities {
            GpuCapabilities {
                backend: Backend::WebGpu,
                features: webgpu_features(),
                limits: GpuLimits::MINIMAL,
            }
        }

        fn create_surface(&mut self, config: SurfaceConfig) -> Result<Self::Surface<'_>, Self::Error> {
            let (width, height) = (self.width(), self.height());
            if config.width != width || config.height != height {
                return Err(WgError::Graphics(format!(
                    "the surface was asked for {}x{} but the context is {width}x{height}",
                    config.width, config.height
                )));
            }
            let format = surface_format_of(self.format());
            Ok(WgSurface {
                context: self,
                config: config.with_format(format),
            })
        }

        fn wait_idle(&mut self) -> Result<(), Self::Error> {
            WgContext::wait_idle(self).map_err(WgError::Graphics)
        }
    }

    /// The canvas target a WebGPU device presents to.
    ///
    /// The surface itself belongs to the [`WgContext`]; this type only
    /// describes it and hands out frame handles that refer to it.
    pub struct WgSurface<'a> {
        /// The device and canvas surface backing every frame.
        context: &'a WgContext,
        /// The configuration adopted for the frames of this surface.
        config: SurfaceConfig,
    }

    impl<'a> WgSurface<'a> {
        /// The pipeline context that owns the canvas surface.
        #[inline]
        #[must_use]
        pub const fn context(&self) -> &'a WgContext {
            self.context
        }

        /// The format of the color attachment.
        #[inline]
        #[must_use]
        pub const fn format(&self) -> SurfaceFormat {
            self.config.format
        }
    }

    impl fmt::Debug for WgSurface<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("WgSurface")
                .field("config", &self.config)
                .finish_non_exhaustive()
        }
    }

    impl GpuSurface for WgSurface<'_> {
        type Frame = WgFrame;
        type Error = WgError;

        fn config(&self) -> SurfaceConfig {
            self.config
        }

        fn resize(&mut self, config: SurfaceConfig) -> Result<(), Self::Error> {
            let mut surface_config = self.context.config().clone();
            surface_config.extent = wgt::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            };
            // SAFETY: `resize` has exclusive access to the surface, no frame
            // is outstanding (a frame is always presented or dropped before
            // the next call) and no commands reference the old textures.
            unsafe {
                self.context
                    .surface()
                    .configure(self.context.device(), &surface_config)
            }
            .map_err(|err| WgError::Graphics(format!("surface reconfigure failed: {err}")))?;
            self.config = config.with_format(self.config.format);
            Ok(())
        }

        fn acquire(&mut self) -> Result<Self::Frame, Self::Error> {
            // SAFETY: the surface is configured on this context's device and
            // the fence is the one every submission of this context signals.
            let acquired = unsafe {
                self.context
                    .surface()
                    .acquire_texture(None, self.context.fence())
            }
            .map_err(|err| WgError::Graphics(format!("surface acquire failed: {err}")))?;
            Ok(WgFrame {
                width: self.config.width,
                height: self.config.height,
                texture: acquired.texture,
            })
        }

        fn present(&mut self, frame: Self::Frame) -> Result<FrameCompletion, Self::Error> {
            // SAFETY: `frame.texture` was acquired from this surface and has
            // not been presented or discarded yet.
            unsafe {
                self.context
                    .queue()
                    .present(self.context.surface(), frame.texture)
            }
            .map_err(|err| WgError::Graphics(format!("surface present failed: {err}")))?;
            // The browser composites the canvas itself once the queue work
            // completes, so no completion handle crosses the boundary.
            Ok(FrameCompletion::Complete)
        }
    }

    /// One acquired frame of a [`WgSurface`].
    pub struct WgFrame {
        /// Width of the target, in pixels.
        width: u32,
        /// Height of the target, in pixels.
        height: u32,
        /// The acquired canvas texture, presented when the frame is consumed.
        texture: Texture,
    }

    impl fmt::Debug for WgFrame {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("WgFrame")
                .field("width", &self.width)
                .field("height", &self.height)
                .finish_non_exhaustive()
        }
    }

    impl GpuFrame for WgFrame {
        #[inline]
        fn width(&self) -> u32 {
            self.width
        }

        #[inline]
        fn height(&self) -> u32 {
            self.height
        }
    }

    #[cfg(test)]
    mod tests {
        //! Platform metadata that does not need a live browser.

        use super::*;

        #[test]
        fn the_platform_reports_webgpu_capabilities() {
            let platform = WgPlatform::new();
            assert_eq!(platform.backend(), Backend::WebGpu);
            let capabilities = platform.capabilities();
            assert_eq!(capabilities.backend, Backend::WebGpu);
            assert!(capabilities.supports(GpuFeature::SrgbSurface));
            assert!(capabilities.supports(GpuFeature::Compute));
            assert!(!capabilities.supports(GpuFeature::DmaBufExport));
        }

        #[test]
        fn byte_formats_map_to_their_neutral_counterpart() {
            assert_eq!(
                surface_format_of(wgt::TextureFormat::Bgra8UnormSrgb),
                SurfaceFormat::Bgra8UnormSrgb
            );
            assert_eq!(
                surface_format_of(wgt::TextureFormat::Rgba8Unorm),
                SurfaceFormat::Rgba8Unorm
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use alloc::string::String;
    use core::fmt;

    use codevar_gpu_core::{
        Backend, FrameCompletion, GpuCapabilities, GpuDevice, GpuFeature, GpuFeatures, GpuFrame, GpuLimits,
        GpuPlatform, GpuSurface, SurfaceConfig,
    };

    use crate::pipeline::{WgConfig, WgContext, WgContextError};

    /// The error type of the WebGPU platform.
    #[derive(Debug)]
    pub enum WgError {
        /// The pipeline context could not be created.
        Context(WgContextError),
        /// A graphics call failed: the name of the call and the driver message.
        Graphics(String),
    }

    impl fmt::Display for WgError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Context(err) => err.fmt(f),
                Self::Graphics(detail) => f.write_str(detail),
            }
        }
    }

    impl core::error::Error for WgError {}

    /// The features the WebGPU backend advertises.
    #[inline]
    #[must_use]
    pub const fn webgpu_features() -> GpuFeatures {
        GpuFeatures::NONE
            .with(GpuFeature::SrgbSurface)
            .with(GpuFeature::DirectDisplayList)
            .with(GpuFeature::Compute)
    }

    /// The WebGPU platform: a stateless factory for [`WgContext`]s.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct WgPlatform;

    impl WgPlatform {
        /// Creates the platform.
        #[must_use]
        pub const fn new() -> Self {
            Self
        }
    }

    impl GpuPlatform for WgPlatform {
        type Config = WgConfig;
        type Device = WgContext;
        type Error = WgError;

        fn backend(&self) -> Backend {
            Backend::WebGpu
        }

        fn capabilities(&self) -> GpuCapabilities {
            GpuCapabilities {
                backend: Backend::WebGpu,
                features: webgpu_features(),
                limits: GpuLimits::MINIMAL,
            }
        }

        fn create_device(
            &self,
            config: &Self::Config,
        ) -> impl core::future::Future<Output = Result<Self::Device, Self::Error>> {
            core::future::ready(WgContext::new(config).map_err(WgError::Context))
        }
    }

    impl GpuDevice for WgContext {
        type Surface<'a> = WgSurface;
        type Error = WgError;

        fn capabilities(&self) -> GpuCapabilities {
            GpuCapabilities {
                backend: Backend::WebGpu,
                features: webgpu_features(),
                limits: GpuLimits::MINIMAL,
            }
        }

        fn create_surface(&mut self, _config: SurfaceConfig) -> Result<Self::Surface<'_>, Self::Error> {
            Err(WgError::Context(WgContextError::UnsupportedTarget))
        }

        fn wait_idle(&mut self) -> Result<(), Self::Error> {
            WgContext::wait_idle(self).map_err(WgError::Graphics)
        }
    }

    /// The canvas target a WebGPU device would present to.
    #[derive(Debug)]
    pub struct WgSurface;

    impl GpuSurface for WgSurface {
        type Frame = WgFrame;
        type Error = WgError;

        fn config(&self) -> SurfaceConfig {
            SurfaceConfig::default()
        }

        fn resize(&mut self, _config: SurfaceConfig) -> Result<(), Self::Error> {
            Err(WgError::Context(WgContextError::UnsupportedTarget))
        }

        fn acquire(&mut self) -> Result<Self::Frame, Self::Error> {
            Err(WgError::Context(WgContextError::UnsupportedTarget))
        }

        fn present(&mut self, _frame: Self::Frame) -> Result<FrameCompletion, Self::Error> {
            Err(WgError::Context(WgContextError::UnsupportedTarget))
        }
    }

    /// One acquired frame of a [`WgSurface`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct WgFrame {
        /// Width of the target, in pixels.
        width: u32,
        /// Height of the target, in pixels.
        height: u32,
    }

    impl GpuFrame for WgFrame {
        #[inline]
        fn width(&self) -> u32 {
            self.width
        }

        #[inline]
        fn height(&self) -> u32 {
            self.height
        }
    }

    #[cfg(test)]
    mod tests {
        //! Platform metadata that does not need a live browser.

        use super::*;

        #[test]
        fn the_platform_reports_webgpu_capabilities() {
            let platform = WgPlatform::new();
            assert_eq!(platform.backend(), Backend::WebGpu);
            let capabilities = platform.capabilities();
            assert_eq!(capabilities.backend, Backend::WebGpu);
            assert!(capabilities.supports(GpuFeature::SrgbSurface));
            assert!(capabilities.supports(GpuFeature::Compute));
            assert!(!capabilities.supports(GpuFeature::DmaBufExport));
        }

        #[test]
        fn a_frame_reports_its_size() {
            let frame = WgFrame {
                width: 10,
                height: 20,
            };
            assert_eq!(frame.width(), 10);
            assert_eq!(frame.height(), 20);
        }
    }
}

pub use imp::{WgError, WgPlatform, WgSurface, webgpu_features};
