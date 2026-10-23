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

//! WebGPU pipeline context: the browser instance, canvas surface,
//! adapter, logical device and queue.
//!
//! It owns everything that outlives a frame. The context is the web
//! counterpart of `codevar-gpu-vk`'s [`PipelineContext`]: on the desktop
//! the "swapchain replacement" is an exported dma-buf image, while in
//! the browser it is the canvas surface the adapter is opened against.
//!
//! The backend is compiled on `wasm32` only. Any other target compiles
//! the stub at the bottom of this file, which never opens a device.
//!
//! [`PipelineContext`]: https://docs.rs/codevar-gpu-vk

#[cfg(target_arch = "wasm32")]
mod imp {
    use alloc::format;
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;
    use core::fmt;

    use wgpu_hal::gles::{Adapter, Device, Fence, Instance, Queue, Surface};
    use wgpu_hal::{Adapter as _, Device as _, Instance as _, OpenDevice, Queue as _, Surface as _};
    use wgpu_types as wgt;

    /// Error returned while opening the WebGPU context.
    #[derive(Debug)]
    pub enum WgContextError {
        /// The browser instance could not be created.
        InstanceCreate(String),
        /// The canvas could not be turned into a surface.
        SurfaceCreate(String),
        /// The instance exposed no adapter for the canvas.
        NoAdapter,
        /// The adapter reports no capabilities for the canvas.
        NoSurfaceCapabilities,
        /// The adapter supports no usable colour format.
        NoSurfaceFormat,
        /// The logical device could not be opened.
        DeviceOpen(String),
        /// The surface could not be configured.
        SurfaceConfigure(String),
    }

    impl fmt::Display for WgContextError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::InstanceCreate(err) => write!(f, "WebGPU instance creation failed: {err}"),
                Self::SurfaceCreate(err) => write!(f, "canvas surface creation failed: {err}"),
                Self::NoAdapter => f.write_str("the WebGPU instance exposed no adapter"),
                Self::NoSurfaceCapabilities => f.write_str("the adapter reports no surface capabilities"),
                Self::NoSurfaceFormat => f.write_str("the adapter supports no usable surface format"),
                Self::DeviceOpen(err) => write!(f, "WebGPU device open failed: {err}"),
                Self::SurfaceConfigure(err) => write!(f, "surface configuration failed: {err}"),
            }
        }
    }

    impl core::error::Error for WgContextError {}

    /// The configuration of a WebGPU device.
    ///
    /// The canvas is the surface the device presents to; the size is the
    /// initial extent the surface is configured with.
    pub struct WgConfig {
        /// The canvas the surface presents to.
        pub canvas: web_sys::HtmlCanvasElement,
        /// The initial surface width, in device pixels.
        pub width: u32,
        /// The initial surface height, in device pixels.
        pub height: u32,
    }

    impl WgConfig {
        /// A configuration presenting to `canvas` at `width` by `height`.
        #[must_use]
        pub fn new(canvas: web_sys::HtmlCanvasElement, width: u32, height: u32) -> Self {
            Self {
                canvas,
                width,
                height,
            }
        }
    }

    impl fmt::Debug for WgConfig {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("WgConfig")
                .field("width", &self.width)
                .field("height", &self.height)
                .finish_non_exhaustive()
        }
    }

    /// The WebGPU pipeline context.
    pub struct WgContext {
        /// The browser instance that created the surface and adapter.
        _instance: Instance,
        /// The canvas surface the device presents to.
        surface: Surface,
        /// The logical device owning every resource of this context.
        device: Device,
        /// The submit/present queue of `device`.
        queue: Queue,
        /// The fence every submission signals; the surface acquires against it.
        fence: Fence,
        /// The color format the surface is configured with.
        format: wgt::TextureFormat,
        /// The configuration the surface currently uses.
        config: wgpu_hal::SurfaceConfiguration,
    }

    impl fmt::Debug for WgContext {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("WgContext")
                .field("format", &self.format)
                .field("extent", &self.config.extent)
                .finish_non_exhaustive()
        }
    }

    /// Picks the canvas format the mix pass should render into.
    ///
    /// An sRGB byte format is preferred so the shader writes linear color;
    /// only if the adapter advertises none is a linear format accepted.
    fn pick_format(formats: &[wgt::SurfaceFormatCapabilities]) -> Result<wgt::TextureFormat, WgContextError> {
        const PREFERRED: [wgt::TextureFormat; 4] = [
            wgt::TextureFormat::Bgra8UnormSrgb,
            wgt::TextureFormat::Rgba8UnormSrgb,
            wgt::TextureFormat::Bgra8Unorm,
            wgt::TextureFormat::Rgba8Unorm,
        ];
        for wanted in PREFERRED {
            if formats.iter().any(|entry| entry.format == wanted) {
                return Ok(wanted);
            }
        }
        formats
            .first()
            .map(|entry| entry.format)
            .ok_or(WgContextError::NoSurfaceFormat)
    }

    /// Prefers vertical-synchronized presentation, then whatever the
    /// adapter lists first.
    fn pick_present_mode(modes: &[wgt::PresentMode]) -> wgt::PresentMode {
        if modes.contains(&wgt::PresentMode::Fifo) {
            wgt::PresentMode::Fifo
        } else {
            modes
                .first()
                .copied()
                .unwrap_or(wgt::PresentMode::Fifo)
        }
    }

    /// Prefers an opaque surface, then whatever the adapter lists first.
    fn pick_alpha_mode(modes: &[wgt::CompositeAlphaMode]) -> wgt::CompositeAlphaMode {
        if modes.contains(&wgt::CompositeAlphaMode::Opaque) {
            wgt::CompositeAlphaMode::Opaque
        } else {
            modes
                .first()
                .copied()
                .unwrap_or(wgt::CompositeAlphaMode::Auto)
        }
    }

    impl WgContext {
        /// Opens the WebGPU context for `config`.
        ///
        /// # Errors
        ///
        /// * [`WgContextError::InstanceCreate`], [`WgContextError::SurfaceCreate`] —
        ///   the browser instance or the canvas surface could not be created.
        /// * [`WgContextError::NoAdapter`], [`WgContextError::NoSurfaceCapabilities`],
        ///   [`WgContextError::NoSurfaceFormat`] — the adapter cannot present to
        ///   the canvas.
        /// * [`WgContextError::DeviceOpen`], [`WgContextError::SurfaceConfigure`] —
        ///   the logical device or the surface configuration failed.
        pub fn new(config: &WgConfig) -> Result<Self, WgContextError> {
            let descriptor = wgpu_hal::InstanceDescriptor {
                name: "codevar",
                flags: wgt::InstanceFlags::default(),
                memory_budget_thresholds: wgt::MemoryBudgetThresholds::default(),
                backend_options: wgt::BackendOptions::default(),
                telemetry: None,
                display: None,
            };
            // SAFETY: the descriptor is well-formed and lives for the call.
            let instance = unsafe { Instance::init(&descriptor) }
                .map_err(|err| WgContextError::InstanceCreate(err.to_string()))?;
            let surface = instance
                .create_surface_from_canvas(config.canvas.clone())
                .map_err(|err| WgContextError::SurfaceCreate(err.to_string()))?;
            // SAFETY: the surface was created by `instance` and outlives the call.
            let exposed = unsafe { instance.enumerate_adapters(Some(&surface)) }
                .into_iter()
                .next()
                .ok_or(WgContextError::NoAdapter)?;

            let adapter: Adapter = exposed.adapter;
            let limits: wgt::Limits = exposed.capabilities.limits;
            // SAFETY: `adapter` belongs to `instance` and `surface` is valid.
            let capabilities = unsafe { adapter.surface_capabilities(&surface) }
                .ok_or(WgContextError::NoSurfaceCapabilities)?;
            let format = pick_format(&capabilities.formats)?;
            // SAFETY: `adapter` belongs to `instance`; the limits are exactly
            // the ones the adapter reported, and empty features are always
            // supported.
            let OpenDevice { device, queue } =
                unsafe { adapter.open(wgt::Features::empty(), &limits, &wgt::MemoryHints::default()) }
                    .map_err(|err| WgContextError::DeviceOpen(err.to_string()))?;

            // SAFETY: the device was just opened by this context and owns the
            // fence; it is destroyed only when the context drops.
            let fence = unsafe { device.create_fence() }
                .map_err(|err| WgContextError::DeviceOpen(format!("fence: {err}")))?;

            let surface_config = wgpu_hal::SurfaceConfiguration {
                maximum_frame_latency: (*capabilities.maximum_frame_latency.start()).max(1),
                present_mode: pick_present_mode(&capabilities.present_modes),
                composite_alpha_mode: pick_alpha_mode(&capabilities.composite_alpha_modes),
                format,
                color_space: wgt::SurfaceColorSpace::Srgb,
                extent: wgt::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                usage: wgt::TextureUses::COLOR_TARGET,
                view_formats: Vec::new(),
            };
            // SAFETY: the surface is not yet configured on any device and no
            // surface texture has been acquired from it.
            unsafe { surface.configure(&device, &surface_config) }
                .map_err(|err| WgContextError::SurfaceConfigure(err.to_string()))?;

            Ok(Self {
                _instance: instance,
                surface,
                device,
                queue,
                fence,
                format,
                config: surface_config,
            })
        }

        /// The logical device.
        #[inline]
        #[must_use]
        pub const fn device(&self) -> &Device {
            &self.device
        }

        /// The submit/present queue.
        #[inline]
        #[must_use]
        pub const fn queue(&self) -> &Queue {
            &self.queue
        }

        /// The fence every submission signals and the surface acquires against.
        #[inline]
        #[must_use]
        pub const fn fence(&self) -> &Fence {
            &self.fence
        }

        /// The canvas surface.
        #[inline]
        #[must_use]
        pub const fn surface(&self) -> &Surface {
            &self.surface
        }

        /// The color format the surface is configured with.
        #[inline]
        #[must_use]
        pub const fn format(&self) -> wgt::TextureFormat {
            self.format
        }

        /// The configuration the surface currently uses.
        #[inline]
        #[must_use]
        pub const fn config(&self) -> &wgpu_hal::SurfaceConfiguration {
            &self.config
        }

        /// Surface width, in device pixels.
        #[inline]
        #[must_use]
        pub const fn width(&self) -> u32 {
            self.config.extent.width
        }

        /// Surface height, in device pixels.
        #[inline]
        #[must_use]
        pub const fn height(&self) -> u32 {
            self.config.extent.height
        }

        /// Blocks until every submitted command finished and the queue is
        /// idle.
        ///
        /// # Errors
        ///
        /// The driver's error, mapped to a string, when the queue is lost.
        pub fn wait_idle(&self) -> Result<(), String> {
            // SAFETY: this context has exclusive access to the queue and no
            // command is being submitted concurrently.
            unsafe { self.queue.wait_for_idle() }.map_err(|err| format!("{err}"))
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use alloc::string::String;
    use core::fmt;

    /// Error returned while opening the WebGPU context.
    #[derive(Debug)]
    pub enum WgContextError {
        /// The context can only be created from a `wasm32` build.
        UnsupportedTarget,
    }

    impl fmt::Display for WgContextError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::UnsupportedTarget => f.write_str("the WebGPU backend only opens a device on wasm32"),
            }
        }
    }

    impl core::error::Error for WgContextError {}

    /// The configuration of a WebGPU device.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct WgConfig {
        /// The initial surface width, in device pixels.
        pub width: u32,
        /// The initial surface height, in device pixels.
        pub height: u32,
    }

    impl WgConfig {
        /// A configuration for a surface of `width` by `height`.
        #[must_use]
        pub const fn new(width: u32, height: u32) -> Self {
            Self { width, height }
        }
    }

    /// The WebGPU pipeline context (unavailable off the web).
    #[derive(Debug)]
    pub struct WgContext {
        width: u32,
        height: u32,
    }

    impl WgContext {
        /// Always [`WgContextError::UnsupportedTarget`]: a WebGPU device only
        /// exists in a browser.
        ///
        /// # Errors
        ///
        /// Always [`WgContextError::UnsupportedTarget`].
        pub fn new(config: &WgConfig) -> Result<Self, WgContextError> {
            let _ = (config.width, config.height);
            Err(WgContextError::UnsupportedTarget)
        }

        /// Surface width, in device pixels.
        #[inline]
        #[must_use]
        pub const fn width(&self) -> u32 {
            self.width
        }

        /// Surface height, in device pixels.
        #[inline]
        #[must_use]
        pub const fn height(&self) -> u32 {
            self.height
        }

        /// Always `Ok(())`; there is no queue off the web.
        #[allow(clippy::unnecessary_wraps)]
        pub fn wait_idle(&self) -> Result<(), String> {
            Ok(())
        }
    }
}

pub use imp::{WgConfig, WgContext, WgContextError};
