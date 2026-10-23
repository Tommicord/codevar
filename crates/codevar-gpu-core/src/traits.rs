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

//! The platform traits a graphics backend implements.
//!
//! A backend crate implements these traits on its own types; the
//! application never names the concrete device, surface or renderer of
//! the API it selected. Vulkan implements them in `codevar-gpu-vk` and
//! WebGPU in `codevar-gpu-wg`.
//!
//! # Lifetimes
//!
//! A Vulkan surface borrows the device that created it, because the
//! device owns the offscreen image the surface presents. The surface
//! associated type of [`GpuDevice`] therefore carries a lifetime, while
//! the WebGPU surface is owned and simply selects the `'a` its device
//! provides.

use core::future::Future;

use crate::backend::{Backend, FrameCompletion, GpuCapabilities, GpuFailure, SurfaceConfig};

/// A graphics API instance that can create a device.
///
/// The configuration is backend-specific on purpose: a Vulkan device
/// needs the DRM modifiers the compositor advertised, while a WebGPU
/// device needs the canvas it will present to. Everything a caller does
/// *after* a device exists is backend-neutral.
pub trait GpuPlatform {
    /// The backend-specific device configuration.
    type Config;

    /// The device type this platform creates.
    type Device: GpuDevice;

    /// The error type the platform reports.
    type Error: GpuFailure;

    /// The graphics API this platform implements.
    fn backend(&self) -> Backend;

    /// The capabilities the platform can promise before a device is
    /// created.
    fn capabilities(&self) -> GpuCapabilities;

    /// Creates the logical device (and its queue).
    ///
    /// The returned future is not required to be `Send`: the WebGPU
    /// device is created asynchronously on the browser event loop, where
    /// the underlying handles are not thread-safe.
    fn create_device(&self, config: &Self::Config)
    -> impl Future<Output = Result<Self::Device, Self::Error>>;
}

/// A logical device that can create surfaces and be waited on.
pub trait GpuDevice {
    /// The surface type this device creates.
    ///
    /// The lifetime lets a surface borrow the device it was created from.
    type Surface<'a>: GpuSurface
    where
        Self: 'a;

    /// The error type the device reports.
    type Error: GpuFailure;

    /// What the device supports and its fixed limits.
    fn capabilities(&self) -> GpuCapabilities;

    /// Creates a surface of `config`.
    ///
    /// A surface may already exist with a different configuration (as is
    /// the case for the swapchain-less Vulkan desktop backend); the
    /// implementation then adopts `config` for the frames it hands out.
    fn create_surface(&mut self, config: SurfaceConfig) -> Result<Self::Surface<'_>, Self::Error>;

    /// Blocks until every submitted frame finished.
    fn wait_idle(&mut self) -> Result<(), Self::Error>;
}

/// A target that frames are rendered into.
pub trait GpuSurface {
    /// The frame handle handed out by [`GpuSurface::acquire`].
    type Frame: GpuFrame;

    /// The error type the surface reports.
    type Error: GpuFailure;

    /// The configuration the surface currently uses.
    #[must_use]
    fn config(&self) -> SurfaceConfig;

    /// Reconfigures the surface (for example after a window resize).
    fn resize(&mut self, config: SurfaceConfig) -> Result<(), Self::Error>;

    /// Acquires the target of the next frame.
    ///
    /// The returned frame is consumed by [`GpuSurface::present`]; it is
    /// the only frame in flight, which is why the method takes `&mut
    /// self`.
    fn acquire(&mut self) -> Result<Self::Frame, Self::Error>;

    /// Presents `frame` and returns the proof that the frame finished.
    fn present(&mut self, frame: Self::Frame) -> Result<FrameCompletion, Self::Error>;
}

/// One acquired frame target.
pub trait GpuFrame {
    /// The width of the target, in device pixels.
    #[must_use]
    fn width(&self) -> u32;

    /// The height of the target, in device pixels.
    #[must_use]
    fn height(&self) -> u32;
}

/// The frame lifecycle of a backend renderer.
///
/// The three steps mirror a Vulkan submission and a WebGPU frame:
///
/// 1. [`GpuRenderer::begin_frame`] retires the previous frame and makes
///    the renderer ready to record a new one.
/// 2. [`GpuRenderer::submit_frame`] records and submits the frame.
/// 3. [`GpuRenderer::end_frame`] produces the completion handle the
///    presentation layer waits on.
pub trait GpuRenderer {
    /// The error type the renderer reports.
    type Error: GpuFailure;

    /// The graphics API the renderer drives.
    fn backend(&self) -> Backend;

    /// What the renderer's device supports.
    fn capabilities(&self) -> GpuCapabilities;

    /// Starts a frame, optionally waiting on `wait` first.
    ///
    /// A backend that cannot use the kind of handle it is given reports
    /// an error rather than silently dropping the dependency.
    fn begin_frame(&mut self, wait: Option<FrameCompletion>) -> Result<(), Self::Error>;

    /// Records and submits the frame begun by [`GpuRenderer::begin_frame`].
    fn submit_frame(&mut self) -> Result<(), Self::Error>;

    /// Finishes the submitted frame and returns its completion handle.
    fn end_frame(&mut self) -> Result<FrameCompletion, Self::Error>;

    /// The number of frames fully rendered so far.
    #[must_use]
    fn frame_index(&self) -> u64;
}

#[cfg(test)]
mod tests {
    //! The traits are exercised by the backend crates; this module only
    //! checks that the bounds compose as documented.

    use super::*;
    use crate::backend::{AlphaMode, Backend, GpuFeatures, GpuLimits, PresentMode, SurfaceFormat};

    /// A complete in-memory backend, used to prove the trait shapes.
    struct TestPlatform;

    struct TestDevice;

    struct TestSurface {
        config: SurfaceConfig,
    }

    struct TestFrame;

    impl GpuPlatform for TestPlatform {
        type Config = ();
        type Device = TestDevice;
        type Error = core::convert::Infallible;

        fn backend(&self) -> Backend {
            Backend::WebGpu
        }

        fn capabilities(&self) -> GpuCapabilities {
            GpuCapabilities {
                backend: Backend::WebGpu,
                features: GpuFeatures::NONE,
                limits: GpuLimits::MINIMAL,
            }
        }

        async fn create_device(&self, _config: &()) -> Result<TestDevice, Self::Error> {
            Ok(TestDevice)
        }
    }

    impl GpuDevice for TestDevice {
        type Surface<'a> = TestSurface;
        type Error = core::convert::Infallible;

        fn capabilities(&self) -> GpuCapabilities {
            GpuCapabilities {
                backend: Backend::WebGpu,
                features: GpuFeatures::NONE,
                limits: GpuLimits::MINIMAL,
            }
        }

        fn create_surface(&mut self, config: SurfaceConfig) -> Result<Self::Surface<'_>, Self::Error> {
            Ok(TestSurface { config })
        }

        fn wait_idle(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    impl GpuSurface for TestSurface {
        type Frame = TestFrame;
        type Error = core::convert::Infallible;

        fn config(&self) -> SurfaceConfig {
            self.config
        }

        fn resize(&mut self, config: SurfaceConfig) -> Result<(), Self::Error> {
            self.config = config;
            Ok(())
        }

        fn acquire(&mut self) -> Result<TestFrame, Self::Error> {
            Ok(TestFrame)
        }

        fn present(&mut self, _frame: TestFrame) -> Result<FrameCompletion, Self::Error> {
            Ok(FrameCompletion::Complete)
        }
    }

    impl GpuFrame for TestFrame {
        fn width(&self) -> u32 {
            0
        }

        fn height(&self) -> u32 {
            0
        }
    }

    #[test]
    fn a_surface_round_trips_its_configuration() {
        let mut device = TestDevice;
        let config = SurfaceConfig {
            width: 640,
            height: 480,
            scale: 2.0,
            format: SurfaceFormat::Rgba8UnormSrgb,
            present_mode: PresentMode::Immediate,
            alpha_mode: AlphaMode::Premultiplied,
        };
        let mut surface = device
            .create_surface(config)
            .expect("the test device creates surfaces");
        assert_eq!(surface.config(), config);
        let resized = config.with_scale(1.0);
        surface
            .resize(resized)
            .expect("the test surface resizes");
        assert_eq!(surface.config(), resized);
        let frame = surface
            .acquire()
            .expect("the test surface acquires");
        assert_eq!(frame.width(), 0);
        assert!(matches!(surface.present(frame), Ok(FrameCompletion::Complete)));
    }
}
