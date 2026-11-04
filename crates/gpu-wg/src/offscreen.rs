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

//! Offscreen color targets: the [`OffscreenFramebuffer`] view handed to
//! the mix pass, the [`PipeOffscreen`] a renderer implements to
//! contribute targets, and the owned [`OffscreenTarget`] allocation.
//!
//! The WebGPU shape mirrors `codevar-gpu-vk`, with one difference the API
//! forces: a Vulkan image and view are copyable handles, while a
//! `wgpu-hal` texture and view are owned values, so
//! [`OffscreenFramebuffer`] borrows them instead of copying them.

use wgpu_types as wgt;

use crate::pipeline::WgContext;
use crate::types::CompositorError;

/// Clamps a mix weight to a finite value in `[0.0, 1.0]`.
///
/// Weights come from pipe implementations and travel to the GPU, so NaN,
/// infinities and out-of-range values are rejected here rather than in
/// the shader. A [`PipeOffscreen`] implementation may call it on the value
/// it is about to return; the compositor applies it unconditionally before
/// pushing the weight to the shader.
#[inline]
#[must_use]
pub fn sanitize_weight(weight: f32) -> f32 {
    if weight.is_finite() {
        weight.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::*;

    use alloc::vec::Vec;
    use core::fmt;
    use core::mem::ManuallyDrop;

    use wgpu_hal::gles::{Texture, TextureView};
    use wgpu_hal::{Device as _, MemoryFlags, TextureDescriptor, TextureViewDescriptor};

    /// View of one offscreen color target exposed to the mix pass.
    ///
    /// A renderer implementing [`PipeOffscreen`] hands these to the
    /// compositor every frame; the compositor owns neither the texture nor
    /// the view, it only binds them for sampling.
    #[derive(Clone, Copy)]
    pub struct OffscreenFramebuffer<'a> {
        /// Color texture backing the target.
        pub texture: &'a Texture,
        /// Texture view bound as a sampled resource by the mix pass.
        pub view: &'a TextureView,
        /// Format of `texture` (must match the mix pipeline's attachment
        /// format for the render pass that writes it).
        pub format: wgt::TextureFormat,
        /// Target width in pixels.
        pub width: u32,
        /// Target height in pixels.
        pub height: u32,
    }

    impl OffscreenFramebuffer<'_> {
        /// Whether the target refers to a real, non-empty texture.
        ///
        /// A borrowed framebuffer is always valid; the method exists so a
        /// renderer can mirror the Vulkan `is_valid` check.
        #[inline]
        #[must_use]
        pub const fn is_valid(&self) -> bool {
            true
        }
    }

    /// Supplies the compositor with the offscreen targets a pipe draws
    /// into.
    ///
    /// Implement this on the renderer registered with the compositor; the
    /// compositor collects the targets once per frame (in queue order) and
    /// folds them into the presentation target with the mix pass.
    pub trait PipeOffscreen {
        /// Number of framebuffers this pipe contributes to the frame.
        fn offscreen_framebuffer_count(&self) -> usize;

        /// The framebuffer at `index`, or `None` when the pipe contributed
        /// fewer targets than advertised.
        #[must_use]
        fn offscreen_framebuffer(&self, index: usize) -> Option<OffscreenFramebuffer<'_>>;

        /// Mix weight of framebuffer `index` in `[0.0, 1.0]`.
        ///
        /// The compositor sanitizes the value with `sanitize_weight` before
        /// pushing it; framebuffer 0 of the whole frame is the base layer
        /// the shader starts from, so its weight is ignored by the mix
        /// itself.
        #[must_use]
        fn mix_weight(&self, index: usize) -> f32 {
            let _ = index;
            1.0
        }
    }

    /// The texture and view an [`OffscreenTarget`] owns.
    ///
    /// The pair is kept behind [`ManuallyDrop`] so [`OffscreenTarget::drop`]
    /// can move the handles out and destroy them; no accessor can observe
    /// the target between the move and the end of `drop`.
    struct TargetHandles {
        /// Color texture.
        texture: Texture,
        /// Sampled view of `texture`.
        view: TextureView,
    }

    /// A pipe-owned offscreen color target: texture and sampled view.
    ///
    /// A renderer exposes it through [`PipeOffscreen`]. The `'p` lifetime
    /// keeps it from outliving the [`WgContext`] that owns the device it
    /// releases on `Drop`.
    pub struct OffscreenTarget<'p> {
        /// Device that owns both handles (`'p`).
        context: &'p WgContext,
        /// The owned texture and view, moved out only on `Drop`.
        handles: ManuallyDrop<TargetHandles>,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Format of the texture.
        format: wgt::TextureFormat,
    }

    impl fmt::Debug for OffscreenTarget<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("OffscreenTarget")
                .field("width", &self.width)
                .field("height", &self.height)
                .field("format", &self.format)
                .finish_non_exhaustive()
        }
    }

    impl<'p> OffscreenTarget<'p> {
        /// Creates an offscreen target of `width` × `height` in `format`.
        ///
        /// # Errors
        ///
        /// [`CompositorError::Graphics`] when the driver rejects the texture
        /// or its view; the partially created texture is released first.
        pub fn new(
            context: &'p WgContext,
            width: u32,
            height: u32,
            format: wgt::TextureFormat,
        ) -> Result<Self, CompositorError> {
            let device = context.device();
            let descriptor = TextureDescriptor {
                label: Some("codevar-offscreen"),
                size: wgt::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgt::TextureDimension::D2,
                format,
                usage: wgt::TextureUses::COLOR_TARGET | wgt::TextureUses::RESOURCE,
                memory_flags: MemoryFlags::empty(),
                view_formats: Vec::new(),
            };
            // SAFETY: the descriptor is well-formed, the extent is non-zero
            // and the device outlives the returned texture.
            let texture = unsafe { device.create_texture(&descriptor) }
                .map_err(|err| CompositorError::graphics("create_texture", err))?;

            let view_descriptor = TextureViewDescriptor {
                label: Some("codevar-offscreen-view"),
                format,
                dimension: wgt::TextureViewDimension::D2,
                usage: wgt::TextureUses::RESOURCE,
                range: wgt::ImageSubresourceRange {
                    aspect: wgt::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                },
            };
            // SAFETY: the view covers the only mip level and layer of a
            // texture created on this device.
            let view = match unsafe { device.create_texture_view(&texture, &view_descriptor) } {
                Ok(view) => view,
                Err(err) => {
                    // SAFETY: the texture is referenced by nothing else yet.
                    unsafe { device.destroy_texture(texture) };
                    return Err(CompositorError::graphics("create_texture_view", err));
                }
            };

            Ok(Self {
                context,
                handles: ManuallyDrop::new(TargetHandles { texture, view }),
                width,
                height,
                format,
            })
        }

        /// The owned handles, for accessors; `ManuallyDrop` derefs to them.
        #[inline]
        fn handles(&self) -> &TargetHandles {
            &self.handles
        }

        /// The target as seen by the compositor.
        #[must_use]
        pub fn framebuffer(&self) -> OffscreenFramebuffer<'_> {
            OffscreenFramebuffer {
                texture: &self.handles().texture,
                view: &self.handles().view,
                format: self.format,
                width: self.width,
                height: self.height,
            }
        }

        /// The color texture.
        #[inline]
        #[must_use]
        pub fn texture(&self) -> &Texture {
            &self.handles().texture
        }

        /// The sampled view of the color texture.
        #[inline]
        #[must_use]
        pub fn view(&self) -> &TextureView {
            &self.handles().view
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

        /// Format of the texture.
        #[inline]
        #[must_use]
        pub const fn format(&self) -> wgt::TextureFormat {
            self.format
        }
    }

    impl Drop for OffscreenTarget<'_> {
        fn drop(&mut self) {
            // SAFETY: `handles` is taken exactly once, here, and only after
            // it was fully initialised by `new`.
            let handles = unsafe { ManuallyDrop::take(&mut self.handles) };
            // SAFETY: both handles were created by this device and are
            // referenced by nothing else; `'p` keeps the device alive for
            // the whole lifetime of `self`.
            unsafe {
                self.context
                    .device()
                    .destroy_texture_view(handles.view);
                self.context
                    .device()
                    .destroy_texture(handles.texture);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use super::*;

    use core::marker::PhantomData;

    /// View of one offscreen color target exposed to the mix pass.
    ///
    /// Off the web there is no texture to borrow, so the `'a` lifetime is
    /// carried by a [`PhantomData`] marker only. Keeping the parameter makes
    /// the type the same shape as the `wasm32` one, so the pipe model is
    /// written once and compiles on both targets.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct OffscreenFramebuffer<'a> {
        /// Format of the target texture.
        pub format: wgt::TextureFormat,
        /// Target width in pixels.
        pub width: u32,
        /// Target height in pixels.
        pub height: u32,
        /// Ties the placeholder to the lifetime of the borrowed target.
        pub(crate) marker: PhantomData<&'a ()>,
    }

    impl OffscreenFramebuffer<'_> {
        /// Whether the target refers to a real, non-empty texture.
        #[inline]
        #[must_use]
        pub const fn is_valid(&self) -> bool {
            self.width != 0 && self.height != 0
        }
    }

    /// Supplies the compositor with the offscreen targets a pipe draws
    /// into.
    pub trait PipeOffscreen {
        /// Number of framebuffers this pipe contributes to the frame.
        fn offscreen_framebuffer_count(&self) -> usize;

        /// The framebuffer at `index`, or `None` when the pipe contributed
        /// fewer targets than advertised.
        #[must_use]
        fn offscreen_framebuffer(&self, index: usize) -> Option<OffscreenFramebuffer<'_>>;

        /// Mix weight of framebuffer `index` in `[0.0, 1.0]`.
        #[must_use]
        fn mix_weight(&self, index: usize) -> f32 {
            let _ = index;
            1.0
        }
    }

    /// A pipe-owned offscreen color target (unavailable off the web).
    #[derive(Debug)]
    pub struct OffscreenTarget<'p> {
        /// Kept for parity with the wasm target, which borrows its context.
        context: &'p WgContext,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Format of the texture.
        format: wgt::TextureFormat,
    }

    impl<'p> OffscreenTarget<'p> {
        /// Constructs the placeholder target; no texture is allocated off
        /// the web.
        ///
        /// # Errors
        ///
        /// Never; the placeholder always succeeds.
        pub fn new(
            context: &'p WgContext,
            width: u32,
            height: u32,
            format: wgt::TextureFormat,
        ) -> Result<Self, CompositorError> {
            Ok(Self {
                context,
                width,
                height,
                format,
            })
        }

        /// The target as seen by the compositor.
        #[must_use]
        pub fn framebuffer(&self) -> OffscreenFramebuffer<'_> {
            OffscreenFramebuffer {
                format: self.format,
                width: self.width,
                height: self.height,
                marker: PhantomData,
            }
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

        /// Format of the texture.
        #[inline]
        #[must_use]
        pub const fn format(&self) -> wgt::TextureFormat {
            self.format
        }

        /// The context the target was created against.
        #[inline]
        #[must_use]
        pub const fn context(&self) -> &'p WgContext {
            self.context
        }
    }
}

pub use imp::{OffscreenFramebuffer, OffscreenTarget, PipeOffscreen};

#[cfg(test)]
mod tests {
    //! Mix weight sanitization, which is API-independent.

    use super::*;

    #[test]
    fn sanitize_weight_rejects_non_finite_and_clamps() {
        assert_eq!(sanitize_weight(f32::NAN), 0.0);
        assert_eq!(sanitize_weight(f32::INFINITY), 0.0);
        assert_eq!(sanitize_weight(f32::NEG_INFINITY), 0.0);
        assert_eq!(sanitize_weight(-1.0), 0.0);
        assert_eq!(sanitize_weight(0.5), 0.5);
        assert_eq!(sanitize_weight(4.0), 1.0);
    }
}
