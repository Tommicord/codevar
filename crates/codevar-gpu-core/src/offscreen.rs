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

use ash::vk;

use crate::base::{Compositor, CompositorError, PipeCtx};
use crate::pipeline::PipelineContext;

/// View of one offscreen color target exposed to the mix pass.
///
/// A renderer implementing [`PipeOffscreen`] hands these to the
/// compositor every frame; the compositor owns neither the image nor the
/// view, it only binds them for sampling.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct OffscreenFramebuffer {
    /// Color image backing the target.
    pub image: vk::Image,
    /// Image view bound as a sampled image by the mix pass.
    pub view: vk::ImageView,
    /// Format of `image` (must match the mix pipeline's attachment
    /// format for the render pass that writes it).
    pub format: vk::Format,
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
}

impl OffscreenFramebuffer {
    /// Placeholder used for unused slots of [`PipeCtx`](PipeCtx)'s snapshot.
    pub const EMPTY: Self = Self {
        image: vk::Image::null(),
        view: vk::ImageView::null(),
        format: vk::Format::UNDEFINED,
        width: 0,
        height: 0,
    };

    /// Whether the view refers to a real, non-empty image.
    #[inline]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.view != vk::ImageView::null()
    }
}

/// Supplies the compositor with the offscreen targets a pipe draws into.
///
/// Implement this on the renderer registered with
/// [`Compositor::add_pipe`](Compositor::add_pipe); the compositor collects the targets once per
/// frame (in queue order) and folds them into the presentation target
/// with the mix pass.
pub trait PipeOffscreen {
    /// Number of framebuffers this pipe contributes to the frame.
    fn offscreen_framebuffer_count(&self) -> usize;

    /// The framebuffer at `index`, or `None` when the pipe contributed
    /// fewer targets than advertised.
    #[must_use]
    fn offscreen_framebuffer(&self, index: usize) -> Option<OffscreenFramebuffer>;

    /// Mix weight of framebuffer `index` in `[0.0, 1.0]`.
    ///
    /// The compositor sanitizes the value with
    /// `sanitize_weight` before pushing it; framebuffer 0 of the whole
    /// frame is the base layer the shader starts from, so its weight is
    /// ignored by the mix itself.
    #[must_use]
    fn mix_weight(&self, index: usize) -> f32 {
        let _ = index;
        1.0
    }
}

/// Clamps a mix weight to a finite value in `[0.0, 1.0]`.
///
/// Weights come from pipe implementations and travel to the GPU as push
/// constants, so NaN, infinities and out-of-range values are rejected
/// here rather than in the shader.
#[inline]
pub(crate) fn sanitize_weight(weight: f32) -> f32 {
    if weight.is_finite() {
        weight.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Creates the color image of an [`OffscreenTarget`].
fn create_color_image(
    context: &PipelineContext,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<vk::Image, CompositorError> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: `info` describes a plain 2D color target and the device
    // outlives the call.
    unsafe { context.device().create_image(&info, None) }.map_err(CompositorError::ImageCreate)
}

/// Allocates and binds device-local memory for `image`.
fn allocate_image_memory(
    context: &PipelineContext,
    image: vk::Image,
) -> Result<vk::DeviceMemory, CompositorError> {
    let device = context.device();
    // SAFETY: `image` was created by [`create_color_image`] on this
    // device, so the driver fills a valid requirements struct.
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    // SAFETY: the instance and physical device outlive the context.
    let properties = unsafe {
        context
            .instance()
            .get_physical_device_memory_properties(context.physical_device())
    };
    let memory_type = crate::pipeline::find_memory_type(
        &properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or(CompositorError::NoSuitableMemoryType)?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    // SAFETY: the allocation matches an advertised memory type and size.
    let memory =
        unsafe { device.allocate_memory(&allocate_info, None) }.map_err(CompositorError::MemoryAllocate)?;
    // SAFETY: `memory` was just allocated for this device and `image` is
    // a valid unbound image.
    if let Err(err) = unsafe { device.bind_image_memory(image, memory, 0) } {
        // SAFETY: nothing else references the allocation yet.
        unsafe { device.free_memory(memory, None) };
        return Err(CompositorError::MemoryBind(err));
    }
    Ok(memory)
}

/// Creates the sampled view of `image`.
fn create_color_view(
    context: &PipelineContext,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, CompositorError> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: the image exists on this device and the subresource range
    // covers its only level and layer.
    unsafe { context.device().create_image_view(&info, None) }.map_err(CompositorError::ImageViewCreate)
}

/// A pipe-owned offscreen color target: image, memory and sampled view.
///
/// The target is created in `UNDEFINED` layout and is transitioned by
/// the compositor's layout pass; a renderer exposes it through
/// [`PipeOffscreen`]. The `'p` lifetime keeps it from outliving the
/// [`PipelineContext`] that owns the device it releases on `Drop`.
pub struct OffscreenTarget<'p> {
    /// Device that owns every handle below (`'p`).
    device: &'p ash::Device,
    /// Color image.
    image: vk::Image,
    /// Device memory bound to `image`.
    memory: vk::DeviceMemory,
    /// View bound as a sampled image by the mix pass.
    view: vk::ImageView,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Format of `image`.
    format: vk::Format,
}

impl<'p> OffscreenTarget<'p> {
    /// Creates an offscreen target of `width` × `height` in `format`.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::ImageCreate`], [`CompositorError::NoSuitableMemoryType`],
    ///   [`CompositorError::MemoryAllocate`], [`CompositorError::MemoryBind`],
    ///   [`CompositorError::ImageViewCreate`] — the target could not be
    ///   created; every partially created object is released first.
    pub fn new(
        context: &'p PipelineContext,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<Self, CompositorError> {
        let device = context.device();
        let image = create_color_image(context, width, height, format)?;
        let memory = match allocate_image_memory(context, image) {
            Ok(memory) => memory,
            Err(err) => {
                // SAFETY: `image` is bound to nothing yet.
                unsafe { device.destroy_image(image, None) };
                return Err(err);
            }
        };
        let view = match create_color_view(context, image, format) {
            Ok(view) => view,
            Err(err) => {
                // SAFETY: both objects were created above and nothing
                // else references them.
                unsafe {
                    device.free_memory(memory, None);
                    device.destroy_image(image, None);
                }
                return Err(err);
            }
        };
        Ok(Self {
            device,
            image,
            memory,
            view,
            width,
            height,
            format,
        })
    }

    /// The target as seen by the compositor.
    #[must_use]
    pub const fn framebuffer(&self) -> OffscreenFramebuffer {
        OffscreenFramebuffer {
            image: self.image,
            view: self.view,
            format: self.format,
            width: self.width,
            height: self.height,
        }
    }

    /// The color image.
    #[inline]
    #[must_use]
    pub const fn image(&self) -> vk::Image {
        self.image
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

    /// Format of the image.
    #[inline]
    #[must_use]
    pub const fn format(&self) -> vk::Format {
        self.format
    }
}

impl Drop for OffscreenTarget<'_> {
    fn drop(&mut self) {
        // SAFETY: the view, memory and image were created by
        // `OffscreenTarget::new` on this device, nothing else references
        // them, and `'p` keeps the device alive for the target's whole
        // lifetime.
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Mix weight sanitization.

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
