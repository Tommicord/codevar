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

//! Frame-level GPU resources: the command buffers, semaphore chain and
//! fence allocated by the compositor, the frame's import/export
//! synchronization, the `sync_file` helpers and the image layout
//! barriers recorded by the frame passes.

use alloc::vec::Vec;
use ash::vk;

use crate::base::{Compositor, CompositorError};
use crate::mix::MixResources;
use crate::pipeline::{OwnedFd, PipelineContext};

/// Everything [`Compositor::start`](Compositor::start) allocates, released on drop.
///
/// `quiesce` must have run before this is dropped so no handle is still
/// referenced by the queue.
pub(crate) struct StartResources<'p> {
    /// Pipeline backing every handle below.
    pub(crate) context: &'p PipelineContext,
    /// One primary command buffer per queue slot.
    pub(crate) pipe_command_buffers: Vec<vk::CommandBuffer>,
    /// Command buffer of the frame's layout pass.
    pub(crate) layout_command_buffer: vk::CommandBuffer,
    /// Command buffer of the frame's mix pass.
    pub(crate) mix_command_buffer: vk::CommandBuffer,
    /// Semaphore chain: `chain_sems[0]` is signalled by the layout pass,
    /// pipe *k* waits on `chain_sems[k]` and signals `chain_sems[k + 1]`.
    pub(crate) chain_sems: Vec<vk::Semaphore>,
    /// Fence of the whole chain, signalled by the mix submission.
    pub(crate) frame_fence: vk::Fence,
    /// Descriptor set, sampler and pipeline of the mix pass.
    pub(crate) mix: MixResources<'p>,
}

impl<'p> StartResources<'p> {
    /// Allocates the command buffers, the semaphore chain, the frame
    /// fence and the mix resources.
    ///
    /// # Errors
    ///
    /// * [`CompositorError::CommandAllocation`], [`CompositorError::SemaphoreCreate`],
    ///   [`CompositorError::FenceCreate`] — a resource could not be
    ///   created; everything allocated before the failure is released by
    ///   this struct's `Drop`.
    pub(crate) fn create(
        context: &'p PipelineContext,
        pipe_count: usize,
        max_framebuffers: u32,
    ) -> Result<Self, CompositorError> {
        let device = context.device();
        let mut resources = Self {
            context,
            pipe_command_buffers: Vec::with_capacity(pipe_count),
            layout_command_buffer: vk::CommandBuffer::null(),
            mix_command_buffer: vk::CommandBuffer::null(),
            chain_sems: Vec::with_capacity(pipe_count + 1),
            frame_fence: vk::Fence::null(),
            mix: MixResources::create(context, max_framebuffers)?,
        };
        if pipe_count > 0 {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(context.command_pool())
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(pipe_count as u32);
            // SAFETY: the pool belongs to `context` and is alive for `'p`.
            let allocated = unsafe { device.allocate_command_buffers(&allocate_info) }
                .map_err(CompositorError::CommandAllocation)?;
            if allocated.len() != pipe_count {
                if !allocated.is_empty() {
                    // SAFETY: the buffers were just allocated from this pool.
                    unsafe { device.free_command_buffers(context.command_pool(), &allocated) };
                }
                return Err(CompositorError::Internal(
                    "the driver allocated an unexpected number of command buffers",
                ));
            }
            resources.pipe_command_buffers = allocated;
        }

        // Allocate layout and mix command buffers
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(context.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(2);
        // SAFETY: the pool belongs to `context` and is alive for `'p`.
        let allocated = unsafe { device.allocate_command_buffers(&allocate_info) }
            .map_err(CompositorError::CommandAllocation)?;
        if allocated.len() != 2 {
            if !allocated.is_empty() {
                // SAFETY: the buffers were just allocated from this pool.
                unsafe { device.free_command_buffers(context.command_pool(), &allocated) };
            }
            return Err(CompositorError::Internal(
                "the driver allocated an unexpected number of command buffers",
            ));
        }
        resources.layout_command_buffer = allocated[0];
        resources.mix_command_buffer = allocated[1];

        // Allocate semaphore chain (one for layout pass + one per pipe)
        let semaphore_count = pipe_count + 1;
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        for _ in 0..semaphore_count {
            // SAFETY: a plain, well-formed create info.
            let semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }
                .map_err(CompositorError::SemaphoreCreate)?;
            resources.chain_sems.push(semaphore);
        }

        // SAFETY: a plain, well-formed create info; the fence starts
        // unsignaled and is waited on by [`Compositor::begin_frame`](super::Compositor::begin_frame).
        resources.frame_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(CompositorError::FenceCreate)?;
        Ok(resources)
    }
}

impl Drop for StartResources<'_> {
    fn drop(&mut self) {
        let context = self.context;
        let device = context.device();
        // SAFETY: `quiesce` waits for the device before this struct is
        // dropped, so no fence, semaphore or command buffer is still
        // referenced by the queue; null handles are never passed on.
        unsafe {
            if self.frame_fence != vk::Fence::null() {
                device.destroy_fence(self.frame_fence, None);
            }
            for semaphore in &self.chain_sems {
                if *semaphore != vk::Semaphore::null() {
                    device.destroy_semaphore(*semaphore, None);
                }
            }
            if !self.pipe_command_buffers.is_empty() {
                device.free_command_buffers(context.command_pool(), &self.pipe_command_buffers);
            }
            if self.layout_command_buffer != vk::CommandBuffer::null() {
                device.free_command_buffers(
                    context.command_pool(),
                    core::slice::from_ref(&self.layout_command_buffer),
                );
            }
            if self.mix_command_buffer != vk::CommandBuffer::null() {
                device.free_command_buffers(
                    context.command_pool(),
                    core::slice::from_ref(&self.mix_command_buffer),
                );
            }
        }
    }
}

/// Synchronization objects of the frame in progress.
pub(crate) struct FrameSync<'p> {
    /// Device that owns both semaphores.
    pub(crate) device: &'p ash::Device,
    /// Wait semaphore importing the caller's `sync_file`; null when the
    /// frame waits on nothing.
    pub(crate) import_sem: vk::Semaphore,
    /// Signal semaphore exported as this frame's `sync_file`.
    pub(crate) present_sem: vk::Semaphore,
}

impl Drop for FrameSync<'_> {
    fn drop(&mut self) {
        // SAFETY: the compositor destroys the frame sync either after
        // the frame fence was waited on or after `device_wait_idle`, so
        // neither semaphore is still referenced by the queue; null
        // handles are skipped.
        unsafe {
            if self.import_sem != vk::Semaphore::null() {
                self.device
                    .destroy_semaphore(self.import_sem, None);
            }
            if self.present_sem != vk::Semaphore::null() {
                self.device
                    .destroy_semaphore(self.present_sem, None);
            }
        }
    }
}

/// Clear color used when a pipe cannot supply one (its target still
/// holds `UNDEFINED` contents).
pub(crate) const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Creates a binary semaphore, optionally with SYNC_FD export enabled.
pub(crate) fn create_binary_semaphore(
    device: &ash::Device,
    export_sync_fd: bool,
) -> Result<vk::Semaphore, CompositorError> {
    let mut export_info =
        vk::ExportSemaphoreCreateInfo::default().handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    // The export info must outlive the create info that chains it, so it
    // is built before the `if` rather than inside a branch.
    let semaphore_info = if export_sync_fd {
        vk::SemaphoreCreateInfo::default().push_next(&mut export_info)
    } else {
        vk::SemaphoreCreateInfo::default()
    };
    // SAFETY: the device is valid and, when exporting, the
    // `VkExportSemaphoreCreateInfo` chain is alive for the call.
    unsafe { device.create_semaphore(&semaphore_info, None) }.map_err(CompositorError::SemaphoreCreate)
}

/// Imports a `sync_file` fd into `semaphore` as a temporary SYNC_FD
/// payload.
///
/// On success ownership of `sync_file` transfers to the implementation;
/// on failure the descriptor is closed here.
pub(crate) fn import_sync_fd(
    context: &PipelineContext,
    semaphore: vk::Semaphore,
    sync_file: OwnedFd,
) -> Result<(), CompositorError> {
    let raw_fd = sync_file.into_raw();
    let import_info = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(semaphore)
        // VUID-VkImportSemaphoreFdInfoKHR-handleType-07307: copy-
        // transference handle types (SYNC_FD) must be TEMPORARY.
        .flags(vk::SemaphoreImportFlags::TEMPORARY)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        .fd(raw_fd);
    // SAFETY: `semaphore` is a fresh binary semaphore (VUID-
    // vkImportSemaphoreFdKHR-semaphore-01142), SYNC_FD import was
    // verified as supported when the pipeline context was created, and
    // the import info is well-formed.
    let result = unsafe {
        context
            .external_semaphore_fd()
            .import_semaphore_fd(&import_info)
    };
    if let Err(err) = result {
        // The implementation only takes ownership on success; rebuild an
        // `OwnedFd` so the descriptor closes on this error path.
        // SAFETY: the fd was moved out above and not consumed because
        // the import failed.
        drop(unsafe { OwnedFd::from_raw(raw_fd) });
        return Err(CompositorError::SemaphoreImport(err));
    }
    Ok(())
}

/// Exports a signaled (or signaling) binary semaphore as a `sync_file`
/// fd representing GPU completion.
pub(crate) fn export_sync_fd(
    context: &PipelineContext,
    semaphore: vk::Semaphore,
) -> Result<OwnedFd, CompositorError> {
    let get_info = vk::SemaphoreGetFdInfoKHR::default()
        .semaphore(semaphore)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    // SAFETY: the semaphore was created with SYNC_FD in
    // `VkExportSemaphoreCreateInfo` and is signalled or has a pending
    // signal operation (it was just submitted), which satisfies
    // VUID-VkSemaphoreGetFdInfoKHR-handleType-01132/01135.
    let raw_fd = unsafe {
        context
            .external_semaphore_fd()
            .get_semaphore_fd(&get_info)
    }
    .map_err(CompositorError::SemaphoreExport)?;
    if raw_fd < 0 {
        return Err(CompositorError::Internal(
            "vkGetSemaphoreFdKHR completed without producing a sync_file fd",
        ));
    }
    // SAFETY: on success ownership of the new descriptor transfers to
    // the caller.
    Ok(unsafe { OwnedFd::from_raw(raw_fd) })
}

/// One image layout transition recorded by [`record_image_transition`].
#[derive(Clone, Copy)]
pub(crate) struct ImageTransition {
    /// Image whose whole subresource is transitioned.
    image: vk::Image,
    /// Layout before the barrier.
    old_layout: vk::ImageLayout,
    /// Layout after the barrier.
    new_layout: vk::ImageLayout,
    /// Pipeline stages whose prior work the barrier waits for.
    src_stage: vk::PipelineStageFlags,
    /// Accesses of `src_stage` made available by the barrier.
    src_access: vk::AccessFlags,
    /// Pipeline stages that observe the transition.
    dst_stage: vk::PipelineStageFlags,
    /// Accesses of `dst_stage` made visible by the barrier.
    dst_access: vk::AccessFlags,
}

impl ImageTransition {
    /// Builds a transition for `image`.
    pub(crate) const fn new(
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_stage: vk::PipelineStageFlags,
        dst_access: vk::AccessFlags,
    ) -> Self {
        Self {
            image,
            old_layout,
            new_layout,
            src_stage,
            src_access,
            dst_stage,
            dst_access,
        }
    }
}

/// Records an image layout transition over the whole subresource.
pub(crate) fn record_image_transition(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    transition: &ImageTransition,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(transition.src_access)
        .dst_access_mask(transition.dst_access)
        .old_layout(transition.old_layout)
        .new_layout(transition.new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(transition.image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: the command buffer is recording, the barrier references a
    // live image of the frame, and both stage and access masks are
    // well-formed for the queue that runs it.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            transition.src_stage,
            transition.dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Source stage and access mask of a layout transition leaving `layout`.
///
/// The masks cover the accesses that produced the current contents; the
/// previous submission's fence was waited on in
/// [`Compositor::begin_frame`](Compositor::begin_frame), so matching the producing stage keeps
/// every barrier valid without over-synchronizing.
pub(crate) fn transition_source(layout: vk::ImageLayout) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match layout {
        vk::ImageLayout::UNDEFINED => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::GENERAL => (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_READ),
        _ => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
    }
}
