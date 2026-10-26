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

//! Mirror of the vendor ICD dispatch table (`struct _cl_icd_dispatch`).
//!
//! Vendor ICD libraries (Intel NEO `libigdrcl.so`, AMD `libamdocl64.so`,
//! NVIDIA `libnvidia-opencl.so.1`, pocl, …) do **not** export the `cl*` API
//! as symbols; they only export `clGetExtensionFunctionAddress` and
//! `clIcdGetPlatformIDsKHR`. All other calls must go through the dispatch
//! table whose pointer is stored as the first field of every platform
//! handle — that is exactly how the Khronos ICD loader dispatches into
//! vendors, and it is the mechanism [`crate::icd`] uses for direct vendor
//! loading.
//!
//! # Layout
//!
//! The struct mirrors `CL/cl_icd.h` field-for-field. Every slot is exactly
//! one pointer wide on every supported target — function pointers, the
//! `intptr_t` `cl_khr_icd` 2.0 tags (which share a slot with
//! `get_platform_ids` and `unload_compiler` via anonymous unions in the C
//! header) and the `void *` placeholders for GL/D3D/EGL/SVM entry points this
//! crate does not use — so the layout is identical across `CL_VERSION_*`
//! and `_WIN32` branches of the header. Unused slots are typed
//! `*mut c_void` and must never be dereferenced; they exist purely to keep
//! the offsets of the used slots correct.
//!
//! Slots that this crate consumes are typed `Option<fn>` so that a null
//! dispatch entry maps to `None` instead of an invalid function pointer.
//!
//! # Reference
//!
//! * <https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_icd.h>
//! * <https://github.com/KhronosGroup/OpenCL-ICD-Loader/blob/main/loader/icd_dispatch.h>

use core::ffi::c_void;
#[cfg(test)]
use core::mem::{offset_of, size_of};

use crate::sys::{
    BuildProgram, CreateBuffer, CreateCommandQueue, CreateContext, CreateKernel, CreateProgramWithIL,
    CreateProgramWithSource, EnqueueCopyBuffer, EnqueueFillBuffer, EnqueueNdRangeKernel, EnqueueReadBuffer,
    EnqueueWriteBuffer, Finish, Flush, GetDeviceIds, GetDeviceInfo, GetEventInfo, GetEventProfilingInfo,
    GetExtensionFunctionAddress, GetKernelInfo, GetKernelWorkGroupInfo, GetPlatformIds, GetPlatformInfo,
    GetProgramBuildInfo, GetProgramInfo, PlatformHandle, ReleaseCommandQueue, ReleaseContext, ReleaseEvent,
    ReleaseKernel, ReleaseMemObject, ReleaseProgram, SetKernelArg, WaitForEvents,
};

/// The vendor ICD dispatch table (`cl_icd_dispatch` in `CL/cl_icd.h`).
///
/// Instances of this type are never constructed by Rust code; they live
/// inside the driver and are read through [`dispatch_of`].
#[repr(C)]
pub(crate) struct IcdDispatch {
    /// Slot 0 (shares its slot with the `cl_khr_icd` 2.0 tag union).
    pub(crate) get_platform_ids: Option<GetPlatformIds>,
    /// Slot 1.
    pub(crate) get_platform_info: Option<GetPlatformInfo>,
    /// Slot 2.
    pub(crate) get_device_ids: Option<GetDeviceIds>,
    /// Slot 3.
    pub(crate) get_device_info: Option<GetDeviceInfo>,
    /// Slot 4.
    pub(crate) create_context: Option<CreateContext>,
    /// Slot 5: `clCreateContextFromType` (unused).
    pub(crate) create_context_from_type: *mut c_void,
    /// Slot 6: `clRetainContext` (unused; the safe layer never retains).
    pub(crate) retain_context: *mut c_void,
    /// Slot 7.
    pub(crate) release_context: Option<ReleaseContext>,
    /// Slot 8: `clGetContextInfo` (unused).
    pub(crate) get_context_info: *mut c_void,
    /// Slot 9.
    pub(crate) create_command_queue: Option<CreateCommandQueue>,
    /// Slot 10: `clRetainCommandQueue` (unused).
    pub(crate) retain_command_queue: *mut c_void,
    /// Slot 11.
    pub(crate) release_command_queue: Option<ReleaseCommandQueue>,
    /// Slot 12: `clGetCommandQueueInfo` (unused).
    pub(crate) get_command_queue_info: *mut c_void,
    /// Slot 13: `clSetCommandQueueProperty` (deprecated, unused).
    pub(crate) set_command_queue_property: *mut c_void,
    /// Slot 14.
    pub(crate) create_buffer: Option<CreateBuffer>,
    /// Slot 15: `clCreateImage2D` (deprecated, unused).
    pub(crate) create_image2d: *mut c_void,
    /// Slot 16: `clCreateImage3D` (deprecated, unused).
    pub(crate) create_image3d: *mut c_void,
    /// Slot 17: `clRetainMemObject` (unused).
    pub(crate) retain_mem_object: *mut c_void,
    /// Slot 18.
    pub(crate) release_mem_object: Option<ReleaseMemObject>,
    /// Slot 19: `clGetSupportedImageFormats` (unused).
    pub(crate) get_supported_image_formats: *mut c_void,
    /// Slot 20: `clGetMemObjectInfo` (unused).
    pub(crate) get_mem_object_info: *mut c_void,
    /// Slot 21: `clGetImageInfo` (unused).
    pub(crate) get_image_info: *mut c_void,
    /// Slot 22: `clCreateSampler` (unused).
    pub(crate) create_sampler: *mut c_void,
    /// Slot 23: `clRetainSampler` (unused).
    pub(crate) retain_sampler: *mut c_void,
    /// Slot 24: `clReleaseSampler` (unused).
    pub(crate) release_sampler: *mut c_void,
    /// Slot 25: `clGetSamplerInfo` (unused).
    pub(crate) get_sampler_info: *mut c_void,
    /// Slot 26.
    pub(crate) create_program_with_source: Option<CreateProgramWithSource>,
    /// Slot 27: `clCreateProgramWithBinary` (unused).
    pub(crate) create_program_with_binary: *mut c_void,
    /// Slot 28: `clRetainProgram` (unused).
    pub(crate) retain_program: *mut c_void,
    /// Slot 29.
    pub(crate) release_program: Option<ReleaseProgram>,
    /// Slot 30.
    pub(crate) build_program: Option<BuildProgram>,
    /// Slot 31: `clUnloadCompiler` (shares its slot with the icd2 tag union).
    pub(crate) unload_compiler: *mut c_void,
    /// Slot 32.
    pub(crate) get_program_info: Option<GetProgramInfo>,
    /// Slot 33.
    pub(crate) get_program_build_info: Option<GetProgramBuildInfo>,
    /// Slot 34.
    pub(crate) create_kernel: Option<CreateKernel>,
    /// Slot 35: `clCreateKernelsInProgram` (unused).
    pub(crate) create_kernels_in_program: *mut c_void,
    /// Slot 36: `clRetainKernel` (unused).
    pub(crate) retain_kernel: *mut c_void,
    /// Slot 37.
    pub(crate) release_kernel: Option<ReleaseKernel>,
    /// Slot 38.
    pub(crate) set_kernel_arg: Option<SetKernelArg>,
    /// Slot 39.
    pub(crate) get_kernel_info: Option<GetKernelInfo>,
    /// Slot 40.
    pub(crate) get_kernel_work_group_info: Option<GetKernelWorkGroupInfo>,
    /// Slot 41.
    pub(crate) wait_for_events: Option<WaitForEvents>,
    /// Slot 42.
    pub(crate) get_event_info: Option<GetEventInfo>,
    /// Slot 43: `clRetainEvent` (unused).
    pub(crate) retain_event: *mut c_void,
    /// Slot 44.
    pub(crate) release_event: Option<ReleaseEvent>,
    /// Slot 45.
    pub(crate) get_event_profiling_info: Option<GetEventProfilingInfo>,
    /// Slot 46.
    pub(crate) flush: Option<Flush>,
    /// Slot 47.
    pub(crate) finish: Option<Finish>,
    /// Slot 48.
    pub(crate) enqueue_read_buffer: Option<EnqueueReadBuffer>,
    /// Slot 49.
    pub(crate) enqueue_write_buffer: Option<EnqueueWriteBuffer>,
    /// Slot 50.
    pub(crate) enqueue_copy_buffer: Option<EnqueueCopyBuffer>,
    /// Slot 51: `clEnqueueReadImage` (unused).
    pub(crate) enqueue_read_image: *mut c_void,
    /// Slot 52: `clEnqueueWriteImage` (unused).
    pub(crate) enqueue_write_image: *mut c_void,
    /// Slot 53: `clEnqueueCopyImage` (unused).
    pub(crate) enqueue_copy_image: *mut c_void,
    /// Slot 54: `clEnqueueCopyImageToBuffer` (unused).
    pub(crate) enqueue_copy_image_to_buffer: *mut c_void,
    /// Slot 55: `clEnqueueCopyBufferToImage` (unused).
    pub(crate) enqueue_copy_buffer_to_image: *mut c_void,
    /// Slot 56: `clEnqueueMapBuffer` (unused).
    pub(crate) enqueue_map_buffer: *mut c_void,
    /// Slot 57: `clEnqueueMapImage` (unused).
    pub(crate) enqueue_map_image: *mut c_void,
    /// Slot 58: `clEnqueueUnmapMemObject` (unused).
    pub(crate) enqueue_unmap_mem_object: *mut c_void,
    /// Slot 59.
    pub(crate) enqueue_nd_range_kernel: Option<EnqueueNdRangeKernel>,
    /// Slot 60: `clEnqueueTask` (deprecated, unused).
    pub(crate) enqueue_task: *mut c_void,
    /// Slot 61: `clEnqueueNativeKernel` (unused).
    pub(crate) enqueue_native_kernel: *mut c_void,
    /// Slot 62: `clEnqueueMarker` (deprecated, unused).
    pub(crate) enqueue_marker: *mut c_void,
    /// Slot 63: `clEnqueueWaitForEvents` (deprecated, unused).
    pub(crate) enqueue_wait_for_events: *mut c_void,
    /// Slot 64: `clEnqueueBarrier` (deprecated, unused).
    pub(crate) enqueue_barrier: *mut c_void,
    /// Slot 65.
    pub(crate) get_extension_function_address: Option<GetExtensionFunctionAddress>,
    /// Slot 66: `clCreateFromGLBuffer` (unused).
    pub(crate) create_from_gl_buffer: *mut c_void,
    /// Slot 67: `clCreateFromGLTexture2D` (unused).
    pub(crate) create_from_gl_texture2d: *mut c_void,
    /// Slot 68: `clCreateFromGLTexture3D` (unused).
    pub(crate) create_from_gl_texture3d: *mut c_void,
    /// Slot 69: `clCreateFromGLRenderbuffer` (unused).
    pub(crate) create_from_gl_renderbuffer: *mut c_void,
    /// Slot 70: `clGetGLObjectInfo` (unused).
    pub(crate) get_gl_object_info: *mut c_void,
    /// Slot 71: `clGetGLTextureInfo` (unused).
    pub(crate) get_gl_texture_info: *mut c_void,
    /// Slot 72: `clEnqueueAcquireGLObjects` (unused).
    pub(crate) enqueue_acquire_gl_objects: *mut c_void,
    /// Slot 73: `clEnqueueReleaseGLObjects` (unused).
    pub(crate) enqueue_release_gl_objects: *mut c_void,
    /// Slot 74: `clGetGLContextInfoKHR` (unused).
    pub(crate) get_gl_context_info_khr: *mut c_void,
    /// Slot 75 (unused).
    pub(crate) get_device_ids_from_d3d10_khr: *mut c_void,
    /// Slot 76 (unused).
    pub(crate) create_from_d3d10_buffer_khr: *mut c_void,
    /// Slot 77 (unused).
    pub(crate) create_from_d3d10_texture2d_khr: *mut c_void,
    /// Slot 78 (unused).
    pub(crate) create_from_d3d10_texture3d_khr: *mut c_void,
    /// Slot 79 (unused).
    pub(crate) enqueue_acquire_d3d10_objects_khr: *mut c_void,
    /// Slot 80 (unused).
    pub(crate) enqueue_release_d3d10_objects_khr: *mut c_void,
    /// Slot 81: `clSetEventCallback` (unused).
    pub(crate) set_event_callback: *mut c_void,
    /// Slot 82: `clCreateSubBuffer` (unused).
    pub(crate) create_sub_buffer: *mut c_void,
    /// Slot 83: `clSetMemObjectDestructorCallback` (unused).
    pub(crate) set_mem_object_destructor_callback: *mut c_void,
    /// Slot 84: `clCreateUserEvent` (unused).
    pub(crate) create_user_event: *mut c_void,
    /// Slot 85: `clSetUserEventStatus` (unused).
    pub(crate) set_user_event_status: *mut c_void,
    /// Slot 86: `clEnqueueReadBufferRect` (unused).
    pub(crate) enqueue_read_buffer_rect: *mut c_void,
    /// Slot 87: `clEnqueueWriteBufferRect` (unused).
    pub(crate) enqueue_write_buffer_rect: *mut c_void,
    /// Slot 88: `clEnqueueCopyBufferRect` (unused).
    pub(crate) enqueue_copy_buffer_rect: *mut c_void,
    /// Slot 89 (unused).
    pub(crate) create_sub_devices_ext: *mut c_void,
    /// Slot 90 (unused).
    pub(crate) retain_device_ext: *mut c_void,
    /// Slot 91 (unused).
    pub(crate) release_device_ext: *mut c_void,
    /// Slot 92 (unused).
    pub(crate) create_event_from_glsync_khr: *mut c_void,
    /// Slot 93: `clCreateSubDevices` (unused).
    pub(crate) create_sub_devices: *mut c_void,
    /// Slot 94: `clRetainDevice` (unused).
    pub(crate) retain_device: *mut c_void,
    /// Slot 95: `clReleaseDevice` (unused).
    pub(crate) release_device: *mut c_void,
    /// Slot 96: `clCreateImage` (unused).
    pub(crate) create_image: *mut c_void,
    /// Slot 97: `clCreateProgramWithBuiltInKernels` (unused).
    pub(crate) create_program_with_built_in_kernels: *mut c_void,
    /// Slot 98: `clCompileProgram` (unused).
    pub(crate) compile_program: *mut c_void,
    /// Slot 99: `clLinkProgram` (unused).
    pub(crate) link_program: *mut c_void,
    /// Slot 100: `clUnloadPlatformCompiler` (unused).
    pub(crate) unload_platform_compiler: *mut c_void,
    /// Slot 101: `clGetKernelArgInfo` (unused).
    pub(crate) get_kernel_arg_info: *mut c_void,
    /// Slot 102.
    pub(crate) enqueue_fill_buffer: Option<EnqueueFillBuffer>,
    /// Slot 103: `clEnqueueFillImage` (unused).
    pub(crate) enqueue_fill_image: *mut c_void,
    /// Slot 104: `clEnqueueMigrateMemObjects` (unused).
    pub(crate) enqueue_migrate_mem_objects: *mut c_void,
    /// Slot 105: `clEnqueueMarkerWithWaitList` (unused).
    pub(crate) enqueue_marker_with_wait_list: *mut c_void,
    /// Slot 106: `clEnqueueBarrierWithWaitList` (unused).
    pub(crate) enqueue_barrier_with_wait_list: *mut c_void,
    /// Slot 107: `clGetExtensionFunctionAddressForPlatform` (unused).
    pub(crate) get_extension_function_address_for_platform: *mut c_void,
    /// Slot 108: `clCreateFromGLTexture` (unused).
    pub(crate) create_from_gl_texture: *mut c_void,
    /// Slot 109 (unused).
    pub(crate) get_device_ids_from_d3d11_khr: *mut c_void,
    /// Slot 110 (unused).
    pub(crate) create_from_d3d11_buffer_khr: *mut c_void,
    /// Slot 111 (unused).
    pub(crate) create_from_d3d11_texture2d_khr: *mut c_void,
    /// Slot 112 (unused).
    pub(crate) create_from_d3d11_texture3d_khr: *mut c_void,
    /// Slot 113 (unused).
    pub(crate) create_from_dx9_media_surface_khr: *mut c_void,
    /// Slot 114 (unused).
    pub(crate) enqueue_acquire_d3d11_objects_khr: *mut c_void,
    /// Slot 115 (unused).
    pub(crate) enqueue_release_d3d11_objects_khr: *mut c_void,
    /// Slot 116 (unused).
    pub(crate) get_device_ids_from_dx9_media_adapter_khr: *mut c_void,
    /// Slot 117 (unused).
    pub(crate) enqueue_acquire_dx9_media_surfaces_khr: *mut c_void,
    /// Slot 118 (unused).
    pub(crate) enqueue_release_dx9_media_surfaces_khr: *mut c_void,
    /// Slot 119 (unused).
    pub(crate) create_from_egl_image_khr: *mut c_void,
    /// Slot 120 (unused).
    pub(crate) enqueue_acquire_egl_objects_khr: *mut c_void,
    /// Slot 121 (unused).
    pub(crate) enqueue_release_egl_objects_khr: *mut c_void,
    /// Slot 122 (unused).
    pub(crate) create_from_egl_sync_khr: *mut c_void,
    /// Slot 123: `clCreateCommandQueueWithProperties` (unused; the safe
    /// layer uses the OpenCL 1.0 `clCreateCommandQueue` for portability).
    pub(crate) create_command_queue_with_properties: *mut c_void,
    /// Slot 124: `clCreatePipe` (unused).
    pub(crate) create_pipe: *mut c_void,
    /// Slot 125: `clGetPipeInfo` (unused).
    pub(crate) get_pipe_info: *mut c_void,
    /// Slot 126: `clSVMAlloc` (unused).
    pub(crate) svm_alloc: *mut c_void,
    /// Slot 127: `clSVMFree` (unused).
    pub(crate) svm_free: *mut c_void,
    /// Slot 128: `clEnqueueSVMFree` (unused).
    pub(crate) enqueue_svm_free: *mut c_void,
    /// Slot 129: `clEnqueueSVMMemcpy` (unused).
    pub(crate) enqueue_svm_memcpy: *mut c_void,
    /// Slot 130: `clEnqueueSVMMemFill` (unused).
    pub(crate) enqueue_svm_mem_fill: *mut c_void,
    /// Slot 131: `clEnqueueSVMMap` (unused).
    pub(crate) enqueue_svm_map: *mut c_void,
    /// Slot 132: `clEnqueueSVMUnmap` (unused).
    pub(crate) enqueue_svm_unmap: *mut c_void,
    /// Slot 133: `clCreateSamplerWithProperties` (unused).
    pub(crate) create_sampler_with_properties: *mut c_void,
    /// Slot 134: `clSetKernelArgSVMPointer` (unused).
    pub(crate) set_kernel_arg_svm_pointer: *mut c_void,
    /// Slot 135: `clSetKernelExecInfo` (unused).
    pub(crate) set_kernel_exec_info: *mut c_void,
    /// Slot 136 (unused).
    pub(crate) get_kernel_sub_group_info_khr: *mut c_void,
    /// Slot 137: `clCloneKernel` (unused).
    pub(crate) clone_kernel: *mut c_void,
    /// Slot 138.
    pub(crate) create_program_with_il: Option<CreateProgramWithIL>,
    /// Slot 139: `clEnqueueSVMMigrateMem` (unused).
    pub(crate) enqueue_svm_migrate_mem: *mut c_void,
    /// Slot 140: `clGetDeviceAndHostTimer` (unused).
    pub(crate) get_device_and_host_timer: *mut c_void,
    /// Slot 141: `clGetHostTimer` (unused).
    pub(crate) get_host_timer: *mut c_void,
    /// Slot 142: `clGetKernelSubGroupInfo` (unused).
    pub(crate) get_kernel_sub_group_info: *mut c_void,
    /// Slot 143: `clSetDefaultDeviceCommandQueue` (unused).
    pub(crate) set_default_device_command_queue: *mut c_void,
    /// Slot 144: `clSetProgramReleaseCallback` (unused).
    pub(crate) set_program_release_callback: *mut c_void,
    /// Slot 145: `clSetProgramSpecializationConstant` (unused).
    pub(crate) set_program_specialization_constant: *mut c_void,
    /// Slot 146: `clCreateBufferWithProperties` (unused).
    pub(crate) create_buffer_with_properties: *mut c_void,
    /// Slot 147: `clCreateImageWithProperties` (unused).
    pub(crate) create_image_with_properties: *mut c_void,
    /// Slot 148: `clSetContextDestructorCallback` (unused).
    pub(crate) set_context_destructor_callback: *mut c_void,
    /// Slot 149: `clGetKernelSuggestedLocalWorkSize` (unused).
    pub(crate) get_kernel_suggested_local_work_size: *mut c_void,
}

/// Reads the dispatch-table pointer out of an ICD platform handle.
///
/// The ICD ABI guarantees that the first word of every platform object is a
/// pointer to the implementation's `cl_icd_dispatch` table; this is how the
/// Khronos loader and `ocl-icd` dispatch every call.
///
/// # Safety
///
/// `platform` must be a platform handle that was produced by
/// `clIcdGetPlatformIDsKHR` (or `clGetPlatformIDs` on a non-ICD
/// implementation) of a library that is still loaded. The returned pointer
/// must only be read while that library remains mapped.
#[inline]
pub(crate) unsafe fn dispatch_of(platform: PlatformHandle) -> *const IcdDispatch {
    // SAFETY: the caller guarantees the handle is a live ICD platform object,
    // whose first field is the dispatch-table pointer. Reading a pointer-sized
    // word from it therefore yields a valid (possibly null) table pointer.
    unsafe {
        platform
            .as_raw()
            .cast::<*const IcdDispatch>()
            .read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;

    /// Size of one dispatch slot: every field of `cl_icd_dispatch` is
    /// pointer-width.
    const SLOT: usize = size_of::<*mut c_void>();

    /// Asserts that a used field sits at the offset its `cl_icd.h` position
    /// requires. A mismatch means the field order drifted from the header
    /// and every call would jump into the wrong entry point.
    #[test]
    fn used_slots_match_the_cl_icd_h_field_order() {
        assert_eq!(offset_of!(IcdDispatch, get_platform_ids), 0);
        assert_eq!(offset_of!(IcdDispatch, get_platform_info), SLOT);
        assert_eq!(offset_of!(IcdDispatch, get_device_ids), 2 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, get_device_info), 3 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, create_context), 4 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, release_context), 7 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, create_command_queue), 9 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, release_command_queue), 11 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, create_buffer), 14 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, release_mem_object), 18 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, create_program_with_source), 26 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, release_program), 29 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, build_program), 30 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, get_program_info), 32 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, get_program_build_info), 33 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, create_kernel), 34 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, release_kernel), 37 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, set_kernel_arg), 38 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, get_kernel_info), 39 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, get_kernel_work_group_info), 40 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, wait_for_events), 41 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, get_event_info), 42 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, release_event), 44 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, get_event_profiling_info), 45 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, flush), 46 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, finish), 47 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, enqueue_read_buffer), 48 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, enqueue_write_buffer), 49 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, enqueue_copy_buffer), 50 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, enqueue_nd_range_kernel), 59 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, get_extension_function_address), 65 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, enqueue_fill_buffer), 102 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, create_program_with_il), 138 * SLOT);
        assert_eq!(offset_of!(IcdDispatch, enqueue_marker_with_wait_list), 105 * SLOT);
        assert_eq!(
            offset_of!(IcdDispatch, create_command_queue_with_properties),
            123 * SLOT
        );
        assert_eq!(offset_of!(IcdDispatch, svm_alloc), 126 * SLOT);
        assert_eq!(
            offset_of!(IcdDispatch, create_sampler_with_properties),
            133 * SLOT
        );
        assert_eq!(
            offset_of!(IcdDispatch, set_context_destructor_callback),
            148 * SLOT
        );
        assert_eq!(
            offset_of!(IcdDispatch, get_kernel_suggested_local_work_size),
            149 * SLOT
        );
    }

    /// The table is exactly the 150 pointer-wide slots of `cl_icd_dispatch`.
    #[test]
    fn table_is_150_pointer_sized_slots() {
        assert_eq!(size_of::<IcdDispatch>(), 150 * SLOT);
    }

    /// `dispatch_of` must return the table pointer stored in word 0 of the
    /// platform object.
    #[test]
    fn dispatch_of_reads_the_first_word_of_the_platform_handle() {
        // SAFETY: zeroing is valid for every field — `Option<fn>` has a null
        // "None" representation and raw pointers accept null — so the
        // resulting value is fully initialized.
        let table: Box<IcdDispatch> = unsafe { Box::new(core::mem::MaybeUninit::zeroed().assume_init()) };

        // A minimal stand-in for a driver platform object: its first word is
        // the dispatch-table pointer, exactly like the ICD ABI requires.
        let mut platform_object: *const IcdDispatch = &*table;
        let platform = PlatformHandle::from_raw((&raw mut platform_object).cast());

        // SAFETY: `platform` points at a live word that holds a pointer to
        // the boxed table, which stays alive for the whole test.
        let read = unsafe { dispatch_of(platform) };
        assert_eq!(read, &*table as *const IcdDispatch);

        // SAFETY: `read` points at the boxed table kept alive above, whose
        // slot was zero-initialized to `None`.
        let is_none = unsafe { (*read).get_platform_ids.is_none() };
        assert!(is_none);
    }
}
