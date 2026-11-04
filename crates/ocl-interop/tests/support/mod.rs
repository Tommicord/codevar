// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

//! A real Vulkan device and a real OpenCL device, for the tests that need
//! drivers rather than logic.
//!
//! Every entry point here answers `None`, or an empty list, when the machine
//! has no loader, no driver or no device that offers the extension, so a test
//! that needs hardware skips instead of failing on a headless runner.
//!
//! The module is compiled into every integration target, and no single one of
//! them uses the whole of it.
#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::raw::c_void;

use ash::vk;
use codevar_ocl_interop::{MemoryKind, SemaphoreKind, Vulkan};

/// The two fd-based extensions a share needs, in the spelling the loader
/// expects in `ppEnabledExtensionNames`.
pub const EXTERNAL_EXTENSIONS: [&CStr; 2] = [c"VK_KHR_external_memory_fd", c"VK_KHR_external_semaphore_fd"];

/// An open handle to the platform Vulkan loader.
///
/// The instance and the device made through it are only valid while the
/// library stays mapped, so a test owns one alongside them. The handle is
/// never closed: a driver keeps pointers into its own image after
/// `vkDestroyInstance`, which is why the loader is left mapped for the whole
/// process, exactly as the production renderer does it.
pub struct Library {
    _handle: *mut c_void,
}

/// The loader and the entry points it exports, or `None` without a driver.
fn loader() -> Option<(Library, ash::Entry)> {
    let name = CString::new("libvulkan.so.1").ok()?;
    // SAFETY: `name` is a valid NUL-terminated path and a failed `dlopen` is
    // reported by a null handle, which is never dereferenced.
    let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        return None;
    }
    // SAFETY: `dlsym` reports either null or the address of the symbol it
    // names, and `vkGetInstanceProcAddr` has the canonical
    // `PFN_vkGetInstanceProcAddr` signature.
    let symbol = unsafe { libc::dlsym(handle, c"vkGetInstanceProcAddr".as_ptr()) };
    if symbol.is_null() {
        return None;
    }
    let get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr = unsafe { core::mem::transmute(symbol) };
    // SAFETY: the pointer is the loader's own entry point and the library is
    // never closed, so it stays valid for the returned entry.
    let entry = unsafe {
        ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    };
    Some((Library { _handle: handle }, entry))
}

/// A logical Vulkan device of one physical device of this machine.
pub struct Machine {
    _library: Library,
    instance: ash::Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    queue: vk::Queue,
}

impl Machine {
    /// Every device the loader reports that has a queue and, when `external`
    /// is set, both fd extensions.
    pub fn all(external: bool) -> Vec<Self> {
        let Some(total) = physical_device_count() else {
            return Vec::new();
        };
        (0..total)
            .filter_map(|index| Self::at(external, index))
            .collect()
    }

    /// The first suitable device, or `None` when the machine has none.
    pub fn first(external: bool) -> Option<Self> {
        Self::all(external).into_iter().next()
    }

    /// The `index`th physical device, when it is suitable.
    fn at(external: bool, index: usize) -> Option<Self> {
        let (library, entry) = loader()?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"codevar-interop-test")
            .api_version(vk::API_VERSION_1_1);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // SAFETY: `app_info` outlives `instance_info`, both are well-formed,
        // and the entry came from a mapped loader.
        let instance = unsafe { entry.create_instance(&instance_info, None) }.ok()?;
        // SAFETY: `instance` is live and the driver fills the returned list.
        let physicals = unsafe { instance.enumerate_physical_devices() }.ok()?;
        let physical = *physicals.get(index)?;
        let wanted: &[&CStr] = if external { &EXTERNAL_EXTENSIONS } else { &[] };
        let family = queue_family(&instance, physical, wanted)?;
        let device = create_device(&instance, physical, family, wanted).ok()?;
        // SAFETY: `family` is the index the device was created with, so queue
        // zero of it exists and lives as long as the device.
        let queue = unsafe { device.get_device_queue(family, 0) };
        Some(Self {
            _library: library,
            instance,
            device,
            physical,
            queue,
        })
    }

    /// The endpoint under test, over this device's entry points.
    pub fn endpoint(&self) -> Vulkan<'_> {
        // SAFETY: `instance` is the instance `device` was created from and
        // both stay live for as long as `&self` does, which is what
        // `Vulkan::new` requires of its caller.
        unsafe { Vulkan::new(&self.instance, &self.device, self.physical) }
    }

    /// Allocates `size` bytes of exportable device memory bound to a storage
    /// buffer, or `None` when the driver refuses the combination.
    pub fn allocate(&self, kind: MemoryKind, size: u64) -> Option<Allocation<'_>> {
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(kind.vk_flag());
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        // SAFETY: the pNext chain outlives the call and the device is live.
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }.ok()?;
        // SAFETY: `buffer` was created by `device`.
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        // SAFETY: `physical` belongs to `instance`.
        let properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical)
        };
        let memory_type = (0..properties.memory_type_count).find(|index| {
            let ty = properties.memory_types[*index as usize];
            requirements.memory_type_bits & (1u64 << index) as u32 != 0
                && ty
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })?;
        // A dedicated allocation is what a dma-buf consumer expects and is
        // always permitted, so the driver's `DEDICATED_ONLY` flag is not
        // consulted first.
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(kind.vk_flag());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size.max(size))
            .memory_type_index(memory_type)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        // SAFETY: the chain outlives the call, the type index came from this
        // buffer's own requirements and the size covers them.
        let memory = unsafe { self.device.allocate_memory(&allocate_info, None) }.ok()?;
        // SAFETY: `memory` satisfies `buffer`'s requirements, so binding it at
        // offset zero is sound.
        if unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
            // SAFETY: nothing is bound to `memory` any more, and `buffer` is
            // still only the object this call made.
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
            }
            return None;
        }
        Some(Allocation {
            machine: self,
            buffer,
            memory,
        })
    }

    /// Signals `semaphore` on this device's queue and waits for the
    /// submission to finish, so a handle exported from it afterwards names an
    /// already signaled payload.
    ///
    /// A real share is always signaled by the exporter before the other API
    /// takes it, and a driver relies on that: its `vkDestroySemaphore` waits
    /// for the payload the semaphore still holds, so an unsignaled binary
    /// semaphore imported from a descriptor never stops waiting. `None` when
    /// the queue refuses the submission, which leaves the semaphore unsignaled
    /// and the test unable to hand it over safely.
    pub fn signal(&self, semaphore: vk::Semaphore) -> Option<()> {
        let submits = [vk::SubmitInfo::default().signal_semaphores(core::slice::from_ref(&semaphore))];
        // SAFETY: `queue` belongs to `device` and `semaphore` was created by
        // it, `submits` outlives the call and a null fence asks for none.
        unsafe {
            self.device
                .queue_submit(self.queue, &submits, vk::Fence::default())
        }
        .ok()?;
        // SAFETY: `queue` is one of `device`'s own queues.
        unsafe { self.device.queue_wait_idle(self.queue) }.ok()?;
        Some(())
    }

    /// Creates a semaphore that may be exported as `kind`, or a plain one
    /// when `kind` is `None`, or `None` when the driver refuses.
    pub fn semaphore(&self, kind: Option<SemaphoreKind>) -> Option<VulkanSemaphore<'_>> {
        let mut export =
            kind.map(|kind| vk::ExportSemaphoreCreateInfo::default().handle_types(kind.vk_flag()));
        let info = match export.as_mut() {
            Some(export) => vk::SemaphoreCreateInfo::default().push_next(export),
            None => vk::SemaphoreCreateInfo::default(),
        };
        // SAFETY: the chain outlives the call and the device is live.
        let semaphore = unsafe { self.device.create_semaphore(&info, None) }.ok()?;
        Some(VulkanSemaphore {
            machine: self,
            semaphore,
        })
    }
}

impl Drop for Machine {
    fn drop(&mut self) {
        // The device goes first: an instance cannot be destroyed while one of
        // its devices is still alive.
        // SAFETY: both handles were made by `self.instance`. The allocations
        // and semaphores of a machine are dropped before it, because they
        // borrow it, so nothing still refers to the device.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// The number of physical devices the loader reports; the counting instance
/// is destroyed again right away.
fn physical_device_count() -> Option<usize> {
    let (_library, entry) = loader()?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"codevar-interop-test")
        .api_version(vk::API_VERSION_1_1);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    // SAFETY: as for `Machine::at`, the chain is well-formed and outlives the
    // call.
    let instance = unsafe { entry.create_instance(&instance_info, None) }.ok()?;
    // SAFETY: `instance` is live.
    let count = unsafe { instance.enumerate_physical_devices() }
        .ok()
        .map(|physicals| physicals.len());
    // SAFETY: no device of this instance was created, so it can go now.
    unsafe { instance.destroy_instance(None) };
    count
}

/// A queue family index to create a device with, when `physical` offers every
/// extension in `wanted` and has a queue.
fn queue_family(instance: &ash::Instance, physical: vk::PhysicalDevice, wanted: &[&CStr]) -> Option<u32> {
    // SAFETY: `physical` came from `instance`'s own enumeration.
    let available = unsafe { instance.enumerate_device_extension_properties(physical) }
        .ok()?
        .iter()
        .map(|properties| {
            // SAFETY: `extension_name` is a fixed NUL-terminated array the
            // driver filled, so reading to the terminator stays inside it.
            unsafe { CStr::from_ptr(properties.extension_name.as_ptr()) }.to_owned()
        })
        .collect::<Vec<CString>>();
    if !wanted
        .iter()
        .all(|name| available.iter().any(|found| found == *name))
    {
        return None;
    }
    // SAFETY: as above, `physical` belongs to `instance`.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    families
        .iter()
        .position(|family| family.queue_count > 0)
        .map(|index| index as u32)
}

/// Creates the logical device with `wanted` enabled.
fn create_device(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    family: u32,
    wanted: &[&CStr],
) -> Result<ash::Device, vk::Result> {
    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities);
    let names: Vec<*const core::ffi::c_char> = wanted.iter().map(|name| name.as_ptr()).collect();
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(core::slice::from_ref(&queue_info))
        .enabled_extension_names(&names);
    // SAFETY: `queue_info`, `priorities` and `names` all outlive the call, and
    // `physical` belongs to `instance`.
    unsafe { instance.create_device(physical, &device_info, None) }
}

/// Device memory bound to a buffer, given back when the test ends.
pub struct Allocation<'m> {
    machine: &'m Machine,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl Allocation<'_> {
    /// The memory to export.
    #[must_use]
    pub const fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }
}

impl Drop for Allocation<'_> {
    fn drop(&mut self) {
        // SAFETY: both handles belong to `machine.device`; destroying the
        // buffer unbinds the memory, which is then free to go.
        unsafe {
            self.machine
                .device
                .destroy_buffer(self.buffer, None);
            self.machine.device.free_memory(self.memory, None);
        }
    }
}

/// A Vulkan semaphore, destroyed when the test ends.
pub struct VulkanSemaphore<'m> {
    machine: &'m Machine,
    semaphore: vk::Semaphore,
}

impl VulkanSemaphore<'_> {
    /// The semaphore to export or to import into.
    #[must_use]
    pub const fn handle(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for VulkanSemaphore<'_> {
    fn drop(&mut self) {
        // SAFETY: the semaphore belongs to `machine.device`. A test that
        // submits work with it waits for the queue to drain first — see
        // `Machine::signal` — so no pending operation refers to it here.
        unsafe {
            self.machine
                .device
                .destroy_semaphore(self.semaphore, None)
        };
    }
}

/// Closes a descriptor this process still owns, because the import on the
/// other side only read it — or refused it.
///
/// The one handle a caller must never close here is a semaphore descriptor a
/// successful `Vulkan::import_semaphore_fd` took over.
///
/// # Safety
///
/// `fd` must be a descriptor this process owns and no other owner may hold —
/// a descriptor is an index into this process's file table, so closing one
/// twice closes whatever took its slot.
pub unsafe fn close_descriptor(fd: i32) {
    drop(unsafe { OwnedFd::from_raw_fd(fd) });
}
