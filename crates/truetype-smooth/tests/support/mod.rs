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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! A real Vulkan device that can export memory, for the tests that check
//! the shared coverage path against a driver.
//!
//! Everything here answers `None` when the machine has no loader, no device
//! or no exportable allocation, so a test that needs drivers skips instead of
//! failing on a headless runner.
//!
//! The allocation is asked for as host-visible and coherent memory because
//! the test reads the glyph's bytes back through the Vulkan side to prove
//! where they landed; a renderer would use device-local-only memory and never
//! map it. That difference is about how the test observes the memory, not
//! about who owns it: on both, the exporting device holds the allocation the
//! whole time and OpenCL only borrows it.

use std::ffi::{CStr, CString};
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::raw::c_void;

use ash::vk;
use codevar_ocl_interop::{MemoryKind, Vulkan};

/// The one device extension an fd memory export needs.
const EXTERNAL_EXTENSIONS: [&CStr; 1] = [c"VK_KHR_external_memory_fd"];

/// An open handle to the platform Vulkan loader.
///
/// The instance and device made through it are valid only while the library
/// stays mapped, so a machine owns one alongside them and never closes it: a
/// driver keeps pointers into its own image after `vkDestroyInstance`.
struct Library {
    _handle: *mut c_void,
}

/// The loader and its entry points, or `None` without a driver.
fn loader() -> Option<(Library, ash::Entry)> {
    let name = CString::new("libvulkan.so.1").ok()?;
    // SAFETY: `name` is a valid NUL-terminated path and a failed `dlopen` is
    // reported by a null handle, which is never dereferenced.
    let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        return None;
    }
    // SAFETY: `dlsym` reports either null or the address of the symbol it
    // names, and `vkGetInstanceProcAddr` has the canonical signature.
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

/// A logical Vulkan device over one physical device of this machine.
pub struct Machine {
    _library: Library,
    instance: ash::Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
}

impl Machine {
    /// Every device that exports memory through a file descriptor.
    ///
    /// One instance per device, as [`at`] builds them: the alternative is a
    /// shared instance that the first machine's `Drop` destroys under the
    /// others.
    pub fn all() -> Vec<Self> {
        let Some(total) = physical_device_count() else {
            return Vec::new();
        };
        (0..total).filter_map(at).collect()
    }

    /// The endpoint over this device's entry points.
    pub fn endpoint(&self) -> Vulkan<'_> {
        // SAFETY: `instance` is the instance `device` was created from and
        // both stay live for as long as `&self` does, which is what
        // `Vulkan::new` requires of its caller.
        unsafe { Vulkan::new(&self.instance, &self.device, self.physical) }
    }

    /// Allocates `size` bytes of exportable memory bound to a storage buffer,
    /// mapped by the test to read the coverage back, or `None` when the driver
    /// refuses the combination.
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
        // Host-visible and coherent, so the test's mapping needs no flush;
        // device-local so the writing GPU, not the CPU, owns the access.
        let wanted = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type = (0..properties.memory_type_count).find(|index| {
            let ty = properties.memory_types[*index as usize];
            requirements.memory_type_bits & (1u64 << index) as u32 != 0 && ty.property_flags.contains(wanted)
        })?;
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
            size: requirements.size.max(size),
        })
    }
}

/// The `index`th physical device of this machine, when it can export memory
/// through a file descriptor.
fn at(index: usize) -> Option<Machine> {
    let (library, entry) = loader()?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"codevar-truetype-smooth-test")
        .api_version(vk::API_VERSION_1_1);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    // SAFETY: `app_info` outlives `instance_info`, both are well-formed, and
    // the entry came from a mapped loader.
    let Ok(instance) = (unsafe { entry.create_instance(&instance_info, None) }) else {
        return None;
    };
    // SAFETY: `instance` is live and the driver fills the returned list.
    let physicals = match unsafe { instance.enumerate_physical_devices() } {
        Ok(physicals) => physicals,
        Err(error) => {
            // SAFETY: no device of this instance was created.
            unsafe { instance.destroy_instance(None) };
            let _ = error;
            return None;
        }
    };
    let Some(&physical) = physicals.get(index) else {
        // SAFETY: as above, nothing refers to the instance any more.
        unsafe { instance.destroy_instance(None) };
        return None;
    };
    let outcome = queue_family(&instance, physical)
        .map(|family| create_device(&instance, physical, family))
        .transpose();
    let device = match outcome {
        Ok(Some(device)) => device,
        // SAFETY: no device was created, or its creation failed.
        _ => {
            unsafe { instance.destroy_instance(None) };
            return None;
        }
    };
    Some(Machine {
        _library: library,
        instance,
        device,
        physical,
    })
}

/// The number of physical devices the loader reports; the counting instance
/// is destroyed again right away.
fn physical_device_count() -> Option<usize> {
    let (_library, entry) = loader()?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"codevar-truetype-smooth-test")
        .api_version(vk::API_VERSION_1_1);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    // SAFETY: as for `at`, the chain is well-formed and outlives the call.
    let instance = unsafe { entry.create_instance(&instance_info, None) }.ok()?;
    // SAFETY: `instance` is live.
    let count = unsafe { instance.enumerate_physical_devices() }
        .ok()
        .map(|physicals| physicals.len());
    // SAFETY: no device of this instance was created, so it can go now.
    unsafe { instance.destroy_instance(None) };
    count
}

impl Drop for Machine {
    fn drop(&mut self) {
        // The device goes first: an instance cannot be destroyed while one of
        // its devices is still alive.
        // SAFETY: both handles were made by `self.instance`. Allocations are
        // dropped before a machine, because they borrow it.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Device memory bound to a buffer, given back when the test ends.
pub struct Allocation<'m> {
    machine: &'m Machine,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

impl Allocation<'_> {
    /// The memory to export.
    #[must_use]
    pub const fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    /// Bytes of the allocation, which is what the pipeline's import is told
    /// it may write.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.size as usize
    }

    /// Reads `len` bytes of this allocation the way its owner does: through a
    /// mapping of the Vulkan memory, with no OpenCL read involved.
    pub fn read(&self, len: usize) -> Option<Vec<u8>> {
        if len as u64 > self.size {
            return None;
        }
        // SAFETY: the memory was allocated as host-visible and coherent, so
        // the mapping covers `self.size` bytes that this call may read, and
        // `unmap_memory` runs before the memory is freed.
        let mapped = unsafe {
            self.machine
                .device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
        }
        .ok()?;
        let bytes = unsafe { core::slice::from_raw_parts(mapped.cast::<u8>(), len) }.to_vec();
        // SAFETY: `mapped` came from the `map_memory` call above and is not
        // used again after this.
        unsafe { self.machine.device.unmap_memory(self.memory) };
        Some(bytes)
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

/// A queue family index to create a device with, when `physical` offers every
/// external-memory extension and has a queue.
fn queue_family(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Option<u32> {
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
    if !EXTERNAL_EXTENSIONS
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

/// Creates the logical device with the export extension enabled.
fn create_device(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    family: u32,
) -> Result<ash::Device, vk::Result> {
    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities);
    let names: Vec<*const core::ffi::c_char> = EXTERNAL_EXTENSIONS
        .iter()
        .map(|name| name.as_ptr())
        .collect();
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(core::slice::from_ref(&queue_info))
        .enabled_extension_names(&names);
    // SAFETY: `queue_info`, `priorities` and `names` all outlive the call, and
    // `physical` belongs to `instance`.
    unsafe { instance.create_device(physical, &device_info, None) }
}

/// Closes a descriptor this process still owns, because the import on the
/// other side only read it — or refused it.
///
/// # Safety
///
/// `fd` must be a descriptor this process owns and no other owner may hold —
/// a descriptor is an index into this process's file table, so closing one
/// twice closes whatever took its slot.
pub unsafe fn close_descriptor(fd: i32) {
    drop(unsafe { OwnedFd::from_raw_fd(fd) });
}
