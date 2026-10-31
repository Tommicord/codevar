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

//! The Vulkan half of the exchange.
//!
//! The memory and semaphore entry points this crate needs belong to
//! `VK_KHR_external_memory_fd` and `VK_KHR_external_semaphore_fd`. Both
//! extensions are old enough that their *structures* are part of Vulkan 1.1,
//! but their *functions* never were, so `ash`'s device table does not carry
//! them and a call needs three steps:
//!
//! 1. ask the instance for the device entry point by name, which a driver
//!    answers with a null pointer when the extension is not enabled on that
//!    device;
//! 2. keep the resolved pointer, so a share that runs every frame does not
//!    look the name up again;
//! 3. call it through the raw signature, with the `VkDevice` handle the
//!    pointer was resolved for.
//!
//! `ash` offers [`ash::khr::external_memory_fd::Device`] for step two, but
//! its loader fills in a missing entry point with a function that
//! `panic!`s, which is not an error a library may hand to a caller — so this
//! module resolves the names itself and reports a null pointer as
//! [`Error::Unsupported`](crate::Error::Unsupported).
//!
//! [`Vulkan`] is that cache. It borrows the instance and the device of the
//! caller and creates nothing: an application that already owns a Vulkan
//! device hands those handles over, and this type only ever answers
//! questions about them (which handle types can be exported, which can be
//! imported) or moves a handle across the boundary.

use core::ffi::{CStr, c_int};

use ash::vk;
use codevar_ocl::Luid;

use crate::error::{Error, Result};
use crate::handle::{MemoryHandle, MemoryKind, SemaphoreHandle, SemaphoreKind};
use crate::pair::Identity;

/// One device entry point, identified by the name it is resolved under.
trait Entry: Copy {
    /// The exact spelling `vkGetDeviceProcAddr` expects.
    const NAME: &'static CStr;
    /// What is missing when the driver answers that name with a null
    /// pointer.
    const EXTENSION: &'static str;

    /// # Safety
    ///
    /// `raw` must be the pointer the loader answered for this trait's
    /// [`NAME`](Self::NAME): either null or the address of a function with
    /// exactly the signature of `Self`.
    unsafe fn from_raw(raw: vk::PFN_vkVoidFunction) -> Option<Self>;
}

impl Entry for vk::PFN_vkGetMemoryFdKHR {
    const NAME: &CStr = c"vkGetMemoryFdKHR";
    const EXTENSION: &str = "VK_KHR_external_memory_fd is not enabled on the Vulkan device";

    unsafe fn from_raw(raw: vk::PFN_vkVoidFunction) -> Option<Self> {
        // SAFETY: the caller guarantees `raw` is this entry point. Both
        // types are one nullable function pointer, so the layout is the
        // same and only the signature differs.
        unsafe { core::mem::transmute::<vk::PFN_vkVoidFunction, Option<Self>>(raw) }
    }
}

impl Entry for vk::PFN_vkGetSemaphoreFdKHR {
    const NAME: &CStr = c"vkGetSemaphoreFdKHR";
    const EXTENSION: &str = "VK_KHR_external_semaphore_fd is not enabled on the Vulkan device";

    unsafe fn from_raw(raw: vk::PFN_vkVoidFunction) -> Option<Self> {
        // SAFETY: as for `vkGetMemoryFdKHR` above.
        unsafe { core::mem::transmute::<vk::PFN_vkVoidFunction, Option<Self>>(raw) }
    }
}

impl Entry for vk::PFN_vkImportSemaphoreFdKHR {
    const NAME: &CStr = c"vkImportSemaphoreFdKHR";
    const EXTENSION: &str = "VK_KHR_external_semaphore_fd is not enabled on the Vulkan device";

    unsafe fn from_raw(raw: vk::PFN_vkVoidFunction) -> Option<Self> {
        // SAFETY: as for `vkGetMemoryFdKHR` above.
        unsafe { core::mem::transmute::<vk::PFN_vkVoidFunction, Option<Self>>(raw) }
    }
}

/// The file descriptor of a handle, when that handle is one.
///
/// `vkImportSemaphoreFdKHR` reads a `sync_file` or an opaque descriptor and
/// nothing else, so a Win32 handle or a D3D12 fence has no answer here: the
/// entry point that would accept it is a different extension.
const fn fd_of(handle: SemaphoreHandle) -> Option<(SemaphoreKind, c_int)> {
    match handle {
        SemaphoreHandle::OpaqueFd(fd) | SemaphoreHandle::SyncFd(fd) => Some((handle.kind(), fd)),
        SemaphoreHandle::OpaqueWin32(_)
        | SemaphoreHandle::OpaqueWin32Kmt(_)
        | SemaphoreHandle::D3D12Fence(_) => None,
    }
}

/// The external-memory and external-semaphore entry points of one device.
///
/// The value is `Copy`, because it is three pointers plus two handles, and a
/// share needs all of them on both sides of a thread boundary.
#[derive(Clone, Copy)]
pub struct Vulkan<'a> {
    instance: &'a ash::Instance,
    device: vk::Device,
    physical: vk::PhysicalDevice,
    get_memory_fd: Option<vk::PFN_vkGetMemoryFdKHR>,
    get_semaphore_fd: Option<vk::PFN_vkGetSemaphoreFdKHR>,
    import_semaphore_fd: Option<vk::PFN_vkImportSemaphoreFdKHR>,
}

impl<'a> Vulkan<'a> {
    /// Resolves every entry point of `device` once, for as long as
    /// `instance` is borrowed.
    ///
    /// A driver that has not enabled one of the two extensions answers with
    /// a null pointer, which is kept and reported by whichever call needs
    /// it. Construction never fails: the capability questions below — and
    /// the pairing question of [`crate`](crate) — are answerable on a device
    /// with neither extension.
    ///
    /// # Safety
    ///
    /// `instance` must be the instance `device` was created from, and
    /// `device` must stay live for the whole lifetime of the returned value:
    /// its entry points are read on every call, and a driver answers for a
    /// device it has already destroyed with undefined results.
    pub unsafe fn new(
        instance: &'a ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Self {
        let handle = device.handle();
        Self {
            instance,
            device: handle,
            physical,
            get_memory_fd: Self::resolve::<vk::PFN_vkGetMemoryFdKHR>(instance, handle),
            get_semaphore_fd: Self::resolve::<vk::PFN_vkGetSemaphoreFdKHR>(instance, handle),
            import_semaphore_fd: Self::resolve::<vk::PFN_vkImportSemaphoreFdKHR>(instance, handle),
        }
    }

    /// Asks the instance for one device entry point by name.
    fn resolve<T: Entry>(instance: &ash::Instance, device: vk::Device) -> Option<T> {
        // SAFETY: `vkGetDeviceProcAddr` reads nothing but the name it is
        // handed, and `device` is a live handle of `instance` — the
        // precondition `new` documents.
        let raw = unsafe { instance.get_device_proc_addr(device, T::NAME.as_ptr()) };
        // SAFETY: the pointer was answered for `T::NAME`, so it is null or
        // the function that name names.
        unsafe { T::from_raw(raw) }
    }

    /// Turns a resolved entry point into a result, naming the extension
    /// that would have provided it.
    fn entry<T: Entry>(slot: Option<T>) -> Result<T> {
        slot.ok_or(Error::Unsupported { what: T::EXTENSION })
    }

    /// Exports `memory` as `kind` (`vkGetMemoryFdKHR`), the handle an
    /// OpenCL buffer is then created from.
    ///
    /// The driver makes a fresh descriptor per call, so the caller owns it:
    /// import it through [`MemoryHandle::as_opencl`] and close it once the
    /// other API has taken its own reference.
    ///
    /// # Errors
    ///
    /// - [`Error::Unsupported`] when the device has not enabled
    ///   `VK_KHR_external_memory_fd`, or when `kind` is one of the Win32
    ///   handles, which no descriptor names.
    /// - [`Error::Vulkan`] when the driver refused the export.
    pub fn export_memory(&self, memory: vk::DeviceMemory, kind: MemoryKind) -> Result<MemoryHandle> {
        // Decided before the driver is reached: `VkMemoryGetFdInfoKHR`
        // accepts the two descriptor handle types and nothing else.
        if !kind.is_descriptor() {
            return Err(Error::Unsupported {
                what: "vkGetMemoryFdKHR exports a dma-buf or an opaque descriptor, not a Win32 handle",
            });
        }
        let get_memory_fd = Self::entry(self.get_memory_fd)?;
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(kind.vk_flag());
        // Initialised, because the descriptor is only ever read after the
        // call said it wrote one.
        let mut fd: c_int = -1;
        // SAFETY: `self.device` is the device `get_memory_fd` was resolved
        // for, `info` is a live `VkMemoryGetFdInfoKHR` whose `memory` member
        // the caller owns, and `fd` is a writable `int`.
        let code = unsafe { get_memory_fd(self.device, &info, &mut fd) };
        Error::check(code, "vkGetMemoryFdKHR")?;
        Ok(if kind == MemoryKind::DmaBuf {
            MemoryHandle::DmaBuf(fd)
        } else {
            MemoryHandle::OpaqueFd(fd)
        })
    }

    /// Exports `semaphore` as `kind` (`vkGetSemaphoreFdKHR`), the handle an
    /// OpenCL semaphore is then created from.
    ///
    /// Ownership works exactly as in [`export_memory`](Self::export_memory).
    ///
    /// # Errors
    ///
    /// - [`Error::Unsupported`] when the device has not enabled
    ///   `VK_KHR_external_semaphore_fd`, or when `kind` is not a descriptor.
    /// - [`Error::Vulkan`] when the driver refused the export.
    pub fn export_semaphore(&self, semaphore: vk::Semaphore, kind: SemaphoreKind) -> Result<SemaphoreHandle> {
        if !kind.is_descriptor() {
            return Err(Error::Unsupported {
                what: "vkGetSemaphoreFdKHR exports an opaque or a sync_file descriptor, not a Win32 handle",
            });
        }
        let get_semaphore_fd = Self::entry(self.get_semaphore_fd)?;
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(kind.vk_flag());
        let mut fd: c_int = -1;
        // SAFETY: as for `vkGetMemoryFdKHR`, with a live
        // `VkSemaphoreGetFdInfoKHR` in place of the memory one.
        let code = unsafe { get_semaphore_fd(self.device, &info, &mut fd) };
        Error::check(code, "vkGetSemaphoreFdKHR")?;
        Ok(if kind == SemaphoreKind::SyncFd {
            SemaphoreHandle::SyncFd(fd)
        } else {
            SemaphoreHandle::OpaqueFd(fd)
        })
    }

    /// Makes `semaphore` wait on and signal like `handle`
    /// (`vkImportSemaphoreFdKHR`), which is how a semaphore created by
    /// OpenCL is signaled into a Vulkan submission.
    ///
    /// Importing does not consume the descriptor: the driver takes its own
    /// reference, and the caller closes the one it passed.
    ///
    /// # Errors
    ///
    /// - [`Error::Unsupported`] for a handle that is not a descriptor, and
    ///   for a device without `VK_KHR_external_semaphore_fd`.
    /// - [`Error::Vulkan`] when the driver refused the handle.
    pub fn import_semaphore_fd(&self, semaphore: vk::Semaphore, handle: SemaphoreHandle) -> Result<()> {
        let (kind, fd) = fd_of(handle).ok_or(Error::Unsupported {
            what: "vkImportSemaphoreFdKHR takes a file descriptor, not a Win32 handle",
        })?;
        let import_semaphore_fd = Self::entry(self.import_semaphore_fd)?;
        let info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore)
            // An opaque share is imported permanently, which is what a share
            // that outlives this call needs. A `sync_file` has no permanent
            // form: the extension supports only a temporary import of it, and
            // a driver that validates the flags refuses anything else.
            .flags(if kind == SemaphoreKind::SyncFd {
                vk::SemaphoreImportFlags::TEMPORARY
            } else {
                vk::SemaphoreImportFlags::empty()
            })
            .handle_type(kind.vk_flag())
            .fd(fd);
        // SAFETY: `self.device` is the device the entry point was resolved
        // for and `info` names a live semaphore of that device.
        let code = unsafe { import_semaphore_fd(self.device, &info) };
        Error::check(code, "vkImportSemaphoreFdKHR")
    }

    /// Reads the identity the physical device reports for itself
    /// (`VkPhysicalDeviceIDProperties`).
    ///
    /// A driver that cannot name its hardware answers with a block of
    /// zeros, which is kept as it is: [`Identity::pairs_with`] decides what
    /// a placeholder means, so no pairing rule lives in two places.
    #[must_use]
    pub fn identity(&self) -> Identity {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        // SAFETY: `physical` is the physical device of the borrowed
        // instance, and the structure chain is well-formed and lives until
        // the end of this block; the driver only writes into it.
        unsafe {
            self.instance
                .get_physical_device_properties2(self.physical, &mut properties)
        };
        Identity {
            device_uuid: id.device_uuid,
            driver_uuid: id.driver_uuid,
            luid: (id.device_luid_valid == vk::TRUE).then_some(Luid {
                bytes: id.device_luid,
                node_mask: id.device_node_mask,
            }),
        }
    }

    /// Whether a buffer used as storage can be exported as `kind`
    /// (`vkGetPhysicalDeviceExternalBufferProperties`).
    ///
    /// The answer is per *usage*, not per device, so a false here means only
    /// that this kind is not shareable as storage memory — the usage this
    /// crate shares buffers with.
    #[must_use]
    pub fn can_export_memory(&self, kind: MemoryKind) -> bool {
        self.memory_features(kind)
            .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
    }

    /// Whether a semaphore can be imported from `kind`
    /// (`vkGetPhysicalDeviceExternalSemaphoreProperties`).
    #[must_use]
    pub fn can_import_semaphore(&self, kind: SemaphoreKind) -> bool {
        self.semaphore_features(kind)
            .contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE)
    }

    /// Whether a semaphore can be exported as `kind`.
    #[must_use]
    pub fn can_export_semaphore(&self, kind: SemaphoreKind) -> bool {
        self.semaphore_features(kind)
            .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
    }

    fn memory_features(&self, kind: MemoryKind) -> vk::ExternalMemoryFeatureFlags {
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .flags(vk::BufferCreateFlags::empty())
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .handle_type(kind.vk_flag());
        let mut properties = vk::ExternalBufferProperties::default();
        // SAFETY: `physical` is the physical device of the borrowed
        // instance, and both structures are live and correctly typed for the
        // whole call; the driver only writes into `properties`.
        unsafe {
            self.instance
                .get_physical_device_external_buffer_properties(self.physical, &info, &mut properties);
        }
        properties
            .external_memory_properties
            .external_memory_features
    }

    fn semaphore_features(&self, kind: SemaphoreKind) -> vk::ExternalSemaphoreFeatureFlags {
        let info = vk::PhysicalDeviceExternalSemaphoreInfo::default().handle_type(kind.vk_flag());
        let mut properties = vk::ExternalSemaphoreProperties::default();
        // SAFETY: as for the buffer query above.
        unsafe {
            self.instance
                .get_physical_device_external_semaphore_properties(self.physical, &info, &mut properties);
        }
        properties.external_semaphore_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::MemoryHandle;

    /// The fd-only rule of `vkImportSemaphoreFdKHR`: a `HANDLE` reaches this
    /// crate as a `usize` and would be narrowed into a descriptor slot,
    /// which the driver then reads as a different object entirely.
    #[test]
    fn only_a_descriptor_handle_can_be_imported_as_an_fd() {
        assert_eq!(
            fd_of(SemaphoreHandle::OpaqueFd(7)),
            Some((SemaphoreKind::OpaqueFd, 7))
        );
        assert_eq!(
            fd_of(SemaphoreHandle::SyncFd(-1)),
            Some((SemaphoreKind::SyncFd, -1))
        );
        for handle in [
            SemaphoreHandle::OpaqueWin32(0x1_0000_0000),
            SemaphoreHandle::OpaqueWin32Kmt(0x1_0000_0000),
            SemaphoreHandle::D3D12Fence(0x1_0000_0000),
        ] {
            assert_eq!(fd_of(handle), None, "{handle:?} is not a descriptor");
        }
    }

    /// A descriptor keeps its sign through the crossing, so a driver handed
    /// `-1` reads `-1` rather than the huge number a truncation would leave.
    #[test]
    fn a_descriptor_of_either_api_crosses_the_same_way() {
        assert_eq!(MemoryHandle::DmaBuf(-1).raw(), SemaphoreHandle::SyncFd(-1).raw());
        assert_eq!(fd_of(SemaphoreHandle::SyncFd(-1)).map(|(_, fd)| fd), Some(-1));
    }

    /// The cache is three function pointers plus two handles, and a share is
    /// set up on the thread that submits to each API, so the value has to be
    /// movable and usable from both.
    #[test]
    fn the_endpoint_can_be_shared_between_threads() {
        fn assert_send_sync<T: Send + Sync + Copy>() {}
        assert_send_sync::<Vulkan<'_>>();
    }
}
