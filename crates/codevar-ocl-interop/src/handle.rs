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

//! The handle vocabulary of the exchange.
//!
//! A shared buffer or semaphore crosses as one number — a file descriptor,
//! a `HANDLE` — plus a token that says what kind of number it is. The two
//! APIs spell that token differently: OpenCL uses `cl_*_KHR` enumerants
//! from `cl_ext.h`, Vulkan a bit in `VkExternalMemoryHandleTypeFlags`, and
//! the two have no numeric relationship at all. Negotiating the exchange is
//! therefore a lookup over this table before it is a call to either driver.
//!
//! [`MemoryKind`] and [`SemaphoreKind`] are the table: each name maps to the
//! OpenCL enumerant, to the Vulkan flag, and back. [`MemoryHandle`] and
//! [`SemaphoreHandle`] add the value to a kind, which is what a driver
//! actually needs, and hand out the matching handle type of either API.
//!
//! A kind is decided before the value is read: importing a dma-buf with the
//! `OPAQUE_FD` token, or a `sync_file` with the `OPAQUE_FD` semaphore token,
//! makes the driver interpret the descriptor as something it is not.

use ash::vk;
use codevar_ocl::{Device, ExternalMemoryHandle, ExternalSemaphoreHandle, sys};

use crate::error::Result;

/// A way to name a shared allocation across an API boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// A Linux `dma_buf` descriptor, the only handle type a Linux driver can
    /// share without an extra export format (`CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR`,
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`).
    DmaBuf,
    /// An opaque POSIX file descriptor
    /// (`CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR`,
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`).
    OpaqueFd,
    /// A Win32 `HANDLE` to an NT handle
    /// (`CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR`,
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`).
    OpaqueWin32,
    /// A Win32 `HANDLE` to a KMT handle
    /// (`CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR`,
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT`).
    OpaqueWin32Kmt,
}

impl MemoryKind {
    /// The `cl_external_memory_handle_type_khr` enumerant of this kind.
    #[must_use]
    pub const fn cl_token(self) -> sys::ExternalMemoryHandleType {
        match self {
            Self::DmaBuf => sys::EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
            Self::OpaqueFd => sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR,
            Self::OpaqueWin32 => sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR,
            Self::OpaqueWin32Kmt => sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR,
        }
    }

    /// The `VkExternalMemoryHandleTypeFlagBits` of this kind.
    #[must_use]
    pub const fn vk_flag(self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            Self::DmaBuf => vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            Self::OpaqueFd => vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
            Self::OpaqueWin32 => vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32,
            Self::OpaqueWin32Kmt => vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT,
        }
    }

    /// Reads a kind back from an OpenCL enumerant, as reported in a device's
    /// importable-handle list.
    ///
    /// The Win32 *object name* form is not representable: it names a shared
    /// object by string rather than by handle, so no single value crosses.
    #[must_use]
    pub const fn from_cl_token(token: sys::ExternalMemoryHandleType) -> Option<Self> {
        match token {
            sys::EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR => Some(Self::DmaBuf),
            sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR => Some(Self::OpaqueFd),
            sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR => Some(Self::OpaqueWin32),
            sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR => Some(Self::OpaqueWin32Kmt),
            _ => None,
        }
    }

    /// Whether `device` can import a memory handle of this kind
    /// (`CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR`).
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::OpenCl`] when the driver rejects the query,
    /// which is also how a device without `cl_khr_external_memory` answers.
    pub fn importable_by_opencl(self, device: &Device) -> Result<bool> {
        Ok(device
            .external_memory_import_handle_types()?
            .contains(&self.cl_token()))
    }
}

/// A memory handle as it crosses: one [`MemoryKind`] plus the value a driver
/// reads for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryHandle {
    /// A `dma_buf` descriptor.
    DmaBuf(i32),
    /// An opaque POSIX file descriptor.
    OpaqueFd(i32),
    /// A Win32 `HANDLE` to an NT handle.
    OpaqueWin32(usize),
    /// A Win32 `HANDLE` to a KMT handle.
    OpaqueWin32Kmt(usize),
}

impl MemoryHandle {
    /// The kind this handle is imported as.
    #[must_use]
    pub const fn kind(self) -> MemoryKind {
        match self {
            Self::DmaBuf(_) => MemoryKind::DmaBuf,
            Self::OpaqueFd(_) => MemoryKind::OpaqueFd,
            Self::OpaqueWin32(_) => MemoryKind::OpaqueWin32,
            Self::OpaqueWin32Kmt(_) => MemoryKind::OpaqueWin32Kmt,
        }
    }

    /// The value as the receiving API reads it: a descriptor widened with
    /// its sign, a `HANDLE` as the platform stores it.
    #[must_use]
    pub const fn raw(self) -> i64 {
        match self {
            Self::DmaBuf(fd) | Self::OpaqueFd(fd) => fd as i64,
            Self::OpaqueWin32(handle) | Self::OpaqueWin32Kmt(handle) => handle as i64,
        }
    }

    /// The OpenCL half of this handle, ready for
    /// [`Buffer::from_external`](codevar_ocl::Buffer::from_external).
    #[must_use]
    pub const fn as_opencl(self) -> ExternalMemoryHandle<'static> {
        match self {
            Self::DmaBuf(fd) => ExternalMemoryHandle::DmaBuf(fd),
            Self::OpaqueFd(fd) => ExternalMemoryHandle::OpaqueFd(fd),
            Self::OpaqueWin32(handle) => ExternalMemoryHandle::OpaqueWin32(handle),
            Self::OpaqueWin32Kmt(handle) => ExternalMemoryHandle::OpaqueWin32Kmt(handle),
        }
    }
}

/// A way to name a shared signal across an API boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemaphoreKind {
    /// An opaque POSIX file descriptor
    /// (`CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR`,
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT`).
    OpaqueFd,
    /// A Linux `sync_file` descriptor, which is how a Vulkan driver shares a
    /// timeline-free signal
    /// (`CL_SEMAPHORE_HANDLE_SYNC_FD_KHR`,
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`).
    SyncFd,
    /// A Win32 `HANDLE` to an NT handle
    /// (`CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR`,
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT`).
    OpaqueWin32,
    /// A Win32 `HANDLE` to a KMT handle
    /// (`CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR`,
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT`).
    OpaqueWin32Kmt,
    /// A `HANDLE` to a D3D12 fence
    /// (`CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR`,
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT`).
    D3D12Fence,
}

impl SemaphoreKind {
    /// The `cl_external_semaphore_handle_type_khr` enumerant of this kind.
    #[must_use]
    pub const fn cl_token(self) -> sys::ExternalSemaphoreHandleType {
        match self {
            Self::OpaqueFd => sys::SEMAPHORE_HANDLE_OPAQUE_FD_KHR,
            Self::SyncFd => sys::SEMAPHORE_HANDLE_SYNC_FD_KHR,
            Self::OpaqueWin32 => sys::SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR,
            Self::OpaqueWin32Kmt => sys::SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR,
            Self::D3D12Fence => sys::SEMAPHORE_HANDLE_D3D12_FENCE_KHR,
        }
    }

    /// The `VkExternalSemaphoreHandleTypeFlagBits` of this kind.
    #[must_use]
    pub const fn vk_flag(self) -> vk::ExternalSemaphoreHandleTypeFlags {
        match self {
            Self::OpaqueFd => vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD,
            Self::SyncFd => vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD,
            Self::OpaqueWin32 => vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32,
            Self::OpaqueWin32Kmt => vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32_KMT,
            Self::D3D12Fence => vk::ExternalSemaphoreHandleTypeFlags::D3D12_FENCE,
        }
    }

    /// Reads a kind back from an OpenCL enumerant, as reported in a device's
    /// importable-handle list.
    #[must_use]
    pub const fn from_cl_token(token: sys::ExternalSemaphoreHandleType) -> Option<Self> {
        match token {
            sys::SEMAPHORE_HANDLE_OPAQUE_FD_KHR => Some(Self::OpaqueFd),
            sys::SEMAPHORE_HANDLE_SYNC_FD_KHR => Some(Self::SyncFd),
            sys::SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR => Some(Self::OpaqueWin32),
            sys::SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR => Some(Self::OpaqueWin32Kmt),
            sys::SEMAPHORE_HANDLE_D3D12_FENCE_KHR => Some(Self::D3D12Fence),
            _ => None,
        }
    }

    /// Whether `device` can import a semaphore handle of this kind
    /// (`CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR`).
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::OpenCl`] when the driver rejects the query,
    /// which is also how a device without `cl_khr_external_semaphore`
    /// answers.
    pub fn importable_by_opencl(self, device: &Device) -> Result<bool> {
        Ok(device
            .semaphore_import_handle_types()?
            .contains(&self.cl_token()))
    }
}

/// A semaphore handle as it crosses: one [`SemaphoreKind`] plus the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemaphoreHandle {
    /// An opaque POSIX file descriptor.
    OpaqueFd(i32),
    /// A `sync_file` descriptor.
    SyncFd(i32),
    /// A Win32 `HANDLE` to an NT handle.
    OpaqueWin32(usize),
    /// A Win32 `HANDLE` to a KMT handle.
    OpaqueWin32Kmt(usize),
    /// A `HANDLE` to a D3D12 fence.
    D3D12Fence(usize),
}

impl SemaphoreHandle {
    /// The kind this handle is imported as.
    #[must_use]
    pub const fn kind(self) -> SemaphoreKind {
        match self {
            Self::OpaqueFd(_) => SemaphoreKind::OpaqueFd,
            Self::SyncFd(_) => SemaphoreKind::SyncFd,
            Self::OpaqueWin32(_) => SemaphoreKind::OpaqueWin32,
            Self::OpaqueWin32Kmt(_) => SemaphoreKind::OpaqueWin32Kmt,
            Self::D3D12Fence(_) => SemaphoreKind::D3D12Fence,
        }
    }

    /// The value as the receiving API reads it.
    #[must_use]
    pub const fn raw(self) -> i64 {
        match self {
            Self::OpaqueFd(fd) | Self::SyncFd(fd) => fd as i64,
            Self::OpaqueWin32(handle) | Self::OpaqueWin32Kmt(handle) | Self::D3D12Fence(handle) => {
                handle as i64
            }
        }
    }

    /// The OpenCL half of this handle, ready for
    /// [`Semaphore::from_external`](codevar_ocl::Semaphore::from_external).
    #[must_use]
    pub const fn as_opencl(self) -> ExternalSemaphoreHandle {
        match self {
            Self::OpaqueFd(fd) => ExternalSemaphoreHandle::OpaqueFd(fd),
            Self::SyncFd(fd) => ExternalSemaphoreHandle::SyncFd(fd),
            Self::OpaqueWin32(handle) => ExternalSemaphoreHandle::OpaqueWin32(handle),
            Self::OpaqueWin32Kmt(handle) => ExternalSemaphoreHandle::OpaqueWin32Kmt(handle),
            Self::D3D12Fence(handle) => ExternalSemaphoreHandle::D3D12Fence(handle),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two enumerations are unrelated numbers, so the mapping is the
    /// whole negotiation: a table entry copied from the wrong header would
    /// make every import fail with an unhelpful driver error.
    #[test]
    fn memory_kinds_carry_the_numbers_of_their_own_headers() {
        assert_eq!(MemoryKind::DmaBuf.cl_token(), 0x2067);
        assert_eq!(MemoryKind::OpaqueFd.cl_token(), 0x2060);
        assert_eq!(MemoryKind::OpaqueWin32.cl_token(), 0x2061);
        assert_eq!(MemoryKind::OpaqueWin32Kmt.cl_token(), 0x2062);
        // A Vulkan flag has no `Debug` in a `no_std` build, so the flags are
        // compared by equality instead of printed on failure.
        assert!(MemoryKind::DmaBuf.vk_flag() == vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        assert!(MemoryKind::OpaqueFd.vk_flag() == vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        assert!(SemaphoreKind::SyncFd.vk_flag() == vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        assert!(SemaphoreKind::D3D12Fence.vk_flag() == vk::ExternalSemaphoreHandleTypeFlags::D3D12_FENCE);
    }

    /// The same kind read back from the token a device reports must be the
    /// kind that was written, or a capability list is checked against a
    /// different handle than the one exported.
    #[test]
    fn every_kind_round_trips_through_its_opencl_token() {
        for kind in [
            MemoryKind::DmaBuf,
            MemoryKind::OpaqueFd,
            MemoryKind::OpaqueWin32,
            MemoryKind::OpaqueWin32Kmt,
        ] {
            assert_eq!(MemoryKind::from_cl_token(kind.cl_token()), Some(kind));
        }
        for kind in [
            SemaphoreKind::OpaqueFd,
            SemaphoreKind::SyncFd,
            SemaphoreKind::OpaqueWin32,
            SemaphoreKind::OpaqueWin32Kmt,
            SemaphoreKind::D3D12Fence,
        ] {
            assert_eq!(SemaphoreKind::from_cl_token(kind.cl_token()), Some(kind));
        }
    }

    /// A token with no counterpart in Vulkan is refused rather than mapped
    /// to the nearest-looking bit: the Win32 object *name* form names a
    /// shared object by string, and an unknown enumerant is a driver bug.
    #[test]
    fn a_token_without_a_vulkan_counterpart_is_not_a_kind() {
        assert_eq!(
            MemoryKind::from_cl_token(sys::EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_NAME_KHR),
            None
        );
        assert_eq!(MemoryKind::from_cl_token(0xFFFF), None);
        assert_eq!(SemaphoreKind::from_cl_token(0xFFFF), None);
    }

    /// A descriptor is a signed `int` in both APIs and a `HANDLE` is a
    /// pointer, so a value that lost its sign or its width would be read as
    /// a different object.
    #[test]
    fn a_handle_keeps_the_sign_and_width_of_its_platform_slot() {
        assert_eq!(MemoryHandle::DmaBuf(-1).raw(), -1);
        assert_eq!(MemoryHandle::OpaqueFd(7).raw(), 7);
        assert_eq!(MemoryHandle::OpaqueWin32(0x1_0000_0000).raw(), 0x1_0000_0000);
        assert_eq!(SemaphoreHandle::SyncFd(-1).raw(), -1);
        assert_eq!(SemaphoreHandle::D3D12Fence(9).raw(), 9);
    }

    /// The value and its kind travel together: a handle that reported the
    /// wrong kind would be imported under the wrong token.
    #[test]
    fn a_handle_reports_the_kind_it_is_imported_as() {
        assert_eq!(MemoryHandle::DmaBuf(3).kind(), MemoryKind::DmaBuf);
        assert_eq!(MemoryHandle::OpaqueWin32Kmt(3).kind(), MemoryKind::OpaqueWin32Kmt);
        assert_eq!(SemaphoreHandle::SyncFd(3).kind(), SemaphoreKind::SyncFd);
        assert_eq!(SemaphoreHandle::D3D12Fence(3).kind(), SemaphoreKind::D3D12Fence);
        // And the OpenCL wrapper agrees with the kind, which is what the
        // property list of an import is built from.
        assert_eq!(
            MemoryHandle::OpaqueFd(3)
                .as_opencl()
                .handle_type(),
            MemoryKind::OpaqueFd.cl_token()
        );
        assert_eq!(
            SemaphoreHandle::SyncFd(3)
                .as_opencl()
                .handle_type(),
            SemaphoreKind::SyncFd.cl_token()
        );
    }
}
