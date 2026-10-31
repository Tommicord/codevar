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

//! Capability questions a driver answers before objects are created.
//!
//! Sharing memory and semaphores with Vulkan is a negotiation, not a
//! feature flag: the exporter and the importer must agree on one handle
//! type, and both must be looking at the same physical device. This module
//! asks the three questions that decision needs, in the order they are
//! cheap to answer:
//!
//! 1. does this implementation export the extension at all
//!    ([`supports_extension`](Device::supports_extension)),
//! 2. which handle types can it import
//!    ([`external_memory_import_handle_types`](Device::external_memory_import_handle_types),
//!    [`semaphore_import_handle_types`](Device::semaphore_import_handle_types)),
//! 3. which physical device is this ([`uuid`](Device::uuid),
//!    [`luid`](Device::luid)) — the identity a
//!    `VkPhysicalDeviceIDProperties` is compared against.
//!
//! Every question is an information query, so none of them creates an
//! object, allocates driver memory, or needs a context. A driver without
//! the extension answers with a status error, which the caller turns into
//! "not supported" rather than a failure of the program.
//!
//! # Examples
//!
//! ```
//! use codevar_ocl::{Runtime, sys};
//!
//! if let Ok(runtime) = Runtime::load() {
//!     for platform in runtime.platforms() {
//!         for device in platform.all_devices().unwrap_or_default() {
//!             // A driver that cannot import dma-bufs cannot share the
//!             // compositor's images, whatever else it supports.
//!             let dmabuf = device
//!                 .external_memory_import_handle_types()
//!                 .unwrap_or_default()
//!                 .contains(&sys::EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR);
//!             let _ = (device.supports_extension(sys::EXT_EXTERNAL_MEMORY), dmabuf);
//!         }
//!     }
//! }
//! ```

use alloc::vec::Vec;
use core::ffi::c_void;

use crate::error::Result;
use crate::platform::{Device, Platform};
use crate::query;
use crate::sys;

/// The Windows identity of a device: a locally unique identifier plus the
/// monitor node it drives (`CL_DEVICE_LUID_KHR` and
/// `CL_DEVICE_NODE_MASK_KHR` of `cl_khr_device_uuid`).
///
/// Vulkan reports the same pair in `VkPhysicalDeviceIDProperties`; on
/// Windows it is the fallback identity for the drivers whose UUID is
/// unusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Luid {
    /// The 8 LUID bytes (`LUID`).
    pub bytes: [u8; sys::LUID_SIZE_KHR],
    /// The monitor node mask of the device.
    pub node_mask: u32,
}

/// Whether `list`, a space-separated OpenCL extension string, names
/// `extension` as one whole token.
///
/// A substring test would report `cl_khr_external_memory` on a driver that
/// only exports `cl_khr_external_memory_opaque_fd`, which is the opposite
/// of what the caller needs to know.
fn lists_extension(list: &str, extension: &str) -> bool {
    list.split_whitespace()
        .any(|token| token == extension)
}

/// Whether `uuid` identifies hardware rather than standing in for "no
/// UUID".
///
/// CPU devices, and drivers that do not implement `cl_khr_device_uuid` but
/// still answer the query, report an all-zero or all-ones block; treating
/// that as an identity would pair every such device with every other.
fn is_identity(uuid: &[u8; sys::UUID_SIZE_KHR]) -> bool {
    !uuid.iter().all(|byte| *byte == 0) && !uuid.iter().all(|byte| *byte == 0xFF)
}

impl Device {
    /// Queries a fixed-size identity block (UUID, LUID).
    fn info_fixed<T: Copy + Default>(&self, param: u32) -> Result<T> {
        let (api, raw) = (self.api.clone(), self.raw);
        let mut call = |size: usize, value: *mut c_void, size_ret: *mut usize| {
            // SAFETY: `raw` is a live device of the driver that resolved
            // `api`, and the query helpers only pass buffers of the size
            // the driver asked for.
            unsafe { (api.get_device_info)(raw, param, size, value, size_ret) }
        };
        query::query_fixed(&mut call, "clGetDeviceInfo")
    }

    /// Queries a zero-terminated list of handle-type tokens.
    fn info_tokens(&self, param: u32) -> Result<Vec<u32>> {
        let (api, raw) = (self.api.clone(), self.raw);
        let mut call = |size: usize, value: *mut c_void, size_ret: *mut usize| {
            // SAFETY: see `info_fixed`.
            unsafe { (api.get_device_info)(raw, param, size, value, size_ret) }
        };
        query::query_u32_list(&mut call, "clGetDeviceInfo")
    }

    /// Whether this device advertises `extension` (a `sys::EXT_*` name or
    /// any other extension string).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`](crate::error::Error::Status) when the
    /// driver rejects the extension query.
    pub fn supports_extension(&self, extension: &str) -> Result<bool> {
        Ok(lists_extension(&self.extensions()?, extension))
    }

    /// The 16-byte UUID of the device (`CL_DEVICE_UUID_KHR`).
    ///
    /// Vulkan exposes the same 16 bytes in `VkPhysicalDeviceIDProperties`,
    /// so two identical arrays mean the same physical device. CPU devices
    /// often report an all-zero block, which is why
    /// [`pair_uuid`](Self::pair_uuid) rejects it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`](crate::error::Error::Status) when the
    /// driver rejects the query, which a device without
    /// `cl_khr_device_uuid` does.
    pub fn uuid(&self) -> Result<[u8; sys::UUID_SIZE_KHR]> {
        self.info_fixed(sys::DEVICE_UUID_KHR)
    }

    /// The 16-byte UUID of the driver serving the device
    /// (`CL_DRIVER_UUID_KHR`).
    ///
    /// # Errors
    ///
    /// See [`uuid`](Self::uuid).
    pub fn driver_uuid(&self) -> Result<[u8; sys::UUID_SIZE_KHR]> {
        self.info_fixed(sys::DRIVER_UUID_KHR)
    }

    /// Whether this device is the one `uuid` names, rejecting the
    /// placeholder blocks that carry no identity.
    ///
    /// # Errors
    ///
    /// See [`uuid`](Self::uuid).
    pub fn pair_uuid(&self, uuid: &[u8; sys::UUID_SIZE_KHR]) -> Result<bool> {
        let mine = self.uuid()?;
        Ok(is_identity(&mine) && &mine == uuid)
    }

    /// The Windows LUID and node mask of the device, when it has one.
    ///
    /// `Ok(None)` is the normal answer on Linux and for drivers that leave
    /// `CL_DEVICE_LUID_VALID_KHR` false; an `Err` means a query was
    /// rejected.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`](crate::error::Error::Status) when the
    /// driver rejects either query.
    pub fn luid(&self) -> Result<Option<Luid>> {
        let (api, raw) = (self.api.clone(), self.raw);
        let mut call = |size: usize, value: *mut c_void, size_ret: *mut usize| {
            // SAFETY: see `info_fixed`.
            unsafe { (api.get_device_info)(raw, sys::DEVICE_LUID_VALID_KHR, size, value, size_ret) }
        };
        if !query::query_bool(&mut call, "clGetDeviceInfo")? {
            return Ok(None);
        }
        let bytes: [u8; sys::LUID_SIZE_KHR] = self.info_fixed(sys::DEVICE_LUID_KHR)?;
        let node_mask = self.info_u32(sys::DEVICE_NODE_MASK_KHR)?;
        Ok(Some(Luid { bytes, node_mask }))
    }

    /// The external memory handle types this device can import
    /// (`CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR`).
    ///
    /// The tokens are the `sys::EXTERNAL_MEMORY_HANDLE_*_KHR` values; a
    /// Vulkan exporter must choose one that also appears in
    /// `VkExternalBufferProperties::compatibleHandleTypes`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`](crate::error::Error::Status) when the
    /// driver rejects the query, which a device without
    /// `cl_khr_external_memory` does.
    pub fn external_memory_import_handle_types(&self) -> Result<Vec<u32>> {
        self.info_tokens(sys::DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR)
    }

    /// The semaphore handle types this device can import
    /// (`CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR`).
    ///
    /// # Errors
    ///
    /// See [`external_memory_import_handle_types`](Self::external_memory_import_handle_types).
    pub fn semaphore_import_handle_types(&self) -> Result<Vec<u32>> {
        self.info_tokens(sys::DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR)
    }

    /// The semaphore handle types this device can export
    /// (`CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR`).
    ///
    /// # Errors
    ///
    /// See [`external_memory_import_handle_types`](Self::external_memory_import_handle_types).
    pub fn semaphore_export_handle_types(&self) -> Result<Vec<u32>> {
        self.info_tokens(sys::DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR)
    }
}

impl Platform {
    /// Queries a zero-terminated list of handle-type tokens of the platform.
    fn info_tokens(&self, param: u32) -> Result<Vec<u32>> {
        let (api, raw) = (self.api.clone(), self.raw);
        let mut call = |size: usize, value: *mut c_void, size_ret: *mut usize| {
            // SAFETY: `raw` is a live platform of the driver that resolved
            // `api`, and the query helpers only pass buffers of the size
            // the driver asked for.
            unsafe { (api.get_platform_info)(raw, param, size, value, size_ret) }
        };
        query::query_u32_list(&mut call, "clGetPlatformInfo")
    }

    /// Whether this platform advertises `extension`.
    ///
    /// A platform-level extension is available on every device underneath
    /// it; [`Device::supports_extension`] stays the narrower, authoritative
    /// answer for one device.
    ///
    /// # Errors
    ///
    /// See [`Device::supports_extension`].
    pub fn supports_extension(&self, extension: &str) -> Result<bool> {
        Ok(lists_extension(&self.extensions()?, extension))
    }

    /// The external memory handle types this platform can import
    /// (`CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR`).
    ///
    /// # Errors
    ///
    /// See [`Device::external_memory_import_handle_types`].
    pub fn external_memory_import_handle_types(&self) -> Result<Vec<u32>> {
        self.info_tokens(sys::PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR)
    }

    /// The semaphore handle types this platform can import
    /// (`CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR`).
    ///
    /// # Errors
    ///
    /// See [`Device::external_memory_import_handle_types`].
    pub fn semaphore_import_handle_types(&self) -> Result<Vec<u32>> {
        self.info_tokens(sys::PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR)
    }

    /// The semaphore handle types this platform can export
    /// (`CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR`).
    ///
    /// # Errors
    ///
    /// See [`Device::external_memory_import_handle_types`].
    pub fn semaphore_export_handle_types(&self) -> Result<Vec<u32>> {
        self.info_tokens(sys::PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extension strings are space-separated token lists; a substring test
    /// would turn a narrower extension into a false positive.
    #[test]
    fn extension_matching_is_token_wise() {
        let list = "cl_khr_icd cl_khr_external_memory_opaque_fd cl_khr_fp64";
        assert!(lists_extension(list, sys::EXT_EXTERNAL_MEMORY_OPAQUE_FD));
        assert!(lists_extension(list, "cl_khr_icd"));
        assert!(!lists_extension(list, sys::EXT_EXTERNAL_MEMORY));
        assert!(!lists_extension(list, sys::EXT_SEMAPHORE));
        assert!(!lists_extension("", sys::EXT_DEVICE_UUID));
    }

    /// A placeholder block is not a device identity, even when the
    /// comparison array is identical.
    #[test]
    fn placeholder_uuids_carry_no_identity() {
        assert!(!is_identity(&[0; sys::UUID_SIZE_KHR]));
        assert!(!is_identity(&[0xFF; sys::UUID_SIZE_KHR]));
        assert!(is_identity(&[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ]));
    }

    /// The LUID is only meaningful together with its node mask; equality
    /// must compare both.
    #[test]
    fn luid_identity_covers_the_node_mask() {
        let bytes = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_ne!(Luid { bytes, node_mask: 1 }, Luid { bytes, node_mask: 2 });
        assert_eq!(Luid { bytes, node_mask: 1 }, Luid { bytes, node_mask: 1 });
    }

    /// On a machine with an OpenCL driver, capability questions must answer
    /// without panicking and without creating any object.
    #[test]
    fn capability_queries_answer_on_a_real_driver() {
        let Ok(runtime) = crate::runtime::Runtime::load() else {
            return;
        };
        for platform in runtime.platforms() {
            let _ = platform.supports_extension(sys::EXT_EXTERNAL_MEMORY);
            let _ = platform.external_memory_import_handle_types();
            let _ = platform.semaphore_export_handle_types();
            for device in platform.all_devices().unwrap_or_default() {
                let _ = device.supports_extension(sys::EXT_SEMAPHORE);
                let _ = device.semaphore_import_handle_types();
                let _ = device.luid();
                // A driver without `cl_khr_device_uuid` reports an error,
                // which must not be mistaken for a device match.
                if let Ok(uuid) = device.uuid() {
                    assert_eq!(
                        device.pair_uuid(&uuid).ok(),
                        Some(is_identity(&uuid)),
                        "a device must pair with its own UUID exactly when it is an identity"
                    );
                }
            }
        }
    }
}
