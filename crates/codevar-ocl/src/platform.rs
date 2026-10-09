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

//! Platforms and devices.
//!
//! A [`Platform`] is one OpenCL implementation (Intel NEO, AMD ROCm,
//! NVIDIA, pocl, …) as exposed either by the ICD loader or by a directly
//! loaded vendor library; a [`Device`] is a compute unit underneath it.
//! Both are cheap, copyable-by-reference handles: they store the resolved
//! [`Api`] of their driver plus the raw driver handle, and they never
//! release anything (platforms and devices are not reference-counted by
//! OpenCL 1.x).

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::ptr::null_mut;

use crate::api::Api;
use crate::error::{Error, Result, status};
use crate::query::{self, MAX_HANDLES};
use crate::sys::{self, DeviceHandle, PlatformHandle};

/// The kind of device to enumerate from a [`Platform`].
///
/// Maps to the `cl_device_type` bitfield; the driver may legitimately
/// return no devices for any kind (for example no GPU on a CPU-only
/// machine), which surfaces as an empty `Vec`, not an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeviceKind {
    /// Every device on the platform.
    #[default]
    All,
    /// The implementation's default device.
    Default,
    /// CPU devices.
    Cpu,
    /// GPU devices.
    Gpu,
    /// Dedicated accelerator devices.
    Accelerator,
    /// Vendor-defined custom devices.
    Custom,
}

impl DeviceKind {
    /// Returns the `cl_device_type` bits this kind maps to.
    #[must_use]
    pub const fn bits(self) -> u64 {
        match self {
            Self::All => sys::DEVICE_TYPE_ALL,
            Self::Default => sys::DEVICE_TYPE_DEFAULT,
            Self::Cpu => sys::DEVICE_TYPE_CPU,
            Self::Gpu => sys::DEVICE_TYPE_GPU,
            Self::Accelerator => sys::DEVICE_TYPE_ACCELERATOR,
            Self::Custom => sys::DEVICE_TYPE_CUSTOM,
        }
    }
}

/// One OpenCL implementation discovered at runtime.
///
/// Clone-cheap? No — cloning costs an `Arc` increment; share references or
/// wrap in [`Arc`] yourself if multiple owners are needed.
#[derive(Clone)]
pub struct Platform {
    pub(crate) api: Arc<Api>,
    pub(crate) raw: PlatformHandle,
}

// SAFETY: `Platform` holds an opaque driver handle plus a shared `Arc<Api>`
// of function pointers. Platform handles are not reference-counted and are
// never released, so moving or sharing a `Platform` between threads performs
// no driver-side state mutation beyond the (thread-safe) info queries.
unsafe impl Send for Platform {}
// SAFETY: see the `Send` implementation; info queries on distinct objects
// are independent per the OpenCL specification.
unsafe impl Sync for Platform {}

impl Platform {
    /// Wraps an already-enumerated platform handle.
    pub(crate) fn new(api: Arc<Api>, raw: PlatformHandle) -> Self {
        Self { api, raw }
    }

    fn info_string(&self, param: u32) -> Result<String> {
        let mut call = |size: usize, value: *mut core::ffi::c_void, size_ret: *mut usize| {
            // SAFETY: the raw handle belongs to the driver that resolved
            // `api`, and `query_string` provides valid buffers.
            unsafe { (self.api.get_platform_info)(self.raw, param, size, value, size_ret) }
        };
        query::query_string(&mut call, "clGetPlatformInfo")
    }

    /// Returns the platform name, e.g. `"Intel(R) OpenCL HD Graphics"`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn name(&self) -> Result<String> {
        self.info_string(sys::PLATFORM_NAME)
    }

    /// Returns the platform vendor name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn vendor(&self) -> Result<String> {
        self.info_string(sys::PLATFORM_VENDOR)
    }

    /// Returns the platform OpenCL version string, e.g. `"OpenCL 3.0 …"`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn version(&self) -> Result<String> {
        self.info_string(sys::PLATFORM_VERSION)
    }

    /// Returns the platform profile (`FULL_PROFILE` or `EMBEDDED_PROFILE`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn profile(&self) -> Result<String> {
        self.info_string(sys::PLATFORM_PROFILE)
    }

    /// Returns the space-separated platform extension list.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn extensions(&self) -> Result<String> {
        self.info_string(sys::PLATFORM_EXTENSIONS)
    }

    /// Enumerates the devices of the given [`DeviceKind`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects enumeration (a
    /// plain [`sys::DEVICE_NOT_FOUND`] is *not* an error: it becomes an
    /// empty vector) and [`Error::InvalidArgument`] when the driver reports
    /// an implausible device count.
    pub fn devices(&self, kind: DeviceKind) -> Result<Vec<Device>> {
        let mut count: u32 = 0;
        // SAFETY: the null/zero first call is the documented way to query
        // the device count; `count` is a valid out-pointer.
        let code = unsafe { (self.api.get_device_ids)(self.raw, kind.bits(), 0, null_mut(), &mut count) };
        if code == sys::DEVICE_NOT_FOUND || (code == sys::SUCCESS && count == 0) {
            return Ok(Vec::new());
        }
        status(code, "clGetDeviceIDs")?;
        if count > MAX_HANDLES {
            return Err(Error::InvalidArgument {
                what: "clGetDeviceIDs reported an implausible device count",
            });
        }
        let mut handles: Vec<DeviceHandle> = (0..count)
            .map(|_| DeviceHandle::from_raw(null_mut()))
            .collect();
        // SAFETY: `handles` has room for `count` handles, matching the
        // count the driver just confirmed.
        let code = unsafe {
            (self.api.get_device_ids)(self.raw, kind.bits(), count, handles.as_mut_ptr(), null_mut())
        };
        status(code, "clGetDeviceIDs")?;
        handles.retain(|handle| !handle.is_null());
        Ok(handles
            .into_iter()
            .map(|raw| Device::new(self.api.clone(), raw))
            .collect())
    }

    /// Enumerates every device of the platform.
    ///
    /// Shorthand for `platform.devices(DeviceKind::All)`.
    ///
    /// # Errors
    ///
    /// See [`devices`](Self::devices).
    pub fn all_devices(&self) -> Result<Vec<Device>> {
        self.devices(DeviceKind::All)
    }
}

impl fmt::Debug for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Platform")
            .field("handle", &self.raw)
            .finish_non_exhaustive()
    }
}

/// One compute device underneath a [`Platform`].
#[derive(Clone)]
pub struct Device {
    pub(crate) api: Arc<Api>,
    pub(crate) raw: DeviceHandle,
}

// SAFETY: the same argument as [`Platform`]: the handle is opaque, never
// released, and the shared `Api` consists of plain function pointers.
unsafe impl Send for Device {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for Device {}

impl Device {
    /// Wraps an already-enumerated device handle.
    pub(crate) fn new(api: Arc<Api>, raw: DeviceHandle) -> Self {
        Self { api, raw }
    }

    /// Returns the resolved entry points of the owning driver.
    pub(crate) fn api(&self) -> &Arc<Api> {
        &self.api
    }

    /// Returns the raw driver handle.
    pub(crate) const fn raw(&self) -> DeviceHandle {
        self.raw
    }

    fn info_string(&self, param: u32) -> Result<String> {
        let mut call = |size: usize, value: *mut core::ffi::c_void, size_ret: *mut usize| {
            // SAFETY: the raw handle belongs to the driver that resolved
            // `api`, and `query_string` provides valid buffers.
            unsafe { (self.api.get_device_info)(self.raw, param, size, value, size_ret) }
        };
        query::query_string(&mut call, "clGetDeviceInfo")
    }

    fn info_u32(&self, param: u32) -> Result<u32> {
        let mut call = |size: usize, value: *mut core::ffi::c_void, size_ret: *mut usize| {
            // SAFETY: see `info_string`.
            unsafe { (self.api.get_device_info)(self.raw, param, size, value, size_ret) }
        };
        query::query_u32(&mut call, "clGetDeviceInfo")
    }

    fn info_u64(&self, param: u32) -> Result<u64> {
        let mut call = |size: usize, value: *mut core::ffi::c_void, size_ret: *mut usize| {
            // SAFETY: see `info_string`.
            unsafe { (self.api.get_device_info)(self.raw, param, size, value, size_ret) }
        };
        query::query_u64(&mut call, "clGetDeviceInfo")
    }

    fn info_usize(&self, param: u32) -> Result<usize> {
        let mut call = |size: usize, value: *mut core::ffi::c_void, size_ret: *mut usize| {
            // SAFETY: see `info_string`.
            unsafe { (self.api.get_device_info)(self.raw, param, size, value, size_ret) }
        };
        query::query_usize(&mut call, "clGetDeviceInfo")
    }

    fn info_bool(&self, param: u32) -> Result<bool> {
        let mut call = |size: usize, value: *mut core::ffi::c_void, size_ret: *mut usize| {
            // SAFETY: see `info_string`.
            unsafe { (self.api.get_device_info)(self.raw, param, size, value, size_ret) }
        };
        query::query_bool(&mut call, "clGetDeviceInfo")
    }

    /// Returns the device name, e.g. `"Intel(R) HD Graphics 530"`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn name(&self) -> Result<String> {
        self.info_string(sys::DEVICE_NAME)
    }

    /// Returns the device vendor name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn vendor(&self) -> Result<String> {
        self.info_string(sys::DEVICE_VENDOR)
    }

    /// Returns the device OpenCL version string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn version(&self) -> Result<String> {
        self.info_string(sys::DEVICE_VERSION)
    }

    /// Returns the vendor's driver version string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn driver_version(&self) -> Result<String> {
        self.info_string(sys::DRIVER_VERSION)
    }

    /// Returns the device profile.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn profile(&self) -> Result<String> {
        self.info_string(sys::DEVICE_PROFILE)
    }

    /// Returns the space-separated device extension list.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn extensions(&self) -> Result<String> {
        self.info_string(sys::DEVICE_EXTENSIONS)
    }

    /// Returns the raw `cl_device_type` bitfield of the device.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn device_type_bits(&self) -> Result<u64> {
        self.info_u64(sys::DEVICE_TYPE)
    }

    /// Whether the device is a GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn is_gpu(&self) -> Result<bool> {
        Ok(self.device_type_bits()? & sys::DEVICE_TYPE_GPU != 0)
    }

    /// Whether the device is a CPU.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn is_cpu(&self) -> Result<bool> {
        Ok(self.device_type_bits()? & sys::DEVICE_TYPE_CPU != 0)
    }

    /// Returns the number of parallel compute units.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn max_compute_units(&self) -> Result<u32> {
        self.info_u32(sys::DEVICE_MAX_COMPUTE_UNITS)
    }

    /// Returns the maximum work-group size for kernels on this device.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn max_work_group_size(&self) -> Result<usize> {
        self.info_usize(sys::DEVICE_MAX_WORK_GROUP_SIZE)
    }

    /// Returns the total global memory in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn global_mem_size(&self) -> Result<u64> {
        self.info_u64(sys::DEVICE_GLOBAL_MEM_SIZE)
    }

    /// Returns the maximum single buffer allocation in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn max_mem_alloc_size(&self) -> Result<u64> {
        self.info_u64(sys::DEVICE_MAX_MEM_ALLOC_SIZE)
    }

    /// Returns the local (on-chip) memory size in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn local_mem_size(&self) -> Result<u64> {
        self.info_u64(sys::DEVICE_LOCAL_MEM_SIZE)
    }

    /// Returns the device address width in bits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn address_bits(&self) -> Result<u32> {
        self.info_u32(sys::DEVICE_ADDRESS_BITS)
    }

    /// Whether the device is currently available.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn available(&self) -> Result<bool> {
        self.info_bool(sys::DEVICE_AVAILABLE)
    }

    /// Whether an online compiler is present (required to build programs).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the driver rejects the query.
    pub fn compiler_available(&self) -> Result<bool> {
        self.info_bool(sys::DEVICE_COMPILER_AVAILABLE)
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("handle", &self.raw)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_kind_bits_match_the_headers() {
        assert_eq!(DeviceKind::All.bits(), 0xFFFF_FFFF);
        assert_eq!(DeviceKind::Default.bits(), 1);
        assert_eq!(DeviceKind::Cpu.bits(), 2);
        assert_eq!(DeviceKind::Gpu.bits(), 4);
        assert_eq!(DeviceKind::Accelerator.bits(), 8);
        assert_eq!(DeviceKind::Custom.bits(), 16);
        assert_eq!(DeviceKind::default(), DeviceKind::All);
    }

    /// On machines without OpenCL this must return an empty list or an
    /// error — never panic; with OpenCL installed it must yield devices
    /// with non-empty names.
    #[test]
    fn enumeration_is_safe_on_any_host() {
        let Ok(runtime) = crate::runtime::Runtime::load() else {
            return;
        };
        for platform in runtime.platforms() {
            let Ok(devices) = platform.all_devices() else {
                continue;
            };
            for device in &devices {
                if let Ok(name) = device.name() {
                    assert!(!name.is_empty());
                }
            }
        }
    }
}
