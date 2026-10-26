//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Platform registration: the one call that arms the `cl*` entry points.
//!
//! A hosted runtime discovers platforms by loading ICD libraries; an image
//! with statically linked drivers discovers them by having been built with
//! them. [`init`] takes the board's description — name, kernel registry and
//! the device back ends — and publishes it for every entry point. Until it
//! runs, `clGetPlatformIDs` reports `CL_PLATFORM_NOT_FOUND_KHR`, exactly as
//! a loader with an empty manifest would.
//!
//! `init` is idempotent: the first successful call wins, later calls are
//! no-ops. That keeps startup code (a test binary, several board bring-up
//! paths compiled into one image) from having to coordinate.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::objects::DeviceObject;
use crate::registry::Registry;

/// Everything the image registers with the driver.
#[derive(Clone, Copy)]
pub struct PlatformConfig {
    /// `CL_PLATFORM_NAME` — the board or product name.
    pub name: &'static str,
    /// `CL_PLATFORM_VENDOR`.
    pub vendor: &'static str,
    /// The kernel modules linked into the image; the registry resolves
    /// `clCreateKernel` names against them.
    pub modules: &'static [&'static crate::registry::KernelModule],
    /// The device back ends this platform exposes (at least one).
    pub devices: &'static [&'static dyn Device],
}

impl core::fmt::Debug for PlatformConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PlatformConfig")
            .field("name", &self.name)
            .field("vendor", &self.vendor)
            .field("modules", &self.modules.len())
            .field("devices", &self.devices.len())
            .finish()
    }
}

/// The published platform: a stable object every `cl_platform_id` points
/// at, plus boxed device handles (a `&dyn` is a fat pointer and cannot
/// round-trip through a C handle, so each device gets a concrete wrapper).
pub struct Platform {
    config: PlatformConfig,
    registry: Registry,
    devices: Vec<&'static DeviceObject>,
}

impl Platform {
    /// Validates `config` and publishes it.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] — no devices were registered.
    /// * [`Error::OutOfMemory`] — a handle allocation failed.
    pub(crate) fn publish(config: PlatformConfig) -> Result<&'static Platform> {
        if config.devices.is_empty() {
            return Err(Error::InvalidArgument);
        }
        let mut devices = Vec::new();
        devices
            .try_reserve_exact(config.devices.len())
            .map_err(|_| Error::OutOfMemory)?;
        for device in config.devices {
            devices.push(DeviceObject::leak(*device));
        }
        let registry = Registry::new(config.modules);
        Ok(Box::leak(Box::new(Self {
            config,
            registry,
            devices,
        })))
    }

    /// `CL_PLATFORM_NAME`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.config.name
    }

    /// `CL_PLATFORM_VENDOR`.
    #[must_use]
    pub fn vendor(&self) -> &'static str {
        self.config.vendor
    }

    /// `CL_PLATFORM_VERSION`.
    #[must_use]
    pub fn version(&self) -> &'static str {
        "OpenCL 1.2 Codevar-bare"
    }

    /// `CL_PLATFORM_PROFILE` — this driver implements the embedded profile.
    #[must_use]
    pub fn profile(&self) -> &'static str {
        "EMBEDDED_PROFILE"
    }

    /// The static kernel registry of the image.
    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// The boxed device handles of the platform (`clGetDeviceIDs`).
    pub fn devices(&self) -> &[&'static DeviceObject] {
        &self.devices
    }

    /// Pointer equality for `clGetPlatformInfo` handle validation.
    #[must_use]
    pub fn matches(&self, other: *const Platform) -> bool {
        core::ptr::eq(self, other)
    }
}

static PLATFORM: spin::Mutex<Option<&'static Platform>> = spin::Mutex::new(None);

/// Registers the platform and its devices — the driver's only required
/// setup call, normally invoked once from the board's `init`.
///
/// Idempotent: a second call returns `Ok` and keeps the first platform, so
/// independently compiled bring-up paths cannot fight over the slot.
///
/// # Errors
///
/// Propagates [`Platform::publish`] failures. The platform must be
/// initialized after the global allocator has its memory, because handles
/// are heap objects.
pub fn init(config: PlatformConfig) -> Result<()> {
    let mut slot = PLATFORM.lock();
    if slot.is_some() {
        return Ok(());
    }
    let platform = Platform::publish(config)?;
    *slot = Some(platform);
    Ok(())
}

/// The published platform (every `cl*` entry point starts here).
///
/// # Errors
///
/// [`Error::NotInitialized`] when [`init`] has not run — surfaced to
/// applications as `CL_PLATFORM_NOT_FOUND_KHR`.
pub fn state() -> Result<&'static Platform> {
    match *PLATFORM.lock() {
        Some(platform) => Ok(platform),
        None => Err(Error::NotInitialized),
    }
}

/// Alias of [`state`] re-exported at the crate root for setup code.
///
/// # Errors
///
/// See [`state`].
pub fn platform() -> Result<&'static Platform> {
    state()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceInfo};
    use crate::error::Result as DriverResult;
    use crate::exec::Launch;
    use crate::objects::{Event, release_ref};
    use crate::sys;

    struct Idle;
    impl Device for Idle {
        fn info(&self) -> &DeviceInfo {
            static INFO: DeviceInfo =
                DeviceInfo::new("idle", "codevar", sys::DEVICE_TYPE_CPU, 1, 64, 1 << 20, 4096, 256);
            &INFO
        }
        fn submit(&self, _launch: &Launch<'_>, event: &'static Event) -> DriverResult<()> {
            event.finish(sys::COMPLETE);
            release_ref(event);
            Ok(())
        }
        fn progress(&self) -> DriverResult<()> {
            Ok(())
        }
        fn inflight(&self) -> usize {
            0
        }
    }

    static IDLE: Idle = Idle;
    static DEVICES: &[&dyn Device] = &[&IDLE];

    #[test]
    fn init_publishes_and_is_idempotent() {
        // The first test to run claims the process-wide slot; every later
        // call must observe the same platform regardless of its config.
        init(PlatformConfig {
            name: "board-a",
            vendor: "codevar",
            modules: &[],
            devices: DEVICES,
        })
        .expect("publish");
        let first = state().expect("published");
        init(PlatformConfig {
            name: "board-b",
            vendor: "other",
            modules: &[],
            devices: &[],
        })
        .expect("idempotent");
        let second = state().expect("still published");
        assert!(first.matches(second));
        assert_eq!(first.name(), "board-a");
        assert_eq!(first.devices().len(), 1);
        assert!(platform().is_ok());
    }
}
