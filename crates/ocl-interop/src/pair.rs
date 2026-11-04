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

//! Whether the two devices are one device.
//!
//! A shared handle names an object inside one GPU, so the first question of
//! an exchange is not whether either API can export it but whether the two
//! devices are the same hardware — a `dma_buf` of one is rejected by another
//! with an error that says nothing about the pairing.
//!
//! Both APIs solve that with the same two identities:
//!
//! * a 16-byte device UUID (`VkPhysicalDeviceIDProperties::deviceUUID` and
//!   `CL_DEVICE_UUID_KHR` of `cl_khr_device_uuid`);
//! * an 8-byte Windows LUID plus the monitor node it drives, which
//!   `VkPhysicalDeviceIDProperties` and `CL_DEVICE_LUID_KHR` also share.
//!
//! The UUID is compared first because it is the identity both APIs
//! document, and because a Linux driver that has no LUID at all still has
//! one. A driver that cannot name its hardware fills the array with zeros,
//! which is not an answer: [`pairs_with`](Identity::pairs_with) falls back to
//! the LUID and, when neither side has one, says so instead of guessing.

use codevar_ocl::{Device, Luid, sys};

use crate::error::{Error, Result};

/// Whether `uuid` identifies hardware instead of standing in for "no
/// identity".
///
/// Vulkan fills an unavailable `deviceUUID` with zeros and `cl_khr_device_uuid`
/// does the same; an all-ones block is what a driver that never wrote the
/// field leaves behind. Either read as an identity would pair every CPU
/// device with every other.
fn identifies(uuid: &[u8; sys::UUID_SIZE_KHR]) -> bool {
    !uuid.iter().all(|byte| *byte == 0) && !uuid.iter().all(|byte| *byte == 0xFF)
}

/// The identity a Vulkan physical device reports for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    /// `VkPhysicalDeviceIDProperties::deviceUUID`.
    pub device_uuid: [u8; sys::UUID_SIZE_KHR],
    /// `VkPhysicalDeviceIDProperties::driverUUID`, which pairs two devices
    /// that run on the same driver even when they are not the same hardware.
    pub driver_uuid: [u8; sys::UUID_SIZE_KHR],
    /// `deviceLUID` and `deviceNodeMask`, present only when the driver set
    /// `deviceLUIDValid`.
    pub luid: Option<Luid>,
}

impl Identity {
    /// Whether the OpenCL `device` is the same hardware as this Vulkan
    /// device.
    ///
    /// A `false` is a decision, not a lack of one: it means both sides
    /// named themselves and named different hardware.
    ///
    /// # Errors
    ///
    /// - [`Error::Unsupported`] when the OpenCL device has no
    ///   `cl_khr_device_uuid`, or when neither side has a usable identity.
    /// - [`Error::OpenCl`](crate::Error::OpenCl) when a driver rejects one
    ///   of the identity queries.
    pub fn pairs_with(&self, device: &Device) -> Result<bool> {
        if !device.supports_extension(sys::EXT_DEVICE_UUID)? {
            return Err(Error::Unsupported {
                what: "the OpenCL device does not advertise cl_khr_device_uuid",
            });
        }
        let theirs = device.uuid()?;
        if identifies(&self.device_uuid) && identifies(&theirs) {
            return Ok(self.device_uuid == theirs);
        }
        // The LUID is the documented fallback for exactly this case, and it
        // is only an identity when both sides have one.
        match (&self.luid, device.luid()?) {
            (Some(mine), Some(other)) => Ok(mine == &other),
            _ => Err(Error::Unsupported {
                what: "neither device reports a UUID or a LUID to compare",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_ocl::Runtime;

    /// A device on the installed driver or the built-in CPU one, whichever
    /// answers first; `None` when the machine exposes no usable device, in
    /// which case the driver-facing test has nothing to check.
    fn device() -> Option<Device> {
        let runtime = Runtime::load().ok()?;
        runtime
            .platforms()
            .first()
            .and_then(|platform| platform.all_devices().ok())
            .and_then(|devices| devices.into_iter().next())
    }

    /// A placeholder block carries no identity, so it must never be the
    /// basis of a pairing.
    #[test]
    fn a_placeholder_uuid_is_not_an_identity() {
        assert!(!identifies(&[0; sys::UUID_SIZE_KHR]));
        assert!(!identifies(&[0xFF; sys::UUID_SIZE_KHR]));
        let mut uuid = [0; sys::UUID_SIZE_KHR];
        uuid[7] = 1;
        assert!(identifies(&uuid));
    }

    /// An identity with nothing to compare is reported as unsupported rather
    /// than answered with a `false` that a caller would read as "different
    /// hardware" — which is what every CPU device would otherwise say about
    /// the machine it runs on.
    #[test]
    fn an_identity_without_a_counterpart_says_so() {
        let Some(device) = device() else { return };
        let identity = Identity {
            device_uuid: [0; sys::UUID_SIZE_KHR],
            driver_uuid: [0; sys::UUID_SIZE_KHR],
            luid: None,
        };
        assert!(matches!(
            identity.pairs_with(&device),
            Err(Error::Unsupported { .. })
        ));
    }
}
