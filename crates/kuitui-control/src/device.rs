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

//! Stable device identity and the description a backend publishes for it.

use alloc::string::String;

use crate::class::{DeviceCapability, DeviceClass};

/// A device identifier that stays valid for as long as the device is
/// connected.
///
/// The value is opaque: a backend chooses it (an evdev minor number, a raw
/// input handle, a WebHID vendor/product pair). A caller compares identifiers
/// but never interprets them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceId(u64);

impl DeviceId {
    /// Creates an identifier from a backend-defined value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The backend-defined value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A connected input device as the backend sees it.
///
/// The name is owned because several operating systems return a UTF-16
/// buffer that has to be decoded at enumeration time; keeping the decoded
/// name avoids re-querying the device on every UI refresh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Stable identifier used by every event from this device.
    pub id: DeviceId,
    /// The class that determines which stream receives its events.
    pub class: DeviceClass,
    /// A human-readable name, empty when the platform hides it.
    pub name: String,
    /// The USB or Bluetooth vendor identifier, zero when unknown.
    pub vendor: u16,
    /// The product identifier, zero when unknown.
    pub product: u16,
    /// The events this particular device can produce.
    pub capabilities: DeviceCapability,
}

impl DeviceInfo {
    /// Creates a description with the capabilities that its class normally
    /// provides.
    #[must_use]
    pub fn new(id: DeviceId, class: DeviceClass, name: String) -> Self {
        Self {
            id,
            class,
            name,
            vendor: 0,
            product: 0,
            capabilities: DeviceCapability::for_class(class),
        }
    }

    /// The description with explicit vendor and product identifiers.
    #[must_use]
    pub fn with_ids(mut self, vendor: u16, product: u16) -> Self {
        self.vendor = vendor;
        self.product = product;
        self
    }

    /// The description with an explicit capability set.
    #[must_use]
    pub fn with_capabilities(mut self, capabilities: DeviceCapability) -> Self {
        self.capabilities = capabilities;
        self
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use super::{DeviceId, DeviceInfo};
    use crate::class::{DeviceCapability, DeviceClass};

    #[test]
    fn new_device_gets_its_class_capabilities() {
        let info = DeviceInfo::new(DeviceId::new(42), DeviceClass::Keyboard, String::from("keys"));
        assert_eq!(info.id, DeviceId::new(42));
        assert_eq!(info.class, DeviceClass::Keyboard);
        assert!(info.capabilities.contains(DeviceCapability::KEY));
        assert_eq!(info.vendor, 0);
        assert_eq!(info.product, 0);
    }

    #[test]
    fn identifiers_and_capabilities_can_be_refined() {
        let info = DeviceInfo::new(DeviceId::new(1), DeviceClass::Pointer, String::new())
            .with_ids(0x046d, 0xc52b)
            .with_capabilities(DeviceCapability::POINTER_MOTION);
        assert_eq!(info.vendor, 0x046d);
        assert_eq!(info.product, 0xc52b);
        assert!(
            info.capabilities
                .contains(DeviceCapability::POINTER_MOTION)
        );
        assert!(
            !info
                .capabilities
                .contains(DeviceCapability::SCROLL)
        );
    }
}
