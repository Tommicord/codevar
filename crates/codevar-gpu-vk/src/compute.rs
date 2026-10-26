//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Vulkan probe for the host↔device link of a discrete accelerator.
//!
//! Core Vulkan cannot report a shader-core count or a clock frequency, so
//! a theoretical FLOP/s figure for a Vulkan device is assembled elsewhere
//! (the OpenCL device queries). What Vulkan *does* report precisely is
//! where the device sits on the bus: `VK_EXT_pci_bus_info` exposes the
//! device's domain, bus, device and function numbers. The operating
//! system then publishes that slot's negotiated signalling rate and link
//! width, which [`probe_host_link`] turns into a
//! [`HostLink`] through [`pcie_link_bandwidth`] — the framing
//! overhead is derived from the rate itself, never a per-device table.
//!
//! The link is read from the same source `lspci -vv` uses, through the
//! best available backend for the target:
//!
//! | Target | Backend |
//! |--------|---------|
//! | Linux  | the kernel's sysfs records |
//! | Windows | the PCI bus driver's device properties (SetupAPI) |
//! | other  | none; the caller keeps its OpenCL-based estimate |
//!
//! The sysfs records are read with [`codevar_pathbuf`], which owns the
//! platform file I/O (libc on Unix, `CreateFileW` on Windows) and
//! degrades to an unsupported-target error on bare metal. The one term
//! no query exposes is the per-transfer start-up latency, which is a
//! documented order-of-magnitude estimate (see
//! [`TRANSFER_STARTUP_SECONDS`]) rather than a device number.
//!
//! The probe is best-effort: no loader, no Vulkan 1.3, no PCI bus info,
//! no OS link record or a target without a backend all yield [`None`] and
//! leave the caller's OpenCL-based estimate in place.

use ash::vk;
use codevar_gpu_core::HostLink;
use codevar_logger::log_debug;

use crate::pipeline::{PipelineError, VulkanLibrary, create_instance};

/// The measured start-up cost of one kernel-mediated DMA transfer.
///
/// PCIe exposes no queuing delay, and the submit plus descriptor setup
/// dominates the transfer time at small sizes, so this term has to be an
/// estimate. It is the small-transfer knee reported by bandwidth probes
/// (CUDA `bandwidthTest`, clpeak) on a kernel-mediated copy, not a
/// per-device figure, and its only role is to keep tiny workloads on the
/// CPU where the fixed cost is not amortised.
pub const TRANSFER_STARTUP_SECONDS: f64 = 5.0e-6;

/// Probes the host↔device link of the most potent PCIe device Vulkan can
/// see.
///
/// Returns [`None`] when the loader, the instance, the PCI bus info
/// extension or the operating system's link record is unavailable; every
/// failure is logged and never fatal, because the caller has a fallback
/// estimate.
#[must_use]
pub fn probe_host_link() -> Option<HostLink> {
    match probe_inner() {
        Ok(link) => link,
        Err(error) => {
            log_debug!("the Vulkan PCIe probe is unavailable: {error}");
            None
        }
    }
}

/// Loads Vulkan, scans the physical devices and destroys the instance.
fn probe_inner() -> Result<Option<HostLink>, PipelineError> {
    // The library handle keeps the loader mapped; it must outlive the
    // instance, so it is bound rather than dropped.
    let (_library, entry) = VulkanLibrary::load()?;
    let instance = create_instance(&entry)?;
    let result = scan_devices(&instance);
    // SAFETY: `instance` is the valid instance created just above, every
    // physical-device handle borrowed from it is out of scope, and it is
    // not used after this call.
    unsafe { instance.destroy_instance(None) };
    result
}

/// Returns the link of a discrete device, or the first link found when no
/// discrete device exposes one.
fn scan_devices(instance: &ash::Instance) -> Result<Option<HostLink>, PipelineError> {
    // SAFETY: `instance` is valid; the driver allocates the returned
    // vector and owns its contents.
    let devices =
        unsafe { instance.enumerate_physical_devices() }.map_err(PipelineError::PhysicalDeviceQuery)?;
    let mut first_link = None;
    for physical in devices {
        let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut pci);
        // SAFETY: `physical` came from enumeration of `instance` and the
        // `properties` chain is well-formed and outlives the call.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let device_type = properties.properties.device_type;
        // The borrow of `pci` ends with the last use of `properties`, so the
        // queried BDF is readable here.
        let slot = (pci.pci_domain, pci.pci_bus, pci.pci_device, pci.pci_function);
        let Some(link) = link_for_bdf(slot.0, slot.1, slot.2, slot.3) else {
            continue;
        };
        if device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            return Ok(Some(link));
        }
        first_link.get_or_insert(link);
    }
    Ok(first_link)
}

/// Builds a [`HostLink`] from a negotiated signalling rate and lane count.
///
/// Both are the raw values the platform publishes (GT/s and lanes), so the
/// bandwidth — including its framing overhead — comes entirely from
/// [`pcie_link_bandwidth`](codevar_gpu_core::pcie_link_bandwidth). A zero or
/// nonsensical pair yields [`None`], which the caller treats as "no link".
#[cfg(any(target_os = "linux", windows, test))]
fn host_link(gigatransfers_per_second: f64, lanes: u32) -> Option<HostLink> {
    use codevar_gpu_core::pcie_link_bandwidth;

    let bandwidth = pcie_link_bandwidth(gigatransfers_per_second, lanes);
    if bandwidth.is_nan() || bandwidth <= 0.0 {
        return None;
    }
    Some(HostLink {
        bandwidth,
        latency: TRANSFER_STARTUP_SECONDS,
    })
}

/// The negotiated link of the device at domain:bus:device.function.
///
/// Dispatched to the target's backend; see the module documentation. On a
/// target without a link backend the probe declines and the caller keeps
/// its OpenCL-based estimate.
#[cfg(target_os = "linux")]
use linux::link_for_bdf;
/// See the Linux alias above.
#[cfg(windows)]
use win32::link_for_bdf;

/// The fallback for targets with no link backend (for example bare metal).
#[cfg(not(any(target_os = "linux", windows)))]
fn link_for_bdf(_domain: u32, _bus: u32, _device: u32, _function: u32) -> Option<HostLink> {
    None
}

/// Linux backend: the kernel's per-slot sysfs records.
#[cfg(target_os = "linux")]
mod linux {
    use alloc::format;
    use alloc::string::String;

    use codevar_pathbuf::{PathBuilder, read_to_string};

    use super::{HostLink, host_link};

    /// Formats a PCI domain:bus:device.function slot as sysfs names it,
    /// for example `0000:01:00.0`.
    fn pci_bdf(domain: u32, bus: u32, device: u32, function: u32) -> String {
        format!("{domain:04x}:{bus:02x}:{device:02x}.{function}")
    }

    /// Reads one field of the sysfs directory of the device at `bdf`.
    ///
    /// The path is assembled and validated with [`PathBuilder`]; the read
    /// itself goes through [`read_to_string`], so no file descriptor or
    /// libc symbol is touched here and the call degrades to an error on a
    /// target without a file backend.
    fn read_field(bdf: &str, field: &str) -> Option<String> {
        let mut path = PathBuilder::new()
            .root()
            .push("sys")
            .push("bus")
            .push("pci")
            .push("devices")
            .push(bdf)
            .build()
            .ok()?;
        path.push(field).ok()?;
        read_to_string(path.as_str()).ok()
    }

    /// Parses the leading token of a `current_link_speed` record as GT/s.
    fn parse_link_speed(text: &str) -> Option<f64> {
        text.split_whitespace()
            .next()?
            .parse::<f64>()
            .ok()
    }

    /// Parses a `current_link_width` record as a lane count.
    fn parse_link_width(text: &str) -> Option<u32> {
        text.trim().parse::<u32>().ok()
    }

    /// Reads the negotiated link of the device at `bdf` from sysfs.
    pub(super) fn link_for_bdf(domain: u32, bus: u32, device: u32, function: u32) -> Option<HostLink> {
        let bdf = pci_bdf(domain, bus, device, function);
        let speed = parse_link_speed(&read_field(&bdf, "current_link_speed")?)?;
        let lanes = parse_link_width(&read_field(&bdf, "current_link_width")?)?;
        host_link(speed, lanes)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn bdf_matches_the_sysfs_layout() {
            assert_eq!(pci_bdf(0, 1, 0, 0), "0000:01:00.0");
            assert_eq!(pci_bdf(0x0001, 0x3b, 0x10, 3), "0001:3b:10.3");
        }

        #[test]
        fn link_speed_parses_the_leading_token() {
            assert_eq!(parse_link_speed("16.0 GT/s PCIe\n"), Some(16.0));
            assert_eq!(parse_link_speed("2.5 GT/s"), Some(2.5));
            assert_eq!(parse_link_speed(""), None);
            assert_eq!(parse_link_speed("unknown"), None);
        }

        #[test]
        fn link_width_parses_the_lane_count() {
            assert_eq!(parse_link_width("16\n"), Some(16));
            assert_eq!(parse_link_width(" 8 "), Some(8));
            assert_eq!(parse_link_width(""), None);
        }
    }
}

/// Windows backend: the PCI bus driver's per-device SetupAPI properties.
///
/// Windows does not expose a file for the negotiated link; the PCI bus
/// driver publishes it as the `DEVPKEY_PciDevice_CurrentLinkSpeed` and
/// `DEVPKEY_PciDevice_CurrentLinkWidth` device properties instead (the
/// same values Device Manager shows). The device is located by its bus
/// number and bus address, which together are the Windows equivalent of
/// the BDF the Vulkan probe reports.
#[cfg(windows)]
mod win32 {
    use windows::Win32::Devices::DeviceAndDriverInstallation::{
        DIGCF_ALLCLASSES, DIGCF_PRESENT, HDEVINFO, SP_DEVINFO_DATA, SetupDiDestroyDeviceInfoList,
        SetupDiEnumDeviceInfo, SetupDiGetClassDevsW, SetupDiGetDevicePropertyW,
    };
    use windows::Win32::Devices::Properties::{DEVPROP_TYPE_UINT32, DEVPROPTYPE};
    use windows::Win32::Foundation::DEVPROPKEY;
    use windows::core::{GUID, PCWSTR};

    use super::{HostLink, host_link};

    /// `{3ab22e31-8264-4b4e-9af5-a8d2d8e33e62}`, the PCI Express device
    /// property set (`DEFINE_PCI_DEVICE_DEVPKEY` in `pciprop.h`).
    const PCI_DEVICE_KEY: GUID = GUID::from_u128(0x3ab22e31_8264_4b4e_9af5_a8d2d8e33e62);
    /// `{a45c254e-df1c-4efd-8020-67d146a850e0}`, the common device
    /// property set (`DEVPKEY_Device_*` in `devpkey.h`).
    const PCI_COMMON_KEY: GUID = GUID::from_u128(0xa45c254e_df1c_4efd_8020_67d146a850e0);

    /// Negotiated link speed; a `DEVPROP_TYPE_UINT32` in the PCIe-spec
    /// encoding decoded by [`link_speed_gts`].
    const CURRENT_LINK_SPEED: DEVPROPKEY = DEVPROPKEY {
        fmtid: PCI_DEVICE_KEY,
        pid: 9,
    };
    /// Negotiated link width in lanes; a `DEVPROP_TYPE_UINT32`.
    const CURRENT_LINK_WIDTH: DEVPROPKEY = DEVPROPKEY {
        fmtid: PCI_DEVICE_KEY,
        pid: 10,
    };
    /// The device's bus-relative address, `(device << 16) | function`.
    const DEVICE_ADDRESS: DEVPROPKEY = DEVPROPKEY {
        fmtid: PCI_COMMON_KEY,
        pid: 30,
    };
    /// The device's bus number.
    const DEVICE_BUS_NUMBER: DEVPROPKEY = DEVPROPKEY {
        fmtid: PCI_COMMON_KEY,
        pid: 23,
    };

    /// Decodes the PCIe-spec link-speed identifier as gigatransfers per
    /// second. Identifiers are defined by `pciprop.h`; an unknown code
    /// yields [`None`] rather than a guess.
    const fn link_speed_gts(code: u32) -> Option<f64> {
        match code {
            1 => Some(2.5),
            2 => Some(5.0),
            3 => Some(8.0),
            4 => Some(16.0),
            5 => Some(32.0),
            6 => Some(64.0),
            _ => None,
        }
    }

    /// Reads a `UINT32` device property, or [`None`] when the property is
    /// absent or has another type.
    fn read_u32(info_set: HDEVINFO, info: &SP_DEVINFO_DATA, key: &DEVPROPKEY) -> Option<u32> {
        let mut property_type = DEVPROPTYPE::default();
        let mut buffer = [0u8; 4];
        // SAFETY: `info_set` and `info` are valid (the set is kept alive by
        // the caller and `info` was filled by `SetupDiEnumDeviceInfo`); the
        // `buffer` is exactly the four bytes a `UINT32` property needs, and
        // the driver writes only the type and the buffer on success.
        unsafe {
            SetupDiGetDevicePropertyW(
                info_set,
                info,
                key,
                &mut property_type,
                Some(&mut buffer),
                None,
                0,
            )
        }
        .ok()?;
        if property_type != DEVPROP_TYPE_UINT32 {
            return None;
        }
        Some(u32::from_ne_bytes(buffer))
    }

    /// Reads the negotiated link of the PCI device whose bus number is
    /// `bus` and whose bus address is `(device << 16) | function`.
    pub(super) fn link_for_bdf(_domain: u32, bus: u32, device: u32, function: u32) -> Option<HostLink> {
        let wanted_address = (device << 16) | (function & 0xffff);
        // SAFETY: no class filter and the null enumerator request every
        // present device; the returned set is owned by this function and
        // destroyed below.
        let info_set =
            unsafe { SetupDiGetClassDevsW(None, None::<&PCWSTR>, None, DIGCF_PRESENT | DIGCF_ALLCLASSES) }
                .ok()?;
        let result = scan(info_set, bus, wanted_address);
        // SAFETY: `info_set` was returned by `SetupDiGetClassDevsW` above
        // and is not used afterwards.
        let _ = unsafe { SetupDiDestroyDeviceInfoList(info_set) };
        result
    }

    /// Walks the device set looking for the slot, returning its link.
    fn scan(info_set: HDEVINFO, bus: u32, wanted_address: u32) -> Option<HostLink> {
        let mut index = 0u32;
        loop {
            let mut info = SP_DEVINFO_DATA {
                cbSize: core::mem::size_of::<SP_DEVINFO_DATA>() as u32,
                ClassGuid: GUID::from_u128(0),
                DevInst: 0,
                Reserved: 0,
            };
            // SAFETY: `info_set` is valid and `info.cbSize` is set; the
            // call writes `info` and returns an error once `index` runs
            // past the last member.
            if unsafe { SetupDiEnumDeviceInfo(info_set, index, &mut info) }.is_err() {
                return None;
            }
            index += 1;
            if read_u32(info_set, &info, &DEVICE_BUS_NUMBER) != Some(bus) {
                continue;
            }
            if read_u32(info_set, &info, &DEVICE_ADDRESS) != Some(wanted_address) {
                continue;
            }
            let code = read_u32(info_set, &info, &CURRENT_LINK_SPEED)?;
            let lanes = read_u32(info_set, &info, &CURRENT_LINK_WIDTH)?;
            return host_link(link_speed_gts(code)?, lanes);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_pcie_speed_identifiers_decode_to_gigatransfers() {
            assert_eq!(link_speed_gts(1), Some(2.5));
            assert_eq!(link_speed_gts(2), Some(5.0));
            assert_eq!(link_speed_gts(3), Some(8.0));
            assert_eq!(link_speed_gts(4), Some(16.0));
            assert_eq!(link_speed_gts(5), Some(32.0));
            assert_eq!(link_speed_gts(6), Some(64.0));
            assert_eq!(link_speed_gts(0), None);
            assert_eq!(link_speed_gts(7), None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_startup_estimate_is_positive_and_finite() {
        const { assert!(TRANSFER_STARTUP_SECONDS > 0.0 && TRANSFER_STARTUP_SECONDS.is_finite()) };
    }

    #[test]
    fn a_zero_or_negative_pair_declines() {
        assert!(host_link(0.0, 16).is_none());
        assert!(host_link(-1.0, 16).is_none());
        assert!(host_link(16.0, 0).is_none());
    }

    #[test]
    fn a_real_pair_yields_the_framed_bandwidth() {
        let link = host_link(16.0, 16).expect("a valid 16 GT/s x16 link");
        // PCIe 4.0 signaling frames with 128b/130b; per-lane bytes are
        // halved to units of bytes rather than bits.
        let expected = 16.0e9 * 16.0 * (128.0 / 130.0) / 8.0;
        assert!((link.bandwidth - expected).abs() < 1.0);
        assert_eq!(link.latency, TRANSFER_STARTUP_SECONDS);
    }
}
