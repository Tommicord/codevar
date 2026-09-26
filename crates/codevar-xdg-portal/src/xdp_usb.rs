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

//! Portal for USB device access (`org.freedesktop.portal.Usb`).
//!
//! Provides methods to create
//! monitoring sessions, enumerate devices, acquire/release device
//! access, and manage USB transfers. Forwards to the
//! `org.freedesktop.impl.portal.Usb` backend for user prompting.

use codevar_base::xml;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use codevar_dbus::DbusReader;

use crate::xdp_context::PortalContext;
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{OptionKey, OptionMap, PortalValue, encode_options, filter_options};

const USB_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.Usb";
const USB_INTERFACE: &str = "org.freedesktop.portal.Usb";
const USB_VERSION: u32 = 1;
const CALL_TIMEOUT: Duration = Duration::from_secs(25);
const MAX_DEVICES_PER_FINISH: usize = 8;

/// Renders a decoded value as a string for the flat device tuples;
/// only string-like values are expected for these keys.
fn value_as_string(value: &PortalValue) -> String {
    match value {
        PortalValue::Str(s) | PortalValue::ObjectPath(s) => s.clone(),
        other => format!("{other:?}"),
    }
}

/// Reads one `a{sv}` dictionary from `reader`.
fn read_options_map(reader: &mut DbusReader<'_>) -> XdpResult<OptionMap> {
    let mut array = reader.read_array(8)?;
    let mut map = OptionMap::new();
    while !array.is_empty() {
        array.read_struct()?;
        let key = array.read_str()?.to_string();
        let value = PortalValue::decode_variant(&mut array)?;
        map.insert(key, value);
    }
    Ok(map)
}

/// Reads one `(sa{sv})` element (device id plus properties) from `reader`.
fn read_device_entry(reader: &mut DbusReader<'_>) -> XdpResult<(String, OptionMap)> {
    reader.read_struct()?;
    let id = reader.read_str()?.to_string();
    let props = read_options_map(reader)?;
    Ok((id, props))
}

const CREATE_SESSION_OPTIONS: &[OptionKey] = &[OptionKey::new("session_handle_token", "s")];

const ENUMERATE_DEVICES_OPTIONS: &[OptionKey] = &[];

const ACQUIRE_DEVICES_OPTIONS: &[OptionKey] = &[OptionKey::new("handle_token", "s")];

const ACQUIRE_DEVICES_DEVICE_OPTIONS: &[OptionKey] =
    &[OptionKey::new("writable", "b"), OptionKey::new("serial", "s")];

const FINISH_ACQUIRE_DEVICES_OPTIONS: &[OptionKey] = &[];

const RELEASE_DEVICES_OPTIONS: &[OptionKey] = &[];

fn handle_create_session<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, CREATE_SESSION_OPTIONS)?;

    let handle = ctx.begin_session(inv, &filtered)?;

    ctx.call_impl(
        USB_IMPL_INTERFACE,
        "CreateSession",
        CALL_TIMEOUT,
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_enumerate_devices<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let _filtered = filter_options(&options, ENUMERATE_DEVICES_OPTIONS)?;
    let app_info = crate::xdp_app_info::AppInfo::host(&inv.sender);
    let reply = ctx.call_impl(USB_IMPL_INTERFACE, "EnumerateDevices", CALL_TIMEOUT, |bw| {
        bw.write_str(app_info.id())?;
        bw.write_array("{sv}", |_| Ok(()))
    })?;

    let mut reply_reader = reply.body_reader();
    let mut devices_array = reply_reader.read_array(8)?;
    let mut devices = Vec::new();
    while !devices_array.is_empty() {
        let (device_id, device_properties) = read_device_entry(&mut devices_array)?;
        devices.push((device_id, device_properties));
    }

    ctx.reply(inv, |bw| {
        bw.write_array("(sa{sv})", |inner| {
            for (id, props) in &devices {
                inner.write_struct("sa{sv}", |s| {
                    s.write_str(id)?;
                    encode_options(s, props)
                })?;
            }
            Ok(())
        })
    })
}

fn handle_acquire_devices<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let parent_window = reader.read_str()?.to_string();
    let mut devices_array = reader.read_array(8)?;
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let _filtered = filter_options(&options, ACQUIRE_DEVICES_OPTIONS)?;

    let mut devices = Vec::new();
    while !devices_array.is_empty() {
        let (device_id, device_options) = read_device_entry(&mut devices_array)?;
        let device_options = filter_options(&device_options, ACQUIRE_DEVICES_DEVICE_OPTIONS)?;
        devices.push((device_id, device_options));
    }

    let handle = ctx.begin_request(inv, &_filtered)?;
    let app_id = handle.app_info.id().to_string();
    ctx.call_impl(
        USB_IMPL_INTERFACE,
        "AcquireDevices",
        CALL_TIMEOUT,
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&parent_window)?;
            bw.write_array("(sa{sv})", |inner| {
                for (id, opts) in &devices {
                    inner.write_struct("sa{sv}", |s| {
                        s.write_str(id)?;
                        encode_options(s, opts)
                    })?;
                }
                Ok(())
            })?;
            encode_options(bw, &_filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_finish_acquire_devices<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let handle_path = reader.read_object_path()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;
    let _filtered = filter_options(&options, FINISH_ACQUIRE_DEVICES_OPTIONS)?;

    let handle = ctx
        .take_request(&handle_path)
        .ok_or_else(|| PortalError::NotFound("Request not found".to_string()))?;
    let reply = ctx.call_impl(USB_IMPL_INTERFACE, "FinishAcquireDevices", CALL_TIMEOUT, |bw| {
        bw.write_object_path(&handle_path)?;
        bw.write_array("{sv}", |_| Ok(()))
    })?;
    let mut results = Vec::new();
    let finished;
    {
        let mut reply_reader = reply.body_reader();
        let mut results_array = reply_reader.read_array(8)?;
        while !results_array.is_empty() && results.len() < MAX_DEVICES_PER_FINISH {
            let (device_id, props) = read_device_entry(&mut results_array)?;
            let result = props
                .get("result")
                .map(value_as_string)
                .unwrap_or_default();
            results.push((device_id, result));
        }
        finished = reply_reader.read_bool().unwrap_or(false);
    }
    ctx.complete_request(&handle, 0, &OptionMap::new())?;
    ctx.reply(inv, |bw| {
        bw.write_array("(sa{sv})", |inner| {
            for (id, res) in &results {
                let mut props = OptionMap::new();
                props.insert(String::from("result"), PortalValue::Str(res.clone()));
                inner.write_struct("sa{sv}", |s| {
                    s.write_str(id)?;
                    encode_options(s, &props)
                })?;
            }
            Ok(())
        })?;
        bw.write_bool(finished)
    })
}

fn handle_release_devices<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let _devices = reader.read_array(4)?;
    let _options = crate::xdp_utils::decode_options(&mut reader)?;
    let _filtered = filter_options(&_options, RELEASE_DEVICES_OPTIONS)?;
    let app_info = crate::xdp_app_info::AppInfo::host(&inv.sender);

    ctx.call_impl(USB_IMPL_INTERFACE, "ReleaseDevices", CALL_TIMEOUT, |bw| {
        bw.write_str(app_info.id())?;
        bw.write_array("s", |_| Ok(()))?;
        bw.write_array("{sv}", |_| Ok(()))
    })?;
    ctx.reply_empty(inv)
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let iface_xml = xml!(interface, attrs: ["name" = "org.freedesktop.portal.Usb"], children: [
        (method, attrs: ["name" = "CreateSession"], children: [
            (arg, attrs: ["type" = "a{sv}", "name" = "options", "direction" = "in"]),
            (arg, attrs: ["type" = "o", "name" = "session_handle", "direction" = "out"])
        ]),
        (method, attrs: ["name" = "EnumerateDevices"], children: [
            (arg, attrs: ["type" = "a{sv}", "name" = "options", "direction" = "in"]),
            (arg, attrs: ["type" = "a(sa{sv})", "name" = "devices", "direction" = "out"])
        ]),
        (method, attrs: ["name" = "AcquireDevices"], children: [
            (arg, attrs: ["type" = "s", "name" = "parent_window", "direction" = "in"]),
            (arg, attrs: ["type" = "a(sa{sv})", "name" = "devices", "direction" = "in"]),
            (arg, attrs: ["type" = "a{sv}", "name" = "options", "direction" = "in"]),
            (arg, attrs: ["type" = "o", "name" = "handle", "direction" = "out"])
        ]),
        (method, attrs: ["name" = "FinishAcquireDevices"], children: [
            (arg, attrs: ["type" = "o", "name" = "handle", "direction" = "in"]),
            (arg, attrs: ["type" = "a{sv}", "name" = "options", "direction" = "in"]),
            (arg, attrs: ["type" = "a(sa{sv})", "name" = "results", "direction" = "out"]),
            (arg, attrs: ["type" = "b", "name" = "finished", "direction" = "out"])
        ]),
        (method, attrs: ["name" = "ReleaseDevices"], children: [
            (arg, attrs: ["type" = "as", "name" = "devices", "direction" = "in"]),
            (arg, attrs: ["type" = "a{sv}", "name" = "options", "direction" = "in"])
        ]),
        (signal, attrs: ["name" = "DeviceEvents"], children: [
            (arg, attrs: ["type" = "o", "name" = "session_handle", "direction" = "out"]),
            (arg, attrs: ["type" = "a(ssa{sv})", "name" = "events", "direction" = "out"])
        ]),
        (property, attrs: ["name" = "version", "type" = "u", "access" = "read"]),
    ]);
    ctx.register_interface(crate::xdp_context::PortalInterface {
        name: USB_INTERFACE,
        version: USB_VERSION,
        introspect_xml: iface_xml,
        methods: &[
            (
                "CreateSession",
                handle_create_session
                    as fn(&mut PortalContext<T>, &crate::xdp_context::MethodInvocation) -> XdpResult<()>,
            ),
            ("EnumerateDevices", handle_enumerate_devices),
            ("AcquireDevices", handle_acquire_devices),
            ("FinishAcquireDevices", handle_finish_acquire_devices),
            ("ReleaseDevices", handle_release_devices),
        ],
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdp_utils::{OptionMap, PortalValue};

    #[test]
    fn validates_create_session_options() {
        let mut options = OptionMap::new();
        options.insert(
            "session_handle_token".to_string(),
            PortalValue::Str("test_session".to_string()),
        );

        let filtered = filter_options(&options, CREATE_SESSION_OPTIONS).unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("session_handle_token"));
    }

    #[test]
    fn validates_acquire_devices_options() {
        let mut options = OptionMap::new();
        options.insert(
            "handle_token".to_string(),
            PortalValue::Str("test_handle".to_string()),
        );

        let filtered = filter_options(&options, ACQUIRE_DEVICES_OPTIONS).unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("handle_token"));
    }

    #[test]
    fn validates_acquire_device_access_options() {
        let mut options = OptionMap::new();
        options.insert("writable".to_string(), PortalValue::Bool(true));

        let filtered = filter_options(&options, ACQUIRE_DEVICES_DEVICE_OPTIONS).unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("writable"));
    }

    #[test]
    fn rejects_unknown_options() {
        let mut options = OptionMap::new();
        options.insert(
            "unknown_option".to_string(),
            PortalValue::Str("value".to_string()),
        );

        let filtered = filter_options(&options, CREATE_SESSION_OPTIONS).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn validates_enumerate_devices_empty_options() {
        let options = OptionMap::new();
        let filtered = filter_options(&options, ENUMERATE_DEVICES_OPTIONS).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn validates_finish_acquire_devices_empty_options() {
        let options = OptionMap::new();
        let filtered = filter_options(&options, FINISH_ACQUIRE_DEVICES_OPTIONS).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn validates_release_devices_empty_options() {
        let options = OptionMap::new();
        let filtered = filter_options(&options, RELEASE_DEVICES_OPTIONS).unwrap();
        assert!(filtered.is_empty());
    }
}
