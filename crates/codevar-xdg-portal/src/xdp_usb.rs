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

use codevar_base::basic_xml::XmlBuilder;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use crate::xdp_context::PortalContext;
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{OptionKey, OptionMap, PortalValue, encode_options, filter_options};

const USB_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.Usb";
const USB_INTERFACE: &str = "org.freedesktop.portal.Usb";
const USB_VERSION: u32 = 1;
const CALL_TIMEOUT: Duration = Duration::from_secs(25);
const MAX_DEVICES_PER_FINISH: usize = 8;

fn validate_usb_bool(_key: &str, _value: &PortalValue, _options: &OptionMap) -> Result<(), PortalError> {
    Ok(())
}

fn validate_usb_string(_key: &str, _value: &PortalValue, _options: &OptionMap) -> Result<(), PortalError> {
    Ok(())
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
        "org.freedesktop.impl.portal.Usb",
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
    let app_info = crate::xdp_app_info::AppInfo::host(&inv.sender)?;
    let reply = ctx.call_impl(
        "org.freedesktop.impl.portal.Usb",
        "EnumerateDevices",
        CALL_TIMEOUT,
        |bw| {
            bw.write_str(app_info.id())?;
            bw.write_array("{sv}", |_| Ok(()))
        },
    )?;

    let mut reply_reader = reply.body_reader();
    let mut devices_array = reply_reader.read_array(1)?;
    let mut devices = Vec::new();
    while !devices_array.is_empty() {
        let mut device_dict = devices_array.read_dict(2)?;
        let mut device_id = String::new();
        let mut device_properties = OptionMap::new();
        while !device_dict.is_empty() {
            let key = device_dict.read_str()?.to_string();
            let value = crate::xdp_utils::decode_variant(&mut device_dict)?;
            if key == "id" {
                device_id = value.to_string();
            } else {
                device_properties.insert(key, value);
            }
        }
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
    let mut devices_array = reader.read_array(1)?;
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let _filtered = filter_options(&options, ACQUIRE_DEVICES_OPTIONS)?;

    let mut devices = Vec::new();
    while !devices_array.is_empty() {
        let mut device_dict = devices_array.read_dict(2)?;
        let mut device_id = String::new();
        let mut device_options = OptionMap::new();
        while !device_dict.is_empty() {
            let key = device_dict.read_str()?.to_string();
            let value = crate::xdp_utils::PortalValue::decode_variant(&mut device_dict)?;
            if key == "id" {
                device_id = value.to_string();
            } else {
                device_options.insert(key, value);
            }
        }
        devices.push((device_id, device_options));
    }

    let handle = ctx.begin_request(inv, &_filtered)?;
    let app_id = handle.app_info.id().to_string();
    ctx.call_impl(
        "org.freedesktop.impl.portal.Usb",
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
    let _options = crate::xdp_utils::decode_options(&mut reader)?;

    let handle = ctx
        .take_request(&handle_path)
        .ok_or_else(|| PortalError::NotFound("Request not found".to_string()))?;
    let reply = ctx.call_impl(
        "org.freedesktop.impl.portal.Usb",
        "FinishAcquireDevices",
        CALL_TIMEOUT,
        |bw| {
            bw.write_object_path(&handle_path)?;
            bw.write_array("{sv}", |_| Ok(()))
        },
    )?;
    let mut reply_reader = reply.body_reader();
    let mut results_array = reply_reader.read_array(1)?;
    let mut results = Vec::new();
    let mut finished = false;
    while !results_array.is_empty() {
        let mut result_dict = results_array.read_dict(3)?;
        let mut device_id = String::new();
        let mut result = String::new();
        while !result_dict.is_empty() {
            let key = result_dict.read_str()?.to_string();
            let value = crate::xdp_utils::PortalValue::decode_variant(&mut result_dict)?;
            if key == "id" {
                device_id = value.to_string();
            } else if key == "result" {
                result = value.to_string();
            }
        }
        results.push((device_id, result));
    }
    finished = reply_reader.read_bool().unwrap_or(false);
    ctx.complete_request(&handle, 0, &OptionMap::new())?;
    ctx.reply(inv, |bw| {
        bw.write_array("(sa{sv})", |inner| {
            for (id, res) in &results {
                inner.write_struct("sa{sv}", |s| {
                    s.write_str(id)?;
                    s.write_dict("a{sv}", |d| {
                        d.write_str("result")?;
                        d.write_str(res)
                    })?;
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
    let _devices = reader.read_array(1)?;
    let _options = crate::xdp_utils::decode_options(&mut reader)?;
    let _filtered = filter_options(&_options, RELEASE_DEVICES_OPTIONS)?;
    let app_info = crate::xdp_app_info::AppInfo::host(&inv.sender);

    ctx.call_impl(
        "org.freedesktop.impl.portal.Usb",
        "ReleaseDevices",
        CALL_TIMEOUT,
        |bw| {
            bw.write_str(app_info.id())?;
            bw.write_array("s", |_| Ok(()))?;
            bw.write_array("{sv}", |_| Ok(()))
        },
    )?;
    let _ = ctx.conn.recv_timeout(CALL_TIMEOUT);
    let _ = device_id;
    ctx.reply_empty(inv)
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    #[rustfmt::skip]
    let iface_xml = XmlBuilder::new("interface")
        .attr("name", "org.freedesktop.portal.Usb")
        .child("method")
            .attr("name", "CreateSession")
            .child("arg")
                .attr("type", "a{sv}")
                .attr("name", "options")
                .attr("direction", "in")
                .end()
            .child("arg")
                .attr("type", "o")
                .attr("name", "session_handle")
                .attr("direction", "out")
                .end()
            .end()
        .child("method")
        .attr("name", "EnumerateDevices")
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a(sa{sv})")
        .attr("name", "devices")
        .attr("direction", "out")
        .end()
        .end()
        .child("method")
        .attr("name", "AcquireDevices")
        .child("arg")
        .attr("type", "s")
        .attr("name", "parent_window")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a(sa{sv})")
        .attr("name", "devices")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "o")
        .attr("name", "handle")
        .attr("direction", "out")
        .end()
        .end()
        .child("method")
        .attr("name", "FinishAcquireDevices")
        .child("arg")
        .attr("type", "o")
        .attr("name", "handle")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a(sa{sv})")
        .attr("name", "results")
        .attr("direction", "out")
        .end()
        .child("arg")
        .attr("type", "b")
        .attr("name", "finished")
        .attr("direction", "out")
        .end()
        .end()
        .child("method")
        .attr("name", "ReleaseDevices")
        .child("arg")
        .attr("type", "as")
        .attr("name", "devices")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .end()
        .child("signal")
        .attr("name", "DeviceEvents")
        .child("arg")
        .attr("type", "o")
        .attr("name", "session_handle")
        .attr("direction", "out")
        .end()
        .child("arg")
        .attr("type", "a(ssa{sv})")
        .attr("name", "events")
        .attr("direction", "out")
        .end()
        .end()
        .child("property")
        .attr("name", "version")
        .attr("type", "u")
        .attr("access", "read")
        .end()
        .build();
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
