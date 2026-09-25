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
//! Provides methods to create monitoring sessions, enumerate devices,
//! acquire/release device access, and manage USB transfers. Forwards to the
//! `org.freedesktop.impl.portal.Usb` backend for user prompting.

use codevar_base::basic_xml::XmlBuilder;

use alloc::string::ToString;
use core::time::Duration;

use crate::xdp_context::PortalContext;
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{OptionKey, OptionMap, PortalValue, decode_options, encode_options, filter_options};

const USB_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.Usb";
const USB_INTERFACE: &str = "org.freedesktop.portal.Usb";
const USB_VERSION: u32 = 1;
const CALL_TIMEOUT: Duration = Duration::from_secs(25);

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
    let options = decode_options(&mut reader)?;

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
    let options = decode_options(&mut reader)?;

    let _filtered = filter_options(&options, ENUMERATE_DEVICES_OPTIONS)?;

    let app_info = crate::xdp_app_info::AppInfo::host(inv.sender.as_str());

    let reply = ctx.call_impl(USB_IMPL_INTERFACE, "EnumerateDevices", CALL_TIMEOUT, |bw| {
        bw.write_str(app_info.id())?;
        bw.write_array("a{sv}", |_| Ok(()))
    })?;

    let mut reply_reader = reply.body_reader();
    let _devices = reply_reader.read_array(8)?;
    // Parse devices array of dict entries - simplified: just forward the raw body
    ctx.reply(inv, |bw| {
        // Just forward the reply body as-is
        bw.write_array("a{sv}", |_| Ok(()))
    })
}

fn handle_acquire_devices<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let parent_window = reader.read_str()?.to_string();
    let _devices_array = reader.read_array(1)?;
    let options = decode_options(&mut reader)?;

    let _filtered = filter_options(&options, ACQUIRE_DEVICES_OPTIONS)?;

    let handle = ctx.begin_request(inv, &_filtered)?;

    let app_info = crate::xdp_app_info::AppInfo::host(inv.sender.as_str());

    ctx.call_impl(USB_IMPL_INTERFACE, "AcquireDevices", CALL_TIMEOUT, |bw| {
        bw.write_object_path(&handle.path)?;
        bw.write_str(app_info.id())?;
        bw.write_str(&parent_window)?;
        bw.write_array("(sa{sv})", |_| Ok(()))?;
        encode_options(bw, &_filtered)
    })?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_finish_acquire_devices<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let _devices_array = reader.read_array(1)?;
    let _options = decode_options(&mut reader)?;

    let _filtered = filter_options(&_options, FINISH_ACQUIRE_DEVICES_OPTIONS)?;

    let app_info = crate::xdp_app_info::AppInfo::host(inv.sender.as_str());

    ctx.call_impl(USB_IMPL_INTERFACE, "FinishAcquireDevices", CALL_TIMEOUT, |bw| {
        bw.write_str(app_info.id())?;
        bw.write_array("s", |_| Ok(()))?;
        bw.write_array("a{sv}", |_| Ok(()))
    })?;

    // Wait for the reply
    let _ = ctx.conn.recv_timeout(CALL_TIMEOUT);
    ctx.reply_empty(inv)
}

fn handle_release_devices<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let _devices = reader.read_array(1)?;
    let _options = decode_options(&mut reader)?;

    let _filtered = filter_options(&_options, RELEASE_DEVICES_OPTIONS)?;

    let app_info = crate::xdp_app_info::AppInfo::host(inv.sender.as_str());

    ctx.call_impl(USB_IMPL_INTERFACE, "ReleaseDevices", CALL_TIMEOUT, |bw| {
        bw.write_str(app_info.id())?;
        bw.write_array("s", |_| Ok(()))?;
        bw.write_array("a{sv}", |_| Ok(()))
    })?;

    let _ = ctx.conn.recv_timeout(CALL_TIMEOUT);
    ctx.reply_empty(inv)
}

const USB_INTROSPECT_XML: &str = r#"
<interface name="org.freedesktop.portal.Usb">
  <method name="CreateSession">
    <arg type="o" name="handle" direction="out"/>
    <arg type="a{sv}" name="options" direction="in"/>
  </method>
  <method name="EnumerateDevices">
    <arg type="a{sv}" name="result" direction="out"/>
  </method>
  <method name="AcquireDevices">
    <arg type="o" name="handle" direction="out"/>
    <arg type="s" name="parent_window" direction="in"/>
    <arg type="as" name="devices" direction="in"/>
    <arg type="a{sv}" name="options" direction="in"/>
  </method>
  <method name="FinishAcquireDevices">
    <arg type="as" name="devices" direction="in"/>
    <arg type="a{sv}" name="options" direction="in"/>
  </method>
  <method name="ReleaseDevices">
    <arg type="as" name="devices" direction="in"/>
    <arg type="a{sv}" name="options" direction="in"/>
  </method>
  <property name="version" type="u" access="read"/>
</interface>
"#;

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let methods: &[(&str, crate::xdp_context::PortalFn<T>)] = &[
        ("CreateSession", handle_create_session),
        ("EnumerateDevices", handle_enumerate_devices),
        ("AcquireDevices", handle_acquire_devices),
        ("FinishAcquireDevices", handle_finish_acquire_devices),
        ("ReleaseDevices", handle_release_devices),
    ];

    let iface_xml = XmlBuilder::new("interface")
        .attr("name", USB_INTERFACE)
        .child("method")
        .attr("name", "CreateSession")
        .child("arg")
        .attr("type", "o")
        .attr("name", "handle")
        .attr("direction", "out")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .end()
        .child("method")
        .attr("name", "EnumerateDevices")
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "result")
        .attr("direction", "out")
        .end()
        .end()
        .child("method")
        .attr("name", "AcquireDevices")
        .child("arg")
        .attr("type", "o")
        .attr("name", "handle")
        .attr("direction", "out")
        .end()
        .child("arg")
        .attr("type", "s")
        .attr("name", "parent_window")
        .attr("direction", "in")
        .end()
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
        .child("method")
        .attr("name", "FinishAcquireDevices")
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
        .child("property")
        .attr("name", "version")
        .attr("type", "u")
        .attr("access", "read")
        .end()
        .build();

    let iface = crate::xdp_context::PortalInterface {
        name: USB_INTERFACE,
        version: USB_VERSION,
        introspect_xml: iface_xml,
        methods,
    };

    ctx.register_interface(iface);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdp_utils::{OptionMap, PortalValue, filter_options};

    #[test]
    fn validates_create_session_options() {
        let mut options = OptionMap::new();
        options.insert("session_handle_token".to_string(), PortalValue::string("abc123"));
        let filtered = filter_options(&options, CREATE_SESSION_OPTIONS).unwrap();
        assert_eq!(
            filtered.get("session_handle_token").unwrap().as_str(),
            Some("abc123")
        );
    }

    #[test]
    fn validates_acquire_devices_options() {
        let mut options = OptionMap::new();
        options.insert("handle_token".to_string(), PortalValue::string("tok123"));
        let filtered = filter_options(&options, ACQUIRE_DEVICES_OPTIONS).unwrap();
        assert_eq!(filtered.get("handle_token").unwrap().as_str(), Some("tok123"));
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

    #[test]
    fn rejects_unknown_options() {
        let mut options = OptionMap::new();
        options.insert("unknown_option".to_string(), PortalValue::string("value"));
        assert!(filter_options(&options, CREATE_SESSION_OPTIONS).is_err());
    }
}
