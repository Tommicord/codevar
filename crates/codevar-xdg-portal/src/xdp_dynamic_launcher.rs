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

//! Dynamic Launcher portal frontend.
//!
//! Implements `org.freedesktop.portal.DynamicLauncher` for installing
//! application launchers (.desktop files) with icons. Supports
//! PrepareInstall (interactive), RequestInstallToken (non-interactive),
//! Install, Uninstall, GetDesktopEntry, GetIcon, and Launch.

use alloc::string::ToString;
use codevar_base::basic_xml::XmlBuilder;

use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn, PortalInterface};
use crate::xdp_error::XdpResult;
use crate::xdp_utils::{OptionKey, encode_options, filter_options};

const DYNAMIC_LAUNCHER_INTERFACE: &str = "org.freedesktop.portal.DynamicLauncher";
const DYNAMIC_LAUNCHER_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.DynamicLauncher";
const DYNAMIC_LAUNCHER_VERSION: u32 = 1;

const INSTALL_OPTIONS: &[OptionKey] = &[];
const PREPARE_INSTALL_OPTIONS: &[OptionKey] = &[
    OptionKey::new("handle_token", "s"),
    OptionKey::new("modal", "b"),
    OptionKey::new("launcher_type", "u"),
    OptionKey::new("target", "s"),
    OptionKey::new("editable_name", "b"),
    OptionKey::new("editable_icon", "b"),
];
const REQUEST_INSTALL_TOKEN_OPTIONS: &[OptionKey] = &[
    OptionKey::new("launcher_type", "u"),
    OptionKey::new("target", "s"),
];
const UNINSTALL_OPTIONS: &[OptionKey] = &[];
const GET_DESKTOP_ENTRY_OPTIONS: &[OptionKey] = &[];
const GET_ICON_OPTIONS: &[OptionKey] = &[];
const LAUNCH_OPTIONS: &[OptionKey] = &[];

fn handle_install<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let token = reader.read_str()?.to_string();
    let desktop_file_id = reader.read_str()?.to_string();
    let desktop_entry = reader.read_str()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, INSTALL_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    ctx.call_impl(
        DYNAMIC_LAUNCHER_IMPL_INTERFACE,
        "Install",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_str(&token)?;
            bw.write_str(&desktop_file_id)?;
            bw.write_str(&desktop_entry)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_prepare_install<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let parent_window = reader.read_str()?.to_string();
    let name = reader.read_str()?.to_string();
    let icon_v = crate::xdp_utils::PortalValue::decode_variant(&mut reader)?;
    let options = crate::xdp_utils::decode_options(&mut reader)?;
    let filtered = filter_options(&options, PREPARE_INSTALL_OPTIONS)?;
    let handle = ctx.begin_request(inv, &filtered)?;
    let app_id = handle.app_info.id().to_string();

    ctx.call_impl(
        DYNAMIC_LAUNCHER_IMPL_INTERFACE,
        "PrepareInstall",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&parent_window)?;
            bw.write_str(&name)?;
            crate::xdp_utils::write_value(bw, &icon_v)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_request_install_token<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let name = reader.read_str()?.to_string();
    let icon_v = crate::xdp_utils::PortalValue::decode_variant(&mut reader)?;
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, REQUEST_INSTALL_TOKEN_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle.app_info.id().to_string();

    ctx.call_impl(
        DYNAMIC_LAUNCHER_IMPL_INTERFACE,
        "RequestInstallToken",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&name)?;
            crate::xdp_utils::write_value(bw, &icon_v)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_uninstall<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let desktop_file_id = reader.read_str()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, UNINSTALL_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    ctx.call_impl(
        DYNAMIC_LAUNCHER_IMPL_INTERFACE,
        "Uninstall",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&desktop_file_id)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_get_desktop_entry<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let desktop_file_id = reader.read_str()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, GET_DESKTOP_ENTRY_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle.app_info.id().to_string();

    ctx.call_impl(
        DYNAMIC_LAUNCHER_IMPL_INTERFACE,
        "GetDesktopEntry",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&desktop_file_id)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_get_icon<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let desktop_file_id = reader.read_str()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, GET_ICON_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle.app_info.id().to_string();

    ctx.call_impl(
        DYNAMIC_LAUNCHER_IMPL_INTERFACE,
        "GetIcon",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&desktop_file_id)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_launch<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let desktop_file_id = reader.read_str()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, LAUNCH_OPTIONS)?;

    let app_info = crate::xdp_app_info::AppInfo::host(&inv.sender);

    ctx.call_impl(
        DYNAMIC_LAUNCHER_IMPL_INTERFACE,
        "Launch",
        core::time::Duration::from_secs(30),
        |bw| {
            bw.write_str(app_info.id())?;
            bw.write_str(&desktop_file_id)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply_empty(inv)
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let iface_xml = XmlBuilder::new("interface")
        .attr("name", "org.freedesktop.portal.DynamicLauncher")
        .child("method")
        .attr("name", "Install")
        .child("arg")
        .attr("type", "s")
        .attr("name", "token")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "s")
        .attr("name", "desktop_file_id")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "s")
        .attr("name", "desktop_entry")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .end()
        .child("method")
        .attr("name", "PrepareInstall")
        .child("arg")
        .attr("type", "s")
        .attr("name", "parent_window")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "s")
        .attr("name", "name")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "v")
        .attr("name", "icon_v")
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
        .attr("name", "RequestInstallToken")
        .child("arg")
        .attr("type", "s")
        .attr("name", "name")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "v")
        .attr("name", "icon_v")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "s")
        .attr("name", "token")
        .attr("direction", "out")
        .end()
        .end()
        .child("method")
        .attr("name", "Uninstall")
        .child("arg")
        .attr("type", "s")
        .attr("name", "desktop_file_id")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .end()
        .child("method")
        .attr("name", "GetDesktopEntry")
        .child("arg")
        .attr("type", "s")
        .attr("name", "desktop_file_id")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "s")
        .attr("name", "contents")
        .attr("direction", "out")
        .end()
        .end()
        .child("method")
        .attr("name", "GetIcon")
        .child("arg")
        .attr("type", "s")
        .attr("name", "desktop_file_id")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "v")
        .attr("name", "icon_v")
        .attr("direction", "out")
        .end()
        .child("arg")
        .attr("type", "s")
        .attr("name", "icon_format")
        .attr("direction", "out")
        .end()
        .child("arg")
        .attr("type", "u")
        .attr("name", "icon_size")
        .attr("direction", "out")
        .end()
        .end()
        .child("method")
        .attr("name", "Launch")
        .child("arg")
        .attr("type", "s")
        .attr("name", "desktop_file_id")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .end()
        .child("property")
        .attr("name", "SupportedLauncherTypes")
        .attr("type", "u")
        .attr("access", "read")
        .end()
        .child("property")
        .attr("name", "version")
        .attr("type", "u")
        .attr("access", "read")
        .end()
        .build();

    let interface = PortalInterface {
        name: DYNAMIC_LAUNCHER_INTERFACE,
        version: DYNAMIC_LAUNCHER_VERSION,
        introspect_xml: iface_xml,
        methods: &[
            ("Install", handle_install as PortalFn<T>),
            ("PrepareInstall", handle_prepare_install as PortalFn<T>),
            ("RequestInstallToken", handle_request_install_token as PortalFn<T>),
            ("Uninstall", handle_uninstall as PortalFn<T>),
            ("GetDesktopEntry", handle_get_desktop_entry as PortalFn<T>),
            ("GetIcon", handle_get_icon as PortalFn<T>),
            ("Launch", handle_launch as PortalFn<T>),
        ],
    };
    ctx.register_interface(interface);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdp_utils::{OptionMap, PortalValue};

    #[test]
    fn filters_prepare_install_options() {
        let mut options = OptionMap::new();
        options.insert("handle_token".to_string(), PortalValue::Str("abc123".to_string()));
        options.insert("modal".to_string(), PortalValue::Bool(true));
        options.insert("launcher_type".to_string(), PortalValue::U32(1));
        options.insert(
            "target".to_string(),
            PortalValue::Str("https://example.com".to_string()),
        );
        options.insert("editable_name".to_string(), PortalValue::Bool(true));
        options.insert("editable_icon".to_string(), PortalValue::Bool(false));

        let filtered = filter_options(&options, PREPARE_INSTALL_OPTIONS).unwrap();
        assert_eq!(filtered.len(), 6);
    }
}
