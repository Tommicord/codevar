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

//! Portal for opening URIs (`org.freedesktop.portal.OpenURI`).
//!
//! Provides methods to open URIs, local files, and directories via an application chooser
//! backend (`org.freedesktop.impl.portal.AppChooser`).

use codevar_base::basic_xml::{XmlBuilder, XmlDocument};

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use crate::xdp_context::PortalContext;
use crate::xdp_documents::{DocumentFlags, plan_register_document};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{OptionKey, OptionMap, PortalValue, decode_options, encode_options, filter_options};

const APP_CHOOSER_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.AppChooser";
const OPEN_URI_INTERFACE: &str = "org.freedesktop.portal.OpenURI";
const OPEN_URI_VERSION: u32 = 5;
const CALL_TIMEOUT: Duration = Duration::from_secs(25);

const OPEN_URI_OPTIONS: &[OptionKey] = &[
    OptionKey::new("parent_window", "s"),
    OptionKey::new("uri", "s"),
    OptionKey::new("handle_token", "s"),
];

const OPEN_FILE_OPTIONS: &[OptionKey] = &[
    OptionKey::new("parent_window", "s"),
    OptionKey::new("fd", "h"),
    OptionKey::new("handle_token", "s"),
];

const OPEN_DIRECTORY_OPTIONS: &[OptionKey] = &[
    OptionKey::new("parent_window", "s"),
    OptionKey::new("fd", "h"),
    OptionKey::new("handle_token", "s"),
];

const SCHEME_SUPPORTED_OPTIONS: &[OptionKey] =
    &[OptionKey::new("scheme", "s"), OptionKey::new("options", "a{sv}")];

fn handle_open_uri<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let parent_window = reader
        .read_str()?
        .to_string();
    let uri = reader
        .read_str()?
        .to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, OPEN_URI_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle
        .app_info
        .id()
        .to_string();

    ctx.call_impl(
        APP_CHOOSER_IMPL_INTERFACE,
        "OpenURI",
        CALL_TIMEOUT,
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&parent_window)?;
            bw.write_str(&uri)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_open_file<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let parent_window = reader
        .read_str()?
        .to_string();
    let fd = reader.read_fd()?;
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, OPEN_FILE_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle
        .app_info
        .id()
        .to_string();

    ctx.call_impl(
        APP_CHOOSER_IMPL_INTERFACE,
        "OpenFile",
        CALL_TIMEOUT,
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&parent_window)?;
            bw.write_fd(fd)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_open_directory<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let parent_window = reader
        .read_str()?
        .to_string();
    let fd = reader.read_fd()?;
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, OPEN_DIRECTORY_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle
        .app_info
        .id()
        .to_string();

    ctx.call_impl(
        APP_CHOOSER_IMPL_INTERFACE,
        "OpenDirectory",
        CALL_TIMEOUT,
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&parent_window)?;
            bw.write_fd(fd)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_scheme_supported<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &crate::xdp_context::MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let scheme = reader
        .read_str()?
        .to_string();
    let _options = crate::xdp_utils::decode_options(&mut reader)?;

    let _filtered = filter_options(&_options, SCHEME_SUPPORTED_OPTIONS)?;

    let app_info = crate::xdp_app_info::AppInfo::host(&inv.sender);

    let reply = ctx
        .call_impl(
            APP_CHOOSER_IMPL_INTERFACE,
            "SchemeSupported",
            CALL_TIMEOUT,
            |bw| {
                bw.write_str(app_info.id())?;
                bw.write_str(&scheme)?;
                bw.write_array("{sv}", |_| Ok(()))
            },
        )
        .map_err(|e| PortalError::InvalidArgument(format!("Failed to call backend: {}", e)))?;

    let mut reply_reader = reply.body_reader();
    let supported = !scheme.is_empty();
    ctx.reply(inv, |bw| bw.write_bool(supported))
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let iface_xml = XmlBuilder::new("interface")
        .attr("name", "org.freedesktop.portal.OpenURI")
        .child("method")
        .attr("name", "OpenURI")
        .child("arg")
        .attr("type", "s")
        .attr("name", "parent_window")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "s")
        .attr("name", "uri")
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
        .attr("name", "OpenFile")
        .child("arg")
        .attr("type", "s")
        .attr("name", "parent_window")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "h")
        .attr("name", "fd")
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
        .attr("name", "OpenDirectory")
        .child("arg")
        .attr("type", "s")
        .attr("name", "parent_window")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "h")
        .attr("name", "fd")
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
        .attr("name", "SchemeSupported")
        .child("arg")
        .attr("type", "s")
        .attr("name", "scheme")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "b")
        .attr("name", "supported")
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
        name: OPEN_URI_INTERFACE,
        version: OPEN_URI_VERSION,
        introspect_xml: iface_xml,
        methods: &[
            (
                "OpenURI",
                handle_open_uri
                    as fn(&mut PortalContext<T>, &crate::xdp_context::MethodInvocation) -> XdpResult<()>,
            ),
            ("OpenFile", handle_open_file),
            ("OpenDirectory", handle_open_directory),
            ("SchemeSupported", handle_scheme_supported),
        ],
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdp_utils::{OptionMap, PortalValue};

    #[test]
    fn filters_open_uri_options() {
        let mut options = OptionMap::new();
        options.insert("parent_window".to_string(), PortalValue::Str("test".to_string()));
        options.insert(
            "uri".to_string(),
            PortalValue::Str("https://example.com".to_string()),
        );
        options.insert("handle_token".to_string(), PortalValue::Str("token".to_string()));
        options.insert("unknown".to_string(), PortalValue::Str("x".to_string()));

        let filtered = filter_options(&options, OPEN_URI_OPTIONS).unwrap();
        assert_eq!(filtered.len(), 3);
    }
}
