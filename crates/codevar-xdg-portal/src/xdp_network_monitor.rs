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

//! org.freedesktop.portal.NetworkMonitor portal implementation.
//!
//! This portal provides network status information to sandboxed
//! applications. It does not involve user interaction.

use codevar_base::basic_xml::{XmlBuilder, XmlDocument};

use alloc::format;
use alloc::string::{String, ToString};

use crate::xdp_app_info::AppInfo;
use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn, PortalInterface};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{OptionKey, OptionMap, PortalValue, decode_options, encode_options};

/// Interface name for the NetworkMonitor portal.
const NETWORK_MONITOR_INTERFACE: &str = "org.freedesktop.portal.NetworkMonitor";

/// Implementation backend interface name.
const NETWORK_MONITOR_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.NetworkMonitor";

/// Timeout for implementation backend calls (1 hour).
const IMPL_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(3600);

/// Supported option keys (none for these methods).
const EMPTY_OPTION_KEYS: &[OptionKey] = &[];

/// Filters an OptionMap to only include supported keys with valid types.
fn filter_options_map(options: &OptionMap, supported: &[OptionKey]) -> Result<OptionMap, PortalError> {
    let mut filtered = OptionMap::new();
    let mut first_error: Option<PortalError> = None;
    for supported_key in supported {
        let Some(value) = options.get(supported_key.key) else {
            continue;
        };
        if !value.matches_signature(supported_key.type_signature) {
            if first_error.is_none() {
                first_error = Some(PortalError::InvalidArgument(format!(
                    "Expected type '{}' for option '{}', got '{}'",
                    supported_key.type_signature,
                    supported_key.key,
                    value.signature()
                )));
            }
            continue;
        }
        if let Some(validate) = supported_key.validate
            && let Err(error) = validate(supported_key.key, value, options)
        {
            if first_error.is_none() {
                first_error = Some(error);
            }
            continue;
        }
        filtered.insert(String::from(supported_key.key), value.clone());
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(filtered),
    }
}

const CAN_REACH_OPTIONS: &[OptionKey] = &[
    OptionKey::new("hostname", "s"),
    OptionKey::new("port", "u"),
];

/// Registers the NetworkMonitor portal interface.
pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let methods: &[(&'static str, PortalFn<T>)] = &[
        ("GetAvailable", handle_get_available as PortalFn<T>),
        ("GetMetered", handle_get_metered as PortalFn<T>),
        ("GetConnectivity", handle_get_connectivity as PortalFn<T>),
        ("GetStatus", handle_get_status as PortalFn<T>),
        ("CanReach", handle_can_reach as PortalFn<T>),
    ];

    let iface_xml = XmlBuilder::new("interface")
        .attr("name", "org.freedesktop.portal.NetworkMonitor")
        .child("signal")
            .attr("name", "changed")
        .end()
        .child("method")
            .attr("name", "GetAvailable")
            .child("arg")
                .attr("type", "b")
                .attr("name", "available")
                .attr("direction", "out")
            .end()
        .end()
        .child("method")
            .attr("name", "GetMetered")
            .child("arg")
                .attr("type", "b")
                .attr("name", "metered")
                .attr("direction", "out")
            .end()
        .end()
        .child("method")
            .attr("name", "GetConnectivity")
            .child("arg")
                .attr("type", "u")
                .attr("name", "connectivity")
                .attr("direction", "out")
            .end()
        .end()
        .child("method")
            .attr("name", "GetStatus")
            .child("arg")
                .attr("type", "a{sv}")
                .attr("name", "status")
                .attr("direction", "out")
            .end()
        .end()
        .child("method")
            .attr("name", "CanReach")
            .child("arg")
                .attr("type", "s")
                .attr("name", "hostname")
                .attr("direction", "in")
            .end()
            .child("arg")
                .attr("type", "u")
                .attr("name", "port")
                .attr("direction", "in")
            .end()
            .child("arg")
                .attr("type", "b")
                .attr("name", "reachable")
                .attr("direction", "out")
            .end()
        .end()
        .child("property")
            .attr("name", "version")
            .attr("type", "u")
            .attr("access", "read")
        .end()
        .build();

    let iface = PortalInterface {
        name: NETWORK_MONITOR_INTERFACE,
        version: 3,
        introspect_xml: iface_xml,
        methods,
    };

    ctx.register_interface(iface);
    Ok(())
}

fn handle_get_available<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let options = decode_options(&mut reader)?;
    let _filtered = filter_options_map(&options, EMPTY_OPTION_KEYS)?;

    let app_info = AppInfo::host(&inv.sender);

    if !app_info.has_network() {
        return Err(PortalError::NotAllowed(String::from(
            "This call is not available inside the sandbox",
        )));
    }

    let reply = ctx.call_impl(
        NETWORK_MONITOR_IMPL_INTERFACE,
        "GetAvailable",
        IMPL_TIMEOUT,
        |bw| {
            bw.write_str("")?; // app_id
            bw.write_str("")?; // parent_window
            bw.write_array("{sv}", |_| Ok(())) // empty options
        },
    )?;

    let mut reply_reader = reply.body_reader();
    let available = reply_reader.read_bool()?;

    ctx.reply(inv, |bw| bw.write_bool(available))
}

fn handle_get_metered<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let options = decode_options(&mut reader)?;
    let _filtered = filter_options_map(&options, EMPTY_OPTION_KEYS)?;

    let app_info = AppInfo::host(&inv.sender);

    if !app_info.has_network() {
        return Err(PortalError::NotAllowed(String::from(
            "This call is not available inside the sandbox",
        )));
    }

    let reply = ctx.call_impl(
        NETWORK_MONITOR_IMPL_INTERFACE,
        "GetMetered",
        IMPL_TIMEOUT,
        |bw| {
            bw.write_str("")?; // app_id
            bw.write_str("")?; // parent_window
            bw.write_array("{sv}", |_| Ok(())) // empty options
        },
    )?;

    let mut reply_reader = reply.body_reader();
    let metered = reply_reader.read_bool()?;

    ctx.reply(inv, |bw| bw.write_bool(metered))
}

fn handle_get_connectivity<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let options = decode_options(&mut reader)?;
    let _filtered = filter_options_map(&options, EMPTY_OPTION_KEYS)?;

    let app_info = AppInfo::host(&inv.sender);

    if !app_info.has_network() {
        return Err(PortalError::NotAllowed(String::from(
            "This call is not available inside the sandbox",
        )));
    }

    let reply = ctx.call_impl(
        NETWORK_MONITOR_IMPL_INTERFACE,
        "GetConnectivity",
        IMPL_TIMEOUT,
        |bw| {
            bw.write_str("")?; // app_id
            bw.write_str("")?; // parent_window
            bw.write_array("{sv}", |_| Ok(())) // empty options
        },
    )?;

    let mut reply_reader = reply.body_reader();
    let connectivity = reply_reader.read_u32()?;

    ctx.reply(inv, |bw| bw.write_u32(connectivity))
}

fn handle_get_status<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let options = decode_options(&mut reader)?;
    let _filtered = filter_options_map(&options, EMPTY_OPTION_KEYS)?;

    let app_info = AppInfo::host(&inv.sender);

    if !app_info.has_network() {
        return Err(PortalError::NotAllowed(String::from(
            "This call is not available inside the sandbox",
        )));
    }

    let reply = ctx.call_impl(
        NETWORK_MONITOR_IMPL_INTERFACE,
        "GetStatus",
        IMPL_TIMEOUT,
        |bw| {
            bw.write_str("")?; // app_id
            bw.write_str("")?; // parent_window
            bw.write_array("{sv}", |_| Ok(())) // empty options
        },
    )?;

    let mut reply_reader = reply.body_reader();
    let status = crate::xdp_utils::decode_options(&mut reply_reader)?;

    ctx.reply(inv, |bw| encode_options(bw, &status))
}

fn handle_can_reach<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let hostname = reader.read_str()?.to_string();
    let port = reader.read_u32()?;
    let options = decode_options(&mut reader)?;

    let _filtered = filter_options_map(&options, CAN_REACH_OPTIONS)?;

    let app_info = AppInfo::host(&inv.sender);

    if !app_info.has_network() {
        return Err(PortalError::NotAllowed(String::from(
            "This call is not available inside the sandbox",
        )));
    }

    let reply = ctx.call_impl(NETWORK_MONITOR_IMPL_INTERFACE, "CanReach", IMPL_TIMEOUT, |bw| {
        bw.write_str(&hostname)?;
        bw.write_u32(port)?;
        bw.write_str("")?;
        bw.write_str("")?;
        bw.write_array("{sv}", |_| Ok(()))
    })?;

    let mut reply_reader = reply.body_reader();
    let reachable = reply_reader.read_bool()?;

    ctx.reply(inv, |bw| bw.write_bool(reachable))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdp_utils::{OptionKey, OptionMap};

    #[test]
    fn test_filter_options_map_empty() {
        let options = OptionMap::new();
        let filtered = filter_options_map(&options, EMPTY_OPTION_KEYS).unwrap();
        assert!(filtered.is_empty());
    }
}