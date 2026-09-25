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

//! org.freedesktop.portal.ProxyResolver portal implementation.
//!
//! This portal provides network proxy information to sandboxed
//! applications. It does not involve user interaction.

use codevar_base::basic_xml::{XmlBuilder, XmlDocument};

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::xdp_app_info::AppInfo;
use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn, PortalInterface};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{OptionKey, OptionMap, decode_options};

/// Interface name for the ProxyResolver portal.
const PROXY_RESOLVER_INTERFACE: &str = "org.freedesktop.portal.ProxyResolver";

/// Implementation backend interface name.
const PROXY_RESOLVER_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.ProxyResolver";

/// Timeout for implementation backend calls (1 hour).
const IMPL_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(3600);

/// Supported option keys (none for Lookup method).
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

/// Registers the ProxyResolver portal interface.
pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let methods: &[(&'static str, PortalFn<T>)] = &[("Lookup", handle_lookup as PortalFn<T>)];

    let iface_xml = XmlBuilder::new("interface")
        .attr("name", "org.freedesktop.portal.ProxyResolver")
        .child("method")
        .attr("name", "Lookup")
        .child("arg")
        .attr("type", "s")
        .attr("name", "uri")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "as")
        .attr("name", "proxies")
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
        name: PROXY_RESOLVER_INTERFACE,
        version: 1,
        introspect_xml: iface_xml,
        methods,
    };

    ctx.register_interface(iface);
    Ok(())
}

fn handle_lookup<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let uri = reader.read_str()?.to_string();
    let options = decode_options(&mut reader)?;
    let _filtered = filter_options_map(&options, EMPTY_OPTION_KEYS)?;

    let app_info = AppInfo::host(&inv.sender);

    if !app_info.has_network() {
        return Err(PortalError::NotAllowed(String::from(
            "This call is not available inside the sandbox",
        )));
    }

    let reply = ctx.call_impl(PROXY_RESOLVER_IMPL_INTERFACE, "Lookup", IMPL_TIMEOUT, |bw| {
        bw.write_str(&uri)?;
        bw.write_str("")?;
        bw.write_str("")?;
        bw.write_array("{sv}", |_| Ok(()))
    })?;

    let mut reply_reader = reply.body_reader();
    let mut proxies_array = reply_reader.read_array(1)?;
    let mut proxies = Vec::new();
    while !proxies_array.is_empty() {
        let proxy = proxies_array.read_str()?.to_string();
        proxies.push(proxy);
    }
    ctx.reply(inv, |bw| {
        bw.write_array("s", |inner| {
            for proxy in &proxies {
                inner.write_str(proxy)?;
            }
            Ok(())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdp_utils::OptionMap;

    #[test]
    fn test_filter_options_map_empty() {
        let options = OptionMap::new();
        let filtered = filter_options_map(&options, EMPTY_OPTION_KEYS).unwrap();
        assert!(filtered.is_empty());
    }
}
