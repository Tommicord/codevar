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

//! org.freedesktop.host.portal.Registry portal implementation.
//!
//! This portal allows unsandboxed applications to register their
//! D-Bus connections and associate them with an application ID that
//! will be used in portal APIs. Only host applications can register.

use codevar_base::basic_xml::{XmlBuilder, XmlDocument};

use crate::xdp_app_info::AppInfo;
use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn, PortalInterface};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{OptionKey, OptionMap, decode_options, encode_options};
use alloc::format;
use alloc::string::{String, ToString};

/// Interface name for the Registry portal.
const REGISTRY_INTERFACE: &str = "org.freedesktop.host.portal.Registry";

/// Implementation backend interface name.
const REGISTRY_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.Registry";

/// Timeout for implementation backend calls (1 hour).
const IMPL_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(3600);

/// Supported option keys for Register method.
const REGISTER_OPTION_KEYS: &[OptionKey] = &[OptionKey::new("handle_token", "s")];

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

/// Registers the Registry portal interface.
pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let methods: &[(&'static str, PortalFn<T>)] = &[("Register", handle_register as PortalFn<T>)];

    let iface_xml = XmlBuilder::new("interface")
        .attr("name", "org.freedesktop.host.portal.Registry")
        .child("method")
        .attr("name", "Register")
        .child("arg")
        .attr("type", "s")
        .attr("name", "app_id")
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

    let iface = PortalInterface {
        name: REGISTRY_INTERFACE,
        version: 1,
        introspect_xml: iface_xml,
        methods,
    };

    ctx.register_interface(iface);
    Ok(())
}

fn handle_register<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();

    let app_id = reader
        .read_str()?
        .to_string();
    let options = decode_options(&mut reader)?;

    let filtered = filter_options_map(&options, REGISTER_OPTION_KEYS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_info = AppInfo::host(&inv.sender);

    ctx.call_impl(REGISTRY_IMPL_INTERFACE, "Register", IMPL_TIMEOUT, |bw| {
        bw.write_object_path(&handle.path)?;
        bw.write_str(app_info.id())?;
        bw.write_str(&app_id)?;
        encode_options(bw, &filtered)
    })?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdp_utils::{OptionMap, PortalValue};

    #[test]
    fn test_filter_options_map() {
        let mut options = OptionMap::new();
        options.insert(
            "handle_token".to_string(),
            PortalValue::Str("valid_token".to_string()),
        );
        options.insert("unknown_key".to_string(), PortalValue::U32(42));
        let filtered = filter_options_map(&options, REGISTER_OPTION_KEYS).unwrap();
        assert!(filtered.contains_key("handle_token"));
        assert!(!filtered.contains_key("unknown_key"));
    }

    #[test]
    fn test_invalid_token_type_rejected() {
        let mut options = OptionMap::new();
        options.insert("handle_token".to_string(), PortalValue::U32(42));

        let err = filter_options_map(&options, REGISTER_OPTION_KEYS).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }
}
