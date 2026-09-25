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

//! Account portal implementation
//!
//! Ported from `desktop-portal/account.c`. Provides the
//! `org.freedesktop.portal.Account` interface with `GetUserInformation`.

use alloc::string::ToString;
use codevar_base::basic_xml::{XmlBuilder, XmlDocument};

use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{OptionKey, OptionMap, PortalValue, filter_options};
use codevar_dbus::BodyWriter;

const ACCOUNT_INTERFACE: &str = "org.freedesktop.portal.Account";
const ACCOUNT_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.Account";
const ACCOUNT_VERSION: u32 = 1;

fn validate_reason(_key: &str, value: &PortalValue, _options: &OptionMap) -> Result<(), PortalError> {
    if let PortalValue::Str(reason) = value
        && reason.len() > 256
    {
        return Err(PortalError::InvalidArgument(
            "Not accepting overly long reasons".to_string(),
        ));
    }
    Ok(())
}

const USER_INFORMATION_OPTIONS: &[OptionKey] = &[OptionKey::with_validate("reason", "s", validate_reason)];

fn handle_get_user_information<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let _window = reader
        .read_str()?
        .to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, USER_INFORMATION_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle
        .app_info
        .id()
        .to_string();

    ctx.call_impl(
        ACCOUNT_IMPL_INTERFACE,
        "GetUserInformation",
        core::time::Duration::from_secs(25),
        |bw: &mut BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&_window)?;
            crate::xdp_utils::encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let methods: &[(&str, PortalFn<T>)] = &[("GetUserInformation", handle_get_user_information)];

    let iface_xml = XmlBuilder::new("interface")
        .attr("name", "org.freedesktop.portal.Account")
        .child("method")
        .attr("name", "GetUserInformation")
        .child("arg")
        .attr("type", "s")
        .attr("name", "window")
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
        .child("property")
        .attr("name", "version")
        .attr("type", "u")
        .attr("access", "read")
        .end()
        .build();

    let iface = crate::xdp_context::PortalInterface {
        name: ACCOUNT_INTERFACE,
        version: ACCOUNT_VERSION,
        introspect_xml: iface_xml,
        methods,
    };

    ctx.register_interface(iface);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdp_utils::{OptionMap, PortalValue};

    #[test]
    fn validates_reason_length() {
        let mut options = OptionMap::new();
        options.insert("reason".to_string(), PortalValue::Str("a".repeat(257)));
        let err = filter_options(&options, USER_INFORMATION_OPTIONS).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }

    #[test]
    fn accepts_valid_reason() {
        let mut options = OptionMap::new();
        options.insert("reason".to_string(), PortalValue::Str("valid reason".to_string()));
        let filtered = filter_options(&options, USER_INFORMATION_OPTIONS).unwrap();
        assert_eq!(
            filtered.get("reason"),
            Some(&PortalValue::Str("valid reason".to_string()))
        );
    }
}
