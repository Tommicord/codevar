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

//! Screenshot portal implementation
//!
//! Ported from `desktop-portal/screenshot.c`. Provides the
//! `org.freedesktop.portal.Screenshot` interface with `Screenshot` and `PickColor`.

use codevar_base::xml;

use alloc::format;
use alloc::string::ToString;

use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{OptionKey, OptionMap, PortalValue, filter_options};
use codevar_dbus::BodyWriter;

const SCREENSHOT_INTERFACE: &str = "org.freedesktop.portal.Screenshot";
const SCREENSHOT_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.Screenshot";
const SCREENSHOT_VERSION: u32 = 3;

const SCREENSHOT_TARGET_SCREEN: u32 = 1u32 << 0;
const SCREENSHOT_TARGET_WINDOW: u32 = 1u32 << 1;
const SCREENSHOT_TARGET_AREA: u32 = 1u32 << 2;
const SCREENSHOT_TARGET_ACTIVE_WINDOW: u32 = 1u32 << 3;
const SCREENSHOT_TARGET_MASK: u32 = SCREENSHOT_TARGET_SCREEN
    | SCREENSHOT_TARGET_WINDOW
    | SCREENSHOT_TARGET_AREA
    | SCREENSHOT_TARGET_ACTIVE_WINDOW;

fn validate_target(_key: &str, value: &PortalValue, _options: &OptionMap) -> Result<(), PortalError> {
    if let PortalValue::U32(target) = value
        && *target & !SCREENSHOT_TARGET_MASK != 0
    {
        return Err(PortalError::InvalidArgument(format!(
            "Invalid screenshot target: {target}"
        )));
    }
    Ok(())
}

const SCREENSHOT_OPTIONS_V3: &[OptionKey] = &[
    OptionKey::new("parent_window", "s"),
    OptionKey::with_validate("target", "u", validate_target),
    OptionKey::new("modal", "b"),
    OptionKey::new("handle_token", "s"),
    OptionKey::new("multiple", "b"),
    OptionKey::new("roundtrip", "b"),
];

fn handle_screenshot<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let parent_window = reader.read_str()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, SCREENSHOT_OPTIONS_V3)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle.app_info.id().to_string();

    ctx.call_impl(
        SCREENSHOT_IMPL_INTERFACE,
        "Screenshot",
        core::time::Duration::from_secs(25),
        |bw: &mut BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&parent_window)?;
            crate::xdp_utils::encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_pick_color<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let parent_window = reader.read_str()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, SCREENSHOT_OPTIONS_V3)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle.app_info.id().to_string();

    ctx.call_impl(
        SCREENSHOT_IMPL_INTERFACE,
        "PickColor",
        core::time::Duration::from_secs(25),
        |bw: &mut BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&parent_window)?;
            crate::xdp_utils::encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let methods: &[(&str, PortalFn<T>)] = &[
        ("Screenshot", handle_screenshot),
        ("PickColor", handle_pick_color),
    ];

    let iface_xml = xml!(interface, attrs: ["name" = "org.freedesktop.portal.Screenshot"], children: [
        (method, attrs: ["name" = "Screenshot"], children: [
            (arg, attrs: ["type" = "s", "name" = "parent_window", "direction" = "in"]),
            (arg, attrs: ["type" = "a{sv}", "name" = "options", "direction" = "in"]),
            (arg, attrs: ["type" = "o", "name" = "handle", "direction" = "out"])
        ]),
        (method, attrs: ["name" = "PickColor"], children: [
            (arg, attrs: ["type" = "s", "name" = "parent_window", "direction" = "in"]),
            (arg, attrs: ["type" = "a{sv}", "name" = "options", "direction" = "in"]),
            (arg, attrs: ["type" = "o", "name" = "handle", "direction" = "out"])
        ]),
        (property, attrs: ["name" = "AvailableTargets", "type" = "u", "access" = "read"]),
        (property, attrs: ["name" = "version", "type" = "u", "access" = "read"]),
    ]);

    let iface = crate::xdp_context::PortalInterface {
        name: SCREENSHOT_INTERFACE,
        version: SCREENSHOT_VERSION,
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
    fn validates_screenshot_targets() {
        let mut options = OptionMap::new();
        options.insert("target".to_string(), PortalValue::U32(999));
        let err = filter_options(&options, SCREENSHOT_OPTIONS_V3).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }
}
