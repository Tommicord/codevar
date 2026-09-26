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

//! Wallpaper portal implementation
//!
//! Ported from `desktop-portal/wallpaper.c`. Provides the
//! `org.freedesktop.portal.Wallpaper` interface with `SetWallpaperURI` and `SetWallpaperFile`.

use codevar_base::xml;

use alloc::format;
use alloc::string::ToString;

use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_permissions::{Permission, get_permission, set_permission};
use crate::xdp_utils::{OptionKey, OptionMap, PortalValue, filter_options};
use codevar_dbus::BodyWriter;

const WALLPAPER_INTERFACE: &str = "org.freedesktop.portal.Wallpaper";
const WALLPAPER_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.Wallpaper";
const WALLPAPER_ACCESS_INTERFACE: &str = "org.freedesktop.impl.portal.Access";
const WALLPAPER_VERSION: u32 = 1;
const DESKTOP_PATH: &str = "/org/freedesktop/portal/desktop";

const WALLPAPER_PERMISSION_TABLE: &str = "wallpaper";
const WALLPAPER_PERMISSION_ID: &str = "wallpaper";

fn validate_set_on(_key: &str, value: &PortalValue, _options: &OptionMap) -> Result<(), PortalError> {
    if let PortalValue::Str(s) = value
        && s != "both"
        && s != "background"
        && s != "lockscreen"
    {
        return Err(PortalError::InvalidArgument(format!(
            "Invalid set-on value '{}', must be 'both', 'background', or 'lockscreen'",
            s
        )));
    }
    Ok(())
}

const WALLPAPER_OPTIONS: &[OptionKey] = &[
    OptionKey::new("parent_window", "s"),
    OptionKey::with_validate("set-on", "s", validate_set_on),
    OptionKey::new("handle_token", "s"),
];

fn handle_set_wallpaper_uri<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let parent_window = reader.read_str()?.to_string();
    let uri = reader.read_str()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, WALLPAPER_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle.app_info.id().to_string();

    ctx.call_impl(
        WALLPAPER_IMPL_INTERFACE,
        "SetWallpaperURI",
        core::time::Duration::from_secs(25),
        |bw: &mut BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&parent_window)?;
            bw.write_str(&uri)?;
            crate::xdp_utils::encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_set_wallpaper_file<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let parent_window = reader.read_str()?.to_string();
    let fd = reader.read_fd()?;
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, WALLPAPER_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle.app_info.id().to_string();

    ctx.call_impl(
        WALLPAPER_IMPL_INTERFACE,
        "SetWallpaperFile",
        core::time::Duration::from_secs(25),
        |bw: &mut BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&parent_window)?;
            bw.write_fd(fd)?;
            crate::xdp_utils::encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let methods: &[(&str, PortalFn<T>)] = &[
        ("SetWallpaperURI", handle_set_wallpaper_uri),
        ("SetWallpaperFile", handle_set_wallpaper_file),
    ];

    let iface_xml = xml!(interface, attrs: ["name" = "org.freedesktop.portal.Wallpaper"], children: [
        (method, attrs: ["name" = "SetWallpaperURI"], children: [
            (arg, attrs: ["type" = "s", "name" = "parent_window", "direction" = "in"]),
            (arg, attrs: ["type" = "s", "name" = "uri", "direction" = "in"]),
            (arg, attrs: ["type" = "a{sv}", "name" = "options", "direction" = "in"]),
            (arg, attrs: ["type" = "o", "name" = "handle", "direction" = "out"])
        ]),
        (method, attrs: ["name" = "SetWallpaperFile"], children: [
            (annotation, attrs: ["name" = "org.gtk.GDBus.C.UnixFD", "value" = "true"]),
            (arg, attrs: ["type" = "s", "name" = "parent_window", "direction" = "in"]),
            (arg, attrs: ["type" = "h", "name" = "fd", "direction" = "in"]),
            (arg, attrs: ["type" = "a{sv}", "name" = "options", "direction" = "in"]),
            (arg, attrs: ["type" = "o", "name" = "handle", "direction" = "out"])
        ]),
        (property, attrs: ["name" = "version", "type" = "u", "access" = "read"]),
    ]);

    let iface = crate::xdp_context::PortalInterface {
        name: WALLPAPER_INTERFACE,
        version: WALLPAPER_VERSION,
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
    fn validates_set_on_values() {
        let mut options = OptionMap::new();
        options.insert("set-on".to_string(), PortalValue::Str("invalid".to_string()));
        let err = filter_options(&options, WALLPAPER_OPTIONS).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }

    #[test]
    fn accepts_valid_set_on() {
        for val in ["both", "background", "lockscreen"] {
            let mut options = OptionMap::new();
            options.insert("set-on".to_string(), PortalValue::Str(val.to_string()));
            let filtered = filter_options(&options, WALLPAPER_OPTIONS).unwrap();
            assert_eq!(filtered.get("set-on"), Some(&PortalValue::Str(val.to_string())));
        }
    }
}
