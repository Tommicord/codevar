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

//! Notification portal frontend.
//!
//! Implements `org.freedesktop.portal.Notification` for sending
//! and withdrawing notifications. Supports icons, sounds, buttons,
//! categories, and markup body.

use codevar_base::xml;

use alloc::format;
use alloc::string::ToString;

use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn, PortalInterface};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{OptionKey, OptionMap, PortalValue, encode_options, filter_options};

const NOTIFICATION_INTERFACE: &str = "org.freedesktop.portal.Notification";
const NOTIFICATION_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.Notification";
const NOTIFICATION_VERSION: u32 = 2;

const SUPPORTED_PRIORITIES: &[&str] = &["low", "normal", "high", "urgent"];

const SUPPORTED_BUTTON_PURPOSES: &[&str] = &[
    "system.custom-alert",
    "im.reply-with-text",
    "im.call-invitation",
    "im.dismiss",
    "im.accept",
    "im.decline",
];

const NOTIFICATION_OPTIONS: &[OptionKey] = &[
    OptionKey::new("title", "s"),
    OptionKey::new("body", "s"),
    OptionKey::new("icon", "s"),
    OptionKey::new("category", "s"),
    OptionKey::new("actions", "a(sv)"),
    OptionKey::new("persistent", "b"),
    OptionKey::new("urgency", "s"),
    OptionKey::new("priority", "s"),
    OptionKey::new("timeout", "i"),
    OptionKey::new("sound", "s"),
    OptionKey::new("image", "s"),
    OptionKey::new("transient", "b"),
    OptionKey::new("resident", "b"),
    OptionKey::new("urgency", "s"),
];

fn validate_priority(_key: &str, value: &PortalValue, _options: &OptionMap) -> Result<(), PortalError> {
    if let PortalValue::Str(s) = value {
        if !SUPPORTED_PRIORITIES.contains(&s.as_str()) {
            return Err(PortalError::InvalidArgument(format!(
                "Invalid priority '{}', must be one of {:?}",
                s, SUPPORTED_PRIORITIES
            )));
        }
    }
    Ok(())
}

fn handle_add_notification<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let id = reader.read_str()?.to_string();
    let notification = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&notification, NOTIFICATION_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;

    let app_id = handle.app_info.id().to_string();

    ctx.call_impl(
        NOTIFICATION_IMPL_INTERFACE,
        "AddNotification",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&id)?;
            encode_options(bw, &filtered)
        },
    )?;

    ctx.reply_empty(inv)
}

fn handle_remove_notification<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let id = reader.read_str()?.to_string();

    let app_info = crate::xdp_app_info::AppInfo::host(&inv.sender);

    ctx.call_impl(
        NOTIFICATION_IMPL_INTERFACE,
        "RemoveNotification",
        core::time::Duration::from_secs(25),
        |bw| {
            bw.write_str(app_info.id())?;
            bw.write_str(&id)
        },
    )?;

    ctx.reply_empty(inv)
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let iface_xml = xml!(interface, attrs: ["name" = "org.freedesktop.portal.Notification"], children: [
        (method, attrs: ["name" = "AddNotification"], children: [
            (annotation, attrs: ["name" = "org.gtk.GDBus.C.UnixFD", "value" = "true"]),
            (arg, attrs: ["type" = "s", "name" = "id", "direction" = "in"]),
            (arg, attrs: ["type" = "a{sv}", "name" = "notification", "direction" = "in"])
        ]),
        (method, attrs: ["name" = "RemoveNotification"], children: [
            (arg, attrs: ["type" = "s", "name" = "id", "direction" = "in"])
        ]),
        (property, attrs: ["name" = "SupportedOptions", "type" = "a{sv}", "access" = "read"], children: [
            (annotation, attrs: ["name" = "org.qtproject.QtDBus.QtTypeName", "value" = "QVariantMap"])
        ]),
        (signal, attrs: ["name" = "ActionInvoked"], children: [
            (arg, attrs: ["type" = "s", "name" = "id"]),
            (arg, attrs: ["type" = "s", "name" = "action"]),
            (arg, attrs: ["type" = "av", "name" = "parameter"])
        ]),
        (property, attrs: ["name" = "version", "type" = "u", "access" = "read"]),
    ]);

    let interface = PortalInterface {
        name: NOTIFICATION_INTERFACE,
        version: NOTIFICATION_VERSION,
        introspect_xml: iface_xml,
        methods: &[
            ("AddNotification", handle_add_notification as PortalFn<T>),
            ("RemoveNotification", handle_remove_notification as PortalFn<T>),
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
    fn filters_notification_options() {
        let mut options = OptionMap::new();
        options.insert("title".to_string(), PortalValue::Str("Test".to_string()));
        options.insert("body".to_string(), PortalValue::Str("Body".to_string()));
        options.insert("priority".to_string(), PortalValue::Str("normal".to_string()));
        options.insert("unknown".to_string(), PortalValue::Str("x".to_string()));

        let filtered = filter_options(&options, NOTIFICATION_OPTIONS).unwrap();
        assert_eq!(filtered.len(), 3);
        assert!(filtered.contains_key("title"));
        assert!(filtered.contains_key("body"));
        assert!(filtered.contains_key("priority"));
    }

    #[test]
    fn validates_priority() {
        let mut options = OptionMap::new();
        options.insert("priority".to_string(), PortalValue::Str("invalid".to_string()));

        let err = filter_options(&options, NOTIFICATION_OPTIONS).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }
}
