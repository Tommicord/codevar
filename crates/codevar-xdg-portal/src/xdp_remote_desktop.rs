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

//! Remote desktop portal frontend.
//!
//! Implements `org.freedesktop.portal.RemoteDesktop` for creating
//! remote desktop sessions and relaying pointer, keyboard and touch
//! events to the `org.freedesktop.impl.portal.RemoteDesktop`
//! backend. Event notifications are forwarded verbatim; the request
//! methods (`CreateSession`, `SelectDevices`, `Start`) complete their
//! `org.freedesktop.portal.Request.Response` signal from the
//! backend's `(u, a{sv})` reply.

use alloc::string::{String, ToString};
use core::time::Duration;

use codevar_base::basic_xml::XmlDocument;
use codevar_base::xml;
use codevar_dbus::{BodyWriter, DbusMessage};

use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn, PortalInterface};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{
    OptionKey, OptionMap, PortalValue, decode_options, encode_options, filter_options,
};

const REMOTE_DESKTOP_INTERFACE: &str = "org.freedesktop.portal.RemoteDesktop";
const REMOTE_DESKTOP_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.RemoteDesktop";
const REMOTE_DESKTOP_VERSION: u32 = 2;
const CALL_TIMEOUT: Duration = Duration::from_secs(25);

const CREATE_SESSION_OPTIONS: &[OptionKey] = &[
    OptionKey::new("handle_token", "s"),
    OptionKey::new("session_handle_token", "s"),
];

const SELECT_DEVICES_OPTIONS: &[OptionKey] = &[
    OptionKey::new("handle_token", "s"),
    OptionKey::new("types", "u"),
    OptionKey::new("restore_token", "s"),
    OptionKey::new("persist_mode", "u"),
];

const START_OPTIONS: &[OptionKey] = &[OptionKey::new("handle_token", "s")];

/// Verifies that `path` names a session registered on this portal.
fn require_session<T: codevar_dbus::DbusTransport>(ctx: &PortalContext<T>, path: &str) -> XdpResult<()> {
    if ctx.has_session(path) {
        Ok(())
    } else {
        Err(PortalError::NotFound(String::from("Session not found")))
    }
}

/// Reads the `(u response, a{sv} results)` reply of an impl call.
fn read_impl_response(reply: &DbusMessage) -> XdpResult<(u32, OptionMap)> {
    let mut reader = reply.body_reader();
    let response = reader.read_u32()?;
    let results = decode_options(&mut reader)?;
    Ok((response, results))
}

fn handle_create_session<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let options = decode_options(&mut reader)?;
    let filtered = filter_options(&options, CREATE_SESSION_OPTIONS)?;

    let handle = ctx.begin_request(inv, &filtered)?;
    let session = ctx.begin_session(inv, &filtered)?;
    let app_id = handle.app_info.id().to_string();

    let mut impl_reply = ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "CreateSession",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_object_path(&session.path)?;
            bw.write_str(&app_id)?;
            encode_options(bw, &filtered)
        },
    )?;
    let (response, mut results) = read_impl_response(&impl_reply)?;
    impl_reply.take_fds();
    // The frontend documents `session_handle` as a string for
    // backwards compatibility, not as an object path.
    results.insert(
        String::from("session_handle"),
        PortalValue::Str(session.path.clone()),
    );

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;
    ctx.complete_request(&handle, response, &results)
}

fn handle_select_devices<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let filtered = filter_options(&options, SELECT_DEVICES_OPTIONS)?;
    require_session(ctx, &session_path)?;

    let handle = ctx.begin_request(inv, &filtered)?;
    let app_id = handle.app_info.id().to_string();

    let mut impl_reply = ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "SelectDevices",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_object_path(&session_path)?;
            bw.write_str(&app_id)?;
            encode_options(bw, &filtered)
        },
    )?;
    let (response, results) = read_impl_response(&impl_reply)?;
    impl_reply.take_fds();

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;
    ctx.complete_request(&handle, response, &results)
}

fn handle_start<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let parent_window = reader.read_str()?.to_string();
    let options = decode_options(&mut reader)?;
    let filtered = filter_options(&options, START_OPTIONS)?;
    require_session(ctx, &session_path)?;

    let handle = ctx.begin_request(inv, &filtered)?;
    let app_id = handle.app_info.id().to_string();

    let mut impl_reply = ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "Start",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_object_path(&session_path)?;
            bw.write_str(&app_id)?;
            bw.write_str(&parent_window)?;
            encode_options(bw, &filtered)
        },
    )?;
    let (response, results) = read_impl_response(&impl_reply)?;
    impl_reply.take_fds();

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;
    ctx.complete_request(&handle, response, &results)
}

fn handle_notify_pointer_motion<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let dx = reader.read_f64()?;
    let dy = reader.read_f64()?;
    require_session(ctx, &session_path)?;

    ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "NotifyPointerMotion",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            encode_options(bw, &options)?;
            bw.write_f64(dx)?;
            bw.write_f64(dy)
        },
    )?;
    ctx.reply_empty(inv)
}

fn handle_notify_pointer_motion_absolute<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let stream = reader.read_u32()?;
    let x = reader.read_f64()?;
    let y = reader.read_f64()?;
    require_session(ctx, &session_path)?;

    ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "NotifyPointerMotionAbsolute",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            encode_options(bw, &options)?;
            bw.write_u32(stream)?;
            bw.write_f64(x)?;
            bw.write_f64(y)
        },
    )?;
    ctx.reply_empty(inv)
}

fn handle_notify_pointer_button<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let button = reader.read_i32()?;
    let state = reader.read_u32()?;
    require_session(ctx, &session_path)?;

    ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "NotifyPointerButton",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            encode_options(bw, &options)?;
            bw.write_i32(button)?;
            bw.write_u32(state)
        },
    )?;
    ctx.reply_empty(inv)
}

fn handle_notify_pointer_axis<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let dx = reader.read_f64()?;
    let dy = reader.read_f64()?;
    require_session(ctx, &session_path)?;

    ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "NotifyPointerAxis",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            encode_options(bw, &options)?;
            bw.write_f64(dx)?;
            bw.write_f64(dy)
        },
    )?;
    ctx.reply_empty(inv)
}

fn handle_notify_pointer_axis_discrete<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let axis = reader.read_u32()?;
    let steps = reader.read_i32()?;
    require_session(ctx, &session_path)?;

    ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "NotifyPointerAxisDiscrete",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            encode_options(bw, &options)?;
            bw.write_u32(axis)?;
            bw.write_i32(steps)
        },
    )?;
    ctx.reply_empty(inv)
}

fn handle_notify_keyboard_keycode<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let keycode = reader.read_i32()?;
    let state = reader.read_u32()?;
    require_session(ctx, &session_path)?;

    ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "NotifyKeyboardKeycode",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            encode_options(bw, &options)?;
            bw.write_i32(keycode)?;
            bw.write_u32(state)
        },
    )?;
    ctx.reply_empty(inv)
}

fn handle_notify_keyboard_keysym<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let keysym = reader.read_i32()?;
    let state = reader.read_u32()?;
    require_session(ctx, &session_path)?;

    ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "NotifyKeyboardKeysym",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            encode_options(bw, &options)?;
            bw.write_i32(keysym)?;
            bw.write_u32(state)
        },
    )?;
    ctx.reply_empty(inv)
}

fn handle_notify_touch_down<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let stream = reader.read_u32()?;
    let slot = reader.read_u32()?;
    let x = reader.read_f64()?;
    let y = reader.read_f64()?;
    require_session(ctx, &session_path)?;

    ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "NotifyTouchDown",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            encode_options(bw, &options)?;
            bw.write_u32(stream)?;
            bw.write_u32(slot)?;
            bw.write_f64(x)?;
            bw.write_f64(y)
        },
    )?;
    ctx.reply_empty(inv)
}

fn handle_notify_touch_motion<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let stream = reader.read_u32()?;
    let slot = reader.read_u32()?;
    let x = reader.read_f64()?;
    let y = reader.read_f64()?;
    require_session(ctx, &session_path)?;

    ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "NotifyTouchMotion",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            encode_options(bw, &options)?;
            bw.write_u32(stream)?;
            bw.write_u32(slot)?;
            bw.write_f64(x)?;
            bw.write_f64(y)
        },
    )?;
    ctx.reply_empty(inv)
}

fn handle_notify_touch_up<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    let slot = reader.read_u32()?;
    require_session(ctx, &session_path)?;

    ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "NotifyTouchUp",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            encode_options(bw, &options)?;
            bw.write_u32(slot)
        },
    )?;
    ctx.reply_empty(inv)
}

fn handle_connect_to_eis<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_path = reader.read_object_path()?.to_string();
    let options = decode_options(&mut reader)?;
    require_session(ctx, &session_path)?;
    let app_id = crate::xdp_app_info::AppInfo::host(&inv.sender)
        .id()
        .to_string();

    let mut impl_reply = ctx.call_impl(
        REMOTE_DESKTOP_IMPL_INTERFACE,
        "ConnectToEIS",
        CALL_TIMEOUT,
        |bw: &mut BodyWriter| {
            bw.write_object_path(&session_path)?;
            bw.write_str(&app_id)?;
            encode_options(bw, &options)
        },
    )?;
    let fds = impl_reply.take_fds();
    if fds.is_empty() {
        return Err(PortalError::Failed(String::from(
            "EIS backend returned no file descriptor",
        )));
    }

    let mut reply = DbusMessage::method_return(inv.serial);
    reply.set_fds(fds);
    reply.build_body(|bw| bw.write_fd(0))?;
    ctx.conn.send_message(reply)?;
    Ok(())
}

/// Appends a `<method>` element with the given arguments.
fn add_method(builder: XmlBuilder, name: &str, args: &[(&str, &str, &str)]) -> XmlBuilder {
    let mut method = builder.child("method").attr("name", name);
    for (arg_name, type_signature, direction) in args {
        method = method
            .child("arg")
            .attr("name", arg_name)
            .attr("type", type_signature)
            .attr("direction", direction)
            .end();
    }
    method.end()
}

fn build_interface_xml() -> XmlDocument {
    let builder = XmlBuilder::new("interface").attr("name", REMOTE_DESKTOP_INTERFACE);
    let builder = add_method(
        builder,
        "CreateSession",
        &[("options", "a{sv}", "in"), ("handle", "o", "out")],
    );
    let builder = add_method(
        builder,
        "SelectDevices",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("handle", "o", "out"),
        ],
    );
    let builder = add_method(
        builder,
        "Start",
        &[
            ("session_handle", "o", "in"),
            ("parent_window", "s", "in"),
            ("options", "a{sv}", "in"),
            ("handle", "o", "out"),
        ],
    );
    let builder = add_method(
        builder,
        "NotifyPointerMotion",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("dx", "d", "in"),
            ("dy", "d", "in"),
        ],
    );
    let builder = add_method(
        builder,
        "NotifyPointerMotionAbsolute",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("stream", "u", "in"),
            ("x", "d", "in"),
            ("y", "d", "in"),
        ],
    );
    let builder = add_method(
        builder,
        "NotifyPointerButton",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("button", "i", "in"),
            ("state", "u", "in"),
        ],
    );
    let builder = add_method(
        builder,
        "NotifyPointerAxis",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("dx", "d", "in"),
            ("dy", "d", "in"),
        ],
    );
    let builder = add_method(
        builder,
        "NotifyPointerAxisDiscrete",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("axis", "u", "in"),
            ("steps", "i", "in"),
        ],
    );
    let builder = add_method(
        builder,
        "NotifyKeyboardKeycode",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("keycode", "i", "in"),
            ("state", "u", "in"),
        ],
    );
    let builder = add_method(
        builder,
        "NotifyKeyboardKeysym",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("keysym", "i", "in"),
            ("state", "u", "in"),
        ],
    );
    let builder = add_method(
        builder,
        "NotifyTouchDown",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("stream", "u", "in"),
            ("slot", "u", "in"),
            ("x", "d", "in"),
            ("y", "d", "in"),
        ],
    );
    let builder = add_method(
        builder,
        "NotifyTouchMotion",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("stream", "u", "in"),
            ("slot", "u", "in"),
            ("x", "d", "in"),
            ("y", "d", "in"),
        ],
    );
    let builder = add_method(
        builder,
        "NotifyTouchUp",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("slot", "u", "in"),
        ],
    );
    let builder = add_method(
        builder,
        "ConnectToEIS",
        &[
            ("session_handle", "o", "in"),
            ("options", "a{sv}", "in"),
            ("fd", "h", "out"),
        ],
    );
    let builder = builder
        .child("property")
        .attr("name", "AvailableDeviceTypes")
        .attr("type", "u")
        .attr("access", "read")
        .end()
        .child("property")
        .attr("name", "version")
        .attr("type", "u")
        .attr("access", "read")
        .end();
    builder.build()
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let methods: &[(&str, PortalFn<T>)] = &[
        ("CreateSession", handle_create_session),
        ("SelectDevices", handle_select_devices),
        ("Start", handle_start),
        ("NotifyPointerMotion", handle_notify_pointer_motion),
        ("NotifyPointerMotionAbsolute", handle_notify_pointer_motion_absolute),
        ("NotifyPointerButton", handle_notify_pointer_button),
        ("NotifyPointerAxis", handle_notify_pointer_axis),
        ("NotifyPointerAxisDiscrete", handle_notify_pointer_axis_discrete),
        ("NotifyKeyboardKeycode", handle_notify_keyboard_keycode),
        ("NotifyKeyboardKeysym", handle_notify_keyboard_keysym),
        ("NotifyTouchDown", handle_notify_touch_down),
        ("NotifyTouchMotion", handle_notify_touch_motion),
        ("NotifyTouchUp", handle_notify_touch_up),
        ("ConnectToEIS", handle_connect_to_eis),
    ];

    let iface = PortalInterface {
        name: REMOTE_DESKTOP_INTERFACE,
        version: REMOTE_DESKTOP_VERSION,
        introspect_xml: build_interface_xml(),
        methods,
    };
    ctx.register_interface(iface);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn introspection_xml_is_valid() {
        let xml = build_interface_xml();
        xml.validate().unwrap();
        let text = xml.to_string();
        assert!(text.contains("org.freedesktop.portal.RemoteDesktop"));
        assert!(text.contains("NotifyPointerMotionAbsolute"));
        assert!(text.contains("ConnectToEIS"));
        assert!(text.contains("AvailableDeviceTypes"));
    }

    #[test]
    fn create_session_keeps_tokens_and_drops_unknown() {
        let mut options = OptionMap::new();
        options.insert(String::from("handle_token"), PortalValue::Str(String::from("abc123")));
        options.insert(
            String::from("session_handle_token"),
            PortalValue::Str(String::from("sess1")),
        );
        options.insert(String::from("bogus"), PortalValue::Bool(true));

        let filtered = filter_options(&options, CREATE_SESSION_OPTIONS).unwrap();
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("handle_token"));
        assert!(filtered.contains_key("session_handle_token"));
    }

    #[test]
    fn select_devices_rejects_wrong_types() {
        let mut options = OptionMap::new();
        options.insert(String::from("types"), PortalValue::Str(String::from("3")));
        let err = filter_options(&options, SELECT_DEVICES_OPTIONS).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }

    #[test]
    fn select_devices_accepts_persist_mode() {
        let mut options = OptionMap::new();
        options.insert(String::from("persist_mode"), PortalValue::U32(2));
        options.insert(String::from("restore_token"), PortalValue::Str(String::from("tok")));
        let filtered = filter_options(&options, SELECT_DEVICES_OPTIONS).unwrap();
        assert_eq!(filtered.len(), 2);
    }
}
