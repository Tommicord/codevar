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

//! Clipboard portal frontend.
//!
//! Implements `org.freedesktop.portal.Clipboard` for session-based
//! clipboard access. The portal does not create its own sessions;
//! instead it extends sessions from RemoteDesktop or InputCapture.

use alloc::string::{String, ToString};
use codevar_base::basic_xml::{XmlBuilder, XmlDocument};

use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn, PortalInterface};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_request::extract_handle_token;
use crate::xdp_utils::{OptionKey, OptionMap, PortalValue, encode_options, filter_options};

const CLIPBOARD_INTERFACE: &str = "org.freedesktop.portal.Clipboard";
const CLIPBOARD_IMPL_INTERFACE: &str = "org.freedesktop.impl.portal.Clipboard";
const CLIPBOARD_VERSION: u32 = 1;

const REQUEST_CLIPBOARD_OPTIONS: &[OptionKey] = &[OptionKey::new("handle_token", "s")];

const SET_SELECTION_OPTIONS: &[OptionKey] = &[OptionKey::new("mime_types", "as")];

fn handle_request_clipboard<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_handle = reader.read_object_path()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, REQUEST_CLIPBOARD_OPTIONS)?;

    let handle = ctx.begin_session(inv, &filtered)?;

    let app_id = handle.app_info.id().to_string();

    ctx.call_impl(
        CLIPBOARD_IMPL_INTERFACE,
        "RequestClipboard",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&app_id)?;
            crate::xdp_utils::encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_set_selection<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_handle = reader.read_object_path()?.to_string();
    let options = crate::xdp_utils::decode_options(&mut reader)?;

    let filtered = filter_options(&options, SET_SELECTION_OPTIONS)?;

    let handle = ctx.begin_session(inv, &filtered)?;

    ctx.call_impl(
        CLIPBOARD_IMPL_INTERFACE,
        "SetSelection",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            crate::xdp_utils::encode_options(bw, &filtered)
        },
    )?;

    ctx.reply(inv, |bw| bw.write_object_path(&handle.path))?;

    Ok(())
}

fn handle_selection_write<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_handle = reader.read_object_path()?.to_string();
    let serial = reader.read_u32()?;

    let handle = ctx
        .take_session(&session_handle)
        .ok_or_else(|| PortalError::NotFound(String::from("Session not found")))?;

    ctx.call_impl(
        CLIPBOARD_IMPL_INTERFACE,
        "SelectionWrite",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_u32(serial)
        },
    )?;
    let mut reply = ctx
        .conn
        .recv_timeout(core::time::Duration::from_secs(25))?;
    let fds = reply.take_fds();
    let mut results = OptionMap::new();

    if !fds.is_empty() {
        results.insert("fd".to_string(), PortalValue::Handle(0));
    } else {
        return Err(PortalError::Failed(String::from(
            "No file descriptor returned from backend",
        )));
    }

    ctx.reply(inv, |bw| encode_options(bw, &results))
}

fn handle_selection_write_done<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_handle = reader.read_object_path()?.to_string();
    let serial = reader.read_u32()?;
    let success = reader.read_bool()?;

    let handle = ctx
        .take_session(&session_handle)
        .ok_or_else(|| PortalError::NotFound(String::from("Session not found")))?;

    ctx.call_impl(
        CLIPBOARD_IMPL_INTERFACE,
        "SelectionWriteDone",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_u32(serial)?;
            bw.write_bool(success)
        },
    )?;

    ctx.reply_empty(inv)
}

fn handle_selection_read<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let session_handle = reader.read_object_path()?.to_string();
    let mime_type = reader.read_str()?.to_string();

    let handle = ctx
        .take_session(&session_handle)
        .ok_or_else(|| PortalError::NotFound(String::from("Session not found")))?;

    ctx.call_impl(
        CLIPBOARD_IMPL_INTERFACE,
        "SelectionRead",
        core::time::Duration::from_secs(25),
        |bw: &mut codevar_dbus::BodyWriter| {
            bw.write_object_path(&handle.path)?;
            bw.write_str(&mime_type)
        },
    )?;
    let mut reply = ctx
        .conn
        .recv_timeout(core::time::Duration::from_secs(25))?;
    let fds = reply.take_fds();
    let mut results = OptionMap::new();

    if !fds.is_empty() {
        results.insert("fd".to_string(), PortalValue::Handle(0));
    } else {
        return Err(PortalError::Failed(String::from(
            "No file descriptor returned from backend",
        )));
    }

    ctx.reply(inv, |bw| encode_options(bw, &results))
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let iface_xml = XmlBuilder::new("interface")
        .attr("name", "org.freedesktop.portal.Clipboard")
        .child("method")
        .attr("name", "RequestClipboard")
        .child("arg")
        .attr("type", "o")
        .attr("name", "session_handle")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .end()
        .child("method")
        .attr("name", "SetSelection")
        .child("arg")
        .attr("type", "o")
        .attr("name", "session_handle")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "in")
        .end()
        .end()
        .child("method")
        .attr("name", "SelectionWrite")
        .child("annotation")
        .attr("name", "org.gtk.GDBus.C.UnixFD")
        .attr("value", "true")
        .end()
        .child("arg")
        .attr("type", "o")
        .attr("name", "session_handle")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "u")
        .attr("name", "serial")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "h")
        .attr("name", "fd")
        .attr("direction", "out")
        .end()
        .end()
        .child("method")
        .attr("name", "SelectionWriteDone")
        .child("arg")
        .attr("type", "o")
        .attr("name", "session_handle")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "u")
        .attr("name", "serial")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "b")
        .attr("name", "success")
        .attr("direction", "in")
        .end()
        .end()
        .child("method")
        .attr("name", "SelectionRead")
        .child("annotation")
        .attr("name", "org.gtk.GDBus.C.UnixFD")
        .attr("value", "true")
        .end()
        .child("arg")
        .attr("type", "o")
        .attr("name", "session_handle")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "s")
        .attr("name", "mime_type")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "h")
        .attr("name", "fd")
        .attr("direction", "out")
        .end()
        .end()
        .child("signal")
        .attr("name", "SelectionOwnerChanged")
        .child("arg")
        .attr("type", "o")
        .attr("name", "session_handle")
        .attr("direction", "out")
        .end()
        .child("arg")
        .attr("type", "a{sv}")
        .attr("name", "options")
        .attr("direction", "out")
        .end()
        .end()
        .child("signal")
        .attr("name", "SelectionTransfer")
        .child("arg")
        .attr("type", "o")
        .attr("name", "session_handle")
        .attr("direction", "out")
        .end()
        .child("arg")
        .attr("type", "s")
        .attr("name", "mime_type")
        .attr("direction", "out")
        .end()
        .child("arg")
        .attr("type", "u")
        .attr("name", "serial")
        .attr("direction", "out")
        .end()
        .end()
        .child("property")
        .attr("name", "version")
        .attr("type", "u")
        .attr("access", "read")
        .end()
        .build();

    let interface = PortalInterface {
        name: CLIPBOARD_INTERFACE,
        version: CLIPBOARD_VERSION,
        introspect_xml: iface_xml,
        methods: &[
            ("RequestClipboard", handle_request_clipboard as PortalFn<T>),
            ("SetSelection", handle_set_selection as PortalFn<T>),
            ("SelectionWrite", handle_selection_write as PortalFn<T>),
            ("SelectionWriteDone", handle_selection_write_done as PortalFn<T>),
            ("SelectionRead", handle_selection_read as PortalFn<T>),
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
    fn filters_request_clipboard_options() {
        let mut options = OptionMap::new();
        options.insert("handle_token".to_string(), PortalValue::Str("abc123".to_string()));
        options.insert("unknown".to_string(), PortalValue::Str("x".to_string()));

        let filtered = filter_options(&options, REQUEST_CLIPBOARD_OPTIONS).unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("handle_token"));
    }

    #[test]
    fn filters_set_selection_options() {
        let mut options = OptionMap::new();
        options.insert(
            "mime_types".to_string(),
            PortalValue::string_array(vec!["text/plain".to_string()]),
        );
        options.insert("unknown".to_string(), PortalValue::Str("x".to_string()));

        let filtered = filter_options(&options, SET_SELECTION_OPTIONS).unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("mime_types"));
    }

    #[test]
    fn rejects_invalid_mime_types_type() {
        let mut options = OptionMap::new();
        options.insert(
            "mime_types".to_string(),
            PortalValue::Str("text/plain".to_string()),
        );

        let err = filter_options(&options, SET_SELECTION_OPTIONS).unwrap_err();
        assert!(matches!(err, PortalError::InvalidArgument(_)));
    }
}
