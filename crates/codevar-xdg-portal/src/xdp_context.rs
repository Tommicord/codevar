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

//! Main dispatcher and run loop for the portal frontend.
//!
//! This module provides the core event loop that receives D-Bus messages,
//! dispatches them to registered portal interfaces, and manages the
//! lifecycle of request and session handles.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use codevar_base::basic_xml::{XmlBuilder, XmlDocument};
use codevar_dbus::{
    BodyWriter, Connection, DbusError, DbusMessage, DbusReader, DbusResult, DbusTransport, MessageKind,
};

use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_portal_config::PortalConfig;
use crate::xdp_request::{REQUEST_BASE_PATH, RequestHandle, build_request_path, extract_handle_token};
use crate::xdp_session::{
    SESSION_BASE_PATH, SessionHandle, build_session_path, close_reason, extract_session_token,
};
use crate::xdp_utils::{OptionMap, encode_options};

/// Desktop portal well-known object path.
const DESKTOP_PATH: &str = "/org/freedesktop/portal/desktop";

/// Introspectable interface name.
const INTROSPECTABLE_INTERFACE: &str = "org.freedesktop.DBus.Introspectable";

/// Properties interface name.
const PROPERTIES_INTERFACE: &str = "org.freedesktop.DBus.Properties";

/// Desktop interface version.
const DESKTOP_VERSION: u32 = 0;

/// A method invocation received from a client.
#[derive(Debug, Clone)]
pub struct MethodInvocation {
    /// The message serial.
    pub serial: u32,
    /// The caller's unique bus name.
    pub sender: String,
    /// The object path the method was called on.
    pub path: String,
    /// The interface the method belongs to.
    pub interface: String,
    /// The method name.
    pub member: String,
    /// The body signature.
    pub body_signature: String,
    /// The raw body bytes.
    pub body: Vec<u8>,
    /// File descriptors attached to the message.
    pub fds: Vec<i32>,
}

impl MethodInvocation {
    /// Creates a `MethodInvocation` from a `DbusMessage`.
    pub fn from_message(message: DbusMessage) -> XdpResult<Self> {
        if message.kind() != MessageKind::MethodCall {
            return Err(PortalError::InvalidArgument(String::from("not a method call")));
        }
        Ok(Self {
            serial: message.serial(),
            sender: message.sender().unwrap_or_default().to_string(),
            path: message.path().unwrap_or_default().to_string(),
            interface: message.interface().unwrap_or_default().to_string(),
            member: message.member().unwrap_or_default().to_string(),
            body_signature: message.signature().to_string(),
            body: message.body().to_vec(),
            fds: message.fds().to_vec(),
        })
    }

    /// Returns a reader over the method body.
    #[must_use]
    pub fn body_reader(&self) -> DbusReader<'_> {
        DbusReader::new(&self.body, codevar_dbus::ByteOrder::Little)
    }
}

/// Function type for a portal method handler.
///
/// The handler receives a mutable reference to the portal context and
/// the method invocation. It must return an `XdpResult` indicating
/// success or a portal error to send back to the caller.
pub type PortalFn<T> = fn(&mut PortalContext<T>, &MethodInvocation) -> XdpResult<()>;

/// Description of a portal interface and its methods.
///
/// Each interface provides its D-Bus name, version, introspection XML,
/// and a slice of method name / handler pairs.
#[derive(Debug, Clone)]
pub struct PortalInterface<T: DbusTransport + 'static> {
    /// The D-Bus interface name (e.g. `org.freedesktop.portal.FileChooser`).
    pub name: &'static str,
    /// The interface version.
    pub version: u32,
    /// Introspection XML for this interface.
    pub introspect_xml: codevar_base::basic_xml::XmlDocument,
    /// Method name and handler pairs.
    pub methods: &'static [(&'static str, PortalFn<T>)],
}

/// Main portal context holding the connection, configuration, and state.
///
/// The context manages request and session handles, registered portal
/// interfaces, and runs the main event loop.
pub struct PortalContext<T: DbusTransport + 'static> {
    pub(crate) conn: Connection<T>,
    config: PortalConfig,
    verbose: bool,
    requests: BTreeMap<String, RequestHandle>,
    sessions: BTreeMap<String, SessionHandle>,
    interfaces: Vec<PortalInterface<T>>,
    quit_status: Option<i32>,
}

impl<T: DbusTransport + 'static> PortalContext<T> {
    /// Creates a new portal context.
    ///
    /// The connection must already be authenticated and have called
    /// `Hello` to obtain its unique name.
    pub fn new(conn: Connection<T>, verbose: bool) -> XdpResult<Self> {
        if conn.unique_name().is_none() {
            return Err(PortalError::InvalidArgument(String::from(
                "connection must have a unique name (call hello())",
            )));
        }
        let config = PortalConfig::load();
        Ok(Self {
            conn,
            config,
            verbose,
            requests: BTreeMap::new(),
            sessions: BTreeMap::new(),
            interfaces: Vec::new(),
            quit_status: None,
        })
    }

    /// Registers a portal interface.
    pub fn register_interface(&mut self, iface: PortalInterface<T>) {
        self.interfaces.push(iface);
    }

    /// Sends a successful method reply with the given body builder.
    pub fn reply<F>(&mut self, inv: &MethodInvocation, body: F) -> XdpResult<()>
    where
        F: FnOnce(&mut BodyWriter) -> DbusResult<()>,
    {
        let mut reply = DbusMessage::method_return(inv.serial);
        reply.build_body(body)?;
        self.conn.send_message(reply)?;
        Ok(())
    }

    /// Sends an empty successful method reply.
    pub fn reply_empty(&mut self, inv: &MethodInvocation) -> XdpResult<()> {
        let reply = DbusMessage::method_return(inv.serial);
        self.conn.send_message(reply)?;
        Ok(())
    }

    /// Sends an error reply with the given error name and message.
    pub fn reply_err(&mut self, inv: &MethodInvocation, name: &str, message: &str) -> XdpResult<()> {
        let mut reply = DbusMessage::error(inv.serial, name)?;
        reply.build_body(|bw| bw.write_str(message))?;
        self.conn.send_message(reply)?;
        Ok(())
    }

    /// Sends an error reply from a `PortalError`.
    pub fn reply_portal_err(&mut self, inv: &MethodInvocation, error: &PortalError) -> XdpResult<()> {
        let dbus_error = error.clone().into_dbus_error();
        let (name, message) = match dbus_error {
            DbusError::Remote { name, message } => (name, message),
            _ => (
                String::from("org.freedesktop.portal.Error.Failed"),
                dbus_error.to_string(),
            ),
        };
        let mut reply = DbusMessage::error(inv.serial, &name)?;
        reply.build_body(|bw| bw.write_str(&message))?;
        self.conn.send_message(reply)?;
        Ok(())
    }

    /// Emits a signal on the given path, interface, and member.
    pub fn emit_signal<F>(&mut self, path: &str, interface: &str, member: &str, body: F) -> XdpResult<()>
    where
        F: FnOnce(&mut BodyWriter) -> DbusResult<()>,
    {
        let mut signal = DbusMessage::signal(path, interface, member)?;
        signal.build_body(body)?;
        self.conn.send_message(signal)?;
        Ok(())
    }

    /// Calls a method on an implementation backend and returns the reply.
    pub fn call_impl<F>(
        &mut self,
        impl_iface: &str,
        member: &str,
        timeout: Duration,
        body: F,
    ) -> XdpResult<DbusMessage>
    where
        F: FnOnce(&mut BodyWriter) -> DbusResult<()>,
    {
        let Some(impl_config) = self.config.find(impl_iface) else {
            return Err(PortalError::NotFound(format!(
                "no implementation for {}",
                impl_iface
            )));
        };
        let reply = self.conn.call(
            &impl_config.dbus_name,
            DESKTOP_PATH,
            impl_iface,
            member,
            body,
            timeout,
        )?;
        Ok(reply)
    }

    /// Creates a new request handle from a method invocation.
    ///
    /// Extracts or generates the handle token, validates it, builds the
    /// request path, and inserts the handle into the request table.
    pub fn begin_request(&mut self, inv: &MethodInvocation, options: &OptionMap) -> XdpResult<RequestHandle> {
        let token = extract_handle_token(options)?;
        let path = build_request_path(&inv.sender, &token, &self.requests)?;
        let app_info = crate::xdp_app_info::AppInfo::host(&inv.sender);
        let created_ms = self.conn.transport().now_ms();
        let handle = RequestHandle::new(&inv.sender, token, app_info, created_ms);
        self.requests.insert(path.clone(), handle.clone());
        Ok(handle)
    }

    /// Completes a request by sending the `Response` signal and removing it.
    ///
    /// `response` is the portal response code (0 = success, 1 = error,
    /// 2 = cancelled). `results` is the `a{sv}` dictionary of result values.
    pub fn complete_request(
        &mut self,
        handle: &RequestHandle,
        response: u32,
        results: &OptionMap,
    ) -> XdpResult<()> {
        if handle.is_closed {
            return Ok(());
        }
        self.emit_signal(&handle.path, "org.freedesktop.portal.Request", "Response", |bw| {
            bw.write_u32(response)?;
            encode_options(bw, results)
        })?;
        self.requests.remove(&handle.path);
        Ok(())
    }

    /// Creates a new session handle from a method invocation.
    ///
    /// Extracts or generates the session handle token, validates it,
    /// builds the session path, and inserts the handle into the session table.
    pub fn begin_session(&mut self, inv: &MethodInvocation, options: &OptionMap) -> XdpResult<SessionHandle> {
        let token = extract_session_token(options)?;
        let path = build_session_path(&inv.sender, &token, &self.sessions)?;
        let app_info = crate::xdp_app_info::AppInfo::host(&inv.sender);
        let created_ms = self.conn.transport().now_ms();
        let handle = SessionHandle::new(&inv.sender, token, app_info, created_ms);
        self.sessions.insert(path.clone(), handle.clone());
        Ok(handle)
    }

    /// Closes a session by emitting the `Closed` signal and removing it.
    pub fn close_session(&mut self, handle: &SessionHandle, reason: u32) -> XdpResult<()> {
        if handle.is_closed {
            return Ok(());
        }
        self.emit_signal(&handle.path, "org.freedesktop.portal.Session", "Closed", |bw| {
            bw.write_u32(reason)
        })?;
        self.sessions.remove(&handle.path);
        Ok(())
    }

    /// Runs the main event loop.
    ///
    /// Receives messages with a 100ms timeout, flushes pending writes,
    /// and dispatches method calls. Returns the quit status when
    /// `quit()` is called.
    pub fn run(&mut self) -> XdpResult<i32> {
        // Register match rules for the interfaces we serve
        self.register_match_rules()?;

        loop {
            if let Some(status) = self.quit_status.take() {
                return Ok(status);
            }

            match self.conn.recv_timeout(Duration::from_millis(100)) {
                Ok(message) => {
                    if let Err(e) = self.handle_message(message) {
                        log::error!("Error handling message: {}", e);
                    }
                }
                Err(DbusError::Timeout) => {
                    // Normal timeout, continue loop
                }
                Err(DbusError::Disconnected) => {
                    return Err(PortalError::Failed(String::from("disconnected from bus")));
                }
                Err(e) => {
                    log::error!("Error receiving message: {}", e);
                }
            }

            if let Err(e) = self.conn.flush() {
                log::error!("Error flushing connection: {}", e);
            }
        }
    }

    /// Requests the run loop to exit with the given status.
    pub fn quit(&mut self, status: i32) {
        self.quit_status = Some(status);
    }

    /// Returns whether a path is a request path.
    pub fn is_request_path(path: &str) -> bool {
        path.starts_with(REQUEST_BASE_PATH)
    }

    /// Returns whether a path is a session path.
    pub fn is_session_path(path: &str) -> bool {
        path.starts_with(SESSION_BASE_PATH)
    }

    /// Takes and removes a request handle by path.
    pub fn take_request(&mut self, path: &str) -> Option<RequestHandle> {
        self.requests.remove(path)
    }

    /// Takes and removes a session handle by path.
    pub fn take_session(&mut self, path: &str) -> Option<SessionHandle> {
        self.sessions.remove(path)
    }

    /// Registers match rules for all registered interfaces.
    fn register_match_rules(&mut self) -> XdpResult<()> {
        // Match rules for method calls on the desktop path and request/session paths
        let rules = [
            format!("type='method_call',path='{}'", DESKTOP_PATH),
            format!("type='method_call',path_namespace='{}'", REQUEST_BASE_PATH),
            format!("type='method_call',path_namespace='{}'", SESSION_BASE_PATH),
        ];
        for rule in rules {
            self.conn.add_match(&rule, Duration::from_secs(5))?;
        }
        Ok(())
    }

    /// Handles a received D-Bus message.
    fn handle_message(&mut self, message: DbusMessage) -> XdpResult<()> {
        if message.kind() != MessageKind::MethodCall {
            // Ignore signals, method returns, and errors (connection handles replies)
            return Ok(());
        }

        let inv = MethodInvocation::from_message(message)?;

        if self.verbose {
            log::debug!(
                "Received method call: {}.{} on {} from {}",
                inv.interface,
                inv.member,
                inv.path,
                inv.sender
            );
        }

        // Properties.Get/GetAll on the desktop path
        if inv.path == DESKTOP_PATH && inv.interface == PROPERTIES_INTERFACE {
            return self.handle_properties(inv);
        }

        // Introspectable.Introspect on the desktop path
        if inv.path == DESKTOP_PATH && inv.interface == INTROSPECTABLE_INTERFACE {
            return self.handle_introspect(inv);
        }

        // Request.Close on a request path
        if Self::is_request_path(&inv.path)
            && inv.interface == "org.freedesktop.portal.Request"
            && inv.member == "Close"
        {
            return self.handle_request_close(inv);
        }

        // Session.Close on a session path
        if Self::is_session_path(&inv.path)
            && inv.interface == "org.freedesktop.portal.Session"
            && inv.member == "Close"
        {
            return self.handle_session_close(inv);
        }

        // Portal method on a registered interface
        if let Some(handler) = self.find_method_handler(&inv.interface, &inv.member) {
            return handler(self, &inv);
        }

        // Unknown method
        self.reply_err(
            &inv,
            "org.freedesktop.DBus.Error.UnknownMethod",
            &format!("Method {} not found on interface {}", inv.member, inv.interface),
        )
    }

    /// Handles Properties.Get and Properties.GetAll.
    fn handle_properties(&mut self, inv: MethodInvocation) -> XdpResult<()> {
        let mut reader = inv.body_reader();
        if inv.member == "Get" {
            let _interface = reader.read_str()?; // interface name
            let property = reader.read_str()?;
            if property == "version" {
                self.reply(&inv, |bw| {
                    bw.write_variant("u", |bw| bw.write_u32(DESKTOP_VERSION))
                })?;
            } else {
                self.reply_err(
                    &inv,
                    "org.freedesktop.DBus.Error.UnknownProperty",
                    &format!("Property {} not found", property),
                )?;
            }
        } else if inv.member == "GetAll" {
            let _interface = reader.read_str()?; // interface name
            self.reply(&inv, |bw| {
                bw.write_array("{sv}", |bw| {
                    bw.write_struct("sv", |bw| {
                        bw.write_str("version")?;
                        bw.write_variant("u", |bw| bw.write_u32(DESKTOP_VERSION))
                    })
                })
            })?;
        } else {
            self.reply_err(
                &inv,
                "org.freedesktop.DBus.Error.UnknownMethod",
                &format!("Method {} not found on interface {}", inv.member, inv.interface),
            )?;
        }
        Ok(())
    }

    /// Handles Introspectable.Introspect.
    fn handle_introspect(&mut self, inv: MethodInvocation) -> XdpResult<()> {
        let iface_xmls: String = self
            .interfaces
            .iter()
            .map(|iface| iface.introspect_xml.to_string())
            .collect();
        let xml = XmlBuilder::new("node")
            .child("interface")
                .attr("name", "org.freedesktop.DBus.Introspectable")
                .child("method")
                    .attr("name", "Introspect")
                    .child("arg")
                        .attr("name", "data")
                        .attr("type", "s")
                        .attr("direction", "out")
                    .end()
                .end()
            .end()
            .child("interface")
                .attr("name", "org.freedesktop.DBus.Properties")
                .child("method")
                    .attr("name", "Get")
                    .child("arg")
                        .attr("name", "interface")
                        .attr("type", "s")
                        .attr("direction", "in")
                    .end()
                    .child("arg")
                        .attr("name", "property")
                        .attr("type", "s")
                        .attr("direction", "in")
                    .end()
                    .child("arg")
                        .attr("name", "value")
                        .attr("type", "v")
                        .attr("direction", "out")
                    .end()
                .end()
                .child("method")
                    .attr("name", "GetAll")
                    .child("arg")
                        .attr("name", "interface")
                        .attr("type", "s")
                        .attr("direction", "in")
                    .end()
                    .child("arg")
                        .attr("name", "props")
                        .attr("type", "a{sv}")
                        .attr("direction", "out")
                    .end()
                .end()
            .end()
            .push(&iface_xmls)
            .child("interface")
                .attr("name", "org.freedesktop.portal.Request")
                .child("method")
                    .attr("name", "Close")
                .end()
                .child("signal")
                    .attr("name", "Response")
                    .child("arg")
                        .attr("name", "response")
                        .attr("type", "u")
                    .end()
                    .child("arg")
                        .attr("name", "results")
                        .attr("type", "a{sv}")
                    .end()
                .end()
            .end()
            .child("interface")
                .attr("name", "org.freedesktop.portal.Session")
                .child("method")
                    .attr("name", "Close")
                    .child("arg")
                        .attr("name", "reason")
                        .attr("type", "u")
                        .attr("direction", "in")
                    .end()
                .end()
                .child("signal")
                    .attr("name", "Closed")
                    .child("arg")
                        .attr("name", "reason")
                        .attr("type", "u")
                    .end()
                .end()
            .end()
            .build();
        self.reply(&inv, |bw| bw.write_str(xml.to_string()))
    }

    /// Handles Request.Close.
    fn handle_request_close(&mut self, inv: MethodInvocation) -> XdpResult<()> {
        if let Some(handle) = self.take_request(&inv.path) {
            // Complete with CANCELLED (2) as per spec
            self.complete_request(&handle, close_reason::CANCELLED, &OptionMap::new())?;
        } else {
            self.reply_err(&inv, "org.freedesktop.portal.Error.NotFound", "Request not found")?;
        }
        Ok(())
    }

    /// Handles Session.Close.
    fn handle_session_close(&mut self, inv: MethodInvocation) -> XdpResult<()> {
        let mut reader = inv.body_reader();
        let reason = if !reader.is_empty() {
            reader.read_u32()?
        } else {
            close_reason::NORMAL
        };

        if let Some(handle) = self.take_session(&inv.path) {
            self.close_session(&handle, reason)?;
        } else {
            self.reply_err(&inv, "org.freedesktop.portal.Error.NotFound", "Session not found")?;
        }
        Ok(())
    }

    /// Finds a method handler for the given interface and member.
    fn find_method_handler(&self, interface: &str, member: &str) -> Option<PortalFn<T>> {
        for iface in &self.interfaces {
            if iface.name == interface {
                for (method_name, handler) in iface.methods {
                    if *method_name == member {
                        return Some(*handler);
                    }
                }
            }
        }
        None
    }
}

/// Convenience function to create a context, register interfaces, and run.
pub fn run_with<T: DbusTransport + 'static, F>(
    conn: Connection<T>,
    verbose: bool,
    register: F,
) -> XdpResult<i32>
where
    F: FnOnce(&mut PortalContext<T>),
{
    let mut ctx = PortalContext::new(conn, verbose)?;
    register(&mut ctx);
    ctx.run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_dbus::DbusMessage;

    fn make_test_message(destination: &str, path: &str, interface: &str, member: &str) -> DbusMessage {
        let mut msg = DbusMessage::method_call(destination, path, interface, member).unwrap();
        msg.set_serial(1).unwrap();
        msg
    }

    #[test]
    fn request_path_helpers() {
        let requests = BTreeMap::new();
        assert!(!PortalContext::<MockTransport>::is_request_path(
            "/org/freedesktop/portal/desktop"
        ));
        assert!(PortalContext::<MockTransport>::is_request_path(
            "/org/freedesktop/portal/desktop/request/_1_42/abc"
        ));
        assert!(build_request_path(":1.42", "abc123", &requests).is_ok());
    }

    #[test]
    fn session_path_helpers() {
        let sessions = BTreeMap::new();
        assert!(!PortalContext::<MockTransport>::is_session_path(
            "/org/freedesktop/portal/desktop"
        ));
        assert!(PortalContext::<MockTransport>::is_session_path(
            "/org/freedesktop/portal/desktop/session/_1_42/abc"
        ));
        assert!(build_session_path(":1.42", "abc123", &sessions).is_ok());
    }

    #[test]
    fn method_invocation_from_message() {
        let msg = make_test_message(
            "org.freedesktop.DBus",
            DESKTOP_PATH,
            "org.freedesktop.portal.FileChooser",
            "OpenFile",
        );
        let inv = MethodInvocation::from_message(msg).unwrap();
        assert_eq!(inv.path, DESKTOP_PATH);
        assert_eq!(inv.interface, "org.freedesktop.portal.FileChooser");
        assert_eq!(inv.member, "OpenFile");
    }

    #[test]
    fn context_new_requires_unique_name() {
        let transport = MockTransport::new();
        let conn = Connection::new(transport);
        // Without hello(), unique_name is None
        assert!(PortalContext::new(conn, false).is_err());
    }

    struct MockTransport {
        rx: alloc::collections::VecDeque<Vec<u8>>,
        rx_fds: Vec<i32>,
        tx: Vec<(Vec<u8>, Vec<i32>)>,
        readable: bool,
        writable: bool,
        now_ms: u64,
    }

    impl MockTransport {
        fn new() -> Self {
            Self {
                rx: alloc::collections::VecDeque::new(),
                rx_fds: Vec::new(),
                tx: Vec::new(),
                readable: true,
                writable: true,
                now_ms: 0,
            }
        }
    }

    impl DbusTransport for MockTransport {
        fn read(&mut self, buf: &mut [u8]) -> DbusResult<usize> {
            if let Some(data) = self.rx.front() {
                let n = data.len().min(buf.len());
                buf[..n].copy_from_slice(&data[..n]);
                if n == data.len() {
                    self.rx.pop_front();
                } else {
                    let remaining = data[n..].to_vec();
                    self.rx.pop_front();
                    self.rx.push_back(remaining);
                }
                Ok(n)
            } else {
                Err(DbusError::WouldBlock)
            }
        }
        fn write(&mut self, buf: &[u8]) -> DbusResult<usize> {
            if self.writable {
                self.tx.push((buf.to_vec(), Vec::new()));
                Ok(buf.len())
            } else {
                Err(DbusError::WouldBlock)
            }
        }
        fn wait(
            &mut self,
            _timeout: Option<Duration>,
            interest: codevar_dbus::DbusPollEvents,
        ) -> DbusResult<codevar_dbus::DbusPollEvents> {
            let mut out = codevar_dbus::DbusPollEvents::EMPTY;
            if interest.contains(codevar_dbus::DbusPollEvents::READABLE) && self.readable {
                out |= codevar_dbus::DbusPollEvents::READABLE;
            }
            if interest.contains(codevar_dbus::DbusPollEvents::WRITABLE) && self.writable {
                out |= codevar_dbus::DbusPollEvents::WRITABLE;
            }
            Ok(out)
        }
        fn now_ms(&self) -> u64 {
            self.now_ms
        }
        fn take_fds(&mut self) -> Vec<i32> {
            core::mem::take(&mut self.rx_fds)
        }
        fn write_with_fds(&mut self, buf: &[u8], fds: &[i32]) -> DbusResult<usize> {
            if self.writable {
                self.tx.push((buf.to_vec(), fds.to_vec()));
                Ok(buf.len())
            } else {
                Err(DbusError::WouldBlock)
            }
        }
    }
}
