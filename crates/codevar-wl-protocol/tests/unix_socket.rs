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

//! Integration tests for the Unix domain socket transport.
//!
//! The tests run real `socketpair` and filesystem sockets through
//! [`WlUnixTransport`] and [`WlUnixPoller`]: byte and descriptor
//! round trips, readiness reporting, session path resolution errors
//! and full [`WlClientDisplay`] to [`WlServerDisplay`] protocol
//! exchanges driven by the `poll(2)` poller.

use std::cell::{Cell, RefCell};
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
use std::rc::Rc;
use std::time::Duration;

use codevar_wl_protocol::{
    WlClientDisplay, WlClock, WlError, WlInterface, WlPollEntry, WlPollEvents, WlPoller, WlRegistryEvent,
    WlServerDisplay, WlTransport, WlUnixPoller, WlUnixTransport,
};

/// Interface announced by the test server.
static TEST_INTERFACE: WlInterface = WlInterface::new("wl_test", 3, &[], &[]);

/// Clock standing still at zero; the fixtures register no timers.
struct FixtureClock;

impl WlClock for FixtureClock {
    fn now_ms(&self) -> u64 {
        0
    }
}

/// Connected client and server over a real socket pair.
struct Fixture {
    server: WlServerDisplay<WlUnixTransport, WlUnixPoller, FixtureClock>,
    client: WlClientDisplay<WlUnixTransport>,
}

impl Fixture {
    fn new() -> Self {
        let (client_transport, server_transport) = WlUnixTransport::pair().unwrap();
        let mut server = WlServerDisplay::new(WlUnixPoller, FixtureClock);
        server.create_client(server_transport).unwrap();
        let client = WlClientDisplay::connect(client_transport).unwrap();
        Self { server, client }
    }

    /// Writes every buffered client request to the server.
    fn flush_client(&mut self) {
        self.client.flush().unwrap();
    }

    /// Lets the server read requests and flush its events.
    fn dispatch_server(&mut self) {
        self.server
            .dispatch(Some(Duration::ZERO))
            .unwrap();
    }

    /// Lets the client read and dispatch pending events.
    fn dispatch_client(&mut self) {
        self.client
            .dispatch(Some(Duration::ZERO))
            .unwrap();
    }
}

#[test]
fn bytes_round_trip_and_would_block() {
    let (mut first, mut second) = WlUnixTransport::pair().unwrap();
    assert_eq!(first.handle(), first.fd() as u64);
    assert!(matches!(
        first.recv(&mut [0u8; 4], &mut Vec::new()),
        Err(WlError::WouldBlock)
    ));

    assert_eq!(first.send(b"hello", &[]).unwrap(), 5);
    let mut buf = [0u8; 16];
    let mut fds = Vec::new();
    assert_eq!(second.recv(&mut buf, &mut fds).unwrap(), 5);
    assert_eq!(&buf[..5], b"hello");
    assert!(fds.is_empty());
    assert!(matches!(
        second.recv(&mut buf, &mut fds),
        Err(WlError::WouldBlock)
    ));

    let events = first
        .wait(Some(Duration::ZERO), WlPollEvents::READABLE)
        .unwrap();
    assert!(events.is_empty());
    second.send(b"x", &[]).unwrap();
    let events = first
        .wait(Some(Duration::ZERO), WlPollEvents::READABLE)
        .unwrap();
    assert!(events.contains(WlPollEvents::READABLE));
}

#[test]
fn wait_expires_without_data() {
    let (mut transport, _peer) = WlUnixTransport::pair().unwrap();
    let events = transport
        .wait(Some(Duration::from_millis(5)), WlPollEvents::READABLE)
        .unwrap();
    assert!(events.is_empty());
}

#[test]
fn file_descriptors_cross_the_socket() {
    let (mut first, mut second) = WlUnixTransport::pair().unwrap();
    let (reader, mut writer) = std::io::pipe().unwrap();
    let payload = b"descriptors ride along";
    assert_eq!(
        first
            .send(payload, &[reader.as_raw_fd()])
            .unwrap(),
        payload.len()
    );
    // The sender copy is released; the receiver must hold a working
    // duplicate installed by the kernel.
    drop(reader);

    let mut buf = [0u8; 32];
    let mut received = Vec::new();
    let n = second.recv(&mut buf, &mut received).unwrap();
    assert_eq!(&buf[..n], payload);
    assert_eq!(received.len(), 1);

    writer.write_all(&[42]).unwrap();
    let mut owned = unsafe { File::from_raw_fd(received[0]) };
    let mut byte = [0u8; 1];
    owned.read_exact(&mut byte).unwrap();
    assert_eq!(byte, [42]);
    drop(owned);

    // `release_fd` closes descriptors the caller discards.
    let (extra, _extra_writer) = std::io::pipe().unwrap();
    assert_eq!(first.send(b"x", &[extra.as_raw_fd()]).unwrap(), 1);
    drop(extra);
    received.clear();
    let n = second.recv(&mut buf, &mut received).unwrap();
    assert_eq!(&buf[..n], b"x");
    assert_eq!(received.len(), 1);
    second.release_fd(received[0]);
    let mut zombie = unsafe { File::from_raw_fd(received[0]) };
    let error = zombie.read(&mut byte).unwrap_err();
    assert_eq!(error.raw_os_error(), Some(9)); // EBADF: the descriptor is gone.
    // The descriptor number is already closed; never close it again.
    let _closed = zombie.into_raw_fd();
}

#[test]
fn send_rejects_more_fds_than_the_limit() {
    let (mut transport, _peer) = WlUnixTransport::pair().unwrap();
    let (reader, _writer) = std::io::pipe().unwrap();
    // The transport refuses lists larger than 16 before any syscall.
    let fds = vec![reader.as_raw_fd(); 17];
    let error = transport.send(b"header", &fds).unwrap_err();
    assert!(matches!(error, WlError::Unsupported(_)));
}

#[test]
fn peer_close_reports_hangup_and_end_of_stream() {
    let (mut transport, peer) = WlUnixTransport::pair().unwrap();
    drop(peer);
    let events = transport
        .wait(Some(Duration::from_secs(2)), WlPollEvents::READABLE)
        .unwrap();
    assert!(events.contains(WlPollEvents::HANGUP));
    let mut buf = [0u8; 8];
    let mut fds = Vec::new();
    assert_eq!(transport.recv(&mut buf, &mut fds).unwrap(), 0);
}

#[test]
fn connect_over_a_filesystem_socket() {
    let dir = std::env::temp_dir().join(format!("codevar-wl-connect-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wayland-test");
    let _ = std::fs::remove_file(&path);

    let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    let mut client_end = WlUnixTransport::connect(path.to_str().unwrap()).unwrap();
    let (stream, _) = listener.accept().unwrap();
    let mut server_end = WlUnixTransport::from_fd(stream.into_raw_fd()).unwrap();

    client_end.send(b"up", &[]).unwrap();
    let mut buf = [0u8; 8];
    let mut fds = Vec::new();
    assert_eq!(server_end.recv(&mut buf, &mut fds).unwrap(), 2);
    assert_eq!(&buf[..2], b"up");

    server_end.send(b"down", &[]).unwrap();
    assert_eq!(client_end.recv(&mut buf, &mut fds).unwrap(), 4);
    assert_eq!(&buf[..4], b"down");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn connect_to_a_missing_socket_fails() {
    let error = WlUnixTransport::connect("/nonexistent/codevar-wayland-0").unwrap_err();
    assert!(matches!(error, WlError::Io(_)));
}

#[test]
fn poller_reports_readiness_and_hangup() {
    let (first, mut peer) = WlUnixTransport::pair().unwrap();
    let mut poller = WlUnixPoller;
    let mut entries = [WlPollEntry {
        handle: first.handle(),
        interest: WlPollEvents::READABLE,
        revents: WlPollEvents::EMPTY,
    }];

    assert_eq!(
        poller
            .poll(&mut [], Some(Duration::ZERO))
            .unwrap(),
        0
    );
    assert_eq!(
        poller
            .poll(&mut entries, Some(Duration::ZERO))
            .unwrap(),
        0
    );
    assert!(entries[0].revents.is_empty());

    peer.send(b"ping", &[]).unwrap();
    assert_eq!(
        poller
            .poll(&mut entries, Some(Duration::ZERO))
            .unwrap(),
        1
    );
    assert!(
        entries[0]
            .revents
            .contains(WlPollEvents::READABLE)
    );

    // A hangup is reported even though only READABLE was requested.
    drop(peer);
    assert_eq!(
        poller
            .poll(&mut entries, Some(Duration::ZERO))
            .unwrap(),
        1
    );
    assert!(entries[0].revents.contains(WlPollEvents::HANGUP));
}

#[test]
fn registry_announces_globals_over_a_real_socket() {
    let globals = Rc::new(RefCell::new(Vec::new()));
    let mut fixture = Fixture::new();
    fixture
        .server
        .add_global(&TEST_INTERFACE, 3)
        .unwrap();

    let registry = fixture.client.get_registry().unwrap();
    let seen = Rc::clone(&globals);
    fixture
        .client
        .add_registry_listener(registry, move |_, event| {
            if let WlRegistryEvent::Global {
                name,
                interface,
                version,
            } = event
            {
                seen.borrow_mut().push((name, interface, version));
            }
            0
        })
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    assert_eq!(*globals.borrow(), [(1u32, String::from("wl_test"), 3u32)]);

    let bound = fixture
        .client
        .registry_bind(registry, 1, &TEST_INTERFACE, 2)
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();

    let (_, server_client) = fixture.server.clients().next().unwrap();
    assert_eq!(server_client.resource_count(), 3);
    assert_eq!(
        server_client
            .resource_interface(bound.id())
            .map(|interface| interface.name),
        Some("wl_test")
    );
    assert_eq!(server_client.resource_version(bound.id()), Some(2));
}

#[test]
fn sync_completes_round_trip_over_a_real_socket() {
    let serial = Rc::new(Cell::new(None));
    let mut fixture = Fixture::new();

    let callback = fixture.client.sync().unwrap();
    let seen = Rc::clone(&serial);
    fixture
        .client
        .add_callback_listener(callback, move |_, value| {
            seen.set(Some(value));
            0
        })
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    assert_eq!(serial.get(), Some(1));
    assert!(fixture.client.is_alive(callback));
    fixture.client.proxy_destroy(callback).unwrap();
    assert!(!fixture.client.is_alive(callback));

    let (_, server_client) = fixture.server.clients().next().unwrap();
    assert_eq!(server_client.resource_count(), 1);
    assert!(
        server_client
            .resource_interface(callback.id())
            .is_none()
    );
}

#[test]
fn protocol_error_from_a_bogus_request_reaches_the_client() {
    let mut fixture = Fixture::new();

    // A `wl_display.sync` from object 42, which does not exist,
    // injected straight into the socket to bypass the client state.
    let bogus = [
        42, 0, 0, 0, // sender id
        0, 0, 12, 0, // (12 << 16) | opcode 0
        0, 0, 0, 0, // new_id argument
    ];
    fixture
        .client
        .connection_mut()
        .transport_mut()
        .send(&bogus, &[])
        .unwrap();
    fixture.dispatch_server();

    let error = fixture
        .client
        .dispatch(Some(Duration::ZERO))
        .unwrap_err();
    let WlError::Protocol(info) = error else {
        panic!("expected a protocol error, got {error:?}");
    };
    assert_eq!(info.code, 0);
    assert_eq!(info.object_id, 1);
    assert_eq!(info.interface, "wl_display");
    assert!(info.message.contains("invalid object 42"));
    assert_eq!(fixture.server.client_count(), 0);
}

#[test]
fn dropping_the_client_removes_it_from_the_server() {
    let (client_transport, server_transport) = WlUnixTransport::pair().unwrap();
    let mut server = WlServerDisplay::new(WlUnixPoller, FixtureClock);
    server.create_client(server_transport).unwrap();
    let client = WlClientDisplay::connect(client_transport).unwrap();
    assert_eq!(server.client_count(), 1);

    drop(client);
    server.dispatch(Some(Duration::ZERO)).unwrap();
    assert_eq!(server.client_count(), 0);
}
