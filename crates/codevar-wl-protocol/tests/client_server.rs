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

//! End-to-end tests wiring a real [`WlClientDisplay`] to a real
//! [`WlServerDisplay`] over an in-memory pipe pair.
//!
//! The two sides are driven by hand: the client flushes its requests,
//! the server dispatches and flushes and the client dispatches the
//! resulting events. A blocking `roundtrip` would deadlock on a single
//! thread, so no test uses it.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Duration;

use codevar_wl_protocol::{
    WlClientDisplay, WlClock, WlError, WlInterface, WlPollEntry, WlPollEvents, WlPoller, WlRegistryEvent,
    WlResult, WlServerDisplay, WlTransport,
};

static TEST_INTERFACE: WlInterface = WlInterface::new("wl_test", 3, &[], &[]);

/// Bytes buffered between the two ends of the pipe.
#[derive(Default)]
struct Wire {
    to_server: Vec<u8>,
    to_server_pos: usize,
    to_client: Vec<u8>,
    to_client_pos: usize,
    client_closed: bool,
}

/// Server end of the pipe; owned by the display's client.
struct ServerEnd {
    wire: Rc<RefCell<Wire>>,
}

impl WlTransport for ServerEnd {
    fn recv(&mut self, buf: &mut [u8], _fds: &mut Vec<i32>) -> WlResult<usize> {
        let mut wire = self.wire.borrow_mut();
        if wire.to_server_pos < wire.to_server.len() {
            let available = wire.to_server.len() - wire.to_server_pos;
            let count = available.min(buf.len());
            let start = wire.to_server_pos;
            buf[..count].copy_from_slice(&wire.to_server[start..start + count]);
            wire.to_server_pos += count;
            return Ok(count);
        }
        if wire.client_closed {
            return Ok(0);
        }
        Err(WlError::WouldBlock)
    }

    fn send(&mut self, data: &[u8], _fds: &[i32]) -> WlResult<usize> {
        self.wire.borrow_mut().to_client.extend_from_slice(data);
        Ok(data.len())
    }

    fn wait(&mut self, _timeout: Option<Duration>, mask: WlPollEvents) -> WlResult<WlPollEvents> {
        let wire = self.wire.borrow();
        let mut events = WlPollEvents::EMPTY;
        if wire.client_closed {
            events.insert(WlPollEvents::HANGUP);
        }
        if wire.to_server_pos < wire.to_server.len() {
            events.insert(WlPollEvents::READABLE);
        }
        Ok(events.intersection(mask))
    }

    fn handle(&self) -> u64 {
        1
    }
}

/// Client end of the pipe; owned by [`WlClientDisplay`].
///
/// Dropping it marks the peer as closed so the server observes a
/// hangup on its next dispatch.
struct ClientEnd {
    wire: Rc<RefCell<Wire>>,
}

impl Drop for ClientEnd {
    fn drop(&mut self) {
        self.wire.borrow_mut().client_closed = true;
    }
}

impl WlTransport for ClientEnd {
    fn recv(&mut self, buf: &mut [u8], _fds: &mut Vec<i32>) -> WlResult<usize> {
        let mut wire = self.wire.borrow_mut();
        if wire.to_client_pos >= wire.to_client.len() {
            return Err(WlError::WouldBlock);
        }
        let available = wire.to_client.len() - wire.to_client_pos;
        let count = available.min(buf.len());
        let start = wire.to_client_pos;
        buf[..count].copy_from_slice(&wire.to_client[start..start + count]);
        wire.to_client_pos += count;
        Ok(count)
    }

    fn send(&mut self, data: &[u8], _fds: &[i32]) -> WlResult<usize> {
        self.wire.borrow_mut().to_server.extend_from_slice(data);
        Ok(data.len())
    }

    fn wait(&mut self, _timeout: Option<Duration>, mask: WlPollEvents) -> WlResult<WlPollEvents> {
        let wire = self.wire.borrow();
        let mut events = WlPollEvents::EMPTY;
        if wire.to_client_pos < wire.to_client.len() {
            events.insert(WlPollEvents::READABLE);
        }
        Ok(events.intersection(mask))
    }

    fn handle(&self) -> u64 {
        2
    }
}

/// Poller reporting the readiness of the server end of the pipe.
///
/// Hangup is reported outside the interest mask, like `poll(2)`,
/// so a closed client can never be missed.
struct WirePoller {
    wire: Rc<RefCell<Wire>>,
}

impl WlPoller for WirePoller {
    fn poll(&mut self, entries: &mut [WlPollEntry], _timeout: Option<Duration>) -> WlResult<usize> {
        let wire = self.wire.borrow();
        let mut ready = 0;
        for entry in entries.iter_mut() {
            let mut events = WlPollEvents::EMPTY;
            if wire.client_closed {
                events.insert(WlPollEvents::HANGUP);
            }
            if wire.to_server_pos < wire.to_server.len() {
                events.insert(WlPollEvents::READABLE);
            }
            if entry.interest.contains(WlPollEvents::WRITABLE) {
                events.insert(WlPollEvents::WRITABLE);
            }
            entry.revents = events;
            if !events.is_empty() {
                ready += 1;
            }
        }
        Ok(ready)
    }
}

/// Clock standing still at zero.
struct WireClock;

impl WlClock for WireClock {
    fn now_ms(&self) -> u64 {
        0
    }
}

type TestServer = WlServerDisplay<ServerEnd, WirePoller, WireClock>;

/// Connected client and server over a shared pipe.
struct Fixture {
    wire: Rc<RefCell<Wire>>,
    server: TestServer,
    client: WlClientDisplay<ClientEnd>,
}

impl Fixture {
    fn new() -> Self {
        let wire = Rc::new(RefCell::new(Wire::default()));
        let mut server = WlServerDisplay::new(
            WirePoller {
                wire: Rc::clone(&wire),
            },
            WireClock,
        );
        let client_id = server
            .create_client(ServerEnd {
                wire: Rc::clone(&wire),
            })
            .unwrap();
        assert_eq!(server.client_count(), 1);
        let client = WlClientDisplay::connect(ClientEnd {
            wire: Rc::clone(&wire),
        })
        .unwrap();
        assert_eq!(
            server
                .client(client_id)
                .map(|client| client.resource_count()),
            Some(1)
        );
        Self { wire, server, client }
    }

    /// Writes every buffered client request to the server.
    fn flush_client(&mut self) {
        self.client.flush().unwrap();
    }

    /// Lets the server read requests and flush its events.
    fn dispatch_server(&mut self) {
        self.server.dispatch(Some(Duration::ZERO)).unwrap();
    }

    /// Lets the client read and dispatch pending events.
    fn dispatch_client(&mut self) {
        self.client.dispatch(Some(Duration::ZERO)).unwrap();
    }
}

#[test]
fn registry_announces_globals_and_binds() {
    let globals = Rc::new(RefCell::new(Vec::new()));
    let mut fixture = Fixture::new();
    fixture.server.add_global(&TEST_INTERFACE, 3).unwrap();

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
fn remove_global_reaches_the_client() {
    let removed = Rc::new(RefCell::new(Vec::new()));
    let mut fixture = Fixture::new();
    fixture.server.add_global(&TEST_INTERFACE, 3).unwrap();

    let registry = fixture.client.get_registry().unwrap();
    let seen = Rc::clone(&removed);
    fixture
        .client
        .add_registry_listener(registry, move |_, event| {
            if let WlRegistryEvent::GlobalRemove { name } = event {
                seen.borrow_mut().push(name);
            }
            0
        })
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    fixture.server.remove_global(1).unwrap();
    fixture.server.flush_clients();
    fixture.dispatch_client();

    assert_eq!(*removed.borrow(), [1u32]);
}

#[test]
fn sync_delivers_done_and_delete_id() {
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
    assert!(server_client.resource_interface(callback.id()).is_none());
}

#[test]
fn protocol_error_from_a_bogus_request_reaches_the_client() {
    let mut fixture = Fixture::new();

    // A `wl_display.sync` from object 42, which does not exist.
    fixture.wire.borrow_mut().to_server.extend_from_slice(&[
        42, 0, 0, 0, // sender id
        0, 0, 12, 0, // (12 << 16) | opcode 0
        0, 0, 0, 0, // new_id argument
    ]);
    fixture.dispatch_server();

    let error = fixture.client.dispatch(Some(Duration::ZERO)).unwrap_err();
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
    let wire = Rc::new(RefCell::new(Wire::default()));
    let mut server = WlServerDisplay::new(
        WirePoller {
            wire: Rc::clone(&wire),
        },
        WireClock,
    );
    server
        .create_client(ServerEnd {
            wire: Rc::clone(&wire),
        })
        .unwrap();
    let client = WlClientDisplay::connect(ClientEnd {
        wire: Rc::clone(&wire),
    })
    .unwrap();
    assert_eq!(server.client_count(), 1);

    drop(client);
    server.dispatch(Some(Duration::ZERO)).unwrap();
    assert_eq!(server.client_count(), 0);
}
