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

//! End-to-end tests for the `wl_compositor`, `wl_surface` and `wl_shm`
//! core interfaces over a real `socketpair`.
//!
//! A scripted compositor records every request a windowing client sends
//! and answers with the events the client needs: `wl_shm.format` right
//! after the bind, `wl_callback.done` for frame callbacks and
//! `wl_buffer.release` after a commit. A second test proves that
//! requests above the negotiated proxy version are rejected before they
//! reach the wire.

use std::cell::{Cell, RefCell};
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::rc::Rc;
use std::time::Duration;

use codevar_wl_protocol::{
    BUFFER_INTERFACE, BUFFER_RELEASE, CALLBACK_DONE, CALLBACK_INTERFACE, COMPOSITOR_CREATE_REGION,
    COMPOSITOR_CREATE_SURFACE, COMPOSITOR_INTERFACE, COMPOSITOR_RELEASE, REGION_INTERFACE, SHM_CREATE_POOL,
    SHM_FORMAT, SHM_FORMAT_ARGB8888, SHM_FORMAT_XRGB8888, SHM_INTERFACE, SHM_POOL_CREATE_BUFFER,
    SHM_POOL_DESTROY, SHM_POOL_INTERFACE, SHM_POOL_RESIZE, SURFACE_ATTACH, SURFACE_COMMIT, SURFACE_DAMAGE,
    SURFACE_DESTROY, SURFACE_FRAME, SURFACE_INTERFACE, SURFACE_OFFSET, SURFACE_SET_INPUT_REGION, WlArgument,
    WlClientDisplay, WlClock, WlDisplayError, WlError, WlProxyId, WlRegistryEvent, WlResult, WlServerDisplay,
    WlUnixPoller, WlUnixTransport,
};

/// Clock standing still at zero; the fixtures register no timers.
struct FixtureClock;

impl WlClock for FixtureClock {
    fn now_ms(&self) -> u64 {
        0
    }
}

type TestServer = WlServerDisplay<WlUnixTransport, WlUnixPoller, FixtureClock>;

/// Connected client and server over a real socket pair.
struct Fixture {
    server: TestServer,
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

/// Requests the scripted compositor observed on the wire.
#[derive(Default)]
struct Record {
    surfaces: Vec<(u32, u32)>,
    regions: Vec<u32>,
    pool_fd_byte: Option<u8>,
    buffers: Vec<(u32, i32, i32, u32)>,
    attached: Option<u32>,
    damage: Vec<(i32, i32, i32, i32)>,
    input_regions: Vec<u32>,
    frames: Vec<u32>,
    commits: usize,
    serial: u32,
}

/// Registers the window related globals and their request handlers.
fn scripted_compositor(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    server.add_global(&COMPOSITOR_INTERFACE, COMPOSITOR_INTERFACE.version)?;
    server.add_global_with(&SHM_INTERFACE, SHM_INTERFACE.version, |client, _, version, id| {
        if client
            .create_resource(id, &SHM_INTERFACE, version)
            .is_err()
        {
            client.post_no_memory();
            return;
        }
        for format in [SHM_FORMAT_ARGB8888, SHM_FORMAT_XRGB8888] {
            if client
                .post_event(id, SHM_FORMAT, vec![WlArgument::Uint(format)])
                .is_err()
            {
                client.post_no_memory();
                return;
            }
        }
    })?;
    install_compositor_handler(server, record)?;
    install_surface_handler(server, record)?;
    install_shm_handlers(server, record)?;
    Ok(())
}

fn install_compositor_handler(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let record = Rc::clone(record);
    server.add_request_handler(&COMPOSITOR_INTERFACE, move |client, _, sender, opcode, args| {
        let version = client.resource_version(sender).unwrap_or(1);
        match opcode {
            COMPOSITOR_CREATE_SURFACE => {
                let Some(WlArgument::NewId(new_id)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "create_surface expects a new_id argument",
                    );
                    return;
                };
                let new_id = *new_id;
                let created = version.min(SURFACE_INTERFACE.version);
                if let Err(error) = client.create_resource(new_id, &SURFACE_INTERFACE, created) {
                    client.post_error(
                        sender,
                        WlDisplayError::NoMemory.code(),
                        format!("create_surface failed: {error}"),
                    );
                    return;
                }
                record
                    .borrow_mut()
                    .surfaces
                    .push((new_id, created));
            }
            COMPOSITOR_CREATE_REGION => {
                let Some(WlArgument::NewId(new_id)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "create_region expects a new_id argument",
                    );
                    return;
                };
                let new_id = *new_id;
                let created = version.min(REGION_INTERFACE.version);
                if let Err(error) = client.create_resource(new_id, &REGION_INTERFACE, created) {
                    client.post_error(
                        sender,
                        WlDisplayError::NoMemory.code(),
                        format!("create_region failed: {error}"),
                    );
                    return;
                }
                record.borrow_mut().regions.push(new_id);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported wl_compositor request",
            ),
        }
    })
}

fn install_surface_handler(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let record = Rc::clone(record);
    server.add_request_handler(
        &SURFACE_INTERFACE,
        move |client, _, sender, opcode, args| match opcode {
            SURFACE_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            SURFACE_ATTACH => {
                let Some(WlArgument::Object(buffer)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "attach expects a buffer argument",
                    );
                    return;
                };
                record.borrow_mut().attached = Some(*buffer);
            }
            SURFACE_DAMAGE => {
                let (
                    Some(WlArgument::Int(x)),
                    Some(WlArgument::Int(y)),
                    Some(WlArgument::Int(width)),
                    Some(WlArgument::Int(height)),
                ) = (args.first(), args.get(1), args.get(2), args.get(3))
                else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "damage expects four int arguments",
                    );
                    return;
                };
                record
                    .borrow_mut()
                    .damage
                    .push((*x, *y, *width, *height));
            }
            SURFACE_FRAME => {
                let Some(WlArgument::NewId(new_id)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "frame expects a new_id argument",
                    );
                    return;
                };
                let new_id = *new_id;
                let version = client
                    .resource_version(sender)
                    .unwrap_or(1)
                    .min(CALLBACK_INTERFACE.version);
                if let Err(error) = client.create_resource(new_id, &CALLBACK_INTERFACE, version) {
                    client.post_error(
                        sender,
                        WlDisplayError::NoMemory.code(),
                        format!("frame callback failed: {error}"),
                    );
                    return;
                }
                record.borrow_mut().frames.push(new_id);
                let serial = {
                    let mut record = record.borrow_mut();
                    record.serial += 1;
                    record.serial
                };
                if client
                    .post_event(new_id, CALLBACK_DONE, vec![WlArgument::Uint(serial)])
                    .is_err()
                    || client.destroy_resource(new_id).is_err()
                {
                    client.post_no_memory();
                }
            }
            SURFACE_SET_INPUT_REGION => {
                let Some(WlArgument::Object(region)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "set_input_region expects a region argument",
                    );
                    return;
                };
                record.borrow_mut().input_regions.push(*region);
            }
            SURFACE_COMMIT => {
                let attached = record.borrow().attached;
                record.borrow_mut().commits += 1;
                if let Some(buffer) = attached
                    && client
                        .post_event(buffer, BUFFER_RELEASE, vec![])
                        .is_err()
                {
                    client.post_no_memory();
                }
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported wl_surface request",
            ),
        },
    )
}

fn install_shm_handlers(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let shm_record = Rc::clone(record);
    server.add_request_handler(&SHM_INTERFACE, move |client, _, sender, opcode, args| {
        if opcode != SHM_CREATE_POOL {
            client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported wl_shm request",
            );
            return;
        }
        let Some(WlArgument::NewId(new_id)) = args.first() else {
            client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "create_pool expects a new_id argument",
            );
            return;
        };
        let new_id = *new_id;
        let version = client.resource_version(sender).unwrap_or(1);
        if let Err(error) = client.create_resource(new_id, &SHM_POOL_INTERFACE, version) {
            client.post_error(
                sender,
                WlDisplayError::NoMemory.code(),
                format!("create_pool failed: {error}"),
            );
            return;
        }
        let Some(fd) = args.get_mut(1).and_then(WlArgument::take_fd) else {
            client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "create_pool expects a descriptor argument",
            );
            return;
        };
        // Safety: the descriptor was demarshalled from the request above
        // and is owned by the server since `take_fd` claimed it.
        let mut pool = unsafe { File::from_raw_fd(fd) };
        let mut byte = [0u8; 1];
        if pool.read_exact(&mut byte).is_ok() {
            shm_record.borrow_mut().pool_fd_byte = Some(byte[0]);
        }
    })?;

    let pool_record = Rc::clone(record);
    server.add_request_handler(&SHM_POOL_INTERFACE, move |client, _, sender, opcode, args| {
        match opcode {
            SHM_POOL_CREATE_BUFFER => {
                // Signature is `niiiiu`: new id, offset, width, height,
                // stride and format.
                let (
                    Some(WlArgument::NewId(new_id)),
                    Some(WlArgument::Int(width)),
                    Some(WlArgument::Int(height)),
                    Some(WlArgument::Uint(format)),
                ) = (args.first(), args.get(2), args.get(3), args.get(5))
                else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "create_buffer expects id, size and format arguments",
                    );
                    return;
                };
                let (new_id, width, height, format) = (*new_id, *width, *height, *format);
                let version = client
                    .resource_version(sender)
                    .unwrap_or(1)
                    .min(BUFFER_INTERFACE.version);
                if let Err(error) = client.create_resource(new_id, &BUFFER_INTERFACE, version) {
                    client.post_error(
                        sender,
                        WlDisplayError::NoMemory.code(),
                        format!("create_buffer failed: {error}"),
                    );
                    return;
                }
                pool_record
                    .borrow_mut()
                    .buffers
                    .push((new_id, width, height, format));
            }
            SHM_POOL_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            SHM_POOL_RESIZE => {}
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported wl_shm_pool request",
            ),
        }
    })?;
    Ok(())
}

/// Listens for `wl_registry.global` announcements of `registry`.
fn collect_globals(
    client: &mut WlClientDisplay<WlUnixTransport>,
    registry: WlProxyId,
) -> Rc<RefCell<Vec<(u32, String, u32)>>> {
    let globals = Rc::new(RefCell::new(Vec::new()));
    let seen = Rc::clone(&globals);
    client
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
    globals
}

/// Looks up the name and advertised version of `interface`.
fn global_named(globals: &[(u32, String, u32)], interface: &str) -> (u32, u32) {
    globals
        .iter()
        .find(|(_, name, _)| name == interface)
        .map(|(name, _, version)| (*name, *version))
        .unwrap_or_else(|| panic!("the server did not announce {interface}"))
}

#[test]
fn window_requests_produce_objects_and_round_trip_events() {
    let record = Rc::new(RefCell::new(Record::default()));
    let mut fixture = Fixture::new();
    scripted_compositor(&mut fixture.server, &record).unwrap();

    let registry = fixture.client.get_registry().unwrap();
    let globals = collect_globals(&mut fixture.client, registry);
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    let (compositor_name, compositor_version) = global_named(&globals.borrow(), "wl_compositor");
    let (shm_name, shm_version) = global_named(&globals.borrow(), "wl_shm");
    assert_eq!((compositor_version, shm_version), (7, 3));

    let compositor = fixture
        .client
        .registry_bind(
            registry,
            compositor_name,
            &COMPOSITOR_INTERFACE,
            compositor_version,
        )
        .unwrap();
    let shm = fixture
        .client
        .registry_bind(registry, shm_name, &SHM_INTERFACE, shm_version)
        .unwrap();

    let formats = Rc::new(RefCell::new(Vec::new()));
    {
        let formats = Rc::clone(&formats);
        fixture
            .client
            .add_listener(shm, move |_, opcode, args| {
                assert_eq!(opcode, SHM_FORMAT);
                if let Some(WlArgument::Uint(format)) = args.first() {
                    formats.borrow_mut().push(*format);
                }
                0
            })
            .unwrap();
    }

    let surface = fixture
        .client
        .marshal_new_id(compositor, COMPOSITOR_CREATE_SURFACE, Vec::new())
        .unwrap();
    let region = fixture
        .client
        .marshal_new_id(compositor, COMPOSITOR_CREATE_REGION, Vec::new())
        .unwrap();

    let (reader, mut writer) = std::io::pipe().unwrap();
    writer.write_all(b"@").unwrap();
    let pool = fixture
        .client
        .marshal_new_id(
            shm,
            SHM_CREATE_POOL,
            vec![WlArgument::Fd(reader.as_raw_fd()), WlArgument::Int(4096)],
        )
        .unwrap();
    let buffer = fixture
        .client
        .marshal_new_id(
            pool,
            SHM_POOL_CREATE_BUFFER,
            vec![
                WlArgument::Int(0),
                WlArgument::Int(8),
                WlArgument::Int(8),
                WlArgument::Int(32),
                WlArgument::Uint(SHM_FORMAT_XRGB8888),
            ],
        )
        .unwrap();

    fixture
        .client
        .marshal_request(
            surface,
            SURFACE_ATTACH,
            vec![
                WlArgument::Object(buffer.id()),
                WlArgument::Int(0),
                WlArgument::Int(0),
            ],
        )
        .unwrap();
    fixture
        .client
        .marshal_request(
            surface,
            SURFACE_DAMAGE,
            vec![
                WlArgument::Int(0),
                WlArgument::Int(0),
                WlArgument::Int(8),
                WlArgument::Int(8),
            ],
        )
        .unwrap();
    fixture
        .client
        .marshal_request(
            surface,
            SURFACE_SET_INPUT_REGION,
            vec![WlArgument::Object(region.id())],
        )
        .unwrap();

    let frame = fixture
        .client
        .marshal_new_id(surface, SURFACE_FRAME, Vec::new())
        .unwrap();
    let frame_serial = Rc::new(Cell::new(None));
    {
        let frame_serial = Rc::clone(&frame_serial);
        fixture
            .client
            .add_callback_listener(frame, move |_, serial| {
                frame_serial.set(Some(serial));
                0
            })
            .unwrap();
    }

    let released = Rc::new(Cell::new(false));
    {
        let released = Rc::clone(&released);
        fixture
            .client
            .add_listener(buffer, move |_, opcode, _| {
                assert_eq!(opcode, BUFFER_RELEASE);
                released.set(true);
                0
            })
            .unwrap();
    }

    fixture
        .client
        .marshal_request(surface, SURFACE_COMMIT, Vec::new())
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    {
        let record = record.borrow();
        assert_eq!(record.surfaces, [(surface.id(), 7)]);
        assert_eq!(record.regions, [region.id()]);
        assert_eq!(record.pool_fd_byte, Some(b'@'));
        assert_eq!(record.buffers, [(buffer.id(), 8, 8, SHM_FORMAT_XRGB8888)]);
        assert_eq!(record.attached, Some(buffer.id()));
        assert_eq!(record.damage, [(0, 0, 8, 8)]);
        assert_eq!(record.input_regions, [region.id()]);
        assert_eq!(record.frames, [frame.id()]);
        assert_eq!(record.commits, 1);
    }

    let (_, server_client) = fixture.server.clients().next().unwrap();
    assert_eq!(
        server_client
            .resource_interface(surface.id())
            .map(|interface| interface.name),
        Some("wl_surface")
    );
    assert_eq!(server_client.resource_version(surface.id()), Some(7));
    assert_eq!(
        server_client
            .resource_interface(region.id())
            .map(|interface| interface.name),
        Some("wl_region")
    );
    assert_eq!(
        server_client
            .resource_interface(pool.id())
            .map(|interface| interface.name),
        Some("wl_shm_pool")
    );
    assert_eq!(server_client.resource_version(pool.id()), Some(3));
    assert_eq!(
        server_client
            .resource_interface(buffer.id())
            .map(|interface| interface.name),
        Some("wl_buffer")
    );

    assert_eq!(*formats.borrow(), [SHM_FORMAT_ARGB8888, SHM_FORMAT_XRGB8888]);
    assert!(released.get());
    assert_eq!(frame_serial.get(), Some(1));
}

#[test]
fn requests_above_the_proxy_version_never_reach_the_wire() {
    let record = Rc::new(RefCell::new(Record::default()));
    let mut fixture = Fixture::new();
    scripted_compositor(&mut fixture.server, &record).unwrap();

    let registry = fixture.client.get_registry().unwrap();
    let globals = collect_globals(&mut fixture.client, registry);
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    let (compositor_name, _) = global_named(&globals.borrow(), "wl_compositor");
    let compositor = fixture
        .client
        .registry_bind(registry, compositor_name, &COMPOSITOR_INTERFACE, 4)
        .unwrap();
    let surface = fixture
        .client
        .marshal_new_id(compositor, COMPOSITOR_CREATE_SURFACE, Vec::new())
        .unwrap();

    // wl_compositor.release is version 7, the proxy knows version 4.
    let error = fixture
        .client
        .marshal_request(compositor, COMPOSITOR_RELEASE, Vec::new())
        .unwrap_err();
    assert!(matches!(error, WlError::InvalidArgument(_)));

    // wl_surface.offset is version 5, the surface inherits version 4.
    let error = fixture
        .client
        .marshal_request(
            surface,
            SURFACE_OFFSET,
            vec![WlArgument::Int(0), WlArgument::Int(0)],
        )
        .unwrap_err();
    assert!(matches!(error, WlError::InvalidArgument(_)));

    let error = fixture
        .client
        .marshal_request(surface, 42, Vec::new())
        .unwrap_err();
    assert!(matches!(
        error,
        WlError::InvalidMethod {
            interface: "wl_surface",
            opcode: 42
        }
    ));

    // wl_surface.attach does not create an object as its first argument.
    let error = fixture
        .client
        .marshal_new_id(
            surface,
            SURFACE_ATTACH,
            vec![WlArgument::Object(0), WlArgument::Int(0), WlArgument::Int(0)],
        )
        .unwrap_err();
    assert!(matches!(error, WlError::InvalidArgument(_)));

    fixture
        .client
        .marshal_request(surface, SURFACE_COMMIT, Vec::new())
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    // A rejected request leaking onto the wire would make the server
    // post a protocol error and tear the connection down here.
    fixture.dispatch_client();
    assert_eq!(fixture.server.client_count(), 1);
    assert_eq!(record.borrow().commits, 1);

    fixture.client.proxy_destroy(surface).unwrap();
    let error = fixture
        .client
        .marshal_request(surface, SURFACE_COMMIT, Vec::new())
        .unwrap_err();
    assert!(matches!(error, WlError::InvalidObject(_)));
}
