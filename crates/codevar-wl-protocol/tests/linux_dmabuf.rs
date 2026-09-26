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

//! End-to-end test for the `linux-dmabuf` protocol over a real
//! `socketpair`.
//!
//! A scripted compositor advertises `zwp_linux_dmabuf_v1` version 5
//! with the deprecated `format` and `modifier` events, imports the
//! plane descriptor of a `zwp_linux_buffer_params_v1`, refuses the
//! legacy `create` with `failed` and honours `create_immed` with a
//! `wl_buffer`. It then publishes a format table and a single scanout
//! tranche over `zwp_linux_dmabuf_feedback_v1` for both the default
//! and the surface feedback. A second test sends
//! `get_surface_feedback` naming an unknown object and observes the
//! compositor's `wl_display.error`.

use std::cell::RefCell;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::rc::Rc;
use std::time::Duration;

use codevar_wl_protocol::{
    BUFFER_INTERFACE, BUFFER_PARAMS_ADD, BUFFER_PARAMS_CREATE, BUFFER_PARAMS_CREATE_IMMED,
    BUFFER_PARAMS_CREATED, BUFFER_PARAMS_DESTROY, BUFFER_PARAMS_FAILED, BUFFER_PARAMS_INTERFACE,
    COMPOSITOR_CREATE_SURFACE, COMPOSITOR_INTERFACE, DMABUF_CREATE_PARAMS, DMABUF_DESTROY,
    DMABUF_FEEDBACK_INTERFACE, DMABUF_FORMAT, DMABUF_GET_DEFAULT_FEEDBACK, DMABUF_GET_SURFACE_FEEDBACK,
    DMABUF_INTERFACE, DMABUF_MODIFIER, DRM_FORMAT_XRGB8888, FEEDBACK_DESTROY, FEEDBACK_DONE,
    FEEDBACK_FORMAT_TABLE, FEEDBACK_MAIN_DEVICE, FEEDBACK_TRANCHE_DONE, FEEDBACK_TRANCHE_FLAGS,
    FEEDBACK_TRANCHE_FORMATS, FEEDBACK_TRANCHE_TARGET_DEVICE, LinuxBufferParamsFlags,
    LinuxDmabufFeedbackTrancheFlags, SURFACE_INTERFACE, WlArgument, WlArray, WlClient, WlClientDisplay,
    WlClock, WlDisplayError, WlError, WlProxyId, WlRegistryEvent, WlResult, WlServerDisplay, WlUnixPoller,
    WlUnixTransport,
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
    /// `(id, version)` of every created `zwp_linux_buffer_params_v1`.
    params: Vec<(u32, u32)>,
    /// `(plane index, descriptor bytes)` of every imported plane.
    planes: Vec<(u32, Vec<u8>)>,
    /// Legacy `create` requests the compositor refused.
    creates: usize,
    /// `(id, version)` of every `wl_buffer` handed out by `create_immed`.
    buffers: Vec<(u32, u32)>,
    /// `(id, version)` of every advertised feedback object.
    feedbacks: Vec<(u32, u32)>,
    /// Surfaces created through `wl_compositor`.
    surfaces: Vec<u32>,
    /// Format tables whose descriptors must stay open until flushed.
    tables: Vec<File>,
}

/// Events the client listeners observed.
#[derive(Default)]
struct ClientLog {
    formats: Vec<u32>,
    modifiers: Vec<(u32, u32, u32)>,
    failed_events: usize,
    created_events: usize,
    format_table: Vec<u8>,
    table_size: u32,
    main_devices: Vec<Vec<u8>>,
    tranche_devices: Vec<Vec<u8>>,
    tranche_formats: Vec<Vec<u8>>,
    tranche_flags: Vec<u32>,
    tranche_dones: usize,
    dones: usize,
}

/// Contents of the advertised format table: one `XR24` row with the
/// linear modifier, laid out in host endianness like the protocol.
fn format_table_bytes() -> Vec<u8> {
    let mut table = Vec::with_capacity(16);
    table.extend_from_slice(&DRM_FORMAT_XRGB8888.to_ne_bytes());
    table.extend_from_slice(&0u32.to_ne_bytes());
    table.extend_from_slice(&0u64.to_ne_bytes());
    table
}

/// Registers the compositor and `zwp_linux_dmabuf_v1` globals with handlers.
fn scripted_session(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    server.add_global(&COMPOSITOR_INTERFACE, COMPOSITOR_INTERFACE.version)?;

    server.add_global_with(
        &DMABUF_INTERFACE,
        DMABUF_INTERFACE.version,
        |client, _, version, id| {
            if client
                .create_resource(id, &DMABUF_INTERFACE, version)
                .is_err()
            {
                client.post_no_memory();
                return;
            }
            // Versions 1 to 3 advertise formats through these deprecated
            // events; version 4 and newer are expected to ignore them.
            if client
                .post_event(id, DMABUF_FORMAT, vec![WlArgument::Uint(DRM_FORMAT_XRGB8888)])
                .is_err()
            {
                client.post_no_memory();
            }
            if client
                .post_event(
                    id,
                    DMABUF_MODIFIER,
                    vec![
                        WlArgument::Uint(DRM_FORMAT_XRGB8888),
                        WlArgument::Uint(0x00FF_FFFF),
                        WlArgument::Uint(0xFFFF_FFFF),
                    ],
                )
                .is_err()
            {
                client.post_no_memory();
            }
        },
    )?;

    install_compositor_handler(server, record)?;
    install_dmabuf_handler(server, record)?;
    install_params_handler(server, record)?;
    install_feedback_handler(server)?;
    Ok(())
}

fn install_compositor_handler(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let record = Rc::clone(record);
    server.add_request_handler(&COMPOSITOR_INTERFACE, move |client, _, sender, opcode, args| {
        if opcode != COMPOSITOR_CREATE_SURFACE {
            client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported wl_compositor request",
            );
            return;
        }
        let Some(WlArgument::NewId(new_id)) = args.first() else {
            client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "create_surface expects a new_id argument",
            );
            return;
        };
        let new_id = *new_id;
        let version = client
            .resource_version(sender)
            .unwrap_or(1)
            .min(SURFACE_INTERFACE.version);
        if let Err(error) = client.create_resource(new_id, &SURFACE_INTERFACE, version) {
            client.post_error(
                sender,
                WlDisplayError::NoMemory.code(),
                format!("create_surface failed: {error}"),
            );
            return;
        }
        record.borrow_mut().surfaces.push(new_id);
    })
}

/// Queues the full feedback sequence: the format table, the main
/// device, one scanout tranche and `done`.
///
/// The descriptor backing `format_table` moves into `record` before
/// the events are queued, so it stays open until the connection has
/// flushed it to the client and the fixture tears down.
fn post_feedback(client: &mut WlClient<WlUnixTransport>, feedback: u32, record: &Rc<RefCell<Record>>) {
    let table = format_table_bytes();
    let fd = unsafe { libc::memfd_create(c"codevar-dmabuf-table".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        client.post_no_memory();
        return;
    }
    // Safety: `fd` is a fresh descriptor owned by this test.
    let mut table_file = unsafe { File::from_raw_fd(fd) };
    if table_file.write_all(&table).is_err() || table_file.rewind().is_err() {
        client.post_no_memory();
        return;
    }
    let table_fd = table_file.as_raw_fd();
    record.borrow_mut().tables.push(table_file);

    let device = 1u64.to_ne_bytes();
    let mut formats = Vec::with_capacity(2);
    formats.extend_from_slice(&0u16.to_ne_bytes());
    let events: [(u32, Vec<WlArgument>); 7] = [
        (
            FEEDBACK_FORMAT_TABLE,
            vec![WlArgument::Fd(table_fd), WlArgument::Uint(table.len() as u32)],
        ),
        (
            FEEDBACK_MAIN_DEVICE,
            vec![WlArgument::Array(Some(WlArray::from_bytes(&device)))],
        ),
        (
            FEEDBACK_TRANCHE_TARGET_DEVICE,
            vec![WlArgument::Array(Some(WlArray::from_bytes(&device)))],
        ),
        (
            FEEDBACK_TRANCHE_FORMATS,
            vec![WlArgument::Array(Some(WlArray::from_bytes(&formats)))],
        ),
        (
            FEEDBACK_TRANCHE_FLAGS,
            vec![WlArgument::Uint(LinuxDmabufFeedbackTrancheFlags::Scanout.code())],
        ),
        (FEEDBACK_TRANCHE_DONE, vec![]),
        (FEEDBACK_DONE, vec![]),
    ];
    for (opcode, args) in events {
        if client.post_event(feedback, opcode, args).is_err() {
            client.post_no_memory();
            return;
        }
    }
}

fn install_dmabuf_handler(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let dmabuf_record = Rc::clone(record);
    server.add_request_handler(&DMABUF_INTERFACE, move |client, _, sender, opcode, args| {
        let version = client.resource_version(sender).unwrap_or(1);
        match opcode {
            DMABUF_CREATE_PARAMS => {
                let Some(WlArgument::NewId(new_id)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "create_params expects a new_id argument",
                    );
                    return;
                };
                let new_id = *new_id;
                let created = version.min(BUFFER_PARAMS_INTERFACE.version);
                if let Err(error) = client.create_resource(new_id, &BUFFER_PARAMS_INTERFACE, created) {
                    client.post_error(
                        sender,
                        WlDisplayError::NoMemory.code(),
                        format!("create_params failed: {error}"),
                    );
                    return;
                }
                dmabuf_record
                    .borrow_mut()
                    .params
                    .push((new_id, created));
            }
            DMABUF_GET_DEFAULT_FEEDBACK | DMABUF_GET_SURFACE_FEEDBACK => {
                let Some(WlArgument::NewId(new_id)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "feedback requests expect a new_id argument",
                    );
                    return;
                };
                let new_id = *new_id;
                let created = version.min(DMABUF_FEEDBACK_INTERFACE.version);
                if let Err(error) = client.create_resource(new_id, &DMABUF_FEEDBACK_INTERFACE, created) {
                    client.post_error(
                        sender,
                        WlDisplayError::NoMemory.code(),
                        format!("feedback request failed: {error}"),
                    );
                    return;
                }
                dmabuf_record
                    .borrow_mut()
                    .feedbacks
                    .push((new_id, created));
                post_feedback(client, new_id, &dmabuf_record);
            }
            DMABUF_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported zwp_linux_dmabuf_v1 request",
            ),
        }
    })
}

fn install_params_handler(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let params_record = Rc::clone(record);
    server.add_request_handler(
        &BUFFER_PARAMS_INTERFACE,
        move |client, _, sender, opcode, args| match opcode {
            BUFFER_PARAMS_ADD => {
                let Some(fd) = args.get_mut(0).and_then(WlArgument::take_fd) else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "add expects a descriptor argument",
                    );
                    return;
                };
                let Some(WlArgument::Uint(plane)) = args.get(1) else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "add expects a plane index",
                    );
                    return;
                };
                let plane = *plane;
                // Safety: the descriptor was demarshalled from the request
                // above and is owned by the server since `take_fd` claimed
                // it, so wrapping it in a `File` closes it exactly once.
                let mut plane_fd = unsafe { File::from_raw_fd(fd) };
                let mut bytes = Vec::new();
                if plane_fd.read_to_end(&mut bytes).is_ok() {
                    params_record
                        .borrow_mut()
                        .planes
                        .push((plane, bytes));
                }
            }
            BUFFER_PARAMS_CREATE => {
                params_record.borrow_mut().creates += 1;
                // The scripted compositor refuses this legacy attempt.
                if client
                    .post_event(sender, BUFFER_PARAMS_FAILED, vec![])
                    .is_err()
                {
                    client.post_no_memory();
                }
            }
            BUFFER_PARAMS_CREATE_IMMED => {
                let Some(WlArgument::NewId(new_id)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "create_immed expects a new_id argument",
                    );
                    return;
                };
                let new_id = *new_id;
                let version = client
                    .resource_version(sender)
                    .unwrap_or(1)
                    .min(BUFFER_INTERFACE.version);
                if let Err(error) = client.create_resource(new_id, &BUFFER_INTERFACE, version) {
                    client.post_error(
                        sender,
                        WlDisplayError::NoMemory.code(),
                        format!("create_immed failed: {error}"),
                    );
                    return;
                }
                params_record
                    .borrow_mut()
                    .buffers
                    .push((new_id, version));
            }
            BUFFER_PARAMS_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported zwp_linux_buffer_params_v1 request",
            ),
        },
    )
}

fn install_feedback_handler(server: &mut TestServer) -> WlResult<()> {
    server.add_request_handler(
        &DMABUF_FEEDBACK_INTERFACE,
        |client, _, sender, opcode, _| match opcode {
            FEEDBACK_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported zwp_linux_dmabuf_feedback_v1 request",
            ),
        },
    )
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

/// Copies the byte array of `args[index]`, or an empty vector.
fn array_arg(args: &[WlArgument], index: usize) -> Vec<u8> {
    match args.get(index) {
        Some(WlArgument::Array(Some(array))) => array.as_bytes().to_vec(),
        _ => Vec::new(),
    }
}

/// Records every event of a `zwp_linux_dmabuf_feedback_v1` proxy.
fn feedback_listener(
    log: Rc<RefCell<ClientLog>>,
) -> impl FnMut(&mut WlClientDisplay<WlUnixTransport>, u32, &mut [WlArgument]) -> i32 {
    move |_, opcode, args| {
        match opcode {
            FEEDBACK_FORMAT_TABLE => {
                // The framework closes descriptors it did not hand over,
                // so ownership of this one moves into the `File` below.
                let fd = args.get_mut(0).and_then(WlArgument::take_fd);
                if let Some(WlArgument::Uint(size)) = args.get(1) {
                    log.borrow_mut().table_size = *size;
                }
                if let Some(fd) = fd {
                    // Safety: the descriptor was demarshalled from the
                    // event and `take_fd` transferred its ownership, so
                    // the `File` closes it exactly once.
                    let mut table = unsafe { File::from_raw_fd(fd) };
                    let mut bytes = Vec::new();
                    if table.read_to_end(&mut bytes).is_ok() {
                        log.borrow_mut().format_table = bytes;
                    }
                }
            }
            FEEDBACK_MAIN_DEVICE => {
                let bytes = array_arg(args, 0);
                log.borrow_mut().main_devices.push(bytes);
            }
            FEEDBACK_TRANCHE_TARGET_DEVICE => {
                let bytes = array_arg(args, 0);
                log.borrow_mut().tranche_devices.push(bytes);
            }
            FEEDBACK_TRANCHE_FORMATS => {
                let bytes = array_arg(args, 0);
                log.borrow_mut().tranche_formats.push(bytes);
            }
            FEEDBACK_TRANCHE_FLAGS => {
                if let Some(WlArgument::Uint(flags)) = args.first() {
                    log.borrow_mut().tranche_flags.push(*flags);
                }
            }
            FEEDBACK_TRANCHE_DONE => log.borrow_mut().tranche_dones += 1,
            FEEDBACK_DONE => log.borrow_mut().dones += 1,
            _ => {}
        }
        0
    }
}

#[test]
fn dmabuf_import_create_and_feedback_round_trip() {
    let record = Rc::new(RefCell::new(Record::default()));
    let mut fixture = Fixture::new();
    scripted_session(&mut fixture.server, &record).unwrap();

    let registry = fixture.client.get_registry().unwrap();
    let globals = collect_globals(&mut fixture.client, registry);
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    let (compositor_name, _) = global_named(&globals.borrow(), "wl_compositor");
    let (dmabuf_name, dmabuf_version) = global_named(&globals.borrow(), "zwp_linux_dmabuf_v1");
    assert_eq!(dmabuf_version, DMABUF_INTERFACE.version);

    let compositor = fixture
        .client
        .registry_bind(
            registry,
            compositor_name,
            &COMPOSITOR_INTERFACE,
            COMPOSITOR_INTERFACE.version,
        )
        .unwrap();
    let dmabuf = fixture
        .client
        .registry_bind(registry, dmabuf_name, &DMABUF_INTERFACE, dmabuf_version)
        .unwrap();

    // The bind answers with the events versions 1 to 3 rely on.
    let log = Rc::new(RefCell::new(ClientLog::default()));
    {
        let log = Rc::clone(&log);
        fixture
            .client
            .add_listener(dmabuf, move |_, opcode, args| {
                match opcode {
                    DMABUF_FORMAT => {
                        if let Some(WlArgument::Uint(format)) = args.first() {
                            log.borrow_mut().formats.push(*format);
                        }
                    }
                    DMABUF_MODIFIER => {
                        if let (
                            Some(WlArgument::Uint(format)),
                            Some(WlArgument::Uint(hi)),
                            Some(WlArgument::Uint(lo)),
                        ) = (args.first(), args.get(1), args.get(2))
                        {
                            log.borrow_mut()
                                .modifiers
                                .push((*format, *hi, *lo));
                        }
                    }
                    _ => {}
                }
                0
            })
            .unwrap();
    }
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();
    {
        let log = log.borrow();
        assert_eq!(log.formats, [DRM_FORMAT_XRGB8888]);
        assert_eq!(log.modifiers, [(DRM_FORMAT_XRGB8888, 0x00FF_FFFF, 0xFFFF_FFFF)]);
    }

    let surface = fixture
        .client
        .marshal_new_id(compositor, COMPOSITOR_CREATE_SURFACE, Vec::new())
        .unwrap();

    // First params: import one plane, then the compositor refuses the
    // legacy create with `failed`.
    let params = fixture
        .client
        .marshal_new_id(dmabuf, DMABUF_CREATE_PARAMS, Vec::new())
        .unwrap();
    {
        let log = Rc::clone(&log);
        fixture
            .client
            .add_listener(params, move |_, opcode, _| {
                match opcode {
                    BUFFER_PARAMS_FAILED => log.borrow_mut().failed_events += 1,
                    BUFFER_PARAMS_CREATED => log.borrow_mut().created_events += 1,
                    _ => {}
                }
                0
            })
            .unwrap();
    }

    let payload: &[u8] = b"DMABUF!";
    let fd = unsafe { libc::memfd_create(c"codevar-dmabuf-plane".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0, "memfd_create must provide the plane descriptor");
    // Safety: `fd` is a fresh descriptor owned by this test.
    let mut plane = unsafe { File::from_raw_fd(fd) };
    plane.write_all(payload).unwrap();
    plane.rewind().unwrap();

    fixture
        .client
        .marshal_request(
            params,
            BUFFER_PARAMS_ADD,
            vec![
                WlArgument::Fd(plane.as_raw_fd()),
                WlArgument::Uint(0),
                WlArgument::Uint(0),
                WlArgument::Uint(2560),
                WlArgument::Uint(0),
                WlArgument::Uint(0),
            ],
        )
        .unwrap();
    fixture
        .client
        .marshal_request(
            params,
            BUFFER_PARAMS_CREATE,
            vec![
                WlArgument::Int(640),
                WlArgument::Int(480),
                WlArgument::Uint(DRM_FORMAT_XRGB8888),
                WlArgument::Uint(0),
            ],
        )
        .unwrap();
    fixture
        .client
        .marshal_request(params, BUFFER_PARAMS_DESTROY, Vec::new())
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    // The descriptor has been sent and imported; this closes the copy
    // the connection kept queued.
    drop(plane);
    fixture.dispatch_client();

    {
        let record = record.borrow();
        assert_eq!(record.params, [(params.id(), 5)]);
        assert_eq!(record.planes, [(0, payload.to_vec())]);
        assert_eq!(record.creates, 1);
        assert_eq!(record.surfaces, [surface.id()]);
    }
    {
        let log = log.borrow();
        assert_eq!(log.failed_events, 1);
        assert_eq!(log.created_events, 0);
    }

    // Second params: the same plane succeeds through `create_immed`.
    let params2 = fixture
        .client
        .marshal_new_id(dmabuf, DMABUF_CREATE_PARAMS, Vec::new())
        .unwrap();
    let payload2: &[u8] = b"DMABUF create_immed";
    let fd2 = unsafe { libc::memfd_create(c"codevar-dmabuf-plane2".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd2 >= 0, "memfd_create must provide the second plane descriptor");
    // Safety: `fd2` is a fresh descriptor owned by this test.
    let mut plane2 = unsafe { File::from_raw_fd(fd2) };
    plane2.write_all(payload2).unwrap();
    plane2.rewind().unwrap();

    fixture
        .client
        .marshal_request(
            params2,
            BUFFER_PARAMS_ADD,
            vec![
                WlArgument::Fd(plane2.as_raw_fd()),
                WlArgument::Uint(0),
                WlArgument::Uint(0),
                WlArgument::Uint(2560),
                WlArgument::Uint(0),
                WlArgument::Uint(0),
            ],
        )
        .unwrap();
    let buffer = fixture
        .client
        .marshal_new_id(
            params2,
            BUFFER_PARAMS_CREATE_IMMED,
            vec![
                WlArgument::Int(640),
                WlArgument::Int(480),
                WlArgument::Uint(DRM_FORMAT_XRGB8888),
                WlArgument::Uint(LinuxBufferParamsFlags::YInvert.code()),
            ],
        )
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    drop(plane2);
    fixture.dispatch_client();

    assert!(fixture.client.protocol_error().is_none());
    assert!(fixture.client.is_alive(dmabuf));
    assert!(fixture.client.is_alive(params2));
    assert!(fixture.client.is_alive(buffer));
    assert_eq!(fixture.server.client_count(), 1);
    {
        let record = record.borrow();
        assert_eq!(record.planes, [(0, payload.to_vec()), (0, payload2.to_vec())]);
        assert_eq!(record.buffers, [(buffer.id(), 1)]);
        assert_eq!(record.params, [(params.id(), 5), (params2.id(), 5)]);
    }
    {
        let (_, server_client) = fixture.server.clients().next().unwrap();
        assert_eq!(
            server_client
                .resource_interface(buffer.id())
                .map(|interface| interface.name),
            Some("wl_buffer")
        );
        assert_eq!(server_client.resource_version(buffer.id()), Some(1));
        assert!(
            server_client
                .resource_interface(params.id())
                .is_none()
        );
        assert_eq!(
            server_client
                .resource_interface(params2.id())
                .map(|interface| interface.name),
            Some("zwp_linux_buffer_params_v1")
        );
        assert_eq!(server_client.resource_version(params2.id()), Some(5));
    }

    // The default feedback carries the format table and one tranche.
    let default_feedback = fixture
        .client
        .marshal_new_id(dmabuf, DMABUF_GET_DEFAULT_FEEDBACK, Vec::new())
        .unwrap();
    fixture
        .client
        .add_listener(default_feedback, feedback_listener(Rc::clone(&log)))
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    // The surface feedback repeats the sequence bound to the surface.
    let surface_feedback = fixture
        .client
        .marshal_new_id(
            dmabuf,
            DMABUF_GET_SURFACE_FEEDBACK,
            vec![WlArgument::Object(surface.id())],
        )
        .unwrap();
    fixture
        .client
        .add_listener(surface_feedback, feedback_listener(Rc::clone(&log)))
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    {
        let record = record.borrow();
        assert_eq!(
            record.feedbacks,
            [(default_feedback.id(), 5), (surface_feedback.id(), 5)]
        );
        assert_eq!(record.tables.len(), 2);
    }
    {
        let log = log.borrow();
        assert_eq!(log.format_table, format_table_bytes());
        assert_eq!(log.table_size, 16);
        // Both feedback objects publish the same sequence.
        let device = 1u64.to_ne_bytes().to_vec();
        assert_eq!(log.main_devices, [device.clone(), device.clone()]);
        assert_eq!(log.tranche_devices, [device.clone(), device]);
        let formats = 0u16.to_ne_bytes().to_vec();
        assert_eq!(log.tranche_formats, [formats.clone(), formats]);
        let scanout = LinuxDmabufFeedbackTrancheFlags::Scanout.code();
        assert_eq!(log.tranche_flags, [scanout, scanout]);
        assert_eq!(log.tranche_dones, 2);
        assert_eq!(log.dones, 2);
    }
    assert!(fixture.client.protocol_error().is_none());
}

#[test]
fn bogus_surface_feedback_reports_a_protocol_error() {
    let record = Rc::new(RefCell::new(Record::default()));
    let mut fixture = Fixture::new();
    scripted_session(&mut fixture.server, &record).unwrap();

    let registry = fixture.client.get_registry().unwrap();
    let globals = collect_globals(&mut fixture.client, registry);
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    let (dmabuf_name, dmabuf_version) = global_named(&globals.borrow(), "zwp_linux_dmabuf_v1");
    let dmabuf = fixture
        .client
        .registry_bind(registry, dmabuf_name, &DMABUF_INTERFACE, dmabuf_version)
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    assert_eq!(fixture.server.client_count(), 1);

    // The compositor rejects the unknown surface with `wl_display.error`.
    fixture
        .client
        .marshal_new_id(
            dmabuf,
            DMABUF_GET_SURFACE_FEEDBACK,
            vec![WlArgument::Object(0xDEAD_BEEF)],
        )
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();

    let dispatch_error = fixture
        .client
        .dispatch(Some(Duration::ZERO))
        .expect_err("the compositor must reject the unknown surface");
    let WlError::Protocol(error) = dispatch_error else {
        panic!("dispatch failed with {dispatch_error}");
    };
    assert_eq!(error.code, WlDisplayError::InvalidMethod.code());
    assert!(error.message.contains("get_surface_feedback"));
    assert_eq!(
        fixture
            .client
            .protocol_error()
            .map(|error| error.code),
        Some(WlDisplayError::InvalidMethod.code())
    );
    assert_eq!(fixture.server.client_count(), 0);
}
