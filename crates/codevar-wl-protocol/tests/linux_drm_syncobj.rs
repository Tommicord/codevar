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

//! End-to-end test for the `linux-drm-syncobj` protocol over a real
//! `socketpair`.
//!
//! A scripted compositor announces `wp_linux_drm_syncobj_manager_v1`,
//! imports a timeline file descriptor written by the client, binds a
//! `wp_linux_drm_syncobj_surface_v1` to a `wl_surface` and records the
//! acquire and release timeline points the client sets. The final
//! `wl_surface.commit` carries pending points but no buffer, so the
//! compositor answers with the `no_buffer` protocol error.

use std::cell::RefCell;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::rc::Rc;
use std::time::Duration;

use codevar_wl_protocol::{
    COMPOSITOR_CREATE_SURFACE, COMPOSITOR_INTERFACE, DRM_SYNCOBJ_MANAGER_DESTROY,
    DRM_SYNCOBJ_MANAGER_GET_SURFACE, DRM_SYNCOBJ_MANAGER_IMPORT_TIMELINE, DRM_SYNCOBJ_MANAGER_INTERFACE,
    DRM_SYNCOBJ_SURFACE_DESTROY, DRM_SYNCOBJ_SURFACE_INTERFACE, DRM_SYNCOBJ_SURFACE_SET_ACQUIRE_POINT,
    DRM_SYNCOBJ_SURFACE_SET_RELEASE_POINT, DRM_SYNCOBJ_TIMELINE_DESTROY, DRM_SYNCOBJ_TIMELINE_INTERFACE,
    DrmSyncobjManagerError, DrmSyncobjSurfaceError, SURFACE_COMMIT, SURFACE_DESTROY, SURFACE_INTERFACE,
    WlArgument, WlClientDisplay, WlClock, WlDisplayError, WlError, WlProxyId, WlRegistryEvent, WlResult,
    WlServerDisplay, WlUnixPoller, WlUnixTransport,
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
    surfaces: Vec<u32>,
    /// `(id, version)` of every imported timeline.
    timelines: Vec<(u32, u32)>,
    /// Bytes read from the imported timeline descriptor.
    fd_bytes: Vec<u8>,
    /// `(id, wl_surface id)` of every synchronization surface.
    syncobj_surfaces: Vec<(u32, u32)>,
    /// `(timeline id, point)` of every acquire point.
    acquire_points: Vec<(u32, u64)>,
    /// `(timeline id, point)` of every release point.
    release_points: Vec<(u32, u64)>,
    commits: usize,
}

/// Registers the compositor and syncobj manager globals with handlers.
fn scripted_session(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    server.add_global(&COMPOSITOR_INTERFACE, COMPOSITOR_INTERFACE.version)?;
    server.add_global_with(
        &DRM_SYNCOBJ_MANAGER_INTERFACE,
        DRM_SYNCOBJ_MANAGER_INTERFACE.version,
        |client, _, version, id| {
            if client
                .create_resource(id, &DRM_SYNCOBJ_MANAGER_INTERFACE, version)
                .is_err()
            {
                client.post_no_memory();
            }
        },
    )?;

    install_compositor_handler(server, record)?;
    install_manager_handler(server, record)?;
    install_timeline_handler(server)?;
    install_syncobj_surface_handler(server, record)?;
    install_surface_handler(server, record)?;
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

fn install_manager_handler(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let manager_record = Rc::clone(record);
    server.add_request_handler(
        &DRM_SYNCOBJ_MANAGER_INTERFACE,
        move |client, _, sender, opcode, args| {
            let version = client.resource_version(sender).unwrap_or(1);
            match opcode {
                DRM_SYNCOBJ_MANAGER_GET_SURFACE => {
                    let (Some(WlArgument::NewId(new_id)), Some(WlArgument::Object(surface))) =
                        (args.first(), args.get(1))
                    else {
                        client.post_error(
                            sender,
                            WlDisplayError::InvalidMethod.code(),
                            "get_surface expects a new_id and a surface",
                        );
                        return;
                    };
                    let (new_id, surface) = (*new_id, *surface);
                    let mut record = manager_record.borrow_mut();
                    if record
                        .syncobj_surfaces
                        .iter()
                        .any(|(_, wl_surface)| *wl_surface == surface)
                    {
                        client.post_error(
                            sender,
                            DrmSyncobjManagerError::SurfaceExists.code(),
                            "the surface already has a synchronization object associated",
                        );
                        return;
                    }
                    let created = version.min(DRM_SYNCOBJ_SURFACE_INTERFACE.version);
                    if let Err(error) =
                        client.create_resource(new_id, &DRM_SYNCOBJ_SURFACE_INTERFACE, created)
                    {
                        client.post_error(
                            sender,
                            WlDisplayError::NoMemory.code(),
                            format!("get_surface failed: {error}"),
                        );
                        return;
                    }
                    record.syncobj_surfaces.push((new_id, surface));
                }
                DRM_SYNCOBJ_MANAGER_IMPORT_TIMELINE => {
                    let Some(WlArgument::NewId(new_id)) = args.first() else {
                        client.post_error(
                            sender,
                            WlDisplayError::InvalidMethod.code(),
                            "import_timeline expects a new_id argument",
                        );
                        return;
                    };
                    let new_id = *new_id;
                    let created = version.min(DRM_SYNCOBJ_TIMELINE_INTERFACE.version);
                    if let Err(error) =
                        client.create_resource(new_id, &DRM_SYNCOBJ_TIMELINE_INTERFACE, created)
                    {
                        client.post_error(
                            sender,
                            WlDisplayError::NoMemory.code(),
                            format!("import_timeline failed: {error}"),
                        );
                        return;
                    }
                    let Some(fd) = args.get_mut(1).and_then(WlArgument::take_fd) else {
                        client.post_error(
                            sender,
                            WlDisplayError::InvalidMethod.code(),
                            "import_timeline expects a descriptor argument",
                        );
                        return;
                    };
                    // Safety: the descriptor was demarshalled from the request
                    // above and is owned by the server since `take_fd` claimed
                    // it, so wrapping it in a `File` closes it exactly once.
                    let mut imported = unsafe { File::from_raw_fd(fd) };
                    let mut bytes = Vec::new();
                    let mut record = manager_record.borrow_mut();
                    if imported.read_to_end(&mut bytes).is_ok() {
                        record.fd_bytes = bytes;
                    }
                    record.timelines.push((new_id, created));
                }
                DRM_SYNCOBJ_MANAGER_DESTROY => {
                    let _ = client.destroy_resource(sender);
                }
                _ => client.post_error(
                    sender,
                    WlDisplayError::InvalidMethod.code(),
                    "unsupported wp_linux_drm_syncobj_manager_v1 request",
                ),
            }
        },
    )
}

fn install_timeline_handler(server: &mut TestServer) -> WlResult<()> {
    server.add_request_handler(
        &DRM_SYNCOBJ_TIMELINE_INTERFACE,
        move |client, _, sender, opcode, _| match opcode {
            DRM_SYNCOBJ_TIMELINE_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported wp_linux_drm_syncobj_timeline_v1 request",
            ),
        },
    )
}

fn install_syncobj_surface_handler(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let syncobj_record = Rc::clone(record);
    server.add_request_handler(
        &DRM_SYNCOBJ_SURFACE_INTERFACE,
        move |client, _, sender, opcode, args| match opcode {
            DRM_SYNCOBJ_SURFACE_SET_ACQUIRE_POINT | DRM_SYNCOBJ_SURFACE_SET_RELEASE_POINT => {
                let (
                    Some(WlArgument::Object(timeline)),
                    Some(WlArgument::Uint(point_hi)),
                    Some(WlArgument::Uint(point_lo)),
                ) = (args.first(), args.get(1), args.get(2))
                else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "set points expects a timeline and two uints",
                    );
                    return;
                };
                let point = (u64::from(*point_hi) << 32) | u64::from(*point_lo);
                let mut record = syncobj_record.borrow_mut();
                if opcode == DRM_SYNCOBJ_SURFACE_SET_ACQUIRE_POINT {
                    record.acquire_points.push((*timeline, point));
                } else {
                    record.release_points.push((*timeline, point));
                }
            }
            DRM_SYNCOBJ_SURFACE_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported wp_linux_drm_syncobj_surface_v1 request",
            ),
        },
    )
}

fn install_surface_handler(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let surface_record = Rc::clone(record);
    server.add_request_handler(
        &SURFACE_INTERFACE,
        move |client, _, sender, opcode, _| match opcode {
            SURFACE_COMMIT => {
                let mut record = surface_record.borrow_mut();
                record.commits += 1;
                // The client never attaches a buffer, so a commit carrying
                // pending timeline points must raise `no_buffer`.
                let syncobj = record
                    .syncobj_surfaces
                    .iter()
                    .find(|(_, wl_surface)| *wl_surface == sender)
                    .map(|(syncobj, _)| *syncobj);
                let pending_points = !record.acquire_points.is_empty() && !record.release_points.is_empty();
                if let Some(syncobj) = syncobj
                    && pending_points
                {
                    client.post_error(
                        syncobj,
                        DrmSyncobjSurfaceError::NoBuffer.code(),
                        "no buffer attached while timeline points are pending",
                    );
                }
            }
            SURFACE_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported wl_surface request",
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

#[test]
fn syncobj_timeline_points_and_fd_round_trip() {
    let record = Rc::new(RefCell::new(Record::default()));
    let mut fixture = Fixture::new();
    scripted_session(&mut fixture.server, &record).unwrap();

    let registry = fixture.client.get_registry().unwrap();
    let globals = collect_globals(&mut fixture.client, registry);
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    let (compositor_name, _) = global_named(&globals.borrow(), "wl_compositor");
    let (manager_name, manager_version) = global_named(&globals.borrow(), "wp_linux_drm_syncobj_manager_v1");
    assert_eq!(manager_version, 1);

    let compositor = fixture
        .client
        .registry_bind(registry, compositor_name, &COMPOSITOR_INTERFACE, 7)
        .unwrap();
    let manager = fixture
        .client
        .registry_bind(
            registry,
            manager_name,
            &DRM_SYNCOBJ_MANAGER_INTERFACE,
            manager_version,
        )
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    let surface = fixture
        .client
        .marshal_new_id(compositor, COMPOSITOR_CREATE_SURFACE, Vec::new())
        .unwrap();

    // A memfd holding a recognizable payload stands in for the syncobj
    // timeline the client would import from the GPU driver.
    let payload: &[u8] = b"codevar syncobj timeline";
    let fd = unsafe { libc::memfd_create(c"codevar-syncobj".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0, "memfd_create must provide the timeline descriptor");
    // Safety: `fd` is a fresh descriptor owned by this test.
    let mut timeline_fd = unsafe { File::from_raw_fd(fd) };
    timeline_fd.write_all(payload).unwrap();
    timeline_fd.rewind().unwrap();

    let timeline = fixture
        .client
        .marshal_new_id(
            manager,
            DRM_SYNCOBJ_MANAGER_IMPORT_TIMELINE,
            vec![WlArgument::Fd(timeline_fd.as_raw_fd())],
        )
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();

    {
        let record = record.borrow();
        assert_eq!(record.surfaces, [surface.id()]);
        assert_eq!(record.timelines, [(timeline.id(), 1)]);
        assert_eq!(record.fd_bytes.as_slice(), payload);
    }

    let syncobj_surface = fixture
        .client
        .marshal_new_id(
            manager,
            DRM_SYNCOBJ_MANAGER_GET_SURFACE,
            vec![WlArgument::Object(surface.id())],
        )
        .unwrap();
    fixture
        .client
        .marshal_request(
            syncobj_surface,
            DRM_SYNCOBJ_SURFACE_SET_ACQUIRE_POINT,
            vec![
                WlArgument::Object(timeline.id()),
                WlArgument::Uint(1),
                WlArgument::Uint(0xDEAD_BEEF),
            ],
        )
        .unwrap();
    fixture
        .client
        .marshal_request(
            syncobj_surface,
            DRM_SYNCOBJ_SURFACE_SET_RELEASE_POINT,
            vec![
                WlArgument::Object(timeline.id()),
                WlArgument::Uint(1),
                WlArgument::Uint(0xDEAD_F00D),
            ],
        )
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    assert!(fixture.client.protocol_error().is_none());
    assert!(fixture.client.is_alive(surface));
    assert!(fixture.client.is_alive(timeline));
    assert!(fixture.client.is_alive(syncobj_surface));

    {
        let record = record.borrow();
        assert_eq!(record.syncobj_surfaces, [(syncobj_surface.id(), surface.id())]);
        assert_eq!(
            record.acquire_points,
            [(timeline.id(), (1u64 << 32) | 0xDEAD_BEEF)]
        );
        assert_eq!(
            record.release_points,
            [(timeline.id(), (1u64 << 32) | 0xDEAD_F00D)]
        );
        assert_eq!(record.commits, 0);
    }

    let (_, server_client) = fixture.server.clients().next().unwrap();
    assert_eq!(
        server_client
            .resource_interface(manager.id())
            .map(|interface| interface.name),
        Some("wp_linux_drm_syncobj_manager_v1")
    );
    assert_eq!(
        server_client
            .resource_interface(timeline.id())
            .map(|interface| interface.name),
        Some("wp_linux_drm_syncobj_timeline_v1")
    );
    assert_eq!(server_client.resource_version(timeline.id()), Some(1));
    assert_eq!(
        server_client
            .resource_interface(syncobj_surface.id())
            .map(|interface| interface.name),
        Some("wp_linux_drm_syncobj_surface_v1")
    );
    assert_eq!(server_client.resource_version(syncobj_surface.id()), Some(1));

    // The commit carries pending points but no buffer: the compositor
    // answers with the no_buffer error of the synchronization surface.
    fixture
        .client
        .marshal_request(surface, SURFACE_COMMIT, Vec::new())
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    let dispatch_error = fixture
        .client
        .dispatch(Some(Duration::ZERO))
        .expect_err("the compositor must report the missing buffer");
    let WlError::Protocol(error) = dispatch_error else {
        panic!("dispatch failed with {dispatch_error}");
    };
    assert_eq!(error.code, DrmSyncobjSurfaceError::NoBuffer.code());
    assert_eq!(error.code, 3);
    assert_eq!(error.object_id, syncobj_surface.id());
    assert_eq!(error.interface, "wp_linux_drm_syncobj_surface_v1");
    assert!(error.message.contains("no buffer"));
    assert_eq!(
        fixture
            .client
            .protocol_error()
            .map(|error| error.code),
        Some(3)
    );
    assert_eq!(record.borrow().commits, 1);
}
