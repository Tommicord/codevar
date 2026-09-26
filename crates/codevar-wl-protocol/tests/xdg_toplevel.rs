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

//! End-to-end test for the `xdg-shell` toplevel handshake over a real
//! `socketpair`.
//!
//! A scripted compositor answers `get_xdg_surface` with a
//! `xdg_surface.configure`, pings the `xdg_wm_base` global when it is
//! bound and observes the client's `ack_configure`, title and initial
//! commit. The client answers the ping, records the configure serial
//! and closes the loop by acking it.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Duration;

use codevar_wl_protocol::{
    COMPOSITOR_CREATE_SURFACE, COMPOSITOR_INTERFACE, SURFACE_COMMIT, SURFACE_DESTROY, SURFACE_INTERFACE,
    WlArgument, WlClientDisplay, WlClock, WlDisplayError, WlProxyId, WlRegistryEvent, WlResult,
    WlServerDisplay, WlUnixPoller, WlUnixTransport, XDG_SURFACE_ACK_CONFIGURE, XDG_SURFACE_CONFIGURE,
    XDG_SURFACE_DESTROY, XDG_SURFACE_GET_TOPLEVEL, XDG_SURFACE_INTERFACE, XDG_TOPLEVEL_CLOSE,
    XDG_TOPLEVEL_DESTROY, XDG_TOPLEVEL_INTERFACE, XDG_TOPLEVEL_SET_APP_ID, XDG_TOPLEVEL_SET_TITLE,
    XDG_WM_BASE_CREATE_POSITIONER, XDG_WM_BASE_GET_XDG_SURFACE, XDG_WM_BASE_INTERFACE, XDG_WM_BASE_PING,
    XDG_WM_BASE_PONG,
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
    serial: u32,
    surfaces: Vec<u32>,
    xdg_surfaces: Vec<(u32, u32)>,
    positioners: Vec<u32>,
    toplevels: Vec<u32>,
    pongs: Vec<u32>,
    configures_sent: Vec<u32>,
    acks: Vec<u32>,
    titles: Vec<String>,
    app_ids: Vec<String>,
    commits: usize,
}

impl Record {
    fn next_serial(&mut self) -> u32 {
        self.serial += 1;
        self.serial
    }
}

/// Registers the compositor and `xdg_wm_base` globals with handlers.
fn scripted_session(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    server.add_global(&COMPOSITOR_INTERFACE, COMPOSITOR_INTERFACE.version)?;

    let ping_record = Rc::clone(record);
    server.add_global_with(
        &XDG_WM_BASE_INTERFACE,
        XDG_WM_BASE_INTERFACE.version,
        move |client, _, version, id| {
            if client
                .create_resource(id, &XDG_WM_BASE_INTERFACE, version)
                .is_err()
            {
                client.post_no_memory();
                return;
            }
            let serial = ping_record.borrow_mut().next_serial();
            if client
                .post_event(id, XDG_WM_BASE_PING, vec![WlArgument::Uint(serial)])
                .is_err()
            {
                client.post_no_memory();
            }
        },
    )?;

    install_compositor_handler(server, record)?;
    install_wm_base_handler(server, record)?;
    install_surface_handlers(server, record)?;
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

fn install_wm_base_handler(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let record = Rc::clone(record);
    server.add_request_handler(&XDG_WM_BASE_INTERFACE, move |client, _, sender, opcode, args| {
        let version = client.resource_version(sender).unwrap_or(1);
        match opcode {
            XDG_WM_BASE_GET_XDG_SURFACE => {
                let (Some(WlArgument::NewId(new_id)), Some(WlArgument::Object(surface))) =
                    (args.first(), args.get(1))
                else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "get_xdg_surface expects a new_id and a surface",
                    );
                    return;
                };
                let (new_id, surface) = (*new_id, *surface);
                let created = version.min(XDG_SURFACE_INTERFACE.version);
                if let Err(error) = client.create_resource(new_id, &XDG_SURFACE_INTERFACE, created) {
                    client.post_error(
                        sender,
                        WlDisplayError::NoMemory.code(),
                        format!("get_xdg_surface failed: {error}"),
                    );
                    return;
                }
                record
                    .borrow_mut()
                    .xdg_surfaces
                    .push((new_id, surface));
            }
            XDG_WM_BASE_CREATE_POSITIONER => {
                let Some(WlArgument::NewId(new_id)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "create_positioner expects a new_id argument",
                    );
                    return;
                };
                let new_id = *new_id;
                if let Err(error) =
                    client.create_resource(new_id, &codevar_wl_protocol::XDG_POSITIONER_INTERFACE, version)
                {
                    client.post_error(
                        sender,
                        WlDisplayError::NoMemory.code(),
                        format!("create_positioner failed: {error}"),
                    );
                    return;
                }
                record.borrow_mut().positioners.push(new_id);
            }
            XDG_WM_BASE_PONG => {
                let Some(WlArgument::Uint(serial)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "pong expects a serial argument",
                    );
                    return;
                };
                record.borrow_mut().pongs.push(*serial);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported xdg_wm_base request",
            ),
        }
    })
}

fn install_surface_handlers(server: &mut TestServer, record: &Rc<RefCell<Record>>) -> WlResult<()> {
    let xdg_record = Rc::clone(record);
    server.add_request_handler(
        &XDG_SURFACE_INTERFACE,
        move |client, _, sender, opcode, args| match opcode {
            XDG_SURFACE_GET_TOPLEVEL => {
                let Some(WlArgument::NewId(new_id)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "get_toplevel expects a new_id argument",
                    );
                    return;
                };
                let new_id = *new_id;
                let version = client
                    .resource_version(sender)
                    .unwrap_or(1)
                    .min(XDG_TOPLEVEL_INTERFACE.version);
                if let Err(error) = client.create_resource(new_id, &XDG_TOPLEVEL_INTERFACE, version) {
                    client.post_error(
                        sender,
                        WlDisplayError::NoMemory.code(),
                        format!("get_toplevel failed: {error}"),
                    );
                    return;
                }
                let serial = {
                    let mut record = xdg_record.borrow_mut();
                    record.toplevels.push(new_id);
                    record.next_serial()
                };
                if client
                    .post_event(sender, XDG_SURFACE_CONFIGURE, vec![WlArgument::Uint(serial)])
                    .is_err()
                {
                    client.post_no_memory();
                }
                xdg_record
                    .borrow_mut()
                    .configures_sent
                    .push(serial);
            }
            XDG_SURFACE_ACK_CONFIGURE => {
                let Some(WlArgument::Uint(serial)) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "ack_configure expects a serial argument",
                    );
                    return;
                };
                xdg_record.borrow_mut().acks.push(*serial);
                let toplevel = xdg_record.borrow().toplevels.first().copied();
                if let Some(toplevel) = toplevel
                    && client
                        .post_event(toplevel, XDG_TOPLEVEL_CLOSE, vec![])
                        .is_err()
                {
                    client.post_no_memory();
                }
            }
            XDG_SURFACE_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported xdg_surface request",
            ),
        },
    )?;

    let toplevel_record = Rc::clone(record);
    server.add_request_handler(
        &XDG_TOPLEVEL_INTERFACE,
        move |client, _, sender, opcode, args| match opcode {
            XDG_TOPLEVEL_SET_TITLE => {
                let Some(WlArgument::Str(Some(title))) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "set_title expects a string argument",
                    );
                    return;
                };
                toplevel_record
                    .borrow_mut()
                    .titles
                    .push(title.clone());
            }
            XDG_TOPLEVEL_SET_APP_ID => {
                let Some(WlArgument::Str(Some(app_id))) = args.first() else {
                    client.post_error(
                        sender,
                        WlDisplayError::InvalidMethod.code(),
                        "set_app_id expects a string argument",
                    );
                    return;
                };
                toplevel_record
                    .borrow_mut()
                    .app_ids
                    .push(app_id.clone());
            }
            XDG_TOPLEVEL_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported xdg_toplevel request",
            ),
        },
    )?;

    let wl_record = Rc::clone(record);
    server.add_request_handler(
        &SURFACE_INTERFACE,
        move |client, _, sender, opcode, _| match opcode {
            SURFACE_COMMIT => wl_record.borrow_mut().commits += 1,
            SURFACE_DESTROY => {
                let _ = client.destroy_resource(sender);
            }
            _ => client.post_error(
                sender,
                WlDisplayError::InvalidMethod.code(),
                "unsupported wl_surface request",
            ),
        },
    )?;
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
fn toplevel_handshake_completes_over_a_real_socket() {
    let record = Rc::new(RefCell::new(Record::default()));
    let mut fixture = Fixture::new();
    scripted_session(&mut fixture.server, &record).unwrap();

    let registry = fixture.client.get_registry().unwrap();
    let globals = collect_globals(&mut fixture.client, registry);
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    let (compositor_name, _) = global_named(&globals.borrow(), "wl_compositor");
    let (wm_base_name, _) = global_named(&globals.borrow(), "xdg_wm_base");
    let compositor = fixture
        .client
        .registry_bind(registry, compositor_name, &COMPOSITOR_INTERFACE, 7)
        .unwrap();
    let wm_base = fixture
        .client
        .registry_bind(registry, wm_base_name, &XDG_WM_BASE_INTERFACE, 7)
        .unwrap();

    // Answer the ping the compositor sends when the global is bound.
    let ping_serials = Rc::new(RefCell::new(Vec::new()));
    {
        let ping_serials = Rc::clone(&ping_serials);
        fixture
            .client
            .add_listener(wm_base, move |client, opcode, args| {
                assert_eq!(opcode, XDG_WM_BASE_PING);
                if let Some(WlArgument::Uint(serial)) = args.first() {
                    let serial = *serial;
                    ping_serials.borrow_mut().push(serial);
                    client
                        .marshal_request(wm_base, XDG_WM_BASE_PONG, vec![WlArgument::Uint(serial)])
                        .unwrap();
                }
                0
            })
            .unwrap();
    }
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();
    fixture.flush_client();
    fixture.dispatch_server();

    let surface = fixture
        .client
        .marshal_new_id(compositor, COMPOSITOR_CREATE_SURFACE, Vec::new())
        .unwrap();
    let xdg_surface = fixture
        .client
        .marshal_new_id(
            wm_base,
            XDG_WM_BASE_GET_XDG_SURFACE,
            vec![WlArgument::Object(surface.id())],
        )
        .unwrap();

    let configure_serial = Rc::new(Cell::new(None));
    {
        let configure_serial = Rc::clone(&configure_serial);
        fixture
            .client
            .add_listener(xdg_surface, move |_, opcode, args| {
                assert_eq!(opcode, XDG_SURFACE_CONFIGURE);
                if let Some(WlArgument::Uint(serial)) = args.first() {
                    configure_serial.set(Some(*serial));
                }
                0
            })
            .unwrap();
    }

    let closed = Rc::new(Cell::new(false));
    let toplevel = fixture
        .client
        .marshal_new_id(xdg_surface, XDG_SURFACE_GET_TOPLEVEL, Vec::new())
        .unwrap();
    {
        let closed = Rc::clone(&closed);
        fixture
            .client
            .add_listener(toplevel, move |_, opcode, _| {
                assert_eq!(opcode, XDG_TOPLEVEL_CLOSE);
                closed.set(true);
                0
            })
            .unwrap();
    }

    fixture
        .client
        .marshal_request(
            toplevel,
            XDG_TOPLEVEL_SET_TITLE,
            vec![WlArgument::Str(Some(String::from("Codevar handshake")))],
        )
        .unwrap();
    fixture
        .client
        .marshal_request(
            toplevel,
            XDG_TOPLEVEL_SET_APP_ID,
            vec![WlArgument::Str(Some(String::from("dev.codevar.hello")))],
        )
        .unwrap();

    // The first commit asks the compositor for the initial configure.
    fixture
        .client
        .marshal_request(surface, SURFACE_COMMIT, Vec::new())
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    let serial = configure_serial
        .get()
        .unwrap_or_else(|| panic!("the compositor sent no configure"));
    fixture
        .client
        .marshal_request(
            xdg_surface,
            XDG_SURFACE_ACK_CONFIGURE,
            vec![WlArgument::Uint(serial)],
        )
        .unwrap();
    fixture.flush_client();
    fixture.dispatch_server();
    fixture.dispatch_client();

    {
        let record = record.borrow();
        assert_eq!(record.surfaces, [surface.id()]);
        assert_eq!(record.xdg_surfaces, [(xdg_surface.id(), surface.id())]);
        assert_eq!(record.toplevels, [toplevel.id()]);
        assert_eq!(record.configures_sent, [serial]);
        assert_eq!(record.acks, [serial]);
        assert_eq!(record.pongs, *ping_serials.borrow());
        assert!(!record.pongs.is_empty());
        assert_eq!(record.titles, ["Codevar handshake"]);
        assert_eq!(record.app_ids, ["dev.codevar.hello"]);
        assert_eq!(record.commits, 1);
    }

    let (_, server_client) = fixture.server.clients().next().unwrap();
    assert_eq!(
        server_client
            .resource_interface(xdg_surface.id())
            .map(|interface| interface.name),
        Some("xdg_surface")
    );
    assert_eq!(
        server_client
            .resource_interface(toplevel.id())
            .map(|interface| interface.name),
        Some("xdg_toplevel")
    );
    assert_eq!(server_client.resource_version(toplevel.id()), Some(7));

    // The scripted compositor closes the window after the ack.
    assert!(closed.get());
}
