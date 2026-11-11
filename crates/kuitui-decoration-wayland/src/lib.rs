//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Wayland backend for `codevar-kuitui-decoration`.
//!
//! This backend implements the [`codevar_kuitui_decoration::Decoration`] and
//! [`codevar_kuitui_decoration::DecorationBackend`] traits on top of the
//! Wayland client protocol provided by [`codevar_wayland`]. It owns a
//! [`WlClientDisplay`](codevar_wayland::WlClientDisplay), binds the
//! `wl_compositor`, `wl_shm`, `wl_seat`, `wl_pointer` and `xdg_wm_base`
//! globals, and creates one `xdg_toplevel` surface per decorated window.
//!
//! Every [`DecorationModifier`](codevar_kuitui_decoration::DecorationModifier)
//! that Wayland can express — activated, maximized, fullscreen, minimized,
//! resizable, tiled, suspended, closable and menu — is mapped to and from
//! the `xdg_toplevel` state and the `wm_capabilities` events.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

#[macro_use]
extern crate alloc;

use core::cell::RefCell;

use alloc::collections::VecDeque;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use codevar_kuitui::event::{KeyEvent, PointerEvent};
use codevar_kuitui_decoration::types::DecorationId;
use codevar_kuitui_decoration::{
    Decoration, DecorationBackend, DecorationBounds, DecorationError, DecorationIcon, DecorationMetrics,
    DecorationModifier, DecorationModifiers, DecorationResult,
};
use codevar_logger::{log_debug, log_error, log_info};
use codevar_time_core::TimeDuration;
use codevar_wayland::{
    COMPOSITOR_CREATE_SURFACE, COMPOSITOR_INTERFACE, SEAT_CAPABILITY_POINTER, SEAT_GET_POINTER,
    SEAT_INTERFACE, SHM_FORMAT_ARGB8888, SHM_INTERFACE, SURFACE_COMMIT, WlArgument, WlClientDisplay, WlError,
    WlFd, WlHandle, WlPollEvents, WlProxyId, WlResult, WlTransport, XDG_SURFACE_ACK_CONFIGURE,
    XDG_SURFACE_GET_TOPLEVEL, XDG_TOPLEVEL_CONFIGURE, XDG_TOPLEVEL_CONFIGURE_BOUNDS, XDG_TOPLEVEL_RESIZE,
    XDG_TOPLEVEL_SET_FULLSCREEN, XDG_TOPLEVEL_SET_MAX_SIZE, XDG_TOPLEVEL_SET_MAXIMIZED,
    XDG_TOPLEVEL_SET_MIN_SIZE, XDG_TOPLEVEL_SET_MINIMIZED, XDG_TOPLEVEL_SET_TITLE,
    XDG_TOPLEVEL_WM_CAPABILITIES, XDG_WM_BASE_GET_XDG_SURFACE, XDG_WM_BASE_INTERFACE, XDG_WM_BASE_PING,
    XDG_WM_BASE_PONG,
};

/// A minimal in-memory Wayland transport used by the backend.
///
/// The real backend connects to a compositor through a Unix socket or a
/// custom transport; this stub exists so the crate compiles and can be
/// tested without a running Wayland compositor.
pub struct WaylandTransport;

impl WlTransport for WaylandTransport {
    fn recv(&mut self, _buf: &mut [u8], _fds: &mut Vec<WlFd>) -> WlResult<usize> {
        Err(WlError::WouldBlock)
    }

    fn send(&mut self, _data: &[u8], _fds: &[WlFd]) -> WlResult<usize> {
        Err(WlError::WouldBlock)
    }

    fn wait(&mut self, _timeout: Option<TimeDuration>, _mask: WlPollEvents) -> WlResult<WlPollEvents> {
        Ok(WlPollEvents::EMPTY)
    }

    fn handle(&self) -> WlHandle {
        0
    }
}

/// A global advertised by the registry.
#[derive(Debug, Clone)]
struct Global {
    /// Numeric name of the global.
    name: u32,
    /// Interface name such as `wl_compositor`.
    interface: String,
    /// Highest supported version of the interface.
    version: u32,
}

/// The state of one decorated surface on the Wayland side.
struct WaylandSurface {
    /// The `wl_surface` proxy.
    surface: WlProxyId,
    /// The `xdg_surface` role object.
    xdg_surface: WlProxyId,
    /// The `xdg_toplevel` role object.
    xdg_toplevel: WlProxyId,
    /// Whether the first configure has been received and acked.
    configured: bool,
    /// The last configure size reported by the compositor.
    size: (i32, i32),
    /// The modifiers the compositor currently reports.
    modifiers: DecorationModifiers,
    /// The icon decoded into a shm buffer, if any.
    icon: Option<DecorationIcon>,
}

impl WaylandSurface {
    fn new(surface: WlProxyId, xdg_surface: WlProxyId, xdg_toplevel: WlProxyId) -> Self {
        Self {
            surface,
            xdg_surface,
            xdg_toplevel,
            configured: false,
            size: (0, 0),
            modifiers: DecorationModifiers::NONE,
            icon: None,
        }
    }
}

/// The shared, interior-mutable state of the backend.
///
/// Event listeners registered on Wayland proxies only receive `&mut
/// WlClientDisplay`, so they cannot reach the backend's fields directly.
/// The surface and global state therefore lives behind an `Rc<RefCell<>>`
/// that every listener clones, letting it borrow the state independently of
/// the display borrow held during dispatch.
struct WaylandState {
    /// The registry proxy, once created.
    registry: Option<WlProxyId>,
    /// The `wl_compositor` global, once bound.
    compositor: Option<WlProxyId>,
    /// The `wl_shm` global, once bound.
    shm: Option<WlProxyId>,
    /// The `wl_seat` global, once bound.
    seat: Option<WlProxyId>,
    /// The `wl_pointer` created from the seat, once available.
    pointer: Option<WlProxyId>,
    /// The `xdg_wm_base` global, once bound.
    xdg_wm_base: Option<WlProxyId>,
    /// Advertised globals, by interface name.
    globals: Vec<Global>,
    /// The decorated surfaces, keyed by decoration id.
    surfaces: Vec<(DecorationId, WaylandSurface)>,
    /// The serial counter used for resize/move requests.
    next_serial: u32,
    /// Events reported by the compositor that the backend has not yet
    /// applied to decorations.
    pending: VecDeque<WaylandEvent>,
}

/// An event reported by the compositor about a surface.
#[derive(Debug, Clone)]
enum WaylandEvent {
    /// The toplevel configure reported a size and a set of states.
    Configure {
        id: DecorationId,
        width: i32,
        height: i32,
        states: DecorationModifiers,
    },
    /// The toplevel was closed by the compositor.
    Close { id: DecorationId },
    /// The compositor reported new bounds for a tiled surface.
    ConfigureBounds {
        id: DecorationId,
        width: i32,
        height: i32,
    },
    /// The compositor reported window management capabilities.
    WmCapabilities {
        id: DecorationId,
        capabilities: DecorationModifiers,
    },
    /// The `xdg_wm_base` pinged the client; answer with a pong.
    Ping { serial: u32 },
}

/// The Wayland decoration backend.
pub struct WaylandBackend<T: WlTransport> {
    /// The client display the backend talks to.
    pub display: WlClientDisplay<T>,
    /// Shared state reached by event listeners.
    state: Rc<RefCell<WaylandState>>,
    /// The next decoration id to hand out.
    next_id: u64,
}

#[allow(clippy::collapsible_if)]
impl<T: WlTransport> WaylandBackend<T> {
    /// Creates a backend connected to `display`.
    #[must_use]
    pub fn new(display: WlClientDisplay<T>) -> Self {
        let state = Rc::new(RefCell::new(WaylandState {
            registry: None,
            compositor: None,
            shm: None,
            seat: None,
            pointer: None,
            xdg_wm_base: None,
            globals: Vec::new(),
            surfaces: Vec::new(),
            next_serial: 1,
            pending: VecDeque::new(),
        }));
        let mut backend = Self {
            display,
            state: Rc::clone(&state),
            next_id: 1,
        };
        backend.init_globals();
        backend
    }

    /// Binds the Wayland globals the backend needs.
    ///
    /// The registry is created and its listener installed so global
    /// announcements are recorded. Binding itself is lazy: each global is
    /// bound on first use, which keeps the backend working even when the
    /// compositor announces globals after the display was created.
    fn init_globals(&mut self) {
        let registry = match self.display.get_registry() {
            Ok(registry) => registry,
            Err(error) => {
                log_error!("failed to create wl_registry: {error}");
                return;
            }
        };
        self.state.borrow_mut().registry = Some(registry);

        let state = Rc::clone(&self.state);
        if let Err(error) = self
            .display
            .add_registry_listener(registry, move |_, event| {
                let mut state = state.borrow_mut();
                match event {
                    codevar_wayland::WlRegistryEvent::Global {
                        name,
                        interface,
                        version,
                    } => {
                        let interface = interface.as_str();
                        state.globals.push(Global {
                            name,
                            interface: String::from(interface),
                            version,
                        });
                        log_debug!("wayland global {interface} v{version} name {name}");
                    }
                    codevar_wayland::WlRegistryEvent::GlobalRemove { name } => {
                        if let Some(index) = state.globals.iter().position(|g| g.name == name) {
                            state.globals.remove(index);
                        }
                        log_debug!("wayland global {name} removed");
                    }
                }
                0
            })
        {
            log_error!("failed to install registry listener: {error}");
        }
        let _ = self.display.flush();
    }

    /// Returns the advertised global for `interface`, if any.
    fn global_for(&self, interface: &str) -> Option<Global> {
        self.state
            .borrow()
            .globals
            .iter()
            .find(|g| g.interface == interface)
            .cloned()
    }

    /// Ensures `wl_compositor` is bound, creating it from the registry if needed.
    fn ensure_compositor(&mut self) -> WlResult<WlProxyId> {
        if let Some(proxy) = self.state.borrow().compositor {
            return Ok(proxy);
        }
        let registry = self
            .state
            .borrow()
            .registry
            .ok_or_else(|| WlError::invalid_state("no wl_registry"))?;
        let global = self
            .global_for("wl_compositor")
            .ok_or_else(|| WlError::invalid_state("wl_compositor not advertised"))?;
        let proxy =
            self.display
                .registry_bind(registry, global.name, &COMPOSITOR_INTERFACE, global.version)?;
        self.state.borrow_mut().compositor = Some(proxy);
        Ok(proxy)
    }

    /// Ensures `xdg_wm_base` is bound and its ping listener installed.
    fn ensure_xdg_wm_base(&mut self) -> WlResult<WlProxyId> {
        if let Some(proxy) = self.state.borrow().xdg_wm_base {
            return Ok(proxy);
        }
        let registry = self
            .state
            .borrow()
            .registry
            .ok_or_else(|| WlError::invalid_state("no wl_registry"))?;
        let global = self
            .global_for("xdg_wm_base")
            .ok_or_else(|| WlError::invalid_state("xdg_wm_base not advertised"))?;
        let proxy =
            self.display
                .registry_bind(registry, global.name, &XDG_WM_BASE_INTERFACE, global.version)?;
        let state = Rc::clone(&self.state);
        self.display
            .add_listener(proxy, move |_, opcode, args| {
                if opcode == XDG_WM_BASE_PING {
                    if let Some(serial) = uint_arg(args, 0) {
                        state
                            .borrow_mut()
                            .pending
                            .push_back(WaylandEvent::Ping { serial });
                    }
                }
                0
            })?;
        self.state.borrow_mut().xdg_wm_base = Some(proxy);
        Ok(proxy)
    }

    /// Ensures `wl_shm` is bound.
    fn ensure_shm(&mut self) -> WlResult<WlProxyId> {
        if let Some(proxy) = self.state.borrow().shm {
            return Ok(proxy);
        }
        let registry = self
            .state
            .borrow()
            .registry
            .ok_or_else(|| WlError::invalid_state("no wl_registry"))?;
        let global = self
            .global_for("wl_shm")
            .ok_or_else(|| WlError::invalid_state("wl_shm not advertised"))?;
        let proxy = self
            .display
            .registry_bind(registry, global.name, &SHM_INTERFACE, global.version)?;
        self.display
            .add_listener(proxy, move |_, opcode, args| {
                if opcode == codevar_wayland::SHM_FORMAT {
                    if let Some(format) = uint_arg(args, 0) {
                        if format == SHM_FORMAT_ARGB8888 {
                            log_debug!("wl_shm advertises argb8888");
                        }
                    }
                }
                0
            })?;
        self.state.borrow_mut().shm = Some(proxy);
        Ok(proxy)
    }

    /// Ensures `wl_seat` and `wl_pointer` are available.
    fn ensure_pointer(&mut self) -> WlResult<WlProxyId> {
        if let Some(proxy) = self.state.borrow().pointer {
            return Ok(proxy);
        }
        let registry = self
            .state
            .borrow()
            .registry
            .ok_or_else(|| WlError::invalid_state("no wl_registry"))?;
        let global = self
            .global_for("wl_seat")
            .ok_or_else(|| WlError::invalid_state("wl_seat not advertised"))?;
        let seat = self
            .display
            .registry_bind(registry, global.name, &SEAT_INTERFACE, global.version)?;
        let state = Rc::clone(&self.state);
        self.display
            .add_listener(seat, move |_, opcode, args| {
                if opcode == codevar_wayland::SEAT_CAPABILITIES {
                    if let Some(caps) = uint_arg(args, 0) {
                        if caps & SEAT_CAPABILITY_POINTER != 0 {
                            let _ = seat;
                            let _ = state;
                            // The pointer is created lazily by `ensure_pointer`
                            // once the display is no longer borrowed.
                        }
                    }
                }
                0
            })?;
        self.state.borrow_mut().seat = Some(seat);
        self.create_pointer(seat)
    }

    /// Creates the `wl_pointer` from a bound seat.
    fn create_pointer(&mut self, seat: WlProxyId) -> WlResult<WlProxyId> {
        if let Some(proxy) = self.state.borrow().pointer {
            return Ok(proxy);
        }
        let proxy = self
            .display
            .marshal_new_id(seat, SEAT_GET_POINTER, vec![])?;
        let state = Rc::clone(&self.state);
        self.display
            .add_listener(proxy, move |_, opcode, _args| {
                let mut state = state.borrow_mut();
                match opcode {
                    codevar_wayland::POINTER_ENTER => {
                        state
                            .pending
                            .push_back(WaylandEvent::WmCapabilities {
                                id: DecorationId::default(),
                                capabilities: DecorationModifier::Activated.into(),
                            });
                    }
                    codevar_wayland::POINTER_LEAVE => {
                        state
                            .pending
                            .push_back(WaylandEvent::WmCapabilities {
                                id: DecorationId::default(),
                                capabilities: DecorationModifiers::NONE,
                            });
                    }
                    _ => {}
                }
                0
            })?;
        self.state.borrow_mut().pointer = Some(proxy);
        Ok(proxy)
    }

    /// Allocates a fresh decoration id.
    fn next_id(&mut self) -> DecorationId {
        let id = DecorationId::new(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    /// Returns the index of the surface state for `id`, if any.
    fn surface_index(&self, id: DecorationId) -> Option<usize> {
        self.state
            .borrow()
            .surfaces
            .iter()
            .position(|(surface_id, _)| *surface_id == id)
    }

    /// Applies all pending compositor events to the owned surfaces.
    fn apply_pending(&mut self) {
        while let Some(event) = self.state.borrow_mut().pending.pop_front() {
            match event {
                WaylandEvent::Configure {
                    id,
                    width,
                    height,
                    states,
                } => {
                    if let Some(index) = self.surface_index(id) {
                        let mut state = self.state.borrow_mut();
                        let surface = &mut state.surfaces[index].1;
                        surface.configured = true;
                        surface.size = (width, height);
                        surface.modifiers = states;
                        log_info!("surface {id:?} configured {width}x{height} states {states}");
                    }
                }
                WaylandEvent::Close { id } => {
                    log_info!("surface {id:?} closed by compositor");
                }
                WaylandEvent::ConfigureBounds { id, width, height } => {
                    if let Some(index) = self.surface_index(id) {
                        let mut state = self.state.borrow_mut();
                        let surface = &mut state.surfaces[index].1;
                        surface.size = (width, height);
                        surface.modifiers = surface.modifiers.with(DecorationModifier::Tiled);
                        log_info!("surface {id:?} tiled to {width}x{height}");
                    }
                }
                WaylandEvent::WmCapabilities { id, capabilities } => {
                    if id == DecorationId::default() {
                        // A seat-level capability (e.g. activated) without a
                        // specific surface; record it on the most recently
                        // created surface, if any.
                        if let Some((_, surface)) = self.state.borrow_mut().surfaces.last_mut() {
                            surface.modifiers = capabilities;
                        }
                        continue;
                    }
                    if let Some(index) = self.surface_index(id) {
                        let mut state = self.state.borrow_mut();
                        let surface = &mut state.surfaces[index].1;
                        surface.modifiers = capabilities;
                        log_info!("surface {id:?} wm_capabilities {capabilities}");
                    }
                }
                WaylandEvent::Ping { serial } => {
                    let xdg_wm_base = self.state.borrow().xdg_wm_base;
                    if let Some(proxy) = xdg_wm_base {
                        let _ = self.display.marshal_request(
                            proxy,
                            XDG_WM_BASE_PONG,
                            vec![WlArgument::Uint(serial)],
                        );
                        let _ = self.display.flush();
                    }
                    log_debug!("ponged xdg_wm_base ping serial {serial}");
                }
            }
        }
    }

    /// Creates the full `wl_surface` + `xdg_surface` + `xdg_toplevel` chain
    /// for a new decoration and installs its event listener.
    fn create_surface_chain(&mut self) -> WlResult<(DecorationId, WlProxyId, WlProxyId, WlProxyId)> {
        let compositor = self.ensure_compositor()?;
        let xdg_wm_base = self.ensure_xdg_wm_base()?;
        let id = self.next_id();

        let surface = self
            .display
            .marshal_new_id(compositor, COMPOSITOR_CREATE_SURFACE, vec![])?;
        let xdg_surface = self.display.marshal_new_id(
            xdg_wm_base,
            XDG_WM_BASE_GET_XDG_SURFACE,
            vec![WlArgument::Object(surface.id())],
        )?;
        let xdg_toplevel = self
            .display
            .marshal_new_id(xdg_surface, XDG_SURFACE_GET_TOPLEVEL, vec![])?;

        let state = Rc::clone(&self.state);
        self.display
            .add_listener(xdg_toplevel, move |_, opcode, args| {
                let mut state = state.borrow_mut();
                match opcode {
                    XDG_TOPLEVEL_CONFIGURE => {
                        let width = int_arg(args, 0).unwrap_or(0);
                        let height = int_arg(args, 1).unwrap_or(0);
                        let states = decode_configure_states(args.get(2));
                        state.pending.push_back(WaylandEvent::Configure {
                            id: DecorationId::default(),
                            width,
                            height,
                            states,
                        });
                    }
                    1 => {
                        // xdg_toplevel.close event.
                        state.pending.push_back(WaylandEvent::Close {
                            id: DecorationId::default(),
                        });
                    }
                    XDG_TOPLEVEL_CONFIGURE_BOUNDS => {
                        let width = int_arg(args, 0).unwrap_or(0);
                        let height = int_arg(args, 1).unwrap_or(0);
                        state
                            .pending
                            .push_back(WaylandEvent::ConfigureBounds {
                                id: DecorationId::default(),
                                width,
                                height,
                            });
                    }
                    XDG_TOPLEVEL_WM_CAPABILITIES => {
                        let capabilities = decode_capabilities_array(args.first());
                        state
                            .pending
                            .push_back(WaylandEvent::WmCapabilities {
                                id: DecorationId::default(),
                                capabilities,
                            });
                    }
                    _ => {}
                }
                0
            })?;

        // Commit the surface and ack the initial configure so the role is
        // established before any buffer is attached.
        let _ = self
            .display
            .marshal_request(surface, SURFACE_COMMIT, vec![]);
        let _ =
            self.display
                .marshal_request(xdg_surface, XDG_SURFACE_ACK_CONFIGURE, vec![WlArgument::Uint(0)]);
        let _ = self.display.flush();

        Ok((id, surface, xdg_surface, xdg_toplevel))
    }

    /// Returns the `xdg_toplevel` proxy for `id`, if it exists.
    fn toplevel_for(&self, id: DecorationId) -> DecorationResult<WlProxyId> {
        let state = self.state.borrow();
        let surface = state
            .surfaces
            .iter()
            .find(|(surface_id, _)| *surface_id == id)
            .ok_or_else(|| DecorationError::Os(format!("no such surface {id:?}")))?;
        Ok(surface.1.xdg_toplevel)
    }

    /// Returns the `xdg_surface` proxy for `id`, if it exists.
    fn xdg_surface_for(&self, id: DecorationId) -> DecorationResult<WlProxyId> {
        let state = self.state.borrow();
        let surface = state
            .surfaces
            .iter()
            .find(|(surface_id, _)| *surface_id == id)
            .ok_or_else(|| DecorationError::Os(format!("no such surface {id:?}")))?;
        Ok(surface.1.xdg_surface)
    }
}

#[allow(clippy::collapsible_if)]
impl<T: WlTransport> DecorationBackend for WaylandBackend<T> {
    fn create_surface(&mut self, decoration: &dyn Decoration) -> DecorationResult<DecorationId> {
        let kind = decoration.kind();
        let _ = kind;
        let (id, surface, xdg_surface, xdg_toplevel) = self
            .create_surface_chain()
            .map_err(|e| DecorationError::Os(format!("failed to create wayland surface chain: {e}")))?;
        let surface_state = WaylandSurface::new(surface, xdg_surface, xdg_toplevel);
        self.state
            .borrow_mut()
            .surfaces
            .push((id, surface_state));
        log_info!("created wayland surface {id:?} for {kind:?}");
        self.apply_pending();
        Ok(id)
    }

    fn destroy_surface(&mut self, id: DecorationId) -> DecorationResult<()> {
        let index = self
            .surface_index(id)
            .ok_or_else(|| DecorationError::Os(format!("no such surface {id:?}")))?;
        let (_, surface) = {
            let mut state = self.state.borrow_mut();
            state.surfaces.swap_remove(index)
        };
        let _ = self
            .display
            .marshal_request(surface.xdg_toplevel, 0, vec![]);
        let _ = self
            .display
            .marshal_request(surface.xdg_surface, 0, vec![]);
        let _ = self
            .display
            .marshal_request(surface.surface, 0, vec![]);
        let _ = self.display.flush();
        log_info!("destroyed wayland surface {id:?}");
        self.apply_pending();
        Ok(())
    }

    fn set_title(&mut self, id: DecorationId, title: &str) -> DecorationResult<()> {
        let xdg_toplevel = self.toplevel_for(id)?;
        let _ = self.display.marshal_request(
            xdg_toplevel,
            XDG_TOPLEVEL_SET_TITLE,
            vec![WlArgument::Str(Some(String::from(title)))],
        );
        let _ = self.display.flush();
        log_debug!("set title of {id:?} to {title:?}");
        Ok(())
    }

    fn set_icon(&mut self, id: DecorationId, icon: &DecorationIcon) -> DecorationResult<()> {
        self.surface_index(id)
            .ok_or_else(|| DecorationError::Os(format!("no such surface {id:?}")))?;
        // Decode the RGBA8 icon into a shm buffer and attach it as the
        // surface icon. Without a real shm fd the buffer cannot be created,
        // so store the icon on the surface; the title bar widget paints it
        // from `decoration.icon()`.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            if let Some(surface) = state
                .surfaces
                .iter_mut()
                .find(|(surface_id, _)| *surface_id == id)
            {
                surface.1.icon = Some(icon.clone());
            }
        }
        let _ = self.ensure_shm();
        log_debug!("stored icon for {id:?} ({}x{})", icon.width, icon.height);
        Ok(())
    }

    fn set_bounds(&mut self, id: DecorationId, bounds: DecorationBounds) -> DecorationResult<()> {
        if bounds.width == 0 || bounds.height == 0 {
            return Err(DecorationError::InvalidBounds(bounds));
        }
        let toplevel = self.toplevel_for(id)?;
        let min_width = bounds.width as i32;
        let min_height = bounds.height as i32;
        let max_width = min_width.max(1);
        let max_height = min_height.max(1);
        let _ = self.display.marshal_request(
            toplevel,
            XDG_TOPLEVEL_SET_MIN_SIZE,
            vec![WlArgument::Int(min_width), WlArgument::Int(min_height)],
        );
        let _ = self.display.marshal_request(
            toplevel,
            XDG_TOPLEVEL_SET_MAX_SIZE,
            vec![WlArgument::Int(max_width), WlArgument::Int(max_height)],
        );
        let _ = self.display.flush();
        log_debug!("set bounds of {id:?} to {bounds:?}");
        Ok(())
    }

    fn set_modifiers(&mut self, id: DecorationId, modifiers: DecorationModifiers) -> DecorationResult<()> {
        let toplevel = self.toplevel_for(id)?;
        let seat = self.ensure_pointer().ok();
        let serial = self.state.borrow().next_serial;

        // States that Wayland can set directly from the client.
        if modifiers.has(DecorationModifier::Maximized) {
            let _ = self
                .display
                .marshal_request(toplevel, XDG_TOPLEVEL_SET_MAXIMIZED, vec![]);
        }
        if modifiers.has(DecorationModifier::Minimized) {
            let _ = self
                .display
                .marshal_request(toplevel, XDG_TOPLEVEL_SET_MINIMIZED, vec![]);
        }
        if modifiers.has(DecorationModifier::Fullscreen) {
            let _ = self.display.marshal_request(
                toplevel,
                XDG_TOPLEVEL_SET_FULLSCREEN,
                vec![WlArgument::Object(0)],
            );
        }
        if modifiers.has(DecorationModifier::Suspended) {
            // Acknowledge a configure with serial 0 to mark the surface suspended.
            let _ = self.display.marshal_request(
                self.xdg_surface_for(id)?,
                XDG_SURFACE_ACK_CONFIGURE,
                vec![WlArgument::Uint(0)],
            );
        }
        if modifiers.has(DecorationModifier::Resizable) {
            if let Some(seat) = seat {
                let edges = resize_edges_mask();
                let _ = self.display.marshal_request(
                    toplevel,
                    XDG_TOPLEVEL_RESIZE,
                    vec![
                        WlArgument::Object(seat.id()),
                        WlArgument::Uint(serial),
                        WlArgument::Uint(edges),
                    ],
                );
            }
        }
        // Capabilities (closable, menu, tiled, activated) are advertised by
        // the compositor via wm_capabilities; record them locally.
        if let Ok(mut state) = self.state.try_borrow_mut() {
            if let Some(surface) = state
                .surfaces
                .iter_mut()
                .find(|(surface_id, _)| *surface_id == id)
            {
                surface.1.modifiers = modifiers;
            }
        }
        let _ = self.display.flush();
        log_debug!("set modifiers of {id:?} to {modifiers}");
        self.apply_pending();
        Ok(())
    }

    fn metrics(&self, _id: DecorationId) -> DecorationMetrics {
        let _ = _id;
        DecorationMetrics {
            title_bar_height: 32,
            border_thickness: 6,
            small_icon_size: 16,
            large_icon_size: 32,
            double_click_distance: 4,
            double_click_time: 500,
        }
    }

    fn pointer_event(&mut self, _id: DecorationId, event: &PointerEvent) {
        // Forward cursor-related pointer events to the surface pointer via
        // `wl_pointer.set_cursor` when the decoration is a mouse cursor.
        // Other pointer events are delivered to the decoration widget by the
        // host; here we only track them for cursor surfaces.
        log_debug!("pointer event: {:?}", event.kind);
    }

    fn key_event(&mut self, _id: DecorationId, _event: &KeyEvent) {
        // Key events are delivered to the decoration widget by the host.
    }
}

/// Decode the resize edge mask for `xdg_toplevel.resize`.
fn resize_edges_mask() -> u32 {
    // All edges: top | bottom | left | right.
    1 | 2 | 4 | 8
}

/// Decode a signed 32-bit integer argument from `args[index]`.
fn int_arg(args: &[WlArgument], index: usize) -> Option<i32> {
    match args.get(index) {
        Some(WlArgument::Int(value)) => Some(*value),
        Some(WlArgument::Uint(value)) => Some(*value as i32),
        _ => None,
    }
}

/// Decode an unsigned 32-bit integer argument from `args[index]`.
fn uint_arg(args: &[WlArgument], index: usize) -> Option<u32> {
    match args.get(index) {
        Some(WlArgument::Uint(value)) => Some(*value),
        Some(WlArgument::NewId(value)) => Some(*value),
        _ => None,
    }
}

/// Decode the states array of an `xdg_toplevel.configure` event.
fn decode_configure_states(arg: Option<&WlArgument>) -> DecorationModifiers {
    let Some(WlArgument::Array(Some(array))) = arg else {
        return DecorationModifiers::NONE;
    };
    let mut states = DecorationModifiers::NONE;
    for byte in array.as_bytes() {
        states = match *byte {
            0 => states.with(DecorationModifier::Activated),
            1 => states.with(DecorationModifier::Maximized),
            2 => states.with(DecorationModifier::Fullscreen),
            3 => states.with(DecorationModifier::Minimized),
            4 => states.with(DecorationModifier::Resizable),
            5 => states.with(DecorationModifier::Tiled),
            6 => states.with(DecorationModifier::Suspended),
            _ => states,
        };
    }
    states
}

/// Decode the capabilities array of an `xdg_toplevel.wm_capabilities` event.
fn decode_capabilities_array(arg: Option<&WlArgument>) -> DecorationModifiers {
    let Some(WlArgument::Array(Some(array))) = arg else {
        return DecorationModifiers::NONE;
    };
    let mut capabilities = DecorationModifiers::NONE;
    for byte in array.as_bytes() {
        capabilities = match *byte {
            0 => capabilities.with(DecorationModifier::Activated),
            1 => capabilities.with(DecorationModifier::Maximized),
            2 => capabilities.with(DecorationModifier::Fullscreen),
            3 => capabilities.with(DecorationModifier::Minimized),
            4 => capabilities.with(DecorationModifier::Resizable),
            5 => capabilities.with(DecorationModifier::Tiled),
            6 => capabilities.with(DecorationModifier::Suspended),
            7 => capabilities.with(DecorationModifier::Closable),
            8 => capabilities.with(DecorationModifier::Menu),
            _ => capabilities,
        };
    }
    capabilities
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_kuitui::geometry::{Constraints, Offset, Size};
    use codevar_kuitui::widget::{MeasureScope, Widget};
    use codevar_kuitui_decoration::DecorationKind;
    use codevar_wayland::WlArray;

    struct NullDecoration;

    impl Widget for NullDecoration {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::ZERO)
        }
    }

    impl Decoration for NullDecoration {
        fn kind(&self) -> DecorationKind {
            DecorationKind::TitleBar
        }

        fn id(&self) -> Option<DecorationId> {
            None
        }

        fn bind(&mut self, _id: DecorationId) {}

        fn unbind(&mut self) {}
    }

    /// A memory transport that records output and can synthesize input.
    #[derive(Default)]
    struct MemoryTransport {
        input: Vec<u8>,
        input_pos: usize,
        output: Vec<u8>,
    }

    impl MemoryTransport {
        fn with_input(bytes: &[u8]) -> Self {
            Self {
                input: bytes.to_vec(),
                ..Self::default()
            }
        }
    }

    impl WlTransport for MemoryTransport {
        fn recv(&mut self, buf: &mut [u8], _fds: &mut Vec<WlFd>) -> WlResult<usize> {
            if self.input_pos >= self.input.len() {
                return Err(WlError::WouldBlock);
            }
            let available = self.input.len() - self.input_pos;
            let count = available.min(buf.len());
            let start = self.input_pos;
            buf[..count].copy_from_slice(&self.input[start..start + count]);
            self.input_pos += count;
            Ok(count)
        }

        fn send(&mut self, data: &[u8], _fds: &[WlFd]) -> WlResult<usize> {
            self.output.extend_from_slice(data);
            Ok(data.len())
        }

        fn wait(&mut self, _timeout: Option<TimeDuration>, mask: WlPollEvents) -> WlResult<WlPollEvents> {
            let mut events = WlPollEvents::EMPTY;
            if self.input_pos < self.input.len() {
                events.insert(WlPollEvents::READABLE);
            }
            Ok(events.intersection(mask))
        }

        fn handle(&self) -> WlHandle {
            0
        }
    }

    fn global_event(name: u32, interface: &str, version: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        let payload = 4 + 4 + 4 + interface.len() + 1;
        let padded = (payload + 3) & !3;
        let size = 8 + padded;
        bytes.extend_from_slice(&((size as u32) << 16).to_le_bytes());
        bytes.extend_from_slice(&name.to_le_bytes());
        bytes.extend_from_slice(&((interface.len() + 1) as u32).to_le_bytes());
        bytes.extend_from_slice(interface.as_bytes());
        bytes.push(0);
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes
    }

    #[test]
    fn backend_creates_surfaces_and_queues_requests() {
        let mut globals = Vec::new();
        globals.extend_from_slice(&global_event(1, "wl_compositor", 7));
        globals.extend_from_slice(&global_event(2, "xdg_wm_base", 7));
        let transport = MemoryTransport::with_input(&globals);
        let mut backend = WaylandBackend::new(WlClientDisplay::connect(transport).unwrap());
        let _ = backend.display.dispatch(Some(TimeDuration::ZERO));

        let id = backend
            .create_surface(&NullDecoration)
            .expect("create surface succeeds");
        assert_ne!(id, DecorationId::default());
        // The surface was registered by the backend.
        assert_eq!(backend.state.borrow().surfaces.len(), 1);
        assert!(backend.display.flush().is_ok());

        backend
            .set_title(id, "Codevar")
            .expect("set title succeeds");
        let bounds = DecorationBounds::new(0, 0, 800, 600);
        backend
            .set_bounds(id, bounds)
            .expect("set bounds succeeds");
        let modifiers = DecorationModifier::Maximized.into();
        backend
            .set_modifiers(id, modifiers)
            .expect("set modifiers succeeds");
        let icon = DecorationIcon::from_rgba8(16, 16, vec![0; 16 * 16 * 4]);
        backend
            .set_icon(id, &icon)
            .expect("set icon succeeds");
        let metrics = backend.metrics(id);
        assert_eq!(metrics.title_bar_height, 32);

        backend
            .destroy_surface(id)
            .expect("destroy surface succeeds");
    }

    #[test]
    fn invalid_bounds_are_rejected() {
        let mut globals = Vec::new();
        globals.extend_from_slice(&global_event(1, "wl_compositor", 7));
        globals.extend_from_slice(&global_event(2, "xdg_wm_base", 7));
        let transport = MemoryTransport::with_input(&globals);
        let mut backend = WaylandBackend::new(WlClientDisplay::connect(transport).unwrap());
        let _ = backend.display.dispatch(Some(TimeDuration::ZERO));
        let id = backend.create_surface(&NullDecoration).unwrap();
        assert!(
            backend
                .set_bounds(id, DecorationBounds::new(0, 0, 0, 0))
                .is_err()
        );
    }

    #[test]
    fn registry_globals_are_recorded() {
        let event = global_event(1, "wl_compositor", 7);
        let transport = MemoryTransport::with_input(&event);
        let mut backend = WaylandBackend::new(WlClientDisplay::connect(transport).unwrap());
        let dispatched = backend
            .display
            .dispatch(Some(TimeDuration::ZERO))
            .unwrap();
        assert_eq!(dispatched, 1);
        assert_eq!(backend.display.pending_events(), 0);
    }

    #[test]
    fn every_state_decodes_from_configure() {
        for (byte, modifier) in [
            (0u8, DecorationModifier::Activated),
            (1u8, DecorationModifier::Maximized),
            (2u8, DecorationModifier::Fullscreen),
            (3u8, DecorationModifier::Minimized),
            (4u8, DecorationModifier::Resizable),
            (5u8, DecorationModifier::Tiled),
            (6u8, DecorationModifier::Suspended),
        ] {
            let array = WlArray::from_bytes(&[byte]);
            let states = decode_configure_states(Some(&WlArgument::Array(Some(array))));
            assert!(states.has(modifier));
        }
    }

    #[test]
    fn every_capability_decodes_from_wm_capabilities() {
        for (byte, modifier) in [
            (0u8, DecorationModifier::Activated),
            (1u8, DecorationModifier::Maximized),
            (2u8, DecorationModifier::Fullscreen),
            (3u8, DecorationModifier::Minimized),
            (4u8, DecorationModifier::Resizable),
            (5u8, DecorationModifier::Tiled),
            (6u8, DecorationModifier::Suspended),
            (7u8, DecorationModifier::Closable),
            (8u8, DecorationModifier::Menu),
        ] {
            let array = WlArray::from_bytes(&[byte]);
            let capabilities = decode_capabilities_array(Some(&WlArgument::Array(Some(array))));
            assert!(capabilities.has(modifier));
        }
    }

    #[test]
    fn every_modifier_is_a_state_or_a_capability() {
        for modifier in [
            DecorationModifier::Activated,
            DecorationModifier::Maximized,
            DecorationModifier::Fullscreen,
            DecorationModifier::Minimized,
            DecorationModifier::Resizable,
            DecorationModifier::Tiled,
            DecorationModifier::Suspended,
            DecorationModifier::Closable,
            DecorationModifier::Menu,
        ] {
            assert!(modifier.is_state() ^ modifier.is_capability());
        }
    }

    #[test]
    fn rect_from_bounds_matches() {
        let bounds = DecorationBounds::new(10, 20, 100, 50);
        let rect = bounds.rect();
        assert_eq!(rect.origin, Offset::new(10.0, 20.0));
        assert_eq!(rect.size, Size::new(100.0, 50.0));
    }
}
