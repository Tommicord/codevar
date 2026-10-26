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

//! Wayland window initialization for the launcher.
//!
//! [`WaylandWindow::open`] performs the whole `xdg-shell` handshake of a
//! toplevel window — registry round trip, global binding, surface
//! creation, the empty first commit, `ack_configure` — and maps an
//! `wl_shm` buffer so the compositor shows the window as soon as the
//! first configure arrives. From then on [`WaylandWindow::pump`] keeps
//! the connection moving: it answers `xdg_wm_base.ping`, acknowledges
//! resizes and reports [`WaylandWindow::is_open`] until the compositor
//! closes the window.
//!
//! The window is generic over [`WlTransport`] so the handshake can be
//! exercised against an in-process compositor; the launcher opens real
//! windows through [`WaylandWindow::open`] on [`WlUnixTransport`].

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::{format, vec};
use core::cell::{Cell, RefCell};

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::wnd::{Window, WndBackend, WndBackendId, WndConfig, WndError, WndEvent, WndSize};
use codevar_env::env_var;
use codevar_logger::log_warn;
use codevar_time_core::TimeDuration;
use codevar_wayland::{
    BUFFER_DESTROY, COMPOSITOR_CREATE_SURFACE, COMPOSITOR_INTERFACE, SHM_CREATE_POOL, SHM_FORMAT_XRGB8888,
    SHM_INTERFACE, SHM_POOL_CREATE_BUFFER, SHM_POOL_DESTROY, SURFACE_ATTACH, SURFACE_COMMIT, SURFACE_DAMAGE,
    SURFACE_DESTROY, WlArgument, WlClientDisplay, WlError, WlInterface, WlProxyId, WlRegistryEvent, WlResult,
    WlTransport, WlUnixTransport, XDG_SURFACE_ACK_CONFIGURE, XDG_SURFACE_CONFIGURE, XDG_SURFACE_DESTROY,
    XDG_SURFACE_GET_TOPLEVEL, XDG_SURFACE_SET_WINDOW_GEOMETRY, XDG_TOPLEVEL_CLOSE, XDG_TOPLEVEL_CONFIGURE,
    XDG_TOPLEVEL_DESTROY, XDG_TOPLEVEL_SET_APP_ID, XDG_TOPLEVEL_SET_TITLE, XDG_WM_BASE_GET_XDG_SURFACE,
    XDG_WM_BASE_INTERFACE, XDG_WM_BASE_PING, XDG_WM_BASE_PONG,
};

/// Dispatch slice used while waiting for compositor answers.
const DISPATCH_SLICE: TimeDuration = TimeDuration::from_millis(8);
/// Dispatch slices spent on `wl_display.sync` before giving up.
const SYNC_SLICES: u32 = 0x100;
/// Dispatch slices spent on the first configure before giving up.
const CONFIGURE_SLICES: u32 = 0x100;
/// Largest `wl_shm` pool the window will ask for (1 GiB).
const MAX_POOL_BYTES: usize = 1 << 30;
/// Background color of the initial buffer, as BGRA bytes.
const BACKGROUND: [u8; 4] = [0x24, 0x1e, 0x1e, 0xff];

/// Registry announcements as `(name, interface, version)` tuples.
type Globals = Rc<RefCell<alloc::vec::Vec<(u32, String, u32)>>>;

/// Protocol state shared between the window and its event listeners.
///
/// The listeners run inside [`WlClientDisplay::dispatch`], so they only
/// record what happened; [`WaylandWindow`] turns those records into
/// requests once the dispatch returned.
#[derive(Default)]
struct WndSignals {
    /// Set when the compositor sent `xdg_toplevel.close`.
    closed: Cell<bool>,
    /// Serial of the newest unacknowledged `xdg_surface.configure`.
    configure_serial: Cell<Option<u32>>,
    /// Size the compositor proposed in `xdg_toplevel.configure`.
    pending_size: Cell<(i32, i32)>,
    /// Number of pings answered.
    pings: Cell<u32>,
}

/// A shared memory pool mapped into this process.
struct ShmPool {
    /// The `wl_shm_pool` the mapping is published as.
    proxy: WlProxyId,
    /// Descriptor the mapping and the pool are backed by.
    fd: libc::c_int,
    /// Start of the `mmap`ed region, `len` bytes long.
    bytes: *mut u8,
    /// Length of the mapped region in bytes.
    len: usize,
}

impl ShmPool {
    /// Maps `width * height` pixels worth of shared memory and publishes
    /// it as a `wl_shm_pool` on `shm`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Unsupported`] when the platform has no
    /// `memfd_create`, [`WlError::Io`] when the descriptor, mapping or
    /// size are unusable, and the marshalling error when the compositor
    /// rejected the pool.
    fn create<T: WlTransport>(
        display: &mut WlClientDisplay<T>,
        shm: WlProxyId,
        width: i32,
        height: i32,
    ) -> WlResult<Self> {
        let len = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        if len == 0 || len > MAX_POOL_BYTES {
            return Err(WlError::invalid_argument(format!(
                "refusing a {len} byte wl_shm pool for {width}x{height}"
            )));
        }
        let fd = create_shm_fd(len)?;
        let bytes = map_shared(fd, len).inspect_err(|_error| {
            // Safety: `fd` is open and owned by this branch until the mapping exists.
            unsafe { libc::close(fd) };
        })?;
        let proxy = match display.marshal_new_id(
            shm,
            SHM_CREATE_POOL,
            vec![WlArgument::Fd(fd), WlArgument::Int(len as i32)],
        ) {
            Ok(proxy) => proxy,
            Err(error) => {
                // Safety: `bytes`/`len` still describe the fresh mapping
                // and `fd` is open, so both are released exactly here.
                unsafe {
                    libc::munmap(bytes.cast::<libc::c_void>(), len);
                    libc::close(fd);
                }
                return Err(error);
            }
        };
        // The descriptor stays open until the pool drops: the wire copy
        // is only taken at the first flush after `create_pool`.
        Ok(Self {
            proxy,
            fd,
            bytes,
            len,
        })
    }

    /// Fills the mapping with [`BACKGROUND`] so the window is opaque.
    fn paint(&self) {
        // Safety: `bytes` points at `len` writable bytes of the live
        // mapping installed by `create` and `chunks_exact_mut(4)` never
        // leaves that range.
        let pixels = unsafe { core::slice::from_raw_parts_mut(self.bytes, self.len) };
        for pixel in pixels.chunks_exact_mut(4) {
            pixel.copy_from_slice(&BACKGROUND);
        }
    }
}

impl Drop for ShmPool {
    fn drop(&mut self) {
        // Safety: `bytes` and `len` describe the mapping created for
        // `fd` in `create` and both are released exactly once.
        unsafe {
            libc::munmap(self.bytes.cast::<libc::c_void>(), self.len);
            libc::close(self.fd);
        }
    }
}

/// Creates the descriptor backing a `wl_shm` pool of `len` bytes.
///
/// # Errors
///
/// Returns [`WlError::Unsupported`] on platforms without
/// `memfd_create` and [`WlError::Io`] when the descriptor could not be
/// created or sized.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn create_shm_fd(len: usize) -> WlResult<libc::c_int> {
    // Safety: the name is a valid C string and the flags are defined.
    let fd = unsafe { libc::memfd_create(c"codevar-wnd".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(WlError::io("memfd_create failed"));
    }
    // Safety: `fd` is open and `len` fits into `off_t` thanks to `MAX_POOL_BYTES`.
    if unsafe { libc::ftruncate(fd, len as libc::off_t) } != 0 {
        // Safety: `fd` is open and owned by this branch.
        unsafe { libc::close(fd) };
        return Err(WlError::io("ftruncate failed"));
    }
    Ok(fd)
}

/// Creates the descriptor backing a `wl_shm` pool of `len` bytes.
///
/// # Errors
///
/// Always returns [`WlError::Unsupported`]: the platform provides no
/// movable shared memory descriptor to hand to `wl_shm`.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn create_shm_fd(_len: usize) -> WlResult<libc::c_int> {
    Err(WlError::unsupported(
        "wl_shm pools need memfd_create on this platform",
    ))
}

/// Maps `len` writable bytes of `fd` into this process.
///
/// # Errors
///
/// Returns [`WlError::Io`] when `mmap` refused the region.
fn map_shared(fd: libc::c_int, len: usize) -> WlResult<*mut u8> {
    // Safety: `len` is bounded by `MAX_POOL_BYTES`, `fd` is open and a
    // fresh mapping has no aliasing readers yet.
    let mapped = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    if mapped == libc::MAP_FAILED {
        return Err(WlError::io("mmap failed"));
    }
    Ok(mapped.cast::<u8>())
}

/// An `xdg-shell` toplevel mapped on a Wayland compositor.
///
/// Dropping the window (or calling [`Self::close`]) destroys the
/// protocol objects in the order the specification prescribes.
pub struct WaylandWindow<T: WlTransport> {
    display: WlClientDisplay<T>,
    signals: Rc<WndSignals>,
    surface: WlProxyId,
    xdg_surface: WlProxyId,
    toplevel: WlProxyId,
    pool: ShmPool,
    buffer: WlProxyId,
    width: i32,
    height: i32,
    mapped: bool,
    torn_down: bool,
}

impl<T: WlTransport> WaylandWindow<T> {
    /// Opens a window from an already connected `transport`.
    ///
    /// The call performs the full handshake, so it only returns once the
    /// window is mapped or the compositor refused to map it.
    ///
    /// # Errors
    ///
    /// Returns [`WndError::NoCompositor`] when the session has no
    /// reachable compositor socket, [`WndError::Timeout`] when the
    /// compositor never answered `wl_display.sync` or the initial
    /// `xdg_surface.configure`, [`WndError::Disconnected`] when it
    /// closed the window first, [`WndError::Unsupported`] when a
    /// required global is missing, and the classified transport error
    /// otherwise.
    pub fn open_with(transport: T, config: &WndConfig) -> Result<Self, WndError> {
        let mut display = WlClientDisplay::connect(transport)?;
        let registry = display.get_registry()?;

        let globals: Globals = Rc::new(RefCell::new(alloc::vec::Vec::new()));
        {
            let globals = Rc::clone(&globals);
            display.add_registry_listener(registry, move |_, event| {
                if let WlRegistryEvent::Global {
                    name,
                    interface,
                    version,
                } = event
                {
                    globals
                        .borrow_mut()
                        .push((name, interface, version));
                }
                0
            })?;
        }
        bounded_roundtrip(&mut display)?;
        let compositor = bind(&mut display, registry, &globals, &COMPOSITOR_INTERFACE)?;
        let shm = bind(&mut display, registry, &globals, &SHM_INTERFACE)?;
        let wm_base = bind(&mut display, registry, &globals, &XDG_WM_BASE_INTERFACE)?;

        let signals = Rc::new(WndSignals::default());
        install_ping_listener(&mut display, wm_base, &signals)?;

        let surface = display.marshal_new_id(compositor, COMPOSITOR_CREATE_SURFACE, Vec::new())?;
        let xdg_surface = display.marshal_new_id(
            wm_base,
            XDG_WM_BASE_GET_XDG_SURFACE,
            vec![WlArgument::Object(surface.id())],
        )?;
        {
            let signals = Rc::clone(&signals);
            display.add_listener(xdg_surface, move |_, opcode, args| {
                if opcode == XDG_SURFACE_CONFIGURE
                    && let Some(WlArgument::Uint(serial)) = args.first()
                {
                    signals.configure_serial.set(Some(*serial));
                }
                0
            })?;
        }

        let toplevel = display.marshal_new_id(xdg_surface, XDG_SURFACE_GET_TOPLEVEL, Vec::new())?;
        {
            let signals = Rc::clone(&signals);
            display.add_listener(toplevel, move |_, opcode, args| match opcode {
                XDG_TOPLEVEL_CONFIGURE => {
                    if let (Some(WlArgument::Int(width)), Some(WlArgument::Int(height))) =
                        (args.first(), args.get(1))
                    {
                        signals.pending_size.set((*width, *height));
                    }
                    0
                }
                XDG_TOPLEVEL_CLOSE => {
                    signals.closed.set(true);
                    0
                }
                _ => 0,
            })?;
        }

        display.marshal_request(
            toplevel,
            XDG_TOPLEVEL_SET_TITLE,
            vec![WlArgument::Str(Some(config.title.clone()))],
        )?;
        display.marshal_request(
            toplevel,
            XDG_TOPLEVEL_SET_APP_ID,
            vec![WlArgument::Str(Some(config.app_id.clone()))],
        )?;
        // The first commit asks the compositor for the initial configure.
        display.marshal_request(surface, SURFACE_COMMIT, Vec::new())?;
        flush_lenient(&mut display)?;
        wait_first_configure(&mut display, &signals)?;

        // The compositor picks the size when it proposes a positive one.
        let mut width = config.geom.width;
        let mut height = config.geom.height;
        let (proposed_width, proposed_height) = signals.pending_size.get();
        if proposed_width > 0 && proposed_height > 0 {
            width = proposed_width;
            height = proposed_height;
        }
        let pool = ShmPool::create(&mut display, shm, width, height)?;
        let buffer = display.marshal_new_id(
            pool.proxy,
            SHM_POOL_CREATE_BUFFER,
            vec![
                WlArgument::Int(0),
                WlArgument::Int(width),
                WlArgument::Int(height),
                WlArgument::Int(width * 4),
                WlArgument::Uint(SHM_FORMAT_XRGB8888),
            ],
        )?;
        pool.paint();
        let mut window = Self {
            display,
            signals,
            surface,
            xdg_surface,
            toplevel,
            pool,
            buffer,
            width,
            height,
            mapped: false,
            torn_down: false,
        };
        window.ack_pending()?;
        window.map()?;
        Ok(window)
    }

    /// Dispatches window events for up to `timeout`.
    ///
    /// Afterward pending configures are acknowledged, which keeps the
    /// window geometry in sync when the compositor resized it.
    ///
    /// # Errors
    ///
    /// Returns the transport error when the connection failed; the
    /// window must be discarded afterward.
    pub fn pump(&mut self, timeout: Option<TimeDuration>) -> WlResult<usize> {
        if self.torn_down {
            return Ok(0);
        }
        let events = self.display.dispatch(timeout)?;
        self.ack_pending()?;
        Ok(events)
    }

    /// Returns `true` while the window exists and the compositor has not
    /// asked it to close.
    #[must_use]
    pub fn is_open(&self) -> bool {
        !self.torn_down && !self.signals.closed.get()
    }

    /// Returns the window size in surface local pixels.
    #[must_use]
    pub const fn size(&self) -> WndSize {
        WndSize::new(self.width, self.height)
    }

    /// Destroys the window; later [`Self::pump`] calls do nothing.
    pub fn close(&mut self) {
        self.teardown();
    }

    /// Acknowledges the newest configure and applies the proposed size.
    fn ack_pending(&mut self) -> WlResult<()> {
        let Some(serial) = self.signals.configure_serial.take() else {
            return Ok(());
        };
        let (pending_width, pending_height) = self.signals.pending_size.get();
        if pending_width > 0 && pending_height > 0 {
            self.width = pending_width;
            self.height = pending_height;
        }
        self.display.marshal_request(
            self.xdg_surface,
            XDG_SURFACE_ACK_CONFIGURE,
            vec![WlArgument::Uint(serial)],
        )?;
        self.display.marshal_request(
            self.xdg_surface,
            XDG_SURFACE_SET_WINDOW_GEOMETRY,
            vec![
                WlArgument::Int(0),
                WlArgument::Int(0),
                WlArgument::Int(self.width),
                WlArgument::Int(self.height),
            ],
        )?;
        if self.mapped {
            // The buffer already covers the new geometry; a bare commit
            // lets the compositor apply it.
            self.display
                .marshal_request(self.surface, SURFACE_COMMIT, Vec::new())?;
        }
        flush_lenient(&mut self.display)
    }

    /// Attaches the buffer and commits it, mapping the window.
    fn map(&mut self) -> WlResult<()> {
        self.display.marshal_request(
            self.surface,
            SURFACE_ATTACH,
            vec![
                WlArgument::Object(self.buffer.id()),
                WlArgument::Int(0),
                WlArgument::Int(0),
            ],
        )?;
        self.display.marshal_request(
            self.surface,
            SURFACE_DAMAGE,
            vec![
                WlArgument::Int(0),
                WlArgument::Int(0),
                WlArgument::Int(self.width),
                WlArgument::Int(self.height),
            ],
        )?;
        self.display
            .marshal_request(self.surface, SURFACE_COMMIT, Vec::new())?;
        self.mapped = true;
        flush_lenient(&mut self.display)
    }

    /// Destroys the protocol objects in the order the specification asks for.
    fn teardown(&mut self) {
        if self.torn_down {
            return;
        }
        self.torn_down = true;
        self.destroy(self.toplevel, XDG_TOPLEVEL_DESTROY);
        self.destroy(self.xdg_surface, XDG_SURFACE_DESTROY);
        self.destroy(self.surface, SURFACE_DESTROY);
        self.destroy(self.buffer, BUFFER_DESTROY);
        self.destroy(self.pool.proxy, SHM_POOL_DESTROY);
        let _ = flush_lenient(&mut self.display);
    }

    /// Queues a destroy request, retrying until the connection takes it.
    ///
    /// An [`WlError::InvalidObject`] answer only says the object is
    /// already gone — which is exactly what a destroy wants — and a
    /// closed connection leaves nothing to destroy either, so both end
    /// quietly. Anything else is logged: `Drop` cannot surface it.
    fn destroy(&mut self, proxy: WlProxyId, opcode: u32) {
        match marshal_retry(&mut self.display, proxy, opcode, Vec::new()) {
            Ok(()) | Err(WlError::InvalidObject(_)) | Err(WlError::Disconnected) => {}
            Err(error) => log_warn!("wnd destroy request failed: {}", error),
        }
    }
}

impl WaylandWindow<WlUnixTransport> {
    /// Opens a window on the compositor of the current session.
    ///
    /// # Errors
    ///
    /// Returns [`WndError::NoCompositor`] when no compositor socket of
    /// the session can be reached, and the classified handshake errors
    /// of [`Self::open_with`] otherwise.
    pub fn open(config: &WndConfig) -> Result<Self, WndError> {
        let transport = WlUnixTransport::connect_session().map_err(|error| WndError::NoCompositor {
            id: WndBackendId::Wayland,
            message: error.to_string(),
        })?;
        Self::open_with(transport, config)
    }
}

impl<T: WlTransport> Drop for WaylandWindow<T> {
    fn drop(&mut self) {
        self.teardown();
    }
}

/// The compiled-in windowing backends, in probe order.
pub(crate) const BACKENDS: &[&dyn WndBackend] = &[&WaylandBackend];

/// Windowing backend that opens [`WaylandWindow`]s on the compositor
/// of the current session.
pub(crate) struct WaylandBackend;

impl WndBackend for WaylandBackend {
    fn id(&self) -> WndBackendId {
        WndBackendId::Wayland
    }

    fn is_available(&self) -> bool {
        advertises_wayland()
    }

    fn open(&self, config: &WndConfig) -> Result<Box<dyn Window>, WndError> {
        Ok(Box::new(WaylandWindow::open(config)?))
    }
}

impl From<WlError> for WndError {
    fn from(error: WlError) -> Self {
        let id = WndBackendId::Wayland;
        let message = error.to_string();
        match error {
            WlError::Disconnected => Self::Disconnected { id, message },
            WlError::Protocol(_) => Self::Protocol { id, message },
            WlError::Unsupported(reason) => Self::Unsupported(reason),
            WlError::WouldBlock
            | WlError::Io(_)
            | WlError::MessageTooBig(_)
            | WlError::InvalidObject(_)
            | WlError::InvalidMethod { .. }
            | WlError::InvalidArgument(_)
            | WlError::InvalidState(_)
            | WlError::TooManyObjects => Self::Backend { id, message },
        }
    }
}

impl<T: WlTransport> Window for WaylandWindow<T> {
    fn backend(&self) -> WndBackendId {
        WndBackendId::Wayland
    }

    fn is_open(&self) -> bool {
        WaylandWindow::is_open(self)
    }

    fn close(&mut self) {
        WaylandWindow::close(self);
    }

    fn size(&self) -> WndSize {
        WaylandWindow::size(self)
    }

    fn set_title(&mut self, title: &str) -> Result<(), WndError> {
        if self.torn_down {
            return Ok(());
        }
        marshal_retry(
            &mut self.display,
            self.toplevel,
            XDG_TOPLEVEL_SET_TITLE,
            vec![WlArgument::Str(Some(title.to_string()))],
        )
        .map_err(WndError::from)?;
        flush_lenient(&mut self.display).map_err(WndError::from)
    }

    fn poll(
        &mut self,
        timeout: Option<TimeDuration>,
        sink: &mut dyn FnMut(WndEvent),
    ) -> Result<usize, WndError> {
        let was_open = WaylandWindow::is_open(self);
        let before = WaylandWindow::size(self);
        let events = WaylandWindow::pump(self, timeout).map_err(WndError::from)?;
        if was_open && !WaylandWindow::is_open(self) {
            sink(WndEvent::CloseRequested);
        }
        let after = WaylandWindow::size(self);
        if after != before {
            sink(WndEvent::Resized(after));
        }
        Ok(events)
    }
}

/// Returns `true` when the environment advertises a Wayland session.
///
/// Mirrors the usual probe: a non-empty `WAYLAND_DISPLAY` means a
/// compositor socket exists, and `XDG_SESSION_TYPE=wayland` is the
/// session manager's marker.
fn advertises_wayland() -> bool {
    if let Some(display) = env_var("WAYLAND_DISPLAY")
        && !display.is_empty()
    {
        return true;
    }
    env_var("XDG_SESSION_TYPE").as_deref() == Some("wayland")
}

/// Answers `xdg_wm_base.ping` and counts the pings seen so far.
fn install_ping_listener<T: WlTransport>(
    display: &mut WlClientDisplay<T>,
    wm_base: WlProxyId,
    signals: &Rc<WndSignals>,
) -> WlResult<()> {
    let signals = Rc::clone(signals);
    display.add_listener(wm_base, move |client, opcode, args| {
        if opcode == XDG_WM_BASE_PING
            && let Some(WlArgument::Uint(serial)) = args.first()
        {
            let serial = *serial;
            let pong = client.marshal_request(wm_base, XDG_WM_BASE_PONG, vec![WlArgument::Uint(serial)]);
            if pong.is_ok() {
                signals.pings.set(signals.pings.get() + 1);
            }
        }
        0
    })
}

/// Runs a bounded `wl_display.sync` round trip.
///
/// [`WlClientDisplay::roundtrip`] would wait forever on a silent
/// compositor; this variant gives up after [`SYNC_SLICES`] dispatches.
///
/// # Errors
///
/// Returns [`WndError::Timeout`] when the callback never arrived and
/// the classified dispatch error when the connection failed.
fn bounded_roundtrip<T: WlTransport>(display: &mut WlClientDisplay<T>) -> Result<(), WndError> {
    let done = Rc::new(Cell::new(false));
    let flag = Rc::clone(&done);
    let callback = display.sync()?;
    display.add_callback_listener(callback, move |_, _| {
        flag.set(true);
        0
    })?;

    let mut slices = SYNC_SLICES;
    while !done.get() {
        if slices == 0 {
            let _ = display.proxy_destroy(callback);
            return Err(WndError::Timeout {
                id: WndBackendId::Wayland,
                message: String::from("timed out waiting for wl_display.sync"),
            });
        }
        slices -= 1;
        display.dispatch(Some(DISPATCH_SLICE))?;
    }
    display
        .proxy_destroy(callback)
        .map_err(WndError::from)
}

/// Waits for the first `xdg_surface.configure` of the window.
///
/// # Errors
///
/// Returns [`WndError::Timeout`] when the compositor did not configure
/// the window within [`CONFIGURE_SLICES`] dispatches,
/// [`WndError::Disconnected`] when it closed the window first, and
/// the classified dispatch error when the connection failed.
fn wait_first_configure<T: WlTransport>(
    display: &mut WlClientDisplay<T>,
    signals: &WndSignals,
) -> Result<(), WndError> {
    let mut slices = CONFIGURE_SLICES;
    while signals.configure_serial.get().is_none() && !signals.closed.get() {
        if slices == 0 {
            return Err(WndError::Timeout {
                id: WndBackendId::Wayland,
                message: String::from("timed out waiting for the initial xdg_surface.configure"),
            });
        }
        slices -= 1;
        display.dispatch(Some(DISPATCH_SLICE))?;
    }
    if signals.closed.get() {
        return Err(WndError::Disconnected {
            id: WndBackendId::Wayland,
            message: String::from("the compositor closed the window before the initial configure"),
        });
    }
    Ok(())
}

/// Binds `interface` to the global of the same name announced by the registry.
///
/// # Errors
///
/// Returns [`WlError::Unsupported`] when the compositor never announced
/// the global and the bind error otherwise.
fn bind<T: WlTransport>(
    display: &mut WlClientDisplay<T>,
    registry: WlProxyId,
    globals: &Globals,
    interface: &'static WlInterface,
) -> WlResult<WlProxyId> {
    let (name, version) = globals
        .borrow()
        .iter()
        .find(|(_, announced, _)| announced == interface.name)
        .map(|(name, _, version)| (*name, *version))
        .ok_or_else(|| WlError::Unsupported(format!("the compositor did not announce {}", interface.name)))?;
    display.registry_bind(registry, name, interface, version.min(interface.version))
}

/// Writes buffered requests, tolerating a partially accepted buffer.
///
/// The remainder is flushed by the next dispatch, so a `WouldBlock`
/// here is not an error.
///
/// # Errors
///
/// Returns the transport error when the connection failed.
fn flush_lenient<T: WlTransport>(display: &mut WlClientDisplay<T>) -> WlResult<()> {
    match display.flush() {
        Ok(_) => Ok(()),
        Err(WlError::WouldBlock) => Ok(()),
        Err(error) => Err(error),
    }
}

/// Queues a request, retrying until the connection accepts it.
///
/// [`WlClientDisplay::marshal_request`] only encodes the message into
/// the outgoing buffer, so the one failure a retry can repair is a
/// buffer that is still full ([`WlError::WouldBlock`]): draining it
/// with [`WlClientDisplay::flush`] makes room and the next attempt
/// fits. Every other error is permanent — a stale object, an unknown
/// request, a message over the wire limit — and is returned at once,
/// because retrying it could never succeed.
///
/// # Errors
///
/// Returns the permanent marshaling error, or the transport error
/// observed while draining the outgoing buffer.
fn marshal_retry<T: WlTransport>(
    display: &mut WlClientDisplay<T>,
    proxy: WlProxyId,
    opcode: u32,
    args: Vec<WlArgument>,
) -> WlResult<()> {
    loop {
        match display.marshal_request(proxy, opcode, args.clone()) {
            Ok(()) => return Ok(()),
            Err(error) if error.is_would_block() => {
                if let Err(error) = display.flush()
                    && !error.is_would_block()
                {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wayland_errors_classify_into_distinct_wnd_error_kinds() {
        let id = WndBackendId::Wayland;
        let disconnected = WndError::from(WlError::Disconnected);
        assert!(matches!(
            disconnected,
            WndError::Disconnected {
                id: WndBackendId::Wayland,
                ..
            }
        ));
        assert!(!disconnected.is_transient());

        let unsupported = WndError::from(WlError::unsupported("the compositor did not announce wl_shm"));
        assert_eq!(
            unsupported,
            WndError::Unsupported(String::from("the compositor did not announce wl_shm"))
        );
        assert!(!unsupported.is_transient());

        let protocol = WndError::from(WlError::protocol(codevar_wayland::WlProtocolError::new(
            2,
            9,
            "wl_surface",
            String::from("buffer too large"),
        )));
        assert!(matches!(
            protocol,
            WndError::Protocol {
                id: WndBackendId::Wayland,
                ..
            }
        ));
        assert!(!protocol.is_transient());

        let transport = WndError::from(WlError::io("read failed"));
        assert_eq!(transport, WndError::backend(id, "transport error: read failed"));
        assert!(transport.is_transient());

        let stale_object = WndError::from(WlError::InvalidObject(42));
        assert!(matches!(
            stale_object,
            WndError::Backend {
                id: WndBackendId::Wayland,
                ..
            }
        ));
    }
}
