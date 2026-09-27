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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Wayland display initialization and window management for the
//! codevar UI stack.
//!
//! This module ties the Wayland protocol layer ([`codevar_wl_protocol`]),
//! the Vulkan pipeline ([`crate::ui_pipeline`]) and the renderer
//! ([`crate::ui_renderer`]) together into a single [`UiDisplay`] handle
//! that initializes a window, renders frames and tears down the
//! connection on drop.
//!
//! # Initialization flow
//!
//! 1. [`UiDisplay::new`] connects to the compositor, binds the
//!    `wl_compositor`, `xdg_wm_base` and `zwp_linux_dmabuf_v1`
//!    globals, creates the surface, xdg-surface and xdg-toplevel,
//!    and drives the `ack_configure` handshake.
//! 2. [`UiDisplay::run`] enters the render loop: each frame waits
//!    for the compositor's release, records, submits and exports a
//!    `sync_file`, then attaches the resulting dma-buf buffer and
//!    commits.
//! 3. [`UiDisplay::stop`] tears down the protocol objects and the
//!    renderer.

use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::ffi::CStr;
use core::fmt;
use core::mem::ManuallyDrop;
use core::time::Duration;

use crate::ui_pipeline::{OwnedFd, PipelineContext, PipelineError};
use crate::ui_renderer::{RendererError, RendererSubsystem};
use codevar_base::basic_signal;
use codevar_base::basic_time::SystemTime;
use codevar_wl_protocol::{
    BUFFER_DESTROY, BUFFER_PARAMS_ADD, BUFFER_PARAMS_CREATE_IMMED, BUFFER_PARAMS_DESTROY, BUFFER_RELEASE,
    COMPOSITOR_CREATE_SURFACE, COMPOSITOR_INTERFACE, DMABUF_CREATE_PARAMS, DMABUF_DESTROY,
    DMABUF_GET_DEFAULT_FEEDBACK, DMABUF_INTERFACE, DMABUF_MODIFIER, DRM_FORMAT_XRGB8888, FEEDBACK_DESTROY,
    FEEDBACK_DONE, FEEDBACK_FORMAT_TABLE, FEEDBACK_MAIN_DEVICE, FEEDBACK_TRANCHE_DONE,
    FEEDBACK_TRANCHE_FLAGS, FEEDBACK_TRANCHE_FORMATS, FEEDBACK_TRANCHE_TARGET_DEVICE, SURFACE_ATTACH,
    SURFACE_COMMIT, SURFACE_DAMAGE, SURFACE_DESTROY, SURFACE_FRAME, WlArgument, WlClientDisplay, WlError,
    WlInterface, WlProxyId, WlRegistryEvent, WlUnixTransport, XDG_SURFACE_ACK_CONFIGURE,
    XDG_SURFACE_CONFIGURE, XDG_SURFACE_DESTROY, XDG_SURFACE_GET_TOPLEVEL, XDG_SURFACE_SET_WINDOW_GEOMETRY,
    XDG_TOPLEVEL_CLOSE, XDG_TOPLEVEL_DESTROY, XDG_TOPLEVEL_SET_APP_ID, XDG_TOPLEVEL_SET_TITLE,
    XDG_WM_BASE_GET_XDG_SURFACE, XDG_WM_BASE_INTERFACE, XDG_WM_BASE_PING, XDG_WM_BASE_PONG,
};
use log::info;

/// Initial window width in surface local pixels (used when the first
/// configure carries `0x0`).
const DEFAULT_WIDTH: i32 = 640;
/// Initial window height in surface local pixels.
const DEFAULT_HEIGHT: i32 = 480;
/// How long a single `dispatch` waits for Wayland events, in
/// milliseconds.
const DISPATCH_TIMEOUT_MS: u64 = 50;
/// How long to wait for `wl_buffer.release` before re-rendering
/// anyway, in milliseconds.
const RELEASE_TIMEOUT_MS: u64 = 500;
/// Milliseconds the renderer waits for the GPU through the
/// exported `sync_file` before committing the buffer.
const GPU_SYNC_TIMEOUT_MS: i32 = 5_000;
/// Largest format table the implementation is willing to allocate.
const MAX_FORMAT_TABLE_BYTES: usize = 1 << 20;

/// Registry globals announced by the compositor: `(name, interface, version)`.
type Globals = Rc<RefCell<Vec<(u32, String, u32)>>>;

/// The Wayland connection pieces produced by [`connect`] before the
/// globals are bound.
struct Connection {
    display: WlClientDisplay<WlUnixTransport>,
    registry: WlProxyId,
    globals: Globals,
    window_state: Rc<RefCell<WindowState>>,
}

/// A pending `wl_surface.frame` callback: its proxy and the flag set when
/// the compositor fires it.
struct FrameCallback {
    proxy: WlProxyId,
    done: Rc<Cell<bool>>,
}

/// Errors returned by [`UiDisplay`] initialization and operation.
#[derive(Debug)]
pub enum UiDisplayError {
    /// The Wayland connection failed.
    Wayland(WlError),
    /// The renderer failed.
    Renderer(RendererError),
    /// The Vulkan pipeline could not be created.
    Pipeline(PipelineError),
    /// The signal handler could not be installed.
    Signal(basic_signal::InstallError),
    /// The compositor closed the window before the first configure.
    WindowClosed,
    /// Timed out waiting for a Wayland event.
    Timeout(String),
}

impl fmt::Display for UiDisplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wayland(err) => write!(f, "wayland error: {err}"),
            Self::Renderer(err) => write!(f, "renderer error: {err}"),
            Self::Pipeline(err) => write!(f, "pipeline error: {err}"),
            Self::Signal(err) => write!(f, "signal error: {err}"),
            Self::WindowClosed => write!(f, "the compositor closed the window"),
            Self::Timeout(msg) => write!(f, "timeout: {msg}"),
        }
    }
}

impl core::error::Error for UiDisplayError {}

impl From<WlError> for UiDisplayError {
    fn from(err: WlError) -> Self {
        Self::Wayland(err)
    }
}

impl From<RendererError> for UiDisplayError {
    fn from(err: RendererError) -> Self {
        Self::Renderer(err)
    }
}

impl From<PipelineError> for UiDisplayError {
    fn from(err: PipelineError) -> Self {
        Self::Pipeline(err)
    }
}

impl From<basic_signal::InstallError> for UiDisplayError {
    fn from(err: basic_signal::InstallError) -> Self {
        Self::Signal(err)
    }
}

/// Configuration for creating a new [`UiDisplay`] window.
pub struct WindowInit {
    /// The title of the window.
    pub title: &'static CStr,
    /// The application ID for the window.
    pub app_id: &'static CStr,
    /// Initial width in surface local pixels.
    pub width: i32,
    /// Initial height in surface local pixels.
    pub height: i32,
}

impl Default for WindowInit {
    fn default() -> Self {
        Self {
            title: c"Codevar",
            app_id: c"dev.codevar.window",
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
        }
    }
}

/// Shared mutable state accessed from Wayland event listeners.
struct WindowState {
    closed: bool,
    width: i32,
    height: i32,
    configure_serial: u32,
    configured: bool,
    pings: u32,
    buffer_released: bool,
    deadline_ns: u64,
}

impl WindowState {
    fn new(width: i32, height: i32) -> Self {
        Self {
            closed: false,
            width,
            height,
            configure_serial: 0,
            configured: false,
            pings: 0,
            buffer_released: true,
            deadline_ns: 0,
        }
    }
}

/// Wayland display and window handle.
pub struct UiDisplay {
    display: Option<WlClientDisplay<WlUnixTransport>>,
    dmabuf: Option<WlProxyId>,
    surface: Option<WlProxyId>,
    xdg_surface: Option<WlProxyId>,
    toplevel: Option<WlProxyId>,
    feedback: Option<WlProxyId>,
    pipeline: Option<PipelineContext>,
    renderer: Option<ManuallyDrop<RendererSubsystem<'static>>>,
    buffer: Option<WlProxyId>,
    window_state: Rc<RefCell<WindowState>>,
    frame_budget: u32,
}

impl UiDisplay {
    /// Creates a new window and initializes the Wayland connection.
    ///
    /// The flow delegates each protocol step to a dedicated helper:
    /// [`connect`] (session + registry), [`bind_protocol_globals`] (globals
    /// and their listeners), [`create_window_objects`] (surface,
    /// toplevel and xdg-shell handshake), [`negotiate_modifiers`] (linux-dmabuf
    /// feedback), then the Vulkan pipeline, renderer and `wl_buffer`.
    pub fn new(init: WindowInit) -> Result<Self, UiDisplayError> {
        basic_signal::install().map_err(UiDisplayError::Signal)?;
        info!("codevar: installing basic signal handler");

        let deadline_ns = SystemTime::monotonic_nanos() + SystemTime::secs_to_nanos(30);
        let Connection {
            mut display,
            registry,
            globals,
            window_state,
        } = connect(init.width, init.height)?;
        let protocol = bind_protocol_globals(&mut display, registry, &globals, &window_state)?;
        let (surface, xdg_surface, toplevel) = create_window_objects(
            &mut display,
            protocol.compositor,
            protocol.wm_base,
            &window_state,
            init,
        )?;
        wait_initial_configure(&mut display, &window_state, deadline_ns)?;
        ack_configure(&mut display, xdg_surface, &window_state)?;

        let width = window_state.borrow().width;
        let height = window_state.borrow().height;
        let (modifiers, feedback) =
            negotiate_modifiers(&mut display, protocol.dmabuf, &protocol.legacy_modifiers)?;

        let pipeline = PipelineContext::new(width as u32, height as u32, &modifiers)
            .map_err(UiDisplayError::Pipeline)?;
        let mut renderer = create_renderer(&pipeline);
        renderer
            .start()
            .map_err(UiDisplayError::Renderer)?;
        let buffer = create_wl_buffer(
            &mut display,
            protocol.dmabuf,
            &pipeline,
            width,
            height,
            &window_state,
        )?;
        Ok(Self {
            display: Some(display),
            dmabuf: Some(protocol.dmabuf),
            surface: Some(surface),
            xdg_surface: Some(xdg_surface),
            toplevel: Some(toplevel),
            feedback,
            pipeline: Some(pipeline),
            renderer: Some(renderer),
            buffer: Some(buffer),
            window_state,
            frame_budget: 0,
        })
    }

    /// Returns the pipeline context.
    #[inline]
    #[must_use]
    pub fn pipeline(&self) -> Option<&PipelineContext> {
        self.pipeline.as_ref()
    }

    /// Returns the current window width.
    #[inline]
    #[must_use]
    pub fn width(&self) -> i32 {
        self.window_state.borrow().width
    }

    /// Returns the current window height.
    #[inline]
    #[must_use]
    pub fn height(&self) -> i32 {
        self.window_state.borrow().height
    }

    /// Returns `true` if the compositor has requested the window to close.
    #[inline]
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.window_state.borrow().closed
    }

    /// Returns the number of frames rendered so far.
    #[inline]
    #[must_use]
    pub fn frames(&self) -> u64 {
        self.renderer
            .as_ref()
            .map_or(0, |r| r.frame_index())
    }

    /// Returns the number of pings answered.
    #[inline]
    #[must_use]
    pub fn pings(&self) -> u32 {
        self.window_state.borrow().pings
    }

    /// Sets a frame budget after which the render loop exits.
    /// A value of `0` means unlimited.
    #[inline]
    pub fn set_frame_budget(&mut self, budget: u32) {
        self.frame_budget = budget;
    }

    /// Sets a deadline for the render loop in nanoseconds.
    #[inline]
    pub fn set_deadline(&mut self, deadline_ns: u64) {
        self.window_state.borrow_mut().deadline_ns = deadline_ns;
    }

    /// Returns a mutable reference to the renderer subsystem.
    #[inline]
    #[must_use]
    pub fn renderer_mut(&mut self) -> Option<&mut RendererSubsystem<'static>> {
        self.renderer.as_deref_mut()
    }

    /// Runs the render loop.
    ///
    /// Each iteration drives one frame through the presentation pipeline:
    /// wait for the compositor to release the previous buffer
    /// ([`Self::wait_for_buffer_release`]), request a `wl_surface.frame`
    /// callback ([`Self::request_frame_callback`]), render offscreen
    /// ([`Self::render_frame`]), attach/damage/commit the dma-buf buffer
    /// ([`Self::present_buffer`]) and wait for the callback
    /// ([`Self::wait_for_frame_callback`]).
    pub fn run(&mut self) -> Result<String, UiDisplayError> {
        let mut frames = 0u32;
        let deadline_ns = if self.window_state.borrow().deadline_ns != 0 {
            self.window_state.borrow().deadline_ns
        } else {
            SystemTime::monotonic_nanos() + SystemTime::secs_to_nanos(30)
        };
        let release_timeout_ns = SystemTime::millis_to_nanos(RELEASE_TIMEOUT_MS);

        while (self.frame_budget == 0 || frames < self.frame_budget)
            && !self.window_state.borrow().closed
            && SystemTime::monotonic_nanos() < deadline_ns
        {
            let release_deadline = SystemTime::monotonic_nanos() + release_timeout_ns;
            self.wait_for_buffer_release(release_deadline)?;

            let Some(callback) = self.request_frame_callback()? else {
                break;
            };

            self.render_frame()?;
            self.present_buffer()?;
            self.wait_for_frame_callback(&callback.done, deadline_ns)?;
            self.destroy_frame_callback(callback.proxy);
            frames += 1;
        }

        let state = self.window_state.borrow();
        Ok(format!(
            "rendered {frames} frames into a {}x{} window, {} pings answered",
            state.width, state.height, state.pings
        ))
    }

    /// Dispatches Wayland events until the compositor releases the
    /// attached buffer, the window closes, or `release_deadline_ns`
    /// passes; the buffer is then considered released either way.
    fn wait_for_buffer_release(&mut self, release_deadline_ns: u64) -> Result<(), UiDisplayError> {
        while !self.window_state.borrow().buffer_released
            && !self.window_state.borrow().closed
            && SystemTime::monotonic_nanos() < release_deadline_ns
        {
            if let Some(display) = &mut self.display {
                display
                    .dispatch(Some(Duration::from_millis(DISPATCH_TIMEOUT_MS)))
                    .map_err(UiDisplayError::Wayland)?;
            }
        }
        self.window_state.borrow_mut().buffer_released = true;
        Ok(())
    }

    /// Requests a `wl_surface.frame` callback for the next compositor
    /// frame and returns its proxy together with the flag the callback
    /// sets when it fires.
    ///
    /// Returns `None` when the surface or the connection is gone, which
    /// ends the render loop.
    fn request_frame_callback(&mut self) -> Result<Option<FrameCallback>, UiDisplayError> {
        let Some(surface) = self.surface else {
            return Ok(None);
        };
        let Some(display) = &mut self.display else {
            return Ok(None);
        };
        let frame_proxy = display
            .marshal_new_id(surface, SURFACE_FRAME, Vec::new())
            .map_err(UiDisplayError::Wayland)?;
        let frame_done = Rc::new(Cell::new(false));
        let done = Rc::clone(&frame_done);
        display
            .add_callback_listener(frame_proxy, move |_, _| {
                done.set(true);
                0
            })
            .map_err(UiDisplayError::Wayland)?;
        Ok(Some(FrameCallback {
            proxy: frame_proxy,
            done: frame_done,
        }))
    }

    /// Renders one frame offscreen through the renderer subsystem and
    /// blocks until the GPU reports completion via the exported
    /// `sync_file`.
    fn render_frame(&mut self) -> Result<(), UiDisplayError> {
        let renderer = self
            .renderer
            .as_mut()
            .ok_or(UiDisplayError::Renderer(RendererError::Internal(
                "renderer not started",
            )))?;
        renderer
            .begin_frame(None)
            .map_err(UiDisplayError::Renderer)?;
        renderer
            .render_frame()
            .map_err(UiDisplayError::Renderer)?;
        let sync_file = renderer
            .end_frame()
            .map_err(UiDisplayError::Renderer)?;
        wait_for_gpu(&sync_file)?;
        drop(sync_file);
        Ok(())
    }

    /// Attaches the dma-buf buffer to the surface, damages the whole
    /// window and commits it to the compositor.
    fn present_buffer(&mut self) -> Result<(), UiDisplayError> {
        let (width, height) = (self.width(), self.height());
        let Some(display) = &mut self.display else {
            return Ok(());
        };
        let surface = self.surface.unwrap_or(WlProxyId(0));
        let buffer = self.buffer.unwrap_or(WlProxyId(0));
        display
            .marshal_request(
                surface,
                SURFACE_ATTACH,
                vec![
                    WlArgument::Object(buffer.id()),
                    WlArgument::Int(0),
                    WlArgument::Int(0),
                ],
            )
            .map_err(UiDisplayError::Wayland)?;
        display
            .marshal_request(
                surface,
                SURFACE_DAMAGE,
                vec![
                    WlArgument::Int(0),
                    WlArgument::Int(0),
                    WlArgument::Int(width),
                    WlArgument::Int(height),
                ],
            )
            .map_err(UiDisplayError::Wayland)?;
        display
            .marshal_request(surface, SURFACE_COMMIT, Vec::new())
            .map_err(UiDisplayError::Wayland)?;
        display.flush().map_err(UiDisplayError::Wayland)?;
        Ok(())
    }

    /// Dispatches Wayland events until the frame callback fires, the
    /// window closes, or `deadline_ns` passes.
    fn wait_for_frame_callback(
        &mut self,
        frame_done: &Rc<Cell<bool>>,
        deadline_ns: u64,
    ) -> Result<(), UiDisplayError> {
        while !frame_done.get()
            && !self.window_state.borrow().closed
            && SystemTime::monotonic_nanos() < deadline_ns
        {
            if let Some(display) = &mut self.display {
                display
                    .dispatch(Some(Duration::from_millis(DISPATCH_TIMEOUT_MS)))
                    .map_err(UiDisplayError::Wayland)?;
            }
        }
        Ok(())
    }

    /// Releases the `wl_surface.frame` proxy; failure is ignored because
    /// the callback has already been awaited.
    fn destroy_frame_callback(&mut self, frame_proxy: WlProxyId) {
        if let Some(display) = &mut self.display {
            let _ = display.proxy_destroy(frame_proxy);
        }
    }

    /// Flushes pending requests to the compositor.
    pub fn flush(&mut self) -> Result<(), UiDisplayError> {
        if let Some(display) = &mut self.display {
            display.flush().map_err(UiDisplayError::Wayland)?;
        }
        Ok(())
    }

    /// Stops the renderer.
    pub fn stop(&mut self) -> Result<(), UiDisplayError> {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer
                .stop()
                .map_err(UiDisplayError::Renderer)?;
        }
        Ok(())
    }
}

impl Drop for UiDisplay {
    fn drop(&mut self) {
        if let Some(display) = &mut self.display {
            if let Some(toplevel) = self.toplevel {
                let _ = display.marshal_request(toplevel, XDG_TOPLEVEL_DESTROY, Vec::new());
            }
            if let Some(xdg_surface) = self.xdg_surface {
                let _ = display.marshal_request(xdg_surface, XDG_SURFACE_DESTROY, Vec::new());
            }
            if let Some(surface) = self.surface {
                let _ = display.marshal_request(surface, SURFACE_DESTROY, Vec::new());
            }
            if let Some(buffer) = self.buffer {
                let _ = display.marshal_request(buffer, BUFFER_DESTROY, Vec::new());
            }
            if let Some(feedback) = self.feedback {
                let _ = display.marshal_request(feedback, FEEDBACK_DESTROY, Vec::new());
            }
            if let Some(dmabuf) = self.dmabuf {
                let _ = display.marshal_request(dmabuf, DMABUF_DESTROY, Vec::new());
            }
            let _ = display.flush();
        }

        if let Some(mut renderer) = self.renderer.take() {
            // SAFETY: the renderer was wrapped with `ManuallyDrop::new` in
            // `UiDisplay::new` and is taken exactly once here, so it has not
            // been dropped before. Dropping it releases the Vulkan frame
            // resources while `self.pipeline` (declared before `renderer`)
            // is still alive.
            unsafe {
                ManuallyDrop::drop(&mut renderer);
            }
        }
    }
}

/// Binds `interface` to the global of the same name announced by the registry.
fn bind(
    display: &mut WlClientDisplay<WlUnixTransport>,
    registry: WlProxyId,
    globals: &Globals,
    interface: &'static WlInterface,
) -> Result<WlProxyId, WlError> {
    let (name, version) = global_named(globals, interface.name)?;
    display.registry_bind(registry, name, interface, version.min(interface.version))
}

/// Globals bound from the registry together with the dmabuf modifier
/// listener state collected while the registry events arrive.
struct ProtocolGlobals {
    compositor: WlProxyId,
    wm_base: WlProxyId,
    dmabuf: WlProxyId,
    /// Modifiers advertised through the legacy `zwp_linux_dmabuf_v1`
    /// `modifier` event (used when no feedback object is available).
    legacy_modifiers: Rc<RefCell<Vec<u64>>>,
}

/// Connects to the Wayland session, creates the registry and performs the
/// first roundtrip so the compositor announces its globals.
///
/// The returned display is ready for [`bind_protocol_globals`].
///
/// # Errors
///
/// * [`UiDisplayError::Wayland`] — the connection, registry or roundtrip
///   failed.
fn connect(width: i32, height: i32) -> Result<Connection, UiDisplayError> {
    let transport = WlUnixTransport::connect_session().map_err(UiDisplayError::Wayland)?;
    let mut display = WlClientDisplay::connect(transport).map_err(UiDisplayError::Wayland)?;
    let registry = display
        .get_registry()
        .map_err(UiDisplayError::Wayland)?;
    info!("codevar: connected to Wayland compositor");
    let window_state: Rc<RefCell<WindowState>> = Rc::new(RefCell::new(WindowState::new(width, height)));
    let globals: Globals = Rc::new(RefCell::new(Vec::new()));
    {
        let globals = Rc::clone(&globals);
        display
            .add_registry_listener(registry, move |_, event| {
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
            })
            .map_err(UiDisplayError::Wayland)?;
    }
    display
        .roundtrip()
        .map_err(UiDisplayError::Wayland)?;
    Ok(Connection {
        display,
        registry,
        globals,
        window_state,
    })
}

/// Binds the `wl_compositor`, `xdg_wm_base` and `zwp_linux_dmabuf_v1`
/// globals and installs their listeners (wm-base ping/pong, dmabuf
/// modifier advertisement).
///
/// # Errors
///
/// * [`UiDisplayError::Wayland`] — a global is missing or a request or
///   listener registration failed.
fn bind_protocol_globals(
    display: &mut WlClientDisplay<WlUnixTransport>,
    registry: WlProxyId,
    globals: &Globals,
    window_state: &Rc<RefCell<WindowState>>,
) -> Result<ProtocolGlobals, UiDisplayError> {
    let compositor =
        bind(display, registry, globals, &COMPOSITOR_INTERFACE).map_err(UiDisplayError::Wayland)?;
    let wm_base =
        bind(display, registry, globals, &XDG_WM_BASE_INTERFACE).map_err(UiDisplayError::Wayland)?;

    let (dmabuf_name, dmabuf_version) =
        global_named(globals, "zwp_linux_dmabuf_v1").map_err(UiDisplayError::Wayland)?;
    let dmabuf = display
        .registry_bind(
            registry,
            dmabuf_name,
            &DMABUF_INTERFACE,
            dmabuf_version.min(DMABUF_INTERFACE.version),
        )
        .map_err(UiDisplayError::Wayland)?;

    {
        let window_state = Rc::clone(window_state);
        display
            .add_listener(wm_base, move |client, opcode, args| {
                if opcode == XDG_WM_BASE_PING
                    && let Some(WlArgument::Uint(serial)) = args.first()
                {
                    let serial = *serial;
                    if client
                        .marshal_request(wm_base, XDG_WM_BASE_PONG, vec![WlArgument::Uint(serial)])
                        .is_ok()
                    {
                        window_state.borrow_mut().pings += 1;
                    }
                }
                0
            })
            .map_err(UiDisplayError::Wayland)?;
    }

    let legacy_modifiers: Rc<RefCell<Vec<u64>>> = Rc::new(RefCell::new(Vec::new()));
    {
        let legacy_modifiers = Rc::clone(&legacy_modifiers);
        display
            .add_listener(dmabuf, move |_, opcode, args| {
                if opcode == DMABUF_MODIFIER
                    && let (
                        Some(WlArgument::Uint(format)),
                        Some(WlArgument::Uint(hi)),
                        Some(WlArgument::Uint(lo)),
                    ) = (args.first(), args.get(1), args.get(2))
                    && *format == DRM_FORMAT_XRGB8888
                {
                    let modifier = (u64::from(*hi) << 32) | u64::from(*lo);
                    let modifier = if modifier == u64::MAX { 0 } else { modifier };
                    legacy_modifiers.borrow_mut().push(modifier);
                }
                0
            })
            .map_err(UiDisplayError::Wayland)?;
    }

    Ok(ProtocolGlobals {
        compositor,
        wm_base,
        dmabuf,
        legacy_modifiers,
    })
}

/// Creates the `wl_surface`, `xdg_surface` and `xdg_toplevel`, installs
/// the configure listener on the `xdg_surface` and the close listener on
/// the `xdg_toplevel`, then publishes the window title, app id and the
/// initial commit.
///
/// The toplevel is created before the first commit so the surface
/// already has its role when it is committed, as required by xdg-shell.
///
/// Returns the `(surface, xdg_surface, toplevel)` triple used by
/// [`ack_configure`] and the render loop.
///
/// # Errors
///
/// * [`UiDisplayError::Wayland`] — object creation, listener
///   registration or the initial commit failed.
fn create_window_objects(
    display: &mut WlClientDisplay<WlUnixTransport>,
    compositor: WlProxyId,
    wm_base: WlProxyId,
    window_state: &Rc<RefCell<WindowState>>,
    init: WindowInit,
) -> Result<(WlProxyId, WlProxyId, WlProxyId), UiDisplayError> {
    let surface = display
        .marshal_new_id(compositor, COMPOSITOR_CREATE_SURFACE, Vec::new())
        .map_err(UiDisplayError::Wayland)?;
    let xdg_surface = display
        .marshal_new_id(
            wm_base,
            XDG_WM_BASE_GET_XDG_SURFACE,
            vec![WlArgument::Object(surface.id())],
        )
        .map_err(UiDisplayError::Wayland)?;

    {
        let window_state = Rc::clone(window_state);
        display
            .add_listener(xdg_surface, move |_, opcode, args| {
                if opcode == XDG_SURFACE_CONFIGURE
                    && let Some(WlArgument::Uint(serial)) = args.first()
                {
                    window_state.borrow_mut().configure_serial = *serial;
                    window_state.borrow_mut().configured = true;
                    if let (Some(WlArgument::Int(width)), Some(WlArgument::Int(height))) =
                        (args.get(1), args.get(2))
                        && *width > 0
                        && *height > 0
                    {
                        window_state.borrow_mut().width = *width;
                        window_state.borrow_mut().height = *height;
                    }
                }
                0
            })
            .map_err(UiDisplayError::Wayland)?;
    }

    let toplevel = create_toplevel(display, xdg_surface)?;

    {
        let window_state = Rc::clone(window_state);
        display
            .add_listener(toplevel, move |_, opcode, _| {
                if opcode == XDG_TOPLEVEL_CLOSE {
                    window_state.borrow_mut().closed = true;
                }
                0
            })
            .map_err(UiDisplayError::Wayland)?;
    }

    display
        .marshal_request(
            toplevel,
            XDG_TOPLEVEL_SET_TITLE,
            vec![WlArgument::Str(Some(
                init.title
                    .to_str()
                    .unwrap_or("Codevar")
                    .to_string(),
            ))],
        )
        .map_err(UiDisplayError::Wayland)?;
    display
        .marshal_request(
            toplevel,
            XDG_TOPLEVEL_SET_APP_ID,
            vec![WlArgument::Str(Some(
                init.app_id
                    .to_str()
                    .unwrap_or("dev.codevar.window")
                    .to_string(),
            ))],
        )
        .map_err(UiDisplayError::Wayland)?;
    display
        .marshal_request(surface, SURFACE_COMMIT, Vec::new())
        .map_err(UiDisplayError::Wayland)?;
    display.flush().map_err(UiDisplayError::Wayland)?;

    Ok((surface, xdg_surface, toplevel))
}

/// Dispatches events until the first `xdg_surface.configure` arrives.
///
/// # Errors
///
/// * [`UiDisplayError::WindowClosed`] — the compositor closed the window
///   before the first configure.
/// * [`UiDisplayError::Timeout`] — `deadline_ns` passed first.
/// * [`UiDisplayError::Wayland`] — dispatching failed.
fn wait_initial_configure(
    display: &mut WlClientDisplay<WlUnixTransport>,
    window_state: &Rc<RefCell<WindowState>>,
    deadline_ns: u64,
) -> Result<(), UiDisplayError> {
    while !window_state.borrow().configured {
        if window_state.borrow().closed {
            return Err(UiDisplayError::WindowClosed);
        }
        if SystemTime::monotonic_nanos() >= deadline_ns {
            return Err(UiDisplayError::Timeout(
                "waiting for xdg_surface.configure".to_string(),
            ));
        }
        display
            .dispatch(Some(Duration::from_millis(DISPATCH_TIMEOUT_MS)))
            .map_err(UiDisplayError::Wayland)?;
    }
    Ok(())
}

/// Acknowledges the first configure and publishes the window geometry.
///
/// # Errors
///
/// * [`UiDisplayError::Wayland`] — the ack or geometry request failed.
fn ack_configure(
    display: &mut WlClientDisplay<WlUnixTransport>,
    xdg_surface: WlProxyId,
    window_state: &Rc<RefCell<WindowState>>,
) -> Result<(), UiDisplayError> {
    let width = window_state.borrow().width;
    let height = window_state.borrow().height;
    let serial = window_state.borrow().configure_serial;
    display
        .marshal_request(
            xdg_surface,
            XDG_SURFACE_ACK_CONFIGURE,
            vec![WlArgument::Uint(serial)],
        )
        .map_err(UiDisplayError::Wayland)?;
    display
        .marshal_request(
            xdg_surface,
            XDG_SURFACE_SET_WINDOW_GEOMETRY,
            vec![
                WlArgument::Int(0),
                WlArgument::Int(0),
                WlArgument::Int(width),
                WlArgument::Int(height),
            ],
        )
        .map_err(UiDisplayError::Wayland)?;
    Ok(())
}

/// Collects the `DRM_FORMAT_XRGB8888` modifiers the compositor accepts
/// and returns them together with the feedback proxy (when one was
/// created).
///
/// Prefers the linux-dmabuf feedback format table (dmabuf v4+), falls
/// back to the legacy `modifier` events, and requests only the linear
/// modifier when the compositor advertises it.
///
/// # Errors
///
/// * [`UiDisplayError::Wayland`] — the feedback object could not be
///   created or dispatched.
/// * [`UiDisplayError::Timeout`] — no modifier at all was advertised.
fn negotiate_modifiers(
    display: &mut WlClientDisplay<WlUnixTransport>,
    dmabuf: WlProxyId,
    legacy_modifiers: &Rc<RefCell<Vec<u64>>>,
) -> Result<(Vec<u64>, Option<WlProxyId>), UiDisplayError> {
    let feedback_state: Rc<RefCell<FeedbackState>> = Rc::new(RefCell::new(FeedbackState::default()));
    let mut feedback_proxy: Option<WlProxyId> = None;
    if display.proxy_version(dmabuf).unwrap_or(0) >= 4 {
        let proxy = display
            .marshal_new_id(dmabuf, DMABUF_GET_DEFAULT_FEEDBACK, Vec::new())
            .map_err(UiDisplayError::Wayland)?;
        {
            let feedback_state = Rc::clone(&feedback_state);
            display
                .add_listener(proxy, move |_, opcode, args: &mut [WlArgument]| {
                    let mut state = feedback_state.borrow_mut();
                    match opcode {
                        FEEDBACK_FORMAT_TABLE => {
                            let fd = args.get_mut(0).and_then(WlArgument::take_fd);
                            let size = match args.get(1) {
                                Some(WlArgument::Uint(size)) => *size as usize,
                                _ => 0,
                            };
                            if let Some(fd) = fd {
                                state.format_table = read_format_table(fd, size);
                            }
                        }
                        FEEDBACK_MAIN_DEVICE
                        | FEEDBACK_TRANCHE_TARGET_DEVICE
                        | FEEDBACK_TRANCHE_FLAGS
                        | FEEDBACK_TRANCHE_DONE => {}
                        FEEDBACK_TRANCHE_FORMATS => {
                            let bytes = array_arg(args, 0);
                            state.tranche_indices.extend(u16_array(&bytes));
                        }
                        FEEDBACK_DONE => state.done = true,
                        _ => {}
                    }
                    0
                })
                .map_err(UiDisplayError::Wayland)?;
        }
        feedback_proxy = Some(proxy);
        display
            .roundtrip()
            .map_err(UiDisplayError::Wayland)?;
    }

    let modifiers = {
        let state = feedback_state.borrow();
        preferred_modifiers(&state.format_table, &state.tranche_indices)
    };
    let modifiers = if modifiers.is_empty() {
        legacy_modifiers.borrow().clone()
    } else {
        modifiers
    };
    if modifiers.is_empty() {
        return Err(UiDisplayError::Timeout(
            "the compositor advertised no modifier for DRM_FORMAT_XRGB8888".to_string(),
        ));
    }

    let linear = [0u64];
    let requested: Vec<u64> = if modifiers.contains(&0) {
        linear.to_vec()
    } else {
        modifiers
    };
    Ok((requested, feedback_proxy))
}

/// Creates the frame orchestrator borrowing `pipeline` for the display's
/// lifetime.
///
/// # Safety of the returned value
///
/// The renderer borrows from `pipeline` which is stored in
/// `UiDisplay::pipeline`. Both are owned by `UiDisplay` and `pipeline`
/// outlives the renderer. The lifetime is transmuted to `'static` because
/// the struct's ownership guarantees validity. `Drop` drops the renderer
/// before the pipeline, preserving the borrow invariant.
fn create_renderer(pipeline: &PipelineContext) -> ManuallyDrop<RendererSubsystem<'static>> {
    let renderer = RendererSubsystem::new(pipeline);
    // SAFETY: see the function documentation above.
    unsafe {
        ManuallyDrop::new(core::mem::transmute::<
            RendererSubsystem<'_>,
            RendererSubsystem<'static>,
        >(renderer))
    }
}

/// Exports the pipeline's render target as a `wl_buffer` through the
/// linux-dmabuf protocol and installs the `wl_buffer.release` listener
/// that drives the render loop's pacing.
///
/// # Errors
///
/// * [`UiDisplayError::Wayland`] — any params or flush request failed.
fn create_wl_buffer(
    display: &mut WlClientDisplay<WlUnixTransport>,
    dmabuf: WlProxyId,
    pipeline: &PipelineContext,
    width: i32,
    height: i32,
    window_state: &Rc<RefCell<WindowState>>,
) -> Result<WlProxyId, UiDisplayError> {
    let target = pipeline.render_target();
    let params = display
        .marshal_new_id(dmabuf, DMABUF_CREATE_PARAMS, Vec::new())
        .map_err(UiDisplayError::Wayland)?;
    display
        .marshal_request(
            params,
            BUFFER_PARAMS_ADD,
            vec![
                WlArgument::Fd(target.dmabuf_fd.as_raw()),
                WlArgument::Uint(0),
                WlArgument::Uint(target.offset),
                WlArgument::Uint(target.stride),
                WlArgument::Uint((target.modifier >> 32) as u32),
                WlArgument::Uint(target.modifier as u32),
            ],
        )
        .map_err(UiDisplayError::Wayland)?;
    let buffer = display
        .marshal_new_id(
            params,
            BUFFER_PARAMS_CREATE_IMMED,
            vec![
                WlArgument::Int(width),
                WlArgument::Int(height),
                WlArgument::Uint(target.drm_format),
                WlArgument::Uint(0),
            ],
        )
        .map_err(UiDisplayError::Wayland)?;
    display
        .marshal_request(params, BUFFER_PARAMS_DESTROY, Vec::new())
        .map_err(UiDisplayError::Wayland)?;
    display.flush().map_err(UiDisplayError::Wayland)?;

    {
        let window_state = Rc::clone(window_state);
        display
            .add_listener(buffer, move |_, opcode, _| {
                if opcode == BUFFER_RELEASE {
                    window_state.borrow_mut().buffer_released = true;
                }
                0
            })
            .map_err(UiDisplayError::Wayland)?;
    }
    Ok(buffer)
}

/// Creates the `xdg_toplevel` role object for `xdg_surface`.
///
/// `get_toplevel` takes no arguments: the sender is the `xdg_surface`
/// and the new id is inserted by [`WlClientDisplay::marshal_new_id`].
///
/// # Errors
///
/// * [`UiDisplayError::Wayland`] — the request failed.
fn create_toplevel(
    display: &mut WlClientDisplay<WlUnixTransport>,
    xdg_surface: WlProxyId,
) -> Result<WlProxyId, UiDisplayError> {
    display
        .marshal_new_id(xdg_surface, XDG_SURFACE_GET_TOPLEVEL, Vec::new())
        .map_err(UiDisplayError::Wayland)
}

/// Returns `(name, version)` of the global called `interface`.
fn global_named(globals: &Globals, interface: &str) -> Result<(u32, u32), WlError> {
    globals
        .borrow()
        .iter()
        .find(|(_, announced, _)| announced == interface)
        .map(|(name, _, version)| (*name, *version))
        .ok_or_else(|| WlError::Unsupported(format!("the compositor did not announce {interface}")))
}

/// Copies the byte array of `args[index]`, or an empty vector.
fn array_arg(args: &[WlArgument], index: usize) -> Vec<u8> {
    match args.get(index) {
        Some(WlArgument::Array(Some(array))) => array.as_bytes().to_vec(),
        _ => Vec::new(),
    }
}

/// Decodes a `tranche_formats` payload as native-endian `u16` indices.
fn u16_array(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(2)
        .map(|word| u32::from(u16::from_ne_bytes([word[0], word[1]])))
        .collect()
}

/// Reads the compositor's `format_table` payload into a fresh buffer.
fn read_format_table(fd: libc::c_int, size: usize) -> Vec<u8> {
    let owned = unsafe { OwnedFd::from_raw(fd) };
    let size = size.min(MAX_FORMAT_TABLE_BYTES);
    let mut bytes = vec![0u8; size];
    let mut total = 0usize;
    while total < size {
        let count = unsafe {
            libc::pread(
                owned.as_raw(),
                bytes[total..].as_mut_ptr().cast(),
                size - total,
                total as libc::off_t,
            )
        };
        if count > 0 {
            total += count as usize;
            continue;
        }
        if count == 0 {
            break;
        }
        break;
    }
    bytes.truncate(total);
    bytes
}

/// Parses the 16-byte entries of a `format_table` into `(fourcc, modifier)` pairs.
fn parse_format_table(table: &[u8]) -> Vec<(u32, u64)> {
    table
        .chunks_exact(16)
        .map(|entry| {
            let format = u32::from_ne_bytes([entry[0], entry[1], entry[2], entry[3]]);
            let modifier = u64::from_ne_bytes([
                entry[8], entry[9], entry[10], entry[11], entry[12], entry[13], entry[14], entry[15],
            ]);
            (format, modifier)
        })
        .collect()
}

/// Returns the `DRM_FORMAT_XRGB8888` modifiers the compositor prefers, in tranche order.
fn preferred_modifiers(table: &[u8], tranche_indices: &[u32]) -> Vec<u64> {
    let entries = parse_format_table(table);
    let mut modifiers: Vec<u64> = Vec::new();
    if tranche_indices.is_empty() {
        for &(format, modifier) in &entries {
            if format == DRM_FORMAT_XRGB8888 && !modifiers.contains(&modifier) {
                modifiers.push(modifier);
            }
        }
        return modifiers;
    }
    for &index in tranche_indices {
        if let Some(&(format, modifier)) = entries.get(index as usize)
            && format == DRM_FORMAT_XRGB8888
            && !modifiers.contains(&modifier)
        {
            modifiers.push(modifier);
        }
    }
    modifiers
}

/// Blocks until the GPU finished the frame the `sync_file` represents.
fn wait_for_gpu(sync_file: &OwnedFd) -> Result<(), UiDisplayError> {
    let mut descriptor = libc::pollfd {
        fd: sync_file.as_raw(),
        events: libc::POLLIN,
        revents: 0,
    };
    let ready = unsafe { libc::poll(&mut descriptor, 1, GPU_SYNC_TIMEOUT_MS) };
    if ready <= 0 {
        return Err(UiDisplayError::Timeout(
            "timed out waiting for the GPU".to_string(),
        ));
    }
    Ok(())
}

/// Raw `zwp_linux_dmabuf_feedback_v1` payload.
#[derive(Default)]
struct FeedbackState {
    format_table: Vec<u8>,
    tranche_indices: Vec<u32>,
    done: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_table_entries_are_decoded() {
        let mut table = Vec::new();
        table.extend_from_slice(&DRM_FORMAT_XRGB8888.to_ne_bytes());
        table.extend_from_slice(&0u32.to_ne_bytes());
        table.extend_from_slice(&0u64.to_ne_bytes());
        table.extend_from_slice(&0xDEAD_BEEFu32.to_ne_bytes());
        table.extend_from_slice(&0u32.to_ne_bytes());
        table.extend_from_slice(&0x00E0_0000_0000_0001u64.to_ne_bytes());
        assert_eq!(
            parse_format_table(&table),
            [(DRM_FORMAT_XRGB8888, 0), (0xDEAD_BEEF, 0x00E0_0000_0000_0001)]
        );
        assert!(parse_format_table(&table[..15]).is_empty());
    }

    #[test]
    fn preferred_modifiers_follow_the_tranche() {
        let mut table = Vec::new();
        for (format, modifier) in [
            (DRM_FORMAT_XRGB8888, 0u64),
            (0x3432_5241, 0x00E0_0000_0000_0001u64),
            (DRM_FORMAT_XRGB8888, 0x00E0_0000_0000_0002u64),
            (DRM_FORMAT_XRGB8888, 0u64),
        ] {
            table.extend_from_slice(&format.to_ne_bytes());
            table.extend_from_slice(&0u32.to_ne_bytes());
            table.extend_from_slice(&modifier.to_ne_bytes());
        }
        assert_eq!(
            preferred_modifiers(&table, &[2, 0, 3]),
            vec![0x00E0_0000_0000_0002, 0]
        );
        assert_eq!(preferred_modifiers(&table, &[]), vec![0, 0x00E0_0000_0000_0002]);
    }
}
