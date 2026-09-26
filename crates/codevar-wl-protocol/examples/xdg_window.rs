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

//! Maps an animated window on the compositor of the current session.
//!
//! The example performs the full `xdg-shell` handshake —
//! `xdg_wm_base.get_xdg_surface`, `get_toplevel`, the empty first
//! commit, `ack_configure` — and then renders an animated gradient
//! into an `wl_shm` buffer, repainting once per frame callback. It
//! answers `xdg_wm_base.ping`, stops when the compositor closes the
//! window and prints a short summary.
//!
//! Run it inside a Wayland session:
//!
//! ```text
//! cargo run -p codevar-wl-protocol --example xdg_window
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use codevar_wl_protocol::{
    BUFFER_DESTROY, COMPOSITOR_CREATE_SURFACE, COMPOSITOR_INTERFACE, SHM_CREATE_POOL, SHM_FORMAT_XRGB8888,
    SHM_INTERFACE, SHM_POOL_CREATE_BUFFER, SHM_POOL_DESTROY, SURFACE_ATTACH, SURFACE_COMMIT, SURFACE_DAMAGE,
    SURFACE_DESTROY, SURFACE_FRAME, WlArgument, WlClientDisplay, WlError, WlProxyId, WlRegistryEvent,
    WlResult, WlUnixTransport, XDG_SURFACE_ACK_CONFIGURE, XDG_SURFACE_CONFIGURE, XDG_SURFACE_DESTROY,
    XDG_SURFACE_GET_TOPLEVEL, XDG_SURFACE_SET_WINDOW_GEOMETRY, XDG_TOPLEVEL_CLOSE, XDG_TOPLEVEL_DESTROY,
    XDG_TOPLEVEL_SET_APP_ID, XDG_TOPLEVEL_SET_TITLE, XDG_WM_BASE_GET_XDG_SURFACE, XDG_WM_BASE_INTERFACE,
    XDG_WM_BASE_PING, XDG_WM_BASE_PONG,
};

/// Width of the demo window in surface local pixels.
const WIDTH: i32 = 640;
/// Height of the demo window in surface local pixels.
const HEIGHT: i32 = 480;
/// Frame callbacks to render before the demo exits.
const FRAME_BUDGET: u32 = 120;
/// Hard stop so a silent compositor cannot hang the example.
const DEADLINE: Duration = Duration::from_secs(15);

fn main() {
    match run() {
        Ok(summary) => println!("{summary}"),
        Err(error) => {
            eprintln!("xdg window demo failed: {error}");
            std::process::exit(1);
        }
    }
}

fn run() -> WlResult<String> {
    let mut display = WlClientDisplay::connect(WlUnixTransport::connect_session()?)?;

    let registry = display.get_registry()?;
    let globals = Rc::new(RefCell::new(Vec::new()));
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
    display.roundtrip()?;

    let compositor = bind(&mut display, registry, &globals, &COMPOSITOR_INTERFACE)?;
    let shm = bind(&mut display, registry, &globals, &SHM_INTERFACE)?;
    let wm_base = bind(&mut display, registry, &globals, &XDG_WM_BASE_INTERFACE)?;

    let pings = Rc::new(Cell::new(0u32));
    {
        let pings = Rc::clone(&pings);
        display.add_listener(wm_base, move |client, opcode, args| {
            if opcode == XDG_WM_BASE_PING
                && let Some(WlArgument::Uint(serial)) = args.first()
            {
                let serial = *serial;
                let result =
                    client.marshal_request(wm_base, XDG_WM_BASE_PONG, vec![WlArgument::Uint(serial)]);
                if result.is_ok() {
                    pings.set(pings.get() + 1);
                }
            }
            0
        })?;
    }

    let surface = display.marshal_new_id(compositor, COMPOSITOR_CREATE_SURFACE, Vec::new())?;
    let xdg_surface = display.marshal_new_id(
        wm_base,
        XDG_WM_BASE_GET_XDG_SURFACE,
        vec![WlArgument::Object(surface.id())],
    )?;

    let configure_serial = Rc::new(Cell::new(None));
    {
        let configure_serial = Rc::clone(&configure_serial);
        display.add_listener(xdg_surface, move |_, opcode, args| {
            if opcode == XDG_SURFACE_CONFIGURE
                && let Some(WlArgument::Uint(serial)) = args.first()
            {
                configure_serial.set(Some(*serial));
            }
            0
        })?;
    }

    let closed = Rc::new(Cell::new(false));
    let toplevel = display.marshal_new_id(xdg_surface, XDG_SURFACE_GET_TOPLEVEL, Vec::new())?;
    {
        let closed = Rc::clone(&closed);
        display.add_listener(toplevel, move |_, opcode, _| {
            if opcode == XDG_TOPLEVEL_CLOSE {
                closed.set(true);
            }
            0
        })?;
    }

    display.marshal_request(
        toplevel,
        XDG_TOPLEVEL_SET_TITLE,
        vec![WlArgument::Str(Some(String::from("Codevar xdg window demo")))],
    )?;
    display.marshal_request(
        toplevel,
        XDG_TOPLEVEL_SET_APP_ID,
        vec![WlArgument::Str(Some(String::from("dev.codevar.xdg-window")))],
    )?;

    // The first commit asks the compositor for the initial configure.
    display.marshal_request(surface, SURFACE_COMMIT, Vec::new())?;
    display.flush()?;

    let deadline = Instant::now() + DEADLINE;
    while configure_serial.get().is_none() {
        if closed.get() {
            return Err(WlError::InvalidState(String::from(
                "the compositor closed the window before the first configure",
            )));
        }
        if Instant::now() >= deadline {
            return Err(WlError::InvalidState(String::from(
                "timed out waiting for xdg_surface.configure",
            )));
        }
        display.dispatch(Some(Duration::from_millis(50)))?;
    }
    let Some(serial) = configure_serial.get() else {
        return Err(WlError::InvalidState(String::from(
            "the configure listener ran without a serial",
        )));
    };

    display.marshal_request(
        xdg_surface,
        XDG_SURFACE_ACK_CONFIGURE,
        vec![WlArgument::Uint(serial)],
    )?;
    display.marshal_request(
        xdg_surface,
        XDG_SURFACE_SET_WINDOW_GEOMETRY,
        vec![
            WlArgument::Int(0),
            WlArgument::Int(0),
            WlArgument::Int(WIDTH),
            WlArgument::Int(HEIGHT),
        ],
    )?;

    let mut pool = create_pool(&mut display, shm)?;
    let buffer = display.marshal_new_id(
        pool.proxy,
        SHM_POOL_CREATE_BUFFER,
        vec![
            WlArgument::Int(0),
            WlArgument::Int(WIDTH),
            WlArgument::Int(HEIGHT),
            WlArgument::Int(WIDTH * 4),
            WlArgument::Uint(SHM_FORMAT_XRGB8888),
        ],
    )?;

    // Safety: the pool mapping is live and `paint` stays inside it.
    unsafe { pool.paint(0) };

    let mut frames = 0u32;
    while frames < FRAME_BUDGET && !closed.get() && Instant::now() < deadline {
        // One batch: attach, damage, frame, commit — the callback must
        // already be pending when the compositor processes the commit,
        // matching what reference clients (weston simple-shm) do.
        let frame = queue_frame(&mut display, surface, buffer)?;
        let frame_done = Rc::new(Cell::new(false));
        {
            let frame_done = Rc::clone(&frame_done);
            display.add_callback_listener(frame, move |_, _| {
                frame_done.set(true);
                0
            })?;
        }
        display.flush()?;
        let mut spins = 0u32;
        while !frame_done.get() && !closed.get() && Instant::now() < deadline {
            let seen = display.dispatch(Some(Duration::from_millis(50)))?;
            spins += 1;
            if std::env::var_os("CODEVAR_XDG_DEBUG").is_some() {
                eprintln!("[dbg] frame={frames} spin={spins} seen={seen} done={} closed={}", frame_done.get(), closed.get());
            }
        }
        if !frame_done.get() {
            break;
        }
        display.proxy_destroy(frame)?;
        frames += 1;
        // Safety: the pool mapping is live and `paint` stays inside it.
        unsafe { pool.paint(frames) };
    }
    display.flush()?;

    // Best effort teardown, in the order the protocol prescribes.
    let _ = display.marshal_request(toplevel, XDG_TOPLEVEL_DESTROY, Vec::new());
    let _ = display.marshal_request(xdg_surface, XDG_SURFACE_DESTROY, Vec::new());
    let _ = display.marshal_request(surface, SURFACE_DESTROY, Vec::new());
    let _ = display.marshal_request(buffer, BUFFER_DESTROY, Vec::new());
    let _ = display.marshal_request(pool.proxy, SHM_POOL_DESTROY, Vec::new());
    let _ = display.flush();

    Ok(format!(
        "xdg window demo: {frames} frames rendered in {WIDTH}x{HEIGHT}, {} pings answered, configure serial {serial}",
        pings.get()
    ))
}

/// Registry announcements as `(name, interface, version)` tuples.
type Globals = Rc<RefCell<Vec<(u32, String, u32)>>>;

/// Binds `interface` to the global of the same name announced by the registry.
fn bind(
    display: &mut WlClientDisplay<WlUnixTransport>,
    registry: WlProxyId,
    globals: &Globals,
    interface: &'static codevar_wl_protocol::WlInterface,
) -> WlResult<WlProxyId> {
    let (name, version) = globals
        .borrow()
        .iter()
        .find(|(_, announced, _)| announced == interface.name)
        .map(|(name, _, version)| (*name, *version))
        .ok_or_else(|| WlError::Unsupported(format!("the compositor did not announce {}", interface.name)))?;
    display.registry_bind(registry, name, interface, version.min(interface.version))
}

/// Queues the frame the mapped buffer will show next in a single batch:
/// `attach`, `damage`, `frame` and `commit` are written to the socket
/// together, so the frame callback is pending before the compositor
/// processes the commit that triggers the repaint.
fn queue_frame(
    display: &mut WlClientDisplay<WlUnixTransport>,
    surface: WlProxyId,
    buffer: WlProxyId,
) -> WlResult<WlProxyId> {
    display.marshal_request(
        surface,
        SURFACE_ATTACH,
        vec![
            WlArgument::Object(buffer.id()),
            WlArgument::Int(0),
            WlArgument::Int(0),
        ],
    )?;
    display.marshal_request(
        surface,
        SURFACE_DAMAGE,
        vec![
            WlArgument::Int(0),
            WlArgument::Int(0),
            WlArgument::Int(WIDTH),
            WlArgument::Int(HEIGHT),
        ],
    )?;
    let frame = display.marshal_new_id(surface, SURFACE_FRAME, Vec::new())?;
    display.marshal_request(surface, SURFACE_COMMIT, Vec::new())?;
    Ok(frame)
}

/// A shared memory pool mapped into this process.
struct Pool {
    proxy: WlProxyId,
    fd: libc::c_int,
    bytes: *mut u8,
    len: usize,
}

impl Pool {
    /// Refills the mapping with an animated gradient for `frame`.
    ///
    /// # Safety
    ///
    /// The pointer covers `len` bytes of the live shared mapping of
    /// `fd` and the method never writes past `len`.
    unsafe fn paint(&mut self, frame: u32) {
        // Safety: guaranteed by the method contract above.
        let pixels = unsafe { core::slice::from_raw_parts_mut(self.bytes, self.len) };
        let shift = frame as usize;
        for y in 0..HEIGHT as usize {
            for x in 0..WIDTH as usize {
                let index = (y * WIDTH as usize + x) * 4;
                pixels[index] = ((x + shift) & 0xff) as u8;
                pixels[index + 1] = ((y + shift * 2) & 0xff) as u8;
                pixels[index + 2] = (((x + y) / 2 + shift * 3) & 0xff) as u8;
                pixels[index + 3] = 0xff;
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // Safety: `bytes` and `len` describe the mapping installed for
        // `fd` in `create_pool` and both are released exactly once.
        unsafe {
            libc::munmap(self.bytes.cast::<libc::c_void>(), self.len);
            libc::close(self.fd);
        }
    }
}

/// Creates a memfd backed `wl_shm_pool` covering the demo window.
fn create_pool(display: &mut WlClientDisplay<WlUnixTransport>, shm: WlProxyId) -> WlResult<Pool> {
    let len = (WIDTH as usize) * (HEIGHT as usize) * 4;
    // Safety: the name is a valid C string and the flags are defined.
    let fd = unsafe { libc::memfd_create(c"codevar-xdg-demo".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(WlError::io(format!(
            "memfd_create failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    // Safety: `fd` is open and `len` fits into `off_t`.
    if unsafe { libc::ftruncate(fd, len as libc::off_t) } != 0 {
        let message = format!("ftruncate failed: {}", std::io::Error::last_os_error());
        // Safety: `fd` is open and owned by this branch.
        unsafe { libc::close(fd) };
        return Err(WlError::io(message));
    }
    // Safety: maps `len` writable shared bytes backed by `fd`.
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
        let message = format!("mmap failed: {}", std::io::Error::last_os_error());
        // Safety: `fd` is open and owned by this branch.
        unsafe { libc::close(fd) };
        return Err(WlError::io(message));
    }
    let proxy = display.marshal_new_id(
        shm,
        SHM_CREATE_POOL,
        vec![WlArgument::Fd(fd), WlArgument::Int(len as i32)],
    );
    let proxy = match proxy {
        Ok(proxy) => proxy,
        Err(error) => {
            // Safety: unmapping the fresh mapping and closing `fd`,
            // both owned by this branch until the pool exists.
            unsafe {
                libc::munmap(mapped, len);
                libc::close(fd);
            }
            return Err(error);
        }
    };
    // The descriptor stays open until the pool drops: the wire copy is
    // only taken at the first flush after `create_pool`.
    Ok(Pool {
        proxy,
        fd,
        bytes: mapped.cast::<u8>(),
        len,
    })
}
