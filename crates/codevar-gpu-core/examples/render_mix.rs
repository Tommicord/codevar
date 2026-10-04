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

//! Renders a three-pipe pixel-mixing test into a real Wayland window.
//!
//! Like `render_hit`, the example drives the dmabuf path
//! (Wayland handshake plus a [`PipelineContext`]), but the frame itself
//! is composited by [`Compositor`] from three independent pipes rather
//! than a single layer:
//!
//! * [`SolidPipe`] — framebuffer 0: a full-screen red base. The mix
//!   shader starts from framebuffer 0, so its weight is ignored.
//! * [`SplitPipe`] (green/yellow halves, weight 0.5) — mixed over the
//!   base, the left half becomes `(0.5, 0.5, 0)` and the right half
//!   `(1.0, 0.5, 0)`.
//! * [`SplitPipe`] (cyan/magenta halves, weight 0.25) — folds each half
//!   again, leaving the four quadrants in four different colors:
//!
//!   | quadrant       | mixed color          |
//!   |----------------|----------------------|
//!   | top-left       | `(0.375, 0.625, 0.25)` |
//!   | top-right      | `(0.750, 0.625, 0.25)` |
//!   | bottom-left    | `(0.500, 0.375, 0.25)` |
//!   | bottom-right   | `(0.875, 0.375, 0.25)` |
//!
//! Every pipe renders into its own [`OffscreenTarget`]; the compositor's
//! layout pass transitions them, the mix pass folds them into the
//! presentation target with the weights above, and the result is
//! committed as the window's dma-buf buffer.
//!
//! Run it inside a Wayland session:
//!
//! ```text
//! cargo run -p codevar-gpu-core --example render_mix
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use ash::vk;
use codevar_gpu_core::ui_base::{
    Compositor, CompositorError, OffscreenFramebuffer, OffscreenTarget, PipeCtx, PipeFuture, PipeOutcome,
    PipeSource, PipeSupplyTraits, block_on,
};
use codevar_gpu_core::ui_pipeline::{OwnedFd, PipelineContext};
use codevar_wl_protocol::{
    BUFFER_DESTROY, BUFFER_PARAMS_ADD, BUFFER_PARAMS_CREATE_IMMED, BUFFER_PARAMS_DESTROY, BUFFER_RELEASE,
    COMPOSITOR_CREATE_SURFACE, COMPOSITOR_INTERFACE, DMABUF_CREATE_PARAMS, DMABUF_DESTROY,
    DMABUF_GET_DEFAULT_FEEDBACK, DMABUF_INTERFACE, DMABUF_MODIFIER, DRM_FORMAT_XRGB8888, FEEDBACK_DESTROY,
    FEEDBACK_DONE, FEEDBACK_FORMAT_TABLE, FEEDBACK_MAIN_DEVICE, FEEDBACK_TRANCHE_DONE,
    FEEDBACK_TRANCHE_FLAGS, FEEDBACK_TRANCHE_FORMATS, FEEDBACK_TRANCHE_TARGET_DEVICE, SURFACE_ATTACH,
    SURFACE_COMMIT, SURFACE_DAMAGE, SURFACE_DESTROY, SURFACE_FRAME, WlArgument, WlClientDisplay, WlError,
    WlInterface, WlProxyId, WlRegistryEvent, WlResult, WlUnixTransport, XDG_SURFACE_ACK_CONFIGURE,
    XDG_SURFACE_CONFIGURE, XDG_SURFACE_DESTROY, XDG_SURFACE_GET_TOPLEVEL, XDG_SURFACE_SET_WINDOW_GEOMETRY,
    XDG_TOPLEVEL_CLOSE, XDG_TOPLEVEL_DESTROY, XDG_TOPLEVEL_SET_APP_ID, XDG_TOPLEVEL_SET_TITLE,
    XDG_WM_BASE_GET_XDG_SURFACE, XDG_WM_BASE_INTERFACE, XDG_WM_BASE_PING, XDG_WM_BASE_PONG,
};

/// Initial window width in surface local pixels (used when the first
/// configure carries `0x0`).
const WIDTH: i32 = 640;
/// Initial window height in surface local pixels.
const HEIGHT: i32 = 480;
/// Frames to render before the example exits on its own.
const FRAME_BUDGET: u32 = 900;
/// Hard stop so a silent compositor cannot hang the example.
const DEADLINE: Duration = Duration::from_secs(30);
/// How long a single `dispatch` waits for Wayland events.
const DISPATCH_TIMEOUT: Duration = Duration::from_millis(50);
/// How long to wait for `wl_buffer.release` before re-rendering anyway
/// (a compositor that never releases single-buffered clients would
/// otherwise stall the demo; the fallback costs at most one torn frame).
const RELEASE_TIMEOUT: Duration = Duration::from_millis(500);
/// Milliseconds the example waits for the GPU through the exported
/// `sync_file` before committing the buffer.
const GPU_SYNC_TIMEOUT_MS: i32 = 5_000;
/// Largest format table the example is willing to allocate.
const MAX_FORMAT_TABLE_BYTES: usize = 1 << 20;

/// Base pipe color: solid red across the whole target.
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
/// First overlay, left half.
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
/// First overlay, right half.
const YELLOW: [f32; 4] = [1.0, 1.0, 0.0, 1.0];
/// Second overlay, top half.
const CYAN: [f32; 4] = [0.0, 1.0, 1.0, 1.0];
/// Second overlay, bottom half.
const MAGENTA: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
/// Mix weight of the left/right overlay.
const WEIGHT_HALF: f32 = 0.5;
/// Mix weight of the top/bottom overlay.
const WEIGHT_QUARTER: f32 = 0.25;

/// Every failure of the example is reported as a boxed error so `?` works
/// for Wayland, Vulkan and compositor errors alike.
type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    match run() {
        Ok(summary) => println!("{summary}"),
        Err(_) => {
            std::process::exit(1);
        }
    }
}

/// Drives the whole example: Wayland window, Vulkan pipeline, three
/// compositor pipes, render loop and teardown.
fn run() -> ExampleResult<String> {
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
    let wm_base = bind(&mut display, registry, &globals, &XDG_WM_BASE_INTERFACE)?;
    let (dmabuf_name, dmabuf_version) = global_named(&globals, "zwp_linux_dmabuf_v1")?;
    let dmabuf = display.registry_bind(
        registry,
        dmabuf_name,
        &DMABUF_INTERFACE,
        dmabuf_version.min(DMABUF_INTERFACE.version),
    )?;
    let pings = Rc::new(Cell::new(0u32));
    {
        let pings = Rc::clone(&pings);
        display.add_listener(wm_base, move |client, opcode, args| {
            if opcode == XDG_WM_BASE_PING
                && let Some(WlArgument::Uint(serial)) = args.first()
            {
                let serial = *serial;
                let answered =
                    client.marshal_request(wm_base, XDG_WM_BASE_PONG, vec![WlArgument::Uint(serial)]);
                if answered.is_ok() {
                    pings.set(pings.get() + 1);
                }
            }
            0
        })?;
    }
    let legacy_modifiers = Rc::new(RefCell::new(Vec::<u64>::new()));
    {
        let legacy_modifiers = Rc::clone(&legacy_modifiers);
        display.add_listener(dmabuf, move |_, opcode, args| {
            if opcode == DMABUF_MODIFIER
                && let (
                    Some(WlArgument::Uint(format)),
                    Some(WlArgument::Uint(hi)),
                    Some(WlArgument::Uint(lo)),
                ) = (args.first(), args.get(1), args.get(2))
                && *format == DRM_FORMAT_XRGB8888
            {
                let modifier = (u64::from(*hi) << 32) | u64::from(*lo);
                // DRM_FORMAT_MOD_INVALID means "any implicit modifier";
                // the only explicit modifier that always works is linear.
                let modifier = if modifier == u64::MAX { 0 } else { modifier };
                let mut modifiers = legacy_modifiers.borrow_mut();
                if !modifiers.contains(&modifier) {
                    modifiers.push(modifier);
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
    let window_size = Rc::new(Cell::new((WIDTH, HEIGHT)));
    let configure_serial = Rc::new(Cell::new(None));
    {
        let window_size = Rc::clone(&window_size);
        let configure_serial = Rc::clone(&configure_serial);
        display.add_listener(xdg_surface, move |_, opcode, args| {
            if opcode == XDG_SURFACE_CONFIGURE
                && let Some(WlArgument::Uint(serial)) = args.first()
            {
                configure_serial.set(Some(*serial));
                // `0x0` means "the client decides the size".
                if let (Some(WlArgument::Int(width)), Some(WlArgument::Int(height))) =
                    (args.get(1), args.get(2))
                    && *width > 0
                    && *height > 0
                {
                    window_size.set((*width, *height));
                }
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
        vec![WlArgument::Str(Some(String::from("Codevar render_mix demo")))],
    )?;
    display.marshal_request(
        toplevel,
        XDG_TOPLEVEL_SET_APP_ID,
        vec![WlArgument::Str(Some(String::from("dev.codevar.render-mix")))],
    )?;

    // The first commit asks the compositor for the initial configure.
    display.marshal_request(surface, SURFACE_COMMIT, Vec::new())?;
    display.flush()?;

    let deadline = Instant::now() + DEADLINE;
    while configure_serial.get().is_none() {
        if closed.get() {
            return Err(WlError::InvalidState(String::from(
                "the compositor closed the window before the first configure",
            ))
            .into());
        }
        if Instant::now() >= deadline {
            return Err(
                WlError::InvalidState(String::from("timed out waiting for xdg_surface.configure")).into(),
            );
        }
        display.dispatch(Some(DISPATCH_TIMEOUT))?;
    }
    let (width, height) = window_size.get();
    let serial = configure_serial.get().unwrap_or(0);
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
            WlArgument::Int(width),
            WlArgument::Int(height),
        ],
    )?;
    let feedback = Rc::new(RefCell::new(Feedback::default()));
    let mut feedback_proxy = None;
    if display.proxy_version(dmabuf).unwrap_or(0) >= 4 {
        let proxy = display.marshal_new_id(dmabuf, DMABUF_GET_DEFAULT_FEEDBACK, Vec::new())?;
        {
            let feedback = Rc::clone(&feedback);
            display.add_listener(proxy, move |_, opcode, args| {
                let mut state = feedback.borrow_mut();
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
            })?;
        }
        feedback_proxy = Some(proxy);
        // The round trip guarantees every feedback event of the request
        // above has been dispatched.
        display.roundtrip()?;
    }

    // Prefer what the feedback advertises for XRGB8888; fall back to the
    // legacy `format`/`modifier` events of a pre-v4 compositor.
    let modifiers = {
        let state = feedback.borrow();
        preferred_modifiers(&state.format_table, &state.tranche_indices)
    };
    let modifiers = if modifiers.is_empty() {
        legacy_modifiers.borrow().clone()
    } else {
        modifiers
    };
    if modifiers.is_empty() {
        return Err(WlError::Unsupported(String::from(
            "the compositor advertised no modifier for DRM_FORMAT_XRGB8888",
        ))
        .into());
    }
    // Prefer the linear layout when both sides offer it: it is the most
    // portable, and without this the driver would pick one of the
    // advertised tiled layouts on its own.
    let linear = [0u64];
    let requested: &[u64] = if modifiers.contains(&0) {
        &linear
    } else {
        &modifiers
    };
    let pipeline = match PipelineContext::new(width as u32, height as u32, requested) {
        Ok(context) => context,
        // The single-entry choice is only an optimisation: fall back to
        // every modifier the compositor advertised if the driver cannot
        // build the image with it.
        Err(_) if requested.len() == 1 => PipelineContext::new(width as u32, height as u32, &modifiers)?,
        Err(error) => return Err(error.into()),
    };
    let target = pipeline.present_target();
    let mut compositor = Compositor::new(&pipeline);
    compositor.start()?;
    compositor.add_pipe(SolidPipe::new(&pipeline, RED)?)?;
    compositor.add_pipe(SplitPipe::new(
        &pipeline,
        GREEN,
        YELLOW,
        Split::LeftRight,
        WEIGHT_HALF,
    )?)?;
    compositor.add_pipe(SplitPipe::new(
        &pipeline,
        CYAN,
        MAGENTA,
        Split::TopBottom,
        WEIGHT_QUARTER,
    )?)?;
    let params = display.marshal_new_id(dmabuf, DMABUF_CREATE_PARAMS, Vec::new())?;
    display.marshal_request(
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
    )?;
    let buffer = display.marshal_new_id(
        params,
        BUFFER_PARAMS_CREATE_IMMED,
        vec![
            WlArgument::Int(width),
            WlArgument::Int(height),
            WlArgument::Uint(target.drm_format),
            WlArgument::Uint(0),
        ],
    )?;
    display.marshal_request(params, BUFFER_PARAMS_DESTROY, Vec::new())?;
    // The descriptor is sent (duplicated over the socket) by this flush;
    // `target` keeps it open until the pipeline context drops.
    display.flush()?;

    let released = Rc::new(Cell::new(true));
    {
        let released = Rc::clone(&released);
        display.add_listener(buffer, move |_, opcode, _| {
            if opcode == BUFFER_RELEASE {
                released.set(true);
            }
            0
        })?;
    }
    let mut frames = 0u32;
    while frames < FRAME_BUDGET && !closed.get() && Instant::now() < deadline {
        // Never rewrite a buffer the compositor may still be reading.
        let release_deadline = Instant::now() + RELEASE_TIMEOUT;
        while !released.get() && !closed.get() && Instant::now() < release_deadline {
            display.dispatch(Some(DISPATCH_TIMEOUT))?;
        }
        released.set(false);

        // Queue the frame callback before the commit that triggers the
        // repaint, exactly like the xdg_window example does.
        let frame = display.marshal_new_id(surface, SURFACE_FRAME, Vec::new())?;
        let frame_done = Rc::new(Cell::new(false));
        {
            let frame_done = Rc::clone(&frame_done);
            display.add_callback_listener(frame, move |_, _| {
                frame_done.set(true);
                0
            })?;
        }
        // One frame: every pipe records into its offscreen target, the
        // mix pass folds them into the presentation target and the
        // export signals completion; wait for the GPU so the committed
        // pixels are final.
        let sync_file = block_on(compositor.composite_frame(None))?;
        wait_for_gpu(&sync_file)?;
        drop(sync_file);

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
                WlArgument::Int(width),
                WlArgument::Int(height),
            ],
        )?;
        display.marshal_request(surface, SURFACE_COMMIT, Vec::new())?;
        display.flush()?;

        while !frame_done.get() && !closed.get() && Instant::now() < deadline {
            display.dispatch(Some(DISPATCH_TIMEOUT))?;
        }
        if !frame_done.get() {
            break;
        }
        display.proxy_destroy(frame)?;
        frames += 1;
    }
    display.flush()?;
    // Best effort teardown, in the order the protocol prescribes.
    let _ = display.marshal_request(toplevel, XDG_TOPLEVEL_DESTROY, Vec::new());
    let _ = display.marshal_request(xdg_surface, XDG_SURFACE_DESTROY, Vec::new());
    let _ = display.marshal_request(surface, SURFACE_DESTROY, Vec::new());
    let _ = display.marshal_request(buffer, BUFFER_DESTROY, Vec::new());
    if let Some(feedback) = feedback_proxy {
        let _ = display.marshal_request(feedback, FEEDBACK_DESTROY, Vec::new());
    }
    let _ = display.marshal_request(dmabuf, DMABUF_DESTROY, Vec::new());
    let _ = display.flush();

    // Stopping waits for the device; dropping the compositor then
    // releases the pipes (and their offscreen targets) before the
    // pipeline that owns the device.
    let _ = compositor.stop();
    drop(compositor);
    drop(pipeline);

    Ok(format!(
        "render_mix: {frames} frames of three-pipe mixing into a {width}x{height} \
         dma-buf window, {} pings answered, configure serial {serial}",
        pings.get()
    ))
}

/// Raw `zwp_linux_dmabuf_feedback_v1` payload, accumulated until `done`.
#[derive(Default)]
struct Feedback {
    /// Contents of the `format_table` file descriptor.
    format_table: Vec<u8>,
    /// Table indices of every tranche, in arrival order.
    tranche_indices: Vec<u32>,
    /// Whether the `done` event arrived.
    done: bool,
}

/// Named Globals for interface binding
type Globals = Rc<RefCell<Vec<(u32, String, u32)>>>;

/// Binds `interface` to the global of the same name announced by the
/// registry.
fn bind(
    display: &mut WlClientDisplay<WlUnixTransport>,
    registry: WlProxyId,
    globals: &Globals,
    interface: &'static WlInterface,
) -> WlResult<WlProxyId> {
    let (name, version) = global_named(globals, interface.name)?;
    display.registry_bind(registry, name, interface, version.min(interface.version))
}

/// Returns `(name, version)` of the global called `interface`.
fn global_named(globals: &Globals, interface: &str) -> WlResult<(u32, u32)> {
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
///
/// The protocol defines each index as a 16-bit unsigned integer in
/// native endianness (not 32-bit like most wayland array payloads).
fn u16_array(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(2)
        .map(|word| u32::from(u16::from_ne_bytes([word[0], word[1]])))
        .collect()
}

/// Reads the compositor's `format_table` payload into a fresh buffer.
///
/// The descriptor received over `SCM_RIGHTS` duplicates the compositor's
/// open file description, so the file offset is *shared* with it and
/// already sits at the end of the table (the compositor had just written
/// it before sending it). A plain `read` would therefore see an
/// immediate end of file, and rewinding with `lseek` would move the
/// offset the compositor still shares with every other client, so the
/// table is read with `pread` from offset zero without touching the
/// shared offset.
fn read_format_table(fd: libc::c_int, size: usize) -> Vec<u8> {
    // SAFETY: `take_fd` transferred ownership of this descriptor to the
    // caller, so `OwnedFd` closes it exactly once when it drops.
    let owned = unsafe { OwnedFd::from_raw(fd) };
    let size = size.min(MAX_FORMAT_TABLE_BYTES);
    let mut bytes = vec![0u8; size];
    let mut total = 0usize;
    while total < size {
        // SAFETY: `bytes[total..]` points at `size - total` writable
        // bytes, `owned` holds an open descriptor, and `total <= size`
        // keeps both the destination and the offset in range.
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
            // End of file before the size the compositor announced.
            break;
        }
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        break;
    }
    bytes.truncate(total);
    bytes
}

/// Parses the 16-byte entries of a `format_table` into
/// `(fourcc, modifier)` pairs (native-endian payload, per the protocol).
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

/// Returns the `DRM_FORMAT_XRGB8888` modifiers the compositor prefers,
/// in tranche order, falling back to every table entry with that format
/// when no tranche listed any.
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
///
/// The example does not wire the descriptor into an explicit-sync
/// acquire point, so waiting here keeps the committed pixels final
/// without relying on implicit dma-buf fencing alone.
fn wait_for_gpu(sync_file: &OwnedFd) -> ExampleResult<()> {
    let mut descriptor = libc::pollfd {
        fd: sync_file.as_raw(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: `descriptor` points at a live `pollfd` holding an open,
    // pollable `sync_file` descriptor owned by the caller; `nfds` is `1`,
    // matching the single entry of the array.
    let ready = unsafe { libc::poll(&mut descriptor, 1, GPU_SYNC_TIMEOUT_MS) };
    if ready <= 0 {
        return Err(format!("timed out waiting for the GPU ({GPU_SYNC_TIMEOUT_MS} ms)").into());
    }
    Ok(())
}

/// Creates an offscreen pipe target the size of the presentation target,
/// so pipe-local rectangles line up with window pixels one to one.
fn new_target<'p>(context: &'p PipelineContext) -> Result<OffscreenTarget<'p>, CompositorError> {
    OffscreenTarget::new(
        context,
        context.width(),
        context.height(),
        PipelineContext::color_format(),
    )
}

/// The base pipe: fills its whole offscreen target with a solid color.
///
/// Registered first, so its framebuffer is framebuffer 0 of the frame —
/// the layer the mix shader starts from, whose weight is ignored.
struct SolidPipe<'p> {
    /// The pipe's offscreen color target.
    target: OffscreenTarget<'p>,
    /// Clear color of the whole target.
    color: [f32; 4],
}

impl<'p> SolidPipe<'p> {
    /// Creates a pipe whose target matches the presentation target.
    ///
    /// # Errors
    ///
    /// Propagates the [`OffscreenTarget::new`] failures (image, memory
    /// or view creation).
    fn new(context: &'p PipelineContext, color: [f32; 4]) -> Result<Self, CompositorError> {
        Ok(Self {
            target: new_target(context)?,
            color,
        })
    }
}

impl PipeSource for SolidPipe<'_> {
    fn pipe_entry<'a>(renderer: &'a mut Self, ctx: &'a mut PipeCtx) -> PipeFuture<'a> {
        Box::pin(async move {
            ctx.begin_render(0, Some(renderer.color))?;
            ctx.end_render();
            Ok(PipeOutcome::Keep)
        })
    }
}

impl PipeSupplyTraits for SolidPipe<'_> {
    fn offscreen_framebuffer_count(&self) -> usize {
        1
    }

    fn offscreen_framebuffer(&self, index: usize) -> Option<OffscreenFramebuffer> {
        if index == 0 {
            Some(self.target.framebuffer())
        } else {
            None
        }
    }
}

/// Which half of a [`SplitPipe`] target receives the accent color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Split {
    /// Accent on the right half.
    LeftRight,
    /// Accent on the bottom half.
    TopBottom,
}

impl Split {
    /// The accent rectangle for a target of `width` × `height`.
    ///
    /// Odd extents keep the remainder on the accent side, so the two
    /// halves always cover the whole target without a gap or overlap.
    fn rect(&self, width: u32, height: u32) -> vk::Rect2D {
        match self {
            Self::LeftRight => vk::Rect2D {
                offset: vk::Offset2D {
                    x: (width / 2) as i32,
                    y: 0,
                },
                extent: vk::Extent2D {
                    width: width - width / 2,
                    height,
                },
            },
            Self::TopBottom => vk::Rect2D {
                offset: vk::Offset2D {
                    x: 0,
                    y: (height / 2) as i32,
                },
                extent: vk::Extent2D {
                    width,
                    height: height - height / 2,
                },
            },
        }
    }
}

/// An overlay pipe: clears its target to `base`, overwrites one half
/// with `accent`, and is folded into the frame at `weight`.
struct SplitPipe<'p> {
    /// The pipe's offscreen color target.
    target: OffscreenTarget<'p>,
    /// Color of the non-accent half.
    base: [f32; 4],
    /// Color of the accent half.
    accent: [f32; 4],
    /// Which half receives `accent`.
    split: Split,
    /// Mix weight of this framebuffer in `[0.0, 1.0]`.
    weight: f32,
}

impl<'p> SplitPipe<'p> {
    /// Creates a pipe whose target matches the presentation target.
    ///
    /// # Errors
    ///
    /// Propagates the [`OffscreenTarget::new`] failures (image, memory
    /// or view creation).
    fn new(
        context: &'p PipelineContext,
        base: [f32; 4],
        accent: [f32; 4],
        split: Split,
        weight: f32,
    ) -> Result<Self, CompositorError> {
        Ok(Self {
            target: new_target(context)?,
            base,
            accent,
            split,
            weight,
        })
    }
}

impl PipeSource for SplitPipe<'_> {
    fn pipe_entry<'a>(renderer: &'a mut Self, ctx: &'a mut PipeCtx) -> PipeFuture<'a> {
        Box::pin(async move {
            ctx.begin_render(0, Some(renderer.base))?;
            let rect = renderer
                .split
                .rect(renderer.target.width(), renderer.target.height());
            let attachment = [vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: renderer.accent,
                    },
                },
            }];
            let rects = [vk::ClearRect {
                rect,
                base_array_layer: 0,
                layer_count: 1,
            }];
            // SAFETY: the rendering scope opened by `begin_render` is
            // still open, attachment 0 is the scope's single color
            // attachment, and `Split::rect` lies inside the framebuffer
            // (it spans exactly one half of it).
            unsafe {
                ctx.device()
                    .cmd_clear_attachments(ctx.command_buffer(), &attachment, &rects);
            }
            ctx.end_render();
            Ok(PipeOutcome::Keep)
        })
    }
}

impl PipeSupplyTraits for SplitPipe<'_> {
    fn offscreen_framebuffer_count(&self) -> usize {
        1
    }

    fn offscreen_framebuffer(&self, index: usize) -> Option<OffscreenFramebuffer> {
        if index == 0 {
            Some(self.target.framebuffer())
        } else {
            None
        }
    }

    fn mix_weight(&self, _index: usize) -> f32 {
        self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `format_table` is a plain array of 16-byte `(format, modifier)`
    /// records; decoding a wrong byte count must not invent entries.
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

    /// Only XRGB8888 entries are offered to the pipeline, tranche order
    /// is preserved and duplicates collapse.
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
        // Tranche: the tiled modifier first, then a repeated linear one.
        assert_eq!(
            preferred_modifiers(&table, &[2, 0, 3]),
            vec![0x00E0_0000_0000_0002, 0]
        );
        // No tranche: every XRGB8888 entry in table order.
        assert_eq!(preferred_modifiers(&table, &[]), vec![0, 0x00E0_0000_0000_0002]);
        // An unknown index is skipped rather than panicking.
        assert_eq!(preferred_modifiers(&table, &[99]), Vec::new());
    }

    /// Both splits put the accent on the far half and keep odd extents
    /// gap-free, which is what the four-quadrant expectation in the
    /// module documentation assumes.
    #[test]
    fn split_rects_cover_their_half_without_gaps() {
        let right = Split::LeftRight.rect(641, 480);
        assert_eq!(
            (
                right.offset.x,
                right.offset.y,
                right.extent.width,
                right.extent.height
            ),
            (320, 0, 321, 480)
        );
        let bottom = Split::TopBottom.rect(640, 481);
        assert_eq!(
            (
                bottom.offset.x,
                bottom.offset.y,
                bottom.extent.width,
                bottom.extent.height
            ),
            (0, 240, 640, 241)
        );
    }
}
