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

//! A button that increments the state a text view reads.
//!
//! The example is the smallest end-to-end picture of reactivity in Kuitui:
//!
//! * **Wayland:** the handshake of `hello_text` — registry, `xdg-shell`,
//!   frame callbacks, `wl_buffer.release` — plus the seat of
//!   `hello_button`, whose `wl_pointer` events are queued by the listener and
//!   replayed into the tree each frame.
//! * **[`ReactiveComposition`]:** the content reads a counter with
//!   `use_state` and renders it in a [`Text`]. The click handler of the
//!   button writes the same state through its handle, so the next frame
//!   renders the new value — the text view and the state cannot disagree,
//!   because they are the same value.
//! * **[`shape_list`]:** rasterizes the text runs of the new list into the
//!   shared atlas and mesh. The list is recorded every frame because the
//!   UI pipe drains it, but the mesh is rebuilt only when the engine
//!   recomposed, which is what makes the changing count cheap.
//!
//! Run it inside a Wayland session and click the button:
//!
//! ```text
//! cargo run -p codevar-gpu-vk --example reactive_counter
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use codevar_gpu_vk::base::{Compositor, block_on};
use codevar_gpu_vk::pipeline::{OwnedFd, PipelineContext};
use codevar_gpu_vk::ui::{UiFrame, UiPipe};
use codevar_kuitui::color::Color;
use codevar_kuitui::event::{Event, PointerButton, PointerButtons, PointerEvent, PointerEventKind};
use codevar_kuitui::geometry::{Offset, Size};
use codevar_kuitui::host::WidgetHost;
use codevar_kuitui::node::{Modifier, Node};
use codevar_kuitui::prelude::{
    Alignment, FontMetrics, InteractionSource, ReactiveComposition, Theme, kuitui_element,
};
use codevar_kuitui::rea::{Element, RenderCtx, State};
use codevar_kuitui_button::{Button, TextButton};
use codevar_kuitui_text::{AtlasConfig, TextAtlas, TextFrame, TextPipe};
use codevar_kuitui_widget::{Font, shape_list};
use codevar_time_core::TimeDuration;
use codevar_wayland::{
    BUFFER_DESTROY, BUFFER_PARAMS_ADD, BUFFER_PARAMS_CREATE_IMMED, BUFFER_PARAMS_DESTROY, BUFFER_RELEASE,
    COMPOSITOR_CREATE_SURFACE, COMPOSITOR_INTERFACE, DMABUF_CREATE_PARAMS, DMABUF_DESTROY,
    DMABUF_GET_DEFAULT_FEEDBACK, DMABUF_INTERFACE, DMABUF_MODIFIER, DRM_FORMAT_XRGB8888, FEEDBACK_DESTROY,
    FEEDBACK_DONE, FEEDBACK_FORMAT_TABLE, FEEDBACK_MAIN_DEVICE, FEEDBACK_TRANCHE_DONE,
    FEEDBACK_TRANCHE_FLAGS, FEEDBACK_TRANCHE_FORMATS, FEEDBACK_TRANCHE_TARGET_DEVICE, POINTER_BUTTON,
    POINTER_ENTER, POINTER_LEAVE, POINTER_MOTION, SEAT_CAPABILITIES, SEAT_CAPABILITY_POINTER,
    SEAT_GET_POINTER, SEAT_INTERFACE, SURFACE_ATTACH, SURFACE_COMMIT, SURFACE_DAMAGE, SURFACE_DESTROY,
    SURFACE_FRAME, WlArgument, WlClientDisplay, WlError, WlInterface, WlProxyId, WlRegistryEvent, WlResult,
    WlUnixTransport, XDG_SURFACE_ACK_CONFIGURE, XDG_SURFACE_CONFIGURE, XDG_SURFACE_DESTROY,
    XDG_SURFACE_GET_TOPLEVEL, XDG_SURFACE_SET_WINDOW_GEOMETRY, XDG_TOPLEVEL_CLOSE, XDG_TOPLEVEL_DESTROY,
    XDG_TOPLEVEL_SET_APP_ID, XDG_TOPLEVEL_SET_TITLE, XDG_WM_BASE_GET_XDG_SURFACE, XDG_WM_BASE_INTERFACE,
    XDG_WM_BASE_PING, XDG_WM_BASE_PONG,
};

/// Initial window width in surface local pixels (used when the first
/// configure carries `0x0`).
const WIDTH: i32 = 560;
/// Initial window height in surface local pixels.
const HEIGHT: i32 = 360;
/// Hard stop for the initial configure, so a silent compositor cannot hang
/// the example before the first frame.
const STARTUP_DEADLINE: Duration = Duration::from_secs(30);
/// How long a single `dispatch` waits for Wayland events.
const DISPATCH_TIMEOUT: TimeDuration = TimeDuration::from_millis(50);
/// How long to wait for `wl_buffer.release` before re-rendering anyway (a
/// compositor that never releases single-buffered clients would otherwise
/// stall the demo; the fallback costs at most one torn frame).
const RELEASE_TIMEOUT: Duration = Duration::from_millis(500);
/// Milliseconds the example waits for the GPU through the exported
/// `sync_file` before committing the buffer.
const GPU_SYNC_TIMEOUT_MS: i32 = 5_000;
/// Largest format table the example is willing to allocate.
const MAX_FORMAT_TABLE_BYTES: usize = 1 << 20;
/// The window background; the UI pipe clears to it, so it is the base layer
/// the text pipe composites over.
const BACKGROUND: [f32; 4] = [0.10, 0.11, 0.14, 1.0];
/// The clear color of the text pipe: transparent, so the glyphs land on the
/// UI layer instead of replacing it.
const TEXT_CLEAR: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
/// Pixels per em the glyphs are rasterized at.
const FONT_PPEM: u16 = 32;
/// Space between the top of the window and the first label.
const TOP_PADDING: f32 = 48.0;
/// The bundled test font (Space Mono for regular text).
const FONT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/tf/codevar.tf.space-mono.regular.ttf"
);

/// Every failure of the example is reported as a boxed error so `?` works for
/// Wayland, Vulkan, compositor and shaping errors alike.
type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    match run() {
        Ok(summary) => println!("{summary}"),
        Err(error) => {
            eprintln!("reactive_counter: {error}");
            std::process::exit(1);
        }
    }
}

/// Drives the whole example: Wayland window, Vulkan pipeline, the two
/// compositor pipes, the render loop and teardown.
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
    // The seat is the input side of the window. The protocol only lets a
    // client ask for a pointer once the seat announces the capability, so
    // the listener below creates it when that event arrives and hands the
    // raw events to `PointerInput`, which the frame loop replays into the
    // tree once the composition exists.
    let input = Rc::new(RefCell::new(PointerInput::default()));
    let seat = bind(&mut display, registry, &globals, &SEAT_INTERFACE)?;
    {
        let input = Rc::clone(&input);
        let pointer = Rc::new(Cell::new(None));
        display.add_listener(seat, move |client, opcode, args| {
            if opcode != SEAT_CAPABILITIES
                || !matches!(args.first(), Some(WlArgument::Uint(capabilities))
                    if capabilities & SEAT_CAPABILITY_POINTER != 0)
                || pointer.get().is_some()
            {
                return 0;
            }
            let Ok(pointer_id) = client.marshal_new_id(seat, SEAT_GET_POINTER, Vec::new()) else {
                return -1;
            };
            pointer.set(Some(pointer_id));
            let input = Rc::clone(&input);
            if client
                .add_listener(pointer_id, move |_, opcode, args| {
                    input.borrow_mut().push(opcode, args);
                    0
                })
                .is_err()
            {
                return -1;
            }
            0
        })?;
    }
    // The round trip delivers the capabilities event (and with it the
    // pointer) before the frame loop starts.
    display.roundtrip()?;
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
                // DRM_FORMAT_MOD_INVALID means "any implicit modifier"; the only
                // explicit modifier that always works is linear.
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
        vec![WlArgument::Str(Some(String::from(
            "Codevar reactive counter demo",
        )))],
    )?;
    display.marshal_request(
        toplevel,
        XDG_TOPLEVEL_SET_APP_ID,
        vec![WlArgument::Str(Some(String::from(
            "dev.codevar.reactive-counter",
        )))],
    )?;

    // The first commit asks the compositor for the initial configure.
    display.marshal_request(surface, SURFACE_COMMIT, Vec::new())?;
    display.flush()?;

    let deadline = Instant::now() + STARTUP_DEADLINE;
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
    let size = Size::new(width as f32, height as f32);
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
        // The round trip guarantees every feedback event of the request above
        // has been dispatched.
        display.roundtrip()?;
    }

    // Prefer what the feedback advertises for XRGB8888; fall back to the legacy
    // `format`/`modifier` events of a pre-v4 compositor.
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
    // portable, and without this the driver would pick one of the advertised
    // tiled layouts on its own.
    let linear = [0u64];
    let requested: &[u64] = if modifiers.contains(&0) {
        &linear
    } else {
        &modifiers
    };
    let pipeline = match PipelineContext::new(width as u32, height as u32, requested) {
        Ok(context) => context,
        // The single-entry choice is only an optimization: fall back to every
        // modifier the compositor advertised if the driver cannot build the
        // image with it.
        Err(_) if requested.len() == 1 => PipelineContext::new(width as u32, height as u32, &modifiers)?,
        Err(error) => return Err(error.into()),
    };
    let target = pipeline.present_target();

    // The UI: a composition measured with the real font, and the two pipes the
    // compositor folds together. The UI pipe is registered first so it is the
    // base of the mix.
    let font = Rc::new(Font::from_bytes(std::fs::read(FONT)?)?.with_ppem(FONT_PPEM));
    let metrics: Rc<dyn FontMetrics> = font.clone();
    let mut composition = ReactiveComposition::new(Theme::dark().with_metrics(metrics));
    // The interaction source outlives a render: the button node is rebuilt
    // from the description of every render, so a source created inside the
    // content would forget where the pointer is each time the count changes.
    composition.set_content(counter_content(InteractionSource::new()));
    composition.recompose(true);
    composition.layout(size, true);

    let ui_frame = UiFrame::new();
    let text_frame = TextFrame::new(TextAtlas::new(AtlasConfig::DEFAULT)?)?;
    let mut compositor = Compositor::new(&pipeline);
    compositor.start()?;
    let mut ui_pipe = UiPipe::new(&pipeline, ui_frame.clone())?;
    ui_pipe.set_clear(BACKGROUND);
    compositor.add_pipe(ui_pipe)?;
    let mut text_pipe = TextPipe::new(&pipeline, text_frame.clone())?;
    text_pipe.set_clear(TEXT_CLEAR);
    compositor.add_pipe(text_pipe)?;

    // The first frame shapes the text of the mounted tree. Later frames shape
    // again only when the engine recomposed, because the mesh persists across
    // frames while the geometry list does not.
    record_frame(&mut composition, &ui_frame, &text_frame, font.as_ref(), true)?;

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
    while !closed.get() {
        // Never rewrite a buffer the compositor may still be reading. The
        // dispatches also collect the pointer events of the last frame.
        let release_deadline = Instant::now() + RELEASE_TIMEOUT;
        while !released.get() && !closed.get() && Instant::now() < release_deadline {
            display.dispatch(Some(DISPATCH_TIMEOUT))?;
        }
        released.set(false);

        // Replay what the pointer did while the last frame was in flight.
        // A click reaches the button's modifier, which calls the handler the
        // content built; the handler writes the counter state.
        {
            let mut input = input.borrow_mut();
            for event in input.events.drain(..) {
                let _ = composition.handle_event(&event);
            }
        }
        // The write woke the root, so render it before the frame below: the
        // label the compositor receives carries the new count.
        let recomposed = composition.is_dirty();
        if recomposed {
            composition.recompose(false);
            composition.layout(size, false);
        }

        let frame = display.marshal_new_id(surface, SURFACE_FRAME, Vec::new())?;
        let frame_done = Rc::new(Cell::new(false));
        {
            let frame_done = Rc::clone(&frame_done);
            display.add_callback_listener(frame, move |_, _| {
                frame_done.set(true);
                0
            })?;
        }
        // A pointer event also changes what a widget draws without touching
        // the engine — hovering the button lights its state layer — so the
        // geometry is recorded every frame while the text is shaped only
        // when the tree changed.
        record_frame(
            &mut composition,
            &ui_frame,
            &text_frame,
            font.as_ref(),
            recomposed,
        )?;

        // One frame: both pipes replay, the mix pass folds them into the
        // presentation target and the export signals completion; wait for the
        // GPU so the committed pixels are final.
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

        let frame_deadline = Instant::now() + RELEASE_TIMEOUT;
        while !frame_done.get() && !closed.get() && Instant::now() < frame_deadline {
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

    // Stopping waits for the device; dropping the compositor then releases the
    // pipes (and their offscreen targets) before the pipeline that owns the
    // device.
    let _ = compositor.stop();
    drop(compositor);
    drop(pipeline);

    Ok(format!(
        "reactive_counter: {frames} frames of a counter into a {width}x{height} \
         dma-buf window, {} pings answered, configure serial {serial}",
        pings.get()
    ))
}

/// The content of the example: a counter the button writes and a text view
/// that reads it.
fn counter_content(
    source: InteractionSource,
) -> impl Fn(&mut RenderCtx<'_>) -> Element<WidgetHost> + 'static {
    move |ctx| {
        let count = ctx.use_state(|| 0_u32);
        let shown = *count.get();
        let source = source.clone();
        kuitui_element! {
            Column(spacing: 20.0, alignment: Alignment::CENTER) {
                Spacer(height: TOP_PADDING)
                Text("Reactive counter", color: Color::WHITE)
                Text(format!("Count: {shown}"), color: Color::WHITE)
                Node(click_me_button(&count, &source))
                Text("Clicking the button writes the state this label reads.",
                    color: Color::from_rgba8(150, 156, 170, 255))
            }
        }
    }
}

/// The button that increments `count` when it is released inside its box.
///
/// The state handle is captured by the handler, so the click writes the same
/// queue the content reads. `source` is the interaction source of the whole
/// example, which keeps the hover and press states across the rebuild the
/// engine performs when the count changes.
fn click_me_button(count: &State<u32>, source: &InteractionSource) -> Node {
    let count = count.clone();
    TextButton::new(move || count.set(count.get().wrapping_add(1)), "Click Me")
        .interaction_source(source.clone())
        .modifier(Modifier::none().size(Some(200.0), Some(56.0)))
        .node()
}

/// Records the laid-out tree and, when `reshape` is set, the text of the new
/// list.
///
/// The geometry list is drained by the UI pipe every frame, so it is always
/// recorded again. The text mesh is not drained: it persists until
/// [`TextFrame::clear`], which is why a frame that did not recompose can skip
/// the shaping pass and still draw the text of the last one.
fn record_frame(
    composition: &mut ReactiveComposition,
    ui_frame: &UiFrame,
    text_frame: &TextFrame,
    font: &Font,
    reshape: bool,
) -> ExampleResult<()> {
    if reshape {
        // The new glyphs enter the atlases, which keep the old ones cached;
        // only the runs of this frame are replaced.
        text_frame.clear();
        text_frame.begin_frame();
    }
    ui_frame.with(|list| -> ExampleResult<()> {
        composition.render(list);
        if reshape {
            shape_list(text_frame, font, list)?;
            // The mesh changed, so the text pipe must rebuild its buffers.
            text_frame.mark_dirty();
        }
        Ok(())
    })
}

/// Linux input code of the primary pointer button.
const BTN_LEFT: u32 = 0x110;
/// Linux input code of the secondary pointer button.
const BTN_RIGHT: u32 = 0x111;
/// Linux input code of the middle pointer button.
const BTN_MIDDLE: u32 = 0x112;
/// The `wl_pointer.button` state code of a press; a release reports `0`.
const BUTTON_PRESSED: u32 = 1;

/// The pointer events a `wl_pointer` listener queues for the frame loop.
///
/// The listener cannot reach the composition — the seat is bound before the
/// tree exists — so it records the events and the frame loop replays them
/// into the tree, where the modifier chain of the button updates the
/// interaction source it owns.
#[derive(Default)]
struct PointerInput {
    /// Where the pointer is, in surface-local pixels.
    position: Offset,
    /// The buttons that are held down.
    buttons: PointerButtons,
    /// The events the tree has not seen yet.
    events: Vec<Event>,
}

impl PointerInput {
    /// Routes one `wl_pointer` event by its opcode.
    fn push(&mut self, opcode: u32, args: &mut [WlArgument]) {
        match opcode {
            POINTER_ENTER => {
                // serial, surface, surface_x, surface_y
                self.moved(args.get(2), args.get(3), PointerEventKind::Enter);
            }
            POINTER_LEAVE => {
                // serial, surface: the event carries no coordinates, so the
                // pointer is reported just outside the window, where every
                // widget clears its hover.
                self.position = Offset::new(-1.0, -1.0);
                self.queue(PointerEventKind::Leave, PointerButton::Primary);
            }
            POINTER_MOTION => {
                // time, surface_x, surface_y
                self.moved(args.get(1), args.get(2), PointerEventKind::Move);
            }
            POINTER_BUTTON => {
                // serial, time, button, state
                let (Some(WlArgument::Uint(button)), Some(WlArgument::Uint(state))) =
                    (args.get(2), args.get(3))
                else {
                    return;
                };
                let code = *button;
                let pressed = *state == BUTTON_PRESSED;
                self.buttons = buttons_after(self.buttons, code, pressed);
                let kind = if pressed {
                    PointerEventKind::Down
                } else {
                    PointerEventKind::Up
                };
                self.queue(kind, pointer_button(code));
            }
            _ => {}
        }
    }

    /// Moves the pointer to the fixed-point coordinates of `x` and `y` and
    /// queues `kind` there.
    fn moved(&mut self, x: Option<&WlArgument>, y: Option<&WlArgument>, kind: PointerEventKind) {
        let (Some(WlArgument::Fixed(x)), Some(WlArgument::Fixed(y))) = (x, y) else {
            return;
        };
        self.position = Offset::new(x.to_double() as f32, y.to_double() as f32);
        self.queue(kind, PointerButton::Primary);
    }

    /// Queues a pointer event at the current position.
    fn queue(&mut self, kind: PointerEventKind, button: PointerButton) {
        let mut event = PointerEvent::new(kind, self.position);
        event.button = button;
        event.buttons = self.buttons;
        self.events.push(Event::Pointer(event));
    }
}

/// The set of buttons held after `code` changed to `pressed`.
fn buttons_after(held: PointerButtons, code: u32, pressed: bool) -> PointerButtons {
    let mut next = PointerButtons::NONE;
    for (button, flag) in [
        (BTN_LEFT, PointerButtons::PRIMARY),
        (BTN_RIGHT, PointerButtons::SECONDARY),
        (BTN_MIDDLE, PointerButtons::MIDDLE),
    ] {
        let down = if button == code {
            pressed
        } else {
            held.contains(flag)
        };
        if down {
            next |= flag;
        }
    }
    next
}

/// The button of the tree that Linux input `code` reports.
fn pointer_button(code: u32) -> PointerButton {
    match code {
        BTN_RIGHT => PointerButton::Secondary,
        BTN_MIDDLE => PointerButton::Middle,
        _ => PointerButton::Primary,
    }
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

/// Named Globals for interface binding.
type Globals = Rc<RefCell<Vec<(u32, String, u32)>>>;

/// Binds `interface` to the global of the same name announced by the registry.
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
/// The protocol defines each index as a 16-bit unsigned integer in native
/// endianness (not 32-bit like most wayland array payloads).
fn u16_array(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(2)
        .map(|word| u32::from(u16::from_ne_bytes([word[0], word[1]])))
        .collect()
}

/// Reads the compositor's `format_table` payload into a fresh buffer.
///
/// The descriptor received over `SCM_RIGHTS` duplicates the compositor's open
/// file description, so the file offset is *shared* with it and already sits at
/// the end of the table. A plain `read` would therefore see an immediate end of
/// file, and rewinding with `lseek` would move the offset the compositor still
/// shares, so the table is read with `pread` from offset zero.
fn read_format_table(fd: libc::c_int, size: usize) -> Vec<u8> {
    // SAFETY: `take_fd` transferred ownership of this descriptor to the caller,
    // so `OwnedFd` closes it exactly once when it drops.
    let owned = unsafe { OwnedFd::from_raw(fd) };
    let size = size.min(MAX_FORMAT_TABLE_BYTES);
    let mut bytes = vec![0u8; size];
    let mut total = 0usize;
    while total < size {
        // SAFETY: `bytes[total..]` points at `size - total` writable bytes,
        // `owned` holds an open descriptor, and `total <= size` keeps both the
        // destination and the offset in range.
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
        if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        break;
    }
    bytes.truncate(total);
    bytes
}

/// Parses the 16-byte entries of a `format_table` into `(fourcc, modifier)`
/// pairs (native-endian payload, per the protocol).
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

/// Returns the `DRM_FORMAT_XRGB8888` modifiers the compositor prefers, in
/// tranche order, falling back to every table entry with that format when no
/// tranche listed any.
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
/// The example does not wire the descriptor into an explicit-sync acquire
/// point, so waiting here keeps the committed pixels final without relying on
/// implicit dma-buf fencing alone.
fn wait_for_gpu(sync_file: &OwnedFd) -> ExampleResult<()> {
    let mut descriptor = libc::pollfd {
        fd: sync_file.as_raw(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: `descriptor` points at a live `pollfd` holding an open, pollable
    // `sync_file` descriptor owned by the caller; `nfds` is `1`, matching the
    // single entry of the array.
    let ready = unsafe { libc::poll(&mut descriptor, 1, GPU_SYNC_TIMEOUT_MS) };
    if ready <= 0 {
        return Err(format!("timed out waiting for the GPU ({GPU_SYNC_TIMEOUT_MS} ms)").into());
    }
    Ok(())
}
