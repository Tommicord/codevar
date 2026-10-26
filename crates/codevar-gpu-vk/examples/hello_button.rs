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

//! Renders a decorated [`TextButton`] tree into a Wayland window.
//!
//! The example is the button-side counterpart of `hello_text`: the same
//! Wayland and compositor plumbing, and a UI whose every widget answers
//! the eleven parameters of the [`Button`] trait.
//!
//! * **Wayland (this file):** the handshake of `render_hit` — registry,
//!   `xdg-shell` (`ack_configure`, `ping`/`pong`, frame callbacks,
//!   `wl_buffer.release`) and a `zwp_linux_dmabuf_v1` buffer built from the
//!   render target's exported dma-buf.
//! * **The tree:** a [`Column`] of buttons laid out at the window
//!   size by [`Composition`]. The first sets every parameter of
//!   [`TextButton`] — flags, shape, fixed palette, elevated shadow, border,
//!   content padding and a `Modifier` that sizes, aligns and ramps its
//!   background. The second is a [`ColorableTextButton`] whose label keeps
//!   the colors of the emoji font and whose modifier outlines and clips the
//!   box instead of painting it.
//! * **GPU decorations:** each button draws its rounded container, its
//!   state layer and its ambient shadow through effect layers, so the
//!   `ui_effect` program runs for them — the corner distance field, the
//!   blurred silhouette of the shadow and the gradient of the decorated
//!   background are all fragment-shader work.
//! * **[`UiPipe`] + [`TextPipe`]:** the compositor replays the polygon UI
//!   pipe and then the text pipe; the mix pass folds the text layer over
//!   the geometry. Both targets clear to the window background because the
//!   mix pass blends the layers with a per-pipe weight, not per-pixel alpha.
//!
//! The example binds no `wl_seat`, so nothing moves the pointer states.
//! One of the buttons starts from an [`InteractionSource`] that already
//! holds `HOVERED`, which is how the state layer of the chrome becomes
//! visible in a static frame.
//!
//! Run it inside a Wayland session:
//!
//! ```text
//! cargo run -p codevar-gpu-vk --example hello_button
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use codevar_gpu_vk::base::{Compositor, block_on};
use codevar_gpu_vk::pipeline::{OwnedFd, PipelineContext};
use codevar_gpu_vk::ui::{UiFrame, UiPipe};
use codevar_kuitui::color::Color;
use codevar_kuitui::geometry::{Offset, Rect, Size};
use codevar_kuitui::paint::{CornerRadii, Paint};
use codevar_kuitui::prelude::{
    Alignment, Column, Composition, EdgeInsets, FontMetrics, Gradient, InteractionSource, InteractionState,
    Modifier, Node, Spacer, Theme,
};
use codevar_kuitui_button::{
    BorderStroke, Button, ButtonColors, ButtonElevation, ButtonFlags, ButtonPalette, Click,
    ColorableTextButton, TextButton,
};
use codevar_kuitui_text::{AtlasConfig, TextAtlas, TextFrame, TextPipe};
use codevar_kuitui_widget::{Font, shape_list_with_color};
use codevar_time_core::TimeDuration;
use codevar_wayland::{
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
/// The window background, shared by the two pipes.
const BACKGROUND: [f32; 4] = [0.10, 0.11, 0.14, 1.0];
/// Pixels per em the glyphs are rasterized at.
const FONT_PPEM: u16 = 32;
/// Space between the top of the window and the first button.
const TOP_PADDING: f32 = 40.0;
/// Space between two buttons of the column.
const ROW_SPACING: f32 = 28.0;
/// The corner radius every button of the example draws with.
const SHAPE: CornerRadii = CornerRadii::uniform(18.0);
/// Width of the decorated button, which its `Modifier` forces on it.
const WIDE_WIDTH: f32 = 420.0;
/// Height of the decorated button, which its `Modifier` forces on it.
const WIDE_HEIGHT: f32 = 84.0;
/// The container color of the fixed palette of the first button.
const CONTAINER: Color = Color::rgb(0.247, 0.322, 0.710);
/// The content color of that palette.
const CONTENT: Color = Color::rgb(0.976, 0.984, 1.0);
/// The state layer color of that same palette; the chrome fades it to the
/// alpha of whichever transient state the button is in.
const STATE_LAYER: Color = Color::rgb(1.0, 0.835, 0.310);
/// The container a selected button of that palette swaps to.
const SELECTED: Color = Color::rgb(0.851, 0.871, 0.973);
/// The outline color of the first button.
const OUTLINE: Color = Color::rgb(0.506, 0.831, 0.980);
/// The colors of the two ends of the gradient that decorates the wide
/// button's background.
const RAMP: Gradient = Gradient::new(
    Offset::ZERO,
    Offset::new(WIDE_WIDTH, 0.0),
    Color::rgb(0.149, 0.651, 0.604),
    Color::rgb(0.051, 0.278, 0.631),
);
/// The bundled test font (Space Mono for regular text).
const FONT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/tf/codevar.tf.space-mono.regular.ttf"
);
/// The color emoji font.
const EMOJI_FONT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/resources/NotoColorEmoji.ttf"
);

/// Every failure of the example is reported as a boxed error so `?` works for
/// Wayland, Vulkan, compositor and shaping errors alike.
type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    match run() {
        Ok(summary) => println!("{summary}"),
        Err(error) => {
            eprintln!("hello_button: {error}");
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
        vec![WlArgument::Str(Some(String::from("Codevar hello_button demo")))],
    )?;
    display.marshal_request(
        toplevel,
        XDG_TOPLEVEL_SET_APP_ID,
        vec![WlArgument::Str(Some(String::from("dev.codevar.hello-button")))],
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
    // Load the color emoji font for color glyph rendering.
    let emoji_font = Rc::new(Font::from_bytes(std::fs::read(EMOJI_FONT)?)?.with_ppem(FONT_PPEM));
    let metrics: Rc<dyn FontMetrics> = font.clone();
    // The handler every button calls: the console gets the same greeting the
    // first button draws as its label.
    let greeting: Click = Rc::new(|| println!("Hello, World!"));
    println!("Hello, World!");
    // The example binds no `wl_seat`, so the sources below are the only way a
    // transient state is ever set — which is exactly what makes the state layer
    // and the ambient shadow of the chrome legible in a static frame.
    let hovered = InteractionSource::with_state(InteractionState::HOVERED);
    let mut composition = Composition::new(Theme::dark().with_metrics(metrics));
    composition.set_content(move |_composer| {
        Node::new(
            Column::new()
                .spacing(ROW_SPACING)
                .alignment(Alignment::CENTER),
        )
        .with_children(vec![
            Node::new(Spacer::new().height(TOP_PADDING)),
            full_button(Rc::clone(&greeting), hovered.clone()),
            emoji_button(Rc::clone(&greeting)),
        ])
    });
    composition.recompose(true);
    composition.layout(Size::new(width as f32, height as f32), true);

    let ui_frame = UiFrame::new();
    let text_frame = TextFrame::new(TextAtlas::new(AtlasConfig::DEFAULT)?)?;
    let mut compositor = Compositor::new(&pipeline);
    compositor.start()?;
    let mut ui_pipe = UiPipe::new(&pipeline, ui_frame.clone())?;
    ui_pipe.set_clear(BACKGROUND);
    compositor.add_pipe(ui_pipe)?;
    let mut text_pipe = TextPipe::new(&pipeline, text_frame.clone())?;
    text_pipe.set_clear(BACKGROUND);
    compositor.add_pipe(text_pipe)?;

    // Record the first frame: the tree into the UI list, a background for the
    // UI pipe, and every text run into the shared atlas and mesh. The text pipe
    // replays that mesh for the rest of the run, so nothing below re-shapes.
    text_frame.begin_frame();
    ui_frame.with(|list| -> ExampleResult<()> {
        composition.render(list);
        list.rect(
            Rect::new(Offset::ZERO, Size::new(width as f32, height as f32)),
            CornerRadii::ZERO,
            Paint::fill(Color::from_rgba8(26, 28, 36, 255)),
        );
        shape_list_with_color(&text_frame, font.as_ref(), emoji_font.as_ref(), list)?;

        // Mark the frame dirty so the color text is rendered on the first frame.
        text_frame.mark_dirty();

        Ok(())
    })?;

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

        let frame = display.marshal_new_id(surface, SURFACE_FRAME, Vec::new())?;
        let frame_done = Rc::new(Cell::new(false));
        {
            let frame_done = Rc::clone(&frame_done);
            display.add_callback_listener(frame, move |_, _| {
                frame_done.set(true);
                0
            })?;
        }
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

    // Stopping waits for the device; dropping the compositor then releases the
    // pipes (and their offscreen targets) before the pipeline that owns the
    // device.
    let _ = compositor.stop();
    drop(compositor);
    drop(pipeline);

    Ok(format!(
        "hello_button: {frames} frames of decorated buttons into a {width}x{height} \
         dma-buf window, {} pings answered, configure serial {serial}",
        pings.get()
    ))
}

/// The button that answers every parameter the trait exposes.
///
/// The palette is a fixed one instead of a theme role because a fixed
/// palette is the only way the state layer of the example is a color the
/// reader can point at. Its `Modifier` sizes and centers the node, and the
/// ramp it names is the `ui_effect` program's gradient over the outer box,
/// behind the container the chrome paints.
fn full_button(greeting: Click, source: InteractionSource) -> Node {
    TextButton::new(move || greeting(), "Hello, World!")
        .flags(ButtonFlags::ALL)
        .shape(SHAPE)
        .colors(ButtonPalette::fixed(
            ButtonColors::layered(CONTAINER, CONTENT, STATE_LAYER).with_selected(SELECTED),
        ))
        .elevation(Some(ButtonElevation::elevated()))
        .border(Some(BorderStroke::new(3.0, OUTLINE).with_radii(SHAPE)))
        .content_padding(EdgeInsets::symmetric(28.0, 18.0))
        .interaction_source(source)
        .modifier(
            Modifier::none()
                .size(Some(WIDE_WIDTH), Some(WIDE_HEIGHT))
                .align(Alignment::CENTER)
                .gradient(RAMP, SHAPE),
        )
        .node()
}

/// The color-font button: the same parameters answered with the outlined
/// role, no shadow, a pill shape and a modifier that clips and outlines the
/// box instead of painting it.
fn emoji_button(greeting: Click) -> Node {
    let pill = CornerRadii::uniform(32.0);
    ColorableTextButton::new(move || greeting(), "Ship it 😀 🚀")
        .flags(ButtonFlags::ENABLED | ButtonFlags::HOVERABLE | ButtonFlags::SELECTABLE)
        .shape(pill)
        .colors(ButtonPalette::role_of(ButtonColors::outlined))
        .elevation(None)
        .border(None)
        .content_padding(EdgeInsets::all(14.0))
        .modifier(
            Modifier::none()
                .size(Some(320.0), Some(64.0))
                .rounded_border(OUTLINE, 2.0, pill)
                .clip(),
        )
        .node()
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
