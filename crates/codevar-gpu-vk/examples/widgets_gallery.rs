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
//! CONDITIONS OF ANY KIND, either express or implied. See the
//! License for the specific language governing permissions and
//! limitations under the License.

//! A window of the interactive Kuitui widgets, rendered through Vulkan.
//!
//! `hello_button` covers the button and `reactive_counter` the reactive
//! content; this example starts with the rest of the widget set — the check
//! boxes — on the same two pipes, so every shape those widgets paint
//! reaches the GPU through the code that will draw the editor:
//!
//! * **[`kuitui_element!`]:** the content is a reactive tree whose widgets
//!   read and write the handles of [`Gallery`]. A press inside a box writes
//!   the [`MutableState`] that the label next to it reads, so the two cannot
//!   disagree; the handles outlive a render.
//! * **[`UiPipe`] + [`TextPipe`]:** the check mark and the outline of a box
//!   are a rounded fill and a rounded stroke — the geometry the UI pipe turns
//!   into scissored spans and signed-distance layers. The labels of the same widgets are text runs,
//!   which the UI pipe deliberately leaves to the text pipe, so both are
//!   registered and the list is shaped every time the tree is laid out again.
//!
//! Run it inside a Wayland session:
//!
//! ```text
//! cargo run -p codevar-gpu-vk --example widgets_gallery
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use codevar_gpu_vk::base::{Compositor, block_on};
use codevar_gpu_vk::pipeline::{OwnedFd, PipelineContext};
use codevar_gpu_vk::ui::{UiFrame, UiPipe};
use codevar_kuitui::color::Color;
use codevar_kuitui::geometry::{Alignment, EdgeInsets, Offset, Rect, Size};
use codevar_kuitui::host::WidgetHost;
use codevar_kuitui::node::Modifier;
use codevar_kuitui::paint::{CornerRadii, Paint};
use codevar_kuitui::prelude::{
    ColorScheme, FontMetrics, InteractionSource, MutableState, ReactiveComposition, Recomposition, Theme,
    kuitui_element,
};
use codevar_kuitui::rea::{Element, RenderCtx};
use codevar_kuitui::widgets::CheckState;
use codevar_kuitui_text::{AtlasConfig, TextAtlas, TextFrame, TextPipe};
use codevar_kuitui_widget::{Font, shape_list};
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

/// Window width in surface local pixels (used when the first configure
/// carries `0x0`).
const WIDTH: i32 = 640;
/// Window height in surface local pixels.
const HEIGHT: i32 = 420;
/// Hard stop for the initial configure, so a silent compositor cannot hang
/// the example before the first frame.
const STARTUP_DEADLINE: Duration = Duration::from_secs(30);
/// How long a single `dispatch` waits for Wayland events.
const DISPATCH_TIMEOUT: TimeDuration = TimeDuration::from_millis(50);
/// How long to wait for `wl_buffer.release` before re-rendering anyway.
const RELEASE_TIMEOUT: Duration = Duration::from_millis(500);
/// Milliseconds the example waits for the GPU through the exported
/// `sync_file` before committing the buffer.
const GPU_SYNC_TIMEOUT_MS: i32 = 5_000;
/// Largest format table the example is willing to allocate.
const MAX_FORMAT_TABLE_BYTES: usize = 1 << 20;
/// The window background, shared by the two pipes.
const BACKGROUND: [f32; 4] = [0.10, 0.11, 0.14, 1.0];
/// The tone of the captions: the muted text color of the dark scheme the
/// widgets are drawn with, so a caption is dimmer than the value it names.
const MUTED: Color = ColorScheme::dark().on_surface_variant;
/// Pixels per em the glyphs are rasterized at.
const FONT_PPEM: u16 = 28;
/// The bundled test font (Space Mono for regular text).
const FONT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/tf/codevar.tf.space-mono.regular.ttf"
);
/// Padding around the whole gallery.
const PAGE_PADDING: f32 = 22.0;
/// Space between two rows of the gallery.
const ROW_SPACING: f32 = 12.0;
/// Space between a control and the label that names it.
const CONTROL_SPACING: f32 = 8.0;
/// Every failure of the example is reported as a boxed error so `?` works for
/// Wayland, Vulkan, compositor and shaping errors alike.
type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

/// The interaction states the gallery keeps, one source per widget.
///
/// A source handed to a widget is a handle, so the hover and press layers the
/// widget reports survive the recomposition that rebuilt it — which is what
/// lets a slider keep its pressed layer while its own value forces a render.
#[derive(Clone)]
struct Sources {
    /// The master box of the header row.
    all: InteractionSource,
}

impl Sources {
    /// The sources of the gallery, each holding nothing yet.
    fn new() -> Self {
        Self {
            all: InteractionSource::new(),
        }
    }
}

/// The values the widgets of the gallery own and the frame loop watches.
///
/// Every field is a handle over shared storage, so the same value is what the
/// widget writes, the label of the same row reads, and the loop polls for a
/// write. A [`MutableState`] that a widget writes does not schedule a render by
/// itself — the engine only knows the states its hooks handed out — so the loop
/// takes the dirty flag of these handles and rebuilds the tree from that, which
/// is what a real host does for a widget it did not create through a hook.
#[derive(Clone)]
struct Gallery {
    /// The tri-state box of the header row.
    all: MutableState<CheckState>,
    /// The interaction states of the widgets.
    sources: Sources,
}

impl Gallery {
    /// A gallery with nothing chosen: the first box indeterminate, the slider
    /// off and disabled, and the middle size selected.
    fn new() -> Self {
        Self {
            all: MutableState::new(CheckState::Indeterminate),
            sources: Sources::new(),
        }
    }

    /// Whether a widget wrote one of the values since the last frame.
    ///
    /// `take_dirty` clears the flag as it reads it, so a frame that wrote
    /// nothing recomposes nothing.
    fn take_dirty(&self) -> bool {
        let mut dirty = false;
        dirty |= self.all.take_dirty();
        dirty
    }
}

fn main() {
    match run() {
        Ok(summary) => println!("{summary}"),
        Err(error) => {
            eprintln!("widgets_gallery: {error}");
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
        vec![WlArgument::Str(Some(String::from("Codevar widget gallery")))],
    )?;
    display.marshal_request(
        toplevel,
        XDG_TOPLEVEL_SET_APP_ID,
        vec![WlArgument::Str(Some(String::from("dev.codevar.widget-gallery")))],
    )?;

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
        display.roundtrip()?;
    }

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
    let linear = [0u64];
    let requested: &[u64] = if modifiers.contains(&0) {
        &linear
    } else {
        &modifiers
    };
    let pipeline = match PipelineContext::new(width as u32, height as u32, requested) {
        Ok(context) => context,
        Err(_) if requested.len() == 1 => PipelineContext::new(width as u32, height as u32, &modifiers)?,
        Err(error) => return Err(error.into()),
    };
    let target = pipeline.present_target();

    // The UI: the gallery measured with the real font, and the two pipes the
    // compositor folds together. The UI pipe is registered first so it is the
    // base of the mix.
    let font = Rc::new(Font::from_bytes(std::fs::read(FONT)?)?.with_ppem(FONT_PPEM));
    let metrics: Rc<dyn FontMetrics> = font.clone();
    let gallery = Gallery::new();
    let mut composition = ReactiveComposition::new(Theme::dark().with_metrics(metrics));
    composition.set_content(gallery_content(gallery.clone()));
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
    text_pipe.set_clear(BACKGROUND);
    compositor.add_pipe(text_pipe)?;

    record_frame(
        &mut composition,
        &ui_frame,
        &text_frame,
        font.as_ref(),
        size,
        true,
    )?;

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
        let release_deadline = Instant::now() + RELEASE_TIMEOUT;
        while !released.get() && !closed.get() && Instant::now() < release_deadline {
            display.dispatch(Some(DISPATCH_TIMEOUT))?;
        }
        released.set(false);

        // The engine schedules the renders its hooks own; a widget writing a
        // handle the gallery created is only seen through that handle.
        let written = gallery.take_dirty();
        let rebuilt = matches!(composition.recompose(written), Recomposition::Recomposed);
        composition.layout(size, rebuilt || written);

        let frame = display.marshal_new_id(surface, SURFACE_FRAME, Vec::new())?;
        let frame_done = Rc::new(Cell::new(false));
        {
            let frame_done = Rc::clone(&frame_done);
            display.add_callback_listener(frame, move |_, _| {
                frame_done.set(true);
                0
            })?;
        }
        // A hover or a press changes what a widget paints without touching the
        // engine, so the geometry is recorded every frame. The text is shaped
        // whenever the tree was laid out again, because a relayout moves the
        // runs the mesh carries.
        record_frame(
            &mut composition,
            &ui_frame,
            &text_frame,
            font.as_ref(),
            size,
            rebuilt || written,
        )?;

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
    let _ = display.marshal_request(toplevel, XDG_TOPLEVEL_DESTROY, Vec::new());
    let _ = display.marshal_request(xdg_surface, XDG_SURFACE_DESTROY, Vec::new());
    let _ = display.marshal_request(surface, SURFACE_DESTROY, Vec::new());
    let _ = display.marshal_request(buffer, BUFFER_DESTROY, Vec::new());
    if let Some(feedback) = feedback_proxy {
        let _ = display.marshal_request(feedback, FEEDBACK_DESTROY, Vec::new());
    }
    let _ = display.marshal_request(dmabuf, DMABUF_DESTROY, Vec::new());
    let _ = display.flush();

    let _ = compositor.stop();
    drop(compositor);
    drop(pipeline);

    Ok(format!(
        "widgets_gallery: {frames} frames of a checkbox row into a {width}x{height} \
         dma-buf window, {} pings answered, configure serial {serial}",
        pings.get()
    ))
}

/// The content of the gallery: four rows of widgets, each reading and writing
/// the handle of the [`Gallery`] it was built with.
///
/// The handles outlive a render, so recomposing the tree — which the engine
/// does whenever a value changes — rebuilds the widgets from their description
/// without losing what the pointer wrote into them: the value, the press and
/// the interaction states all live behind the clone the widget was handed.
fn gallery_content(gallery: Gallery) -> impl Fn(&mut RenderCtx<'_>) -> Element<WidgetHost> + 'static {
    move |_ctx| {
        let all = gallery.all.clone();
        let sources = gallery.sources.clone();

        // The values the labels read. A `MutableState` read here is a plain
        // borrow of the same cell the widget writes, so a label and its box
        // cannot disagree.
        let all_state = all.get();

        kuitui_element! {
            Column(
                modifier: Modifier::none().padding(EdgeInsets::all(PAGE_PADDING)),
                spacing: ROW_SPACING,
                alignment: Alignment::TOP_START
            ) {
                Text("Kuitui widgets on Vulkan", color: Color::WHITE)
                Text("Checkbox rows through UiPipe and TextPipe.", color: MUTED)
                Row(spacing: CONTROL_SPACING, alignment: Alignment::CENTER) {
                    Checkbox(
                        toggle: all.clone(),
                        tri_state: true,
                        interaction_with: sources.all.clone(),
                    )
                    Text(all_caption(all_state))
                }
            }
        }
    }
}

/// The caption of the master box of the gallery.
fn all_caption(state: CheckState) -> String {
    match state {
        CheckState::Checked => "All rows selected".to_string(),
        CheckState::Unchecked => "No row selected".to_string(),
        CheckState::Indeterminate => "Some rows selected — click to select all".to_string(),
    }
}

/// Records the laid-out tree and, when `reshape` is set, the text of the new
/// list.
///
/// The geometry list is drained by the UI pipe every frame, so it is always
/// recorded again. The text mesh persists until [`TextFrame::clear`], which is
/// why a frame that laid nothing out can skip the shaping pass and still draw
/// the text of the last one.
fn record_frame(
    composition: &mut ReactiveComposition,
    ui_frame: &UiFrame,
    text_frame: &TextFrame,
    font: &Font,
    size: Size,
    reshape: bool,
) -> ExampleResult<()> {
    if reshape {
        text_frame.clear();
        text_frame.begin_frame();
    }
    ui_frame.with(|list| -> ExampleResult<()> {
        composition.render(list);
        list.rect(
            Rect::new(Offset::ZERO, size),
            CornerRadii::ZERO,
            Paint::fill(Color::from_rgba8(26, 28, 36, 255)),
        );
        if reshape {
            shape_list(text_frame, font, list)?;
            text_frame.mark_dirty();
        }
        Ok(())
    })
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
/// the end of the table: the table is read with `pread` from offset zero.
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
