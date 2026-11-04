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
//! content; this example is the rest of the widget set — the checkboxes,
//! the [`Radio`] group and the [`Slider`] — on the same two pipes, so every
//! shape those widgets paint reaches the GPU through the code that will draw
//! the editor:
//!
//! * **[`kuitui_element!`]:** the content is a reactive tree whose widgets
//!   read and write the handles of [`Gallery`]. A press inside a box writes
//!   the [`MutableState`] that the label next to it reads, so the two cannot
//!   disagree; the handles outlive a render, which is what lets a drag survive
//!   the recomposition its own value writes ask for.
//! * **The gesture stream:** the `wl_pointer` listener queues motion, button
//!   *and wheel* events with the millisecond timestamp the compositor
//!   reported. [`GestureRegion`] needs a clock: without one a widget drags but
//!   never flings.
//! * **[`UiPipe`] + [`TextPipe`]:** the check mark and the outline of a box,
//!   the ring and dot of a radio, the track, ticks and thumb of a slider and
//!   the thumb of the scroll bar are rounded fills, strokes and lines — the
//!   geometry the UI pipe turns into scissored spans and signed-distance
//!   layers. The labels of the same widgets are text runs, which the UI pipe
//!   deliberately leaves to the text pipe, so both are registered and the list
//!   is shaped every time the tree is laid out again.
//! * **[`Viewport`] + [`Scrollbar`]:** the block of lines at the foot of the
//!   page is taller than the box it is given, so its content is laid out
//!   whole, clipped to the box, and shifted by the offset the gestures write.
//!   The box is bounded and clipped from the *modifier chain* of its node, not
//!   from the viewport: a viewport measured with an unbounded height has
//!   nothing to scroll, and the framework only cuts what a chain asks to be
//!   cut.
//! * **The present pool:** the page is drawn through three render targets, each
//!   exported as its own dma-buf and wrapped as its own `wl_buffer`, so a commit
//!   *supersedes* the image on screen instead of overwriting it. The loop
//!   renders the next frame while the compositor still holds the last, sleeps on
//!   the compositor's socket while the GPU writes the one it just submitted, and
//!   commits at the vsync — which is what holds it at the refresh rate.
//!
//! The regions are handles the [`Gallery`] owns rather than fields the widgets
//! build, because the widgets are rebuilt by every recomposition and a drag
//! outlives the one that wrote its first pixel: a slider writes on the press,
//! so the move that follows would reach a region that never saw it.
//!
//! Run it inside a Wayland session, click the boxes, choose a size, drag the
//! slider, and scroll the block of lines with the wheel or a drag of it:
//!
//! ```text
//! cargo run -p codevar-gpu-vk --example widgets_gallery
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use codevar_gpu_vk::base::{Compositor, block_on};
use codevar_gpu_vk::pipeline::{OwnedFd, PipelineContext, RenderTarget};
use codevar_gpu_vk::ui::{UiFrame, UiPipe};
use codevar_kuitui::color::Color;
use codevar_kuitui::event::{Event, PointerButton, PointerButtons, PointerEvent, PointerEventKind};
use codevar_kuitui::geometry::{Alignment, EdgeInsets, Offset, Size};
use codevar_kuitui::host::WidgetHost;
use codevar_kuitui::input::Timestamp;
use codevar_kuitui::node::{Modifier, Node};
use codevar_kuitui::prelude::{
    ColorScheme, FontMetrics, InteractionSource, MutableState, ReactiveComposition, Recomposition, Theme,
    kuitui_element,
};
use codevar_kuitui::rea::{Element, RenderCtx};
use codevar_kuitui::widgets::{
    CheckState, Column, GestureRegion, Radio, RadioGroup, Row, ScrollAxes, ScrollState, Scrollbar, Slider,
    Stack, Text, Viewport,
};
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
/// Frames the loop renders before it reports how it paced them.
const MEASURE_FRAMES: u32 = 400;
/// Wall-clock stop for the measurement, so a compositor that stops answering
/// frame callbacks cannot hold the example open forever.
const MEASURE_WINDOW: Duration = Duration::from_secs(5);
/// How long a beat waits for the compositor to answer the last commit before
/// the loop draws the next frame anyway.
///
/// One refresh period, not the round trip: the wait is a pacing beat, and a
/// frame that misses it is counted and retried instead of costing the loop a
/// timeout's worth of stalled input.
const FRAME_PACE: Duration = Duration::from_millis(20);
/// Render targets the window pools, and so the frames it may keep in flight.
///
/// A target is a rendered image exported as its own dma-buf and wrapped as its
/// own `wl_buffer`, so committing the next one *supersedes* the one on screen —
/// which is the only way a compositor is ever going to send `wl_buffer.release`
/// back. One target cannot: the buffer the loop renders into is the current
/// surface state, so the loop would have to wait for the display before it
/// could write again. Three is what `MAX_PRESENT_SLOTS` pools in the editor,
/// and it is enough to cover the round trip the compositor takes to retire a
/// superseded buffer.
const PRESENT_SLOTS: usize = 3;
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
/// The width the slider is given, in logical pixels.
const SLIDER_WIDTH: f32 = 260.0;
/// The side of a clickable pill.
const BUTTON_HEIGHT: f32 = 34.0;
/// The width of the scrolled block, which is the page inside its padding.
const LIST_WIDTH: f32 = 596.0;
/// The height of the box the lines scroll inside.
///
/// The block is deliberately taller than this, since a box that fits its
/// content has nothing to scroll and no thumb to paint.
const LIST_HEIGHT: f32 = 132.0;
/// The lines the scrolled block holds.
const LIST_LINES: usize = 20;
/// Pixels the measurement creeps the block of lines by each frame.
///
/// A window with nothing changing has no frame to draw and no rate to measure,
/// so the loop drives the one animation that touches every stage: the viewport
/// moves, which dirties the tree, which lays it out again, which reshapes the
/// text the mesh carries.
const DRIVE_PIXELS: f32 = 1.0;

/// Every failure of the example is reported as a boxed error so `?` works for
/// Wayland, Vulkan, compositor and shaping errors alike.
type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

/// The option a radio of the density group stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Density {
    /// The compact spacing.
    Compact,
    /// The default spacing.
    Comfortable,
    /// The roomy spacing.
    Roomy,
}

impl Density {
    /// The caption of the option.
    const fn label(self) -> &'static str {
        match self {
            Self::Compact => "Compact",
            Self::Comfortable => "Comfortable",
            Self::Roomy => "Roomy",
        }
    }

    /// The extra spacing the option asks the gallery for, in pixels.
    const fn spacing(self) -> f32 {
        match self {
            Self::Compact => -4.0,
            Self::Comfortable => 0.0,
            Self::Roomy => 8.0,
        }
    }
}

/// The interaction states the gallery keeps, one source per widget.
///
/// A source handed to a widget is a handle, so the hover and press layers the
/// widget reports survive the recomposition that rebuilt it — which is what
/// lets a slider keep its pressed layer while its own value forces a render.
#[derive(Clone)]
struct Sources {
    /// The master box of the header row.
    all: InteractionSource,
    /// The box that enables the slider.
    enabled: InteractionSource,
    /// The box that snaps the slider to its ticks.
    stepped: InteractionSource,
    /// The three radios, in the order the row lays them out.
    radios: [InteractionSource; 3],
    /// The slider.
    slider: InteractionSource,
    /// The reset button.
    button: InteractionSource,
}

impl Sources {
    /// The sources of the gallery, each holding nothing yet.
    fn new() -> Self {
        Self {
            all: InteractionSource::new(),
            enabled: InteractionSource::new(),
            stepped: InteractionSource::new(),
            radios: [
                InteractionSource::new(),
                InteractionSource::new(),
                InteractionSource::new(),
            ],
            slider: InteractionSource::new(),
            button: InteractionSource::new(),
        }
    }
}

/// The gesture regions the widgets of the gallery read through.
///
/// A region is a handle too, over the recognition of the press in flight, and
/// it is hoisted for the same reason [`Sources`] is: a content closure rebuilds
/// the widget every render. The widgets that answer a press without writing
/// anything — the boxes and the radios — survive either way, because their
/// press and release land in the same drain of events. The slider and the
/// viewport do not: each writes a value the tree reads back, so the move that
/// follows a press arrives after the recomposition that built a new widget.
#[derive(Clone)]
struct Regions {
    /// The slider of the volume row.
    slider: GestureRegion,
    /// The scrolled block of lines.
    list: GestureRegion,
}

impl Regions {
    /// The regions of the gallery, each recognizing nothing yet.
    fn new() -> Self {
        Self {
            slider: GestureRegion::default(),
            list: GestureRegion::default(),
        }
    }
}

/// One present target of the pool: the image the GPU renders, the compositor
/// that folds the two pipes into it, and the `wl_buffer` that shows the image.
///
/// The three belong together because a target is more than memory. Each owns a
/// Vulkan device and the pipelines built on it, and its pipes read a
/// [`TextFrame`] of their own: the text pipe uploads the atlas pages to the
/// device it was registered on, so a second compositor handed the first one's
/// frame would sample an atlas it never filled. The [`UiFrame`] is shared —
/// the geometry list is drained by whichever compositor is run, and the loop
/// records exactly once per target.
struct Slot<'p> {
    /// The renderer that writes this target's image.
    compositor: Compositor<'p>,
    /// The text the frame draws, with the atlas this device uploads.
    text_frame: TextFrame,
    /// The `wl_buffer` wrapping the exported dma-buf.
    buffer: WlProxyId,
    /// Whether the compositor has handed this buffer back. A commit clears it,
    /// and the release of the superseded buffer sets it again.
    released: Rc<Cell<bool>>,
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
    /// The box that enables the slider.
    enabled: MutableState<CheckState>,
    /// The box that snaps the slider to its ticks.
    stepped: MutableState<CheckState>,
    /// The thumb position of the slider.
    progress: MutableState<f32>,
    /// The option the density group holds.
    density: RadioGroup<Density>,
    /// The presses the reset button completed.
    resets: MutableState<u32>,
    /// The density selection the loop has already seen, as a revision count.
    ///
    /// A [`RadioGroup`] exposes a revision rather than the state behind it, so
    /// the loop notices a click on a radio by watching that number move.
    density_seen: Cell<u64>,
    /// The interaction states of the widgets.
    sources: Sources,
    /// The scroll position of the block of lines.
    list: ScrollState,
    /// The gesture regions the widgets read through.
    regions: Regions,
}

impl Gallery {
    /// A gallery with nothing chosen: the first box indeterminate, the slider
    /// off and disabled, and the middle size selected.
    fn new() -> Self {
        Self {
            all: MutableState::new(CheckState::Indeterminate),
            enabled: MutableState::new(CheckState::Unchecked),
            stepped: MutableState::new(CheckState::Checked),
            progress: MutableState::new(0.35),
            density: RadioGroup::with_value(Density::Comfortable),
            resets: MutableState::new(0_u32),
            density_seen: Cell::new(0),
            sources: Sources::new(),
            list: ScrollState::new(ScrollAxes::VERTICAL),
            regions: Regions::new(),
        }
    }

    /// The state [`Self::new`] starts with, except for the reset count, which
    /// is the history of the session.
    fn reset(&self) {
        self.all.set(CheckState::Indeterminate);
        self.enabled.set(CheckState::Unchecked);
        self.stepped.set(CheckState::Checked);
        self.progress.set(0.35);
        self.density.select(&Density::Comfortable);
        // The block goes back to its first line, which is the one place the
        // offset is written by something other than a gesture.
        let top = self.list.offset().y;
        self.list.scroll_by(Offset::new(0.0, -top));
    }

    /// Move the block of lines down by `step`, and back to its first line once
    /// its last one has left the box.
    ///
    /// This is the animation the frame measurement runs on: the scroll writes
    /// the offset the layout reads, so every frame rebuilds, lays out and
    /// reshapes the whole page instead of reporting the rate of a window that
    /// had nothing to draw. Wrapping keeps the block scrolling for as long as
    /// the loop runs rather than parking against its bottom.
    fn drive(&self, step: f32) {
        let offset = self.list.offset().y;
        let bottom = self.list.max().y;
        if offset + step >= bottom {
            self.list.scroll_by(Offset::new(0.0, -offset));
        } else {
            self.list.scroll_by(Offset::new(0.0, step));
        }
    }

    /// Whether a widget wrote one of the values since the last frame.
    ///
    /// `take_dirty` clears the flag as it reads it, so a frame that wrote
    /// nothing recomposes nothing.
    fn take_dirty(&self) -> bool {
        let mut dirty = false;
        dirty |= self.all.take_dirty();
        dirty |= self.enabled.take_dirty();
        dirty |= self.stepped.take_dirty();
        // The progress is written on every move of a drag, so it asks for a
        // rebuild on its own; a rebuild is what re-asks the slider whether it
        // snaps and whether it answers the pointer at all.
        dirty |= self.progress.take_dirty();
        dirty |= self.resets.take_dirty();
        // A scroll writes the offset the layout reads, so it asks for the tree
        // to be placed again; the recomposition it costs is what the host
        // regions make harmless.
        dirty |= self.list.take_dirty();
        let revision = self.density.revision();
        dirty |= revision != self.density_seen.replace(revision);
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
    // The seat is the input side of the window. The protocol only lets a
    // client ask for a pointer once the seat announces the capability, so the
    // listener below creates it when that event arrives and hands the raw
    // events to `PointerInput`, which the frame loop replays into the tree.
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
    // The present pool. Each target is a render image of its own, exported as a
    // dma-buf of its own and wrapped as a `wl_buffer` of its own, so a commit
    // *supersedes* the buffer on screen instead of overwriting it: the loop can
    // render the next frame while the compositor still holds the last, and the
    // compositor has something to release.
    let mut pipelines = Vec::with_capacity(PRESENT_SLOTS);
    for _ in 0..PRESENT_SLOTS {
        let pipeline = match PipelineContext::new(width as u32, height as u32, requested) {
            Ok(context) => context,
            Err(_) if requested.len() == 1 => PipelineContext::new(width as u32, height as u32, &modifiers)?,
            Err(error) => return Err(error.into()),
        };
        pipelines.push(pipeline);
    }

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
    let mut slots = Vec::with_capacity(PRESENT_SLOTS);
    for pipeline in &pipelines {
        let text_frame = TextFrame::new(TextAtlas::new(AtlasConfig::DEFAULT)?)?;
        let mut compositor = Compositor::new(pipeline);
        compositor.start()?;
        let mut ui_pipe = UiPipe::new(pipeline, ui_frame.clone())?;
        ui_pipe.set_clear(BACKGROUND);
        compositor.add_pipe(ui_pipe)?;
        let mut text_pipe = TextPipe::new(pipeline, text_frame.clone())?;
        text_pipe.set_clear(TEXT_CLEAR);
        compositor.add_pipe(text_pipe)?;
        let (buffer, released) =
            import_buffer(&mut display, dmabuf, pipeline.present_target(), width, height)?;
        slots.push(Slot {
            compositor,
            text_frame,
            buffer,
            released,
        });
    }

    // Every target is shaped before the loop starts. A frame that does not
    // reshape leaves the last mesh of *that* target on screen, so a target the
    // loop had not drawn yet would draw the page without its text — and the
    // cold rasterization each atlas pays once belongs here, not inside the
    // frames being measured.
    for slot in &mut slots {
        record_frame(&mut composition, &ui_frame, &slot.text_frame, font.as_ref(), true)?;
    }
    let mut frames = 0_u32;
    // Where a frame's time goes, accumulated so the report names the stage
    // that binds the loop instead of only counting the frames it made.
    let mut free_wait = Duration::ZERO;
    let mut pace_wait = Duration::ZERO;
    let mut place_time = Duration::ZERO;
    let mut record_time = Duration::ZERO;
    let mut submit_time = Duration::ZERO;
    let mut gpu_wait = Duration::ZERO;
    // Frames that started with no target the compositor had handed back, and
    // beats that found no frame callback waiting for them. The first says the
    // pool is too shallow to hide the display's round trip; the second, that
    // the compositor stopped answering commits.
    let mut held = 0_u32;
    let mut stalled = 0_u32;
    let measured = Instant::now();
    // The clock of the gallery is the compositor's: the `time` of every
    // pointer event is milliseconds since the client connected, so the
    // elapsed step handed to the widgets is measured the same way.
    let mut last_time = Timestamp::UNSPECIFIED;
    // The commit that still owes the loop its frame callback, as the callback
    // proxy and the flag its `done` sets. Waiting on that callback is what
    // paces the loop to the refresh rate rather than to a timeout.
    let mut pending: Option<(WlProxyId, Rc<Cell<bool>>)> = None;
    // The target the next frame renders into, walked in commit order so the
    // image the loop is offered is always the oldest one on screen.
    let mut cursor = 0_usize;
    while !closed.get() && frames < MEASURE_FRAMES && measured.elapsed() < MEASURE_WINDOW {
        // A target the compositor has handed back — the release of the oldest
        // buffer, which arrives as soon as committing a newer one supersedes
        // it. This is the wait that keeps the GPU from writing into the image
        // still on screen, and the reason for the pool: a single target is
        // never superseded, so its release never comes.
        let stage = Instant::now();
        let count = slots.len();
        let ready = wait_for(&mut display, &closed, FRAME_PACE, || {
            (0..count).any(|offset| slots[(cursor + offset) % count].released.get())
        })?;
        free_wait += stage.elapsed();
        if !ready {
            held += 1;
        }
        let offset = (0..count)
            .find(|offset| slots[(cursor + offset) % count].released.get())
            .unwrap_or(0);
        let slot = &mut slots[(cursor + offset) % count];
        cursor = (cursor + offset + 1) % count;

        // Replay what the pointer did while the last frame was in flight, in
        // the order the compositor reported it. A press that completes inside
        // a box writes that box's state; a move past the slop drags the
        // slider's thumb.
        let stage = Instant::now();
        let mut consumed = 0_u32;
        let mut frame_time = last_time;
        {
            let mut input = input.borrow_mut();
            for event in input.events.drain(..) {
                if let Event::Pointer(pointer) = &event
                    && pointer.time.is_specified()
                {
                    frame_time = pointer.time;
                }
                if composition.handle_event(&event).is_consumed() {
                    consumed += 1;
                }
            }
        }
        // A release fast enough to carry the content moves a widget by itself,
        // so the animation runs before the rebuild: this frame draws from the
        // position the gesture reached.
        let elapsed = if last_time.is_specified() && frame_time.is_specified() {
            Timestamp::from_micros(
                frame_time
                    .micros()
                    .saturating_sub(last_time.micros())
                    .max(1_000),
            )
        } else {
            Timestamp::UNSPECIFIED
        };
        last_time = frame_time;
        let animated = elapsed.is_specified() && composition.advance(elapsed);
        // The block of lines creeps on its own, because a window with nothing
        // changing has no frame to draw and no rate to measure.
        gallery.drive(DRIVE_PIXELS);
        // The engine schedules the renders its hooks own; a widget writing a
        // handle the gallery created is only seen through that handle.
        let written = gallery.take_dirty();
        let rebuilt = matches!(composition.recompose(written), Recomposition::Recomposed);
        composition.layout(size, rebuilt || written || animated);
        place_time += stage.elapsed();

        // A hover or a press changes what a widget paints without touching the
        // engine, so the geometry is recorded every frame. The text is shaped
        // whenever the tree was laid out again, because a relayout moves the
        // runs the mesh carries.
        let stage = Instant::now();
        record_frame(
            &mut composition,
            &ui_frame,
            &slot.text_frame,
            font.as_ref(),
            rebuilt || written || animated,
        )?;
        record_time += stage.elapsed();
        let stage = Instant::now();
        let sync_file = block_on(slot.compositor.composite_frame(None))?;
        submit_time += stage.elapsed();

        // The frame is in flight, so the display's own wait runs beside it:
        // `dispatch` sleeps on the compositor's socket while the GPU writes the
        // image, and the two overlap instead of adding up. The wait is what
        // paces the loop to the refresh period — the commit lands at the vsync
        // rather than after it, because the work that had to finish first is
        // already finished by the time the callback answers.
        let stage = Instant::now();
        let shown = wait_for(&mut display, &closed, FRAME_PACE, || {
            pending
                .as_ref()
                .is_none_or(|(_, done)| done.get())
        })?;
        pace_wait += stage.elapsed();
        if shown {
            if let Some((callback, _)) = pending.take() {
                display.proxy_destroy(callback)?;
            }
        } else {
            stalled += 1;
        }

        // Only now may the buffer be offered. The compositor reads the dma-buf
        // as soon as it is attached and `codevar-wayland` has no in-fence
        // protocol to tell it to hold off, so the image has to be finished by
        // the time this returns — which it nearly always is, since the wait
        // above gave the GPU the whole round trip to get there.
        let stage = Instant::now();
        wait_for_gpu(&sync_file)?;
        drop(sync_file);
        gpu_wait += stage.elapsed();

        let callback = display.marshal_new_id(surface, SURFACE_FRAME, Vec::new())?;
        let done = Rc::new(Cell::new(false));
        {
            let done = Rc::clone(&done);
            display.add_callback_listener(callback, move |_, _| {
                done.set(true);
                0
            })?;
        }
        display.marshal_request(
            surface,
            SURFACE_ATTACH,
            vec![
                WlArgument::Object(slot.buffer.id()),
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
        slot.released.set(false);
        pending = Some((callback, done));
        frames += 1;
        let _ = consumed;
    }
    let measured_for = measured.elapsed();
    display.flush()?;
    let _ = display.marshal_request(toplevel, XDG_TOPLEVEL_DESTROY, Vec::new());
    let _ = display.marshal_request(xdg_surface, XDG_SURFACE_DESTROY, Vec::new());
    let _ = display.marshal_request(surface, SURFACE_DESTROY, Vec::new());
    for slot in &slots {
        let _ = display.marshal_request(slot.buffer, BUFFER_DESTROY, Vec::new());
    }
    if let Some(feedback) = feedback_proxy {
        let _ = display.marshal_request(feedback, FEEDBACK_DESTROY, Vec::new());
    }
    let _ = display.marshal_request(dmabuf, DMABUF_DESTROY, Vec::new());
    let _ = display.flush();

    for slot in &mut slots {
        let _ = slot.compositor.stop();
    }
    drop(slots);
    drop(pipelines);

    // The distance the block can be scrolled is reported because it is the one
    // number that says whether the box was bounded: a viewport measured with an
    // unbounded height has a content as tall as its box, a maximum of zero, and
    // gestures that answer a wheel which moves nothing.
    let scrollable = gallery.list.max().y;
    let per_frame = if frames == 0 {
        0.0
    } else {
        measured_for.as_secs_f64() * 1_000.0 / f64::from(frames)
    };
    let fps = if measured_for.is_zero() {
        0.0
    } else {
        f64::from(frames) / measured_for.as_secs_f64()
    };
    let average = |wait: Duration| {
        if frames == 0 {
            0.0
        } else {
            wait.as_secs_f64() * 1_000.0 / f64::from(frames)
        }
    };
    Ok(format!(
        "widgets_gallery: {frames} frames in {:.1} ms at {fps:.1} fps, {per_frame:.3} per frame \
         over {PRESENT_SLOTS} present targets (free {:.2}, pace {:.2}, place {:.2}, record \
         {:.2}, submit {:.2}, gpu {:.2} ms each), {stalled} beats without a frame callback, \
         {held} frames with no released target, {} pings answered, configure serial {serial}, \
         {scrollable:.0} of scroll to report",
        measured_for.as_secs_f64() * 1_000.0,
        average(free_wait),
        average(pace_wait),
        average(place_time),
        average(record_time),
        average(submit_time),
        average(gpu_wait),
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
        let enabled = gallery.enabled.clone();
        let stepped = gallery.stepped.clone();
        let progress = gallery.progress.clone();
        let density_group = gallery.density.clone();
        let resets = gallery.resets.clone();
        let sources = gallery.sources.clone();

        // The values the labels read. A `MutableState` read here is a plain
        // borrow of the same cell the widget writes, so a label and its box
        // cannot disagree.
        let all_state = all.get();
        let enabled_state = enabled.get();
        let stepped_state = stepped.get();
        let shown = progress.get();
        let chosen = density_group.current();
        let clicks = resets.get();
        let on = enabled_state.is_checked();
        let snapped = stepped_state.is_checked();
        let extra = chosen.map_or(0.0, Density::spacing);

        // The master box answers for the two below it: a click that leaves it
        // checked or unchecked writes both children, and an indeterminate one
        // leaves them alone. Each handler owns the clones it writes with,
        // because a closure that moved a handle would take it out of the render.
        let master = all.clone();
        let master_enabled = enabled.clone();
        let master_stepped = stepped.clone();
        let cascade = move |state: CheckState| {
            master.set(state);
            if state == CheckState::Indeterminate {
                return;
            }
            master_enabled.set(state);
            master_stepped.set(state);
        };
        let recount = || recount_handler(all.clone(), enabled.clone(), stepped.clone());

        // A `Node(..)` splice is a factory the engine owns, so the handles it
        // reads move into that factory and cannot be borrowed from the render:
        // each splice gets a clone of its own, made before the macro runs.
        let radio_group = density_group.clone();
        let radio_sources = sources.clone();
        let slider_value = progress.clone();
        let slider_source = sources.slider.clone();
        let reset_gallery = gallery.clone();
        let reset_count = resets.clone();
        let reset_source = sources.button.clone();
        let slider_region = gallery.regions.slider.clone();
        let list_state = gallery.list.clone();
        let list_region = gallery.regions.list.clone();

        kuitui_element! {
            Column(
                modifier: Modifier::none().padding(EdgeInsets::all(PAGE_PADDING)),
                spacing: ROW_SPACING + extra,
                alignment: Alignment::TOP_START
            ) {
                Text("Kuitui widgets on Vulkan", color: Color::WHITE)
                Row(spacing: CONTROL_SPACING, alignment: Alignment::CENTER) {
                    Checkbox(
                        toggle: all.clone(),
                        tri_state: true,
                        interaction_with: sources.all.clone(),
                        on_change: cascade
                    )
                    Text(all_caption(all_state))
                }
                Row(spacing: CONTROL_SPACING, alignment: Alignment::CENTER) {
                    Text("Density")
                    Node(radio_row(radio_group.clone(), radio_sources.clone()))
                    Text(chosen.map_or("none".to_string(), |option| option.label().to_string()))
                }
                Row(spacing: CONTROL_SPACING, alignment: Alignment::CENTER) {
                    Checkbox(
                        toggle: enabled.clone(),
                        interaction_with: sources.enabled.clone(),
                        on_change: recount()
                    )
                    Text(if on { "Volume" } else { "Volume (disabled)" }, color: Color::WHITE)
                    Node(slider_node(
                        slider_value.clone(),
                        SLIDER_WIDTH,
                        on,
                        snapped,
                        slider_source.clone(),
                        slider_region.clone(),
                    ))
                    Text(format!("{shown:>5.1}"), color: Color::WHITE)
                }
                Row(spacing: CONTROL_SPACING, alignment: Alignment::CENTER) {
                    Checkbox(
                        toggle: stepped.clone(),
                        interaction_with: sources.stepped.clone(),
                        on_change: recount()
                    )
                    Text("Snap to ticks")
                    Node(reset_node(reset_gallery.clone(), reset_count.clone(), reset_source.clone()))
                    Text(format!("{clicks} resets"), color: MUTED)
                }
                Text("A viewport: scroll the lines with the wheel, or drag them.", color: MUTED)
                Node(list_node(
                    list_state.clone(),
                    list_region.clone(),
                    ROW_SPACING + extra,
                ))
            }
        }
    }
}

/// The handler a child box calls to recompute what the master box reports.
///
/// The two children answer independently, so the master is checked when both
/// agree, unchecked is both empty, and indeterminate while they disagree.
fn recount_handler(
    all: MutableState<CheckState>,
    enabled: MutableState<CheckState>,
    stepped: MutableState<CheckState>,
) -> impl Fn(CheckState) + 'static {
    move |_| {
        let enabled = enabled.get();
        let stepped = stepped.get();
        all.set(if enabled == stepped {
            enabled
        } else {
            CheckState::Indeterminate
        });
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

/// The row of radios that answers for the density group.
///
/// A radio is built with two arguments — the option it stands for and the
/// group it writes — so it is described as a node and spliced into the macro
/// with `Node(..)`. The group is a handle shared by all three buttons, which is
/// how choosing one draws the dot in that button and clears the other two.
///
/// The ring of a radio measures the touch target and paints nothing else, so
/// each option pairs its button with the caption the same row lays out.
fn radio_row(group: RadioGroup<Density>, sources: Sources) -> Node {
    let mut options = Vec::new();
    for (index, option) in [Density::Compact, Density::Comfortable, Density::Roomy]
        .into_iter()
        .enumerate()
    {
        options.push(
            Node::new(
                Row::new()
                    .spacing(4.0)
                    .alignment(Alignment::CENTER),
            )
            .with_children([
                Node::new(Radio::new(option, group.clone()).interaction_with(sources.radios[index].clone())),
                Node::new(Text::new(option.label()).color(MUTED)),
            ]),
        );
    }
    Node::new(
        Row::new()
            .spacing(10.0)
            .alignment(Alignment::CENTER),
    )
    .with_children(options)
}

/// The slider of the volume row, given `width` and the two switches the boxes
/// above it write: a disabled slider answers nothing, and a stepped one snaps
/// to its ticks.
fn slider_node(
    progress: MutableState<f32>,
    width: f32,
    on: bool,
    snapped: bool,
    source: InteractionSource,
    region: GestureRegion,
) -> Node {
    Node::new(
        Slider::new(progress)
            .range(0.0, 100.0)
            .steps(if snapped { 10 } else { 0 })
            .enabled(on)
            .interaction_with(source)
            .region(region),
    )
    .modifier(Modifier::none().size(Some(width), None))
}

/// The block of lines that scrolls: a viewport over the state the gallery
/// owns, with the bar that reports where in them the box stands.
///
/// The two are siblings of a [`Stack`] rather than parent and child. The bar
/// reads the same [`ScrollState`] handle the viewport writes, so it needs
/// nothing from the viewport, and a stack paints its children in the order of
/// the tree: the bar last, over the lines it reports on. It is not
/// [`hit_testable`](codevar_kuitui::widget::Widget::hit_testable), so a press
/// on it is a press on the content underneath and drags the lines.
///
/// The bounded box comes from the *modifier chain* of the viewport node, not
/// from the viewport: `size` is what keeps the height from growing to the
/// whole document — a viewport measured with the unbounded height a
/// [`Column`] offers its children has nothing left to scroll — and `clip` is
/// what keeps the lines below the box painted over the rest of the page.
fn list_node(state: ScrollState, region: GestureRegion, spacing: f32) -> Node {
    let mut lines = Vec::new();
    for index in 0..LIST_LINES {
        lines.push(Node::new(
            Text::new(format!("{:>2}  vkQueueSubmit(q, n);", index + 1)).color(MUTED),
        ));
    }
    let content = Node::new(
        Column::new()
            .spacing(spacing)
            .alignment(Alignment::TOP_START),
    )
    .with_children(lines);
    let scrolled = Node::new(
        Viewport::new()
            .scroll(state.clone())
            .region(region),
    )
    .modifier(
        Modifier::none()
            .size(Some(LIST_WIDTH), Some(LIST_HEIGHT))
            .clip(),
    )
    .child(content);
    Node::new(Stack::new()).with_children([scrolled, Node::new(Scrollbar::new(state))])
}

/// The button that puts the gallery back the way
/// [`Gallery::new`] left it, counting how many times it was pressed.
fn reset_node(gallery: Gallery, resets: MutableState<u32>, source: InteractionSource) -> Node {
    TextButton::new(
        move || {
            gallery.reset();
            resets.set(resets.get().wrapping_add(1));
        },
        "Reset",
    )
    .interaction_source(source)
    .modifier(Modifier::none().size(None, Some(BUTTON_HEIGHT)))
    .node()
}

/// Dispatches Wayland events until `ready` answers or the beat is spent.
///
/// The wait is on the socket, not on a clock: `dispatch` returns as soon as the
/// compositor sends anything — which is also what lets a pointer event reach
/// the loop between two frames — and only the time left in `beat` bounds how
/// long the wait can hold the loop when nothing arrives. `false` is the beat
/// ending unanswered, never an error: a compositor that stops answering the
/// frame callback costs the loop one frame of latency, not the window.
fn wait_for(
    display: &mut WlClientDisplay<WlUnixTransport>,
    closed: &Cell<bool>,
    beat: Duration,
    ready: impl Fn() -> bool,
) -> ExampleResult<bool> {
    let deadline = Instant::now() + beat;
    loop {
        if ready() {
            return Ok(true);
        }
        if closed.get() {
            return Ok(false);
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Ok(false);
        }
        let millis = u64::try_from(left.as_millis())
            .unwrap_or(1)
            .clamp(1, 1_000);
        display.dispatch(Some(TimeDuration::from_millis(millis)))?;
    }
}

/// Wraps one exported render target as a `wl_buffer` and watches its release.
///
/// The params object is a one-shot builder — created, given the plane of the
/// dma-buf, consumed by `create_immed`, destroyed — so each target of the pool
/// is imported through its own. The released flag starts set because nothing
/// has committed the buffer yet, and the listener is what a superseded commit
/// answers: the release of one buffer is the permission to render into it.
fn import_buffer(
    display: &mut WlClientDisplay<WlUnixTransport>,
    dmabuf: WlProxyId,
    target: &RenderTarget,
    width: i32,
    height: i32,
) -> ExampleResult<(WlProxyId, Rc<Cell<bool>>)> {
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
    Ok((buffer, released))
}

/// Records the laid-out tree and, when `reshape` is set, the text of the new
/// list.
///
/// The geometry list is drained by the UI pipe every frame, so it is always
/// recorded again. The text mesh persists until [`TextFrame::clear`], which is
/// why a frame that laid nothing out can skip the shaping pass and still draw
/// the text of the last one.
///
/// `text_frame` is the frame of the target about to be composited, not of the
/// window: a text pipe uploads its atlas to the device it was registered on, so
/// each target shapes and caches the same glyphs on its own.
fn record_frame(
    composition: &mut ReactiveComposition,
    ui_frame: &UiFrame,
    text_frame: &TextFrame,
    font: &Font,
    reshape: bool,
) -> ExampleResult<()> {
    if reshape {
        text_frame.clear();
        text_frame.begin_frame();
    }
    ui_frame.with(|list| -> ExampleResult<()> {
        composition.render(list);
        if reshape {
            shape_list(text_frame, font, list)?;
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
/// The opcode of `wl_pointer.axis`: time, axis, value.
///
/// The `codevar-wayland` crate names the four events of version 1 that the
/// other examples need; `axis` is the fifth event of that interface, and it is
/// the only one this gallery adds: the continuous value it carries is already
/// in surface-local pixels, so the discrete and value120 events of version 5
/// and 8 would double the same scroll.
const POINTER_AXIS: u32 = 4;
/// The `wl_pointer.axis` enum code of the vertical axis; `1` is horizontal.
const AXIS_VERTICAL: u32 = 0;

/// The pointer events a `wl_pointer` listener queues for the frame loop.
///
/// The listener cannot reach the composition — the seat is bound before the
/// tree exists — so it records the events and the frame loop replays them into
/// the tree, where the gesture region of each widget updates the state it owns.
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
                self.moved(args.get(2), args.get(3), PointerEventKind::Enter, None);
            }
            POINTER_LEAVE => {
                // serial, surface: the event carries no coordinates, so the
                // pointer is reported just outside the window, where every
                // widget clears its hover.
                self.position = Offset::new(-1.0, -1.0);
                self.queue(PointerEventKind::Leave, PointerButton::Primary, None);
            }
            POINTER_MOTION => {
                // time, surface_x, surface_y
                self.moved(args.get(1), args.get(2), PointerEventKind::Move, args.first());
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
                self.queue(kind, pointer_button(code), args.get(1));
            }
            POINTER_AXIS => {
                // time, axis, value
                let (Some(WlArgument::Uint(axis)), Some(WlArgument::Fixed(value))) =
                    (args.get(1), args.get(2))
                else {
                    return;
                };
                let delta = value.to_double() as f32;
                if delta == 0.0 {
                    return;
                }
                let position = self.position;
                let mut event = Event::Scroll(codevar_kuitui::event::ScrollEvent::new(
                    if *axis == AXIS_VERTICAL {
                        Offset::new(0.0, delta)
                    } else {
                        Offset::new(delta, 0.0)
                    },
                    position,
                ));
                if let Event::Scroll(scroll) = &mut event {
                    scroll.time = timestamp_arg(args.first());
                }
                self.events.push(event);
            }
            _ => {}
        }
    }

    /// Moves the pointer to the fixed-point coordinates of `x` and `y` and
    /// queues `kind` there.
    fn moved(
        &mut self,
        x: Option<&WlArgument>,
        y: Option<&WlArgument>,
        kind: PointerEventKind,
        time: Option<&WlArgument>,
    ) {
        let (Some(WlArgument::Fixed(x)), Some(WlArgument::Fixed(y))) = (x, y) else {
            return;
        };
        self.position = Offset::new(x.to_double() as f32, y.to_double() as f32);
        self.queue(
            kind,
            PointerButton::Primary,
            Some(time.unwrap_or(&WlArgument::Uint(0))),
        );
    }

    /// Queues a pointer event at the current position, stamped with `time`.
    fn queue(&mut self, kind: PointerEventKind, button: PointerButton, time: Option<&WlArgument>) {
        let mut event = PointerEvent::new(kind, self.position).with_time(timestamp_arg(time));
        event.button = button;
        event.buttons = self.buttons;
        self.events.push(Event::Pointer(event));
    }
}

/// The compositor timestamp of an event argument, in the microseconds the
/// [`Timestamp`] of the tree counts.
///
/// A gesture region measures the speed of a release from these stamps, so a
/// missing one costs the fling rather than the drag.
fn timestamp_arg(arg: Option<&WlArgument>) -> Timestamp {
    match arg {
        Some(WlArgument::Uint(millis)) => Timestamp::from_micros(u64::from(*millis) * 1_000),
        _ => Timestamp::UNSPECIFIED,
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
