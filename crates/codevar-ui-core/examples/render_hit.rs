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

//! Renders the codevar placeholder triangle into a real Wayland window.
//!
//! The example ties the three pieces of the UI stack together the way
//! `codevar-wl-protocol/examples/xdg_window.rs` drives its `wl_shm`
//! demo, but with the dmabuf path of the editor:
//!
//! * **Wayland (this file):** registry, `xdg-shell` handshake
//!   (`ack_configure`, `xdg_wm_base.ping`/`pong`, frame callbacks,
//!   `wl_buffer.release`) and a `zwp_linux_dmabuf_v1` buffer built from
//!   the exported dma-buf of the render target.
//! * **`ui_pipeline`:** a swapchain-less [`PipelineContext`] whose
//!   offscreen image is created with a modifier the compositor
//!   advertised in its linux-dmabuf feedback and exported as a dma-buf
//!   file descriptor.
//! * **`ui_renderer`:** the [`RendererSubsystem`] frame lifecycle
//!   (`begin_frame` → `render_frame` → `end_frame`), here driven with
//!   the [`TriangleLayer`] of this file as its only layer.
//!
//! Run it inside a Wayland session:
//!
//! ```text
//! cargo run -p codevar-launcher --example render_hit
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use ash::vk;
use codevar_ui_core::ui_pipeline::{OwnedFd, PipelineContext};
use codevar_ui_core::ui_renderer::{
    FrameContext, RenderLayer, RendererError, RendererSubsystem, load_spir_v,
};
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

/// Embedded vertex shader SPIR-V.
static VERT_SPIRV: &[u8] = include_bytes!("shaders/triangle.vert.spv");
/// Embedded fragment shader SPIR-V.
static FRAG_SPIRV: &[u8] = include_bytes!("shaders/triangle.frag.spv");

/// Every failure of the example is reported as a boxed error so `?` works
/// for Wayland, Vulkan and renderer errors alike.
type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    match run() {
        Ok(summary) => println!("{summary}"),
        Err(_) => {
            std::process::exit(1);
        }
    }
}

/// Drives the whole example: Wayland window, Vulkan pipeline, render
/// loop and teardown.
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
        vec![WlArgument::Str(Some(String::from("Codevar render_hit demo")))],
    )?;
    display.marshal_request(
        toplevel,
        XDG_TOPLEVEL_SET_APP_ID,
        vec![WlArgument::Str(Some(String::from("dev.codevar.render-hit")))],
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
    let target = pipeline.render_target();
    let mut renderer = RendererSubsystem::new(&pipeline);
    renderer.start()?;
    renderer
        .layers_mut()
        .add(Box::new(TriangleLayer::new(&pipeline)?))?;
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
        // One frame: record, submit, export completion, wait for the GPU
        // so the committed pixels are final (the alternative would be
        // implicit dma-buf fencing alone).
        renderer.begin_frame(None)?;
        renderer.render_frame()?;
        let sync_file = renderer.end_frame()?;
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

    let _ = renderer.stop();
    drop(renderer);
    drop(pipeline);

    Ok(format!(
        "render_hit: {frames} frames rendered into a {width}x{height} dma-buf window, \
         {} pings answered, configure serial {serial}",
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

/// Creates a shader module from validated SPIR-V words.
fn create_shader_module(device: &ash::Device, words: &[u32]) -> Result<vk::ShaderModule, RendererError> {
    let module_info = vk::ShaderModuleCreateInfo::default().code(words);
    // SAFETY: `words` is a validated, 4-byte-aligned SPIR-V module that
    // outlives the call.
    unsafe { device.create_shader_module(&module_info, None) }.map_err(RendererError::ShaderModuleCreate)
}

/// Creates the graphics pipeline for the placeholder triangle: embedded
/// SPIR-V, no vertex input, dynamic viewport/scissor, dynamic rendering
/// with a single `B8G8R8A8_UNORM` color attachment, no depth.
fn create_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
) -> Result<vk::Pipeline, RendererError> {
    let color_format = PipelineContext::color_format();
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A,
    };
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(core::slice::from_ref(&color_blend_attachment));
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(core::slice::from_ref(&color_format));
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .push_next(&mut rendering_info);
    // SAFETY: every referenced state structure outlives `pipeline_info`,
    // the shader modules belong to `device` and `pipeline_layout` was
    // created on the same device; the color format matches the render
    // target because both come from `PipelineContext::color_format`.
    let result = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            core::slice::from_ref(&pipeline_info),
            None,
        )
    };
    let pipelines = result.map_err(|(_, err)| RendererError::PipelineCreate(err))?;
    pipelines
        .first()
        .copied()
        .ok_or(RendererError::Internal("driver returned no graphics pipeline"))
}

/// The one layer of this example: the embedded placeholder triangle.
///
/// The layer owns its graphics pipeline and therefore borrows the
/// [`PipelineContext`] it was created from; registering it with the
/// renderer keeps it inside the context's lifetime.
struct TriangleLayer<'p> {
    context: &'p PipelineContext,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl<'p> TriangleLayer<'p> {
    /// Loads the embedded SPIR-V, builds an empty pipeline layout and a
    /// dynamic-rendering graphics pipeline matching the render target.
    ///
    /// # Errors
    ///
    /// * [`RendererError::ShaderLoad`] — an embedded module failed
    ///   validation (corrupted build artifacts).
    /// * [`RendererError::ShaderModuleCreate`], [`RendererError::PipelineLayoutCreate`],
    ///   [`RendererError::PipelineCreate`] — the Vulkan call failed;
    ///   everything created so far is destroyed again.
    fn new(context: &'p PipelineContext) -> Result<Self, RendererError> {
        let device = context.device();
        let vertex_words = load_spir_v(VERT_SPIRV)?;
        let fragment_words = load_spir_v(FRAG_SPIRV)?;
        let vertex_module = create_shader_module(device, &vertex_words)?;
        let fragment_module = match create_shader_module(device, &fragment_words) {
            Ok(module) => module,
            Err(err) => {
                // SAFETY: the vertex module was created on this device
                // and is not referenced by anything yet.
                unsafe { device.destroy_shader_module(vertex_module, None) };
                return Err(err);
            }
        };

        // No descriptor sets and no push constants are needed.
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        // SAFETY: a plain, well-formed create info with empty ranges.
        let pipeline_layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
            Ok(layout) => layout,
            Err(err) => {
                // SAFETY: both modules were created on this device and are
                // not referenced by anything yet.
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(fragment_module, None);
                }
                return Err(RendererError::PipelineLayoutCreate(err));
            }
        };

        let pipeline = match create_pipeline(device, pipeline_layout, vertex_module, fragment_module) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                // SAFETY: the layout and both modules were created on
                // this device; the pipeline creation failed, so nothing
                // else references them.
                unsafe {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(fragment_module, None);
                }
                return Err(err);
            }
        };
        // SAFETY: shader modules may be destroyed once the pipeline
        // completed; the pipeline object does not reference them.
        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(fragment_module, None);
        }

        Ok(Self {
            context,
            pipeline_layout,
            pipeline,
        })
    }
}

impl RenderLayer for TriangleLayer<'_> {
    fn name(&self) -> &str {
        "triangle"
    }

    fn priority(&self) -> i32 {
        // Draws underneath every layer a real UI would add later.
        100
    }

    fn render(&mut self, frame: &mut FrameContext<'_>) -> Result<(), RendererError> {
        let device = frame.device();
        let command_buffer = frame.command_buffer();
        // SAFETY: the command buffer is inside the dynamic-rendering
        // scope opened by the renderer and the pipeline was created on
        // this device with the render target's color format.
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
        Ok(())
    }
}

impl Drop for TriangleLayer<'_> {
    fn drop(&mut self) {
        let device = self.context.device();
        // SAFETY: the renderer waits for the device to go idle before its
        // layer stack drops with it, both handles were created by `new`
        // on this device and the pipeline context outlives the layer.
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
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
            (DRM_FORMAT_XRGB8888, 0),
            (0x3432_5241, 0x00E0_0000_0000_0001),
            (DRM_FORMAT_XRGB8888, 0x00E0_0000_0000_0002),
            (DRM_FORMAT_XRGB8888, 0),
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

    /// The shaders embedded in the crate must always decode: this catches
    /// corrupted build artifacts before any window is opened.
    #[test]
    fn embedded_shaders_are_valid_spir_v() {
        assert!(load_spir_v(VERT_SPIRV).is_ok());
        assert!(load_spir_v(FRAG_SPIRV).is_ok());
    }
}
