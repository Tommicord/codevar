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

//! Draws the codevar placeholder triangle in a browser tab.
//!
//! The example ties the two pieces of the web stack together:
//!
//! * **`WgContext`** opens the canvas surface, the adapter and the
//!   logical device, and owns the fence the frame signals.
//! * **the recording loop (this file):** acquire → transition →
//!   `begin_render_pass` → `set_render_pipeline` → `draw` → submit →
//!   present, once per `requestAnimationFrame`.
//!
//! It bypasses `codevar_gpu_wg::renderer::WgRenderer` on purpose: that
//! subsystem only records a clear pass until the mix pass is ported, so
//! the triangle is recorded directly against the acquired canvas texture.
//!
//! The workspace configures `wasm-server-runner` as the runner for
//! `wasm32-unknown-unknown`, so a plain `cargo run` starts a local server
//! and opens the page:
//!
//! ```text
//! cargo run -p codevar-gpu-wg --example triangle --target wasm32-unknown-unknown
//! ```

/// Every failure of the example is reported as a boxed error so `?` works
/// for the browser, the context and the raw `wgpu-hal` calls alike.
#[cfg(target_arch = "wasm32")]
type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

#[cfg(target_arch = "wasm32")]
mod web {
    use std::cell::RefCell;
    use std::rc::Rc;

    use codevar_gpu_wg::pipeline::{WgConfig, WgContext};
    use wasm_bindgen::prelude::*;
    use web_sys::{HtmlCanvasElement, Window};
    use wgpu_hal::gles::{CommandEncoder, RenderPipeline};
    use wgpu_hal::{
        Attachment, AttachmentOps, ColorAttachment, CommandEncoder as _, CommandEncoderDescriptor,
        Device as _, PipelineLayoutDescriptor, PipelineLayoutFlags, ProgrammableStage, Queue as _,
        RenderPassDescriptor, RenderPipelineDescriptor, ShaderInput, ShaderModuleDescriptor, StateTransition,
        Surface as _, TextureBarrier, TextureViewDescriptor, VertexProcessor,
    };
    use wgpu_types as wgt;

    use super::ExampleResult;

    /// The canvas the example draws into. Created on the fly when the page
    /// does not ship one.
    const CANVAS_ID: &str = "codevar-triangle";

    /// Vertex shader GLSL ES 3.00, passed through to the WebGL2 backend.
    const VERTEX_GLSL: &str = include_str!("shaders/triangle.vert.glsl");
    /// Fragment shader GLSL ES 3.00, passed through to the WebGL2 backend.
    const FRAGMENT_GLSL: &str = include_str!("shaders/triangle.frag.glsl");

    /// The width the canvas falls back to when the window size is unknown.
    const FALLBACK_WIDTH: u32 = 640;
    /// The height the canvas falls back to when the window size is unknown.
    const FALLBACK_HEIGHT: u32 = 480;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        pub fn log(message: &str);
    }

    /// The slot a frame callback is parked in.
    ///
    /// A `requestAnimationFrame` callback must be `'static`, so the
    /// closure is owned by this slot for as long as the frame loop runs.
    /// Each tick clones the `Rc` and reschedules the same slot, which is
    /// the standard way to keep a browser callback alive from Rust.
    type FrameSlot = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

    /// Formats a raw `wgpu-hal` failure as the example's boxed error.
    fn failure(operation: &str, error: impl core::fmt::Display) -> Box<dyn std::error::Error> {
        format!("{operation}: {error}").into()
    }

    /// Formats a browser `JsValue` failure as the example's boxed error.
    ///
    /// `JsValue` is not a `std::error::Error`, so the `?` operator cannot
    /// convert it for free; its `Debug` form is the browser's message.
    fn js_failure(error: JsValue) -> Box<dyn std::error::Error> {
        format!("{error:?}").into()
    }

    /// The full subresource range of a single-level, single-layer texture.
    fn full_range() -> wgt::ImageSubresourceRange {
        wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        }
    }

    /// The view descriptor of a canvas texture the pipeline renders into.
    fn surface_view(format: wgt::TextureFormat) -> TextureViewDescriptor<'static> {
        TextureViewDescriptor {
            label: Some("codevar-triangle-view"),
            format,
            dimension: wgt::TextureViewDimension::D2,
            usage: wgt::TextureUses::COLOR_TARGET,
            range: full_range(),
        }
    }

    /// One open device plus the triangle pipeline recorded every frame.
    ///
    /// The context is kept alive for the life of the page: the frame loop
    /// holds this value through an `Rc`, so it is never dropped.
    struct Triangle {
        /// Device, queue, canvas surface and fence.
        context: WgContext,
        /// The single command encoder, reset and reused every frame.
        encoder: CommandEncoder,
        /// The triangle pipeline itself.
        pipeline: RenderPipeline,
        /// Value the next submission stores in the context fence.
        fence_value: u64,
    }

    impl Triangle {
        /// Opens the context and builds the triangle pipeline.
        fn new(context: WgContext) -> ExampleResult<Self> {
            let device = context.device();
            let format = context.format();

            let shader_module = |label: &str, source: &'static str| {
                // SAFETY: `source` is a complete GLSL ES 3.00 program and the
                // descriptor carries no borrowed state past the call.
                unsafe {
                    device.create_shader_module(
                        &ShaderModuleDescriptor {
                            label: Some(label),
                            runtime_checks: wgt::ShaderRuntimeChecks::unchecked(),
                        },
                        ShaderInput::Glsl { shader: source },
                    )
                }
                .map_err(|error| failure("create_shader_module", error))
            };
            let vertex = shader_module("codevar-triangle.vert", VERTEX_GLSL)?;
            let fragment = shader_module("codevar-triangle.frag", FRAGMENT_GLSL)?;

            // SAFETY: the layout borrows no bind-group layout (the triangle
            // needs none) and no immediate data.
            let pipeline_layout = unsafe {
                device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("codevar-triangle-layout"),
                    flags: PipelineLayoutFlags::empty(),
                    bind_group_layouts: &[],
                    immediate_size: 0,
                })
            }
            .map_err(|error| failure("create_pipeline_layout", error))?;

            // The inferred type is `naga::back::PipelineConstants`; the
            // triangle declares no overridable constants.
            let constants = Default::default();
            let vertex_stage = ProgrammableStage {
                module: &vertex,
                entry_point: "main",
                constants: &constants,
                zero_initialize_workgroup_memory: false,
            };
            let fragment_stage = ProgrammableStage {
                module: &fragment,
                entry_point: "main",
                constants: &constants,
                zero_initialize_workgroup_memory: false,
            };

            // SAFETY: both shader modules are live for the call, the layout
            // outlives the pipeline, and the color target matches the
            // configured surface format.
            let pipeline = unsafe {
                device.create_render_pipeline(&RenderPipelineDescriptor {
                    label: Some("codevar-triangle"),
                    layout: &pipeline_layout,
                    vertex_processor: VertexProcessor::Standard {
                        vertex_buffers: &[],
                        vertex_stage,
                    },
                    primitive: wgt::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgt::MultisampleState::default(),
                    fragment_stage: Some(fragment_stage),
                    color_targets: &[Some(wgt::ColorTargetState {
                        format,
                        blend: Some(wgt::BlendState::REPLACE),
                        write_mask: wgt::ColorWrites::ALL,
                    })],
                    multiview_mask: None,
                    cache: None,
                })
            }
            .map_err(|error| failure("create_render_pipeline", error))?;

            // The program is linked; the modules and the layout have no
            // further use. SAFETY: the pipeline no longer references any of
            // them.
            unsafe {
                device.destroy_shader_module(vertex);
                device.destroy_shader_module(fragment);
                device.destroy_pipeline_layout(pipeline_layout);
            }

            // SAFETY: the queue is owned by `context` and outlives the
            // encoder.
            let encoder = unsafe {
                device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("codevar-triangle"),
                    queue: context.queue(),
                })
            }
            .map_err(|error| failure("create_command_encoder", error))?;

            Ok(Self {
                context,
                encoder,
                pipeline,
                fence_value: 0,
            })
        }

        /// Records and presents one triangle frame.
        fn draw_frame(&mut self) -> ExampleResult<()> {
            let device = self.context.device();
            let queue = self.context.queue();
            let surface = self.context.surface();
            let fence = self.context.fence();
            let extent = self.context.config().extent;

            // SAFETY: no frame is in flight, so the surface has no acquired
            // texture, and the fence is the one every submission signals.
            let acquired = unsafe { surface.acquire_texture(None, fence) }
                .map_err(|error| failure("acquire_texture", error))?;
            let texture = acquired.texture;

            // SAFETY: the view covers the whole acquired surface texture,
            // which was created on this device.
            let view = unsafe { device.create_texture_view(&texture, &surface_view(self.context.format())) }
                .map_err(|error| failure("create_texture_view", error))?;

            // SAFETY: the encoder is not recording; the previous frame was
            // reset after its submission.
            unsafe {
                self.encoder
                    .begin_encoding(Some("codevar-triangle"))
            }
            .map_err(|error| failure("begin_encoding", error))?;

            let barrier = TextureBarrier {
                texture: &texture,
                range: full_range(),
                usage: StateTransition {
                    from: wgt::TextureUses::PRESENT,
                    to: wgt::TextureUses::COLOR_TARGET,
                },
            };
            // SAFETY: the barrier names the acquired surface texture and the
            // usage it is being moved to.
            unsafe {
                self.encoder
                    .transition_textures(core::iter::once(barrier))
            };

            let attachment = ColorAttachment {
                target: Attachment {
                    view: &view,
                    usage: wgt::TextureUses::COLOR_TARGET,
                },
                depth_slice: None,
                resolve_target: None,
                ops: AttachmentOps::LOAD_CLEAR | AttachmentOps::STORE,
                clear_value: wgt::Color {
                    r: 0.05,
                    g: 0.05,
                    b: 0.08,
                    a: 1.0,
                },
            };
            let attachments = [Some(attachment)];
            let pass = RenderPassDescriptor {
                label: Some("codevar-triangle-pass"),
                extent,
                sample_count: 1,
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            };
            // SAFETY: the attachment view is alive for the whole pass and is
            // a color target of the pass's only attachment slot.
            unsafe { self.encoder.begin_render_pass(&pass) }
                .map_err(|error| failure("begin_render_pass", error))?;

            let viewport = wgpu_hal::Rect {
                x: 0.0,
                y: 0.0,
                w: extent.width as f32,
                h: extent.height as f32,
            };
            let scissor = wgpu_hal::Rect {
                x: 0,
                y: 0,
                w: extent.width,
                h: extent.height,
            };
            // SAFETY: a render pass is recording, the pipeline is compatible
            // with the pass's only color target, and the draw reads the
            // three vertices the vertex shader generates from `gl_VertexID`.
            unsafe {
                self.encoder.set_viewport(&viewport, 0.0..1.0);
                self.encoder.set_scissor_rect(&scissor);
                self.encoder.set_render_pipeline(&self.pipeline);
                self.encoder.draw(0, 3, 0, 1);
                self.encoder.end_render_pass();
            }

            // SAFETY: a render pass was begun and ended on this encoder.
            let command_buffer =
                unsafe { self.encoder.end_encoding() }.map_err(|error| failure("end_encoding", error))?;

            self.fence_value += 1;
            // SAFETY: the command buffer writes only to the surface texture,
            // which is listed as a surface texture so `submit` transitions it
            // to the present state; the fence belongs to this context.
            unsafe { queue.submit(&[&command_buffer], &[&texture], (fence, self.fence_value)) }
                .map_err(|error| failure("submit", error))?;
            // SAFETY: the browser backend executes submissions synchronously,
            // so the command buffer is free to be recycled immediately.
            unsafe {
                self.encoder
                    .reset_all(core::iter::once(command_buffer))
            };
            // SAFETY: the render pass that referenced the view has ended and
            // the view is referenced by nothing else.
            unsafe { device.destroy_texture_view(view) };

            // SAFETY: the texture was acquired from this surface, submitted
            // (which moved it to the present state) and is not referenced
            // elsewhere.
            unsafe { queue.present(surface, texture) }.map_err(|error| failure("present", error))?;
            Ok(())
        }
    }

    /// Returns the window's inner size in device pixels, or a fixed fallback.
    fn window_extent(window: &Window) -> (u32, u32) {
        let width = window
            .inner_width()
            .ok()
            .and_then(|value| value.as_f64())
            .unwrap_or(f64::from(FALLBACK_WIDTH)) as u32;
        let height = window
            .inner_height()
            .ok()
            .and_then(|value| value.as_f64())
            .unwrap_or(f64::from(FALLBACK_HEIGHT)) as u32;
        (width.max(1), height.max(1))
    }

    /// Returns the page's canvas, creating and appending one when absent.
    ///
    /// `wasm-server-runner`'s own page has no `<canvas>`, so the example
    /// makes its own instead of relying on the host document.
    fn canvas(window: &Window) -> ExampleResult<HtmlCanvasElement> {
        let document = window
            .document()
            .ok_or("the page has no document")?;
        if let Some(element) = document.get_element_by_id(CANVAS_ID) {
            return element
                .dyn_into::<HtmlCanvasElement>()
                .map_err(|_| format!("the {CANVAS_ID} element is not a canvas").into());
        }

        let element = document
            .create_element("canvas")
            .map_err(js_failure)?;
        element.set_id(CANVAS_ID);
        let canvas = element
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| "create_element(\"canvas\") did not return a canvas")?;
        let (width, height) = window_extent(window);
        canvas.set_width(width);
        canvas.set_height(height);
        document
            .body()
            .ok_or("the page has no body")?
            .append_child(&canvas)
            .map_err(js_failure)?;
        Ok(canvas)
    }

    /// Schedules `slot`'s closure as the next animation frame.
    fn request_frame(window: &Window, slot: &FrameSlot) {
        if let Ok(slot) = slot.try_borrow()
            && let Some(closure) = slot.as_ref()
        {
            let _ = window.request_animation_frame(closure.as_ref().unchecked_ref());
        }
    }

    /// Opens the device, builds the pipeline and starts the frame loop.
    pub fn run() -> ExampleResult<()> {
        let window = web_sys::window().ok_or("the page has no window")?;
        let canvas = canvas(&window)?;
        let (width, height) = (canvas.width(), canvas.height());
        let context = WgContext::new(&WgConfig::new(canvas, width, height))
            .map_err(|error| failure("WgContext::new", error))?;
        let frame = Rc::new(RefCell::new(Triangle::new(context)?));

        let slot: FrameSlot = Rc::new(RefCell::new(None));
        let next = Rc::clone(&slot);
        let ticking_window = window.clone();
        *slot.borrow_mut() = Some(Closure::new(move |_timestamp: f64| {
            if let Ok(mut frame) = frame.try_borrow_mut()
                && let Err(error) = frame.draw_frame()
            {
                log(&format!("codevar-gpu-wg triangle: {error}"));
            }
            request_frame(&ticking_window, &next);
        }));

        request_frame(&window, &slot);
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    if let Err(error) = web::run() {
        web::log(&format!("codevar-gpu-wg triangle: {error}"));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!(
        "the WebGPU triangle only runs in a browser; build it for wasm32 with:\n    \
         cargo run -p codevar-gpu-wg --example triangle --target wasm32-unknown-unknown"
    );
}
