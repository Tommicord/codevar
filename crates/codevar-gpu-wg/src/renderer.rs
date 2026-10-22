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

//! The renderer that replays a display list with WebGPU.
//!
//! The renderer walks the display list of `codevar-kuitui` and fills
//! the vertex buffer of the frame with one quad per primitive. Every
//! quad carries its own color and its own corner radii, so a frame is
//! a single draw call.
//!
//! # The shape of a frame
//!
//! 1. [`WgRenderer::render`] turns the display list into a mesh,
//!    uploads it and records the draw call.
//! 2. [`WgRenderer::present`] submits the frame and, in a browser,
//!    hands it to the canvas.
//!
//! On the desktop there is no canvas, so the frame is rendered into an
//! offscreen texture; the application reads it back or hands it to
//! another backend.

use codevar_kuitui::DrawList;

use crate::config::WgConfig;
use crate::error::WgError;
use crate::mesh::{Mesh, build_mesh};
use crate::text::GlyphAtlas;
use crate::upload::{pad_coverage, vertex_bytes, vertex_layout};

/// The shader of the user interface.
const SHADER: &str = include_str!("shaders/ui.wgsl");

/// The number of bytes of the uniform of a frame.
const UNIFORM_SIZE: u64 = 16;

/// The renderer that replays a display list with WebGPU.
pub struct WgRenderer {
    /// The configuration of the renderer.
    config: WgConfig,
    /// The device every resource of the renderer lives on.
    device: wgpu::Device,
    /// The queue the frames are submitted to.
    queue: wgpu::Queue,
    /// The pipeline that draws the quads of a frame.
    pipeline: Option<wgpu::RenderPipeline>,
    /// The group that holds the uniform, the atlas and the sampler.
    bind_group: Option<wgpu::BindGroup>,
    /// The uniform of the frame in flight.
    uniform_buffer: Option<wgpu::Buffer>,
    /// The vertices of the frame.
    vertex_buffer: Option<wgpu::Buffer>,
    /// The indices of the frame.
    index_buffer: Option<wgpu::Buffer>,
    /// The texture of the atlas of the glyphs.
    atlas_texture: Option<wgpu::Texture>,
    /// The view of the texture of the atlas.
    atlas_view: Option<wgpu::TextureView>,
    /// The sampler of the atlas.
    sampler: Option<wgpu::Sampler>,
    /// The mesh of the frame.
    mesh: Mesh,
    /// The atlas of the glyphs.
    atlas: GlyphAtlas,
    /// The revision of the atlas the texture was filled with.
    atlas_revision: u32,
    /// The surface the frames are presented to, in a browser.
    surface: Option<wgpu::Surface<'static>>,
    /// The texture of the offscreen target, on the desktop.
    offscreen: Option<wgpu::TextureView>,
    /// The encoder that records the frame.
    encoder: Option<wgpu::CommandEncoder>,
    /// The view of the texture the frame is rendered into.
    target: Option<wgpu::TextureView>,
    /// The texture of the surface the frame is rendered into.
    surface_texture: Option<wgpu::SurfaceTexture>,
    /// The width of the target, in device pixels.
    width: u32,
    /// The height of the target, in device pixels.
    height: u32,
    /// The number of device pixels per logical pixel.
    scale: f32,
    /// The format of the target of the renderer.
    color_format: wgpu::TextureFormat,
}

impl WgRenderer {
    /// Creates a renderer for `config`.
    ///
    /// The device is asked for asynchronously: in a browser the
    /// adapter is chosen by the page, which needs a turn of the event
    /// loop.
    ///
    /// # Errors
    ///
    /// Returns a [`WgError`] when no adapter can render, when the
    /// device cannot be created, when the canvas of the configuration
    /// is not in the document or when the surface of the canvas cannot
    /// be created.
    pub async fn new(config: WgConfig) -> Result<Self, WgError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|error| WgError::AdapterNotFound(error.to_string()))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("codevar-kuitui"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| WgError::DeviceCreate(error.to_string()))?;

        let surface = create_surface(&instance, &config)?;
        let color_format = match &surface {
            Some(surface) => {
                let capabilities = surface.get_capabilities(&adapter);
                capabilities
                    .formats
                    .iter()
                    .copied()
                    .find(|format| format.is_srgb())
                    .unwrap_or(capabilities.formats[0])
            }
            None => wgpu::TextureFormat::Rgba8Unorm,
        };

        let mut renderer = Self::empty(config, device, queue, color_format, surface);
        renderer.create_resources(atlas_size())?;
        if let Some(surface) = renderer.surface.as_ref() {
            let configuration = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: color_format,
                width: renderer.width.max(1),
                height: renderer.height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: renderer.alpha_mode(),
                color_space: wgpu::SurfaceColorSpace::Srgb,
                view_formats: Vec::new(),
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&renderer.device, &configuration);
        }
        Ok(renderer)
    }

    /// Creates a renderer without any resource.
    fn empty(
        config: WgConfig,
        device: wgpu::Device,
        queue: wgpu::Queue,
        color_format: wgpu::TextureFormat,
        surface: Option<wgpu::Surface<'static>>,
    ) -> Self {
        let width = config.width.max(1);
        let height = config.height.max(1);
        Self {
            config,
            device,
            queue,
            pipeline: None,
            bind_group: None,
            uniform_buffer: None,
            vertex_buffer: None,
            index_buffer: None,
            atlas_texture: None,
            atlas_view: None,
            sampler: None,
            mesh: Mesh::new(),
            atlas: GlyphAtlas::new(atlas_size().0, atlas_size().1),
            atlas_revision: 0,
            surface,
            offscreen: None,
            encoder: None,
            target: None,
            surface_texture: None,
            width,
            height,
            scale: 1.0,
            color_format,
        }
    }

    /// The configuration of the renderer.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &WgConfig {
        &self.config
    }

    /// The atlas of the rasterized glyphs.
    #[inline]
    #[must_use]
    pub const fn atlas(&self) -> &GlyphAtlas {
        &self.atlas
    }

    /// The atlas of the rasterized glyphs, for mutation.
    ///
    /// A glyph that is added to the atlas is uploaded to the texture of
    /// the atlas at the beginning of the next frame.
    #[inline]
    pub fn atlas_mut(&mut self) -> &mut GlyphAtlas {
        &mut self.atlas
    }

    /// The number of quads of the last frame.
    #[inline]
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.mesh.quad_count()
    }

    /// The width of the target, in device pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The height of the target, in device pixels.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Resizes the target.
    ///
    /// The size is in device pixels; the number of device pixels per
    /// logical pixel follows from it.
    pub fn resize(&mut self, width: u32, height: u32, scale: f32) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.scale = scale.max(f32::MIN_POSITIVE);
        if let Some(surface) = self.surface.as_ref() {
            let configuration = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.color_format,
                width: self.width,
                height: self.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: self.alpha_mode(),
                color_space: wgpu::SurfaceColorSpace::Srgb,
                view_formats: Vec::new(),
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&self.device, &configuration);
        } else {
            self.offscreen = None;
        }
    }

    /// Draws the display list into the target of the renderer.
    ///
    /// The frame is recorded but not submitted; [`WgRenderer::present`]
    /// submits it.
    ///
    /// # Errors
    ///
    /// Returns a [`WgError`] when the target of the frame could not
    /// be acquired or when the buffers of the frame could not be
    /// filled.
    pub fn render(&mut self, list: &DrawList) -> Result<(), WgError> {
        let (target, surface_texture) = self.acquire_target()?;
        let viewport =
            codevar_kuitui::Size::new(self.width as f32 / self.scale, self.height as f32 / self.scale);
        build_mesh(list, &self.atlas, viewport, &mut self.mesh);
        self.upload_atlas();
        self.upload_mesh();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("codevar-kuitui-frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("codevar-kuitui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let pipeline = self
                .pipeline
                .as_ref()
                .ok_or_else(|| WgError::Pipeline(String::from("the pipeline is missing")))?;
            let bind_group = self
                .bind_group
                .as_ref()
                .ok_or_else(|| WgError::Pipeline(String::from("the bind group is missing")))?;
            let vertex_buffer = self
                .vertex_buffer
                .as_ref()
                .ok_or_else(|| WgError::Buffer(String::from("the vertex buffer is missing")))?;
            let index_buffer = self
                .index_buffer
                .as_ref()
                .ok_or_else(|| WgError::Buffer(String::from("the index buffer is missing")))?;
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.mesh.index_count() as u32, 0, 0..1);
        }
        self.encoder = Some(encoder);
        self.target = Some(target);
        self.surface_texture = surface_texture;
        Ok(())
    }

    /// Submits the frame and presents it.
    ///
    /// # Errors
    ///
    /// Returns a [`WgError`] when the frame could not be submitted.
    pub fn present(&mut self) -> Result<(), WgError> {
        let Some(encoder) = self.encoder.take() else {
            return Ok(());
        };
        self.queue.submit(Some(encoder.finish()));
        self.target = None;
        // The surface texture is presented when it is dropped.
        self.surface_texture = None;
        Ok(())
    }

    /// Creates the buffers, the pipeline and the texture of the atlas.
    fn create_resources(&mut self, atlas: (u32, u32)) -> Result<(), WgError> {
        self.uniform_buffer = Some(
            self.device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("codevar-kuitui-uniform"),
                    size: UNIFORM_SIZE,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
        );
        self.vertex_buffer = Some(
            self.device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("codevar-kuitui-vertices"),
                    size: VERTEX_CAPACITY,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
        );
        self.index_buffer = Some(
            self.device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("codevar-kuitui-indices"),
                    size: INDEX_CAPACITY,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
        );

        let atlas_texture = self
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("codevar-kuitui-atlas"),
                size: wgpu::Extent3d {
                    width: atlas.0.max(1),
                    height: atlas.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
        self.atlas_view = Some(atlas_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.atlas_texture = Some(atlas_texture);

        self.sampler = Some(
            self.device
                .create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("codevar-kuitui-atlas-sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..wgpu::SamplerDescriptor::default()
                }),
        );

        let bind_group_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("codevar-kuitui-bind-group"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(UNIFORM_SIZE),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        self.bind_group =
            Some(
                self.device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("codevar-kuitui-bind-group"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self
                                    .uniform_buffer
                                    .as_ref()
                                    .ok_or_else(|| {
                                        WgError::Buffer(String::from("the uniform buffer is missing"))
                                    })?
                                    .as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    self.atlas_view.as_ref().ok_or_else(|| {
                                        WgError::Buffer(String::from("the atlas is missing"))
                                    })?,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(self.sampler.as_ref().ok_or_else(
                                    || WgError::Buffer(String::from("the sampler is missing")),
                                )?),
                            },
                        ],
                    }),
            );

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("codevar-kuitui-shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("codevar-kuitui-layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        self.pipeline = Some(
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("codevar-kuitui-pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[Some(vertex_layout())],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: self.color_format,
                            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                }),
        );
        Ok(())
    }

    /// The alpha mode of the surface.
    fn alpha_mode(&self) -> wgpu::CompositeAlphaMode {
        match self.config.alpha_mode {
            crate::config::WgAlphaMode::Opaque => wgpu::CompositeAlphaMode::Opaque,
            crate::config::WgAlphaMode::Premultiplied => wgpu::CompositeAlphaMode::PreMultiplied,
        }
    }

    /// Acquires the texture the frame is rendered into.
    fn acquire_target(&mut self) -> Result<(wgpu::TextureView, Option<wgpu::SurfaceTexture>), WgError> {
        if let Some(surface) = self.surface.as_ref() {
            let frame = surface.get_current_texture();
            let texture = match frame {
                wgpu::CurrentSurfaceTexture::Success(texture) => texture,
                wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                wgpu::CurrentSurfaceTexture::Timeout => {
                    return Err(WgError::Submit(String::from("the surface timed out")));
                }
                wgpu::CurrentSurfaceTexture::Occluded => {
                    return Err(WgError::Submit(String::from("the surface is occluded")));
                }
                wgpu::CurrentSurfaceTexture::Outdated => {
                    return Err(WgError::Submit(String::from("the surface is outdated")));
                }
                other => {
                    return Err(WgError::Submit(alloc::format!(
                        "the surface cannot be acquired: {other:?}"
                    )));
                }
            };
            let view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            return Ok((view, Some(texture)));
        }
        if self.offscreen.is_none() {
            let texture = self
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("codevar-kuitui-offscreen"),
                    size: wgpu::Extent3d {
                        width: self.width,
                        height: self.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
            self.offscreen = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        }
        let view = self
            .offscreen
            .clone()
            .ok_or_else(|| WgError::Submit(String::from("the target is missing")))?;
        Ok((view, None))
    }

    /// Uploads the atlas of the glyphs when it changed.
    fn upload_atlas(&mut self) {
        if self.atlas.revision() == self.atlas_revision {
            return;
        }
        self.atlas_revision = self.atlas.revision();
        let (width, height) = atlas_size();
        let bytes = self.atlas.coverage();
        let padded = pad_coverage(bytes, width, height);
        let Some(texture) = self.atlas_texture.as_ref() else {
            return;
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Uploads the mesh of the frame.
    fn upload_mesh(&mut self) {
        let vertices = self.mesh.vertices();
        let indices = self.mesh.indices();
        if !vertices.is_empty()
            && let Some(buffer) = self.vertex_buffer.as_ref()
        {
            let bytes = vertex_bytes(vertices);
            self.queue.write_buffer(buffer, 0, &bytes);
        }
        if !indices.is_empty()
            && let Some(buffer) = self.index_buffer.as_ref()
        {
            let mut bytes = Vec::with_capacity(indices.len() * 4);
            for index in indices {
                bytes.extend_from_slice(&index.to_le_bytes());
            }
            self.queue.write_buffer(buffer, 0, &bytes);
        }
        let mut uniform = Vec::with_capacity(UNIFORM_SIZE as usize);
        uniform.extend_from_slice(&(self.width as f32).to_le_bytes());
        uniform.extend_from_slice(&(self.height as f32).to_le_bytes());
        uniform.extend_from_slice(&self.scale.to_le_bytes());
        uniform.extend_from_slice(&0.0_f32.to_le_bytes());
        if let Some(buffer) = self.uniform_buffer.as_ref() {
            self.queue.write_buffer(buffer, 0, &uniform);
        }
    }
}

/// The size of the atlas of the glyphs.
fn atlas_size() -> (u32, u32) {
    (1024, 1024)
}

/// The capacity of the vertex buffer, in bytes.
const VERTEX_CAPACITY: u64 = 1 << 20;

/// The capacity of the index buffer, in bytes.
const INDEX_CAPACITY: u64 = 1 << 18;

/// Creates the surface of the canvas of the configuration.
///
/// On the desktop there is no canvas, so the renderer draws into an
/// offscreen texture.
#[cfg(target_arch = "wasm32")]
fn create_surface(
    instance: &wgpu::Instance,
    config: &WgConfig,
) -> Result<Option<wgpu::Surface<'static>>, WgError> {
    let canvas = find_canvas(&config.canvas_id)?;
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(|error| WgError::SurfaceCreate(error.to_string()))?;
    Ok(Some(surface))
}

/// Creates the surface of the canvas of the configuration.
///
/// On the desktop there is no canvas, so the renderer draws into an
/// offscreen texture.
#[cfg(not(target_arch = "wasm32"))]
fn create_surface(
    _instance: &wgpu::Instance,
    _config: &WgConfig,
) -> Result<Option<wgpu::Surface<'static>>, WgError> {
    Ok(None)
}

/// Finds the canvas of the document that the configuration names.
#[cfg(target_arch = "wasm32")]
fn find_canvas(id: &str) -> Result<wgpu::web_sys::HtmlCanvasElement, WgError> {
    use wasm_bindgen::JsCast;

    let window = wgpu::web_sys::window().ok_or_else(|| WgError::CanvasNotFound(String::from(id)))?;
    let document = window
        .document()
        .ok_or_else(|| WgError::CanvasNotFound(String::from(id)))?;
    let element = document
        .get_element_by_id(id)
        .ok_or_else(|| WgError::CanvasNotFound(String::from(id)))?;
    element
        .dyn_into::<wgpu::web_sys::HtmlCanvasElement>()
        .map_err(|_| WgError::CanvasNotFound(String::from(id)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_kuitui::Color;
    use codevar_kuitui::geometry::{Offset, Rect};
    use codevar_kuitui::paint::{CornerRadii, Paint};

    #[test]
    fn the_mesh_of_a_list_holds_a_quad_per_primitive() {
        let mut list = codevar_kuitui::DrawList::new();
        list.rect(
            Rect::from_min_size(Offset::ZERO, 10.0, 10.0),
            CornerRadii::ZERO,
            Paint::fill(Color::RED),
        );
        let mut mesh = Mesh::new();
        build_mesh(
            &list,
            &GlyphAtlas::new(64, 64),
            codevar_kuitui::Size::new(100.0, 100.0),
            &mut mesh,
        );
        assert_eq!(mesh.quad_count(), 1);
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.index_count(), 6);
    }
}
